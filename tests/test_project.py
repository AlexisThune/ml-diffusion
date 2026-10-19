import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.dataset import SyntheticOceanDataset
from models.diffusion import DiffusionUNet, GaussianDiffusion
from models.unet import SimpleUNet
from utils.benchmark import evaluate_models, save_comparison_figure, save_metrics_csv
from utils.metrics import calculate_metrics


class DatasetTests(unittest.TestCase):
    def test_samples_are_reproducible_and_actually_downscaled(self):
        dataset = SyntheticOceanDataset(3, size=32, downscale_factor=4, seed=17)
        inputs, target = dataset[0]

        self.assertEqual(inputs.shape, (3, 32, 32))
        self.assertEqual(target.shape, (1, 32, 32))
        self.assertTrue(torch.equal(inputs, dataset[0][0]))
        self.assertTrue(torch.equal(target, dataset[0][1]))
        self.assertLess(inputs[0].var(), target[0].var())
        self.assertFalse(torch.equal(target, dataset[1][1]))

    def test_invalid_downscale_factor_is_rejected(self):
        with self.assertRaises(ValueError):
            SyntheticOceanDataset(size=30, downscale_factor=4)


class ModelTests(unittest.TestCase):
    def test_regression_unet_forward_and_backward(self):
        model = SimpleUNet(base_channels=8)
        output = model(torch.randn(2, 3, 32, 32))

        self.assertEqual(output.shape, (2, 1, 32, 32))
        output.square().mean().backward()
        self.assertIsNotNone(model.output.weight.grad)

    def test_diffusion_loss_and_reproducible_sampling(self):
        diffusion = GaussianDiffusion(DiffusionUNet(base_channels=8), num_steps=8)
        condition = torch.randn(2, 3, 16, 16)
        target = torch.randn(2, 1, 16, 16)

        loss = diffusion.training_loss(target, condition)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        first = diffusion.sample(
            condition, 3, generator=torch.Generator().manual_seed(9)
        )
        second = diffusion.sample(
            condition, 3, generator=torch.Generator().manual_seed(9)
        )
        self.assertEqual(first.shape, target.shape)
        self.assertTrue(torch.equal(first, second))
        self.assertLess(diffusion.alpha_bars[-1].item(), 0.01)


class EvaluationTests(unittest.TestCase):
    def test_metrics_are_exact_for_constant_bias(self):
        target = torch.zeros(1, 1, 16, 16)
        metrics = calculate_metrics(target + 2, target)

        self.assertAlmostEqual(metrics["mae"], 2.0)
        self.assertAlmostEqual(metrics["rmse"], 2.0)
        self.assertAlmostEqual(metrics["bias"], 2.0)

    def test_uncertainty_metrics_and_artifacts(self):
        dataset = SyntheticOceanDataset(2, size=16, downscale_factor=4, seed=22)
        unet = SimpleUNet(base_channels=8)
        diffusion = GaussianDiffusion(DiffusionUNet(base_channels=8), num_steps=4)
        metrics, examples = evaluate_models(
            unet,
            diffusion,
            DataLoader(dataset, batch_size=2),
            torch.device("cpu"),
            sampling_steps=2,
            seed=31,
            uncertainty_samples=3,
        )

        self.assertEqual(examples["diffusion_samples"].shape, (3, 2, 1, 16, 16))
        self.assertGreater(metrics["Diffusion"]["mean_std"], 0)
        self.assertGreaterEqual(metrics["Diffusion"]["coverage_95"], 0)
        self.assertLessEqual(metrics["Diffusion"]["coverage_95"], 1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save_metrics_csv(metrics, output / "metrics.csv")
            save_comparison_figure(examples, output / "comparison.png")
            self.assertTrue((output / "metrics.csv").is_file())
            self.assertTrue((output / "comparison.png").is_file())


if __name__ == "__main__":
    unittest.main()