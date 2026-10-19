import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.dataset import SyntheticOceanDataset
from models.diffusion import DiffusionUNet, GaussianDiffusion
from models.unet import SimpleUNet
from utils.benchmark import (
    evaluate_models,
    print_metrics_table,
    save_comparison_figure,
    save_metrics_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained downscaling models on held-out synthetic fields.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/inference"))
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--uncertainty-samples", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")
    if args.sampling_steps is not None and args.sampling_steps < 1:
        parser.error("--sampling-steps must be positive")
    if args.uncertainty_samples is not None and args.uncertainty_samples < 2:
        parser.error("--uncertainty-samples must be at least 2")
    return args


def main():
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir
    with (checkpoint_dir / "experiment.json").open(encoding="utf-8") as source:
        experiment = json.load(source)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    unet_checkpoint = torch.load(
        checkpoint_dir / "unet.pt", map_location=device, weights_only=True
    )
    diffusion_checkpoint = torch.load(
        checkpoint_dir / "diffusion.pt", map_location=device, weights_only=True
    )
    base_channels = unet_checkpoint["model_config"]["base_channels"]
    unet = SimpleUNet(base_channels=base_channels).to(device)
    unet.load_state_dict(unet_checkpoint["model_state_dict"])
    diffusion_model = DiffusionUNet(base_channels=base_channels).to(device)
    diffusion_model.load_state_dict(diffusion_checkpoint["model_state_dict"])
    diffusion = GaussianDiffusion(
        diffusion_model, num_steps=diffusion_checkpoint["diffusion_steps"]
    ).to(device)

    sampling_steps = args.sampling_steps or experiment["sampling_steps"]
    uncertainty_samples = args.uncertainty_samples or experiment.get("uncertainty_samples", 8)
    dataset = SyntheticOceanDataset(
        num_samples=args.num_samples,
        size=experiment["size"],
        downscale_factor=experiment["downscale_factor"],
        seed=experiment["seed"] + 200_000,
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    metrics, examples = evaluate_models(
        unet,
        diffusion,
        dataloader,
        device,
        sampling_steps=sampling_steps,
        seed=experiment["seed"] + 300_000,
        uncertainty_samples=uncertainty_samples,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_metrics_csv(metrics, args.output_dir / "metrics.csv")
    save_comparison_figure(examples, args.output_dir / "comparison.png")
    print_metrics_table(metrics)
    print(f"Résultats sauvegardés dans {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()