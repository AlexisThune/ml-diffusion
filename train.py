import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    parser = argparse.ArgumentParser(description="Compare U-Net regression and conditional diffusion for downscaling.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--val-samples", type=int, default=64)
    parser.add_argument("--test-samples", type=int, default=64)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--downscale-factor", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--sampling-steps", type=int, default=15)
    parser.add_argument("--uncertainty-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if args.base_channels % 8:
        parser.error("--base-channels must be divisible by 8")
    if args.size < 16 or args.size % 4:
        parser.error("--size must be at least 16 and divisible by 4")
    if args.epochs < 1 or args.sampling_steps < 1 or args.uncertainty_samples < 2:
        parser.error("epochs and sampling steps must be positive; uncertainty samples must be at least 2")
    if args.diffusion_steps < 2 or args.batch_size < 1:
        parser.error("--diffusion-steps must be at least 2 and --batch-size must be positive")
    if min(args.train_samples, args.val_samples, args.test_samples) < 1:
        parser.error("all dataset splits must contain at least one sample")
    if args.downscale_factor < 2 or args.size % args.downscale_factor:
        parser.error("--size must be divisible by --downscale-factor, which must be at least 2")
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def evaluate_unet_loss(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_pixels = 0
    with torch.inference_mode():
        for inputs, target in dataloader:
            inputs, target = inputs.to(device), target.to(device)
            total_loss += F.mse_loss(model(inputs), target, reduction="sum").item()
            total_pixels += target.numel()
    return total_loss / total_pixels


def evaluate_diffusion_loss(diffusion, dataloader, device, seed):
    diffusion.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    total_loss = 0.0
    total_pixels = 0
    with torch.inference_mode():
        for inputs, target in dataloader:
            inputs, target = inputs.to(device), target.to(device)
            batch_size = target.shape[0]
            timesteps = torch.randint(
                diffusion.num_steps, (batch_size,), device=device, generator=generator
            )
            noise = torch.randn(target.shape, device=device, generator=generator)
            alpha_bar = diffusion.alpha_bars[timesteps].view(batch_size, 1, 1, 1)
            noisy_target = alpha_bar.sqrt() * target + (1 - alpha_bar).sqrt() * noise
            prediction = diffusion.model(noisy_target, inputs, timesteps)
            total_loss += F.mse_loss(prediction, noise, reduction="sum").item()
            total_pixels += noise.numel()
    return total_loss / total_pixels


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}; split seed: {args.seed}")

    dataset_args = {
        "size": args.size,
        "downscale_factor": args.downscale_factor,
    }
    train_dataset = SyntheticOceanDataset(args.train_samples, seed=args.seed, **dataset_args)
    val_dataset = SyntheticOceanDataset(args.val_samples, seed=args.seed + 100_000, **dataset_args)
    test_dataset = SyntheticOceanDataset(args.test_samples, seed=args.seed + 200_000, **dataset_args)
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, generator=loader_generator
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    unet = SimpleUNet(base_channels=args.base_channels).to(device)
    diffusion = GaussianDiffusion(
        DiffusionUNet(base_channels=args.base_channels), num_steps=args.diffusion_steps
    ).to(device)
    unet_optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-3, weight_decay=1e-4)
    diffusion_optimizer = torch.optim.AdamW(diffusion.parameters(), lr=1e-3, weight_decay=1e-4)

    best_unet_loss = float("inf")
    best_diffusion_loss = float("inf")
    best_unet_state = None
    best_diffusion_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        unet.train()
        diffusion.train()
        unet_train_loss = 0.0
        diffusion_train_loss = 0.0

        for inputs, target in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            inputs, target = inputs.to(device), target.to(device)

            unet_optimizer.zero_grad(set_to_none=True)
            unet_loss = F.mse_loss(unet(inputs), target)
            unet_loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            unet_optimizer.step()
            unet_train_loss += unet_loss.item()

            diffusion_optimizer.zero_grad(set_to_none=True)
            diffusion_loss = diffusion.training_loss(target, inputs)
            diffusion_loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)
            diffusion_optimizer.step()
            diffusion_train_loss += diffusion_loss.item()

        unet_train_loss /= len(train_loader)
        diffusion_train_loss /= len(train_loader)
        unet_val_loss = evaluate_unet_loss(unet, val_loader, device)
        diffusion_val_loss = evaluate_diffusion_loss(
            diffusion, val_loader, device, seed=args.seed + epoch
        )
        history.append(
            {
                "epoch": epoch,
                "unet_train_mse": unet_train_loss,
                "unet_val_mse": unet_val_loss,
                "diffusion_train_noise_mse": diffusion_train_loss,
                "diffusion_val_noise_mse": diffusion_val_loss,
            }
        )

        if unet_val_loss < best_unet_loss:
            best_unet_loss = unet_val_loss
            best_unet_state = copy.deepcopy(unet.state_dict())
            best_unet_epoch = epoch
        if diffusion_val_loss < best_diffusion_loss:
            best_diffusion_loss = diffusion_val_loss
            best_diffusion_state = copy.deepcopy(diffusion.state_dict())
            best_diffusion_epoch = epoch

        print(
            f"Epoch {epoch:02d}: U-Net val MSE={unet_val_loss:.5f}; "
            f"diffusion val noise MSE={diffusion_val_loss:.5f}"
        )

    unet.load_state_dict(best_unet_state)
    diffusion.load_state_dict(best_diffusion_state)
    model_config = {"base_channels": args.base_channels}
    torch.save(
        {"model_state_dict": unet.state_dict(), "model_config": model_config},
        args.output_dir / "unet.pt",
    )
    torch.save(
        {
            "model_state_dict": diffusion.model.state_dict(),
            "model_config": model_config,
            "diffusion_steps": args.diffusion_steps,
        },
        args.output_dir / "diffusion.pt",
    )

    metrics, examples = evaluate_models(
        unet,
        diffusion,
        test_loader,
        device,
        sampling_steps=args.sampling_steps,
        seed=args.seed + 300_000,
        uncertainty_samples=args.uncertainty_samples,
    )
    save_metrics_csv(metrics, args.output_dir / "metrics.csv")
    save_comparison_figure(examples, args.output_dir / "comparison.png")
    with (args.output_dir / "training_history.json").open("w", encoding="utf-8") as output:
        json.dump(history, output, indent=2)
    metadata = {
        "seed": args.seed,
        "device": str(device),
        "train_samples": args.train_samples,
        "validation_samples": args.val_samples,
        "test_samples": args.test_samples,
        "size": args.size,
        "downscale_factor": args.downscale_factor,
        "epochs": args.epochs,
        "best_unet_epoch": best_unet_epoch,
        "best_diffusion_epoch": best_diffusion_epoch,
        "diffusion_steps": args.diffusion_steps,
        "sampling_steps": args.sampling_steps,
        "uncertainty_samples": args.uncertainty_samples,
        "test_metrics": metrics,
    }
    with (args.output_dir / "experiment.json").open("w", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2)

    print("\nRésultats sur le jeu de test indépendant:")
    print_metrics_table(metrics)
    print(f"\nArtefacts sauvegardés dans {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()