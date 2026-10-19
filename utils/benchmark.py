import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from utils.metrics import calculate_metrics


def evaluate_models(
    unet, diffusion, dataloader, device, sampling_steps, seed, uncertainty_samples=8
):
    if uncertainty_samples < 2:
        raise ValueError("uncertainty_samples must be at least 2")
    unet.eval()
    diffusion.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    inputs_all = []
    targets_all = []
    predictions = {"Bicubic": [], "U-Net": [], "Diffusion": []}
    diffusion_samples = []

    with torch.inference_mode():
        for inputs, target in dataloader:
            inputs = inputs.to(device)
            target = target.to(device)
            inputs_all.append(inputs.cpu())
            targets_all.append(target.cpu())
            predictions["Bicubic"].append(inputs[:, :1].cpu())
            predictions["U-Net"].append(unet(inputs).cpu())
            batch_samples = torch.stack(
                [
                    diffusion.sample(inputs, sampling_steps, generator=generator)
                    for _ in range(uncertainty_samples)
                ]
            )
            diffusion_samples.append(batch_samples.cpu())
            predictions["Diffusion"].append(batch_samples.mean(dim=0).cpu())

    inputs = torch.cat(inputs_all)
    target = torch.cat(targets_all)
    predictions = {name: torch.cat(values) for name, values in predictions.items()}
    diffusion_samples = torch.cat(diffusion_samples, dim=1)
    uncertainty = diffusion_samples.std(dim=0, unbiased=False)
    metrics = {name: calculate_metrics(prediction, target) for name, prediction in predictions.items()}
    lower_bound = torch.quantile(diffusion_samples, 0.025, dim=0)
    upper_bound = torch.quantile(diffusion_samples, 0.975, dim=0)
    metrics["Diffusion"].update(
        {
            "coverage_95": ((target >= lower_bound) & (target <= upper_bound)).float().mean().item(),
            "mean_std": uncertainty.mean().item(),
            "interval_width_95": (upper_bound - lower_bound).mean().item(),
        }
    )
    return metrics, {
        "inputs": inputs,
        "target": target,
        "predictions": predictions,
        "diffusion_samples": diffusion_samples,
        "uncertainty": uncertainty,
    }


def save_metrics_csv(metrics, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = list(dict.fromkeys(name for values in metrics.values() for name in values))
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=("method", *metric_names))
        writer.writeheader()
        for method, values in metrics.items():
            writer.writerow({"method": method, **values})


def print_metrics_table(metrics):
    metric_names = list(dict.fromkeys(name for values in metrics.values() for name in values))
    headers = ["Méthode", *[name.upper() for name in metric_names]]
    rows = [
        [method, *[f"{values[name]:.4f}" if name in values else "-" for name in metric_names]]
        for method, values in metrics.items()
    ]
    widths = [max(len(row[column]) for row in [headers, *rows]) for column in range(len(headers))]
    print(" | ".join(value.ljust(width) for value, width in zip(headers, widths)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(width) for value, width in zip(row, widths)))


def save_comparison_figure(examples, path, max_examples=3):
    inputs = examples["inputs"]
    target = examples["target"]
    predictions = examples["predictions"]
    count = min(max_examples, target.shape[0])
    figure, axes = plt.subplots(count, 9, figsize=(21, 3 * count), squeeze=False)
    columns = (
        "Bicubic (entrée)",
        "U-Net",
        "DDIM tirage A",
        "DDIM tirage B",
        "Moyenne diffusion",
        "Cible fine",
        "Erreur absolue U-Net",
        "Erreur absolue moyenne",
        "Ecart-type diffusion",
    )

    for row in range(count):
        target_image = target[row, 0].numpy()
        low, high = float(target_image.min()), float(target_image.max())
        images = (
            inputs[row, 0].numpy(),
            predictions["U-Net"][row, 0].numpy(),
            examples["diffusion_samples"][0, row, 0].numpy(),
            examples["diffusion_samples"][min(1, examples["diffusion_samples"].shape[0] - 1), row, 0].numpy(),
            predictions["Diffusion"][row, 0].numpy(),
            target_image,
            (predictions["U-Net"][row, 0] - target[row, 0]).abs().numpy(),
            (predictions["Diffusion"][row, 0] - target[row, 0]).abs().numpy(),
            examples["uncertainty"][row, 0].numpy(),
        )
        error_limit = max(float(images[6].max()), float(images[7].max()), 1e-6)
        uncertainty_limit = max(float(images[8].max()), 1e-6)
        for column, (axis, image) in enumerate(zip(axes[row], images)):
            if column < 6:
                axis.imshow(image, cmap="coolwarm", vmin=low, vmax=high)
            elif column < 8:
                axis.imshow(image, cmap="magma", vmin=0, vmax=error_limit)
            else:
                axis.imshow(image, cmap="magma", vmin=0, vmax=uncertainty_limit)
            axis.set_title(columns[column])
            axis.axis("off")

    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)