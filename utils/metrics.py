import torch
import torch.nn.functional as F


def calculate_metrics(prediction, target):
    """Return complementary pixel, structure, and amplitude metrics."""
    prediction = prediction.float()
    target = target.float()
    error = prediction - target
    mse = torch.mean(error.square())
    rmse = mse.sqrt()
    data_range = (target.max() - target.min()).clamp_min(1e-8)

    window_size = min(7, target.shape[-2], target.shape[-1])
    mean_prediction = F.avg_pool2d(prediction, window_size, stride=1)
    mean_target = F.avg_pool2d(target, window_size, stride=1)
    variance_prediction = F.avg_pool2d(prediction.square(), window_size, stride=1) - mean_prediction.square()
    variance_target = F.avg_pool2d(target.square(), window_size, stride=1) - mean_target.square()
    covariance = F.avg_pool2d(prediction * target, window_size, stride=1) - mean_prediction * mean_target
    c1 = (0.01 * data_range).square()
    c2 = (0.03 * data_range).square()
    ssim = (
        (2 * mean_prediction * mean_target + c1)
        * (2 * covariance + c2)
        / (
            (mean_prediction.square() + mean_target.square() + c1)
            * (variance_prediction + variance_target + c2)
        ).clamp_min(1e-8)
    ).mean()

    prediction_centered = prediction.flatten() - prediction.mean()
    target_centered = target.flatten() - target.mean()
    correlation = torch.sum(prediction_centered * target_centered) / (
        torch.linalg.vector_norm(prediction_centered)
        * torch.linalg.vector_norm(target_centered)
    ).clamp_min(1e-8)

    return {
        "rmse": rmse.item(),
        "mae": error.abs().mean().item(),
        "bias": error.mean().item(),
        "psnr": (20 * torch.log10(data_range / rmse.clamp_min(1e-8))).item(),
        "ssim": ssim.item(),
        "correlation": correlation.item(),
    }