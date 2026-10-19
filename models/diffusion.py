import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps, dimension):
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10000)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half, 1)
    )
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)
    if dimension % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class TimeResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_projection = nn.Linear(time_channels, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x, time_embedding):
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.time_projection(F.silu(time_embedding))[:, :, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return (hidden + self.skip(x)) / math.sqrt(2)


class DiffusionUNet(nn.Module):
    """Conditional U-Net that predicts noise in a noisy target field."""

    def __init__(self, condition_channels=3, out_channels=1, base_channels=32):
        super().__init__()
        time_channels = base_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(time_channels, time_channels),
            nn.SiLU(),
            nn.Linear(time_channels, time_channels),
        )
        self.input = nn.Conv2d(condition_channels + out_channels, base_channels, 3, padding=1)
        self.enc1 = TimeResidualBlock(base_channels, base_channels, time_channels)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1)
        self.enc2 = TimeResidualBlock(base_channels * 2, base_channels * 2, time_channels)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1)
        self.middle = TimeResidualBlock(base_channels * 4, base_channels * 4, time_channels)
        self.dec2 = TimeResidualBlock(base_channels * 6, base_channels * 2, time_channels)
        self.dec1 = TimeResidualBlock(base_channels * 3, base_channels, time_channels)
        self.output = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, noisy_target, condition, timesteps):
        time_embedding = self.time_mlp(timestep_embedding(timesteps, self.time_mlp[0].in_features))
        enc1 = self.enc1(self.input(torch.cat((noisy_target, condition), dim=1)), time_embedding)
        enc2 = self.enc2(self.down1(enc1), time_embedding)
        middle = self.middle(self.down2(enc2), time_embedding)
        dec2 = F.interpolate(middle, size=enc2.shape[-2:], mode="bilinear", align_corners=False)
        dec2 = self.dec2(torch.cat((dec2, enc2), dim=1), time_embedding)
        dec1 = F.interpolate(dec2, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
        dec1 = self.dec1(torch.cat((dec1, enc1), dim=1), time_embedding)
        return self.output(dec1)


class GaussianDiffusion(nn.Module):
    """DDPM training objective with deterministic DDIM sampling."""

    def __init__(self, model, num_steps=100):
        super().__init__()
        if num_steps < 2:
            raise ValueError("num_steps must be at least 2")
        self.model = model
        self.num_steps = num_steps
        steps = torch.arange(num_steps + 1, dtype=torch.float32) / num_steps
        offset = 0.008
        alpha_bars = torch.cos((steps + offset) / (1 + offset) * math.pi / 2).square()
        alpha_bars = alpha_bars / alpha_bars[0]
        betas = (1 - alpha_bars[1:] / alpha_bars[:-1]).clamp(max=0.999)
        alphas = 1.0 - betas
        self.register_buffer("alpha_bars", torch.cumprod(alphas, dim=0))

    def training_loss(self, target, condition):
        batch_size = target.shape[0]
        timesteps = torch.randint(self.num_steps, (batch_size,), device=target.device)
        noise = torch.randn_like(target)
        alpha_bar = self.alpha_bars[timesteps].view(batch_size, 1, 1, 1)
        noisy_target = alpha_bar.sqrt() * target + (1 - alpha_bar).sqrt() * noise
        predicted_noise = self.model(noisy_target, condition, timesteps)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, condition, sampling_steps=25, generator=None):
        if sampling_steps < 1:
            raise ValueError("sampling_steps must be positive")
        batch_size, _, height, width = condition.shape
        device = condition.device
        sample = torch.randn(
            (batch_size, 1, height, width), device=device, generator=generator
        )
        step_count = min(sampling_steps, self.num_steps)
        timesteps = torch.linspace(
            self.num_steps - 1, 0, step_count, device=device
        ).round().long().unique_consecutive()

        for index, timestep in enumerate(timesteps):
            time_batch = timestep.expand(batch_size)
            alpha_bar = self.alpha_bars[timestep]
            predicted_noise = self.model(sample, condition, time_batch)
            predicted_target = (sample - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
            if index + 1 < len(timesteps):
                previous_alpha_bar = self.alpha_bars[timesteps[index + 1]]
            else:
                previous_alpha_bar = torch.ones((), device=device)
            sample = (
                previous_alpha_bar.sqrt() * predicted_target
                + (1 - previous_alpha_bar).sqrt() * predicted_noise
            )
        return sample