import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SyntheticOceanDataset(Dataset):
    """Deterministic synthetic fields for a reproducible downscaling benchmark."""

    def __init__(self, num_samples=1000, size=64, downscale_factor=4, seed=0):
        if downscale_factor < 2:
            raise ValueError("downscale_factor must be at least 2")
        if size < 1 or size % downscale_factor != 0:
            raise ValueError("size must be positive and divisible by downscale_factor")
        if num_samples < 1:
            raise ValueError("num_samples must be positive")

        self.num_samples = num_samples
        self.size = size
        self.downscale_factor = downscale_factor
        self.seed = seed
        coordinates = np.linspace(-1.0, 1.0, size, dtype=np.float32)
        self.xx, self.yy = np.meshgrid(coordinates, coordinates)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed + idx)
        xx, yy = self.xx, self.yy

        angle = rng.uniform(0, 2 * np.pi)
        phase = rng.uniform(0, 2 * np.pi)
        along_front = xx * np.cos(angle) + yy * np.sin(angle)
        across_front = -xx * np.sin(angle) + yy * np.cos(angle)
        eddy_x, eddy_y = rng.uniform(-0.45, 0.45, size=2)
        eddy_radius = rng.uniform(0.18, 0.42)
        eddy = np.exp(-((xx - eddy_x) ** 2 + (yy - eddy_y) ** 2) / eddy_radius**2)

        fine_sss = (
            0.42 * np.sin(2 * np.pi * (1.1 * along_front) + phase)
            + 0.24 * np.sin(2 * np.pi * (1.8 * across_front) - phase / 2)
            + 0.22 * eddy * np.cos(5 * np.pi * across_front + phase)
            + 0.12 * np.tanh(8 * (along_front - rng.uniform(-0.25, 0.25)))
            + rng.normal(0, 0.015, size=(self.size, self.size))
        ).astype(np.float32)

        sst = (
            0.68 * fine_sss
            + 0.25 * np.cos(np.pi * across_front + phase)
            + rng.normal(0, 0.06, size=(self.size, self.size))
        ).astype(np.float32)
        ssh = (
            0.3 * fine_sss
            + 0.5 * np.sin(np.pi * (along_front + across_front) - phase)
            + 0.2 * np.cos(2 * np.pi * across_front)
            + rng.normal(0, 0.05, size=(self.size, self.size))
        ).astype(np.float32)

        factor = self.downscale_factor
        low_resolution = fine_sss.reshape(
            self.size // factor, factor, self.size // factor, factor
        ).mean(axis=(1, 3))
        low_resolution = torch.from_numpy(low_resolution)[None, None]
        coarse_sss = F.interpolate(
            low_resolution,
            size=(self.size, self.size),
            mode="bicubic",
            align_corners=False,
        )[0, 0].numpy()

        inputs = np.stack((coarse_sss, sst, ssh), axis=0)
        target = fine_sss[None]
        return torch.from_numpy(inputs), torch.from_numpy(target)