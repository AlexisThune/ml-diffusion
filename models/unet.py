import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class SimpleUNet(nn.Module):
    """Deterministic regression baseline for coarse-to-fine field reconstruction."""

    def __init__(self, in_channels=3, out_channels=1, base_channels=32):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)
        self.dec2 = ConvBlock(base_channels * 6, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 3, base_channels)
        self.output = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        bottleneck = self.bottleneck(self.pool(enc2))
        dec2 = F.interpolate(bottleneck, size=enc2.shape[-2:], mode="bilinear", align_corners=False)
        dec2 = self.dec2(torch.cat((dec2, enc2), dim=1))
        dec1 = F.interpolate(dec2, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
        dec1 = self.dec1(torch.cat((dec1, enc1), dim=1))
        return self.output(dec1)