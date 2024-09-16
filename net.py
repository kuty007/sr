import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return out + residual


class EDSR(nn.Module):
    def __init__(self, num_channels=3, num_residual_blocks=16, upscale_factor=2):
        super(EDSR, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, padding=1)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residual_blocks)]
        )
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.upscale = nn.Sequential(
            nn.Conv2d(64, 64 * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(64, num_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.residual_blocks(out)
        out = self.conv2(out)
        out = self.upscale(out)
        return out
