import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding_mode="reflect"):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, padding_mode=padding_mode)
        self.norm1 = nn.GroupNorm(4, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1, padding_mode=padding_mode)
        self.norm2 = nn.GroupNorm(4, out_channels)
        self.se = SEBlock(out_channels)
        self.activation = nn.SiLU()

        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual_conv(x)  # Ensure matching channels
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x = self.se(x) + residual
        return self.activation(x)

class SEResNet(nn.Module):
    def __init__(self, in_channels=12, padding_mode="reflect"):
        super(SEResNet, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels, 12, kernel_size=3, padding=1, padding_mode=padding_mode)

        self.encoder = nn.Sequential(
            ResidualBlock(12, 64),  # Adjust input size to match `initial_conv` output
            ResidualBlock(64, 128),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 64, kernel_size=1)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(76, 64, kernel_size=3, padding=1, padding_mode=padding_mode),
            ResidualBlock(64, 64),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, padding_mode=padding_mode),
            ResidualBlock(32, 32),
            nn.Conv2d(32, 1, kernel_size=3, padding=1, padding_mode=padding_mode)
        )



    def forward(self, x):
        original_x = self.initial_conv(x)
        compressed = self.encoder(original_x)
        compressed = compressed.expand(-1, -1, x.shape[2], x.shape[3])
        x = torch.cat([compressed, original_x], dim=1)
        x = self.decoder(x)
        return x.squeeze(1)