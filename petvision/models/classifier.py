"""A compact CNN classifier implemented for the course project."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SmallCNN(nn.Module):
    """Four convolutional stages followed by global average pooling."""

    def __init__(self, num_classes: int = 37, base_channels: int = 32) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4]
        self.features = nn.Sequential(
            ConvBlock(3, channels[0]),
            nn.MaxPool2d(2),
            ConvBlock(channels[0], channels[1]),
            nn.MaxPool2d(2),
            ConvBlock(channels[1], channels[2]),
            nn.MaxPool2d(2),
            ConvBlock(channels[2], channels[2] * 2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(channels[2] * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))
