"""Small CNN baseline."""
from __future__ import annotations

import torch
from torch import nn


def conv_block(in_channels: int, out_channels: int, kernel_size: int = 3, pool: bool = True) -> nn.Sequential:
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class SmallCNN(nn.Module):
    """A small convolutional network with 3 blocks."""

    def __init__(self, num_channels: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(num_channels, 32),
            conv_block(32, 64),
            conv_block(64, 128, pool=False),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.features(x)
        return self.classifier(x)
