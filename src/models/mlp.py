"""MLP baseline for wafer map classification."""
from __future__ import annotations

import torch
from torch import nn


class MLPClassifier(nn.Module):
    """A simple multilayer perceptron classifier."""

    def __init__(self, input_shape: tuple[int, int, int], num_classes: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        c, h, w = input_shape
        input_dim = c * h * w
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(x)
