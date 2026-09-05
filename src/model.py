"""Feedforward neural network used in the reported experiments."""

from __future__ import annotations

import torch
from torch import nn


class ResidualMinimizationFNN(nn.Module):
    """Dimension-specific FNN for parameter-to-solution approximation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout_probability: float = 0.2,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
            nn.Linear(32, output_dim),
        )

    def forward(self, parameters: torch.Tensor) -> torch.Tensor:
        """Predict a solution from standardized system parameters."""
        return self.network(parameters)
