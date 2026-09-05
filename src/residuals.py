"""Differentiable residual evaluation for the cyclic nonlinear system."""

from __future__ import annotations

import torch

EPSILON = 0.2


def phi(
    x: torch.Tensor,
    coefficients: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Evaluate Phi(x; A, b) for a batch of system instances."""
    next_variable = torch.roll(x, shifts=-1, dims=-1)
    next_next_variable = torch.roll(x, shifts=-2, dims=-1)

    return offsets + EPSILON * (
        coefficients[..., 0] * torch.sin(next_variable)
        + coefficients[..., 1] * torch.cos(next_next_variable)
    )


def residual(
    x: torch.Tensor,
    coefficients: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Evaluate F(x; A, b) = x - Phi(x; A, b)."""
    return x - phi(x, coefficients, offsets)


def mean_squared_residual(
    x: torch.Tensor,
    coefficients: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Return the batch mean of squared Euclidean residual norms."""
    residual_values = residual(x, coefficients, offsets)
    return torch.mean(torch.sum(residual_values.square(), dim=-1))
