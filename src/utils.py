"""Utilities for reproducible experiments and evaluation."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def relative_solution_error(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Compute per-sample relative Euclidean solution errors."""
    numerator = torch.linalg.vector_norm(prediction - reference, dim=-1)
    denominator = torch.linalg.vector_norm(reference, dim=-1) + epsilon
    return numerator / denominator
