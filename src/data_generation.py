"""Data generation for the contractive cyclic nonlinear system family."""

from __future__ import annotations

import numpy as np

EPSILON = 0.2


def phi(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Evaluate the fixed-point map Phi(x; A, b)."""
    n = x.shape[-1]
    output = np.zeros_like(x)

    for i in range(n):
        j = (i + 1) % n
        k = (i + 2) % n
        output[..., i] = b[..., i] + EPSILON * (
            a[..., i, 0] * np.sin(x[..., j])
            + a[..., i, 1] * np.cos(x[..., k])
        )

    return output


def residual(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Evaluate F(x; A, b) = x - Phi(x; A, b)."""
    return x - phi(x, a, b)


def fixed_point_solution(
    a: np.ndarray,
    b: np.ndarray,
    iterations: int = 150,
) -> np.ndarray:
    """Compute the reference solution by fixed-point iteration from zero."""
    x = np.zeros_like(b, dtype=np.float64)

    for _ in range(iterations):
        x = phi(x, a, b)

    return x


def generate_dataset(
    n: int,
    num_samples: int,
    seed: int,
    iterations: int = 150,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate parameters and reference solutions for a fixed dimension.

    Returns
    -------
    parameters : ndarray, shape (num_samples, 3 * n)
        Concatenated parameter vectors [vec(A), b].
    solutions : ndarray, shape (num_samples, n)
        Reference solutions computed by fixed-point iteration.
    coefficients : ndarray, shape (num_samples, n, 2)
        Sampled coefficient matrices A.
    offsets : ndarray, shape (num_samples, n)
        Sampled offset vectors b.
    """
    rng = np.random.default_rng(seed)

    coefficients = rng.uniform(0.5, 2.0, size=(num_samples, n, 2))
    offsets = rng.uniform(-1.0, 1.0, size=(num_samples, n))

    solutions = fixed_point_solution(
        coefficients,
        offsets,
        iterations=iterations,
    )

    parameters = np.concatenate(
        [coefficients.reshape(num_samples, -1), offsets],
        axis=1,
    )

    return (
        parameters.astype(np.float32),
        solutions.astype(np.float32),
        coefficients.astype(np.float32),
        offsets.astype(np.float32),
    )
