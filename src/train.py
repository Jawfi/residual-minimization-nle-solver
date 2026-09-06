"""Training utilities for residual-only and supervised FNN objectives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data_generation import generate_dataset
from .model import ResidualMinimizationFNN
from .residuals import mean_squared_residual, residual
from .utils import relative_solution_error, set_seed


@dataclass
class ExperimentConfig:
    """Configuration matching the reported experimental protocol."""

    dimension: int
    seed: int
    objective: str
    num_samples: int = 3000
    fixed_point_iterations: int = 150
    batch_size: int = 64
    max_epochs: int = 300
    learning_rate: float = 1e-3
    lr_patience: int = 5
    lr_factor: float = 0.5
    early_stopping_patience: int = 25
    minimum_improvement: float = 1e-7


def split_and_standardize(
    parameters: np.ndarray,
    solutions: np.ndarray,
    coefficients: np.ndarray,
    offsets: np.ndarray,
    seed: int,
) -> dict[str, tuple[torch.Tensor, ...]]:
    """Create a 70/15/15 split and standardize inputs from train statistics."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(parameters))

    train_end = int(0.70 * len(indices))
    validation_end = int(0.85 * len(indices))

    split_indices = {
        "train": indices[:train_end],
        "validation": indices[train_end:validation_end],
        "test": indices[validation_end:],
    }

    train_parameters = parameters[split_indices["train"]]
    mean = train_parameters.mean(axis=0, keepdims=True)
    standard_deviation = train_parameters.std(axis=0, keepdims=True)
    standard_deviation = np.where(standard_deviation == 0.0, 1.0, standard_deviation)

    splits: dict[str, tuple[torch.Tensor, ...]] = {}
    for split_name, split_index in split_indices.items():
        standardized_parameters = (
            parameters[split_index] - mean
        ) / standard_deviation

        splits[split_name] = (
            torch.from_numpy(standardized_parameters.astype(np.float32)),
            torch.from_numpy(solutions[split_index].astype(np.float32)),
            torch.from_numpy(coefficients[split_index].astype(np.float32)),
            torch.from_numpy(offsets[split_index].astype(np.float32)),
        )

    return splits


def evaluate(
    model: nn.Module,
    dataset: tuple[torch.Tensor, ...],
    device: torch.device,
) -> dict[str, float]:
    """Evaluate residual norm, squared residual, and relative solution error."""
    parameters, solutions, coefficients, offsets = (
        value.to(device) for value in dataset
    )

    model.eval()
    with torch.no_grad():
        predictions = model(parameters)
        residual_values = residual(predictions, coefficients, offsets)

        residual_norm = torch.linalg.vector_norm(residual_values, dim=-1).mean()
        residual_squared = torch.sum(residual_values.square(), dim=-1).mean()
        relative_error = relative_solution_error(predictions, solutions).mean()

    return {
        "residual_norm": float(residual_norm.cpu()),
        "residual_squared": float(residual_squared.cpu()),
        "relative_error": float(relative_error.cpu()),
    }


def train_experiment(config: ExperimentConfig) -> tuple[nn.Module, dict[str, float], dict[str, list[float]]]:
    """Train one model and return the retained model, test metrics, and history."""
    if config.objective not in {"residual", "supervised"}:
        raise ValueError("objective must be 'residual' or 'supervised'.")

    set_seed(config.seed)
    device = torch.device("cpu")

    parameters, solutions, coefficients, offsets = generate_dataset(
        n=config.dimension,
        num_samples=config.num_samples,
        seed=config.seed,
        iterations=config.fixed_point_iterations,
    )
    splits = split_and_standardize(
        parameters,
        solutions,
        coefficients,
        offsets,
        seed=config.seed,
    )

    train_loader = DataLoader(
        TensorDataset(*splits["train"]),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )

    model = ResidualMinimizationFNN(
        input_dim=3 * config.dimension,
        output_dim=config.dimension,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_factor,
        patience=config.lr_patience,
    )

    best_validation_residual = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history = {"training_residual_squared": [], "validation_residual_squared": []}

    for _ in range(config.max_epochs):
        model.train()
        batch_losses = []

        for batch in train_loader:
            batch_parameters, batch_solutions, batch_coefficients, batch_offsets = (
                value.to(device) for value in batch
            )

            predictions = model(batch_parameters)
            if config.objective == "residual":
                loss = mean_squared_residual(
                    predictions,
                    batch_coefficients,
                    batch_offsets,
                )
            else:
                loss = torch.sum((predictions - batch_solutions).square(), dim=-1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        training_metrics = evaluate(model, splits["train"], device)
        validation_metrics = evaluate(model, splits["validation"], device)

        history["training_residual_squared"].append(
            training_metrics["residual_squared"]
        )
        history["validation_residual_squared"].append(
            validation_metrics["residual_squared"]
        )

        validation_residual = validation_metrics["residual_squared"]
        scheduler.step(validation_residual)

        if best_validation_residual - validation_residual > config.minimum_improvement:
            best_validation_residual = validation_residual
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.early_stopping_patience:
            break

    if best_state is None:
        raise RuntimeError("No model state was retained during training.")

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, splits["test"], device)

    return model, test_metrics, history
