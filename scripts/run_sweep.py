"""Run the full experiment sweep reported in the manuscript."""

from __future__ import annotations

import csv
from pathlib import Path

from src.train import ExperimentConfig, train_experiment


DIMENSIONS = (2, 5, 10, 20)
OBJECTIVES = ("residual", "supervised")
SEEDS = (0, 1, 2, 3, 4)


def main() -> None:
    """Train all configurations and save run-level test metrics."""
    output_directory = Path("results")
    output_directory.mkdir(exist_ok=True)
    output_file = output_directory / "run_level_results.csv"

    fieldnames = [
        "dimension",
        "objective",
        "seed",
        "residual_norm",
        "relative_solution_error",
    ]

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for dimension in DIMENSIONS:
            for objective in OBJECTIVES:
                for seed in SEEDS:
                    config = ExperimentConfig(
                        dimension=dimension,
                        objective=objective,
                        seed=seed,
                    )
                    _, test_metrics, _ = train_experiment(config)

                    writer.writerow(
                        {
                            "dimension": dimension,
                            "objective": objective,
                            "seed": seed,
                            "residual_norm": test_metrics["residual_norm"],
                            "relative_solution_error": (
                                100.0 * test_metrics["relative_error"]
                            ),
                        }
                    )

                    print(
                        f"n={dimension}, objective={objective}, seed={seed}: "
                        f"residual={test_metrics['residual_norm']:.6f}, "
                        f"relative error="
                        f"{100.0 * test_metrics['relative_error']:.2f}%"
                    )


if __name__ == "__main__":
    main()
