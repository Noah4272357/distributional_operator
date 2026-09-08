#!/usr/bin/env python3
"""Summarize seeded DeepSets/MLP runs with a kernel-regression baseline."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


METRICS = {
    "NLL": "nll",
    "W_2": "w2_distance",
    "KL Divergence": "kl_divergence",
    "Hellinger Distance": "hellinger_distance",
}
SEEDED_MODELS = ("deepsets", "momentmlp")


def _read_validation_metrics(path: Path) -> dict[str, float]:
    result = json.loads(path.read_text(encoding="utf-8"))
    return result["metrics_by_split"]["val"]["observable_metrics"]


def _mean_std(values: list[float]) -> str:
    return f"{statistics.mean(values):.6f}+-{statistics.stdev(values):.6f}"


def build_rows(experiments_dir: Path) -> list[dict[str, str]]:
    """Build two seed summaries and one fixed kernel-regression result row."""
    rows: list[dict[str, str]] = []
    for model_name in SEEDED_MODELS:
        seeded_metrics = [
            _read_validation_metrics(
                experiments_dir
                / f"seed_sweep_{model_name}_seed{seed}"
                / "results.json"
            )
            for seed in range(5)
        ]
        row = {"model_name": model_name}
        for column, metric_name in METRICS.items():
            row[column] = _mean_std(
                [float(metrics[metric_name]) for metrics in seeded_metrics]
            )
        rows.append(row)

    kernel_metrics = _read_validation_metrics(
        experiments_dir / "kernel_regression_h1_all1000" / "results.json"
    )
    kernel_row = {"model_name": "kernel_regression"}
    for column, metric_name in METRICS.items():
        kernel_row[column] = f"{float(kernel_metrics[metric_name]):.6f}"
    rows.append(kernel_row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--output", type=Path, default=Path("summary.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(args.experiments_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model_name", *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    print(args.output.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
