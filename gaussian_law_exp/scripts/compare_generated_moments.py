#!/usr/bin/env python3
"""Compare empirical output-distribution moments with their target parameters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _error_summary(error: np.ndarray, target: np.ndarray) -> dict[str, float]:
    flat_error = error.reshape(error.shape[0], -1)
    flat_target = target.reshape(target.shape[0], -1)
    absolute_norm = np.linalg.norm(flat_error, axis=1)
    target_norm = np.linalg.norm(flat_target, axis=1)
    relative_norm = absolute_norm / np.maximum(target_norm, 1.0e-12)
    return {
        "element_mae": float(np.mean(np.abs(error))),
        "element_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "absolute_norm_mean": float(np.mean(absolute_norm)),
        "absolute_norm_median": float(np.median(absolute_norm)),
        "absolute_norm_p95": float(np.quantile(absolute_norm, 0.95)),
        "absolute_norm_max": float(np.max(absolute_norm)),
        "relative_norm_mean": float(np.mean(relative_norm)),
        "relative_norm_median": float(np.median(relative_norm)),
        "relative_norm_p95": float(np.quantile(relative_norm, 0.95)),
        "relative_norm_max": float(np.max(relative_norm)),
    }


def compare_moments(dataset_path: Path) -> dict[str, Any]:
    """Calculate sample moments and compare them with stored population targets."""
    with h5py.File(dataset_path, "r") as handle:
        output_dist = handle["output_dist"][...].astype(np.float64)
        target_mean = handle["target_mean"][...].astype(np.float64)
        target_cov = handle["target_cov"][...].astype(np.float64)

    data_size, sample_size, dimension = output_dist.shape
    if sample_size < 2:
        raise ValueError("sample_size must be at least two to estimate covariance")

    empirical_mean = np.mean(output_dist, axis=1)
    centered = output_dist - empirical_mean[:, None, :]
    empirical_cov = np.einsum("nsi,nsj->nij", centered, centered)
    empirical_cov /= sample_size - 1

    mean_error = empirical_mean - target_mean
    covariance_error = empirical_cov - target_cov
    result = {
        "dataset": str(dataset_path.resolve()),
        "data_size": int(data_size),
        "sample_size": int(sample_size),
        "dimension": int(dimension),
        "covariance_estimator": "unbiased (denominator sample_size - 1)",
        "mean_error": _error_summary(mean_error, target_mean),
        "covariance_error": _error_summary(covariance_error, target_cov),
        "aggregate_bias": {
            "mean": np.mean(mean_error, axis=0).tolist(),
            "covariance": np.mean(covariance_error, axis=0).tolist(),
        },
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/dataset.h5"),
        help="generated HDF5 dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/moment_covariance_comparison.json"),
        help="JSON output path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compare_moments(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved comparison to {args.output}")


if __name__ == "__main__":
    main()
