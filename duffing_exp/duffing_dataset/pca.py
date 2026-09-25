#!/usr/bin/env python3
"""Compute global PCA truncation dimensions for Duffing X and Y.

Each dataset is flattened from ``(data_size, sample_size, grid_size)`` to
``(data_size * sample_size, grid_size)``. PCA is then applied over the grid
features, and the smallest dimension with cumulative explained variance
strictly greater than the requested ratio is reported.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass(frozen=True)
class PCAResult:
    dimension: int
    achieved_ratio: float
    previous_ratio: float
    observations: int
    grid_size: int


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir / "duffing_dataset.h5",
        help="Input HDF5 file (default: duffing_dataset.h5 beside this script).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "pca_dim.csv",
        help="Output CSV file (default: pca_dim.csv beside this script).",
    )
    parser.add_argument(
        "--explained-ratio",
        type=float,
        default=0.99,
        help="Required cumulative explained-variance ratio (default: 0.99).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of data fields read per batch (default: 64).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Scatter-matrix device; auto uses CUDA when available.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float64",
        help="Accumulation and eigendecomposition precision (default: float64).",
    )
    return parser.parse_args()


def _select_backend(device: str, dtype: str) -> tuple[str, Any, Any]:
    if device == "cpu":
        return "cpu", np, np.dtype(dtype)
    try:
        import torch
    except ImportError:
        if device == "cuda":
            raise RuntimeError("--device cuda requires PyTorch with CUDA support")
        return "cpu", np, np.dtype(dtype)
    if not torch.cuda.is_available():
        if device == "cuda":
            raise RuntimeError("CUDA was requested but is not available")
        return "cpu", np, np.dtype(dtype)
    torch_dtype = torch.float64 if dtype == "float64" else torch.float32
    return "cuda", torch, torch_dtype


def _validate_dataset(dataset: h5py.Dataset) -> tuple[int, int, int]:
    if dataset.ndim != 3:
        raise ValueError(
            f"{dataset.name} must have shape (data_size, sample_size, grid_size)"
        )
    data_size, sample_size, grid_size = map(int, dataset.shape)
    if min(data_size, sample_size, grid_size) < 1:
        raise ValueError(f"{dataset.name} dimensions must all be positive")
    if data_size * sample_size < 2 or grid_size < 2:
        raise ValueError(f"{dataset.name} is too small for PCA")
    return data_size, sample_size, grid_size


def _merge_numpy_statistics(
    count: int,
    mean: np.ndarray,
    scatter: np.ndarray,
    values: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    batch_count = int(values.shape[0])
    batch_mean = values.mean(axis=0)
    centered = values - batch_mean
    batch_scatter = centered.T @ centered
    if count == 0:
        return batch_count, batch_mean, batch_scatter
    total_count = count + batch_count
    delta = batch_mean - mean
    scatter += batch_scatter
    scatter += np.outer(delta, delta) * (count * batch_count / total_count)
    mean += delta * (batch_count / total_count)
    return total_count, mean, scatter


def _global_scatter_numpy(
    dataset: h5py.Dataset,
    batch_size: int,
    dtype: np.dtype,
) -> tuple[np.ndarray, int, int]:
    data_size, sample_size, grid_size = _validate_dataset(dataset)
    count = 0
    mean = np.zeros(grid_size, dtype=dtype)
    scatter = np.zeros((grid_size, grid_size), dtype=dtype)
    for start in range(0, data_size, batch_size):
        stop = min(start + batch_size, data_size)
        fields = np.asarray(dataset[start:stop], dtype=dtype)
        if not np.isfinite(fields).all():
            raise ValueError(f"{dataset.name}[{start}:{stop}] contains NaN or infinity")
        values = fields.reshape(-1, grid_size)
        count, mean, scatter = _merge_numpy_statistics(
            count, mean, scatter, values
        )
        print(
            f"{dataset.name}: processed {stop}/{data_size} fields "
            f"({count}/{data_size * sample_size} flattened observations)",
            flush=True,
        )
    return scatter, count, grid_size


def _merge_torch_statistics(
    count: int,
    mean: Any,
    scatter: Any,
    values: Any,
) -> tuple[int, Any, Any]:
    batch_count = int(values.shape[0])
    batch_mean = values.mean(dim=0)
    centered = values - batch_mean
    batch_scatter = centered.T @ centered
    if count == 0:
        return batch_count, batch_mean, batch_scatter
    total_count = count + batch_count
    delta = batch_mean - mean
    scatter.add_(batch_scatter)
    scatter.add_(torch_outer(delta) * (count * batch_count / total_count))
    mean.add_(delta * (batch_count / total_count))
    return total_count, mean, scatter


def torch_outer(values: Any) -> Any:
    """Return an outer product without importing PyTorch at module import time."""
    return values[:, None] * values[None, :]


def _global_scatter_cuda(
    dataset: h5py.Dataset,
    batch_size: int,
    torch: Any,
    dtype: Any,
) -> tuple[np.ndarray, int, int]:
    data_size, sample_size, grid_size = _validate_dataset(dataset)
    count = 0
    mean = torch.zeros(grid_size, device="cuda", dtype=dtype)
    scatter = torch.zeros((grid_size, grid_size), device="cuda", dtype=dtype)
    for start in range(0, data_size, batch_size):
        stop = min(start + batch_size, data_size)
        fields = np.asarray(dataset[start:stop])
        if not np.isfinite(fields).all():
            raise ValueError(f"{dataset.name}[{start}:{stop}] contains NaN or infinity")
        values = torch.as_tensor(fields, device="cuda", dtype=dtype).reshape(
            -1, grid_size
        )
        count, mean, scatter = _merge_torch_statistics(
            count, mean, scatter, values
        )
        print(
            f"{dataset.name}: processed {stop}/{data_size} fields "
            f"({count}/{data_size * sample_size} flattened observations)",
            flush=True,
        )
    eigenvalues = torch.linalg.eigvalsh(scatter).flip(dims=(-1,)).clamp_min_(0)
    return eigenvalues.cpu().numpy(), count, grid_size


def _select_dimension(
    descending_eigenvalues: np.ndarray,
    explained_ratio: float,
    observations: int,
    grid_size: int,
) -> PCAResult:
    eigenvalues = np.maximum(
        np.asarray(descending_eigenvalues, dtype=np.float64), 0.0
    )
    total_variance = float(eigenvalues.sum())
    if not np.isfinite(total_variance) or total_variance <= 0:
        raise ValueError("PCA total variance must be finite and positive")
    cumulative_ratio = np.cumsum(eigenvalues) / total_variance
    dimension = int(np.count_nonzero(cumulative_ratio <= explained_ratio) + 1)
    dimension = min(dimension, grid_size)
    achieved_ratio = float(cumulative_ratio[dimension - 1])
    previous_ratio = 0.0 if dimension == 1 else float(cumulative_ratio[dimension - 2])
    if not achieved_ratio > explained_ratio:
        raise RuntimeError("failed to find a PCA dimension above the target ratio")
    if dimension > 1 and previous_ratio > explained_ratio:
        raise RuntimeError("selected PCA dimension is not minimal")
    return PCAResult(
        dimension=dimension,
        achieved_ratio=achieved_ratio,
        previous_ratio=previous_ratio,
        observations=observations,
        grid_size=grid_size,
    )


def global_pca_dimension(
    dataset: h5py.Dataset,
    *,
    explained_ratio: float,
    batch_size: int,
    backend: str,
    module: Any,
    dtype: Any,
) -> PCAResult:
    if backend == "cuda":
        eigenvalues, observations, grid_size = _global_scatter_cuda(
            dataset, batch_size, module, dtype
        )
    else:
        scatter, observations, grid_size = _global_scatter_numpy(
            dataset, batch_size, dtype
        )
        eigenvalues = np.maximum(np.linalg.eigvalsh(scatter)[::-1], 0.0)
    return _select_dimension(
        eigenvalues, explained_ratio, observations, grid_size
    )


def write_dimensions(output: Path, x_dimension: int, y_dimension: int) -> None:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.tmp")
    with temporary_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("pca_x_dim", "pca_y_dim"))
        writer.writerow((x_dimension, y_dimension))
    temporary_output.replace(output)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.explained_ratio < 1.0:
        raise ValueError("--explained-ratio must lie strictly between zero and one")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    input_path = args.input.expanduser().resolve()
    backend, module, dtype = _select_backend(args.device, args.dtype)
    print(
        f"input={input_path}; backend={backend}; dtype={args.dtype}; "
        f"required explained ratio > {args.explained_ratio}",
        flush=True,
    )
    with h5py.File(input_path, "r") as source:
        if "X" not in source or "Y" not in source:
            raise KeyError("input HDF5 file must contain X and Y datasets")
        if source["X"].shape != source["Y"].shape:
            raise ValueError("X and Y must have identical shapes")
        x_result = global_pca_dimension(
            source["X"],
            explained_ratio=args.explained_ratio,
            batch_size=args.batch_size,
            backend=backend,
            module=module,
            dtype=dtype,
        )
        y_result = global_pca_dimension(
            source["Y"],
            explained_ratio=args.explained_ratio,
            batch_size=args.batch_size,
            backend=backend,
            module=module,
            dtype=dtype,
        )

    write_dimensions(args.output, x_result.dimension, y_result.dimension)
    print(
        f"saved one row to {args.output.expanduser().resolve()}\n"
        f"flattened shape: ({x_result.observations}, {x_result.grid_size})\n"
        f"X dimension={x_result.dimension}; previous ratio="
        f"{x_result.previous_ratio:.10f}; achieved ratio="
        f"{x_result.achieved_ratio:.10f}\n"
        f"Y dimension={y_result.dimension}; previous ratio="
        f"{y_result.previous_ratio:.10f}; achieved ratio="
        f"{y_result.achieved_ratio:.10f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
