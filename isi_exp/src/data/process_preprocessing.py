"""Training-only PCA preprocessing for stochastic-process paths."""

from __future__ import annotations

from typing import Any

import h5py
import torch


PCAState = dict[str, Any]


def _read_law_chunk(dataset: h5py.Dataset, row_indices: torch.Tensor) -> torch.Tensor:
    """Read arbitrary HDF5 rows and restore their requested order."""
    indices = row_indices.detach().cpu().to(torch.long)
    order = torch.argsort(indices)
    sorted_indices = indices[order].numpy()
    sorted_values = torch.from_numpy(dataset[sorted_indices, ...])
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel())
    return sorted_values[inverse]


def fit_process_pca(
    process_samples: h5py.Dataset,
    train_row_indices: torch.Tensor,
    *,
    explained_variance_threshold: float,
    truncate_dim: str | int,
    chunk_laws: int,
    enforce_minimum_dim: bool,
) -> PCAState:
    """Fit covariance PCA from training paths without loading the full array."""
    if process_samples.ndim != 3:
        raise ValueError("process_samples must have shape [law, path, time]")
    if not 0.0 < explained_variance_threshold < 1.0:
        raise ValueError("explained_variance_threshold must be in (0, 1)")
    if chunk_laws < 1:
        raise ValueError("chunk_laws must be positive")
    if train_row_indices.numel() < 1:
        raise ValueError("PCA requires at least one training law")

    time_dim = int(process_samples.shape[-1])
    total = torch.zeros(time_dim, dtype=torch.float64)
    second_moment = torch.zeros((time_dim, time_dim), dtype=torch.float64)
    path_count = 0
    for start in range(0, train_row_indices.numel(), chunk_laws):
        rows = train_row_indices[start : start + chunk_laws]
        paths = _read_law_chunk(process_samples, rows).reshape(-1, time_dim).to(torch.float64)
        if not torch.isfinite(paths).all():
            raise ValueError("process_samples contains non-finite training values")
        total += paths.sum(dim=0)
        second_moment += paths.T @ paths
        path_count += int(paths.shape[0])
    if path_count < 2:
        raise ValueError("PCA requires at least two training paths")

    mean = total / path_count
    covariance = (second_moment - path_count * torch.outer(mean, mean)) / (path_count - 1)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0.0)
    eigenvectors = eigenvectors[:, order]
    variance_total = eigenvalues.sum()
    if variance_total <= 0.0:
        raise ValueError("training process paths have zero total variance")
    explained_ratio = eigenvalues / variance_total
    cumulative_ratio = explained_ratio.cumsum(dim=0)
    qualifying = torch.nonzero(cumulative_ratio > explained_variance_threshold, as_tuple=True)[0]
    if qualifying.numel() == 0:
        raise RuntimeError("no PCA dimension exceeds the explained-variance threshold")
    minimum_dim = int(qualifying[0]) + 1

    if str(truncate_dim).lower() == "auto":
        selected_dim = minimum_dim
    else:
        selected_dim = int(truncate_dim)
        if not 1 <= selected_dim <= time_dim:
            raise ValueError(f"truncate_dim must be between 1 and {time_dim}")
        if enforce_minimum_dim and selected_dim != minimum_dim:
            raise ValueError(
                f"configured truncate_dim={selected_dim}, but the minimum dimension above "
                f"{explained_variance_threshold} is {minimum_dim}"
            )

    return {
        "format": "process_pca_v1",
        "mean": mean.to(torch.float32),
        "components": eigenvectors[:, :selected_dim].T.contiguous().to(torch.float32),
        "explained_variance": eigenvalues.to(torch.float32),
        "explained_variance_ratio": explained_ratio.to(torch.float32),
        "cumulative_explained_variance_ratio": cumulative_ratio.to(torch.float32),
        "selected_dim": selected_dim,
        "minimum_dim": minimum_dim,
        "explained_variance_threshold": float(explained_variance_threshold),
        "achieved_explained_variance_ratio": float(cumulative_ratio[selected_dim - 1]),
        "training_path_count": path_count,
    }


def validate_pca_state(state: PCAState, time_dim: int) -> None:
    if state.get("format") != "process_pca_v1":
        raise ValueError("unsupported or missing process PCA state format")
    mean = state.get("mean")
    components = state.get("components")
    if not torch.is_tensor(mean) or tuple(mean.shape) != (time_dim,):
        raise ValueError(f"PCA mean must have shape ({time_dim},)")
    if not torch.is_tensor(components) or components.ndim != 2 or components.shape[1] != time_dim:
        raise ValueError(f"PCA components must have shape [truncate_dim, {time_dim}]")
    if int(state.get("selected_dim", -1)) != int(components.shape[0]):
        raise ValueError("PCA selected_dim does not match its component matrix")
    if not torch.isfinite(mean).all() or not torch.isfinite(components).all():
        raise ValueError("PCA state contains non-finite values")


def transform_process_rows(
    process_samples: h5py.Dataset,
    row_indices: torch.Tensor,
    state: PCAState,
    *,
    chunk_laws: int,
) -> torch.Tensor:
    """Transform selected laws with an already-fitted PCA state."""
    time_dim = int(process_samples.shape[-1])
    sample_size = int(process_samples.shape[1])
    validate_pca_state(state, time_dim)
    mean = state["mean"].to(torch.float32)
    components = state["components"].to(torch.float32)
    result = torch.empty(
        (row_indices.numel(), sample_size, components.shape[0]),
        dtype=torch.float32,
    )
    for start in range(0, row_indices.numel(), chunk_laws):
        stop = min(start + chunk_laws, row_indices.numel())
        paths = _read_law_chunk(process_samples, row_indices[start:stop]).to(torch.float32)
        if not torch.isfinite(paths).all():
            raise ValueError("process_samples contains non-finite values")
        result[start:stop] = (paths - mean) @ components.T
    return result
