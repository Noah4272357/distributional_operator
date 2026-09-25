"""Train-only PCA and feature standardization for paired Duffing distributions."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np


STATE_VERSION = 1


def _indices_digest(indices: np.ndarray) -> str:
    return hashlib.sha256(indices.astype("<i8", copy=False).tobytes()).hexdigest()


def _merge_statistics(
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
    new_count = count + batch_count
    delta = batch_mean - mean
    scatter += batch_scatter
    scatter += np.outer(delta, delta) * (count * batch_count / new_count)
    mean += delta * (batch_count / new_count)
    return new_count, mean, scatter


def _fit_pca(
    dataset: h5py.Dataset,
    train_indices: np.ndarray,
    dimension: int,
    fields_per_batch: int,
) -> dict[str, np.ndarray | float | int]:
    grid_size = int(dataset.shape[-1])
    count = 0
    mean = np.zeros(grid_size, dtype=np.float64)
    scatter = np.zeros((grid_size, grid_size), dtype=np.float64)
    for start in range(0, train_indices.size, fields_per_batch):
        batch_indices = train_indices[start : start + fields_per_batch]
        fields = np.asarray(dataset[batch_indices], dtype=np.float64)
        if not np.isfinite(fields).all():
            raise ValueError(
                f"{dataset.name} training fields {start}:"
                f"{start + batch_indices.size} contain NaN or infinity"
            )
        values = fields.reshape(-1, grid_size)
        count, mean, scatter = _merge_statistics(count, mean, scatter, values)

    eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    total_variance = float(eigenvalues.sum())
    if total_variance <= 0 or not np.isfinite(total_variance):
        raise ValueError(f"{dataset.name} training variance must be positive and finite")
    components = eigenvectors[:, order[:dimension]].T
    explained_ratio = float(eigenvalues[:dimension].sum() / total_variance)
    return {
        "mean": mean.astype(np.float32),
        "components": components.astype(np.float32),
        "explained_variance_ratio": explained_ratio,
        "observations": count,
    }


def transform_pca(
    values: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    coefficient_mean: np.ndarray | None = None,
    coefficient_scale: np.ndarray | None = None,
) -> np.ndarray:
    """Project the last axis, optionally standardizing with training statistics."""
    coefficients = (values - mean) @ components.T
    coefficient_mean, coefficient_scale = _normalization_parameters(
        coefficients.shape[-1], coefficient_mean, coefficient_scale
    )
    if coefficient_mean is not None:
        coefficients = (coefficients - coefficient_mean) / coefficient_scale
    return coefficients.astype(np.float32, copy=False)


def inverse_pca(
    coefficients: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    coefficient_mean: np.ndarray | None = None,
    coefficient_scale: np.ndarray | None = None,
) -> np.ndarray:
    """Undo coefficient standardization and map back to the original grid."""
    coefficient_mean, coefficient_scale = _normalization_parameters(
        coefficients.shape[-1], coefficient_mean, coefficient_scale
    )
    if coefficient_mean is not None:
        coefficients = coefficients * coefficient_scale + coefficient_mean
    return (coefficients @ components + mean).astype(np.float32, copy=False)


def _normalization_parameters(dimension, mean, scale):
    if (mean is None) != (scale is None):
        raise ValueError("coefficient_mean and coefficient_scale must be supplied together")
    if mean is None:
        return None, None
    mean, scale = np.asarray(mean), np.asarray(scale)
    if mean.shape != (dimension,) or scale.shape != (dimension,):
        raise ValueError("normalization vectors must match the last PCA dimension")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("normalization means must be finite and scales positive/finite")
    return mean, scale


def _prepare_normalization(state, input_path, target_path, train_indices, batch, epsilon):
    """Load or fit a sidecar using only training trajectories in PCA space."""
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("normalization_epsilon must be positive and finite")
    sidecar = Path(state["state_path"]).with_suffix(".normalization.npz")
    basis_hash = hashlib.sha256()
    for key in ("input_mean", "input_components", "target_mean", "target_components"):
        basis_hash.update(np.ascontiguousarray(state[key]).tobytes())
    digest = basis_hash.hexdigest()
    if sidecar.exists():
        normalization = _load_state(sidecar)
        if (
            int(_scalar(normalization, "version")) != 1
            or str(_scalar(normalization, "train_indices_sha256")) != _indices_digest(train_indices)
            or str(_scalar(normalization, "pca_basis_sha256")) != digest
            or float(_scalar(normalization, "epsilon")) != epsilon
        ):
            raise ValueError("normalization sidecar does not match the training split/PCA/epsilon")
        action = "loaded"
    else:
        normalization = {
            "version": np.int64(1), "epsilon": np.float64(epsilon),
            "train_indices_sha256": np.asarray(_indices_digest(train_indices)),
            "pca_basis_sha256": np.asarray(digest),
        }
        with h5py.File(input_path, "r") as xf, h5py.File(target_path, "r") as yf:
            for prefix, handle in (("input", xf), ("target", yf)):
                dimension = int(_scalar(state, f"{prefix}_dimension"))
                count = 0
                mean = np.zeros(dimension, dtype=np.float64)
                scatter = np.zeros((dimension, dimension), dtype=np.float64)
                for start in range(0, train_indices.size, batch):
                    raw = np.asarray(handle[str(_scalar(state, f"{prefix}_dataset"))][
                        train_indices[start:start + batch]
                    ], dtype=np.float32)
                    values = transform_pca(raw, state[f"{prefix}_mean"], state[f"{prefix}_components"])
                    if not np.isfinite(values).all():
                        raise ValueError("nonfinite training PCA coefficients")
                    count, mean, scatter = _merge_statistics(
                        count, mean, scatter, values.reshape(-1, dimension).astype(np.float64)
                    )
                # Population std (ddof=0) gives unit training variance.
                std = np.sqrt(np.maximum(np.diag(scatter) / count, 0))
                normalization[f"{prefix}_coefficient_mean"] = mean.astype(np.float32)
                normalization[f"{prefix}_coefficient_scale"] = np.where(std > epsilon, std, 1).astype(np.float32)
        _save_state(sidecar, normalization)
        action = "fitted and saved"
    for prefix in ("input", "target"):
        mean, scale = _normalization_parameters(
            int(_scalar(state, f"{prefix}_dimension")),
            normalization[f"{prefix}_coefficient_mean"],
            normalization[f"{prefix}_coefficient_scale"],
        )
        state[f"{prefix}_coefficient_mean"] = mean
        state[f"{prefix}_coefficient_scale"] = scale
    state["normalization_state_path"] = str(sidecar)
    print(f"{action} training-only PCA normalization: {sidecar}", flush=True)


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("wb") as handle:
        np.savez_compressed(handle, **state)
    os.replace(temporary_path, path)


def _load_state(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name] for name in stored.files}


def _scalar(state: dict, name: str):
    if name not in state:
        raise ValueError(f"PCA state is missing {name}")
    return np.asarray(state[name]).item()


def prepare_duffing_pca(
    input_path: str | Path,
    target_path: str | Path,
    state_path: str | Path,
    train_indices: Sequence[int],
    *,
    input_dataset: str = "X",
    target_dataset: str = "Y",
    input_dimension: int = 24,
    target_dimension: int = 13,
    fields_per_batch: int = 64,
    rebuild: bool = False,
    normalize: bool = False,
    normalization_epsilon: float = 1.0e-8,
) -> dict:
    """Fit/load training PCA and optionally training coefficient normalization.

    Normalization uses one mean/std per retained feature over all training
    fields and trajectories (not per field). Set ``normalize=True`` to fit or
    load a separate sidecar. Validation/test must reuse the returned training
    state, never supply their own indices to fit a new transform. Near-constant
    coordinates use scale=1 to avoid division by zero. Legacy callers default
    to unnormalized PCA coordinates.
    """
    input_path = Path(input_path).expanduser().resolve()
    target_path = Path(target_path).expanduser().resolve()
    state_path = Path(state_path).expanduser().resolve()
    train_indices_array = np.sort(np.asarray(train_indices, dtype=np.int64))
    if train_indices_array.ndim != 1 or train_indices_array.size < 1:
        raise ValueError("train_indices must be a non-empty one-dimensional sequence")
    if np.unique(train_indices_array).size != train_indices_array.size:
        raise ValueError("train_indices must not contain duplicates")
    if fields_per_batch < 1:
        raise ValueError("fields_per_batch must be positive")

    with h5py.File(input_path, "r") as input_file, h5py.File(
        target_path, "r"
    ) as target_file:
        if input_dataset not in input_file or target_dataset not in target_file:
            raise KeyError("configured PCA datasets are missing")
        input_shape = tuple(input_file[input_dataset].shape)
        target_shape = tuple(target_file[target_dataset].shape)
    if input_shape != target_shape or len(input_shape) != 3:
        raise ValueError("PCA datasets must share (fields, samples, grid) shape")
    if train_indices_array.min() < 0 or train_indices_array.max() >= input_shape[0]:
        raise ValueError("training indices exceed the PCA datasets")
    if not 1 <= input_dimension <= input_shape[-1]:
        raise ValueError("input PCA dimension is outside the grid dimension")
    if not 1 <= target_dimension <= target_shape[-1]:
        raise ValueError("target PCA dimension is outside the grid dimension")

    digest = _indices_digest(train_indices_array)
    if rebuild or not state_path.exists():
        with h5py.File(input_path, "r") as input_file, h5py.File(
            target_path, "r"
        ) as target_file:
            input_pca = _fit_pca(
                input_file[input_dataset],
                train_indices_array,
                input_dimension,
                fields_per_batch,
            )
            target_pca = _fit_pca(
                target_file[target_dataset],
                train_indices_array,
                target_dimension,
                fields_per_batch,
            )
        state = {
            "version": np.int64(STATE_VERSION),
            "input_mean": input_pca["mean"],
            "input_components": input_pca["components"],
            "target_mean": target_pca["mean"],
            "target_components": target_pca["components"],
            "input_explained_variance_ratio": np.float64(
                input_pca["explained_variance_ratio"]
            ),
            "target_explained_variance_ratio": np.float64(
                target_pca["explained_variance_ratio"]
            ),
            "training_observations": np.int64(input_pca["observations"]),
            "raw_shape": np.asarray(input_shape, dtype=np.int64),
            "input_dataset": np.asarray(input_dataset),
            "target_dataset": np.asarray(target_dataset),
            "input_dimension": np.int64(input_dimension),
            "target_dimension": np.int64(target_dimension),
            "train_indices_sha256": np.asarray(digest),
        }
        _save_state(state_path, state)
        action = "fitted and saved"
    else:
        state = _load_state(state_path)
        compatible = (
            int(_scalar(state, "version")) == STATE_VERSION
            and tuple(np.asarray(state["raw_shape"]).tolist()) == input_shape
            and str(_scalar(state, "input_dataset")) == input_dataset
            and str(_scalar(state, "target_dataset")) == target_dataset
            and int(_scalar(state, "input_dimension")) == input_dimension
            and int(_scalar(state, "target_dimension")) == target_dimension
            and str(_scalar(state, "train_indices_sha256")) == digest
        )
        if not compatible:
            raise ValueError(
                "saved Duffing PCA state does not match this training split or "
                "configuration; choose another state_path or enable rebuild"
            )
        action = "loaded"

    state["state_path"] = str(state_path)
    state["train_indices"] = train_indices_array
    state["normalization_enabled"] = bool(normalize)
    if normalize:
        _prepare_normalization(
            state, input_path, target_path, train_indices_array,
            fields_per_batch, normalization_epsilon,
        )
    print(
        f"{action} PCA state: {state_path}; training fields="
        f"{train_indices_array.size}; observations="
        f"{int(_scalar(state, 'training_observations'))}; "
        f"X dim={input_dimension} explained="
        f"{float(_scalar(state, 'input_explained_variance_ratio')):.10f}; "
        f"Y dim={target_dimension} explained="
        f"{float(_scalar(state, 'target_explained_variance_ratio')):.10f}",
        flush=True,
    )
    return state
