"""Fit fixed-rank PCA on training data and cache encoded samples."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch


def _values(raw_values: np.ndarray, solution_statistics: bool) -> np.ndarray:
    if solution_statistics:
        return np.stack(
            (raw_values.mean(axis=1), raw_values.var(axis=1)), axis=1
        ).reshape(-1, raw_values.shape[-1])
    return raw_values.reshape(-1, raw_values.shape[-1])


def _min_max(
    dataset: h5py.Dataset,
    indices: np.ndarray,
    fields_per_batch: int,
    *,
    solution_statistics: bool = False,
) -> tuple[float, float]:
    value_min, value_max = np.inf, -np.inf
    for start in range(0, indices.size, fields_per_batch):
        raw_values = np.asarray(
            dataset[indices[start : start + fields_per_batch]], dtype=np.float64
        )
        values = _values(raw_values, solution_statistics)
        value_min = min(value_min, float(values.min()))
        value_max = max(value_max, float(values.max()))
    if not np.isfinite(value_min) or not np.isfinite(value_max):
        raise ValueError("data contains non-finite values")
    if value_max <= value_min:
        raise ValueError("cannot min-max scale constant data")
    return value_min, value_max


def _fit_pca(
    dataset: h5py.Dataset,
    train_indices: np.ndarray,
    dimension: int,
    fields_per_batch: int,
    *,
    solution_statistics: bool,
    value_min: float,
    value_max: float,
) -> dict[str, np.ndarray | float]:
    n_channels = int(dataset.shape[-1])
    value_range = value_max - value_min
    count = 0
    mean = np.zeros(n_channels, dtype=np.float64)
    scatter = np.zeros((n_channels, n_channels), dtype=np.float64)
    for start in range(0, train_indices.size, fields_per_batch):
        batch_indices = train_indices[start : start + fields_per_batch]
        values = _values(
            np.asarray(dataset[batch_indices], dtype=np.float64), solution_statistics
        )
        values = (values - value_min) / value_range
        batch_count = values.shape[0]
        batch_mean = values.mean(axis=0)
        centered = values - batch_mean
        batch_scatter = centered.T @ centered
        if count == 0:
            mean, scatter, count = batch_mean, batch_scatter, batch_count
        else:
            new_count = count + batch_count
            delta = batch_mean - mean
            scatter += batch_scatter + np.outer(delta, delta) * (
                count * batch_count / new_count
            )
            mean += delta * (batch_count / new_count)
            count = new_count

    covariance = scatter / (count - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    components = eigenvectors[:, order[:dimension]].T
    ratio = float(eigenvalues[:dimension].sum() / eigenvalues.sum())

    residual_squared = data_squared = 0.0
    element_count = 0
    for start in range(0, train_indices.size, fields_per_batch):
        batch_indices = train_indices[start : start + fields_per_batch]
        values = _values(
            np.asarray(dataset[batch_indices], dtype=np.float64), solution_statistics
        )
        values = (values - value_min) / value_range
        reconstruction = ((values - mean) @ components.T) @ components + mean
        residual_squared += float(np.square(values - reconstruction).sum())
        data_squared += float(np.square(values).sum())
        element_count += values.size

    scaled_rmse = math.sqrt(residual_squared / element_count)
    return {
        "mean": mean.astype(np.float32),
        "components": components.astype(np.float32),
        "explained_variance_ratio": ratio,
        "reconstruction_relative_l2": math.sqrt(residual_squared / data_squared),
        "reconstruction_scaled_rmse": scaled_rmse,
        "reconstruction_original_rmse": scaled_rmse * value_range,
        "value_min": value_min,
        "value_max": value_max,
    }


def _indices_digest(indices: np.ndarray) -> str:
    return hashlib.sha256(indices.astype("<i8", copy=False).tobytes()).hexdigest()


def prepare_pca_data(
    raw_path: str | Path,
    cache_path: str | Path,
    state_path: str | Path,
    train_indices: Sequence[int],
    *,
    initial_dataset: str,
    solution_dataset: str,
    dimension: int,
    fields_per_batch: int,
    rebuild: bool = False,
) -> tuple[Path, dict]:
    raw_path = Path(raw_path).expanduser().resolve()
    cache_path = Path(cache_path).expanduser().resolve()
    state_path = Path(state_path).expanduser().resolve()
    train_indices = np.sort(np.asarray(train_indices, dtype=np.int64))
    with h5py.File(raw_path, "r") as raw:
        if initial_dataset not in raw or solution_dataset not in raw:
            raise KeyError("configured HDF5 datasets are missing")
        shape = tuple(raw[initial_dataset].shape)
        solution_shape = tuple(raw[solution_dataset].shape)
    if shape != solution_shape or len(shape) != 3:
        raise ValueError("raw datasets must share shape (field,sample,x)")
    if not 1 <= dimension <= shape[-1]:
        raise ValueError("PCA dimension must be between 1 and nx")
    if train_indices.size == 0 or train_indices.min() < 0 or train_indices.max() >= shape[0]:
        raise ValueError("invalid training indices")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    digest = _indices_digest(train_indices)

    if rebuild or not state_path.exists():
        with h5py.File(raw_path, "r") as raw:
            inputs, solutions = raw[initial_dataset], raw[solution_dataset]
            input_min, input_max = _min_max(inputs, train_indices, fields_per_batch)
            output_min, output_max = _min_max(
                solutions, train_indices, fields_per_batch, solution_statistics=True
            )
            input_pca = _fit_pca(
                inputs,
                train_indices,
                dimension,
                fields_per_batch,
                solution_statistics=False,
                value_min=input_min,
                value_max=input_max,
            )
            output_pca = _fit_pca(
                solutions,
                train_indices,
                dimension,
                fields_per_batch,
                solution_statistics=True,
                value_min=output_min,
                value_max=output_max,
            )
        output_range = output_max - output_min
        state = {
            "dimension": dimension,
            "input_mean": torch.from_numpy(input_pca["mean"]),
            "input_components": torch.from_numpy(input_pca["components"]),
            "output_scaled_mean": torch.from_numpy(output_pca["mean"]),
            "output_scaled_components": torch.from_numpy(output_pca["components"]),
            "input_explained_variance_ratio": input_pca["explained_variance_ratio"],
            "output_explained_variance_ratio": output_pca["explained_variance_ratio"],
            "input_reconstruction_relative_l2": input_pca["reconstruction_relative_l2"],
            "output_reconstruction_relative_l2": output_pca["reconstruction_relative_l2"],
            "input_reconstruction_original_rmse": input_pca["reconstruction_original_rmse"],
            "output_reconstruction_original_rmse": output_pca["reconstruction_original_rmse"],
            "input_data_min": input_min,
            "input_data_max": input_max,
            "output_data_min": output_min,
            "output_data_max": output_max,
            "raw_shape": shape,
            "initial_dataset": initial_dataset,
            "solution_dataset": solution_dataset,
            "train_indices_sha256": digest,
        }
        state["output_mean"] = torch.from_numpy(
            output_pca["mean"] * output_range + output_min
        )
        state["output_components"] = torch.from_numpy(
            output_pca["components"] * output_range
        )
        temporary_state = state_path.with_suffix(state_path.suffix + ".tmp")
        torch.save(state, temporary_state)
        os.replace(temporary_state, state_path)
    else:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        expected = (
            int(state["dimension"]) == dimension
            and tuple(state["raw_shape"]) == shape
            and state["initial_dataset"] == initial_dataset
            and state["solution_dataset"] == solution_dataset
            and state["train_indices_sha256"] == digest
        )
        if not expected:
            raise ValueError("existing PCA state is incompatible; rebuild the cache")

    if rebuild or not cache_path.exists():
        temporary_cache = cache_path.with_suffix(cache_path.suffix + ".tmp")
        input_mean = state["input_mean"].numpy()
        input_components = state["input_components"].numpy()
        output_mean = state["output_scaled_mean"].numpy()
        output_components = state["output_scaled_components"].numpy()
        input_min = float(state["input_data_min"])
        input_range = float(state["input_data_max"]) - input_min
        output_min = float(state["output_data_min"])
        output_range = float(state["output_data_max"]) - output_min
        with h5py.File(raw_path, "r") as raw, h5py.File(temporary_cache, "w") as target:
            input_encoded = target.create_dataset(
                "input_pca", shape=(shape[0], shape[1], dimension), dtype="f4",
                chunks=(1, shape[1], dimension)
            )
            output_encoded = target.create_dataset(
                "output_pca", shape=(shape[0], 2, dimension), dtype="f4",
                chunks=(1, 2, dimension)
            )
            for start in range(0, shape[0], fields_per_batch):
                stop = min(start + fields_per_batch, shape[0])
                inputs = np.asarray(raw[initial_dataset][start:stop], dtype=np.float32)
                inputs = inputs.reshape(-1, shape[-1])
                inputs = (inputs - input_min) / input_range
                samples = np.asarray(raw[solution_dataset][start:stop], dtype=np.float32)
                outputs = np.stack(
                    (samples.mean(axis=1, dtype=np.float64), samples.var(axis=1, dtype=np.float64)),
                    axis=1,
                ).astype(np.float32).reshape(-1, shape[-1])
                outputs = (outputs - output_min) / output_range
                input_encoded[start:stop] = (
                    (inputs - input_mean) @ input_components.T
                ).reshape(stop - start, shape[1], dimension)
                output_encoded[start:stop] = (
                    (outputs - output_mean) @ output_components.T
                ).reshape(stop - start, 2, dimension)
            target.attrs["dimension"] = dimension
            target.attrs["train_indices_sha256"] = digest
        os.replace(temporary_cache, cache_path)
    else:
        with h5py.File(cache_path, "r") as cache:
            if tuple(cache["input_pca"].shape) != (shape[0], shape[1], dimension):
                raise ValueError("existing input PCA cache is incompatible")
            if tuple(cache["output_pca"].shape) != (shape[0], 2, dimension):
                raise ValueError("existing output PCA cache is incompatible")
            if cache.attrs.get("train_indices_sha256") != digest:
                raise ValueError("existing PCA cache uses a different training split")

    print(
        f"PCA dimension={dimension}; input explained={state['input_explained_variance_ratio']:.10f}; "
        f"output explained={state['output_explained_variance_ratio']:.10f}", flush=True
    )
    return cache_path, state
