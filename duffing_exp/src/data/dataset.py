"""Lazy HDF5 dataset for PCA-encoded Burgers random fields."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class RandomFieldDataset(Dataset[tuple[Tensor, Tensor]]):
    """Return encoded inputs and physical solution mean/population variance."""

    def __init__(
        self,
        raw_path: str | Path,
        pca_cache_path: str | Path,
        indices: Sequence[int],
        *,
        initial_dataset: str,
        solution_dataset: str,
    ) -> None:
        self.raw_path = str(Path(raw_path).expanduser().resolve())
        self.pca_cache_path = str(Path(pca_cache_path).expanduser().resolve())
        self.indices = np.asarray(indices, dtype=np.int64)
        self.initial_dataset = initial_dataset
        self.solution_dataset = solution_dataset
        self._raw_handle: h5py.File | None = None
        self._cache_handle: h5py.File | None = None

        with h5py.File(self.raw_path, "r") as raw, h5py.File(
            self.pca_cache_path, "r"
        ) as cache:
            initial_shape = tuple(raw[self.initial_dataset].shape)
            solution_shape = tuple(raw[self.solution_dataset].shape)
            cache_input_shape = tuple(cache["input_pca"].shape)
            cache_output_shape = tuple(cache["output_pca"].shape)
        if initial_shape != solution_shape or len(initial_shape) != 3:
            raise ValueError("initial and solution data must share (field,sample,x)")
        if cache_input_shape[:2] != initial_shape[:2]:
            raise ValueError("input PCA cache is incompatible with raw data")
        if cache_output_shape[:2] != (solution_shape[0], 2):
            raise ValueError("output PCA cache is incompatible with raw data")
        if self.indices.size and (
            self.indices.min() < 0 or self.indices.max() >= initial_shape[0]
        ):
            raise IndexError("dataset indices are out of range")
        self.raw_shape = initial_shape
        self.pca_dim_in = cache_input_shape[-1]
        self.pca_dim_out = cache_output_shape[-1]

    def __len__(self) -> int:
        return int(self.indices.size)

    def _files(self) -> tuple[h5py.File, h5py.File]:
        if self._raw_handle is None:
            self._raw_handle = h5py.File(self.raw_path, "r")
        if self._cache_handle is None:
            self._cache_handle = h5py.File(self.pca_cache_path, "r")
        return self._raw_handle, self._cache_handle

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        field_index = int(self.indices[index])
        raw, cache = self._files()
        encoded_input = np.asarray(cache["input_pca"][field_index], dtype=np.float32)
        solution_samples = np.asarray(
            raw[self.solution_dataset][field_index], dtype=np.float32
        )
        sample_mean = solution_samples.mean(axis=0, dtype=np.float64)
        sample_variance = solution_samples.var(axis=0, dtype=np.float64)
        target = np.stack((sample_mean, sample_variance), axis=0).astype(np.float32)
        return torch.from_numpy(encoded_input), torch.from_numpy(target)

    def close(self) -> None:
        for name in ("_raw_handle", "_cache_handle"):
            handle = getattr(self, name)
            if handle is not None:
                handle.close()
                setattr(self, name, None)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_raw_handle"] = None
        state["_cache_handle"] = None
        return state

    def __del__(self) -> None:
        self.close()
