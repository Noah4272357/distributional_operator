"""Dataset interfaces for generated distribution pairs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import torch


DATA_KEYS = ("input_dist", "output_dist", "target_mean", "target_cov")


def validate_payload(payload: Any) -> dict[str, torch.Tensor]:
    """Validate and return a payload produced by ``data.generation``."""
    if not isinstance(payload, dict):
        raise TypeError("dataset payload must be a dictionary")
    if set(payload) != set(DATA_KEYS):
        missing = set(DATA_KEYS) - set(payload)
        extra = set(payload) - set(DATA_KEYS)
        raise ValueError(
            f"dataset keys do not match the data API; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    if not all(torch.is_tensor(payload[key]) for key in DATA_KEYS):
        raise TypeError("all dataset values must be tensors")

    data_size = int(payload["input_dist"].shape[0])
    if data_size < 1:
        raise ValueError("dataset must contain at least one distribution")
    if any(int(value.shape[0]) != data_size for value in payload.values()):
        raise ValueError("all dataset tensors must have the same leading dimension")
    if payload["input_dist"].ndim != 3 or payload["output_dist"].ndim != 3:
        raise ValueError("input_dist and output_dist must have shape [data, samples, q]")
    if payload["target_mean"].ndim != 2 or payload["target_cov"].ndim != 3:
        raise ValueError("target_mean and target_cov have invalid dimensions")

    input_dimension = int(payload["input_dist"].shape[-1])
    output_dimension = int(payload["output_dist"].shape[-1])
    if tuple(payload["target_mean"].shape) != (data_size, output_dimension):
        raise ValueError("target_mean shape must match output_dist")
    if tuple(payload["target_cov"].shape) != (
        data_size,
        output_dimension,
        output_dimension,
    ):
        raise ValueError("target_cov shape must match output_dist")
    if input_dimension <= 0 or output_dimension <= 0:
        raise ValueError("distribution dimensions must be positive")
    return payload


def load_dataset(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Load and validate one generated HDF5 dataset."""
    with h5py.File(Path(path), "r") as handle:
        payload = {
            key: torch.from_numpy(handle[key][...]).to(map_location)
            for key in handle.keys()
        }
    return validate_payload(payload)


def split_payload(
    payload: dict[str, torch.Tensor],
    *,
    train_size: int | None = None,
    test_size: int | None = None,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> dict[str, dict[str, torch.Tensor]]:
    """Create deterministic training and held-out tensor subsets.

    When explicit sizes are supplied, the held-out test subset is also used
    for validation because this experiment defines only train and test sets.
    """
    payload = validate_payload(payload)
    data_size = int(payload["input_dist"].shape[0])
    if data_size < 3:
        raise ValueError("dataset must contain at least three distributions to split")
    permutation = torch.randperm(
        data_size,
        generator=torch.Generator().manual_seed(int(seed)),
    )

    if train_size is not None or test_size is not None:
        if train_size is None or test_size is None:
            raise ValueError("train_size and test_size must be specified together")
        train_size = int(train_size)
        test_size = int(test_size)
        if train_size <= 0 or test_size <= 0:
            raise ValueError("train_size and test_size must be positive")
        if train_size + test_size != data_size:
            raise ValueError(
                "train_size + test_size must equal the dataset size "
                f"({train_size} + {test_size} != {data_size})"
            )
        train_indices = permutation[:train_size]
        test_indices = permutation[train_size:]
        indices = {
            "train": train_indices,
            "val": test_indices,
            "test": test_indices,
        }
        return {
            split: {
                key: value.index_select(0, split_indices)
                for key, value in payload.items()
            }
            for split, split_indices in indices.items()
        }

    train_fraction = float(train_fraction)
    val_fraction = float(val_fraction)
    test_fraction = 1.0 - train_fraction - val_fraction
    if train_fraction <= 0.0 or val_fraction <= 0.0 or test_fraction <= 0.0:
        raise ValueError("train, validation, and test fractions must be positive")

    val_size = max(1, int(round(data_size * val_fraction)))
    test_size = max(1, int(round(data_size * test_fraction)))
    train_size = data_size - val_size - test_size
    if train_size < 1:
        raise ValueError("split fractions leave no training examples")

    train_end = train_size
    val_end = train_size + val_size
    indices = {
        "train": permutation[:train_end],
        "val": permutation[train_end:val_end],
        "test": permutation[val_end:],
    }
    return {
        split: {key: value.index_select(0, split_indices) for key, value in payload.items()}
        for split, split_indices in indices.items()
    }


class LawDataset(torch.utils.data.Dataset):
    """Dataset view over one generated distribution-pair payload."""

    def __init__(
        self,
        payload: dict[str, torch.Tensor],
        input_indices: torch.Tensor | None = None,
    ) -> None:
        self.payload = validate_payload(payload)
        self.input_indices = (
            None
            if input_indices is None
            else torch.as_tensor(input_indices, dtype=torch.long)
        )

    def __len__(self) -> int:
        return int(self.payload["input_dist"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: value[index] for key, value in self.payload.items()}
        if self.input_indices is not None:
            indices = self.input_indices.to(device=item["input_dist"].device)
            item["input_dist"] = item["input_dist"].index_select(0, indices)
        return item


def make_datasets(
    split_payloads: dict[str, dict[str, torch.Tensor]],
    input_indices: torch.Tensor | None = None,
) -> dict[str, LawDataset]:
    """Create dataset objects for all three subsets."""
    if set(split_payloads) != {"train", "val", "test"}:
        raise ValueError("split_payloads must contain train, val, and test")
    return {
        name: LawDataset(payload, input_indices=input_indices)
        for name, payload in split_payloads.items()
    }
