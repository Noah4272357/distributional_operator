"""Input/output helpers for generated law datasets and manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import yaml


def ensure_parent_dir(path: str | Path) -> None:
    """Create the parent directory for ``path`` when needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_hdf5(
    path: str | Path,
    arrays: dict[str, torch.Tensor],
    split_indices: dict[str, torch.Tensor],
    attributes: dict[str, Any],
) -> None:
    """Save generated arrays and split membership in one compressed HDF5 file."""
    ensure_parent_dir(path)
    with h5py.File(Path(path), "w") as handle:
        for key, value in attributes.items():
            handle.attrs[key] = value
        for key, value in arrays.items():
            handle.create_dataset(key, data=value.detach().cpu().numpy(), compression="gzip")
        split_group = handle.create_group("split_indices")
        for split, value in split_indices.items():
            split_group.create_dataset(split, data=value.detach().cpu().numpy())


def load_hdf5_split(path: str | Path, split: str) -> dict[str, Any]:
    """Load one logical split from a root-array HDF5 dataset."""
    with h5py.File(Path(path), "r") as handle:
        if split not in handle["split_indices"]:
            raise ValueError(f"unknown split {split!r}")
        indices = np.asarray(handle["split_indices"][split], dtype=np.int64)
        payload: dict[str, Any] = {
            "schema_version": int(handle.attrs["schema_version"]),
            "split": split,
            "target_name": str(handle.attrs["target_name"]),
        }
        for key, value in handle.items():
            if key == "split_indices":
                continue
            payload[key] = torch.from_numpy(np.asarray(value)[indices])
    return payload


def save_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    """Save a YAML metadata payload."""
    ensure_parent_dir(path)
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML metadata payload as a resolved plain container."""
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return loaded
