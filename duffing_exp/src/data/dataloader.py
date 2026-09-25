"""Deterministic split, preprocessing, dataset, and DataLoader assembly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import DataLoader

from .dataset import RandomFieldDataset
from .preprocessing import prepare_pca_data


@dataclass
class DataBundle:
    train_loader: DataLoader
    validation_loader: DataLoader
    train_dataset: RandomFieldDataset
    validation_dataset: RandomFieldDataset
    pca_state: dict

    def close(self) -> None:
        self.train_dataset.close()
        self.validation_dataset.close()


def build_dataloaders(config: dict, project_root: Path) -> DataBundle:
    data_cfg = config["data"]
    raw_path = (project_root / data_cfg["path"]).resolve()
    initial_dataset = data_cfg["initial_dataset"]
    solution_dataset = data_cfg["solution_dataset"]
    with h5py.File(raw_path, "r") as raw:
        count = int(raw[initial_dataset].shape[0])
    rng = np.random.default_rng(int(config["seed"]))
    indices = rng.permutation(count)
    train_count = int(round(count * float(data_cfg["train_fraction"])))
    train_indices = np.sort(indices[:train_count])
    validation_indices = np.sort(indices[train_count:])
    cache_path = (project_root / data_cfg["cache_path"]).resolve()
    state_path = (project_root / data_cfg["state_path"]).resolve()
    cache_path, pca_state = prepare_pca_data(
        raw_path,
        cache_path,
        state_path,
        train_indices,
        initial_dataset=initial_dataset,
        solution_dataset=solution_dataset,
        dimension=int(data_cfg["pca_dimension"]),
        fields_per_batch=int(data_cfg["pca_fields_per_batch"]),
    )
    common = dict(
        raw_path=raw_path,
        pca_cache_path=cache_path,
        initial_dataset=initial_dataset,
        solution_dataset=solution_dataset,
    )
    train_dataset = RandomFieldDataset(indices=train_indices, **common)
    validation_dataset = RandomFieldDataset(indices=validation_indices, **common)
    loader_common = dict(
        batch_size=int(data_cfg["batch_size"]),
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=bool(data_cfg["pin_memory"]),
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_common)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_common)
    return DataBundle(
        train_loader, validation_loader, train_dataset, validation_dataset, pca_state
    )
