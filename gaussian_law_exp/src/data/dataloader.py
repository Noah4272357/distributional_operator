"""Dataset loading, splitting, and DataLoader assembly."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from src.data.dataset import LawDataset, load_dataset, make_datasets, split_payload
from src.utils.config import project_path


def load_split_payloads(
    data_cfg: Any,
    map_location: str | torch.device = "cpu",
) -> dict[str, dict[str, torch.Tensor]]:
    """Load one generated dataset and partition it reproducibly."""
    payload = load_dataset(project_path(data_cfg.file), map_location=map_location)
    return split_payload(
        payload,
        train_size=data_cfg.get("train_size"),
        test_size=data_cfg.get("test_size"),
        train_fraction=float(data_cfg.get("train_fraction", 0.8)),
        val_fraction=float(data_cfg.get("val_fraction", 0.1)),
        seed=int(data_cfg.get("split_seed", 0)),
    )


def build_dataloaders(
    data_cfg: Any,
    split_payloads: dict[str, dict[str, torch.Tensor]],
    input_indices: torch.Tensor | None = None,
) -> dict[str, DataLoader]:
    """Construct a shuffled training loader and deterministic evaluation loaders."""
    datasets: dict[str, LawDataset] = make_datasets(
        split_payloads,
        input_indices=input_indices,
    )
    return {
        split: DataLoader(
            dataset,
            batch_size=int(data_cfg.batch_size),
            shuffle=split == "train",
            num_workers=int(data_cfg.get("num_workers", 0)),
            pin_memory=bool(data_cfg.get("pin_memory", False)),
        )
        for split, dataset in datasets.items()
    }
