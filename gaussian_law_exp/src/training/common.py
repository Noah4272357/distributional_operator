"""Small tensor-batch helpers shared by training and validation."""

from __future__ import annotations

from typing import Any

import torch


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor values to a device while preserving metadata values."""
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def batch_size(batch: dict[str, Any]) -> int:
    """Infer the leading batch dimension from the first tensor."""
    for value in batch.values():
        if torch.is_tensor(value):
            return int(value.shape[0])
    raise ValueError("batch must contain at least one tensor")

