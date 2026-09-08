"""Single-epoch optimization."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.common import batch_size, move_batch_to_device


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float | None = None,
) -> float:
    """Optimize the model for exactly one pass over the training split."""
    model.train()
    total_loss = 0.0
    total_items = 0
    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)
        device_batch = move_batch_to_device(batch, device)
        loss = criterion(model(device_batch), device_batch)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
        optimizer.step()
        count = batch_size(device_batch)
        total_items += count
        total_loss += float(loss.detach().cpu().item()) * count
    return float(total_loss / total_items) if total_items else float("nan")

