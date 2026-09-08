"""One epoch of ISI categorical optimization."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.losses import compute_isi_loss
from src.training.validate import move_batch_to_device


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float | None = None,
) -> float:
    model.train()
    total = 0.0
    samples = 0
    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)
        device_batch = move_batch_to_device(batch, device)
        loss = compute_isi_loss(model(device_batch), device_batch)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        batch_size = int(device_batch["bin_counts"].shape[0])
        total += float(loss.detach().cpu()) * batch_size
        samples += batch_size
    return total / samples if samples else float("nan")

