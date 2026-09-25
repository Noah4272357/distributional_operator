"""One optimization epoch for conditional diffusion models."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def _diffusion_loss(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor):
    loss_function = getattr(model, "diffusion_loss", None)
    if not callable(loss_function):
        raise TypeError(
            "the diffusion training pipeline requires a model with "
            "diffusion_loss(inputs, targets)"
        )
    return loss_function(inputs, targets)


def train_diffusion_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float | None,
) -> float:
    """Optimize the DDPM noise-prediction objective for one epoch."""
    model.train()
    loss_sum = 0.0
    samples = 0
    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = _diffusion_loss(model, inputs, targets)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        batch_size = inputs.shape[0]
        loss_sum += float(loss.detach()) * batch_size
        samples += batch_size
    if samples == 0:
        raise ValueError("the diffusion training dataloader is empty")
    return loss_sum / samples
