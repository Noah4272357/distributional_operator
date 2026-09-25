"""Validation objective and generated-distribution metrics for diffusion models."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.utils.metrics import distribution_metrics

from .diffusion_train_epoch import _diffusion_loss


@torch.inference_mode()
def validate_diffusion(
    model: nn.Module,
    dataloader: DataLoader,
    distribution_criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate denoising loss and samples from the reverse diffusion process."""
    model.eval()
    diffusion_loss_sum = 0.0
    sinkhorn_sum = 0.0
    distribution_sums = {
        "mmd": 0.0,
        "sliced_wasserstein": 0.0,
        "energy_distance": 0.0,
    }
    samples = 0
    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        diffusion_loss = _diffusion_loss(model, inputs, targets)
        predictions = model(inputs)
        sinkhorn_distance = distribution_criterion(predictions, targets)
        batch_metrics = distribution_metrics(predictions, targets)
        batch_size = inputs.shape[0]
        diffusion_loss_sum += float(diffusion_loss) * batch_size
        sinkhorn_sum += float(sinkhorn_distance) * batch_size
        for name in distribution_sums:
            distribution_sums[name] += float(batch_metrics[name]) * batch_size
        samples += batch_size
    if samples == 0:
        raise ValueError("the diffusion validation dataloader is empty")
    metrics = {
        "diffusion_loss": diffusion_loss_sum / samples,
        "sinkhorn_distance": sinkhorn_sum / samples,
    }
    metrics.update(
        {name: value / samples for name, value in distribution_sums.items()}
    )
    return metrics
