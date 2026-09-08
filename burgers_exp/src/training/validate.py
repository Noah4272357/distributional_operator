"""Validation and domain metrics."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.utils.metrics import distribution_metrics


@torch.inference_mode()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
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
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        batch_metrics = distribution_metrics(predictions, targets)
        batch = inputs.shape[0]
        sinkhorn_sum += float(loss) * batch
        for name, value in batch_metrics.items():
            distribution_sums[name] += float(value) * batch
        samples += batch
    metrics = {"sinkhorn_distance": sinkhorn_sum / samples}
    metrics.update({name: value / samples for name, value in distribution_sums.items()})
    return metrics
