"""Split-level evaluation for observable and synthetic diagnostic metrics."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.utils.metrics import (
    VALIDATION_METRICS,
    gaussian_nll,
    gaussian_validation_metrics,
)
from src.training.common import batch_size, move_batch_to_device


def _mean(total: float, count: int) -> float:
    return float(total / count) if count else float("nan")


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    metrics_cfg: Any | None = None,
) -> dict[str, Any]:
    """Evaluate one split without changing the caller's model mode."""
    del metrics_cfg
    was_training = model.training
    model.eval()
    total_items = 0
    total_training_loss = 0.0
    metric_totals = {name: 0.0 for name in VALIDATION_METRICS}

    for batch in dataloader:
        device_batch = move_batch_to_device(batch, device)
        prediction = model(device_batch)
        count = batch_size(device_batch)
        training_loss = criterion(prediction, device_batch)
        metrics = {
            "nll": float(
                gaussian_nll(
                    prediction["pred_mean"],
                    prediction["pred_scale_tril"],
                    device_batch["output_dist"],
                ).cpu().item()
            )
        }
        metrics.update(gaussian_validation_metrics(prediction, device_batch))

        total_items += count
        total_training_loss += float(training_loss.cpu().item()) * count
        for name, value in metrics.items():
            metric_totals[name] += float(value) * count

    if was_training:
        model.train()
    return {
        "training_loss": _mean(total_training_loss, total_items),
        "observable_metrics": {
            name: _mean(total, total_items)
            for name, total in metric_totals.items()
        },
        "synthetic_diagnostic_metrics": {},
    }
