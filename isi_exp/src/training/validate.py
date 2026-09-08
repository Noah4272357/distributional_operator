"""Evaluation loop for train, validation, and test splits."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.metrics import _per_law_metrics

_REGIME_NAMES = ("subthreshold", "balanced_near", "suprathreshold")


@torch.no_grad()
def evaluate_isi_nll(model: nn.Module, dataloader: DataLoader, device: torch.device | str) -> dict[str, Any]:
    """Evaluate only observation NLL for inexpensive checkpoint selection."""
    target_device = torch.device(device)
    was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    for batch in dataloader:
        device_batch = move_batch_to_device(batch, target_device)
        logits = model(device_batch)["logits"]
        counts = device_batch["bin_counts"].to(dtype=logits.dtype)
        per_law_events = counts.sum(dim=-1).clamp_min(1.0)
        per_law_nll = -(counts * torch.log_softmax(logits, dim=-1)).sum(dim=-1) / per_law_events
        total += float(per_law_nll.sum().detach().cpu())
        count += int(per_law_nll.numel())
    if was_training:
        model.train()
    value = _mean_weighted(total, count)
    return {"training_loss": value, "observable_metrics": {"observation_nll": value}}

def move_batch_to_device(batch: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    """Copy a collated batch dict to a device, preserving non-tensor values."""
    target = torch.device(device)
    return {key: value.to(target) if torch.is_tensor(value) else value for key, value in batch.items()}


def _mean_weighted(total: float, count: int) -> float:
    return float(total / count) if count else float("nan")


@torch.no_grad()
def evaluate_isi_model(model: nn.Module, dataloader: DataLoader, device: torch.device | str) -> dict[str, Any]:
    """Evaluate observable metrics for one split."""
    target_device = torch.device(device)
    was_training = model.training
    model.eval()
    metric_totals: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    by_regime_totals: dict[int, dict[str, float]] = {}
    by_regime_counts: dict[int, int] = {}

    for batch in dataloader:
        device_batch = move_batch_to_device(batch, target_device)
        prediction = model(device_batch)
        per_law = _per_law_metrics(prediction, device_batch)
        batch_size = int(device_batch["bin_counts"].shape[0])
        for metric_name, values in per_law.items():
            metric_totals[metric_name] = metric_totals.get(metric_name, 0.0) + float(values.sum().detach().cpu())
            metric_counts[metric_name] = metric_counts.get(metric_name, 0) + batch_size
        for regime_label in torch.unique(device_batch["regime_label"]).tolist():
            label = int(regime_label)
            mask = device_batch["regime_label"] == label
            by_regime_counts[label] = by_regime_counts.get(label, 0) + int(mask.sum())
            totals = by_regime_totals.setdefault(label, {})
            for metric_name, values in per_law.items():
                totals[metric_name] = totals.get(metric_name, 0.0) + float(values[mask].sum().detach().cpu())

    if was_training:
        model.train()
    observable_metrics = {
        key: _mean_weighted(value, metric_counts[key])
        for key, value in metric_totals.items()
    }
    observable_metrics_by_regime = {
        _REGIME_NAMES[label] if label < len(_REGIME_NAMES) else str(label): {
            key: _mean_weighted(value, by_regime_counts[label])
            for key, value in totals.items()
        }
        for label, totals in by_regime_totals.items()
    }
    return {
        "training_loss": observable_metrics["observation_nll"],
        "observable_metrics": observable_metrics,
        "observable_metrics_by_regime": observable_metrics_by_regime,
    }
