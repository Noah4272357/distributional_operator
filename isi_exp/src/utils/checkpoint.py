"""Resumable checkpoints that remain readable by the original experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.utils.config import config_container


CHECKPOINT_SELECTION_METRIC = "val.observable_metrics.observation_nll"


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    epoch: int,
    best_val_observation_nll: float,
    cfg: Any,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": cpu_state_dict(model),
            "model_name": str(cfg.model.name),
            "model_config": config_container(cfg)["model"],
            "epoch": int(epoch),
            "checkpoint_selection_metric": CHECKPOINT_SELECTION_METRIC,
            "best_val_observation_nll": float(best_val_observation_nll),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "config": config_container(cfg),
            "torch_rng_state": torch.get_rng_state(),
        },
        destination,
    )


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint is missing model_state_dict")
    return checkpoint


def restore_checkpoint(
    checkpoint: dict[str, Any],
    model: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
) -> tuple[int, float]:
    model.load_state_dict({key: value.to(device) for key, value in checkpoint["model_state_dict"].items()})
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if checkpoint.get("torch_rng_state") is not None:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    return (
        int(checkpoint.get("epoch", 0)),
        float(checkpoint.get("best_val_observation_nll", float("inf"))),
    )
