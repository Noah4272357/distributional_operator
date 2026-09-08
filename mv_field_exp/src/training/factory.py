"""Optimizer and learning-rate scheduler factories."""

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def build_optimizer(model: nn.Module, cfg: Any) -> torch.optim.Optimizer:
    """Build the configured optimizer for the model's trainable parameters."""
    name = str(_get(cfg, "name", "adamw")).lower()
    kwargs = {
        "lr": float(_get(cfg, "lr", 1.0e-3)),
        "weight_decay": float(_get(cfg, "weight_decay", 0.01)),
    }

    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    raise ValueError(f"Unsupported optimizer: {name!r}. Choose 'adamw' or 'adam'.")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Any,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    """Build the configured learning-rate scheduler."""
    name = str(_get(cfg, "name", "none")).lower()
    if name == "none":
        return None
    if name == "reduce_lr_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(_get(cfg, "mode", "min")),
            factor=float(_get(cfg, "factor", 0.5)),
            patience=int(_get(cfg, "patience", 5)),
            threshold=float(_get(cfg, "threshold", 1.0e-4)),
            threshold_mode=str(_get(cfg, "threshold_mode", "rel")),
            cooldown=int(_get(cfg, "cooldown", 0)),
            min_lr=float(_get(cfg, "min_lr", 1.0e-6)),
            eps=float(_get(cfg, "eps", 1.0e-8)),
        )
    raise ValueError(
        f"Unsupported scheduler: {name!r}. Choose 'reduce_lr_on_plateau' or 'none'."
    )


def step_scheduler(
    scheduler: torch.optim.lr_scheduler.LRScheduler
    | torch.optim.lr_scheduler.ReduceLROnPlateau
    | None,
    validation_metric: float,
) -> None:
    """Advance a scheduler using the validation checkpoint metric when required."""
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(float(validation_metric))
    else:
        scheduler.step()
