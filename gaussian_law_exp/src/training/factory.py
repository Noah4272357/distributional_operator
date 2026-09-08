"""Optimizer and scheduler factories."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def build_optimizer(model: nn.Module, optimizer_cfg: Any) -> torch.optim.Optimizer | None:
    """Build the configured optimizer."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        return None
    name = str(optimizer_cfg.name).lower()
    if name == "adam":
        optimizer_class = torch.optim.Adam
    elif name == "adamw":
        optimizer_class = torch.optim.AdamW
    else:
        raise ValueError(f"unsupported optimizer: {name}")
    return optimizer_class(
        parameters,
        lr=float(optimizer_cfg.get("lr", 1.0e-3)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer | None,
    scheduler_cfg: Any,
) -> (
    torch.optim.lr_scheduler.LRScheduler
    | torch.optim.lr_scheduler.ReduceLROnPlateau
    | None
):
    """Build the configured scheduler, or skip it for parameter-free models."""
    name = str(scheduler_cfg.name).lower()
    if name == "none":
        return None
    if optimizer is None:
        return None
    if name in {"cosine_annealing", "cosineannealing", "cosine_annealing_lr"}:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_cfg.t_max),
            eta_min=float(scheduler_cfg.get("eta_min", 0.0)),
        )
    if name == "step_lr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_cfg.step_size),
            gamma=float(scheduler_cfg.gamma),
        )
    if name in {"reduce_lr_on_plateau", "reduce_on_plateau", "plateau"}:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(scheduler_cfg.get("mode", "min")),
            factor=float(scheduler_cfg.get("factor", 0.1)),
            patience=int(scheduler_cfg.get("patience", 10)),
            threshold=float(scheduler_cfg.get("threshold", 1.0e-4)),
            threshold_mode=str(scheduler_cfg.get("threshold_mode", "rel")),
            cooldown=int(scheduler_cfg.get("cooldown", 0)),
            min_lr=float(scheduler_cfg.get("min_lr", 0.0)),
            eps=float(scheduler_cfg.get("eps", 1.0e-8)),
        )
    raise ValueError(f"unsupported scheduler: {name}")
