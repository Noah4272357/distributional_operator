"""Optimizer and scheduler factories."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def build_optimizer(model: nn.Module, cfg: Any) -> torch.optim.Optimizer:
    name = str(cfg.name).lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    raise ValueError(f"unsupported optimizer: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Any,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    name = str(cfg.name).lower()
    if name == "none":
        return None
    if name == "step_lr":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(cfg.step_size), gamma=float(cfg.gamma))
    if name in {"cosineannealing", "cosine_annealing", "cosine_annealing_lr"}:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg.t_max),
            eta_min=float(cfg.get("eta_min", 0.0)),
        )
    if name in {"reduce_lr_on_plateau", "reduce_on_plateau", "plateau"}:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(cfg.get("mode", "min")),
            factor=float(cfg.get("factor", 0.1)),
            patience=int(cfg.get("patience", 10)),
            threshold=float(cfg.get("threshold", 1.0e-4)),
            threshold_mode=str(cfg.get("threshold_mode", "rel")),
            cooldown=int(cfg.get("cooldown", 0)),
            min_lr=float(cfg.get("min_lr", 0.0)),
            eps=float(cfg.get("eps", 1.0e-8)),
        )
    raise ValueError(f"unsupported scheduler: {name}")
