"""Optimizer and scheduler factories."""

from __future__ import annotations

import torch
from torch import nn


def build_optimizer(config: dict, model: nn.Module) -> torch.optim.Optimizer:
    if config["name"].lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
        )
    raise ValueError(f"unsupported optimizer: {config['name']}")


def build_scheduler(
    config: dict, optimizer: torch.optim.Optimizer, epochs: int
) -> torch.optim.lr_scheduler.LRScheduler:
    name = config["name"].lower()
    if name == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=float(config["eta_min"])
        )
    if name in {"reduce_lr_on_plateau", "reduce_on_plateau"}:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(config.get("mode", "min")),
            factor=float(config.get("factor", 0.1)),
            patience=int(config.get("patience", 10)),
            threshold=float(config.get("threshold", 1.0e-4)),
            threshold_mode=str(config.get("threshold_mode", "rel")),
            cooldown=int(config.get("cooldown", 0)),
            min_lr=float(config.get("min_lr", 0.0)),
            eps=float(config.get("eps", 1.0e-8)),
        )
    raise ValueError(f"unsupported scheduler: {config['name']}")
