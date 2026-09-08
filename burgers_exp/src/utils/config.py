"""Load, override, validate, and save resolved JSON configuration."""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def resolve_config(
    config: dict, *, epochs: int | None, device: str | None, seed: int | None = None
) -> dict:
    resolved = copy.deepcopy(config)
    if epochs is not None:
        resolved["training"]["epochs"] = epochs
    if device is not None:
        resolved["training"]["device"] = device
    if seed is not None:
        resolved["seed"] = seed
    validate_config(resolved)
    return resolved


def validate_config(config: dict) -> None:
    if int(config["training"]["epochs"]) < 1:
        raise ValueError("training.epochs must be positive")
    data_config = config["data"]
    if int(data_config["batch_size"]) < 1:
        raise ValueError("data.batch_size must be positive")
    if "train_size" in data_config or "validation_size" in data_config:
        if int(data_config["train_size"]) < 1:
            raise ValueError("data.train_size must be positive")
        if int(data_config["validation_size"]) < 1:
            raise ValueError("data.validation_size must be positive")
        if int(data_config.get("test_size", 0)) < 0:
            raise ValueError("data.test_size must be non-negative")
    elif not 0.0 < float(data_config["train_fraction"]) < 1.0:
        raise ValueError("data.train_fraction must lie between zero and one")
    if float(config["optimizer"]["lr"]) <= 0:
        raise ValueError("optimizer.lr must be positive")
    scheduler_config = config.get("scheduler")
    scheduler_name = (
        scheduler_config["name"].lower() if scheduler_config is not None else None
    )
    if scheduler_name in {"reduce_lr_on_plateau", "reduce_on_plateau"}:
        if scheduler_config["mode"] not in {"min", "max"}:
            raise ValueError("scheduler.mode must be 'min' or 'max'")
        if not 0.0 < float(scheduler_config["factor"]) < 1.0:
            raise ValueError("scheduler.factor must lie between zero and one")
        if int(scheduler_config["patience"]) < 0:
            raise ValueError("scheduler.patience must be non-negative")
        if float(scheduler_config["threshold"]) < 0:
            raise ValueError("scheduler.threshold must be non-negative")
        if scheduler_config["threshold_mode"] not in {"rel", "abs"}:
            raise ValueError("scheduler.threshold_mode must be 'rel' or 'abs'")
        if int(scheduler_config["cooldown"]) < 0:
            raise ValueError("scheduler.cooldown must be non-negative")
        if float(scheduler_config["min_lr"]) < 0:
            raise ValueError("scheduler.min_lr must be non-negative")
        if float(scheduler_config["eps"]) < 0:
            raise ValueError("scheduler.eps must be non-negative")


def create_run_dir(project_root: Path, config: dict, run_name: str | None) -> Path:
    root = (project_root / config["experiment"]["root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    name = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = root / name
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


def save_config(config: dict, path: Path) -> None:
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
