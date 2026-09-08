"""Run-directory and experiment logging helpers."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.config import project_path


def create_run_dir(experiment_cfg: Any, model_name: str, loss_name: str, run_name: str | None = None) -> Path:
    """Create a collision-resistant experiment directory."""
    output_root = project_path(experiment_cfg.output_root)
    if run_name is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        run_name = f"{experiment_cfg.name}_{model_name}_{loss_name}_seed{int(experiment_cfg.seed)}_{timestamp}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_logger(run_dir: Path, name: str = "gaussian_law_exp") -> logging.Logger:
    """Log to both stdout and the run-local train.log file."""
    logger = logging.getLogger(f"{name}.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger
