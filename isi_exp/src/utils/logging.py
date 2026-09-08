"""Run-directory creation and file/console logging."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.config import project_path


def create_run_dir(experiment_cfg: Any, model_name: str, run_name: str | None = None) -> Path:
    root = project_path(str(experiment_cfg.output_root))
    if run_name is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        run_name = f"{model_name}_seed{int(experiment_cfg.seed)}_{stamp}"
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"isi_exp.{run_dir.resolve()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

