"""Project logging configuration."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(run_dir: str | Path) -> logging.Logger:
    logger = logging.getLogger("mv_field_exp")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(Path(run_dir) / "train.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger
