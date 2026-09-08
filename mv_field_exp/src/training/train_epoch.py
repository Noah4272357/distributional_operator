"""Single-epoch optimization API."""

from __future__ import annotations

from src.training.pipeline import _train_one_epoch as train_one_epoch

__all__ = ["train_one_epoch"]
