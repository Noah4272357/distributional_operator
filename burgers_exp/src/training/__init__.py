"""Training loops, factories, and orchestration."""

from .losses import SinkhornLoss, build_loss

__all__ = ["SinkhornLoss", "build_loss"]
