"""Factory for training losses."""

from __future__ import annotations

from typing import Any

from torch import nn

from src.utils.metrics import ConfiguredLoss


def build_loss(loss_cfg: Any) -> nn.Module:
    """Build the configured training objective."""
    return ConfiguredLoss(loss_cfg)
