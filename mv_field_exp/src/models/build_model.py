"""Stable model-factory interface."""

from __future__ import annotations

from typing import Any

from torch import nn


def build_model(
    model_cfg: Any,
    loss_cfg: Any,
    train_payload: dict[str, Any],
    input_representation: str,
) -> nn.Module:
    """Build the configured model while preserving the upstream implementation."""
    from src.training.pipeline import build_random_field_model

    return build_random_field_model(model_cfg, loss_cfg, train_payload, input_representation)
