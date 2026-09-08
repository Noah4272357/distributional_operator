"""Shared GeomLoss Sinkhorn construction and distance conversion."""

from __future__ import annotations

import torch
from geomloss import SamplesLoss


def build_sinkhorn(
    *,
    p: int = 2,
    blur: float = 0.05,
    scaling: float = 0.9,
    debias: bool = True,
    backend: str = "tensorized",
) -> SamplesLoss:
    if int(p) not in (1, 2):
        raise ValueError("GeomLoss Sinkhorn p must be 1 or 2")
    return SamplesLoss(
        loss="sinkhorn",
        p=int(p),
        blur=float(blur),
        scaling=float(scaling),
        debias=bool(debias),
        backend=str(backend),
    )


def sinkhorn_distance(value: torch.Tensor, p: int) -> torch.Tensor:
    """Convert GeomLoss's p-ground-cost divergence to distance units."""
    value = value.clamp_min(0.0)
    if int(p) == 2:
        return (2.0 * value).sqrt()
    return value
