"""Reproducibility utilities."""

from __future__ import annotations

import random

import torch


def seed_everything(seed: int) -> None:
    """Seed Python and PyTorch CPU/CUDA generators."""
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(requested: str) -> torch.device:
    """Resolve `auto` or validate an explicitly requested device."""
    normalized = str(requested).lower()
    if normalized == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {requested}")
    return device

