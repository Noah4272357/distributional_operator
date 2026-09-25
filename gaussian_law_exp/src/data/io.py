"""Input/output helpers for generated law datasets and manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf


def ensure_parent_dir(path: str | Path) -> None:
    """Create the parent directory for ``path`` when needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_split(path: str | Path, payload: dict[str, Any]) -> None:
    """Save a generated dataset split payload."""
    ensure_parent_dir(path)
    torch.save(payload, Path(path))


def load_split(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a generated dataset split payload."""
    return torch.load(Path(path), map_location=map_location)


def save_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    """Save a YAML metadata payload."""
    ensure_parent_dir(path)
    OmegaConf.save(config=OmegaConf.create(payload), f=str(path))


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML metadata payload as a resolved plain container."""
    loaded = OmegaConf.load(path)
    return OmegaConf.to_container(loaded, resolve=True)
