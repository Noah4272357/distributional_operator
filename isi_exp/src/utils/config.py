"""Configuration loading, validation, path resolution, and persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load a YAML file, apply dot-list overrides, resolve it, and validate it."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(cfg)
    validate_training_config(cfg)
    return cfg


def _require(cfg: Any, keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if OmegaConf.select(cfg, key) is None]
    if missing:
        raise ValueError(f"missing required configuration values: {', '.join(missing)}")


def validate_training_config(cfg: Any) -> None:
    _require(
        cfg,
        (
            "experiment.name",
            "experiment.output_root",
            "experiment.seed",
            "experiment.device",
            "data.file",
            "data.train_size",
            "data.val_size",
            "data.test_size",
            "data.batch_size",
            "model.name",
            "optimizer.name",
            "scheduler.name",
            "training.epochs",
            "training.checkpoint_selection_metric",
        ),
    )
    if int(cfg.data.batch_size) <= 0:
        raise ValueError("data.batch_size must be positive")
    if int(cfg.data.train_size) <= 0 or int(cfg.data.val_size) <= 0 or int(cfg.data.test_size) < 0:
        raise ValueError("train_size and val_size must be positive; test_size cannot be negative")
    if int(cfg.training.epochs) < 0:
        raise ValueError("training.epochs cannot be negative")
    if str(cfg.training.checkpoint_selection_metric) != "val.observable_metrics.observation_nll":
        raise ValueError("checkpoint selection must remain val.observable_metrics.observation_nll")


def project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def save_resolved_config(cfg: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    OmegaConf.save(config=OmegaConf.create(resolved), f=destination)


def config_container(cfg: Any) -> dict[str, Any]:
    result = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(result, dict):
        raise TypeError("configuration root must be a mapping")
    return result
