"""Configuration loading, validation, overrides, and persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load a YAML config and merge OmegaConf dot-list overrides."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(cfg)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Any) -> None:
    """Fail early when a required experiment choice is absent or invalid."""
    required = (
        "experiment.name",
        "experiment.output_root",
        "experiment.seed",
        "experiment.device",
        "data.file",
        "data.batch_size",
        "model.name",
        "loss.name",
        "optimizer.name",
        "scheduler.name",
        "training.epochs",
        "training.checkpoint_selection_metric",
    )
    missing = [key for key in required if OmegaConf.select(cfg, key) is None]
    if missing:
        raise ValueError(f"missing required configuration values: {', '.join(missing)}")
    if int(cfg.data.batch_size) <= 0:
        raise ValueError("data.batch_size must be positive")
    if int(cfg.training.epochs) < 0:
        raise ValueError("training.epochs cannot be negative")
    if int(cfg.training.get("validation_interval", 10)) <= 0:
        raise ValueError("training.validation_interval must be positive")
    supported_schedulers = {
        "none",
        "cosine_annealing",
        "cosineannealing",
        "cosine_annealing_lr",
        "step_lr",
        "reduce_lr_on_plateau",
        "reduce_on_plateau",
        "plateau",
    }
    if str(cfg.scheduler.name).lower() not in supported_schedulers:
        raise ValueError(f"unsupported scheduler: {cfg.scheduler.name}")
    if str(cfg.training.checkpoint_selection_metric) != "val.observable_metrics.nll":
        raise ValueError("checkpoint selection must remain val.observable_metrics.nll")


def project_path(path: str | Path) -> Path:
    """Resolve a configured project-relative path."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def save_resolved_config(cfg: Any, path: str | Path) -> None:
    """Save the exact resolved configuration used by a run."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)), f=destination)


def config_container(cfg: Any) -> dict[str, Any]:
    """Return a resolved plain dictionary suitable for checkpoints."""
    result = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(result, dict):
        raise TypeError("the root configuration must resolve to a mapping")
    return result
