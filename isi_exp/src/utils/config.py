"""Configuration loading, validation, path resolution, and persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EARLY_STOPPING_DEFAULTS = {
    "enabled": True,
    "patience": 30,
    "min_delta": 0.001,
}


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
    if OmegaConf.select(cfg, "training.early_stopping") is None:
        cfg.training.early_stopping = OmegaConf.create({})
    early_stopping = cfg.training.early_stopping
    if not OmegaConf.is_dict(early_stopping):
        raise ValueError("training.early_stopping must be a mapping")
    for key, default in EARLY_STOPPING_DEFAULTS.items():
        if early_stopping.get(key) is None:
            early_stopping[key] = default
    required = [
            "experiment.name",
            "experiment.output_root",
            "experiment.seed",
            "experiment.device",
            "data.train_size",
            "data.val_size",
            "data.test_size",
            "data.batch_size",
            "model.name",
            "optimizer.name",
            "scheduler.name",
            "training.epochs",
    ]
    process_data = OmegaConf.select(cfg, "data.process_file") is not None
    if process_data:
        required.extend(
            (
                "data.process_file",
                "data.target_file",
                "data.pca.explained_variance_threshold",
                "model.truncate_dim",
            )
        )
    else:
        required.append("data.file")
    _require(cfg, tuple(required))
    if int(cfg.data.batch_size) <= 0:
        raise ValueError("data.batch_size must be positive")
    if int(cfg.data.train_size) <= 0 or int(cfg.data.val_size) < 0 or int(cfg.data.test_size) < 0:
        raise ValueError("train_size must be positive; val_size and test_size cannot be negative")
    has_validation = int(cfg.data.val_size) > 0
    if not has_validation and int(cfg.data.test_size) <= 0:
        raise ValueError("at least one validation or test law is required")
    if int(cfg.training.epochs) < 0:
        raise ValueError("training.epochs cannot be negative")
    if not isinstance(cfg.training.early_stopping.enabled, bool):
        raise ValueError("training.early_stopping.enabled must be true or false")
    if int(cfg.training.early_stopping.patience) < 1:
        raise ValueError("training.early_stopping.patience must be positive")
    if float(cfg.training.early_stopping.min_delta) < 0.0:
        raise ValueError("training.early_stopping.min_delta cannot be negative")
    selection_metric = cfg.training.get("checkpoint_selection_metric")
    if has_validation:
        if str(selection_metric) != "val.observable_metrics.observation_nll":
            raise ValueError("checkpoint selection must remain val.observable_metrics.observation_nll")
    else:
        if selection_metric is not None:
            raise ValueError("checkpoint_selection_metric must be null without a validation split")
        if bool(cfg.training.early_stopping.enabled):
            raise ValueError("early stopping requires a validation split")
        if str(cfg.scheduler.name).lower() in {"reduce_lr_on_plateau", "reduce_on_plateau", "plateau"}:
            raise ValueError("ReduceLROnPlateau requires a validation split")
    if process_data:
        threshold = float(cfg.data.pca.explained_variance_threshold)
        if not 0.0 < threshold < 1.0:
            raise ValueError("data.pca.explained_variance_threshold must be in (0, 1)")
        truncate_dim = str(cfg.model.truncate_dim).lower()
        if truncate_dim != "auto" and int(cfg.model.truncate_dim) <= 0:
            raise ValueError("model.truncate_dim must be auto or a positive integer")


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
