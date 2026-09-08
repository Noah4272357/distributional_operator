"""Compatibility facade for the former monolithic training module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from scripts.train import main, run_training
from src.models.build_model import build_isi_model, model_metadata
from src.models.isi_models import (
    ISIContextDeepSets,
    ISIFeatureMLP,
    ISIParamMLP,
)
from src.training.losses import compute_isi_loss
from src.training.metrics import (
    _finite_renormalized_piecewise_uniform_w2,
    categorical_metrics_against_empirical,
)
from src.training.validate import evaluate_isi_model, move_batch_to_device
from src.utils.checkpoint import CHECKPOINT_SELECTION_METRIC
from src.utils.config import load_config


def _adapt_legacy_config(cfg: Any) -> Any:
    """Map the original Hydra layout to the standalone layout when needed."""
    if not OmegaConf.is_config(cfg):
        cfg = OmegaConf.create(cfg)
    if OmegaConf.select(cfg, "data") is not None and OmegaConf.select(cfg, "training") is not None:
        return cfg
    if OmegaConf.select(cfg, "generated_data") is None:
        raise ValueError("configuration must define either data or generated_data")
    adapted = load_config("configs/config.yaml")
    experiment = OmegaConf.select(cfg, "experiment")
    adapted.data = OmegaConf.create(OmegaConf.to_container(cfg.generated_data, resolve=True))
    adapted.data.batch_size = int(OmegaConf.select(experiment, "batch_size", default=64))
    adapted.data.num_workers = int(OmegaConf.select(experiment, "num_workers", default=0))
    adapted.data.pin_memory = bool(OmegaConf.select(experiment, "pin_memory", default=False))
    adapted.model = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    if OmegaConf.select(cfg, "optimizer") is not None:
        adapted.optimizer = OmegaConf.create(OmegaConf.to_container(cfg.optimizer, resolve=True))
    if OmegaConf.select(cfg, "scheduler") is not None:
        adapted.scheduler = OmegaConf.create(OmegaConf.to_container(cfg.scheduler, resolve=True))
    adapted.experiment.name = str(OmegaConf.select(experiment, "name", default="isi_law_to_law"))
    adapted.experiment.seed = int(OmegaConf.select(experiment, "seed", default=0))
    adapted.experiment.device = str(OmegaConf.select(experiment, "device", default="cpu"))
    adapted.experiment.output_root = str(OmegaConf.select(experiment, "save_path", default="experiments"))
    adapted.training.epochs = int(OmegaConf.select(experiment, "epochs", default=1))
    return adapted


def run_isi_experiment(cfg: Any, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Preserve the original return shape while using the modular trainer."""
    results, _ = run_training(_adapt_legacy_config(cfg), run_dir=output_dir)
    return results


if __name__ == "__main__":
    main()
