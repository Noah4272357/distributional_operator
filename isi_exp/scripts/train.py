#!/usr/bin/env python3
"""Train an ISI categorical law-to-law model from a resolved YAML configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import build_dataloaders, load_split_payloads  # noqa: E402
from src.models.build_model import build_model  # noqa: E402
from src.training.factory import build_optimizer, build_scheduler  # noqa: E402
from src.training.trainer import Trainer  # noqa: E402
from src.utils.config import load_config, save_resolved_config  # noqa: E402
from src.utils.logging import build_logger, create_run_dir  # noqa: E402
from src.utils.seed import resolve_device, seed_everything  # noqa: E402


def run_training(
    cfg: Any,
    *,
    resume: str | Path | None = None,
    run_name: str | None = None,
    run_dir: str | Path | None = None,
) -> tuple[dict[str, Any], Path]:
    """Assemble configured components and delegate coordination to Trainer."""
    seed_everything(int(cfg.experiment.seed))
    device = resolve_device(str(cfg.experiment.device))
    payloads = load_split_payloads(
        cfg.data,
        map_location="cpu",
        truncate_dim=cfg.model.get("truncate_dim", "auto"),
    )
    preprocessing_state = payloads.get("_preprocessing_state")
    if preprocessing_state is not None:
        cfg.model.truncate_dim = int(preprocessing_state["selected_dim"])
        cfg.data.pca.selected_dim = int(preprocessing_state["selected_dim"])
        cfg.data.pca.achieved_explained_variance_ratio = float(
            preprocessing_state["achieved_explained_variance_ratio"]
        )
    dataloaders = build_dataloaders(cfg.data, payloads)
    model = build_model(cfg.model, payloads["train"]).to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = build_optimizer(model, cfg.optimizer) if trainable_parameters else None
    scheduler = build_scheduler(optimizer, cfg.scheduler) if optimizer is not None else None
    output_dir = (
        create_run_dir(cfg.experiment, str(cfg.model.name), run_name=run_name)
        if run_dir is None
        else Path(run_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if preprocessing_state is not None:
        torch.save(preprocessing_state, output_dir / str(cfg.data.pca.get("state_file", "pca_state.pt")))
    save_resolved_config(cfg, output_dir / "config.yaml")
    logger = build_logger(output_dir)
    logger.info("run_dir=%s device=%s model=%s", output_dir, device, cfg.model.name)
    trainer = Trainer(
        cfg,
        model,
        dataloaders,
        optimizer,
        scheduler,
        device,
        output_dir,
        logger,
        preprocessing_state=preprocessing_state,
    )
    return trainer.fit(resume=resume), output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("overrides", nargs="*", help="OmegaConf dot-list overrides.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    results, run_dir = run_training(cfg, resume=args.resume, run_name=args.run_name)
    print(f"model_name: {results['model_name']}")
    if "val" in results["metrics_by_split"]:
        print(
            "best_val_observation_nll: "
            f"{results['metrics_by_split']['val']['observable_metrics']['observation_nll']}"
        )
    elif "test" in results["metrics_by_split"]:
        print(
            "test_observation_nll: "
            f"{results['metrics_by_split']['test']['observable_metrics']['observation_nll']}"
        )
    print(f"results_json: {run_dir / 'results.json'}")


if __name__ == "__main__":
    main()
