#!/usr/bin/env python3
"""Thin training CLI for the standalone coefficient-Gaussian experiment."""

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
from src.models.factory import build_model  # noqa: E402
from src.training.factory import build_optimizer, build_scheduler  # noqa: E402
from src.training.losses import build_loss  # noqa: E402
from src.training.trainer import Trainer  # noqa: E402
from src.utils.config import load_config, save_resolved_config  # noqa: E402
from src.utils.logging import build_logger, create_run_dir  # noqa: E402
from src.utils.metrics import loss_metadata  # noqa: E402
from src.utils.seed import resolve_device, seed_everything  # noqa: E402


def run_training(
    cfg: Any,
    *,
    resume: str | Path | None = None,
    run_name: str | None = None,
    run_dir: str | Path | None = None,
    input_indices: torch.Tensor | None = None,
) -> tuple[dict[str, Any], Path]:
    """Assemble configured components and delegate coordination to Trainer."""
    loss_metadata(str(cfg.loss.name), str(cfg.model.name))
    seed_everything(int(cfg.experiment.seed))
    device = resolve_device(str(cfg.experiment.device))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    split_payloads = load_split_payloads(cfg.data, map_location="cpu")
    dataloaders = build_dataloaders(
        cfg.data,
        split_payloads,
        input_indices=input_indices,
    )
    model = build_model(cfg.model, cfg.data, cfg.loss, split_payloads["train"]).to(device)
    criterion = build_loss(cfg.loss)
    optimizer = build_optimizer(model, cfg.optimizer)
    scheduler = build_scheduler(optimizer, cfg.scheduler)

    if run_dir is None:
        output_dir = create_run_dir(
            cfg.experiment,
            str(cfg.model.name),
            str(cfg.loss.name),
            run_name=run_name,
        )
    else:
        output_dir = Path(run_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(cfg, output_dir / "config.yaml")
    logger = build_logger(output_dir)
    logger.info("run_dir=%s", output_dir)
    logger.info("device=%s model=%s loss=%s", device, cfg.model.name, cfg.loss.name)

    trainer = Trainer(
        cfg=cfg,
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        run_dir=output_dir,
        logger=logger,
    )
    return trainer.fit(resume=resume), output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml", help="Project-relative YAML configuration.")
    parser.add_argument("--resume", type=Path, default=None, help="New or legacy checkpoint to resume.")
    parser.add_argument("--run-name", default=None, help="Optional explicit experiment directory name.")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dot-list overrides, e.g. training.epochs=20.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    results, run_dir = run_training(cfg, resume=args.resume, run_name=args.run_name)
    print(f"model_name: {results['model_name']}")
    print(f"loss_name: {results['loss_name']}")
    print(f"best_val_nll: {results['metrics_by_split']['val']['observable_metrics']['nll']}")
    print(f"results_json: {run_dir / 'results.json'}")
    if results["peak_cuda_memory_bytes"] is not None:
        peak_gib = results["peak_cuda_memory_bytes"] / 1024**3
        print(f"peak_cuda_memory_gib: {peak_gib:.6f}")


if __name__ == "__main__":
    main()
