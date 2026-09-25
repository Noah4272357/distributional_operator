#!/usr/bin/env python3
"""Evaluate an ISI checkpoint without retraining."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import build_dataloaders, load_split_payloads  # noqa: E402
from src.models.build_model import build_model  # noqa: E402
from src.training.validate import evaluate_isi_model  # noqa: E402
from src.utils.checkpoint import load_checkpoint, restore_checkpoint  # noqa: E402
from src.utils.config import load_config, validate_training_config  # noqa: E402
from src.utils.reporting import _write_json  # noqa: E402
from src.utils.seed import resolve_device, seed_everything  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", default=None, help="Needed for legacy checkpoints without embedded config.")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("overrides", nargs="*", help="Optional OmegaConf dot-list overrides.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    checkpoint = load_checkpoint(args.checkpoint)
    if args.config:
        cfg = load_config(args.config, args.overrides)
    elif checkpoint.get("config") is not None:
        cfg = OmegaConf.create(checkpoint["config"])
        if args.overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
        OmegaConf.resolve(cfg)
        validate_training_config(cfg)
    else:
        raise ValueError("legacy checkpoint has no embedded config; pass --config")
    seed_everything(int(cfg.experiment.seed))
    device = resolve_device(str(cfg.experiment.device))
    preprocessing_state = checkpoint.get("preprocessing_state")
    if cfg.data.get("process_file") is not None and preprocessing_state is None:
        raise ValueError("distribution-operator checkpoint is missing its PCA preprocessing state")
    payloads = load_split_payloads(
        cfg.data,
        preprocessing_state=preprocessing_state,
        truncate_dim=cfg.model.get("truncate_dim", "auto"),
    )
    dataloaders = build_dataloaders(cfg.data, payloads, shuffle_train=False)
    model = build_model(cfg.model, payloads["train"]).to(device)
    restore_checkpoint(checkpoint, model, device)
    selected = dataloaders if args.split == "all" else {args.split: dataloaders[args.split]}
    metrics = {split: evaluate_isi_model(model, loader, device) for split, loader in selected.items()}
    output = args.output or args.checkpoint.resolve().parents[1] / f"evaluation_{args.split}.json"
    _write_json(output, {"checkpoint": str(args.checkpoint), "metrics_by_split": metrics})
    print(f"evaluation_json: {output}")


if __name__ == "__main__":
    main()
