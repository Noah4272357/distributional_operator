#!/usr/bin/env python3
"""Evaluate a saved coefficient-Gaussian checkpoint without retraining."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import build_dataloaders, load_split_payloads  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.training.losses import build_loss  # noqa: E402
from src.training.validate import evaluate_model  # noqa: E402
from src.utils.artifacts import write_json  # noqa: E402
from src.utils.checkpoint import load_checkpoint, restore_checkpoint  # noqa: E402
from src.utils.config import load_config, validate_config  # noqa: E402
from src.utils.metrics import loss_metadata  # noqa: E402
from src.utils.seed import resolve_device, seed_everything  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", default=None, help="Required only for legacy checkpoints without embedded config.")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("overrides", nargs="*", help="Optional OmegaConf dot-list overrides.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    if args.config is not None:
        cfg = load_config(args.config, args.overrides)
    elif checkpoint.get("config") is not None:
        cfg = OmegaConf.create(checkpoint["config"])
        if args.overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
        validate_config(cfg)
    else:
        raise ValueError("legacy checkpoint has no embedded config; pass --config")

    loss_metadata(str(cfg.loss.name), str(cfg.model.name))
    seed_everything(int(cfg.experiment.seed))
    device = resolve_device(str(cfg.experiment.device))
    split_payloads = load_split_payloads(cfg.data, map_location="cpu")
    dataloaders = build_dataloaders(cfg.data, split_payloads)
    model = build_model(cfg.model, cfg.data, cfg.loss, split_payloads["train"]).to(device)
    restore_checkpoint(checkpoint, model, device)
    criterion = build_loss(cfg.loss)
    selected = dataloaders if args.split == "all" else {args.split: dataloaders[args.split]}
    metrics = {
        split: evaluate_model(model, loader, criterion, device, cfg.get("metrics"))
        for split, loader in selected.items()
    }
    output = args.output or args.checkpoint.resolve().parents[1] / f"evaluation_{args.split}.json"
    write_json(output, {"checkpoint": str(args.checkpoint), "metrics_by_split": metrics})
    print(f"evaluation_json: {output}")


if __name__ == "__main__":
    main()
