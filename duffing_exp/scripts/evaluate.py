#!/usr/bin/env python3
"""Evaluate a checkpoint without retraining."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.models import build_model
from src.training.losses import build_loss
from src.training.validate import validate
from src.utils.checkpoint import load_checkpoint
from src.utils.seed import seed_everything

from scripts.train import build_training_dataloaders


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--split", choices=("validation", "test"), default="validation"
    )
    args = parser.parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if args.device:
        config["training"]["device"] = args.device
    seed_everything(int(config["seed"]))
    project_root = Path(__file__).resolve().parents[1]
    device = torch.device(config["training"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    bundle = build_training_dataloaders(config, project_root)
    try:
        model = build_model(config["model"], bundle.pca_state).to(device)
        load_checkpoint(checkpoint_path, model)
        if args.split == "test":
            if not hasattr(bundle, "test_loader") or bundle.test_loader is None:
                raise ValueError("this dataset configuration does not define a test split")
            loader = bundle.test_loader
        else:
            loader = bundle.validation_loader
        metrics = validate(
            model,
            loader,
            build_loss(config["loss"]).to(device),
            device,
        )
        result = {
            "checkpoint": str(checkpoint_path),
            "split": args.split,
            "fields": len(loader.dataset),
            **metrics,
        }
        rendered = json.dumps(result, indent=2) + "\n"
        if args.output:
            output_path = args.output.expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
    finally:
        bundle.close()


if __name__ == "__main__":
    main()
