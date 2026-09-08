#!/usr/bin/env python
"""Evaluate a checkpoint without retraining."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataloader import build_dataloaders
from src.data.dataset import load_random_field_dataset_splits, make_random_field_datasets
from src.training import pipeline
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    device = torch.device(args.device or str(cfg.experiment.device))
    split_payloads = load_random_field_dataset_splits(cfg.generated_data)
    datasets = make_random_field_datasets(split_payloads, str(cfg.random_field_view.input_representation))
    loaders = build_dataloaders(datasets, cfg.experiment)
    stats = pipeline._standardization_stats(
        split_payloads["train"],
        str(cfg.random_field_view.input_representation),
        pipeline._standardization_metadata(cfg.get("standardization")),
    )
    init_payload = pipeline._standardized_model_init_payload(split_payloads["train"], stats)
    base_model = pipeline.build_random_field_model(cfg.model, cfg.loss, init_payload, str(cfg.random_field_view.input_representation))
    model = pipeline._StandardizedRandomFieldModel(base_model, **stats).to(device)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else run_dir / "artifacts" / "model_ckpt" / "best.pt"
    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = pipeline.evaluate_random_field_model(
        model,
        loaders[args.split],
        str(cfg.loss.name),
        cfg.loss,
        device,
        pipeline._load_field_metric_context(cfg.generated_data),
        cfg.get("metrics"),
    )
    print(json.dumps({"split": args.split, "metrics": metrics}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
