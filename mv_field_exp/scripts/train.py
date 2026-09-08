#!/usr/bin/env python
"""Train an MV random-field model from a resolved configuration."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.trainer import Trainer
from src.utils.config import load_config, save_config
from src.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("overrides", nargs="*", help="YAML dot-list overrides, e.g. experiment.epochs=10")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_dir = Path(args.run_dir) if args.run_dir else Path(str(cfg.experiment.save_path)) / str(cfg.experiment.name) / timestamp
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    save_config(cfg, run_dir)
    logger = configure_logging(run_dir)
    logger.info("run_dir=%s", run_dir)
    logger.info("epochs=%s device=%s", cfg.experiment.epochs, cfg.experiment.device)
    results = Trainer(cfg, run_dir).fit()
    logger.info("best_epoch=%s val_score=%s", results["best_epoch"], results["metrics_by_split"]["val"]["training_loss"])
    print(json.dumps({"run_dir": str(run_dir), "best_epoch": results["best_epoch"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
