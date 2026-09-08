#!/usr/bin/env python3
"""Run the controlled Gaussian-law model comparison across random seeds."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_LOSSES = {
    "global_constant": "gaussian_nll",
    "oracle_feature": "gaussian_nll",
    "pathwise_mlp": "pathwise_mse",
    "cylindrical": "gaussian_nll",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODEL_LOSSES, default=list(MODEL_LOSSES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--prefix", default="cosine5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for seed in args.seeds:
        for model in args.models:
            run_name = f"{args.prefix}_{model}_seed{seed}"
            result_path = PROJECT_ROOT / "experiments" / run_name / "results.json"
            if result_path.is_file():
                print(f"skip completed run: {run_name}", flush=True)
                continue
            command = [
                sys.executable,
                "scripts/train.py",
                "--run-name",
                run_name,
                f"model.name={model}",
                f"loss.name={MODEL_LOSSES[model]}",
                f"experiment.seed={seed}",
            ]
            print(f"run: {run_name}", flush=True)
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
