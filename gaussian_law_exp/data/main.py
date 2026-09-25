"""Command-line entry point for synthetic distribution-pair generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import h5py


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.generation import generate_dataset  # noqa: E402


OUTPUT_PATH = Path(__file__).with_name("dataset.h5")


def build_parser() -> argparse.ArgumentParser:
    """Build the dataset-generation argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate paired synthetic distributions."
    )
    parser.add_argument(
        "--data_size",
        "--data-size",
        dest="data_size",
        type=int,
        default=10,
        help="number of distributions to generate (default: 10)",
    )
    parser.add_argument(
        "--sample_size",
        "--sample-size",
        dest="sample_size",
        type=int,
        default=5,
        help="number of particles per distribution (default: 5)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    """Generate and save ``data/dataset.h5``."""
    args = build_parser().parse_args(argv)
    dataset = generate_dataset(args.data_size, args.sample_size)
    with h5py.File(OUTPUT_PATH, "w") as handle:
        for name, value in dataset.items():
            handle.create_dataset(
                name,
                data=value.detach().cpu().numpy(),
                compression="gzip",
            )

    print(f"Saved dataset to {OUTPUT_PATH}")
    for name, value in dataset.items():
        print(f"{name}: {tuple(value.shape)}")
    return OUTPUT_PATH


if __name__ == "__main__":
    main()
