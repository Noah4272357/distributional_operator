#!/usr/bin/env python3
"""Generate a compact dataset for active ISI distribution-learning models."""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
    from .isi_generation import generate_isi_lif_dataset
else:
    from isi_generation import generate_isi_lif_dataset


DATA_ROOT = Path(__file__).resolve().parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-size", "--data_size", dest="data_size", type=int, default=1200)
    parser.add_argument("--sample-size", "--sample_size", dest="sample_size", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("generated/isi_lif_laws.h5"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-isi", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--t-max", type=float, default=8.0)
    parser.add_argument("--finite-bins", type=int, default=48)
    parser.add_argument("--feature-frequencies", type=int, default=8)
    parser.add_argument("--rff-scale", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output.expanduser()
    if not output.is_absolute():
        output = DATA_ROOT / output
    result = generate_isi_lif_dataset(
        data_size=args.data_size,
        sample_size=args.sample_size,
        output_path=output,
        seed=args.seed,
        device=args.device,
        n_isi=args.n_isi,
        dt=args.dt,
        t_max=args.t_max,
        finite_bins=args.finite_bins,
        feature_frequencies=args.feature_frequencies,
        rff_scale=args.rff_scale,
    )
    print(f"dataset: {result['dataset_path']}")
    print(f"input_particles_shape: {tuple(result['payload']['input_particles'].shape)}")
    print(f"bin_counts_shape: {tuple(result['payload']['bin_counts'].shape)}")
    print(f"input_features_shape: {tuple(result['payload']['input_features'].shape)}")


if __name__ == "__main__":
    main()
