#!/usr/bin/env python
"""Generate the configured McKean_Vlasov random-field dataset as HDF5."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.mckean_vlasov_generation import generate_mckean_vlasov_dataset
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/generate.yaml")
    parser.add_argument("--data_size", "--data-size", dest="data_size", type=int, default=None)
    parser.add_argument("--sample_size", "--sample-size", dest="sample_size", type=int, default=None)
    parser.add_argument("--grid_size", "--grid-size", dest="grid_size", type=int, default=None)
    parser.add_argument("overrides", nargs="*", help="YAML dot-list overrides")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    output_dir = Path(str(cfg.output_dir))
    if not output_dir.is_absolute():
        cfg.output_dir = str(ROOT / output_dir)
    data_size = int(args.data_size if args.data_size is not None else cfg.data_size)
    sample_size = int(args.sample_size if args.sample_size is not None else cfg.sample_size)
    grid_size = int(args.grid_size if args.grid_size is not None else cfg.grid_size)
    result = generate_mckean_vlasov_dataset(
        cfg,
        data_size=data_size,
        sample_size=sample_size,
        grid_size=grid_size,
    )
    print(f"hdf5_path: {result['hdf5_path']}", flush=True)
    print(f"input_field_particles_shape: {result['input_field_particles_shape']}", flush=True)


if __name__ == "__main__":
    main()
