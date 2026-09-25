#!/usr/bin/env python3
"""Generate drifted Brownian paths from parameter pairs in an ISI dataset."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/generated/isi_distribution_dataset.h5"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/generated/isi_process_data.h5"),
    )
    parser.add_argument("--paths-per-law", type=int, default=200)
    parser.add_argument("--time-points", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--law-batch-size", type=int, default=8)
    parser.add_argument("--compression-level", type=int, default=4)
    return parser.parse_args(argv)


def _read_source(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        if "params" not in handle or "bin_edges" not in handle:
            raise ValueError("source must contain params and bin_edges")
        params = handle["params"][...]
        bin_edges = handle["bin_edges"][...]
        if params.ndim != 2 or params.shape[1] != 2:
            raise ValueError(f"params must have shape [law_count, 2], got {params.shape}")
        if params.shape[0] != 1200:
            raise ValueError(f"expected 1200 parameter laws, got {params.shape[0]}")
        if not np.isfinite(params).all():
            raise ValueError("params contains non-finite values")
        if np.any(params[:, 1] < 0.0):
            raise ValueError("diffusion variance q must be nonnegative")
        if bin_edges.ndim != 1 or bin_edges.size < 2:
            raise ValueError("bin_edges must be a one-dimensional array")
        if not np.isclose(bin_edges[0], 0.0):
            raise ValueError("bin_edges must start at zero")
        horizon = float(bin_edges[-1])
        if not math.isfinite(horizon) or horizon <= 0.0:
            raise ValueError("the source time horizon must be positive and finite")

        law_ids = (
            handle["law_ids"][...]
            if "law_ids" in handle
            else np.arange(params.shape[0], dtype=np.int64)
        )
        regime_labels = handle["regime_labels"][...] if "regime_labels" in handle else None
        source_attrs = {str(key): handle.attrs[key] for key in handle.attrs}
    if law_ids.shape != (params.shape[0],):
        raise ValueError("law_ids must have one entry per parameter law")
    if regime_labels is not None and regime_labels.shape != (params.shape[0],):
        raise ValueError("regime_labels must have one entry per parameter law")
    return {
        "params": params.astype(np.float32, copy=False),
        "bin_edges": bin_edges,
        "horizon": horizon,
        "law_ids": law_ids,
        "regime_labels": regime_labels,
        "source_attrs": source_attrs,
    }


def _law_paths(
    m: float,
    q: float,
    law_id: int,
    *,
    time_grid: np.ndarray,
    paths_per_law: int,
    master_seed: int,
) -> np.ndarray:
    time_points = int(time_grid.size)
    dt = float(time_grid[1] - time_grid[0])
    seed_sequence = np.random.SeedSequence([int(master_seed), int(law_id)])
    generator = np.random.default_rng(seed_sequence)
    standard_normals = generator.standard_normal((paths_per_law, time_points - 1))
    brownian_increments = standard_normals.astype(np.float32) * np.float32(math.sqrt(dt))
    brownian = np.empty((paths_per_law, time_points), dtype=np.float32)
    brownian[:, 0] = 0.0
    np.cumsum(brownian_increments, axis=1, dtype=np.float32, out=brownian[:, 1:])
    return (
        np.float32(m) * time_grid[None, :]
        + np.float32(math.sqrt(q)) * brownian
    ).astype(np.float32, copy=False)


def _validation_summary(
    path: Path,
    source: dict[str, Any],
    *,
    paths_per_law: int,
    time_points: int,
    law_batch_size: int,
) -> dict[str, float]:
    params = source["params"].astype(np.float64)
    horizon = float(source["horizon"])
    dt = horizon / (time_points - 1)
    max_terminal_mean_z = 0.0
    terminal_variance_ratios: list[np.ndarray] = []
    max_increment_mean_z = 0.0
    increment_variance_ratios: list[np.ndarray] = []

    with h5py.File(path, "r") as handle:
        samples = handle["process_samples"]
        time_grid = handle["time_grid"][...]
        expected_shape = (params.shape[0], paths_per_law, time_points)
        if samples.shape != expected_shape:
            raise RuntimeError(f"process_samples has shape {samples.shape}, expected {expected_shape}")
        if time_grid.shape != (time_points,):
            raise RuntimeError("time_grid has the wrong shape")
        if not np.isclose(time_grid[0], 0.0) or not np.isclose(time_grid[-1], horizon):
            raise RuntimeError("time_grid endpoints do not match the source horizon")
        if not np.allclose(
            np.diff(time_grid.astype(np.float64)),
            dt,
            rtol=1.0e-5,
            atol=5.0e-7,
        ):
            raise RuntimeError("time_grid is not uniformly spaced")
        if not np.array_equal(handle["params"][...], source["params"]):
            raise RuntimeError("copied params do not match the source")
        if not np.array_equal(handle["law_ids"][...], source["law_ids"]):
            raise RuntimeError("copied law_ids do not match the source")

        for start in range(0, params.shape[0], law_batch_size):
            stop = min(start + law_batch_size, params.shape[0])
            block = samples[start:stop].astype(np.float64)
            if not np.isfinite(block).all():
                raise RuntimeError(f"non-finite process value in laws {start}:{stop}")
            if not np.allclose(block[:, :, 0], 0.0, atol=1.0e-7):
                raise RuntimeError(f"nonzero initial value in laws {start}:{stop}")

            m = params[start:stop, 0]
            q = params[start:stop, 1]
            terminal = block[:, :, -1]
            expected_terminal_mean = m * horizon
            expected_terminal_variance = q * horizon
            terminal_mean_se = np.sqrt(expected_terminal_variance / paths_per_law)
            terminal_mean_z = np.abs(terminal.mean(axis=1) - expected_terminal_mean) / terminal_mean_se
            terminal_variance_ratio = terminal.var(axis=1, ddof=1) / expected_terminal_variance
            max_terminal_mean_z = max(max_terminal_mean_z, float(terminal_mean_z.max()))
            terminal_variance_ratios.append(terminal_variance_ratio)

            increments = np.diff(block, axis=2)
            increment_count = paths_per_law * (time_points - 1)
            expected_increment_mean = m * dt
            expected_increment_variance = q * dt
            increment_mean_se = np.sqrt(expected_increment_variance / increment_count)
            increment_mean = increments.mean(axis=(1, 2))
            increment_mean_z = np.abs(increment_mean - expected_increment_mean) / increment_mean_se
            increment_variance = increments.var(axis=(1, 2), ddof=1)
            increment_variance_ratio = increment_variance / expected_increment_variance
            max_increment_mean_z = max(max_increment_mean_z, float(increment_mean_z.max()))
            increment_variance_ratios.append(increment_variance_ratio)

    terminal_ratios = np.concatenate(terminal_variance_ratios)
    increment_ratios = np.concatenate(increment_variance_ratios)
    if max_terminal_mean_z > 6.0:
        raise RuntimeError(f"terminal mean validation failed: max z-score {max_terminal_mean_z:.3f}")
    if terminal_ratios.min() < 0.5 or terminal_ratios.max() > 1.7:
        raise RuntimeError("terminal variance validation failed")
    if max_increment_mean_z > 6.0:
        raise RuntimeError(f"increment mean validation failed: max z-score {max_increment_mean_z:.3f}")
    if increment_ratios.min() < 0.9 or increment_ratios.max() > 1.1:
        raise RuntimeError("increment variance validation failed")
    return {
        "max_terminal_mean_z": max_terminal_mean_z,
        "min_terminal_variance_ratio": float(terminal_ratios.min()),
        "max_terminal_variance_ratio": float(terminal_ratios.max()),
        "max_increment_mean_z": max_increment_mean_z,
        "min_increment_variance_ratio": float(increment_ratios.min()),
        "max_increment_variance_ratio": float(increment_ratios.max()),
    }


def generate(
    source_path: Path,
    output_path: Path,
    *,
    paths_per_law: int,
    time_points: int,
    seed: int,
    law_batch_size: int,
    compression_level: int,
) -> dict[str, Any]:
    if paths_per_law < 1:
        raise ValueError("paths_per_law must be positive")
    if time_points < 2:
        raise ValueError("time_points must be at least two")
    if law_batch_size < 1:
        raise ValueError("law_batch_size must be positive")
    if not 0 <= compression_level <= 9:
        raise ValueError("compression_level must be between zero and nine")

    source_path = source_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    source = _read_source(source_path)
    law_count = int(source["params"].shape[0])
    horizon = float(source["horizon"])
    time_grid = np.linspace(0.0, horizon, time_points, dtype=np.float32)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with h5py.File(temporary_path, "w") as handle:
            handle.attrs["format"] = "isi_drifted_brownian_paths"
            handle.attrs["format_version"] = 1
            handle.attrs["source_path"] = str(source_path)
            handle.attrs["source_format"] = str(source["source_attrs"].get("format", "unknown"))
            handle.attrs["source_format_version"] = int(
                source["source_attrs"].get("format_version", 0)
            )
            handle.attrs["equation"] = "dX_t = m dt + sqrt(q) dW_t"
            handle.attrs["initial_value"] = 0.0
            handle.attrs["t_max"] = horizon
            handle.attrs["time_points"] = time_points
            handle.attrs["dt"] = horizon / (time_points - 1)
            handle.attrs["paths_per_law"] = paths_per_law
            handle.attrs["master_seed"] = int(seed)
            handle.attrs["seed_rule"] = "SeedSequence([master_seed, law_id])"
            handle.attrs["completed"] = False

            handle.create_dataset("time_grid", data=time_grid)
            handle.create_dataset("params", data=source["params"])
            handle.create_dataset("law_ids", data=source["law_ids"])
            if source["regime_labels"] is not None:
                handle.create_dataset("regime_labels", data=source["regime_labels"])
            process_samples = handle.create_dataset(
                "process_samples",
                shape=(law_count, paths_per_law, time_points),
                dtype=np.float32,
                chunks=(1, paths_per_law, time_points),
                compression="gzip" if compression_level > 0 else None,
                compression_opts=compression_level if compression_level > 0 else None,
                shuffle=compression_level > 0,
            )

            for start in range(0, law_count, law_batch_size):
                stop = min(start + law_batch_size, law_count)
                batch = np.empty((stop - start, paths_per_law, time_points), dtype=np.float32)
                for offset, index in enumerate(range(start, stop)):
                    m, q = source["params"][index]
                    batch[offset] = _law_paths(
                        float(m),
                        float(q),
                        int(source["law_ids"][index]),
                        time_grid=time_grid,
                        paths_per_law=paths_per_law,
                        master_seed=seed,
                    )
                process_samples[start:stop] = batch
                if stop == law_count or stop % 100 == 0:
                    print(f"generated_laws: {stop}/{law_count}", flush=True)
            handle.flush()

        validation = _validation_summary(
            temporary_path,
            source,
            paths_per_law=paths_per_law,
            time_points=time_points,
            law_batch_size=law_batch_size,
        )
        with h5py.File(temporary_path, "r+") as handle:
            handle.attrs["validation_summary"] = json.dumps(validation, sort_keys=True)
            handle.attrs["completed"] = True
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    return {
        "output": output_path,
        "shape": (law_count, paths_per_law, time_points),
        "time_range": (0.0, horizon),
        "validation": validation,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = generate(
        args.source,
        args.output,
        paths_per_law=int(args.paths_per_law),
        time_points=int(args.time_points),
        seed=int(args.seed),
        law_batch_size=int(args.law_batch_size),
        compression_level=int(args.compression_level),
    )
    print(f"output: {result['output']}")
    print(f"process_samples_shape: {result['shape']}")
    print(f"time_range: {result['time_range']}")
    print(f"validation: {json.dumps(result['validation'], sort_keys=True)}")


if __name__ == "__main__":
    main()
