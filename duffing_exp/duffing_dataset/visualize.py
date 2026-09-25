"""Create diagnostic figures and a Markdown report for a Duffing HDF5 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save_path_figure(handle: h5py.File, output_dir: Path) -> Path:
    time = handle["time"][:]
    x_paths = handle["X"][0]
    y_paths = handle["Y"][0]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for row, (paths, label, color) in enumerate(((x_paths, "X", "tab:blue"), (y_paths, "Y", "tab:red"))):
        for path in paths[: min(12, len(paths))]:
            axes[row, 0].plot(time, path, color=color, alpha=0.3, linewidth=0.8)
        mean, std = paths.mean(axis=0), paths.std(axis=0)
        axes[row, 1].plot(time, mean, color=color, label="mean")
        axes[row, 1].fill_between(time, mean - std, mean + std, color=color, alpha=0.25, label="mean ± std")
        axes[row, 0].set_ylabel(label)
        axes[row, 1].legend(loc="upper right")
    axes[0, 0].set_title("Representative ensemble paths")
    axes[0, 1].set_title("Pointwise ensemble statistics")
    axes[-1, 0].set_xlabel("time")
    axes[-1, 1].set_xlabel("time")
    figure.tight_layout()
    path = output_dir / "ensemble_diagnostics.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _save_parameter_figure(handle: h5py.File, output_dir: Path) -> Path:
    theta = handle["theta"][:]
    names = json.loads(handle.attrs["parameter_names"])
    figure, axes = plt.subplots(2, 3, figsize=(13, 8))
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    for axis, (left, right) in zip(axes.flat, pairs):
        axis.scatter(theta[:, left], theta[:, right], s=14, alpha=0.65)
        axis.set_xlabel(names[left])
        axis.set_ylabel(names[right])
    figure.suptitle("Latin-hypercube parameter coverage")
    figure.tight_layout()
    path = output_dir / "parameter_coverage.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def create_report(dataset_path: str | Path, output_dir: str | Path | None = None) -> Path:
    """Validate basic structure, create figures, and write a concise report."""
    dataset_path = Path(dataset_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve() if output_dir else dataset_path.parent / "report"
    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(dataset_path, "r") as handle:
        required = {"X", "Y", "theta", "time"}
        missing = required.difference(handle)
        if missing:
            raise KeyError(f"dataset is missing: {sorted(missing)}")
        x_shape, y_shape = handle["X"].shape, handle["Y"].shape
        if x_shape[0] != y_shape[0] or x_shape[2] != y_shape[2]:
            raise ValueError("X and Y measure/time dimensions do not match")
        x = handle["X"][:]
        y = handle["Y"][:]
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise FloatingPointError("dataset contains non-finite values")
        ensemble_figure = _save_path_figure(handle, output_dir)
        parameter_figure = _save_parameter_figure(handle, output_dir)
        seconds = float(handle.attrs.get("generation_seconds", np.nan))
        peak_bytes = int(handle.attrs.get("peak_process_tree_rss_bytes", 0))
        full_seconds = seconds * 1400.0 / x_shape[0]
        theta = handle["theta"][:]
        names = json.loads(handle.attrs["parameter_names"])

    parameter_lines = "\n".join(
        f"- `{name}`: [{theta[:, index].min():.6g}, {theta[:, index].max():.6g}]"
        for index, name in enumerate(names)
    )
    report = f"""# Duffing dataset generation report

- Dataset: `{dataset_path}`
- X shape: `{x_shape}`
- Y shape: `{y_shape}`
- Finite-value check: passed
- File size: {dataset_path.stat().st_size / 2**20:.2f} MiB
- Generation time: {seconds:.3f} seconds
- Aggregate peak process-tree RSS: {peak_bytes / 2**20:.2f} MiB
- Linear estimate for 1,400 measures: {full_seconds / 3600:.3f} hours

## Sample statistics

- X mean/std/range: {x.mean():.6g} / {x.std():.6g} / [{x.min():.6g}, {x.max():.6g}]
- Y mean/std/range: {y.mean():.6g} / {y.std():.6g} / [{y.min():.6g}, {y.max():.6g}]

## Sampled parameter ranges

{parameter_lines}

## Figures

- [{ensemble_figure.name}]({ensemble_figure.name})
- [{parameter_figure.name}]({parameter_figure.name})
"""
    report_path = output_dir / "report.md"
    report_path.write_text(report)
    print(report)
    print(f"Saved report to {report_path}")
    return report_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    create_report(arguments.dataset, arguments.output_dir)
