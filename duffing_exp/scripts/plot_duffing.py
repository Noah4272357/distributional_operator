#!/usr/bin/env python3
"""Plot Duffing predictive means and pointwise 90% quantile envelopes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from scripts.train import PairedHdf5Dataset, build_training_dataloaders
from src.models import build_model
from src.utils.checkpoint import load_checkpoint
from src.utils.seed import seed_everything


TARGET_COLOR = "#1F4E79"
PREDICTION_COLOR = "#D97706"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="trained Duffing conditional-flow checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="PNG output path (default: duffing_mean_quantile_envelopes.png beside checkpoint)",
    )
    parser.add_argument(
        "--test-indices",
        type=int,
        nargs="+",
        default=(0, 66, 133),
        metavar="INDEX",
        help="exactly three unique zero-based positions within the checkpoint's test split (default: 0 66 133)",
    )
    parser.add_argument(
        "--generated-samples",
        type=int,
        default=1000,
        help="conditional output paths drawn for each test measure (default: 1000)",
    )
    parser.add_argument("--device", help="PyTorch device; defaults to checkpoint device")
    parser.add_argument(
        "--seed",
        type=int,
        help="sampling seed; defaults to the seed stored in the checkpoint",
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args(argv)


def _duffing_time_grid(dataset: PairedHdf5Dataset, dimension: int) -> np.ndarray:
    """Read and validate the time grid from Duffing HDF5."""
    for path in dict.fromkeys((dataset.target_path, dataset.input_path)):
        with h5py.File(path, "r") as handle:
            if "time" not in handle:
                continue
            time_grid = np.asarray(handle["time"][:], dtype=np.float64)
            if time_grid.shape != (dimension,):
                raise ValueError(
                    f"Duffing time grid must have shape ({dimension},), got {time_grid.shape}"
                )
            if not np.all(np.isfinite(time_grid)) or not np.all(np.diff(time_grid) > 0):
                raise ValueError("Duffing time grid must be finite and strictly increasing")
            return time_grid
    raise KeyError("configured HDF5 file does not contain the Duffing 'time' dataset")


def _summaries(samples: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = samples.detach().float().cpu()
    mean = samples.mean(dim=0)
    lower, upper = torch.quantile(samples, torch.tensor((0.05, 0.95)), dim=0)
    return mean.numpy(), lower.numpy(), upper.numpy()


def _panel_limits(
    target: tuple[np.ndarray, np.ndarray, np.ndarray],
    prediction: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[float, float]:
    lower = min(float(target[1].min()), float(prediction[1].min()))
    upper = max(float(target[2].max()), float(prediction[2].max()))
    padding = max(0.06 * (upper - lower), 1.0e-4)
    return lower - padding, upper + padding


def _style_axis(axis: Axes, title: str) -> None:
    axis.set_title(title, fontsize=8.7, pad=5)
    axis.grid(axis="y", color="#D9DDE3", linewidth=0.55)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#4B5563")
    axis.spines["bottom"].set_color("#4B5563")
    axis.tick_params(direction="out", width=0.7, length=3, labelsize=8)
    axis.margins(x=0)


def _resolve_output(checkpoint: Path, requested: Path | None) -> Path:
    png_path = (
        requested.expanduser().resolve()
        if requested is not None
        else checkpoint.parent / "duffing_mean_quantile_envelopes.png"
    )
    if png_path.suffix.lower() != ".png":
        png_path = png_path.with_suffix(".png")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    return png_path


def main() -> None:
    args = parse_args()
    if args.generated_samples < 1:
        raise ValueError("--generated-samples must be positive")
    if args.dpi < 1:
        raise ValueError("--dpi must be positive")

    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    sampling_seed = int(config["seed"] if args.seed is None else args.seed)
    seed_everything(sampling_seed)

    requested_device = args.device or config["training"]["device"]
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    project_root = Path(__file__).resolve().parent
    bundle = build_training_dataloaders(config, project_root)
    try:
        dataset = bundle.test_dataset
        if dataset is None:
            raise ValueError("checkpoint configuration must specify data.test_size > 0")
        if not isinstance(dataset, PairedHdf5Dataset):
            raise TypeError("this plot requires the paired HDF5 data API")
        if dataset.input_dataset != "X" or dataset.target_dataset != "Y":
            raise ValueError("Duffing plotting requires input_dataset='X' and target_dataset='Y'")

        positions = list(dict.fromkeys(args.test_indices))
        if len(positions) != 3:
            raise ValueError("exactly three unique test indices are required for the 1x3 figure")
        if min(positions) < 0 or max(positions) >= len(dataset):
            raise IndexError(
                f"test indices must be between 0 and {len(dataset) - 1}"
            )

        model = build_model(config["model"], bundle.pca_state).to(device)
        load_checkpoint(checkpoint_path, model)
        model.eval()
        if not hasattr(model, "sample"):
            raise TypeError("checkpoint model does not expose conditional sampling")

        dimension = int(dataset.sample_shape[-1])
        time_grid = _duffing_time_grid(dataset, dimension)
        rows: list[tuple[int, tuple[np.ndarray, ...], tuple[np.ndarray, ...]]] = []
        with torch.inference_mode():
            for position in positions:
                inputs, targets = dataset[position]
                predictions = model.sample(
                    inputs.unsqueeze(0).to(device),
                    num_samples=args.generated_samples,
                ).squeeze(0)
                rows.append(
                    (
                        position,
                        _summaries(targets),
                        _summaries(predictions),
                    )
                )

        with plt.rc_context(
            {
                "font.family": "serif",
                "font.serif": ["DejaVu Serif"],
                "font.size": 9,
                "axes.labelsize": 9,
                "axes.titlesize": 8.7,
            }
        ):
            figure, axes = plt.subplots(
                1,
                3,
                figsize=(10.8, 3.5),
                sharex=True,
                squeeze=False,
            )
            figure.patch.set_facecolor("white")

            for panel_column, (test_index, target, prediction) in enumerate(rows):
                axis = axes[0, panel_column]
                target_mean, target_lower, target_upper = target
                pred_mean, pred_lower, pred_upper = prediction
                axis.fill_between(
                    time_grid,
                    target_lower,
                    target_upper,
                    color=TARGET_COLOR,
                    alpha=0.14,
                    linewidth=0,
                )
                axis.fill_between(
                    time_grid,
                    pred_lower,
                    pred_upper,
                    facecolor=PREDICTION_COLOR,
                    edgecolor=PREDICTION_COLOR,
                    alpha=0.12,
                    hatch="////",
                    linewidth=0.35,
                )
                axis.plot(time_grid, target_mean, color=TARGET_COLOR, linewidth=1.8)
                axis.plot(
                    time_grid,
                    pred_mean,
                    color=PREDICTION_COLOR,
                    linestyle="--",
                    linewidth=1.8,
                )
                axis.set_xlim(float(time_grid[0]), float(time_grid[-1]))
                axis.set_ylim(*_panel_limits(target, prediction))
                _style_axis(axis, f"Test Index {test_index}")
                axis.set_xlabel(r"Time $t$")
                if panel_column == 0:
                    axis.set_ylabel(r"Displacement $Y(t)$")

            legend_handles = (
                Line2D([], [], color=TARGET_COLOR, linewidth=1.8, label="Target"),
                Line2D(
                    [],
                    [],
                    color=PREDICTION_COLOR,
                    linestyle="--",
                    linewidth=1.8,
                    label="Conditional flow",
                ),
            )
            figure.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.99),
                ncol=2,
                frameon=False,
                handlelength=2.8,
            )
            figure.text(
                0.5,
                0.89,
                "Lines show pointwise means; bands show empirical 5th–95th percentiles.",
                ha="center",
                va="top",
                fontsize=8.2,
                color="#374151",
            )
            figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.79), w_pad=1.8)

            png_path = _resolve_output(checkpoint_path, args.output)
            figure.savefig(png_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
            plt.close(figure)

        print(f"checkpoint: {checkpoint_path}")
        print(f"checkpoint_epoch: {int(checkpoint['epoch']) + 1}")
        print(f"test_indices: {', '.join(str(row[0]) for row in rows)}")
        print(f"target_paths_per_measure: {int(dataset.sample_shape[0])}")
        print(f"generated_paths_per_measure: {args.generated_samples}")
        print(f"time_interval: [{time_grid[0]:.6g}, {time_grid[-1]:.6g}]")
        print(f"png: {png_path}")
    finally:
        bundle.close()


if __name__ == "__main__":
    main()
