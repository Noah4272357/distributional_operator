#!/usr/bin/env python3
"""Plot predictive means and pointwise 90% quantile envelopes."""

from __future__ import annotations

import argparse
import json
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
from matplotlib.patches import Patch

from scripts.train import PairedHdf5Dataset, build_training_dataloaders
from src.models import build_model
from src.utils.checkpoint import load_checkpoint
from src.utils.seed import seed_everything


TARGET_COLOR = "#0072B2"
PREDICTION_COLOR = "#D55E00"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="trained checkpoint containing the model configuration and weights",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="PNG output path (default: mean_quantile_envelopes.png beside checkpoint)",
    )
    parser.add_argument(
        "--validation-indices",
        type=int,
        nargs="+",
        default=(0, 99, 199),
        metavar="INDEX",
        help="zero-based positions within the checkpoint's validation split",
    )
    parser.add_argument(
        "--generated-samples",
        type=int,
        default=1000,
        help="conditional samples drawn for each validation field (default: 1000)",
    )
    parser.add_argument("--device", help="PyTorch device; defaults to checkpoint device")
    parser.add_argument(
        "--seed",
        type=int,
        help="sampling seed; defaults to the seed stored in the checkpoint",
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args(argv)


def _spatial_grid(dataset: PairedHdf5Dataset, dimension: int) -> np.ndarray:
    """Recover the spatial domain from HDF5 metadata, with a safe index fallback."""
    for path in (dataset.target_path, dataset.input_path):
        with h5py.File(path, "r") as handle:
            raw_config = handle.attrs.get("config")
            if raw_config is None:
                continue
            if isinstance(raw_config, bytes):
                raw_config = raw_config.decode("utf-8")
            try:
                domain = json.loads(str(raw_config)).get("domain")
            except (json.JSONDecodeError, TypeError):
                continue
            if (
                isinstance(domain, list)
                and len(domain) == 2
                and float(domain[0]) < float(domain[1])
            ):
                # Periodic simulations use an endpoint-exclusive spatial grid.
                return np.linspace(
                    float(domain[0]), float(domain[1]), dimension, endpoint=False
                )
    return np.arange(dimension, dtype=np.float64)


def _summaries(samples: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = samples.detach().float().cpu()
    mean = samples.mean(dim=0)
    lower, upper = torch.quantile(
        samples, torch.tensor((0.05, 0.95)), dim=0
    )
    return mean.numpy(), lower.numpy(), upper.numpy()


def _style_axis(axis: Axes, title: str, show_xlabel: bool) -> None:
    axis.set_title(title, loc="left", fontsize=9, fontweight="semibold", pad=5)
    axis.set_ylabel(r"Initial condition $u_0(x)$")
    if show_xlabel:
        axis.set_xlabel(r"Spatial coordinate $x$")
    axis.grid(axis="y", color="#D9DEE7", linewidth=0.55, alpha=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#7A8493")
    axis.spines["bottom"].set_color("#7A8493")
    axis.tick_params(direction="out", width=0.7, length=3, labelsize=8)
    axis.margins(x=0)


def _resolve_output(checkpoint: Path, requested: Path | None) -> tuple[Path, Path]:
    png_path = (
        requested.expanduser().resolve()
        if requested is not None
        else checkpoint.parent / "mean_quantile_envelopes.png"
    )
    if png_path.suffix.lower() != ".png":
        png_path = png_path.with_suffix(".png")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    return png_path, png_path.with_suffix(".pdf")


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
        dataset = bundle.validation_dataset
        if not isinstance(dataset, PairedHdf5Dataset):
            raise TypeError("this plot requires the current paired HDF5 data API")
        positions = list(dict.fromkeys(args.validation_indices))
        if not positions:
            raise ValueError("at least one validation index is required")
        if min(positions) < 0 or max(positions) >= len(dataset):
            raise IndexError(
                f"validation indices must be between 0 and {len(dataset) - 1}"
            )

        model = build_model(config["model"], bundle.pca_state).to(device)
        load_checkpoint(checkpoint_path, model)
        model.eval()

        dimension = int(dataset.sample_shape[-1])
        x = _spatial_grid(dataset, dimension)
        rows: list[tuple[int, tuple[np.ndarray, ...], tuple[np.ndarray, ...]]] = []
        with torch.inference_mode():
            for position in positions:
                inputs, targets = dataset[position]
                if not hasattr(model, "sample"):
                    raise TypeError("checkpoint model does not expose conditional sampling")
                predictions = model.sample(
                    inputs.unsqueeze(0).to(device),
                    num_samples=args.generated_samples,
                ).squeeze(0)
                rows.append(
                    (
                        int(dataset.indices[position]),
                        _summaries(targets),
                        _summaries(predictions),
                    )
                )

        with plt.rc_context(
            {
                "font.family": "serif",
                "font.serif": ["DejaVu Serif"],
                "font.size": 8.5,
                "axes.labelsize": 8.5,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        ):
            figure, axes = plt.subplots(
                len(rows),
                1,
                figsize=(7.15, 2.05 * len(rows) + 0.65),
                sharex=True,
                squeeze=False,
            )
            figure.patch.set_facecolor("white")
            figure.subplots_adjust(
                left=0.105,
                right=0.985,
                bottom=0.085,
                top=0.845,
                hspace=0.22,
            )
            figure.suptitle(
                "Conditional-flow predictions on held-out random fields",
                fontsize=11,
                fontweight="semibold",
                y=0.985,
            )

            for row_index, (field_index, target, prediction) in enumerate(rows):
                axis = axes[row_index, 0]
                target_mean, target_lower, target_upper = target
                pred_mean, pred_lower, pred_upper = prediction
                axis.fill_between(
                    x,
                    target_lower,
                    target_upper,
                    color=TARGET_COLOR,
                    alpha=0.17,
                    linewidth=0,
                    zorder=1,
                )
                axis.fill_between(
                    x,
                    pred_lower,
                    pred_upper,
                    color=PREDICTION_COLOR,
                    alpha=0.14,
                    linewidth=0,
                    zorder=2,
                )
                axis.plot(
                    x,
                    target_mean,
                    color=TARGET_COLOR,
                    linewidth=1.65,
                    zorder=4,
                )
                axis.plot(
                    x,
                    pred_mean,
                    color=PREDICTION_COLOR,
                    linestyle=(0, (5, 2.4)),
                    linewidth=1.65,
                    zorder=5,
                )
                _style_axis(
                    axis,
                    f"Validation field {field_index}",
                    show_xlabel=row_index == len(rows) - 1,
                )

            legend_handles = (
                Line2D([], [], color=TARGET_COLOR, linewidth=1.65, label="Target mean"),
                Line2D(
                    [],
                    [],
                    color=PREDICTION_COLOR,
                    linestyle=(0, (5, 2.4)),
                    linewidth=1.65,
                    label="Generated mean",
                ),
                Patch(
                    facecolor=TARGET_COLOR,
                    alpha=0.17,
                    edgecolor="none",
                    label="Target 5–95% interval",
                ),
                Patch(
                    facecolor=PREDICTION_COLOR,
                    alpha=0.14,
                    edgecolor="none",
                    label="Generated 5–95% interval",
                ),
            )
            figure.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.945),
                ncol=4,
                frameon=False,
                fontsize=7.7,
                handlelength=2.6,
                columnspacing=1.25,
            )

            png_path, pdf_path = _resolve_output(checkpoint_path, args.output)
            figure.savefig(
                png_path,
                dpi=args.dpi,
                bbox_inches="tight",
                facecolor="white",
            )
            figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
            plt.close(figure)

        print(f"checkpoint: {checkpoint_path}")
        print(f"checkpoint_epoch: {int(checkpoint['epoch']) + 1}")
        print(f"validation_fields: {', '.join(str(row[0]) for row in rows)}")
        print(f"target_samples_per_field: {int(dataset.sample_shape[0])}")
        print(f"generated_samples_per_field: {args.generated_samples}")
        print(f"png: {png_path}")
        print(f"pdf: {pdf_path}")
    finally:
        bundle.close()


if __name__ == "__main__":
    main()
