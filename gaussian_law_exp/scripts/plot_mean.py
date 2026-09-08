#!/usr/bin/env python3
"""Plot predicted versus true Gaussian means for a trained checkpoint."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import build_dataloaders, load_split_payloads  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.training.common import move_batch_to_device  # noqa: E402
from src.utils.checkpoint import load_checkpoint, restore_checkpoint  # noqa: E402
from src.utils.config import validate_config  # noqa: E402
from src.utils.seed import resolve_device, seed_everything  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a checkpoint containing the trained model and its config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output filename or stem. Both PDF and 400-dpi PNG files are written. "
            "Defaults to <experiment>/mean_prediction_scatter."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override (for example, cpu or cuda:0).",
    )
    return parser.parse_args(argv)


def _output_stem(checkpoint_path: Path, output: Path | None) -> Path:
    if output is None:
        return checkpoint_path.resolve().parents[1] / "mean_prediction_scatter"
    output = output.expanduser().resolve()
    return output.with_suffix("") if output.suffix.lower() in {".pdf", ".png"} else output


@torch.inference_mode()
def collect_means(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect target and predicted means for every distribution in a loader."""
    target_batches: list[torch.Tensor] = []
    prediction_batches: list[torch.Tensor] = []
    model.eval()
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        prediction = model(batch)
        if "pred_mean" not in prediction:
            raise KeyError("model output must contain 'pred_mean'")
        target_batches.append(batch["target_mean"].detach().cpu())
        prediction_batches.append(prediction["pred_mean"].detach().cpu())

    if not target_batches:
        raise ValueError("the test data loader is empty")
    targets = torch.cat(target_batches).numpy()
    predictions = torch.cat(prediction_batches).numpy()
    if targets.shape != predictions.shape:
        raise ValueError(
            f"target and predicted means differ in shape: {targets.shape} != {predictions.shape}"
        )
    return targets, predictions


def _component_statistics(target: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    mae = float(np.mean(np.abs(prediction - target)))
    residual_sum = float(np.sum((prediction - target) ** 2))
    total_sum = float(np.sum((target - np.mean(target)) ** 2))
    r_squared = float("nan") if total_sum == 0.0 else 1.0 - residual_sum / total_sum
    return mae, r_squared


def plot_means(targets: np.ndarray, predictions: np.ndarray, output_stem: Path) -> None:
    """Render and export the four-component publication figure."""
    if targets.ndim != 2 or targets.shape[1] != 4:
        raise ValueError(f"plot_mean expects four-dimensional means; received {targets.shape}")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(2, 2, figsize=(7.1, 6.7), constrained_layout=True)
    point_color = "#35618D"
    reference_color = "#303030"
    panel_labels = ("(a)", "(b)", "(c)", "(d)")

    for component, ax in enumerate(axes.flat):
        target = targets[:, component]
        prediction = predictions[:, component]
        data_min = float(min(target.min(), prediction.min()))
        data_max = float(max(target.max(), prediction.max()))
        span = data_max - data_min
        padding = 0.06 * span if span > 0.0 else max(abs(data_min) * 0.06, 0.1)
        limits = (data_min - padding, data_max + padding)
        mae, r_squared = _component_statistics(target, prediction)

        ax.scatter(
            target,
            prediction,
            s=18,
            color=point_color,
            alpha=0.72,
            edgecolors="white",
            linewidths=0.25,
            rasterized=True,
            zorder=2,
        )
        ax.plot(limits, limits, linestyle=(0, (4, 3)), color=reference_color, linewidth=1.0, zorder=1)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(rf"True mean, $\mu_{component + 1}$")
        ax.set_ylabel(rf"Predicted mean, $\hat{{\mu}}_{component + 1}$")
        ax.set_title(rf"Mean component $\mu_{component + 1}$", pad=7)
        ax.grid(True, color="#D8D8D8", linewidth=0.5, alpha=0.65)
        ax.set_axisbelow(True)
        ax.text(
            0.04,
            0.95,
            panel_labels[component],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
        r2_text = "undefined" if math.isnan(r_squared) else f"{r_squared:.3f}"
        ax.text(
            0.96,
            0.05,
            rf"MAE = {mae:.3f}" + "\n" + rf"$R^2$ = {r2_text}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color="#202020",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.0},
        )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=400, bbox_inches="tight")
    plt.close(figure)
    print(f"test_distributions: {targets.shape[0]}")
    print(f"pdf: {pdf_path}")
    print(f"png: {png_path}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if checkpoint.get("config") is None:
        raise ValueError("checkpoint has no embedded config; it cannot reproduce the trained data split")

    cfg = OmegaConf.create(checkpoint["config"])
    validate_config(cfg)
    seed_everything(int(cfg.experiment.seed))
    device = resolve_device(args.device or str(cfg.experiment.device))
    split_payloads = load_split_payloads(cfg.data, map_location="cpu")
    dataloaders = build_dataloaders(cfg.data, split_payloads)
    model = build_model(cfg.model, cfg.data, cfg.loss, split_payloads["train"]).to(device)
    restore_checkpoint(checkpoint, model, device)

    targets, predictions = collect_means(model, dataloaders["test"], device)
    plot_means(targets, predictions, _output_stem(checkpoint_path, args.output))


if __name__ == "__main__":
    main()
