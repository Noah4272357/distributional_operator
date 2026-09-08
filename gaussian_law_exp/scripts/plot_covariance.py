#!/usr/bin/env python3
"""Plot elementwise covariance MAE for a trained Gaussian checkpoint."""

from __future__ import annotations

import argparse
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
            "Defaults to <experiment>/covariance_mae_heatmap."
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
        return checkpoint_path.resolve().parents[1] / "covariance_mae_heatmap"
    output = output.expanduser().resolve()
    return output.with_suffix("") if output.suffix.lower() in {".pdf", ".png"} else output


@torch.inference_mode()
def collect_covariances(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect target and predicted covariance matrices from a loader."""
    target_batches: list[torch.Tensor] = []
    prediction_batches: list[torch.Tensor] = []
    model.eval()
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        prediction = model(batch)
        if "pred_cov" not in prediction:
            raise KeyError("model output must contain 'pred_cov'")
        target_batches.append(batch["target_cov"].detach().cpu())
        prediction_batches.append(prediction["pred_cov"].detach().cpu())

    if not target_batches:
        raise ValueError("the test data loader is empty")
    targets = torch.cat(target_batches).numpy()
    predictions = torch.cat(prediction_batches).numpy()
    if targets.shape != predictions.shape:
        raise ValueError(
            "target and predicted covariances differ in shape: "
            f"{targets.shape} != {predictions.shape}"
        )
    return targets, predictions


def covariance_mae(targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Calculate elementwise MAE across test covariance matrices."""
    if targets.ndim != 3 or targets.shape[1:] != (4, 4):
        raise ValueError(
            "plot_covariance expects covariance arrays with shape [K, 4, 4]; "
            f"received {targets.shape}"
        )
    return np.mean(np.abs(predictions - targets), axis=0)


def plot_covariance_error(error: np.ndarray, sample_count: int, output_stem: Path) -> None:
    """Render and export a publication-quality covariance-error heatmap."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, ax = plt.subplots(figsize=(5.4, 4.7), constrained_layout=True)
    image = ax.imshow(error, cmap="Blues", vmin=0.0, interpolation="nearest", aspect="equal")
    component_labels = [str(index) for index in range(1, 5)]
    ax.set_xticks(np.arange(4), labels=component_labels)
    ax.set_yticks(np.arange(4), labels=component_labels)
    ax.set_xlabel("Covariance column")
    ax.set_ylabel("Covariance row")
    ax.set_title("Mean absolute covariance error", pad=24, fontweight="bold")
    ax.text(
        0.5,
        1.025,
        f"Elementwise average across $K={sample_count}$ test distributions",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color="#4A4A4A",
        fontsize=9,
    )

    threshold = float(error.max()) * 0.52
    for row in range(4):
        for column in range(4):
            ax.text(
                column,
                row,
                f"{error[row, column]:.3f}",
                ha="center",
                va="center",
                color="white" if error[row, column] > threshold else "#202020",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = figure.colorbar(image, ax=ax, fraction=0.047, pad=0.04)
    colorbar.set_label("Mean absolute error")
    colorbar.outline.set_linewidth(0.8)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=400, bbox_inches="tight")
    plt.close(figure)
    print(f"test_distributions: {sample_count}")
    print(f"mean_absolute_covariance_error: {float(error.mean()):.6f}")
    print("elementwise_mae:")
    print(np.array2string(error, precision=6, suppress_small=False))
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

    targets, predictions = collect_covariances(model, dataloaders["test"], device)
    error = covariance_mae(targets, predictions)
    plot_covariance_error(error, targets.shape[0], _output_stem(checkpoint_path, args.output))


if __name__ == "__main__":
    main()
