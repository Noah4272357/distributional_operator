"""Plot training and distribution-validation metrics from an experiment CSV."""

from __future__ import annotations

import sys
from pathlib import Path

# When this file is executed directly, Python prepends ``src/utils`` to
# sys.path. That makes the sibling ``logging.py`` shadow the standard-library
# logging module imported by Pillow/Matplotlib. Package execution does not
# have this problem, so only sanitize the direct-script path.
if __package__ in {None, ""}:
    script_directory = Path(__file__).resolve().parent
    sys.path[:] = [
        entry
        for entry in sys.path
        if Path(entry or ".").resolve() != script_directory
    ]

import argparse
import csv
import math
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


REQUIRED_COLUMNS = (
    "epoch",
    "train_loss",
    "validation_sinkhorn_distance",
    "validation_mmd",
    "validation_sliced_wasserstein",
    "validation_energy_distance",
)


def resolve_metrics_csv(path: str | Path) -> Path:
    """Resolve a CSV, experiment directory, or experiments root to metrics.csv."""
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        return candidate
    direct_metrics = candidate / "metrics.csv"
    if direct_metrics.is_file():
        return direct_metrics
    if not candidate.is_dir():
        raise FileNotFoundError(f"metrics path does not exist: {candidate}")
    metrics_files = list(candidate.rglob("metrics.csv"))
    if not metrics_files:
        raise FileNotFoundError(f"no metrics.csv found under: {candidate}")
    return max(metrics_files, key=lambda item: item.stat().st_mtime)


def load_metrics(metrics_csv: str | Path) -> dict[str, list[float]]:
    """Load and validate the columns required by the visualization."""
    path = resolve_metrics_csv(metrics_csv)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = [name for name in REQUIRED_COLUMNS if name not in columns]
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {', '.join(missing)}"
            )
        values = {name: [] for name in REQUIRED_COLUMNS}
        for row_number, row in enumerate(reader, start=2):
            try:
                for name in REQUIRED_COLUMNS:
                    values[name].append(float(row[name]))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{path}:{row_number} contains a non-numeric metric"
                ) from error
    if not values["epoch"]:
        raise ValueError(f"{path} contains no metric rows")
    if any(
        not math.isfinite(value)
        for column_values in values.values()
        for value in column_values
    ):
        raise ValueError(f"{path} contains a non-finite metric")
    if any(
        current <= previous
        for previous, current in zip(values["epoch"], values["epoch"][1:])
    ):
        raise ValueError("epoch values must be strictly increasing")
    return values


def _style_axis(axis: Axes, title: str, ylabel: str) -> None:
    axis.set_title(title, loc="left", fontsize=11, fontweight="semibold")
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#D9DEE7", linewidth=0.8, alpha=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#8B95A7")
    axis.spines["bottom"].set_color("#8B95A7")
    axis.tick_params(colors="#4A5568", labelsize=9)


def plot_metrics(
    metrics_csv: str | Path = "experiments",
    output_path: str | Path | None = None,
    *,
    dpi: int = 180,
    show: bool = False,
) -> Path:
    """Create a four-panel training-metrics figure and return its output path."""
    if dpi < 1:
        raise ValueError("dpi must be positive")
    csv_path = resolve_metrics_csv(metrics_csv)
    metrics = load_metrics(csv_path)
    epochs = metrics["epoch"]
    output = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else csv_path.with_name("training_metrics.png")
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    figure.patch.set_facecolor("white")
    figure.suptitle(
        f"Training and Validation Metrics — {csv_path.parent.name}",
        fontsize=15,
        fontweight="semibold",
        color="#1F2937",
    )

    loss_axis = axes[0, 0]
    loss_axis.plot(
        epochs,
        metrics["train_loss"],
        label="Train",
        color="#2563EB",
        linewidth=2,
    )
    loss_axis.plot(
        epochs,
        metrics["validation_sinkhorn_distance"],
        label="Validation Sinkhorn",
        color="#D97706",
        linewidth=2,
    )
    loss_values = metrics["train_loss"] + metrics["validation_sinkhorn_distance"]
    loss_ylabel = "Loss"
    if min(loss_values) > 0 and max(loss_values) / min(loss_values) >= 100:
        loss_axis.set_yscale("log")
        loss_ylabel = "Loss (log scale)"
    _style_axis(loss_axis, "Training Loss and Validation Sinkhorn", loss_ylabel)
    loss_axis.legend(frameon=False, ncol=2, fontsize=9)

    panels = (
        (axes[0, 1], "validation_mmd", "Validation MMD", "MMD", "#2563EB"),
        (
            axes[1, 0],
            "validation_sliced_wasserstein",
            "Validation Sliced Wasserstein",
            "Distance",
            "#B45309",
        ),
        (
            axes[1, 1],
            "validation_energy_distance",
            "Validation Energy Distance",
            "Distance",
            "#6B7280",
        ),
    )
    for axis, column, title, ylabel, color in panels:
        axis.plot(epochs, metrics[column], color=color, linewidth=2)
        _style_axis(axis, title, ylabel)

    for axis in axes.flat:
        axis.margins(x=0.02)

    figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(figure)
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metrics",
        nargs="?",
        default=Path("experiments"),
        type=Path,
        help="metrics.csv, experiment directory, or experiments root",
    )
    parser.add_argument("--output", type=Path, help="output PNG path")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    metrics_csv = resolve_metrics_csv(args.metrics)
    output = plot_metrics(
        metrics_csv,
        args.output,
        dpi=args.dpi,
        show=args.show,
    )
    print(f"metrics: {metrics_csv}")
    print(f"figure: {output}")


if __name__ == "__main__":
    main()
