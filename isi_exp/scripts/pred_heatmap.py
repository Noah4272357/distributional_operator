#!/usr/bin/env python3
"""Create publication-ready ground-truth and prediction heatmaps for an ISI checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import load_split_payloads  # noqa: E402
from src.models.build_model import build_model  # noqa: E402
from src.utils.checkpoint import load_checkpoint, restore_checkpoint  # noqa: E402
from src.utils.config import load_config, validate_training_config  # noqa: E402
from src.utils.seed import resolve_device, seed_everything  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None, help="Required only for legacy checkpoints.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0, help="Seed used to select rows from the split.")
    parser.add_argument("--device", default=None, help="Override the checkpoint device, e.g. cpu or cuda.")
    parser.add_argument("--output", type=Path, default=None, help="Output PDF path.")
    parser.add_argument("--dpi", type=int, default=600, help="Resolution of the companion PNG.")
    return parser.parse_args(argv)


def _checkpoint_config(checkpoint: dict[str, Any], config_path: Path | None) -> Any:
    if config_path is not None:
        return load_config(config_path)
    if checkpoint.get("config") is None:
        raise ValueError("checkpoint has no embedded config; pass --config")
    cfg = OmegaConf.create(checkpoint["config"])
    OmegaConf.resolve(cfg)
    validate_training_config(cfg)
    return cfg


def _selected_indices(
    empirical_mass: torch.Tensor,
    bin_edges: torch.Tensor,
    *,
    num_samples: int,
    seed: int,
) -> torch.Tensor:
    """Sample rows without replacement, then order them by empirical mean ISI."""
    split_size = int(empirical_mass.shape[0])
    if not 1 <= num_samples <= split_size:
        raise ValueError(f"num_samples must be between 1 and {split_size}")
    generator = torch.Generator().manual_seed(int(seed))
    selected = torch.randperm(split_size, generator=generator)[:num_samples]
    finite_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    support = torch.cat((finite_centers, bin_edges[-1:]))
    empirical_mean = (empirical_mass[selected] * support).sum(dim=-1)
    return selected[torch.argsort(empirical_mean, descending=False)]


def _plot_heatmaps(
    empirical_mass: torch.Tensor,
    predicted_mass: torch.Tensor,
    bin_edges: torch.Tensor,
    output: Path,
    *,
    dpi: int,
) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )

    truth = empirical_mass.detach().cpu().to(torch.float32).numpy()
    prediction = predicted_mass.detach().cpu().to(torch.float32).numpy()
    edges = bin_edges.detach().cpu().to(torch.float32).numpy()
    sample_count = truth.shape[0]
    vmax = max(float(truth.max()), float(prediction.max()), 1.0e-8)
    norm = Normalize(vmin=0.0, vmax=vmax)

    figure = plt.figure(figsize=(7.2, 4.5), constrained_layout=True)
    grid = figure.add_gridspec(
        1,
        6,
        width_ratios=(1.0, 0.055, 0.10, 1.0, 0.055, 0.045),
        wspace=0.08,
    )
    ax_truth = figure.add_subplot(grid[0, 0])
    ax_truth_tail = figure.add_subplot(grid[0, 1], sharey=ax_truth)
    ax_prediction = figure.add_subplot(grid[0, 3], sharey=ax_truth)
    ax_prediction_tail = figure.add_subplot(grid[0, 4], sharey=ax_truth)
    ax_colorbar = figure.add_subplot(grid[0, 5])

    image_kwargs = {
        "aspect": "auto",
        "interpolation": "nearest",
        "origin": "upper",
        "cmap": "magma",
        "norm": norm,
        "rasterized": True,
    }
    extent = (float(edges[0]), float(edges[-1]), sample_count + 0.5, 0.5)
    truth_image = ax_truth.imshow(truth[:, :-1], extent=extent, **image_kwargs)
    ax_truth_tail.imshow(truth[:, -1:], extent=(0.0, 1.0, sample_count + 0.5, 0.5), **image_kwargs)
    ax_prediction.imshow(prediction[:, :-1], extent=extent, **image_kwargs)
    ax_prediction_tail.imshow(
        prediction[:, -1:],
        extent=(0.0, 1.0, sample_count + 0.5, 0.5),
        **image_kwargs,
    )

    for axis, title in ((ax_truth, "Ground truth"), (ax_prediction, "Prediction")):
        axis.set_title(title, fontweight="semibold", pad=7)
        axis.set_xlabel("ISI time")
        axis.set_xlim(float(edges[0]), float(edges[-1]))
        axis.tick_params(direction="out", length=2.5, width=0.6)
        for spine in axis.spines.values():
            spine.set_linewidth(0.6)
    ax_truth.set_ylabel("Selected sample (ordered by empirical mean ISI)")
    y_ticks = torch.linspace(1, sample_count, min(6, sample_count)).round().unique().tolist()
    ax_truth.set_yticks(y_ticks)

    for axis in (ax_truth_tail, ax_prediction_tail):
        axis.set_title("No\nspike", fontsize=7, pad=3)
        axis.set_xticks([])
        axis.tick_params(axis="y", left=False, labelleft=False)
        for spine in axis.spines.values():
            spine.set_linewidth(0.6)
    ax_prediction.tick_params(axis="y", left=False, labelleft=False)

    colorbar = figure.colorbar(truth_image, cax=ax_colorbar)
    colorbar.set_label("Probability mass", labelpad=5)
    colorbar.outline.set_linewidth(0.6)
    colorbar.ax.tick_params(length=2.5, width=0.6)

    output = output.expanduser().resolve()
    if output.suffix.lower() != ".pdf":
        output = output.with_suffix(".pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    png_output = output.with_suffix(".png")
    metadata = {"Creator": "pred_heatmap.py", "Title": "ISI prediction heatmaps"}
    figure.savefig(output, bbox_inches="tight", metadata=metadata)
    figure.savefig(png_output, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)
    return output, png_output


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    cfg = _checkpoint_config(checkpoint, args.config)
    seed_everything(int(cfg.experiment.seed))
    device = resolve_device(args.device or str(cfg.experiment.device))
    preprocessing_state = checkpoint.get("preprocessing_state")
    if cfg.data.get("process_file") is not None and preprocessing_state is None:
        raise ValueError("distribution-operator checkpoint is missing its PCA preprocessing state")
    payloads = load_split_payloads(
        cfg.data,
        map_location="cpu",
        preprocessing_state=preprocessing_state,
        truncate_dim=cfg.model.get("truncate_dim", "auto"),
    )
    if args.split not in payloads:
        available = ", ".join(payloads)
        raise ValueError(f"split {args.split!r} is unavailable; choose one of: {available}")

    split_payload = payloads[args.split]
    selected = _selected_indices(
        split_payload["empirical_bin_mass"],
        split_payload["bin_edges"],
        num_samples=int(args.num_samples),
        seed=int(args.seed),
    )
    model = build_model(cfg.model, payloads["train"]).to(device)
    restore_checkpoint(checkpoint, model, device)
    model.eval()
    model_batch = {
        key: split_payload[key][selected].to(device)
        for key in (
            "normalized_params",
            "input_features",
            "input_particles",
            "process_features",
            "params",
            "bin_counts",
            "empirical_bin_mass",
        )
        if key in split_payload
    }
    model_batch["law_id"] = split_payload["law_ids"][selected].to(device)
    model_batch["regime_label"] = split_payload["regime_labels"][selected].to(device)
    model_batch["bin_edges"] = split_payload["bin_edges"].to(device)
    with torch.no_grad():
        prediction = model(model_batch)

    output = args.output or checkpoint_path.parents[1] / f"pred_heatmap_{args.split}.pdf"
    pdf_path, png_path = _plot_heatmaps(
        split_payload["empirical_bin_mass"][selected],
        prediction["pred_bin_mass"],
        split_payload["bin_edges"],
        output,
        dpi=int(args.dpi),
    )
    print(f"pdf: {pdf_path}")
    print(f"png: {png_path}")


if __name__ == "__main__":
    main()
