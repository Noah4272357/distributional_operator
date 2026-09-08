"""Visualization helpers for random-field law sample diagnostics."""

# ruff: noqa: PLR0913

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


FIELD_TENSOR_DIMS = 2
MIN_BAND_SAMPLE_COUNT = 2
TRUE_COLOR = "#2563eb"
PREDICTED_COLOR = "#dc2626"


@dataclass(frozen=True)
class FieldSamplePanel:
    """One law-level field sample panel."""

    title: str
    grid: Sequence[float] | torch.Tensor
    true_fields: torch.Tensor
    predicted_fields: torch.Tensor | None = None


def _as_2d_float_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu()
    if tensor.ndim != FIELD_TENSOR_DIMS:
        raise ValueError(f"{name} must have shape [particles, grid]")
    return tensor


def _as_1d_float_tensor(value: Sequence[float] | torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu()
    if tensor.ndim != 1:
        raise ValueError(f"{name} must have shape [grid]")
    return tensor


def _validate_panel(panel: FieldSamplePanel) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    grid = _as_1d_float_tensor(panel.grid, "grid")
    true_fields = _as_2d_float_tensor(panel.true_fields, "true_fields")
    if true_fields.shape[-1] != grid.shape[0]:
        raise ValueError("true_fields grid dimension must match grid")

    predicted_fields = None
    if panel.predicted_fields is not None:
        predicted_fields = _as_2d_float_tensor(panel.predicted_fields, "predicted_fields")
        if predicted_fields.shape[-1] != grid.shape[0]:
            raise ValueError("predicted_fields grid dimension must match grid")
    return grid, true_fields, predicted_fields


def _y_limits(field_sets: Sequence[torch.Tensor]) -> tuple[float, float]:
    stacked = torch.cat(list(field_sets), dim=0)
    y_min = float(stacked.min().item())
    y_max = float(stacked.max().item())
    padding = max((y_max - y_min) * 0.08, 1.0e-4)
    return y_min - padding, y_max + padding


def _sample_indices(total_count: int, max_samples: int) -> list[int]:
    sample_count = min(int(max_samples), int(total_count))
    if sample_count <= 0:
        return []
    if sample_count == total_count:
        return list(range(total_count))
    return torch.linspace(0, total_count - 1, steps=sample_count).round().to(dtype=torch.long).unique().tolist()


def _plot_sample_set(
    ax: plt.Axes,
    grid: torch.Tensor,
    fields: torch.Tensor,
    color: str,
    label: str,
    max_samples: int,
) -> None:
    x_values = grid.numpy()
    if fields.shape[0] >= MIN_BAND_SAMPLE_COUNT:
        lower = fields.quantile(0.1, dim=0).numpy()
        upper = fields.quantile(0.9, dim=0).numpy()
        ax.fill_between(x_values, lower, upper, color=color, alpha=0.12, linewidth=0)

    for sample_index in _sample_indices(int(fields.shape[0]), max_samples):
        ax.plot(x_values, fields[sample_index].numpy(), color=color, alpha=0.25, linewidth=0.8)
    ax.plot(x_values, fields.mean(dim=0).numpy(), color=color, linewidth=2.0, label=label)


def _plot_panel(
    ax: plt.Axes,
    panel: FieldSamplePanel,
    max_true_samples: int,
    max_predicted_samples: int,
    show_legend: bool,
) -> None:
    grid, true_fields, predicted_fields = _validate_panel(panel)
    field_sets: list[torch.Tensor] = [true_fields]
    if predicted_fields is not None:
        field_sets.append(predicted_fields)
    y_min, y_max = _y_limits(field_sets)

    _plot_sample_set(ax, grid, true_fields, TRUE_COLOR, "true", max_true_samples)
    if predicted_fields is not None:
        _plot_sample_set(ax, grid, predicted_fields, PREDICTED_COLOR, "pred", max_predicted_samples)

    ax.set_title(panel.title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.set_xlim(float(grid.min().item()), float(grid.max().item()))
    ax.set_ylim(y_min, y_max)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.tick_params(labelsize=8)
    if show_legend:
        ax.legend(loc="best", fontsize=8, frameon=False)


def plot_field_sample_panels(
    panels: Sequence[FieldSamplePanel],
    output_path: str | Path,
    figure_title: str | None = None,
    max_true_samples: int = 8,
    max_predicted_samples: int = 8,
    columns: int = 2,
) -> Path:
    """Render law-level true/predicted output field samples."""
    if not panels:
        raise ValueError("panels must be non-empty")
    if int(columns) <= 0:
        raise ValueError("columns must be positive")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ncols = min(int(columns), len(panels))
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.4 * ncols, 3.8 * nrows),
        squeeze=False,
    )
    for index, panel in enumerate(panels):
        row = index // ncols
        col = index % ncols
        _plot_panel(
            axes[row][col],
            panel,
            max_true_samples=max_true_samples,
            max_predicted_samples=max_predicted_samples,
            show_legend=index == 0,
        )

    for index in range(len(panels), nrows * ncols):
        row = index // ncols
        col = index % ncols
        fig.delaxes(axes[row][col])

    # Keep the legacy parameter in the public API, but do not draw a figure-level
    # title so filenames carry experiment context and each panel stays focused.
    _ = figure_title
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.95, wspace=0.22, hspace=0.35)
    fig.savefig(output)
    plt.close(fig)
    return output
