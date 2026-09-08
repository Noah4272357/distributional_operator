"""Covariance visualization helpers for random-field law diagnostics."""

# ruff: noqa: PLR0913

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402


REFERENCE_COLOR = "#2563eb"
PREDICTED_COLOR = "#dc2626"
PARTICLE_COLOR = "#4b5563"
GRID_COLOR = "#e5e7eb"
EPSILON = 1.0e-8
MATRIX_DIMS = 2
PAIR_TOKEN_PARTS = 2
PAIR_DIMS = 2
STD_MARKER_SIZE = 4.0
COLORBAR_FRACTION = 0.046
COLORBAR_PAD = 0.04


@dataclass(frozen=True)
class SelectedLaw:
    """One selected law and its selection bucket."""

    bucket: str
    index: int
    law_id: int
    score: float


@dataclass(frozen=True)
class CovarianceEllipse:
    """Parameters for a 2D covariance ellipse."""

    center: tuple[float, float]
    width: float
    height: float
    angle_degrees: float


def covariance_to_correlation(covariance: torch.Tensor, eps: float = EPSILON) -> torch.Tensor:
    """Convert a covariance matrix to a finite correlation matrix."""
    tensor = torch.as_tensor(covariance, dtype=torch.float32).detach().cpu()
    if tensor.ndim != MATRIX_DIMS or tensor.shape[0] != tensor.shape[1]:
        raise ValueError("covariance must have shape [dim, dim]")

    diagonal = torch.diagonal(tensor).clamp_min(0.0)
    positive = diagonal > float(eps)
    denom = torch.sqrt(torch.outer(diagonal, diagonal)).clamp_min(float(eps))
    valid = torch.outer(positive, positive)
    correlation = torch.where(valid, tensor / denom, torch.zeros_like(tensor))
    correlation = torch.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=-1.0, max=1.0)
    diagonal_values = torch.where(positive, torch.ones_like(diagonal), torch.zeros_like(diagonal))
    indices = torch.arange(tensor.shape[0])
    correlation[indices, indices] = diagonal_values
    return correlation


def anchor_token(anchor: float) -> str:
    """Return a compact stable filename token for a field anchor."""
    scaled = int(round(float(anchor) * 100.0))
    return f"x{scaled:03d}"


def field_anchor_index(grid: torch.Tensor, anchor: float) -> int:
    """Return the nearest grid index for a continuous anchor location."""
    values = torch.as_tensor(grid, dtype=torch.float32).detach().cpu()
    if values.ndim != 1:
        raise ValueError("grid must have shape [points]")
    return int(torch.argmin((values - float(anchor)).abs()).item())


def parse_pair_token(value: str, value_type: type[int] | type[float]) -> tuple[int, int] | tuple[float, float]:
    """Parse a CLI pair token of the form ``a,b``."""
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != PAIR_TOKEN_PARTS or not all(parts):
        raise ValueError(f"pair value must have form a,b: {value}")
    return value_type(parts[0]), value_type(parts[1])


def select_top_correlation_pairs(correlation: torch.Tensor, num_pairs: int) -> list[tuple[int, int]]:
    """Select off-diagonal pairs with the largest absolute correlation."""
    matrix = torch.as_tensor(correlation, dtype=torch.float32).detach().cpu()
    if matrix.ndim != MATRIX_DIMS or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("correlation must have shape [dim, dim]")
    requested = int(num_pairs)
    if requested <= 0:
        return []

    candidates: list[tuple[float, int, int]] = []
    dim = int(matrix.shape[0])
    for i in range(dim):
        for j in range(i + 1, dim):
            candidates.append((float(abs(matrix[i, j]).item()), i, j))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [(i, j) for _, i, j in candidates[:requested]]


def select_law_buckets(
    scores: torch.Tensor,
    law_ids: torch.Tensor,
    best_k: int,
    middle_k: int,
    worst_k: int,
) -> list[SelectedLaw]:
    """Select stable best, middle, and worst law buckets without duplicates."""
    score_values = torch.as_tensor(scores, dtype=torch.float32).detach().cpu()
    law_id_values = torch.as_tensor(law_ids, dtype=torch.long).detach().cpu()
    if score_values.ndim != 1 or law_id_values.ndim != 1:
        raise ValueError("scores and law_ids must be one-dimensional")
    if score_values.numel() != law_id_values.numel():
        raise ValueError("scores and law_ids must have matching lengths")
    if score_values.numel() == 0:
        raise ValueError("cannot select laws from an empty split")

    order = sorted(range(int(score_values.numel())), key=lambda idx: (float(score_values[idx]), int(law_id_values[idx])))
    selected_indices: set[int] = set()
    selected: list[SelectedLaw] = []

    def add_bucket(bucket_name: str, candidate_indices: list[int], count: int) -> None:
        if int(count) <= 0:
            return
        rank = 1
        for index in candidate_indices:
            if index in selected_indices:
                continue
            selected_indices.add(index)
            selected.append(
                SelectedLaw(
                    bucket=f"{bucket_name}{rank:02d}",
                    index=int(index),
                    law_id=int(law_id_values[index].item()),
                    score=float(score_values[index].item()),
                )
            )
            rank += 1
            if rank > int(count):
                break

    add_bucket("best", order, best_k)
    median = float(score_values.median().item())
    middle_order = sorted(
        range(int(score_values.numel())),
        key=lambda idx: (abs(float(score_values[idx]) - median), float(score_values[idx]), int(law_id_values[idx])),
    )
    add_bucket("middle", middle_order, middle_k)
    add_bucket("worst", list(reversed(order)), worst_k)
    return selected


def ellipse_from_moments(mean: torch.Tensor, covariance: torch.Tensor, sigma: float = 1.0) -> CovarianceEllipse:
    """Return ellipse parameters from 2D mean and covariance moments."""
    mean_values = torch.as_tensor(mean, dtype=torch.float32).detach().cpu()
    covariance_values = torch.as_tensor(covariance, dtype=torch.float32).detach().cpu()
    if mean_values.shape != (PAIR_DIMS,):
        raise ValueError("mean must have shape [2]")
    if covariance_values.shape != (PAIR_DIMS, PAIR_DIMS):
        raise ValueError("covariance must have shape [2, 2]")

    symmetric = 0.5 * (covariance_values + covariance_values.T)
    eigvals, eigvecs = torch.linalg.eigh(symmetric)
    eigvals = eigvals.clamp_min(0.0)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    major_vector = eigvecs[:, 0]
    angle = float(torch.atan2(major_vector[1], major_vector[0]).item() * 180.0 / torch.pi)
    axis_lengths = 2.0 * float(sigma) * eigvals.sqrt()
    return CovarianceEllipse(
        center=(float(mean_values[0].item()), float(mean_values[1].item())),
        width=float(axis_lengths[0].item()),
        height=float(axis_lengths[1].item()),
        angle_degrees=angle,
    )


def _prepare_output_path(output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _as_1d_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu()
    if tensor.ndim != 1:
        raise ValueError(f"{name} must have shape [dim]")
    if tensor.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    return tensor


def _as_2d_square_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu()
    if tensor.ndim != MATRIX_DIMS or tensor.shape[0] != tensor.shape[1]:
        raise ValueError(f"{name} must have shape [dim, dim]")
    if tensor.shape[0] == 0:
        raise ValueError(f"{name} must be non-empty")
    return tensor


def _as_pair_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    tensor = _as_1d_tensor(value, name)
    if tensor.shape != (PAIR_DIMS,):
        raise ValueError(f"{name} must have shape [2]")
    return tensor


def _as_particle_pair_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu()
    if tensor.ndim != MATRIX_DIMS or tensor.shape[1] != PAIR_DIMS:
        raise ValueError(f"{name} must have shape [particles, 2]")
    if tensor.shape[0] == 0:
        raise ValueError(f"{name} must be non-empty")
    return tensor


def _validate_matching_covariance_inputs(
    reference_mean: torch.Tensor,
    reference_cov: torch.Tensor,
    predicted_mean: torch.Tensor,
    predicted_cov: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    reference_mean_values = _as_1d_tensor(reference_mean, "reference_mean")
    predicted_mean_values = _as_1d_tensor(predicted_mean, "predicted_mean")
    reference_cov_values = _as_2d_square_tensor(reference_cov, "reference_cov")
    predicted_cov_values = _as_2d_square_tensor(predicted_cov, "predicted_cov")
    dim = int(reference_cov_values.shape[0])
    if predicted_cov_values.shape != reference_cov_values.shape:
        raise ValueError("reference_cov and predicted_cov must have matching shapes")
    if reference_mean_values.shape[0] != dim or predicted_mean_values.shape[0] != dim:
        raise ValueError("means must match covariance dimension")
    return reference_mean_values, reference_cov_values, predicted_mean_values, predicted_cov_values


def _plot_heatmap(
    ax: plt.Axes,
    matrix: torch.Tensor,
    title: str,
    *,
    vmin: float,
    vmax: float,
    cmap: str,
    colorbar_label: str,
) -> None:
    image = ax.imshow(matrix.numpy(), origin="lower", vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("coefficient")
    ax.set_ylabel("coefficient")
    ax.tick_params(labelsize=8)
    ax.figure.colorbar(image, ax=ax, fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD, label=colorbar_label)


def coefficient_covariance_heatmap_limits(
    reference_cov: torch.Tensor,
    predicted_cov: torch.Tensor,
    eps: float = EPSILON,
) -> tuple[float, float]:
    """Return shared symmetric color limits for coefficient covariance heatmaps."""
    reference_cov_values = _as_2d_square_tensor(reference_cov, "reference_cov")
    predicted_cov_values = _as_2d_square_tensor(predicted_cov, "predicted_cov")
    if predicted_cov_values.shape != reference_cov_values.shape:
        raise ValueError("reference_cov and predicted_cov must have matching shapes")
    bound = max(
        float(reference_cov_values.abs().max().item()),
        float(predicted_cov_values.abs().max().item()),
        float(eps),
    )
    return -bound, bound


def plot_coefficient_covariance_heatmaps(
    output_path: str | Path,
    title: str,
    reference_label: str,
    reference_cov: torch.Tensor,
    predicted_cov: torch.Tensor,
    max_coefficients: int = 32,
) -> Path:
    """Render reference and predicted coefficient covariance matrices with one shared colorbar."""
    output = _prepare_output_path(output_path)
    reference_cov_values = _as_2d_square_tensor(reference_cov, "reference_cov")
    predicted_cov_values = _as_2d_square_tensor(predicted_cov, "predicted_cov")
    if predicted_cov_values.shape != reference_cov_values.shape:
        raise ValueError("reference_cov and predicted_cov must have matching shapes")
    if int(max_coefficients) <= 0:
        raise ValueError("max_coefficients must be positive")

    full_dim = int(reference_cov_values.shape[0])
    display_dim = min(int(max_coefficients), full_dim)
    reference_cov_display = reference_cov_values[:display_dim, :display_dim]
    predicted_cov_display = predicted_cov_values[:display_dim, :display_dim]
    vmin, vmax = coefficient_covariance_heatmap_limits(reference_cov_display, predicted_cov_display)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), constrained_layout=True)
    matrices = [
        (reference_cov_display, f"{reference_label} covariance"),
        (predicted_cov_display, "predicted covariance"),
    ]
    image = None
    for ax, (matrix, axis_title) in zip(axes, matrices, strict=True):
        image = ax.imshow(matrix.numpy(), origin="lower", vmin=vmin, vmax=vmax, cmap="RdBu_r", aspect="equal")
        ax.set_title(axis_title, fontsize=10)
        ax.set_xlabel("coefficient")
        ax.set_ylabel("coefficient")
        ax.tick_params(labelsize=8)
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD, label="covariance")
    if display_dim < full_dim:
        fig.suptitle(f"{title}\nfirst {display_dim} output-basis coefficients", fontsize=12)
    else:
        fig.suptitle(title, fontsize=12)
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_coefficient_covariance_summary(
    output_path: str | Path,
    title: str,
    reference_label: str,
    reference_mean: torch.Tensor,
    reference_cov: torch.Tensor,
    predicted_mean: torch.Tensor,
    predicted_cov: torch.Tensor,
    max_coefficients: int = 32,
    residual_limit: float | None = None,
) -> Path:
    """Render a 2x2 coefficient covariance summary figure."""
    output = _prepare_output_path(output_path)
    _, reference_cov_values, _, predicted_cov_values = _validate_matching_covariance_inputs(
        reference_mean,
        reference_cov,
        predicted_mean,
        predicted_cov,
    )
    if int(max_coefficients) <= 0:
        raise ValueError("max_coefficients must be positive")

    display_dim = min(int(max_coefficients), int(reference_cov_values.shape[0]))
    reference_cov_display = reference_cov_values[:display_dim, :display_dim]
    predicted_cov_display = predicted_cov_values[:display_dim, :display_dim]
    reference_correlation = covariance_to_correlation(reference_cov_display)
    predicted_correlation = covariance_to_correlation(predicted_cov_display)
    residual = predicted_correlation - reference_correlation
    if residual_limit is None:
        residual_bound = max(float(residual.abs().max().item()), EPSILON)
    else:
        residual_bound = float(residual_limit)
        if residual_bound <= 0.0:
            raise ValueError("residual_limit must be positive")

    reference_std = torch.diagonal(reference_cov_display).clamp_min(0.0).sqrt()
    predicted_std = torch.diagonal(predicted_cov_display).clamp_min(0.0).sqrt()
    indices = torch.arange(display_dim, dtype=torch.float32)
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2), constrained_layout=True)

    ax_std = axes[0][0]
    ax_std.plot(
        indices.numpy(),
        reference_std.numpy(),
        color=REFERENCE_COLOR,
        marker="o",
        markersize=STD_MARKER_SIZE,
        linewidth=1.8,
        label=reference_label,
    )
    ax_std.plot(
        indices.numpy(),
        predicted_std.numpy(),
        color=PREDICTED_COLOR,
        marker="o",
        markersize=STD_MARKER_SIZE,
        linewidth=1.8,
        label="predicted",
    )
    ax_std.set_title("Coefficient std", fontsize=10)
    ax_std.set_xlabel("coefficient")
    ax_std.set_ylabel("std")
    ax_std.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax_std.legend(loc="best", fontsize=8, frameon=False)
    ax_std.tick_params(labelsize=8)

    _plot_heatmap(
        axes[0][1],
        reference_correlation,
        f"{reference_label} correlation",
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
        colorbar_label="corr",
    )
    _plot_heatmap(
        axes[1][0],
        predicted_correlation,
        "predicted correlation",
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
        colorbar_label="corr",
    )
    _plot_heatmap(
        axes[1][1],
        residual,
        "correlation residual",
        vmin=-residual_bound,
        vmax=residual_bound,
        cmap="RdBu_r",
        colorbar_label="pred - ref",
    )
    fig.suptitle(title, fontsize=12)
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_field_covariance_slice(
    output_path: str | Path,
    title: str,
    grid: torch.Tensor,
    reference_values: torch.Tensor,
    predicted_values: torch.Tensor,
    reference_label: str,
    anchor: float,
) -> Path:
    """Render one function-space covariance slice against a field grid."""
    output = _prepare_output_path(output_path)
    grid_values = _as_1d_tensor(grid, "grid")
    reference_slice = _as_1d_tensor(reference_values, "reference_values")
    predicted_slice = _as_1d_tensor(predicted_values, "predicted_values")
    if reference_slice.shape != grid_values.shape or predicted_slice.shape != grid_values.shape:
        raise ValueError("field covariance slices must match grid shape")

    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    ax.plot(grid_values.numpy(), reference_slice.numpy(), color=REFERENCE_COLOR, linewidth=2.0, label=reference_label)
    ax.plot(grid_values.numpy(), predicted_slice.numpy(), color=PREDICTED_COLOR, linewidth=2.0, label="predicted")
    ax.axvline(float(anchor), color=PARTICLE_COLOR, linestyle=":", linewidth=1.2, label="anchor")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("covariance")
    ax.set_xlim(float(grid_values.min().item()), float(grid_values.max().item()))
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.legend(loc="best", fontsize=8, frameon=False)
    ax.tick_params(labelsize=8)
    fig.savefig(output)
    plt.close(fig)
    return output


def _ellipse_patch(
    ellipse: CovarianceEllipse,
    color: str,
    label: str,
    linestyle: str,
) -> Ellipse:
    return Ellipse(
        xy=ellipse.center,
        width=ellipse.width,
        height=ellipse.height,
        angle=ellipse.angle_degrees,
        fill=False,
        edgecolor=color,
        linewidth=1.8,
        linestyle=linestyle,
        label=label,
    )


def _set_pair_axis_limits(
    ax: plt.Axes,
    particles: torch.Tensor,
    reference_mean: torch.Tensor,
    predicted_mean: torch.Tensor,
    reference_ellipse: CovarianceEllipse,
    predicted_ellipse: CovarianceEllipse,
) -> None:
    points = torch.cat([particles, reference_mean.unsqueeze(0), predicted_mean.unsqueeze(0)], dim=0)
    x_min = float(points[:, 0].min().item())
    x_max = float(points[:, 0].max().item())
    y_min = float(points[:, 1].min().item())
    y_max = float(points[:, 1].max().item())
    ellipse_padding = 0.5 * max(
        reference_ellipse.width,
        reference_ellipse.height,
        predicted_ellipse.width,
        predicted_ellipse.height,
        EPSILON,
    )
    x_span = max(x_max - x_min, EPSILON)
    y_span = max(y_max - y_min, EPSILON)
    padding = 0.12 * max(x_span, y_span) + ellipse_padding
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)


def plot_pair_scatter_with_ellipses(
    output_path: str | Path,
    title: str,
    x_label: str,
    y_label: str,
    particles: torch.Tensor,
    reference_mean: torch.Tensor,
    reference_cov: torch.Tensor,
    predicted_mean: torch.Tensor,
    predicted_cov: torch.Tensor,
    reference_label: str,
    ellipse_sigma: float = 1.0,
) -> Path:
    """Render a 2D particle scatter plot with reference and predicted covariance ellipses."""
    output = _prepare_output_path(output_path)
    particle_values = _as_particle_pair_tensor(particles, "particles")
    reference_mean_values = _as_pair_tensor(reference_mean, "reference_mean")
    predicted_mean_values = _as_pair_tensor(predicted_mean, "predicted_mean")
    reference_cov_values = _as_2d_square_tensor(reference_cov, "reference_cov")
    predicted_cov_values = _as_2d_square_tensor(predicted_cov, "predicted_cov")
    if reference_cov_values.shape != (PAIR_DIMS, PAIR_DIMS) or predicted_cov_values.shape != (PAIR_DIMS, PAIR_DIMS):
        raise ValueError("pair covariance matrices must have shape [2, 2]")

    reference_ellipse = ellipse_from_moments(reference_mean_values, reference_cov_values, sigma=ellipse_sigma)
    predicted_ellipse = ellipse_from_moments(predicted_mean_values, predicted_cov_values, sigma=ellipse_sigma)
    fig, ax = plt.subplots(figsize=(5.4, 5.0), constrained_layout=True)
    ax.scatter(
        particle_values[:, 0].numpy(),
        particle_values[:, 1].numpy(),
        s=18,
        color=PARTICLE_COLOR,
        alpha=0.28,
        edgecolors="none",
        label="particles",
    )
    ax.add_patch(_ellipse_patch(reference_ellipse, REFERENCE_COLOR, f"{reference_label} ellipse", "solid"))
    ax.add_patch(_ellipse_patch(predicted_ellipse, PREDICTED_COLOR, "predicted ellipse", "dashed"))
    ax.scatter(
        [float(reference_mean_values[0].item())],
        [float(reference_mean_values[1].item())],
        s=42,
        color=REFERENCE_COLOR,
        edgecolors="white",
        linewidths=0.8,
        label=f"{reference_label} mean",
        zorder=3,
    )
    ax.scatter(
        [float(predicted_mean_values[0].item())],
        [float(predicted_mean_values[1].item())],
        s=48,
        color=PREDICTED_COLOR,
        marker="X",
        edgecolors="white",
        linewidths=0.8,
        label="predicted mean",
        zorder=3,
    )
    _set_pair_axis_limits(
        ax,
        particle_values,
        reference_mean_values,
        predicted_mean_values,
        reference_ellipse,
        predicted_ellipse,
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.legend(loc="best", fontsize=8, frameon=False)
    ax.tick_params(labelsize=8)
    fig.savefig(output)
    plt.close(fig)
    return output
