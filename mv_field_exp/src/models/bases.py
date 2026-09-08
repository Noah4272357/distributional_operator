"""Basis function helpers for coefficient and field representations."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LearnedBasis:
    center: torch.Tensor
    values: torch.Tensor
    weights: torch.Tensor
    explained_variance: torch.Tensor
    explained_variance_ratio: torch.Tensor
    cumulative_explained_variance_ratio: torch.Tensor
    centering_policy: str
    normalization_policy: str


def uniform_grid(num_points: int, domain_min: float = 0.0, domain_max: float = 1.0) -> torch.Tensor:
    if int(num_points) < 2:
        raise ValueError("num_points must be at least 2")
    if float(domain_max) <= float(domain_min):
        raise ValueError("domain_max must be greater than domain_min")
    return torch.linspace(float(domain_min), float(domain_max), int(num_points), dtype=torch.float32)


def trapezoid_weights(grid: torch.Tensor) -> torch.Tensor:
    if grid.ndim != 1 or grid.numel() < 2:
        raise ValueError("grid must be one-dimensional with at least two points")

    deltas = grid[1:] - grid[:-1]
    if bool(torch.any(deltas <= 0)):
        raise ValueError("grid must be strictly increasing")

    weights = grid.new_empty(grid.shape)
    weights[0] = 0.5 * deltas[0]
    weights[-1] = 0.5 * deltas[-1]
    if grid.numel() > 2:
        weights[1:-1] = 0.5 * (deltas[:-1] + deltas[1:])
    return weights


def sine_basis_values(grid: torch.Tensor, rank: int) -> torch.Tensor:
    if grid.ndim != 1:
        raise ValueError("grid must be one-dimensional")
    if int(rank) <= 0:
        raise ValueError("rank must be positive")

    modes = torch.arange(1, int(rank) + 1, dtype=grid.dtype, device=grid.device)
    return torch.sqrt(torch.tensor(2.0, dtype=grid.dtype, device=grid.device)) * torch.sin(
        torch.pi * grid.unsqueeze(-1) * modes
    )


def select_pca_rank(
    explained_variance_ratio: torch.Tensor,
    threshold: float,
    min_rank: int = 1,
    max_rank: int | None = None,
) -> int:
    """Select the smallest PCA rank meeting a cumulative variance threshold."""
    if explained_variance_ratio.ndim != 1:
        raise ValueError("explained_variance_ratio must be one-dimensional")
    if explained_variance_ratio.numel() == 0:
        raise ValueError("explained_variance_ratio must not be empty")

    threshold = float(threshold)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")

    min_rank = int(min_rank)
    if min_rank <= 0:
        raise ValueError("min_rank must be positive")

    available_rank = int(explained_variance_ratio.numel())
    if max_rank is None:
        max_rank = available_rank
    max_rank = int(max_rank)
    if max_rank <= 0:
        raise ValueError("max_rank must be positive")
    if min_rank > max_rank:
        raise ValueError("min_rank must not exceed max_rank")
    if max_rank > available_rank:
        raise ValueError("max_rank must not exceed available rank")

    cumulative = explained_variance_ratio.cumsum(dim=0)
    meets_threshold = torch.nonzero(cumulative >= threshold, as_tuple=False)
    selected = available_rank if meets_threshold.numel() == 0 else int(meets_threshold[0, 0].item()) + 1
    return max(min_rank, min(selected, max_rank))


def fit_weighted_pca_basis(  # noqa: PLR0913
    particles: torch.Tensor,
    weights: torch.Tensor,
    rank: int | None,
    centering_policy: str = "global_train_particle_mean",
    normalization_policy: str = "none",
    explained_variance_threshold: float | None = None,
    min_rank: int = 1,
    max_rank: int | None = None,
) -> LearnedBasis:
    if particles.ndim != 2:
        raise ValueError("particles must have shape [samples, grid_points]")
    if weights.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    if particles.shape[-1] != weights.numel():
        raise ValueError("weights length must match particles grid dimension")
    if bool(torch.any(weights <= 0)):
        raise ValueError("weights must be positive")
    if centering_policy != "global_train_particle_mean":
        raise ValueError("centering_policy must be global_train_particle_mean")
    if normalization_policy != "none":
        raise ValueError("normalization_policy must be none")

    samples, grid_points = particles.shape

    weights = weights.to(device=particles.device, dtype=particles.dtype)
    sqrt_weights = weights.sqrt()
    center = particles.mean(dim=0)
    weighted_centered = (particles - center) * sqrt_weights.unsqueeze(0)

    _, singular_values, vh = torch.linalg.svd(weighted_centered, full_matrices=False)
    all_explained_variance = singular_values.square() / max(samples - 1, 1)
    denominator = all_explained_variance.sum().clamp_min(torch.finfo(all_explained_variance.dtype).eps)
    all_explained_variance_ratio = all_explained_variance / denominator

    available_rank = int(min(samples, grid_points))
    if rank is None:
        if explained_variance_threshold is None:
            raise ValueError("explained_variance_threshold is required when rank is None")
        rank = select_pca_rank(
            all_explained_variance_ratio,
            threshold=float(explained_variance_threshold),
            min_rank=int(min_rank),
            max_rank=available_rank if max_rank is None else int(max_rank),
        )
    else:
        rank = int(rank)
        if rank <= 0:
            raise ValueError("rank must be positive")
        if rank > available_rank:
            raise ValueError("rank must not exceed min(samples, grid_points)")

    values = vh[:rank].transpose(0, 1) / sqrt_weights.unsqueeze(-1)
    explained_variance = all_explained_variance[:rank]
    explained_variance_ratio = all_explained_variance_ratio[:rank]
    cumulative_explained_variance_ratio = explained_variance_ratio.cumsum(dim=0)

    return LearnedBasis(
        center=center,
        values=values,
        weights=weights,
        explained_variance=explained_variance,
        explained_variance_ratio=explained_variance_ratio,
        cumulative_explained_variance_ratio=cumulative_explained_variance_ratio,
        centering_policy=centering_policy,
        normalization_policy=normalization_policy,
    )


def project_onto_basis(
    particles: torch.Tensor,
    center: torch.Tensor,
    basis_values: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if basis_values.ndim != 2:
        raise ValueError("basis_values must have shape [grid_points, rank]")
    if center.ndim != 1 or weights.ndim != 1:
        raise ValueError("center and weights must be one-dimensional")
    if center.numel() != basis_values.shape[0] or weights.numel() != basis_values.shape[0]:
        raise ValueError("center, weights, and basis_values grid dimensions must match")
    if particles.shape[-1] != basis_values.shape[0]:
        raise ValueError("particles last dimension must match basis grid dimension")

    weights = weights.to(device=particles.device, dtype=particles.dtype)
    center = center.to(device=particles.device, dtype=particles.dtype)
    basis_values = basis_values.to(device=particles.device, dtype=particles.dtype)
    return (particles - center) @ (weights.unsqueeze(-1) * basis_values)


def reconstruct_from_basis(coefficients: torch.Tensor, center: torch.Tensor, basis_values: torch.Tensor) -> torch.Tensor:
    if basis_values.ndim != 2:
        raise ValueError("basis_values must have shape [grid_points, rank]")
    if center.ndim != 1:
        raise ValueError("center must be one-dimensional")
    if center.numel() != basis_values.shape[0]:
        raise ValueError("center length must match basis grid dimension")
    if coefficients.shape[-1] != basis_values.shape[1]:
        raise ValueError("coefficients last dimension must match basis rank")

    center = center.to(device=coefficients.device, dtype=coefficients.dtype)
    basis_values = basis_values.to(device=coefficients.device, dtype=coefficients.dtype)
    return center + coefficients @ basis_values.transpose(0, 1)
