"""Weighted-grid sample metrics and field diagnostics for random-field laws."""

from __future__ import annotations

from collections.abc import Sequence

import torch


FIELD_METRIC_EPSILON = 1.0e-8


def _validate_field_samples(left: torch.Tensor, right: torch.Tensor, weights: torch.Tensor) -> None:
    if left.ndim != 3 or right.ndim != 3:
        raise ValueError("field samples must have shape [batch, particles, grid]")
    if left.shape[0] != right.shape[0] or left.shape[-1] != right.shape[-1]:
        raise ValueError("field sample sets must share batch and grid dimensions")
    if weights.ndim != 1 or weights.shape[0] != left.shape[-1]:
        raise ValueError("weights must have shape [grid]")
    if bool(torch.any(weights <= 0)):
        raise ValueError("weights must be positive")


def _weights_like(weights: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return weights.to(device=reference.device, dtype=reference.dtype)


def _weighted_l2_pairwise_squared(left: torch.Tensor, right: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    _validate_field_samples(left, right, weights)
    weights = _weights_like(weights, left)
    right = right.to(device=left.device, dtype=left.dtype)
    diff = left.unsqueeze(2) - right.unsqueeze(1)
    return (diff.square() * weights.view(1, 1, 1, -1)).sum(dim=-1).clamp_min(0.0)


def weighted_l2_pairwise_distances(left: torch.Tensor, right: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(_weighted_l2_pairwise_squared(left, right, weights))


def field_energy_distance(predicted: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    cross = weighted_l2_pairwise_distances(predicted, target, weights).mean(dim=(1, 2))
    pred_self = weighted_l2_pairwise_distances(predicted, predicted, weights).mean(dim=(1, 2))
    target_self = weighted_l2_pairwise_distances(target, target, weights).mean(dim=(1, 2))
    return 2.0 * cross - pred_self - target_self


def field_mmd(
    predicted: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    scales: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
) -> torch.Tensor:
    if not scales:
        raise ValueError("scales must be non-empty")

    pred_self_sq = _weighted_l2_pairwise_squared(predicted, predicted, weights)
    target_self_sq = _weighted_l2_pairwise_squared(target, target, weights)
    cross_sq = _weighted_l2_pairwise_squared(predicted, target, weights)
    pooled = torch.cat(
        [
            pred_self_sq.reshape(predicted.shape[0], -1),
            target_self_sq.reshape(predicted.shape[0], -1),
            cross_sq.reshape(predicted.shape[0], -1),
        ],
        dim=1,
    )
    median_bandwidth = torch.sqrt(pooled.median(dim=1).values.clamp_min(0.0)).clamp_min(FIELD_METRIC_EPSILON)

    total = pred_self_sq.new_zeros(predicted.shape[0])
    for scale in scales:
        bandwidth = (median_bandwidth * float(scale)).clamp_min(FIELD_METRIC_EPSILON)
        bandwidth_sq = bandwidth.square().view(-1, 1, 1)
        pred_kernel = torch.exp(-pred_self_sq / (2.0 * bandwidth_sq)).mean(dim=(1, 2))
        target_kernel = torch.exp(-target_self_sq / (2.0 * bandwidth_sq)).mean(dim=(1, 2))
        cross_kernel = torch.exp(-cross_sq / (2.0 * bandwidth_sq)).mean(dim=(1, 2))
        total = total + pred_kernel + target_kernel - 2.0 * cross_kernel

    return (total / float(len(scales))).clamp_min(0.0)


def integrated_crps(predicted: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    _validate_field_samples(predicted, target, weights)
    target = target.to(device=predicted.device, dtype=predicted.dtype)
    weights = _weights_like(weights, predicted)
    abs_cross = (predicted.unsqueeze(2) - target.unsqueeze(1)).abs().mean(dim=(1, 2))
    abs_pred = (predicted.unsqueeze(2) - predicted.unsqueeze(1)).abs().mean(dim=(1, 2))
    abs_target = (target.unsqueeze(2) - target.unsqueeze(1)).abs().mean(dim=(1, 2))
    per_grid = abs_cross - 0.5 * abs_pred - 0.5 * abs_target
    return (per_grid * weights.view(1, -1)).sum(dim=-1)


def field_variogram_discrepancy(
    predicted: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    p: float = 1.0,
    pair_indices: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    _validate_field_samples(predicted, target, weights)
    if float(p) <= 0.0:
        raise ValueError("p must be positive")

    target = target.to(device=predicted.device, dtype=predicted.dtype)
    weights = _weights_like(weights, predicted)
    grid = predicted.shape[-1]

    if pair_indices is None:
        left_idx, right_idx = torch.triu_indices(grid, grid, offset=1, device=predicted.device)
    elif isinstance(pair_indices, tuple):
        left_idx, right_idx = pair_indices
        left_idx = left_idx.to(device=predicted.device, dtype=torch.long)
        right_idx = right_idx.to(device=predicted.device, dtype=torch.long)
    else:
        if pair_indices.ndim != 2 or pair_indices.shape[0] != 2:
            raise ValueError("pair_indices must have shape [2, pairs]")
        pair_indices = pair_indices.to(device=predicted.device, dtype=torch.long)
        left_idx, right_idx = pair_indices[0], pair_indices[1]

    if left_idx.numel() == 0:
        raise ValueError("pair_indices must include at least one pair")

    pred_increment = (predicted[..., left_idx] - predicted[..., right_idx]).abs().pow(float(p)).mean(dim=1)
    target_increment = (target[..., left_idx] - target[..., right_idx]).abs().pow(float(p)).mean(dim=1)
    pair_weights = weights[left_idx] * weights[right_idx]
    return ((pred_increment - target_increment).square() * pair_weights.view(1, -1)).sum(dim=-1)


def sample_predicted_field_particles(
    pred_mean: torch.Tensor,
    pred_scale_tril: torch.Tensor,
    output_center: torch.Tensor,
    output_basis_values: torch.Tensor,
    num_samples: int,
    seed: int,
) -> torch.Tensor:
    if pred_mean.ndim != 2:
        raise ValueError("pred_mean must have shape [batch, rank]")
    if pred_scale_tril.ndim != 3 or pred_scale_tril.shape[0] != pred_mean.shape[0]:
        raise ValueError("pred_scale_tril must have shape [batch, rank, rank]")
    if pred_scale_tril.shape[-2:] != (pred_mean.shape[-1], pred_mean.shape[-1]):
        raise ValueError("pred_scale_tril rank dimensions must match pred_mean")
    if output_center.ndim != 1:
        raise ValueError("output_center must have shape [grid]")
    if output_basis_values.ndim != 2:
        raise ValueError("output_basis_values must have shape [grid, rank]")
    if output_basis_values.shape[0] != output_center.shape[0]:
        raise ValueError("output_center and output_basis_values grid dimensions must match")
    if output_basis_values.shape[1] != pred_mean.shape[-1]:
        raise ValueError("output_basis_values rank must match pred_mean")
    if int(num_samples) <= 0:
        raise ValueError("num_samples must be positive")

    try:
        generator = torch.Generator(device=pred_mean.device)
        generator.manual_seed(int(seed))
        eps = torch.randn(
            (pred_mean.shape[0], int(num_samples), pred_mean.shape[-1]),
            dtype=pred_mean.dtype,
            device=pred_mean.device,
            generator=generator,
        )
    except (RuntimeError, TypeError):
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        eps = torch.randn(
            (pred_mean.shape[0], int(num_samples), pred_mean.shape[-1]),
            dtype=pred_mean.dtype,
            generator=generator,
        ).to(device=pred_mean.device)

    pred_scale_tril = pred_scale_tril.to(device=pred_mean.device, dtype=pred_mean.dtype)
    output_center = output_center.to(device=pred_mean.device, dtype=pred_mean.dtype)
    output_basis_values = output_basis_values.to(device=pred_mean.device, dtype=pred_mean.dtype)
    coeffs = pred_mean.unsqueeze(1) + eps @ pred_scale_tril.transpose(-1, -2)
    return output_center + coeffs @ output_basis_values.transpose(0, 1)
