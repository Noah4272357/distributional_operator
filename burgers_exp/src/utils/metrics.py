"""Metrics for comparing batches of empirical distributions."""

from __future__ import annotations

import torch
from torch import Tensor


def _validate_samples(prediction: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    if prediction.ndim == 2:
        prediction = prediction.unsqueeze(0)
    if target.ndim == 2:
        target = target.unsqueeze(0)
    if prediction.ndim != 3 or target.ndim != 3:
        raise ValueError("samples must have shape (batch, samples, features)")
    if prediction.shape[0] != target.shape[0]:
        raise ValueError("prediction and target batch sizes differ")
    if prediction.shape[2] != target.shape[2]:
        raise ValueError("prediction and target feature dimensions differ")
    if prediction.shape[1] < 1 or target.shape[1] < 1:
        raise ValueError("empirical distributions must contain at least one sample")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("distribution metrics require floating-point tensors")
    return prediction, target.to(device=prediction.device, dtype=prediction.dtype)


def maximum_mean_discrepancy(
    prediction: Tensor,
    target: Tensor,
    *,
    bandwidth: float | Tensor | None = None,
) -> Tensor:
    """Return biased RBF-kernel MMD, averaged across the batch.

    When ``bandwidth`` is omitted, a separate median-heuristic bandwidth is
    estimated for every batch element.
    """
    prediction, target = _validate_samples(prediction, target)
    prediction_distance = torch.cdist(prediction, prediction).square()
    target_distance = torch.cdist(target, target).square()
    cross_distance = torch.cdist(prediction, target).square()
    batch_size = prediction.shape[0]

    if bandwidth is None:
        prediction_count = prediction.shape[1]
        target_count = target.shape[1]
        prediction_off_diagonal = ~torch.eye(
            prediction_count, device=prediction.device, dtype=torch.bool
        )
        target_off_diagonal = ~torch.eye(
            target_count, device=prediction.device, dtype=torch.bool
        )
        candidates = [cross_distance.reshape(batch_size, -1)]
        if prediction_count > 1:
            candidates.append(prediction_distance[:, prediction_off_diagonal])
        if target_count > 1:
            candidates.append(target_distance[:, target_off_diagonal])
        bandwidth_squared = torch.cat(candidates, dim=1).median(dim=1).values
    else:
        bandwidth_tensor = torch.as_tensor(
            bandwidth, device=prediction.device, dtype=prediction.dtype
        )
        if torch.any(bandwidth_tensor <= 0):
            raise ValueError("bandwidth must be positive")
        bandwidth_squared = bandwidth_tensor.square().expand(batch_size)

    epsilon = torch.finfo(prediction.dtype).eps
    bandwidth_squared = bandwidth_squared.clamp_min(epsilon).view(-1, 1, 1)
    prediction_kernel = torch.exp(-prediction_distance / (2 * bandwidth_squared))
    target_kernel = torch.exp(-target_distance / (2 * bandwidth_squared))
    cross_kernel = torch.exp(-cross_distance / (2 * bandwidth_squared))
    squared_mmd = (
        prediction_kernel.mean(dim=(1, 2))
        + target_kernel.mean(dim=(1, 2))
        - 2 * cross_kernel.mean(dim=(1, 2))
    )
    return squared_mmd.clamp_min(0).sqrt().mean()


def mmd(
    prediction: Tensor,
    target: Tensor,
    *,
    bandwidth: float | Tensor | None = None,
) -> Tensor:
    """Short alias for :func:`maximum_mean_discrepancy`."""
    return maximum_mean_discrepancy(prediction, target, bandwidth=bandwidth)


def _projected_quantiles(values: Tensor, quantile_count: int) -> Tensor:
    if values.shape[1] == quantile_count:
        return values.sort(dim=1).values
    quantiles = torch.linspace(
        0,
        1,
        quantile_count,
        device=values.device,
        dtype=values.dtype,
    )
    return torch.quantile(values, quantiles, dim=1).permute(1, 0, 2)


def sliced_wasserstein_distance(
    prediction: Tensor,
    target: Tensor,
    *,
    num_projections: int = 128,
    p: float = 2.0,
    seed: int = 0,
) -> Tensor:
    """Return the projection-averaged sliced Wasserstein-p distance."""
    prediction, target = _validate_samples(prediction, target)
    if num_projections < 1:
        raise ValueError("num_projections must be positive")
    if p <= 0:
        raise ValueError("p must be positive")
    generator = torch.Generator(device=prediction.device)
    generator.manual_seed(seed)
    projections = torch.randn(
        prediction.shape[2],
        num_projections,
        generator=generator,
        device=prediction.device,
        dtype=prediction.dtype,
    )
    projections = projections / projections.norm(dim=0, keepdim=True).clamp_min(
        torch.finfo(prediction.dtype).eps
    )
    prediction_projected = prediction @ projections
    target_projected = target @ projections
    quantile_count = max(prediction.shape[1], target.shape[1])
    prediction_quantiles = _projected_quantiles(
        prediction_projected, quantile_count
    )
    target_quantiles = _projected_quantiles(target_projected, quantile_count)
    distance = (
        (prediction_quantiles - target_quantiles)
        .abs()
        .pow(p)
        .mean(dim=(1, 2))
        .pow(1.0 / p)
    )
    return distance.mean()


def energy_distance(prediction: Tensor, target: Tensor) -> Tensor:
    """Return the biased energy distance, averaged across the batch."""
    prediction, target = _validate_samples(prediction, target)
    cross_term = 2 * torch.cdist(prediction, target).mean(dim=(1, 2))
    prediction_term = torch.cdist(prediction, prediction).mean(dim=(1, 2))
    target_term = torch.cdist(target, target).mean(dim=(1, 2))
    return (cross_term - prediction_term - target_term).clamp_min(0).mean()


def distribution_metrics(
    prediction: Tensor,
    target: Tensor,
    *,
    mmd_bandwidth: float | Tensor | None = None,
    sliced_projections: int = 128,
    sliced_seed: int = 0,
) -> dict[str, Tensor]:
    """Compute all supported empirical-distribution metrics."""
    return {
        "mmd": maximum_mean_discrepancy(
            prediction, target, bandwidth=mmd_bandwidth
        ),
        "sliced_wasserstein": sliced_wasserstein_distance(
            prediction,
            target,
            num_projections=sliced_projections,
            seed=sliced_seed,
        ),
        "energy_distance": energy_distance(prediction, target),
    }
