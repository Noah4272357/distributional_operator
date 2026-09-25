"""Metric namespace for law-to-law experiment reporting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from src.models.gaussian_ops import gaussian_w2


COEFFICIENT_METRICS = ("e_m", "e_C", "gaussian_w2")
FIELD_METRICS = ("e_m_y", "e_C_y")
METRIC_EPSILON = 1.0e-8


def relative_l2_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    metric_epsilon: float = METRIC_EPSILON,
) -> torch.Tensor:
    """Compute per-item relative L2 error with a clamped target norm."""
    numerator = torch.linalg.vector_norm(prediction - target, dim=-1)
    denominator = torch.linalg.vector_norm(target, dim=-1).clamp_min(float(metric_epsilon))
    return numerator / denominator


def relative_frobenius_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    metric_epsilon: float = METRIC_EPSILON,
) -> torch.Tensor:
    """Compute per-item relative Frobenius error with a clamped target norm."""
    numerator = torch.linalg.matrix_norm(prediction - target, ord="fro", dim=(-2, -1))
    denominator = torch.linalg.matrix_norm(target, ord="fro", dim=(-2, -1)).clamp_min(float(metric_epsilon))
    return numerator / denominator


def _mean_float(value: torch.Tensor) -> float:
    return float(value.detach().mean().cpu().item())


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def coefficient_metrics(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Compute coefficient-space synthetic diagnostics for a batch."""
    mean_error = relative_l2_error(prediction["pred_mean"], batch["target_mean"])
    covariance_error = relative_frobenius_error(prediction["pred_cov"], batch["target_cov"])
    w2_distance = gaussian_w2(
        prediction["pred_mean"],
        prediction["pred_cov"],
        batch["target_mean"],
        batch["target_cov"],
    )
    return {
        "e_m": _mean_float(mean_error),
        "e_C": _mean_float(covariance_error),
        "gaussian_w2": _mean_float(w2_distance),
    }


def sliced_w2_distance(
    left_particles: torch.Tensor,
    right_particles: torch.Tensor,
    num_projections: int = 128,
    seed: int = 0,
) -> torch.Tensor:
    """Compute per-batch sliced W2 distances between equal-size sample sets."""
    if left_particles.ndim != 3 or right_particles.ndim != 3:
        raise ValueError("sliced_w2_distance requires sample sets with shape [batch, particles, q]")
    if left_particles.shape != right_particles.shape:
        raise ValueError("sliced_w2_distance requires sample sets with the same shape")

    _, _, q = left_particles.shape
    try:
        generator = torch.Generator(device=left_particles.device)
        generator.manual_seed(int(seed))
        directions = torch.randn(
            (int(num_projections), q),
            dtype=left_particles.dtype,
            device=left_particles.device,
            generator=generator,
        )
    except (RuntimeError, TypeError):
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        directions = torch.randn(
            (int(num_projections), q),
            dtype=left_particles.dtype,
            generator=generator,
        ).to(device=left_particles.device)

    directions = directions / torch.linalg.vector_norm(directions, dim=-1, keepdim=True).clamp_min(METRIC_EPSILON)
    left_projected = torch.einsum("bnq,kq->bnk", left_particles, directions).sort(dim=1).values
    right_projected = torch.einsum("bnq,kq->bnk", right_particles, directions).sort(dim=1).values
    projection_w2_squared = (left_projected - right_projected).square().mean(dim=1)
    return torch.sqrt(projection_w2_squared.mean(dim=-1).clamp_min(0.0))


def sample_set_metrics(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    metrics_cfg: object | None = None,
) -> dict[str, float]:
    """Compute observable sample-set diagnostics when predicted particles exist."""
    if "pred_particles" not in prediction:
        return {}

    sliced_w2_cfg = _cfg_get(metrics_cfg, "sliced_w2", {})
    distance = sliced_w2_distance(
        prediction["pred_particles"],
        batch["output_dist"],
        num_projections=int(_cfg_get(sliced_w2_cfg, "num_projections", 128)),
        seed=int(_cfg_get(sliced_w2_cfg, "seed", 0)),
    )
    return {"sliced_w2": _mean_float(distance)}


def average_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    """Average a non-empty list of metric dictionaries key by key."""
    if not metric_dicts:
        return {}
    return {
        key: float(sum(metric_dict[key] for metric_dict in metric_dicts) / len(metric_dicts))
        for key in metric_dicts[0]
    }
