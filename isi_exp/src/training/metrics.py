"""Observable metrics for categorical ISI laws."""

from __future__ import annotations

import torch
import torch.nn.functional as F

BATCHED_BIN_EDGES_NDIM = 2
_METRIC_EPS = 1.0e-12


def _normalize_mass(mass: torch.Tensor) -> torch.Tensor:
    """Normalize nonnegative categorical masses along the final dimension."""
    nonnegative = mass.clamp_min(0.0)
    return nonnegative / nonnegative.sum(dim=-1, keepdim=True).clamp_min(_METRIC_EPS)


def _kl_term(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Compute ``left * log(left / right)`` with the zero-mass convention."""
    return torch.where(
        left > 0.0,
        left * torch.log(left / right.clamp_min(_METRIC_EPS)),
        torch.zeros_like(left),
    )


def _hellinger_distance_against_empirical(
    pred: torch.Tensor,
    empirical: torch.Tensor,
) -> torch.Tensor:
    """Return per-law categorical Hellinger distance to the empirical law."""
    pred = _normalize_mass(pred)
    empirical = _normalize_mass(empirical)
    squared_distance = 0.5 * (pred.sqrt() - empirical.sqrt()).square().sum(dim=-1)
    return squared_distance.clamp_min(0.0).sqrt()


def _kl_divergence_against_empirical(
    pred: torch.Tensor,
    empirical: torch.Tensor,
) -> torch.Tensor:
    """Return per-law KL(empirical || prediction) in natural-log units."""
    pred = _normalize_mass(pred)
    empirical = _normalize_mass(empirical)
    return _kl_term(empirical, pred).sum(dim=-1)


def _piecewise_uniform_w2_single(
    left_mass: list[float],
    right_mass: list[float],
    edges: list[float],
) -> float:
    """Compute exact 1D W2 for two finite-bin piecewise-uniform histograms."""
    bin_count = len(left_mass)
    left_total = sum(left_mass)
    right_total = sum(right_mass)
    if left_total <= _METRIC_EPS or right_total <= _METRIC_EPS:
        return 0.0
    left_mass = [max(value, 0.0) / left_total for value in left_mass]
    right_mass = [max(value, 0.0) / right_total for value in right_mass]

    i = 0
    j = 0
    u = 0.0
    left_cdf_prev = 0.0
    right_cdf_prev = 0.0
    left_cdf_next = left_mass[0] if bin_count else 1.0
    right_cdf_next = right_mass[0] if bin_count else 1.0
    cost = 0.0

    while u < 1.0 - _METRIC_EPS and i < bin_count and j < bin_count:
        while i < bin_count - 1 and left_cdf_next <= u + _METRIC_EPS:
            left_cdf_prev = left_cdf_next
            i += 1
            left_cdf_next += left_mass[i]
        while j < bin_count - 1 and right_cdf_next <= u + _METRIC_EPS:
            right_cdf_prev = right_cdf_next
            j += 1
            right_cdf_next += right_mass[j]

        next_u = min(left_cdf_next, right_cdf_next, 1.0)
        if next_u <= u + _METRIC_EPS:
            u = next_u
            continue

        left_bin_mass = max(left_mass[i], _METRIC_EPS)
        right_bin_mass = max(right_mass[j], _METRIC_EPS)
        left_slope = (edges[i + 1] - edges[i]) / left_bin_mass
        right_slope = (edges[j + 1] - edges[j]) / right_bin_mass
        left_intercept = edges[i] - left_slope * left_cdf_prev
        right_intercept = edges[j] - right_slope * right_cdf_prev
        intercept_delta = left_intercept - right_intercept
        slope_delta = left_slope - right_slope
        cost += (
            intercept_delta * intercept_delta * (next_u - u)
            + intercept_delta * slope_delta * (next_u * next_u - u * u)
            + slope_delta * slope_delta * (next_u**3 - u**3) / 3.0
        )
        u = next_u

    return max(cost, 0.0) ** 0.5


def _finite_renormalized_piecewise_uniform_w2(
    pred: torch.Tensor,
    empirical: torch.Tensor,
    bin_edges: torch.Tensor,
) -> torch.Tensor:
    """Return per-law W2 on finite bins after dropping and renormalizing tail mass."""
    pred_finite = pred[:, :-1].detach().cpu().to(dtype=torch.float64)
    empirical_finite = empirical[:, :-1].detach().cpu().to(dtype=torch.float64)
    edges = bin_edges.detach().cpu().to(dtype=torch.float64).tolist()
    values = [
        _piecewise_uniform_w2_single(
            empirical_finite[index].tolist(),
            pred_finite[index].tolist(),
            edges,
        )
        for index in range(pred_finite.shape[0])
    ]
    return torch.tensor(values, dtype=pred.dtype, device=pred.device)


def _per_law_metrics(prediction: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    counts = batch["bin_counts"].to(dtype=prediction["logits"].dtype)
    empirical = batch["empirical_bin_mass"].to(dtype=prediction["pred_bin_mass"].dtype)
    pred = prediction["pred_bin_mass"]
    log_probs = F.log_softmax(prediction["logits"], dim=-1)
    per_law_events = counts.sum(dim=-1).clamp_min(1.0)
    observation_nll = -(counts * log_probs).sum(dim=-1) / per_law_events
    bin_edges = batch["bin_edges"]
    if bin_edges.ndim == BATCHED_BIN_EDGES_NDIM:
        bin_edges = bin_edges[0]
    return {
        "observation_nll": observation_nll,
        "tail_bin_error_against_empirical": (pred[:, -1] - empirical[:, -1]).abs(),
        "hellinger_distance_against_empirical": _hellinger_distance_against_empirical(
            pred,
            empirical,
        ),
        "kl_divergence_against_empirical": _kl_divergence_against_empirical(
            pred,
            empirical,
        ),
        "finite_renormalized_w2_against_empirical": _finite_renormalized_piecewise_uniform_w2(
            pred,
            empirical,
            bin_edges,
        ),
    }


def categorical_metrics_against_empirical(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Compute observable categorical metrics against empirical split targets."""
    per_law = _per_law_metrics(prediction, batch)
    return {key: float(value.mean().detach().cpu()) for key, value in per_law.items()}
