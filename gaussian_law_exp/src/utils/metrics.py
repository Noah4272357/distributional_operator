"""Gaussian validation metrics for distribution-to-distribution prediction."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


VALIDATION_METRICS = (
    "nll",
    "nll_diff",
    "w2_distance",
    "kl_divergence",
    "hellinger_distance",
)
MAIN_RESULT_MODELS = frozenset(
    {
        "distributional_operator",
        # Retained so checkpoints written before the model rename remain readable.
        "deepsets",
        "momentmlp",
        "moment_mlp",
        "kernel_regression",
        "kernelregression",
    }
)
SUPPORTED_LOSSES = frozenset(
    {"gaussian_nll", "param_supervised", "gaussian_nll_plus_param", "pathwise_mse"}
)
LOSS_ROLES = {
    "gaussian_nll": "main_objective",
    "param_supervised": "debug_only",
    "gaussian_nll_plus_param": "synthetic_oracle_assisted_ablation",
    "pathwise_mse": "misspecified_pathwise_baseline",
}
_PARTICLE_BATCH_NDIM = 3


def fill_lower_triangular(values: torch.Tensor, q: int) -> torch.Tensor:
    """Fill lower-triangular matrices from flat row-major values."""
    q = int(q)
    expected_values = q * (q + 1) // 2
    if values.shape[-1] != expected_values:
        raise ValueError(
            f"expected last dimension {expected_values} for q={q}, "
            f"got {values.shape[-1]}"
        )
    result = values.new_zeros((*values.shape[:-1], q, q))
    row_indices, col_indices = torch.tril_indices(q, q, device=values.device)
    result[..., row_indices, col_indices] = values
    return result


def scale_tril_to_covariance(scale_tril: torch.Tensor) -> torch.Tensor:
    """Convert lower-triangular scale factors to covariance matrices."""
    return scale_tril @ scale_tril.transpose(-1, -2)


def raw_gaussian_prediction_to_scale_tril(
    raw_values: torch.Tensor,
    q: int,
    covariance_epsilon: float,
) -> torch.Tensor:
    """Convert raw full-covariance values to valid Cholesky factors."""
    scale_tril = fill_lower_triangular(raw_values, q)
    diag_indices = torch.arange(int(q), device=raw_values.device)
    raw_diag = scale_tril[..., diag_indices, diag_indices]
    scale_tril[..., diag_indices, diag_indices] = (
        F.softplus(raw_diag) + float(covariance_epsilon)
    )
    return scale_tril


def raw_diagonal_gaussian_prediction_to_scale_tril(
    raw_values: torch.Tensor,
    q: int,
    covariance_epsilon: float,
) -> torch.Tensor:
    """Convert raw diagonal-covariance values to Cholesky factors."""
    q = int(q)
    if raw_values.shape[-1] != q:
        raise ValueError(
            f"expected last dimension {q} for diagonal q={q}, "
            f"got {raw_values.shape[-1]}"
        )
    diagonal = F.softplus(raw_values) + float(covariance_epsilon)
    return torch.diag_embed(diagonal)


def raw_low_rank_diagonal_gaussian_prediction_to_scale_tril(
    raw_values: torch.Tensor,
    q: int,
    rank: int,
    covariance_epsilon: float,
) -> torch.Tensor:
    """Convert low-rank-plus-diagonal values to Cholesky factors."""
    q = int(q)
    rank = int(rank)
    if rank <= 0:
        raise ValueError("rank must be positive for low_rank_diagonal covariance")
    expected_values = q + q * rank
    if raw_values.shape[-1] != expected_values:
        raise ValueError(
            f"expected last dimension {expected_values} for q={q}, rank={rank}, "
            f"got {raw_values.shape[-1]}"
        )
    raw_diagonal = raw_values[..., :q]
    raw_factor = raw_values[..., q:].reshape(*raw_values.shape[:-1], q, rank)
    diagonal_scale = F.softplus(raw_diagonal) + float(covariance_epsilon)
    covariance = (
        torch.diag_embed(diagonal_scale.square())
        + raw_factor @ raw_factor.transpose(-1, -2)
    )
    return covariance_to_scale_tril(covariance)


def covariance_to_scale_tril(
    covariance: torch.Tensor,
    covariance_epsilon: float = 0.0,
) -> torch.Tensor:
    """Convert covariance matrices to Cholesky factors with optional jitter."""
    _, scale_tril = _stabilize_covariance_for_cholesky(
        covariance, covariance_epsilon
    )
    return scale_tril


def _stabilize_covariance_for_cholesky(
    covariance: torch.Tensor,
    covariance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    symmetric = 0.5 * (covariance + covariance.transpose(-1, -2))
    q = covariance.shape[-1]
    eye = torch.eye(q, dtype=covariance.dtype, device=covariance.device)
    adjusted = symmetric + float(covariance_epsilon) * eye
    try:
        return adjusted, torch.linalg.cholesky(adjusted)
    except RuntimeError:
        working = symmetric.to(dtype=torch.float64)
        working_eye = torch.eye(q, dtype=working.dtype, device=working.device)
        min_eigenvalue = torch.linalg.eigvalsh(working).amin(dim=-1)
        jitter = (-min_eigenvalue).clamp_min(0.0) + float(covariance_epsilon)
        adjusted64 = working + jitter.reshape((*jitter.shape, 1, 1)) * working_eye
        scale_tril64 = torch.linalg.cholesky(adjusted64)
        return (
            adjusted64.to(dtype=covariance.dtype),
            scale_tril64.to(dtype=covariance.dtype),
        )


def gaussian_nll(
    mean: torch.Tensor,
    scale_tril: torch.Tensor,
    samples: torch.Tensor,
) -> torch.Tensor:
    """Return empirical Gaussian NLL averaged over batches and samples."""
    distribution = torch.distributions.MultivariateNormal(
        loc=mean.unsqueeze(-2),
        scale_tril=scale_tril.unsqueeze(-3),
    )
    return -distribution.log_prob(samples).mean()


def nll_diff(
    predicted_mean: torch.Tensor,
    predicted_scale_tril: torch.Tensor,
    target_mean: torch.Tensor,
    target_cov: torch.Tensor,
    output_samples: torch.Tensor,
) -> torch.Tensor:
    """Return predicted NLL minus target-Gaussian NLL on the same samples."""
    predicted_nll = gaussian_nll(
        predicted_mean,
        predicted_scale_tril,
        output_samples,
    )
    target_scale_tril = covariance_to_scale_tril(target_cov)
    target_nll = gaussian_nll(target_mean, target_scale_tril, output_samples)
    return predicted_nll - target_nll


def fit_global_gaussian(
    output_particles: torch.Tensor,
    covariance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit one full empirical Gaussian to all supplied particles."""
    flat_particles = output_particles.reshape(-1, output_particles.shape[-1])
    mean = flat_particles.mean(dim=0)
    centered = flat_particles - mean
    covariance = centered.transpose(-1, -2) @ centered / flat_particles.shape[0]
    covariance, scale_tril = _stabilize_covariance_for_cholesky(
        covariance, covariance_epsilon
    )
    return mean, covariance, scale_tril


def fit_batched_empirical_gaussian(
    particles: torch.Tensor,
    covariance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit one empirical Gaussian per batch item."""
    if particles.ndim != _PARTICLE_BATCH_NDIM:
        raise ValueError("particles must have shape [batch, particles, q]")
    mean = particles.mean(dim=1)
    centered = particles - mean.unsqueeze(1)
    covariance = centered.transpose(-1, -2) @ centered / particles.shape[1]
    covariance, scale_tril = _stabilize_covariance_for_cholesky(
        covariance, covariance_epsilon
    )
    return mean, covariance, scale_tril


def _symmetric_matrix_sqrt(matrix: torch.Tensor) -> torch.Tensor:
    symmetric = 0.5 * (matrix + matrix.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    sqrt_eigenvalues = torch.sqrt(eigenvalues.clamp_min(0.0))
    return (
        eigenvectors
        @ torch.diag_embed(sqrt_eigenvalues)
        @ eigenvectors.transpose(-1, -2)
    )


def gaussian_w2(
    mean_left: torch.Tensor,
    cov_left: torch.Tensor,
    mean_right: torch.Tensor,
    cov_right: torch.Tensor,
) -> torch.Tensor:
    """Return Gaussian Wasserstein-2 distances per batch item."""
    result_dtype = mean_left.dtype
    mean_left = mean_left.to(dtype=torch.float64)
    cov_left = cov_left.to(dtype=torch.float64)
    mean_right = mean_right.to(dtype=torch.float64)
    cov_right = cov_right.to(dtype=torch.float64)
    mean_distance_squared = torch.sum((mean_left - mean_right).square(), dim=-1)
    cov_right_sqrt = _symmetric_matrix_sqrt(cov_right)
    inner_sqrt = _symmetric_matrix_sqrt(
        cov_right_sqrt @ cov_left @ cov_right_sqrt
    )
    trace_term = torch.diagonal(
        cov_left + cov_right - 2.0 * inner_sqrt, dim1=-2, dim2=-1
    ).sum(dim=-1)
    return torch.sqrt(
        (mean_distance_squared + trace_term).clamp_min(0.0)
    ).to(dtype=result_dtype)


def is_spd(covariance: torch.Tensor, atol: float = 1e-7) -> torch.Tensor:
    """Return whether every covariance is symmetric positive definite."""
    symmetric = torch.isclose(
        covariance,
        covariance.transpose(-1, -2),
        atol=atol,
        rtol=0.0,
    ).all(dim=(-1, -2))
    min_eigenvalue = torch.linalg.eigvalsh(covariance).amin(dim=-1)
    return symmetric & (min_eigenvalue > atol)


def sample_gaussian(
    mean: torch.Tensor,
    scale_tril: torch.Tensor,
    num_samples: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample batches of Gaussians from means and Cholesky factors."""
    num_laws, q = mean.shape
    eps = torch.randn(
        (num_laws, num_samples, q),
        dtype=mean.dtype,
        device=mean.device,
        generator=generator,
    )
    return mean[:, None, :] + eps @ scale_tril.transpose(-1, -2)


def _positive_logdet(covariance: torch.Tensor) -> torch.Tensor:
    """Return log determinants after checking covariance definiteness."""
    sign, logdet = torch.linalg.slogdet(covariance)
    if not bool(torch.all(sign > 0)):
        raise ValueError("covariance matrices must be positive definite")
    return logdet


def gaussian_kl_divergence(
    mean_p: torch.Tensor,
    cov_p: torch.Tensor,
    mean_q: torch.Tensor,
    cov_q: torch.Tensor,
) -> torch.Tensor:
    """Return ``KL(N_p || N_q)`` for each batch item.

    Computation uses float64 linear solves instead of explicit inverses for
    numerical stability. The returned tensor uses the input mean's dtype.
    """
    result_dtype = mean_p.dtype
    mean_p = mean_p.to(dtype=torch.float64)
    cov_p = cov_p.to(dtype=torch.float64)
    mean_q = mean_q.to(dtype=torch.float64)
    cov_q = cov_q.to(dtype=torch.float64)

    dimension = mean_p.shape[-1]
    mean_delta = mean_q - mean_p
    trace_term = torch.diagonal(
        torch.linalg.solve(cov_q, cov_p), dim1=-2, dim2=-1
    ).sum(dim=-1)
    solved_delta = torch.linalg.solve(cov_q, mean_delta.unsqueeze(-1)).squeeze(-1)
    quadratic_term = torch.sum(mean_delta * solved_delta, dim=-1)
    logdet_term = _positive_logdet(cov_q) - _positive_logdet(cov_p)
    divergence = 0.5 * (
        trace_term + quadratic_term - dimension + logdet_term
    )
    return divergence.clamp_min(0.0).to(dtype=result_dtype)


def gaussian_hellinger_distance(
    mean_left: torch.Tensor,
    cov_left: torch.Tensor,
    mean_right: torch.Tensor,
    cov_right: torch.Tensor,
) -> torch.Tensor:
    """Return the Hellinger distance between Gaussian pairs per batch item."""
    result_dtype = mean_left.dtype
    mean_left = mean_left.to(dtype=torch.float64)
    cov_left = cov_left.to(dtype=torch.float64)
    mean_right = mean_right.to(dtype=torch.float64)
    cov_right = cov_right.to(dtype=torch.float64)

    average_cov = 0.5 * (cov_left + cov_right)
    mean_delta = mean_left - mean_right
    solved_delta = torch.linalg.solve(
        average_cov, mean_delta.unsqueeze(-1)
    ).squeeze(-1)
    quadratic_term = torch.sum(mean_delta * solved_delta, dim=-1)
    log_coefficient = (
        0.25 * (_positive_logdet(cov_left) + _positive_logdet(cov_right))
        - 0.5 * _positive_logdet(average_cov)
        - 0.125 * quadratic_term
    )
    bhattacharyya_coefficient = torch.exp(log_coefficient.clamp_max(0.0))
    distance = torch.sqrt((1.0 - bhattacharyya_coefficient).clamp_min(0.0))
    return distance.to(dtype=result_dtype)


def _mean_float(value: torch.Tensor) -> float:
    return float(value.detach().mean().cpu().item())


def gaussian_validation_metrics(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Compute target-Gaussian validation distances for one batch."""
    predicted_mean = prediction["pred_mean"]
    predicted_cov = prediction["pred_cov"]
    target_mean = batch["target_mean"]
    target_cov = batch["target_cov"]
    return {
        "nll_diff": _mean_float(
            nll_diff(
                predicted_mean,
                prediction["pred_scale_tril"],
                target_mean,
                target_cov,
                batch["output_dist"],
            )
        ),
        "w2_distance": _mean_float(
            gaussian_w2(predicted_mean, predicted_cov, target_mean, target_cov)
        ),
        "kl_divergence": _mean_float(
            gaussian_kl_divergence(
                target_mean,
                target_cov,
                predicted_mean,
                predicted_cov,
            )
        ),
        "hellinger_distance": _mean_float(
            gaussian_hellinger_distance(
                predicted_mean,
                predicted_cov,
                target_mean,
                target_cov,
            )
        ),
    }


def loss_metadata(loss_name: str, model_name: str) -> dict[str, Any]:
    """Describe a configured loss and its role in result reporting."""
    loss_name = str(loss_name).lower()
    model_name = str(model_name).lower()
    if loss_name not in SUPPORTED_LOSSES:
        raise ValueError(f"unsupported loss: {loss_name}")
    if model_name not in MAIN_RESULT_MODELS:
        raise ValueError(
            f"unsupported model: {model_name}; expected distributional_operator, momentmlp, "
            "or kernel_regression"
        )
    uses_artificial_pairing = loss_name == "pathwise_mse"
    return {
        "loss_name": loss_name,
        "model_name": model_name,
        "loss_role": LOSS_ROLES[loss_name],
        "main_result_eligible": (
            loss_name == "gaussian_nll" and model_name in MAIN_RESULT_MODELS
        ),
        "uses_artificial_pairing": uses_artificial_pairing,
        "pairing_policy": "index_aligned" if uses_artificial_pairing else None,
    }


class ConfiguredLoss(nn.Module):
    """Apply the configured training objective."""

    def __init__(self, loss_cfg: Any) -> None:
        super().__init__()
        self.name = str(loss_cfg.name).lower()
        if self.name not in SUPPORTED_LOSSES:
            raise ValueError(f"unsupported loss: {self.name}")
        self.param_weight = float(loss_cfg.get("param_weight", 1.0))
        self.mean_weight = float(loss_cfg.get("mean_weight", 1.0))
        self.covariance_weight = float(loss_cfg.get("covariance_weight", 1.0))

    def _parameter_loss(
        self,
        prediction: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        mean_loss = F.mse_loss(prediction["pred_mean"], batch["target_mean"])
        covariance_loss = F.mse_loss(
            prediction["pred_cov"], batch["target_cov"]
        )
        return (
            self.mean_weight * mean_loss
            + self.covariance_weight * covariance_loss
        )

    def forward(
        self,
        prediction: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.name == "pathwise_mse":
            if "pred_particles" not in prediction:
                raise ValueError("pathwise_mse requires prediction['pred_particles']")
            if prediction["pred_particles"].shape != batch["output_dist"].shape:
                raise ValueError(
                    "pathwise_mse requires predicted and output particles "
                    "to have the same shape"
                )
            return F.mse_loss(prediction["pred_particles"], batch["output_dist"])
        if self.name == "param_supervised":
            return self._parameter_loss(prediction, batch)

        nll = gaussian_nll(
            prediction["pred_mean"],
            prediction["pred_scale_tril"],
            batch["output_dist"],
        )
        if self.name == "gaussian_nll":
            return nll
        return nll + self.param_weight * self._parameter_loss(prediction, batch)
