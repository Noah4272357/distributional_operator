"""Gaussian parameter and distance operations for coefficient laws."""

from __future__ import annotations

import torch
import torch.nn.functional as F

_PARTICLE_BATCH_NDIM = 3


def fill_lower_triangular(values: torch.Tensor, q: int) -> torch.Tensor:
    """Fill lower-triangular matrices from row-major flat lower-triangle values."""
    q = int(q)
    expected_values = q * (q + 1) // 2
    if values.shape[-1] != expected_values:
        raise ValueError(f"expected last dimension {expected_values} for q={q}, got {values.shape[-1]}")

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
    """Convert raw lower-triangular head values to positive-diagonal Cholesky factors."""
    scale_tril = fill_lower_triangular(raw_values, q)
    diag_indices = torch.arange(int(q), device=raw_values.device)
    raw_diag = scale_tril[..., diag_indices, diag_indices]
    scale_tril[..., diag_indices, diag_indices] = F.softplus(raw_diag) + float(covariance_epsilon)
    return scale_tril


def raw_diagonal_gaussian_prediction_to_scale_tril(
    raw_values: torch.Tensor,
    q: int,
    covariance_epsilon: float,
) -> torch.Tensor:
    """Convert raw diagonal scale values to diagonal Cholesky factors."""
    q = int(q)
    if raw_values.shape[-1] != q:
        raise ValueError(f"expected last dimension {q} for diagonal q={q}, got {raw_values.shape[-1]}")
    diagonal = F.softplus(raw_values) + float(covariance_epsilon)
    return torch.diag_embed(diagonal)


def raw_low_rank_diagonal_gaussian_prediction_to_scale_tril(
    raw_values: torch.Tensor,
    q: int,
    rank: int,
    covariance_epsilon: float,
) -> torch.Tensor:
    """Convert raw diagonal and factor values to a low-rank plus diagonal Cholesky factor."""
    q = int(q)
    rank = int(rank)
    if rank <= 0:
        raise ValueError("rank must be positive for low_rank_diagonal covariance")
    expected_values = q + q * rank
    if raw_values.shape[-1] != expected_values:
        raise ValueError(f"expected last dimension {expected_values} for q={q}, rank={rank}, got {raw_values.shape[-1]}")
    raw_diagonal = raw_values[..., :q]
    raw_factor = raw_values[..., q:].reshape(*raw_values.shape[:-1], q, rank)
    diagonal_scale = F.softplus(raw_diagonal) + float(covariance_epsilon)
    covariance = torch.diag_embed(diagonal_scale.square()) + raw_factor @ raw_factor.transpose(-1, -2)
    return covariance_to_scale_tril(covariance, covariance_epsilon=0.0)


def covariance_to_scale_tril(covariance: torch.Tensor, covariance_epsilon: float = 0.0) -> torch.Tensor:
    """Convert covariance matrices to Cholesky factors, optionally adding diagonal jitter."""
    _, scale_tril = _stabilize_covariance_for_cholesky(covariance, covariance_epsilon)
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
        return adjusted64.to(dtype=covariance.dtype), scale_tril64.to(dtype=covariance.dtype)


def gaussian_nll(mean: torch.Tensor, scale_tril: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
    """Return particle empirical negative log likelihood averaged over batch and particles."""
    distribution = torch.distributions.MultivariateNormal(
        loc=mean.unsqueeze(-2),
        scale_tril=scale_tril.unsqueeze(-3),
    )
    return -distribution.log_prob(samples).mean()


def fit_global_gaussian(
    output_particles: torch.Tensor,
    covariance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit one full empirical Gaussian to all output particles in a split."""
    flat_particles = output_particles.reshape(-1, output_particles.shape[-1])
    mean = flat_particles.mean(dim=0)
    centered = flat_particles - mean
    covariance = centered.transpose(-1, -2) @ centered / flat_particles.shape[0]
    covariance, scale_tril = _stabilize_covariance_for_cholesky(covariance, covariance_epsilon)
    return mean, covariance, scale_tril


def fit_batched_empirical_gaussian(
    particles: torch.Tensor,
    covariance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit one empirical Gaussian per batch item from particle samples."""
    if particles.ndim != _PARTICLE_BATCH_NDIM:
        raise ValueError("particles must have shape [batch, particles, q]")
    mean = particles.mean(dim=1)
    centered = particles - mean.unsqueeze(1)
    covariance = centered.transpose(-1, -2) @ centered / particles.shape[1]
    covariance, scale_tril = _stabilize_covariance_for_cholesky(covariance, covariance_epsilon)
    return mean, covariance, scale_tril


def _symmetric_matrix_sqrt(matrix: torch.Tensor) -> torch.Tensor:
    symmetric = 0.5 * (matrix + matrix.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    sqrt_eigenvalues = torch.sqrt(eigenvalues.clamp_min(0.0))
    return eigenvectors @ torch.diag_embed(sqrt_eigenvalues) @ eigenvectors.transpose(-1, -2)


def gaussian_w2(
    mean_left: torch.Tensor,
    cov_left: torch.Tensor,
    mean_right: torch.Tensor,
    cov_right: torch.Tensor,
) -> torch.Tensor:
    """Return square-rooted Gaussian Wasserstein-2 distances per batch item."""
    result_dtype = mean_left.dtype
    mean_left = mean_left.to(dtype=torch.float64)
    cov_left = cov_left.to(dtype=torch.float64)
    mean_right = mean_right.to(dtype=torch.float64)
    cov_right = cov_right.to(dtype=torch.float64)

    mean_distance_squared = torch.sum((mean_left - mean_right).square(), dim=-1)
    cov_right_sqrt = _symmetric_matrix_sqrt(cov_right)
    inner = cov_right_sqrt @ cov_left @ cov_right_sqrt
    inner_sqrt = _symmetric_matrix_sqrt(inner)
    trace_term = torch.diagonal(cov_left + cov_right - 2.0 * inner_sqrt, dim1=-2, dim2=-1).sum(dim=-1)
    w2_squared = (mean_distance_squared + trace_term).clamp_min(0.0)
    return torch.sqrt(w2_squared).to(dtype=result_dtype)


def is_spd(covariance: torch.Tensor, atol: float = 1e-7) -> torch.Tensor:
    """Return one boolean per covariance matrix indicating symmetric positive definiteness."""
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
    """Sample a batch of Gaussians parameterized by mean and Cholesky scale factors."""
    num_laws, q = mean.shape
    eps = torch.randn(
        (num_laws, num_samples, q),
        dtype=mean.dtype,
        device=mean.device,
        generator=generator,
    )
    return mean[:, None, :] + eps @ scale_tril.transpose(-1, -2)
