"""The single supported McKean--Vlasov Gaussian law map."""

from __future__ import annotations

from typing import Any

import torch

TARGET_NAME = "McKean_Vlasov"


def _cfg_get(cfg: Any, key: str) -> Any:
    if isinstance(cfg, dict):
        return cfg[key]
    return getattr(cfg, key)


def sample_gaussian_laws(
    data_size: int,
    coefficient_rank: int,
    seed: int,
    mean_low: float,
    mean_high: float,
    sigma_low: float,
    sigma_high: float,
    jitter: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample reproducible full-covariance Gaussian input laws."""
    if data_size <= 0 or coefficient_rank <= 0:
        raise ValueError("data_size and coefficient_rank must be positive")
    if mean_high <= mean_low:
        raise ValueError("mean_high must be greater than mean_low")
    if not 0.0 < sigma_low < sigma_high:
        raise ValueError("sigma bounds must satisfy 0 < sigma_low < sigma_high")
    if jitter < 0.0:
        raise ValueError("jitter must be non-negative")

    generator = torch.Generator().manual_seed(int(seed))
    mean = torch.empty((data_size, coefficient_rank), dtype=torch.float32).uniform_(
        float(mean_low), float(mean_high), generator=generator
    )
    raw = torch.randn((data_size, coefficient_rank, coefficient_rank), generator=generator)
    orthogonal, _ = torch.linalg.qr(raw)
    signs = torch.sign(torch.diagonal(orthogonal, dim1=-2, dim2=-1))
    signs = torch.where(signs == 0.0, torch.ones_like(signs), signs)
    orthogonal = orthogonal * signs.unsqueeze(-2)
    sigmas = torch.empty((data_size, coefficient_rank), dtype=torch.float32).uniform_(
        float(sigma_low), float(sigma_high), generator=generator
    )
    covariance = orthogonal @ torch.diag_embed(sigmas.square()) @ orthogonal.transpose(-1, -2)
    identity = torch.eye(coefficient_rank, dtype=torch.float32).unsqueeze(0)
    covariance = covariance + float(jitter) * identity
    return mean, 0.5 * (covariance + covariance.transpose(-1, -2))


def sample_particles(
    mean: torch.Tensor,
    covariance: torch.Tensor,
    sample_size: int,
    seed: int,
) -> torch.Tensor:
    """Draw particles with shape ``[data_size, sample_size, coefficient_rank]``."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    generator = torch.Generator().manual_seed(int(seed))
    scale_tril = torch.linalg.cholesky(covariance)
    noise = torch.randn(
        (mean.shape[0], int(sample_size), mean.shape[-1]),
        dtype=mean.dtype,
        generator=generator,
    )
    return mean.unsqueeze(1) + noise @ scale_tril.transpose(-1, -2)


def apply_mckean_vlasov(
    input_mean: torch.Tensor,
    input_covariance: torch.Tensor,
    cfg: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the closed-form ``McKean_Vlasov`` Gaussian law map."""
    coefficient_rank = int(input_mean.shape[-1])
    modes = torch.arange(1, coefficient_rank + 1, dtype=input_mean.dtype, device=input_mean.device)
    laplacian_eigenvalues = (torch.pi * modes).square()
    kappa = float(_cfg_get(cfg, "kappa"))
    a = float(_cfg_get(cfg, "a"))
    b = float(_cfg_get(cfg, "b"))
    time_horizon = float(_cfg_get(cfg, "T"))

    mean_decay = torch.exp(-(kappa * laplacian_eigenvalues + a - b) * time_horizon)
    fluctuation_decay = torch.exp(-(kappa * laplacian_eigenvalues + a) * time_horizon)
    output_mean = input_mean * mean_decay
    decay = torch.diag(fluctuation_decay).unsqueeze(0)
    output_covariance = decay @ input_covariance @ decay
    return output_mean, 0.5 * (output_covariance + output_covariance.transpose(-1, -2))
