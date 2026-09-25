"""Deterministic features derived from finite input-law samples."""

from __future__ import annotations

import torch


def build_moment_rff_features(
    input_particles: torch.Tensor,
    *,
    frequencies: int = 8,
    rff_scale: float = 1.0,
    seed: int = 404,
) -> torch.Tensor:
    """Return mean, variance, and empirical sine/cosine Fourier moments."""
    if input_particles.ndim != 2:
        raise ValueError("input_particles must have shape [data_size, sample_size]")
    if frequencies < 1:
        raise ValueError("frequencies must be positive")
    if rff_scale <= 0:
        raise ValueError("rff_scale must be positive")
    generator = torch.Generator(device=input_particles.device).manual_seed(int(seed))
    omega = torch.randn(
        frequencies,
        dtype=input_particles.dtype,
        device=input_particles.device,
        generator=generator,
    ) * float(rff_scale)
    mean = input_particles.mean(dim=1, keepdim=True)
    variance = input_particles.var(dim=1, unbiased=False, keepdim=True)
    projected = input_particles.unsqueeze(-1) * omega.view(1, 1, -1)
    return torch.cat(
        (mean, variance, projected.sin().mean(dim=1), projected.cos().mean(dim=1)),
        dim=-1,
    )
