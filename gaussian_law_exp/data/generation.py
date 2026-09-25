"""Generate paired input and output probability-distribution samples.

Each input distribution is a randomly parameterized diagonal Gaussian mixture.
A fixed nonlinear feature map turns its analytic statistics into the mean and
covariance of a Gaussian output distribution. The returned samples therefore
contain different distributions for every row and are reproducible by seed.
"""

from __future__ import annotations

import math

import torch


DIMENSION = 4
DEFAULT_SEED = 0
_MAX_COMPONENTS = 3
_FOURIER_FEATURES = 8


def _sample_input_parameters(
    data_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample padded parameters for diagonal Gaussian mixtures."""
    counts = torch.randint(
        1,
        _MAX_COMPONENTS + 1,
        (data_size,),
        generator=generator,
    )
    active = torch.arange(_MAX_COMPONENTS).unsqueeze(0) < counts.unsqueeze(1)

    unnormalized_weights = -torch.log(
        torch.rand((data_size, _MAX_COMPONENTS), generator=generator).clamp_min(
            torch.finfo(torch.float32).tiny
        )
    )
    weights = torch.where(active, unnormalized_weights, 0.0)
    weights = weights / weights.sum(dim=1, keepdim=True)

    means = -2.0 + 4.0 * torch.rand(
        (data_size, _MAX_COMPONENTS, DIMENSION), generator=generator
    )
    means = torch.where(active.unsqueeze(-1), means, 0.0)

    standard_deviations = 0.15 + 0.65 * torch.rand(
        (data_size, _MAX_COMPONENTS, DIMENSION), generator=generator
    )
    variances = torch.where(active.unsqueeze(-1), standard_deviations.square(), 0.0)
    return weights, means, variances


def _sample_mixtures(
    weights: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
    sample_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draw ``sample_size`` particles from every Gaussian mixture."""
    data_size, _, dimension = means.shape
    component_indices = torch.multinomial(
        weights,
        num_samples=sample_size,
        replacement=True,
        generator=generator,
    )
    law_indices = torch.arange(data_size).unsqueeze(1)
    selected_means = means[law_indices, component_indices]
    selected_std = variances[law_indices, component_indices].sqrt()
    noise = torch.randn((data_size, sample_size, dimension), generator=generator)
    return selected_means + selected_std * noise


def _mixture_features(
    weights: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
    frequencies: torch.Tensor,
) -> torch.Tensor:
    """Compute analytic moments and Fourier features of each input mixture."""
    expanded_weights = weights.unsqueeze(-1)
    mixture_mean = (expanded_weights * means).sum(dim=1)
    second_moment = (expanded_weights * (variances + means.square())).sum(dim=1)
    mixture_variance = (second_moment - mixture_mean.square()).clamp_min(0.0)

    phase = means @ frequencies.transpose(0, 1)
    damping = torch.exp(
        -0.5 * torch.einsum("ncd,fd->ncf", variances, frequencies.square())
    )
    weighted_damping = expanded_weights * damping
    sine = (weighted_damping * torch.sin(phase)).sum(dim=1)
    cosine = (weighted_damping * torch.cos(phase)).sum(dim=1)
    return torch.cat((mixture_mean, mixture_variance, sine, cosine), dim=-1)


def _make_targets(
    features: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map input-law features to Gaussian means and positive-definite covariances."""
    feature_size = features.shape[-1]
    weight_scale = 1.0 / math.sqrt(feature_size)

    mean_weight = weight_scale * torch.randn(
        (DIMENSION, feature_size), generator=generator
    )
    target_mean = features @ mean_weight.transpose(0, 1)

    diagonal_weight = weight_scale * torch.randn(
        (DIMENSION, feature_size), generator=generator
    )
    diagonal = 0.25 + 0.75 * torch.sigmoid(
        features @ diagonal_weight.transpose(0, 1)
    )

    off_diagonal_count = DIMENSION * (DIMENSION - 1) // 2
    off_diagonal_weight = weight_scale * torch.randn(
        (off_diagonal_count, feature_size), generator=generator
    )
    off_diagonal = 0.2 * torch.tanh(
        features @ off_diagonal_weight.transpose(0, 1)
    )

    scale_tril = torch.diag_embed(diagonal)
    rows, columns = torch.tril_indices(DIMENSION, DIMENSION, offset=-1)
    scale_tril[:, rows, columns] = off_diagonal
    target_cov = scale_tril @ scale_tril.transpose(-1, -2)
    target_cov = target_cov + 1.0e-4 * torch.eye(DIMENSION).unsqueeze(0)
    return target_mean, target_cov, torch.linalg.cholesky(target_cov)


def generate_dataset(
    data_size: int = 10,
    sample_size: int = 5,
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, torch.Tensor]:
    """Generate a dataset containing only the four tensors used for training."""
    data_size = int(data_size)
    sample_size = int(sample_size)
    if data_size <= 0:
        raise ValueError("data_size must be positive")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    generator = torch.Generator().manual_seed(int(seed))
    weights, means, variances = _sample_input_parameters(data_size, generator)
    input_dist = _sample_mixtures(
        weights, means, variances, sample_size, generator
    )

    frequencies = torch.randn(
        (_FOURIER_FEATURES, DIMENSION), generator=generator
    )
    features = _mixture_features(weights, means, variances, frequencies)
    target_mean, target_cov, target_scale_tril = _make_targets(features, generator)

    noise = torch.randn(
        (data_size, sample_size, DIMENSION), generator=generator
    )
    output_dist = (
        target_mean.unsqueeze(1) + noise @ target_scale_tril.transpose(-1, -2)
    )

    return {
        "input_dist": input_dist,
        "output_dist": output_dist,
        "target_mean": target_mean,
        "target_cov": target_cov,
    }
