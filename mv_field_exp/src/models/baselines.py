"""Baseline model definitions for law-to-law comparison experiments."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from src.models.gaussian_ops import (
    fit_batched_empirical_gaussian,
    fit_global_gaussian,
    raw_diagonal_gaussian_prediction_to_scale_tril,
    raw_gaussian_prediction_to_scale_tril,
    raw_low_rank_diagonal_gaussian_prediction_to_scale_tril,
    scale_tril_to_covariance,
)


def activation_module(name: str) -> nn.Module:
    """Return an activation module by config name."""
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "silu":
        return nn.SiLU()
    if normalized == "identity":
        return nn.Identity()
    raise ValueError(f"unsupported activation: {name}")


def make_mlp(
    input_dim: int,
    output_dim: int,
    hidden_width: int,
    hidden_layers: int,
    activation: str,
) -> nn.Sequential:
    """Build a compact feed-forward network with configurable hidden depth."""
    input_dim = int(input_dim)
    output_dim = int(output_dim)
    hidden_width = int(hidden_width)
    hidden_layers = int(hidden_layers)
    if hidden_layers < 0:
        raise ValueError("hidden_layers cannot be negative")

    if hidden_layers == 0:
        return nn.Sequential(nn.Linear(input_dim, output_dim))

    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_width), activation_module(activation)]
    for _ in range(hidden_layers - 1):
        layers.extend([nn.Linear(hidden_width, hidden_width), activation_module(activation)])
    layers.append(nn.Linear(hidden_width, output_dim))
    return nn.Sequential(*layers)


def _lower_triangular_diagonal_positions(q: int) -> list[int]:
    positions: list[int] = []
    index = 0
    for row in range(q):
        for col in range(row + 1):
            if row == col:
                positions.append(index)
            index += 1
    return positions


def _normalize_covariance_structure(covariance_structure: str) -> str:
    normalized = str(covariance_structure).lower()
    if normalized not in {"full", "diagonal", "low_rank_diagonal"}:
        raise ValueError(f"unsupported covariance_structure: {covariance_structure}")
    return normalized


class GaussianHead(nn.Module):
    """Map features to the shared Gaussian prediction protocol."""

    def __init__(
        self,
        input_dim: int,
        q_out: int,
        covariance_epsilon: float = 1.0e-5,
        initial_scale: float = 0.1,
        covariance_structure: str = "full",
        covariance_rank: int = 4,
    ) -> None:
        super().__init__()
        self.q_out = int(q_out)
        self.covariance_epsilon = float(covariance_epsilon)
        self.initial_scale = float(initial_scale)
        self.covariance_structure = _normalize_covariance_structure(covariance_structure)
        self.covariance_rank = int(covariance_rank)
        if self.initial_scale <= 0.0:
            raise ValueError("initial_scale must be positive")
        if self.covariance_structure == "low_rank_diagonal" and self.covariance_rank <= 0:
            raise ValueError("covariance_rank must be positive for low_rank_diagonal covariance")
        raw_scale_dim = self._raw_scale_dim()
        self.output = nn.Linear(int(input_dim), self.q_out + raw_scale_dim)
        self._initialize_scale_bias()

    def _raw_scale_dim(self) -> int:
        if self.covariance_structure == "diagonal":
            return self.q_out
        if self.covariance_structure == "low_rank_diagonal":
            return self.q_out + self.q_out * self.covariance_rank
        return self.q_out * (self.q_out + 1) // 2

    def _initialize_scale_bias(self) -> None:
        target_scale = self.initial_scale
        raw_diag_bias = math.log(math.exp(target_scale) - 1.0)
        raw_offset = self.q_out
        if self.covariance_structure in {"diagonal", "low_rank_diagonal"}:
            diagonal_positions = range(self.q_out)
        else:
            diagonal_positions = _lower_triangular_diagonal_positions(self.q_out)
        with torch.no_grad():
            for raw_position in diagonal_positions:
                self.output.bias[raw_offset + raw_position] = raw_diag_bias
            if self.covariance_structure == "low_rank_diagonal":
                factor_start = raw_offset + self.q_out
                self.output.weight[factor_start:].mul_(0.01)
                self.output.bias[factor_start:] = 0.0

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        raw_prediction = self.output(features)
        pred_mean = raw_prediction[..., : self.q_out]
        raw_scale = raw_prediction[..., self.q_out :]
        if self.covariance_structure == "diagonal":
            pred_scale_tril = raw_diagonal_gaussian_prediction_to_scale_tril(
                raw_scale,
                q=self.q_out,
                covariance_epsilon=self.covariance_epsilon,
            )
        elif self.covariance_structure == "low_rank_diagonal":
            pred_scale_tril = raw_low_rank_diagonal_gaussian_prediction_to_scale_tril(
                raw_scale,
                q=self.q_out,
                rank=self.covariance_rank,
                covariance_epsilon=self.covariance_epsilon,
            )
        else:
            pred_scale_tril = raw_gaussian_prediction_to_scale_tril(
                raw_scale,
                q=self.q_out,
                covariance_epsilon=self.covariance_epsilon,
            )
        pred_cov = scale_tril_to_covariance(pred_scale_tril)
        return {
            "pred_mean": pred_mean,
            "pred_scale_tril": pred_scale_tril,
            "pred_cov": pred_cov,
        }


def _batch_size_from_batch(batch: dict[str, Any]) -> int:
    for value in batch.values():
        if torch.is_tensor(value):
            return int(value.shape[0])
    raise ValueError("batch must contain at least one tensor")


class GlobalConstantPredictor(nn.Module):
    """Predict one train-split Gaussian for every law instance."""

    def __init__(self, pred_mean: torch.Tensor, pred_cov: torch.Tensor, pred_scale_tril: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("pred_mean", pred_mean)
        self.register_buffer("pred_cov", pred_cov)
        self.register_buffer("pred_scale_tril", pred_scale_tril)

    @classmethod
    def fit_from_output_particles(
        cls,
        output_particles: torch.Tensor,
        covariance_epsilon: float,
    ) -> GlobalConstantPredictor:
        pred_mean, pred_cov, pred_scale_tril = fit_global_gaussian(output_particles, covariance_epsilon)
        return cls(pred_mean=pred_mean, pred_cov=pred_cov, pred_scale_tril=pred_scale_tril)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch_size = _batch_size_from_batch(batch)
        return {
            "pred_mean": self.pred_mean.unsqueeze(0).expand(batch_size, -1),
            "pred_scale_tril": self.pred_scale_tril.unsqueeze(0).expand(batch_size, -1, -1),
            "pred_cov": self.pred_cov.unsqueeze(0).expand(batch_size, -1, -1),
        }


class MomentOnlyGaussianPredictor(nn.Module):
    """Predict output Gaussian parameters from context-particle mean and diagonal covariance."""

    def __init__(
        self,
        q_in: int,
        q_out: int,
        hidden_width: int,
        hidden_layers: int,
        activation: str,
        covariance_epsilon: float = 1.0e-5,
        initial_scale: float = 0.1,
        covariance_structure: str = "full",
        covariance_rank: int = 4,
    ) -> None:
        super().__init__()
        feature_dim = 2 * int(q_in)
        self.encoder = make_mlp(feature_dim, int(hidden_width), hidden_width, hidden_layers, activation)
        self.head = GaussianHead(
            int(hidden_width),
            q_out,
            covariance_epsilon=covariance_epsilon,
            initial_scale=initial_scale,
            covariance_structure=covariance_structure,
            covariance_rank=covariance_rank,
        )

    @staticmethod
    def moment_features(context_particles: torch.Tensor) -> torch.Tensor:
        sample_mean = context_particles.mean(dim=1)
        sample_diag_cov = context_particles.var(dim=1, unbiased=False)
        return torch.cat([sample_mean, sample_diag_cov], dim=-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        features = self.moment_features(batch["context_particles"])
        return self.head(self.encoder(features))


class OracleFeatureGaussianPredictor(nn.Module):
    """Predict output Gaussian parameters from synthetic oracle features."""

    def __init__(
        self,
        feature_dim: int,
        q_out: int,
        hidden_width: int,
        hidden_layers: int,
        activation: str,
        covariance_epsilon: float = 1.0e-5,
        initial_scale: float = 0.1,
        covariance_structure: str = "full",
        covariance_rank: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = make_mlp(int(feature_dim), int(hidden_width), hidden_width, hidden_layers, activation)
        self.head = GaussianHead(
            int(hidden_width),
            q_out,
            covariance_epsilon=covariance_epsilon,
            initial_scale=initial_scale,
            covariance_structure=covariance_structure,
            covariance_rank=covariance_rank,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.head(self.encoder(batch["oracle_features"]))


class PathwiseMLPPredictor(nn.Module):
    """Map index-aligned context particles to output particles, then fit a Gaussian."""

    def __init__(
        self,
        q_in: int,
        q_out: int,
        hidden_width: int,
        hidden_layers: int,
        activation: str,
        covariance_epsilon: float = 1.0e-5,
    ) -> None:
        super().__init__()
        self.map = make_mlp(q_in, q_out, hidden_width, hidden_layers, activation)
        self.covariance_epsilon = float(covariance_epsilon)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        context_particles = batch["context_particles"]
        if "output_particles" in batch and context_particles.shape[1] != batch["output_particles"].shape[1]:
            raise ValueError("pathwise_mlp requires N_ctx == N_out for index_aligned pairing")

        batch_size, particle_count, q_in = context_particles.shape
        flat_prediction = self.map(context_particles.reshape(batch_size * particle_count, q_in))
        pred_particles = flat_prediction.reshape(batch_size, particle_count, -1)
        pred_mean, pred_cov, pred_scale_tril = fit_batched_empirical_gaussian(
            pred_particles,
            covariance_epsilon=self.covariance_epsilon,
        )
        return {
            "pred_particles": pred_particles,
            "pred_mean": pred_mean,
            "pred_scale_tril": pred_scale_tril,
            "pred_cov": pred_cov,
        }
