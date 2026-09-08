"""Moment-based MLP for distribution-to-Gaussian prediction."""

from __future__ import annotations

import torch
from torch import nn

from src.models.modelutils import GaussianHead, make_mlp


class MomentMLP(nn.Module):
    """Predict a Gaussian from the empirical mean and covariance of input samples."""

    def __init__(
        self,
        input_dim: int,
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
        input_dim = int(input_dim)
        hidden_width = int(hidden_width)
        moment_dim = input_dim + input_dim * input_dim
        self.encoder = make_mlp(
            input_dim=moment_dim,
            output_dim=hidden_width,
            hidden_width=hidden_width,
            hidden_layers=int(hidden_layers),
            activation=activation,
        )
        self.head = GaussianHead(
            hidden_width,
            int(q_out),
            covariance_epsilon=covariance_epsilon,
            initial_scale=initial_scale,
            covariance_structure=covariance_structure,
            covariance_rank=covariance_rank,
        )

    @staticmethod
    def moment_features(input_dist: torch.Tensor) -> torch.Tensor:
        """Return empirical means and flattened unbiased covariance matrices."""
        if input_dist.ndim != 3:
            raise ValueError("input_dist must have shape [batch, sample_size, q]")
        sample_size = int(input_dist.shape[1])
        if sample_size < 2:
            raise ValueError("sample_size must be at least two to compute covariance")

        sample_mean = input_dist.mean(dim=1)
        centered = input_dist - sample_mean.unsqueeze(1)
        sample_cov = centered.transpose(-1, -2) @ centered / (sample_size - 1)
        return torch.cat((sample_mean, sample_cov.flatten(start_dim=1)), dim=-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        features = self.moment_features(batch["input_dist"])
        return self.head(self.encoder(features))
