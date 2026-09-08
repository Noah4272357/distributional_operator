"""DeepSets empirical-law encoder."""

from __future__ import annotations

import torch
from torch import nn

from src.models.baselines import GaussianHead, make_mlp


class DeepSets(nn.Module):
    """Encode particles independently, mean-pool, and predict a Gaussian law."""

    def __init__(
        self,
        input_dim: int,
        q_out: int,
        inner_width: int,
        inner_layers: int,
        embedding_dim: int,
        outer_width: int,
        outer_layers: int,
        activation: str,
        covariance_epsilon: float = 1.0e-5,
        initial_scale: float = 0.1,
        covariance_structure: str = "full",
        covariance_rank: int = 4,
    ) -> None:
        super().__init__()
        self.phi = make_mlp(
            input_dim=int(input_dim),
            output_dim=int(embedding_dim),
            hidden_width=int(inner_width),
            hidden_layers=int(inner_layers),
            activation=activation,
        )
        self.rho = make_mlp(
            input_dim=int(embedding_dim),
            output_dim=int(outer_width),
            hidden_width=int(outer_width),
            hidden_layers=int(outer_layers),
            activation=activation,
        )
        self.head = GaussianHead(
            int(outer_width),
            int(q_out),
            covariance_epsilon=covariance_epsilon,
            initial_scale=initial_scale,
            covariance_structure=covariance_structure,
            covariance_rank=covariance_rank,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        particle_embeddings = self.phi(batch["context_particles"])
        pooled_embedding = particle_embeddings.mean(dim=1)
        return self.head(self.rho(pooled_embedding))
