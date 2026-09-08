"""Categorical predictors for ISI law-to-law learning."""

from __future__ import annotations

import torch
from torch import nn


def activation_module(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
        "identity": nn.Identity,
    }
    normalized = str(name).lower()
    if normalized not in activations:
        raise ValueError(f"unsupported activation: {name}")
    return activations[normalized]()


def make_mlp(
    input_dim: int,
    output_dim: int,
    hidden_width: int,
    hidden_layers: int,
    activation: str,
) -> nn.Sequential:
    """Build an MLP whose width, depth, and activation are constructor choices."""
    if int(hidden_layers) < 0:
        raise ValueError("hidden_layers cannot be negative")
    if int(hidden_layers) == 0:
        return nn.Sequential(nn.Linear(int(input_dim), int(output_dim)))
    layers: list[nn.Module] = [
        nn.Linear(int(input_dim), int(hidden_width)),
        activation_module(activation),
    ]
    for _ in range(int(hidden_layers) - 1):
        layers.extend([nn.Linear(int(hidden_width), int(hidden_width)), activation_module(activation)])
    layers.append(nn.Linear(int(hidden_width), int(output_dim)))
    return nn.Sequential(*layers)


def prediction_from_logits(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    pred_bin_mass = torch.softmax(logits, dim=-1)
    return {
        "logits": logits,
        "pred_bin_mass": pred_bin_mass,
        "pred_cdf": pred_bin_mass.cumsum(dim=-1),
    }


def batch_size_from_batch(batch: dict[str, torch.Tensor]) -> int:
    for value in batch.values():
        if torch.is_tensor(value):
            return int(value.shape[0])
    raise ValueError("batch must contain at least one tensor")


class ISIParamMLP(nn.Module):
    """Predict ISI bin logits from normalized drive parameters."""

    def __init__(self, output_dim: int, hidden_width: int, hidden_layers: int, activation: str) -> None:
        super().__init__()
        self.net = make_mlp(2, output_dim, hidden_width, hidden_layers, activation)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return prediction_from_logits(self.net(batch["normalized_params"]))


class ISIFeatureMLP(nn.Module):
    """Predict ISI bin logits from fixed finite input-law features."""

    def __init__(
        self,
        feature_dim: int,
        output_dim: int,
        hidden_width: int,
        hidden_layers: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.net = make_mlp(feature_dim, output_dim, hidden_width, hidden_layers, activation)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return prediction_from_logits(self.net(batch["input_features"]))


class ISIContextDeepSets(nn.Module):
    """Predict ISI bin logits from proxy particles using a DeepSets encoder."""

    def __init__(
        self,
        output_dim: int,
        inner_width: int,
        inner_layers: int,
        embedding_dim: int,
        outer_width: int,
        outer_layers: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.phi = make_mlp(1, embedding_dim, inner_width, inner_layers, activation)
        self.rho = make_mlp(embedding_dim, output_dim, outer_width, outer_layers, activation)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        particles = batch["input_particles"]
        if particles.ndim != 2:
            raise ValueError("input_particles must have shape [batch_size, sample_size]")
        particles = particles.unsqueeze(-1)
        batch_size, particle_count, input_dim = particles.shape
        embedded = self.phi(particles.reshape(batch_size * particle_count, input_dim))
        pooled = embedded.reshape(batch_size, particle_count, -1).mean(dim=1)
        return prediction_from_logits(self.rho(pooled))
