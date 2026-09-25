"""Shared neural-network components for active ISI model pipelines."""

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
        layers.extend(
            [nn.Linear(int(hidden_width), int(hidden_width)), activation_module(activation)]
        )
    layers.append(nn.Linear(int(hidden_width), int(output_dim)))
    return nn.Sequential(*layers)


def prediction_from_logits(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    pred_bin_mass = torch.softmax(logits, dim=-1)
    return {
        "logits": logits,
        "pred_bin_mass": pred_bin_mass,
        "pred_cdf": pred_bin_mass.cumsum(dim=-1),
    }
