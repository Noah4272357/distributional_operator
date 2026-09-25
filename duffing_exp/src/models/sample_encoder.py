"""Encoders that map sampled trajectories to distribution features."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _activation(name: str) -> type[nn.Module]:
    activations: dict[str, type[nn.Module]] = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
    }
    try:
        return activations[name.lower()]
    except KeyError as error:
        supported = ", ".join(sorted(activations))
        raise ValueError(
            f"unsupported activation '{name}'; choose from: {supported}"
        ) from error


class SampleEncoder(nn.Module):
    """Common interface for encoders that return ``(batch, samples, features)``."""

    input_dim: int
    feature_dim: int


class SampleMLPEncoder(SampleEncoder):
    """Map each raw grid trajectory to a learned feature independently."""

    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        *,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if min(input_dim, feature_dim, hidden_dim) < 1:
            raise ValueError("encoder dimensions must be positive")
        if hidden_layers < 0:
            raise ValueError("hidden_layers must be non-negative")
        activation_type = _activation(activation)
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(current_dim, hidden_dim), activation_type()))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, feature_dim))
        self.input_dim = int(input_dim)
        self.feature_dim = int(feature_dim)
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_dim:
            raise ValueError(
                "inputs must have shape (batch, num_samples, input_dim); "
                f"expected last dimension {self.input_dim}, got {tuple(inputs.shape)}"
            )
        return self.network(inputs)


class PCAEncoder(SampleEncoder):
    """Fit and apply a truncated PCA basis independently to each input item.

    For each ``x`` with shape ``(num_samples, input_dim)``, the encoder centers
    the samples, computes the right singular vectors of the centered matrix,
    and projects onto the leading ``feature_dim`` vectors. The resulting shape
    is ``(num_samples, feature_dim)``.
    """

    def __init__(self, input_dim: int, feature_dim: int) -> None:
        super().__init__()
        if min(input_dim, feature_dim) < 1:
            raise ValueError("encoder dimensions must be positive")
        if feature_dim > input_dim:
            raise ValueError("feature_dim cannot exceed input_dim")
        self.input_dim = int(input_dim)
        self.feature_dim = int(feature_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_dim:
            raise ValueError(
                "inputs must have shape (batch, num_samples, input_dim); "
                f"expected last dimension {self.input_dim}, got {tuple(inputs.shape)}"
            )
        if inputs.shape[1] < self.feature_dim:
            raise ValueError(
                "num_samples must be at least feature_dim for per-item PCA; "
                f"got {inputs.shape[1]} samples and feature_dim={self.feature_dim}"
            )
        centered = inputs - inputs.mean(dim=1, keepdim=True)
        _, _, right_singular_vectors = torch.linalg.svd(
            centered, full_matrices=False
        )
        components = right_singular_vectors[..., : self.feature_dim, :]
        return centered @ components.transpose(-2, -1)


def build_sample_encoder(
    name: str,
    *,
    input_dim: int,
    feature_dim: int,
    **parameters,
) -> SampleEncoder:
    """Construct a sample encoder from configuration."""
    normalized_name = name.lower().replace("-", "_")
    if normalized_name in {"mlp", "sample_mlp", "pointwise_mlp"}:
        return SampleMLPEncoder(
            input_dim=input_dim,
            feature_dim=feature_dim,
            **parameters,
        )
    if normalized_name in {"pca", "pca_encoder"}:
        return PCAEncoder(
            input_dim=input_dim,
            feature_dim=feature_dim,
            **parameters,
        )
    raise ValueError(
        f"unsupported sample encoder '{name}'; choose 'mlp' or 'pca_encoder'"
    )


__all__ = [
    "SampleEncoder",
    "SampleMLPEncoder",
    "PCAEncoder",
    "build_sample_encoder",
]
