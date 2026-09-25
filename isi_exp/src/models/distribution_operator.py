"""Process-path distribution operator with permutation-invariant aggregation."""

from __future__ import annotations

import torch
from torch import nn

from src.models.components import activation_module, make_mlp, prediction_from_logits


def _element_mlp(
    input_dim: int,
    output_dim: int,
    hidden_width: int,
    hidden_layers: int,
    activation: str,
    dropout: float,
) -> nn.Sequential:
    """Build the shared DeepSets element network."""
    if hidden_layers < 0:
        raise ValueError("hidden_layers cannot be negative")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if hidden_layers == 0:
        return nn.Sequential(nn.Linear(input_dim, output_dim))

    layers: list[nn.Module] = []
    current_dim = input_dim
    for _ in range(hidden_layers):
        layers.extend(
            [
                nn.Linear(current_dim, hidden_width),
                activation_module(activation),
            ]
        )
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_width
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class DeepSetsEncoder(nn.Module):
    """Encode a set of path features into one permutation-invariant context."""

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        hidden_width: int,
        hidden_layers: int,
        activation: str,
        aggregation: str = "mean",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.context_dim = int(context_dim)
        self.aggregation = str(aggregation).lower()
        if self.aggregation != "mean":
            raise ValueError(f"unsupported DeepSets aggregation: {aggregation}")
        self.element_net = _element_mlp(
            self.input_dim,
            self.context_dim,
            int(hidden_width),
            int(hidden_layers),
            activation,
            float(dropout),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("DeepSetsEncoder input must have shape [batch, set_size, feature_dim]")
        batch_size, set_size, feature_dim = features.shape
        if feature_dim != self.input_dim:
            raise ValueError(f"expected feature_dim={self.input_dim}, received {feature_dim}")
        encoded = self.element_net(features.reshape(batch_size * set_size, feature_dim))
        encoded = encoded.reshape(batch_size, set_size, self.context_dim)
        return encoded.mean(dim=1, keepdim=True)


class DistributionOperator(nn.Module):
    """Predict an ISI law from PCA-compressed stochastic-process paths."""

    def __init__(
        self,
        *,
        output_dim: int,
        truncate_dim: int,
        feature_dim: int,
        context_dim: int,
        path_mlp_hidden_width: int,
        path_mlp_activation: str,
        deepsets_hidden_width: int,
        deepsets_hidden_layers: int,
        deepsets_activation: str,
        deepsets_aggregation: str,
        deepsets_dropout: float,
        rho_hidden_width: int,
        rho_hidden_layers: int,
        rho_activation: str,
    ) -> None:
        super().__init__()
        self.truncate_dim = int(truncate_dim)
        self.feature_dim = int(feature_dim)
        if min(self.truncate_dim, self.feature_dim, int(context_dim), int(output_dim)) <= 0:
            raise ValueError("model dimensions must be positive")

        self.path_mlp = nn.Sequential(
            nn.Linear(self.truncate_dim, int(path_mlp_hidden_width)),
            activation_module(path_mlp_activation),
            nn.Linear(int(path_mlp_hidden_width), self.feature_dim),
        )
        self.encoder = DeepSetsEncoder(
            self.feature_dim,
            int(context_dim),
            int(deepsets_hidden_width),
            int(deepsets_hidden_layers),
            deepsets_activation,
            aggregation=deepsets_aggregation,
            dropout=float(deepsets_dropout),
        )
        self.rho = make_mlp(
            int(context_dim),
            int(output_dim),
            int(rho_hidden_width),
            int(rho_hidden_layers),
            rho_activation,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        paths = batch["process_features"]
        if paths.ndim != 3:
            raise ValueError("process_features must have shape [batch, sample_size, truncate_dim]")
        batch_size, sample_size, path_dim = paths.shape
        if path_dim != self.truncate_dim:
            raise ValueError(f"expected truncate_dim={self.truncate_dim}, received {path_dim}")
        features = self.path_mlp(paths.reshape(batch_size * sample_size, path_dim))
        features = features.reshape(batch_size, sample_size, self.feature_dim)
        context = self.encoder(features)
        logits = self.rho(context.squeeze(1))
        return prediction_from_logits(logits)
