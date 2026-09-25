"""Reusable permutation-invariant encoders for empirical distributions."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    hidden_layers: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for _ in range(hidden_layers):
        layers.extend((nn.Linear(current_dim, hidden_dim), nn.SiLU()))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class DeepSetsEncoder(nn.Module):
    """Encode a set of samples into one permutation-invariant context vector."""

    def __init__(
        self,
        input_dim: int,
        *,
        sample_embed_dim: int,
        context_dim: int,
        hidden_dim: int,
        phi_layers: int,
        rho_layers: int,
        aggregation: str = "mean_std",
    ) -> None:
        super().__init__()
        if aggregation not in {"mean", "sum", "mean_std"}:
            raise ValueError("aggregation must be 'mean', 'sum', or 'mean_std'")

        self.input_dim = input_dim
        self.aggregation = aggregation
        self.phi = _build_mlp(
            input_dim, sample_embed_dim, hidden_dim, phi_layers
        )
        pooled_dim = 2 * sample_embed_dim if aggregation == "mean_std" else sample_embed_dim
        self.rho = _build_mlp(pooled_dim, context_dim, hidden_dim, rho_layers)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_dim:
            raise ValueError(
                "input must have shape (batch, samples, input_dim); "
                f"got {tuple(inputs.shape)}"
            )
        embeddings = self.phi(inputs)
        if self.aggregation == "mean":
            pooled = embeddings.mean(dim=1)
        elif self.aggregation == "sum":
            pooled = embeddings.sum(dim=1)
        else:
            mean = embeddings.mean(dim=1)
            variance = (embeddings - mean[:, None]).square().mean(dim=1)
            pooled = torch.cat((mean, torch.sqrt(variance + 1.0e-6)), dim=-1)
        return self.rho(pooled)


class MultiheadAttentionBlock(nn.Module):
    """Residual multihead attention followed by a row-wise feed-forward block."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        *,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            model_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(model_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
        )
        self.feed_forward_dropout = nn.Dropout(dropout)
        self.feed_forward_norm = nn.LayerNorm(model_dim)

    def forward(self, queries: Tensor, keys_and_values: Tensor) -> Tensor:
        attended, _ = self.attention(
            queries,
            keys_and_values,
            keys_and_values,
            need_weights=False,
        )
        hidden = self.attention_norm(queries + self.attention_dropout(attended))
        return self.feed_forward_norm(
            hidden + self.feed_forward_dropout(self.feed_forward(hidden))
        )


class InducedSetAttentionBlock(nn.Module):
    """Set attention with learned inducing points and linear set-size scaling."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        num_inducing_points: int,
        *,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.inducing_points = nn.Parameter(
            torch.empty(1, num_inducing_points, model_dim)
        )
        nn.init.xavier_uniform_(self.inducing_points)
        self.inducing_attention = MultiheadAttentionBlock(
            model_dim,
            num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.output_attention = MultiheadAttentionBlock(
            model_dim,
            num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        inducing = self.inducing_points.expand(inputs.shape[0], -1, -1)
        induced_summary = self.inducing_attention(inducing, inputs)
        return self.output_attention(inputs, induced_summary)


class PoolingByMultiheadAttention(nn.Module):
    """Pool a set into a fixed number of learned seed-vector representations."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        num_seeds: int,
        *,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seed_vectors = nn.Parameter(torch.empty(1, num_seeds, model_dim))
        nn.init.xavier_uniform_(self.seed_vectors)
        self.attention = MultiheadAttentionBlock(
            model_dim,
            num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        seeds = self.seed_vectors.expand(inputs.shape[0], -1, -1)
        return self.attention(seeds, inputs)


class SetTransformerEncoder(nn.Module):
    """Encode an empirical distribution into a permutation-invariant context."""

    def __init__(
        self,
        input_dim: int,
        *,
        model_dim: int,
        num_heads: int,
        num_inducing_points: int,
        num_attention_layers: int,
        num_pooling_seeds: int,
        attention_hidden_dim: int,
        context_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if input_dim < 1 or model_dim < 1 or context_dim < 1:
            raise ValueError("input, model, and context dimensions must be positive")
        if num_heads < 1 or model_dim % num_heads:
            raise ValueError("num_heads must be positive and divide model_dim")
        if min(num_inducing_points, num_attention_layers, num_pooling_seeds) < 1:
            raise ValueError("inducing points, attention layers, and pooling seeds must be positive")
        if attention_hidden_dim < 1:
            raise ValueError("attention_hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

        self.input_dim = input_dim
        self.num_pooling_seeds = num_pooling_seeds
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )
        self.attention_blocks = nn.ModuleList(
            InducedSetAttentionBlock(
                model_dim,
                num_heads,
                num_inducing_points,
                hidden_dim=attention_hidden_dim,
                dropout=dropout,
            )
            for _ in range(num_attention_layers)
        )
        self.pooling = PoolingByMultiheadAttention(
            model_dim,
            num_heads,
            num_pooling_seeds,
            hidden_dim=attention_hidden_dim,
            dropout=dropout,
        )
        self.context_projection = nn.Sequential(
            nn.Linear(num_pooling_seeds * model_dim, attention_hidden_dim),
            nn.GELU(),
            nn.Linear(attention_hidden_dim, context_dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_dim:
            raise ValueError(
                "input must have shape (batch, samples, input_dim); "
                f"got {tuple(inputs.shape)}"
            )
        if inputs.shape[1] < 1:
            raise ValueError("input sets must contain at least one sample")
        hidden = self.input_projection(inputs)
        for block in self.attention_blocks:
            hidden = block(hidden)
        pooled = self.pooling(hidden)
        return self.context_projection(pooled.flatten(start_dim=1))


def build_distribution_encoder(
    name: str,
    *,
    input_dim: int,
    context_dim: int,
    **parameters,
) -> nn.Module:
    """Construct a set-level distribution encoder from configuration."""
    normalized_name = name.lower().replace("-", "_")
    if normalized_name in {"deepsets", "deep_sets", "deep_set"}:
        return DeepSetsEncoder(
            input_dim=input_dim,
            context_dim=context_dim,
            sample_embed_dim=int(parameters.get("sample_embed_dim", 128)),
            hidden_dim=int(parameters.get("hidden_dim", 256)),
            phi_layers=int(parameters.get("phi_layers", 2)),
            rho_layers=int(parameters.get("rho_layers", 2)),
            aggregation=str(parameters.get("aggregation", "mean_std")),
        )
    if normalized_name in {"set_transformer", "settransformer"}:
        return SetTransformerEncoder(
            input_dim=input_dim,
            context_dim=context_dim,
            model_dim=int(parameters.get("model_dim", 128)),
            num_heads=int(parameters.get("num_heads", 4)),
            num_inducing_points=int(parameters.get("num_inducing_points", 32)),
            num_attention_layers=int(parameters.get("num_attention_layers", 2)),
            num_pooling_seeds=int(parameters.get("num_pooling_seeds", 1)),
            attention_hidden_dim=int(parameters.get("attention_hidden_dim", 256)),
            dropout=float(parameters.get("dropout", 0.0)),
        )
    raise ValueError(
        f"unsupported distribution encoder '{name}'; "
        "choose 'deepsets' or 'set_transformer'"
    )


__all__ = [
    "DeepSetsEncoder",
    "SetTransformerEncoder",
    "MultiheadAttentionBlock",
    "InducedSetAttentionBlock",
    "PoolingByMultiheadAttention",
    "build_distribution_encoder",
]
