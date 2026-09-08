"""DeepSets-conditioned RealNVP model for distribution-to-distribution learning."""

from __future__ import annotations

import math
from typing import Optional

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


class ConditionalAffineCoupling(nn.Module):
    """One conditional affine coupling transformation."""

    def __init__(
        self,
        data_dim: int,
        context_dim: int,
        *,
        hidden_dim: int,
        mask: Tensor,
        hidden_layers: int,
        scale_limit: float,
    ) -> None:
        super().__init__()
        if mask.shape != (data_dim,):
            raise ValueError(f"mask must have shape ({data_dim},)")
        self.scale_limit = scale_limit
        self.register_buffer("mask", mask.float())
        self.network = _build_mlp(
            data_dim + context_dim, 2 * data_dim, hidden_dim, hidden_layers
        )
        final_layer = self.network[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)

    def _scale_shift(self, values: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        mask = self.mask.unsqueeze(0)
        scale, shift = self.network(
            torch.cat((values * mask, context), dim=-1)
        ).chunk(2, dim=-1)
        scale = self.scale_limit * torch.tanh(scale / self.scale_limit)
        inverse_mask = 1.0 - mask
        return scale * inverse_mask, shift * inverse_mask

    def forward(self, values: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        mask = self.mask.unsqueeze(0)
        scale, shift = self._scale_shift(values, context)
        transformed = mask * values + (1.0 - mask) * (
            values * torch.exp(scale) + shift
        )
        return transformed, scale.sum(dim=-1)

    def inverse(self, values: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        mask = self.mask.unsqueeze(0)
        scale, shift = self._scale_shift(values, context)
        transformed = mask * values + (1.0 - mask) * (
            (values - shift) * torch.exp(-scale)
        )
        return transformed, -scale.sum(dim=-1)


class ConditionalRealNVP(nn.Module):
    """Conditional RealNVP with an isotropic standard-normal base."""

    def __init__(
        self,
        data_dim: int,
        context_dim: int,
        *,
        num_layers: int,
        hidden_dim: int,
        coupling_hidden_layers: int,
        scale_limit: float,
    ) -> None:
        super().__init__()
        if data_dim < 2:
            raise ValueError("RealNVP requires data_dim >= 2")
        self.data_dim = data_dim
        self.context_dim = context_dim
        base_mask = (torch.arange(data_dim) % 2).float()
        self.layers = nn.ModuleList(
            ConditionalAffineCoupling(
                data_dim,
                context_dim,
                hidden_dim=hidden_dim,
                mask=(base_mask if index % 2 == 0 else 1.0 - base_mask).clone(),
                hidden_layers=coupling_hidden_layers,
                scale_limit=scale_limit,
            )
            for index in range(num_layers)
        )

    def to_latent(self, values: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        log_determinant = values.new_zeros(values.shape[0])
        for layer in self.layers:
            values, layer_log_determinant = layer(values, context)
            log_determinant = log_determinant + layer_log_determinant
        return values, log_determinant

    def from_latent(self, values: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        log_determinant = values.new_zeros(values.shape[0])
        for layer in reversed(self.layers):
            values, layer_log_determinant = layer.inverse(values, context)
            log_determinant = log_determinant + layer_log_determinant
        return values, log_determinant

    def log_prob(self, values: Tensor, context: Tensor) -> Tensor:
        latent, log_determinant = self.to_latent(values, context)
        log_base = -0.5 * (latent.square() + math.log(2.0 * math.pi)).sum(dim=-1)
        return log_base + log_determinant

    def sample(self, context: Tensor, num_samples: int, temperature: float) -> Tensor:
        if context.ndim != 2:
            raise ValueError("context must have shape (batch, context_dim)")
        batch_size = context.shape[0]
        latent = temperature * torch.randn(
            batch_size,
            num_samples,
            self.data_dim,
            device=context.device,
            dtype=context.dtype,
        )
        expanded_context = context[:, None, :].expand(
            batch_size, num_samples, self.context_dim
        )
        generated, _ = self.from_latent(
            latent.reshape(-1, self.data_dim),
            expanded_context.reshape(-1, self.context_dim),
        )
        return generated.reshape(batch_size, num_samples, self.data_dim)


class DeepSetConditionalFlow(nn.Module):
    """Map an empirical input distribution to a conditional output distribution."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        sample_embed_dim: int = 128,
        context_dim: int = 128,
        encoder_hidden_dim: int = 256,
        encoder_phi_layers: int = 2,
        encoder_rho_layers: int = 2,
        aggregation: str = "mean_std",
        num_flow_layers: int = 8,
        flow_hidden_dim: int = 256,
        coupling_hidden_layers: int = 2,
        scale_limit: float = 2.0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if input_dim < 1 or output_dim < 2:
            raise ValueError("input_dim must be positive and output_dim must be at least 2")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.output_dim = output_dim
        self.context_dim = context_dim
        self.temperature = temperature
        self.encoder = DeepSetsEncoder(
            input_dim,
            sample_embed_dim=sample_embed_dim,
            context_dim=context_dim,
            hidden_dim=encoder_hidden_dim,
            phi_layers=encoder_phi_layers,
            rho_layers=encoder_rho_layers,
            aggregation=aggregation,
        )
        self.flow = ConditionalRealNVP(
            output_dim,
            context_dim,
            num_layers=num_flow_layers,
            hidden_dim=flow_hidden_dim,
            coupling_hidden_layers=coupling_hidden_layers,
            scale_limit=scale_limit,
        )

    def encode(self, inputs: Tensor) -> Tensor:
        return self.encoder(inputs)

    def log_prob(self, inputs: Tensor, targets: Tensor) -> Tensor:
        if targets.ndim != 3 or targets.shape[-1] != self.output_dim:
            raise ValueError(
                "target must have shape (batch, samples, output_dim); "
                f"got {tuple(targets.shape)}"
            )
        batch_size, num_samples, _ = targets.shape
        if inputs.shape[0] != batch_size:
            raise ValueError("input and target batch sizes differ")
        context = self.encode(inputs)
        expanded_context = context[:, None, :].expand(
            batch_size, num_samples, self.context_dim
        )
        values = targets.reshape(-1, self.output_dim)
        return self.flow.log_prob(
            values, expanded_context.reshape(-1, self.context_dim)
        ).reshape(batch_size, num_samples)

    def nll(self, inputs: Tensor, targets: Tensor, reduction: str = "mean") -> Tensor:
        losses = -self.log_prob(inputs, targets)
        if reduction == "mean":
            return losses.mean()
        if reduction == "sum":
            return losses.sum()
        if reduction == "none":
            return losses
        raise ValueError("reduction must be 'mean', 'sum', or 'none'")

    def sample(
        self,
        inputs: Tensor,
        num_samples: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Tensor:
        if num_samples is None:
            num_samples = inputs.shape[1]
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        effective_temperature = self.temperature if temperature is None else temperature
        if effective_temperature <= 0:
            raise ValueError("temperature must be positive")
        return self.flow.sample(
            self.encode(inputs), num_samples, effective_temperature
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Generate as many output samples as there are input samples."""
        return self.sample(inputs, num_samples=inputs.shape[1])
