"""Conditional MLP and RealNVP generators used by retained experiments."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    hidden_layers: int,
) -> nn.Sequential:
    if min(input_dim, output_dim, hidden_dim) < 1:
        raise ValueError("MLP dimensions must be positive")
    if hidden_layers < 0:
        raise ValueError("hidden_layers must be non-negative")
    layers: list[nn.Module] = []
    current_dim = input_dim
    for _ in range(hidden_layers):
        layers.extend((nn.Linear(current_dim, hidden_dim), nn.SiLU()))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class ConditionalGenerator(nn.Module, ABC):
    """Common sampling interface for retained conditional generators."""

    context_dim: int
    data_dim: int

    def _validate_context(self, context: Tensor) -> None:
        if context.ndim != 2 or context.shape[-1] != self.context_dim:
            raise ValueError(
                "context must have shape (batch, context_dim); "
                f"expected last dimension {self.context_dim}, got {tuple(context.shape)}"
            )

    @staticmethod
    def _validate_sampling(num_samples: int, temperature: float) -> None:
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

    @abstractmethod
    def sample(
        self, context: Tensor, num_samples: int, temperature: float = 1.0
    ) -> Tensor:
        """Generate samples with shape (batch, num_samples, data_dim)."""

    def forward(
        self, context: Tensor, num_samples: int, temperature: float = 1.0
    ) -> Tensor:
        return self.sample(context, num_samples, temperature)


class ConditionalMLPGenerator(ConditionalGenerator):
    """Implicit generator that decodes noise concatenated with context."""

    def __init__(
        self,
        data_dim: int,
        context_dim: int,
        *,
        latent_dim: int | None = None,
        hidden_dim: int = 256,
        hidden_layers: int = 3,
    ) -> None:
        super().__init__()
        latent_dim = data_dim if latent_dim is None else latent_dim
        if min(data_dim, context_dim, latent_dim, hidden_dim) < 1:
            raise ValueError("generator dimensions must be positive")
        self.data_dim = int(data_dim)
        self.context_dim = int(context_dim)
        self.latent_dim = int(latent_dim)
        self.network = _build_mlp(
            self.latent_dim + self.context_dim,
            self.data_dim,
            hidden_dim,
            hidden_layers,
        )

    def sample(
        self, context: Tensor, num_samples: int, temperature: float = 1.0
    ) -> Tensor:
        self._validate_context(context)
        self._validate_sampling(num_samples, temperature)
        batch_size = context.shape[0]
        noise = temperature * torch.randn(
            batch_size,
            num_samples,
            self.latent_dim,
            device=context.device,
            dtype=context.dtype,
        )
        expanded_context = context[:, None, :].expand(
            batch_size, num_samples, self.context_dim
        )
        return self.network(torch.cat((noise, expanded_context), dim=-1))


class ConditionalAffineCoupling(nn.Module):
    """One affine coupling transformation conditioned on an encoded set."""

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


class ConditionalRealNVP(ConditionalGenerator):
    """Conditional RealNVP with an isotropic normal base distribution."""

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
        if context_dim < 1:
            raise ValueError("context_dim must be positive")
        self.data_dim = int(data_dim)
        self.context_dim = int(context_dim)
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
        if values.ndim != 2 or values.shape[-1] != self.data_dim:
            raise ValueError("values must have shape (items, data_dim)")
        self._validate_context(context)
        if values.shape[0] != context.shape[0]:
            raise ValueError("values and context batch sizes differ")
        latent, log_determinant = self.to_latent(values, context)
        log_base = -0.5 * (latent.square() + math.log(2.0 * math.pi)).sum(dim=-1)
        return log_base + log_determinant

    def sample(
        self, context: Tensor, num_samples: int, temperature: float = 1.0
    ) -> Tensor:
        self._validate_context(context)
        self._validate_sampling(num_samples, temperature)
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


def build_conditional_generator(
    name: str,
    *,
    data_dim: int,
    context_dim: int,
    **parameters,
) -> ConditionalGenerator:
    """Construct a generator used by one of the retained pipelines."""
    normalized_name = name.lower().replace("-", "_")
    generators: dict[str, type[ConditionalGenerator]] = {
        "mlp": ConditionalMLPGenerator,
        "implicit_mlp": ConditionalMLPGenerator,
        "realnvp": ConditionalRealNVP,
        "conditional_realnvp": ConditionalRealNVP,
    }
    try:
        generator_type = generators[normalized_name]
    except KeyError as error:
        supported = ", ".join(sorted(generators))
        raise ValueError(
            f"unsupported retained conditional generator '{name}'; "
            f"choose from: {supported}"
        ) from error
    return generator_type(data_dim=data_dim, context_dim=context_dim, **parameters)


__all__ = [
    "ConditionalGenerator",
    "ConditionalMLPGenerator",
    "ConditionalAffineCoupling",
    "ConditionalRealNVP",
    "build_conditional_generator",
]
