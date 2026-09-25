"""Composable distribution-to-distribution model with a stable tensor pipeline."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .conditional_generator import ConditionalGenerator
from .decoder import Decoder
from .sample_encoder import SampleEncoder


class ModularConditionalModel(nn.Module):
    """Compose sample, distribution, generation, and grid-decoding modules.

    Tensor flow::

        (B,N,G) -> (B,N,H) -> (B,1,C) -> (B,K,D) -> (B,K,G)
    """

    def __init__(
        self,
        *,
        sample_encoder: SampleEncoder,
        distribution_encoder: nn.Module,
        generator: ConditionalGenerator,
        decoder: Decoder,
        grid_points: Tensor,
        num_samples: Optional[int] = None,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if grid_points.ndim not in {1, 2}:
            raise ValueError(
                "grid_points must have shape (grid_size,) or "
                "(grid_size, coordinate_dim)"
            )
        if grid_points.shape[0] != decoder.grid_size:
            raise ValueError("grid_points length must match decoder.grid_size")
        if num_samples is not None and num_samples < 1:
            raise ValueError("num_samples must be positive or None")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        distribution_input_dim = getattr(distribution_encoder, "input_dim", None)
        if distribution_input_dim != sample_encoder.feature_dim:
            raise ValueError(
                "sample encoder feature dimension must match distribution "
                "encoder input dimension"
            )
        if generator.data_dim != decoder.data_dim:
            raise ValueError("generator.data_dim must match decoder.data_dim")
        self.sample_encoder = sample_encoder
        self.distribution_encoder = distribution_encoder
        self.generator = generator
        self.decoder = decoder
        self.num_samples = num_samples
        self.temperature = float(temperature)
        self.register_buffer("grid_points", grid_points.float())

    def encode_samples(self, inputs: Tensor) -> Tensor:
        """Return pointwise trajectory features with shape ``(B,N,H)``."""
        return self.sample_encoder(inputs)

    def encode_distribution(self, sample_features: Tensor) -> Tensor:
        """Return a singleton set context with shape ``(B,1,C)``."""
        context = self.distribution_encoder(sample_features)
        if context.ndim != 2 or context.shape[-1] != self.generator.context_dim:
            raise ValueError(
                "distribution encoder must return (batch, context_dim) matching "
                "the conditional generator"
            )
        return context.unsqueeze(1)

    def generate(
        self,
        context: Tensor,
        num_samples: int,
        temperature: Optional[float] = None,
    ) -> Tensor:
        """Generate coefficient samples with shape ``(B,K,data_dim)``."""
        if context.ndim != 3 or context.shape[1] != 1:
            raise ValueError("context must have shape (batch, 1, context_dim)")
        effective_temperature = (
            self.temperature if temperature is None else float(temperature)
        )
        return self.generator.sample(
            context.squeeze(1), num_samples, effective_temperature
        )

    def decode(self, generated_samples: Tensor) -> Tensor:
        """Decode coefficients to target-grid trajectories."""
        return self.decoder(generated_samples, self.grid_points)

    def sample(
        self,
        inputs: Tensor,
        num_samples: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Tensor:
        """Run the complete conditional generation pipeline."""
        count = self.num_samples if num_samples is None else num_samples
        if count is None:
            count = inputs.shape[1]
        features = self.encode_samples(inputs)
        context = self.encode_distribution(features)
        generated = self.generate(context, count, temperature)
        return self.decode(generated)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sample(inputs)


__all__ = ["ModularConditionalModel"]
