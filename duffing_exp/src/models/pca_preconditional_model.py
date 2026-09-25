"""PCA-preprocessed conditional distribution model."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .conditional_generator import ConditionalGenerator


class PCAPreconditionalModel(nn.Module):
    """Generate target PCA coefficients from PCA-compressed input samples.

    Tensor flow::

        (B,N,X_pca) -> (B,1,C) -> (B,K,Y_pca) -> (B,K,G_y)

    Input PCA is applied by the data pipeline using state fitted only on the
    training split. This model stores the corresponding target inverse-PCA
    transform so generated coefficients are returned on the original grid.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        distribution_encoder: nn.Module,
        generator: ConditionalGenerator,
        output_mean: Tensor,
        output_components: Tensor,
        num_samples: Optional[int] = None,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        output_mean = torch.as_tensor(output_mean, dtype=torch.float32)
        output_components = torch.as_tensor(output_components, dtype=torch.float32)
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if output_mean.ndim != 1:
            raise ValueError("output_mean must have shape (output_grid_dim,)")
        if output_components.ndim != 2:
            raise ValueError(
                "output_components must have shape "
                "(output_pca_dim, output_grid_dim)"
            )
        if output_components.shape[1] != output_mean.shape[0]:
            raise ValueError("output PCA mean and components have incompatible shapes")
        distribution_input_dim = getattr(distribution_encoder, "input_dim", None)
        if distribution_input_dim != input_dim:
            raise ValueError(
                "distribution encoder input dimension must match input PCA dimension"
            )
        if generator.data_dim != output_components.shape[0]:
            raise ValueError(
                "generator.data_dim must match the target PCA dimension"
            )
        if num_samples is not None and num_samples < 1:
            raise ValueError("num_samples must be positive or None")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.input_dim = int(input_dim)
        self.output_pca_dim = int(output_components.shape[0])
        self.output_dim = int(output_components.shape[1])
        self.distribution_encoder = distribution_encoder
        self.generator = generator
        self.num_samples = num_samples
        self.temperature = float(temperature)
        self.register_buffer("output_mean", output_mean.clone())
        self.register_buffer("output_components", output_components.clone())

    def _validate_inputs(self, inputs: Tensor) -> None:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"inputs must have shape (batch, samples, {self.input_dim}); "
                f"got {tuple(inputs.shape)}"
            )

    def encode_distribution(self, inputs: Tensor) -> Tensor:
        """Return a singleton context set with shape ``(batch, 1, context_dim)``."""
        self._validate_inputs(inputs)
        context = self.distribution_encoder(inputs)
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
        """Generate target PCA samples with shape ``(batch, K, Y_pca)``."""
        if context.ndim != 3 or context.shape[1] != 1:
            raise ValueError("context must have shape (batch, 1, context_dim)")
        effective_temperature = (
            self.temperature if temperature is None else float(temperature)
        )
        return self.generator.sample(
            context.squeeze(1), num_samples, effective_temperature
        )

    def inverse_output_pca(self, coefficients: Tensor) -> Tensor:
        """Map target PCA coefficients back to the original output grid."""
        if coefficients.ndim != 3 or coefficients.shape[-1] != self.output_pca_dim:
            raise ValueError(
                "coefficients must have shape (batch, samples, output_pca_dim)"
            )
        return coefficients @ self.output_components + self.output_mean

    def sample(
        self,
        inputs: Tensor,
        num_samples: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Tensor:
        """Generate target trajectories on the original output grid."""
        count = self.num_samples if num_samples is None else num_samples
        if count is None:
            count = inputs.shape[1]
        context = self.encode_distribution(inputs)
        coefficients = self.generate(context, count, temperature)
        return self.inverse_output_pca(coefficients)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sample(inputs)


__all__ = ["PCAPreconditionalModel"]
