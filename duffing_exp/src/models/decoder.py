"""Decoders from generator coordinates to values on a target grid.

Both decoders use the same public API::

    predictions = decoder(generator_samples, grid_points)

``generator_samples`` must have shape ``(batch, num_samples, data_dim)`` and
the result has shape ``(batch, num_samples, grid_size)``. This lets callers
place either decoder directly after any generator in
``conditional_generator.py`` without changing the surrounding tensor flow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

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


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    hidden_layers: int,
    activation: str,
) -> nn.Sequential:
    if min(input_dim, output_dim, hidden_dim) < 1:
        raise ValueError("MLP dimensions must be positive")
    if hidden_layers < 0:
        raise ValueError("hidden_layers must be non-negative")
    activation_type = _activation(activation)
    layers: list[nn.Module] = []
    current_dim = input_dim
    for _ in range(hidden_layers):
        layers.extend((nn.Linear(current_dim, hidden_dim), activation_type()))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class Decoder(nn.Module, ABC):
    """Common interface for decoding generator samples onto a fixed-size grid."""

    def __init__(self, data_dim: int, grid_size: int) -> None:
        super().__init__()
        if min(data_dim, grid_size) < 1:
            raise ValueError("data_dim and grid_size must be positive")
        self.data_dim = int(data_dim)
        self.grid_size = int(grid_size)

    def _validate_samples(self, samples: Tensor) -> None:
        if samples.ndim != 3 or samples.shape[-1] != self.data_dim:
            raise ValueError(
                "samples must have shape (batch, num_samples, data_dim); "
                f"expected last dimension {self.data_dim}, got {tuple(samples.shape)}"
            )

    def _validate_grid_size(self, grid_points: Tensor) -> None:
        if grid_points.ndim not in {1, 2}:
            raise ValueError(
                "grid_points must have shape (grid_size,) or "
                "(grid_size, coordinate_dim)"
            )
        if grid_points.shape[0] != self.grid_size:
            raise ValueError(
                f"expected {self.grid_size} grid points, got {grid_points.shape[0]}"
            )

    @abstractmethod
    def decode(
        self, samples: Tensor, grid_points: Tensor | None = None
    ) -> Tensor:
        """Decode ``(B,N,data_dim)`` samples to ``(B,N,grid_size)``."""

    def forward(
        self, samples: Tensor, grid_points: Tensor | None = None
    ) -> Tensor:
        return self.decode(samples, grid_points)


class DirectMLPDecoder(Decoder):
    """Map each generated coordinate vector directly onto the target grid."""

    def __init__(
        self,
        data_dim: int,
        grid_size: int,
        *,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
        activation: str = "silu",
    ) -> None:
        super().__init__(data_dim, grid_size)
        self.network = _build_mlp(
            data_dim,
            grid_size,
            hidden_dim,
            hidden_layers,
            activation,
        )

    def decode(
        self, samples: Tensor, grid_points: Tensor | None = None
    ) -> Tensor:
        self._validate_samples(samples)
        if grid_points is not None:
            self._validate_grid_size(grid_points)
        return self.network(samples)


class ImplicitNeuralRepresentationDecoder(Decoder):
    r"""Decode with a learned coordinate-dependent basis.

    An MLP represents

    .. math::

        t \mapsto \phi(t) = (\phi_1(t), \ldots, \phi_D(t)),

    where ``D=data_dim``. For generated coefficients ``z``, the prediction at
    a grid point is ``<z, phi(t)>``.
    """

    def __init__(
        self,
        data_dim: int,
        grid_size: int,
        *,
        coordinate_dim: int = 1,
        hidden_dim: int = 256,
        hidden_layers: int = 3,
        activation: str = "silu",
    ) -> None:
        super().__init__(data_dim, grid_size)
        if coordinate_dim < 1:
            raise ValueError("coordinate_dim must be positive")
        self.coordinate_dim = int(coordinate_dim)
        self.basis_network = _build_mlp(
            coordinate_dim,
            data_dim,
            hidden_dim,
            hidden_layers,
            activation,
        )

    def _prepare_grid_points(self, grid_points: Tensor, samples: Tensor) -> Tensor:
        self._validate_grid_size(grid_points)
        if grid_points.ndim == 1:
            if self.coordinate_dim != 1:
                raise ValueError(
                    "one-dimensional grid_points require coordinate_dim=1"
                )
            grid_points = grid_points.unsqueeze(-1)
        if grid_points.shape[-1] != self.coordinate_dim:
            raise ValueError(
                "grid point coordinate dimension differs from coordinate_dim; "
                f"expected {self.coordinate_dim}, got {grid_points.shape[-1]}"
            )
        return grid_points.to(device=samples.device, dtype=samples.dtype)

    def basis(self, grid_points: Tensor, *, reference: Tensor | None = None) -> Tensor:
        """Evaluate ``phi`` and return shape ``(grid_size, data_dim)``."""
        if reference is None:
            parameter = next(self.parameters())
            reference = parameter
        prepared_points = self._prepare_grid_points(grid_points, reference)
        return self.basis_network(prepared_points)

    def decode(
        self, samples: Tensor, grid_points: Tensor | None = None
    ) -> Tensor:
        self._validate_samples(samples)
        if grid_points is None:
            raise ValueError("grid_points are required by the implicit decoder")
        basis_values = self.basis(grid_points, reference=samples)
        return torch.einsum("bnd,gd->bng", samples, basis_values)


def build_decoder(
    name: str,
    *,
    data_dim: int,
    grid_size: int,
    **parameters,
) -> Decoder:
    """Construct a decoder from a compact configuration name."""
    normalized_name = name.lower().replace("-", "_")
    decoders: dict[str, type[Decoder]] = {
        "direct_mlp": DirectMLPDecoder,
        "mlp": DirectMLPDecoder,
        "implicit": ImplicitNeuralRepresentationDecoder,
        "implicit_neural_representation": ImplicitNeuralRepresentationDecoder,
        "inr": ImplicitNeuralRepresentationDecoder,
    }
    try:
        decoder_type = decoders[normalized_name]
    except KeyError as error:
        supported = ", ".join(sorted(decoders))
        raise ValueError(
            f"unsupported decoder '{name}'; choose from: {supported}"
        ) from error
    return decoder_type(
        data_dim=data_dim,
        grid_size=grid_size,
        **parameters,
    )


__all__ = [
    "Decoder",
    "DirectMLPDecoder",
    "ImplicitNeuralRepresentationDecoder",
    "build_decoder",
]
