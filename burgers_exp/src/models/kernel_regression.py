"""Nadaraya-Watson regression between empirical probability distributions."""

from __future__ import annotations

import torch
from geomloss import SamplesLoss
from torch import Tensor, nn


class KernelRegression(nn.Module):
    """Interpolate historical output laws using Sinkhorn input distances.

    Historical input laws have shape ``(history, samples, dimension)``.  The
    corresponding output laws may have any common trailing shape, such as
    ``(history, samples, dimension)`` or ``(history, bins)``.  Predictions are
    Nadaraya-Watson weighted averages of those aligned output representations.
    """

    def __init__(
        self,
        *,
        bandwidth: float = 1.0,
        sinkhorn_p: int = 2,
        sinkhorn_blur: float = 0.05,
        sinkhorn_debias: bool = True,
        sinkhorn_scaling: float = 0.5,
        sinkhorn_backend: str = "auto",
        reference_chunk_size: int = 16,
    ) -> None:
        super().__init__()
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        if sinkhorn_p not in {1, 2}:
            raise ValueError("sinkhorn_p must be 1 or 2")
        if sinkhorn_blur <= 0:
            raise ValueError("sinkhorn_blur must be positive")
        if not 0 < sinkhorn_scaling < 1:
            raise ValueError("sinkhorn_scaling must lie between zero and one")
        if reference_chunk_size < 1:
            raise ValueError("reference_chunk_size must be positive")

        self.bandwidth = float(bandwidth)
        self.reference_chunk_size = int(reference_chunk_size)
        self.sinkhorn = SamplesLoss(
            loss="sinkhorn",
            p=sinkhorn_p,
            blur=float(sinkhorn_blur),
            debias=bool(sinkhorn_debias),
            scaling=float(sinkhorn_scaling),
            backend=sinkhorn_backend,
        )
        self.register_buffer("reference_inputs", torch.empty(0), persistent=True)
        self.register_buffer("reference_targets", torch.empty(0), persistent=True)

    @property
    def is_fitted(self) -> bool:
        """Whether historical input/output laws have been attached."""
        return self.reference_inputs.numel() > 0

    def fit(self, inputs: Tensor, targets: Tensor) -> "KernelRegression":
        """Store historical pairs and return this model.

        The stored tensors are detached snapshots, so fitting does not retain a
        caller's autograd graph.  Calling ``fit`` again replaces the history.
        """
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape (history, samples, dimension)")
        if targets.ndim < 2:
            raise ValueError("targets must have shape (history, ...)")
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError("inputs and targets must contain the same history size")
        if inputs.shape[0] == 0 or inputs.shape[1] == 0 or inputs.shape[2] == 0:
            raise ValueError("historical input laws must be non-empty")
        if not inputs.is_floating_point() or not targets.is_floating_point():
            raise TypeError("inputs and targets must be floating-point tensors")
        if not torch.isfinite(inputs).all() or not torch.isfinite(targets).all():
            raise ValueError("inputs and targets must contain only finite values")

        device = self.reference_inputs.device
        dtype = inputs.dtype
        self.reference_inputs = inputs.detach().to(device=device, dtype=dtype).clone()
        self.reference_targets = targets.detach().to(device=device, dtype=dtype).clone()
        return self

    def squared_distances(self, inputs: Tensor) -> Tensor:
        """Compute GeomLoss Sinkhorn divergences to all historical input laws."""
        self._validate_queries(inputs)
        batch_size = inputs.shape[0]
        distances: list[Tensor] = []
        for start in range(
            0, self.reference_inputs.shape[0], self.reference_chunk_size
        ):
            references = self.reference_inputs[
                start : start + self.reference_chunk_size
            ]
            chunk_size = references.shape[0]
            query_pairs = inputs[:, None].expand(
                batch_size, chunk_size, *inputs.shape[1:]
            )
            reference_pairs = references[None].expand(
                batch_size, chunk_size, *references.shape[1:]
            )
            chunk_distances = self.sinkhorn(
                query_pairs.reshape(-1, *inputs.shape[1:]).contiguous(),
                reference_pairs.reshape(-1, *references.shape[1:]).contiguous(),
            ).reshape(batch_size, chunk_size)
            distances.append(chunk_distances.clamp_min(0.0))
        return torch.cat(distances, dim=1)

    def weights_from_squared_distances(self, squared_distances: Tensor) -> Tensor:
        """Convert a precomputed ``(batch, history)`` distance matrix to weights."""
        if squared_distances.ndim != 2:
            raise ValueError("squared_distances must have shape (batch, history)")
        if squared_distances.shape[1] != self.reference_targets.shape[0]:
            raise ValueError("distance history dimension does not match fitted targets")
        logits = -squared_distances / (2.0 * self.bandwidth**2)
        return torch.softmax(logits, dim=1)

    def kernel_weights(self, inputs: Tensor) -> Tensor:
        """Return normalized Gaussian-kernel weights over the history."""
        return self.weights_from_squared_distances(self.squared_distances(inputs))

    def predict_from_squared_distances(self, squared_distances: Tensor) -> Tensor:
        """Predict outputs while reusing precomputed Sinkhorn divergences."""
        weights = self.weights_from_squared_distances(squared_distances)
        flattened_targets = self.reference_targets.reshape(
            self.reference_targets.shape[0], -1
        )
        prediction = weights @ flattened_targets
        return prediction.reshape(
            squared_distances.shape[0], *self.reference_targets.shape[1:]
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Predict output-law representations for a batch of query laws."""
        return self.predict_from_squared_distances(self.squared_distances(inputs))

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # Fitted histories are dynamic-size buffers. Resize empty buffers before
        # nn.Module performs its otherwise shape-strict state restoration.
        for name in ("reference_inputs", "reference_targets"):
            key = prefix + name
            if (
                key in state_dict
                and getattr(self, name).shape != state_dict[key].shape
            ):
                current = getattr(self, name)
                setattr(
                    self,
                    name,
                    torch.empty_like(state_dict[key], device=current.device),
                )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _validate_queries(self, inputs: Tensor) -> None:
        if not self.is_fitted:
            raise RuntimeError("KernelRegression must be fitted before prediction")
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape (batch, samples, dimension)")
        if inputs.shape[1] == 0:
            raise ValueError("query laws must contain at least one sample")
        if inputs.shape[-1] != self.reference_inputs.shape[-1]:
            raise ValueError("query and historical input dimensions differ")
        if inputs.device != self.reference_inputs.device:
            raise ValueError("queries and fitted history must be on the same device")
        if inputs.dtype != self.reference_inputs.dtype:
            raise ValueError("queries and fitted history must have the same dtype")
        if not torch.isfinite(inputs).all():
            raise ValueError("inputs must contain only finite values")
