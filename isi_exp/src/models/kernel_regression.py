"""Nadaraya-Watson regression between empirical input distributions."""

from __future__ import annotations

import torch
from torch import nn

from src.training.metrics import _finite_renormalized_piecewise_uniform_w2


def _histogram_mass(samples: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Convert scalar samples to finite-bin masses plus an empty tail bin."""
    if samples.ndim != 2:
        raise ValueError("input particles must have shape [law_count, sample_size]")
    finite_bins = edges.numel() - 1
    indices = torch.bucketize(samples, edges[1:-1]).clamp(0, finite_bins - 1)
    counts = torch.zeros(
        samples.shape[0], finite_bins, dtype=samples.dtype, device=samples.device
    )
    counts.scatter_add_(1, indices, torch.ones_like(samples))
    finite_mass = counts / counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return torch.cat((finite_mass, torch.zeros_like(finite_mass[:, :1])), dim=-1)


class ISIKernelRegression(nn.Module):
    """Interpolate historical output laws using input-law Wasserstein distance."""

    def __init__(
        self,
        train_payload: dict[str, torch.Tensor],
        *,
        bandwidth: float,
        input_bins: int = 32,
        distance_chunk_size: int = 256,
    ) -> None:
        super().__init__()
        if bandwidth <= 0:
            raise ValueError("kernel bandwidth must be positive")
        if input_bins < 2:
            raise ValueError("input_bins must be at least 2")
        if distance_chunk_size < 1:
            raise ValueError("distance_chunk_size must be positive")
        particles = train_payload["input_particles"].detach().to(torch.float32)
        lower = particles.amin()
        upper = particles.amax()
        margin = (upper - lower).clamp_min(torch.finfo(particles.dtype).eps) * 1.0e-6
        edges = torch.linspace(lower - margin, upper + margin, input_bins + 1)
        self.register_buffer("input_edges", edges)
        self.register_buffer("history_input_mass", _histogram_mass(particles, edges))
        target = train_payload["empirical_bin_mass"].detach().to(torch.float32)
        self.register_buffer(
            "history_output_mass",
            target / target.sum(-1, keepdim=True).clamp_min(1.0e-12),
        )
        self.register_buffer(
            "history_law_ids", train_payload["law_ids"].detach().to(torch.long)
        )
        self.register_buffer("bandwidth", torch.tensor(float(bandwidth), dtype=torch.float32))
        self.distance_chunk_size = int(distance_chunk_size)

    def _distances(self, query_mass: torch.Tensor) -> torch.Tensor:
        chunks = []
        batch_size = query_mass.shape[0]
        for start in range(0, self.history_input_mass.shape[0], self.distance_chunk_size):
            history = self.history_input_mass[start : start + self.distance_chunk_size]
            history_count = history.shape[0]
            left = query_mass[:, None, :].expand(-1, history_count, -1)
            right = history[None, :, :].expand(batch_size, -1, -1)
            distances = _finite_renormalized_piecewise_uniform_w2(
                left.reshape(-1, query_mass.shape[-1]),
                right.reshape(-1, history.shape[-1]),
                self.input_edges,
            )
            chunks.append(distances.reshape(batch_size, history_count))
        return torch.cat(chunks, dim=1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        query_mass = _histogram_mass(
            batch["input_particles"].to(torch.float32), self.input_edges
        )
        distances = self._distances(query_mass)
        if "law_id" in batch and self.history_law_ids.numel() > 1:
            same_law = batch["law_id"][:, None] == self.history_law_ids[None, :]
            distances = distances.masked_fill(same_law, torch.inf)
        kernel_logits = -(distances.square()) / (2.0 * self.bandwidth.square())
        weights = torch.softmax(kernel_logits, dim=-1)
        pred_bin_mass = weights @ self.history_output_mass
        pred_bin_mass = pred_bin_mass.clamp_min(1.0e-12)
        pred_bin_mass = pred_bin_mass / pred_bin_mass.sum(dim=-1, keepdim=True)
        logits = pred_bin_mass.log()
        return {
            "logits": logits,
            "pred_bin_mass": pred_bin_mass,
            "pred_cdf": pred_bin_mass.cumsum(dim=-1),
        }
