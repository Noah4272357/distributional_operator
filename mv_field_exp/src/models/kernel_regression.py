"""Sinkhorn-kernel regression between empirical input probability laws."""

from __future__ import annotations

import torch
from torch import nn

from src.utils.sinkhorn import build_sinkhorn, sinkhorn_distance


class KernelRegression(nn.Module):
    """Nadaraya--Watson regression with Sinkhorn distances between input laws."""

    def __init__(
        self,
        historical_inputs: torch.Tensor,
        historical_target_mean: torch.Tensor,
        historical_target_covariance: torch.Tensor,
        bandwidth: float,
        sinkhorn_p: int = 2,
        sinkhorn_blur: float = 0.05,
        sinkhorn_scaling: float = 0.9,
        sinkhorn_debias: bool = True,
        sinkhorn_backend: str = "tensorized",
        history_chunk_size: int = 64,
        covariance_epsilon: float = 1.0e-5,
    ) -> None:
        super().__init__()
        if historical_inputs.ndim != 3:
            raise ValueError("historical_inputs must have shape [laws, samples, dimension]")
        if bandwidth <= 0.0:
            raise ValueError("kernel-regression bandwidth must be positive")
        if history_chunk_size <= 0:
            raise ValueError("history_chunk_size must be positive")
        self.bandwidth = float(bandwidth)
        self.history_chunk_size = int(history_chunk_size)
        self.covariance_epsilon = float(covariance_epsilon)
        self.sinkhorn_p = int(sinkhorn_p)
        self.sinkhorn = build_sinkhorn(
            p=int(sinkhorn_p),
            blur=float(sinkhorn_blur),
            scaling=float(sinkhorn_scaling),
            debias=bool(sinkhorn_debias),
            backend=str(sinkhorn_backend),
        )
        self.register_buffer("historical_inputs", historical_inputs.detach().clone())
        self.register_buffer("historical_target_mean", historical_target_mean.detach().clone())
        self.register_buffer("historical_target_covariance", historical_target_covariance.detach().clone())

    def _distances(self, query: torch.Tensor) -> torch.Tensor:
        batch_size, query_samples, dimension = query.shape
        history_size, history_samples, history_dimension = self.historical_inputs.shape
        if dimension != history_dimension:
            raise ValueError("query and historical input dimensions must match")
        chunks = []
        for start in range(0, history_size, self.history_chunk_size):
            history = self.historical_inputs[start : start + self.history_chunk_size]
            chunk_size = int(history.shape[0])
            query_pairs = query[:, None].expand(-1, chunk_size, -1, -1).reshape(
                batch_size * chunk_size, query_samples, dimension
            ).contiguous()
            history_pairs = history[None].expand(batch_size, -1, -1, -1).reshape(
                batch_size * chunk_size, history_samples, dimension
            ).contiguous()
            divergence = self.sinkhorn(query_pairs, history_pairs)
            chunks.append(sinkhorn_distance(divergence, self.sinkhorn_p).reshape(batch_size, chunk_size))
        return torch.cat(chunks, dim=-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        distances = self._distances(batch["context_particles"])
        weights = torch.softmax(-distances.square() / (2.0 * self.bandwidth**2), dim=-1)
        pred_mean = weights @ self.historical_target_mean
        historical_second_moment = self.historical_target_covariance + torch.einsum(
            "ki,kj->kij", self.historical_target_mean, self.historical_target_mean
        )
        pred_second_moment = torch.einsum("bk,kij->bij", weights, historical_second_moment)
        pred_covariance = pred_second_moment - torch.einsum("bi,bj->bij", pred_mean, pred_mean)
        pred_covariance = 0.5 * (pred_covariance + pred_covariance.transpose(-1, -2))
        identity = torch.eye(pred_covariance.shape[-1], device=pred_covariance.device, dtype=pred_covariance.dtype)
        pred_covariance = pred_covariance + self.covariance_epsilon * identity.unsqueeze(0)
        pred_scale_tril = torch.linalg.cholesky(pred_covariance)
        return {
            "pred_mean": pred_mean,
            "pred_cov": pred_covariance,
            "pred_scale_tril": pred_scale_tril,
            "kernel_weights": weights,
            "sinkhorn_distances": distances,
        }
