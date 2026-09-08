"""Distribution-space Nadaraya-Watson kernel regression."""

from __future__ import annotations

import torch
from torch import nn

from src.utils.metrics import covariance_to_scale_tril, gaussian_w2


class KernelRegression(nn.Module):
    """Interpolate empirical output moments using input-distribution W2 kernels.

    Historical output means and covariances are estimated exclusively from
    ``output_dist`` samples. Stored target parameters are intentionally not
    accepted by this model.
    """

    def __init__(
        self,
        reference_input_dist: torch.Tensor,
        reference_output_dist: torch.Tensor,
        bandwidth: float,
        covariance_epsilon: float = 1.0e-5,
        top_k: int | None = None,
    ) -> None:
        super().__init__()
        bandwidth = float(bandwidth)
        if bandwidth <= 0.0:
            raise ValueError("kernel bandwidth must be positive")
        if reference_input_dist.shape[0] != reference_output_dist.shape[0]:
            raise ValueError("reference input and output counts must match")

        input_mean, input_cov = self.empirical_moments(reference_input_dist)
        output_mean, output_cov = self.empirical_moments(reference_output_dist)
        reference_count = int(reference_input_dist.shape[0])
        if top_k is not None:
            top_k = int(top_k)
            if top_k <= 0 or top_k > reference_count:
                raise ValueError(
                    f"kernel top_k must be between 1 and {reference_count}"
                )

        self.bandwidth = bandwidth
        self.covariance_epsilon = float(covariance_epsilon)
        self.top_k = top_k
        self.register_buffer("reference_input_mean", input_mean)
        self.register_buffer("reference_input_cov", input_cov)
        self.register_buffer("reference_output_mean", output_mean)
        self.register_buffer("reference_output_cov", output_cov)

    @staticmethod
    def empirical_moments(samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return sample means and unbiased sample covariance matrices."""
        if samples.ndim != 3:
            raise ValueError("samples must have shape [laws, sample_size, q]")
        sample_size = int(samples.shape[1])
        if sample_size < 2:
            raise ValueError("sample_size must be at least two")
        mean = samples.mean(dim=1)
        centered = samples - mean.unsqueeze(1)
        covariance = centered.transpose(-1, -2) @ centered / (sample_size - 1)
        return mean, covariance

    def kernel_weights(self, input_dist: torch.Tensor) -> torch.Tensor:
        """Return normalized RBF weights over historical input distributions."""
        query_mean, query_cov = self.empirical_moments(input_dist)
        distances = gaussian_w2(
            query_mean.unsqueeze(1),
            query_cov.unsqueeze(1),
            self.reference_input_mean.unsqueeze(0),
            self.reference_input_cov.unsqueeze(0),
        )
        logits = -distances.square() / (2.0 * self.bandwidth**2)
        if self.top_k is not None and self.top_k < logits.shape[1]:
            top_logits, top_indices = torch.topk(logits, self.top_k, dim=1)
            masked_logits = torch.full_like(logits, -torch.inf)
            logits = masked_logits.scatter(1, top_indices, top_logits)
        return torch.softmax(logits, dim=1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        weights = self.kernel_weights(batch["input_dist"])
        pred_mean = weights @ self.reference_output_mean
        pred_cov = torch.einsum(
            "bk,kij->bij", weights, self.reference_output_cov
        )
        dimension = pred_cov.shape[-1]
        eye = torch.eye(
            dimension,
            dtype=pred_cov.dtype,
            device=pred_cov.device,
        )
        pred_cov = 0.5 * (pred_cov + pred_cov.transpose(-1, -2))
        pred_cov = pred_cov + self.covariance_epsilon * eye
        pred_scale_tril = covariance_to_scale_tril(pred_cov)
        return {
            "pred_mean": pred_mean,
            "pred_scale_tril": pred_scale_tril,
            "pred_cov": pred_cov,
        }
