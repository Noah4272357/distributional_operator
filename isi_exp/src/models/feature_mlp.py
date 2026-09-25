"""Fixed-feature MLP for categorical ISI-law prediction."""

from __future__ import annotations

import torch
from torch import nn

from src.models.components import make_mlp, prediction_from_logits


class ISIFeatureMLP(nn.Module):
    """Predict ISI bin logits from finite-sample moment and Fourier features."""

    def __init__(
        self,
        feature_dim: int,
        output_dim: int,
        hidden_width: int,
        hidden_layers: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.net = make_mlp(feature_dim, output_dim, hidden_width, hidden_layers, activation)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return prediction_from_logits(self.net(batch["input_features"]))
