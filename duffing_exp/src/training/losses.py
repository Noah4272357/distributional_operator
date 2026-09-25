"""Loss factory."""

from __future__ import annotations

from torch import Tensor, nn


class SinkhornLoss(nn.Module):
    """Sinkhorn divergence with optional sample-mean MSE regularization."""

    def __init__(
        self,
        *,
        p: int = 2,
        blur: float = 0.05,
        debias: bool = True,
        scaling: float = 0.5,
        backend: str = "auto",
        reduction: str = "mean",
        mean_loss_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        if mean_loss_weight < 0:
            raise ValueError("mean_loss_weight must be non-negative")
        try:
            from geomloss import SamplesLoss
        except ImportError as error:
            raise ImportError(
                "the Sinkhorn loss requires the optional 'geomloss' package"
            ) from error
        self.reduction = reduction
        self.mean_loss_weight = float(mean_loss_weight)
        self.loss = SamplesLoss(
            loss="sinkhorn",
            p=p,
            blur=blur,
            debias=debias,
            scaling=scaling,
            backend=backend,
        )

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        if prediction.shape != target.shape or prediction.ndim != 3:
            raise ValueError(
                "prediction and target must share shape "
                "(batch, samples, dimension)"
            )
        values = self.loss(prediction.contiguous(), target.contiguous())
        prediction_mean = prediction.mean(dim=1)
        target_mean = target.mean(dim=1)
        mean_loss = (prediction_mean - target_mean).square().mean(dim=-1)
        values = values + self.mean_loss_weight * mean_loss
        if self.reduction == "mean":
            return values.mean()
        if self.reduction == "sum":
            return values.sum()
        return values


def build_loss(config: dict) -> nn.Module:
    """Build a loss from its configuration mapping."""
    name = config["name"].lower()
    if name == "mse":
        return nn.MSELoss(reduction=str(config.get("reduction", "mean")))
    if name in {"sinkhorn", "samples_loss"}:
        return SinkhornLoss(
            p=int(config.get("p", 2)),
            blur=float(config.get("blur", 0.05)),
            debias=bool(config.get("debias", True)),
            scaling=float(config.get("scaling", 0.5)),
            backend=str(config.get("backend", "auto")),
            reduction=str(config.get("reduction", "mean")),
            mean_loss_weight=float(config.get("mean_loss_weight", 0.0)),
        )
    raise ValueError(f"unsupported loss: {config['name']}")
