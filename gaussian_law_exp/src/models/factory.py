"""Factory for distribution-to-Gaussian models."""

from __future__ import annotations

from typing import Any

from torch import nn

from src.models.deepsets import DeepSets
from src.models.kernel_regression import KernelRegression
from src.models.momentmlp import MomentMLP


def build_model(
    model_cfg: Any,
    data_cfg: Any,
    loss_cfg: Any,
    train_payload: dict,
) -> nn.Module:
    """Build the configured model with dimensions inferred from the dataset."""
    del data_cfg
    name = str(model_cfg.name).lower()
    covariance_epsilon = float(loss_cfg.get("covariance_epsilon", 1.0e-5))
    if name in {"kernel_regression", "kernelregression"}:
        return KernelRegression(
            reference_input_dist=train_payload["input_dist"],
            reference_output_dist=train_payload["output_dist"],
            bandwidth=float(model_cfg.get("kernel_bandwidth", 1.0)),
            covariance_epsilon=covariance_epsilon,
            top_k=model_cfg.get("kernel_top_k"),
        )

    input_dim = int(train_payload["input_dist"].shape[-1])
    output_dim = int(train_payload["output_dist"].shape[-1])
    head_kwargs = {
        "covariance_epsilon": covariance_epsilon,
        "initial_scale": float(model_cfg.get("initial_scale", 0.1)),
        "covariance_structure": str(model_cfg.get("covariance_structure", "full")),
        "covariance_rank": int(model_cfg.get("covariance_rank", 4)),
    }
    if name == "deepsets":
        return DeepSets(
            input_dim=input_dim,
            q_out=output_dim,
            inner_width=int(model_cfg.inner_width),
            inner_layers=int(model_cfg.inner_layers),
            embedding_dim=int(model_cfg.embedding_dim),
            outer_width=int(model_cfg.outer_width),
            outer_layers=int(model_cfg.outer_layers),
            activation=str(model_cfg.activation),
            **head_kwargs,
        )
    if name in {"momentmlp", "moment_mlp"}:
        return MomentMLP(
            input_dim=input_dim,
            q_out=output_dim,
            hidden_width=int(model_cfg.hidden_width),
            hidden_layers=int(model_cfg.hidden_layers),
            activation=str(model_cfg.activation),
            **head_kwargs,
        )
    raise ValueError(
        f"unsupported model: {name}; expected deepsets, momentmlp, "
        "or kernel_regression"
    )
