"""Model factory."""

from __future__ import annotations

from torch import nn

from .deep_set_conditional_flow import DeepSetConditionalFlow
from .kernel_regression import KernelRegression


def build_model(config: dict, pca_state: dict) -> nn.Module:
    name = config["name"].lower()
    dimension = int(pca_state["dimension"])

    if name in {
        "deep_set_conditional_flow",
        "deepset_conditional_flow",
        "deepsetconditionalflow",
    }:
        return DeepSetConditionalFlow(
            input_dim=dimension,
            output_dim=dimension,
            sample_embed_dim=int(config.get("sample_embed_dim", 128)),
            context_dim=int(config.get("context_dim", 128)),
            encoder_hidden_dim=int(config.get("encoder_hidden_dim", 256)),
            encoder_phi_layers=int(config.get("encoder_phi_layers", 2)),
            encoder_rho_layers=int(config.get("encoder_rho_layers", 2)),
            aggregation=str(config.get("aggregation", "mean_std")),
            num_flow_layers=int(config.get("num_flow_layers", 8)),
            flow_hidden_dim=int(config.get("flow_hidden_dim", 256)),
            coupling_hidden_layers=int(config.get("coupling_hidden_layers", 2)),
            scale_limit=float(config.get("scale_limit", 2.0)),
            temperature=float(config.get("temperature", 1.0)),
        )
    if name in {"kernel_regression", "kernelregression"}:
        return KernelRegression(
            bandwidth=float(config.get("bandwidth", 1.0)),
            sinkhorn_p=int(config.get("sinkhorn_p", 2)),
            sinkhorn_blur=float(config.get("sinkhorn_blur", 0.05)),
            sinkhorn_debias=bool(config.get("sinkhorn_debias", True)),
            sinkhorn_scaling=float(config.get("sinkhorn_scaling", 0.5)),
            sinkhorn_backend=str(config.get("sinkhorn_backend", "auto")),
            reference_chunk_size=int(config.get("reference_chunk_size", 16)),
        )
    raise ValueError(f"unsupported model: {config['name']}")
