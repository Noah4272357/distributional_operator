"""Stable model factory for every preserved ISI predictor family."""

from __future__ import annotations

from typing import Any

from torch import nn

from src.models.distribution_operator import DistributionOperator
from src.models.feature_mlp import ISIFeatureMLP
from src.models.kernel_regression import ISIKernelRegression


MAIN_RESULT_MODELS = frozenset({"isi_feature_mlp", "distribution_operator"})
REFERENCE_MODELS = frozenset({"kernel_regression"})
SUPPORTED_MODELS = MAIN_RESULT_MODELS | REFERENCE_MODELS


def model_metadata(model_name: str) -> dict[str, Any]:
    normalized = str(model_name).lower()
    if normalized not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported ISI model: {model_name}")
    main_result_eligible = normalized in MAIN_RESULT_MODELS
    return {
        "model_name": normalized,
        "main_result_eligible": main_result_eligible,
        "result_role": (
            "main_result"
            if main_result_eligible
            else "nonparametric_reference_baseline"
            if normalized == "kernel_regression"
            else "parametric_reference_baseline"
        ),
        "primary_model": normalized == "distribution_operator",
        "fixed_feature_main_baseline": normalized == "isi_feature_mlp",
    }


def build_isi_model(model_cfg: Any, train_payload: dict[str, Any]) -> nn.Module:
    name = str(model_cfg.name).lower()
    output_dim = int(train_payload["bin_counts"].shape[-1])
    if name == "kernel_regression":
        return ISIKernelRegression(
            train_payload,
            bandwidth=float(model_cfg.bandwidth),
            input_bins=int(model_cfg.get("input_bins", 32)),
            distance_chunk_size=int(model_cfg.get("distance_chunk_size", 256)),
        )
    if name == "isi_feature_mlp":
        feature_dim = int(train_payload["input_features"].shape[-1])
        return ISIFeatureMLP(
            feature_dim,
            output_dim,
            model_cfg.hidden_width,
            model_cfg.hidden_layers,
            model_cfg.activation,
        )
    if name == "distribution_operator":
        if str(model_cfg.truncate_dim).lower() == "auto":
            raise ValueError("distribution_operator requires resolved integer model.truncate_dim")
        return DistributionOperator(
            output_dim=output_dim,
            truncate_dim=int(model_cfg.truncate_dim),
            feature_dim=int(model_cfg.feature_dim),
            context_dim=int(model_cfg.context_dim),
            path_mlp_hidden_width=int(model_cfg.path_mlp.hidden_width),
            path_mlp_activation=str(model_cfg.path_mlp.activation),
            deepsets_hidden_width=int(model_cfg.deepsets.hidden_width),
            deepsets_hidden_layers=int(model_cfg.deepsets.hidden_layers),
            deepsets_activation=str(model_cfg.deepsets.activation),
            deepsets_aggregation=str(model_cfg.deepsets.get("aggregation", "mean")),
            deepsets_dropout=float(model_cfg.deepsets.get("dropout", 0.0)),
            rho_hidden_width=int(model_cfg.rho.hidden_width),
            rho_hidden_layers=int(model_cfg.rho.hidden_layers),
            rho_activation=str(model_cfg.rho.activation),
        )
    raise ValueError(f"unsupported ISI model: {name}")


build_model = build_isi_model
