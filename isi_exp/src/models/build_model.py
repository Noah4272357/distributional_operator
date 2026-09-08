"""Stable model factory for every preserved ISI predictor family."""

from __future__ import annotations

from typing import Any

from torch import nn

from src.models.kernel_regression import ISIKernelRegression
from src.models.isi_models import (
    ISIContextDeepSets,
    ISIFeatureMLP,
    ISIParamMLP,
)


MAIN_RESULT_MODELS = frozenset({"isi_feature_mlp", "isi_context_deepsets"})
REFERENCE_MODELS = frozenset({"isi_param_mlp", "kernel_regression"})
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
        "primary_model": normalized == "isi_context_deepsets",
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
    if name == "isi_param_mlp":
        return ISIParamMLP(output_dim, model_cfg.hidden_width, model_cfg.hidden_layers, model_cfg.activation)
    if name == "isi_feature_mlp":
        feature_dim = int(train_payload["input_features"].shape[-1])
        return ISIFeatureMLP(
            feature_dim,
            output_dim,
            model_cfg.hidden_width,
            model_cfg.hidden_layers,
            model_cfg.activation,
        )
    if name == "isi_context_deepsets":
        return ISIContextDeepSets(
            output_dim,
            model_cfg.inner_width,
            model_cfg.inner_layers,
            model_cfg.embedding_dim,
            model_cfg.outer_width,
            model_cfg.outer_layers,
            model_cfg.activation,
        )
    raise ValueError(f"unsupported ISI model: {name}")


build_model = build_isi_model
