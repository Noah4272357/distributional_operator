"""Factory for the three retained Duffing experiment pipelines."""

from __future__ import annotations

import torch
from torch import nn

from .conditional_generator import build_conditional_generator
from .decoder import build_decoder
from .distribution_encoder import build_distribution_encoder
from .kernel_regression import KernelRegression
from .modular_conditional_model import ModularConditionalModel
from .pca_preconditional_model import PCAPreconditionalModel
from .sample_encoder import build_sample_encoder


def _output_pca_transform(state: dict):
    """Fold Y unstandardization into the inverse-PCA buffers."""
    mean, components = state["target_mean"], state["target_components"]
    if state.get("normalization_enabled", False):
        mean = mean + state["target_coefficient_mean"] @ components
        components = state["target_coefficient_scale"][:, None] * components
    return mean, components


def _build_modular_conditional(config: dict, pca_state: dict) -> nn.Module:
    if "input_dimension" in pca_state or "target_components" in pca_state:
        raise ValueError(
            "modular_conditional requires raw grid data; disable PCA preprocessing"
        )
    dimension = int(pca_state["dimension"])
    input_encoder_config = dict(config["input_encoder"])
    input_encoder_name = str(input_encoder_config.pop("name"))
    feature_dim = int(input_encoder_config.pop("feature_dim"))
    sample_encoder = build_sample_encoder(
        input_encoder_name, input_dim=dimension, feature_dim=feature_dim,
        **input_encoder_config,
    )
    distribution_config = dict(config["distribution_encoder"])
    distribution_name = str(distribution_config.pop("name"))
    context_dim = int(distribution_config.pop("context_dim"))
    distribution_encoder = build_distribution_encoder(
        distribution_name, input_dim=feature_dim, context_dim=context_dim,
        **distribution_config,
    )
    generator_config = dict(config["generator"])
    generator_name = str(generator_config.pop("name"))
    data_dim = int(generator_config.pop("data_dim"))
    generator = build_conditional_generator(
        generator_name, data_dim=data_dim, context_dim=context_dim,
        **generator_config,
    )
    decoder_config = dict(config["decoder"])
    decoder_name = str(decoder_config.pop("name"))
    decoder = build_decoder(
        decoder_name, data_dim=data_dim, grid_size=dimension, **decoder_config,
    )
    grid_min = float(config.get("grid_min", 0.0))
    grid_max = float(config.get("grid_max", 1.0))
    if grid_max <= grid_min:
        raise ValueError("model.grid_max must be greater than model.grid_min")
    num_samples = config.get("num_samples")
    return ModularConditionalModel(
        sample_encoder=sample_encoder,
        distribution_encoder=distribution_encoder,
        generator=generator,
        decoder=decoder,
        grid_points=torch.linspace(grid_min, grid_max, dimension),
        num_samples=None if num_samples is None else int(num_samples),
        temperature=float(config.get("temperature", 1.0)),
    )


def _build_pca_preconditional(config: dict, pca_state: dict) -> nn.Module:
    required_state = ("input_dimension", "target_mean", "target_components")
    missing = [key for key in required_state if key not in pca_state]
    if missing:
        raise ValueError(
            "pca_preconditional requires enabled PCA preprocessing; missing state: "
            + ", ".join(missing)
        )
    input_dim = int(pca_state["input_dimension"])
    output_mean, output_components = _output_pca_transform(pca_state)
    distribution_config = dict(config["distribution_encoder"])
    distribution_name = str(distribution_config.pop("name"))
    context_dim = int(distribution_config.pop("context_dim"))
    distribution_encoder = build_distribution_encoder(
        distribution_name, input_dim=input_dim, context_dim=context_dim,
        **distribution_config,
    )
    generator_config = dict(config["generator"])
    generator_name = str(generator_config.pop("name"))
    data_dim = int(generator_config.pop("data_dim"))
    generator = build_conditional_generator(
        generator_name, data_dim=data_dim, context_dim=context_dim,
        **generator_config,
    )
    num_samples = config.get("num_samples")
    return PCAPreconditionalModel(
        input_dim=input_dim,
        distribution_encoder=distribution_encoder,
        generator=generator,
        output_mean=output_mean,
        output_components=output_components,
        num_samples=None if num_samples is None else int(num_samples),
        temperature=float(config.get("temperature", 1.0)),
    )


def build_model(config: dict, pca_state: dict) -> nn.Module:
    """Build one of the retained, reproducible experiment models."""
    name = str(config["name"]).lower()
    if name in {
        "momentmlp",
        "moment_mlp",
        "modular_conditional",
        "modular_conditional_model",
        "conditional_pipeline",
    }:
        return _build_modular_conditional(config, pca_state)
    if name in {
        "distributional_operator",
        "pca_preconditional",
        "pca_preconditional_model",
        "pcapreconditional",
    }:
        return _build_pca_preconditional(config, pca_state)
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
    raise ValueError(
        f"unsupported retained model: {config['name']}; supported pipelines are "
        "kernel_regression, momentMLP, and distributional_operator"
    )
