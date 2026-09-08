"""Shared helpers for Phase 4/5 random-field law-to-law training."""

from __future__ import annotations

import csv
import json
import random
import shutil
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from dataset.io import load_yaml
from src.data.dataset import load_random_field_dataset_splits, make_random_field_datasets
from src.models.gaussian_ops import fit_batched_empirical_gaussian, gaussian_nll, gaussian_w2, scale_tril_to_covariance
from src.models.baselines import GlobalConstantPredictor, MomentOnlyGaussianPredictor
from src.models.deepsets import DeepSets
from src.models.kernel_regression import KernelRegression
from src.training.factory import build_optimizer, build_scheduler, step_scheduler
from src.utils.artifacts import (
    BASIS_METADATA_YAML,
    BEST_METRICS_ARTIFACT,
    EMPIRICAL_GAUSSIAN_ORACLE_FLOOR_JSON,
    EVAL_BY_SPLIT_ARTIFACT,
    GENERATION_MANIFEST_COPY_YAML,
    GENERATION_SUMMARY_YAML,
    PROJECTION_SUMMARY_YAML,
    RESULT_ARTIFACT,
    RESULT_TABLE_ARTIFACT,
    TRAINING_HISTORY_CSV,
)
from src.utils.field_metrics import (
    field_energy_distance,
    field_mmd,
    field_variogram_discrepancy,
    integrated_crps,
    sample_predicted_field_particles,
)
from src.utils.metrics import relative_frobenius_error, relative_l2_error, sliced_w2_distance
from src.utils.sinkhorn import build_sinkhorn, sinkhorn_distance

ROOT = Path(__file__).resolve().parents[2]
MAIN_RESULT_MODELS = frozenset({"global_constant", "moment_only", "deepsets", "kernel_regression"})
SUPPORTED_MODELS = MAIN_RESULT_MODELS
SUPPORTED_LOSSES = frozenset(
    {
        "gaussian_nll",
        "param_supervised",
        "gaussian_nll_plus_param",
        "field_marginal_ce",
        "gaussian_nll_plus_field_marginal_ce",
        "coefficient_energy",
        "gaussian_nll_plus_energy",
    }
)
LOSS_ROLES = {
    "gaussian_nll": "main_objective",
    "param_supervised": "debug_only",
    "gaussian_nll_plus_param": "synthetic_oracle_assisted_ablation",
    "field_marginal_ce": "function_space_pseudo_likelihood",
    "gaussian_nll_plus_field_marginal_ce": "coefficient_and_function_space_objective",
    "coefficient_energy": "sample_based_objective",
    "gaussian_nll_plus_energy": "coefficient_nll_and_sample_energy_objective",
}
CHECKPOINT_SELECTION_METRIC = "val.observable_metrics.projected_coeff_nll"
FIELD_MARGINAL_CHECKPOINT_SELECTION_METRIC = "val.training_loss"
TRAINING_LOSS_CHECKPOINT_LOSSES = frozenset(
    {
        "param_supervised",
        "gaussian_nll_plus_param",
        "field_marginal_ce",
        "gaussian_nll_plus_field_marginal_ce",
        "coefficient_energy",
        "gaussian_nll_plus_energy",
    }
)
OUTPUT_PROTOCOL = "learned_output_basis_gaussian"
PARTICLE_TENSOR_DIM = 3
TRAINING_HISTORY_FIELDNAMES = [
    "epoch",
    "learning_rate",
    "train_loss",
    "val_loss",
    "val_checkpoint_score",
    "is_best",
    "val_projected_coeff_nll",
    "val_field_energy_distance",
    "val_field_sample_sliced_w2",
    "val_field_sinkhorn_distance",
    "val_field_marginal_ce",
    "val_field_mmd",
    "val_integrated_crps",
    "val_field_variogram_discrepancy",
    "val_field_gaussian_w2",
    "val_field_mean_l2_error",
    "val_field_cov_hs_error",
]


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _required_cfg_get(cfg: Any, key: str) -> Any:
    value = _cfg_get(cfg, key)
    if value is None:
        raise ValueError(f"missing required config value: {key}")
    return value


def _covariance_epsilon(loss_cfg: Any) -> float:
    return float(_cfg_get(loss_cfg, "covariance_epsilon", 1.0e-5))


def _batch_size(batch: dict[str, Any]) -> int:
    for value in batch.values():
        if torch.is_tensor(value):
            return int(value.shape[0])
    raise ValueError("batch must contain at least one tensor")


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _load_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor], device: torch.device) -> None:
    model.load_state_dict({key: value.to(device) for key, value in state_dict.items()})


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "phase",
        "input_representation",
        "output_protocol",
        "model_name",
        "loss_name",
        "loss_role",
        "main_result_eligible",
        "metric_group",
        "metric_name",
        "metric_value",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_training_history_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRAINING_HISTORY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def _infer_input_dim(train_payload: dict[str, torch.Tensor], input_representation: str) -> int:
    if input_representation == "learned_basis_projected":
        return int(train_payload["input_projected_particles"].shape[-1])
    if input_representation == "direct_grid":
        return int(train_payload["input_field_particles"].shape[-1])
    raise ValueError(f"unsupported input_representation: {input_representation}")


def _infer_q_out(train_payload: dict[str, torch.Tensor]) -> int:
    return int(train_payload["output_projected_particles"].shape[-1])


def build_random_field_model(
    model_cfg: Any,
    loss_cfg: Any,
    train_payload: dict[str, torch.Tensor],
    input_representation: str,
) -> nn.Module:
    """Build a configured random-field model from split dimensions."""
    model_name = str(_required_cfg_get(model_cfg, "name")).lower()
    covariance_epsilon = _covariance_epsilon(loss_cfg)
    initial_scale = float(_cfg_get(model_cfg, "initial_scale", 0.1))
    covariance_structure = str(_cfg_get(model_cfg, "covariance_structure", "full"))
    covariance_rank = int(_cfg_get(model_cfg, "covariance_rank", 4))
    input_dim = _infer_input_dim(train_payload, input_representation)
    q_out = _infer_q_out(train_payload)

    if model_name == "global_constant":
        return GlobalConstantPredictor.fit_from_output_particles(
            train_payload["output_projected_particles"],
            covariance_epsilon=covariance_epsilon,
        )

    if model_name == "moment_only":
        return MomentOnlyGaussianPredictor(
            q_in=input_dim,
            q_out=q_out,
            hidden_width=int(_required_cfg_get(model_cfg, "hidden_width")),
            hidden_layers=int(_required_cfg_get(model_cfg, "hidden_layers")),
            activation=str(_required_cfg_get(model_cfg, "activation")),
            covariance_epsilon=covariance_epsilon,
            initial_scale=initial_scale,
            covariance_structure=covariance_structure,
            covariance_rank=covariance_rank,
        )

    if model_name == "deepsets":
        return DeepSets(
            input_dim=input_dim,
            q_out=q_out,
            inner_width=int(_required_cfg_get(model_cfg, "inner_width")),
            inner_layers=int(_required_cfg_get(model_cfg, "inner_layers")),
            embedding_dim=int(_required_cfg_get(model_cfg, "embedding_dim")),
            outer_width=int(_required_cfg_get(model_cfg, "outer_width")),
            outer_layers=int(_required_cfg_get(model_cfg, "outer_layers")),
            activation=str(_required_cfg_get(model_cfg, "activation")),
            covariance_epsilon=covariance_epsilon,
            initial_scale=initial_scale,
            covariance_structure=covariance_structure,
            covariance_rank=covariance_rank,
        )

    if model_name == "kernel_regression":
        sinkhorn_cfg = _cfg_get(model_cfg, "sinkhorn", {})
        context_key = _input_particles_key(input_representation)
        return KernelRegression(
            historical_inputs=train_payload[context_key],
            historical_target_mean=train_payload["target_projected_output_mean"],
            historical_target_covariance=train_payload["target_projected_output_cov"],
            bandwidth=float(_required_cfg_get(model_cfg, "bandwidth")),
            sinkhorn_p=int(_cfg_get(sinkhorn_cfg, "p", 2)),
            sinkhorn_blur=float(_cfg_get(sinkhorn_cfg, "blur", 0.05)),
            sinkhorn_scaling=float(_cfg_get(sinkhorn_cfg, "scaling", 0.9)),
            sinkhorn_debias=bool(_cfg_get(sinkhorn_cfg, "debias", True)),
            sinkhorn_backend=str(_cfg_get(sinkhorn_cfg, "backend", "tensorized")),
            history_chunk_size=int(_cfg_get(model_cfg, "history_chunk_size", 64)),
            covariance_epsilon=covariance_epsilon,
        )

    if model_name == "oracle_feature":
        raise ValueError("oracle_feature is not supported for random-field Phase 3-5 v1")
    raise ValueError(f"unsupported model: {model_name}")


def _param_supervised_loss(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    loss_cfg: Any,
) -> torch.Tensor:
    required_keys = ("target_projected_output_mean", "target_projected_output_cov")
    missing = [key for key in required_keys if key not in batch]
    if missing:
        raise ValueError(f"param_supervised requires diagnostic keys: {', '.join(missing)}")
    mean_weight = float(_cfg_get(loss_cfg, "mean_weight", 1.0))
    covariance_weight = float(_cfg_get(loss_cfg, "covariance_weight", 1.0))
    mean_loss = F.mse_loss(prediction["pred_mean"], batch["target_projected_output_mean"])
    covariance_loss = F.mse_loss(prediction["pred_cov"], batch["target_projected_output_cov"])
    return mean_weight * mean_loss + covariance_weight * covariance_loss


def _field_marginal_variance_floor(loss_cfg: Any) -> float:
    variance_floor = float(_cfg_get(loss_cfg, "field_marginal_variance_floor", 1.0e-4))
    if variance_floor < 0.0:
        raise ValueError("field_marginal_variance_floor cannot be negative")
    return variance_floor


def _require_field_metric_context(
    field_metric_context: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    if field_metric_context is None:
        raise ValueError("field_marginal_ce requires field_metric_context")
    return field_metric_context


def field_marginal_cross_entropy(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    loss_cfg: Any,
    field_metric_context: dict[str, torch.Tensor] | None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Weighted pointwise Gaussian cross-entropy on decoded field marginals."""
    context = _require_field_metric_context(field_metric_context)
    target_device = torch.device(device) if device is not None else prediction["pred_mean"].device
    pred_grid_mean, pred_grid_cov = _predicted_grid_moments(prediction, context, target_device)
    pred_var = pred_grid_cov.diagonal(dim1=-2, dim2=-1).clamp_min(0.0) + _field_marginal_variance_floor(loss_cfg)

    if "target_output_grid_mean" in batch and "target_output_grid_cov" in batch:
        target_mean = batch["target_output_grid_mean"].to(device=target_device, dtype=pred_grid_mean.dtype)
        target_cov = batch["target_output_grid_cov"].to(device=target_device, dtype=pred_grid_mean.dtype)
        target_var = target_cov.diagonal(dim1=-2, dim2=-1).clamp_min(0.0)
    elif "output_field_particles" in batch:
        target_particles = batch["output_field_particles"].to(device=target_device, dtype=pred_grid_mean.dtype)
        target_mean = target_particles.mean(dim=1)
        target_var = target_particles.var(dim=1, unbiased=False).clamp_min(0.0)
    else:
        raise ValueError(
            "field_marginal_ce requires target_output_grid_mean/target_output_grid_cov or output_field_particles"
        )
    output_weights = context["output_weights"].to(device=target_device, dtype=pred_grid_mean.dtype)
    normalized_weights = output_weights / output_weights.sum().clamp_min(1.0e-12)

    mean_error_sq = (target_mean - pred_grid_mean).square()
    pointwise_ce = 0.5 * (pred_var.log() + (target_var + mean_error_sq) / pred_var)
    return (pointwise_ce * normalized_weights.view(1, -1)).sum(dim=-1).mean()


def coefficient_energy_distance(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Energy distance between predicted and target coefficient particles for each law."""
    if predicted.ndim != PARTICLE_TENSOR_DIM or target.ndim != PARTICLE_TENSOR_DIM:
        raise ValueError("coefficient_energy_distance expects [batch, samples, dim] tensors")
    if predicted.shape[0] != target.shape[0] or predicted.shape[-1] != target.shape[-1]:
        raise ValueError("predicted and target coefficient particles must share batch size and dimension")
    cross = torch.cdist(predicted, target).mean(dim=(1, 2))
    predicted_self = torch.cdist(predicted, predicted).mean(dim=(1, 2))
    target_self = torch.cdist(target, target).mean(dim=(1, 2))
    return 2.0 * cross - predicted_self - target_self


def _coefficient_energy_sample_count(loss_cfg: Any, batch: dict[str, torch.Tensor]) -> int:
    samples = int(_cfg_get(loss_cfg, "energy_samples", batch["output_particles"].shape[1]))
    if samples <= 0:
        raise ValueError("energy_samples must be positive")
    return samples


def _sample_predicted_coefficient_particles(
    prediction: dict[str, torch.Tensor],
    num_samples: int,
    seed: int | None,
) -> torch.Tensor:
    pred_mean = prediction["pred_mean"]
    pred_scale_tril = prediction["pred_scale_tril"]
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pred_mean.device)
        generator.manual_seed(int(seed))
    eps = torch.randn(
        pred_mean.shape[0],
        int(num_samples),
        pred_mean.shape[-1],
        device=pred_mean.device,
        dtype=pred_mean.dtype,
        generator=generator,
    )
    return pred_mean.unsqueeze(1) + torch.matmul(eps, pred_scale_tril.transpose(-1, -2))


def coefficient_energy_loss(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    loss_cfg: Any,
) -> torch.Tensor:
    """Sample-based coefficient-space energy distance for a predicted Gaussian law."""
    target_particles = batch["output_particles"].to(
        device=prediction["pred_mean"].device,
        dtype=prediction["pred_mean"].dtype,
    )
    seed_value = _cfg_get(loss_cfg, "energy_seed", 0)
    seed = None if seed_value is None else int(seed_value)
    predicted_particles = _sample_predicted_coefficient_particles(
        prediction,
        _coefficient_energy_sample_count(loss_cfg, batch),
        seed,
    )
    return coefficient_energy_distance(predicted_particles, target_particles).mean()


def compute_random_field_loss(
    loss_name: str,
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    loss_cfg: Any,
    field_metric_context: dict[str, torch.Tensor] | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Compute a random-field training objective from projected output coefficients."""
    normalized_loss = str(loss_name).lower()
    if normalized_loss == "gaussian_nll":
        loss = gaussian_nll(prediction["pred_mean"], prediction["pred_scale_tril"], batch["output_particles"])
    elif normalized_loss == "param_supervised":
        loss = _param_supervised_loss(prediction, batch, loss_cfg)
    elif normalized_loss == "gaussian_nll_plus_param":
        param_weight = float(_cfg_get(loss_cfg, "param_weight", 1.0))
        loss = compute_random_field_loss("gaussian_nll", prediction, batch, loss_cfg) + param_weight * _param_supervised_loss(
            prediction,
            batch,
            loss_cfg,
        )
    elif normalized_loss == "field_marginal_ce":
        loss = field_marginal_cross_entropy(prediction, batch, loss_cfg, field_metric_context, device)
    elif normalized_loss == "gaussian_nll_plus_field_marginal_ce":
        field_weight = float(_cfg_get(loss_cfg, "field_marginal_weight", 1.0))
        loss = compute_random_field_loss("gaussian_nll", prediction, batch, loss_cfg) + field_weight * compute_random_field_loss(
            "field_marginal_ce",
            prediction,
            batch,
            loss_cfg,
            field_metric_context,
            device,
        )
    elif normalized_loss == "coefficient_energy":
        loss = coefficient_energy_loss(prediction, batch, loss_cfg)
    elif normalized_loss == "gaussian_nll_plus_energy":
        energy_weight = float(_cfg_get(loss_cfg, "energy_weight", 1.0))
        loss = compute_random_field_loss("gaussian_nll", prediction, batch, loss_cfg) + energy_weight * coefficient_energy_loss(
            prediction,
            batch,
            loss_cfg,
        )
    else:
        raise ValueError(f"unsupported loss: {loss_name}")
    return loss


def loss_metadata(loss_name: str, model_name: str) -> dict[str, Any]:
    """Return result-selection metadata for a random-field loss/model combination."""
    normalized_loss = str(loss_name).lower()
    normalized_model = str(model_name).lower()
    if normalized_loss not in SUPPORTED_LOSSES:
        raise ValueError(f"unsupported loss: {loss_name}")
    if normalized_model == "oracle_feature":
        raise ValueError("oracle_feature is not supported for random-field Phase 3-5 v1")
    if normalized_model not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported model: {model_name}")
    return {
        "loss_name": normalized_loss,
        "model_name": normalized_model,
        "loss_role": LOSS_ROLES[normalized_loss],
        "main_result_eligible": normalized_loss == "gaussian_nll" and normalized_model in MAIN_RESULT_MODELS,
    }


def _checkpoint_selection_metric(loss_name: str) -> str:
    normalized_loss = str(loss_name).lower()
    if normalized_loss in TRAINING_LOSS_CHECKPOINT_LOSSES:
        return FIELD_MARGINAL_CHECKPOINT_SELECTION_METRIC
    return CHECKPOINT_SELECTION_METRIC


def _checkpoint_score(metrics: dict[str, Any], metric_path: str) -> float:
    if metric_path == "val.training_loss":
        return float(metrics["training_loss"])
    if metric_path == "val.observable_metrics.projected_coeff_nll":
        return float(metrics["observable_metrics"]["projected_coeff_nll"])
    raise ValueError(f"unsupported checkpoint metric: {metric_path}")


def move_batch_to_device(batch: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    """Copy a collated batch dict to a device, preserving non-tensor values."""
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


class _StandardizedRandomFieldModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        input_mean: torch.Tensor | None = None,
        input_scale: torch.Tensor | None = None,
        output_mean: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.has_input_standardization = input_mean is not None and input_scale is not None
        self.has_output_standardization = output_mean is not None and output_scale is not None
        self.register_buffer("input_mean", input_mean if input_mean is not None else torch.empty(0))
        self.register_buffer("input_scale", input_scale if input_scale is not None else torch.empty(0))
        self.register_buffer("output_mean", output_mean if output_mean is not None else torch.empty(0))
        self.register_buffer("output_scale", output_scale if output_scale is not None else torch.empty(0))

    @staticmethod
    def _feature_view(statistic: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        return statistic.reshape((1,) * (tensor.ndim - 1) + (-1,))

    @staticmethod
    def _scale_tril_view(statistic: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        return statistic.reshape((1,) * (tensor.ndim - 2) + (-1, 1))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        model_batch = dict(batch)
        if self.has_input_standardization:
            context_particles = model_batch["context_particles"]
            model_batch["context_particles"] = (
                context_particles - self._feature_view(self.input_mean, context_particles)
            ) / self._feature_view(self.input_scale, context_particles)

        prediction = dict(self.base_model(model_batch))
        if not self.has_output_standardization:
            return prediction

        pred_mean = prediction["pred_mean"]
        pred_scale_tril = prediction["pred_scale_tril"]
        output_mean = self._feature_view(self.output_mean, pred_mean)
        output_scale = self._feature_view(self.output_scale, pred_mean)
        scale_tril_scale = self._scale_tril_view(self.output_scale, pred_scale_tril)
        prediction["pred_mean"] = output_mean + pred_mean * output_scale
        prediction["pred_scale_tril"] = pred_scale_tril * scale_tril_scale
        prediction["pred_cov"] = scale_tril_to_covariance(prediction["pred_scale_tril"])
        if "pred_particles" in prediction:
            pred_particles = prediction["pred_particles"]
            pred_particle_mean = self._feature_view(self.output_mean, pred_particles)
            pred_particle_scale = self._feature_view(self.output_scale, pred_particles)
            prediction["pred_particles"] = pred_particle_mean + pred_particles * pred_particle_scale
        return prediction


def _standardization_metadata(standardization_cfg: Any) -> dict[str, Any]:
    enabled = bool(_cfg_get(standardization_cfg, "enabled", False))
    epsilon = float(_cfg_get(standardization_cfg, "epsilon", 1.0e-6))
    if epsilon <= 0.0:
        raise ValueError("standardization.epsilon must be positive")
    return {
        "enabled": enabled,
        "input": enabled and bool(_cfg_get(standardization_cfg, "input", True)),
        "output": enabled and bool(_cfg_get(standardization_cfg, "output", True)),
        "epsilon": epsilon,
    }


def _fit_feature_standardization(values: torch.Tensor, epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
    flat_values = values.reshape(-1, values.shape[-1])
    mean = flat_values.mean(dim=0)
    scale = flat_values.std(dim=0, unbiased=False).clamp_min(float(epsilon))
    return mean, scale


def _input_particles_key(input_representation: str) -> str:
    if input_representation == "learned_basis_projected":
        return "input_projected_particles"
    if input_representation == "direct_grid":
        return "input_field_particles"
    raise ValueError(f"unsupported input_representation: {input_representation}")


def _standardization_stats(
    train_payload: dict[str, torch.Tensor],
    input_representation: str,
    metadata: dict[str, Any],
) -> dict[str, torch.Tensor | None]:
    if not metadata["enabled"]:
        return {"input_mean": None, "input_scale": None, "output_mean": None, "output_scale": None}
    stats: dict[str, torch.Tensor | None] = {
        "input_mean": None,
        "input_scale": None,
        "output_mean": None,
        "output_scale": None,
    }
    if metadata["input"]:
        input_mean, input_scale = _fit_feature_standardization(
            train_payload[_input_particles_key(input_representation)],
            float(metadata["epsilon"]),
        )
        stats["input_mean"] = input_mean
        stats["input_scale"] = input_scale
    if metadata["output"]:
        output_mean, output_scale = _fit_feature_standardization(
            train_payload["output_projected_particles"],
            float(metadata["epsilon"]),
        )
        stats["output_mean"] = output_mean
        stats["output_scale"] = output_scale
    return stats


def _standardized_model_init_payload(
    train_payload: dict[str, torch.Tensor],
    stats: dict[str, torch.Tensor | None],
) -> dict[str, torch.Tensor]:
    output_mean = stats["output_mean"]
    output_scale = stats["output_scale"]
    if output_mean is None or output_scale is None:
        return train_payload
    payload = dict(train_payload)
    payload["output_projected_particles"] = (
        train_payload["output_projected_particles"] - output_mean.view(1, 1, -1)
    ) / output_scale.view(1, 1, -1)
    return payload


def _optional_particle_count(view_cfg: Any, key: str) -> int | None:
    value = _cfg_get(view_cfg, key, None)
    if value is None:
        return None
    count = int(value)
    if count <= 0:
        raise ValueError(f"random_field_view.{key} must be positive")
    return count


def _limit_random_field_split_payload(
    split_payload: dict[str, Any],
    input_representation: str,
    context_particle_count: int | None,
    output_particle_count: int | None,
) -> dict[str, Any]:
    limited_payload = dict(split_payload)
    if context_particle_count is not None:
        context_key = _input_particles_key(input_representation)
        available_context = int(split_payload[context_key].shape[1])
        if context_particle_count > available_context:
            raise ValueError(
                f"random_field_view.context_particles={context_particle_count} "
                f"exceeds available context particles {available_context}"
            )
        for key in ("input_projected_particles", "input_field_particles"):
            if key in split_payload:
                limited_payload[key] = split_payload[key][:, :context_particle_count]
    if output_particle_count is not None:
        available_output = int(split_payload["output_projected_particles"].shape[1])
        if output_particle_count > available_output:
            raise ValueError(
                f"random_field_view.output_particles={output_particle_count} "
                f"exceeds available output particles {available_output}"
            )
        for key in ("output_projected_particles", "output_field_particles"):
            if key in split_payload:
                limited_payload[key] = split_payload[key][:, :output_particle_count]
    return limited_payload


def _limit_random_field_split_payloads(
    split_payloads: dict[str, dict[str, Any]],
    input_representation: str,
    context_particle_count: int | None,
    output_particle_count: int | None,
) -> dict[str, dict[str, Any]]:
    if context_particle_count is None and output_particle_count is None:
        return split_payloads
    return {
        split_name: _limit_random_field_split_payload(
            split_payload,
            input_representation,
            context_particle_count,
            output_particle_count,
        )
        for split_name, split_payload in split_payloads.items()
    }


def _mean_weighted(total: float, count: int) -> float:
    if count == 0:
        return float("nan")
    return float(total / count)


def _tensor_mean_float(value: torch.Tensor) -> float:
    return float(value.detach().mean().cpu().item())


def _cfg_enabled(cfg: Any, key: str) -> bool:
    nested = _cfg_get(cfg, key, None)
    return bool(_cfg_get(nested, "enabled", False))


def _make_variogram_pair_indices(grid: int, variogram_cfg: Any, device: torch.device) -> torch.Tensor | None:
    max_pairs = _cfg_get(variogram_cfg, "max_pairs", None)
    if max_pairs is None:
        return None
    left, right = torch.triu_indices(int(grid), int(grid), offset=1)
    pair_count = int(left.numel())
    max_pairs = int(max_pairs)
    if max_pairs <= 0:
        raise ValueError("variogram.max_pairs must be positive")
    if pair_count <= max_pairs:
        return torch.stack([left, right]).to(device=device)
    generator = torch.Generator().manual_seed(int(_cfg_get(variogram_cfg, "seed", 0)))
    selected = torch.randperm(pair_count, generator=generator)[:max_pairs]
    return torch.stack([left[selected], right[selected]]).to(device=device)


def _load_field_metric_context(generated_data_cfg: Any) -> dict[str, torch.Tensor]:
    files_cfg = _required_cfg_get(generated_data_cfg, "files")
    basis_metadata = load_yaml(_resolve_path(_required_cfg_get(files_cfg, "basis_metadata")))
    output_basis = basis_metadata["output_basis"]
    return {
        "output_center": torch.tensor(output_basis["center"], dtype=torch.float32),
        "output_basis_values": torch.tensor(output_basis["basis_values"], dtype=torch.float32),
        "output_weights": torch.tensor(output_basis["quadrature_weights"], dtype=torch.float32),
    }


def _is_non_gaussian_sample_based(generated_data_cfg: Any) -> bool:
    target_cfg = _cfg_get(generated_data_cfg, "target", {})
    return str(_cfg_get(target_cfg, "target_distribution_role", "")) == "non_gaussian_sample_based"


def _predicted_grid_moments(
    prediction: dict[str, torch.Tensor],
    field_metric_context: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    output_center = field_metric_context["output_center"].to(device=device, dtype=prediction["pred_mean"].dtype)
    output_basis_values = field_metric_context["output_basis_values"].to(device=device, dtype=prediction["pred_mean"].dtype)
    pred_grid_mean = output_center.unsqueeze(0) + prediction["pred_mean"] @ output_basis_values.transpose(0, 1)
    pred_grid_cov = torch.einsum("gq,bqr,hr->bgh", output_basis_values, prediction["pred_cov"], output_basis_values)
    return pred_grid_mean, pred_grid_cov


def _field_gaussian_diagnostics(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    field_metric_context: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    diagnostics: dict[str, float] = {}
    if "target_projected_output_mean" in batch and "target_projected_output_cov" in batch:
        diagnostics["z_e_m"] = _tensor_mean_float(
            relative_l2_error(prediction["pred_mean"], batch["target_projected_output_mean"])
        )
        diagnostics["z_e_C"] = _tensor_mean_float(
            relative_frobenius_error(prediction["pred_cov"], batch["target_projected_output_cov"])
        )
        diagnostics["z_gaussian_w2"] = _tensor_mean_float(
            gaussian_w2(
                prediction["pred_mean"],
                prediction["pred_cov"],
                batch["target_projected_output_mean"],
                batch["target_projected_output_cov"],
            )
        )
    if "target_output_grid_mean" in batch and "target_output_grid_cov" in batch:
        weights = field_metric_context["output_weights"].to(device=device, dtype=prediction["pred_mean"].dtype)
        pred_grid_mean, pred_grid_cov = _predicted_grid_moments(prediction, field_metric_context, device)
        target_grid_mean = batch["target_output_grid_mean"]
        target_grid_cov = batch["target_output_grid_cov"]
        mean_num = ((pred_grid_mean - target_grid_mean).square() * weights.view(1, -1)).sum(dim=-1).sqrt()
        mean_den = (target_grid_mean.square() * weights.view(1, -1)).sum(dim=-1).sqrt().clamp_min(1.0e-8)
        cov_weights = weights.view(1, -1, 1) * weights.view(1, 1, -1)
        cov_num = ((pred_grid_cov - target_grid_cov).square() * cov_weights).sum(dim=(-2, -1)).sqrt()
        cov_den = (target_grid_cov.square() * cov_weights).sum(dim=(-2, -1)).sqrt().clamp_min(1.0e-8)
        diagnostics["field_gaussian_w2"] = _tensor_mean_float(
            gaussian_w2(pred_grid_mean, pred_grid_cov, target_grid_mean, target_grid_cov)
        )
        diagnostics["field_mean_l2_error"] = _tensor_mean_float(mean_num / mean_den)
        diagnostics["field_cov_hs_error"] = _tensor_mean_float(cov_num / cov_den)
    return diagnostics


@torch.no_grad()
def evaluate_random_field_model(
    model: nn.Module,
    dataloader: DataLoader,
    loss_name: str,
    loss_cfg: Any,
    device: torch.device | str,
    field_metric_context: dict[str, torch.Tensor],
    metrics_cfg: Any | None = None,
) -> dict[str, Any]:
    """Evaluate training loss, projected coefficient NLL, and field sample metrics."""
    target_device = torch.device(device)
    was_training = model.training
    model.eval()

    total_items = 0
    total_training_loss = 0.0
    observable_totals: dict[str, float] = {}
    observable_counts: dict[str, int] = {}
    diagnostic_totals: dict[str, float] = {}
    diagnostic_counts: dict[str, int] = {}

    output_center = field_metric_context["output_center"].to(target_device)
    output_basis_values = field_metric_context["output_basis_values"].to(target_device)
    output_weights = field_metric_context["output_weights"].to(target_device)
    sqrt_weights = output_weights.sqrt().view(1, 1, -1)
    sliced_w2_cfg = _cfg_get(metrics_cfg, "sliced_w2", {})
    mmd_cfg = _cfg_get(metrics_cfg, "mmd", {})
    variogram_cfg = _cfg_get(metrics_cfg, "variogram", {})
    sinkhorn_cfg = _cfg_get(metrics_cfg, "sinkhorn", {})
    sinkhorn_metric = None
    sinkhorn_p = int(_cfg_get(sinkhorn_cfg, "p", 2))
    if bool(_cfg_get(sinkhorn_cfg, "enabled", False)):
        sinkhorn_metric = build_sinkhorn(
            p=sinkhorn_p,
            blur=float(_cfg_get(sinkhorn_cfg, "blur", 0.05)),
            scaling=float(_cfg_get(sinkhorn_cfg, "scaling", 0.9)),
            debias=bool(_cfg_get(sinkhorn_cfg, "debias", True)),
            backend=str(_cfg_get(sinkhorn_cfg, "backend", "tensorized")),
        )
    variogram_pair_indices: torch.Tensor | None = None

    for batch_index, batch in enumerate(dataloader):
        device_batch = move_batch_to_device(batch, target_device)
        prediction = model(device_batch)
        batch_size = _batch_size(device_batch)
        training_loss = compute_random_field_loss(
            loss_name,
            prediction,
            device_batch,
            loss_cfg,
            field_metric_context,
            target_device,
        )
        projected_coeff_nll = compute_random_field_loss("gaussian_nll", prediction, device_batch, loss_cfg)
        num_predicted_samples = int(
            _cfg_get(metrics_cfg, "predicted_field_samples", device_batch["output_field_particles"].shape[1])
        )
        predicted_fields = sample_predicted_field_particles(
            prediction["pred_mean"],
            prediction["pred_scale_tril"],
            output_center,
            output_basis_values,
            num_samples=num_predicted_samples,
            seed=int(_cfg_get(sliced_w2_cfg, "seed", 0)) + batch_index,
        )
        target_fields = device_batch["output_field_particles"]
        min_particles = min(int(predicted_fields.shape[1]), int(target_fields.shape[1]))
        weighted_predicted = predicted_fields[:, :min_particles] * sqrt_weights
        weighted_target = target_fields[:, :min_particles] * sqrt_weights
        observable_metrics = {
            "projected_coeff_nll": float(projected_coeff_nll.detach().cpu().item()),
            "field_energy_distance": _tensor_mean_float(field_energy_distance(predicted_fields, target_fields, output_weights)),
            "field_sample_sliced_w2": _tensor_mean_float(
                sliced_w2_distance(
                    weighted_predicted,
                    weighted_target,
                    num_projections=int(_cfg_get(sliced_w2_cfg, "num_projections", 128)),
                    seed=int(_cfg_get(sliced_w2_cfg, "seed", 0)),
                )
            ),
        }
        if sinkhorn_metric is not None:
            observable_metrics["field_sinkhorn_distance"] = _tensor_mean_float(
                sinkhorn_distance(sinkhorn_metric(weighted_predicted, weighted_target), sinkhorn_p)
            )
        has_target_grid_moments = "target_output_grid_mean" in device_batch and "target_output_grid_cov" in device_batch
        if has_target_grid_moments or "output_field_particles" in device_batch:
            field_marginal_ce = compute_random_field_loss(
                "field_marginal_ce",
                prediction,
                device_batch,
                loss_cfg,
                field_metric_context,
                target_device,
            )
            observable_metrics["field_marginal_ce"] = float(field_marginal_ce.detach().cpu().item())
        if _cfg_enabled(metrics_cfg, "mmd"):
            observable_metrics["field_mmd"] = _tensor_mean_float(
                field_mmd(predicted_fields, target_fields, output_weights, scales=_cfg_get(mmd_cfg, "scales", (0.5, 1.0, 2.0, 4.0)))
            )
        if _cfg_enabled(metrics_cfg, "integrated_crps"):
            observable_metrics["integrated_crps"] = _tensor_mean_float(
                integrated_crps(predicted_fields, target_fields, output_weights)
            )
        if _cfg_enabled(metrics_cfg, "variogram"):
            if variogram_pair_indices is None:
                variogram_pair_indices = _make_variogram_pair_indices(target_fields.shape[-1], variogram_cfg, target_device)
            observable_metrics["field_variogram_discrepancy"] = _tensor_mean_float(
                field_variogram_discrepancy(
                    predicted_fields,
                    target_fields,
                    output_weights,
                    p=float(_cfg_get(variogram_cfg, "p", 1.0)),
                    pair_indices=variogram_pair_indices,
                )
            )

        diagnostics = _field_gaussian_diagnostics(prediction, device_batch, field_metric_context, target_device)
        total_items += batch_size
        total_training_loss += float(training_loss.detach().cpu().item()) * batch_size
        for metric_name, metric_value in observable_metrics.items():
            observable_totals[metric_name] = observable_totals.get(metric_name, 0.0) + float(metric_value) * batch_size
            observable_counts[metric_name] = observable_counts.get(metric_name, 0) + batch_size
        for metric_name, metric_value in diagnostics.items():
            diagnostic_totals[metric_name] = diagnostic_totals.get(metric_name, 0.0) + float(metric_value) * batch_size
            diagnostic_counts[metric_name] = diagnostic_counts.get(metric_name, 0) + batch_size

    if was_training:
        model.train()

    return {
        "training_loss": _mean_weighted(total_training_loss, total_items),
        "observable_metrics": {
            key: _mean_weighted(value, observable_counts[key]) for key, value in observable_totals.items()
        },
        "synthetic_diagnostic_metrics": {
            key: _mean_weighted(value, diagnostic_counts[key]) for key, value in diagnostic_totals.items()
        },
    }


@torch.no_grad()
def evaluate_empirical_gaussian_oracle_floor(
    dataloader: DataLoader,
    loss_cfg: Any,
    device: torch.device | str,
    field_metric_context: dict[str, torch.Tensor],
    metrics_cfg: Any | None = None,
) -> dict[str, Any]:
    """Evaluate the per-law empirical Gaussian floor from true output particles."""

    class _BatchOracle(nn.Module):
        def __init__(self, covariance_epsilon: float) -> None:
            super().__init__()
            self.covariance_epsilon = float(covariance_epsilon)

        def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            mean, cov, scale_tril = fit_batched_empirical_gaussian(
                batch["output_particles"],
                covariance_epsilon=self.covariance_epsilon,
            )
            return {"pred_mean": mean, "pred_cov": cov, "pred_scale_tril": scale_tril}

    oracle = _BatchOracle(_covariance_epsilon(loss_cfg)).to(device)
    return evaluate_random_field_model(
        oracle,
        dataloader,
        "gaussian_nll",
        loss_cfg,
        device,
        field_metric_context,
        metrics_cfg=metrics_cfg,
    )


def _evaluate_empirical_gaussian_oracle_floor_by_split(
    dataloaders: dict[str, DataLoader],
    loss_cfg: Any,
    device: torch.device | str,
    field_metric_context: dict[str, torch.Tensor],
    metrics_cfg: Any | None = None,
) -> dict[str, Any]:
    return {
        split_name: evaluate_empirical_gaussian_oracle_floor(
            dataloader,
            loss_cfg,
            device,
            field_metric_context,
            metrics_cfg=metrics_cfg,
        )
        for split_name, dataloader in dataloaders.items()
    }


def flatten_eval_rows(
    results: dict[str, Any],
    model_name: str,
    loss_name: str,
    loss_role: str,
    main_result_eligible: bool,
) -> list[dict[str, Any]]:
    """Flatten random-field split metrics into CSV rows."""
    rows: list[dict[str, Any]] = []
    for split_name, split_metrics in results["metrics_by_split"].items():
        base_row = {
            "split": split_name,
            "phase": results["phase"],
            "input_representation": results["input_representation"],
            "output_protocol": results["output_protocol"],
            "model_name": model_name,
            "loss_name": loss_name,
            "loss_role": loss_role,
            "main_result_eligible": bool(main_result_eligible),
        }
        rows.append(
            {
                **base_row,
                "metric_group": "training",
                "metric_name": "loss",
                "metric_value": float(split_metrics["training_loss"]),
            }
        )
        for metric_group in ("observable_metrics", "synthetic_diagnostic_metrics"):
            for metric_name, metric_value in split_metrics[metric_group].items():
                rows.append(
                    {
                        **base_row,
                        "metric_group": metric_group,
                        "metric_name": metric_name,
                        "metric_value": float(metric_value),
                    }
                )
    return rows


def _copy_random_field_metadata_artifacts(generated_data_cfg: Any, output_dir: Path) -> None:
    files_cfg = _required_cfg_get(generated_data_cfg, "files")
    copies = {
        "manifest": GENERATION_MANIFEST_COPY_YAML,
        "basis_metadata": BASIS_METADATA_YAML,
        "projection_summary": PROJECTION_SUMMARY_YAML,
        "generation_summary": GENERATION_SUMMARY_YAML,
    }
    for source_key, target_name in copies.items():
        shutil.copyfile(_resolve_path(_required_cfg_get(files_cfg, source_key)), output_dir / target_name)


def _resolve_output_dir(cfg: Any, output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    experiment_cfg = _required_cfg_get(cfg, "experiment")
    save_path = Path(str(_cfg_get(experiment_cfg, "save_path", "logs")))
    experiment_name = str(_cfg_get(experiment_cfg, "name", "random_field_experiment"))
    return save_path / experiment_name


def _make_dataloaders(
    datasets: dict[str, torch.utils.data.Dataset],
    experiment_cfg: Any,
) -> dict[str, DataLoader]:
    batch_size = int(_cfg_get(experiment_cfg, "batch_size", 64))
    num_workers = int(_cfg_get(experiment_cfg, "num_workers", 0))
    pin_memory = bool(_cfg_get(experiment_cfg, "pin_memory", False))
    return {
        split_name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split_name == "train",
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        for split_name, dataset in datasets.items()
    }


def _has_trainable_parameters(model: nn.Module) -> bool:
    return any(parameter.requires_grad for parameter in model.parameters())


def _history_metric(metrics: dict[str, Any], group: str, metric_name: str) -> float | str:
    value = metrics[group].get(metric_name)
    if value is None:
        return ""
    return float(value)


def _training_history_row(
    epoch: int,
    learning_rate: float,
    train_loss: float,
    val_metrics: dict[str, Any],
    val_checkpoint_score: float,
    is_best: bool,
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "learning_rate": float(learning_rate),
        "train_loss": float(train_loss),
        "val_loss": float(val_metrics["training_loss"]),
        "val_checkpoint_score": float(val_checkpoint_score),
        "is_best": "true" if is_best else "false",
        "val_projected_coeff_nll": _history_metric(val_metrics, "observable_metrics", "projected_coeff_nll"),
        "val_field_energy_distance": _history_metric(val_metrics, "observable_metrics", "field_energy_distance"),
        "val_field_sample_sliced_w2": _history_metric(val_metrics, "observable_metrics", "field_sample_sliced_w2"),
        "val_field_sinkhorn_distance": _history_metric(
            val_metrics, "observable_metrics", "field_sinkhorn_distance"
        ),
        "val_field_marginal_ce": _history_metric(val_metrics, "observable_metrics", "field_marginal_ce"),
        "val_field_mmd": _history_metric(val_metrics, "observable_metrics", "field_mmd"),
        "val_integrated_crps": _history_metric(val_metrics, "observable_metrics", "integrated_crps"),
        "val_field_variogram_discrepancy": _history_metric(
            val_metrics,
            "observable_metrics",
            "field_variogram_discrepancy",
        ),
        "val_field_gaussian_w2": _history_metric(val_metrics, "synthetic_diagnostic_metrics", "field_gaussian_w2"),
        "val_field_mean_l2_error": _history_metric(val_metrics, "synthetic_diagnostic_metrics", "field_mean_l2_error"),
        "val_field_cov_hs_error": _history_metric(val_metrics, "synthetic_diagnostic_metrics", "field_cov_hs_error"),
    }


def _train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_name: str,
    loss_cfg: Any,
    device: torch.device,
    field_metric_context: dict[str, torch.Tensor],
) -> float:
    model.train()
    total_items = 0
    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)
        device_batch = move_batch_to_device(batch, device)
        prediction = model(device_batch)
        loss = compute_random_field_loss(loss_name, prediction, device_batch, loss_cfg, field_metric_context, device)
        batch_size = _batch_size(device_batch)
        total_items += batch_size
        total_loss += float(loss.detach().cpu().item()) * batch_size
        loss.backward()
        optimizer.step()
    return _mean_weighted(total_loss, total_items)


def _phase_name(input_representation: str) -> str:
    if input_representation == "learned_basis_projected":
        return "phase4"
    if input_representation == "direct_grid":
        return "phase5"
    raise ValueError(f"unsupported input_representation: {input_representation}")


def run_random_field_experiment(cfg: Any, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Run a Phase 4/5 random-field helper-level training/evaluation job."""
    generated_data_cfg = _required_cfg_get(cfg, "generated_data")
    model_cfg = _required_cfg_get(cfg, "model")
    loss_cfg = _required_cfg_get(cfg, "loss")
    experiment_cfg = _required_cfg_get(cfg, "experiment")
    random_field_view_cfg = _required_cfg_get(cfg, "random_field_view")
    model_name = str(_required_cfg_get(model_cfg, "name")).lower()
    loss_name = str(_required_cfg_get(loss_cfg, "name")).lower()
    input_representation = str(_required_cfg_get(random_field_view_cfg, "input_representation"))
    output_protocol = str(_cfg_get(random_field_view_cfg, "output_protocol", _cfg_get(generated_data_cfg, "output_protocol", OUTPUT_PROTOCOL)))
    if output_protocol != OUTPUT_PROTOCOL:
        raise ValueError(f"unsupported output_protocol: {output_protocol}")

    metadata = loss_metadata(loss_name, model_name)
    metrics_cfg = _cfg_get(cfg, "metrics", None)
    standardization = _standardization_metadata(_cfg_get(cfg, "standardization", None))
    if model_name == "kernel_regression" and standardization["enabled"]:
        raise ValueError("kernel_regression currently requires standardization.enabled=false")
    device = torch.device(str(_cfg_get(experiment_cfg, "device", "cpu")))
    output_path = _resolve_output_dir(cfg, output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _set_seed(int(_cfg_get(experiment_cfg, "seed", 0)))
    split_payloads = load_random_field_dataset_splits(generated_data_cfg, map_location="cpu")
    context_particle_count = _optional_particle_count(random_field_view_cfg, "context_particles")
    output_particle_count = _optional_particle_count(random_field_view_cfg, "output_particles")
    split_payloads = _limit_random_field_split_payloads(
        split_payloads,
        input_representation,
        context_particle_count,
        output_particle_count,
    )
    datasets = make_random_field_datasets(split_payloads, input_representation=input_representation)
    dataloaders = _make_dataloaders(datasets, experiment_cfg)
    field_metric_context = _load_field_metric_context(generated_data_cfg)
    standardization_stats = _standardization_stats(split_payloads["train"], input_representation, standardization)
    model_init_payload = _standardized_model_init_payload(split_payloads["train"], standardization_stats)
    base_model = build_random_field_model(model_cfg, loss_cfg, model_init_payload, input_representation)
    model = _StandardizedRandomFieldModel(base_model, **standardization_stats).to(device)
    checkpoint_selection_metric = _checkpoint_selection_metric(loss_name)

    best_epoch = 0
    best_val_metrics = evaluate_random_field_model(
        model,
        dataloaders["val"],
        loss_name,
        loss_cfg,
        device,
        field_metric_context,
        metrics_cfg=metrics_cfg,
    )
    best_val_score = _checkpoint_score(best_val_metrics, checkpoint_selection_metric)
    best_state = _cpu_state_dict(model)
    best_optimizer_state = None
    best_scheduler_state = None
    training_history_rows: list[dict[str, Any]] = []

    if model_name != "global_constant" and _has_trainable_parameters(model):
        optimizer = build_optimizer(model, _cfg_get(cfg, "optimizer", None))
        scheduler = build_scheduler(optimizer, _cfg_get(cfg, "scheduler", None))
        for epoch in range(1, int(_cfg_get(experiment_cfg, "epochs", 1)) + 1):
            train_loss = _train_one_epoch(
                model,
                dataloaders["train"],
                optimizer,
                loss_name,
                loss_cfg,
                device,
                field_metric_context,
            )
            val_metrics = evaluate_random_field_model(
                model,
                dataloaders["val"],
                loss_name,
                loss_cfg,
                device,
                field_metric_context,
                metrics_cfg=metrics_cfg,
            )
            val_score = _checkpoint_score(val_metrics, checkpoint_selection_metric)
            is_best = val_score <= best_val_score
            learning_rate = float(optimizer.param_groups[0]["lr"])
            training_history_rows.append(
                _training_history_row(epoch, learning_rate, train_loss, val_metrics, val_score, is_best)
            )
            if is_best:
                best_epoch = epoch
                best_val_score = val_score
                best_val_metrics = deepcopy(val_metrics)
                best_state = _cpu_state_dict(model)
            step_scheduler(scheduler, val_score)
            if is_best:
                best_optimizer_state = deepcopy(optimizer.state_dict())
                best_scheduler_state = deepcopy(scheduler.state_dict()) if scheduler is not None else None

    checkpoint_dir = output_path / "model_ckpt"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "optimizer_state_dict": best_optimizer_state,
            "scheduler_state_dict": best_scheduler_state,
            "epoch": best_epoch,
            "checkpoint_selection_metric": checkpoint_selection_metric,
            "best_val_checkpoint_score": best_val_score,
            "best_val_projected_coeff_nll": float(best_val_metrics["observable_metrics"]["projected_coeff_nll"]),
        },
        checkpoint_path,
    )
    _load_state_dict(model, best_state, device)

    metrics_by_split = {
        split_name: evaluate_random_field_model(
            model,
            dataloader,
            loss_name,
            loss_cfg,
            device,
            field_metric_context,
            metrics_cfg=metrics_cfg,
        )
        for split_name, dataloader in dataloaders.items()
    }
    empirical_gaussian_oracle_floor = None
    if _is_non_gaussian_sample_based(generated_data_cfg):
        empirical_gaussian_oracle_floor = _evaluate_empirical_gaussian_oracle_floor_by_split(
            dataloaders,
            loss_cfg,
            device,
            field_metric_context,
            metrics_cfg=metrics_cfg,
        )

    results = {
        "phase": _phase_name(input_representation),
        "model_name": model_name,
        "loss_name": loss_name,
        "loss_role": metadata["loss_role"],
        "input_representation": input_representation,
        "output_protocol": output_protocol,
        "particle_view": {
            "context_particles": context_particle_count,
            "output_particles": output_particle_count,
        },
        "main_result_eligible": bool(metadata["main_result_eligible"]),
        "checkpoint_selection_metric": checkpoint_selection_metric,
        "best_epoch": best_epoch,
        "best_checkpoint_path": str(checkpoint_path),
        "metrics_by_split": metrics_by_split,
        "target": _json_safe(_cfg_get(generated_data_cfg, "target", {})),
    }
    if standardization["enabled"]:
        results["standardization"] = standardization
    if empirical_gaussian_oracle_floor is not None:
        results["empirical_gaussian_oracle_floor"] = empirical_gaussian_oracle_floor

    best_metrics = {
        "checkpoint_selection_metric": checkpoint_selection_metric,
        "best_epoch": best_epoch,
        "val": best_val_metrics,
    }
    rows = flatten_eval_rows(
        results,
        model_name=model_name,
        loss_name=loss_name,
        loss_role=metadata["loss_role"],
        main_result_eligible=bool(metadata["main_result_eligible"]),
    )

    _write_json(output_path / RESULT_ARTIFACT, results)
    if empirical_gaussian_oracle_floor is not None:
        _write_json(
            output_path / EMPIRICAL_GAUSSIAN_ORACLE_FLOOR_JSON,
            {"metrics_by_split": empirical_gaussian_oracle_floor},
        )
    _write_json(output_path / BEST_METRICS_ARTIFACT, best_metrics)
    _write_csv(output_path / RESULT_TABLE_ARTIFACT, rows)
    _write_csv(output_path / EVAL_BY_SPLIT_ARTIFACT, rows)
    _write_training_history_csv(output_path / TRAINING_HISTORY_CSV, training_history_rows)
    _copy_random_field_metadata_artifacts(generated_data_cfg, output_path)
    return results
