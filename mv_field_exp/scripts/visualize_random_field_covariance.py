"""Render covariance diagnostics for random-field law-to-law runs."""

# ruff: noqa: E402,PLR0912,PLR0913

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import RandomFieldSplitDataset, load_random_field_dataset_splits
from src.training.pipeline import (
    _StandardizedRandomFieldModel,
    _standardization_metadata,
    _standardization_stats,
    _standardized_model_init_payload,
    build_random_field_model,
    move_batch_to_device,
)
from src.models.gaussian_ops import scale_tril_to_covariance
from scripts.visualize_phase3_5_samples import (
    _align_projected_input_to_run_basis,
    _load_checkpoint_path,
    _load_generated_basis_metadata,
    _load_json,
    _load_run_basis_metadata,
    _output_basis_context_from_metadata,
    _stack_samples,
)
from src.utils.artifacts import RESULT_ARTIFACT
from src.utils.config import load_config
from src.utils.random_field_covariance_visualization import (
    SelectedLaw,
    anchor_token,
    covariance_to_correlation,
    field_anchor_index,
    parse_pair_token,
    plot_coefficient_covariance_heatmaps,
    plot_coefficient_covariance_summary,
    plot_field_covariance_slice,
    plot_pair_scatter_with_ellipses,
    select_law_buckets,
    select_top_correlation_pairs,
)

FIGURE_KINDS = (
    "all",
    "coefficient-summary",
    "coefficient-cov-heatmap",
    "field-cov-slice",
    "coefficient-scatter",
    "field-scatter",
)
EXPANDED_FIGURE_KINDS = frozenset(FIGURE_KINDS[1:])
SELECTION_SCORES = ("coeff-cov", "field-cov", "combined")
OUTPUT_FORMATS = ("pdf", "png")
DEFAULT_FIELD_SLICE_ANCHORS = (0.25, 0.5, 0.75)
DEFAULT_FIELD_PAIRS = ("0.25,0.50", "0.25,0.75", "0.50,0.75")
COEFFICIENT_INDEX_DIMS = 2


@dataclass(frozen=True)
class RunVisualizationContext:
    """Loaded run state needed by covariance visualization dispatch."""

    run_dir: Path
    cfg: Any
    results: dict[str, Any]
    split_payload: dict[str, Any]
    dataset: RandomFieldSplitDataset
    model: torch.nn.Module
    output_grid: torch.Tensor
    output_center: torch.Tensor
    output_basis_values: torch.Tensor
    output_weights: torch.Tensor
    target_metadata: dict[str, Any]
    reference_label: str
    run_label: str


@dataclass(frozen=True)
class PredictionBatch:
    """One predicted batch plus the original CPU batch tensors."""

    batch: dict[str, torch.Tensor]
    pred_mean: torch.Tensor
    pred_cov: torch.Tensor
    pred_scale_tril: torch.Tensor


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Run directory containing config.yaml and artifacts/.")
    parser.add_argument("--output-dir", required=True, help="Directory where covariance figures are written.")
    parser.add_argument("--split", default="test", help="Dataset split to visualize.")
    parser.add_argument("--device", default="cpu", help="Torch device for model prediction.")
    parser.add_argument("--output-format", choices=OUTPUT_FORMATS, default="pdf", help="Figure output format.")
    parser.add_argument(
        "--figure-kind",
        action="append",
        choices=FIGURE_KINDS,
        default=None,
        help="Figure kind to render. May be repeated.",
    )
    parser.add_argument("--law-selection", choices=["stratified", "explicit"], default="stratified")
    parser.add_argument("--best-k", type=int, default=2)
    parser.add_argument("--middle-k", type=int, default=2)
    parser.add_argument("--worst-k", type=int, default=2)
    parser.add_argument("--selection-score", choices=SELECTION_SCORES, default="coeff-cov")
    parser.add_argument("--law-ids", nargs="*", type=int, default=None)
    parser.add_argument("--max-coefficients", type=int, default=32)
    parser.add_argument("--num-coeff-pairs", type=int, default=3)
    parser.add_argument("--coeff-pairs", nargs="*", default=None, help="Coefficient index pairs like 0,1 0,2.")
    parser.add_argument("--corr-residual-limit", type=float, default=None)
    parser.add_argument("--field-slice-anchors", nargs="*", type=float, default=list(DEFAULT_FIELD_SLICE_ANCHORS))
    parser.add_argument("--field-pairs", nargs="*", default=list(DEFAULT_FIELD_PAIRS))
    parser.add_argument("--ellipse-sigma", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.figure_kind is None:
        args.figure_kind = ["all"]
    return args


def _reference_label(target_metadata: Any) -> str:
    if isinstance(target_metadata, Mapping):
        role = str(target_metadata.get("target_distribution_role", ""))
    else:
        role = str(getattr(target_metadata, "target_distribution_role", ""))
    return "empirical reference" if role == "non_gaussian_sample_based" else "target"


def _plain_target_metadata(target_metadata: Any) -> dict[str, Any]:
    if isinstance(target_metadata, Mapping):
        return dict(target_metadata)
    return {}


def _normalize_output_format(output_format: str) -> str:
    normalized = str(output_format).lower().lstrip(".")
    if normalized not in OUTPUT_FORMATS:
        raise ValueError("output_format must be one of: pdf, png")
    return normalized


def _lift_covariance(basis_values: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
    return torch.einsum("gq,bqr,hr->bgh", basis_values, covariance, basis_values)


def _lift_mean(center: torch.Tensor, basis_values: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    return center.unsqueeze(0) + torch.einsum("gq,bq->bg", basis_values, mean)


def _relative_frobenius_scores(pred: torch.Tensor, ref: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    numerator = torch.linalg.matrix_norm(pred - ref, ord="fro", dim=(-2, -1))
    denominator = torch.linalg.matrix_norm(ref, ord="fro", dim=(-2, -1)).clamp_min(float(eps))
    return numerator / denominator


def _weighted_field_covariance_scores(
    pred: torch.Tensor,
    ref: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    cov_weights = weights.view(1, -1, 1) * weights.view(1, 1, -1)
    numerator = ((pred - ref).square() * cov_weights).sum(dim=(-2, -1)).sqrt()
    denominator = (ref.square() * cov_weights).sum(dim=(-2, -1)).sqrt().clamp_min(float(eps))
    return numerator / denominator


def _build_model_for_context(
    cfg: Any,
    train_payload: dict[str, torch.Tensor],
    input_representation: str,
) -> torch.nn.Module:
    standardization = _standardization_metadata(_cfg_get(cfg, "standardization", None))
    standardization_stats = _standardization_stats(train_payload, input_representation, standardization)
    model_init_payload = _standardized_model_init_payload(train_payload, standardization_stats)
    base_model = build_random_field_model(cfg.model, cfg.loss, model_init_payload, input_representation)
    return _StandardizedRandomFieldModel(base_model, **standardization_stats)


def _load_model_state_dict_compatible(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    device: torch.device,
) -> None:
    device_state = {key: value.to(device) for key, value in state_dict.items()}
    try:
        model.load_state_dict(device_state)
        return
    except RuntimeError as wrapper_error:
        base_model = getattr(model, "base_model", None)
        if base_model is None or any(key.startswith("base_model.") for key in device_state):
            raise wrapper_error
        try:
            base_model.load_state_dict(device_state)
        except RuntimeError:
            raise wrapper_error from None


def _load_context(
    run_dir: str | Path,
    split: str,
    device: torch.device,
    repo_root: Path | None = None,
) -> RunVisualizationContext:
    run_path = Path(run_dir)
    root = repo_root if repo_root is not None else Path.cwd()
    cfg_path = run_path / "config.yaml"
    results_path = run_path / "artifacts" / RESULT_ARTIFACT
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing run config: {cfg_path}")
    if not results_path.exists():
        raise FileNotFoundError(f"missing results artifact: {results_path}")

    cfg = load_config(cfg_path)
    results = _load_json(results_path)
    input_representation = cfg.random_field_view.input_representation
    generated_basis_metadata = _load_generated_basis_metadata(cfg.generated_data, root)
    run_basis_metadata = _load_run_basis_metadata(run_path, cfg.generated_data, root)
    split_payloads = load_random_field_dataset_splits(cfg.generated_data, map_location="cpu")
    split_payloads = _align_projected_input_to_run_basis(
        split_payloads,
        input_representation=input_representation,
        generated_basis_metadata=generated_basis_metadata,
        run_basis_metadata=run_basis_metadata,
    )
    if split not in split_payloads:
        raise ValueError(f"unknown split {split!r}; available splits: {sorted(split_payloads)}")

    dataset = RandomFieldSplitDataset(split_payloads[split], input_representation=input_representation)
    model = _build_model_for_context(cfg, split_payloads["train"], input_representation)
    checkpoint = torch.load(_load_checkpoint_path(results, run_path), map_location=device)
    _load_model_state_dict_compatible(model, checkpoint["model_state_dict"], device)
    model.to(device)
    model.eval()

    output_basis_context = _output_basis_context_from_metadata(run_basis_metadata)
    output_basis = run_basis_metadata["output_basis"]
    target_metadata = _plain_target_metadata(_cfg_get(cfg.generated_data, "target", results.get("target", {})))
    run_label = str(_cfg_get(cfg.experiment, "name", run_path.name))
    return RunVisualizationContext(
        run_dir=run_path,
        cfg=cfg,
        results=results,
        split_payload=split_payloads[split],
        dataset=dataset,
        model=model,
        output_grid=output_basis_context["grid"],
        output_center=output_basis_context["center"],
        output_basis_values=output_basis_context["basis_values"],
        output_weights=torch.tensor(output_basis["quadrature_weights"], dtype=torch.float32),
        target_metadata=target_metadata,
        reference_label=_reference_label(target_metadata),
        run_label=run_label,
    )


@torch.no_grad()
def _predict_indices(
    context: RunVisualizationContext,
    indices: Sequence[int],
    device: torch.device,
) -> PredictionBatch:
    if not indices:
        raise ValueError("indices must be non-empty")
    batch = _stack_samples(context.dataset, indices)
    device_batch = move_batch_to_device(batch, device)
    was_training = context.model.training
    context.model.eval()
    prediction = context.model(device_batch)
    if was_training:
        context.model.train()

    pred_mean = prediction["pred_mean"].detach().cpu()
    pred_scale_tril = prediction["pred_scale_tril"].detach().cpu()
    pred_cov = prediction.get("pred_cov")
    if pred_cov is None:
        pred_cov = scale_tril_to_covariance(pred_scale_tril)
    else:
        pred_cov = pred_cov.detach().cpu()
    return PredictionBatch(
        batch=batch,
        pred_mean=pred_mean,
        pred_cov=pred_cov,
        pred_scale_tril=pred_scale_tril,
    )


def _select_explicit_laws(split_payload: dict[str, Any], law_ids: Sequence[int]) -> list[SelectedLaw]:
    available = {int(law_id.item()): index for index, law_id in enumerate(split_payload["law_ids"])}
    selected: list[SelectedLaw] = []
    for rank, law_id in enumerate(law_ids, start=1):
        normalized_law_id = int(law_id)
        if normalized_law_id not in available:
            raise ValueError(f"law_id {normalized_law_id} is not present in selected split")
        selected.append(
            SelectedLaw(
                bucket=f"explicit{rank:02d}",
                index=int(available[normalized_law_id]),
                law_id=normalized_law_id,
                score=0.0,
            )
        )
    return selected


def _select_stratified_laws(
    context: RunVisualizationContext,
    device: torch.device,
    best_k: int,
    middle_k: int,
    worst_k: int,
    selection_score: str,
) -> list[SelectedLaw]:
    all_indices = list(range(int(context.split_payload["law_ids"].numel())))
    prediction = _predict_indices(context, all_indices, device)
    coeff_scores = _relative_frobenius_scores(prediction.pred_cov, prediction.batch["target_projected_output_cov"])
    if selection_score == "coeff-cov":
        scores = coeff_scores
    else:
        pred_field_cov = _lift_covariance(context.output_basis_values, prediction.pred_cov)
        ref_field_cov = prediction.batch["target_output_grid_cov"]
        field_scores = _weighted_field_covariance_scores(pred_field_cov, ref_field_cov, context.output_weights)
        if selection_score == "field-cov":
            scores = field_scores
        elif selection_score == "combined":
            coeff_std = (coeff_scores - coeff_scores.mean()) / coeff_scores.std(unbiased=False).clamp_min(1.0e-8)
            field_std = (field_scores - field_scores.mean()) / field_scores.std(unbiased=False).clamp_min(1.0e-8)
            scores = 0.5 * (coeff_std + field_std)
        else:
            raise ValueError(f"unsupported selection_score: {selection_score}")
    return select_law_buckets(
        scores,
        context.split_payload["law_ids"],
        best_k=best_k,
        middle_k=middle_k,
        worst_k=worst_k,
    )


def _expanded_figure_kinds(values: Sequence[str]) -> set[str]:
    kinds = set(values)
    if "all" in kinds:
        return set(EXPANDED_FIGURE_KINDS)
    unknown = kinds - EXPANDED_FIGURE_KINDS
    if unknown:
        raise ValueError(f"unsupported figure kind(s): {sorted(unknown)}")
    return kinds


def _coefficient_covariance_heatmap_title(selected: SelectedLaw) -> str:
    """Return the intentionally compact in-figure title for raw covariance heatmaps."""
    return f"law_id={selected.law_id}"


def _coefficient_pairs(
    reference_cov: torch.Tensor,
    max_coefficients: int,
    num_pairs: int,
    manual_pairs: Sequence[str] | None,
) -> list[tuple[int, int]]:
    if manual_pairs:
        return [parse_pair_token(value, value_type=int) for value in manual_pairs]  # type: ignore[list-item]
    display_dim = min(int(max_coefficients), int(reference_cov.shape[0]))
    correlation = covariance_to_correlation(reference_cov[:display_dim, :display_dim])
    return select_top_correlation_pairs(correlation, num_pairs=num_pairs)


def _field_pairs(values: Sequence[str]) -> list[tuple[float, float]]:
    return [parse_pair_token(value, value_type=float) for value in values]  # type: ignore[list-item]


def _index_tensor(pair: Sequence[int], device: torch.device | None = None) -> torch.Tensor:
    if len(pair) != COEFFICIENT_INDEX_DIMS:
        raise ValueError("pair must contain exactly two indices")
    return torch.tensor([int(pair[0]), int(pair[1])], dtype=torch.long, device=device)


def _covariance_block(covariance: torch.Tensor, pair: Sequence[int]) -> torch.Tensor:
    index = _index_tensor(pair, device=covariance.device)
    return covariance.index_select(0, index).index_select(1, index)


def _vector_pair(values: torch.Tensor, pair: Sequence[int]) -> torch.Tensor:
    return values.index_select(0, _index_tensor(pair, device=values.device))


def _particle_pair(particles: torch.Tensor, pair: Sequence[int]) -> torch.Tensor:
    return particles.index_select(1, _index_tensor(pair, device=particles.device))


def render_covariance_visualizations(
    run_dir: str | Path,
    output_dir: str | Path,
    split: str = "test",
    device: str = "cpu",
    output_format: str = "pdf",
    figure_kind: Sequence[str] = ("all",),
    law_selection: str = "stratified",
    best_k: int = 2,
    middle_k: int = 2,
    worst_k: int = 2,
    selection_score: str = "coeff-cov",
    law_ids: Sequence[int] | None = None,
    max_coefficients: int = 32,
    num_coeff_pairs: int = 3,
    coeff_pairs: Sequence[str] | None = None,
    corr_residual_limit: float | None = None,
    field_slice_anchors: Sequence[float] = DEFAULT_FIELD_SLICE_ANCHORS,
    field_pairs: Sequence[str] = DEFAULT_FIELD_PAIRS,
    ellipse_sigma: float = 1.0,
) -> list[Path]:
    """Render requested covariance diagnostics for one saved random-field run."""
    normalized_output_format = _normalize_output_format(output_format)
    target_device = torch.device(device)
    context = _load_context(run_dir, split=split, device=target_device)
    if law_selection == "explicit":
        if not law_ids:
            raise ValueError("explicit law selection requires --law-ids")
        selected_laws = _select_explicit_laws(context.split_payload, law_ids)
    elif law_selection == "stratified":
        selected_laws = _select_stratified_laws(context, target_device, best_k, middle_k, worst_k, selection_score)
    else:
        raise ValueError("law_selection must be one of: stratified, explicit")

    selected_indices = [law.index for law in selected_laws]
    prediction = _predict_indices(context, selected_indices, target_device)
    kinds = _expanded_figure_kinds(figure_kind)
    output_root = Path(output_dir)
    output_paths: list[Path] = []
    pred_field_cov = _lift_covariance(context.output_basis_values, prediction.pred_cov)
    pred_field_mean = _lift_mean(context.output_center, context.output_basis_values, prediction.pred_mean)

    for row, selected in enumerate(selected_laws):
        prefix = f"{context.run_label}__{split}__{selected.bucket}__law_{selected.law_id:03d}"
        ref_mean = prediction.batch["target_projected_output_mean"][row]
        ref_cov = prediction.batch["target_projected_output_cov"][row]
        pred_mean = prediction.pred_mean[row]
        pred_cov = prediction.pred_cov[row]

        if "coefficient-summary" in kinds:
            output_paths.append(
                plot_coefficient_covariance_summary(
                    output_path=output_root
                    / "coefficient_summary"
                    / f"{prefix}__coeff_covariance_summary.{normalized_output_format}",
                    title=f"{context.run_label} {split} {selected.bucket} law {selected.law_id}",
                    reference_label=context.reference_label,
                    reference_mean=ref_mean,
                    reference_cov=ref_cov,
                    predicted_mean=pred_mean,
                    predicted_cov=pred_cov,
                    max_coefficients=max_coefficients,
                    residual_limit=corr_residual_limit,
                )
            )

        if "coefficient-cov-heatmap" in kinds:
            output_paths.append(
                plot_coefficient_covariance_heatmaps(
                    output_path=output_root
                    / "coefficient_cov_heatmaps"
                    / f"{prefix}__coeff_covariance_heatmaps.{normalized_output_format}",
                    title=_coefficient_covariance_heatmap_title(selected),
                    reference_label=context.reference_label,
                    reference_cov=ref_cov,
                    predicted_cov=pred_cov,
                    max_coefficients=max_coefficients,
                )
            )

        if "field-cov-slice" in kinds:
            for anchor in field_slice_anchors:
                anchor_index = field_anchor_index(context.output_grid, anchor)
                output_paths.append(
                    plot_field_covariance_slice(
                        output_path=output_root
                        / "field_cov_slices"
                        / f"{prefix}__field_cov_slice_{anchor_token(anchor)}.{normalized_output_format}",
                        title=f"Field covariance slice x0={anchor:.2f}; law {selected.law_id}",
                        grid=context.output_grid,
                        reference_values=prediction.batch["target_output_grid_cov"][row, anchor_index],
                        predicted_values=pred_field_cov[row, anchor_index],
                        reference_label=context.reference_label,
                        anchor=float(context.output_grid[anchor_index].item()),
                    )
                )

        if "coefficient-scatter" in kinds:
            for first, second in _coefficient_pairs(ref_cov, max_coefficients, num_coeff_pairs, coeff_pairs):
                pair = (int(first), int(second))
                output_paths.append(
                    plot_pair_scatter_with_ellipses(
                        output_path=output_root
                        / "coefficient_scatter"
                        / f"{prefix}__coeff_pair_z{first + 1:02d}_z{second + 1:02d}.{normalized_output_format}",
                        title=f"Coefficient pair z{first + 1}, z{second + 1}; law {selected.law_id}",
                        x_label=f"z{first + 1}",
                        y_label=f"z{second + 1}",
                        particles=_particle_pair(prediction.batch["output_particles"][row], pair),
                        reference_mean=_vector_pair(ref_mean, pair),
                        reference_cov=_covariance_block(ref_cov, pair),
                        predicted_mean=_vector_pair(pred_mean, pair),
                        predicted_cov=_covariance_block(pred_cov, pair),
                        reference_label=context.reference_label,
                        ellipse_sigma=ellipse_sigma,
                    )
                )

        if "field-scatter" in kinds:
            for anchor_a, anchor_b in _field_pairs(field_pairs):
                idx_a = field_anchor_index(context.output_grid, anchor_a)
                idx_b = field_anchor_index(context.output_grid, anchor_b)
                pair = (idx_a, idx_b)
                output_paths.append(
                    plot_pair_scatter_with_ellipses(
                        output_path=output_root
                        / "field_scatter"
                        / f"{prefix}__field_pair_{anchor_token(anchor_a)}_{anchor_token(anchor_b)}.{normalized_output_format}",
                        title=f"Field pair x={anchor_a:.2f}, x={anchor_b:.2f}; law {selected.law_id}",
                        x_label=f"u({anchor_a:.2f})",
                        y_label=f"u({anchor_b:.2f})",
                        particles=_particle_pair(prediction.batch["output_field_particles"][row], pair),
                        reference_mean=_vector_pair(prediction.batch["target_output_grid_mean"][row], pair),
                        reference_cov=_covariance_block(prediction.batch["target_output_grid_cov"][row], pair),
                        predicted_mean=_vector_pair(pred_field_mean[row], pair),
                        predicted_cov=_covariance_block(pred_field_cov[row], pair),
                        reference_label=context.reference_label,
                        ellipse_sigma=ellipse_sigma,
                    )
                )
    return output_paths


def main(argv: Sequence[str] | None = None) -> None:
    """Run the covariance visualization CLI."""
    args = _parse_args(argv)
    paths = render_covariance_visualizations(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        split=args.split,
        device=args.device,
        output_format=args.output_format,
        figure_kind=args.figure_kind,
        law_selection=args.law_selection,
        best_k=args.best_k,
        middle_k=args.middle_k,
        worst_k=args.worst_k,
        selection_score=args.selection_score,
        law_ids=args.law_ids,
        max_coefficients=args.max_coefficients,
        num_coeff_pairs=args.num_coeff_pairs,
        coeff_pairs=args.coeff_pairs,
        corr_residual_limit=args.corr_residual_limit,
        field_slice_anchors=args.field_slice_anchors,
        field_pairs=args.field_pairs,
        ellipse_sigma=args.ellipse_sigma,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
