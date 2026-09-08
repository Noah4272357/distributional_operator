"""Render Phase 3-5 law-level true/predicted random-field samples."""

# ruff: noqa: E402,PLR0913

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.io import load_yaml
from src.data.dataset import RandomFieldSplitDataset, load_random_field_dataset_splits
from src.training.pipeline import (
    _StandardizedRandomFieldModel,
    _standardization_metadata,
    _standardization_stats,
    _standardized_model_init_payload,
    build_random_field_model,
    move_batch_to_device,
)
from src.utils.artifacts import RESULT_ARTIFACT
from src.utils.config import load_config
from src.utils.field_metrics import sample_predicted_field_particles
from src.utils.random_field_visualization import FieldSamplePanel, plot_field_sample_panels


BASIS_ALIGNMENT_MIN_ABS_DOT = 0.9


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


def _resolve_repo_path(path_value: str | Path, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_generated_basis_metadata(generated_data_cfg: Any, repo_root: Path) -> dict[str, Any]:
    basis_metadata_path = _resolve_repo_path(_required_cfg_get(generated_data_cfg.files, "basis_metadata"), repo_root)
    return load_yaml(basis_metadata_path)


def _load_run_basis_metadata(run_dir: Path, generated_data_cfg: Any, repo_root: Path) -> dict[str, Any]:
    artifact_metadata_path = run_dir / "artifacts" / "basis_metadata.yaml"
    if artifact_metadata_path.exists():
        return load_yaml(artifact_metadata_path)
    return _load_generated_basis_metadata(generated_data_cfg, repo_root)


def _basis_column_signs(source_metadata: dict[str, Any], target_metadata: dict[str, Any], basis_key: str) -> torch.Tensor:
    source_basis = source_metadata[basis_key]
    target_basis = target_metadata[basis_key]
    source_values = torch.tensor(source_basis["basis_values"], dtype=torch.float32)
    target_values = torch.tensor(target_basis["basis_values"], dtype=torch.float32)
    weights = torch.tensor(source_basis["quadrature_weights"], dtype=torch.float32)
    if source_values.shape != target_values.shape:
        raise ValueError(f"{basis_key} shape mismatch: {tuple(source_values.shape)} vs {tuple(target_values.shape)}")
    if weights.numel() != source_values.shape[0]:
        raise ValueError(f"{basis_key} quadrature weights do not match basis grid")

    overlaps = (source_values * target_values * weights.unsqueeze(-1)).sum(dim=0)
    if bool(torch.any(overlaps.abs() < BASIS_ALIGNMENT_MIN_ABS_DOT)):
        raise ValueError(
            f"{basis_key} from generated data is not column-wise sign-compatible with run artifact basis; "
            f"weighted diagonal overlaps={overlaps.tolist()}"
        )
    return torch.where(overlaps < 0.0, -torch.ones_like(overlaps), torch.ones_like(overlaps))


def _align_projected_input_to_run_basis(
    split_payloads: dict[str, dict[str, Any]],
    input_representation: str,
    generated_basis_metadata: dict[str, Any],
    run_basis_metadata: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if input_representation != "learned_basis_projected":
        return split_payloads

    signs = _basis_column_signs(generated_basis_metadata, run_basis_metadata, "input_basis")
    if bool(torch.all(signs > 0.0)):
        return split_payloads

    aligned_payloads: dict[str, dict[str, Any]] = {}
    for split_name, payload in split_payloads.items():
        aligned_payload = dict(payload)
        projected = payload["input_projected_particles"]
        aligned_payload["input_projected_particles"] = projected * signs.to(
            device=projected.device,
            dtype=projected.dtype,
        ).view(1, 1, -1)
        aligned_payloads[split_name] = aligned_payload
    return aligned_payloads


def _output_basis_context_from_metadata(metadata: dict[str, Any]) -> dict[str, torch.Tensor]:
    output_basis = metadata["output_basis"]
    return {
        "grid": torch.tensor(metadata["output_grid"], dtype=torch.float32),
        "center": torch.tensor(output_basis["center"], dtype=torch.float32),
        "basis_values": torch.tensor(output_basis["basis_values"], dtype=torch.float32),
    }


def _load_checkpoint_path(results: dict[str, Any], run_dir: Path) -> Path:
    raw_path = results.get("best_checkpoint_path")
    if raw_path is None:
        raise ValueError("results.json does not include best_checkpoint_path")
    path = Path(raw_path)
    if path.exists():
        return path
    candidate = run_dir / "artifacts" / "model_ckpt" / "best.pt"
    if candidate.exists():
        return candidate
    return path


def _select_law_indices(split_payload: dict[str, Any], laws_per_split: int, seed: int) -> list[int]:
    law_count = int(split_payload["law_ids"].numel())
    requested = min(int(laws_per_split), law_count)
    generator = torch.Generator().manual_seed(int(seed))
    return torch.randperm(law_count, generator=generator)[:requested].tolist()


def _build_visualization_model(
    cfg: Any,
    train_payload: dict[str, torch.Tensor],
    input_representation: str,
) -> torch.nn.Module:
    standardization = _standardization_metadata(_cfg_get(cfg, "standardization", None))
    standardization_stats = _standardization_stats(train_payload, input_representation, standardization)
    model_init_payload = _standardized_model_init_payload(train_payload, standardization_stats)
    base_model = build_random_field_model(cfg.model, cfg.loss, model_init_payload, input_representation)
    return _StandardizedRandomFieldModel(base_model, **standardization_stats)


def _stack_samples(dataset: RandomFieldSplitDataset, indices: Sequence[int]) -> dict[str, torch.Tensor]:
    samples = [dataset[int(index)] for index in indices]
    keys = samples[0].keys()
    return {key: torch.stack([sample[key] for sample in samples]) for key in keys}


@torch.no_grad()
def _predict_fields_for_indices(
    model: torch.nn.Module,
    dataset: RandomFieldSplitDataset,
    indices: Sequence[int],
    device: torch.device,
    output_basis_context: dict[str, torch.Tensor],
    predicted_samples: int,
    seed: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    batch = _stack_samples(dataset, indices)
    device_batch = move_batch_to_device(batch, device)
    model.eval()
    prediction = model(device_batch)
    predicted_fields = sample_predicted_field_particles(
        prediction["pred_mean"],
        prediction["pred_scale_tril"],
        output_basis_context["center"].to(device=device),
        output_basis_context["basis_values"].to(device=device),
        num_samples=int(predicted_samples),
        seed=int(seed),
    ).cpu()
    return batch, predicted_fields


def render_run_sample_visualizations(
    run_dir: str | Path,
    output_dir: str | Path,
    splits: Sequence[str] = ("train", "test"),
    laws_per_split: int = 4,
    predicted_samples: int = 16,
    seed: int = 0,
    device: str = "cpu",
    repo_root: str | Path | None = None,
    output_format: str = "pdf",
) -> list[Path]:
    """Render train/test true-vs-predicted sample panels for one run directory."""
    normalized_output_format = str(output_format).lower().lstrip(".")
    if normalized_output_format not in {"pdf", "png"}:
        raise ValueError("output_format must be one of: pdf, png")
    run_path = Path(run_dir)
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    cfg_path = run_path / "config.yaml"
    results_path = run_path / "artifacts" / RESULT_ARTIFACT
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing run config: {cfg_path}")
    if not results_path.exists():
        raise FileNotFoundError(f"missing results artifact: {results_path}")

    cfg = load_config(cfg_path)
    results = _load_json(results_path)
    target_device = torch.device(device)
    generated_basis_metadata = _load_generated_basis_metadata(cfg.generated_data, root)
    run_basis_metadata = _load_run_basis_metadata(run_path, cfg.generated_data, root)
    split_payloads = load_random_field_dataset_splits(cfg.generated_data, map_location="cpu")
    split_payloads = _align_projected_input_to_run_basis(
        split_payloads,
        input_representation=cfg.random_field_view.input_representation,
        generated_basis_metadata=generated_basis_metadata,
        run_basis_metadata=run_basis_metadata,
    )
    datasets = {
        split_name: RandomFieldSplitDataset(split_payload, input_representation=cfg.random_field_view.input_representation)
        for split_name, split_payload in split_payloads.items()
    }
    model = _build_visualization_model(cfg, split_payloads["train"], cfg.random_field_view.input_representation)
    checkpoint = torch.load(_load_checkpoint_path(results, run_path), map_location=target_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(target_device)
    output_basis_context = _output_basis_context_from_metadata(run_basis_metadata)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    run_label = str(_cfg_get(cfg.experiment, "name", run_path.name))
    for split_offset, split_name in enumerate(splits):
        split_payload = split_payloads[split_name]
        indices = _select_law_indices(split_payload, laws_per_split=laws_per_split, seed=int(seed) + split_offset)
        batch, predicted_fields = _predict_fields_for_indices(
            model=model,
            dataset=datasets[split_name],
            indices=indices,
            device=target_device,
            output_basis_context=output_basis_context,
            predicted_samples=predicted_samples,
            seed=int(seed) + 1000 + split_offset,
        )
        panels = []
        for panel_index, law_index in enumerate(indices):
            law_id = int(split_payload["law_ids"][law_index].item())
            panels.append(
                FieldSamplePanel(
                    title=f"law_id={law_id}",
                    grid=output_basis_context["grid"],
                    true_fields=batch["output_field_particles"][panel_index],
                    predicted_fields=predicted_fields[panel_index],
                )
            )
        output_path = output / f"{run_label}__{split_name}.{normalized_output_format}"
        plot_field_sample_panels(
            panels,
            output_path=output_path,
            figure_title=None,
            max_true_samples=8,
            max_predicted_samples=min(8, int(predicted_samples)),
        )
        output_paths.append(output_path)
    return output_paths


def find_run_dirs(runs_root: str | Path) -> list[Path]:
    """Find run directories with random-field result artifacts under a root."""
    root = Path(runs_root)
    result_paths = sorted(root.glob("*/**/artifacts/results.json"))
    return [path.parents[1] for path in result_paths]


def render_all_runs(
    runs_root: str | Path,
    output_dir: str | Path,
    splits: Sequence[str],
    laws_per_split: int,
    predicted_samples: int,
    seed: int,
    device: str,
    repo_root: str | Path | None = None,
    output_format: str = "pdf",
) -> list[Path]:
    """Render visualizations for every run under a Phase 3-5 result root."""
    output_paths: list[Path] = []
    for run_dir in find_run_dirs(runs_root):
        output_paths.extend(
            render_run_sample_visualizations(
                run_dir=run_dir,
                output_dir=output_dir,
                splits=splits,
                laws_per_split=laws_per_split,
                predicted_samples=predicted_samples,
                seed=seed,
                device=device,
                repo_root=repo_root,
                output_format=output_format,
            )
        )
    return output_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True, help="Root containing random-field run directories.")
    parser.add_argument("--output-dir", required=True, help="Directory for PDF visualizations.")
    parser.add_argument("--splits", nargs="+", default=["train", "test"], help="Dataset splits to visualize.")
    parser.add_argument("--laws-per-split", type=int, default=4, help="Number of laws to sample per split.")
    parser.add_argument("--predicted-samples", type=int, default=16, help="Predicted field samples per law.")
    parser.add_argument("--seed", type=int, default=0, help="Law/prediction sampling seed.")
    parser.add_argument("--device", default="cpu", help="Torch device for model prediction.")
    parser.add_argument("--repo-root", default=None, help="Repository root for resolving relative data paths.")
    parser.add_argument("--output-format", choices=["pdf", "png"], default="pdf", help="Visualization file format.")
    return parser.parse_args()


def main() -> None:
    """Run the sample visualization CLI."""
    args = _parse_args()
    output_paths = render_all_runs(
        runs_root=args.runs_root,
        output_dir=args.output_dir,
        splits=args.splits,
        laws_per_split=args.laws_per_split,
        predicted_samples=args.predicted_samples,
        seed=args.seed,
        device=args.device,
        repo_root=args.repo_root,
        output_format=args.output_format,
    )
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
