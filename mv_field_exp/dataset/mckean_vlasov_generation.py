"""Generate the single McKean_Vlasov random-field dataset in HDF5 format."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from dataset.io import save_hdf5, save_yaml
from src.models.bases import (
    LearnedBasis,
    fit_weighted_pca_basis,
    project_onto_basis,
    sine_basis_values,
    trapezoid_weights,
    uniform_grid,
)
from src.models.mckean_vlasov import TARGET_NAME, apply_mckean_vlasov, sample_gaussian_laws, sample_particles
from src.utils.config import to_plain


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _split_indices(data_size: int, train_fraction: float, val_fraction: float) -> dict[str, torch.Tensor]:
    if data_size < 3:
        raise ValueError("data_size must be at least 3 to create train, val, and test splits")
    if train_fraction <= 0.0 or val_fraction <= 0.0 or train_fraction + val_fraction >= 1.0:
        raise ValueError("split fractions must be positive and sum to less than 1")
    train_size = max(1, int(data_size * train_fraction))
    val_size = max(1, int(data_size * val_fraction))
    if train_size + val_size >= data_size:
        val_size = 1
        train_size = data_size - 2
    return {
        "train": torch.arange(0, train_size, dtype=torch.long),
        "val": torch.arange(train_size, train_size + val_size, dtype=torch.long),
        "test": torch.arange(train_size + val_size, data_size, dtype=torch.long),
    }


def _project_particles(particles: torch.Tensor, basis: LearnedBasis) -> torch.Tensor:
    data_size, sample_size, _ = particles.shape
    projected = project_onto_basis(
        particles.reshape(data_size * sample_size, -1),
        basis.center,
        basis.values,
        basis.weights,
    )
    return projected.reshape(data_size, sample_size, -1)


def _project_grid_moments(
    mean: torch.Tensor,
    covariance: torch.Tensor,
    basis: LearnedBasis,
) -> tuple[torch.Tensor, torch.Tensor]:
    projection = basis.weights.unsqueeze(-1) * basis.values
    projected_mean = (mean - basis.center) @ projection
    projected_covariance = torch.einsum("gq,bgh,hr->bqr", projection, covariance, projection)
    return projected_mean, projected_covariance


def _basis_metadata(grid: torch.Tensor, basis: LearnedBasis) -> dict[str, Any]:
    return {
        "grid": grid.tolist(),
        "center": basis.center.tolist(),
        "basis_values": basis.values.tolist(),
        "quadrature_weights": basis.weights.tolist(),
        "rank": int(basis.values.shape[1]),
        "explained_variance": basis.explained_variance.tolist(),
        "explained_variance_ratio": basis.explained_variance_ratio.tolist(),
        "cumulative_explained_variance_ratio": basis.cumulative_explained_variance_ratio.tolist(),
        "centering_policy": basis.centering_policy,
        "normalization_policy": basis.normalization_policy,
    }


def generate_mckean_vlasov_dataset(
    cfg: Any,
    *,
    data_size: int,
    sample_size: int,
    grid_size: int,
) -> dict[str, Any]:
    """Generate all arrays, save one HDF5 file, and return artifact paths."""
    data_size = int(data_size)
    sample_size = int(sample_size)
    grid_size = int(grid_size)
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")

    coefficient_rank = int(cfg.coefficient_rank)
    if coefficient_rank > grid_size:
        raise ValueError("coefficient_rank cannot exceed grid_size")
    seed = int(cfg.seed)
    indices = _split_indices(data_size, float(cfg.split.train_fraction), float(cfg.split.val_fraction))
    grid = uniform_grid(grid_size, float(cfg.domain_min), float(cfg.domain_max))
    weights = trapezoid_weights(grid)
    generator_basis = sine_basis_values(grid, coefficient_rank)

    input_mean, input_covariance = sample_gaussian_laws(
        data_size,
        coefficient_rank,
        seed,
        float(cfg.input_law.mean_low),
        float(cfg.input_law.mean_high),
        float(cfg.input_law.sigma_low),
        float(cfg.input_law.sigma_high),
        float(cfg.input_law.jitter),
    )
    input_coefficients = sample_particles(input_mean, input_covariance, sample_size, seed + 1_000)
    input_field_particles = input_coefficients @ generator_basis.transpose(0, 1)

    output_mean, output_covariance = apply_mckean_vlasov(input_mean, input_covariance, cfg.mckean_vlasov)
    output_coefficients = sample_particles(output_mean, output_covariance, sample_size, seed + 2_000)
    output_field_particles = output_coefficients @ generator_basis.transpose(0, 1)
    output_grid_mean = output_mean @ generator_basis.transpose(0, 1)
    output_grid_covariance = torch.einsum(
        "gq,bqr,hr->bgh", generator_basis, output_covariance, generator_basis
    )

    train_indices = indices["train"]
    input_basis = fit_weighted_pca_basis(
        input_field_particles[train_indices].reshape(-1, grid_size),
        weights,
        rank=_cfg_get(cfg.basis, "rank"),
        explained_variance_threshold=float(cfg.basis.explained_variance_threshold),
        min_rank=int(cfg.basis.min_rank),
        max_rank=min(int(cfg.basis.max_rank), grid_size, int(train_indices.numel()) * sample_size),
    )
    output_basis = fit_weighted_pca_basis(
        output_field_particles[train_indices].reshape(-1, grid_size),
        weights,
        rank=_cfg_get(cfg.basis, "rank"),
        explained_variance_threshold=float(cfg.basis.explained_variance_threshold),
        min_rank=int(cfg.basis.min_rank),
        max_rank=min(int(cfg.basis.max_rank), grid_size, int(train_indices.numel()) * sample_size),
    )
    input_projected = _project_particles(input_field_particles, input_basis)
    output_projected = _project_particles(output_field_particles, output_basis)
    target_projected_mean, target_projected_covariance = _project_grid_moments(
        output_grid_mean, output_grid_covariance, output_basis
    )

    arrays = {
        "law_ids": torch.arange(data_size, dtype=torch.long),
        "input_field_particles": input_field_particles,
        "output_field_particles": output_field_particles,
        "input_projected_particles": input_projected,
        "output_projected_particles": output_projected,
        "input_law_mean": input_mean,
        "input_law_cov": input_covariance,
        "target_mean": output_mean,
        "target_cov": output_covariance,
        "target_scale_tril": torch.linalg.cholesky(output_covariance),
        "target_output_grid_mean": output_grid_mean,
        "target_output_grid_cov": output_grid_covariance,
        "target_projected_output_mean": target_projected_mean,
        "target_projected_output_cov": target_projected_covariance,
    }
    output_dir = Path(str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = output_dir / str(cfg.filename)
    attributes = {
        "schema_version": int(cfg.schema_version),
        "target_name": TARGET_NAME,
        "seed": seed,
        "data_size": data_size,
        "sample_size": sample_size,
        "grid_size": grid_size,
        "coefficient_rank": coefficient_rank,
    }
    save_hdf5(hdf5_path, arrays, indices, attributes)

    basis_metadata = {
        "input_grid": grid.tolist(),
        "output_grid": grid.tolist(),
        "input_basis": _basis_metadata(grid, input_basis),
        "output_basis": _basis_metadata(grid, output_basis),
    }
    manifest = {
        "name": TARGET_NAME,
        "schema_version": int(cfg.schema_version),
        "path": str(hdf5_path),
        "format": "hdf5",
        "data_size": data_size,
        "sample_size": sample_size,
        "grid_size": grid_size,
        "splits": {name: int(value.numel()) for name, value in indices.items()},
        "datasets": {name: list(value.shape) for name, value in arrays.items()},
        "target": {
            "name": TARGET_NAME,
            "target_distribution_role": "gaussian_exact",
            "main_gaussian_nll_interpretation": "matched",
            "parameters": to_plain(cfg.mckean_vlasov),
        },
    }
    projection_summary = {
        "input_basis": {"rank": int(input_basis.values.shape[1])},
        "output_basis": {"rank": int(output_basis.values.shape[1])},
    }
    generation_summary = {
        "input_field_particles_shape": list(input_field_particles.shape),
        "output_field_particles_shape": list(output_field_particles.shape),
        "finite_tensors": all(bool(torch.isfinite(value).all()) for value in arrays.values()),
    }
    save_yaml(output_dir / "manifest.yaml", manifest)
    save_yaml(output_dir / "basis_metadata.yaml", basis_metadata)
    save_yaml(output_dir / "projection_summary.yaml", projection_summary)
    save_yaml(output_dir / "generation_summary.yaml", generation_summary)
    return {
        "hdf5_path": hdf5_path,
        "manifest": manifest,
        "input_field_particles_shape": tuple(input_field_particles.shape),
    }
