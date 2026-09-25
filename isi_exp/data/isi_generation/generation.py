"""Compact ISI LIF dataset generation for distribution-learning experiments."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import h5py
import torch

from .features import build_moment_rff_features
from .simulation import bin_isi_observations, make_isi_bin_edges, simulate_lif_isi

REGIME_NAMES = ("subthreshold", "balanced_near", "suprathreshold")
REGIME_RANGES = ((0.75, 0.95), (0.95, 1.05), (1.05, 1.25))
Q_RANGE = (0.1, 0.35)


def sample_drive_parameters(data_size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample an approximately equal number of laws from each drive regime."""
    if data_size < len(REGIME_NAMES):
        raise ValueError(f"data_size must be at least {len(REGIME_NAMES)}")
    generator = torch.Generator().manual_seed(int(seed) + 101)
    base, remainder = divmod(int(data_size), len(REGIME_NAMES))
    counts = [base + (index < remainder) for index in range(len(REGIME_NAMES))]
    params = []
    labels = []
    for label, ((r_min, r_max), count) in enumerate(zip(REGIME_RANGES, counts)):
        r = r_min + (r_max - r_min) * torch.rand(count, generator=generator)
        log_q = math.log(Q_RANGE[0]) + math.log(Q_RANGE[1] / Q_RANGE[0]) * torch.rand(
            count, generator=generator
        )
        params.append(torch.stack((r, torch.exp(log_q)), dim=-1))
        labels.append(torch.full((count,), label, dtype=torch.long))
    return torch.cat(params).to(torch.float32), torch.cat(labels)


def sample_input_particles(params: torch.Tensor, sample_size: int, seed: int) -> torch.Tensor:
    """Sample scalar input-current proxies with shape ``[data_size, sample_size]``."""
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    generator = torch.Generator(device=params.device).manual_seed(int(seed) + 202)
    noise = torch.randn(
        (params.shape[0], int(sample_size)),
        dtype=params.dtype,
        device=params.device,
        generator=generator,
    )
    return params[:, :1] + torch.sqrt(params[:, 1:2]) * noise


def _validate_payload(
    payload: dict[str, torch.Tensor],
    data_size: int,
    sample_size: int,
    feature_dim: int,
) -> None:
    category_count = int(payload["bin_edges"].numel())
    expected = {
        "law_ids": (data_size,),
        "regime_labels": (data_size,),
        "params": (data_size, 2),
        "normalized_params": (data_size, 2),
        "input_particles": (data_size, sample_size),
        "input_features": (data_size, feature_dim),
        "bin_counts": (data_size, category_count),
        "empirical_bin_mass": (data_size, category_count),
    }
    for key, shape in expected.items():
        if tuple(payload[key].shape) != shape:
            raise RuntimeError(f"{key} has shape {tuple(payload[key].shape)}, expected {shape}")
    if not torch.allclose(payload["empirical_bin_mass"].sum(-1), torch.ones(data_size)):
        raise RuntimeError("empirical bin masses are not normalized")


def save_hdf5_dataset(path: str | Path, payload: dict[str, torch.Tensor]) -> Path:
    """Persist each payload tensor as a named HDF5 dataset."""
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(destination, "w") as handle:
        handle.attrs["format"] = "isi_lif_laws"
        handle.attrs["format_version"] = 1
        for key, tensor in payload.items():
            handle.create_dataset(key, data=tensor.numpy(), compression="gzip", shuffle=True)
    return destination


def generate_isi_lif_dataset(
    *,
    data_size: int,
    sample_size: int,
    output_path: str | Path,
    seed: int = 0,
    device: str = "cpu",
    n_isi: int = 512,
    dt: float = 0.002,
    t_max: float = 8.0,
    finite_bins: int = 48,
    feature_frequencies: int = 8,
    rff_scale: float = 1.0,
) -> dict[str, Any]:
    """Generate and save one unsplit dataset consumed by ``src.data.dataloader``."""
    data_size = int(data_size)
    sample_size = int(sample_size)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA dataset generation was requested but CUDA is unavailable")
    params, regime_labels = sample_drive_parameters(data_size, seed)
    params = params.to(target_device)
    regime_labels = regime_labels.to(target_device)
    normalized_params = torch.stack((params[:, 0], params[:, 1].log()), dim=-1)
    input_particles = sample_input_particles(params, sample_size, seed)
    input_features = build_moment_rff_features(
        input_particles,
        frequencies=int(feature_frequencies),
        rff_scale=float(rff_scale),
        seed=int(seed) + 404,
    )
    simulation = simulate_lif_isi(
        params, dt=dt, t_max=t_max, n_isi=n_isi, seed=int(seed) + 303
    )
    bin_edges = make_isi_bin_edges(finite_bins=finite_bins, t_max=t_max).to(target_device)
    bin_counts, empirical_bin_mass = bin_isi_observations(
        simulation["isi_times"], simulation["isi_censored"], bin_edges
    )
    payload = {
        "law_ids": torch.arange(data_size, device=target_device, dtype=torch.long),
        "regime_labels": regime_labels,
        "params": params,
        "normalized_params": normalized_params,
        "input_particles": input_particles,
        "input_features": input_features,
        "bin_counts": bin_counts,
        "empirical_bin_mass": empirical_bin_mass,
        "bin_edges": bin_edges,
    }
    payload = {key: value.detach().cpu() for key, value in payload.items()}
    _validate_payload(payload, data_size, sample_size, 2 + 2 * int(feature_frequencies))
    destination = save_hdf5_dataset(output_path, payload)
    return {"dataset_path": destination.resolve(), "payload": payload}
