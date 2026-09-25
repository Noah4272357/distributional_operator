"""Load one generated dataset and create deterministic data splits."""

from __future__ import annotations

from typing import Any

import h5py
import torch
from torch.utils.data import DataLoader

from src.data.dataset import ISILawDataset
from src.data.process_preprocessing import (
    PCAState,
    fit_process_pca,
    transform_process_rows,
    validate_pca_state,
)
from src.utils.config import project_path

SPLIT_NAMES = ("train", "val", "test")
ROW_KEYS = (
    "law_ids",
    "regime_labels",
    "params",
    "normalized_params",
    "input_particles",
    "input_features",
    "bin_counts",
    "empirical_bin_mass",
)


def split_dataset_payload(
    payload: dict[str, torch.Tensor],
    *,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
) -> dict[str, dict[str, torch.Tensor]]:
    """Shuffle once and split all law-level tensors into train, val, and optional test sets."""
    missing = [key for key in (*ROW_KEYS, "bin_edges") if key not in payload]
    if missing:
        raise ValueError(f"dataset is missing required keys: {', '.join(missing)}")
    data_size = int(payload["law_ids"].numel())
    sizes = (int(train_size), int(val_size), int(test_size))
    if sizes[0] < 1 or sizes[1] < 0 or sizes[2] < 0:
        raise ValueError("train_size must be positive; val_size and test_size cannot be negative")
    if sum(sizes) != data_size:
        raise ValueError(f"split sizes sum to {sum(sizes)}, but dataset contains {data_size} laws")
    for key in ROW_KEYS:
        if payload[key].shape[0] != data_size:
            raise ValueError(f"{key} first dimension must equal data size {data_size}")
    generator = torch.Generator().manual_seed(int(seed))
    available = {
        int(label): indices[torch.randperm(indices.numel(), generator=generator)]
        for label in torch.unique(payload["regime_labels"]).tolist()
        for indices in [(payload["regime_labels"] == int(label)).nonzero(as_tuple=True)[0]]
    }
    result = {}
    active_splits = [(name, size) for name, size in zip(SPLIT_NAMES, sizes) if size > 0]
    for split_index, (split_name, size) in enumerate(active_splits):
        if split_index == len(active_splits) - 1:
            selected = torch.cat(list(available.values()))
        else:
            remaining = sum(indices.numel() for indices in available.values())
            exact = {label: size * indices.numel() / remaining for label, indices in available.items()}
            take = {label: int(value) for label, value in exact.items()}
            shortfall = size - sum(take.values())
            order = sorted(exact, key=lambda label: exact[label] - take[label], reverse=True)
            for label in order[:shortfall]:
                take[label] += 1
            chunks = []
            for label, count in take.items():
                chunks.append(available[label][:count])
                available[label] = available[label][count:]
            selected = torch.cat(chunks)
        selected = selected[torch.randperm(selected.numel(), generator=generator)]
        if selected.numel() != size:
            raise RuntimeError(f"internal split error for {split_name}: expected {size} rows")
        result[split_name] = {key: payload[key][selected] for key in ROW_KEYS}
        result[split_name]["bin_edges"] = payload["bin_edges"]
    return result


def _validate_unique_ids(ids: torch.Tensor, label: str) -> None:
    if ids.ndim != 1 or torch.unique(ids).numel() != ids.numel():
        raise ValueError(f"{label} law_ids must be a one-dimensional unique array")


def _process_rows_for_law_ids(process_law_ids: torch.Tensor, law_ids: torch.Tensor) -> torch.Tensor:
    row_by_id = {int(law_id): row for row, law_id in enumerate(process_law_ids.tolist())}
    missing = [int(law_id) for law_id in law_ids.tolist() if int(law_id) not in row_by_id]
    if missing:
        raise ValueError(f"process file is missing law_ids: {missing[:5]}")
    return torch.tensor([row_by_id[int(law_id)] for law_id in law_ids.tolist()], dtype=torch.long)


def load_process_split_payloads(
    data_cfg: Any,
    *,
    map_location: str | torch.device = "cpu",
    preprocessing_state: PCAState | None = None,
    truncate_dim: str | int = "auto",
) -> dict[str, Any]:
    """Join process inputs to ISI targets, split by law, and apply training PCA."""
    target_path = project_path(str(data_cfg.target_file))
    process_path = project_path(str(data_cfg.process_file))
    with h5py.File(target_path, "r") as target_handle:
        if target_handle.attrs.get("format") != "isi_lif_laws":
            raise ValueError("target HDF5 file is not an isi_lif_laws dataset")
        target_payload = {
            key: torch.from_numpy(target_handle[key][...]).to(map_location)
            for key in target_handle.keys()
        }

    target_ids = target_payload["law_ids"].cpu().to(torch.long)
    _validate_unique_ids(target_ids, "target")
    for key in ("bin_counts", "empirical_bin_mass"):
        values = target_payload[key]
        if not torch.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"target {key} must contain finite nonnegative values")
    mass_sums = target_payload["empirical_bin_mass"].sum(dim=-1)
    if not torch.allclose(mass_sums, torch.ones_like(mass_sums), rtol=1.0e-5, atol=1.0e-6):
        raise ValueError("each empirical_bin_mass row must sum to one")
    split_payloads = split_dataset_payload(
        target_payload,
        train_size=int(data_cfg.train_size),
        val_size=int(data_cfg.val_size),
        test_size=int(data_cfg.test_size),
        seed=int(data_cfg.get("split_seed", 0)),
    )

    with h5py.File(process_path, "r") as process_handle:
        if process_handle.attrs.get("format") != "isi_drifted_brownian_paths":
            raise ValueError("process HDF5 file is not an isi_drifted_brownian_paths dataset")
        required = ("law_ids", "params", "regime_labels", "process_samples", "time_grid")
        missing = [key for key in required if key not in process_handle]
        if missing:
            raise ValueError(f"process HDF5 file is missing required keys: {', '.join(missing)}")
        process_ids = torch.from_numpy(process_handle["law_ids"][...]).to(torch.long)
        _validate_unique_ids(process_ids, "process")
        if set(process_ids.tolist()) != set(target_ids.tolist()):
            raise ValueError("process and target files must contain exactly the same law_ids")
        target_process_rows = _process_rows_for_law_ids(process_ids, target_ids)
        process_params = torch.from_numpy(process_handle["params"][...])[target_process_rows]
        process_regimes = torch.from_numpy(process_handle["regime_labels"][...])[target_process_rows]
        if not torch.allclose(process_params, target_payload["params"].cpu(), rtol=1.0e-6, atol=1.0e-7):
            raise ValueError("matched process and target params do not agree")
        if not torch.equal(process_regimes.to(torch.long), target_payload["regime_labels"].cpu().to(torch.long)):
            raise ValueError("matched process and target regime_labels do not agree")
        samples = process_handle["process_samples"]
        if samples.ndim != 3 or samples.shape[0] != process_ids.numel():
            raise ValueError("process_samples must have shape [number_of_laws, sample_size, time_dim]")

        split_process_rows = {
            split: _process_rows_for_law_ids(process_ids, payload["law_ids"].cpu())
            for split, payload in split_payloads.items()
        }
        pca_cfg = data_cfg.pca
        paths_per_law = int(samples.shape[1])
        fit_chunk_laws = max(1, int(pca_cfg.get("fit_chunk_size", 8192)) // paths_per_law)
        if preprocessing_state is None:
            preprocessing_state = fit_process_pca(
                samples,
                split_process_rows["train"],
                explained_variance_threshold=float(pca_cfg.explained_variance_threshold),
                truncate_dim=truncate_dim,
                chunk_laws=fit_chunk_laws,
                enforce_minimum_dim=bool(pca_cfg.get("enforce_minimum_dim", True)),
            )
            preprocessing_state["training_law_ids"] = split_payloads["train"]["law_ids"].cpu().clone()
            preprocessing_state["split_seed"] = int(data_cfg.get("split_seed", 0))
            preprocessing_state["process_file"] = str(process_path.resolve())
            preprocessing_state["target_file"] = str(target_path.resolve())
        else:
            validate_pca_state(preprocessing_state, int(samples.shape[-1]))
            expected_ids = preprocessing_state.get("training_law_ids")
            if not torch.is_tensor(expected_ids) or not torch.equal(
                expected_ids.cpu().to(torch.long),
                split_payloads["train"]["law_ids"].cpu().to(torch.long),
            ):
                raise ValueError("checkpoint PCA state does not match the configured training split")

        transform_chunk = max(
            1,
            int(pca_cfg.get("transform_chunk_size", pca_cfg.get("fit_chunk_size", 8192)))
            // paths_per_law,
        )
        for split, payload in split_payloads.items():
            payload["process_features"] = transform_process_rows(
                samples,
                split_process_rows[split],
                preprocessing_state,
                chunk_laws=transform_chunk,
            ).to(map_location)
    result: dict[str, Any] = dict(split_payloads)
    result["_preprocessing_state"] = preprocessing_state
    return result


def load_split_payloads(
    data_cfg: Any,
    map_location: str | torch.device = "cpu",
    preprocessing_state: PCAState | None = None,
    truncate_dim: str | int = "auto",
) -> dict[str, Any]:
    """Load ``data.file`` and split it according to the training configuration."""
    if data_cfg.get("process_file") is not None:
        return load_process_split_payloads(
            data_cfg,
            map_location=map_location,
            preprocessing_state=preprocessing_state,
            truncate_dim=truncate_dim,
        )
    with h5py.File(project_path(str(data_cfg.file)), "r") as handle:
        if handle.attrs.get("format") != "isi_lif_laws":
            raise ValueError("HDF5 file is not an isi_lif_laws dataset")
        payload = {key: torch.from_numpy(handle[key][...]).to(map_location) for key in handle.keys()}
    return split_dataset_payload(
        payload,
        train_size=int(data_cfg.train_size),
        val_size=int(data_cfg.val_size),
        test_size=int(data_cfg.test_size),
        seed=int(data_cfg.get("split_seed", 0)),
    )


def build_dataloaders(
    data_cfg: Any,
    split_payloads: dict[str, Any],
    *,
    shuffle_train: bool = True,
) -> dict[str, DataLoader]:
    return {
        split: DataLoader(
            ISILawDataset(payload),
            batch_size=int(data_cfg.batch_size),
            shuffle=split == "train" and shuffle_train,
            num_workers=int(data_cfg.get("num_workers", 0)),
            pin_memory=bool(data_cfg.get("pin_memory", False)),
        )
        for split, payload in split_payloads.items()
        if split in SPLIT_NAMES
    }
