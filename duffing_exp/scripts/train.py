#!/usr/bin/env python3
"""Thin training entry point."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from duffing_dataset.preprocessing import (
    inverse_pca,
    prepare_duffing_pca,
    transform_pca,
)
from src.data import build_dataloaders as build_pca_dataloaders
from src.models import build_model
from src.training.diffusion_trainer import DiffusionTrainer
from src.training.factories import build_optimizer, build_scheduler
from src.training.losses import build_loss
from src.training.trainer import Trainer
from src.utils.checkpoint import load_checkpoint
from src.utils.config import create_run_dir, load_config, resolve_config, save_config
from src.utils.seed import seed_everything


class PairedHdf5Dataset(Dataset[tuple[Tensor, Tensor]]):
    """Read paired empirical distributions lazily from separate HDF5 files."""

    def __init__(
        self,
        input_path: Path,
        input_dataset: str,
        target_path: Path,
        target_dataset: str,
        indices: np.ndarray,
        preprocessing_state: dict | None = None,
    ) -> None:
        self.input_path = str(input_path)
        self.input_dataset = input_dataset
        self.target_path = str(target_path)
        self.target_dataset = target_dataset
        self.indices = indices.astype(np.int64, copy=False)
        self._input_file: h5py.File | None = None
        self._target_file: h5py.File | None = None
        with h5py.File(self.input_path, "r") as input_file, h5py.File(
            self.target_path, "r"
        ) as target_file:
            input_shape = tuple(input_file[input_dataset].shape)
            target_shape = tuple(target_file[target_dataset].shape)
        if input_shape != target_shape or len(input_shape) != 3:
            raise ValueError(
                "input and target datasets must share shape "
                "(fields, samples, dimension)"
            )
        if self.indices.size and (
            self.indices.min() < 0 or self.indices.max() >= input_shape[0]
        ):
            raise IndexError("dataset split exceeds the available fields")
        self.sample_shape = target_shape[1:]
        self.preprocessing_enabled = preprocessing_state is not None
        self.input_coefficient_mean = None
        self.input_coefficient_scale = None
        self.target_coefficient_mean = None
        self.target_coefficient_scale = None
        if preprocessing_state is None:
            self.input_mean = None
            self.input_components = None
            self.target_mean = None
            self.target_components = None
            self.input_sample_shape = input_shape[1:]
        else:
            if preprocessing_state.get("normalization_enabled", False):
                self.input_coefficient_mean = preprocessing_state["input_coefficient_mean"]
                self.input_coefficient_scale = preprocessing_state["input_coefficient_scale"]
                self.target_coefficient_mean = preprocessing_state["target_coefficient_mean"]
                self.target_coefficient_scale = preprocessing_state["target_coefficient_scale"]
            self.input_mean = np.asarray(
                preprocessing_state["input_mean"], dtype=np.float32
            )
            self.input_components = np.asarray(
                preprocessing_state["input_components"], dtype=np.float32
            )
            self.target_mean = np.asarray(
                preprocessing_state["target_mean"], dtype=np.float32
            )
            self.target_components = np.asarray(
                preprocessing_state["target_components"], dtype=np.float32
            )
            if self.input_mean.shape != (input_shape[-1],):
                raise ValueError("input PCA mean does not match the input grid")
            if self.input_components.ndim != 2 or self.input_components.shape[1] != input_shape[-1]:
                raise ValueError("input PCA components do not match the input grid")
            if self.target_mean.shape != (target_shape[-1],):
                raise ValueError("target PCA mean does not match the target grid")
            if self.target_components.ndim != 2 or self.target_components.shape[1] != target_shape[-1]:
                raise ValueError("target PCA components do not match the target grid")
            self.input_sample_shape = (
                input_shape[1],
                int(self.input_components.shape[0]),
            )

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if self._input_file is None:
            self._input_file = h5py.File(self.input_path, "r")
            self._target_file = h5py.File(self.target_path, "r")
        field_index = int(self.indices[index])
        inputs = np.asarray(
            self._input_file[self.input_dataset][field_index], dtype=np.float32
        )
        targets = np.asarray(
            self._target_file[self.target_dataset][field_index], dtype=np.float32
        )
        if self.preprocessing_enabled:
            inputs = transform_pca(
                inputs, self.input_mean, self.input_components,
                self.input_coefficient_mean, self.input_coefficient_scale,
            )
            target_coefficients = transform_pca(
                targets, self.target_mean, self.target_components,
                self.target_coefficient_mean, self.target_coefficient_scale,
            )
            targets = inverse_pca(
                target_coefficients, self.target_mean, self.target_components,
                self.target_coefficient_mean, self.target_coefficient_scale,
            )
        return (
            torch.from_numpy(np.ascontiguousarray(inputs)),
            torch.from_numpy(np.ascontiguousarray(targets)),
        )

    def close(self) -> None:
        for name in ("_input_file", "_target_file"):
            handle = getattr(self, name)
            if handle is not None:
                handle.close()
                setattr(self, name, None)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_input_file"] = None
        state["_target_file"] = None
        return state


@dataclass
class PairedDataBundle:
    train_loader: DataLoader
    validation_loader: DataLoader
    test_loader: DataLoader | None
    train_dataset: PairedHdf5Dataset
    validation_dataset: PairedHdf5Dataset
    test_dataset: PairedHdf5Dataset | None
    pca_state: dict

    def close(self) -> None:
        self.train_dataset.close()
        self.validation_dataset.close()
        if self.test_dataset is not None:
            self.test_dataset.close()


def build_paired_dataloaders(config: dict, project_root: Path) -> PairedDataBundle:
    """Build deterministic, disjoint train/validation and optional test splits."""
    data_config = config["data"]
    input_path = (project_root / data_config["input_path"]).resolve()
    target_path = (project_root / data_config["target_path"]).resolve()
    input_dataset = str(data_config["input_dataset"])
    target_dataset = str(data_config["target_dataset"])
    with h5py.File(input_path, "r") as input_file, h5py.File(
        target_path, "r"
    ) as target_file:
        input_shape = tuple(input_file[input_dataset].shape)
        target_shape = tuple(target_file[target_dataset].shape)
    if input_shape != target_shape or len(input_shape) != 3:
        raise ValueError("configured datasets must share (fields, samples, dimension)")
    train_size = int(data_config["train_size"])
    validation_size = int(data_config["validation_size"])
    test_size = int(data_config.get("test_size", 0))
    if min(train_size, validation_size) < 1 or test_size < 0:
        raise ValueError("train/validation must be positive and test non-negative")
    if train_size + validation_size + test_size > input_shape[0]:
        raise ValueError("configured splits exceed the available fields")
    generator = torch.Generator().manual_seed(int(config["seed"]))
    indices = np.random.default_rng(int(config["seed"])).permutation(input_shape[0])
    train_indices = indices[:train_size]
    preprocessing_config = data_config.get("preprocessing", {})
    preprocessing_state = None
    pca_state = {"dimension": int(input_shape[-1])}
    if bool(preprocessing_config.get("enabled", False)):
        if str(preprocessing_config.get("name", "pca")).lower() != "pca":
            raise ValueError("paired Duffing preprocessing currently supports only PCA")
        input_dimension = int(preprocessing_config["input_dimension"])
        target_dimension = int(preprocessing_config["target_dimension"])
        state_template = str(preprocessing_config["state_path"])
        state_relative_path = state_template.format(
            seed=int(config["seed"]),
            train_size=train_size,
            input_dim=input_dimension,
            target_dim=target_dimension,
        )
        state_path = (project_root / state_relative_path).resolve()
        preprocessing_state = prepare_duffing_pca(
            input_path,
            target_path,
            state_path,
            train_indices,
            input_dataset=input_dataset,
            target_dataset=target_dataset,
            input_dimension=input_dimension,
            target_dimension=target_dimension,
            fields_per_batch=int(
                preprocessing_config.get("fields_per_batch", 64)
            ),
            rebuild=bool(preprocessing_config.get("rebuild", False)),
            normalize=bool(preprocessing_config.get("normalize", False)),
            normalization_epsilon=float(preprocessing_config.get("normalization_epsilon", 1e-8)),
        )
        pca_state = {
            "dimension": input_dimension,
            "input_dimension": input_dimension,
            "target_pca_dimension": target_dimension,
            "output_dimension": int(target_shape[-1]),
            "input_mean": preprocessing_state["input_mean"],
            "input_components": preprocessing_state["input_components"],
            "target_mean": preprocessing_state["target_mean"],
            "target_components": preprocessing_state["target_components"],
            "input_explained_variance_ratio": float(
                np.asarray(
                    preprocessing_state["input_explained_variance_ratio"]
                ).item()
            ),
            "target_explained_variance_ratio": float(
                np.asarray(
                    preprocessing_state["target_explained_variance_ratio"]
                ).item()
            ),
            "state_path": preprocessing_state["state_path"],
        }
        if preprocessing_state.get("normalization_enabled", False):
            pca_state.update({key: preprocessing_state[key] for key in (
                "normalization_enabled", "normalization_state_path",
                "input_coefficient_mean", "input_coefficient_scale",
                "target_coefficient_mean", "target_coefficient_scale",
            )})
    common = (
        input_path,
        input_dataset,
        target_path,
        target_dataset,
    )
    train_dataset = PairedHdf5Dataset(
        *common,
        indices=train_indices,
        preprocessing_state=preprocessing_state,
    )
    validation_dataset = PairedHdf5Dataset(
        *common,
        indices=indices[train_size : train_size + validation_size],
        preprocessing_state=preprocessing_state,
    )
    test_dataset = (
        PairedHdf5Dataset(
            *common,
            indices=indices[
                train_size + validation_size : train_size
                + validation_size
                + test_size
            ],
            preprocessing_state=preprocessing_state,
        )
        if test_size
        else None
    )
    loader_options = {
        "batch_size": int(data_config["batch_size"]),
        "num_workers": int(data_config["num_workers"]),
        "pin_memory": bool(data_config["pin_memory"]),
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_options
    )
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, **loader_options
    )
    test_loader = (
        DataLoader(test_dataset, shuffle=False, **loader_options)
        if test_dataset is not None
        else None
    )
    return PairedDataBundle(
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        pca_state=pca_state,
    )


def build_training_dataloaders(config: dict, project_root: Path):
    """Use paired distributions when configured, otherwise the legacy PCA path."""
    if "input_path" in config["data"]:
        return build_paired_dataloaders(config, project_root)
    return build_pca_dataloaders(config, project_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/conditional_flow.json")
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--run-name")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config = resolve_config(
        load_config((project_root / args.config).resolve()),
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
    )
    seed_everything(int(config["seed"]))
    device = torch.device(config["training"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.empty(0, device=device)
        torch.cuda.reset_peak_memory_stats()
    total_start = time.perf_counter()
    run_dir = create_run_dir(project_root, config, args.run_name)
    save_config(config, run_dir / "config.json")
    bundle = build_training_dataloaders(config, project_root)
    try:
        model = build_model(config["model"], bundle.pca_state).to(device)
        criterion = build_loss(config["loss"]).to(device)
        optimizer = build_optimizer(config["optimizer"], model)
        scheduler = build_scheduler(
            config["scheduler"], optimizer, int(config["training"]["epochs"])
        )
        start_epoch, best_validation = 0, float("inf")
        if args.resume:
            checkpoint = load_checkpoint(
                args.resume.expanduser().resolve(), model, optimizer, scheduler
            )
            start_epoch = int(checkpoint["epoch"]) + 1
            best_validation = float(checkpoint["best_validation"])
        pipeline_name = str(config["training"].get("pipeline", "standard")).lower()
        trainer_class = DiffusionTrainer if pipeline_name == "diffusion" else Trainer
        trainer = trainer_class(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            train_loader=bundle.train_loader,
            validation_loader=bundle.validation_loader,
            device=device,
            config=config,
            run_dir=run_dir,
            start_epoch=start_epoch,
            best_validation=best_validation,
        )
        summary = trainer.fit()
        summary["total_seconds_including_preprocessing"] = time.perf_counter() - total_start
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2), flush=True)
    finally:
        bundle.close()


if __name__ == "__main__":
    main()
