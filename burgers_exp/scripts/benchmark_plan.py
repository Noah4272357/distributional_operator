#!/usr/bin/env python3
"""Run the distributional-flow benchmark specified by plan.md."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.models import build_model
from src.training.factories import build_optimizer
from src.training.losses import build_loss
from src.utils.metrics import distribution_metrics


class PairedHdf5Dataset(Dataset[tuple[Tensor, Tensor]]):
    """Read paired empirical distributions lazily from two HDF5 files."""

    def __init__(
        self,
        input_path: Path,
        input_dataset: str,
        target_path: Path,
        target_dataset: str,
        indices: np.ndarray,
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
                "input and target datasets must share shape (fields, samples, dimension)"
            )
        if self.indices.size and self.indices.max() >= input_shape[0]:
            raise IndexError("dataset split exceeds the available fields")
        self.sample_shape = input_shape[1:]

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
        return torch.from_numpy(inputs), torch.from_numpy(targets)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_input_file"] = None
        state["_target_file"] = None
        return state


def _loader(
    dataset: Dataset,
    config: dict,
    *,
    shuffle: bool,
    generator: torch.Generator,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config["num_workers"]),
        pin_memory=bool(config["pin_memory"]),
        generator=generator,
    )


def _epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float | None,
) -> tuple[float, dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    field_count = 0
    metric_sums = {
        "mmd": 0.0,
        "sliced_wasserstein": 0.0,
        "energy_distance": 0.0,
    }
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            if training:
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            batch_size = inputs.shape[0]
            total_loss += float(loss.detach()) * batch_size
            if not training:
                batch_metrics = distribution_metrics(predictions, targets)
                for name, value in batch_metrics.items():
                    metric_sums[name] += float(value) * batch_size
            field_count += batch_size
    metrics = {
        name: value / field_count for name, value in metric_sums.items()
    } if not training else {}
    return total_loss / field_count, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/conditional_flow.json")
    )
    parser.add_argument("--output", type=Path, default=Path("plan_benchmark.json"))
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run one train batch and one validation batch for integration testing",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config = json.loads((project_root / args.config).read_text(encoding="utf-8"))
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_config = config["data"]
    train_size = int(data_config["train_size"])
    validation_size = int(data_config["validation_size"])
    rng = np.random.default_rng(seed)
    indices = rng.permutation(train_size + validation_size)
    if args.smoke_test:
        smoke_size = int(data_config["batch_size"])
        train_indices = indices[:smoke_size]
        validation_indices = indices[train_size : train_size + smoke_size]
    else:
        train_indices = indices[:train_size]
        validation_indices = indices[train_size:]
    dataset_args = (
        project_root / data_config["input_path"],
        data_config["input_dataset"],
        project_root / data_config["target_path"],
        data_config["target_dataset"],
    )
    train_dataset = PairedHdf5Dataset(
        *dataset_args, indices=train_indices
    )
    validation_dataset = PairedHdf5Dataset(
        *dataset_args, indices=validation_indices
    )
    num_samples, dimension = train_dataset.sample_shape
    generator = torch.Generator().manual_seed(seed)
    train_loader = _loader(
        train_dataset, data_config, shuffle=True, generator=generator
    )
    validation_loader = _loader(
        validation_dataset, data_config, shuffle=False, generator=generator
    )

    device = torch.device(config["training"]["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires the configured CUDA device")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    model = build_model(config["model"], {"dimension": dimension}).to(device)
    criterion = build_loss(config["loss"]).to(device)
    optimizer = build_optimizer(config["optimizer"], model)
    epochs = 1 if args.smoke_test else int(config["training"]["epochs"])
    grad_clip_value = config["training"].get("grad_clip")
    grad_clip = None if grad_clip_value is None else float(grad_clip_value)

    started = time.perf_counter()
    epoch_records = []
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        train_loss, _ = _epoch(
            model, train_loader, criterion, device, optimizer, grad_clip
        )
        validation_sinkhorn, validation_metrics = _epoch(
            model, validation_loader, criterion, device, None, None
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_sinkhorn_distance": validation_sinkhorn,
            "validation_mmd": validation_metrics["mmd"],
            "validation_sliced_wasserstein": validation_metrics[
                "sliced_wasserstein"
            ],
            "validation_energy_distance": validation_metrics["energy_distance"],
            "seconds": time.perf_counter() - epoch_started,
        }
        epoch_records.append(record)
        print(json.dumps(record), flush=True)

    torch.cuda.synchronize(device)
    summary = {
        "config": config,
        "shape": {
            "batch_size": int(data_config["batch_size"]),
            "num_samples": num_samples,
            "input_dimension": dimension,
            "output_dimension": dimension,
        },
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "total_seconds": time.perf_counter() - started,
        "mean_epoch_seconds": sum(item["seconds"] for item in epoch_records) / epochs,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "epochs": epoch_records,
    }
    output_path = project_root / args.output
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(output_path), **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
