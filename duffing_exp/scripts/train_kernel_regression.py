#!/usr/bin/env python3
"""Fit and evaluate the nonparametric distributional kernel regressor."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import KernelRegression, build_model
from src.training.losses import build_loss
from src.utils.metrics import distribution_metrics


def _load_laws(
    input_path: Path,
    input_dataset: str,
    target_path: Path,
    target_dataset: str,
    indices: np.ndarray,
    sample_indices: np.ndarray,
) -> tuple[Tensor, Tensor]:
    with h5py.File(input_path, "r") as input_file, h5py.File(
        target_path, "r"
    ) as target_file:
        inputs = np.asarray(input_file[input_dataset][indices], dtype=np.float32)
        targets = np.asarray(target_file[target_dataset][indices], dtype=np.float32)
    return (
        torch.from_numpy(inputs[:, sample_indices]),
        torch.from_numpy(targets[:, sample_indices]),
    )


@torch.inference_mode()
def _distances(
    model: KernelRegression,
    inputs: Tensor,
    batch_size: int,
) -> Tensor:
    chunks = []
    for start in range(0, inputs.shape[0], batch_size):
        chunks.append(model.squared_distances(inputs[start : start + batch_size]))
    return torch.cat(chunks)


def _validation_metrics(
    predictions: Tensor,
    targets: Tensor,
    sinkhorn: torch.nn.Module,
) -> dict[str, float]:
    return {
        "sinkhorn_distance": float(sinkhorn(predictions, targets)),
        **{
            name: float(value)
            for name, value in distribution_metrics(predictions, targets).items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/kernel_regression.json")
    )
    parser.add_argument("--run-name")
    parser.add_argument("--device", help="Override the device from the configuration")
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    config = json.loads((project_root / args.config).read_text(encoding="utf-8"))
    if args.device:
        config["training"]["device"] = args.device
    seed = int(config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    data_config = config["data"]
    input_path = (project_root / data_config["input_path"]).resolve()
    target_path = (project_root / data_config["target_path"]).resolve()
    with h5py.File(input_path, "r") as input_file, h5py.File(
        target_path, "r"
    ) as target_file:
        input_shape = tuple(input_file[data_config["input_dataset"]].shape)
        target_shape = tuple(target_file[data_config["target_dataset"]].shape)
    if input_shape != target_shape or len(input_shape) != 3:
        raise ValueError("input and target laws must share (laws, samples, dimension)")

    split_size = int(data_config["train_size"]) + int(
        data_config["validation_size"]
    )
    if split_size > input_shape[0]:
        raise ValueError("configured splits exceed the number of available laws")
    law_sample_count = int(data_config["law_sample_count"])
    if not 1 <= law_sample_count <= input_shape[1]:
        raise ValueError("law_sample_count is outside the available sample range")
    rng = np.random.default_rng(seed)
    law_indices = rng.permutation(input_shape[0])[:split_size]
    sample_indices = np.sort(
        rng.choice(input_shape[1], size=law_sample_count, replace=False)
    )
    train_end = int(data_config["train_size"])
    train_indices = np.sort(law_indices[:train_end])
    validation_indices = np.sort(law_indices[train_end:])
    train_inputs, train_targets = _load_laws(
        input_path,
        data_config["input_dataset"],
        target_path,
        data_config["target_dataset"],
        train_indices,
        sample_indices,
    )
    validation_inputs, validation_targets = _load_laws(
        input_path,
        data_config["input_dataset"],
        target_path,
        data_config["target_dataset"],
        validation_indices,
        sample_indices,
    )
    device = torch.device(config["training"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model_config = dict(config["model"])
    candidates = [float(value) for value in model_config.pop("bandwidth_candidates")]
    model_config["bandwidth"] = candidates[0]
    model = build_model(model_config, {"dimension": input_shape[-1]}).to(device)
    if not isinstance(model, KernelRegression):
        raise TypeError("configuration did not construct KernelRegression")
    model.fit(train_inputs, train_targets)
    sinkhorn = build_loss(config["loss"]).to(device)
    validation_inputs = validation_inputs.to(device)
    validation_targets = validation_targets.to(device)

    run_name = args.run_name or config["experiment"]["run_name"]
    run_dir = (project_root / config["experiment"]["root"] / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    validation_distances = _distances(
        model, validation_inputs, int(data_config["query_batch_size"])
    )
    sweep = []
    best_sinkhorn = float("inf")
    best_bandwidth = candidates[0]
    for bandwidth in candidates:
        model.bandwidth = bandwidth
        predictions = model.predict_from_squared_distances(validation_distances)
        sinkhorn_distance = float(sinkhorn(predictions, validation_targets))
        sweep.append(
            {
                "bandwidth": bandwidth,
                "validation_sinkhorn_distance": sinkhorn_distance,
            }
        )
        if sinkhorn_distance < best_sinkhorn:
            best_sinkhorn = sinkhorn_distance
            best_bandwidth = bandwidth

    model.bandwidth = best_bandwidth
    validation_predictions = model.predict_from_squared_distances(validation_distances)
    validation_metrics = _validation_metrics(
        validation_predictions,
        validation_targets,
        sinkhorn,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result = {
        "best_bandwidth": best_bandwidth,
        "validation_sweep": sweep,
        "validation_metrics": validation_metrics,
        "train_laws": train_inputs.shape[0],
        "validation_laws": validation_inputs.shape[0],
        "samples_per_law": law_sample_count,
        "dimension": input_shape[-1],
        "elapsed_seconds": elapsed,
        "peak_gpu_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }
    saved_config = dict(config)
    saved_config["model"] = {**config["model"], "bandwidth": best_bandwidth}
    saved_config["experiment"] = {
        **config["experiment"],
        "run_name": run_name,
    }
    (run_dir / "config.json").write_text(
        json.dumps(saved_config, indent=2) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "model": model.state_dict(),
            "bandwidth": best_bandwidth,
            "train_indices": train_indices,
            "validation_indices": validation_indices,
            "sample_indices": sample_indices,
            "validation_distances": validation_distances.cpu(),
        },
        run_dir / "model.pt",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"run_dir": str(run_dir), **result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
