"""Training result tables, metadata copies, and prediction curves."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.metrics import BATCHED_BIN_EDGES_NDIM
from src.training.validate import move_batch_to_device


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "model_name",
        "result_role",
        "main_result_eligible",
        "metric_group",
        "regime",
        "metric_name",
        "metric_value",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_training_history_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_observation_nll",
        "val_observation_nll",
        "val_tail_bin_error_against_empirical",
        "val_hellinger_distance_against_empirical",
        "val_kl_divergence_against_empirical",
        "val_finite_renormalized_w2_against_empirical",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _flatten_eval_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for split_name, split_metrics in results["metrics_by_split"].items():
        base = {
            "split": split_name,
            "model_name": results["model_name"],
            "result_role": results["result_role"],
            "main_result_eligible": bool(results["main_result_eligible"]),
        }
        rows.append(
            {
                **base,
                "metric_group": "training",
                "regime": "",
                "metric_name": "loss",
                "metric_value": float(split_metrics["training_loss"]),
            }
        )
        for metric_name, metric_value in split_metrics["observable_metrics"].items():
            rows.append(
                {
                    **base,
                    "metric_group": "observable_metrics",
                    "regime": "",
                    "metric_name": metric_name,
                    "metric_value": float(metric_value),
                }
            )
        for regime_name, regime_metrics in split_metrics["observable_metrics_by_regime"].items():
            for metric_name, metric_value in regime_metrics.items():
                rows.append(
                    {
                        **base,
                        "metric_group": "observable_metrics_by_regime",
                        "regime": regime_name,
                        "metric_name": metric_name,
                        "metric_value": float(metric_value),
                    }
                )
    return rows


@torch.no_grad()
def _prediction_curves(model: nn.Module, dataloaders: dict[str, DataLoader], device: torch.device) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    payload: dict[str, Any] = {}
    for split_name, dataloader in dataloaders.items():
        chunks: dict[str, list[torch.Tensor]] = {
            "law_ids": [],
            "regime_labels": [],
            "params": [],
            "logits": [],
            "pred_bin_mass": [],
            "pred_cdf": [],
            "empirical_bin_mass": [],
        }
        bin_edges = None
        for batch in dataloader:
            device_batch = move_batch_to_device(batch, device)
            prediction = model(device_batch)
            batch_key_map = {
                "law_ids": "law_id",
                "regime_labels": "regime_label",
                "params": "params",
                "empirical_bin_mass": "empirical_bin_mass",
            }
            for key, batch_key in batch_key_map.items():
                chunks[key].append(batch[batch_key].detach().cpu())
            chunks["logits"].append(prediction["logits"].detach().cpu())
            chunks["pred_bin_mass"].append(prediction["pred_bin_mass"].detach().cpu())
            chunks["pred_cdf"].append(prediction["pred_cdf"].detach().cpu())
            if bin_edges is None:
                batch_bin_edges = batch["bin_edges"]
                if batch_bin_edges.ndim == BATCHED_BIN_EDGES_NDIM:
                    batch_bin_edges = batch_bin_edges[0]
                bin_edges = batch_bin_edges.detach().cpu()
        split_payload = {key: torch.cat(values, dim=0) for key, values in chunks.items()}
        split_payload["empirical_cdf"] = split_payload["empirical_bin_mass"].cumsum(dim=-1)
        if bin_edges is None:
            raise RuntimeError(f"empty dataloader for split {split_name}")
        finite_widths = bin_edges[1:] - bin_edges[:-1]
        split_payload["pred_piecewise_density"] = split_payload["pred_bin_mass"][:, :-1] / finite_widths.view(1, -1)
        split_payload["bin_edges"] = bin_edges
        payload[split_name] = split_payload
    if was_training:
        model.train()
    return payload
