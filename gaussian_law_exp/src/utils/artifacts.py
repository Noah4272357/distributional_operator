"""Stable result artifact names and serialization helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch


RESULT_ARTIFACT = "results.json"
RESULT_TABLE_ARTIFACT = "results.csv"
BEST_METRICS_ARTIFACT = "best_metrics.json"
EVAL_BY_SPLIT_ARTIFACT = "eval_by_split.csv"
TRAINING_HISTORY_ARTIFACT = "training_history.csv"


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


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a deterministic, JSON-safe object."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def flatten_eval_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten split metrics using the original comparison-table schema."""
    rows: list[dict[str, Any]] = []
    for split_name, split_metrics in results["metrics_by_split"].items():
        base = {
            "split": split_name,
            "model_name": results["model_name"],
            "loss_name": results["loss_name"],
            "loss_role": results["loss_role"],
            "main_result_eligible": bool(results["main_result_eligible"]),
            "uses_artificial_pairing": bool(results["uses_artificial_pairing"]),
            "pairing_policy": results["pairing_policy"],
        }
        rows.append({**base, "metric_group": "training", "metric_name": "loss", "metric_value": float(split_metrics["training_loss"])})
        for group in ("observable_metrics", "synthetic_diagnostic_metrics"):
            for name, value in split_metrics[group].items():
                rows.append({**base, "metric_group": group, "metric_name": name, "metric_value": float(value)})
    return rows


def write_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write flattened metric rows to CSV."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "model_name",
        "loss_name",
        "loss_role",
        "main_result_eligible",
        "uses_artificial_pairing",
        "pairing_policy",
        "metric_group",
        "metric_name",
        "metric_value",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_history(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write epoch-level training/validation history."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "val_training_loss",
        "val_nll",
        "val_nll_diff",
        "val_w2_distance",
        "val_kl_divergence",
        "val_hellinger_distance",
        "learning_rate",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
