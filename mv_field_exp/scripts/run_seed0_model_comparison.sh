#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
SOURCE_DATA="$ROOT/dataset/generated/McKean_Vlasov/McKean_Vlasov.h5"
SPLIT_DATA="$ROOT/dataset/generated/McKean_Vlasov/McKean_Vlasov_train1000_val200.h5"
RUN_ROOT="$ROOT/experiments/comparison_seed0_200epochs_1000train_200val"

if [[ -e "$RUN_ROOT" ]]; then
  echo "Refusing to overwrite existing run directory: $RUN_ROOT" >&2
  exit 1
fi
mkdir -p "$RUN_ROOT"
exec > >(tee "$RUN_ROOT/run.log") 2> >(tee "$RUN_ROOT/error.log" >&2)

"$PYTHON_BIN" - "$SOURCE_DATA" "$SPLIT_DATA" <<'PY'
import shutil
import sys

import h5py
import numpy as np

source, target = sys.argv[1:]
shutil.copy2(source, target)
with h5py.File(target, "r+") as handle:
    split_group = handle["split_indices"]
    train = split_group["train"][:]
    val = split_group["val"][:]
    test = split_group["test"][:]
    if (len(train), len(val), len(test)) != (800, 200, 200):
        raise RuntimeError(f"expected source splits (800, 200, 200), got {(len(train), len(val), len(test))}")
    train_1000 = np.concatenate([train, test])
    if len(np.intersect1d(train_1000, val)) != 0:
        raise RuntimeError("derived train and validation indices overlap")
    for name in list(split_group):
        del split_group[name]
    split_group.create_dataset("train", data=train_1000)
    split_group.create_dataset("val", data=val)
    split_group.create_dataset("test", data=val)
print("prepared splits: train=1000 val=200 (test aliases val for required evaluation)")
PY

common=(
  experiment.seed=0
  experiment.epochs=200
  generated_data.file="$SPLIT_DATA"
)

"$PYTHON_BIN" scripts/train.py --config configs/config.yaml \
  --run-dir "$RUN_ROOT/deepsets" "${common[@]}" model.name=deepsets

"$PYTHON_BIN" scripts/train.py --config configs/config.yaml \
  --run-dir "$RUN_ROOT/kernel_regression" "${common[@]}" \
  model.name=kernel_regression model.bandwidth=0.5 \
  model.sinkhorn.p=2 model.sinkhorn.blur=0.05 model.sinkhorn.scaling=0.9 \
  model.sinkhorn.debias=true model.sinkhorn.backend=tensorized \
  model.history_chunk_size=64

"$PYTHON_BIN" scripts/train.py --config configs/config.yaml \
  --run-dir "$RUN_ROOT/moment_only" "${common[@]}" \
  model.name=moment_only model.hidden_width=128 model.hidden_layers=3

"$PYTHON_BIN" - "$RUN_ROOT" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for model_name in ("deepsets", "kernel_regression", "moment_only"):
    payload = json.loads((root / model_name / "artifacts" / "best_metrics.json").read_text())
    val = payload["val"]
    row = {
        "model_name": model_name,
        "best_epoch": payload["best_epoch"],
        "training_loss": val["training_loss"],
    }
    row.update(val.get("observable_metrics", {}))
    row.update(val.get("synthetic_diagnostic_metrics", {}))
    rows.append(row)

fieldnames = ["model_name", "best_epoch"] + sorted(
    {key for row in rows for key in row if key not in {"model_name", "best_epoch"}}
)
with (root / "best_results.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(root / "best_results.csv")
PY
