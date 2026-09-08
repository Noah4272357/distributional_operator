#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
DATA="$ROOT/dataset/generated/McKean_Vlasov/McKean_Vlasov_train1000_val200.h5"
BASE="$ROOT/experiments/comparison_seed0_200epochs_1000train_200val"

for model in deepsets moment_only; do
  test ! -e "$BASE/${model}_700"
done

"$PYTHON_BIN" scripts/train.py --config configs/config.yaml \
  --run-dir "$BASE/deepsets_700" experiment.seed=0 experiment.epochs=700 \
  generated_data.file="$DATA" model.name=deepsets

"$PYTHON_BIN" scripts/train.py --config configs/config.yaml \
  --run-dir "$BASE/moment_only_700" experiment.seed=0 experiment.epochs=700 \
  generated_data.file="$DATA" model.name=moment_only \
  model.hidden_width=128 model.hidden_layers=3

"$PYTHON_BIN" - "$BASE" <<'PY'
import csv
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
runs = {
    "deepsets": root / "deepsets_700",
    "kernel_regression": root / "kernel_regression",
    "moment_only": root / "moment_only_700",
}
rows = []
for model_name, run in runs.items():
    payload = json.loads((run / "artifacts/best_metrics.json").read_text())
    val = payload["val"]
    row = {"model_name": model_name, "best_epoch": payload["best_epoch"], "training_loss": val["training_loss"]}
    row.update(val.get("observable_metrics", {}))
    row.update(val.get("synthetic_diagnostic_metrics", {}))
    rows.append(row)

fields = ["model_name", "best_epoch"] + sorted({k for row in rows for k in row if k not in {"model_name", "best_epoch"}})
target = root / "best_results.csv"
temporary = target.with_suffix(".csv.tmp")
with temporary.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
os.replace(temporary, target)
PY
