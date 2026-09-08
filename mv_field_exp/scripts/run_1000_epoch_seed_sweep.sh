#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
DATA="$ROOT/dataset/generated/McKean_Vlasov/McKean_Vlasov_train1000_val200.h5"
RUN_ROOT="$ROOT/experiments/sweep_1000epochs_seeds0-4"

test -f "$DATA"
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT"

run_model() {
  local model="$1" gpu="$2"
  shift 2
  for seed in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/train.py \
      --config configs/config.yaml \
      --run-dir "$RUN_ROOT/$model/seed_$seed" \
      experiment.seed="$seed" experiment.epochs=1000 \
      generated_data.file="$DATA" model.name="$model" "$@"
  done
}

run_model deepsets 0 >"$RUN_ROOT/deepsets.log" 2>"$RUN_ROOT/deepsets.error.log" &
deepsets_pid=$!
run_model moment_only 1 model.hidden_width=128 model.hidden_layers=3 \
  >"$RUN_ROOT/moment_only.log" 2>"$RUN_ROOT/moment_only.error.log" &
moment_pid=$!

status=0
wait "$deepsets_pid" || status=1
wait "$moment_pid" || status=1
exit "$status"
