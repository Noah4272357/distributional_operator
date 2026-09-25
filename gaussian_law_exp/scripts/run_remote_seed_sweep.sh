#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
LOG_DIR="experiments/seed_sweep_logs"
mkdir -p "$LOG_DIR"

for seed in 0 1 2 3 4; do
  for specification in \
    "distributional_operator:configs/distributional_operator_config.yaml" \
    "momentmlp:configs/mlp_config.yaml"; do
    model_name="${specification%%:*}"
    config_path="${specification#*:}"
    run_name="seed_sweep_${model_name}_seed${seed}"
    echo "Starting ${run_name}"
    "$PYTHON_BIN" scripts/train.py \
      --config "$config_path" \
      --run-name "$run_name" \
      "experiment.seed=${seed}" \
      > "${LOG_DIR}/${run_name}.stdout.log" \
      2> "${LOG_DIR}/${run_name}.stderr.log"
    echo "Finished ${run_name}"
  done
done

"$PYTHON_BIN" scripts/summarize_model_seed_runs.py \
  --experiments-dir experiments \
  --output summary.csv
