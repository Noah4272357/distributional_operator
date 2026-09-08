#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$PROJECT_ROOT"

PYTHON_BIN=${PYTHON_BIN:-"$PROJECT_ROOT/.venv_remote/bin/python"}
RUN_ROOT=${RUN_ROOT:-"experiments/all_models_5seeds_20260906_195636"}
models=(
    isi_context_deepsets
    isi_feature_mlp
    isi_param_mlp
    kernel_regression
)

resume_model() {
    local model=$1
    local seed=$2
    local gpu=$3
    local total_epochs=1000
    local run_dir="$RUN_ROOT/${model}_seed${seed}"
    if [[ "$model" == "kernel_regression" ]]; then
        total_epochs=0
    fi
    [[ -f "$run_dir/config.yaml" ]]
    [[ -f "$run_dir/model_ckpt/last.pt" ]]
    CUDA_VISIBLE_DEVICES=$gpu "$PYTHON_BIN" - "$run_dir" "$total_epochs" \
        >> "$RUN_ROOT/${model}_seed${seed}.log" 2>&1 <<'PY'
import sys
from pathlib import Path

from scripts.train import run_training
from src.utils.config import load_config

run_dir = Path(sys.argv[1])
total_epochs = int(sys.argv[2])
cfg = load_config(run_dir / "config.yaml", [f"training.epochs={total_epochs}"])
run_training(
    cfg,
    resume=run_dir / "model_ckpt" / "last.pt",
    run_dir=run_dir,
)
PY
}

worker() {
    local gpu=$1
    local index=0
    local model
    local seed
    for model in "${models[@]}"; do
        for seed in 0 1 2 3 4; do
            if (( index % 2 == gpu )); then
                resume_model "$model" "$seed" "$gpu"
            fi
            ((index += 1))
        done
    done
}

worker 0 &
worker0_pid=$!
worker 1 &
worker1_pid=$!
wait "$worker0_pid"
wait "$worker1_pid"
