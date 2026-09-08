#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$PROJECT_ROOT"

PYTHON_BIN=${PYTHON_BIN:-"$PROJECT_ROOT/.venv_remote/bin/python"}
CONFIG="configs/remote_1200.yaml"
DATA_FILE="data/generated/isi_lif_laws_n1200_s200_seed0.h5"
RUN_TAG=${RUN_TAG:-"all_models_5seeds_$(date +%Y%m%d_%H%M%S)"}
RUN_ROOT="experiments/$RUN_TAG"

[[ -x "$PYTHON_BIN" ]]
[[ -f "$CONFIG" ]]
[[ -f "$DATA_FILE" ]]
mkdir -p "$RUN_ROOT"

models=(
    isi_context_deepsets
    isi_feature_mlp
    isi_param_mlp
    kernel_regression
)

run_model() {
    local model=$1
    local seed=$2
    local gpu=$3
    local epochs=300
    if [[ "$model" == "kernel_regression" ]]; then
        epochs=0
    fi
    CUDA_VISIBLE_DEVICES=$gpu "$PYTHON_BIN" scripts/train.py \
        --config "$CONFIG" \
        --run-name "${model}_seed${seed}" \
        "experiment.output_root=$RUN_ROOT" \
        "experiment.seed=$seed" \
        "model.name=$model" \
        "training.epochs=$epochs" \
        > "$RUN_ROOT/${model}_seed${seed}.log" 2>&1
}

worker() {
    local gpu=$1
    local index=0
    local model
    local seed
    for model in "${models[@]}"; do
        for seed in 0 1 2 3 4; do
            if (( index % 2 == gpu )); then
                run_model "$model" "$seed" "$gpu"
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
