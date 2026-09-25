#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$PROJECT_ROOT"

PYTHON_BIN=${PYTHON_BIN:-"$PROJECT_ROOT/.venv_remote/bin/python"}
CONFIG=${CONFIG:-"configs/process_train_1000.yaml"}
GPU=${GPU:-0}
RUN_TAG=${RUN_TAG:-"distribution_operator_5seeds_$(date +%Y%m%d_%H%M%S)"}
RUN_ROOT=${RUN_ROOT:-"experiments/$RUN_TAG"}

[[ -x "$PYTHON_BIN" ]]
[[ -f "$CONFIG" ]]
[[ -f "data/generated/isi_process_data.h5" ]]
[[ -f "data/generated/isi_distribution_dataset.h5" ]]
mkdir -p "$RUN_ROOT"

echo "Run directory: $RUN_ROOT"
echo "Python: $PYTHON_BIN"
echo "GPU: $GPU"
echo "Each seed first fits/transforms the training PCA, which may be quiet for a short time."

for seed in 0 1 2 3 4; do
    log_file="$RUN_ROOT/distribution_operator_seed${seed}.log"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting seed $seed; log: $log_file"
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" scripts/train.py \
        --config "$CONFIG" \
        --run-name "distribution_operator_seed${seed}" \
        "experiment.output_root=$RUN_ROOT" \
        "experiment.seed=$seed" \
        2>&1 | tee "$log_file"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished seed $seed"
done

echo "All five seeds finished: $RUN_ROOT"
