#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$PROJECT_ROOT"

exec > >(tee -a table3_run.log) 2> >(tee -a table3_error.log >&2)

REMOTE_PYTHON=${REMOTE_PYTHON:-"$HOME/anaconda3/envs/trail/bin/python"}
REMOTE_VENV="$PROJECT_ROOT/.venv_remote"
if [[ ! -x "$REMOTE_VENV/bin/python" ]]; then
    "$REMOTE_PYTHON" -m venv --system-site-packages "$REMOTE_VENV"
fi

PYTHON_BIN="$REMOTE_VENV/bin/python"
if ! "$PYTHON_BIN" -c "import omegaconf" >/dev/null 2>&1; then
    "$PYTHON_BIN" -m pip install "omegaconf>=2.3,<3"
fi

"$PYTHON_BIN" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available in the selected remote environment")
print(f"torch={torch.__version__}")
print(f"cuda_devices={[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
PY

MATRIX_ROOT="experiments/table3_matrix"
mkdir -p "$MATRIX_ROOT"

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" data/generate_isi_lif_laws.py \
    --data-size 1239 \
    --sample-size 64 \
    --output generated/isi_lif_laws.h5 \
    --device cuda

run_model() {
    local model_name=$1
    local gpu=$2
    local seed
    for seed in 0 1 2 3 4; do
        CUDA_VISIBLE_DEVICES=$gpu "$PYTHON_BIN" scripts/train.py \
            --config configs/table3.yaml \
            --run-name "${model_name}_seed${seed}" \
            "model.name=${model_name}" \
            "experiment.seed=${seed}" \
            > "$MATRIX_ROOT/${model_name}_seed${seed}.stdout.log" 2>&1
    done
}

run_model isi_context_deepsets 0 &
context_pid=$!

"$PYTHON_BIN" scripts/train.py \
    --config configs/kernel_regression.yaml \
    --run-name kernel_regression_seed0 \
    > "$MATRIX_ROOT/kernel_regression_seed0.stdout.log" 2>&1 &
kernel_pid=$!

status=0
wait "$context_pid" || status=$?
wait "$kernel_pid" || status=$?
if [[ "$status" -ne 0 ]]; then
    echo "one or more Table 3 training runs failed" >&2
    exit "$status"
fi

"$PYTHON_BIN" scripts/aggregate_table3.py --root "$MATRIX_ROOT"
