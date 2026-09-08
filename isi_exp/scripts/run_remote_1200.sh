#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$PROJECT_ROOT"

REMOTE_PYTHON=${REMOTE_PYTHON:-"$HOME/anaconda3/envs/trail/bin/python"}
REMOTE_VENV="$PROJECT_ROOT/.venv_remote"
if [[ ! -x "$REMOTE_VENV/bin/python" ]]; then
    "$REMOTE_PYTHON" -m venv --system-site-packages "$REMOTE_VENV"
fi
PYTHON_BIN="$REMOTE_VENV/bin/python"
if ! "$PYTHON_BIN" -c "import omegaconf" >/dev/null 2>&1; then
    "$PYTHON_BIN" -m pip install "omegaconf>=2.3,<3"
fi

"$PYTHON_BIN" -c 'import h5py, omegaconf, torch; assert torch.cuda.device_count() >= 2; print(f"torch={torch.__version__} gpus={torch.cuda.device_count()} h5py={h5py.__version__}")'

RUN_TAG=${RUN_TAG:-"n1200_s200_seed0_e300_$(date +%Y%m%d_%H%M%S)"}
RUN_ROOT="experiments/$RUN_TAG"
DATA_FILE="data/generated/isi_lif_laws_n1200_s200_seed0.h5"
mkdir -p "$RUN_ROOT"
printf '%s\n' "$RUN_ROOT" > remote_run_path.txt

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" data/generate_isi_lif_laws.py \
    --data-size 1200 \
    --sample-size 200 \
    --output generated/isi_lif_laws_n1200_s200_seed0.h5 \
    --seed 0 \
    --device cuda \
    > "$RUN_ROOT/generation.log" 2>&1

run_model() {
    local model_name=$1
    local gpu=$2
    local epochs=$3
    CUDA_VISIBLE_DEVICES=$gpu "$PYTHON_BIN" scripts/train.py \
        --config configs/remote_1200.yaml \
        --run-name "$model_name" \
        "experiment.output_root=$RUN_ROOT" \
        "experiment.seed=0" \
        "data.file=$DATA_FILE" \
        "data.train_size=1000" \
        "data.val_size=200" \
        "data.test_size=0" \
        "data.split_seed=0" \
        "model.name=$model_name" \
        "training.epochs=$epochs" \
        > "$RUN_ROOT/${model_name}.stdout.log" 2>&1
}

run_model isi_context_deepsets 0 300 &
gpu0_pid=$!
(
    run_model isi_feature_mlp 1 300
    run_model isi_param_mlp 1 300
    run_model kernel_regression 1 0
) &
gpu1_pid=$!

status=0
wait "$gpu0_pid" || status=$?
wait "$gpu1_pid" || status=$?
if [[ "$status" -ne 0 ]]; then
    echo "one or more model runs failed; inspect $RUN_ROOT/*.stdout.log" >&2
    exit "$status"
fi

"$PYTHON_BIN" - "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
models = (
    "isi_context_deepsets",
    "isi_feature_mlp",
    "isi_param_mlp",
    "kernel_regression",
)
summary = {"run_root": str(root), "models": {}}
for model in models:
    with (root / model / "results.json").open(encoding="utf-8") as handle:
        result = json.load(handle)
    summary["models"][model] = {
        "best_epoch": result["best_epoch"],
        "validation": result["metrics_by_split"]["val"],
        "best_checkpoint_path": result["best_checkpoint_path"],
    }
with (root / "summary.json").open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
    handle.write("\n")
for model, result in summary["models"].items():
    nll = result["validation"]["observable_metrics"]["observation_nll"]
    print(f"{model}\tbest_epoch={result['best_epoch']}\tval_observation_nll={nll:.10f}")
PY
