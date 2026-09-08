#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-$project_root/.venv/bin/python}"
run_name="${RUN_NAME:-student_t_1400_train1000_valid200}"
run_dir="$project_root/experiments/$run_name"
cd "$project_root"

if [[ ! -x "$python_bin" ]]; then
    echo "Python environment is missing: $python_bin" >&2
    exit 1
fi
if [[ -e "$run_dir" ]]; then
    echo "Run directory already exists: $run_dir" >&2
    exit 1
fi

exec > >(tee -a training_ssh3_1400.log) \
     2> >(tee -a training_ssh3_1400.error.log >&2)

echo "Started: $(date --iso-8601=seconds)"
echo "Python: $python_bin"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

/usr/bin/time -v -o training_ssh3_1400.resource.txt \
    env PYTHON_BIN="$python_bin" bash scripts/train.sh \
    --config configs/conditional_flow.json --run-name "$run_name"

env PYTHON_BIN="$python_bin" bash scripts/evaluate.sh \
    --checkpoint "$run_dir/best.pt" --split validation \
    --output "$run_dir/validation_metrics.json"

echo "Finished: $(date --iso-8601=seconds)"
