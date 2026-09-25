#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-$project_root/.venv/bin/python}"
config_path="configs/momentMLP.json"

cd "$project_root"

if [[ ! -x "$python_bin" ]]; then
    echo "Python environment is missing or not executable: $python_bin" >&2
    exit 1
fi
if [[ ! -f "$config_path" ]]; then
    echo "Configuration file is missing: $project_root/$config_path" >&2
    exit 1
fi

for seed in 0 1 2 3 4; do
    run_name="momentMLP_seed${seed}"
    echo "Starting $run_name"
    "$python_bin" -m scripts.train \
        --config "$config_path" \
        --seed "$seed" \
        --run-name "$run_name"
done
