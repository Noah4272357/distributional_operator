#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    python_bin="$PYTHON_BIN"
elif [[ -x "$project_root/.venv/bin/python" ]]; then
    python_bin="$project_root/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    python_bin="$(command -v python3)"
else
    python_bin="$(command -v python)"
fi

if [[ ! -x "$python_bin" ]]; then
    echo "Python environment is missing or not executable: $python_bin" >&2
    exit 1
fi

configs=(
    configs/momentMLP.json
    configs/distributional_operator.json
)

run_prefix="${RUN_PREFIX:-}"
dry_run="${DRY_RUN:-0}"
if [[ "$dry_run" != "0" && "$dry_run" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1; got: $dry_run" >&2
    exit 1
fi

run_names=()
for config_path in "${configs[@]}"; do
    if [[ ! -f "$config_path" ]]; then
        echo "Configuration file is missing: $project_root/$config_path" >&2
        exit 1
    fi
    config_stem="$(basename "$config_path" .json)"
    run_name="${config_stem#duffing_}"
    if [[ -n "$run_prefix" ]]; then
        run_name="${run_prefix}_${run_name}"
    fi
    if [[ -e "$project_root/experiments/$run_name" ]]; then
        echo "Run directory already exists: $project_root/experiments/$run_name" >&2
        echo "Set RUN_PREFIX to a new value or move the existing result." >&2
        exit 1
    fi
    run_names+=("$run_name")
done

echo "Python: $python_bin"
echo "Experiments: ${#configs[@]}"
echo "Started: $(date '+%Y-%m-%dT%H:%M:%S%z')"

for index in "${!configs[@]}"; do
    config_path="${configs[$index]}"
    run_name="${run_names[$index]}"
    echo "[$((index + 1))/${#configs[@]}] $run_name"
    if [[ "$dry_run" == "1" ]]; then
        printf '  %q' "$python_bin" scripts/train.py --config "$config_path" "$@" \
            --run-name "$run_name"
        printf '\n'
        continue
    fi
    "$python_bin" scripts/train.py \
        --config "$config_path" \
        "$@" \
        --run-name "$run_name"
done

echo "Finished: $(date '+%Y-%m-%dT%H:%M:%S%z')"
