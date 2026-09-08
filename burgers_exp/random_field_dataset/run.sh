#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-$HOME/miniconda3/envs/d2l/bin/python}"
cd "$script_dir"

exec > >(tee -a run.log) 2> >(tee -a error.log >&2)

echo "Started: $(date --iso-8601=seconds)"
echo "Python: $python_bin"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

/usr/bin/time -v -o generation_resource.txt \
  "$python_bin" generate_dataset.py data_generation_config.json --large-experiment

echo "Finished: $(date --iso-8601=seconds)"
