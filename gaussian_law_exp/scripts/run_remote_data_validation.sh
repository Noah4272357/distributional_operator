#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
"$PYTHON_BIN" data/main.py --data_size 1200 --sample_size 200
"$PYTHON_BIN" scripts/compare_generated_moments.py \
  --dataset data/dataset.h5 \
  --output data/moment_covariance_comparison.json \
  > data/moment_covariance_comparison.log
