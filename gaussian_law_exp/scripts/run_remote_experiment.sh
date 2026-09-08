#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_NAME="adamw_train1000_test200_epochs300"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
mkdir -p experiments

/usr/bin/time -v \
  -o "experiments/${RUN_NAME}_time.txt" \
  "$PYTHON_BIN" scripts/train.py --run-name "$RUN_NAME" \
  > "experiments/${RUN_NAME}_stdout.log" \
  2> error.log
