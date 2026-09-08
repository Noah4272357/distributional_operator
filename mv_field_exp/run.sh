#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" scripts/generate.py --config configs/generate.yaml
"$PYTHON_BIN" scripts/train.py --config configs/config.yaml "$@"
