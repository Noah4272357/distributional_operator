#!/usr/bin/env sh
set -eu
DATA_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/" && pwd)
cd "$DATA_ROOT"
exec "${PYTHON_BIN:-python}" generate_isi_lif_laws.py "$@"
