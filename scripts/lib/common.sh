#!/usr/bin/env bash
# Shared helpers for SenseNova-U1.5-ROCm scripts.
# Sourced, not executed.

PROJECT_NAME="SenseNova-U1.5-ROCm"
MODEL_REPO="SenseNova/SenseNova-U1.5-8B-MoT"
MODEL_DIR_NAME="SenseNova-U1.5-8B-MoT"
UPSTREAM_PINNED_COMMIT="76c32c2"   # third_party/SenseNova-U1 @ feat/u1.5

# Project root (scripts/ is always one level below it)
ROOT="$(cd "$(dirname "${BASH_SOURCE[1]}")/.." && pwd)"

# Virtualenv created by scripts/01-setup-venv.sh
VENV="$ROOT/.venv"
PY="$VENV/bin/python"

# Where the 50.2 GB checkpoint lives. Default lands inside the HF cache
# because that is the persistent host-disk mount on the reference machine;
# override with MODEL_DIR=/path if your layout differs.
MODEL_BASE="${MODEL_BASE:-${HF_HOME:-$HOME/.cache/huggingface}/modelscope}"
MODEL_DIR="${MODEL_DIR:-$MODEL_BASE/$MODEL_DIR_NAME}"

# Generated images / answers / receipts go here
OUT_DIR="${OUT_DIR:-$ROOT/outputs}"

log()  { echo "[$PROJECT_NAME] $*"; }
warn() { echo "[$PROJECT_NAME] WARNING: $*" >&2; }
die()  { echo "[$PROJECT_NAME] FAIL: $*" >&2; exit 1; }

require_venv() {
    [ -x "$PY" ] || die "virtualenv missing at $VENV — run: bash scripts/01-setup-venv.sh"
}

require_model() {
    [ -f "$MODEL_DIR/config.json" ] || die "model not found at $MODEL_DIR — run: bash scripts/02-fetch-model.sh"
}

# Activate ROCm userspace for this shell if not already present.
add_rocm_path() {
    if [ -d /opt/rocm ] && ! echo "$PATH" | grep -q '/opt/rocm/bin'; then
        export PATH="/opt/rocm/bin:$PATH"
    fi
}
