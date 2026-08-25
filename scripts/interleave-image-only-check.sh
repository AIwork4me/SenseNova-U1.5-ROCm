#!/usr/bin/env bash
# interleave-image-only-check.sh — run one interleave_gen_image_only scenario
# (tuple vs list image_size) on the real model, with the project's ROCm env.
#
# Usage: bash scripts/interleave-image-only-check.sh <tag> <tuple|list> [num_steps] [seed]
# Writes log + summary.json + PNGs under outputs/validation/interleave-image-size/<tag>/
# Exit codes: 0 = completed, 3 = model raised (expected for list on pre-fix code).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/lib/common.sh
source "$ROOT/scripts/lib/common.sh"
add_rocm_path
require_venv
require_model

TAG="${1:?usage: interleave-image-only-check.sh <tag> <tuple|list> [num_steps] [seed]}"
MODE="${2:?usage: interleave-image-only-check.sh <tag> <tuple|list> [num_steps] [seed]}"
STEPS="${3:-5}"
SEED="${4:-42}"
case "$MODE" in
    tuple|list) ;;
    *) die "mode must be 'tuple' or 'list', got '$MODE'" ;;
esac

OUT="$OUT_DIR/validation/interleave-image-size/$TAG"
mkdir -p "$OUT"

# Same env assembly as scripts/run-task.sh (PYTHONPATH brings in the ROCm
# wrapper module; sensenova_u1 itself resolves via the editable install).
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

log "tag=$TAG mode=$MODE steps=$STEPS seed=$SEED model=$MODEL_DIR ld_preload=${LD_PRELOAD:-none}"
# errexit would kill the script on the *expected* rc=3 before PIPESTATUS is
# captured — disable it around the pipeline only.
set +e
set -o pipefail
"$PY" -m sensenova_u1_rocm "$ROOT/scripts/interleave-image-only-check.py" \
    --model_path "$MODEL_DIR" \
    --mode "$MODE" \
    --tag "$TAG" \
    --num_steps "$STEPS" \
    --seed "$SEED" \
    --out_dir "$OUT" 2>&1 | tee "$OUT/log.txt"
rc="${PIPESTATUS[0]}"
set -e
log "tag=$TAG rc=$rc (0=completed, 3=model exception)"
log "summary: $OUT/summary.json"
exit "$rc"
