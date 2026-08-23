#!/usr/bin/env bash
# run-task.sh — run one SenseNova-U1.5 task (t2i | edit | vqa | interleave)
# on the AMD GPU with this project's validated defaults.
#
# Thin wrapper over the upstream example scripts (third_party/SenseNova-U1):
# injects --model_path, --vram_mode, --attn_backend and output paths, then
# forwards every extra flag to the upstream script (later flags win, so you
# can still override anything).
#
# Usage:
#   bash scripts/run-task.sh t2i  --prompt "a cat astronaut" --width 2048 --height 2048
#   bash scripts/run-task.sh vqa  --image <img> --question "what is this?"
#   bash scripts/run-task.sh edit --image <img> --prompt "make it snow"
#   bash scripts/run-task.sh interleave --prompt "illustrated tutorial for ..."
#
# Env knobs:
#   VRAM_MODE   full | fast | balanced | low   (default: balanced — see README)
#   TASK_ARGS... extra upstream flags are forwarded as-is
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/run-task.sh <t2i|edit|vqa|interleave> [upstream flags...]

Tasks:
  t2i         text-to-image          (needs --prompt or --jsonl)
  edit        image editing          (needs --prompt and --image)
  vqa         visual understanding   (needs --image and --question)
  interleave  text+image interleaved (needs --prompt, optional --image)

Project defaults injected (override by passing the flag again):
  --model_path $MODEL_DIR --vram_mode $VRAM_MODE --attn_backend sdpa
Examples:
  bash scripts/run-task.sh t2i --prompt "A cinematic mountain village at sunrise" \
      --width 2048 --height 2048 --seed 42 --output outputs/t2i/village.png
EOF
}

[ $# -ge 1 ] || { usage >&2; exit 2; }
case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/lib/common.sh
source "$ROOT/scripts/lib/common.sh"

# Validate the task name BEFORE touching venv/model — `run-task.sh nonsense`
# must fail with a usage error even on a fresh clone (CI tests this).
TASK="$1"; shift
case "$TASK" in
    t2i)        SCRIPT="examples/t2i/inference.py" ;;
    edit)       SCRIPT="examples/editing/inference.py" ;;
    vqa)        SCRIPT="examples/vqa/inference.py" ;;
    interleave) SCRIPT="examples/interleave/inference.py" ;;
    *) die "unknown task '$TASK' (expected t2i | edit | vqa | interleave)" ;;
esac

add_rocm_path
require_venv
require_model

UPSTREAM="$ROOT/third_party/SenseNova-U1"
VRAM_MODE="${VRAM_MODE:-balanced}"

mkdir -p "$OUT_DIR/$TASK"
# Our ROCm fixes (MIOpen bypass in BLAS mode; BLAS/full-stack preloads come
# from common.sh — in full-stack mode MIOpen stays enabled) wrap the
# upstream script without touching it.
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
log "task=$TASK  model=$MODEL_DIR  vram_mode=$VRAM_MODE  attn=sdpa"
exec "$PY" -m sensenova_u1_rocm \
    "$UPSTREAM/$SCRIPT" \
    --model_path "$MODEL_DIR" \
    --vram_mode "$VRAM_MODE" \
    --attn_backend sdpa \
    "$@"
