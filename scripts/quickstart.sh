#!/usr/bin/env bash
# quickstart.sh — ONE COMMAND from a bare ROCm host to a generated image
# (or a VQA answer) from SenseNova-U1.5-8B-MoT.
#
#   bash scripts/quickstart.sh              # text-to-image (default)
#   bash scripts/quickstart.sh vqa          # ask the model about an image
#   bash scripts/quickstart.sh edit         # edit the bundled sample image
#   bash scripts/quickstart.sh interleave   # interleaved text+image answer
#
# Stages (each idempotent — re-run resumes where you left off):
#   1. environment check      (scripts/00-check-env.sh)
#   2. Python + ROCm PyTorch  (scripts/01-setup-venv.sh, ~10 GiB)
#   3. model download         (scripts/02-fetch-model.sh, ~50 GiB, SHA256-verified)
#   4. run the demo task      (scripts/run-task.sh)
#
# Env knobs:
#   PROMPT="..."    custom prompt for t2i / interleave
#   QUESTION="..."  custom question for vqa
#   VRAM_MODE=...   full | fast | balanced | low  (default balanced)
#   SKIP_CHECKS=1   skip stage 1 (you know your host is fine)
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/quickstart.sh [t2i|vqa|edit|interleave]

One command: check environment, build the venv, download the 50 GiB
checkpoint (first run only), and generate. Default task: t2i.
EOF
}

TASK="${1:-t2i}"
case "$TASK" in
    -h|--help) usage; exit 0 ;;
    t2i|vqa|edit|interleave) ;;
    *) echo "ERROR: unknown task '$TASK'" >&2; usage >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/lib/common.sh
source "$ROOT/scripts/lib/common.sh"

echo "==================================================================="
echo " SenseNova-U1.5-ROCm quickstart — task: $TASK"
echo "==================================================================="

if [ "${SKIP_CHECKS:-0}" != "1" ]; then
    log "stage 1/4: environment check"
    bash "$ROOT/scripts/00-check-env.sh"
fi

if [ ! -x "$PY" ] || ! "$PY" -c 'import torch, sensenova_u1' 2>/dev/null; then
    log "stage 2/4: building Python environment (ROCm PyTorch 2.8.0)"
    bash "$ROOT/scripts/01-setup-venv.sh"
else
    log "stage 2/4: Python environment already ready"
fi

if [ ! -f "$MODEL_DIR/model-00016-of-00016.safetensors" ]; then
    log "stage 3/4: downloading checkpoint (~50 GiB, resumable, SHA256-verified)"
    bash "$ROOT/scripts/02-fetch-model.sh"
else
    log "stage 3/4: checkpoint already present at $MODEL_DIR"
fi

log "stage 4/4: running $TASK demo"
mkdir -p "$OUT_DIR/quickstart"
STAMP="$(date +%Y%m%d-%H%M%S)"

case "$TASK" in
    t2i)
        PROMPT="${PROMPT:-A cinematic mountain village at sunrise, golden light over slate roofs, drifting mist between timber houses, ultra detailed}"
        bash "$ROOT/scripts/run-task.sh" t2i \
            --prompt "$PROMPT" \
            --width 2048 --height 2048 \
            --cfg_scale 4.0 --cfg_norm none --timestep_shift 3.0 --num_steps 50 \
            --seed 42 \
            --output "$OUT_DIR/quickstart/t2i-$STAMP.png" \
            --profile
        OUT="$OUT_DIR/quickstart/t2i-$STAMP.png"
        ;;
    vqa)
        UP="$ROOT/third_party/SenseNova-U1"
        bash "$ROOT/scripts/run-task.sh" vqa \
            --image "$UP/examples/vqa/data/images/menu.jpg" \
            --question "${QUESTION:-Describe this image in one sentence.}" \
            --max_new_tokens 512 \
            --output "$OUT_DIR/quickstart/vqa-$STAMP.txt" \
            --profile
        OUT="$OUT_DIR/quickstart/vqa-$STAMP.txt"
        ;;
    edit)
        UP="$ROOT/third_party/SenseNova-U1"
        bash "$ROOT/scripts/run-task.sh" edit \
            --image "$UP/examples/editing/data/images/1.webp" \
            --prompt "${PROMPT:-Change the jacket of the person on the left to bright yellow.}" \
            --cfg_scale 4.0 --img_cfg_scale 1.0 --cfg_norm none \
            --timestep_shift 3.0 --num_steps 50 \
            --output "$OUT_DIR/quickstart/edit-$STAMP.png" \
            --profile
        OUT="$OUT_DIR/quickstart/edit-$STAMP.png"
        ;;
    interleave)
        bash "$ROOT/scripts/run-task.sh" interleave \
            --prompt "${PROMPT:-I want to learn how to cook tomato and egg stir-fry. Please give me a beginner-friendly illustrated tutorial.}" \
            --resolution "16:9" \
            --output_dir "$OUT_DIR/quickstart/interleave-$STAMP" \
            --stem demo
        OUT="$OUT_DIR/quickstart/interleave-$STAMP"
        ;;
esac

echo "==================================================================="
log "done — output: $OUT"
echo "==================================================================="
