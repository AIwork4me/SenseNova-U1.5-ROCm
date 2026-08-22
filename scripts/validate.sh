#!/usr/bin/env bash
# validate.sh — full validation suite: every SenseNova-U1.5 task on the AMD
# GPU, measured, with receipts.
#
# What gets validated (one receipt per block, docs/results/validation/):
#   vqa         visual understanding, greedy (deterministic)
#   t2i         text-to-image 2048x2048 @ 50 steps, seed 42
#   t2i-think   same + reasoning mode (--think)
#   edit        image editing on the upstream sample
#   interleave  interleaved text+image generation
#   determinism t2i seed 42 run twice -> byte-identical PNG (sha256)
#   vram-modes  balanced vs fast vs low per-step latency @ 2048x2048, 10 steps
#
# Every block records: exact command, wall time, per-step time (image tasks),
# peak VRAM (device-level, rocm-smi sampler), output sha256, and the run log.
# Env knobs: VRAM_MODE (default balanced), ONLY=block-name to run one block.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/validate.sh [ONLY=<block>|--only <block>]

Runs the full validation suite and writes receipts + logs under
docs/results/. Blocks: vqa t2i t2i-think edit interleave determinism
vram-modes (vram-mode-balanced/fast/low are addressable individually too).
EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    "") ;;
    ONLY=*) ONLY="${1#ONLY=}" ;;
    --only) [ $# -ge 2 ] || { echo "ERROR: --only needs a block name" >&2; exit 2; }; ONLY="$2" ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/lib/common.sh
source "$ROOT/scripts/lib/common.sh"
add_rocm_path
require_venv
require_model

VRAM_MODE="${VRAM_MODE:-balanced}"
RESULTS="$ROOT/docs/results"
LOGS="$RESULTS/logs"
RECEIPTS="$RESULTS/validation"
mkdir -p "$LOGS" "$RECEIPTS" "$OUT_DIR/validation"
cd "$ROOT"
UP="$ROOT/third_party/SenseNova-U1"

# ---- device-level VRAM sampler (captures torch + MIOpen + everything) ----
VRAM_LOG=""
start_vram_sampler() {
    VRAM_LOG="$(mktemp /tmp/vram-sample.XXXXXX)"
    (
        while true; do
            rocm-smi --showmeminfo vram 2>/dev/null \
                | awk '/VRAM Total Used Memory/ {print $NF; exit}' >> "$VRAM_LOG"
            sleep 2
        done
    ) &
    VRAM_SAMPLER_PID=$!
}
stop_vram_sampler() {
    kill "$VRAM_SAMPLER_PID" 2>/dev/null || true
    wait "$VRAM_SAMPLER_PID" 2>/dev/null || true
}
peak_vram_bytes() {
    [ -s "$VRAM_LOG" ] && sort -n "$VRAM_LOG" | tail -1 || echo 0
}

# Wrap one task run: sampler + wall clock + receipt + log capture.
# GROUP (optional) lets ONLY=<group> select a set of related blocks.
# run_block <name> <cmd...>
run_block() {
    local name="$1"; shift
    if [ -n "${ONLY:-}" ] && [ "$ONLY" != "$name" ] && [ "$ONLY" != "${GROUP:-}" ]; then
        log "skipping block '$name' (ONLY=${ONLY})"
        return 0
    fi
    log "=== block: $name ==="
    local log_file="$LOGS/$name.log"
    local t0 t1 rc
    local artifact_args=()
    for a in ${BLOCK_ARTIFACTS:-}; do artifact_args+=("sha256:$a"); done
    BLOCK_ARTIFACTS=""
    start_vram_sampler
    t0=$(date +%s.%N)
    set +e
    "$@" 2>&1 | tee "$log_file"
    rc=${PIPESTATUS[0]}
    set -e
    t1=$(date +%s.%N)
    stop_vram_sampler
    local wall peak
    wall=$("$PY" -c "print(f'{$t1 - $t0:.1f}')")
    peak=$(peak_vram_bytes)
    "$ROOT/scripts/receipt.py" "$RECEIPTS/$name.json" \
        "block=$name" \
        "wall_seconds=$wall" \
        "peak_vram_bytes=$peak" \
        "peak_vram_gib=$("$PY" -c "print(round($peak / 2**30, 2))")" \
        "vram_mode=$VRAM_MODE" \
        "command=$(printf '%q ' "$@")" \
        "log=logs/$name.log" \
        ${artifact_args[@]+"${artifact_args[@]}"} \
        "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ -d "@$t0")"
    if [ "$rc" -ne 0 ]; then
        log "block '$name' FAILED (rc=$rc) — see $log_file"
        exit "$rc"
    fi
    log "block '$name' ok: wall=${wall}s peak_vram=$(python3 -c "print(round($peak/2**30,1))") GiB"
}

T2I_ARGS=(--width 2048 --height 2048 --cfg_scale 4.0 --cfg_norm none
          --timestep_shift 3.0 --seed 42 --profile)

# ---- 1. VQA (visual understanding) ----
run_block vqa bash "$ROOT/scripts/run-task.sh" vqa \
    --image "$UP/examples/vqa/data/images/menu.jpg" \
    --question "List every dish you can read on this menu with its price. Answer as a plain list." \
    --max_new_tokens 768 \
    --output "$OUT_DIR/validation/vqa-menu.txt"

# ---- 2. T2I 2048x2048 @ 50 steps ----
BLOCK_ARTIFACTS="$OUT_DIR/validation/t2i-2048.png" \
run_block t2i bash "$ROOT/scripts/run-task.sh" t2i \
    --prompt "A cinematic mountain village at sunrise, golden light over slate roofs, drifting mist between timber houses, ultra detailed" \
    "${T2I_ARGS[@]}" --num_steps 50 \
    --output "$OUT_DIR/validation/t2i-2048.png"

# ---- 3. T2I + reasoning mode ----
BLOCK_ARTIFACTS="$OUT_DIR/validation/t2i-think.png $OUT_DIR/validation/t2i-think.think.txt" \
run_block t2i-think bash "$ROOT/scripts/run-task.sh" t2i \
    --prompt "A male peacock trying to attract a female" \
    "${T2I_ARGS[@]}" --num_steps 50 --think --print_think \
    --output "$OUT_DIR/validation/t2i-think.png"

# ---- 4. Image editing ----
BLOCK_ARTIFACTS="$OUT_DIR/validation/edit-jacket.png" \
run_block edit bash "$ROOT/scripts/run-task.sh" edit \
    --image "$UP/examples/editing/data/images/1.webp" \
    --prompt "Change the jacket of the person on the left to bright yellow." \
    --cfg_scale 4.0 --img_cfg_scale 1.0 --cfg_norm none --timestep_shift 3.0 \
    --num_steps 50 --seed 42 --profile \
    --output "$OUT_DIR/validation/edit-jacket.png"

# ---- 5. Interleave ----
run_block interleave bash "$ROOT/scripts/run-task.sh" interleave \
    --prompt "I want to learn how to cook tomato and egg stir-fry. Please give me a beginner-friendly illustrated tutorial." \
    --resolution "16:9" --num_steps 50 --seed 42 \
    --output_dir "$OUT_DIR/validation/interleave" --stem tutorial

# ---- 6. Determinism: same seed twice -> byte-identical PNG ----
if [ -z "${ONLY:-}" ] || [ "$ONLY" = determinism ]; then
    log "=== block: determinism ==="
    for run in a b; do
        bash "$ROOT/scripts/run-task.sh" t2i \
            --prompt "A red paper lantern hanging in a snowy street at dusk" \
            "${T2I_ARGS[@]}" --num_steps 20 \
            --output "$OUT_DIR/validation/determinism-$run.png" \
            > "$LOGS/determinism-$run.log" 2>&1
    done
    "$ROOT/scripts/receipt.py" "$RECEIPTS/determinism.json" \
        "sha256:$OUT_DIR/validation/determinism-a.png" \
        "sha256:$OUT_DIR/validation/determinism-b.png" \
        "identical=$("$PY" - <<'PYEOF'
import hashlib, sys
def h(p):
    d = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            d.update(c)
    return d.hexdigest()
a = h("outputs/validation/determinism-a.png")
b = h("outputs/validation/determinism-b.png")
print("true" if a == b else "false")
PYEOF
)"
    grep -o '"identical": "[a-z]*"' "$RECEIPTS/determinism.json" | head -1 || true
fi

# ---- 7. VRAM mode comparison (10 steps, per-step latency focus) ----
if [ -z "${ONLY:-}" ] || [ "$ONLY" = vram-modes ] || [[ "$ONLY" == vram-mode-* ]]; then
    log "=== block: vram-modes (10-step probe) ==="
    for mode in balanced fast low; do
        GROUP=vram-modes VRAM_MODE="$mode" run_block "vram-mode-$mode" bash "$ROOT/scripts/run-task.sh" t2i \
            --prompt "A red paper lantern hanging in a snowy street at dusk" \
            "${T2I_ARGS[@]}" --num_steps 10 \
            --output "$OUT_DIR/validation/vrammode-$mode.png"
    done
fi

log "validation suite complete — receipts in $RECEIPTS"
