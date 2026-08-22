#!/usr/bin/env bash
# 00-check-env.sh — verify the host can run SenseNova-U1.5-8B-MoT on ROCm.
#
# Checks: python3, git, curl, ROCm userspace + /dev/kfd access, an AMD GPU,
# free disk for the venv (~10 GiB) and the 50.2 GiB checkpoint destination,
# and host RAM for the layer-offload path (>= 64 GiB recommended; the
# bf16 checkpoint alone is ~47 GiB and `--vram_mode balanced` streams
# layers from host memory).
#
# Exits non-zero when a hard requirement is missing. Warnings never fail.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/00-check-env.sh

Verifies the host environment for SenseNova-U1.5-ROCm:
  - host tools (python3 >= 3.10, git, curl)
  - ROCm userspace (any 6.x / 7.x; reference: 7.2.1) and /dev/kfd access
  - an AMD GPU is present (reported, not gated: any gfx arch works,
    this project was validated on gfx1100 / 48 GB)
  - disk space: ~10 GiB for the venv, ~51 GiB for the checkpoint
  - host RAM >= 64 GiB recommended for the layer-offload path
EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    "") ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/lib/common.sh
source "$ROOT/scripts/lib/common.sh"

fail_count=0
hard_fail() { echo "FAIL: $1" >&2; fail_count=$((fail_count + 1)); }
ok()   { echo "  ok: $*"; }
note() { echo "  ..: $*"; }

echo "== host tools =="
command -v python3 >/dev/null 2>&1 || hard_fail "python3 not found"
pyver="$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo 0)"
if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"; then
    ok "python3 $pyver"
else
    hard_fail "python3 >= 3.10 required (found $pyver)"
fi
command -v git  >/dev/null 2>&1 && ok "git $(git --version | awk '{print $3}')"  || hard_fail "git not found"
command -v curl >/dev/null 2>&1 && ok "curl $(curl --version | awk '{print $2}' | head -1)" || hard_fail "curl not found"

echo "== ROCm =="
ROCM_PATH=""
for p in /opt/rocm /usr/local/rocm "${ROCM_PATH:-}"; do
    [ -n "$p" ] && [ -x "$p/bin/rocminfo" ] && ROCM_PATH="$p" && break
done
if [ -n "$ROCM_PATH" ]; then
    add_rocm_path
    ver="$(cat "$ROCM_PATH/.info/version" 2>/dev/null || echo unknown)"
    ok "ROCm userspace at $ROCM_PATH (version $ver)"
else
    hard_fail "ROCm not found (looked in /opt/rocm, /usr/local/rocm)"
fi
[ -e /dev/kfd ] && ok "/dev/kfd present" || hard_fail "/dev/kfd missing (ROCm compute node)"
[ -w /dev/kfd ] && ok "/dev/kfd writable" || warn "/dev/kfd not writable — GPU compute may fail (video/render group?)"

echo "== GPU =="
if command -v rocminfo >/dev/null 2>&1; then
    gfx="$(rocminfo 2>/dev/null | awk '/^ *Marketing Name/ {m=$0} /^ *Name: *gfx/ {gsub(/ /,"",$2); print $2}' | head -1)"
    if [ -n "$gfx" ]; then
        ok "AMD GPU arch: $gfx"
        if [ "$gfx" = "gfx1100" ]; then
            note "reference platform (Radeon W7800/W7900-class, 48 GB)"
        else
            note "not the reference arch (gfx1100) — YMMV, we welcome evidence"
        fi
    else
        warn "rocminfo present but no gfx agent found"
    fi
else
    warn "rocminfo not on PATH — cannot enumerate GPU arch"
fi
if command -v rocm-smi >/dev/null 2>&1; then
    vram_bytes="$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/VRAM Total Memory/ {print $(NF)}' | head -1)"
    if [ -n "$vram_bytes" ] && [ "$vram_bytes" -gt 0 ] 2>/dev/null; then
        vram_gib=$(( vram_bytes / 1073741824 ))
        ok "VRAM: ${vram_gib} GiB"
        if [ "$vram_gib" -lt 32 ]; then
            warn "less than 32 GiB VRAM — expect heavy offload (slow image generation)"
        fi
    fi
fi

echo "== disk =="
free_workspace="$(df -Pk "$ROOT" 2>/dev/null | awk 'NR==2 {print $4}')"
[ -n "$free_workspace" ] && [ "$free_workspace" -gt $(( 12 * 1024 * 1024 )) ] \
    && ok "~$(( free_workspace / 1024 / 1024 )) GiB free under project root (venv needs ~10 GiB)" \
    || warn "low disk under project root (venv needs ~10 GiB)"
mkdir -p "$MODEL_BASE" 2>/dev/null || true
free_model="$(df -Pk "$MODEL_BASE" 2>/dev/null | awk 'NR==2 {print $4}')"
[ -n "$free_model" ] && [ "$free_model" -gt $(( 52 * 1024 * 1024 )) ] \
    && ok "~$(( free_model / 1024 / 1024 )) GiB free at $MODEL_BASE (checkpoint needs ~51 GiB)" \
    || warn "low disk at $MODEL_BASE — checkpoint needs ~51 GiB (override with MODEL_BASE=...)"

echo "== host RAM (layer-offload path) =="
ram_kib="$(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
ram_gib=$(( ram_kib / 1024 / 1024 ))
if   [ "$ram_gib" -ge 64 ]; then ok "${ram_gib} GiB RAM"
elif [ "$ram_gib" -ge 48 ]; then warn "${ram_gib} GiB RAM — workable, but >= 64 GiB recommended for --vram_mode balanced"
else hard_fail "only ${ram_gib} GiB RAM — the offload path needs the ~47 GiB bf16 checkpoint resident in host memory"; fi

echo
if [ "$fail_count" -gt 0 ]; then
    echo "RESULT: $fail_count hard failure(s). Fix the items above, then re-run."
    exit 1
fi
echo "RESULT: environment looks good. Next: bash scripts/01-setup-venv.sh"
