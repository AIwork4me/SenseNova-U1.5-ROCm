#!/usr/bin/env bash
# Shared helpers for SenseNova-U1.5-ROCm scripts.
# Sourced, not executed.

PROJECT_NAME="SenseNova-U1.5-ROCm"
# shellcheck disable=SC2034  # consumed by scripts that source this file
MODEL_REPO="SenseNova/SenseNova-U1.5-8B-MoT"
MODEL_DIR_NAME="SenseNova-U1.5-8B-MoT"
# shellcheck disable=SC2034  # consumed by scripts that source this file
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

# --- ROCm math-library workaround (measured on the reference host) ---------
# The PyTorch rocm6.3 wheel's bundled BLAS stack is broken on gfx1100:
#   (a) rocBLAS segfaults in Tensile::PlaceholderLibrary::loadPlaceholderLibrary()
#       on any half-precision GEMM routed through it (repro: any bf16 Conv2d);
#   (b) even redirected to the system kernel set, its 6.3-era decompressor
#       cannot read ROCm 7.x Tensile .dat files ("Unbundle Objects Error:
#       ... Unknown frame descriptor") and large bf16 GEMMs (the first LLM
#       q_proj) fail with HIPBLAS_STATUS_INTERNAL_ERROR.
# Fix: route ALL BLAS calls through the system ROCm install by preloading
# its hipBLAS + rocBLAS (symbol interposition beats the wheel's bundled
# copies) and pointing rocBLAS at the system Tensile kernels. Verified
# numerically: bf16 conv/GEMM outputs match CPU fp32 references at bf16
# rounding level. See docs/results/findings/rocm6.3-wheel-blas-on-gfx1100.md
# Override: set BLAS_FIX=0 to disable (e.g. on CUDA hosts or a future fixed
# wheel), or point ROCM_HOME at a different ROCm install.
if [ "${BLAS_FIX:-1}" = "1" ]; then
    _rocm_home="${ROCM_HOME:-}"
    if [ -z "$_rocm_home" ]; then
        for _p in /opt/rocm /usr/local/rocm; do
            [ -e "$_p/lib/librocblas.so" ] && _rocm_home="$_p" && break
        done
    fi
    if [ -n "$_rocm_home" ] && [ -e "$_rocm_home/lib/librocblas.so" ]; then
        case ":${LD_PRELOAD:-}:" in
            *":$_rocm_home/lib/librocblas.so:"*) ;;
            *) export LD_PRELOAD="$_rocm_home/lib/libhipblas.so:$_rocm_home/lib/librocblas.so${LD_PRELOAD:+:$LD_PRELOAD}" ;;
        esac
        if [ -z "${ROCBLAS_TENSILE_LIBPATH:-}" ] && [ -d "$_rocm_home/lib/rocblas/library" ]; then
            export ROCBLAS_TENSILE_LIBPATH="$_rocm_home/lib/rocblas/library"
        fi
    fi
fi

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
