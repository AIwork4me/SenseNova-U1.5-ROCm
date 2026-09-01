#!/usr/bin/env bash
# Shared helpers for SenseNova-U1.5-ROCm scripts.
# Sourced, not executed.

PROJECT_NAME="SenseNova-U1.5-ROCm"

log()  { echo "[$PROJECT_NAME] $*"; }
warn() { echo "[$PROJECT_NAME] WARNING: $*" >&2; }
die()  { echo "[$PROJECT_NAME] FAIL: $*" >&2; exit 1; }
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
# The PyTorch rocm6.3 wheel's bundled math stack is broken on gfx1100:
#   (a) rocBLAS segfaults in Tensile::PlaceholderLibrary::loadPlaceholderLibrary()
#       on any half-precision GEMM routed through it (repro: any bf16 Conv2d);
#   (b) even redirected to the system kernel set, its 6.3-era decompressor
#       cannot read ROCm 7.x Tensile .dat files ("Unbundle Objects Error:
#       ... Unknown frame descriptor") and large bf16 GEMMs (the first LLM
#       q_proj) fail with HIPBLAS_STATUS_INTERNAL_ERROR;
#   (c) the wheel's MIOpen JIT-compiles kernels through a comgr that
#       segfaults mid-model (ViT dense_embedding).
# Fix (verified 2026-08-22/23; see docs/results/findings/):
#   - BLAS mode: preload the system ROCm's hipBLAS+rocBLAS and point rocBLAS
#     at the system Tensile kernels; bypass MIOpen convs (cudnn off) due to (c).
#   - FULL-STACK mode (auto when a ROCm >= 7.14 install is found, e.g. from
#     scripts/install-rocm-7.14-gfx110x.sh, or — default since the torch
#     2.12.0+rocm7.14.0 migration — the ROCm 7.14 wheel SDK pip installs
#     into the venv under site-packages/_rocm_sdk_*): preload that stack's
#     MIOpen+comgr+hipBLAS+rocBLAS — bug (c) is fixed in 7.14's comgr, so
#     MIOpen convs stay ENABLED (no unfold+GEMM penalty). Verified: the model
#     runs end-to-end with cudnn on; numerics identical at bf16 level.
# Knobs:
#   ROCM_FULL_STACK=auto|1|0   default auto — engages when a valid 7.14 stack
#                              is found (prefix dir or venv wheel SDK);
#                              0 forces BLAS mode; 1 requires it
#   ROCM714_PREFIX=/path       explicit full-stack prefix
#   BLAS_FIX=0                 disable the workaround entirely (CUDA hosts /
#                              a future fixed wheel)
_gpu_arch_cached() {
    # Detect the host GPU arch once (rocminfo), cache in $TMPDIR. Prints e.g.
    # gfx1100. Returns 1 when undetectable (caller falls back to any-gfx).
    local cache arch
    if [ -n "${GPU_ARCH:-}" ]; then printf '%s' "$GPU_ARCH"; return 0; fi
    cache="${TMPDIR:-/tmp}/senu15-gpu-arch"
    if [ -s "$cache" ]; then printf '%s' "$(cat "$cache")"; return 0; fi
    arch=""
    if command -v rocminfo >/dev/null 2>&1; then
        arch="$(rocminfo 2>/dev/null | awk '/^ *Name: *gfx/ {gsub(/ /,"",$2); print $2; exit}')"
    fi
    if [ -n "$arch" ]; then
        printf '%s' "$arch" > "$cache"
        printf '%s' "$arch"
        return 0
    fi
    return 1
}

_rocm_prefix_valid() {
    # $1 = prefix: needs the full preload set + Tensile kernels for THIS
    # GPU arch (a gfx1151-only dist must not be picked on a gfx1100 host).
    [ -e "$1/lib/libMIOpen.so" ] && [ -e "$1/lib/libamd_comgr.so" ] \
        && [ -e "$1/lib/libhipblas.so" ] && [ -e "$1/lib/librocblas.so" ] || return 1
    local arch
    if arch="$(_gpu_arch_cached)"; then
        ls "$1"/lib/rocblas/library/*"${arch}"*.dat >/dev/null 2>&1
    else
        # arch undetectable: accept any gfx kernels (best effort)
        ls "$1"/lib/rocblas/library/*gfx*.dat >/dev/null 2>&1
    fi
}

_find_fullstack_prefix() {
    local p
    # An explicit ROCM714_PREFIX is authoritative: use it or fail (no
    # silent fallback to wildcard candidates).
    if [ -n "${ROCM714_PREFIX:-}" ]; then
        if _rocm_prefix_valid "$ROCM714_PREFIX"; then
            printf '%s' "$ROCM714_PREFIX"
            return 0
        fi
        return 1
    fi
    for p in "$HOME"/rocm-7.14* /root/rocm-7.14* /opt/rocm-7.14*; do
        [ -n "$p" ] || continue
        if _rocm_prefix_valid "$p"; then
            printf '%s' "$p"
            return 0
        fi
    done
    return 1
}

_sdk_dso() {
    # $1 = dir, $2 = DSO basename without extension: prints the first of
    # "$1/$2.so", "$1/$2.so.<N>" that exists. Wheel SDKs ship versioned
    # sonames only (libMIOpen.so.1, librocblas.so.5, ...).
    local f
    for f in "$1/$2.so" "$1/$2".so.*; do
        [ -e "$f" ] && { printf '%s' "$f"; return 0; }
    done
    return 1
}

_setup_wheel_fullstack() {
    # The torch 2.12.0+rocm7.14.0 wheels install a complete ROCm 7.14 SDK
    # into the venv: core runtime + comgr under _rocm_sdk_core/lib, the math
    # stack (MIOpen, hipBLAS, rocBLAS, Tensile kernels) under
    # _rocm_sdk_libraries/lib. Sets _fs_preload/_fs_tensile from it and
    # returns 0; returns 1 when the venv carries no complete wheel SDK.
    # This must engage INSTEAD of the BLAS fallback below: preloading the
    # host's older hipBLAS/rocBLAS drags in its libhsa-runtime64, which
    # shadows the wheel's and makes import torch die with "undefined symbol:
    # hsa_amd_vmem_export_fabric_handle, version ROCR_1" (observed 2026-09-01,
    # host ROCm 7.2.1 vs wheel 7.14).
    local sp core math miopen comgr hipblas rocblas
    for sp in "$VENV"/lib/python*/site-packages; do
        core="$sp/_rocm_sdk_core/lib"
        math="$sp/_rocm_sdk_libraries/lib"
        [ -d "$core" ] && [ -d "$math" ] || continue
        miopen="$(_sdk_dso "$math" libMIOpen)" || continue
        comgr="$(_sdk_dso "$core" libamd_comgr)" || continue
        hipblas="$(_sdk_dso "$math" libhipblas)" || continue
        rocblas="$(_sdk_dso "$math" librocblas)" || continue
        ls "$math"/rocblas/library/*gfx*.dat >/dev/null 2>&1 || continue
        _fs_preload="$miopen:$comgr:$hipblas:$rocblas"
        _fs_tensile="$math/rocblas/library"
        return 0
    done
    return 1
}

if [ "${BLAS_FIX:-1}" = "1" ]; then
    _fs_mode="${ROCM_FULL_STACK:-auto}"
    _fs_prefix=""
    _fs_preload=""
    _fs_tensile=""
    if [ "$_fs_mode" != "0" ]; then
        _fs_prefix="$(_find_fullstack_prefix)" || _fs_prefix=""
        if [ -n "$_fs_prefix" ]; then
            _fs_preload="$_fs_prefix/lib/libMIOpen.so:$_fs_prefix/lib/libamd_comgr.so:$_fs_prefix/lib/libhipblas.so:$_fs_prefix/lib/librocblas.so"
            _fs_tensile="$_fs_prefix/lib/rocblas/library"
        else
            _setup_wheel_fullstack || true
        fi
    fi
    if [ -n "$_fs_preload" ]; then
        case ":${LD_PRELOAD:-}:" in
            *":$_fs_preload:"*) ;;   # already applied (double-sourced)
            *) export LD_PRELOAD="$_fs_preload${LD_PRELOAD:+:$LD_PRELOAD}" ;;
        esac
        export ROCBLAS_TENSILE_LIBPATH="$_fs_tensile"
        export SENU15_MIOPEN=1          # keep MIOpen enabled: 7.14 comgr JIT works
        export ROCM_FULL_STACK_ACTIVE=1
    elif [ "$_fs_mode" = "1" ]; then
        die "ROCM_FULL_STACK=1 but no valid ROCm >=7.14 stack found (looked at \$ROCM714_PREFIX, ~/rocm-7.14*, /root/rocm-7.14*, /opt/rocm-7.14*, and the venv's rocm7.14 wheel SDK) — run scripts/01-setup-venv.sh (torch 2.12 rocm7.14 wheels) or scripts/install-rocm-7.14-gfx110x.sh, or set ROCM_FULL_STACK=0"
    else
        # BLAS mode: system hipBLAS/rocBLAS only; MIOpen bypassed (see (c))
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
fi


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
