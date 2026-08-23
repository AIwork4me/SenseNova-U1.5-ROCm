#!/usr/bin/env bash
# install-rocm-7.14-gfx110x.sh — install AMD's official TheRock ROCm 7.14.0
# gfx110X userspace next to (not over) your system ROCm, for the project's
# full-stack mode (MIOpen enabled; see scripts/lib/common.sh).
#
# Why: the torch rocm6.3 wheel's bundled rocBLAS/comgr are broken on gfx1100
# (three native bugs — docs/results/findings/rocm63-wheel-blas-on-gfx1100.md).
# Preloading this 7.14 stack fixes all three AND restores the MIOpen conv
# path. Size: ~2.31 GB download, ~7 GB extracted.
#
# Usage: bash scripts/install-rocm-7.14-gfx110x.sh [PREFIX]
#   PREFIX defaults to $HOME/rocm-7.14-gfx110x (override with $1 or
#   ROCM714_PREFIX). SHA256-verified, idempotent re-runs.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/install-rocm-7.14-gfx110x.sh [PREFIX]

Downloads AMD's therock-dist-linux-gfx110X-all-7.14.0.tar.gz (repo.amd.com),
verifies size + SHA256, extracts to PREFIX (default ~/rocm-7.14-gfx110x).
Re-runs verify and skip. After installing, scripts automatically engage
full-stack mode (override: ROCM_FULL_STACK=0).
EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    "") ;;
    *) PREFIX="$1" ;;
esac
PREFIX="${PREFIX:-${ROCM714_PREFIX:-$HOME/rocm-7.14-gfx110x}}"

URL="https://repo.amd.com/rocm/tarball-multi-arch/therock-dist-linux-gfx110X-all-7.14.0.tar.gz"
SIZE=2313039793
SHA256="e78a4445c52d879fbd0765f24e7fa9df1e262a8baf681b118a13e75340120127"  # recorded from the verified download used in docs/results/findings/transcripts/rocm714-verification.md
ARCHIVE="${ROCM714_ARCHIVE:-${TMPDIR:-/tmp}/therock-dist-linux-gfx110X-all-7.14.0.tar.gz}"

if [ -e "$PREFIX/lib/libMIOpen.so" ] && ls "$PREFIX"/lib/rocblas/library/*gfx1100*.dat >/dev/null 2>&1; then
    echo "[install-rocm-7.14] $PREFIX already valid; nothing to do."
    exit 0
fi
if [ -e "$PREFIX" ]; then
    echo "ERROR: $PREFIX exists but is incomplete — move it aside and re-run." >&2
    exit 1
fi

echo "[install-rocm-7.14] downloading ~2.31 GB from repo.amd.com ..."
curl -fL --retry 5 --retry-delay 3 -C - -o "$ARCHIVE" "$URL"
actual_size=$(stat -c%s "$ARCHIVE")
[ "$actual_size" = "$SIZE" ] || { echo "ERROR: size mismatch ($actual_size != $SIZE)" >&2; exit 1; }
echo "$SHA256  $ARCHIVE" | sha256sum -c -

mkdir -p "$(dirname "$PREFIX")" "$PREFIX"
echo "[install-rocm-7.14] extracting to $PREFIX ..."
tar -xzf "$ARCHIVE" -C "$PREFIX"
ls "$PREFIX"/lib/rocblas/library/*gfx1100*.dat >/dev/null 2>&1 \
    || { echo "ERROR: no gfx1100 Tensile kernels in $PREFIX — unexpected archive layout" >&2; exit 1; }
echo "[install-rocm-7.14] done. Scripts will auto-engage full-stack mode (ROCM_FULL_STACK=auto)."
