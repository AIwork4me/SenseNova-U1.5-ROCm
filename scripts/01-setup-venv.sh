#!/usr/bin/env bash
# 01-setup-venv.sh — build the Python environment for SenseNova-U1.5 on ROCm.
#
# Creates .venv/ inside the project and installs:
#   - PyTorch 2.8.0 + torchvision 0.23.0 from the ROCm wheel index
#     (gfx1100-class GPUs are natively supported by these wheels)
#   - the upstream inference stack (transformers, accelerate, ...) at
#     versions compatible with the pinned SenseNova-U1 checkout
#   - the `sensenova_u1` package itself (editable, --no-deps) from
#     third_party/SenseNova-U1 @ 76c32c2
#
# Ends with a GPU smoke test: torch must see the AMD GPU and run a matmul.
#
# Env knobs:
#   PY_INDEX_ROCM   torch wheel index (default: rocm6.3 — matches torch 2.8.0)
#   SKIP_SMOKE=1    skip the GPU smoke test
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/01-setup-venv.sh

Creates .venv/ with ROCm PyTorch 2.8.0 and the SenseNova-U1 inference stack,
then verifies torch can see and compute on the AMD GPU.
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
add_rocm_path

PY_INDEX_ROCM="${PY_INDEX_ROCM:-https://download.pytorch.org/whl/rocm6.3}"
UPSTREAM="$ROOT/third_party/SenseNova-U1"

[ -d "$UPSTREAM" ] || die "upstream checkout missing at $UPSTREAM
  run: git clone https://github.com/OpenSenseNova/SenseNova-U1.git $UPSTREAM
       git -C $UPSTREAM checkout $UPSTREAM_PINNED_COMMIT"

log "creating virtualenv at $VENV"
[ -d "$VENV" ] || python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Upgrade pip quietly first so the extra index parses cleanly.
pip install -q --upgrade pip >/dev/null

log "installing ROCm PyTorch (this downloads ~3 GiB of wheels)"
pip install "torch==2.8.0" "torchvision==0.23.0" \
    --index-url "$PY_INDEX_ROCM"

log "installing the SenseNova-U1 inference stack"
# Mirror of upstream pyproject [project].dependencies (cu128-specific index
# stripped), with transformers pinned to the line validated on ROCm 7.2.1.
pip install \
    "transformers==4.57.1" \
    "accelerate>=1.1,<2" \
    "huggingface-hub>=0.34,<2" \
    "safetensors>=0.4.3,<1" \
    "sentencepiece==0.2.1" \
    "numpy>=1.24,<3" \
    "pillow>=10,<13" \
    "tqdm==4.67.1" \
    "packaging>=24" \
    "httpx>=0.27,<1" \
    "modelscope>=1.39"

log "installing sensenova_u1 (editable, no deps — deps installed above)"
upstream_commit="$(git -C "$UPSTREAM" rev-parse --short=7 HEAD)"
[ "$upstream_commit" = "$UPSTREAM_PINNED_COMMIT" ] \
    || warn "upstream HEAD is $upstream_commit, project validated @ $UPSTREAM_PINNED_COMMIT"
pip install -e "$UPSTREAM" --no-deps

log "writing environment fingerprint"
mkdir -p "$ROOT/docs/results"
"$PY" - <<'PYEOF' > "$ROOT/docs/results/environment.json"
import json, subprocess, sys, torch, transformers, accelerate
info = {
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "torch_backend": "hip" if torch.version.hip else "cuda",
    "torch_hip_version": torch.version.hip,
    "transformers": transformers.__version__,
    "accelerate": accelerate.__version__,
    "gpu_available": torch.cuda.is_available(),
    "gpu_count": torch.cuda.device_count(),
}
if torch.cuda.is_available():
    info["gpu_name"] = torch.cuda.get_device_name(0)
    info["gpu_arch"] = torch.cuda.get_device_capability(0)
    info["gpu_total_vram_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
try:
    info["rocm_smi"] = subprocess.run(
        ["rocm-smi", "--showproductname", "--showid"],
        capture_output=True, text=True, timeout=10).stdout.strip()
except Exception:
    pass
print(json.dumps(info, indent=2))
PYEOF
cat "$ROOT/docs/results/environment.json"

if [ "${SKIP_SMOKE:-0}" != "1" ]; then
    log "GPU smoke test (bf16 matmul on device 0)"
    "$PY" - <<'PYEOF'
import torch
assert torch.cuda.is_available(), "torch cannot see the GPU — check ROCm install / /dev/kfd"
x = torch.randn(2048, 2048, dtype=torch.bfloat16, device="cuda")
y = (x @ x.T).float().mean().item()
print(f"smoke ok: mean={y:.4f} on {torch.cuda.get_device_name(0)}")
PYEOF
fi

log "done. Next: bash scripts/02-fetch-model.sh"
