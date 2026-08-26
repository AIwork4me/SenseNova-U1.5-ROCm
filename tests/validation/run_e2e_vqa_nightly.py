#!/usr/bin/env python3
"""E2E launcher (Phase 10) — print the no-workaround validation context, then
run the upstream SenseNova-U1 VQA example exactly as run-task.sh would
(same flags), but under the PyTorch nightly ROCm 7.14 interpreter.

Usage:
  SENU15_MIOPEN=1 PYTHONPATH=<nightly-site-packages>:<repo-.venv-site-packages>:<upstream-src> \
    <nightly-python> run_e2e_vqa_nightly.py <upstream example>/inference.py [flags...]

SENU15_MIOPEN=1 opts OUT of this repo's cudnn/MIOpen bypass so convolutions
run through MIOpen (the historical workaround is disabled, not engaged).
"""
import os
import runpy
import sys

import torch

print("=== E2E validation context (pytorch nightly rocm7.14, no workarounds) ===")
print("torch:", torch.__version__)
print("torch.version.hip:", torch.version.hip)
print("torch.__file__:", torch.__file__)
print("device:", torch.cuda.get_device_name(0))
print("device capability:", torch.cuda.get_device_capability(0))
print("MIOpen/cuDNN backend enabled:", torch.backends.cudnn.enabled)
print("LD_PRELOAD:", os.environ.get("LD_PRELOAD", "<unset>"))
print("ROCBLAS_TENSILE_LIBPATH:", os.environ.get("ROCBLAS_TENSILE_LIBPATH", "<unset>"))
assert torch.backends.cudnn.enabled, "E2E must run with MIOpen enabled"
assert not os.environ.get("LD_PRELOAD"), "LD_PRELOAD must be unset"
assert not os.environ.get("ROCBLAS_TENSILE_LIBPATH"), "ROCBLAS_TENSILE_LIBPATH must be unset"
print("========================================================================")

if len(sys.argv) < 2:
    print("usage: run_e2e_vqa_nightly.py <script.py> [args...]", file=sys.stderr)
    sys.exit(2)
target, *rest = sys.argv[1:]
sys.argv = [target, *rest]
runpy.run_path(target, run_name="__main__")
