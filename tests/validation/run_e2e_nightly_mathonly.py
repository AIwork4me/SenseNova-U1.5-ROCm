#!/usr/bin/env python3
"""DIAGNOSTIC launcher: nightly stack with fused SDPA backends DISABLED
(forces the MATH fallback everywhere, replicating the semantics of the
historical patch 0002 without touching model code). Used ONLY to bisect
the nightly t2i corruption: if MATH-only output is a coherent image, the
fault is in a fused SDPA kernel configuration; if it is still garbage,
the fault lies elsewhere in the nightly build. Not a shipped workaround.
"""
import os
import runpy
import sys

import torch

print("=== E2E diagnostic context (nightly rocm7.14, MATH-only SDPA) ===")
print("torch:", torch.__version__, "| hip:", torch.version.hip)
print("LD_PRELOAD:", os.environ.get("LD_PRELOAD", "<unset>"))
print("ROCBLAS_TENSILE_LIBPATH:", os.environ.get("ROCBLAS_TENSILE_LIBPATH", "<unset>"))
print("MIOpen/cuDNN backend enabled:", torch.backends.cudnn.enabled)
assert torch.backends.cudnn.enabled

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
print("flash_sdp:", torch.backends.cuda.flash_sdp_enabled())
print("mem_efficient_sdp:", torch.backends.cuda.mem_efficient_sdp_enabled())
print("math_sdp:", torch.backends.cuda.math_sdp_enabled())
print("===================================================================")

if len(sys.argv) < 2:
    print("usage: run_e2e_nightly_mathonly.py <script.py> [args...]", file=sys.stderr)
    sys.exit(2)
target, *rest = sys.argv[1:]
sys.argv = [target, *rest]
runpy.run_path(target, run_name="__main__")
