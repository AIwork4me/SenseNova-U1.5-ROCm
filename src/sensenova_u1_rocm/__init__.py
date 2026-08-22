"""ROCm runtime fixes for SenseNova-U1.5 on gfx1100-class GPUs.

Importing this package applies (and records) the workarounds measured on the
reference host — see docs/results/findings/rocm63-wheel-blas-on-gfx1100.md:

1. Nothing here replaces libraries; the shell layer (scripts/lib/common.sh)
   preloads the system ROCm hipBLAS/rocBLAS over the torch wheel's broken
   copies and points rocBLAS at the system Tensile kernels.
2. This module disables the cuDNN/MIOpen conv path
   (``torch.backends.cudnn.enabled = False``) because the wheel's MIOpen
   JIT-compiles kernels through a comgr that segfaults mid-model; with
   MIOpen off, convolutions run as unfold+GEMM through the (now healthy)
   rocBLAS GEMMs.

Set SENU15_MIOPEN=1 to keep MIOpen enabled (e.g. on hosts where the wheel's
JIT path works) and SENU15_ROCM_FIXES=0 to disable this module entirely.
"""
import os

_fixes_enabled = os.environ.get("SENU15_ROCM_FIXES", "1") == "1"
_miopen_forced_on = os.environ.get("SENU15_MIOPEN", "0") == "1"


def apply() -> bool:
    """Apply the ROCm fixes. Returns True when the MIOpen bypass is active."""
    if not _fixes_enabled or _miopen_forced_on:
        return False
    try:
        import torch
    except ImportError:
        return False
    if torch.version.hip is None:
        return False  # CUDA hosts don't need this
    torch.backends.cudnn.enabled = False
    return True


_bypass_active = apply()
