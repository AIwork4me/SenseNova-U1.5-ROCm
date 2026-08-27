#!/usr/bin/env python3
"""Bug 2 exact repro (pytorch/pytorch#194447) — large bf16 GEMM.

Original failure on the torch 2.8.0+rocm6.3 wheel (with only
ROCBLAS_TENSILE_LIBPATH redirected to system kernels):
  "Unbundle Objects Error: Failed to decompress ... Unknown frame descriptor"
  RuntimeError: HIPBLAS_STATUS_INTERNAL_ERROR
on the first LLM q_proj GEMM: Linear(4096 -> 4096, bias=True) on
(16060, 4096) bf16. 16060 = number of ViT patches for menu.jpg
(smart-resized to 1760x2336, patch 16 -> grid 110x146).

No LD_PRELOAD, no ROCBLAS_TENSILE_LIBPATH, no backend disabling.
"""
import sys

import pytest

pytest.importorskip("torch", reason="GPU repro script — run on a torch host")

import torch

print(f"torch: {torch.__version__} | hip: {torch.version.hip}")
print(f"device: {torch.cuda.get_device_name(0)}")

torch.manual_seed(0)

lin = torch.nn.Linear(
    4096,
    4096,
    bias=True,
).to("cuda", torch.bfloat16).requires_grad_(False)

x = torch.randn(
    16060,
    4096,
    device="cuda",
    dtype=torch.bfloat16,
)

with torch.no_grad():
    out = lin(x)

torch.cuda.synchronize()

print(out.shape)
print("finite:", torch.isfinite(out).all().item())
assert out.shape == (16060, 4096), f"unexpected output shape {out.shape}"
assert torch.isfinite(out).all().item(), "non-finite output values"
print("PASS")
sys.exit(0)
