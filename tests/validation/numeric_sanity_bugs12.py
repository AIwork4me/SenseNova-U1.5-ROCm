#!/usr/bin/env python3
"""Numerical sanity (Phase 11) — GPU bf16 vs CPU fp32 reference for Bugs 1-2.

Goal: evidence that outputs are numerically reasonable for bf16 (not merely
"did not crash"). Reference computed on CPU in fp32 from the same seeded
weights/inputs; bf16 rounding dominates, so ~1e-2 relative errors are
expected and acceptable; only clearly abnormal errors would fail this check.
"""
import copy

import torch

print(f"torch: {torch.__version__} | hip: {torch.version.hip}")


def stats(name, gpu_bf16, cpu_fp32):
    gpu = gpu_bf16.to("cpu", torch.float32)
    ref = cpu_fp32
    abs_err = (gpu - ref).abs()
    rel_err = abs_err / ref.abs().clamp_min(1e-3)
    print(f"[{name}] shape={tuple(gpu.shape)}")
    print(f"[{name}] max abs err : {abs_err.max().item():.6e}")
    print(f"[{name}] mean abs err: {abs_err.mean().item():.6e}")
    print(f"[{name}] median abs err (elementwise over all): {abs_err.median().item():.6e}")
    print(f"[{name}] median rel err (|ref|>=1e-3 denominator clamp): {rel_err.median().item():.6e}")


torch.manual_seed(0)

# --- Bug 1 shape: Conv2d(3, 768, 16, 16) on (16, 3, 16, 16) ---
conv = torch.nn.Conv2d(3, 768, 16, 16).requires_grad_(False)
conv_ref = copy.deepcopy(conv)          # CPU fp32 reference module
x = torch.randn(16, 3, 16, 16)
conv_gpu = conv.to("cuda", torch.bfloat16)
x_gpu = x.to("cuda", torch.bfloat16)
with torch.no_grad():
    out_gpu = conv_gpu(x_gpu)
torch.cuda.synchronize()
with torch.no_grad():
    out_ref = conv_ref(x)               # CPU fp32 reference
stats("bug1 conv", out_gpu, out_ref)

# --- Bug 2 shape: Linear(4096, 4096, bias=True) on (16060, 4096) ---
lin = torch.nn.Linear(4096, 4096, bias=True).requires_grad_(False)
lin_ref = copy.deepcopy(lin)            # CPU fp32 reference module
x2 = torch.randn(16060, 4096)
lin_gpu = lin.to("cuda", torch.bfloat16)
x2_gpu = x2.to("cuda", torch.bfloat16)
with torch.no_grad():
    out2_gpu = lin_gpu(x2_gpu)
torch.cuda.synchronize()
with torch.no_grad():                             # CPU fp32 reference, chunked
    outs = []
    for i in range(0, 16060, 2048):
        outs.append(lin_ref(x2[i : i + 2048]))
    out2_ref = torch.cat(outs, dim=0)
stats("bug2 gemm", out2_gpu, out2_ref)

print("NUMERIC-SANITY-DONE")
