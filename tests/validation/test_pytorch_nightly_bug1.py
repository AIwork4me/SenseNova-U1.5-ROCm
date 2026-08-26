#!/usr/bin/env python3
"""Bug 1 exact repro (pytorch/pytorch#194447) — bf16 Conv2d.

Original failure on the torch 2.8.0+rocm6.3 wheel: SIGSEGV in
Tensile::PlaceholderLibrary::loadPlaceholderLibrary() inside the wheel's
bundled librocblas.so on the first bf16 conv (the SenseNova-U1.5 ViT
patch-embedding shape family).

Exact input shape from the original report: Conv2d(3, 768, 16, 16) on
(16, 3, 16, 16) bf16. MIOpen must stay enabled; no LD_PRELOAD, no
ROCBLAS_TENSILE_LIBPATH, no backend disabling.
"""
import sys

import torch

print(f"torch: {torch.__version__} | hip: {torch.version.hip}")
print(f"device: {torch.cuda.get_device_name(0)}")
assert torch.backends.cudnn.enabled, "MIOpen/cuDNN backend must remain enabled"
print(f"cudnn (MIOpen) enabled: {torch.backends.cudnn.enabled}")

torch.manual_seed(0)

conv = (
    torch.nn.Conv2d(3, 768, 16, 16)
    .to("cuda", torch.bfloat16)
    .requires_grad_(False)
)

x = torch.randn(
    16, 3, 16, 16,
    device="cuda",
    dtype=torch.bfloat16,
)

with torch.no_grad():
    out = conv(x)

torch.cuda.synchronize()

print(out.shape)
print("finite:", torch.isfinite(out).all().item())
assert out.shape == (16, 768, 1, 1), f"unexpected output shape {out.shape}"
assert torch.isfinite(out).all().item(), "non-finite output values"
print("PASS")
sys.exit(0)
