#!/usr/bin/env python3
"""Bug 3 exact repro (pytorch/pytorch#194447) — MIOpen JIT / comgr segfault.

Original failure on the torch 2.8.0+rocm6.3 wheel: full SenseNova-U1.5 VQA
forward crashed with SIGSEGV in COMGR::DataObject::clearData() inside the
wheel's bundled libamd_comgr.so, via hiprtcCompileProgram ->
miopen::hiprtc::BuildHip — the ViT dense_embedding conv has no precompiled
MIOpen kernel and must be JIT-compiled for its dynamic per-image shape.

Exact shape derivation (third_party/SenseNova-U1,
sensenova_u1.models.neo_unify):
  - menu.jpg is 4096x3072 (examples/vqa/data/images/menu.jpg)
  - load_image_native(max_pixels=4194304, patch_size=16, downsample_ratio=0.5)
    smart_resizes with factor=32: h_bar=floor_by_factor(3072/sqrt(3),32)=1760,
    w_bar=floor_by_factor(4096/sqrt(3),32)=2336  (4_111_360 px <= max)
  - patch_embedding Conv2d(3->768, k16 s16) flattens to grid 110x146
  - dense_embedding Conv2d(768->4096, kernel_size=2, stride=2) runs per image
    on (1, 768, 110, 146) bf16  ->  (1, 4096, 55, 73)

MIOpen must stay enabled (torch.backends.cudnn.enabled is asserted); the
110x146 spatial size has no precompiled kernel and forces the hiprtc JIT
path that used to segfault.
"""
import sys

import pytest

pytest.importorskip("torch", reason="GPU repro script — run on a torch host")

import torch

print(f"torch: {torch.__version__} | hip: {torch.version.hip}")
print(f"device: {torch.cuda.get_device_name(0)}")
assert torch.backends.cudnn.enabled, "MIOpen/cuDNN backend must remain enabled"
print(f"cudnn (MIOpen) enabled: {torch.backends.cudnn.enabled}")

torch.manual_seed(0)

conv = (
    torch.nn.Conv2d(768, 4096, kernel_size=2, stride=2)
    .to("cuda", torch.bfloat16)
    .requires_grad_(False)
)

# dynamic per-image shape: menu.jpg patch grid 110x146 at 768 channels
x = torch.randn(1, 768, 110, 146, device="cuda", dtype=torch.bfloat16)

with torch.no_grad():
    out = conv(x)

torch.cuda.synchronize()

print(out.shape)
print("finite:", torch.isfinite(out).all().item())
assert out.shape == (1, 4096, 55, 73), f"unexpected output shape {out.shape}"
assert torch.isfinite(out).all().item(), "non-finite output values"
print("PASS")
sys.exit(0)
