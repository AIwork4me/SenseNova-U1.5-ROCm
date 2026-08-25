# [ROCm] SDPA fused backends (FLASH and mem-efficient) fail to launch on gfx1100 with torch 2.12.0+rocm7.14.0 — deferred hipErrorInvalidValue

## 🐛 Describe the bug

On the official `torch==2.12.0+rocm7.14.0` wheel (gfx1100 / Radeon Pro
W7900D, 48 GB), **both fused SDPA backends fail kernel launch for every
configuration we tested** (bf16, various shapes: kv 1024–4096, power-of-two
and non-power-of-two, head_dim 64/128, causal and not, contiguous and
transposed-view inputs). Only the MATH backend works.

Two properties make this especially confusing to debug:

1. The `hipErrorInvalidValue` is a **launch-time error that is deferred**:
  it surfaces at the *next* checked CUDA operation — not at the SDPA call,
  not even at `torch.cuda.synchronize()` or with `AMD_SERIALIZE_KERNEL=3`.
  In real models it appears at an innocent elementwise op (we first saw it
  blamed on a decoder residual add).
2. Within one process the error state can contaminate subsequent probes
  (an unrelated follow-up op reports the deferred failure), which easily
  produces misleading "shape-dependent" hypotheses during bisection.

### Minimal repro (each case in a FRESH process)

```python
# fresh python process
import torch, torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

q = torch.randn(1, 32, 1024, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 32, 1281, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, 32, 1281, 128, device="cuda", dtype=torch.bfloat16)
with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):        # same for EFFICIENT_ATTENTION
    o = F.scaled_dot_product_attention(q, k, v)
    print(o.float().abs().sum().item())                # trailing checked op — raises here
# -> torch.AcceleratorError: CUDA error: invalid argument (hipErrorInvalidValue)
```

Swap in `SDPBackend.MATH` and the same call succeeds (and matches a CPU
fp32 reference at ~1e-3). The trailing consumer op matters: without it
the failing case can exit 0 because the launch error is deferred.

## Backend matrix (gfx1100, torch 2.12.0+rocm7.14.0, bf16; per-case fresh process + trailing checked op)

| Backend | result |
|---|---|
| FLASH_ATTENTION | **FAIL — every config tested** (with and without explicit `scale`; kv 1024/1152/1281/1536/2048/4096; head_dim 64/128; causal/non-causal; contig/transposed) |
| EFFICIENT_ATTENTION | **FAIL — every config tested** (same sweep) |
| MATH | OK — every config tested; numerically sane vs CPU fp32 |

Profiler output for a FLASH-only call shows **zero kernels launched** —
the failure is at launch/argument validation, not inside a kernel.

## Impact

The default dispatcher picks a fused backend for ordinary decoder-style
shapes, so attention crashes on this stack with a misleading traceback.
We carry a temporary compatibility patch restricting the dispatcher to
MATH on ROCm for one such model (SenseNova-U1.5-8B-MoT); with that, the
full 50 GB model runs end-to-end on this card.

## Environment

- GPU: AMD Radeon Pro W7900D (gfx1100, 48 GB)
- torch 2.12.0+rocm7.14.0 (official AMD wheel,
  `https://repo.amd.com/rocm/whl-multi-arch/` + `amd-torch-device-gfx1100`
  device wheel), Python 3.12
- Model-level comparison: the same model code with torch 2.8.0+rocm6.3
  (SDPA path included) runs the equivalent workload to completion on the
  same host

Full debugging story and transcripts:
https://github.com/AIwork4me/SenseNova-U1.5-ROCm/blob/main/docs/results/findings/rocm63-wheel-blas-on-gfx1100.md

Happy to run any further diagnostics on this hardware.

## Resolution (2026-08-25)

Root-caused by liminfei-amd
([comment](https://github.com/pytorch/pytorch/issues/194498#issuecomment-5406837588)):
wheel metadata — the `amd-torch-device-gfx1100` leaf lacks its dependency
on the `amd-torch-device-gfx11` family wheel (AOTriton images). Fix:
[ROCm/rocm-systems#10685](https://github.com/ROCm/rocm-systems/pull/10685).
Independently verified A/B on the reporting host with a single package
delta: fused 0/8 → 8/8, and the real-model t2i workload that motivated
this issue completes 1.94× faster than the MATH fallback with the gfx11
wheel installed. Receipts:
[../results/validation/sdpa-gfx11/README.md](../results/validation/sdpa-gfx11/README.md).
