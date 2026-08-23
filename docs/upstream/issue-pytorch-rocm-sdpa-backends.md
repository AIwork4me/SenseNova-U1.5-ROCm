# [ROCm] SDPA on gfx1100: FLASH backend fails with explicit `scale` kwarg; mem-efficient backend fails several shapes — hipErrorInvalidValue (torch 2.12.0+rocm7.14.0)

## 🐛 Describe the bug

Two independent SDPA backend failures on the official
`torch==2.12.0+rocm7.14.0` wheel (gfx1100 / Radeon Pro W7900D, 48 GB).
Both return `hipErrorInvalidValue` at kernel launch. Because the error is
a launch-time (non-sticky) error, it surfaces at the next checked CUDA
call — in real models this appears far from the actual site (we first saw
it blamed on a decoder residual add), which makes it very confusing to
debug.

### Bug A — FLASH backend + explicit `scale` kwarg

```python
import torch, torch.nn.functional as F, math
from torch.nn.attention import SDPBackend, sdpa_kernel
q = torch.randn(1, 32, 1024, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 32, 1281, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, 32, 1281, 128, device="cuda", dtype=torch.bfloat16)
with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
    o = F.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(128))
    torch.cuda.synchronize()
# -> torch.AcceleratorError: CUDA error: invalid argument (hipErrorInvalidValue)
```

The same call **without** `scale` succeeds. Non-contiguous
(transpose-view) inputs behave the same way.

### Bug B — mem-efficient backend, multiple shapes

```python
import torch, torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
q = torch.randn(1, 32, 1024, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 32, 1281, 128, device="cuda", dtype=torch.bfloat16)   # kv_len NOT a power of two
v = torch.randn(1, 32, 1281, 128, device="cuda", dtype=torch.bfloat16)
with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
    o = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
# -> torch.AcceleratorError: CUDA error: invalid argument (hipErrorInvalidValue)
```

Fails for `kv_len` ∈ {1152, 1281, 1536} (works for 1024/2048/4096),
`head_dim=64` (kv_len=1024), and causal with S=4096. Works for the
power-of-two / head_dim-128 non-causal shapes.

## Backend matrix (gfx1100, torch 2.12.0+rocm7.14.0)

| Backend | result |
|---|---|
| FLASH, no `scale` | OK |
| FLASH, `scale=1/√128` | **FAIL (Bug A)** |
| EFFICIENT (several shapes) | **FAIL (Bug B)** |
| MATH | OK (every shape tested) |

## Impact

The default dispatcher selects a failing backend for common
decoder-style shapes, so any model passing `scale` (standard for
GPT-style attention, `1/sqrt(head_dim)`) or hitting non-power-of-two
kv lengths crashes on this stack. With a full multimodal
model (SenseNova-U1.5-8B-MoT) the crash surfaces as
`hipErrorInvalidValue` at an innocent elementwise add. Restricting the
dispatcher to MATH makes the model run end-to-end (we carry that as a
temporary compatibility patch).

## Environment

- GPU: AMD Radeon Pro W7900D (gfx1100, 48 GB)
- torch 2.12.0+rocm7.14.0 (official AMD wheel,
  `https://repo.amd.com/rocm/whl-multi-arch/`, plus the
  `amd-torch-device-gfx1100` device wheel), Python 3.12
- torch 2.8.0+rocm6.3 does not show either failure

Full debugging story, gdb/probe transcripts and the model-level
verification of the MATH-backend workaround:
https://github.com/AIwork4me/SenseNova-U1.5-ROCm/blob/main/docs/results/findings/rocm63-wheel-blas-on-gfx1100.md

Happy to run any further diagnostics on this hardware.
