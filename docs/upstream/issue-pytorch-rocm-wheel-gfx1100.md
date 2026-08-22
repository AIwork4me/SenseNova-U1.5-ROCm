# ROCm wheel 2.8.0+rocm6.3: three native crashes in the bundled math stack on gfx1100 (bf16 conv / large bf16 GEMM / MIOpen JIT)

## 🐛 Describe the bug

On a gfx1100 (Radeon W7900-class, 48 GB; ROCm 7.2.1 system userspace at
/opt/rocm), the official `torch==2.8.0` wheel from
`https://download.pytorch.org/whl/rocm6.3` (cp312, Python 3.12.3) crashes
natively in three distinct places, all inside libraries bundled with the
wheel. Plain bf16 matmuls (hipBLASLt path) work; the failures hit any
workload that routes through rocBLAS or MIOpen JIT. All three reproduce
standalone on every run; gdb transcripts and the full repro matrix are
linked below.

### Bug 1 — rocBLAS lazy Tensile load segfaults on half-precision GEMMs routed through rocBLAS

```python
import torch
conv = torch.nn.Conv2d(3, 768, 16, 16).to("cuda", torch.bfloat16).requires_grad_(False)
out = conv(torch.randn(16, 3, 16, 16, device="cuda", dtype=torch.bfloat16))
# -> SIGSEGV (no Python traceback); fp32 runs fine; same crash with torch.backends.cudnn.enabled = False
```

gdb backtrace (full transcript:
[bug1-rocblas-tensile-segv.txt](https://github.com/AIwork4me/SenseNova-U1.5-ROCm/blob/main/docs/results/findings/transcripts/bug1-rocblas-tensile-segv.txt)):

```
#0 Tensile::PlaceholderLibrary<Tensile::ContractionProblem, Tensile::ContractionSolution>::loadPlaceholderLibrary() const
   from .../torch/lib/librocblas.so
#2 Tensile::ExactLogicLibrary<...>::findBestSolution(...)
#15 rocblas_gemm_ex () from .../torch/lib/librocblas.so
#16 miopen::CallGemm(...) from .../torch/lib/libMIOpen.so
```

### Bug 2 — the wheel's 6.3-era rocBLAS cannot decompress ROCm 7.x Tensile data

Redirecting to the system kernel set
(`ROCBLAS_TENSILE_LIBPATH=/opt/rocm/lib/rocblas/library`) fixes small
GEMMs but large bf16 GEMMs then fail (transcript:
[bug2-unbundle-decompress-error.txt](https://github.com/AIwork4me/SenseNova-U1.5-ROCm/blob/main/docs/results/findings/transcripts/bug2-unbundle-decompress-error.txt)):

```python
lin = torch.nn.Linear(4096, 4096).to("cuda", torch.bfloat16)
out = lin(torch.randn(16060, 4096, device="cuda", dtype=torch.bfloat16))
```

```
Unbundle Objects Error: Failed to decompress input: Could not decompress embedded file contents: Unknown frame descriptor
RuntimeError: CUDA error: HIPBLAS_STATUS_INTERNAL_ERROR when calling `hipblasSgemm( handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc)`
```

i.e. the 6.3 decompressor cannot read the 7.x `.dat` format.

### Bug 3 — MIOpen JIT kernel compile segfaults in the bundled comgr

With bugs 1–2 worked around, a conv with no precompiled MIOpen kernel
(e.g. the SenseNova-U1.5 ViT `dense_embedding`: 768→4096, k2 s2,
per-image dynamic shape) crashes inside the wheel's `libamd_comgr.so`
(full transcript:
[bug3-miopen-comgr-segv.txt](https://github.com/AIwork4me/SenseNova-U1.5-ROCm/blob/main/docs/results/findings/transcripts/bug3-miopen-comgr-segv.txt)):

```
#0 COMGR::DataObject::clearData() from .../torch/lib/libamd_comgr.so
#4 amd_comgr_do_action () from .../torch/lib/libamd_comgr.so
#5 hiprtc::helpers::compileToExecutable(...) from .../torch/lib/libamdhip64.so
#7 hiprtcCompileProgram () from .../torch/lib/libamdhip64.so
#9 miopen::hiprtc::BuildHip(...) from .../torch/lib/libMIOpen.so
```

`LD_PRELOAD`-ing the system comgr does **not** interpose (symbol-version
binding keeps the calls on the wheel's copy).

## Working workaround (verified)

Route BLAS through the system ROCm and bypass MIOpen JIT entirely:

```bash
export LD_PRELOAD=/opt/rocm/lib/libhipblas.so:/opt/rocm/lib/librocblas.so
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm/lib/rocblas/library
```

```python
import torch
torch.backends.cudnn.enabled = False   # conv via unfold+GEMM; avoids bug 3
```

With this, the conv workaround output matches a CPU fp32 reference at
median rel err 0.32 % (bf16 rounding level) and the large GEMM at median
abs diff 4e-4; a 50.2 GB bf16 unified multimodal model
(SenseNova-U1.5-8B-MoT) then runs end-to-end on the card. Full matrix:
[repro-matrix.md](https://github.com/AIwork4me/SenseNova-U1.5-ROCm/blob/main/docs/results/findings/repro-matrix.md).

## Environment

- GPU: AMD Radeon gfx1100 (Device ID 0x744b, 48 GB)
- ROCm system userspace: 7.2.1 at /opt/rocm
- torch 2.8.0+rocm6.3 (cp312 wheel), Python 3.12.3
- Reproduced on every run; standalone snippets above

Full root-cause analysis with all transcripts:
https://github.com/AIwork4me/SenseNova-U1.5-ROCm/blob/main/docs/results/findings/rocm63-wheel-blas-on-gfx1100.md

Happy to run any additional diagnostics you need on this hardware.
