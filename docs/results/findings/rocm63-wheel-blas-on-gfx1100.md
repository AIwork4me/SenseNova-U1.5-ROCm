# Finding: the PyTorch rocm6.3 wheel's math stack is broken on gfx1100

Date: 2026-08-22 · Host: gfx1100 (48 GB) · ROCm 7.2.1 system ·
torch 2.8.0+rocm6.3 (cp312 wheel) · Status: **root-caused, worked around,
model runs correctly**

## Summary

Running SenseNova-U1.5-8B-MoT (bf16) through the official PyTorch ROCm
wheel hits **three distinct native bugs in the wheel's bundled math
libraries**, all reproducible standalone, all independent of the model
code:

| # | Bug | Trigger | Native crash site |
|---|---|---|---|
| 1 | rocBLAS lazy Tensile segfault | any bf16/fp16 GEMM routed through the bundled rocBLAS (the ViT patch-embedding conv is the first) | `Tensile::PlaceholderLibrary::loadPlaceholderLibrary()` in the wheel's `librocblas.so` |
| 2 | rocBLAS 6.3 cannot read ROCm 7.x Tensile data | redirecting to system kernel files (partial fix for #1) then running a large bf16 GEMM (first LLM `q_proj`, M=16060 K=N=4096) | `Unbundle Objects Error: Failed to decompress ... Unknown frame descriptor` → `HIPBLAS_STATUS_INTERNAL_ERROR` |
| 3 | MIOpen JIT kernel compile segfault | a conv with no precompiled MIOpen kernel (the ViT `dense_embedding`, 768→4096, k2 s2, dynamic per-image shape) | `COMGR::DataObject::clearData()` in the wheel's `libamd_comgr.so`, via `hiprtcCompileProgram` |

The wheel's `libamd_comgr.so` also **cannot be shadowed by preloading the
system copy** — symbol-version binding keeps the calls on the wheel's
comgr (verified with gdb).

## The workaround (what this project ships)

Applied automatically, no upstream code touched:

1. **`LD_PRELOAD` the system ROCm's `libhipblas.so` + `librocblas.so`** —
   symbol interposition routes every BLAS call to the ROCm 7.2.1 stack,
   which reads its own native Tensile data format correctly.
   (`scripts/lib/common.sh`)
2. **`ROCBLAS_TENSILE_LIBPATH=/opt/rocm/lib/rocblas/library`** — points
   rocBLAS at the system (full, non-lazy) gfx1100 kernel set.
3. **`torch.backends.cudnn.enabled = False`** — convolutions bypass
   MIOpen's JIT entirely and run as unfold+GEMM through the now-healthy
   rocBLAS GEMMs. (`src/sensenova_u1_rocm/`, opt out with `SENU15_MIOPEN=1`)

## Verification

- Standalone conv repro: bf16 Conv2d(3→768, k16 s16), N=16..16060 —
  segfault before, OK after (both MIOpen on and off).
- Standalone GEMM repro: bf16 Linear M=16060 K=N=4096 —
  `HIPBLAS_STATUS_INTERNAL_ERROR` before, OK after.
- Numerics: GPU bf16 conv vs CPU fp32 reference — median rel err
  0.32 % (bf16 rounding level); GPU bf16 GEMM vs CPU fp32 — median abs
  diff 4e-4. Not a numerics-breaking workaround.
- End-to-end: VQA on the upstream `menu.jpg` returns a correct, coherent
  answer (receipt `docs/results/validation/vqa.json`).

## Reproduction snippets (before the fix)

```python
# Bug 1/3 — segfault on the wheel alone:
conv = torch.nn.Conv2d(3, 768, 16, 16).to("cuda", torch.bfloat16)
conv(torch.randn(16, 3, 16, 16, device="cuda", dtype=torch.bfloat16))

# Bug 2 — with only ROCBLAS_TENSILE_LIBPATH redirected:
lin = torch.nn.Linear(4096, 4096).to("cuda", torch.bfloat16)
lin(torch.randn(16060, 4096, device="cuda", dtype=torch.bfloat16))
# → "Unbundle Objects Error: Failed to decompress ... Unknown frame descriptor"
# → RuntimeError: HIPBLAS_STATUS_INTERNAL_ERROR
```

Primary evidence artifacts:

- [`transcripts/bug1-rocblas-tensile-segv.txt`](transcripts/bug1-rocblas-tensile-segv.txt) — gdb backtrace (wheel-only bf16 conv → SIGSEGV in `Tensile::PlaceholderLibrary::loadPlaceholderLibrary`, wheel `librocblas.so`)
- [`transcripts/bug2-unbundle-decompress-error.txt`](transcripts/bug2-unbundle-decompress-error.txt) — full decompressor error + `HIPBLAS_STATUS_INTERNAL_ERROR` (libpath-only, large bf16 GEMM)
- [`transcripts/bug3-miopen-comgr-segv.txt`](transcripts/bug3-miopen-comgr-segv.txt) — gdb backtrace (full-model forward → SIGSEGV in `COMGR::DataObject::clearData`, wheel `libamd_comgr.so`, via `hiprtcCompileProgram` / MIOpen JIT)
- [`repro-matrix.md`](repro-matrix.md) — the full standalone matrix (dtypes × MIOpen on/off × workaround combos) with numeric-sanity results

## Open questions / upstream value

- Bugs 1–3 likely affect **every** gfx1100 user of the official
  `torch==2.8.0` rocm6.3 wheel doing half-precision conv/GEMM through
  rocBLAS or MIOpen JIT (plain matmuls survive because they route through
  hipBLASLt). Filed upstream:
  [pytorch/pytorch#194447](https://github.com/pytorch/pytorch/issues/194447).
- The companion upstream fix for SenseNova-U1's interleave bug is filed as
  [OpenSenseNova/SenseNova-U1#260](https://github.com/OpenSenseNova/SenseNova-U1/pull/260).
- A wheel built against a ROCm ≥ 7.x userspace, or bundling a fixed comgr,
  should make the preloads unnecessary. The `cudnn.enabled=False` bypass
  costs some conv performance (unfold+GEMM instead of MIOpen's compiled
  kernels) — measured impact is dominated by the LLM/GEMM phases of this
  model either way.
