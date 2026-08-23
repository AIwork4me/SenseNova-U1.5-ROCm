# ROCm 7.14.0 verification — are the three wheel bugs fixed upstream? (2026-08-23)

Spike on the reference host (gfx1100 48 GB). ROCm 7.14.0 obtained as the
official TheRock dist `therock-dist-linux-gfx110X-all-7.14.0.tar.gz`
(repo.amd.com/rocm/tarball-multi-arch, 2.31 GB; contains 600 rocBLAS
library files incl. 75 gfx1100 `TensileLibrary_*_gfx1100.dat` kernels,
`librocblas.so.5`, `libhipblas.so.3`, `libamd_comgr.so.3.3.0`,
`libMIOpen.so.1`), extracted to `/root/rocm-7.14-gfx110x` (throwaway,
not part of the pipeline). Torch side unchanged: the official
`torch==2.8.0+rocm6.3` wheel.

Notation: "7.14 BLAS" = `LD_PRELOAD=<7.14>/lib/libhipblas.so:<7.14>/lib/librocblas.so`
(+ `ROCBLAS_TENSILE_LIBPATH=<7.14>/lib/rocblas/library` where noted);
"wheel" = no preload (the wheel's bundled 6.3 libs).

| # | Repro | Stack | Result |
|---|---|---|---|
| A | bug-1 conv bf16 N=16 (wheel-alone repro) | 7.14 BLAS preload, no LIBPATH | **SIGSEGV** — same as 7.2.1 in this preload-without-libpath configuration; the default search path of a preloaded rocBLAS does not resolve its kernel dir |
| A2 | same | 7.14 BLAS + LIBPATH | **OK** |
| A3 | conv bf16 N=16060 | 7.14 BLAS + LIBPATH | **OK** |
| B | bug-2 big Linear bf16 M=16060 K=N=4096 | 7.14 BLAS + LIBPATH | **OK**, median abs diff 4e-4 vs CPU fp32 (identical to 7.2.1) |
| B2 | numeric sanity conv bf16 | 7.14 BLAS + LIBPATH | median rel err **0.32 %** — identical to the 7.2.1 workaround |
| C | full model, MIOpen enabled, wheel comgr | 7.14 BLAS + LIBPATH (comgr untouched = wheel 6.3) | **SIGSEGV in wheel comgr** (bug 3 reproduces with 7.14 BLAS too — the crash is the wheel's comgr, independent of which rocBLAS is loaded) |
| C1 | full model, MIOpen enabled, comgr swapped | 7.14 BLAS + wheel MIOpen 6.3 + **7.14 comgr symlinked over the wheel's** | **no segfault** — fails cleanly with `miopenStatusUnknownError` (mixed 6.3-MIOpen/7.14-comgr is not a supported combo, but the native crash is gone) |
| C2 | full model, MIOpen enabled | **full 7.14 stack preload**: 7.14 MIOpen + comgr + hipBLAS + rocBLAS + LIBPATH | **✅ model generates correctly** (menu VQA answer matches), with `torch.backends.cudnn.enabled = True` — **no bypass needed at all** |

## Verdict per bug

- **Bug 1 (rocBLAS lazy Tensile segfault): the wheel-6.3 bug is absent
  from ROCm 7.14's rocBLAS.** With 7.14's rocBLAS + its own kernel dir,
  every bf16 conv/GEMM repro passes with correct numerics. (The
  preload-without-LIBPATH crash in row A is a preload-usage artifact
  present in 7.2.1 and 7.14 alike, not the wheel bug.)
- **Bug 2 (6.3 decompressor vs 7.x data): does not exist in a 7.14 stack**
  — it is a wheel-6.3/data-7.x mixing problem. 7.14 rocBLAS reads its own
  data fine.
- **Bug 3 (MIOpen JIT comgr segfault): fixed in 7.14's comgr** (row C1:
  segfault becomes a clean error; row C2: with 7.14's MIOpen too, JIT
  works and the model runs with MIOpen enabled).

## Practical corollaries

1. All three bugs are **properties of the torch rocm6.3 wheel**, not of
   gfx1100 hardware. A torch wheel built against ROCm 7.14 should need
   **none** of this project's workarounds.
2. Until such a wheel exists, the workaround can be upgraded on hosts
   with ROCm 7.14: preloading the full 7.14 stack (MIOpen + comgr + BLAS)
   restores the MIOpen conv path (`cudnn.enabled` can stay true), removing
   the unfold+GEMM perf penalty.
3. The existing shipped workaround (system 7.2.1 BLAS + `cudnn off`)
   remains correct for hosts with only 7.2.x installed.

## Reproduction

```bash
R=/root/rocm-7.14-gfx110x   # from therock-dist-linux-gfx110X-all-7.14.0.tar.gz
LD_PRELOAD=$R/lib/libMIOpen.so:$R/lib/libamd_comgr.so:$R/lib/libhipblas.so:$R/lib/librocblas.so \
ROCBLAS_TENSILE_LIBPATH=$R/lib/rocblas/library python repro_vqa.py chat
# -> correct answer, MIOpen enabled, no bypass
```
