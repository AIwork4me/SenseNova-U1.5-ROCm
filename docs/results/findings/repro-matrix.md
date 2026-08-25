# Repro matrix — torch 2.8.0+rocm6.3 wheel on gfx1100 (2026-08-22)

Standalone matrix captured during the debugging session on the reference
host (gfx1100 48 GB, ROCm 7.2.1 system, Python 3.12, cp312 wheel).
Conv repro: `Conv2d(3→768, k16 s16)` bf16/fp16/fp32, N=16..16060.
GEMM repro: `Linear(4096→4096)` on `[16060, 4096]` bf16.
"Preload" = `LD_PRELOAD=/opt/rocm/lib/libhipblas.so:/opt/rocm/lib/librocblas.so`
"Libpath" = `ROCBLAS_TENSILE_LIBPATH=/opt/rocm/lib/rocblas/library`

| # | Repro | MIOpen | Preload | Libpath | Result |
|---|---|---|---|---|---|
| 1 | conv bf16, N=16..16060 | on | – | – | **SIGSEGV** (bug 1; transcript `transcripts/bug1-rocblas-tensile-segv.txt`) |
| 2 | conv bf16 | off | – | – | **SIGSEGV** (slow-conv fallback GEMM hits the same rocBLAS path) |
| 3 | conv fp32 | on | – | – | OK |
| 4 | plain matmul / Linear-with-bias / addmm, bf16 | – | – | – | OK (hipBLASLt path) |
| 5 | big Linear bf16 | – | – | Libpath only | **Unbundle decompress error → HIPBLAS_STATUS_INTERNAL_ERROR** (bug 2; transcript `transcripts/bug2-unbundle-decompress-error.txt`) |
| 6 | conv bf16 N=16..16060 | on/off | ✓ | ✓ | OK |
| 7 | big Linear bf16 | – | ✓ | ✓ | OK, median abs diff 4e-4 vs CPU fp32 |
| 8 | full model VQA forward | – | ✓ | ✓ | **SIGSEGV in wheel comgr** at ViT dense_embedding JIT (bug 3; transcript `transcripts/bug3-miopen-comgr-segv.txt`) |
| 9 | full model VQA forward | – | ✓ | ✓ | OK once `torch.backends.cudnn.enabled=False` (final config; receipt `../validation/vqa.json`) |

Numeric sanity of the workaround (rows 6–7): conv GPU bf16 vs CPU fp32
median rel err **0.32 %** (bf16 rounding level); GEMM median abs diff
**4e-4**. The preloaded system libraries produce correct results.

Full-model pre-patch interleave failure (upstream bug, not ROCm):
`../logs/interleave-prepatch-typeerror.txt`.
