# Draft closing comment for pytorch/pytorch#194447 (NOT posted)

> Status: DRAFT ONLY — do not post without explicit human approval.
> Prepared 2026-08-26 after the native nightly validation:
> [`docs/results/findings/pytorch-nightly-rocm714-native-validation.md`](../results/findings/pytorch-nightly-rocm714-native-validation.md)

---

Thanks @naromero77amd.

I verified this against the current upstream PyTorch nightly ROCm 7.14 wheel
(`torch-2.15.0.dev20260825+rocm7.14`, `torch.version.hip = 7.14.60850`) on
the same gfx1100 system, together with the ROCm 7.14 pip runtime
(`rocm-sdk-core/libraries/device-gfx1100` 7.14.0) the wheel's RPATH expects.

Tested natively, with:
- no `LD_PRELOAD`
- no `ROCBLAS_TENSILE_LIBPATH`
- MIOpen enabled (`torch.backends.cudnn.enabled = True` throughout)

Results (5 independent process runs each):
- Bug 1 — bf16 `Conv2d(3,768,16,16)`: rocBLAS/Tensile lazy-load SIGSEGV → **PASS** (5/5, correct shape, finite, 0.32 % median rel err vs CPU fp32)
- Bug 2 — large bf16 GEMM `Linear(4096,4096)` on `(16060,4096)`: Tensile decompression / `HIPBLAS_STATUS_INTERNAL_ERROR` → **PASS** (5/5)
- Bug 3 — MIOpen JIT conv `Conv2d(768,4096,k2,s2)` on the dynamic ViT `dense_embedding` shape `(1,768,110,146)`: comgr SIGSEGV → **PASS** (5/5; MIOpen kernel compilation verified — its user kernel DB is populated)
- SenseNova-U1.5-8B-MoT end-to-end VQA inference (the workload from the report): **PASS**, correct output

This confirms the failures were specific to the 2.8.0+rocm6.3 wheel's bundled
stack and are no longer present with the ROCm 7.14 nightly wheel. One install
note for other gfx1100 users: the nightly wheel does not bundle the ROCm
userspace — install the ROCm 7.14 runtime alongside it (AMD pip distribution
`rocm-sdk-core`/`rocm-sdk-libraries`/`rocm-sdk-device-<gfx>` from
repo.amd.com, or a system ROCm 7.14); without it `import torch` fails on
`libhipfile.so.0`.

Full validation log and repro scripts:
[AIwork4me/SenseNova-U1.5-ROCm — pytorch-nightly-rocm714-native-validation.md](https://github.com/AIwork4me/SenseNova-U1.5-ROCm/blob/main/docs/results/findings/pytorch-nightly-rocm714-native-validation.md)

Closing as resolved. Thanks!
