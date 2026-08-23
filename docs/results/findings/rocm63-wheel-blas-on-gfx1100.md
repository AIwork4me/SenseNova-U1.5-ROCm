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

## Update 2026-08-23 — verified against ROCm 7.14.0

All three bugs are **absent from the ROCm 7.14.0 stack itself** — they are
properties of the torch rocm6.3 wheel's bundled libraries. Verified with
the official TheRock `gfx110X-all-7.14.0` dist on this host: 7.14's
rocBLAS runs every bf16 conv/GEMM repro with identical numerics (0.32 %
median rel err), and preloading the full 7.14 stack (MIOpen + comgr +
BLAS) lets the model run with **MIOpen enabled and no bypass at all**.
Full matrix and verdicts:
[`transcripts/rocm714-verification.md`](transcripts/rocm714-verification.md).
Practical upshot: a torch wheel built against ROCm 7.14 should need none
of the workarounds in this repo; until one ships, hosts with 7.14 can
preload the full 7.14 stack and drop the `cudnn off` penalty.

## Update 2026-08-23 (b) — AMD official torch 2.12.0+rocm7.14.0 wheels

AMD now ships torch wheels built against ROCm 7.14 via
`https://repo.amd.com/rocm/whl-multi-arch/` (pip-distribution model:
`rocm`/`rocm-sdk-core`/`rocm-sdk-libraries` metapackages + a per-arch
`amd-torch-device-gfx1100` device wheel that carries the gfx1100 kernels).
Install (cp312):

```bash
pip install "torch==2.12.0+rocm7.14.0" "torchvision==0.27.0+rocm7.14.0" \
            "amd-torch-device-gfx1100==2.12.0+rocm7.14.0" \
            --extra-index-url https://repo.amd.com/rocm/whl-multi-arch/
```

Measured on the reference host (W7900D, correctly identified by the new
torch device table):

- **All three wheel-6.3 bugs are absent — with ZERO workarounds** (no
  LD_PRELOAD, no ROCBLAS_TENSILE_LIBPATH, MIOpen enabled by default):
  bf16 conv (incl. N=16060) OK, large bf16 GEMM OK (4e-4), numerics
  identical (0.32 % median rel err), and full-model VQA answers correctly
  (receipt `../validation/vqa-torch212.json`).
- **The image-generation path (`forward_gen`) failed on torch 2.12 —
  root cause: two ROCm SDPA backend bugs** (found after per-op probes;
  final analysis 2026-08-23):
  on this stack **both fused SDPA backends (FLASH and mem-efficient)
  fail kernel launch for every configuration tested** (bf16; kv
  1024–4096 power-of-two or not; head_dim 64/128; causal or not;
  contiguous or transposed inputs; with or without an explicit `scale`
  kwarg — the generation path passes `1/sqrt(head_dim)` and kv lengths
  like 1281).
  Both are launch-time errors that surface at the next checked CUDA call
  (the decoder residual add) — misleading tracebacks; HIP_LAUNCH_BLOCKING
  does not relocate them and `hipGetLastError` is reset by intervening
  successful calls — and same-process probes contaminate each other,
  which briefly produced false "scale/shape-dependent" readings. Neither
  `torch.cuda.synchronize()` nor `AMD_SERIALIZE_KERNEL=3` relocates the
  deferred error. The understanding path happens never to select a fused
  backend on this stack (layout-eligibility dependent — explicit masks
  alone do not prevent the failing dispatch) — hence VQA worked unpatched. Fix shipped as
  [`patches/0002`](../../../patches/0002-sdpa-rocm-math-backend-compat.patch):
  on ROCm torch ≥ 2.9 restrict `_sdpa_attn_func` to the MATH backend and pre-scale
  `q`. Verified end-to-end: 2048×2048@50 = 687.7 s / 27.8 GiB
  (receipt `../validation/t2i-torch212-fixed.json`; torch-2.8 baseline
  420.1 s / 22.3 GiB — MATH costs ~64 % on this cell, correctness-first).
  Filed upstream: [pytorch/pytorch#194498](https://github.com/pytorch/pytorch/issues/194498)
  (SDPA fused-backend launch failures) and
  [OpenSenseNova/SenseNova-U1#261](https://github.com/OpenSenseNova/SenseNova-U1/issues/261)
  (compatibility issue with the verified patch).
- Practical guidance until upstream adapts to torch 2.12: **use torch
  2.8.0+rocm6.3 + this repo's workarounds for generation tasks**; torch
  2.12.0+rocm7.14.0 is a zero-workaround option for the understanding /
  VQA path.
