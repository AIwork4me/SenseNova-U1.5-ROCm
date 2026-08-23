# Validation report — 2026-08-22

Reference host: AMD Radeon gfx1100 (48 GB VRAM, Device ID 0x744b) ·
ROCm 7.2.1 (`/opt/rocm`) · PyTorch 2.8.0+rocm6.3 (hip 6.3.42131) ·
transformers 4.57.1 · accelerate 1.14.0 · Python 3.12.3 · 1 TiB host RAM.
Full fingerprint: [`environment.json`](environment.json).

> **Scope note (2026-08-23):** this report is the 2026-08-22 full-suite
> run on torch 2.8.0+rocm6.3 in BLAS mode. Later same-host results live
> alongside it: `vqa-torch212.json` (torch 2.12, zero workarounds, VQA
> correct), `t2i-torch212.json` (torch 2.12 unpatched — generation fails;
> root-caused to ROCm SDPA fused-backend launch bugs,
> [pytorch/pytorch#194498](https://github.com/pytorch/pytorch/issues/194498)),
> and `t2i-torch212-fixed.json` (torch 2.12 + patches/0002: 687.7 s /
> 27.8 GiB, zero BLAS workarounds). See the findings doc's 2026-08-23
> updates for the full story.

Executive summary: **all four SenseNova-U1.5-8B-MoT task families run
correctly on a single 48 GB gfx1100 through the upstream transformers
path with layer offload**, with three wheel-level ROCm bugs root-caused
and worked around, and one upstream cross-platform bug found and patched.

## Receipts

Every row links to a machine-written receipt (exact command, wall time,
device-level peak VRAM via rocm-smi sampler, output sha256) and its raw
log in [`logs/`](logs/).

| Block | Wall | Peak VRAM | Result | Receipt / log |
|---|---:|---:|---|---|
| vqa | 601.8 s | 24.75 GiB | ✅ correct bilingual menu read-out (greedy) | [vqa.json](validation/vqa.json) · [log](logs/vqa.log) |
| t2i | 420.1 s | 22.28 GiB | ✅ 2048×2048 @ 50 steps (65.6 s load + 347.9 s gen) | [t2i.json](validation/t2i.json) · [log](logs/t2i.log) |
| t2i-think | 546.6 s | 22.34 GiB | ✅ reasoning text + image | [t2i-think.json](validation/t2i-think.json) · [log](logs/t2i-think.log) |
| edit | 483.8 s | 29.82 GiB | ✅ jacket recolored | [edit.json](validation/edit.json) · [log](logs/edit.log) |
| interleave | 3392.0 s | 47.66 GiB | ✅ 7-image illustrated tutorial | [interleave.json](validation/interleave.json) · [log](logs/interleave.log) |
| determinism | — | — | ✅ identical sha256 `d24ae824c575492c…` | [determinism.json](validation/determinism.json) |
| vram-mode-balanced | 200.5 s | 22.28 GiB | ✅ 10-step probe | [vram-mode-balanced.json](validation/vram-mode-balanced.json) |
| vram-mode-fast | 199.2 s | 22.28 GiB | ✅ 10-step probe | [vram-mode-fast.json](validation/vram-mode-fast.json) |
| vram-mode-low | 208.4 s | 3.39 GiB | ✅ 10-step probe | [vram-mode-low.json](validation/vram-mode-low.json) |

Checkpoint integrity: 24/24 files match size + SHA256 against
[`configs/artifact-manifest.json`](../../configs/artifact-manifest.json)
(ModelScope revision `27fc42dc`).

Generated-output gallery: [`gallery/`](gallery/README.md).

## Notable engineering findings

1. **The rocm6.3 wheel's math stack crashes natively on gfx1100** — three
   distinct bugs (rocBLAS lazy Tensile segfault; 6.3-era decompressor
   cannot read ROCm 7.x Tensile data → `HIPBLAS_STATUS_INTERNAL_ERROR` on
   large bf16 GEMMs; MIOpen JIT compile segfault in comgr). Workaround:
   preload system hipBLAS/rocBLAS + `ROCBLAS_TENSILE_LIBPATH` +
   `cudnn.enabled=False`; numerics verified vs CPU fp32 (median rel err
   0.3 % on conv). Full story:
   [findings/rocm63-wheel-blas-on-gfx1100.md](findings/rocm63-wheel-blas-on-gfx1100.md).
2. **Upstream bug**: `interleave_gen` / `interleave_gen_image_only` omit
   `image_size` when calling `_t2i_predict_v`, which is a hard `TypeError`
   with `use_pixel_head: true` (U1.5) — on every platform. Fixed by
   [patches/0001](../../patches/0001-interleave-pass-image_size-to-_t2i_predict_v.patch).
3. **`low` VRAM mode is the sleeper feature**: 3.39 GiB peak at only
   +4 % wall time (10-step probe) — the 50 GB model is runnable on small
   cards.
4. **Interleave is the memory-heavy path**: 7 sequential images with CFG
   condition branches peaked at 47.66 GiB device memory (GTT absorbed the
   spike on this host).

## How to reproduce

```bash
bash scripts/00-check-env.sh
bash scripts/01-setup-venv.sh          # applies upstream patches + smoke test
bash scripts/02-fetch-model.sh         # SHA256-verifies (skips if present)
bash scripts/validate.sh               # ~2.5 h on gfx1100
python3 scripts/summarize_results.py   # prints the table above
```
