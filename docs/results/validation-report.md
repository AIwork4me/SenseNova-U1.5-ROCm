# Validation report

Reference host: AMD Radeon gfx1100 (48 GB VRAM, Device ID 0x744b) ·
ROCm 7.2.1 (`/opt/rocm`) · PyTorch 2.8.0+rocm6.3 (hip 6.3.42131) ·
transformers 4.57.1 · accelerate 1.14.0 · Python 3.12.3 · 1 TiB host RAM.
Full fingerprint: [`environment.json`](environment.json).

## 2026-08-23 — full-stack ROCm 7.14 re-validation (current default mode)

Same host, same torch 2.8.0+rocm6.3 wheel, **full-stack mode**
(`ROCM_FULL_STACK=auto` → detected `/root/rocm-7.14-gfx110x`): that stack's
MIOpen + comgr + hipBLAS + rocBLAS are preloaded (receipt field
`rocm_stack: full-stack`, `ld_preload` lists all four libs), MIOpen stays
enabled — the BLAS-mode unfold+GEMM detour is gone. Patch 0001 active;
patch 0002 gates itself off on torch 2.8.

| Block | Wall | Δ vs BLAS | Peak VRAM | Result | Receipt / log |
|---|---:|---:|---:|---|---|
| vqa | 628.0 s | +4.4 % | 25.26 GiB | ✅ correct bilingual menu read-out (greedy) | [vqa.json](validation/vqa.json) · [log](logs/vqa.log) |
| t2i | 379.6 s | −9.6 % | 22.31 GiB | ✅ 2048×2048 @ 50 steps (67.1 s load + 297.8 s gen ≈ 6.0 s/step) | [t2i.json](validation/t2i.json) · [log](logs/t2i.log) |
| t2i-think | 504.6 s | −7.7 % | 22.30 GiB | ✅ reasoning text + image | [t2i-think.json](validation/t2i-think.json) · [log](logs/t2i-think.log) |
| edit | 460.9 s | −4.7 % | 29.90 GiB | ✅ jacket recolored | [edit.json](validation/edit.json) · [log](logs/edit.log) |
| interleave | 3250.5 s | −4.2 % | 47.89 GiB | ✅ 7-image illustrated tutorial | [interleave.json](validation/interleave.json) · [log](logs/interleave.log) |
| determinism | — | — | — | ✅ identical sha256 `49e9f9b86160…` | [determinism.json](validation/determinism.json) |
| vram-mode-balanced | 207.5 s | +3.5 % | 22.25 GiB | ✅ 10-step probe | [vram-mode-balanced.json](validation/vram-mode-balanced.json) |
| vram-mode-fast | 210.1 s | +5.5 % | 22.31 GiB | ✅ 10-step probe | [vram-mode-fast.json](validation/vram-mode-fast.json) |
| vram-mode-low | 217.8 s | +4.5 % | 3.67 GiB | ✅ 10-step probe | [vram-mode-low.json](validation/vram-mode-low.json) |

Reading the deltas: every generation path got **faster with MIOpen back in
the loop** (−4.2 to −9.6 % — the unfold+GEMM penalty is gone); decode-bound
VQA and the short 10-step probes sit at +3.5 to +5.5 % (probe blocks are
load-dominated, so the full-stack library preload shows up as small
overhead). `low` mode's peak grew 3.39 → 3.67 GiB (MIOpen runtime resident)
— still ~13× smaller than the checkpoint (46.8 GiB of bf16 weights).

Determinism note: the full-stack sha256 (`49e9f9b8…`) differs from the
BLAS baseline's (`d24ae824…`) — different math libraries, different
rounding. Within one stack, same seed stays byte-identical.

Harness fix shipped with this round: the `validate.sh` interleave block now
registers its (variable-count) output pngs in the receipt (artifact glob
expanded after the run, stale outputs cleared first). The 2026-08-23
interleave receipt carries all 7 pngs with sha256; baseline receipts (and
all pre-2026-08-23 interleave receipts) have an empty `artifacts` map.

## 2026-08-22 — BLAS-mode baseline

> **Scope note (2026-08-23):** this section is the original full-suite run
> on torch 2.8.0+rocm6.3 in BLAS mode; its receipts are archived in
> [`matrix-blas-20260822/`](validation/matrix-blas-20260822/) and the
> table links below now point at the current (full-stack) receipts.
> Later same-host results also live alongside: `vqa-torch212.json`
> (torch 2.12, zero workarounds, VQA correct), `t2i-torch212.json`
> (torch 2.12 unpatched — generation fails; root-caused to ROCm SDPA
> fused-backend launch bugs,
> [pytorch/pytorch#194498](https://github.com/pytorch/pytorch/issues/194498)),
> and `t2i-torch212-fixed.json` (torch 2.12 + patches/0002: 687.7 s /
> 27.8 GiB, zero BLAS workarounds). See the findings doc's 2026-08-23
> updates for the full story.

Executive summary: **all four SenseNova-U1.5-8B-MoT task families run
correctly on a single 48 GB gfx1100 through the upstream transformers
path with layer offload**, with three wheel-level ROCm bugs root-caused
and worked around, and one upstream cross-platform bug found and patched.

### Baseline receipts (archived)

Every row links to a machine-written receipt (exact command, wall time,
device-level peak VRAM via rocm-smi sampler, output sha256) and its raw
log in [`logs/`](logs/).

| Block | Wall | Peak VRAM | Result | Archived receipt |
|---|---:|---:|---|---|
| vqa | 601.8 s | 24.75 GiB | ✅ correct bilingual menu read-out (greedy) | [vqa.json](validation/matrix-blas-20260822/vqa.json) |
| t2i | 420.1 s | 22.28 GiB | ✅ 2048×2048 @ 50 steps (65.6 s load + 347.9 s gen) | [t2i.json](validation/matrix-blas-20260822/t2i.json) |
| t2i-think | 546.6 s | 22.34 GiB | ✅ reasoning text + image | [t2i-think.json](validation/matrix-blas-20260822/t2i-think.json) |
| edit | 483.8 s | 29.82 GiB | ✅ jacket recolored | [edit.json](validation/matrix-blas-20260822/edit.json) |
| interleave | 3392.0 s | 47.66 GiB | ✅ 7-image illustrated tutorial | [interleave.json](validation/matrix-blas-20260822/interleave.json) |
| determinism | — | — | ✅ identical sha256 `d24ae824c575492c…` | [determinism.json](validation/matrix-blas-20260822/determinism.json) |
| vram-mode-balanced | 200.5 s | 22.28 GiB | ✅ 10-step probe | [vram-mode-balanced.json](validation/matrix-blas-20260822/vram-mode-balanced.json) |
| vram-mode-fast | 199.2 s | 22.28 GiB | ✅ 10-step probe | [vram-mode-fast.json](validation/matrix-blas-20260822/vram-mode-fast.json) |
| vram-mode-low | 208.4 s | 3.39 GiB | ✅ 10-step probe | [vram-mode-low.json](validation/matrix-blas-20260822/vram-mode-low.json) |

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
   +4 % wall time (10-step probe; full-stack re-check 2026-08-23: 3.67 GiB,
   +5 %) — the 50 GB model is runnable on small cards.
4. **Interleave is the memory-heavy path**: 7 sequential images with CFG
   condition branches peaked at 47.66 GiB device memory (2026-08-22 BLAS
   baseline; 47.89 GiB in the 2026-08-23 full-stack run) — GTT absorbed the
   spike on this host.

## How to reproduce

```bash
bash scripts/00-check-env.sh
bash scripts/01-setup-venv.sh          # applies upstream patches + smoke test
bash scripts/02-fetch-model.sh         # SHA256-verifies (skips if present)
bash scripts/validate.sh               # ~1.75 h full-stack / ~2.5 h BLAS on gfx1100
python3 scripts/summarize_results.py   # prints the tables above
```
