# Finding: upstream PyTorch nightly rocm7.14 on gfx1100 — fused SDPA healthy, but t2i generation silently corrupts (2026-08-26)

Two results from the same venv as the [native nightly validation](pytorch-nightly-rocm714-native-validation.md) (torch 2.15.0.dev20260825+rocm7.14, hip 7.14.60850, AMD pip ROCm 7.14 runtime, zero workarounds):

1. **The pytorch#194498 failure mode (fused SDPA deferred-launch failures) is ABSENT on the upstream nightly packaging.** The nightly wheel bundles `torch/lib/aotriton.images/` (incl. `amd-gfx110x/flash/`) and `libaotriton_v2.so.0.13.0` itself, so the missing-AOTriton-images defect of the AMD multi-arch leaf wheels cannot occur. Fresh-process matrix (phase C): FLASH 4/4, EFFICIENT 4/4, MATH 3/3 — 11/11 ok; forced-fused outputs match MATH with norm_rel ≤ 5.4e-6, median_rel 0, max_abs at exact bf16 quanta (2^-9 / 2^-6). MATH reference sums are bit-identical across phases A/B/C (three different torch builds, seed 42), and phase-C fused sums differ slightly from phase B's — independent-run corroboration, not copied output.
2. **NEW REGRESSION (previously unreported): the t2i/generation path silently produces corrupted output on this nightly build.** The process exits 0 and saves a PNG, but the image is a washed-out repeating grid texture, not the prompted scene. This is NOT the #194498 mechanism: the corruption is bit-identical with fused SDPA backends disabled (pure MATH), so fused kernels are exonerated; it is stack-specific (identical code/seed/config on the validated torch 2.8.0+rocm6.3 stack produces a natural image); and it is path-specific (nightly VQA/understanding produces a correct, coherent answer).

## Evidence: SDPA probe matrix, phase C (nightly)

Receipts: [sdpa-gfx11/phaseC-nightly-matrix.json](../validation/sdpa-gfx11/phaseC-nightly-matrix.json), per-case logs in [phaseC-cases.tar.gz](../validation/sdpa-gfx11/phaseC-cases.tar.gz), pip freeze [pip-freeze-nightly-rocm714.txt](../validation/sdpa-gfx11/pip-freeze-nightly-rocm714.txt), raw transcript [transcripts/sdpa-nightly-rocm714/phaseC-nightly-matrix.txt](transcripts/sdpa-nightly-rocm714/phaseC-nightly-matrix.txt).

| phase | stack | fused (FLASH+EFFICIENT) | MATH |
|---|---|---|---|
| A | AMD 2.12.0+rocm7.14.0 leaf-only (no gfx11 family wheel) | **0/8** — deferred `hipErrorInvalidValue` | 3/3 |
| B | A + `amd-torch-device-gfx11` (only delta) | 8/8 | 3/3 |
| C | **upstream nightly 2.15.0.dev20260825+rocm7.14** (AOTriton bundled in wheel) | **8/8** | 3/3 |

Fused-vs-MATH numerics (same-process, numerics-only; launch success proven fresh-process): [transcripts/sdpa-nightly-rocm714/phaseC-fused-vs-math.txt](transcripts/sdpa-nightly-rocm714/phaseC-fused-vs-math.txt), generator committed as `tests/validation/sdpa_fused_vs_math_nightly.py`.

## Evidence: t2i silent corruption on nightly

All runs: upstream `examples/t2i/inference.py`, `--vram_mode balanced --attn_backend sdpa`, prompt "A cinematic mountain village at sunrise, golden light over slate roofs, drifting mist between timber houses, ultra detailed", seed 42, MIOpen enabled, no LD_PRELOAD / no ROCBLAS_TENSILE_LIBPATH. Stats method: horizontal neighbor-pixel correlation (hcorr) and correlation of horizontally adjacent 64 px block means (block64-corr) — a natural image has hcorr ≈ 0.99 and block64-corr well below 1; a repeating grid texture shows hcorr depressed and block64-corr → 1.

| run | stack | wall | image sha256 (prefix) | mean | std | hcorr | block64-corr | verdict |
|---|---|---|---|---|---|---|---|---|
| 2048×2048 @ 50 | nightly, stock dispatcher | 677 s (load 66 s) | `970cd11c…` | 129.8 | 29.1 | 0.570 | **0.998** | CORRUPT (grid) |
| 1024×1024 @ 25 | nightly, stock | 195 s | `ad517645…` | 219.8 | 33.1 | 0.739 | **0.985** | CORRUPT (grid) |
| 1024×1024 @ 25 | **torch 2.8.0+rocm6.3 + repo workarounds** (control, `run-task.sh`) | 175 s | `1ce9d682…` | 66.6 | 64.8 | **0.988** | 0.908 | GOOD (natural) |
| 1024×1024 @ 25 | nightly, **fused SDPA disabled** (MATH-only diagnostic) | 189 s | `ad517645…` (**bit-identical to stock**) | 219.8 | 33.1 | 0.739 | 0.985 | CORRUPT (same image) |

Isolation chain, in order: 2048@50 corrupt → 1024@25 reproduces (not transient, not size/steps-specific) → identical config on the validated torch 2.8 stack is good (stack-specific) → MATH-only nightly is bit-identical to stock nightly (fused SDPA exonerated; also implies the stock dispatcher was already choosing MATH for these layouts, consistent with the 677 s ≈ MATH-baseline wall time and with the layout-eligibility behavior documented for the understanding path) → nightly VQA is correct (path-specific) → op-level probes (bf16 conv / large GEMM / MIOpen JIT / SDPA incl. explicit-scale and kv-1281 variants) all match references. The fault therefore lies in some op or configuration exercised by the generation path (DiT blocks / VAE decode / conditioning) on this nightly build; the corrupt signature (uniform repeating block grid, flattened contrast) suggests a positional/patch-assembly-style miscomputation, but no op has been convicted — bisection is future work.

Logs: [transcripts/sdpa-nightly-rocm714/t2i-e2e.txt](transcripts/sdpa-nightly-rocm714/t2i-e2e.txt) (2048, ends `T2I_EXIT=0`), [t2i-1024-nightly-ab.txt](transcripts/sdpa-nightly-rocm714/t2i-1024-nightly-ab.txt), [t2i-1024-torch28-ab.txt](transcripts/sdpa-nightly-rocm714/t2i-1024-torch28-ab.txt), [t2i-1024-nightly-mathonly.txt](transcripts/sdpa-nightly-rocm714/t2i-1024-nightly-mathonly.txt) (`flash_sdp: False / mem_efficient_sdp: False / math_sdp: True`). Receipt with full sha256s: [t2i-nightly-receipt.json](../validation/sdpa-gfx11/t2i-nightly-receipt.json). Originals under `outputs/validation/sdpa-gfx11/` (gitignored). Launchers committed: `tests/validation/run_e2e_nightly.py`, `tests/validation/run_e2e_nightly_mathonly.py` (diagnostic).

Honest scope limits: single seed/prompt; probes cover kv ≤ 2048, hd 64/128, 32 heads — not the model's full generation-length configs; per-run rc recorded only for the 2048 run (the others end at `[saved]`, no traceback); exact CLI flags live in this doc, not in the raw logs.

## Verdicts and follow-ups

- **pytorch#194498**: the reported failure mode does not reproduce on the upstream nightly wheel — probes pass and numerics are sound. Worth reporting upstream as a data point; the underlying packaging defect (leaf-wheel metadata) is tracked by ROCm/rocm-systems#10685.
- **Nightly t2i corruption: FAIL — do not use this nightly build for generation on gfx1100.** The understanding/VQA path is fine. Next step if pursued: op-level bisection (hook the DiT forward, hash intermediate activations against the torch 2.8 stack, bisect the first divergent op), then file a new upstream issue with the minimal repro. Until then this repo's guidance stands: torch 2.8.0+rocm6.3 + workarounds (default) or AMD 2.12.0+rocm7.14.0 + gfx11 family wheel for generation **(SUPERSEDED 2026-09-01 — the stable 2.12 wheel also silently corrupts t2i on gfx1100; see the Update section below)**; nightly for understanding only.

## Update (2026-09-01): the "AMD 2.12.0+rocm7.14.0 wheel for generation" guidance above is falsified on gfx1100

The workspace-wide ROCm 7.14 convergence regression (2026-09-01) reproduced the same silent grid corruption on the **stable AMD 2.12.0+rocm7.14.0 wheel stack** (AMD multi-arch index, `[device-gfx1100]` extras, full-stack mode via the common.sh wheel-SDK fix from commit e8467ce): exit 0, valid PNG container, but hcorr 0.510 / block64-corr 0.995 (mean 130.6 / std 30.1) — the identical signature family as the nightly rows above. The corruption is independent of the workaround mode (full-stack LD_PRELOAD+MIOpen on, and the 2026-08-23 zero-workaround run, are both corrupt), while the three historical #194447-family repros (bug1/2/3) and the unit suite all pass on 2.12 — exactly the generation-path-specific pattern established for the nightly, now confirmed on the stable wheel. Conversely, the same prompt/seed/pipeline on the validated torch 2.8.0+rocm6.3 stack rendered byte-identical (sha256 `3285c15f…c00fffe`) to the archived known-good image.

Consequences applied on 2026-09-01: the corrupt 2.12 artifact is preserved at [`outputs/regression-714/corrupt-2.12/`](../../../outputs/regression-714/corrupt-2.12/) (image + run.log + stats); the 2026-08-23 receipt `t2i-torch212-fixed.json` was annotated as archiving a corrupt image (its sha256 is corruption evidence, not success evidence); this repo's generation stack retreated to gen-validated torch 2.8.0+rocm6.3 via the `STACK` knob (commit fcfa48f), with 7.14 remaining the script default for setup/understanding/tests. Corrected guidance: **torch 2.8.0+rocm6.3 + workarounds for generation on gfx1100; 2.12.0+rocm7.14.0 for understanding/tests only; nightly for understanding only.** Upstream report (AMD/pytorch, minimal repro pending op-level bisection) is the open follow-up.
