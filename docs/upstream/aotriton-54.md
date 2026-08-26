# Upstream: AOTriton flash SDPA silently wrong for head_dim=72, seq>1024, non-causal (ROCm/aotriton#54 — fix unmerged)

verified on: gfx1151 (Strix Halo / Radeon 8060S); no gfx1100 receipt yet.
The defect is kernel-level, so any host whose torch wheel bundles the
unfixed AOTriton release line is presumed affected until proven otherwise.

## Summary

The AOTriton flash SDPA forward kernel returns **silently wrong output**
for one narrow but real configuration class — no launch error, no
exception, just garbage (worse than a crash, because nothing surfaces):

- head_dim = **72** exactly (64/80/88/96/104/112/120/128 all measured OK)
- seq_len **> 1024** (1024 OK; 1025 already fails)
- **non-causal** attention only (causal at the same shape is correct —
  a different tile configuration)

This is the shape of typical ViT patch attention: the Qwen-Image-Bench
judge ViT is 16 heads × hidden 1152 (= head_dim 72) and a 1024² image
yields 4096 patches — squarely inside the envelope. SenseNova's own
attention shapes (head_dim 64/128) are **unaffected**; that is why
AOTriton-enabled SenseNova runs stay correct while the judge ViT sharing
the same venv produced garbage ("乱码") — the observation that opened this
investigation.

Note the contrast with the gfx1100 fused-backend *launch* failure tracked
in [issue-pytorch-rocm-sdpa-backends.md](issue-pytorch-rocm-sdpa-backends.md):
this kernel launches fine and lies.

## Symptoms (speedup2 re-verification, 2026-08-25/26)

- relative error vs the MATH backend (bf16, b=1/16 heads):
  **1.6e-01 … 7.3e+00 (≈16–730%)** — dominantly *finite* garbage, not NaN
- error magnitude **drifts run-to-run even at fixed seed** (single-run
  snapshots are not an invariant; one re-verification capped hd72@4096 at
  4.9e-01)
- **sporadic NaN in ~1/5 of calls** (5 consecutive same-input calls:
  [3.9e-1, nan, 5.5e-1, 1.4e-1, 3.4e-1]) — "not NaN" is not an invariant;
  report as interval + occasional NaN
- the **LSE itself is corrupted** (softmax denominator polluted); errors
  spread across the whole sequence
- reproduces via direct `aten::_scaled_dot_product_flash_attention` —
  kernel-level, not an SDPA dispatch artifact
- fp16 also affected; bad across the whole batch × heads matrix
  (occupancy-independent — split-KV hypothesis excluded)

## Root cause and upstream status

Upstream [ROCm/aotriton issue #54](https://github.com/ROCm/aotriton/issues/54)
(open since 2024-11): the fwd kernel's
`qk += (Qk_scale * tl.dot(q0, k0))` lacks an explicit
`out_dtype=tl.float32`, so on gfx1151 the dot result overflows / drops
mantissa in bf16/fp16 before the scale multiply → garbage or NaN.
head_dim 72 is the size that takes the BLOCK_DMODEL tiled-dot path
(the `BLOCK_DMODEL1>0` branch) — the necessary condition to reach the
defective line.

- Fix commit `8232d69672` (2026-06-01, branch
  `fix/gfx1151-bf16-dot-fp32-accumulator`) — **still unmerged**: main and
  every release tag 0.11.2b…0.13.50tp carry the unfixed code; the tested
  wheel bundles `libaotriton_v2.so.0.11.2` (unfixed). The fix's commit
  notes cite the BLOCK_M=128/BLOCK_N=64 trigger, matching the measured
  envelope.
- Production repro + failure envelope + causal control reported upstream
  on 2026-08-26
  ([issuecomment-5420432049](https://github.com/ROCm/aotriton/issues/54#issuecomment-5420432049)),
  requesting that the fix branch be merged and released. This is an
  upstream defect — nothing to patch in this repo; this entry tracks it
  until a release carries `8232d69672`.

## Cross-stack verification (gfx1151)

| stack | hd=72 result |
|---|---|
| TheRock `torch 2.10.0+rocm7.13.0a20260513` (HIP 7.13.26183, vLLM venv) | seq ≥1088 BAD (1.6e-01…7.3e+00); hd 64/128 and hd72@1024 OK |
| AMD official `torch 2.12.0+rocm7.14.0` (HIP 7.14.60850) | seq 1088/2048/4096 → rel err 1.9e-1 / 3.2e-1 / 9.1e+0, all BAD; hd 64/128 and hd72@1024 OK |

The defect spans at least ROCm 7.13-alpha (TheRock) through 7.14.0
(official). (The V2 note initially mis-attributed the repro to
2.12+rocm7.14; corrected 2026-08-26 — attribution above is final.)

## Workaround until fixed upstream

SenseNova's own attention (head_dim 64/128) needs no action. For
**foreign models with other head dims (hd≠64/128 — classically a
16-head/1152-hidden ViT)** run in a venv where AOTriton is enabled:

- **disable AOTriton for that model / pin SDPA to the MATH backend**
  (the 8060S repo's judge service keeps AOTriton off as its standing
  policy; the run scripts there disable it with `SENSENOVA_NO_AOTRITON=1`)
- do not treat "no NaN" as evidence of health — finite garbage dominates

## Evidence

- Verification chain (V2 root-cause closure, failure envelope, upstream
  filing; commits `d0af6efc` / `987015a1` / `28e4f3af`):
  [evidence/speedup2/VERIFICATION.md §V2](https://github.com/AIwork4me/SenseNova-U1.5-ROCm-8060S/blob/master/evidence/speedup2/VERIFICATION.md)
  (directory:
  [evidence/speedup2/](https://github.com/AIwork4me/SenseNova-U1.5-ROCm-8060S/tree/master/evidence/speedup2))
- User-facing caveat (exact guidance quoted above):
  [docs/TROUBLESHOOTING.md §5](https://github.com/AIwork4me/SenseNova-U1.5-ROCm-8060S/blob/master/docs/TROUBLESHOOTING.md)
