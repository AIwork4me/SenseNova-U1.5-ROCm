# Upstream patches

Patches applied by `scripts/01-setup-venv.sh` on top of the pinned
SenseNova-U1 checkout (`76c32c2`, branch `feat/u1.5`). Each is minimal,
platform-independent, and worth offering upstream.

## 0001-interleave-pass-image_size-to-_t2i_predict_v.patch

**Bug (upstream, affects all platforms):** `NEOChatModel.interleave_gen`
and `interleave_gen_image_only` call `self._t2i_predict_v(...)` without
the `image_size` argument. With U1.5-8B-MoT (`use_pixel_head: true` in
config.json) `_t2i_predict_v` dereferences `image_size[1]` — plain
`TypeError: 'NoneType' object is not subscriptable` before any image is
denoised. The plain `generate()` path (t2i/edit) passes `image_size`
correctly, which is why those tasks work unpatched.

**Fix:** pass `image_size=image_size` at the 10 call sites inside the two
interleave functions (the variable is already in scope — the progress-bar
description uses it a few lines above). **2026-08-25 follow-up (upstream
review by yl-1993, PR#260):** in `interleave_gen_image_only` the five
calls (and the progress-bar description) now pass the per-image
`cur_image_size` instead of the `image_size` parameter — the function
accepts `image_size` as a `list[tuple]`, and forwarding the original
list still crashed the pixel-head branch (`image_size[1] // int` on a
tuple). `interleave_gen` is unaffected: it rebinds `image_size` to the
current tuple inside its loop. Same-seed tuple outputs are byte-identical
before/after the follow-up; list input now completes. Full V1–V4 matrix:
[../docs/results/validation/interleave-image-size/README.md](../docs/results/validation/interleave-image-size/README.md).

**Symptom before the patch** (pre-patch run transcript:
[../docs/results/logs/interleave-prepatch-typeerror.txt](../docs/results/logs/interleave-prepatch-typeerror.txt)):

```
File ".../modeling_neo_chat.py", line 614, in _t2i_predict_v
    token_h = image_size[1] // (self.patch_size * merge_size)
TypeError: 'NoneType' object is not subscriptable
```

## 0002-sdpa-rocm-math-backend-compat.patch

**Bug (ROCm, torch ≥ 2.9; verified on the official torch 2.12.0+rocm7.14.0
wheel, gfx1100 — [pytorch/pytorch#194498](https://github.com/pytorch/pytorch/issues/194498)):**
the image-generation path crashes with `hipErrorInvalidValue` surfacing
at the decoder residual add, far from the real site (the launch error is
deferred to the next checked CUDA call). Root cause, per fresh-process
minimal repros: **both fused SDPA backends (FLASH and mem-efficient) fail
kernel launch for every configuration tested** (bf16; kv 1024–4096,
power-of-two or not; head_dim 64/128; causal or not; contiguous or
transposed; with or without an explicit `scale`). Only the MATH backend
is healthy. The understanding path is spared because its tensor layouts
happen to be ineligible for the fused kernels — explicit masks alone do
NOT prevent the failing dispatch — which is why VQA works unpatched.
(Earlier "scale-dependent"/"shape-dependent" readings were artifacts of
same-process probe contamination by the deferred error.)

**Fix:** in `_sdpa_attn_func`, on ROCm with torch ≥ 2.9 only, restrict
the dispatcher to the MATH backend (proven healthy for every shape
tested) and pre-scale `q` instead of passing `scale`. torch 2.8 and CUDA
hosts keep the stock (fast) behavior.
Cost on the reference host: 2048×2048@50 steps runs 687.7 s vs 420.1 s
on torch 2.8+BLAS-preload (receipt `docs/results/validation/t2i-torch212-fixed.json`)
— a correctness-first trade until the ROCm SDPA backends are fixed
upstream ([pytorch/pytorch#194498](https://github.com/pytorch/pytorch/issues/194498);
compatibility write-up:
[OpenSenseNova/SenseNova-U1#261](https://github.com/OpenSenseNova/SenseNova-U1/issues/261)).
