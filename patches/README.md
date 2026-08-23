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
description uses it a few lines above).

**Symptom before the patch** (pre-patch run transcript:
[../docs/results/logs/interleave-prepatch-typeerror.txt](../docs/results/logs/interleave-prepatch-typeerror.txt)):

```
File ".../modeling_neo_chat.py", line 614, in _t2i_predict_v
    token_h = image_size[1] // (self.patch_size * merge_size)
TypeError: 'NoneType' object is not subscriptable
```

## 0002-sdpa-rocm-math-backend-compat.patch

**Bug (ROCm, torch ≥ 2.9 SDPA backend bugs; reproduced on the official
torch 2.12.0+rocm7.14.0 wheel, gfx1100):** the image-generation path
crashes with `hipErrorInvalidValue` surfacing at the decoder residual add,
far from the real site. Root cause (minimal repros in
`docs/results/findings/`):

1. the **FLASH** SDPA backend fails launch whenever an explicit `scale`
   kwarg is passed (the model always passes `scale = 1/sqrt(head_dim)`);
2. the **mem-efficient** SDPA backend fails launch for several shapes on
   this stack (`kv_len` not a power of two — e.g. the generation path's
   `kv=1281`; `head_dim=64`; long causal sequences).

The understanding path never hits either (transformers' standard SDPA
path with explicit masks), which is why VQA works unpatched.

**Fix:** in `_sdpa_attn_func`, on ROCm only, restrict the dispatcher to
the MATH backend (proven healthy for every shape tested) and pre-scale
`q` instead of passing `scale`. CUDA hosts keep the stock behavior.
Cost on the reference host: 2048×2048@50 steps runs 687.7 s vs 420.1 s
on torch 2.8+BLAS-preload (receipt `docs/results/validation/t2i-torch212-fixed.json`)
— a correctness-first trade until the ROCm SDPA backends are fixed
upstream (pytorch/pytorch issue linked in the findings doc).
