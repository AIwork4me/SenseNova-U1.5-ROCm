# Fix `interleave_gen` / `interleave_gen_image_only`: pass `image_size` to `_t2i_predict_v`

## Summary

`NEOChatModel.interleave_gen` and `interleave_gen_image_only` call
`self._t2i_predict_v(...)` **without the `image_size` argument** (10 call
sites). With `use_pixel_head: true` — i.e. SenseNova-U1.5-8B-MoT —
`_t2i_predict_v` dereferences `image_size[1]` in the pixel-head branch:

```python
# modeling_neo_chat.py, _t2i_predict_v
if self.use_pixel_head:
    merge_size = int(1 / self.downsample_ratio)
    token_h = image_size[1] // (self.patch_size * merge_size)   # <- image_size is None
```

so every interleave run crashes before the first image is denoised:

```
File ".../modeling_neo_chat.py", line 1266, in interleave_gen
    out_cond = self._t2i_predict_v(image_embeds, indexes_image_condition, ..., timestep_embeddings=timestep_embeddings)
File ".../modeling_neo_chat.py", line 614, in _t2i_predict_v
    token_h = image_size[1] // (self.patch_size * merge_size)
TypeError: 'NoneType' object is not subscriptable
```

This is platform-independent (reproduces on AMD ROCm; nothing in the path
is CUDA-specific). The plain `generate()` path (t2i / editing) already
passes `image_size` correctly, which is why those tasks work unpatched.

## Fix

Pass `image_size=image_size` at the 10 call sites inside the two interleave
functions. The variable is already in scope in both — the progress-bar
description uses it a few lines above each call:

```python
step_iter = _tqdm(step_iter, desc=f"image {img_count + 1} ({image_size[0]}x{image_size[1]})", ...)
```

## Review follow-up (2026-08-24/25, yl-1993) — APPLIED

Reviewer yl-1993 confirmed the direction but caught one real issue
([comment](https://github.com/OpenSenseNova/SenseNova-U1/pull/260#issuecomment-5399345254)):
in `interleave_gen_image_only` the calls should pass the per-image
`cur_image_size`, not the `image_size` parameter — the function accepts
`image_size` as a `list[tuple]` (modeling_neo_chat.py:691-698), and with a
list the pixel-head branch still crashes (`image_size[1]` is then a tuple).
`interleave_gen` was already correct (rebinds `image_size` per loop).

Follow-up commit `e2f2c865` ("interleave_gen_image_only: pass
cur_image_size to _t2i_predict_v", pushed 2026-08-25): the five call sites
plus the progress-bar description (which printed tuple reprs for list
input) now use `cur_image_size`. CI green, PR MERGEABLE; PR #260 remains
open, awaiting maintainer merge.

Verified on the real model (U1.5-8B-MoT, pixel head, gfx1100/ROCm) with a
direct driver — the list input has no in-repo caller:

| code | `image_size` | result |
|------|--------------|--------|
| PR 80991edd | list `[(256,256),(320,256)]` | `TypeError: unsupported operand type(s) for //: 'tuple' and 'int'` at L614 via the L921 call (first step of first image) |
| follow-up `e2f2c865` | same list | completes; images `[1,3,256,256]` + `[1,3,256,320]`, per-image tqdm |
| PR 80991edd | tuple `(256,256)` | completes (baseline) |
| follow-up `e2f2c865` | same tuple | **byte-identical** to baseline (raw+PNG sha256) |

Full matrix and hashes:
[../results/validation/interleave-image-size/README.md](../results/validation/interleave-image-size/README.md).
Local patch `patches/0001` regenerated to match the new PR head (applies
cleanly on pinned `76c32c2`).

## Validation

On a single AMD gfx1100 (48 GB, ROCm 7.2.1, torch 2.8.0+rocm6.3) with this
patch applied, `examples/interleave/inference.py` completes end-to-end:
a tomato-and-egg stir-fry tutorial produced interleaved text plus **7**
2048×1152 images in one response (3392 s wall, 47.7 GiB peak device
memory). Full receipts and logs:
https://github.com/AIwork4me/SenseNova-U1.5-ROCm (see
`docs/results/validation/interleave.json` and `patches/README.md`).

Without the patch the same script fails with the `TypeError` above (short repro: prompt "hello", 5 steps) —
pre-patch transcript:
[interleave-prepatch-typeerror.txt](https://github.com/AIwork4me/SenseNova-U1.5-ROCm/blob/main/docs/results/logs/interleave-prepatch-typeerror.txt).
