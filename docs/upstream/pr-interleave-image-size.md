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

One-line-per-call-site change; no behavior change for the paths that
already passed it. One caveat reviewers may want to weigh: in
`interleave_gen_image_only` the per-image size lives in `cur_image_size`
while this passes the `image_size` parameter — identical for the tuple
input `examples/interleave/inference.py` uses (and consistent with the
existing tqdm description, which also uses `image_size`); only a caller
passing a list of per-image tuples would see a difference.

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
