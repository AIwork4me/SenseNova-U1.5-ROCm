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

**Symptom before the patch** (from
`docs/results/logs/interleave.log` of 2026-08-22):

```
File ".../modeling_neo_chat.py", line 614, in _t2i_predict_v
    token_h = image_size[1] // (self.patch_size * merge_size)
TypeError: 'NoneType' object is not subscriptable
```
