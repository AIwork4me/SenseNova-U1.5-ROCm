# interleave-image-size — V1–V4 verification matrix (2026-08-25)

Follow-up verification for upstream PR
[OpenSenseNova/SenseNova-U1#260](https://github.com/OpenSenseNova/SenseNova-U1/pull/260),
review comment by **yl-1993** (2026-08-24): in `interleave_gen_image_only`
the `_t2i_predict_v` calls should pass `image_size=cur_image_size`
(per-image) instead of the `image_size` parameter, because the function
accepts `image_size` as a `list[tuple]` and the pixel-head branch of
`_t2i_predict_v` expects `image_size[0]/[1]` to be the current `W/H`.

This matrix was produced by
[`scripts/interleave-image-only-check.sh`](../../../scripts/interleave-image-only-check.sh)
(+ `.py`), which calls `interleave_gen_image_only` directly on
SenseNova-U1.5-8B-MoT (`use_pixel_head: true`) — the list-of-tuples input
has no in-repo caller, so the reviewer's scenario needed a direct driver.
One fresh process per scenario; identical recipe everywhere:
`num_steps=5, cfg_scale=1.0, img_cfg_scale=1.0, max_images=2, seed=42,
vram_mode=balanced, attn=sdpa`, sizes from `[(256,256),(320,256)]`
(multiples of `patch_size*merge_size = 32`).

| run | code | `image_size` input | result |
|-----|------|--------------------|--------|
| [v1](v1-prefix-list.json) | PR 80991edd (`c02073d1…`) | list `[(256,256),(320,256)]` | **crash** — `TypeError: unsupported operand type(s) for //: 'tuple' and 'int'` at `modeling_neo_chat.py:614` via the L921 call (first step of first image); tqdm desc prints tuple reprs `image 1 ((256, 256)x(320, 256))` |
| [v2](v2-postfix-list.json) | follow-up `e2f2c865` (`5cd28513…`) | list `[(256,256),(320,256)]` | **ok** — 2 images, shapes `[1,3,256,256]` + `[1,3,256,320]`; tqdm shows per-image `image 1 (256x256)` / `image 2 (320x256)` |
| [v3](v3-prefix-tuple.json) | PR 80991edd (`c02073d1…`) | tuple `(256,256)` | **ok** — 2 images `[1,3,256,256]` (baseline) |
| [v4](v4-postfix-tuple.json) | follow-up `e2f2c865` (`5cd28513…`) | tuple `(256,256)` | **ok** — **byte-identical to v3** (see hashes) |

## Bit-identity receipts (sha256)

| artifact | v3 (pre-fix) | v4 (post-fix) | identical |
|---|---|---|---|
| image 0 raw tensor | `c869cbae695c61b8fc5aaf54a7fca9be4194bb4b4c539f86daaf505da60990d5` | same | ✅ |
| image 0 PNG | `a2e45fa5ee4c19d0840bc135499815a41b1f679804ad039052af9b2f0f08ca76` | same | ✅ |
| image 1 raw tensor | `436a8b7190f40c8e91ac750ef85b35115d71c8c7472fd7edca2f746ae7ed95c1` | same | ✅ |
| image 1 PNG | `b44028bd7a3a7023309f61e4e7181f09c2c51b119519c797770176042df68515` | same | ✅ |

v3 and v4 ran **different code** (per-summary `code.modeling_neo_chat_sha256`
fingerprints differ) yet produced identical bytes — the follow-up commit
changes nothing for tuple input (`cur_image_size` is the same object as
`image_size` there). Consistency bonus: v2's first image (256×256 in list
mode, same first noise draw from the shared seed) is byte-identical to
v3/v4's first image, while its second image (`[1,3,256,320]`) differs —
proving the list is honored per image.

Logs: `../logs/interleave-image-size-v{1,2,3,4}-*.log`
(full v1 traceback included in v1 log + summary).

Environment: AMD gfx1100 (48 GB), ROCm 7.2.1 BLAS-mode preloads,
torch 2.8.0+rocm6.3, MIOpen bypass active; generation wall ≈ 79 s per
success run (small images, 5 steps); peak torch alloc ≈ 3.8 GiB.
