# Porting Notes: SenseNova-U1.5 on AMD GPUs (ROCm)

How the umbrella repo runs SenseNova-U1.5-8B-MoT on AMD GPUs, and why it
stays thin. Hardware-specific depth lives in
[docs/hardware/](hardware/) profiles; evidence-first claims live in
[docs/results/](results/).

**TL;DR: the upstream code is already ROCm-ready** (one hardware-independent
interleave hotfix aside — §6). The NEO-unify reference implementation is
written against the `torch.cuda` namespace (which ROCm PyTorch provides) and
falls back to PyTorch SDPA when flash-attn is absent. The real work is
(1) getting PyTorch ROCm wheels that contain kernels for your gfx target,
and (2) fitting a ~50 GB bf16 checkpoint into your GPU's memory budget.
That's what `scripts/01-setup-venv.sh` and the `--vram_mode` flag take
care of.

## 1. What "ROCm support" means in torch

ROCm PyTorch wheels re-expose the GPU as the `cuda` device type:

- `torch.cuda.is_available()` → `True`
- `device="cuda"` / `.to("cuda")` → runs on the AMD GPU
- `torch.version.hip` → the ROCm version the wheel was built against

So any code that sticks to the `torch.cuda` API (like SenseNova-U1's
`src/sensenova_u1/utils/accel.py`, whose docstring explicitly says
"CUDA (incl. ROCm via the cuda namespace)") works unmodified.

The one thing that does **not** carry over automatically is compiled
extension coverage: each wheel only contains GPU kernels for the gfx
architectures it was built for. `scripts/rocm_check.py` verifies your wheel
actually has kernels for your GPU before you waste an hour debugging a
`HIP error` deep inside model loading.

## 2. Attention: SDPA instead of flash-attn

Upstream prefers flash-attn (`--attn_backend auto` → flash if importable).
There is no flash-attn build for most ROCm gfx targets (including gfx1151).
Upstream's `auto` therefore selects **PyTorch SDPA** on ROCm, and this
repo's run scripts pin `--attn_backend sdpa` explicitly for determinism of
behavior.

On ROCm, SDPA dispatches to an AOTriton flash kernel, an MIOpen fused
kernel, or the math fallback — all correct bf16 attention for the shapes
this model uses.

Backend selection and determinism notes: see the per-hardware profile pages.

## 3. Fitting a 50 GB checkpoint into smaller memories

The bf16 checkpoint is ~50 GB. Two complementary mechanisms are used:

### a) Unified-memory GTT spill (APUs)

Unified-memory GTT spill on APUs: see
[docs/hardware/strix-halo/README.md](hardware/strix-halo/README.md).

### b) Layer offload tiers (any GPU)

Upstream ships a tiered layer-offload engine
(`sensenova_u1/utils/layer_offload.py`) exposed as `--vram_mode`:

| vram_mode | strategy | suitable GPU memory |
|---|---|---|
| `full` | everything GPU-resident | ~52+ GiB discrete VRAM; on APUs the VRAM+GTT pool counts — see the [strix-halo profile](hardware/strix-halo/README.md) |
| `fast` | most layers resident; generation layers swapped per step | ~24 GB |
| `balanced` | heavier offload with async prefetch | ~16-24 GB |
| `low` | most aggressive offload | ~12-16 GB |

In this repo `scripts/run-task.sh` injects `--vram_mode` for you and
defaults to `balanced`; select another tier with the `VRAM_MODE`
environment variable or by passing `--vram_mode` yourself.

## 4. PyTorch wheel matrix (known ROCm wheel sources)

| source | index | covers |
|---|---|---|
| **AMD official multi-arch** | `https://repo.amd.com/rocm/whl-multi-arch/` (`torch 2.12.0+rocm7.14.0`) | multi-gfx family wheels; gfx1100 and gfx1151 covered — gfx1151 runs without the HSA preload (verified on: gfx1151; see the [strix-halo profile](hardware/strix-halo/README.md)) |
| Official PyTorch ROCm 7 | `https://download.pytorch.org/whl/rocm7.0` (`torch 2.10.0+rocm7.0`) | gfx942, gfx950, gfx1100/1101, gfx1151, gfx1200 — gfx1151 needs the HSA preload (see the [strix-halo profile](hardware/strix-halo/README.md)) |
| AMD arch-specific nightlies | `https://rocm.nightlies.amd.com/v2/<gfx>/` | fallback for arches the official wheels miss |

`scripts/01-setup-venv.sh` installs from a single index — `PY_INDEX_ROCM`
overrides; the default is the official PyTorch rocm6.3 index (`torch
2.8.0+rocm6.3`, the stack validated on gfx1100 in this repo) — and ends
with a GPU smoke test. `scripts/rocm_check.py` is the deeper check that
your wheel carries kernels for your gfx target. Validated stacks are
recorded with receipts under [docs/results/](results/).

## 5. A real pitfall: the wheel's bundled HSA runtime (gfx1151)

This pitfall is gfx1151-specific — see
[docs/hardware/strix-halo/README.md](hardware/strix-halo/README.md).

## 6. The one upstream change: interleave hotfix

Everything in the model code runs unmodified **except one upstream bugfix
touching 10 `_t2i_predict_v` call sites** we carry as a patch
([patches/0001-interleave-pass-image_size-to-_t2i_predict_v.patch](../patches/0001-interleave-pass-image_size-to-_t2i_predict_v.patch)):

- `_t2i_predict_v()` dereferences `image_size` when the pixel head is
  enabled (`use_pixel_head=true` in the released checkpoint), but the
  interleave generation loops (`interleave_gen`,
  `interleave_gen_image_only`) never pass it →
  `TypeError: 'NoneType' object is not subscriptable` on the first
  generated image. `t2i_generate` already passes it correctly.
- The bug is **hardware-independent** (the same code path exists on CUDA;
  reproduced independently on gfx1100 and gfx1151) — see upstream
  [PR #260](https://github.com/OpenSenseNova/SenseNova-U1/pull/260)
  (open; local copy:
  [docs/upstream/pr-interleave-image-size.md](upstream/pr-interleave-image-size.md)),
  which also carries an independent cross-platform confirmation. This
  repo's 0001 is regenerated to that PR's head — the reviewed version that
  passes the per-image `cur_image_size` inside `interleave_gen_image_only`.
- `scripts/01-setup-venv.sh` applies the patch automatically after cloning
  the pinned upstream checkout (`git apply`, idempotent — see
  [patches/README.md](../patches/README.md)).

If upstream merges the fix, delete the patch file and the setup step.

## 7. What this repository changes vs upstream

Almost nothing in the model code: three small patches, all documented in
[patches/README.md](../patches/README.md) — the interleave hotfix of §6
(0001), an SDPA math-backend fallback for installs whose fused-SDPA wheels
are broken (0002, gfx1100 leaf wheels), and an opt-in
torch.compile/cudagraph acceleration (0003, off by default).
`scripts/01-setup-venv.sh`:

1. installs ROCm PyTorch itself (default: the official PyTorch rocm6.3
   index; `PY_INDEX_ROCM` overrides — see §4) — upstream pins
   `torch==2.8.0` on a cu128 index, an NVIDIA build, bypassed with
   `pip install -e ... --no-deps`;
2. installs upstream from git at a **pinned commit** for reproducibility
   and applies the patches;
3. wraps the four upstream example CLIs (`t2i`, `editing`, `vqa`,
   `interleave`) with ROCm-verified defaults
   ([scripts/run-task.sh](../scripts/run-task.sh)).

If upstream ever ships ROCm wheels of its own, this repo shrinks to just
the verification harness — by design.
