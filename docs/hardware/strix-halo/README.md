# Hardware Profile: Strix Halo (Radeon 8060S iGPU, gfx1151)

verified on: gfx1151 — Ryzen AI MAX+ PRO 395, ROCm 7.2.1 system +
torch 2.12.0+rocm7.14.0 wheels. Full evidence corpus (V1–V8 speedup,
quality bench, determinism, tier tests):
[SenseNova-U1.5-ROCm-8060S](https://github.com/AIwork4me/SenseNova-U1.5-ROCm-8060S).
gfx1100 receipts live in [docs/results/](../../results/) here.
Generic background: [porting notes](../../porting.md).

## Why full bf16 fits a "32 GB" iGPU — GTT spill

On AMD APUs the GPU shares physical memory with the CPU. The driver
exposes two pools: dedicated **VRAM** (BIOS carve-out, e.g. 32 GB on the
reference machine) and **GTT** (the rest of system memory the GPU may
map, e.g. 80 GB). ROCm's allocator can allocate past dedicated VRAM by
spilling into GTT, so `torch.empty(50 GiB, device="cuda")` succeeds.
Since both pools are the same physical LPDDR5X on package, the penalty
is modest — in the 8060S repo's suite runs the allocator put ~1.2 GiB in
VRAM and up to 41.6 GiB in GTT (112 GiB total GPU-addressable), with
full-mode peaks ≤ 42.9 GiB, so ~45+ GiB addressable works in practice.

`scripts/rocm_check.py --alloc-test 48` verifies exactly this spill
(48 GiB allocation; archived as `evidence/stack714/env_check.json` in
the 8060S repo). This is what makes `--vram_mode full` (entire ~50 GB
checkpoint GPU-resident, no layer offload) possible on a "32 GB" GPU.
Note when choosing tiers on the 7.14 stack: `full` is also the fastest
tier there — do not override to `fast`/`balanced` for speed (8060S
PERFORMANCE.md, 2026-08-25 correction; tiers remain the tool for VRAM
capacity on smaller GPUs).

## HSA runtime pitfall (torch +rocm7.0 wheels, driver ≥ 7.1)

Official `torch 2.10.0+rocm7.0` wheels bundle a ROCm 7.0 HSA runtime
that segfaults on gfx1151 (Strix Halo) when the kernel driver is from
ROCm ≥ 7.1: crash in `GpuAgent::QueueCreate` → `ReleaseQueueMainScratch`
while freezing the first code object — every GPU op dies. Fix: preload
the system ROCm runtime
(`LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so`), which always matches
the kernel driver.

This profile ships the shim as [hsa_fix.sh](hsa_fix.sh) — source it
(not execute) and call `hsa_fix_apply "$PY" "$REPO_ROOT"`; it exports
`LD_PRELOAD` only when the system runtime is NEWER than the wheel's
bundled one, so machines where the bundled runtime already works (or is
newer) are untouched. The recommended 7.14 wheels run bare and never
trigger it.

## Wheel guidance for gfx1151

- **Default (recommended):** AMD official multi-arch
  `torch 2.12.0+rocm7.14.0` from
  `https://repo.amd.com/rocm/whl-multi-arch/` — verified bare-run on
  gfx1151 (no HSA preload needed) and faster than the torch-2.10 build
  in the 8060S A/B (167.9 s vs 232.1 s for a 1024×1024 t2i in `fast`
  mode; both sessions were contended, so treat the magnitude as
  indicative). These wheels bundle their own ROCm userspace — a system
  ROCm mainly provides the kernel-driver tooling (`rocminfo`,
  `rocm-smi`).
- **Fallback:** official PyTorch rocm7.0 index
  (`https://download.pytorch.org/whl/rocm7.0`, `torch 2.10.0+rocm7.0`;
  covers gfx942/950/1100/1101/1151/1200) — on gfx1151 it needs the HSA
  preload above.
- **Last resort:** AMD arch-specific nightlies
  (`https://rocm.nightlies.amd.com/v2/<gfx>/`) for arches the official
  wheels miss.

In this repo `scripts/01-setup-venv.sh` installs from a single index —
point `PY_INDEX_ROCM` at the source you want; the deeper
wheel-has-kernels-for-your-gfx check is `scripts/rocm_check.py`
(see [porting notes §4](../../porting.md#4-pytorch-wheel-matrix-known-rocm-wheel-sources)).

## AOTriton caveat (hd=72 ViT heads)

On gfx1151 with torch 2.12+rocm7.14, AOTriton flash SDPA is *silently
wrong* for head_dim=72 with seq>1024, **non-causal only** (causal
attention at the same shape is correct) — errors drift run-to-run,
typically 16–730% relative error, dominantly finite garbage with
sporadic NaN in ~1/5 of calls. SenseNova's own attention shapes
(head_dim 64/128) are unaffected, but disable AOTriton for a foreign
model with a 16-head/1152-hidden ViT. Root cause: upstream AOTriton
[issue #54](https://github.com/ROCm/aotriton/issues/54) (missing
`out_dtype=tl.float32` in the fwd kernel); a fix exists (commit
`8232d69672`) but is unmerged in every release so far — tracked in
[docs/upstream/aotriton-54.md](../../upstream/aotriton-54.md).

## Companion repo

The daily-driver run scripts, one-command quickstart, and the full
evidence corpus live in the
[8060S repo](https://github.com/AIwork4me/SenseNova-U1.5-ROCm-8060S) —
the Strix Halo lab where these numbers were measured. Experiments there
graduate into this umbrella via PR when they meet the graduation
criteria ([CONTRIBUTING.md](../../../CONTRIBUTING.md)).
