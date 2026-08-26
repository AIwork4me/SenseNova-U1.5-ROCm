# SenseNova-U1.5-ROCm

[![CI](https://github.com/AIwork4me/SenseNova-U1.5-ROCm/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**Run [SenseNova-U1.5-8B-MoT](https://modelscope.cn/models/SenseNova/SenseNova-U1.5-8B-MoT) — the
native unified multimodal model (understanding *and* image generation, one set
of weights) — locally on AMD RDNA3 GPUs via ROCm. One command, fully measured,
every claim linked to a receipt.**

- ✅ **All four tasks validated on gfx1100 (48 GB)** in both run modes —
  BLAS workaround and full-stack ROCm 7.14 (MIOpen on): text-to-image,
  image editing, visual understanding (VQA), and interleaved text+image
  generation — including the `--think` reasoning mode.
- ✅ **48 GB card runs a 50.2 GB bf16 checkpoint** through the upstream
  layer-offload path (`--vram_mode`), with the trade-offs measured, not guessed.
- ✅ **20.6% faster t2i, quality verified at n=36** — `--cfg_interval 0 0.2`
  skips CFG on 56% of steps; paired Qwen-Image-Bench scoring (36 prompts,
  27B judge) shows no detectable loss (+0.34 ± 2.57). The more aggressive
  `-32%` interval was **rejected** by the same expanded validation
  (significant Quality/Aesthetics regression) —
  [receipt](docs/results/validation/cfg-interval/cfg-interval36.json).
- ✅ **SHA256-verified checkpoint** — the fetch script checks every one of the
  24 files against a committed manifest.
- ✅ **Deterministic where it claims to be**: fixed-seed t2i is byte-identical
  across runs (sha256-proven).

```
bash scripts/quickstart.sh          # t2i | vqa | edit | interleave
```

[Quick start](#quick-start) · [Showcase](#showcase-movie-poster-style) ·
[What was validated](#what-was-validated) · [Performance](#performance) ·
[VRAM modes](#vram-modes) ·
[Evidence & receipts](#evidence--receipts) ·
[Troubleshooting](docs/getting-started.md#troubleshooting) ·
[Contributing](CONTRIBUTING.md)

## Hardware coverage

| hardware | status | where |
|---|---|---|
| gfx1100 (RDNA3 dGPU) | fully validated — receipts in [docs/results/](docs/results/) | this repo |
| gfx1151 (Strix Halo APU) | fully validated — evidence in the [8060S repo](https://github.com/AIwork4me/SenseNova-U1.5-ROCm-8060S); profile: [docs/hardware/strix-halo/](docs/hardware/strix-halo/) | this repo + Strix Halo lab |

Other AMD GPUs: the pipeline is hardware-generic (see
[Porting Notes](docs/porting.md)) — validation reports welcome
(issue template `validation_report`).

## Showcase: movie-poster style

Generated on the reference gfx1100 host with this repo's pipeline
(`bash scripts/run-task.sh t2i --jsonl examples/posters-2026-08.jsonl …`,
1664×2496 bucket, 50 steps, ~5 min per poster, 13.4 img-tok/s —
[log](docs/results/logs/posters.log)). Themes are August-2026 box-office
hits; these are **original AI-generated artworks** in poster style — not
official posters, no actor likenesses, no studio logos, no affiliation.

| 《八仙！》 animated fantasy | THE ODYSSEY epic mythology |
|:---:|:---:|
| ![八仙！](docs/results/gallery/posters/baxian.webp) | ![The Odyssey](docs/results/gallery/posters/the-odyssey.webp) |

| 《功夫女足》 sports comedy | 《欢迎来龙餐馆》 fantasy comedy |
|:---:|:---:|
| ![功夫女足](docs/results/gallery/posters/kungfu-girls.webp) | ![欢迎来龙餐馆](docs/results/gallery/posters/dragon-restaurant.webp) |

Title glyphs verified character-by-character (including both Chinese
titles); the 功夫女足 poster was regenerated once to remove a brand-like
mark from the kit and re-verified clean. Every prompt + seed is in
[`examples/posters-2026-08.jsonl`](examples/posters-2026-08.jsonl) —
reproduce with:

```bash
bash scripts/run-task.sh t2i --jsonl examples/posters-2026-08.jsonl \
    --output_dir outputs/posters/ --cfg_scale 4.0 --cfg_norm none \
    --timestep_shift 3.0 --num_steps 50 --profile
```

---

## Why this project

SenseNova-U1.5-8B-MoT (NEO-Unify architecture: a Qwen3-42L backbone with an
integrated flow-matching image head — no separate VAE, no diffusion UNet)
ships CUDA-first tooling: the production serving stack is LightLLM +
LightX2V with FlashAttention-3 kernels. Nothing of that runs on ROCm today.

The upstream `transformers` path, however, is self-contained — and it has a
built-in layer-offload mode designed exactly for cards smaller than the
checkpoint. This project wraps that path into a reproducible pipeline,
validates it end-to-end on a single gfx1100, and publishes the measurements.

**Bonus finding** (worth it even if you never touch this model): the stock
PyTorch rocm6.3 wheel's math stack is broken on gfx1100 — three distinct
native crashes (rocBLAS Tensile segfault, unreadable ROCm 7.x Tensile data,
MIOpen JIT compiler segfault). This project root-causes all three and ships
a two-environment-variable + one-flag workaround with numerics verified
against CPU fp32. Full debugging story:
[`docs/results/findings/rocm63-wheel-blas-on-gfx1100.md`](docs/results/findings/rocm63-wheel-blas-on-gfx1100.md).
We also found and patched an upstream bug that breaks `interleave` on
**all** platforms ([`patches/README.md`](patches/README.md)).

**What this is not:** not a serving stack (no OpenAI-compatible server), not
a quantization project, not training support. It is the reference for
*running the full model, full precision, on one Radeon card* — with numbers
you can check.

## Quick start

Prerequisites: Linux, AMD GPU (RDNA3-class; validated on gfx1100), ROCm
userspace, ~62 GiB free disk, ≥ 64 GiB RAM (the offload path hosts the
47 GiB of bf16 weights in system memory). Full list:
[getting started](docs/getting-started.md).

```bash
git clone https://github.com/AIwork4me/SenseNova-U1.5-ROCm.git && cd SenseNova-U1.5-ROCm
bash scripts/quickstart.sh                 # first run: env + venv (~10 GiB) + model (~50 GiB)
# ... or pick a task:
bash scripts/quickstart.sh vqa             # ask about an image
bash scripts/quickstart.sh edit            # edit the bundled sample
bash scripts/quickstart.sh interleave      # illustrated answer
```

Every stage is idempotent and resumable — interrupt anywhere, re-run the
same command. After the first run, generation starts in seconds:

```bash
bash scripts/run-task.sh t2i --prompt "A cinematic mountain village at sunrise" \
    --width 2048 --height 2048 --seed 42 --output outputs/t2i/village.png
```

Reproducibility of the one-command path itself: the quickstart t2i demo
(same prompt/seed as the validation block, BLAS mode) produced a
**byte-identical PNG, sha256 `39688e22c66059aa…`** — matching the archived
baseline receipt
[`matrix-blas-20260822/t2i.json`](docs/results/validation/matrix-blas-20260822/t2i.json).

## What was validated

On the reference host — **AMD Radeon gfx1100 (48 GB) / ROCm 7.2.1 /
PyTorch 2.8.0+rocm6.3 / transformers 4.57.1 / Python 3.12.3** (full
fingerprint: [`docs/results/environment.json`](docs/results/environment.json)).
Numbers below are the **2026-08-23 full-stack re-validation** — the repo's
default mode (`ROCM_FULL_STACK=auto`: full-stack when an ROCm ≥ 7.14
install is present, BLAS fallback otherwise; ROCm 7.14 MIOpen+comgr+BLAS
preloaded over the torch 2.8 wheel; every receipt tagged
`rocm_stack: full-stack`). The 2026-08-22
BLAS-mode baseline is kept as the comparison value and archived in
[`matrix-blas-20260822/`](docs/results/validation/matrix-blas-20260822/).

| # | Capability | Result | Evidence |
|---|---|---|---|
| 1 | Checkpoint integrity (24 files, 50.23 GB) | ✅ all SHA256-verified | [`configs/artifact-manifest.json`](configs/artifact-manifest.json) |
| 2 | VQA / visual understanding (greedy, 16k-patch image) | ✅ 628 s wall, 25.3 GiB peak — reads a full bilingual menu with prices (BLAS: 602 s) | [`vqa.json`](docs/results/validation/vqa.json) |
| 3 | Text-to-image 2048×2048 @ 50 steps | ✅ 380 s wall (67 s load + 298 s gen + ~15 s overhead ≈ 6.0 s/step), 22.3 GiB (BLAS: 420 s) | [`t2i.json`](docs/results/validation/t2i.json) |
| 4 | T2I with reasoning (`--think`) | ✅ 505 s, structured reasoning text + image (BLAS: 547 s) | [`t2i-think.json`](docs/results/validation/t2i-think.json) |
| 5 | Image editing (it2i) | ✅ 461 s, 29.9 GiB (BLAS: 484 s) | [`edit.json`](docs/results/validation/edit.json) |
| 6 | Interleaved text+image (7-image illustrated tutorial, 2048×1152) | ✅ 3251 s, 47.9 GiB peak (BLAS: 3392 s) | [`interleave.json`](docs/results/validation/interleave.json) |
| 7 | Determinism (same seed twice) | ✅ byte-identical PNG, sha256 `49e9f9b86160…` | [`determinism.json`](docs/results/validation/determinism.json) |
| 8 | Layer-offload modes (10-step probe) | ✅ balanced 207.5 s / fast 210.1 s / low 217.8 s (BLAS: 200.5 / 199.2 / 208.4 s) | [`vram-mode-*.json`](docs/results/validation/) |
| 9 | Upstream interleave bug found & fixed | ✅ minimal patch, all platforms | [`patches/README.md`](patches/README.md) |
| 10 | CFG-interval speedup, quality-verified (n=10) *(superseded by #11)* | ✅ `--cfg_interval 0 0.2`: 295.4→235.7 s/img (−20.2%), n=10 paired-t +0.88 (n.s.); `0.7 1.0` n.s. at n=10 | [`cfg-interval.json`](docs/results/validation/cfg-interval/cfg-interval.json) |
| 11 | CFG-interval **expanded re-validation (n=36, authoritative)** | ✅ `0 0.2`: −20.6%, paired +0.34 ± 2.57 (n.s., all dims n.s.) → **recommended**; `0.7 1.0`: −32.4% but **−6.06 ± 3.85 (t=−3.19, p=0.003; Quality & Aesthetics sig. drop) → retracted** | [`cfg-interval36.json`](docs/results/validation/cfg-interval/cfg-interval36.json) |

Generated images from these runs: [gallery](docs/results/gallery/README.md)
(thumbnail pairs in `determinism-*.webp` are byte-identical too).

## Performance

Single gfx1100, bf16, SDPA attention, `--vram_mode balanced` unless noted.
2048×2048 is the upstream's canonical bucket; 50 steps the default quality.
Wall times are the 2026-08-23 full-stack run; the BLAS-mode baseline
(2026-08-22) is shown for comparison.

| Task | Config | Wall | BLAS baseline | Peak VRAM | Receipt |
|---|---|---:|---:|---:|---|
| T2I | 2048×2048, 50 steps, seed 42 | 380 s | 420 s | 22.3 GiB | `t2i.json` |
| T2I + think | same + reasoning | 505 s | 547 s | 22.3 GiB | `t2i-think.json` |
| Edit | 2048-class, 50 steps | 461 s | 484 s | 29.9 GiB | `edit.json` |
| VQA | greedy, ≤768 new tokens, 16k-patch image | 628 s | 602 s | 25.3 GiB | `vqa.json` |
| Interleave | 7 × 2048×1152 + text | 3251 s | 3392 s | 47.9 GiB | `interleave.json` |

Model load is ~67 s from warm page cache (first-ever load reads 50 GB from
disk). Generation dominates: ~6 s/denoising-step at 2048×2048 (BLAS
baseline ~7 s), rising to ~4.7 s/step late in long interleave runs as the
KV cache grows.

### Faster t2i: `--cfg_interval` (−20.6%, quality-verified at n=36)

The t2i head re-runs the model twice per step for classifier-free guidance
(cond + uncond). The upstream `--cfg_interval LO HI` restricts CFG to
timesteps inside `[LO, HI]` and runs uncond-only elsewhere — on this
compute-bound offload path that directly removes per-step compute.
**The interval is checked against the `timestep_shift=3.0`-warped
t′ = 1 − 3(1−t)/(3−2t)**, not raw t — at 50 steps that warping compresses
step points toward the low-noise end, so intuitions from raw t mislead
(e.g. `(0, 0.7)` skips only 6 of 50 steps; do not use it).

Two validation rounds, same judge (official
[Qwen-Image-Bench](https://github.com/QwenLM/Qwen-Image-Bench), 27B
Qwen3.5 VL, temperature 0, thinking on), same seeds, paired per-prompt:

**n=10 (initial, 2026-08-24/25)** — both intervals looked quality-neutral
(receipt `cfg-interval.json`).

**n=36 (expanded, 2026-08-25/26 — authoritative, receipt
[`cfg-interval36.json`](docs/results/validation/cfg-interval/cfg-interval36.json))**
— 36 prompts (the original 10 kept, generation byte-identical by
determinism, + 26 stratified additions covering all five bench
dimensions; 153 judge tasks per set, 0 parse failures; stats independently
recomputed, two aggregation conventions agree):

| Config | s/img | vs baseline | Paired total delta | Verdict |
|---|---:|---:|---|---|
| baseline (CFG all 50 steps) | 295.9 | — | — | — |
| `--cfg_interval 0 0.2` (CFG on first 22 steps) | 234.8 | **−20.6%** | **+0.34 ± 2.57 (t=+0.26, n.s.)**; all dims n.s. | ✅ **recommended** |
| `--cfg_interval 0.7 1.0` (CFG on last 6 steps) | 200.0 | −32.4% | **−6.06 ± 3.85 (t=−3.19, p=0.003)**; Quality −6.5*, Aesthetics −10.8* | ❌ **retracted: real quality cost** |

The n=10 result for `0.7 1.0` was under-powered (CI ±5.5): the expanded
set resolves its true effect — a significant regression, worst on
aesthetic detail. This is exactly why the recommendation is `0 0.2`:

```bash
bash scripts/run-task.sh t2i --jsonl examples/posters-2026-08.jsonl \
    --output_dir outputs/posters/ --cfg_scale 4.0 --cfg_norm none \
    --timestep_shift 3.0 --num_steps 50 --cfg_interval 0 0.2
```

Caveats: validated at 2048²/50 steps; n=36 detects ~2.6-point total
shifts, so "no loss" means "within ±2.6 at 95% confidence". Judge runs
were local int8 (paired design cancels judge bias). Full data, per-task
outputs, per-dimension scores and both receipts:
[`docs/results/validation/cfg-interval/`](docs/results/validation/cfg-interval/cfg-interval36.json).

### The VRAM story (why `--vram_mode` is the whole game)

The checkpoint's **46.8 GiB of bf16 weights** nearly fill the card's
**48 GB VRAM** (~1.2 GiB to spare) — no headroom for activations, so
`full` mode OOMs by construction. The
upstream offload path streams layers over PCIe from host RAM. Measured
(10-step probe, 2048×2048, seed 42 — receipts `vram-mode-*.json`):

| Mode | How it works | Wall (10 steps) | Peak VRAM |
|---|---|---:|---:|
| `balanced` (default) | async prefetch, H2D overlapped with compute | 207.5 s | 22.3 GiB |
| `fast` | prefetch + retain generation layers in VRAM budget | 210.1 s | 22.3 GiB |
| `low` | synchronous per-layer swap | 217.8 s | **3.7 GiB** |

BLAS-mode baseline for the same probes: 200.5 / 199.2 / 208.4 s
(`low`: 3.39 GiB) — the ~4 % slower probes carry full-stack library
load; `low`'s peak grew by MIOpen's resident footprint.

Practical guidance (measured, see receipts):

- **Interactive single images** → `balanced` (default).
- **Batch of similar-size images** → `fast` (retention amortizes; parity
  with balanced on this host).
- **Small card or sharing the GPU** → `low` — only **~5 % slower** while
  fitting in **~13× less memory than the checkpoint** (3.7 GiB vs its
  46.8 GiB of bf16 weights): the model
  runs on much smaller GPUs than the checkpoint size suggests. On 8–16 GB
  cards, drop resolution/step count too.

## One-command inventory

| Script | What it does |
|---|---|
| `scripts/00-check-env.sh` | verifies ROCm, GPU arch, disk, RAM |
| `scripts/01-setup-venv.sh` | builds `.venv` (ROCm torch 2.8.0, upstream stack, pinned `sensenova_u1`, upstream patches) + GPU smoke test |
| `scripts/02-fetch-model.sh` | downloads + SHA256-verifies the checkpoint (resumable) |
| `scripts/install-rocm-7.14-gfx110x.sh` | optional: AMD TheRock ROCm 7.14 gfx110X userspace for full-stack mode |
| `scripts/run-task.sh` | `t2i \| edit \| vqa \| interleave` dispatcher with validated defaults + ROCm fixes |
| `scripts/quickstart.sh` | the one command (env → venv → model → demo) |
| `scripts/validate.sh` | full validation suite, writes receipts under `docs/results/` |
| `scripts/summarize_results.py` | regenerates the evidence tables from receipts |
| `scripts/make_gallery.py` | builds the webp gallery from validation outputs |

## Evidence & receipts

Every number above traces to a machine-written receipt
(`docs/results/validation/*.json`) capturing the exact command, wall time,
device-level peak VRAM (rocm-smi sampler), and output SHA256 — plus the raw
log (`docs/results/logs/*.log`). Regenerate everything:

```bash
bash scripts/validate.sh                       # ~2–3 h on gfx1100
python3 scripts/summarize_results.py           # prints a table from the receipts
```

The repo's default path is **full-stack mode** (`ROCM_FULL_STACK=auto`,
BLAS fallback when no ROCm ≥ 7.14 install is present): the 2026-08-23 receipts
above all carry `rocm_stack: full-stack` (ROCm 7.14 MIOpen+comgr+BLAS
preloaded over the torch 2.8 wheel, MIOpen enabled, patch 0001 active —
0002 only activates on ROCm torch ≥ 2.9, so 2.8 runs are unaffected).
The 2026-08-22 BLAS-mode baseline is archived in
[`matrix-blas-20260822/`](docs/results/validation/matrix-blas-20260822/);
the torch-2.12 receipts (`*-torch212*.json`) are zero-BLAS-workaround runs.

If your measured numbers differ wildly on the same hardware, open an issue
with your receipts — that's a bug in our claims.

## Known good / known bad

**Known good** (reference host, receipts linked above): all four tasks;
2048×2048@50 generation; think mode; fixed-seed determinism; greedy VQA;
`low` VRAM mode (3.7 GiB peak).

**Known limitations**:

- `--vram_mode full` OOMs by construction on 48 GB (weights leave no
  room for activations).
- No flash-attn on ROCm → SDPA everywhere (`--attn_backend sdpa` forced).
  Per-step latency carries that cost; on CUDA the same code with flash-attn
  is faster.
- **BLAS mode** (no ROCm ≥ 7.14 install found): BLAS calls are routed to
  the system ROCm's hipBLAS/rocBLAS via `LD_PRELOAD` and convolutions run
  through unfold+GEMM (MIOpen bypassed) because the torch-2.8 wheel's
  MIOpen JIT segfaults — `scripts/lib/common.sh` + `src/sensenova_u1_rocm/`,
  opt out with `BLAS_FIX=0`.
- **Full-stack mode** (auto when an ROCm ≥ 7.14 install is present, e.g.
  via `scripts/install-rocm-7.14-gfx110x.sh`): that stack's
  MIOpen+comgr+BLAS is preloaded and MIOpen stays **enabled** — the
  unfold+GEMM penalty disappears. Control with `ROCM_FULL_STACK=0|1`.
- Interleave with many images is the memory-heavy path: 7 × 2048×1152
  peaked at 47.9 GiB (GTT absorbs the spikes — don't try this on a card
  with little GTT headroom).
- VQA decode speed with the full 16k-patch context is ~1 tok/s — the
  offload path re-streams all 42 layers per token. Use smaller images
  (`--max_pixels` upstream flag) for snappier understanding.
- The production LightLLM/LightX2V serving stack is CUDA-only — not part of
  this project.
- AMD's official `torch 2.12.0+rocm7.14.0` wheels fix all three wheel bugs
  with zero workarounds; VQA works out of the box. **Update (2026-08-25):
  the fused-SDPA launch failure is a wheel-metadata bug, root-caused
  upstream (fix PR open)** — `amd-torch-device-gfx1100` was missing its dependency
  on the `amd-torch-device-gfx11` family wheel (AOTriton images); verified
  A/B on this host ([pytorch#194498 comment by
  liminfei-amd](https://github.com/pytorch/pytorch/issues/194498#issuecomment-5406837588),
  fix [ROCm/rocm-systems#10685](https://github.com/ROCm/rocm-systems/pull/10685),
  receipts `docs/results/validation/sdpa-gfx11/`): installing
  `amd-torch-device-gfx11==2.12.0+rocm7.14.0` flips fused SDPA from 0/8
  (hipErrorInvalidValue) to 8/8, and t2i 2048×2048@50 drops to **355 s**
  (1.94× whole-script vs the MATH-patch 687.7 s, 2.22× generation-only)
  with [patches/0002](patches/0002-sdpa-rocm-math-backend-compat.patch)
  removed. On installs that already carry the gfx11 wheel, patch 0002 is
  unnecessary; keep it for broken installs. Baseline receipts:
  `docs/results/validation/t2i-torch212-fixed.json` vs `sdpa-gfx11/`;
  compatibility write-up:
  [SenseNova-U1#261](https://github.com/OpenSenseNova/SenseNova-U1/issues/261).
  torch 2.8 remains the fully validated default stack (all tasks,
  determinism, VRAM modes); on t2i alone, 2.12+gfx11 is now the faster
  path (355 s vs 379.6 s).

## Contributing hardware evidence

Validated on gfx1100 so far. Ran it on another AMD card?
[CONTRIBUTING.md](CONTRIBUTING.md) explains exactly what to paste —
successes and failures both land in `docs/results/`.

## Acknowledgments

- [OpenSenseNova/SenseNova-U1](https://github.com/OpenSenseNova/SenseNova-U1)
  — the model, the inference code this project wraps (pinned at `76c32c2`),
  and the layer-offload machinery that makes 48 GB work.
- AMD's ROCm / PyTorch ROCm wheel teams.

## License

Apache-2.0 (this repo). The wrapped upstream is Apache-2.0; model weights
follow [their license](https://modelscope.cn/models/SenseNova/SenseNova-U1.5-8B-MoT).
