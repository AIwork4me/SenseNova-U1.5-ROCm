# SenseNova-U1.5-ROCm

[![CI](https://github.com/AIwork4me/SenseNova-U1.5-ROCm/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**Run [SenseNova-U1.5-8B-MoT](https://modelscope.cn/models/SenseNova/SenseNova-U1.5-8B-MoT) — the
native unified multimodal model (understanding *and* image generation, one set
of weights) — locally on AMD RDNA3 GPUs via ROCm. One command, fully measured,
every claim linked to a receipt.**

- ✅ **All four tasks validated on gfx1100 (48 GB, ROCm 7.2.1)**: text-to-image,
  image editing, visual understanding (VQA), and interleaved text+image
  generation — including the `--think` reasoning mode.
- ✅ **48 GB card runs a 50.2 GB bf16 checkpoint** through the upstream
  layer-offload path (`--vram_mode`), with the trade-offs measured, not guessed.
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
(same prompt/seed as the validation block) produced a **byte-identical
PNG, sha256 `39688e22c66059aa…`** — matching
[`t2i.json`](docs/results/validation/t2i.json).

<!-- VALIDATION_RESULTS: filled from docs/results/ by scripts/summarize_results.py -->

## What was validated

On the reference host — **AMD Radeon gfx1100 (48 GB) / ROCm 7.2.1 /
PyTorch 2.8.0+rocm6.3 / transformers 4.57.1 / Python 3.12.3** (full
fingerprint: [`docs/results/environment.json`](docs/results/environment.json)):

| # | Capability | Result | Evidence |
|---|---|---|---|
| 1 | Checkpoint integrity (24 files, 50.23 GB) | ✅ all SHA256-verified | [`configs/artifact-manifest.json`](configs/artifact-manifest.json) |
| 2 | VQA / visual understanding (greedy, 16k-patch image) | ✅ 602 s wall, 24.8 GiB peak — reads a full bilingual menu with prices | [`vqa.json`](docs/results/validation/vqa.json) |
| 3 | Text-to-image 2048×2048 @ 50 steps | ✅ 420 s wall (65.6 s load + 348 s gen ≈ 7.0 s/step), 22.3 GiB | [`t2i.json`](docs/results/validation/t2i.json) |
| 4 | T2I with reasoning (`--think`) | ✅ 547 s, structured reasoning text + image | [`t2i-think.json`](docs/results/validation/t2i-think.json) |
| 5 | Image editing (it2i) | ✅ 484 s, 29.8 GiB | [`edit.json`](docs/results/validation/edit.json) |
| 6 | Interleaved text+image (7-image illustrated tutorial, 2048×1152) | ✅ 3392 s, 47.7 GiB peak | [`interleave.json`](docs/results/validation/interleave.json) |
| 7 | Determinism (same seed twice) | ✅ byte-identical PNG, sha256 `d24ae824c575…` | [`determinism.json`](docs/results/validation/determinism.json) |
| 8 | Layer-offload modes (10-step probe) | ✅ balanced 200.5 s / fast 199.2 s / low 208.4 s | [`vram-mode-*.json`](docs/results/validation/) |
| 9 | Upstream interleave bug found & fixed | ✅ minimal patch, all platforms | [`patches/README.md`](patches/README.md) |

Generated images from these runs: [gallery](docs/results/gallery/README.md)
(thumbnail pairs in `determinism-*.webp` are byte-identical too).

## Performance

Single gfx1100, bf16, SDPA attention, `--vram_mode balanced` unless noted.
2048×2048 is the upstream's canonical bucket; 50 steps the default quality.

| Task | Config | Wall | Peak VRAM | Receipt |
|---|---|---:|---:|---|
| T2I | 2048×2048, 50 steps, seed 42 | 420 s | 22.3 GiB | `t2i.json` |
| T2I + think | same + reasoning | 547 s | 22.3 GiB | `t2i-think.json` |
| Edit | 2048-class, 50 steps | 484 s | 29.8 GiB | `edit.json` |
| VQA | greedy, ≤768 new tokens, 16k-patch image | 602 s | 24.8 GiB | `vqa.json` |
| Interleave | 7 × 2048×1152 + text | 3392 s | 47.7 GiB | `interleave.json` |

Model load is ~66 s from warm page cache (first-ever load reads 50 GB from
disk). Generation dominates: ~7 s/denoising-step at 2048×2048, rising to
~5.5 s/step late in long interleave runs as the KV cache grows.

### The VRAM story (why `--vram_mode` is the whole game)

The checkpoint is **50.23 GB bf16**; the reference card has **48 GB VRAM**.
`full` mode cannot even hold the weights, let alone activations. The
upstream offload path streams layers over PCIe from host RAM. Measured
(10-step probe, 2048×2048, seed 42 — receipts `vram-mode-*.json`):

| Mode | How it works | Wall (10 steps) | Peak VRAM |
|---|---|---:|---:|
| `balanced` (default) | async prefetch, H2D overlapped with compute | 200.5 s | 22.3 GiB |
| `fast` | prefetch + retain generation layers in VRAM budget | 199.2 s | 22.3 GiB |
| `low` | synchronous per-layer swap | 208.4 s | **3.4 GiB** |

Practical guidance (measured, see receipts):

- **Interactive single images** → `balanced` (default).
- **Batch of similar-size images** → `fast` (retention amortizes; parity
  with balanced on this host).
- **Small card or sharing the GPU** → `low` — only **4 % slower** while
  using 15× less VRAM (3.4 GiB): the model runs on much smaller GPUs than
  the checkpoint size suggests. On 8–16 GB cards, drop resolution/step
  count too.

## One-command inventory

| Script | What it does |
|---|---|
| `scripts/00-check-env.sh` | verifies ROCm, GPU arch, disk, RAM |
| `scripts/01-setup-venv.sh` | builds `.venv` (ROCm torch 2.8.0, upstream stack, pinned `sensenova_u1`, upstream patches) + GPU smoke test |
| `scripts/02-fetch-model.sh` | downloads + SHA256-verifies the checkpoint (resumable) |
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
python3 scripts/summarize_results.py           # prints the tables
```

If your measured numbers differ wildly on the same hardware, open an issue
with your receipts — that's a bug in our claims.

## Known good / known bad

**Known good** (reference host, receipts linked above): all four tasks;
2048×2048@50 generation; think mode; fixed-seed determinism; greedy VQA;
`low` VRAM mode (3.4 GiB peak).

**Known limitations**:

- `--vram_mode full` OOMs by construction on 48 GB (checkpoint > VRAM).
- No flash-attn on ROCm → SDPA everywhere (`--attn_backend sdpa` forced).
  Per-step latency carries that cost; on CUDA the same code with flash-attn
  is faster.
- Convolutions run through unfold+GEMM (`torch.backends.cudnn.enabled=False`,
  applied by `src/sensenova_u1_rocm/`) because the wheel's MIOpen JIT
  segfaults (see findings doc). Opt out with `SENU15_MIOPEN=1`.
- BLAS calls are routed to the **system** ROCm's hipBLAS/rocBLAS via
  `LD_PRELOAD` (applied by `scripts/lib/common.sh`); disable with
  `BLAS_FIX=0` on hosts where the wheel works as-is.
- Interleave with many images is the memory-heavy path: 7 × 2048×1152
  peaked at 47.7 GiB (GTT absorbs the spikes — don't try this on a card
  with little GTT headroom).
- VQA decode speed with the full 16k-patch context is ~1 tok/s — the
  offload path re-streams all 42 layers per token. Use smaller images
  (`--max_pixels` upstream flag) for snappier understanding.
- The production LightLLM/LightX2V serving stack is CUDA-only — not part of
  this project.

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
