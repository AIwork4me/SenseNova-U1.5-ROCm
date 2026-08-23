# Getting started

From a bare Linux host with an AMD GPU to your first SenseNova-U1.5 output.

## What you need

| Requirement | Why | Reference host |
|---|---|---|
| AMD GPU, RDNA3-class (gfx1100 tested) | compute | Radeon W7900-class, 48 GB |
| ROCm ≥ 6.x userspace + `/dev/kfd` access | HIP runtime | ROCm 7.2.1 at `/opt/rocm` |
| Python 3.10–3.13 | inference stack | 3.12.3 |
| ~62 GiB free disk | 10 GiB venv + 51 GiB checkpoint | — |
| ≥ 64 GiB host RAM (128+ better) | layer offload streams the 47 GiB bf16 weights through host memory | 1 TiB |

The checkpoint is bf16 and **larger than the reference card's 48 GB VRAM**,
so all tasks run through the upstream layer-offload path (`--vram_mode`).
More VRAM still helps: `fast` mode pins the generation layers that fit.

## The one-command path

```bash
git clone <this-repo> && cd SenseNova-U1.5-ROCm
bash scripts/quickstart.sh            # t2i demo; also: vqa | edit | interleave
```

Stages (idempotent — safe to re-run after any interruption):

1. `00-check-env.sh` — verifies ROCm, GPU, disk, RAM; prints what's missing.
2. `01-setup-venv.sh` — builds `.venv/`: ROCm PyTorch 2.8.0, the upstream
   inference stack, and `sensenova_u1` (editable, from `third_party/`,
   auto-cloned and pinned to `76c32c2`).
3. `02-fetch-model.sh` — downloads the 24-file / 50.2 GiB checkpoint and
   verifies every file against `configs/artifact-manifest.json` (SHA256).
   Resumable; re-runs only verify.
4. `run-task.sh` — runs the demo task and prints where the output landed.

## Running each task

```bash
# text-to-image (2048x2048 @ 50 steps by default is what we validate)
bash scripts/run-task.sh t2i --prompt "A cinematic mountain village at sunrise" \
    --width 2048 --height 2048 --seed 42 --output outputs/t2i/village.png

# ask about an image (visual understanding)
bash scripts/run-task.sh vqa --image path/to/img.jpg --question "what is on the table?"

# edit an image
bash scripts/run-task.sh edit --image path/to/img.jpg --prompt "make it snow"

# interleaved text+image answer (illustrated tutorial etc.)
bash scripts/run-task.sh interleave --prompt "give me an illustrated pancake tutorial"
```

`run-task.sh` injects `--model_path`, `--vram_mode` and `--attn_backend sdpa`
before your flags — pass any upstream flag after the task name to override
(see the upstream
[examples README](https://github.com/OpenSenseNova/SenseNova-U1/blob/feat/u1.5/examples/README.md)
for the full flag surface: LoRA, think mode, JSONL batching, CFG, ...).

## Knobs that matter

| Knob | Default | Meaning |
|---|---|---|
| `VRAM_MODE` | `balanced` | `full` (needs > 52 GB VRAM), `fast` (prefetch + retain gen layers), `balanced` (async prefetch), `low` (sync swap, smallest VRAM) |
| `--num_steps` | 50 | denoising steps; 20 halves time, costs detail |
| `--width/--height` | 2048×2048 | trained buckets: 1:1, 16:9, 3:2, 4:3, 2:1, 3:1 (+ rotations) |
| `MODEL_DIR` | `$HF_HOME/modelscope/SenseNova-U1.5-8B-MoT` | checkpoint location |
| `ROCM_FULL_STACK` | `auto` | with a ROCm ≥ 7.14 install present (see `scripts/install-rocm-7.14-gfx110x.sh`), preload its MIOpen+comgr+BLAS and keep MIOpen enabled; `0` = BLAS-only workaround |

**torch flavors**: the validated full-task path is `torch 2.8.0+rocm6.3`
+ this repo's workarounds. AMD's official `torch 2.12.0+rocm7.14.0`
wheels (`pip ... --extra-index-url https://repo.amd.com/rocm/whl-multi-arch/`)
need **zero workarounds and fix all three wheel bugs**, and the
understanding/VQA path works — but image generation (`forward_gen`)
currently fails on torch 2.12 (upstream code pins torch 2.8). Details:
[findings](results/findings/rocm63-wheel-blas-on-gfx1100.md).

## Where things land

```
outputs/<task>/...            generated images / answers
docs/results/validation/      one JSON receipt per validation block
docs/results/logs/            raw stdout of every measured run
docs/results/environment.json torch / ROCm / GPU fingerprint
```

## Troubleshooting

- **`/dev/kfd not writable`** — add yourself to the `render` + `video`
  groups (`sudo usermod -aG render,video $USER`), re-login.
- **OOM during load** — you chose `VRAM_MODE=full` on a < 52 GB card;
  don't. Use `balanced` (default).
- **First run feels hung** — the model load reads 50 GiB from disk and the
  first image runs MIOpen kernel autotuning; watch `rocm-smi` / give it a
  few minutes. Later images reuse warm caches.
- **`torch.cuda.is_available()` false in the smoke test** — ROCm userspace
  not found; check `/opt/rocm` exists and `rocminfo` lists a `gfx*` agent.
