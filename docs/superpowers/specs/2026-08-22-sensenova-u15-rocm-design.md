# SenseNova-U1.5-ROCm — Design

Date: 2026-08-22
Status: approved-for-implementation (user request defines scope; autonomous session)

## Goal

An open-source, evidence-first reference for running
**SenseNova-U1.5-8B-MoT** (native unified multimodal understanding +
generation) on **AMD RDNA3 GPUs via ROCm**, fully validated on the reference
host (gfx1100, 48 GB, ROCm 7.2.1), with one-command usability.

## Research findings that shaped the design

1. **The only viable serving path on ROCm is the upstream `transformers`
   stack.** The official production stack (LightLLM + LightX2V, FA3 kernels)
   is CUDA-only. The upstream repo ships self-contained example scripts
   (`examples/{t2i,editing,vqa,interleave}/inference.py`) that depend only on
   torch/transformers — these are what we wrap.
2. **Weights are 50.23 GB bf16 (13 shards; the `of-00016` suffix is legacy —
   shards 2–4 were never uploaded), which exceeds the 48 GB VRAM of the
   reference card.** The upstream ships a purpose-built layer-offload path
   (`--vram_mode full|fast|balanced|low`) that streams layers from host RAM.
   The reference host has 1 TiB RAM — `balanced` (async prefetch) is the
   default; modes are benchmarked against each other.
3. **ModelScope publishes SHA256 for every file** — the fetch script verifies
   the full checkpoint against a committed manifest.
4. **Machine-local precedent projects** (`Qwen3.8-27B-ROCm`,
   `Muse-Glimmer-30B-ROCm`) established conventions this project follows:
   numbered pipeline scripts, manifest-driven fetch, receipts under
   `docs/results/`, README claims that link to measured evidence.

## Non-goals

- No vLLM/sglang/LightLLM integration (CUDA-bound or unsupported for this
  architecture on ROCm today).
- No GGUF/quantization path (upstream supports GGUF loading, but no U1.5 GGUF
  artifact exists; noted as future work).
- No training/finetuning support.

## Architecture

```
SenseNova-U1.5-ROCm/
├── scripts/
│   ├── 00-check-env.sh     host verification (ROCm, /dev/kfd, GPU, disk, RAM)
│   ├── 01-setup-venv.sh    .venv: ROCm torch 2.8.0 + upstream stack + smoke test
│   ├── 02-fetch-model.sh   manifest-driven, SHA256-verified, resumable download
│   ├── run-task.sh         dispatcher: t2i|edit|vqa|interleave + project defaults
│   ├── quickstart.sh       ONE command: env → venv → model → demo output
│   ├── validate.sh         full validation suite with receipts
│   ├── receipt.py          JSON receipt writer (timing, VRAM, sha256)
│   └── lib/common.sh       shared paths/log helpers
├── configs/artifact-manifest.json   24 files, sizes + SHA256 (from ModelScope API)
├── docs/results/           environment.json, validation/*.json, logs/, gallery/
├── third_party/SenseNova-U1         pinned upstream @ 76c32c2 (feat/u1.5)
└── tests/                  unit tests for scripts + manifest integrity
```

Data flow: `quickstart.sh` → (00 → 01 → 02) → `run-task.sh` → upstream
`examples/<task>/inference.py` with `--model_path $MODEL_DIR --vram_mode
balanced --attn_backend sdpa` injected. Validation adds a device-level VRAM
sampler (rocm-smi polling) and writes one receipt per block.

## Key decisions

| Decision | Choice | Why |
|---|---|---|
| Serving path | upstream transformers examples | only ROCm-viable path; self-contained |
| attn backend | `sdpa` (explicit) | flash-attn has no ROCm wheel; `auto` would silently fall back anyway |
| default vram mode | `balanced` | 50 GB bf16 > 48 GB VRAM; async prefetch overlaps H2D with compute |
| checkpoint home | `$HF_HOME/modelscope/` | the persistent host-disk mount on the reference machine; overridable |
| torch | 2.8.0 + rocm6.3 wheels | upstream pin; gfx1100 natively supported |
| upstream pin | 76c32c2 (feat/u1.5) | the branch that adds U1.5-8B-MoT support |

## Validation matrix (what "fully validated" means here)

1. Environment fingerprint (torch/hip/gpu captured to `docs/results/environment.json`).
2. All four tasks produce correct-form outputs: vqa (text), t2i + think (PNG +
   reasoning text), edit (PNG), interleave (text + PNGs).
3. Determinism: greedy VQA + fixed-seed t2i twice → byte-identical sha256.
4. Performance: per-step latency, end-to-end wall time, tok/s, peak VRAM
   (device-level), for 2048×2048@50 canonical + 10-step mode comparison.
5. Every number in the README links to a receipt/log under `docs/results/`.

## Testing

- `tests/` unit-test the shell surface that can be tested offline: manifest
  integrity (all 24 files, sizes sum), receipt.py behavior, common.sh path
  resolution, script --help exits 0, run-task.sh rejects unknown tasks.
- The GPU validation itself is `scripts/validate.sh`, run on the reference
  host, receipts committed.
