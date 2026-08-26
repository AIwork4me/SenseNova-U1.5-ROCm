# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `--cfg_interval` t2i speedup, validated in two rounds (36-prompt paired
  Qwen-Image-Bench scoring, 27B judge, official params):
  - **`0 0.2` recommended**: −20.6% per 2048²×50-step image, paired total
    delta +0.34 ± 2.57 (all five dimensions n.s.) — quality-neutral;
  - **`0.7 1.0` retracted**: −32.4% wall but a significant quality cost
    (−6.06 ± 3.85, t=−3.19, p=0.003; Quality −6.5* and Aesthetics −10.8*
    individually significant). An initial n=10 pass missed it (CI ±5.5);
    the expanded n=36 set resolved the effect and the earlier "no loss"
    claim for this interval was withdrawn.
  Receipts: docs/results/validation/cfg-interval/ (authoritative:
  `cfg-interval36.json`; superseded n=10: `cfg-interval.json`).
- PR#260 review follow-up: `interleave_gen_image_only` now passes the
  per-image `cur_image_size` (review by yl-1993); patch 0001 regenerated
  to PR head e2f2c865; V1–V4 on-model verification (list input fixed,
  tuple output byte-identical) — receipts
  docs/results/validation/interleave-image-size/.
- Patch 0003: optional torch.compile/cudagraph safety (env-gated
  `SENSENOVA_COMPILE*`); ported from the Strix Halo sibling repo —
  verified on gfx1151 (V1–V8 + quality bench there); gfx1100 receipt
  pending, compile stays opt-in.
- `scripts/gpu_monitor.py`: sample VRAM + GTT while a workload runs
  (JSONL + peak summary) — GTT visibility is what evidences APU
  GTT-spill runs.
- `scripts/rocm_check.py`: env sanity check (HIP build, wheel gfx
  coverage, bf16 matmul + SDPA smoke, optional GTT-spill alloc test).
- Hardware profile `docs/hardware/strix-halo/` (EN+CN): GTT-spill
  memory model, HSA runtime shim (`hsa_fix.sh`, now CI-shellchecked),
  gfx1151 wheel guidance, AOTriton hd72 caveat — the umbrella's first
  APU profile, ported from the 8060S repo.
- `docs/porting.md` + `docs/porting_CN.md`: generic ROCm porting notes,
  split from the 8060S repo's porting doc (APU-specific depth moved to
  the strix-halo profile).
- `docs/upstream/aotriton-54.md`: upstream tracking entry for the
  AOTriton hd=72 silent-wrong-output bug (issue #54, fix unmerged);
  verified on gfx1151, evidence in the 8060S repo.
- Governance: Code of Conduct, `SECURITY.md`, bug/validation issue
  templates, PR template (evidence gates), CONTRIBUTING cross-repo
  section (graduation model), README/README_CN hardware-coverage table.

### Changed
- pytorch#194498 root cause verified on-host: SDPA fused-backend launch
  failures are a wheel-metadata defect (amd-torch-device-gfx1100 missing
  its amd-torch-device-gfx11 family dependency; diagnosis by liminfei-amd,
  fix ROCm/rocm-systems#10685). Controlled A/B: single package delta flips
  the fresh-process matrix from fused 0/8 (hipErrorInvalidValue) to 11/11;
  real-model t2i 2048²@50 with patch 0002 removed: 355 s (1.94× whole-
  script / 2.22× generation-only vs the 687.7 s MATH baseline). Patch 0002
  reclassified as fallback-only; receipts docs/results/validation/sdpa-gfx11/.

## [0.2.0] — 2026-08-23

Full-stack ROCm 7.14 mode as the validated default path, torch 2.12
support (patched), and four upstream filings.

### Added
- Full-stack ROCm 7.14 mode: `ROCM_FULL_STACK=auto|1|0` (arch-validated
  prefix detection, idempotent preload of MIOpen+comgr+BLAS, MIOpen kept
  enabled) + `scripts/install-rocm-7.14-gfx110x.sh` (SHA256-verified
  TheRock gfx110X dist); receipts now record `rocm_stack`/`ld_preload`.
- Patch 0002: ROCm torch ≥ 2.9 SDPA compatibility (both fused backends
  fail launch — pytorch/pytorch#194498); MATH-only dispatch + q
  pre-scaling; verified end-to-end on torch 2.12.0+rocm7.14.0
  (t2i 2048×2048@50 = 687.7 s / 27.8 GiB, zero BLAS workarounds).
- torch 2.12.0+rocm7.14.0 verification: all three wheel bugs absent with
  zero workarounds; VQA healthy unpatched; receipts `*-torch212*.json`.
- Upstream feedback filed: PR OpenSenseNova/SenseNova-U1#260 (interleave
  image_size fix), issues pytorch/pytorch#194447 (rocm6.3 wheel math
  stack), pytorch/pytorch#194498 (SDPA fused backends),
  OpenSenseNova/SenseNova-U1#261 (torch 2.12 compatibility + patch).
- Evidence artifacts: gdb transcripts for the three wheel bugs, repro
  matrix, ROCm 7.14 verification matrix, torch-2.12 failure transcript;
  poster showcase with reproducible prompts/seeds.
- Full-stack re-validation (2026-08-23): the entire suite re-run under
  full-stack ROCm 7.14 — every generation path 4.2–9.6 % faster than the
  BLAS baseline (t2i 2048×2048@50: 379.6 s / 22.3 GiB; interleave 7-img:
  3250.5 s / 47.9 GiB); `low` vram mode at 3.7 GiB; determinism holds
  (byte-identical within the stack); BLAS-mode baseline archived under
  `docs/results/validation/matrix-blas-20260822/`.

### Fixed
- `validate.sh` interleave block now registers its output images in the
  receipt (artifact glob expanded after the run, stale outputs cleared
  first); `make_gallery.py` labels the run mode from the receipt's
  `rocm_stack` field.

## [0.1.0] — 2026-08-22

First release: full validation of SenseNova-U1.5-8B-MoT on AMD gfx1100
(48 GB, ROCm 7.2.1) through the upstream transformers path.

### Added
- Pipeline scripts: environment check, ROCm venv bootstrap (torch 2.8.0
  rocm6.3 + GPU smoke test incl. bf16-conv guard), manifest-driven
  SHA256-verified checkpoint fetch, task dispatcher
  (t2i / edit / vqa / interleave), one-command quickstart.
- Full validation suite with per-block JSON receipts (wall time, peak VRAM,
  output SHA256, raw logs): all four tasks + think mode + determinism +
  vram-mode comparison. All green; numbers in README trace to receipts.
- ROCm gfx1100 fixes: system hipBLAS/rocBLAS preload + Tensile libpath +
  MIOpen JIT bypass (`src/sensenova_u1_rocm/`), numerics verified.
- Upstream patch 0001: `interleave_gen*` passes `image_size` to
  `_t2i_predict_v` (fixes a cross-platform `TypeError` with
  `use_pixel_head: true`).
- Findings doc: three native bugs in the torch rocm6.3 wheel's math stack
  on gfx1100, root-caused with gdb backtraces and minimal repros.
- Bilingual README (EN/CN), validation report, webp evidence gallery,
  offline unit tests + CI.
