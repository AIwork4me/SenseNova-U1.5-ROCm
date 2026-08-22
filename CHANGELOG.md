# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
