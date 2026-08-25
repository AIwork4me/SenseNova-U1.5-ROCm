# sdpa-gfx11 — root-cause A/B verification for pytorch#194498 (2026-08-25)

Root cause identified by **liminfei-amd**
([comment](https://github.com/pytorch/pytorch/issues/194498#issuecomment-5406837588),
fix [ROCm/rocm-systems#10685](https://github.com/ROCm/rocm-systems/pull/10685)):
the `amd-torch-device-gfx1100` leaf wheel (pip multi-arch layout) does not
declare a dependency on the **family wheel** `amd-torch-device-gfx11`,
which carries the AOTriton images the fused SDPA backends need. Installs
that follow the published leaf-wheel-only recipe are therefore missing
those images: FLASH and mem-efficient fail kernel launch with a DEFERRED
`hipErrorInvalidValue`; MATH (plain PyTorch kernels) is unaffected.

This directory verifies that diagnosis on the issue reporter's host
(W7900D / gfx1100, 48 GB) with a controlled single-variable A/B:

| phase | venv delta vs original install | fresh-process SDPA matrix (11 cases) |
|---|---|---|
| A | none — faithful rebuild of the original install (`torch 2.12.0+rocm7.14.0` + `torchvision 0.27.0` + `amd-torch-device-gfx1100`, cp312, no workarounds) | **fused 0/8** — `AcceleratorError: CUDA error: invalid argument` (hipErrorInvalidValue) at the trailing checked op, every config; **MATH 3/3 ok** |
| B | + `amd-torch-device-gfx11==2.12.0+rocm7.14.0` (the ONLY package delta — see the two pip-freeze snapshots) | **11/11 ok**; fused outputs match MATH to bf16 tolerance (rel. diff ≤ 2e-6); MATH sums bit-identical across phases (same seed) |

Matrix runner/probe: `scripts/sdpa-gfx11-matrix.sh` + `scripts/sdpa-gfx11-probe.py`
(one FRESH python process per case; trailing checked op after the SDPA
call so the deferred launch error surfaces; causal cases use q_len==kv_len
so flash eligibility is not a confound). Cases: FLASH/EFFICIENT ×
{kv 1024/1281/2048, hd 128/64, causal/non, ±explicit scale} + MATH × 3.
Per-case logs: `phaseA-cases.tar.gz` / `phaseB-cases.tar.gz`.

## Real-model confirmation (t2i 2048×2048@50, balanced, seed 42)

With gfx11 installed and [patches/0002](../../../patches/README.md)
(MATH-forcing) REMOVED — i.e. the stock dispatcher free to pick fused
backends — the exact workload that used to crash now completes, and is
much faster than the MATH-patch baseline (the previously recommended
workaround):

| | wall (whole script) | generation only | peak reserved |
|---|---|---|---|
| torch 2.12 + patch 0002 (MATH) — `t2i-torch212-fixed.json` | 687.7 s | ~607 s | 27.3 GiB |
| torch 2.12 + gfx11 wheel, no patch — `t2i-flash-receipt.json` | **355 s** | **273 s** | 21.6 GiB |
| speedup | **1.94×** | **2.22×** | −21% |

Artifact `t2i-2048-flash.png` sha256 `56a58cf3…d4e68b` (full value in the
receipt); log `../../logs/sdpa-gfx11-t2i-flash.log`, zero errors.

## Practical guidance

- On torch 2.12.0+rocm7.14.0 multi-arch installs missing the family wheel:
  `pip install "amd-torch-device-gfx11==2.12.0+rocm7.14.0" --extra-index-url https://repo.amd.com/rocm/whl-multi-arch/`
  (or wait for the leaf wheel to carry the dependency — 2.13.0 still
  lacked it at the time of the comment).
- With gfx11 present, patch 0002 is unnecessary (and leaves fused-backend
  performance on the table); keep it for installs you cannot fix.
- torch 2.8.0+rocm6.3 (this repo's validated default) is unaffected
  either way — 0002 only activates on ROCm torch ≥ 2.9.
