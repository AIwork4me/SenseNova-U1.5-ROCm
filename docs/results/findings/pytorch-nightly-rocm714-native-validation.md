# Native validation — upstream PyTorch nightly ROCm 7.14 wheel on gfx1100 (2026-08-26)

Closes the loop on [pytorch/pytorch#194447](https://github.com/pytorch/pytorch/issues/194447)
("three native crashes of the torch 2.8.0+rocm6.3 wheel's bundled math stack
on gfx1100"). Upstream maintainer @naromero77amd
([comment](https://github.com/pytorch/pytorch/issues/194447#issuecomment-5416018384),
2026-08-25): the reported issues are fixed in ROCm 7.14 and the upstream
PyTorch nightly wheels are now on ROCm 7.14. Prior verification in this repo
([rocm714-verification.md](transcripts/rocm714-verification.md)) had only
proven that substituting/preloading ROCm 7.14 libraries over the old 6.3
wheel removes the crashes. This run proves the **official upstream nightly
wheel works natively, by itself, with none of this repo's historical
workarounds**.

## 1. Executive verdict

```text
PASS — all three #194447 regressions are absent in upstream PyTorch nightly ROCm 7.14 with no workaround.
```

15/15 standalone repro executions (3 bugs × 5 independent processes) passed;
real SenseNova-U1.5-8B-MoT VQA inference completed with a correct answer;
numerics match the historical baselines at bf16 rounding level.

## 2. Exact environment

| Item | Value |
|---|---|
| Date | 2026-08-26 (10:36–11:19 UTC) |
| OS | Ubuntu 24.04.4 LTS, kernel 6.8.0-79-generic |
| CPU / RAM | AMD EPYC 9334 (128 logical), 1 TiB |
| Python | 3.12.3 (`/usr/bin/python3`), fresh venv `.venv-pytorch-nightly-rocm714` |
| GPU | AMD Radeon Pro W7900D, gfx1100 (capability 11,0), 48 GiB VRAM (51 522 830 336 B) |
| GPU id | Device ID 0x744b, ROCk module 6.14.14 |
| System ROCm | 7.2.1 at `/opt/rocm -> /opt/rocm-7.2.1` (untouched, not used by the nightly wheel) |
| **torch under test** | **2.15.0.dev20260825+rocm7.14** (cp312 manylinux_2_28_x86_64) |
| torch.version.hip | **7.14.60850** |
| torchvision | 0.30.0.dev20260825+rocm7.14 |
| triton | triton-rocm 3.8.0+git3f6e4113 (torch METADATA's exact pin) |
| ROCm userspace | rocm-sdk-core/libraries/device-gfx1100 7.14.0 (official AMD pip runtime, in-venv) |

Install commands (network note: this host cannot reach PyPI's
files.pythonhosted.org nor download-r2.pytorch.org — the R2 host the nightly
index hrefs point at; `download.pytorch.org` itself serves the identical wheel
paths directly, so the wheels were fetched from there with curl and installed
locally; pip validated each wheel's internal RECORD hashes):

```bash
# official nightly wheel (metadata resolution: pip install torch --index-url
# https://download.pytorch.org/whl/nightly/rocm7.14 ; bytes fetched directly):
python -m pip install --no-deps \
  /tmp/torch-2.15.0.dev20260825+rocm7.14-cp312-cp312-manylinux_2_28_x86_64.whl \
  /tmp/torchvision-0.30.0.dev20260825+rocm7.14-cp312-cp312-manylinux_2_28_x86_64.whl
# the wheel's own RPATH resolves the ROCm userspace from the venv:
python -m pip install 'rocm-sdk-core==7.14.*' 'rocm-sdk-libraries==7.14.*' \
  'rocm-sdk-device-gfx1100==7.14.*' --extra-index-url https://repo.amd.com/rocm/whl-multi-arch/
# triton pin from torch METADATA:
python -m pip install --no-deps triton_rocm-3.8.0+git3f6e4113-cp312-cp312-linux_x86_64.whl
```

Full log: [`transcripts/pytorch-nightly-rocm714-native/install.txt`](transcripts/pytorch-nightly-rocm714-native/install.txt)

### Packaging model (documented, not a workaround)

The upstream nightly ROCm wheel **does not bundle the ROCm userspace**;
`torch/lib` carries only `libc10_hip.so` / `libtorch_hip.so` (+ rocshmem,
aotriton), and `libtorch_hip.so`'s RPATH is
`$ORIGIN/../../_rocm_sdk_core/lib:$ORIGIN/../../_rocm_sdk_core/lib/rocm_sysdeps/lib:$ORIGIN/../../_rocm_sdk_libraries/lib:$ORIGIN`
— i.e. the supported model is the AMD pip ROCm distribution in the same venv
(or an equivalent system ROCm 7.14). Without it, libs fall back to system
`/opt/rocm-7.2.1`, which lacks `libhipfile.so.0` (ROCm ≥ 7.14) and `import
torch` fails. Link- and runtime-level proof that the tested process used the
venv's ROCm 7.14 (never system 7.2.1, never a preload):
[`transcripts/pytorch-nightly-rocm714-native/libraries.txt`](transcripts/pytorch-nightly-rocm714-native/libraries.txt)
(ldd + `/proc/self/maps`: libMIOpen/librocblas/libhipblas/libamd_comgr/libamdhip64/libhiprtc/libhipfile/libroctx64 all from `_rocm_sdk_*`).

## 3. Workaround status (all three historical workarounds absent)

```text
LD_PRELOAD: unset
ROCBLAS_TENSILE_LIBPATH: unset
torch.backends.cudnn.enabled: True (MIOpen enabled, asserted in-process in every run)
```

Additionally absent: no MIOpen overrides (only logging-only vars in one
clearly-labeled supplementary diagnostic), no LD_LIBRARY_PATH, no patch 0002
(SDPA MATH fallback), no `torch.backends.cudnn.enabled = False` anywhere
(`SENU15_MIOPEN=1` explicitly opts OUT of this repo's cudnn bypass for the
E2E). Checks before and after testing:
[`transcripts/pytorch-nightly-rocm714-native/environment.txt`](transcripts/pytorch-nightly-rocm714-native/environment.txt),
[`transcripts/pytorch-nightly-rocm714-native/post-test-contamination.txt`](transcripts/pytorch-nightly-rocm714-native/post-test-contamination.txt).

## 4. Results

| Test | Runs | Passed | Failed | Native crash | Workaround |
|---|---:|---:|---:|---|---|
| Bug 1 bf16 Conv (`Conv2d(3,768,16,16)` on `(16,3,16,16)`) | 5 | 5 | 0 | 0 | none |
| Bug 2 large bf16 GEMM (`Linear(4096,4096)` on `(16060,4096)`) | 5 | 5 | 0 | 0 | none |
| Bug 3 MIOpen JIT (`Conv2d(768,4096,k2,s2)` on `(1,768,110,146)`) | 5 | 5 | 0 | 0 | none |

| Case | Original failure (rocm6.3 wheel) | Nightly ROCm 7.14 result | Verdict |
|---|---|---|---|
| Bug 1 bf16 Conv | SIGSEGV `Tensile::PlaceholderLibrary::loadPlaceholderLibrary()` | rc=0, shape OK, finite, MIOpen on | **PASS** |
| Bug 2 large bf16 GEMM | `Unbundle Objects Error: Failed to decompress … Unknown frame descriptor` → `HIPBLAS_STATUS_INTERNAL_ERROR` | rc=0, shape OK, finite | **PASS** |
| Bug 3 MIOpen JIT | SIGSEGV `COMGR::DataObject::clearData()` via `hiprtcCompileProgram`/`miopen::hiprtc::BuildHip` | rc=0, shape OK, finite, MIOpen on | **PASS** |
| SenseNova-U1.5-8B-MoT E2E VQA | required full preload stack + `cudnn off` | real inference, correct coherent answer | **PASS** |

### Bug 3 shape provenance (not re-derived from memory)

From the repo's own model code + image: menu.jpg 4096×3072 →
`load_image_native(max_pixels=4194304)` smart-resize (factor 32) →
1760×2336 → patch k16 s16 → grid 110×146 → `dense_embedding =
Conv2d(768→4096, k2, s2)` on `(1, 768, 110, 146)` (this is also where Bug 2's
M=16060 comes from). MIOpen's own log confirms it handled exactly this
descriptor set. Cold-cache JIT probe: rerunning the exact bug-3 repro with
a fresh `HOME` (empty `~/.cache/miopen`) **passes and writes a brand-new
user kernel DB** (`$HOME/.cache/miopen/3.5.2.cd957402/gfx1100_48.ukdb`,
114 688 B, fresh mtime — transcript
`bug3-cold-cache-probe-supplementary.txt`) — MIOpen demonstrably compiled
the kernel through the 7.14 comgr/hiprtc stack (the historical crash
site) rather than loading a warm cache. A second probe with an irregular
111×147 shape also passes.

### E2E detail

Same invocation as the repo's documented VQA workflow (`--vram_mode balanced
--attn_backend sdpa`, menu.jpg, 768 max new tokens), run via
`tests/validation/run_e2e_vqa_nightly.py` under the nightly interpreter with
deps provided by `PYTHONPATH=<nightly-sp>:<repo-sp>:<upstream-src>`
(torch/torchvision resolve from the nightly venv; transformers 4.57.1 /
accelerate 1.14.0 from the repo venv — cp312-identical). Model loaded (13/13
shards, ≈24.9 GiB VRAM observed via a mid-run `rocm-smi` sample
(26 768 035 840 B ≈ 24.9 GiB binary / 26.8 GB decimal)) and produced a
correct, coherent menu transcription (23 dishes listed, 22 with prices —
the output truncates at the 768-token cap exactly like the baseline run).
Diff vs the
historical workaround-stack answer (`outputs/validation/vqa-menu.txt`): 4
trivial OCR-ambiguity character differences on small print; the 22 prices
that appear in both (each run truncates mid-item-23 at the token cap) and
the overall structure are identical — expected bf16-stack-level
divergence. The answer text is preserved in-repo as
[`transcripts/pytorch-nightly-rocm714-native/e2e-answer.txt`](transcripts/pytorch-nightly-rocm714-native/e2e-answer.txt).

### Numerical sanity (GPU bf16 vs CPU fp32, same seeded inputs)

| Case | max abs | mean abs | median abs | median rel |
|---|---|---|---|---|
| Bug 1 conv | 9.29e-3 | 1.48e-3 | 1.19e-3 | **3.16e-3 (0.32 %)** |
| Bug 2 GEMM | 1.19e-2 | 1.26e-3 | 1.04e-3 | 2.76e-3 |

Bug 1's 0.32 % median relative error is **identical** to both the historical
7.2.1-workaround and 7.14-TheRock measurements — bf16 rounding level, no
numeric regression.

## 5. Evidence

All raw transcripts in [`transcripts/pytorch-nightly-rocm714-native/`](transcripts/pytorch-nightly-rocm714-native/):
`environment.txt`, `install.txt` (includes the failed attempt-1 and the
network diagnosis), `libraries.txt`, `bug{1,2,3}-run{1..5}.txt`,
`bug3-miopen-logging.txt`, `bug3-jit-probe-supplementary.txt`,
`bug3-cold-cache-probe-supplementary.txt`, `matrix.txt`,
`numeric-sanity.txt`, `e2e.txt` (attempt 1, import-time triton failure —
kept, not hidden), `e2e-run2.txt` (PASS), `e2e-answer.txt` (model output,
in-repo copy since `outputs/` is gitignored), `post-test-contamination.txt`.
Repro scripts: `tests/validation/` (`test_pytorch_nightly_bug{1,2,3}.py`,
`run_all_nightly.sh`, `numeric_sanity_bugs12.py`, `run_e2e_vqa_nightly.py`).
Model answer (working tree): `outputs/validation/vqa-menu-nightly.txt`.

Honest failure notes, for the record:
1. E2E attempt 1 failed at import: nightly dynamo touches
   `triton.language.dtype` and the only triton on the PYTHONPATH chain was
   the repo venv's ancient `pytorch-triton-rocm 3.4.0`. Fixed by installing
   torch's own pinned dependency (`triton-rocm==3.8.0+git3f6e4113`) into the
   nightly venv — a dependency-completion step, not a behavior workaround.
2. On a host with only system ROCm 7.2.1, `import torch` of the nightly wheel
   fails on `libhipfile.so.0` until the ROCm 7.14 userspace is provided —
   see "Packaging model" above; users need a ROCm 7.14 runtime (AMD pip
   distribution or system install).

## 6. Conclusion — can #194447 be closed?

**Yes.** The three reported crashes were properties of the torch
2.8.0+rocm6.3 wheel's bundled math stack (its rocBLAS lazy-Tensile loader,
its 6.3-era Tensile decompressor, and its comgr/MIOpen JIT). On the same
gfx1100 host, the official upstream nightly wheel built against ROCm 7.14 —
run natively with its supported ROCm 7.14 pip runtime, MIOpen enabled, and
none of LD_PRELOAD / ROCBLAS_TENSILE_LIBPATH / cudnn-off — passes all three
exact reproducers across repeated independent processes, executes MIOpen
kernel compilation (verified on a cold cache: fresh user kernel DB
written), and completes real
SenseNova-U1.5 multimodal inference with correct output. Nothing else is
needed from the PyTorch side for the issues as reported.

Two non-blocking notes for anyone following the same path: (a) install a
ROCm 7.14 runtime alongside the nightly wheel (its RPATH expects the AMD pip
distribution or an equivalent system 7.14); (b) on network-restricted hosts,
the nightly index's download links point at `download-r2.pytorch.org` — if
unreachable, the identical paths on `download.pytorch.org` serve the wheels.
