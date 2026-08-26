# Umbrella Backport 8060S — Wave 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute wave-1 of the 8060S→umbrella backport: port the torch.compile patch + GPU tools + porting docs + Strix Halo hardware profile + upstream-tracking entry into this repo, close the governance gaps (CoC, SECURITY, templates, Discussions-ready CONTRIBUTING), and commit the alignment roadmap on the 8060S side.

**Architecture:** All work happens in two local checkouts: the umbrella repo (this repo, at `/home/amd/Desktop/sensnova-merge/main-repo`, branch `backport/8060s-wave1`) and the 8060S repo (at `/home/amd/Desktop/SenseNova-U1.5-ROCm`, real git history + working venv, branch `docs/upstream-alignment`). Ports carry `Ported from` trailers; every task ends green on the repo's offline gates (`pytest tests/ -q`, `shellcheck`). No push happens in this plan — the final task prepares the PR and stops for user confirmation.

**Tech Stack:** bash (strict mode, shellcheck-clean), stdlib-only Python 3, pytest offline (CPU-only, no GPU/network in CI), git plain-diff patches applied by the idempotent loop in `scripts/01-setup-venv.sh`.

**Spec:** `docs/superpowers/specs/2026-08-26-umbrella-backport-8060s-design.md` — read it first; this plan argues from it.

## Global Constraints

- Repo layout gates: CI runs `python3 -m pytest tests/ -v` and `shellcheck scripts/*.sh scripts/lib/common.sh` — every task must keep both green offline (CI runners have no GPU, no torch, no network).
- Python ported into `scripts/` is stdlib-only and must not import torch at module level (CI has no torch; `rocm_check.py` imports torch only inside `main()` — preserve that).
- Patches in `patches/` are plain `git apply`-able diffs against upstream pin `76c32c2` (`scripts/lib/common.sh:14`); the setup loop auto-applies every `patches/*.patch`, so file 0003 needs no script change.
- Attribution: every ported commit message ends with a trailer line `Ported from AIwork4me/SenseNova-U1.5-ROCm-8060S@<sha>`.
- Bilingual discipline: every ported **doc** ships EN + CN in the same commit.
- Evidence labels: ported claims that are only gfx1151-verified carry `verified on: gfx1151 (Strix Halo)` plus a link into the 8060S repo until a gfx1100 receipt exists.
- Commit style: lowercase conventional prefixes used by this repo's history (`feat:`, `docs:`, `fix:`, `changelog:`).
- Git identity for commits: use `-c user.name=... -c user.email=...` only if repo config lacks identity; otherwise the configured identity.
- **Network constraint:** git transport to github.com is currently broken from this host (TLS resets). Work stays local; push/PR is a post-plan, user-confirmed step.

## Environment Map (verify before starting)

| what | path | state |
|---|---|---|
| umbrella repo (this repo) | `/home/amd/Desktop/sensnova-merge/main-repo` | tarball baseline `df59e51` + spec commit `b6c36a5` on `main`; **no real origin history** (graft before push, Task 10) |
| 8060S repo (port source, real history @ `28e4f3a`) | `/home/amd/Desktop/SenseNova-U1.5-ROCm` | clean tree expected; has `.venv` (torch 2.12.0+rocm7.14.0) |
| clean upstream checkout @ `76c32c2` | `/home/amd/SenseNova-U1.5-ROCm-work/upstream` | clean; use for `apply --check` only — never modify |
| gfx hardware | local iGPU gfx1151 (Strix Halo) | the 8060S reference host |

Attribution SHAs (8060S history): tools + porting doc initial release `6046587f`; torch.compile patch `ceefff1a`; AOTriton#54 chain `d0af6efc`, `987015a1`, `28e4f3af`.

---

### Task 1: Branch + patch 0003 (torch.compile/cudagraph safety)

**Files:**
- Create: `patches/0003-optional-torch-compile-and-cudagraph-safety.patch`
- Modify: `patches/README.md` (append 0003 section)
- Modify: `CHANGELOG.md` (Unreleased → Added)

**Interfaces:**
- Consumes: 8060S source `/home/amd/Desktop/SenseNova-U1.5-ROCm/patches/0002-optional-torch-compile-and-cudagraph-safety.patch` (60-line plain diff).
- Produces: `patches/0003-*.patch` auto-applied by `scripts/01-setup-venv.sh`'s glob loop; env knobs `SENSENOVA_COMPILE`, `SENSENOVA_COMPILE_SUPPRESS`, `SENSENOVA_COMPILE_FEATURES` documented in patches/README.md.

- [ ] **Step 1: Create the wave-1 branch in the umbrella repo**

```bash
cd /home/amd/Desktop/sensnova-merge/main-repo
git checkout -b backport/8060s-wave1
```

Expected: `Switched to a new branch` (carries the spec commit `b6c36a5`).

- [ ] **Step 2: Copy the patch with renumbered filename; content ports verbatim**

```bash
cp /home/amd/Desktop/SenseNova-U1.5-ROCm/patches/0002-optional-torch-compile-and-cudagraph-safety.patch \
   patches/0003-optional-torch-compile-and-cudagraph-safety.patch
```

- [ ] **Step 3: Verify it applies cleanly to the pinned upstream (no tree modification)**

```bash
git -C /home/amd/SenseNova-U1.5-ROCm-work/upstream apply --check \
    patches/0003-optional-torch-compile-and-cudagraph-safety.patch && echo APPLY-OK
```

Expected: `APPLY-OK`. If it fails: copy the upstream checkout to `/tmp/upstream-scratch`, `git -C /tmp/upstream-scratch apply --3way` the patch, fix hunks, regenerate with `git -C /tmp/upstream-scratch diff > patches/0003-…patch`, and record the regeneration in the commit message.

- [ ] **Step 4: Runtime smoke on gfx1151 (the patch's original verification host)**

```bash
cd /home/amd/Desktop/SenseNova-U1.5-ROCm
SENSENOVA_COMPILE=1 SENSENOVA_SIZE=1024x1024 SENSENOVA_STEPS=4 SENSENOVA_SEED=42 \
bash scripts/run-t2i.sh "a red lighthouse on a rocky cliff, smoke test" /tmp/wave1-compile-smoke.png
```

Expected: log line `[compile] enabled {'dynamic': False}` and a saved PNG. (This runs the 8060S venv, whose installed tree carries the identical change — Step 3 proved our file is the same content against the same pin.) If the model cache/venv is unavailable, skip with a note in the PR description and cite the existing V1–V8 evidence in the 8060S repo.

- [ ] **Step 5: Append the 0003 section to `patches/README.md`**

Add after the 0002 section (match its heading style):

```markdown
## 0003-optional-torch-compile-and-cudagraph-safety.patch

**Opt-in acceleration (all platforms).** Adds env-gated `torch.compile`
around `_t2i_predict_v` (and optionally `extract_feature`/`patchify`):
`SENSENOVA_COMPILE=1|default|reduce-overhead|…` enables,
`SENSENOVA_COMPILE_SUPPRESS=1` sets dynamo suppress-errors,
`SENSENOVA_COMPILE_FEATURES=1` also compiles the feature/patchify path.
The model-side hunk marks cudagraph step boundaries and materializes the
conditioned prediction before the unconditioned re-run of the same
compiled callable — without it, cudagraph output pooling overwrites the
CFG condition with the uncondition result.

verified on: gfx1151 (Strix Halo) — V1–V8 speedup + Qwen-Image-Bench
paired quality validation live in the
[8060S repo](https://github.com/AIwork4me/SenseNova-U1.5-ROCm-8060S)
(`evidence/speedup2/`, PASS-with-caveat: compile stays opt-in until a
gfx1100 receipt exists here).
```

- [ ] **Step 6: CHANGELOG Unreleased entry**

Add under `## [Unreleased]` → `### Added`:

```markdown
- Patch 0003: optional torch.compile/cudagraph safety (env-gated
  `SENSENOVA_COMPILE*`); ported from the Strix Halo sibling repo —
  verified on gfx1151 (V1–V8 + quality bench there); gfx1100 receipt
  pending, compile stays opt-in.
```

- [ ] **Step 7: Commit**

```bash
cd /home/amd/Desktop/sensnova-merge/main-repo
git add patches/0003-optional-torch-compile-and-cudagraph-safety.patch patches/README.md CHANGELOG.md
git commit -m "feat: patch 0003 — optional torch.compile/cudagraph safety (env-gated)

Ported from AIwork4me/SenseNova-U1.5-ROCm-8060S@ceefff1a"
```

---

### Task 2: Port `gpu_monitor.py` (VRAM+GTT sampling)

**Files:**
- Create: `scripts/gpu_monitor.py`
- Test: `tests/test_scripts.py` (append `test_gpu_monitor_*`)

**Interfaces:**
- Consumes: 8060S source `/home/amd/Desktop/SenseNova-U1.5-ROCm/python/gpu_monitor.py` (100 lines, stdlib-only).
- Produces: CLI `scripts/gpu_monitor.py --interval <s> --out <jsonl>`; JSONL lines `{"ts":…, "vram_used_gib":…, "gtt_used_gib":…}` + final `{"summary":true, "samples":n, "peak_vram_gib":…, "peak_gtt_gib":…, "peak_total_gib":…}`; importable functions `_used_bytes(which:str)->int|None`, `sample_rocm_smi()->tuple[float,float]|None`.

- [ ] **Step 1: Write the failing tests (append to `tests/test_scripts.py`)**

```python
def test_gpu_monitor_help():
    r = subprocess.run(
        [sys.executable, f"{ROOT}/scripts/gpu_monitor.py", "--help"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "VRAM" in r.stdout and "GTT" in r.stdout


def test_gpu_monitor_parses_rocm_smi_csv():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gpu_monitor", f"{ROOT}/scripts/gpu_monitor.py")
    gm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gm)

    fake_csv = ("device,vram total (B),vram used (B)\n"
                "card0,34359738368,10737418240\n"
                "card1,34359738368,2147483648\n")
    class FakeProc:
        returncode = 0
        stdout = fake_csv
    real_run = gm.subprocess.run
    gm.subprocess.run = lambda *a, **k: FakeProc()
    try:
        assert gm._used_bytes("vram") == 10737418240  # max across cards
        vram, gtt = gm.sample_rocm_smi()
        assert abs(vram - 10.0) < 0.01 and abs(gtt - 10.0) < 0.01  # /2**30
    finally:
        gm.subprocess.run = real_run
```

(No pytest fixtures: this file has a `__main__` runner that calls every `test_*` with no arguments — new tests must stay fixture-free.)

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/amd/Desktop/sensnova-merge/main-repo && python3 -m pytest tests/ -q -k gpu_monitor`
Expected: FAIL (file not found / import error).

- [ ] **Step 3: Port the file with two doc adaptations**

```bash
cp /home/amd/Desktop/SenseNova-U1.5-ROCm/python/gpu_monitor.py scripts/gpu_monitor.py
```

Then edit `scripts/gpu_monitor.py` docstring (code body unchanged):
- Usage line: `python gpu_monitor.py --interval 0.5 --out evidence/mem_usage.jsonl` → `python scripts/gpu_monitor.py --interval 0.5 --out docs/results/logs/mem_usage.jsonl`
- Last docstring line: `…aggregation by scripts/verify-all.sh.` → `…aggregation by scripts/validate.sh receipts.`

- [ ] **Step 4: Run tests to verify pass + full suite green**

Run: `python3 -m pytest tests/ -q`
Expected: PASS (all, including pre-existing).

- [ ] **Step 5: Real-GPU sanity on gfx1151 (not a CI gate)**

```bash
timeout 5 python3 scripts/gpu_monitor.py --interval 0.5 --out /tmp/mem.jsonl; tail -1 /tmp/mem.jsonl
```

Expected: a summary JSON with nonzero `peak_vram_gib` (timeout exit 124 is fine — summary prints only on SIGINT/SIGTERM, so check sampled lines: `head -2 /tmp/mem.jsonl`).

- [ ] **Step 6: CHANGELOG + commit**

Add to Unreleased → Added:

```markdown
- `scripts/gpu_monitor.py`: sample VRAM + GTT while a workload runs
  (JSONL + peak summary) — GTT visibility is what evidences APU
  GTT-spill runs.
```

```bash
git add scripts/gpu_monitor.py tests/test_scripts.py CHANGELOG.md
git commit -m "feat: port gpu_monitor.py (VRAM+GTT sampling) from 8060S

Ported from AIwork4me/SenseNova-U1.5-ROCm-8060S@6046587f"
```

---

### Task 3: Port `rocm_check.py` (env sanity, GPU-dependent parts stay runtime-only)

**Files:**
- Create: `scripts/rocm_check.py`
- Test: `tests/test_scripts.py` (append `test_rocm_check_*`)

**Interfaces:**
- Consumes: 8060S source `/home/amd/Desktop/SenseNova-U1.5-ROCm/python/rocm_check.py` (171 lines; imports torch ONLY inside `main()` — keep it that way).
- Produces: CLI `scripts/rocm_check.py [--alloc-test GIB] [--json]`; exit 0 = env OK; `finish(report:dict, as_json:bool, ok:bool)->int` importable offline.

- [ ] **Step 1: Write the failing tests**

```python
def test_rocm_check_help():
    r = subprocess.run(
        [sys.executable, f"{ROOT}/scripts/rocm_check.py", "--help"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "alloc" in r.stdout.lower()


def test_rocm_check_finish_formats_report():
    import importlib.util, io, contextlib
    spec = importlib.util.spec_from_file_location(
        "rocm_check", f"{ROOT}/scripts/rocm_check.py")
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = rc.finish({"torch": "x", "ok": False}, as_json=False, ok=False)
    assert code == 1 and "[FAIL]" in buf.getvalue()
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/ -q -k rocm_check`
Expected: FAIL (file not found).

- [ ] **Step 3: Port with doc adaptations**

```bash
cp /home/amd/Desktop/SenseNova-U1.5-ROCm/python/rocm_check.py scripts/rocm_check.py
```

Docstring edits only (body unchanged):
- `(see scripts/setup-rocm.sh)` → `(see scripts/01-setup-venv.sh)`
- Line 5-8 numbered list stays; line about `--alloc-test` keeps the APU/GTT-spill explanation (that is the strix-halo profile hook).

- [ ] **Step 4: Run full suite**

Run: `python3 -m pytest tests/ -q`
Expected: PASS. (If a test imports torch accidentally it fails on CI-like hosts — the module import in Step 1 must stay torch-free; it is, torch is inside `main()`.)

- [ ] **Step 5: Real run on gfx1151 (not a CI gate)**

```bash
python3 scripts/rocm_check.py --alloc-test 8 --json | head -20
```

Expected: `"ok": true`, `alloc_test: "ok"` (8 GiB fits VRAM carve-out; a `--alloc-test 48` run belongs to the strix-halo profile docs, not here).

- [ ] **Step 6: CHANGELOG + commit**

Unreleased → Added:

```markdown
- `scripts/rocm_check.py`: env sanity check (HIP build, wheel gfx
  coverage, bf16 matmul + SDPA smoke, optional GTT-spill alloc test).
```

```bash
git add scripts/rocm_check.py tests/test_scripts.py CHANGELOG.md
git commit -m "feat: port rocm_check.py (environment sanity) from 8060S

Ported from AIwork4me/SenseNova-U1.5-ROCm-8060S@6046587f"
```

---

### Task 4: `receipt.py` hardware field (test + doc only — code is already generic)

**Files:**
- Modify: `tests/test_scripts.py` (`test_receipt_writer` extension)
- Modify: `scripts/receipt.py` (docstring line only)
- Modify: `CONTRIBUTING.md` (evidence-policy note — full section lands in Task 8)

**Interfaces:**
- Consumes: `receipt.py` already stores arbitrary `key=value` pairs (`parse_value` tries JSON, else string) — no code change.
- Produces: documented convention `hardware=gfx1100|gfx1151|…` in receipts; asserted round-trip in tests.

- [ ] **Step 1: Extend the failing test — in `test_receipt_writer`, change the receipt invocation and assertions**

Replace the `subprocess.run` arg list entry `"note=plain text"` args with:

```python
            [sys.executable, f"{ROOT}/scripts/receipt.py", out,
             "block=t2i", "wall_seconds=12.5", "hardware=gfx1151",
             f"sha256:{art}", "note=plain text"],
```

and add after `assert receipt["note"] == "plain text"`:

```python
        assert receipt["hardware"] == "gfx1151"
```

- [ ] **Step 2: Run — this PASSES immediately (proves genericity); that is the point**

Run: `python3 -m pytest tests/ -q -k receipt`
Expected: PASS. If it fails, `receipt.py` regressed — fix before proceeding.

- [ ] **Step 3: Document the field in `scripts/receipt.py` docstring**

Add one line to the module docstring after the "Special keys" paragraph:

```markdown
Conventional keys recorded by validate.sh runs: `block`, `hardware`
(gfx arch the run executed on, e.g. gfx1100 / gfx1151 — lets
cross-hardware evidence be queried), `rocm_stack`.
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_scripts.py scripts/receipt.py
git commit -m "test+docs: receipt hardware field convention (code already generic)"
```

---

### Task 5: `docs/porting.md` + `docs/porting_CN.md` (generic split of ROCm_PORTING)

**Files:**
- Create: `docs/porting.md` (EN), `docs/porting_CN.md` (CN)
- Modify: `docs/getting-started.md` (one cross-link line)

**Interfaces:**
- Consumes: `/home/amd/Desktop/SenseNova-U1.5-ROCm/docs/ROCm_PORTING.md` (146 lines, EN) and `…/ROCm_PORTING_CN.md` (107 lines, CN).
- Produces: umbrella-generic porting notes; APU-specific sections are NOT here (they move in Task 6).

- [ ] **Step 1: Write `docs/porting.md` from the EN source with this section mapping**

| 8060S source section | destination |
|---|---|
| header + TL;DR | keep, reword "this repository" → umbrella repo; drop `setup-rocm.sh`/`vram_mode tiering automate` phrasing → `scripts/01-setup-venv.sh` + `--vram_mode` |
| §1 What "ROCm support" means | keep; `python/rocm_check.py` → `scripts/rocm_check.py` |
| §2 Attention: SDPA | keep through "math fallback"; DELETE the AOTriton-enablement + hd72 caveat paragraph (8060S-specific stack guidance) and the determinism notes paragraph (that is their PERFORMANCE.md content) — replace with one line: `Backend selection and determinism notes: see the per-hardware profile pages.` |
| §3 intro + §3b layer offload tiers | keep (generic, any GPU) |
| §3a GTT spill (APUs) | NOT here — one pointer line: `Unified-memory GTT spill on APUs: see docs/hardware/strix-halo/README.md.` |
| §4 wheel matrix | keep table shell, but generic: drop the "default since 2026-08-23 / 8060S A/B" framing; cover the official pytorch rocm7.0 index + AMD multi-arch + nightlies rows; gfx1151-specific HSA note → pointer to strix-halo profile |
| §5 HSA runtime pitfall | NOT here — pointer to strix-halo profile |
| §6 interleave hotfix | keep, but reference THIS repo's patch 0001 (same fix, our regenerated version) and upstream PR#260 |
| §7 what this repo changes | keep, adapted: `setup-rocm.sh` → `01-setup-venv.sh`; mention patches 0001–0003 (0003 = opt-in compile) |

Header of the new file:

```markdown
# Porting Notes: SenseNova-U1.5 on AMD GPUs (ROCm)

How the umbrella repo runs SenseNova-U1.5-8B-MoT on AMD GPUs, and why it
stays thin. Hardware-specific depth lives in
[docs/hardware/](hardware/) profiles; evidence-first claims live in
[docs/results/](results/).
```

- [ ] **Step 2: Write `docs/porting_CN.md` by applying the same mapping to the CN source** (`ROCm_PORTING_CN.md`, on disk). Same deletions/pointers, translated consistently with the EN file's terminology.

- [ ] **Step 3: Cross-link from `docs/getting-started.md`**

Add near the top of getting-started.md (after its intro line):

```markdown
For how this port works (wheels, attention backend, memory tiers):
[Porting Notes](porting.md) · [移植笔记](porting_CN.md)
```

- [ ] **Step 4: Commit**

```bash
git add docs/porting.md docs/porting_CN.md docs/getting-started.md
git commit -m "docs: porting notes (EN+CN) — generic split from 8060S ROCm_PORTING

Ported from AIwork4me/SenseNova-U1.5-ROCm-8060S@6046587f"
```

---

### Task 6: Strix Halo hardware profile (`docs/hardware/strix-halo/`)

**Files:**
- Create: `docs/hardware/strix-halo/README.md` (EN), `docs/hardware/strix-halo/README_CN.md` (CN)
- Create: `docs/hardware/strix-halo/hsa_fix.sh` (ported)
- Modify: `.github/workflows/ci.yml` (shellcheck the new script)
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: `/home/amd/Desktop/SenseNova-U1.5-ROCm/scripts/hsa_fix.sh` (34 lines, shell function `hsa_fix_apply`); 8060S porting doc §3a (GTT spill) + §5 (HSA pitfall) content.
- Produces: profile dir referenced by `docs/porting.md` (Task 5 pointer); `hsa_fix_apply "$PY" "$REPO_ROOT"` shell function, sourced not executed.

- [ ] **Step 1: Port `hsa_fix.sh` with header rewording**

```bash
mkdir -p docs/hardware/strix-halo
cp /home/amd/Desktop/SenseNova-U1.5-ROCm/scripts/hsa_fix.sh docs/hardware/strix-halo/hsa_fix.sh
```

Edit the header comment block (function body unchanged): replace `sourced by setup-rocm.sh and common.sh` with `Strix Halo (gfx1151) hardware profile — source this file to get hsa_fix_apply()`; keep the mechanism comment intact.

- [ ] **Step 2: Extend CI shellcheck to cover it**

In `.github/workflows/ci.yml`, change:

```yaml
          shellcheck scripts/*.sh scripts/lib/common.sh
```

to:

```yaml
          shellcheck scripts/*.sh scripts/lib/common.sh docs/hardware/strix-halo/hsa_fix.sh
```

Local gate: `shellcheck docs/hardware/strix-halo/hsa_fix.sh` → no output (clean).

- [ ] **Step 3: Write `docs/hardware/strix-halo/README.md`**

Structure (content adapted from 8060S porting §3a + §5 + their PERFORMANCE/INSTALL facts on disk at `/home/amd/Desktop/SenseNova-U1.5-ROCm/docs/`):

```markdown
# Hardware Profile: Strix Halo (Radeon 8060S iGPU, gfx1151)

verified on: gfx1151 — Ryzen AI MAX+ PRO 395, ROCm 7.2.1 system +
torch 2.12.0+rocm7.14.0 wheels. Full evidence corpus (V1–V8 speedup,
quality bench, determinism, tier tests):
[SenseNova-U1.5-ROCm-8060S](https://github.com/AIwork4me/SenseNova-U1.5-ROCm-8060S).
gfx1100 receipts live in [docs/results/](../../results/) here.

## Why full bf16 fits a "32 GB" iGPU — GTT spill
<§3a content: VRAM carve-out vs GTT pools, same-package LPDDR5X, allocator
spill, `scripts/rocm_check.py --alloc-test 48` as the verifier>

## HSA runtime pitfall (torch +rocm7.0 wheels, driver ≥ 7.1)
<§5 content: GpuAgent::QueueCreate segfault, system-runtime preload,
hsa_fix.sh applies it only when the system runtime is newer>

## Wheel guidance for gfx1151
<AMD multi-arch rocm7.14 wheels default (bare-run, no HSA preload);
official rocm7.0 index + HSA preload as fallback; nightly chain>

## AOTriton caveat (hd=72 ViT heads)
<one paragraph: upstream aotriton#54, unmerged fix branch, see
[../../upstream/aotriton-54.md](../../upstream/aotriton-54.md)>

## Companion repo
daily-driver scripts, one-command quickstart, and the full evidence
corpus live in the 8060S repo (the Strix Halo lab); experiments there
graduate into this umbrella via PR when they meet the graduation
criteria (CONTRIBUTING.md).
```

- [ ] **Step 4: Write `README_CN.md` (same structure, CN terminology consistent with 8060S `ROCm_PORTING_CN.md`)**

- [ ] **Step 5: CHANGELOG + commit**

Unreleased → Added:

```markdown
- Hardware profile `docs/hardware/strix-halo/` (EN+CN): GTT-spill
  memory model, HSA runtime shim (`hsa_fix.sh`, now CI-shellchecked),
  gfx1151 wheel guidance, AOTriton hd72 caveat — the umbrella's first
  APU profile, ported from the 8060S repo.
```

```bash
git add docs/hardware .github/workflows/ci.yml CHANGELOG.md
git commit -m "feat: strix-halo hardware profile (EN+CN docs + hsa_fix shim, CI-covered)

Ported from AIwork4me/SenseNova-U1.5-ROCm-8060S@6046587f"
```

---

### Task 7: `docs/upstream/aotriton-54.md` (upstream tracking entry)

**Files:**
- Create: `docs/upstream/aotriton-54.md`

**Interfaces:**
- Consumes: 8060S evidence chain at `/home/amd/Desktop/SenseNova-U1.5-ROCm/` — `evidence/` V2 notes, `docs/superpowers/specs/2026-08-23-speedup-and-qwen-image-bench-design.md` (its VERIFICATION notes), commits `d0af6efc`/`987015a1`/`28e4f3af`.
- Produces: standalone tracking doc parallel to the existing `docs/upstream/` filings (read one for format before writing).

- [ ] **Step 1: Read an existing `docs/upstream/` file for format, then write the entry**

Content requirements (facts to verify against the 8060S sources above — do not invent): affected component (AOTriton flash kernel, hd=72 head-dim ViT configs), symptom (launch failure / NaN drift observed in speedup2 re-verification), root-cause status (aotriton issue #54 filed 2026-08-26, fix branch unmerged), cross-verified on the 7.14 stack, link to the 8060S evidence directory, and the caveat chain for foreign models (hd≠64 configs until fixed upstream).

- [ ] **Step 2: Commit**

```bash
git add docs/upstream/aotriton-54.md
git commit -m "docs: upstream tracking — AOTriton hd72 kernel bug (issue #54)

Ported from AIwork4me/SenseNova-U1.5-ROCm-8060S@d0af6efc 987015a1 28e4f3af"
```

---

### Task 8: Governance additions (CoC, SECURITY, templates, CONTRIBUTING, README)

**Files:**
- Create: `CODE_OF_CONDUCT.md`, `SECURITY.md`
- Create: `.github/ISSUE_TEMPLATE/bug_report.md`, `.github/ISSUE_TEMPLATE/validation_report.md`, `.github/PULL_REQUEST_TEMPLATE.md`
- Modify: `CONTRIBUTING.md` (cross-repo section), `README.md` + `README_CN.md` (hardware coverage + sibling link)

**Interfaces:**
- Consumes: CONTRIBUTING's existing "validation: <GPU>" issue convention.
- Produces: governance surface a community expects; CONTRIBUTING §"Contributing between the two repos" referenced by the 8060S-side roadmap (Task 9).

- [ ] **Step 1: `CODE_OF_CONDUCT.md`** — concise, bilingual note at end:

```markdown
# Code of Conduct

## Our pledge
Be evidence-first about claims and kind to people. We welcome
contributors of every background and skill level.

## Standards
**Encouraged:** citing receipts for claims, asking questions, reporting
both successes and failures, reviewing others' work respectfully.
**Unacceptable:** harassment or discriminatory language, hostile
responses to newcomers, presenting unverified numbers as measured,
doxxing, spam.

## Enforcement
Report violations by opening a private GitHub security advisory
("Report a vulnerability" → select "not applicable") or contacting a
maintainer via GitHub. Maintainers will respond within 7 days and may
remove comments/commits/ban repeat offenders.

## Attribution
Adapted in spirit from the Contributor Covenant 2.1.
本行为准则同样适用于中文社区讨论；违反行为可同样通过上述渠道举报。
```

- [ ] **Step 2: `SECURITY.md`**

```markdown
# Security Policy

## Supported versions
The tip of `main` only.

## Reporting a vulnerability
Please do NOT open a public issue. Use GitHub's private vulnerability
reporting ("Security" tab → "Report a vulnerability"). Include
reproduction steps and affected components (scripts/, patches/, docs/
pipeline). Expect a response within 7 days.

## Scope
This repo ships shell/python scripts and upstream patches; it does not
serve anything to the network. Model weights come from upstream
ModelScope — weight integrity is verified by
configs/artifact-manifest.json SHA256s (any tampering report is
welcome but belongs upstream).
```

- [ ] **Step 3: Issue templates** — `.github/ISSUE_TEMPLATE/bug_report.md`:

```markdown
---
name: Bug report
about: Something broke on your AMD GPU
title: "bug: "
labels: bug
---
**Hardware / stack** (paste `scripts/00-check-env.sh` output or
docs/results/environment.json): 
**Exact command:** 
**Full log** (pastebin/gist if long): 
**What you expected vs what happened:** 
```

`.github/ISSUE_TEMPLATE/validation_report.md`:

```markdown
---
name: Validation report
about: Share results from YOUR hardware (successes AND failures)
title: "validation: <GPU> / ROCm <version>"
labels: validation
---
**GPU / ROCm / torch:** 
**`00-check-env.sh` output:** 
**Summary table** (`python3 scripts/summarize_results.py`): 
**Receipts** (docs/results/… if you can share): 
**Anything that failed, with the log:** 
```

`.github/PULL_REQUEST_TEMPLATE.md`:

```markdown
**What & why** (one paragraph; link the spec/design doc if any): 

**Evidence** (receipts under docs/results/, or `verified on: <gfx>` +
link for ports from sibling repos — see CONTRIBUTING): 

**Gates** (must both be green):
- [ ] `python3 -m pytest tests/ -q`
- [ ] `shellcheck scripts/*.sh scripts/lib/common.sh docs/hardware/strix-halo/hsa_fix.sh`

**If ported from SenseNova-U1.5-ROCm-8060S:** source commit SHA:
```

- [ ] **Step 4: CONTRIBUTING.md cross-repo section** (append at end):

```markdown
## Contributing between the two repos

This repo is the **umbrella** (hardware-generic assets + hardware
profiles). [SenseNova-U1.5-ROCm-8060S](…) is the **Strix Halo lab**:
APU-first experiments live there; verified work graduates here.

**8060S → umbrella (graduation).** A change graduates when it is
① hardware-independent or absorbed as a hardware profile, ② verified on
gfx1151 with evidence in the 8060S repo, ③ passes this repo's offline
CI, ④ documented EN+CN, ⑤ not a duplicate of an existing tool here.
Open a PR labelled `backport`; the commit carries
`Ported from …@<sha>` and the claim is labelled
`verified on: gfx1151` until a gfx1100 receipt lands here.

**Umbrella → 8060S (flow-down).** Changes to shared surfaces
(`patches/`, `scripts/`) are cherry-picked by the 8060S repo as
needed; that repo records `Merged-upstream: <sha>`.

**Evidence across repos.** `receipt.py` records `hardware=<gfx>` on
every run; cross-hardware claims must link the producing repo's
evidence.
```

Fill the `…` link with `https://github.com/AIwork4me/SenseNova-U1.5-ROCm-8060S`.

- [ ] **Step 5: README hardware coverage** — in `README.md`, after the intro paragraph (before the showcase section), insert:

```markdown
## Hardware coverage

| hardware | status | where |
|---|---|---|
| gfx1100 (RDNA3 dGPU) | fully validated — receipts in [docs/results/](docs/results/) | this repo |
| gfx1151 (Strix Halo APU) | fully validated — evidence in the [8060S repo](https://github.com/AIwork4me/SenseNova-U1.5-ROCm-8060S); profile: [docs/hardware/strix-halo/](docs/hardware/strix-halo/) | this repo + Strix Halo lab |

Other AMD GPUs: the pipeline is hardware-generic (see
[Porting Notes](docs/porting.md)) — validation reports welcome
(issue template `validation_report`).
```

Mirror the same table in `README_CN.md` (translated, same links). Add the 8060S link line to README_CN's intro too.

- [ ] **Step 6: Full gates + commit**

Run: `python3 -m pytest tests/ -q && shellcheck scripts/*.sh scripts/lib/common.sh docs/hardware/strix-halo/hsa_fix.sh`
Expected: all green.

```bash
git add CODE_OF_CONDUCT.md SECURITY.md .github CONTRIBUTING.md README.md README_CN.md
git commit -m "docs: governance — CoC, SECURITY, issue/PR templates, cross-repo contributing, hardware coverage"
```

---

### Task 9: 8060S-side alignment roadmap (docs-only, separate repo)

**Files (in `/home/amd/Desktop/SenseNova-U1.5-ROCm`, branch `docs/upstream-alignment`):**
- Create: `docs/UPSTREAM.md`
- Modify: `README.md` (upstream note after the badges)

**Interfaces:**
- Consumes: the spec §4 (three alignment steps) + §5 (graduation criteria — final wording lives in the umbrella CONTRIBUTING from Task 8; reference, don't fork).
- Produces: the 8060S-side commitment doc; no behavior change in that repo.

- [ ] **Step 1: Branch + write `docs/UPSTREAM.md`**

```bash
cd /home/amd/Desktop/SenseNova-U1.5-ROCm && git checkout -b docs/upstream-alignment
```

```markdown
# Upstream relationship & alignment roadmap

Upstream umbrella: [SenseNova-U1.5-ROCm](https://github.com/AIwork4me/SenseNova-U1.5-ROCm).
This repo stays **independent** — its own quickstart, evidence corpus,
announcements, releases — while aligning structure so work flows both
ways cheaply.

## Graduation (this repo → umbrella)
Experiments here graduate into the umbrella when they meet the
criteria in the umbrella's CONTRIBUTING ("Contributing between the two
repos"): hardware-independent or profile-absorbed, gfx1151 evidence,
CI-green, EN+CN docs, no duplicate tool. Graduating PRs carry
`Ported from …@<sha>`; this repo records `Merged-upstream: <sha>`.

## Alignment roadmap (this repo, own pace)
1. **Governance**: CONTRIBUTING, CHANGELOG (backfilled), Code of
   Conduct, SECURITY, issue/PR templates — adapted from the umbrella.
2. **Structure**: `python/*` tools adopt the umbrella's `scripts/`
   names once wave-1 lands there (gpu_monitor, rocm_check); root
   `quickstart.sh` stays as a thin wrapper.
3. **Quality gates**: the umbrella's CI (shellcheck strict + offline
   pytest) and a `tests/` skeleton.

## Flow-down (umbrella → this repo)
Umbrella changes to `patches/`/`scripts/` are cherry-picked here as
needed.
```

- [ ] **Step 2: README upstream note** — after the two badge lines add:

```markdown
> Upstream umbrella (multi-hardware): [SenseNova-U1.5-ROCm](https://github.com/AIwork4me/SenseNova-U1.5-ROCm) — this repo is the independent Strix Halo lab; see [docs/UPSTREAM.md](docs/UPSTREAM.md).
```

Mirror in `README_CN.md` (same position, CN wording).

- [ ] **Step 3: Commit (8060S repo)**

```bash
git add docs/UPSTREAM.md README.md README_CN.md
git commit -m "docs: upstream relationship & alignment roadmap (umbrella graduation model)"
```

---

### Task 10: Final verification, PR preparation, STOP

**Files:**
- Create: `/home/amd/Desktop/sensnova-merge/WAVE1-PR.md` (PR description, outside both repos)

- [ ] **Step 1: Full offline gates in the umbrella repo**

```bash
cd /home/amd/Desktop/sensnova-merge/main-repo
python3 -m pytest tests/ -v
shellcheck scripts/*.sh scripts/lib/common.sh docs/hardware/strix-halo/hsa_fix.sh
bash -n scripts/*.sh
```

Expected: pytest all green; shellcheck silent; bash -n silent.

- [ ] **Step 2: Verify every commit carries its trailer**

```bash
git log --format='%h %s%n%b' main..backport/8060s-wave1 | grep -B2 'Ported from'
```

Expected: 4 ported commits (Tasks 1, 2, 3, 5/6/7 share trailers) each showing the trailer.

- [ ] **Step 3: Write `WAVE1-PR.md`**

Must contain: summary paragraph (umbrella wave-1), per-commit list with SHA ranges, the `verified on: gfx1151` claims inventory (patch 0003, strix-halo profile) with 8060S evidence links, explicit "gfx1100 re-validation pending" note, and the graduation-criteria mapping (CONTRIBUTING §cross-repo).

- [ ] **Step 4: STOP — user gate**

Do NOT push. Report to the user: branch names, commit list, gate results, smoke-test outcomes, and the push procedure for when they confirm:

```bash
# when git transport to github.com works again (TLS currently broken on this host):
cd /home/amd/Desktop/sensnova-merge/main-repo
git fetch origin && git rebase --onto origin/main <baseline-sha> backport/8060s-wave1
git push -u origin backport/8060s-wave1
# then open the PR from WAVE1-PR.md; likewise push 8060S docs/upstream-alignment
```

(`<baseline-sha>` = the commit before the first wave-1 commit; recorded at execution time.)

---

## Self-Review (done at plan-writing time)

- **Spec coverage:** inventory items 1–6 → Tasks 1–7; repo-side adaptations (receipt hardware field, governance, README) → Tasks 4+8; 8060S roadmap → Task 9; evidence/attribution rules → trailers + labels throughout; execution boundary/STOP → Task 10. Deferred items (bench_report, evidence bulk, gfx1100 re-validation, 8060S restructuring execution) match the spec's deferrals.
- **Type consistency:** `gpu_monitor._used_bytes/sample_rocm_smi`, `rocm_check.finish(report, as_json, ok)`, `hsa_fix_apply "$PY" "$REPO_ROOT"` — names match source files verbatim.
- **Placeholders:** none — every new file's content is embedded or mapped section-by-section from an on-disk source; no "TBD"/"add tests later".
