# Contributing

Thanks for helping make SenseNova-U1.5 run well on AMD GPUs!

## The one rule: evidence first

Every performance or compatibility claim in this project must trace to a
receipt under `docs/results/` — a JSON file recording the exact command,
wall time, peak VRAM, and output hashes, plus the raw log. If you claim a
number, produce the receipt.

## Ways to contribute

### 1. Validate on your hardware

Run the pipeline on your AMD GPU and share the results:

```bash
bash scripts/00-check-env.sh     # confirm your host is sane
bash scripts/01-setup-venv.sh
bash scripts/02-fetch-model.sh
bash scripts/validate.sh         # full suite; receipts land in docs/results/
```

Open an issue titled `validation: <GPU> / ROCm <version>` and paste:

- the output of `scripts/00-check-env.sh`
- `docs/results/environment.json`
- the summary table from `python3 scripts/summarize_results.py`
- anything that failed, with the log

Both successes and failures are valuable — failures get preserved as
findings, not hidden.

### 2. Bug reports

Include: GPU model, ROCm version, torch version (`docs/results/environment.json`
format), the exact command, and the full log. `bash scripts/validate.sh ONLY=<block>`
re-runs a single block for a minimal reproduction.

### 3. Code

- Shell scripts must pass `bash -n` (shellcheck in CI must pass).
- Python must stay stdlib-only where it is today (no new runtime deps
  without discussion).
- Unit tests (`python3 -m pytest tests/ -q`) must pass offline, without a
  GPU — CI runs on CPU-only runners.

## Style

- Scripts: `set -euo pipefail`, a `usage()` with `--help`, clear `[project]`
  log prefixes — follow the existing scripts.
- Numbers in docs: cite the receipt file they came from.

## Contributing between the two repos

This repo is the **umbrella** (hardware-generic assets + hardware
profiles). [SenseNova-U1.5-ROCm-8060S](https://github.com/AIwork4me/SenseNova-U1.5-ROCm-8060S) is the **Strix Halo lab**:
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
