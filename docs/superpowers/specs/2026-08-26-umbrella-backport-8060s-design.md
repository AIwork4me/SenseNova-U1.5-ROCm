# Umbrella repo + 8060S backport — Design

Date: 2026-08-26
Status: implemented — wave-1 shipped as PR #1 (executed 2026-08-26; per-task reviews + final whole-branch review clean)

## Goal

Merge the valuable contributions of
[`SenseNova-U1.5-ROCm-8060S`](https://github.com/AIwork4me/SenseNova-U1.5-ROCm-8060S)
(the Strix Halo / gfx1151 sibling repo) into this repo, and establish a
durable two-repo collaboration model, **while keeping the 8060S repo
independent** — its own identity, quickstart UX, evidence corpus, and
release cadence.

This repo becomes the **multi-hardware umbrella**: community-facing,
hardware-generic assets plus per-hardware profiles. The 8060S repo becomes
the **Strix Halo lab**: first to receive APU-specific work; verified
experiments graduate upstream into this repo via PR.

## Research findings that shaped the design

1. **The two repos are same-origin siblings, not forks.** Both were created
   2026-08-22 (one hour apart), never git-linked, and carry the same
   `patches/0001` (interleave image_size hotfix — ours regenerated to the
   upstream PR#260 head e2f2c865, theirs still the 76c32c2-era format-patch).
   Git-level unification (forking / history grafting) is therefore both
   painful and pointless: the trees diverged structurally from day one.
2. **Structural divergence map.**

   | | this repo (gfx1100) | 8060S (gfx1151) |
   |---|---|---|
   | default branch | `main` | `master` |
   | python code | `src/sensenova_u1_rocm/` | `python/` (loose tools) |
   | scripts | numbered pipeline + `scripts/lib/common.sh` | `run-*.sh` per task + root `quickstart.sh` |
   | evidence | `docs/results/` receipts (receipt.py, sha256) | `evidence/` (results.json per run, 178 files) |
   | quality gates | CI (shellcheck strict, offline pytest), CONTRIBUTING, CHANGELOG, tests | none of these |

*Update 2026-08-26(b): the 8060S repo completed its alignment wave-2 (PR #6) — governance, `python/`→`scripts/`, CI + tests. The table above reflects design-time state.*

3. **What 8060S has that we lack** (verified by reading its tree):
   a hardware-independent `torch.compile`/cudagraph-safety patch
   (env-gated, V1–V8 verified), `gpu_monitor.py` (VRAM **and GTT** sampling —
   we sample neither), `rocm_check.py` (env sanity incl. APU GTT-spill
   alloc test), `ROCm_PORTING.md` (mixed generic + APU knowledge),
   `hsa_fix.sh` (Strix Halo wheel-HSA crash workaround), and the AOTriton
   hd72 root-cause chain (their issue #54, mirrors our pytorch#194498 work).
4. **Overlap to avoid**: their `bench_report.py` duplicates our
   `summarize_results.py` purpose (aggregate run results into a summary
   table) — porting both would create a two-headed tool.
5. **Both repos pin the same upstream checkout** (`feat/u1.5` @ `76c32c2`,
   patches applied by the venv setup script) — so their patch applies (or
   regenerates cleanly) against our pin with high probability; must be
   verified, not assumed.
6. **The working machine for this effort is the 8060S reference host**
   (Ryzen AI MAX+ PRO 395, gfx1151). gfx1100 re-validation is possible only
   later, on that host.

## Decisions (user-approved 2026-08-26)

| decision point | choice |
|---|---|
| merge scope | umbrella repo: generic assets in-tree + Strix Halo as a hardware profile; 8060S keeps evidence/announcements/showcase |
| structure relation | 8060S aligns to this repo's structure over time (its own pace, three steps) |
| evidence policy | layered attribution: merge with `verified on: gfx1151` labels linking 8060S evidence; drop labels only after gfx1100 receipts exist |
| execution scope | design + wave-1 executed locally (branch, port, test on gfx1151); push/PR only after user confirmation; gfx1100 re-validation deferred |

## Design

### 1. Governance and repo relationship

- This repo: **no rename** (already hardware-generic), description and
  README scope broadened to "AMD GPUs" covering gfx1100 dGPU + gfx1151 APU.
  New `docs/hardware/strix-halo/` profile directory; existing gfx1100
  content stays where it is (minimal churn).
- 8060S keeps: independent repo identity, root `quickstart.sh` one-command
  UX, its `evidence/` corpus, announcements, showcase, own releases.
- Cross-linking: 8060S README gains an explicit "upstream: this repo" note;
  this repo's README hardware table links both. GitHub topics unified
  (`rocm`, `amd`, `sensenova`, …).
- Relationship model: classic downstream–upstream. 8060S experiments
  graduate into this repo via PR when they meet the graduation criteria
  (§5); this repo's shared-dir changes flow down to 8060S via cherry-pick.

### 2. Wave-1 backport inventory

Ordered by value × independence; every item lands as its own commit with
`Ported from AIwork4me/SenseNova-U1.5-ROCm-8060S@<sha>`.

| # | from 8060S | to this repo | action |
|---|---|---|---|
| 1 | `patches/0002-optional-torch-compile…patch` | `patches/0003-…` (**renumbered**; our 0002 is SDPA) | adapt to our patch format; verify clean apply on pin `76c32c2`; smoke `SENSENOVA_COMPILE=1` on gfx1151 |
| 2 | `python/gpu_monitor.py` | `scripts/gpu_monitor.py` | adopt script conventions (stdlib-only, usage(), `set -euo pipefail` n/a for py); add offline unit tests; GTT sampling is new capability |
| 3 | `python/rocm_check.py` | `scripts/rocm_check.py` | generic env-check logic; alloc-test section labelled APU/GTT-scoped |
| 4 | `docs/ROCm_PORTING.md` generic sections | `docs/porting.md` (standalone new file — the source has 7 substantive sections; folding into getting-started would bloat it) | split: generic → here, APU-specific → `docs/hardware/strix-halo/` |
| 5 | `scripts/hsa_fix.sh` + GTT-spill knowledge | `docs/hardware/strix-halo/` (doc + script) | hardware-profile absorption |
| 6 | AOTriton #54 / V2 root-cause chain | `docs/upstream/` entry | alongside pytorch#194498 tracking |
| 7 | `python/bench_report.py` | — | **deferred** (overlaps `summarize_results.py`) |
| 8 | `evidence/`, announcements, showcase | — | **not merged**; referenced by link |

Docs ported in wave 1 ship EN + CN variants together (bilingual discipline).

Wave 1 also includes two **this-repo-side adaptations** (not ports, listed
here so the plan is exhaustive): the `receipt.py` `hardware` field with its
offline test (§3), and the governance additions (CoC, SECURITY.md, issue/PR
templates, CONTRIBUTING cross-repo section, README hardware-table update).

### 3. Evidence and attribution rules

- Layered labels: merged hardware-independent claims carry
  `verified on: gfx1151 (Strix Halo) — evidence: <8060S link>` until a
  gfx1100 receipt lands in `docs/results/`, then the label is removed.
- `receipt.py` gains an optional `hardware` field (e.g. `"gfx1151"`) so
  cross-hardware evidence is machine-queryable; offline unit test included.
- Attribution: every ported commit carries `Ported from …@<sha>`; original
  authors get `Co-authored-by:` where applicable. 8060S records
  `Merged-upstream: <this-repo sha>` on its side — bidirectional trace.

### 4. 8060S alignment roadmap (executed in the 8060S repo, its own pace)

1. **Governance**: CONTRIBUTING (incl. graduation flow), CHANGELOG
   (backfilled from git history), Code of Conduct, SECURITY.md, issue/PR
   templates — adapted from this repo, bilingual.
2. **Structure**: `python/*` → `scripts/*` with this repo's names (after
   wave 1 lands here, 8060S adopts the merged versions verbatim — sync cost
   drops to cherry-pick); root `quickstart.sh` remains as a thin wrapper.
3. **Quality gates**: this repo's CI (shellcheck strict + offline CPU
   pytest) and a `tests/` skeleton.

Also in this repo (governance gaps found in self-audit): Code of Conduct,
SECURITY.md, issue/PR templates, GitHub Discussions enabled (8060S already
uses Discussions), CONTRIBUTING gains a cross-repo contribution section.

### 5. Ongoing sync process (bidirectional)

- **8060S → here (graduation)**: experiment verified on gfx1151 → meets
  graduation criteria → PR to this repo labelled `backport`, CI green →
  merged → 8060S records `Merged-upstream`.
- **Here → 8060S (flow-down)**: changes to shared dirs (`patches/`,
  `scripts/`) are cherry-picked / replayed by 8060S as needed.
- **Graduation criteria (written down)**: ① hardware-independent or
  absorbed as a hardware profile ② gfx1151 evidence exists ③ passes this
  repo's offline CI ④ EN+CN docs complete ⑤ no two-headed tool conflict.
- Each CONTRIBUTING documents the other direction; PRs/issues cross-linked
  between repos.

### 6. Testing, risks, execution boundary

Wave-1 tests (all doable on the gfx1151 host):
- `git apply --check` of the renumbered torch.compile patch against the
  pinned upstream checkout; real `SENSENOVA_COMPILE=1` smoke run.
- `gpu_monitor.py` / `rocm_check.py` real runs on gfx1151 + new offline
  tests under `tests/`.
- This repo's `python3 -m pytest tests/ -q` green + shellcheck clean
  (same gates as CI).

Risks & mitigations:
- torch.compile patch drift vs our pin → regenerate first, verify apply,
  rework in our format if conflicting.
- two-headed tools → bench_report deferred.
- doc-reorg churn → zero moves of existing content, only new dirs.
- network: git transport to github.com currently failing from this host
  (TLS resets); work proceeds on local clones initialised from `gh api`
  tarballs; push/PR deferred until both network and user confirmation.

## Non-goals

- No repo rename, no fork conversion, no git-history grafting.
- No bulk import of 8060S `evidence/` (178 files) or announcements.
- No 8060S restructuring *inside* this repo's PRs — that happens in the
  8060S repo on its own schedule.
- No gfx1100 re-validation inside wave 1 (deferred; needs that host).

## Success criteria

1. Wave-1 items 1–6 merged as individually-attributed commits; CI green;
   bilingual docs in place.
2. Governance gaps closed in this repo (CoC, SECURITY, templates,
   Discussions, CONTRIBUTING cross-repo section).
3. Graduation criteria + sync process written into both CONTRIBUTINGs.
4. 8060S still fully functional and independent (quickstart UX unchanged),
   with its alignment roadmap committed as its own docs.
