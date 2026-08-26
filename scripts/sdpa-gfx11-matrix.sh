#!/usr/bin/env bash
# sdpa-gfx11-matrix.sh — run the fresh-process SDPA backend matrix for
# pytorch#194498 verification (phases: without / with amd-torch-device-gfx11).
#
# Usage: bash scripts/sdpa-gfx11-matrix.sh <phase-label> <venv-python>
# Writes outputs/validation/sdpa-gfx11/<phase-label>/{cases/, matrix.json}
# Each case = one FRESH python process (deferred launch errors are
# per-process; same-process probing contaminates results).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PHASE="${1:?usage: sdpa-gfx11-matrix.sh <phase-label> <venv-python>}"
PY="${2:?usage: sdpa-gfx11-matrix.sh <phase-label> <venv-python>}"

OUT="$ROOT/outputs/validation/sdpa-gfx11/$PHASE"
mkdir -p "$OUT/cases"

# backend kv head_dim causal with_scale   (representative subset of the
# issue's sweep: contig inputs; kv pow2/non-pow2; hd 64/128; causal/not;
# with/without explicit scale)
CASES=(
    "FLASH_ATTENTION 1024 128 0 0"
    "FLASH_ATTENTION 1281 128 0 0"
    "FLASH_ATTENTION 2048 64 1 0"
    "FLASH_ATTENTION 1024 128 0 1"
    "EFFICIENT_ATTENTION 1024 128 0 0"
    "EFFICIENT_ATTENTION 1281 128 0 0"
    "EFFICIENT_ATTENTION 2048 64 1 0"
    "EFFICIENT_ATTENTION 1024 128 0 1"
    "MATH 1024 128 0 0"
    "MATH 1281 128 0 0"
    "MATH 2048 64 1 0"
)

echo "phase=$PHASE py=$PY"
# shellcheck disable=SC2116
for c in "${CASES[@]}"; do
    # shellcheck disable=SC2086
    set -- $c
    backend=$1; kv=$2; hd=$3; causal=$4; scale=$5
    tag="${backend}_kv${kv}_hd${hd}_c${causal}_s${scale}"
    log="$OUT/cases/$tag.txt"
    "$PY" "$ROOT/scripts/sdpa-gfx11-probe.py" "$backend" "$kv" "$hd" "$causal" "$scale" >"$log" 2>&1
    printf '%-40s %s\n' "$tag" "$(tail -1 "$log")"
done

"$PY" - "$OUT" <<'PYEOF'
import json, re, sys
from pathlib import Path

out = Path(sys.argv[1])
rows = []
for log in sorted((out / "cases").glob("*.txt")):
    m = re.match(r"([A-Z_]+)_kv(\d+)_hd(\d+)_c([01])_s([01])", log.stem)
    backend, kv, hd, causal, scale = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4), m.group(5)
    result = next((ln for ln in log.read_text().splitlines() if ln.startswith("RESULT")), "?")
    ok = result.startswith("RESULT ok")
    rows.append({
        "case": log.stem, "backend": backend, "kv": kv, "head_dim": hd,
        "causal": causal == "1", "explicit_scale": scale == "1",
        "ok": ok, "result": result,
    })
summary = {
    "phase": out.name,
    "torch": __import__("torch").__version__,
    "total": len(rows),
    "ok": sum(r["ok"] for r in rows),
    "by_backend": {
        b: {"total": sum(1 for r in rows if r["backend"] == b),
            "ok": sum(r["ok"] for r in rows if r["backend"] == b)}
        for b in sorted({r["backend"] for r in rows})
    },
    "cases": rows,
}
(out / "matrix.json").write_text(json.dumps(summary, indent=1) + "\n")
print(json.dumps(summary["by_backend"], indent=1))
print(f"matrix -> {out / 'matrix.json'}")
PYEOF
