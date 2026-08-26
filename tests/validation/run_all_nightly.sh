#!/usr/bin/env bash
# Phase 9 runner — each bug repro in clean independent processes, >=5 runs.
# Usage: PY=<python> OUT=<dir> RUNS=<n> bash tests/validation/run_all_nightly.sh
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:?set PY to the python under test}"
OUT="${OUT:?set OUT to the transcript output dir}"
RUNS="${RUNS:-5}"

# Defensive: the historical workarounds must NOT be active.
unset LD_PRELOAD ROCBLAS_TENSILE_LIBPATH
echo "runner: PY=$PY OUT=$OUT RUNS=$RUNS"
echo "runner: LD_PRELOAD=${LD_PRELOAD:-<unset>} ROCBLAS_TENSILE_LIBPATH=${ROCBLAS_TENSILE_LIBPATH:-<unset>}"

mkdir -p "$OUT"
summary="$OUT/matrix.txt"
: > "$summary"

for bug in bug1 bug2 bug3; do
  passed=0; failed=0; crash=0
  for i in $(seq 1 "$RUNS"); do
    log="$OUT/${bug}-run${i}.txt"
    "$PY" "$HERE/test_pytorch_nightly_${bug}.py" > "$log" 2>&1
    rc=$?
    echo "exit_code=$rc" >> "$log"
    if [ "$rc" -eq 0 ] && grep -q "^PASS$" "$log"; then
      passed=$((passed + 1)); status=PASS
    else
      failed=$((failed + 1)); status=FAIL
      case "$rc" in
        139|134|135|136|138) crash=$((crash + 1)); status="FAIL(native-signal rc=$rc)";;
      esac
      if grep -qE "SIGSEGV|Segmentation fault|HIPBLAS_STATUS|Unbundle Objects|Failed to decompress|Unknown frame descriptor|miopenStatusUnknownError" "$log"; then
        crash=$((crash + 1)); status="FAIL(historical-signature)"
      fi
    fi
    printf '%s run%d: %s (rc=%d)\n' "$bug" "$i" "$status" "$rc" | tee -a "$summary"
  done
  printf '| %-6s | %d runs | passed=%d failed=%d native-crash-signature=%d | workaround=none |\n' \
    "$bug" "$RUNS" "$passed" "$failed" "$crash" >> "$summary"
done

echo; echo "=== matrix ==="; cat "$summary"
