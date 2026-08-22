#!/usr/bin/env bash
# 02-fetch-model.sh — download the SenseNova-U1.5-8B-MoT checkpoint.
#
# Manifest-driven: every file is verified against configs/artifact-manifest.json
# (size + SHA256, both from the ModelScope API) after download; verified
# files are skipped on re-run, partial downloads resume.
#
# Destination: $MODEL_BASE/SenseNova-U1.5-8B-MoT (default: inside the HF
# cache — on the reference machine that is the persistent host-disk mount).
# Override with MODEL_BASE=... or MODEL_DIR=...
#
# Env knobs:
#   MODEL_BASE / MODEL_DIR   where the checkpoint lands
#   NCONNS=N                 parallel shard downloads (default 4)
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/02-fetch-model.sh

Downloads SenseNova/SenseNova-U1.5-8B-MoT (~50.2 GiB, bf16, 13 safetensors
shards) from ModelScope and verifies every file against the SHA256 manifest.
Resumable; already-verified files are skipped.
EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    "") ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/lib/common.sh
source "$ROOT/scripts/lib/common.sh"
MANIFEST="$ROOT/configs/artifact-manifest.json"
NCONNS="${NCONNS:-4}"

mkdir -p "$MODEL_DIR"
log "fetching $MODEL_REPO -> $MODEL_DIR (verified against $MANIFEST)"

python3 - "$MANIFEST" "$MODEL_DIR" "$NCONNS" "$MODEL_REPO" <<'PYEOF'
import hashlib, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

manifest_path, dest, nconns, repo = sys.argv[1:5]
files = json.load(open(manifest_path))["files"]
base = f"https://modelscope.cn/models/{repo}/resolve/master"

def verified(entry):
    f = os.path.join(dest, entry["path"])
    if not os.path.isfile(f) or os.path.getsize(f) != entry["size"]:
        return False
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest() == entry["sha256"]

def fetch(entry):
    url, out = f"{base}/{entry['path']}", os.path.join(dest, entry["path"])
    tmp = out + ".part"
    for attempt in range(5):
        r = subprocess.run(
            ["curl", "-fL", "--retry", "5", "--retry-delay", "3", "-C", "-",
             "-o", tmp, "--connect-timeout", "30", url],
            capture_output=True)
        if r.returncode == 0 and os.path.getsize(tmp) == entry["size"]:
            os.replace(tmp, out)
            return entry["path"], entry["size"], "downloaded"
        time.sleep(3 * (attempt + 1))
    return entry["path"], 0, "FAILED (curl rc=%d)" % r.returncode

pending = [e for e in files if not verified(e)]
skipped = len(files) - len(pending)
if skipped:
    print(f"{skipped}/{len(files)} files already verified — skipping")
total = sum(e["size"] for e in pending)
print(f"downloading {len(pending)} files, {total/1e9:.1f} GB total")

done_bytes, failures = 0, []
with ThreadPoolExecutor(max_workers=int(nconns)) as ex:
    for path, size, status in ex.map(fetch, pending):
        if status == "downloaded":
            done_bytes += size
            print(f"  ok {path} ({size/1e9:.2f} GB) — {done_bytes/1e9:.1f}/{total/1e9:.1f} GB")
        else:
            failures.append((path, status))
            print(f"  !! {path}: {status}")

bad = [e for e in files if not verified(e)]
if failures or bad:
    for e in bad:
        print(f"VERIFY FAILED: {e['path']}", file=sys.stderr)
    sys.exit(1)
print(f"all {len(files)} files verified (size + SHA256)")
PYEOF

log "checkpoint ready at $MODEL_DIR"
