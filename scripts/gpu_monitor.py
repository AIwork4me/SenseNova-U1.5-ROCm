#!/usr/bin/env python3
"""Sample AMD GPU memory usage (VRAM + GTT) while another process runs.

Usage:
    python scripts/gpu_monitor.py --interval 0.5 --out docs/results/logs/mem_usage.jsonl

Writes one JSON object per sample:
    {"ts": 1724..., "vram_used_gib": 12.3, "gtt_used_gib": 4.1}

Run it in the background, start the workload, then send SIGINT/SIGTERM (or
call stop()) — it prints a summary (peak VRAM / peak GTT / peak total) and
exits. The summary line is also appended to the output file for easy
aggregation by scripts/validate.sh receipts.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path


def _used_bytes(which: str) -> int | None:
    """rocm-smi --csv emits: device,<label total>,<label used> + data rows."""
    out = subprocess.run(
        ["rocm-smi", "--showmeminfo", which, "--csv"],
        capture_output=True, text=True, timeout=10,
    )
    if out.returncode != 0:
        return None
    lines = [l for l in out.stdout.splitlines() if l.strip()]
    if len(lines) < 2:
        return None
    header = [c.strip() for c in lines[0].split(",")]
    try:
        col = next(i for i, h in enumerate(header) if "Used" in h)
    except StopIteration:
        return None
    try:
        return max(int(row.split(",")[col]) for row in lines[1:])
    except (ValueError, IndexError):
        return None


def sample_rocm_smi() -> tuple[float, float] | None:
    vram = _used_bytes("vram")
    gtt = _used_bytes("gtt")
    if vram is None or gtt is None:
        return None
    return vram / (1 << 30), gtt / (1 << 30)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=0.5, help="seconds between samples")
    parser.add_argument("--out", type=Path, required=True, help="output JSONL path")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    running = True

    def stop(*_):  # noqa: ANN001
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    peak_vram = peak_gtt = 0.0
    n = 0
    with args.out.open("w") as fh:
        while running:
            s = sample_rocm_smi()
            if s is not None:
                vram, gtt = s
                peak_vram = max(peak_vram, vram)
                peak_gtt = max(peak_gtt, gtt)
                fh.write(json.dumps({"ts": time.time(), "vram_used_gib": round(vram, 2), "gtt_used_gib": round(gtt, 2)}) + "\n")
                fh.flush()
                n += 1
            time.sleep(args.interval)
        summary = {
            "summary": True,
            "samples": n,
            "peak_vram_gib": round(peak_vram, 2),
            "peak_gtt_gib": round(peak_gtt, 2),
            "peak_total_gib": round(peak_vram + peak_gtt, 2),
        }
        fh.write(json.dumps(summary) + "\n")

    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
