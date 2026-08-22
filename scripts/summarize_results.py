#!/usr/bin/env python3
"""Summarize validation receipts + logs into the README evidence tables.

Reads docs/results/validation/*.json and docs/results/logs/*.log, extracts
per-step latency / throughput from the upstream profiler output, and prints
a markdown table. Run after scripts/validate.sh; the README's measured
numbers come from here so they can't drift from the receipts.
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECEIPTS = f"{ROOT}/docs/results/validation"
LOGS = f"{ROOT}/docs/results/logs"


def load_receipt(name):
    path = f"{RECEIPTS}/{name}.json"
    if not os.path.isfile(path):
        return None
    return json.load(open(path))


def parse_profile(log_path):
    """Extract upstream 'Profile summary' fields from a run log."""
    if not os.path.isfile(log_path):
        return {}
    text = open(log_path, errors="replace").read()
    out = {}
    m = re.search(r"model load\s*:\s*([\d.]+) s", text)
    if m:
        out["load_s"] = float(m.group(1))
    m = re.search(r"avg per image\s*:\s*([\d.]+) s", text)
    if m:
        out["per_image_s"] = float(m.group(1))
    m = re.search(r"throughput\s*:\s*([\d.]+) tok/s", text)
    if m:
        out["img_tok_per_s"] = float(m.group(1))
    m = re.search(r"generation peak mem\s*:\s*([\d.]+)\s*(GiB|MiB)", text)
    if m:
        val, unit = float(m.group(1)), m.group(2)
        out["torch_peak_gib"] = val if unit == "GiB" else val / 1024
    return out


def fmt_receipt(name, label):
    r = load_receipt(name)
    if not r:
        return None
    row = {
        "block": label or name,
        "wall_s": r.get("wall_seconds"),
        "peak_vram_gib": r.get("peak_vram_gib"),
        "vram_mode": r.get("vram_mode"),
    }
    row.update(parse_profile(f"{LOGS}/{name}.log"))
    arts = r.get("artifacts", {})
    row["artifacts"] = [os.path.basename(p) for p in arts]
    return row


def main():
    names = sys.argv[1:] or [
        "vqa", "t2i", "t2i-think", "edit", "interleave",
        "determinism", "vram-mode-balanced", "vram-mode-fast", "vram-mode-low",
    ]
    rows = []
    for name in names:
        row = fmt_receipt(name, None)
        if row:
            rows.append((name, row))
    cols = ["block", "wall_s", "load_s", "per_image_s", "img_tok_per_s",
            "torch_peak_gib", "peak_vram_gib", "vram_mode"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for name, row in rows:
        cells = []
        for c in cols:
            v = row.get(c)
            cells.append("-" if v is None else str(v))
        print("| " + " | ".join(cells) + " |")
    # determinism verdict
    det = load_receipt("determinism")
    if det:
        print()
        print(f"determinism identical={det.get('identical')}")


if __name__ == "__main__":
    main()
