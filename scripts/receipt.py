#!/usr/bin/env python3
"""Write a validation receipt (JSON) for one measured run.

Usage: receipt.py <receipt.json> <key>=<value> [<key>=<value> ...]

Values are parsed as JSON when possible, else kept as strings.
Special keys computed automatically when the referenced files exist:
  sha256:<path>   -> {path: {sha256, bytes}}
Used by scripts/validate.sh; every claim in docs/results/ must trace to one
of the receipts this script writes.
"""
import hashlib
import json
import os
import sys


def parse_value(raw: str):
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def file_digest(path: str):
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
            n += len(chunk)
    return {"sha256": h.hexdigest(), "bytes": n}


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    out_path, pairs = sys.argv[1], sys.argv[2:]
    receipt: dict = {}
    for pair in pairs:
        if pair.startswith("sha256:"):
            path = pair[len("sha256:"):]
            if os.path.isfile(path):
                receipt.setdefault("artifacts", {})[path] = file_digest(path)
            else:
                print(f"warn: artifact missing: {path}", file=sys.stderr)
            continue
        if "=" not in pair:
            print(f"bad pair (expected key=value): {pair}", file=sys.stderr)
            return 2
        key, raw = pair.split("=", 1)
        receipt[key] = parse_value(raw)
    receipt.setdefault("artifacts", {})
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(receipt, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"receipt -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
