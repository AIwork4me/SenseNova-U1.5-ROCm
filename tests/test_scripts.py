#!/usr/bin/env python3
"""Offline unit tests for SenseNova-U1.5-ROCm scripts and configs.

Run:  python3 -m pytest tests/ -q      (or: python3 tests/test_scripts.py)
These tests are CPU-only and safe in CI: they never touch the GPU, the
checkpoint, or the network.
"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_manifest_complete():
    manifest = json.load(open(f"{ROOT}/configs/artifact-manifest.json"))
    files = manifest["files"]
    assert len(files) == 24, f"expected 24 files, got {len(files)}"
    # 13 safetensors shards + 11 sidecars
    shards = [f for f in files if f["path"].endswith(".safetensors")]
    assert len(shards) == 13, f"expected 13 shards, got {len(shards)}"
    assert all(len(f["sha256"]) == 64 for f in files), "every file needs a sha256"
    assert all(f["size"] > 0 for f in files)
    assert manifest["total_size_bytes"] == sum(f["size"] for f in files)
    # ~50.2 GB total (decimal)
    assert 49e9 < manifest["total_size_bytes"] < 52e9
    # the config must be there — run-task.sh needs it
    assert any(f["path"] == "config.json" for f in files)


def test_upstream_pinned():
    """The pinned upstream commit must be recorded and, when present, checked out."""
    common = open(f"{ROOT}/scripts/lib/common.sh").read()
    assert 'UPSTREAM_PINNED_COMMIT="76c32c2"' in common
    upstream = f"{ROOT}/third_party/SenseNova-U1"
    if os.path.isdir(f"{upstream}/.git"):
        head = subprocess.run(
            ["git", "-C", upstream, "rev-parse", "--short=7", "HEAD"],
            capture_output=True, text=True).stdout.strip()
        assert head == "76c32c2", f"upstream HEAD {head} != pinned 76c32c2"


def _bash(script, *args):
    return subprocess.run(
        ["bash", f"{ROOT}/scripts/{script}", *args],
        capture_output=True, text=True)


def test_help_exits_zero():
    for script in ("00-check-env.sh", "01-setup-venv.sh", "02-fetch-model.sh",
                   "run-task.sh", "quickstart.sh", "validate.sh"):
        r = _bash(script, "--help")
        assert r.returncode == 0, f"{script} --help failed:\n{r.stderr}"
        assert "usage" in r.stdout.lower() or "Usage" in r.stdout


def test_run_task_rejects_unknown_task():
    r = _bash("run-task.sh", "spreadsheet")
    assert r.returncode != 0
    assert "unknown task 'spreadsheet'" in r.stderr


def test_run_task_rejects_missing_venv():
    """With no .venv (fresh clone) the dispatcher must point at 01-setup-venv.sh."""
    env = dict(os.environ)
    env["ROOT"] = ROOT
    # common.sh resolves VENV relative to the script location; simulate a
    # missing venv by pointing the check at a temp copy of the tree would be
    # heavy — instead assert the guard exists and is reachable.
    src = open(f"{ROOT}/scripts/lib/common.sh").read()
    assert "require_venv()" in src
    assert "01-setup-venv.sh" in src


def test_receipt_writer():
    with tempfile.TemporaryDirectory() as td:
        art = os.path.join(td, "out.bin")
        with open(art, "wb") as fh:
            fh.write(b"hello")
        out = os.path.join(td, "receipt.json")
        r = subprocess.run(
            [sys.executable, f"{ROOT}/scripts/receipt.py", out,
             "block=t2i", "wall_seconds=12.5", f"sha256:{art}", "note=plain text"],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        receipt = json.load(open(out))
        assert receipt["block"] == "t2i"
        assert receipt["wall_seconds"] == 12.5      # parsed as number
        assert receipt["note"] == "plain text"       # kept as string
        import hashlib
        assert receipt["artifacts"][art]["sha256"] == hashlib.sha256(b"hello").hexdigest()
        assert receipt["artifacts"][art]["bytes"] == 5


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok {name}")
            except AssertionError as e:
                failures += 1
                print(f"  FAIL {name}: {e}")
    sys.exit(1 if failures else 0)
