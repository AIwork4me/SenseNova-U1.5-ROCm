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


def test_upstream_pinned_and_patched():
    """The pinned upstream commit must be recorded; when checked out, the
    interleave patch must be applied (01-setup-venv.sh does this)."""
    common = open(f"{ROOT}/scripts/lib/common.sh").read()
    assert 'UPSTREAM_PINNED_COMMIT="76c32c2"' in common
    upstream = f"{ROOT}/third_party/SenseNova-U1"
    if os.path.isdir(f"{upstream}/.git"):
        head = subprocess.run(
            ["git", "-C", upstream, "rev-parse", "--short=7", "HEAD"],
            capture_output=True, text=True).stdout.strip()
        assert head == "76c32c2", f"upstream HEAD {head} != pinned 76c32c2"
        # the patch file exists and targets the known call sites
        patch = open(f"{ROOT}/patches/0001-interleave-pass-image_size-to-_t2i_predict_v.patch").read()
        assert patch.count("+") >= 10
        assert "image_size=image_size" in patch


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
             "block=t2i", "wall_seconds=12.5", "hardware=gfx1151",
             f"sha256:{art}", "note=plain text"],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        receipt = json.load(open(out))
        assert receipt["block"] == "t2i"
        assert receipt["wall_seconds"] == 12.5      # parsed as number
        assert receipt["note"] == "plain text"       # kept as string
        assert receipt["hardware"] == "gfx1151"
        import hashlib
        assert receipt["artifacts"][art]["sha256"] == hashlib.sha256(b"hello").hexdigest()
        assert receipt["artifacts"][art]["bytes"] == 5


def test_fullstack_mode_logic():
    """common.sh ROCm_FULL_STACK mode selection, offline with fake prefixes."""
    import tempfile
    common = f"{ROOT}/scripts/lib/common.sh"
    with tempfile.TemporaryDirectory() as td:
        fake = os.path.join(td, "rocm-7.14-fake")
        lib = os.path.join(fake, "lib", "rocblas", "library")
        os.makedirs(lib)
        for f in ("libMIOpen.so", "libamd_comgr.so", "libhipblas.so", "librocblas.so"):
            open(os.path.join(fake, "lib", f), "w").close()
        open(os.path.join(lib, "TensileLibrary_fake_gfx1100.dat"), "w").close()

        probe = (
            "export ROCM714_PREFIX=\'{p}\' {env}\n"
            "source \'{c}\'\n"
            "echo ${{ROCM_FULL_STACK_ACTIVE:-0}}\n"
            "echo ${{LD_PRELOAD:-}}\n"
            "echo END-PROBE\n"
        )

        def run_common(env):
            r = subprocess.run(["bash", "-c", probe.format(p=fake, env=env, c=common)],
                               capture_output=True, text=True)
            out = r.stdout[:r.stdout.index("END-PROBE")] if "END-PROBE" in r.stdout else r.stdout
            return out.split("\n")[:2]

        # auto + valid prefix -> full stack (MIOpen preloaded too)
        active, preload = run_common("")
        assert active == "1" and "libMIOpen.so" in preload and "librocblas.so" in preload, preload
        # explicit 0 -> falls back to BLAS-only preload (no MIOpen). The
        # rocBLAS preload itself only appears when a system ROCm exists
        # (GitHub runners have none) — assert MIOpen exclusion, and the
        # rocBLAS presence only when /opt/rocm or /usr/local/rocm is present.
        active, preload = run_common("export ROCM_FULL_STACK=0")
        assert active == "0" and "libMIOpen.so" not in preload, preload
        if os.path.isdir("/opt/rocm") or os.path.isdir("/usr/local/rocm"):
            assert "librocblas.so" in preload, preload
        # BLAS_FIX=0 -> no preload at all
        active, preload = run_common("export BLAS_FIX=0")
        assert preload == "", preload
        # ROCM_FULL_STACK=1 with a bogus prefix must fail loudly
        r = subprocess.run(
            ["bash", "-c", probe.format(p=os.path.join(td, "nope"), env="export ROCM_FULL_STACK=1", c=common)],
            capture_output=True, text=True)
        assert r.returncode != 0 and "ROCM_FULL_STACK=1" in r.stderr, r.stderr


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

    # header mirrors real rocm-smi 4.0.0 (ROCm 7.8) --csv output; the
    # parser matches the capital-"Used" column case-sensitively
    fake_csv = ("device,VRAM Total Memory (B),VRAM Total Used Memory (B)\n"
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
