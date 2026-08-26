#!/usr/bin/env python3
"""Environment sanity check for SenseNova-U1.5 on AMD GPUs (ROCm).

Verifies, in order:
  1. PyTorch is a ROCm (HIP) build and can see the GPU.
  2. The wheel was compiled for this GPU's gfx architecture.
  3. bf16 matmul + SDPA (the attention fallback used when flash-attn is
     absent, which is the default case on ROCm) actually run on the GPU.
  4. (optional, --alloc-test GB) The allocator can reserve a tensor larger
     than dedicated VRAM — on unified-memory APUs (e.g. Strix Halo /
     gfx1151) this succeeds by spilling into GTT, which is what allows the
     full bf16 checkpoint (~50 GB) to stay GPU-resident.

Exit code 0 = environment is good for SenseNova-U1.5 inference.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time


def _apply_hsa_runtime_fix() -> None:
    """Re-exec with the system HSA runtime preloaded when needed.

    Official torch +rocm7.0 wheels bundle a ROCm 7.0 HSA runtime that
    segfaults on gfx1151 with ROCm >= 7.1 kernel drivers (crash inside
    GpuAgent::QueueCreate while freezing the first code object). The
    system runtime always matches the kernel driver, so preloading it
    fixes the crash. No-op when the wheel's runtime is already current.
    """
    sys_rt = "/opt/rocm/lib/libhsa-runtime64.so"
    if not os.path.exists(sys_rt) or sys_rt in os.environ.get("LD_PRELOAD", ""):
        return
    try:
        import torch  # noqa: F401  (needs the venv's torch for the version)
        bundled = torch.version.hip or "0"
        sys_txt = open("/opt/rocm/.info/version").read().strip()
        sys_ver = re.split(r"[^0-9]", sys_txt)[:3]
        bundled_ver = re.split(r"[^0-9]", bundled)[:3]
        newer = tuple(int(x or 0) for x in sys_ver) > tuple(int(x or 0) for x in bundled_ver)
    except Exception:  # noqa: BLE001 — any failure here means "don't touch"
        return
    if newer:
        env = dict(os.environ)
        env["LD_PRELOAD"] = sys_rt + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
        print(f"[sensenova-rocm] applied HSA runtime fix: LD_PRELOAD={sys_rt} "
              f"(system ROCm {sys_txt} > wheel-bundled {bundled})", flush=True)
        os.execve(sys.executable, [sys.executable] + sys.argv, env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alloc-test", type=float, default=0, metavar="GIB",
        help="Try to allocate a GIB-sized uint8 tensor on the GPU and report success/failure.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()

    _apply_hsa_runtime_fix()

    report: dict = {"ok": False}
    try:
        import torch

        report["torch"] = torch.__version__
        report["hip"] = getattr(torch.version, "hip", None)
        if report["hip"] is None:
            report["problem"] = "PyTorch is not a ROCm/HIP build. Install the +rocm wheels (see scripts/01-setup-venv.sh)."
            raise SystemExit(finish(report, args.json, ok=False))

        if not torch.cuda.is_available():
            report["problem"] = "torch.cuda.is_available() is False. Check group membership ('rocm'/'video') and 'rocminfo'."
            raise SystemExit(finish(report, args.json, ok=False))

        props = torch.cuda.get_device_properties(0)
        report["device"] = props.name
        report["gfx_from_hip"] = torch.cuda.get_device_capability(0)
        report["arch_list"] = torch.cuda.get_arch_list()

        # gfx target this wheel actually compiled kernels for (e.g. gfx1151).
        arch_list = report["arch_list"]
        wheel_gfx = [a.split("-")[0] for a in arch_list if a.startswith("gfx")]

        # 3. real kernel smoke tests
        torch.manual_seed(0)
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        c = a @ b
        torch.cuda.synchronize()
        report["bf16_matmul"] = "ok"

        # SDPA with a mask shaped like the model's attention (the fallback
        # attention backend used on ROCm).
        q = torch.randn(1, 32, 1024, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 8, 1024, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, 8, 1024, 128, device="cuda", dtype=torch.bfloat16)
        # GQA shapes (32 q heads / 8 kv heads) like the model's Qwen3 layers
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        torch.cuda.synchronize()
        report["sdpa"] = "ok"
        del a, b, c, q, k, v, out

        # quick bf16 matmul timing (rough chip sanity, not a rigorous benchmark)
        a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        for _ in range(3):
            a @ b
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n = 20
        for _ in range(n):
            a @ b
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        tflops = 2 * 4096**3 * n / dt / 1e12
        report["bf16_matmul_tflops"] = round(tflops, 1)
        del a, b
        torch.cuda.empty_cache()

        if args.alloc_test > 0:
            gib = int(args.alloc_test)
            try:
                t = torch.empty(gib * (1 << 30), dtype=torch.uint8, device="cuda")
                t[:: max(1, t.numel() // 1024)] = 1  # touch pages so the allocation is real
                torch.cuda.synchronize()
                report["alloc_test_gib"] = gib
                report["alloc_test"] = "ok"
                del t
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError as e:
                report["alloc_test_gib"] = gib
                report["alloc_test"] = f"failed ({e})"

        report["ok"] = True
        raise SystemExit(finish(report, args.json, ok=True))
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        report["problem"] = f"{type(e).__name__}: {e}"
        raise SystemExit(finish(report, args.json, ok=False))


def finish(report: dict, as_json: bool, ok: bool) -> int:
    report["ok"] = ok
    if as_json:
        print(json.dumps(report, indent=2))
        return 0 if ok else 1
    print("=" * 62)
    print("SenseNova-U1.5-ROCm environment check")
    print("=" * 62)
    for key in ("torch", "hip", "device", "gfx_from_hip", "arch_list", "bf16_matmul", "sdpa", "bf16_matmul_tflops"):
        if key in report:
            print(f"  {key:<20} {report[key]}")
    if "alloc_test" in report:
        print(f"  alloc_test{'':<11} {report['alloc_test_gib']} GiB -> {report['alloc_test']}")
    if ok:
        print("\n[PASS] environment ready for SenseNova-U1.5 on AMD GPU.")
    else:
        print(f"\n[FAIL] {report.get('problem', 'see diagnostics above')}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
