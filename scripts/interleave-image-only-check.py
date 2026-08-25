#!/usr/bin/env python3
"""Exercise ``NEOChatModel.interleave_gen_image_only`` with tuple vs list ``image_size``.

Purpose (upstream PR OpenSenseNova/SenseNova-U1#260, review by yl-1993):
``interleave_gen_image_only`` accepts ``image_size`` as a single (W, H) tuple
or a per-image ``list[tuple]`` (see modeling_neo_chat.py:691-698). The PR under
review passes the *original parameter* down to ``_t2i_predict_v``; with a list
input the pixel-head branch (``image_size[1] // int``, modeling_neo_chat.py:614)
receives a tuple instead of an int and must crash. This driver makes that
scenario runnable end-to-end on the real model:

  mode=tuple -> image_size=(256, 256)          (baseline, PR-validated path)
  mode=list  -> image_size=[(256, 256), (320, 256)]  (reviewer's scenario)

Run through scripts/interleave-image-only-check.sh (inherits the project's
ROCm env). One scenario per process. Writes summary.json + PNGs + sha256 into
--out_dir; exits 0 on success, 3 on a caught model exception (expected for
list mode on the pre-fix code — the traceback lands in the tee'd log).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sensenova_u1
from sensenova_u1.utils import (
    load_model_and_tokenizer,
    make_offload_ctx,
    seed_all_accelerators,
    vram_mode_keeps_generation_resident,
    vram_mode_to_prefetch_count,
)

SIZES = [(256, 256), (320, 256)]  # multiples of patch_size*merge_size = 16*2 = 32
PROMPT = "Give me a simple illustration."
GT_TEXT = "Sure, here you go. <image> And a second one. <image> All done."
UPSTREAM_DIR = Path(__file__).resolve().parent.parent / "third_party" / "SenseNova-U1"
MODELING_REL = "src/sensenova_u1/models/neo_unify/modeling_neo_chat.py"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_fingerprint() -> dict:
    """Pin the exact code state the run executed (editable install -> worktree)."""
    fp: dict = {"modeling_neo_chat_sha256": sha256_file(UPSTREAM_DIR / MODELING_REL)}
    try:
        fp["upstream_commit"] = subprocess.run(
            ["git", "-C", str(UPSTREAM_DIR), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        fp["upstream_dirty"] = bool(subprocess.run(
            ["git", "-C", str(UPSTREAM_DIR), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip())
    except Exception as exc:  # git missing / not a repo — non-fatal
        fp["git_error"] = repr(exc)
    return fp


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Same denorm as examples/interleave/inference.py:_to_pil (mean/std 0.5)."""
    arr = (t.detach().float().cpu() * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    return Image.fromarray((arr[0] * 255.0).round().astype(np.uint8))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--mode", choices=["tuple", "list"], required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--num_steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vram_mode", default="balanced",
                    choices=["full", "fast", "balanced", "low"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_size = SIZES[0] if args.mode == "tuple" else list(SIZES)
    prefetch_count = vram_mode_to_prefetch_count(args.vram_mode)
    summary = {
        "tag": args.tag,
        "mode": args.mode,
        "image_size": [list(s) for s in ([image_size] if isinstance(image_size, tuple) else image_size)],
        "prompt": PROMPT,
        "gt_text": GT_TEXT,
        "num_steps": args.num_steps,
        "seed": args.seed,
        "cfg_scale": 1.0,
        "img_cfg_scale": 1.0,
        "max_images": 2,
        "vram_mode": args.vram_mode,
        "prefetch_count": prefetch_count,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "code": code_fingerprint(),
    }

    sensenova_u1.set_attn_backend("sdpa")
    print(f"[{args.tag}] attn={sensenova_u1.effective_attn_backend()!r} "
          f"mode={args.mode} image_size={image_size!r} steps={args.num_steps} "
          f"seed={args.seed} vram_mode={args.vram_mode}")

    t_load0 = time.time()
    # Same load/offload assembly as examples/interleave/inference.py: the
    # offload wrapper proxies arbitrary methods, and every validated receipt
    # in this repo runs through it (full-residency is not a validated mode).
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, dtype=torch.bfloat16, device="cuda",
        for_offload=prefetch_count > 0,
    )
    model.eval()
    summary["load_seconds"] = round(time.time() - t_load0, 1)
    print(f"[{args.tag}] model loaded in {summary['load_seconds']}s")

    # interleave_gen_image_only draws its initial noise from the *global* RNG
    # (modeling_neo_chat.py:860, no generator=) — seed right before the call.
    torch.manual_seed(args.seed)
    seed_all_accelerators(args.seed)

    t0 = time.time()
    try:
        with make_offload_ctx(
            model,
            prefetch_count,
            "cuda",
            keep_generation_resident=vram_mode_keeps_generation_resident(args.vram_mode),
        ) as offloaded:
            images = offloaded.interleave_gen_image_only(
                tokenizer,
                PROMPT,
                GT_TEXT,
                image_size=image_size,
                cfg_scale=1.0,
                img_cfg_scale=1.0,
                num_steps=args.num_steps,
                max_images=2,
                verbose=True,
            )
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[{args.tag}] EXCEPTION ({'expected TypeError for list mode on pre-fix code' if args.mode == 'list' else 'UNEXPECTED'}):")
        print(tb, end="")
        summary["status"] = "crashed"
        summary["exception_type"] = type(exc).__name__
        summary["exception"] = str(exc)
        summary["traceback"] = tb
        summary["expected_list_crash"] = (
            args.mode == "list" and type(exc).__name__ == "TypeError"
        )
        summary["wall_seconds"] = round(time.time() - t0, 1)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        print(f"[{args.tag}] summary -> {out_dir / 'summary.json'}")
        return 3

    wall = time.time() - t0
    summary["status"] = "ok"
    summary["wall_seconds"] = round(wall, 1)
    summary["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    summary["images"] = []
    for i, t in enumerate(images):
        png = out_dir / f"{args.tag}_image_{i}.png"
        tensor_to_pil(t).save(png)
        raw = t.detach().float().cpu().contiguous().numpy().tobytes()
        summary["images"].append({
            "shape": list(t.shape),
            "png": png.name,
            "png_sha256": sha256_file(png),
            "raw_sha256": sha256_bytes(raw),
            "raw_min_max": [float(t.min()), float(t.max())],
        })
        print(f"[{args.tag}] image {i}: shape={list(t.shape)} png={png.name}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"[{args.tag}] OK wall={summary['wall_seconds']}s peak_alloc={summary['peak_alloc_gib']}GiB")
    print(f"[{args.tag}] summary -> {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
