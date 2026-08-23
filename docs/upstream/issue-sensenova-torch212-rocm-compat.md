# Compatibility: image generation crashes on ROCm with torch >= 2.9 (SDPA backend bugs) — patch included

## Summary

With the official AMD `torch 2.12.0+rocm7.14.0` wheel on gfx1100
(Radeon Pro W7900D), every image-generation task (t2i / editing /
interleave) crashes early in denoising:

```
File ".../modeling_qwen3.py", line 973, in forward_gen
    hidden_states = residual + hidden_states
torch.AcceleratorError: CUDA error: invalid argument   (hipErrorInvalidValue)
```

The crash is **misleading**: the residual add is fine. Root cause (found
by per-op probes + standalone minimal repros) is two ROCm SDPA backend
bugs that the generation path hits through `_sdpa_attn_func`:

1. the FLASH backend fails launch whenever an explicit `scale` kwarg is
   passed — the model always passes `scale = 1/sqrt(head_dim)`;
2. the mem-efficient backend fails launch for several shapes, including
   the generation path's actual `kv_len` (e.g. 1281 = prefix + current
   image tokens, not a power of two).

Both are launch-time errors, so they surface at the next checked CUDA
call (the residual add), far from the real site. The understanding path
(`forward_und`, transformers' standard SDPA with explicit masks) is
unaffected — VQA works unpatched on the same stack. Filed upstream at
pytorch/pytorch (SDPA backend bugs, minimal repros:
https://github.com/AIwork4me/SenseNova-U1.5-ROCm/blob/main/docs/results/findings/rocm63-wheel-blas-on-gfx1100.md).

## Minimal compatibility patch (verified end-to-end)

Restrict `_sdpa_attn_func` to the MATH backend on ROCm (proven healthy
for every shape) and pre-scale `q` instead of passing `scale`. CUDA
hosts keep the stock behavior:

```diff
--- a/src/sensenova_u1/models/neo_unify/modeling_qwen3.py
+++ b/src/sensenova_u1/models/neo_unify/modeling_qwen3.py
@@ def _sdpa_attn_func(...)
+    _on_hip = torch.version.hip is not None
+    _scale = softmax_scale
+    _sdpa_ctx = contextlib.nullcontext()
+    if _on_hip:
+        try:
+            from torch.nn.attention import SDPBackend, sdpa_kernel
+            _sdpa_ctx = sdpa_kernel([SDPBackend.MATH])
+        except ImportError:
+            pass
+        if _scale is not None:
+            q_bhsd = q_bhsd * _scale
+            _scale = None
+    with _sdpa_ctx:
         ... existing scaled_dot_product_attention calls, scale=_scale ...
```

Full patch: https://github.com/AIwork4me/SenseNova-U1.5-ROCm/blob/main/patches/0002-sdpa-rocm-math-backend-compat.patch

## Verification (gfx1100, torch 2.12.0+rocm7.14.0, zero BLAS workarounds)

- t2i 2048×2048 @ 50 steps, seed 42: completes, 687.7 s wall, 27.8 GiB
  peak (receipt:
  `docs/results/validation/t2i-torch212-fixed.json` in the repo above)
- t2i 1024×1024 @ 2 steps: completes
- VQA on the bundled menu.jpg: correct answer, unpatched

Cost of the MATH restriction on this host: 687.7 s vs 420.1 s for the
same canonical cell on torch 2.8+rocm6.3 with this project's BLAS
preloads — a correctness-first trade until the ROCm SDPA backends are
fixed. Once fixed upstream, the backend restriction can be lifted.
