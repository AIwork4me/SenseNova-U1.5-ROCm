# Compatibility: image generation crashes on ROCm with torch 2.12 (SDPA fused-backend launch bugs) — verified patch included

## Summary

With the official AMD `torch 2.12.0+rocm7.14.0` wheel on gfx1100
(Radeon Pro W7900D), text-to-image generation crashes early in
denoising (the same mechanism applies to the other generation tasks —
they share the affected code path):

```
File ".../modeling_qwen3.py", line 973, in forward_gen
    hidden_states = residual + hidden_states
torch.AcceleratorError: CUDA error: invalid argument   (hipErrorInvalidValue)
```

The crash is **misleading**: the residual add is fine. Root cause (found
by per-op probes + standalone minimal repros) is two ROCm SDPA backend
bugs that the generation path hits through `_sdpa_attn_func`:

on this stack **both fused SDPA backends fail kernel launch for every
configuration tested** (bf16; kv 1024–4096, power-of-two or not;
head_dim 64/128; causal or not; contiguous or transposed inputs) — and
the generation path's `_sdpa_attn_func` always passes
`scale = 1/sqrt(head_dim)` with kv lengths like 1281 (prefix + image
tokens). Only the MATH backend is healthy. (torch 2.8.0+rocm6.3 is
unaffected at model level on the same host.)

Both are launch-time errors, so they surface at the next checked CUDA
call (the residual add), far from the real site. The understanding path
(`forward_und`, transformers' standard SDPA with explicit masks) is
unaffected — VQA works unpatched on the same stack. Filed upstream as
[pytorch/pytorch#194498](https://github.com/pytorch/pytorch/issues/194498)
(SDPA backend bugs, minimal repros:
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
- VQA on the bundled menu.jpg: correct answer, unpatched (observed: the
  understanding path never selected a fused backend on this stack; note
  explicit masks alone do NOT prevent the failing fused dispatch —
  standalone mask+SDPA still fails — the sparing is
  layout/eligibility-dependent, e.g. the mask/attention layouts used by
  the understanding path are ineligible for the fused kernels)

Cost of the MATH restriction on this host: 687.7 s vs 420.1 s for the
same canonical cell on torch 2.8+rocm6.3 with this project's BLAS
preloads — a correctness-first trade until the ROCm SDPA backends are
fixed. Once fixed upstream, the backend restriction can be lifted.
