# torch 2.12.0+rocm7.14.0 — forward_gen (image generation) failure transcript (2026-08-23)

Collected during the torch-2.12 verification session (runs on the reference
host, W7900D / torch 2.12.0+rocm7.14.0, zero workarounds). The VQA /
understanding path is healthy (receipt `../../validation/vqa-torch212.json`).

Primary failure (2048×2048, 50 steps, vram_mode balanced — log
`../../logs/t2i-torch212.log`, wall 182.7 s):

```
File ".../modeling_qwen3.py", line 973, in forward_gen
    hidden_states = residual + hidden_states
torch.AcceleratorError: CUDA error: invalid argument
Search for `hipErrorInvalidValue' in https://rocm.docs.amd.com/projects/HIP/en/latest/index.html for more details.
```

Scope probes (same error, all runs):

| Probe | Result |
|---|---|
| 2048×2048, `vram_mode low` (sync offload) | same `hipErrorInvalidValue` in forward_gen |
| 1024×1024, 4 steps | same error |
| 1408×1408, 4 steps | same error |
| `AMD_SERIALIZE_KERNEL=3` (2048×2048, 10 steps) | same error, still surfacing at the forward_gen residual add |

Not saved as separate files at the time; recorded here from the session
log. Conclusion: failure is independent of resolution, offload mode and
kernel-serialization; the understanding path (`forward_und`) completes
correctly in the same environment. Upstream model code pins torch 2.8 —
`forward_gen` needs upstream adaptation for torch 2.12 before the AMD
official wheel can serve the generation tasks.


## UPDATE 2026-08-23 (later same day) — root cause found

The probes above were doubly misleading: (1) the add/randn failures were
deferred launch-time `hipErrorInvalidValue` from SDPA surfacing at the
next checked call, and (2) our same-process bisection was contaminated by
that deferred error, which briefly produced false "scale-dependent" /
"shape-dependent" hypotheses. Fresh-process standalone repros (final
matrix in the findings doc and
[pytorch/pytorch#194498](https://github.com/pytorch/pytorch/issues/194498)):

- **both fused backends — FLASH and mem-efficient — fail kernel launch
  for every configuration tested** (bf16; kv 1024–4096 power-of-two or
  not; head_dim 64/128; causal or not; contiguous or transposed; with or
  without an explicit `scale` kwarg);
- MATH backend: healthy for all shapes;
- the error is deferred: it does not surface at the SDPA call, at
  `torch.cuda.synchronize()`, with `AMD_SERIALIZE_KERNEL=3`, or with
  `HIP_LAUNCH_BLOCKING=1`.

Model-level fix: `patches/0002` (MATH-only on ROCm torch ≥ 2.9 + q
pre-scaling) — t2i 2048×2048@50 completes (687.7 s, receipt
`../../validation/t2i-torch212-fixed.json`).

## UPDATE 2026-08-25 — actual root cause (wheel metadata)

The "both fused backends broken / model needs adaptation" reading above is
superseded. Root cause
([pytorch#194498 comment by liminfei-amd](https://github.com/pytorch/pytorch/issues/194498#issuecomment-5406837588)):
the `amd-torch-device-gfx1100` leaf wheel omits its dependency on the
`amd-torch-device-gfx11` family wheel (AOTriton images) — one packaging
defect explains everything logged here. With
`amd-torch-device-gfx11==2.12.0+rocm7.14.0` installed, a controlled A/B
(single package delta) flips the fresh-process matrix from fused 0/8 to
8/8, and the same t2i workload completes with the STOCK dispatcher — no
model adaptation, no patch — in 355 s (vs 687.7 s under patch 0002).
Upstream fix: [ROCm/rocm-systems#10685](https://github.com/ROCm/rocm-systems/pull/10685)
(open). Patch 0002 is now fallback-only. Receipts:
[../../validation/sdpa-gfx11/](../../validation/sdpa-gfx11/README.md).
