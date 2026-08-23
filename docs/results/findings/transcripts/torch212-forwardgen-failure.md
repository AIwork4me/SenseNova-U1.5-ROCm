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
