#!/usr/bin/env python3
"""SDPA fused-vs-MATH numerics comparison (pytorch#194498 phase C).

Same-process backend switching is acceptable here ONLY because launch
success was independently proven by the fresh-process matrix
(scripts/sdpa-gfx11-matrix.sh — deferred launch errors are per-process and
same-process probing contaminates failure detection). This script compares
OUTPUTS only: forced-backend SDPA vs forced-MATH reference, bf16.

Stats rationale: median_rel and norm_rel are the meaningful agreement
measures; a raw clamped max_rel is dominated by near-zero denominators,
and max over |ref|>0.1 is reported instead of it.
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

print("torch:", torch.__version__, "| hip:", torch.version.hip)
configs = [(1024, 128, False, None), (1281, 128, False, 1.0 / 128**0.5), (2048, 64, True, None)]
for kv, hd, causal, scale in configs:
    torch.manual_seed(42)
    q_len = kv if causal else 1024
    q = torch.randn(1, 32, q_len, hd, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 32, kv, hd, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 32, kv, hd, device="cuda", dtype=torch.bfloat16)
    kwargs = {"is_causal": causal}
    if scale:
        kwargs["scale"] = scale
    with sdpa_kernel([SDPBackend.MATH]):
        ref = F.scaled_dot_product_attention(q, k, v, **kwargs)
    for name in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION"):
        with sdpa_kernel([getattr(SDPBackend, name)]):
            out = F.scaled_dot_product_attention(q, k, v, **kwargs)
        torch.cuda.synchronize()
        d = (out.float() - ref.float()).abs()
        rf = ref.float().abs()
        mask = rf > 0.1
        med_rel = (d / rf.clamp_min(1e-6)).median().item()
        max_rel_filtered = (d[mask] / rf[mask]).max().item()
        norm_rel = (out.float().norm() - ref.float().norm()).abs().item() / ref.float().norm().item()
        print(
            f"kv={kv} hd={hd} causal={int(causal)} scale={int(scale is not None)}: "
            f"{name:20s} vs MATH: max_abs={d.max().item():.3e} median_rel={med_rel:.3e} "
            f"max_rel(|ref|>0.1)={max_rel_filtered:.3e} norm_rel={norm_rel:.3e}"
        )
print("DONE")
