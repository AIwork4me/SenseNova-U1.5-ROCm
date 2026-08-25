#!/usr/bin/env python3
"""Single-case SDPA probe for pytorch#194498 / ROCm rocm-systems#10685.

One FRESH process per case (caller loops). Runs one forced-backend
scaled_dot_product_attention call followed by a trailing checked op — the
fused-backend failure is a DEFERRED launch error (hipErrorInvalidValue)
that only surfaces at the next checked CUDA op, so the trailing consumer
is load-bearing.

Usage: sdpa-gfx11-probe.py BACKEND kv head_dim causal(0|1) [with_scale(0|1)]
Prints: RESULT ok sum=...  |  RESULT fail ExcType: msg
Exit 0 either way (the runner parses RESULT lines).
"""
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

backend = getattr(SDPBackend, sys.argv[1])  # 2.12: pybind enum, not subscriptable
kv = int(sys.argv[2])
hd = int(sys.argv[3])
causal = sys.argv[4] == "1"
with_scale = len(sys.argv) > 5 and sys.argv[5] == "1"

torch.manual_seed(42)
q_len = kv if causal else 1024  # flash-family: causal requires q_len == kv_len
q = torch.randn(1, 32, q_len, hd, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 32, kv, hd, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, 32, kv, hd, device="cuda", dtype=torch.bfloat16)

try:
    kwargs = {"is_causal": causal}
    if with_scale:
        kwargs["scale"] = 1.0 / (hd ** 0.5)
    with sdpa_kernel([backend]):
        o = F.scaled_dot_product_attention(q, k, v, **kwargs)
    s = o.float().abs().sum().item()  # trailing checked op
    print(f"RESULT ok sum={s:.4f}")
except Exception as exc:
    print(f"RESULT fail {type(exc).__name__}: {str(exc)[:200]}")
    sys.exit(0)
