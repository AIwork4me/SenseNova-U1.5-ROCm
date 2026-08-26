# Draft comment for pytorch/pytorch#194498 (NOT posted)

> Status: DRAFT ONLY — awaiting explicit human approval before posting.
> Prepared 2026-08-26 after phase C of the SDPA probe matrix:
> [`docs/results/findings/pytorch-nightly-rocm714-sdpa-t2i.md`](../results/findings/pytorch-nightly-rocm714-sdpa-t2i.md)
> Note: an earlier draft flow for #194447 was posted after explicit approval; this one has NOT been authorized yet.

---

Thanks @liminfei-amd for the root cause (leaf wheel missing the `amd-torch-device-gfx11` family dependency with the AOTriton images) — we reproduced the full A/B on the reporter host (phase A: fused 0/8 with the deferred `hipErrorInvalidValue`, phase B: 11/11 after adding only the family wheel; receipts in the repo linked below).

An additional data point for this issue: the failure mode **does not occur on the upstream nightly ROCm 7.14 packaging**, because that wheel bundles the AOTriton pieces itself — `torch/lib/aotriton.images/` (including `amd-gfx110x/flash/`) and `libaotriton_v2.so.0.13.0` — so there is no leaf/family split to get wrong. On `torch-2.15.0.dev20260825+rocm7.14` (hip 7.14.60850, gfx1100, no env workarounds) the same fresh-process matrix passes 11/11 (FLASH 4/4, mem-efficient 4/4, MATH 3/3), and forced-fused outputs match forced-MATH references at bf16 level (norm rel diff ≤ 5.4e-6, median rel 0, max abs exactly at bf16 quanta). Full matrix, per-case logs and the numerics comparison: [phase C receipts and transcripts](https://github.com/AIwork4me/SenseNova-U1.5-ROCm/tree/main/docs/results/validation/sdpa-gfx11).

One honest caveat from the same session, so nobody reads "nightly" as a blanket fix: on this particular nightly build the SenseNova-U1.5 **text-to-image path silently produces corrupted output** (exits 0, saves a washed-out repeating grid texture; bit-identical with fused SDPA disabled, so it is NOT this issue's mechanism and not any fused kernel — the stock dispatcher picks MATH for these layouts anyway; the same code/seed/config renders correctly on torch 2.8). The understanding/VQA path on nightly is correct. Isolation chain and receipts: [pytorch-nightly-rocm714-sdpa-t2i.md](https://github.com/AIwork4me/SenseNova-U1.5-ROCm/blob/main/docs/results/findings/pytorch-nightly-rocm714-sdpa-t2i.md). We plan to bisect the mis-computing op before filing that as its own issue.
