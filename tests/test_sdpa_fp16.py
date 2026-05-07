"""Accuracy + determinism tests for sdpa_kernel_mma_fp16.

These exercise the fp16 MMA SDPA path against PyTorch's fp16 SDPA reference.
The configs include long-S, small-d cases that previously surfaced a race
condition (output non-deterministic across repeated calls on identical
inputs) — those configs are pinned here so the regression doesn't recur.

Skips cleanly when CUDA / PyTorch are unavailable.
"""
from __future__ import annotations

import importlib

import pytest


_torch = importlib.util.find_spec("torch")
_tensorax = importlib.util.find_spec("tensorax")
torch = importlib.import_module("torch") if _torch else None
ts = importlib.import_module("tensorax") if _tensorax else None
F = importlib.import_module("tensorax.functional") if _tensorax else None


_CUDA_OK = (
    _torch is not None
    and _tensorax is not None
    and torch.cuda.is_available()
    and ts.cuda_is_available()
)


pytestmark = pytest.mark.skipif(
    not _CUDA_OK, reason="sdpa_mma_fp16 requires CUDA + tensorax with CUDA build"
)


# (B, H, Sq, Sk, Dk, Dv). The sweep covers small + long S × all supported d_v.
# Long-S small-d entries (S>=2048, d in {64, 128}) are the configs that
# previously raced at high CTA-per-SM occupancy — keeping them here ensures
# the smem-bloat fix in sdpa_mma_fp16_cuda stays effective.
ACCURACY_CONFIGS = [
    # tiny / boundary
    (1, 1,   64,   64,  64,  64),
    (1, 1,   80,   80,  64,  64),  # non-multiple-of-Q_ROWS
    (1, 1,  128,  128,  64,  64),
    # reference workload (the journal's historical S=256)
    (2, 4,  256,  256,  64,  64),
    (2, 4,  256,  256, 128, 128),
    (2, 4,  256,  256, 256, 256),
    (2, 8,  256,  256, 512, 512),
    (4, 8,  256,  256, 512, 512),
    # long-S small-d — the configs the race used to hit
    (4, 8, 2048, 2048, 128, 128),
    (4, 8, 4096, 4096, 128, 128),
    (1, 8, 8192, 8192,  64,  64),
    (4, 8, 8192, 8192,  64,  64),
    # long-S larger-d
    (2, 4, 1024, 1024, 256, 256),
    (4, 8, 4096, 4096, 256, 256),
]

# fp16 SDPA accumulates over O(seq_len_k) terms in fp16/fp32 mixed precision.
# The reference (PyTorch fp16) uses its own summation order, so a few-millis
# absolute error is expected. Tighten if the kernel ever does fp32 oacc with
# bit-exact reduction.
ATOL = 5e-3
RTOL = 5e-3


def _mma_fp16(q, k, v):
    """Run tensorax MMA fp16 SDPA from PyTorch fp32 inputs, return fp32 result."""
    import numpy as np

    qt = ts.Tensor(q.cpu().numpy(), dtype="float32", device="cuda")
    kt = ts.Tensor(k.cpu().numpy(), dtype="float32", device="cuda")
    vt = ts.Tensor(v.cpu().numpy(), dtype="float32", device="cuda")
    qh = F.cast_to_fp16(qt)
    kh = F.cast_to_fp16(kt)
    vh = F.cast_to_fp16(vt)
    out = F.scaled_dot_product_attention_mma_fp16(qh, kh, vh)
    return torch.tensor(np.asarray(out.cpu().tolist()), dtype=torch.float32, device="cuda")


@pytest.mark.parametrize(("B", "H", "Sq", "Sk", "Dk", "Dv"), ACCURACY_CONFIGS)
def test_sdpa_mma_fp16_matches_pytorch(B, H, Sq, Sk, Dk, Dv):
    torch.manual_seed(0)
    q = torch.randn((B, H, Sq, Dk), device="cuda", dtype=torch.float32)
    k = torch.randn((B, H, Sk, Dk), device="cuda", dtype=torch.float32)
    v = torch.randn((B, H, Sk, Dv), device="cuda", dtype=torch.float32)

    ref = torch.nn.functional.scaled_dot_product_attention(
        q.half(), k.half(), v.half()
    ).float()

    got = _mma_fp16(q, k, v)
    assert got.shape == ref.shape

    diff = (got - ref).abs().max().item()
    assert diff < ATOL, (
        f"sdpa_mma_fp16 vs PyTorch fp16 max-abs error {diff:.4g} > {ATOL} "
        f"(B={B} H={H} Sq={Sq} Sk={Sk} Dk={Dk} Dv={Dv})"
    )
    assert torch.allclose(got, ref, atol=ATOL, rtol=RTOL)


# A dedicated subset of long-S configs to exercise the determinism property.
# At high CTA-per-SM occupancy the kernel previously produced different output
# across repeated calls on identical inputs (max-abs self-diff up to ~0.02).
# The smem-bloat workaround pins occupancy at 2 CTAs/SM where the race goes
# away; this test asserts bitwise self-consistency at those configs.
DETERMINISM_CONFIGS = [
    (4, 8, 2048, 2048, 128, 128),
    (4, 8, 4096, 4096, 128, 128),
    (4, 8, 8192, 8192,  64,  64),
    (1, 8, 8192, 8192,  64,  64),
    (4, 8, 4096, 4096, 256, 256),
]


@pytest.mark.parametrize(("B", "H", "Sq", "Sk", "Dk", "Dv"), DETERMINISM_CONFIGS)
def test_sdpa_mma_fp16_deterministic(B, H, Sq, Sk, Dk, Dv):
    torch.manual_seed(0)
    q = torch.randn((B, H, Sq, Dk), device="cuda", dtype=torch.float32)
    k = torch.randn((B, H, Sk, Dk), device="cuda", dtype=torch.float32)
    v = torch.randn((B, H, Sk, Dv), device="cuda", dtype=torch.float32)

    out0 = _mma_fp16(q, k, v)
    for _ in range(3):
        out_n = _mma_fp16(q, k, v)
        diff = (out_n - out0).abs().max().item()
        assert diff == 0.0, (
            f"sdpa_mma_fp16 produced different output across identical calls: "
            f"max self-diff {diff:.4g} (B={B} H={H} Sq={Sq} Dk={Dk})"
        )
