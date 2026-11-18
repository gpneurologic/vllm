# SPDX-License-Identifier: Apache-2.0
"""Grouped FP8 GEMV with 32x32 power-of-two block scales (decode wo_a).

z[t, g, r] = sum_k x[t, g, k] * W[g*R + r, k] * S[(g*R + r) // 32, k // 32]

W is the stored FP8 e4m3 weight [G*R, K]; S its block scales (any float or
e8m0 dtype, any strides). Split-K partial sums are added afterwards."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gemv_kernel(x_ptr, w_ptr, s_ptr, y_ptr, R, K, G,
                 sx_t, sx_g, sw_n, ss_n, ss_k, n_tok,
                 BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, K_PER: tl.constexpr):
    t = tl.program_id(0)
    pn = tl.program_id(1)  # over G*R rows
    pk = tl.program_id(2)
    rows = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    g = (pn * BLOCK_N) // R  # BLOCK_N divides R
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for i in range(K_PER):
        k0 = (pk * K_PER + i) * BLOCK_K
        ks = k0 + tl.arange(0, BLOCK_K)
        km = ks < K
        xv = tl.load(x_ptr + t * sx_t + g * sx_g + ks, mask=km, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + rows[:, None] * sw_n + ks[None, :], mask=km[None, :], other=0.0)
        sc = tl.load(s_ptr + (rows // 32)[:, None] * ss_n + (ks // 32)[None, :] * ss_k,
                     mask=km[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(w.to(tl.float32) * sc * xv[None, :], axis=1)
    tl.store(y_ptr + (pk * n_tok + t) * (G * R) + rows, acc)


def fp8_grouped_gemv(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor, groups: int,
                     block_n: int = 32, block_k: int = 256, k_per: int = 2) -> torch.Tensor:
    """x [T, G, K] (any float dtype), w [G*R, K] fp8, s [G*R/32, K/32].
    Returns float32 [T, G, R]."""
    T, G, K = x.shape
    N = w.shape[0]
    assert G == groups and N % G == 0 and w.shape[1] == K and w.stride(1) == 1
    R = N // G
    assert R % block_n == 0 and block_n % 32 == 0 or 32 % block_n == 0
    if s.dtype == torch.float8_e8m0fnu:
        s = s.view(torch.uint8).to(torch.int32).sub(127).float().exp2()
    x = x.contiguous()
    split = triton.cdiv(triton.cdiv(K, block_k), k_per)
    y = torch.empty(split, T, N, dtype=torch.float32, device=x.device)
    _gemv_kernel[(T, N // block_n, split)](
        x, w, s, y, R, K, G, x.stride(0), x.stride(1), w.stride(0), s.stride(0), s.stride(1), T,
        BLOCK_N=block_n, BLOCK_K=block_k, K_PER=k_per,
    )
    y = y.sum(0) if split > 1 else y[0]
    return y.view(T, G, R)
