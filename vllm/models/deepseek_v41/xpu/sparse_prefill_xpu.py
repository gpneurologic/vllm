# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA attention for DeepSeek V4.1 prefill on XPU (one shared KV head).

Same math as vllm/v1/attention/ops/xpu_mla_sparse.py (bf16 KV rows gathered by
index, online softmax, attention sink), but each gathered [BLOCK_N, 512] tile
is loaded once and used as both K and V (they are the same rows here), and one
program can cover all of a rank's heads, so a token's rows are gathered once
instead of once per 16-head block.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_prefill_kernel(q_ptr, kv_ptr, idx_ptr, len_ptr, sink_ptr, out_ptr,
                           H, seq_kv, topk, stride_qt, stride_qh, stride_it,
                           stride_ot, stride_oh, sm_scale_log2,
                           D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr,
                           HAS_SINK: tl.constexpr, HAS_LEN: tl.constexpr):
    t = tl.program_id(0)
    hb = tl.program_id(1)
    heads = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    hmask = heads < H
    d = tl.arange(0, D)
    q = tl.load(q_ptr + t * stride_qt + heads[:, None] * stride_qh + d[None, :],
                mask=hmask[:, None], other=0.0)
    m = tl.zeros([BLOCK_H], dtype=tl.float32) - 1.0e30
    s = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, D], dtype=tl.float32)
    # Valid indices are packed at the front of each row; stop at the row's
    # length instead of walking the padding (rows are 768 wide, <= 640 used).
    L = topk
    if HAS_LEN:
        L = tl.minimum(tl.load(len_ptr + t), topk)
    for n0 in range(0, L, BLOCK_N):
        offs = n0 + tl.arange(0, BLOCK_N)
        ind = tl.load(idx_ptr + t * stride_it + offs, mask=offs < L, other=-1)
        valid = (ind >= 0) & (ind < seq_kv)
        rows = tl.where(valid, ind, 0).to(tl.int64)
        kv = tl.load(kv_ptr + rows[:, None] * D + d[None, :], mask=valid[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(kv)) * sm_scale_log2  # [BH, BN], log2 domain
        qk = tl.where(valid[None, :], qk, -1.0e30)
        m_new = tl.maximum(m, tl.max(qk, 1))
        alpha = tl.exp2(m - m_new)
        p = tl.exp2(qk - m_new[:, None])
        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        s = s * alpha + tl.sum(p, 1)
        m = m_new
    if HAS_SINK:
        sink2 = tl.load(sink_ptr + heads, mask=hmask, other=float("-inf")) * 1.4426950408889634
        m_f = tl.maximum(m, sink2)
        sc = tl.exp2(m - m_f)
        acc = acc * sc[:, None]
        s = s * sc + tl.exp2(sink2 - m_f)
    s = tl.where(s == 0.0, 1.0, s)
    acc = acc / s[:, None]
    tl.store(out_ptr + t * stride_ot + heads[:, None] * stride_oh + d[None, :],
             acc.to(tl.bfloat16), mask=hmask[:, None])


def sparse_prefill_attn(q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor,
                        sm_scale: float, attn_sink: torch.Tensor | None,
                        block_h: int = 32, block_n: int = 16,
                        num_warps: int = 16,
                        lens: torch.Tensor | None = None) -> torch.Tensor:
    """q [T, H, 512] bf16, kv [M, 512] bf16 rows, indices [T, topk] int32 (-1 =
    none). lens [T] int32 (optional): number of valid indices at the front of
    each row; the rest are skipped. Returns bf16 [T, H, 512].

    Length-bounded loop at T = 1024, 24 heads: 4.97 -> 4.65 ms with 640 of 768
    valid, 5.29 -> 2.19 ms with 300 valid (bit-identical output).

    Defaults from a sweep on Arc Pro B70 (T = 512, 22 heads): 0.67 ms vs 4.32 ms
    for the generic kernel at topk 128, 2.12 ms vs 16.09 ms at topk 640; 16
    warps are needed to hold the [32, 512] fp32 accumulator without spilling."""
    T, H, D = q.shape
    topk = indices.shape[-1]
    assert topk % block_n == 0, (topk, block_n)
    out = torch.empty_like(q)
    kv = kv.reshape(-1, D)
    idx = indices.reshape(T, topk)
    _sparse_prefill_kernel[(T, triton.cdiv(H, block_h))](
        q, kv, idx, lens if lens is not None else idx,
        attn_sink if attn_sink is not None else q, out,
        H, kv.shape[0], topk, q.stride(0), q.stride(1), idx.stride(0),
        out.stride(0), out.stride(1), sm_scale * 1.4426950408889634,
        D=D, BLOCK_H=block_h, BLOCK_N=block_n, HAS_SINK=attn_sink is not None,
        HAS_LEN=lens is not None, num_warps=num_warps,
    )
    return out
