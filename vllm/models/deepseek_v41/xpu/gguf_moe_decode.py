# SPDX-License-Identifier: Apache-2.0
"""Decode-time routed MoE for GGUF experts (IQ1_M / IQ2_XXS / IQ2_XS) on XPU.

For a handful of (token, expert) pairs the batched fused-MoE kernel is
latency bound: its grid is sized for padded batches and each program walks the
whole K dimension serially. Here each program computes BLOCK_N output rows of
one pair over a slice of K (split-K), dequantizing whole 256-value superblocks
at once; the split-K partial sums are added afterwards.

Pairs whose expert is not on this rank have expert id -1: their programs exit
at once and their outputs stay zero, so there is no host sync on routing.
The per-element dequant formulas follow vllm-gguf-plugin's dequant kernels
(and ggml's reference)."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

QT_IQ2_XXS, QT_IQ2_XS, QT_IQ1_M = 16, 17, 29
_BLOCK_BYTES = {QT_IQ2_XXS: 66, QT_IQ2_XS: 74, QT_IQ1_M: 56}


@triton.jit
def _u16(ptrs):
    lo = tl.load(ptrs).to(tl.uint32)
    hi = tl.load(ptrs + 1).to(tl.uint32)
    return lo | (hi << 8)


@triton.jit
def _f16(ptrs):
    lo = tl.load(ptrs).to(tl.uint16)
    hi = tl.load(ptrs + 1).to(tl.uint16)
    return tl.cast(lo | (hi << 8), tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _u32(ptrs):
    b0 = tl.load(ptrs).to(tl.uint32)
    b1 = tl.load(ptrs + 1).to(tl.uint32)
    b2 = tl.load(ptrs + 2).to(tl.uint32)
    b3 = tl.load(ptrs + 3).to(tl.uint32)
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)


@triton.jit
def _deq_block(bp, pos, grid_ptr, sign_ptr, QTYPE: tl.constexpr):
    """bp: [BN, 1] pointers to one superblock per row; pos: [1, 256].
    Returns float32 [BN, 256]."""
    ib = pos // 32
    il = (pos % 32) // 8
    k = pos % 8
    if QTYPE == 29:  # IQ1_M, 56 bytes: qs[32] qh[16] scales[8]
        sc0 = _u16(bp + 48)
        sc1 = _u16(bp + 50)
        sc2 = _u16(bp + 52)
        sc3 = _u16(bp + 54)
        base_bits = (sc0 >> 12) | ((sc1 >> 8) & 0x00F0) | ((sc2 >> 4) & 0x0F00) | (sc3 & 0xF000)
        base = tl.cast(base_bits.to(tl.uint16), tl.float16, bitcast=True).to(tl.float32)
        qh = tl.load(bp + 32 + 2 * ib + (il // 2)).to(tl.int32)
        idx = tl.load(bp + 4 * ib + il).to(tl.int32)
        idx = idx | (((qh >> (4 * (il % 2))) & 0x07) << 8)
        delta = tl.where((qh & (0x08 << (4 * (il % 2)))) != 0, -9.0, -7.0)
        ib16 = 2 * ib + (il // 2)
        sc_sel = _u16(bp + 48 + 2 * (ib16 // 4))
        scale = base * (2.0 * ((sc_sel >> (3 * (ib16 % 4))) & 0x07).to(tl.float32) + 1.0) * 0.125
        g = tl.load(grid_ptr + idx * 8 + k).to(tl.float32)
        return (g * 8.0 + delta) * scale
    elif QTYPE == 16:  # IQ2_XXS, 66 bytes: d, then 8 x (4 grid bytes + u32 signs/scale)
        q2 = bp + 2 + 8 * ib
        aux = _u32(q2 + 4)
        d = _f16(bp)
        dscale = d * ((aux >> 28).to(tl.float32) + 0.5) * 0.25
        gi = tl.load(q2 + il).to(tl.int32)
        signs = tl.load(sign_ptr + ((aux >> (7 * il)) & 127))
        g = tl.load(grid_ptr + gi * 8 + k).to(tl.float32)
        s = tl.where((signs & (1 << k)) != 0, -1.0, 1.0)
        return g * s * dscale
    else:  # IQ2_XS, 74 bytes: d, qs[32] u16, scales[8]
        q2 = _u16(bp + 2 + 2 * (4 * ib + il)).to(tl.int32)
        gi = q2 & 0x1FF
        signs = tl.load(sign_ptr + (q2 >> 9))
        sb = tl.load(bp + 66 + ib)
        d = _f16(bp)
        scale = (((sb >> (4 * (il // 2))) & 0x0F).to(tl.float32) + 0.5) * 0.25
        g = tl.load(grid_ptr + gi * 8 + k).to(tl.float32)
        s = tl.where((signs & (1 << k)) != 0, -1.0, 1.0)
        return g * s * d * scale


@triton.jit
def _gemv_kernel(x_ptr, w_ptr, eid_ptr, y_ptr, grid_ptr, sign_ptr,
                 N, K, row_bytes, expert_stride, n_pairs,
                 QTYPE: tl.constexpr, BB: tl.constexpr, BLOCK_N: tl.constexpr,
                 KB_PER: tl.constexpr):
    pid_p = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    e = tl.load(eid_ptr + pid_p)
    if e < 0:
        return
    rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    rows = tl.where(rmask, rows, 0)
    pos = tl.arange(0, 256)
    n_kb = K // 256
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    row_base = w_ptr + e.to(tl.int64) * expert_stride + rows.to(tl.int64) * row_bytes
    for i in range(KB_PER):
        kb = pid_k * KB_PER + i
        if kb < n_kb:
            bp = (row_base + kb * BB)[:, None]
            vals = _deq_block(bp, pos[None, :], grid_ptr, sign_ptr, QTYPE)
            xv = tl.load(x_ptr + pid_p * K + kb * 256 + pos).to(tl.float32)
            acc += tl.sum(vals * xv[None, :], axis=1)
    tl.store(y_ptr + (pid_k * n_pairs + pid_p) * N + rows, acc, mask=rmask)


_tables: dict = {}


def _iq_tables(dev, qtype):
    key = (dev, qtype)
    if key not in _tables:
        from vllm_gguf_plugin.triton.gemm.iq_quant.iq_tables import get_iq_table_tensors

        t = get_iq_table_tensors(dev)
        grid = {QT_IQ1_M: "iq1s_grid", QT_IQ2_XXS: "iq2xxs_grid", QT_IQ2_XS: "iq2xs_grid"}[qtype]
        _tables[key] = (t[grid], t["ksigns_iq2xs"])
    return _tables[key]


def gguf_gemv(x: torch.Tensor, w: torch.Tensor, eid: torch.Tensor, qtype: int,
              block_n: int = 4, kb_per: int = 1) -> torch.Tensor:
    """x [P, K] (bf16/fp32), w [E, N, row_bytes] uint8, eid [P] int32 (-1 = skip).
    Returns float32 [P, N]: row p = W[eid[p]] @ x[p] (zeros where eid = -1)."""
    P, K = x.shape
    E, N, rb = w.shape
    assert K % 256 == 0 and rb == K // 256 * _BLOCK_BYTES[qtype], (K, rb, qtype)
    n_kb = K // 256
    split = triton.cdiv(n_kb, kb_per)
    y = torch.zeros(split, P, N, dtype=torch.float32, device=x.device)
    grid_t, sign_t = _iq_tables(x.device, qtype)
    x = x.contiguous()
    _gemv_kernel[(P, triton.cdiv(N, block_n), split)](
        x, w, eid, y, grid_t, sign_t, N, K, rb, w.stride(0), P,
        QTYPE=qtype, BB=_BLOCK_BYTES[qtype], BLOCK_N=block_n, KB_PER=kb_per,
    )
    return y.sum(0) if split > 1 else y[0]


def moe_decode(x, idx, w, out, w13, w2, t13, t2, e_start, n_local, limit):
    """Add the local experts' outputs for all (token, slot) pairs to out
    (float32 [T, H]); no host sync."""
    T, k = idx.shape
    local = (idx >= e_start) & (idx < e_start + n_local)
    eid = torch.where(local, idx - e_start, -1).to(torch.int32).reshape(-1)
    xs = x.repeat_interleave(k, dim=0)  # [T*k, H]
    h = gguf_gemv(xs, w13, eid, t13)  # [P, 2I]
    gate, up = h.chunk(2, dim=-1)
    if limit:
        gate = gate.clamp(max=limit)
        up = up.clamp(-limit, limit)
    a = torch.nn.functional.silu(gate) * up
    y = gguf_gemv(a, w2, eid, t2)  # [P, H]
    y = y * w.reshape(-1, 1).float()
    out += y.view(T, k, -1).sum(1)
