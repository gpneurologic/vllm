# SPDX-License-Identifier: Apache-2.0
"""Prefill (many tokens) GGUF MoE GEMM for DeepSeek V4.1 on XPU.

Pairs (token, expert) are sorted by expert and padded to BLOCK_M rows per
block (moe_align_block_size). Each program decodes a [BLOCK_N, 256] tile of
IQ weights (one superblock per row) to bf16 in registers and multiplies it
with the block's [BLOCK_M, 256] activations with tl.dot, which Triton maps to
the XMX (DPAS) units. Weights are read once per BLOCK_M pairs, not per pair.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

try:
    from .gguf_moe_decode import (_BLOCK_BYTES, _deq_groups, _f16, _grid8, _iq_tables,
                                  _signs8, _u16, _u32)
except ImportError:  # loaded by path in kernel tests
    from gguf_moe_decode import (_BLOCK_BYTES, _deq_groups, _f16, _grid8, _iq_tables,
                                 _signs8, _u16, _u32)


@triton.jit
def _deq_part(bp, G0: tl.constexpr, NG: tl.constexpr, grid64_ptr, sign_ptr,
              QTYPE: tl.constexpr):
    """Groups G0 .. G0+NG-1 (8 values each) of the superblocks at bp [BN, 1].
    Returns float32 [BN, NG, 8]."""
    g = G0 + tl.arange(0, NG)[None, :]
    ib = g // 4
    il = g % 4
    if QTYPE == 29:  # IQ1_M: qs[32] qh[16] scales[8]
        sc0 = _u16(bp + 48)
        sc1 = _u16(bp + 50)
        sc2 = _u16(bp + 52)
        sc3 = _u16(bp + 54)
        base_bits = (sc0 >> 12) | ((sc1 >> 8) & 0x00F0) | ((sc2 >> 4) & 0x0F00) | (sc3 & 0xF000)
        base = tl.cast(base_bits.to(tl.uint16), tl.float16, bitcast=True).to(tl.float32)
        qh = tl.load(bp + 32 + g // 2).to(tl.int32)
        sh = 4 * (g % 2)
        gi = tl.load(bp + g).to(tl.int32) | (((qh >> sh) & 0x07) << 8)
        delta = tl.where((qh & (0x08 << sh)) != 0, -9.0, -7.0)
        ib16 = g // 2
        sc_sel = _u16(bp + 48 + 2 * (ib16 // 4))
        scale = base * (2.0 * ((sc_sel >> (3 * (ib16 % 4)).to(tl.uint32)) & 0x07).to(tl.float32)
                        + 1.0) * 0.125
        gv = _grid8(grid64_ptr, gi)
        return (gv * 8.0 + delta[:, :, None]) * scale[:, :, None]
    elif QTYPE == 16:  # IQ2_XXS
        d = _f16(bp)
        q2 = bp + 2 + 8 * ib
        aux = _u32(q2 + 4)
        scale = d * ((aux >> 28).to(tl.float32) + 0.5) * 0.25
        gi = tl.load(q2 + il).to(tl.int32)
        si = ((aux >> (7 * il).to(tl.uint32)) & 127).to(tl.int32)
        return _grid8(grid64_ptr, gi) * _signs8(sign_ptr, si) * scale[:, :, None]
    else:  # IQ2_XS
        d = _f16(bp)
        q2 = _u16(bp + 2 + 2 * g).to(tl.int32)
        sb = tl.load(bp + 66 + ib).to(tl.int32)
        scale = d * (((sb >> (4 * (il // 2))) & 0x0F).to(tl.float32) + 0.5) * 0.25
        return _grid8(grid64_ptr, q2 & 0x1FF) * _signs8(sign_ptr, q2 >> 9) * scale[:, :, None]


@triton.jit
def _mmq_kernel_k(x_ptr, w_ptr, y_ptr, sorted_ptr, eblk_ptr, ntpp_ptr, grid64_ptr, sign_ptr,
                  N, K, row_bytes, expert_stride, n_pairs, n_experts,
                  QTYPE: tl.constexpr, BB: tl.constexpr, BLOCK_M: tl.constexpr,
                  BLOCK_N: tl.constexpr, NG: tl.constexpr):
    """Like _mmq_kernel, but decodes and multiplies NG*8 values of K at a time
    (smaller decoded tile: fewer registers)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # moe_align_block_size allocates more blocks than it fills and leaves the
    # expert ids of the unused ones uninitialized: skip them, and any id out
    # of range, or the kernel reads weights past the end of w.
    if pid_m * BLOCK_M >= tl.load(ntpp_ptr):
        return
    e = tl.load(eblk_ptr + pid_m)
    if (e < 0) | (e >= n_experts):
        return
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    pair = tl.load(sorted_ptr + offs_m)
    pmask = pair < n_pairs
    pair = tl.where(pmask, pair, 0)
    rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    rows = tl.where(rmask, rows, 0)
    row_base = w_ptr + e.to(tl.int64) * expert_stride + rows.to(tl.int64) * row_bytes
    xk = tl.arange(0, NG * 8)
    x_rows = x_ptr + pair.to(tl.int64)[:, None] * K + xk[None, :]
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for kb in range(0, K // 256):
        bp = (row_base + kb * BB)[:, None]
        for c in tl.static_range(0, 32 // NG):
            w = tl.reshape(_deq_part(bp, c * NG, NG, grid64_ptr, sign_ptr, QTYPE),
                           (BLOCK_N, NG * 8))
            xv = tl.load(x_rows + kb * 256 + c * NG * 8, mask=pmask[:, None], other=0.0)
            acc += tl.dot(xv.to(tl.bfloat16), tl.trans(w.to(tl.bfloat16)))
    tl.store(y_ptr + pair.to(tl.int64)[:, None] * N + rows[None, :], acc,
             mask=pmask[:, None] & rmask[None, :])


@triton.jit
def _mmq_kernel(x_ptr, w_ptr, y_ptr, sorted_ptr, eblk_ptr, ntpp_ptr, grid64_ptr, sign_ptr,
                N, K, row_bytes, expert_stride, n_pairs, n_experts,
                QTYPE: tl.constexpr, BB: tl.constexpr, BLOCK_M: tl.constexpr,
                BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # moe_align_block_size allocates more blocks than it fills and leaves the
    # expert ids of the unused ones uninitialized: skip them, and any id out
    # of range, or the kernel reads weights past the end of w.
    if pid_m * BLOCK_M >= tl.load(ntpp_ptr):
        return
    e = tl.load(eblk_ptr + pid_m)
    if (e < 0) | (e >= n_experts):
        return
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    pair = tl.load(sorted_ptr + offs_m)
    pmask = pair < n_pairs
    pair = tl.where(pmask, pair, 0)
    rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    rows = tl.where(rmask, rows, 0)
    row_base = w_ptr + e.to(tl.int64) * expert_stride + rows.to(tl.int64) * row_bytes
    xk = tl.arange(0, 256)
    x_rows = x_ptr + pair.to(tl.int64)[:, None] * K + xk[None, :]
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for kb in range(0, K // 256):
        bp = (row_base + kb * BB)[:, None]
        w = tl.reshape(_deq_groups(bp, grid64_ptr, sign_ptr, QTYPE), (BLOCK_N, 256))
        xv = tl.load(x_rows + kb * 256, mask=pmask[:, None], other=0.0)
        acc += tl.dot(xv.to(tl.bfloat16), tl.trans(w.to(tl.bfloat16)))
    tl.store(y_ptr + pair.to(tl.int64)[:, None] * N + rows[None, :], acc,
             mask=pmask[:, None] & rmask[None, :])


QT_IQ2_XXS, QT_IQ2_XS, QT_IQ1_M = 16, 17, 29
BLOCK_M = 32
# (BLOCK_N, num_warps, NG) per (type, K), from a sweep on Arc Pro B70 with
# 512-token chunks (w13: K = 5120, w2: K = 2304). Only settings that matched
# the dequantized reference are listed: Triton's XPU code generation gave
# wrong results for BLOCK_M = 64 and for IQ2_XXS with NG = 4.
_CFG = {
    (QT_IQ1_M, 5120): (128, 4, 8), (QT_IQ1_M, 2304): (64, 4, 4),
    (QT_IQ2_XXS, 5120): (128, 8, 8), (QT_IQ2_XXS, 2304): (128, 8, 8),
    (QT_IQ2_XS, 5120): (128, 4, 4), (QT_IQ2_XS, 2304): (128, 4, 4),
}


def gguf_moe_mmq(x: torch.Tensor, w: torch.Tensor, qtype: int, sorted_ids: torch.Tensor,
                 expert_blocks: torch.Tensor, ntpp: torch.Tensor, block_m: int, n_pairs: int,
                 block_n: int | None = None, num_warps: int | None = None,
                 ng: int = -1) -> torch.Tensor:
    """x [P, K] bf16 (pair rows), w [E, N, rb] uint8; sorted_ids, expert_blocks
    and ntpp (num_tokens_post_padded) from moe_align_block_size(eid, block_m, E).
    Returns float32 [P, N]."""
    P, K = x.shape
    E, N, rb = w.shape
    assert K % 256 == 0 and rb == K // 256 * _BLOCK_BYTES[qtype], (K, rb, qtype)
    bn, nw, g = _CFG.get((qtype, K), (64, 4, 8))
    block_n = block_n or bn
    num_warps = num_warps or nw
    ng = ng if ng >= 0 else g
    y = torch.empty(P, N, dtype=torch.float32, device=x.device)
    _, grid64, sign_t = _iq_tables(x.device, qtype)
    n_blocks = expert_blocks.numel()
    args = (x.contiguous(), w, y, sorted_ids, expert_blocks, ntpp, grid64, sign_t,
            N, K, rb, w.stride(0), n_pairs, E)
    if ng:
        _mmq_kernel_k[(n_blocks, triton.cdiv(N, block_n))](
            *args, QTYPE=qtype, BB=_BLOCK_BYTES[qtype], BLOCK_M=block_m, BLOCK_N=block_n,
            NG=ng, num_warps=num_warps)
    else:
        _mmq_kernel[(n_blocks, triton.cdiv(N, block_n))](
            *args, QTYPE=qtype, BB=_BLOCK_BYTES[qtype], BLOCK_M=block_m, BLOCK_N=block_n,
            num_warps=num_warps)
    return y


# ------------------------------------------------------------- self-check
_checked: dict = {}


def _plugin_moe():
    from vllm_gguf_plugin.triton.fused_moe.interface import ggml_moe_a8_triton
    from vllm_gguf_plugin.triton.fused_moe.utils import get_triton_moe_block_m

    return ggml_moe_a8_triton, get_triton_moe_block_m


def mmq_ok(w: torch.Tensor, qtype: int) -> bool:
    """Once per (type, K, N): compare the new kernel with the plugin's on random
    activations spread over the first experts. Triton's XPU code generation has
    produced wrong results for some tile settings; this guards against that."""
    E, N, rb = w.shape
    K = rb // _BLOCK_BYTES[qtype] * 256
    key = (qtype, K, N)
    if key not in _checked:
        from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

        moe, block_m = _plugin_moe()
        g = torch.Generator(device=w.device).manual_seed(0)
        P = 256
        eid = torch.arange(P, device=w.device, dtype=torch.int32).remainder(min(E, 8)).view(-1, 1)
        x = (torch.randn(P, K, device=w.device, generator=g) * 0.5).to(torch.bfloat16)
        bm = block_m(qtype)
        s0, e0, n0 = moe_align_block_size(eid, bm, E)
        ref = moe(x, w, s0, e0, n0, qtype, N, 1, P).float()
        # Garbage in the unused expert-id slots, as in a real run, so the
        # check covers the skip logic too.
        s1, e1, n1 = moe_align_block_size(eid, BLOCK_M, E)
        nused = -(-int(n1.item()) // BLOCK_M)
        e1[nused:] = 1 << 20
        got = gguf_moe_mmq(x, w, qtype, s1, e1, n1, BLOCK_M, P)
        rel = ((got - ref).norm() / ref.norm().clamp_min(1e-12)).item()
        ok = rel < 0.02
        _checked[key] = ok
        from vllm.logger import init_logger

        log = init_logger(__name__)
        if ok:
            log.info("DeepSeek V4.1 GGUF prefill MoE: tl.dot kernel for type %d K=%d N=%d "
                     "(rel diff vs plugin %.4f)", qtype, K, N, rel)
        else:
            log.warning("DeepSeek V4.1 GGUF prefill MoE: tl.dot kernel mismatch for type %d "
                        "K=%d N=%d (rel diff %.4f); using the plugin kernel", qtype, K, N, rel)
    return _checked[key]
