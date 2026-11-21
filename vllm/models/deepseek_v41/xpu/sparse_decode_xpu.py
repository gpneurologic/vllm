# SPDX-License-Identifier: Apache-2.0
"""Fused sparse MLA decode for DeepSeek V4.1 on XPU, straight from the FP8
caches (flash-decoding split-K).

Replaces xpu_sparse_decode_fp8 (gather + dequant into a bf16 workspace,
~15 index ops, a bf16 attention kernel and a copy) for small batches.

Keys of a token: its top-k compressed slots (global slot ids into the
compressed cache, `topk_lens` valid) followed by its sliding-window slots
(into the SWA cache, `swa_lens` valid); -1 slots are skipped. Each cached
token (fp8_ds_mla layout, 576 bytes + 8 scale bytes per token):
448 FP8 e4m3 values with one UE8M0 scale per 64, then 64 bf16 RoPE values;
block layout [block_size * 576 token bytes][block_size * 8 scale bytes].
The 512-value latent is both key and value; out = sum_i p_i v_i with the
softmax over the selected keys plus one sink logit per head.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

try:
    from .kv_int4 import RECORD as _I4_RECORD
    from .kv_int4 import load_kv_i4
except ImportError:  # loaded by path in kernel tests
    from kv_int4 import RECORD as _I4_RECORD
    from kv_int4 import load_kv_i4

LOG2E = 1.4426950408889634


@triton.jit
def _load_kv(cache_ptr, slot, ok, bs, stride, d):
    """[BK] slots -> float32 [BK, 512] latents (0 where not ok)."""
    slot = tl.where(ok, slot, 0).to(tl.int64)
    blk = slot // bs
    pos = slot % bs
    tok = cache_ptr + blk * stride + pos * 576  # [BK]
    is_fp8 = d < 448
    raw = tl.load(tok[:, None] + d[None, :], mask=ok[:, None] & is_fp8[None, :], other=0)
    f8 = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    sc = tl.load(cache_ptr + blk[:, None] * stride + bs.to(tl.int64) * 576 + pos[:, None] * 8
                 + (d // 64)[None, :], mask=ok[:, None] & is_fp8[None, :], other=127)
    f8 = f8 * tl.exp2(sc.to(tl.float32) - 127.0)
    bptr = (tok + 448).to(tl.pointer_type(tl.bfloat16))
    bf = tl.load(bptr[:, None] + (d - 448)[None, :], mask=ok[:, None] & (~is_fp8)[None, :],
                 other=0.0).to(tl.float32)
    return tl.where(is_fp8[None, :], f8, bf)


@triton.jit
def _decode_kernel(q_ptr, ckv_ptr, skv_ptr, topk_ptr, tlen_ptr, swa_ptr, slen_ptr,
                   acc_ptr, ml_ptr,
                   sq_t, sq_h, H, max_topk, max_swa, c_bs, c_stride, s_bs, s_stride,
                   sm_scale_log2,
                   HAS_TOPK: tl.constexpr, BH: tl.constexpr, BK: tl.constexpr,
                   KEYS_PER: tl.constexpr, NSPLIT: tl.constexpr,
                   C_INT4: tl.constexpr = False):
    t = tl.program_id(0)
    hb = tl.program_id(1)
    sp = tl.program_id(2)
    heads = hb * BH + tl.arange(0, BH)
    hm = heads < H
    d = tl.arange(0, 512)
    q = tl.load(q_ptr + t * sq_t + heads[:, None] * sq_h + d[None, :], mask=hm[:, None],
                other=0.0).to(tl.bfloat16)
    if HAS_TOPK:
        tlen = tl.load(tlen_ptr + t)
    else:
        tlen = 0
    slen = tl.load(slen_ptr + t)
    m = tl.full([BH], float("-inf"), tl.float32)
    lsum = tl.zeros([BH], tl.float32)
    acc = tl.zeros([BH, 512], tl.float32)
    total = max_topk + max_swa
    for kk in range(0, KEYS_PER, BK):
        v = sp * KEYS_PER + kk + tl.arange(0, BK)
        in_c = (v < max_topk) & (v < tlen)
        u = v - max_topk
        in_s = (v >= max_topk) & (u < slen) & (v < total)
        ci = tl.load(topk_ptr + t * max_topk + v, mask=in_c, other=-1)
        si = tl.load(swa_ptr + t * max_swa + u, mask=in_s, other=-1)
        ok_c = in_c & (ci >= 0)
        ok_s = in_s & (si >= 0)
        if C_INT4:
            kv_c = load_kv_i4(ckv_ptr, ci, ok_c, c_bs, c_stride, d)
        else:
            kv_c = _load_kv(ckv_ptr, ci, ok_c, c_bs, c_stride, d)
        kv = kv_c + _load_kv(skv_ptr, si, ok_s, s_bs, s_stride, d)
        kvb = kv.to(tl.bfloat16)
        s = tl.dot(q, tl.trans(kvb)) * sm_scale_log2  # [BH, BK]
        valid = ok_c | ok_s
        s = tl.where(valid[None, :], s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        lsum = lsum * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kvb)
        m = m_new
    base = (t * NSPLIT + sp) * H + heads
    tl.store(acc_ptr + base[:, None] * 512 + d[None, :], acc, mask=hm[:, None])
    tl.store(ml_ptr + base * 2, m, mask=hm)
    tl.store(ml_ptr + base * 2 + 1, lsum, mask=hm)


@triton.jit
def _combine_kernel(acc_ptr, ml_ptr, sink_ptr, out_ptr, so_t, so_h, H,
                    NSPLIT: tl.constexpr, NSP_P: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    sp = tl.arange(0, NSP_P)
    spm = sp < NSPLIT
    base = (t * NSPLIT + sp) * H + h
    m = tl.load(ml_ptr + base * 2, mask=spm, other=float("-inf"))
    lsum = tl.load(ml_ptr + base * 2 + 1, mask=spm, other=0.0)
    sink = tl.load(sink_ptr + h) * 1.4426950408889634
    mx = tl.maximum(tl.max(m, axis=0), sink)
    w = tl.where(m == float("-inf"), 0.0, tl.exp2(m - mx))
    denom = tl.sum(w * lsum, axis=0) + tl.exp2(sink - mx)
    d = tl.arange(0, 512)
    a = tl.load(acc_ptr + base[:, None] * 512 + d[None, :], mask=spm[:, None], other=0.0)
    o = tl.sum(a * w[:, None], axis=0) / tl.where(denom > 0, denom, 1.0)
    tl.store(out_ptr + t * so_t + h * so_h + d, o.to(out_ptr.dtype.element_ty))


def sparse_decode_fused(q, kv_cache, swa_kv_cache, topk_indices, topk_lens, swa_indices,
                        swa_lens, attn_sink, softmax_scale, out, nsplit: int = 8,
                        bh: int = 16, bk: int = 16, num_warps: int = 8) -> None:
    """q [T, H, 512]; caches [num_blocks, block_size, 584] uint8 views;
    topk_indices [T, 1, max_topk] or None; swa_indices [T, 1, max_swa];
    out [T, H, 512]."""
    T, H, D = q.shape
    assert D == 512
    if T == 0:
        return
    has_topk = topk_indices is not None and kv_cache is not None
    swa = swa_indices.reshape(T, -1).to(torch.int32).contiguous()
    max_swa = swa.shape[1]
    if has_topk:
        topk = topk_indices.reshape(T, -1).to(torch.int32).contiguous()
        max_topk = topk.shape[1]
        tl_ = topk_lens.reshape(-1).to(torch.int32).contiguous()
        ckv, c_bs, c_stride = kv_cache, kv_cache.shape[1], kv_cache.stride(0)
    else:
        topk, max_topk, tl_ = swa, 0, swa_lens
        ckv, c_bs, c_stride = swa_kv_cache, swa_kv_cache.shape[1], swa_kv_cache.stride(0)
    sl = swa_lens.reshape(-1).to(torch.int32).contiguous()
    total = max_topk + max_swa
    keys_per = triton.cdiv(triton.cdiv(total, nsplit), bk) * bk
    nsplit = triton.cdiv(total, keys_per)
    acc = torch.empty(T, nsplit, H, 512, dtype=torch.float32, device=q.device)
    ml = torch.empty(T, nsplit, H, 2, dtype=torch.float32, device=q.device)
    _decode_kernel[(T, triton.cdiv(H, bh), nsplit)](
        q, ckv, swa_kv_cache, topk, tl_, swa, sl, acc, ml,
        q.stride(0), q.stride(1), H, max_topk, max_swa, c_bs, c_stride,
        swa_kv_cache.shape[1], swa_kv_cache.stride(0), softmax_scale * LOG2E,
        HAS_TOPK=has_topk, BH=bh, BK=bk, KEYS_PER=keys_per, NSPLIT=nsplit,
        C_INT4=bool(has_topk and ckv.shape[-1] == _I4_RECORD),
        num_warps=num_warps,
    )
    _combine_kernel[(T, H)](acc, ml, attn_sink, out, out.stride(0), out.stride(1), H,
                            NSPLIT=nsplit, NSP_P=triton.next_power_of_2(nsplit), num_warps=4)
