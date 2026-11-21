# SPDX-License-Identifier: Apache-2.0
"""INT4 compressed KV cache for DeepSeek V4.1 on XPU (VLLM_DS41_KV_INT4=1).

The compressed cache (the part that grows with context) uses a 384-byte record
instead of fp8_ds_mla's 584 bytes, so it holds ~1.5x the tokens; the 128-token
sliding-window cache keeps the fp8_ds_mla record.

Per token: the 448 NoPE values as symmetric INT4 (scale = amax / 7 per group
of 32, values q + 8 in [1, 15], two per byte, low nibble first), then the 64
RoPE values as bf16 (after RoPE). Page layout, like fp8_ds_mla's:
[CACHE_BLOCK * 352 value bytes (224 INT4 + 128 RoPE)][CACHE_BLOCK * 32 scale
bytes (14 fp16 scales + 4 pad)]. Quality was measured first with the
simulation in kv_int4_sim.py (group 32: perplexity +0.4-0.5%, needles 3/3).
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

ENABLED = os.environ.get("VLLM_DS41_KV_INT4", "0") == "1"
RECORD = 384
VAL_BYTES = 352
SCALE_BYTES = 32


def is_int4_cache(cache: torch.Tensor | None) -> bool:
    return cache is not None and cache.dtype == torch.uint8 and cache.shape[-1] == RECORD


# ------------------------------------------------------------------- writer
@triton.jit
def _insert_kernel(latent, positions, cos_sin, cache, cache_slots,
                   COS_STRIDE: tl.constexpr, CACHE_STRIDE: tl.constexpr,
                   CACHE_BLOCK: tl.constexpr, COMPRESS_RATIO: tl.constexpr):
    t = tl.program_id(0)
    slot = tl.load(cache_slots + t)
    if slot < 0:
        return
    position = tl.load(positions + t)
    if (position + 1) % COMPRESS_RATIO != 0:
        return
    d = tl.arange(0, 512)
    normed = tl.load(latent + t.to(tl.int64) * 512 + d).to(tl.float32)
    page = cache + (slot // CACHE_BLOCK).to(tl.int64) * CACHE_STRIDE
    values = page + (slot % CACHE_BLOCK) * 352
    scales = (page + CACHE_BLOCK * 352 + (slot % CACHE_BLOCK) * 32).to(tl.pointer_type(tl.float16))

    g = tl.reshape(normed, (16, 32))  # groups 14, 15 are RoPE dims: not stored as INT4
    amax = tl.max(tl.abs(g), 1)
    sc = (tl.maximum(amax, 1e-8) * (1.0 / 7.0)).to(tl.float16)
    sf = sc.to(tl.float32)
    q = tl.clamp(tl.floor(g / sf[:, None] + 0.5), -7.0, 7.0) + 8.0
    qi = tl.reshape(q.to(tl.int32), (256, 2))
    lo, hi = tl.split(qi)
    packed = (lo | (hi << 4)).to(tl.uint8)
    b = tl.arange(0, 256)
    tl.store(values + b, packed, mask=b < 224)
    s = tl.arange(0, 16)
    tl.store(scales + s, sc, mask=s < 14)

    even, odd = tl.split(tl.reshape(normed, (256, 2)))
    pair = tl.arange(0, 256) - 224
    cs = cos_sin + (position // COMPRESS_RATIO * COMPRESS_RATIO) * COS_STRIDE
    c = tl.load(cs + tl.maximum(pair, 0), pair >= 0, other=1.0).to(tl.float32)
    sn = tl.load(cs + 32 + tl.maximum(pair, 0), pair >= 0, other=0.0).to(tl.float32)
    rotated = tl.interleave(even * c - odd * sn, odd * c + even * sn)
    rope_dst = (values + 224).to(tl.pointer_type(tl.bfloat16))
    tl.store(rope_dst + d - 448, rotated.to(tl.bfloat16), d >= 448)


def insert(latent: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor,
           kv_cache: torch.Tensor, slot_mapping: torch.Tensor, compress_ratio: int) -> None:
    n = slot_mapping.numel()
    if n == 0:
        return
    assert latent.shape[1] == 512 and latent.is_contiguous()
    _insert_kernel[(n,)](
        latent, positions, cos_sin_cache, kv_cache, slot_mapping,
        COS_STRIDE=cos_sin_cache.stride(0), CACHE_STRIDE=kv_cache.stride(0),
        CACHE_BLOCK=kv_cache.shape[1], COMPRESS_RATIO=compress_ratio, num_warps=4,
    )


# ------------------------------------------------------------- record read
@triton.jit
def load_kv_i4(cache_ptr, slot, ok, bs, stride, d):
    """[BK] slots of an INT4 compressed cache -> float32 [BK, 512] (0 where not ok)."""
    slot = tl.where(ok, slot, 0).to(tl.int64)
    blk = slot // bs
    pos = slot % bs
    tok = cache_ptr + blk * stride + pos * 352  # [BK]
    is_q = d < 448
    byte = tl.load(tok[:, None] + (d // 2)[None, :], mask=ok[:, None] & is_q[None, :], other=136)
    nib = ((byte.to(tl.int32) >> ((d % 2) * 4)[None, :]) & 15).to(tl.float32) - 8.0
    sptr = (cache_ptr + blk * stride + bs.to(tl.int64) * 352 + pos * 32).to(
        tl.pointer_type(tl.float16))
    sc = tl.load(sptr[:, None] + (d // 32)[None, :], mask=ok[:, None] & is_q[None, :],
                 other=0.0).to(tl.float32)
    bptr = (tok + 224).to(tl.pointer_type(tl.bfloat16))
    bf = tl.load(bptr[:, None] + (d - 448)[None, :], mask=ok[:, None] & (~is_q)[None, :],
                 other=0.0).to(tl.float32)
    return tl.where(is_q[None, :], nib * sc, bf)


# ---------------------------------------------------------- prefill gather
@triton.jit
def _gather_kernel(out_ptr, cache_ptr, seq_lens_ptr, bt_ptr, stride_or, stride_ot,
                   stride_bt, bs, stride_cb, BLOCK_T: tl.constexpr):
    r = tl.program_id(0)
    tb = tl.program_id(1)
    n = tl.load(seq_lens_ptr + r)
    j = tb * BLOCK_T + tl.arange(0, BLOCK_T)
    ok = j < n
    blk = tl.load(bt_ptr + r * stride_bt + j // bs, mask=ok, other=0)
    slot = blk.to(tl.int64) * bs + j % bs
    d = tl.arange(0, 512)
    kv = load_kv_i4(cache_ptr, slot, ok, bs, stride_cb, d)
    tl.store(out_ptr + r * stride_or + j[:, None].to(tl.int64) * stride_ot + d[None, :],
             kv.to(out_ptr.dtype.element_ty), mask=ok[:, None])


def gather(out: torch.Tensor, cache: torch.Tensor, seq_lens: torch.Tensor,
           block_table: torch.Tensor, block_size: int, max_len: int | None = None) -> None:
    """out [R, maxN, 512] bf16: out[r, j] = record j of request r, j < seq_lens[r]
    (the compressed-cache use of dequantize_and_gather_k_cache: offset 0, all
    tokens)."""
    R = seq_lens.numel()
    if R == 0:
        return
    maxn = max_len if max_len is not None else out.shape[1]
    if maxn <= 0:
        return
    BT = 16
    _gather_kernel[(R, triton.cdiv(maxn, BT))](
        out, cache, seq_lens, block_table, out.stride(0), out.stride(1),
        block_table.stride(0), block_size, cache.stride(0), BLOCK_T=BT, num_warps=4,
    )


# ------------------------------------------------------------ installation
def install() -> bool:
    """Route the compressor's cache writes for INT4 caches to `insert`."""
    if not ENABLED:
        return False
    import vllm.models.deepseek_v41.compressor as comp

    orig = comp.rope_quant_insert
    if getattr(orig, "_kv_int4", False):
        return True

    def rope_quant_insert(latent, positions, cos_sin_cache, kv_cache, slot_mapping,
                          compress_ratio, fp8_scale=None):
        if is_int4_cache(kv_cache):
            return insert(latent, positions, cos_sin_cache, kv_cache, slot_mapping,
                          compress_ratio)
        return orig(latent, positions, cos_sin_cache, kv_cache, slot_mapping,
                    compress_ratio, fp8_scale=fp8_scale)

    rope_quant_insert._kv_int4 = True
    comp.rope_quant_insert = rope_quant_insert
    return True
