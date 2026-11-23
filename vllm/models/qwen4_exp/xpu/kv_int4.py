# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT4 main K/V cache for XPU QSA attention (`VLLM_QWEN4_KV_INT4=1`).

Used with ``--kv-cache-dtype fp8``. One packed slot per (token, KV head):

    [0, R)                 K RoPE dims as FP8 e4m3 (R = ROPE_DIM bytes)
    [R, R + (D-R)/2)       K non-RoPE dims, INT4, two per byte (low nibble first)
    [.., + D/2)            V, INT4, two per byte
    [.., + 2*(D-R)/G)      K scales, fp16, one per G dims
    [.., + 2*D/G)          V scales, fp16
    padded to 16 bytes

INT4 is symmetric: q = round(x / s) in [-7, 7], s = amax / 7, stored as q + 8.
For D = 256, R = 64, G = 32 a slot is 320 bytes instead of FP8's 512. This is
the format measured by ``VLLM_QWEN4_SIM_KV=int4g32`` (RoPE dims kept, then
rounded to FP8 by the cache): +0.20% / +0.41% perplexity against bf16.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton

ENABLED = os.environ.get("VLLM_QWEN4_KV_INT4", "0") == "1"
GROUP = 32
ROPE_DIM = int(os.environ.get("VLLM_QWEN4_KV_INT4_ROPE", "64"))


@dataclass(frozen=True)
class Layout:
    head_dim: int
    rope: int
    group: int
    k_nib: int  # byte offset of K nibbles
    v_nib: int
    k_scale: int
    v_scale: int
    slot: int


def layout(head_dim: int) -> Layout:
    rope, group = ROPE_DIM, GROUP
    assert head_dim % group == 0 and (head_dim - rope) % group == 0
    k_nib = rope
    v_nib = k_nib + (head_dim - rope) // 2
    k_scale = v_nib + head_dim // 2
    v_scale = k_scale + 2 * ((head_dim - rope) // group)
    end = v_scale + 2 * (head_dim // group)
    slot = (end + 15) // 16 * 16
    return Layout(head_dim, rope, group, k_nib, v_nib, k_scale, v_scale, slot)


def slot_bytes(head_dim: int) -> int:
    return layout(head_dim).slot


@triton.jit
def _round_clamp(x):
    r = tl.where(x >= 0, tl.floor(x + 0.5), -tl.floor(0.5 - x))
    return tl.minimum(tl.maximum(r, -7.0), 7.0)


@triton.jit(do_not_specialize=["num_tokens"])
def _kv_int4_store_kernel(
    k_ptr,
    v_ptr,
    cache_ptr,
    slot_ptr,
    stride_k_tok,
    stride_k_head,
    stride_v_tok,
    stride_v_head,
    stride_c_block,
    stride_c_token,
    stride_c_head,
    num_tokens,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE: tl.constexpr,
    GROUP: tl.constexpr,
    K_NIB: tl.constexpr,
    V_NIB: tl.constexpr,
    K_SCALE: tl.constexpr,
    V_SCALE: tl.constexpr,
    NGK: tl.constexpr,
    NGK_P2: tl.constexpr,
    NGV: tl.constexpr,
    NGV_P2: tl.constexpr,
) -> None:
    t = tl.program_id(0)
    h = tl.program_id(1)
    slot = tl.load(slot_ptr + t).to(tl.int64)
    if slot < 0:
        return
    page = slot // PAGE_SIZE
    off = slot % PAGE_SIZE
    base = cache_ptr + page * stride_c_block + off * stride_c_token + h * stride_c_head
    kp = k_ptr + t * stride_k_tok + h * stride_k_head
    vp = v_ptr + t * stride_v_tok + h * stride_v_head

    # K RoPE dims: FP8 e4m3.
    r = tl.arange(0, ROPE)
    kr = tl.load(kp + r).to(tl.float32).to(tl.float8e4nv)
    tl.store(base + r, kr.to(tl.uint8, bitcast=True))

    half: tl.constexpr = GROUP // 2
    j = tl.arange(0, half)[None, :]

    # K non-RoPE dims.
    g = tl.arange(0, NGK_P2)[:, None]
    mk = g < NGK
    ke = tl.load(kp + ROPE + g * GROUP + 2 * j, mask=mk, other=0.0).to(tl.float32)
    ko = tl.load(kp + ROPE + g * GROUP + 2 * j + 1, mask=mk, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(ke), axis=1), tl.max(tl.abs(ko), axis=1))
    scale = tl.maximum(amax / 7.0, 1e-8).to(tl.float16)
    sf = scale.to(tl.float32)[:, None]
    qe = _round_clamp(ke / sf).to(tl.int32) + 8
    qo = _round_clamp(ko / sf).to(tl.int32) + 8
    tl.store(base + K_NIB + g * half + j, (qe | (qo << 4)).to(tl.uint8), mask=mk)
    gs = tl.arange(0, NGK_P2)
    tl.store(
        (base + K_SCALE + 2 * gs).to(tl.pointer_type(tl.float16)),
        scale,
        mask=gs < NGK,
    )

    # V.
    g = tl.arange(0, NGV_P2)[:, None]
    mv = g < NGV
    ve = tl.load(vp + g * GROUP + 2 * j, mask=mv, other=0.0).to(tl.float32)
    vo = tl.load(vp + g * GROUP + 2 * j + 1, mask=mv, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(ve), axis=1), tl.max(tl.abs(vo), axis=1))
    scale = tl.maximum(amax / 7.0, 1e-8).to(tl.float16)
    sf = scale.to(tl.float32)[:, None]
    qe = _round_clamp(ve / sf).to(tl.int32) + 8
    qo = _round_clamp(vo / sf).to(tl.int32) + 8
    tl.store(base + V_NIB + g * half + j, (qe | (qo << 4)).to(tl.uint8), mask=mv)
    gs = tl.arange(0, NGV_P2)
    tl.store(
        (base + V_SCALE + 2 * gs).to(tl.pointer_type(tl.float16)),
        scale,
        mask=gs < NGV,
    )


def cache_view(kv_cache: torch.Tensor) -> torch.Tensor:
    """(B, H, N, C) manager view -> (B, N, H, C) uint8, C = slot bytes."""
    if kv_cache.dtype != torch.uint8:
        kv_cache = kv_cache.view(torch.uint8)
    return kv_cache.transpose(1, 2)


def store(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    head_dim: int,
) -> None:
    n = slot_mapping.shape[0]
    if n == 0:
        return
    lay = layout(head_dim)
    k = key[:n].reshape(n, -1, head_dim)
    v = value[:n].reshape(n, -1, head_dim)
    if k.stride(-1) != 1:
        k = k.contiguous()
    if v.stride(-1) != 1:
        v = v.contiguous()
    cache = cache_view(kv_cache)
    assert cache.shape[-1] == lay.slot, (cache.shape, lay.slot)
    assert cache.stride(-1) == 1
    ngk = (head_dim - lay.rope) // lay.group
    ngv = head_dim // lay.group
    _kv_int4_store_kernel[(n, k.shape[1])](
        k,
        v,
        cache,
        slot_mapping,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        n,
        PAGE_SIZE=cache.shape[1],
        HEAD_DIM=head_dim,
        ROPE=lay.rope,
        GROUP=lay.group,
        K_NIB=lay.k_nib,
        V_NIB=lay.v_nib,
        K_SCALE=lay.k_scale,
        V_SCALE=lay.v_scale,
        NGK=ngk,
        NGK_P2=triton.next_power_of_2(ngk),
        NGV=ngv,
        NGV_P2=triton.next_power_of_2(ngv),
        num_warps=1,
    )


def reference_dequant(cache: torch.Tensor, head_dim: int) -> tuple[torch.Tensor, ...]:
    """(..., slot) uint8 -> K, V float32 (..., head_dim). For tests."""
    lay = layout(head_dim)
    c = cache
    rope = c[..., : lay.rope].contiguous().view(torch.float8_e4m3fn).float()

    def nib(start: int, count: int) -> torch.Tensor:
        b = c[..., start : start + count // 2].to(torch.int32)
        q = torch.stack([b & 15, b >> 4], dim=-1).flatten(-2) - 8
        return q.float()

    def scales(start: int, count: int) -> torch.Tensor:
        return c[..., start : start + 2 * count].contiguous().view(torch.float16).float()

    nk = head_dim - lay.rope
    ks = scales(lay.k_scale, nk // lay.group).repeat_interleave(lay.group, -1)
    vs = scales(lay.v_scale, head_dim // lay.group).repeat_interleave(lay.group, -1)
    k = torch.cat([rope, nib(lay.k_nib, nk) * ks], dim=-1)
    v = nib(lay.v_nib, head_dim) * vs
    return k, v
