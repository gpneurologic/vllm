# SPDX-License-Identifier: Apache-2.0
"""Triton GPT-J (interleaved pair) RoPE for DeepSeek V4.1 on XPU.

- `gptj_rope_`: in place on the last `rope_dim` of x [T, H, D] (one launch;
  the torch version was ~15 launches);
- `inv_rope_fp8_round`: inverse RoPE plus the FP8 e4m3 round trip with
  power-of-two scales per 32 values that DeepSeek applies to the attention
  output before wo_a, into a new bf16 [T, H * D] tensor (one launch, was
  ~25).
cos_sin: [max_pos, rope_dim], first half cos, second half sin."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(x_ptr, pos_ptr, cs_ptr, s_t, s_h, D, cs_stride,
                 ROPE: tl.constexpr, INVERSE: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    p = tl.load(pos_ptr + t)
    half = ROPE // 2
    i = tl.arange(0, ROPE)
    base = x_ptr + t * s_t + h * s_h + (D - ROPE)
    x = tl.load(base + i).to(tl.float32)
    xp = tl.load(base + (i ^ 1)).to(tl.float32)  # pair partner
    pair = i // 2
    c = tl.load(cs_ptr + p * cs_stride + pair).to(tl.float32)
    s = tl.load(cs_ptr + p * cs_stride + half + pair).to(tl.float32)
    if INVERSE:
        s = -s
    even = (i % 2) == 0
    # even: e*c - o*s ; odd: e*s + o*c  (partner of odd is e, of even is o)
    out = tl.where(even, x * c - xp * s, xp * s + x * c)
    tl.store(base + i, out.to(x_ptr.dtype.element_ty))


def gptj_rope_(x: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor,
               inverse: bool = False) -> None:
    """In place on x [T, H, D] or [T, D] (last dim contiguous)."""
    x3 = x if x.dim() == 3 else x.unsqueeze(1)
    T, H, D = x3.shape
    if T == 0:
        return
    assert x3.stride(2) == 1
    rope = cos_sin.shape[-1]
    _rope_kernel[(T, H)](x3, positions, cos_sin, x3.stride(0), x3.stride(1), D,
                         cos_sin.stride(0), ROPE=rope, INVERSE=inverse, num_warps=1)


@triton.jit
def _inv_rope_round_kernel(x_ptr, pos_ptr, cs_ptr, y_ptr, s_t, s_h, H, cs_stride,
                           D: tl.constexpr, ROPE: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    p = tl.load(pos_ptr + t)
    half = ROPE // 2
    nb: tl.constexpr = D // 32
    i = tl.arange(0, nb)[:, None] * 32 + tl.arange(0, 32)[None, :]  # [nb, 32]
    base = x_ptr + t * s_t + h * s_h
    x = tl.load(base + i).to(tl.float32)
    r = i - (D - ROPE)
    in_rope = r >= 0
    rr = tl.where(in_rope, r, 0)
    xp = tl.load(base + (i ^ 1), mask=in_rope, other=0.0).to(tl.float32)
    pair = rr // 2
    c = tl.load(cs_ptr + p * cs_stride + pair, mask=in_rope, other=1.0).to(tl.float32)
    s = -tl.load(cs_ptr + p * cs_stride + half + pair, mask=in_rope, other=0.0).to(tl.float32)
    even = (i % 2) == 0
    rot = tl.where(even, x * c - xp * s, xp * s + x * c)
    x = tl.where(in_rope, rot, x)
    # FP8 round trip, power-of-two scale per 32 values:
    # s = 2^ceil(log2(max(amax, 1e-4) / 448))
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4)  # [nb]
    e = tl.ceil(tl.log2(amax / 448.0))
    sc = tl.exp2(e)[:, None]
    q = tl.minimum(tl.maximum(x / sc, -448.0), 448.0)
    q = q.to(tl.float8e4nv).to(tl.float32) * sc
    tl.store(y_ptr + (t * H + h) * D + i, q.to(y_ptr.dtype.element_ty))


def inv_rope_fp8_round(o: torch.Tensor, positions: torch.Tensor,
                       cos_sin: torch.Tensor) -> torch.Tensor:
    """o [T, H, D] (last dim contiguous) -> bf16 [T, H * D]."""
    T, H, D = o.shape
    y = torch.empty(T, H * D, dtype=torch.bfloat16, device=o.device)
    if T == 0:
        return y
    assert o.stride(2) == 1 and D % 32 == 0
    _inv_rope_round_kernel[(T, H)](o, positions, cos_sin, y, o.stride(0), o.stride(1), H,
                                   cos_sin.stride(0), D=D, ROPE=cos_sin.shape[-1],
                                   num_warps=4)
    return y
