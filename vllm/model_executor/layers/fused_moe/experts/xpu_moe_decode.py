# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT4 MoE for decode-sized batches on XPU (W4A16, symmetric, per-group scales).

The grouped GEMM behind `XpuFusedMoe` costs ~90 us per layer at 1 token where
the expert bytes alone would take ~20 us. For a few tokens this runs the MoE
as two GEMV-style Triton kernels over the weights in the layout `XpuFusedMoe`
already rewrote them to (no second copy):

    w13 int8 [E, 2I, H/2]  two's-complement int4, two per byte (k even = low)
    s13      [E, 2I, H/G]
    w2  int8 [E, H, I/2],  s2 [E, H, I/G]

1. gate/up: program per (token, top-k slot, I tile): silu(gate) * up -> act.
2. down: program per (token, H tile): sum over the token's local experts of
   router weight * (act @ w2^T). Deterministic, no atomics.

`VLLM_XPU_MOE_DECODE=1` enables it for batches of at most
`VLLM_XPU_MOE_DECODE_MAX_TOKENS` tokens (default 10). B70, Flash Next shapes
(171 local experts, H 2560, I 640, group 128), graph replay: 1 token 42 vs 143 us,
5 tokens 114 vs 160 us, 10 tokens 193 vs 204 us; slower from 16 tokens.
"""

import os

import torch

from vllm.triton_utils import tl, triton

ENABLED = os.environ.get("VLLM_XPU_MOE_DECODE", "0") == "1"
MAX_TOKENS = int(os.environ.get("VLLM_XPU_MOE_DECODE_MAX_TOKENS", "10"))
LOW_FIRST = os.environ.get("VLLM_XPU_MOE_DECODE_LOW_FIRST", "1") == "1"
GATE_FIRST = True


@triton.jit
def _nibbles(b, LOW_FIRST: tl.constexpr):
    bi = b.to(tl.int32)
    lo = ((bi & 15) ^ 8) - 8
    hi = (((bi >> 4) & 15) ^ 8) - 8
    if LOW_FIRST:
        return lo.to(tl.float32), hi.to(tl.float32)
    return hi.to(tl.float32), lo.to(tl.float32)


@triton.jit
def _row_dot(w_ptr, s_ptr, rows, x_even, x_odd, k0, K: tl.constexpr, G: tl.constexpr,
             BK: tl.constexpr, LOW_FIRST: tl.constexpr):
    """sum_k W[rows, k0:k0+BK] * x for int4 rows, per-group scaled. -> [BN] fp32."""
    BN: tl.constexpr = rows.shape[0]
    half = tl.arange(0, BK // 2)
    b = tl.load(w_ptr + rows[:, None] * (K // 2) + k0 // 2 + half[None, :])
    w_even, w_odd = _nibbles(b, LOW_FIRST)
    prod = w_even * x_even[None, :] + w_odd * x_odd[None, :]
    per_group = tl.sum(tl.reshape(prod, (BN, BK // G, G // 2)), axis=2)
    sc = tl.load(s_ptr + rows[:, None] * (K // G) + k0 // G + tl.arange(0, BK // G)[None, :])
    return tl.sum(per_group * sc.to(tl.float32), axis=1)


@triton.jit
def _local_id(ids_ptr, map_ptr, t, k, TOPK: tl.constexpr, HAS_MAP: tl.constexpr,
              local_base, num_local):
    gid = tl.load(ids_ptr + t * TOPK + k)
    if HAS_MAP:
        lid = tl.load(map_ptr + gid)
    else:
        lid = gid - local_base
        lid = tl.where((lid >= 0) & (lid < num_local), lid, -1)
    return lid


@triton.jit(do_not_specialize=["local_base", "num_local"])
def _moe_up_kernel(
    x_ptr, ids_ptr, map_ptr, w13_ptr, s13_ptr, act_ptr,
    local_base, num_local,
    H: tl.constexpr, I: tl.constexpr, G: tl.constexpr, TOPK: tl.constexpr,
    HAS_MAP: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    LOW_FIRST: tl.constexpr, GATE_FIRST: tl.constexpr,
):
    pair = tl.program_id(0)
    t = pair // TOPK
    k = pair % TOPK
    lid = _local_id(ids_ptr, map_ptr, t, k, TOPK, HAS_MAP, local_base, num_local)
    if lid >= 0:
        rows = tl.program_id(1) * BN + tl.arange(0, BN)
        e = lid.to(tl.int64)
        w_e = w13_ptr + e * (2 * I) * (H // 2)
        s_e = s13_ptr + e * (2 * I) * (H // G)
        if GATE_FIRST:
            g_rows = rows
            u_rows = rows + I
        else:
            g_rows = rows + I
            u_rows = rows
        acc_g = tl.zeros((BN,), dtype=tl.float32)
        acc_u = tl.zeros((BN,), dtype=tl.float32)
        half = tl.arange(0, BK // 2)
        for k0 in range(0, H, BK):
            x_even = tl.load(x_ptr + t * H + k0 + 2 * half).to(tl.float32)
            x_odd = tl.load(x_ptr + t * H + k0 + 2 * half + 1).to(tl.float32)
            acc_g += _row_dot(w_e, s_e, g_rows, x_even, x_odd, k0, H, G, BK, LOW_FIRST)
            acc_u += _row_dot(w_e, s_e, u_rows, x_even, x_odd, k0, H, G, BK, LOW_FIRST)
        # Round like the grouped GEMM's bf16 outputs before the activation.
        acc_g = acc_g.to(tl.bfloat16).to(tl.float32)
        acc_u = acc_u.to(tl.bfloat16).to(tl.float32)
        act = acc_g / (1.0 + tl.exp(-acc_g)) * acc_u
        tl.store(act_ptr + pair * I + rows, act.to(tl.bfloat16))


@triton.jit(do_not_specialize=["local_base", "num_local"])
def _moe_down_kernel(
    ids_ptr, map_ptr, wts_ptr, act_ptr, w2_ptr, s2_ptr, out_ptr,
    local_base, num_local,
    H: tl.constexpr, I: tl.constexpr, G: tl.constexpr, TOPK: tl.constexpr,
    HAS_MAP: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    LOW_FIRST: tl.constexpr,
):
    t = tl.program_id(0)
    rows = tl.program_id(1) * BN + tl.arange(0, BN)
    half = tl.arange(0, BK // 2)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k in range(TOPK):
        lid = _local_id(ids_ptr, map_ptr, t, k, TOPK, HAS_MAP, local_base, num_local)
        if lid >= 0:
            e = lid.to(tl.int64)
            w_e = w2_ptr + e * H * (I // 2)
            s_e = s2_ptr + e * H * (I // G)
            a_base = act_ptr + (t * TOPK + k) * I
            part = tl.zeros((BN,), dtype=tl.float32)
            for k0 in range(0, I, BK):
                a_even = tl.load(a_base + k0 + 2 * half).to(tl.float32)
                a_odd = tl.load(a_base + k0 + 2 * half + 1).to(tl.float32)
                part += _row_dot(w_e, s_e, rows, a_even, a_odd, k0, I, G, BK, LOW_FIRST)
            wt = tl.load(wts_ptr + t * TOPK + k).to(tl.float32)
            acc += wt * part.to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + t * H + rows, acc.to(tl.bfloat16))


def _profile(num_tokens: int) -> tuple[int, int, int, int]:
    """(up rows, up K chunk, down rows, down K chunk), from a sweep on a B70."""
    if num_tokens == 1:
        return 8, 256, 32, 128
    if num_tokens <= 4:
        return 32, 256, 16, 128
    if num_tokens <= 6:
        return 32, 512, 32, 128
    return 32, 256, 32, 128


def applies(num_tokens: int) -> bool:
    return ENABLED and 0 < num_tokens <= MAX_TOKENS


def moe_decode(
    output: torch.Tensor,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13: torch.Tensor,
    s13: torch.Tensor,
    w2: torch.Tensor,
    s2: torch.Tensor,
    expert_map: torch.Tensor | None,
    local_base: int,
    profile: tuple[int, int, int, int] | None = None,
) -> None:
    T, H = x.shape
    topk = topk_ids.shape[1]
    E, two_i, _ = w13.shape
    inter = two_i // 2
    group = H // s13.shape[2]
    assert I_ok(inter, group) and w2.shape == (E, H, inter // 2)
    assert x.is_contiguous() and output.is_contiguous()
    ids = topk_ids.to(torch.int32).contiguous()
    wts = topk_weights.contiguous()
    act = torch.empty((T * topk, inter), dtype=torch.bfloat16, device=x.device)
    bn1, bk1, bn2, bk2 = profile or _profile(T)
    bk1 = min(bk1, H)
    bk2 = min(bk2, inter)
    has_map = expert_map is not None
    emap = expert_map if has_map else ids
    _moe_up_kernel[(T * topk, triton.cdiv(inter, bn1))](
        x, ids, emap, w13, s13, act, local_base, E,
        H=H, I=inter, G=group, TOPK=topk, HAS_MAP=has_map, BN=bn1, BK=bk1,
        LOW_FIRST=LOW_FIRST, GATE_FIRST=GATE_FIRST, num_warps=4,
    )
    _moe_down_kernel[(T, triton.cdiv(H, bn2))](
        ids, emap, wts, act, w2, s2, output, local_base, E,
        H=H, I=inter, G=group, TOPK=topk, HAS_MAP=has_map, BN=bn2, BK=bk2,
        LOW_FIRST=LOW_FIRST, num_warps=4,
    )


def I_ok(inter: int, group: int) -> bool:  # noqa: N802
    return inter % group == 0 and inter % 2 == 0
