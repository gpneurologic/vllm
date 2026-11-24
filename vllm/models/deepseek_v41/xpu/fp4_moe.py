# SPDX-License-Identifier: Apache-2.0
"""Decode-time routed MoE for streamed FP4 experts (DeepSeek V4.1, XPU).

The experts live in one device pool of fixed-size slots (`stream.StreamCache`),
each slot holding one expert in the repacked layout (see tools/ds41_repack.py):
  w13 weight [2I, H/2] | w2 weight [H, I/2] | w13 scale [2I, H/32] | w2 scale [H, I/32]
FP4 is e2m1 (low nibble = even k), scales ue8m0 per 32 values.

One program computes BLOCK_N output rows of one (token, top-k slot) pair over a
slice of K; the expert's pool slot comes from a per-layer slot table
(slot_of[expert], -1 = not resident on this rank: the pair adds zero). No host
sync. Numerics follow the reference `offload.run_expert`: activations rounded
through fp8 e4m3 with ue8m0 scales per 32 before each GEMV, routing weight
applied before the w2 input is rounded, fp32 accumulation.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _e2m1(c):
    """uint8 codes 0..15 -> float32 e2m1 values."""
    m = c & 7
    e = (m >> 1).to(tl.float32)
    mant = (m & 1).to(tl.float32)
    mag = tl.where(m < 2, 0.5 * mant, (1.0 + 0.5 * mant) * tl.exp2(e - 1.0))
    return tl.where((c & 8) != 0, -mag, mag)


@triton.jit
def _fp8_round(xe, xo):
    """Reference activation rounding, in registers: per group of 32 values
    (row g of xe / xo, the even / odd halves) a ue8m0 scale from amax / 448,
    values rounded through fp8 e4m3. Bit-exact with offload.act_fp8_roundtrip."""
    amax = tl.maximum(tl.maximum(tl.max(tl.abs(xe), axis=1), tl.max(tl.abs(xo), axis=1)), 1e-4)
    bits = (amax / 448.0).to(tl.int32, bitcast=True)
    e = ((bits >> 23) & 255) - 127 + tl.where((bits & 0x7FFFFF) != 0, 1, 0)
    sc = tl.exp2(e.to(tl.float32))[:, None]
    qe = tl.minimum(tl.maximum(xe / sc, -448.0), 448.0).to(tl.float8e4nv).to(tl.float32) * sc
    qo = tl.minimum(tl.maximum(xo / sc, -448.0), 448.0).to(tl.float8e4nv).to(tl.float32) * sc
    return qe, qo


@triton.jit
def _fp4_gemv_kernel(x_ptr, pool_ptr, slot_ptr, idx_ptr, y_ptr,
                     N, K, slot_stride, w_off, s_off, n_pairs,
                     XDIV: tl.constexpr, BLOCK_N: tl.constexpr, KB_PER: tl.constexpr,
                     QX: tl.constexpr = False):
    """y[pk, p, n] = sum_k W[slot(p)][n, k] * x[p // XDIV, k] over this K split.
    One K block = 256 values = 128 weight bytes = 8 scales."""
    pid_p = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    e = tl.load(idx_ptr + pid_p).to(tl.int32)
    slot = tl.load(slot_ptr + e)
    rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    rows = tl.where(rmask, rows, 0)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    if slot >= 0:
        base = pool_ptr + slot.to(tl.int64) * slot_stride
        rb = K // 2
        sb = K // 32
        wrow = base + w_off + rows.to(tl.int64)[:, None, None] * rb   # [BN,1,1]
        srow = base + s_off + rows.to(tl.int64)[:, None] * sb          # [BN,1]
        g = tl.arange(0, 8)
        b = tl.arange(0, 16)
        boff = g[:, None] * 16 + b[None, :]                            # [8,16]
        xrow = x_ptr + (pid_p // XDIV) * K
        n_kb = K // 256
        for i in range(KB_PER):
            kb = pid_k * KB_PER + i
            if kb < n_kb:
                wb = tl.load(wrow + kb * 128 + boff[None, :, :])        # [BN,8,16] u8
                sc = tl.load(srow + kb * 8 + g[None, :]).to(tl.float32)  # [BN,8]
                xe = tl.load(xrow + kb * 256 + 2 * boff).to(tl.float32)  # [8,16]
                xo = tl.load(xrow + kb * 256 + 2 * boff + 1).to(tl.float32)
                if QX:
                    xe, xo = _fp8_round(xe, xo)
                lo = _e2m1(wb & 15)
                hi = _e2m1(wb >> 4)
                part = tl.sum(lo * xe[None, :, :] + hi * xo[None, :, :], axis=2)  # [BN,8]
                acc += tl.sum(part * tl.exp2(sc - 127.0), axis=1)
    tl.store(y_ptr + (pid_k * n_pairs + pid_p) * N + rows, acc, mask=rmask)


@triton.jit
def _swiglu_w_kernel(h_ptr, w_ptr, a_ptr, I, limit, HAS_LIMIT: tl.constexpr,
                     BLOCK: tl.constexpr):
    """a[p] = silu(min(g, L)) * clamp(u, -L, L) * w[p]  (h[p] = [g | u])."""
    p = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = c < I
    g = tl.load(h_ptr + p * 2 * I + c, mask=m, other=0.0)
    u = tl.load(h_ptr + p * 2 * I + I + c, mask=m, other=0.0)
    if HAS_LIMIT:
        g = tl.minimum(g, limit)
        u = tl.minimum(tl.maximum(u, -limit), limit)
    wv = tl.load(w_ptr + p)
    tl.store(a_ptr + p * I + c, (g / (1.0 + tl.exp(-g)) * u * wv).to(a_ptr.dtype.element_ty), mask=m)


# (BLOCK_N, KB_PER, num_warps) per K: w13 K = 5120 (20 blocks), w2 K = 2304 (9).
_CFG = {5120: (8, 4, 4), 2304: (8, 3, 4)}


def fp4_gemv(x, pool, slot_row, idx_flat, N, K, w_off, s_off, xdiv, cfg=None, qx=False):
    P = idx_flat.shape[0]
    bn, kp, nw = cfg or _CFG.get(K, (8, 4, 4))
    split = triton.cdiv(K // 256, kp)
    y = torch.empty(split, P, N, dtype=torch.float32, device=x.device)
    _fp4_gemv_kernel[(P, triton.cdiv(N, bn), split)](
        x, pool, slot_row, idx_flat, y, N, K, pool.stride(0), w_off, s_off, P,
        XDIV=xdiv, BLOCK_N=bn, KB_PER=kp, QX=qx, num_warps=nw)
    return y.sum(0) if split > 1 else y[0]


def act_fp8_roundtrip(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    from .offload import act_fp8_roundtrip as f

    return f(x, block)


_FUSED_Q = __import__("os").environ.get("VLLM_DS41_FP4_FUSED_Q", "1") == "1"


def moe_decode_fp4(x, idx, w, out, pool, slot_row, lay, limit):
    """out (fp32 [T, H]) += this rank's routed experts for all (token, slot)
    pairs. x [T, H] bf16, idx [T, k] global expert ids, w [T, k] fp32 routing
    weights, slot_row [n_experts] int32 device (slot or -1)."""
    T, k = idx.shape
    H, I = lay["hidden"], lay["inter"]
    o = lay["offsets"]
    idx_flat = idx.reshape(-1).to(torch.int32)
    if _FUSED_Q:
        # Activation rounding inside the GEMVs (bit-exact with the unfused path).
        h = fp4_gemv(x.contiguous(), pool, slot_row, idx_flat, 2 * I, H, o["w13_weight"],
                     o["w13_scale"], k, qx=True)
        P = T * k
        a = torch.empty(P, I, dtype=x.dtype, device=x.device)
        _swiglu_w_kernel[(P, triton.cdiv(I, 1024))](
            h, w.reshape(-1).float().contiguous(), a, I, float(limit or 0.0),
            HAS_LIMIT=bool(limit), BLOCK=1024)
        y = fp4_gemv(a, pool, slot_row, idx_flat, H, I, o["w2_weight"], o["w2_scale"], 1, qx=True)
        out += y.view(T, k, H).sum(1)
        return
    xq = act_fp8_roundtrip(x).contiguous()
    h = fp4_gemv(xq, pool, slot_row, idx_flat, 2 * I, H, o["w13_weight"], o["w13_scale"], k)
    P = T * k
    a = torch.empty(P, I, dtype=torch.float32, device=x.device)
    _swiglu_w_kernel[(P, triton.cdiv(I, 1024))](
        h, w.reshape(-1).float().contiguous(), a, I, float(limit or 0.0),
        HAS_LIMIT=bool(limit), BLOCK=1024)
    aq = act_fp8_roundtrip(a.to(x.dtype)).contiguous()
    y = fp4_gemv(aq, pool, slot_row, idx_flat, H, I, o["w2_weight"], o["w2_scale"], 1)
    out += y.view(T, k, H).sum(1)


def expert_parts(pool, slot: int, lay):
    """Views (w1, s1, w3, s3, w2, s2) of one pool slot, shaped like the
    checkpoint tensors (for the per-expert prefill path, offload.run_expert)."""
    H, I = lay["hidden"], lay["inter"]
    o = lay["offsets"]
    b = pool[slot]
    w13 = b[o["w13_weight"]: o["w13_weight"] + 2 * I * H // 2].view(2 * I, H // 2)
    w2 = b[o["w2_weight"]: o["w2_weight"] + H * I // 2].view(H, I // 2)
    s13 = b[o["w13_scale"]: o["w13_scale"] + 2 * I * H // 32].view(2 * I, H // 32)
    s2 = b[o["w2_scale"]: o["w2_scale"] + H * I // 32].view(H, I // 32)
    return w13[:I], s13[:I], w13[I:], s13[I:], w2, s2
