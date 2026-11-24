# SPDX-License-Identifier: Apache-2.0
"""Routed MoE for streamed GGUF experts (all 384 experts at ~2 bits, e.g.
IQ2_XXS gate/up + Q2_K down) held in the StreamCache pool.

Decode (few tokens): split-K GEMVs over pool slots, as gguf_moe_decode's v2
kernel but indexed by cache slot (slot -1: expert not resident on this rank,
output stays zero). Prefill: each needed expert is dequantized to bf16 and
multiplied with torch matmul.

Q2_K (84-byte superblock of 256 values: scales[16] (low nibble scale, high
nibble min), qs[64] 2-bit, d f16, dmin f16) follows ggml's
dequantize_row_q2_K; the IQ formats reuse gguf_moe_decode._deq_groups."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .gguf_moe_decode import _BLOCK_BYTES, _deq_groups, _f16, _iq_tables
from .fp4_moe import _swiglu_w_kernel

QT_Q2_K = 10
BLOCK_BYTES = dict(_BLOCK_BYTES)
BLOCK_BYTES[QT_Q2_K] = 84


@triton.jit
def _deq_any(bp, grid64_ptr, sign_ptr, QTYPE: tl.constexpr):
    """bp: [BN, 1] superblock pointers -> float32 [BN, 32, 8] (value g*8+j)."""
    if QTYPE == 10:  # Q2_K
        g = tl.arange(0, 32)[:, None]
        j = tl.arange(0, 8)[None, :]
        v = (g * 8 + j)[None, :, :]  # [1, 32, 8]
        h = v // 128
        r = v % 128
        sh = 2 * (r // 32)
        s = r % 32
        bp3 = bp[:, :, None]  # [BN, 1, 1]
        q = tl.load(bp3 + 16 + 32 * h + s).to(tl.int32)
        sc = tl.load(bp3 + 8 * h + 2 * (r // 32) + s // 16).to(tl.int32)
        d = _f16(bp3 + 80)
        dm = _f16(bp3 + 82)
        return d * (sc & 0xF).to(tl.float32) * ((q >> sh) & 3).to(tl.float32) \
            - dm * (sc >> 4).to(tl.float32)
    else:
        return _deq_groups(bp, grid64_ptr, sign_ptr, QTYPE)


@triton.jit
def _gemv_slot_kernel(x_ptr, w_ptr, sidx_ptr, y_ptr, grid64_ptr, sign_ptr,
                      N, K, row_bytes, slot_stride, n_pairs,
                      XDIV: tl.constexpr, QTYPE: tl.constexpr, BB: tl.constexpr,
                      BLOCK_N: tl.constexpr, KB_PER: tl.constexpr):
    pid_p = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    s = tl.load(sidx_ptr + pid_p)
    rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    rows = tl.where(rmask, rows, 0)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    if s >= 0:
        n_kb = K // 256
        row_base = w_ptr + s.to(tl.int64) * slot_stride + rows.to(tl.int64) * row_bytes
        xo = tl.arange(0, 32)[:, None] * 8 + tl.arange(0, 8)[None, :]
        xrow = (pid_p // XDIV) * K
        for i in range(KB_PER):
            kb = pid_k * KB_PER + i
            if kb < n_kb:
                bp = (row_base + kb * BB)[:, None]
                wv = _deq_any(bp, grid64_ptr, sign_ptr, QTYPE)
                xv = tl.load(x_ptr + xrow + kb * 256 + xo).to(tl.float32)
                acc += tl.sum(tl.sum(wv * xv[None, :, :], axis=2), axis=1)
    tl.store(y_ptr + (pid_k * n_pairs + pid_p) * N + rows, acc, mask=rmask)


@triton.jit
def _w2_fused_kernel(h_ptr, wts_ptr, w_ptr, sidx_ptr, out_ptr, grid64_ptr, sign_ptr,
                     H, I, row_bytes, slot_stride, limit,
                     K_TOP: tl.constexpr, HAS_LIMIT: tl.constexpr, QTYPE: tl.constexpr,
                     BB: tl.constexpr, BLOCK_N: tl.constexpr):
    """out[t, rows] += sum_j W2[slot(t, j)] @ (silu(min(g, L)) * clamp(u) * w[t, j]),
    h[p] = [g | u] fp32 [P, 2I], p = t * K_TOP + j. One launch replaces swiglu,
    the split-K GEMV, both sums and the add."""
    t = tl.program_id(0)
    rows = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < H
    rows = tl.where(rmask, rows, 0)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    xo = tl.arange(0, 32)[:, None] * 8 + tl.arange(0, 8)[None, :]
    n_kb = I // 256
    for j in range(K_TOP):
        p = t * K_TOP + j
        s = tl.load(sidx_ptr + p)
        if s >= 0:
            wt = tl.load(wts_ptr + p).to(tl.float32)
            row_base = w_ptr + s.to(tl.int64) * slot_stride + rows.to(tl.int64) * row_bytes
            hb = h_ptr + p.to(tl.int64) * 2 * I
            for kb in range(n_kb):
                g = tl.load(hb + kb * 256 + xo)
                u = tl.load(hb + I + kb * 256 + xo)
                if HAS_LIMIT:
                    g = tl.minimum(g, limit)
                    u = tl.minimum(tl.maximum(u, -limit), limit)
                a = g / (1.0 + tl.exp(-g)) * u * wt
                wv = _deq_any((row_base + kb * BB)[:, None], grid64_ptr, sign_ptr, QTYPE)
                acc += tl.sum(tl.sum(wv * a[None, :, :], axis=2), axis=1)
    optr = out_ptr + t.to(tl.int64) * H + rows
    tl.store(optr, tl.load(optr, mask=rmask, other=0.0) + acc, mask=rmask)


@triton.jit
def _dequant_iq_kernel(w_ptr, o_ptr, grid64_ptr, sign_ptr, N, K, row_bytes,
                       QTYPE: tl.constexpr, BB: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_n = tl.program_id(0)
    kb = tl.program_id(1)
    rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    rows = tl.where(rmask, rows, 0)
    bp = (w_ptr + rows.to(tl.int64) * row_bytes + kb * BB)[:, None]
    wv = _deq_any(bp, grid64_ptr, sign_ptr, QTYPE)
    xo = (tl.arange(0, 32)[:, None] * 8 + tl.arange(0, 8)[None, :])[None, :, :]
    optr = o_ptr + rows.to(tl.int64)[:, None, None] * K + kb * 256 + xo
    tl.store(optr, wv.to(o_ptr.dtype.element_ty), mask=rmask[:, None, None])


# (BLOCK_N, KB_PER, num_warps) per (type, K); IQ2_XXS from gguf_moe_decode's sweep
_CFG = {(16, 5120): (4, 4, 2), (16, 2304): (8, 1, 2),
        (10, 5120): (4, 4, 2), (10, 2304): (4, 3, 2)}


_TAB: dict = {}


def _tables(dev, qtype):
    """(grid as int64 [n] (8 grid bytes per entry), sign table uint8) on dev.
    Built from llama.cpp's gguf-py (VLLM_DS41_GGUF_PY), which the serving
    image does not ship; falls back to vllm_gguf_plugin."""
    key = (str(dev), qtype)
    if key in _TAB:
        return _TAB[key]
    if qtype == QT_Q2_K:
        z = torch.zeros(1, dtype=torch.int64, device=dev)
        _TAB[key] = (z, z.view(torch.uint8))
        return _TAB[key]
    try:
        _, g64, sg = _iq_tables(dev, qtype)
    except ImportError:
        import os
        import sys

        import numpy as np

        sys.path.insert(0, os.environ.get("VLLM_DS41_GGUF_PY", "/g/llama.cpp/gguf-py"))
        from gguf.quants import IQ2_XXS

        assert qtype == 16, qtype
        IQ2_XXS.init_grid()
        g8 = np.ascontiguousarray(IQ2_XXS.grid[0, 0].astype(np.uint8)).reshape(-1)
        assert g8.size % 8 == 0
        g64 = torch.from_numpy(g8.copy()).to(dev).view(torch.int64)
        sg = torch.from_numpy(np.frombuffer(IQ2_XXS.ksigns, dtype=np.uint8).copy()).to(dev)
    _TAB[key] = (g64, sg)
    return _TAB[key]


def gemv_slots(x, pool, sidx, N, K, off, row_bytes, qtype, xdiv, cfg=None):
    """y [P, N] fp32: row p = W[slot sidx[p]] @ x[p // xdiv] (zeros for slot -1).
    W of a slot: bytes pool[slot, off : off + N * row_bytes], N rows."""
    P = sidx.shape[0]
    bn, kp, nw = cfg or _CFG.get((qtype, K), (4, 4, 2))
    split = triton.cdiv(K // 256, kp)
    y = torch.empty(split, P, N, dtype=torch.float32, device=x.device)
    g64, sg = _tables(x.device, qtype)
    _gemv_slot_kernel[(P, triton.cdiv(N, bn), split)](
        x, pool[:, off:], sidx, y, g64, sg, N, K, row_bytes, pool.stride(0), P,
        XDIV=xdiv, QTYPE=qtype, BB=BLOCK_BYTES[qtype], BLOCK_N=bn, KB_PER=kp, num_warps=nw)
    return y.sum(0) if split > 1 else y[0]


def dequant(raw: torch.Tensor, N: int, K: int, qtype: int, dtype=torch.bfloat16) -> torch.Tensor:
    """raw: uint8 bytes of N GGUF rows over K -> [N, K] dtype."""
    rb = K // 256 * BLOCK_BYTES[qtype]
    assert raw.numel() >= N * rb, (raw.numel(), N, rb)
    o = torch.empty(N, K, dtype=dtype, device=raw.device)
    bn = 16
    g64, sg = _tables(raw.device, qtype)
    _dequant_iq_kernel[(triton.cdiv(N, bn), K // 256)](
        raw, o, g64, sg, N, K, rb, QTYPE=qtype, BB=BLOCK_BYTES[qtype], BLOCK_N=bn, num_warps=4)
    return o


_SYCL = __import__("os").environ.get("VLLM_DS41_MOE_SYCL", "1") == "1"
_ROWS = int(__import__("os").environ.get("VLLM_DS41_MOE_ROWS", "4"))
_sycl_cache: dict = {}
# rows per program of the fused w2 kernel (0: old swiglu + split-K path)
_FUSED_W2 = int(__import__("os").environ.get("VLLM_DS41_FUSED_W2", "8"))
_FUSED_W2_WARPS = int(__import__("os").environ.get("VLLM_DS41_FUSED_W2_WARPS", "2"))


def _sycl_ok(qtype: int) -> bool:
    """SYCL kernels exist for IQ1_M / IQ2_XXS / IQ2_XS (not Q2_K)."""
    if qtype not in (16, 17, 29) or not _SYCL:
        return False
    if "ok" not in _sycl_cache:
        from . import iq_mmvq as iq

        _sycl_cache["ok"] = iq.available()
    return _sycl_cache["ok"]


def moe_decode_gguf(x, idx, w, out, pool, slot_row, lay, limit):
    """out (fp32 [T, H]) += this rank's routed experts for all (token, slot)
    pairs; same contract as fp4_moe.moe_decode_fp4."""
    T, k = idx.shape
    H, I = lay["hidden"], lay["inter"]
    o, qt, rb = lay["offsets"], lay["qtypes"], lay["row_bytes"]
    sidx = slot_row.index_select(0, idx.reshape(-1))
    P = T * k
    if _sycl_ok(qt["w13"]):
        # llama.cpp-style SYCL GEMV (q8_1 activations), as the live GGUF build;
        # slots act as experts 0..n_slots-1 of a strided view of the pool.
        from . import iq_mmvq as iq

        n = pool.shape[0]
        w13 = pool.as_strided((n, 2 * I, rb["w13"]), (pool.stride(0), rb["w13"], 1),
                              pool.storage_offset() + o["w13_weight"])
        xq = iq.quantize_q8_1(x.contiguous())
        if _ROWS and iq.has_rows():
            h = iq.moe_mmvq2(xq, H, w13, qt["w13"], sidx, 0, n, k, _ROWS)
        else:
            h = iq.moe_mmvq(xq, H, w13, qt["w13"], sidx, 0, n, k)
    else:
        h = gemv_slots(x.contiguous(), pool, sidx, 2 * I, H, o["w13_weight"], rb["w13"],
                       qt["w13"], k)
    if _FUSED_W2 and out.dtype == torch.float32 and out.is_contiguous():
        g64, sg = _tables(x.device, qt["w2"])
        _w2_fused_kernel[(T, triton.cdiv(H, _FUSED_W2))](
            h, w.reshape(-1), pool[:, o["w2_weight"]:], sidx, out, g64, sg,
            H, I, rb["w2"], pool.stride(0), float(limit or 0.0),
            K_TOP=k, HAS_LIMIT=bool(limit), QTYPE=qt["w2"], BB=BLOCK_BYTES[qt["w2"]],
            BLOCK_N=_FUSED_W2, num_warps=_FUSED_W2_WARPS)
        return
    a = torch.empty(P, I, dtype=torch.float32, device=x.device)
    _swiglu_w_kernel[(P, triton.cdiv(I, 1024))](
        h, w.reshape(-1).float().contiguous(), a, I, float(limit or 0.0),
        HAS_LIMIT=bool(limit), BLOCK=1024)
    y = gemv_slots(a, pool, sidx, H, I, o["w2_weight"], rb["w2"], qt["w2"], 1)
    out += y.view(T, k, H).sum(1)


def run_expert_gguf(x, pool, slot: int, lay, limit: float, weights) -> torch.Tensor:
    """One expert for n tokens (prefill). x [n, H] bf16, weights [n, 1] fp32.
    Returns fp32 [n, H]."""
    H, I = lay["hidden"], lay["inter"]
    o, qt, rb = lay["offsets"], lay["qtypes"], lay["row_bytes"]
    b = pool[slot]
    w13 = dequant(b[o["w13_weight"]: o["w13_weight"] + 2 * I * rb["w13"]], 2 * I, H, qt["w13"],
                  x.dtype)
    w2 = dequant(b[o["w2_weight"]: o["w2_weight"] + H * rb["w2"]], H, I, qt["w2"], x.dtype)
    h = (x @ w13.t()).float()
    gate, up = h[:, :I], h[:, I:]
    if limit and limit > 0:
        up = up.clamp(-limit, limit)
        gate = gate.clamp(max=limit)
    a = (torch.nn.functional.silu(gate) * up * weights).to(x.dtype)
    return (a @ w2.t()).float()
