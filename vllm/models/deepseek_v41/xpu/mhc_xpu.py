# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4.1 delayed mHC pre on XPU in two Triton launches.

`MHCPreDelayedOp` has no XPU path and falls back to the torch reference
(about a hundred small ops per call: 20 Sinkhorn iterations on 4x4 matrices).
Here:
1. `_mix_kernel`: split-K GEMV of the flattened residual (or the entry `x`)
   with fn [24, D], plus its sum of squares, into per-split partials;
2. `_gate_kernel`: one program per (token, hidden chunk) adds the partials,
   applies the RMS scale, sigmoid gates and Sinkhorn (programs with chunk 0
   write post / comb / pre), and collapses the residual with the carried
   pre-mix (or takes stream 0 when there is none).
Same math as `mhc_pre_delayed_torch`."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from vllm.model_executor.layers.mhc import MHCPostOp


@triton.jit
def _mix_kernel(src_ptr, fn_ptr, part_ptr, D, n_split,
                NMIX: tl.constexpr, NMIX_P: tl.constexpr, BLOCK_D: tl.constexpr):
    t = tl.program_id(0)
    s = tl.program_id(1)
    offs = s * BLOCK_D + tl.arange(0, BLOCK_D)
    m = offs < D
    x = tl.load(src_ptr + t * D + offs, mask=m, other=0.0).to(tl.float32)
    rows = tl.arange(0, NMIX_P)
    rm = rows < NMIX
    f = tl.load(fn_ptr + rows[:, None] * D + offs[None, :], mask=rm[:, None] & m[None, :], other=0.0)
    acc = tl.sum(f * x[None, :], axis=1)  # [NMIX_P]
    sq = tl.sum(x * x, axis=0)
    acc = tl.where(rows == NMIX, sq, acc)  # slot NMIX holds the sum of squares
    tl.store(part_ptr + (t * n_split + s) * NMIX_P + rows, acc)


@triton.jit
def _gate_kernel(part_ptr, scale_ptr, base_ptr, res_ptr, prev_ptr,
                 post_ptr, comb_ptr, pre_ptr, out_ptr,
                 n_split, D, H, rms_eps, pre_eps, sk_eps, post_mult,
                 SK_REPEAT: tl.constexpr, HC: tl.constexpr, NMIX: tl.constexpr,
                 NMIX_P: tl.constexpr, SPLIT_P: tl.constexpr, HAS_PREV: tl.constexpr,
                 BLOCK_H: tl.constexpr):
    t = tl.program_id(0)
    c = tl.program_id(1)
    if c == 0:
        sp = tl.arange(0, SPLIT_P)
        rows = tl.arange(0, NMIX_P)
        part = tl.load(part_ptr + (t * n_split + sp[:, None]) * NMIX_P + rows[None, :],
                       mask=(sp < n_split)[:, None], other=0.0)
        mix = tl.sum(part, axis=0)  # [NMIX_P]
        sq = tl.sum(tl.where(rows == NMIX, mix, 0.0), axis=0)
        r = tl.rsqrt(sq / D + rms_eps)
        mix = mix * r
        s0 = tl.load(scale_ptr)
        s1 = tl.load(scale_ptr + 1)
        s2 = tl.load(scale_ptr + 2)
        base = tl.load(base_ptr + rows, mask=rows < NMIX, other=0.0)
        # pre / post: rows [0, HC) and [HC, 2HC)
        h = tl.arange(0, HC)
        mix_pre = tl.sum(tl.where(rows[None, :] == h[:, None], mix[None, :], 0.0), axis=1)
        mix_post = tl.sum(tl.where(rows[None, :] == (h[:, None] + HC), mix[None, :], 0.0), axis=1)
        b_pre = tl.load(base_ptr + h)
        b_post = tl.load(base_ptr + HC + h)
        pre = tl.sigmoid(mix_pre * s0 + b_pre) + pre_eps
        post = tl.sigmoid(mix_post * s1 + b_post) * post_mult
        tl.store(pre_ptr + t * HC + h, pre)
        tl.store(post_ptr + t * HC + h, post)
        # comb: rows [2HC, 2HC + HC*HC) as [HC, HC]
        i = tl.arange(0, HC)[:, None]
        j = tl.arange(0, HC)[None, :]
        idx = 2 * HC + i * HC + j  # [HC, HC]
        cm = tl.zeros([HC, HC], dtype=tl.float32)
        for q in range(HC * HC):
            v = tl.sum(tl.where(rows == 2 * HC + q, mix, 0.0), axis=0)
            cm = tl.where(idx == 2 * HC + q, v, cm)
        cb = tl.load(base_ptr + idx)
        cm = cm * s2 + cb
        cm = cm - tl.max(cm, axis=1)[:, None]
        cm = tl.exp(cm)
        cm = cm / tl.sum(cm, axis=1)[:, None] + sk_eps
        cm = cm / (tl.sum(cm, axis=0)[None, :] + sk_eps)
        for _ in range(SK_REPEAT - 1):
            cm = cm / (tl.sum(cm, axis=1)[:, None] + sk_eps)
            cm = cm / (tl.sum(cm, axis=0)[None, :] + sk_eps)
        tl.store(comb_ptr + t * HC * HC + i * HC + j, cm)
    # collapse with the carried pre-mix
    offs = c * BLOCK_H + tl.arange(0, BLOCK_H)
    m = offs < H
    hh = tl.arange(0, HC)
    res = tl.load(res_ptr + t * HC * H + hh[:, None] * H + offs[None, :],
                  mask=m[None, :], other=0.0).to(tl.float32)
    if HAS_PREV:
        pm = tl.load(prev_ptr + t * HC + hh)
        out = tl.sum(res * pm[:, None], axis=0)
    else:
        out = tl.sum(tl.where(hh[:, None] == 0, res, 0.0), axis=0)
    tl.store(out_ptr + t * H + offs, out.to(out_ptr.dtype.element_ty), mask=m)


def mhc_pre_delayed_triton(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                           hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                           pre_mix=None, x=None, block_d: int = 1024, block_h: int = 1024):
    """Returns (post [T, HC, 1], comb [T, HC, HC], layer_input [T, H], pre [T, HC])."""
    T, HC, H = residual.shape
    src = (residual.reshape(T, HC * H) if x is None else x).contiguous()
    D = src.shape[1]
    NMIX = fn.shape[0]
    assert fn.shape[1] == D and NMIX == 2 * HC + HC * HC, (fn.shape, D)
    NMIX_P = triton.next_power_of_2(NMIX + 1)
    n_split = triton.cdiv(D, block_d)
    dev = residual.device
    part = torch.empty(T, n_split, NMIX_P, dtype=torch.float32, device=dev)
    post = torch.empty(T, HC, 1, dtype=torch.float32, device=dev)
    comb = torch.empty(T, HC, HC, dtype=torch.float32, device=dev)
    pre = torch.empty(T, HC, dtype=torch.float32, device=dev)
    out = torch.empty(T, H, dtype=residual.dtype, device=dev)
    if T == 0:
        return post, comb, out, pre
    fn = fn.contiguous()
    res = residual.contiguous()
    _mix_kernel[(T, n_split)](src, fn, part, D, n_split,
                              NMIX=NMIX, NMIX_P=NMIX_P, BLOCK_D=block_d)
    prev = pre_mix.float().contiguous() if pre_mix is not None else pre
    _gate_kernel[(T, triton.cdiv(H, block_h))](
        part, hc_scale, hc_base, res, prev, post, comb, pre, out,
        n_split, D, H, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
        SK_REPEAT=sinkhorn_repeat, HC=HC, NMIX=NMIX, NMIX_P=NMIX_P,
        SPLIT_P=triton.next_power_of_2(n_split), HAS_PREV=pre_mix is not None,
        BLOCK_H=block_h,
    )
    return post, comb, out, pre


_MAX_TOKENS = int(__import__("os").environ.get("VLLM_DS41_MHC_TRITON_MAX_TOKENS", "64"))


class XpuMHCPreDelayed(torch.nn.Module):
    """Drop-in for `MHCPreDelayedOp` (same call signature and outputs): the
    Triton path up to `_MAX_TOKENS` tokens (decode), the original op above
    (prefill, where the torch reference is faster)."""

    def __init__(self) -> None:
        super().__init__()
        from vllm.model_executor.layers.mhc import MHCPreDelayedOp

        self._post = MHCPostOp()
        self._ref = MHCPreDelayedOp()

    def forward(self, residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                hc_post_mult_value, sinkhorn_repeat, pre_mix=None, x=None, norm_weight=None,
                norm_eps=1e-6, sublayer_out=None, post_layer_mix=None, comb_res_mix=None,
                fn_packed=None):
        if norm_weight is not None or residual.shape[0] > _MAX_TOKENS:
            return self._ref(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                             hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                             pre_mix=pre_mix, x=x, norm_weight=norm_weight, norm_eps=norm_eps,
                             sublayer_out=sublayer_out, post_layer_mix=post_layer_mix,
                             comb_res_mix=comb_res_mix, fn_packed=fn_packed)
        if sublayer_out is not None:
            residual = self._post(sublayer_out, residual, post_layer_mix, comb_res_mix)
        post, comb, layer_input, pre = mhc_pre_delayed_triton(
            residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
            hc_post_mult_value, sinkhorn_repeat, pre_mix=pre_mix, x=x,
        )
        return residual, post, comb, layer_input, pre
