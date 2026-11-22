# SPDX-License-Identifier: Apache-2.0
"""FP8 GEMV with 32x32 power-of-two block scales, for decode.

z[t, g, r] = sum_k x[t, g, k] * W[g*R + r, k] * S[(g*R + r) // 32, k // 32]

W is the stored FP8 e4m3 weight [G*R, K]; S its block scales (float, any
strides). `fp8_grouped_gemv` serves wo_a (G groups); `DecodeFp8Linear`
puts the G = 1 case in front of every block-FP8 linear for small batches,
replacing the per-token activation quant + oneDNN GEMM (several launches)
with one kernel."""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _gemv_kernel(x_ptr, w_ptr, s_ptr, y_ptr, R, K, G,
                 sx_t, sx_g, sw_n, ss_n, ss_k, n_tok,
                 BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, K_PER: tl.constexpr):
    t = tl.program_id(0)
    pn = tl.program_id(1)  # over G*R rows
    pk = tl.program_id(2)
    rows = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    g = (pn * BLOCK_N) // R  # BLOCK_N divides R
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for i in range(K_PER):
        k0 = (pk * K_PER + i) * BLOCK_K
        ks = k0 + tl.arange(0, BLOCK_K)
        km = ks < K
        xv = tl.load(x_ptr + t * sx_t + g * sx_g + ks, mask=km, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + rows[:, None] * sw_n + ks[None, :], mask=km[None, :], other=0.0)
        sc = tl.load(s_ptr + (rows // 32)[:, None] * ss_n + (ks // 32)[None, :] * ss_k,
                     mask=km[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(w.to(tl.float32) * sc * xv[None, :], axis=1)
    tl.store(y_ptr + (pk * n_tok + t) * (G * R) + rows, acc.to(y_ptr.dtype.element_ty))


def _launch(x3, w, s, groups, out_dtype, block_n, block_k, k_per, num_warps=4):
    T, G, K = x3.shape
    N = w.shape[0]
    R = N // G
    split = triton.cdiv(triton.cdiv(K, block_k), k_per)
    y = torch.empty(split, T, N, dtype=out_dtype if split == 1 else torch.float32,
                    device=x3.device)
    _gemv_kernel[(T, N // block_n, split)](
        x3, w, s, y, R, K, G, x3.stride(0), x3.stride(1), w.stride(0), s.stride(0), s.stride(1),
        T, BLOCK_N=block_n, BLOCK_K=block_k, K_PER=k_per, num_warps=num_warps,
    )
    if split == 1:
        return y[0]
    return y.sum(0).to(out_dtype)


def fp8_grouped_gemv(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor, groups: int,
                     block_n: int | None = None, block_k: int | None = None,
                     k_per: int | None = None, num_warps: int | None = None) -> torch.Tensor:
    """x [T, G, K] (any float dtype), w [G*R, K] fp8, s [G*R/32, K/32] float.
    Returns float32 [T, G, R].

    Default launch from a graph-replay sweep on Arc Pro B70 for wo_a at TP=3
    (G=3, R=1024, K=4096): 1 token 8 rows x whole K per program (no split-K
    reduction), 52.8 -> 27.3 us; 2+ tokens 8 rows, 4 x 256 K per program,
    63.7 -> 42.9 us."""
    T, G, K = x.shape
    if block_n is None:
        block_n, block_k, k_per, num_warps = ((8, 512, triton.cdiv(K, 512), 4) if T == 1
                                              else (8, 256, 4, 2))
    N = w.shape[0]
    assert G == groups and N % G == 0 and w.shape[1] == K and w.stride(1) == 1
    assert (N // G) % block_n == 0
    if s.dtype == torch.float8_e8m0fnu:
        s = s.view(torch.uint8).to(torch.int32).sub(127).float().exp2()
    y = _launch(x.contiguous(), w, s, G, torch.float32, block_n, block_k, k_per,
                num_warps or 4)
    return y.view(T, G, N // G)


def block_scales_f32(s: torch.Tensor) -> torch.Tensor:
    """Block scales in any stored dtype -> contiguous float32."""
    if s.dtype == torch.float8_e8m0fnu:
        s = s.view(torch.uint8).to(torch.int32).sub(127).float().exp2()
    elif s.dtype == torch.uint8:
        s = s.to(torch.int32).sub(127).float().exp2()
    return s.float().contiguous()


_MAX_TOKENS = int(os.environ.get("VLLM_DS41_FP8_GEMV_MAX_TOKENS", "16"))
_TUNE = os.environ.get("VLLM_DS41_FP8_GEMV_TUNE", "1") == "1"


# (BLOCK_N, BLOCK_K, K_PER or 0 = whole K, num_warps) candidates for the
# per-shape tuning at load time (the best one differs a lot by shape).
_CANDIDATES = [(bn, bk, kp, nw) for bn, bk, kp in
               ((8, 128, 1), (8, 512, 4), (16, 128, 2), (32, 256, 0), (64, 128, 1),
                (64, 128, 2), (64, 256, 4), (64, 128, 0)) for nw in (2, 4, 8)]
_tuned: dict[tuple, tuple] = {}


def _tune(w: torch.Tensor, s: torch.Tensor) -> tuple:
    """Fastest launch config for this weight shape (one token), cached."""
    import time

    N, K = w.shape
    key = (N, K, w.device)
    if key in _tuned:
        return _tuned[key]
    x = torch.randn(1, 1, K, device=w.device).to(torch.bfloat16)
    best = None
    for bn, bk, kp, nw in _CANDIDATES:
        if N % bn:
            continue
        kpe = triton.cdiv(K, bk) if kp == 0 else min(kp, triton.cdiv(K, bk))
        try:
            for _ in range(3):
                _launch(x, w, s, 1, torch.bfloat16, bn, bk, kpe, nw)
            torch.accelerator.synchronize()
            t = time.perf_counter()
            for _ in range(20):
                _launch(x, w, s, 1, torch.bfloat16, bn, bk, kpe, nw)
            torch.accelerator.synchronize()
            dt = time.perf_counter() - t
        except Exception:  # noqa: BLE001 - a config may not compile
            continue
        if best is None or dt < best[0]:
            best = (dt, (bn, bk, kpe, nw))
    _tuned[key] = best[1] if best else (32, 256, triton.cdiv(K, 256), 4)
    return _tuned[key]


class DecodeFp8Linear:
    """Per-layer proxy for a block-FP8 linear's quant method: up to
    `_MAX_TOKENS` tokens run one Triton GEMV on the stored weight; larger
    batches (and anything unexpected) use the original method."""

    def __init__(self, orig, layer: torch.nn.Module, scales: torch.Tensor):
        self._orig = orig
        self._sf = scales
        self._cfg = _tune(layer.weight.data, scales) if _TUNE else (
            32, 256, triton.cdiv(layer.weight.shape[1], 256), 4)

    def __getattr__(self, name):
        return getattr(self._orig, name)

    def apply(self, layer, x, bias=None):
        if x.dim() == 2 and x.shape[0] <= _MAX_TOKENS and x.shape[0] > 0:
            bn, bk, kp, nw = self._cfg
            y = _launch(x.contiguous().view(x.shape[0], 1, -1), layer.weight, self._sf, 1,
                        x.dtype, bn, bk, kp, nw)
            return y if bias is None else y + bias
        return self._orig.apply(layer, x, bias)


def install_decode_fp8_linears(model: torch.nn.Module, skip=()) -> int:
    """Put DecodeFp8Linear in front of every 32x32 block-FP8 linear layer."""
    if os.environ.get("VLLM_DS41_FP8_GEMV", "1") != "1":
        return 0
    from vllm.model_executor.layers.linear import LinearBase

    n = 0
    for name, m in model.named_modules():
        if not isinstance(m, LinearBase) or any(k in name for k in skip):
            continue
        w = getattr(m, "weight", None)
        if w is None or w.dtype != torch.float8_e4m3fn or w.dim() != 2 or w.stride(1) != 1:
            continue
        s = getattr(m, "weight_scale_inv", None)
        if s is None:
            s = getattr(m, "weight_scale", None)
        N, K = w.shape
        if s is None or tuple(s.shape) != (N // 32, K // 32) or N % 32 or K % 32:
            continue
        if isinstance(m.quant_method, DecodeFp8Linear):
            continue
        m.quant_method = DecodeFp8Linear(m.quant_method, m, block_scales_f32(s.data))
        n += 1
    from vllm.logger import init_logger

    init_logger(__name__).info("decode FP8 GEMV configs (N, K) -> (BLOCK_N, BLOCK_K, "
                               "K_PER, warps): %s",
                               {k[:2]: v for k, v in _tuned.items()})
    return n
