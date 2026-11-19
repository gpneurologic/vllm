# SPDX-License-Identifier: Apache-2.0
"""Store block-FP8 linears as INT4 (W4A16) after loading, to free VRAM.

Each selected linear's FP8 weight (32x32 power-of-two block scales) is
dequantized on the device, rounded to symmetric INT4 per group of
`group_size` input values (q = round(w / s) + 8, s = amax / 7), packed GPTQ
style (8 values per int32 along K), and the FP8 weight and scales are
dropped. The layer's quant method is replaced by a proxy that runs
torch.ops._xpu_C.int4_gemm_w4a16, the kernel vLLM's XPU W4A16 path uses
(XPUwNa16LinearKernel), for every batch size.

VLLM_DS41_INT4_DENSE=1 turns it on; VLLM_DS41_INT4_GROUP sets the group
size (default 128); VLLM_DS41_INT4_SKIP lists comma-separated name
substrings to keep in FP8."""
from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)
ENABLED = os.environ.get("VLLM_DS41_INT4_DENSE") == "1"
GROUP = int(os.environ.get("VLLM_DS41_INT4_GROUP", "128"))
SKIP = tuple(s for s in os.environ.get("VLLM_DS41_INT4_SKIP", "").split(",") if s)


def _dequant_fp8_block(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """w [N, K] fp8, s [N/32, K/32] (float or e8m0) -> float32 [N, K]."""
    from .fp8_gemv import block_scales_f32

    sf = block_scales_f32(s)
    N, K = w.shape
    return (w.float().view(N // 32, 32, K // 32, 32) * sf[:, None, :, None]).view(N, K)


def quantize_int4(wf: torch.Tensor, group: int) -> tuple[torch.Tensor, torch.Tensor]:
    """float [N, K] -> (int32 [N, K/8] packed along K, bf16 scales [K/group, N])."""
    N, K = wf.shape
    g = wf.view(N, K // group, group)
    s = (g.abs().amax(-1) / 7.0).clamp(min=1e-12)  # [N, K/group]
    q = (torch.round(g / s[..., None]).clamp(-8, 7) + 8).to(torch.int64).view(N, K // 8, 8)
    shifts = torch.arange(8, device=wf.device, dtype=torch.int64) * 4
    p64 = (q << shifts).sum(-1)  # disjoint bits, < 2^32
    packed = (p64 - (p64 >= 2**31).to(torch.int64) * 2**32).to(torch.int32)  # same bits
    return packed.contiguous(), s.t().contiguous().to(torch.bfloat16)


def dequant_int4(packed: torch.Tensor, scales: torch.Tensor, group: int) -> torch.Tensor:
    """Reference inverse of quantize_int4 -> float32 [N, K]."""
    N, K8 = packed.shape
    shifts = torch.arange(8, device=packed.device, dtype=torch.int64) * 4
    q = ((packed.to(torch.int64)[..., None] >> shifts) & 0xF).view(N, K8 * 8).float() - 8
    return (q.view(N, -1, group) * scales.t().float()[..., None]).view(N, K8 * 8)


class Int4LinearMethod:
    """Quant-method proxy for one layer stored as INT4."""

    def __init__(self, orig, group: int, n_out: int):
        self._orig = orig
        self.group = group
        self.n_out = n_out

    def __getattr__(self, name):
        return getattr(self._orig, name)

    def apply(self, layer, x, bias=None):
        x2 = x.reshape(-1, x.shape[-1])
        if x2.dtype != torch.bfloat16:
            x2 = x2.to(torch.bfloat16)
        out = torch.ops._xpu_C.int4_gemm_w4a16(
            x2, layer.weight.t(), bias, layer.int4_scales, layer.int4_zp, self.group, None)
        return out.reshape(*x.shape[:-1], self.n_out)


def convert_dense_to_int4(model: torch.nn.Module, skip=()) -> tuple[int, int, int]:
    """Convert every 32x32 block-FP8 LinearBase (except `skip` names) to INT4.
    Returns (layers, bytes before, bytes after)."""
    from vllm.model_executor.layers.linear import LinearBase

    n = before = after = 0
    group = GROUP
    for name, m in model.named_modules():
        if not isinstance(m, LinearBase) or any(k in name for k in tuple(skip) + SKIP):
            continue
        w = getattr(m, "weight", None)
        if w is None or w.dtype != torch.float8_e4m3fn or w.dim() != 2:
            continue
        s_name = "weight_scale_inv" if hasattr(m, "weight_scale_inv") else "weight_scale"
        s = getattr(m, s_name, None)
        N, K = w.shape
        if s is None or tuple(s.shape) != (N // 32, K // 32) or K % group or K % 8:
            continue
        before += w.numel() + s.numel() * s.element_size()
        packed_rows, scale_rows = [], []
        for r0 in range(0, N, 4096):  # bounded temporary memory
            r1 = min(N, r0 + 4096)
            wf = _dequant_fp8_block(w.data[r0:r1], s.data[r0 // 32:r1 // 32])
            p, sc = quantize_int4(wf, group)
            packed_rows.append(p)
            scale_rows.append(sc)
            del wf
        packed = torch.cat(packed_rows, 0)
        scales = torch.cat(scale_rows, 1).contiguous()
        m.weight = torch.nn.Parameter(packed, requires_grad=False)
        delattr(m, s_name)
        m.register_buffer("int4_scales", scales, persistent=False)
        m.register_buffer("int4_zp", torch.tensor([8], dtype=torch.int8, device=packed.device),
                          persistent=False)
        m.quant_method = Int4LinearMethod(m.quant_method, group, N)
        after += packed.numel() * 4 + scales.numel() * 2
        n += 1
    torch.accelerator.empty_cache()
    return n, before, after
