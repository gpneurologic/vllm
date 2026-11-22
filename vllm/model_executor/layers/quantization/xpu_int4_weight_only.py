# SPDX-License-Identifier: Apache-2.0
"""Load-time INT4 weight-only (W4A16) conversion for 16-bit layers on XPU.

Symmetric INT4 with one bf16 scale per group of `VLLM_XPU_INT4_GROUP` (128)
input dims, packed GPTQ-style for oneDNN ``_xpu_C.int4_gemm_w4a16`` (zero point
8). Same role as ``xpu_fp8_weight_only`` at half the bytes. Used for:

  VLLM_MTP_INT4=1           the MTP draft layer (drafts only: the target verifies
                            every token, so the served text cannot change)
  VLLM_MTP_DRAFT_HEAD=int4  the reduced-vocabulary draft head rows

The op has a fixed cost of ~11-20 us per call on a B70, so it pays on large
matrices and can lose on small ones; time it in graph mode.
"""

import os

import torch
from torch.nn import Parameter

from vllm.logger import init_logger

logger = init_logger(__name__)

GROUP = int(os.environ.get("VLLM_XPU_INT4_GROUP", "128"))


def quantize_int4(w: torch.Tensor, group: int = GROUP) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, K] -> (int32 [N, K/8] packed along K, bf16 scales [K/group, N])."""
    n, k = w.shape
    packed = torch.empty((n, k // 8), dtype=torch.int32, device=w.device)
    scales = torch.empty((k // group, n), dtype=torch.bfloat16, device=w.device)
    shifts = torch.arange(8, device=w.device, dtype=torch.int64) * 4
    step = 4096  # keep the float32 temporaries small
    for r0 in range(0, n, step):
        r1 = min(n, r0 + step)
        g = w[r0:r1].float().view(r1 - r0, k // group, group)
        s = (g.abs().amax(-1) / 7.0).clamp(min=1e-12)
        s = s.to(torch.bfloat16).float()  # quantize against the stored scale
        q = (torch.round(g / s[..., None]).clamp(-8, 7) + 8).to(torch.int64)
        p64 = (q.view(r1 - r0, k // 8, 8) << shifts).sum(-1)
        packed[r0:r1] = (p64 - (p64 >= 2**31).to(torch.int64) * 2**32).to(torch.int32)
        scales[:, r0:r1] = s.t().to(torch.bfloat16)
    return packed, scales


def dequantize_int4(packed: torch.Tensor, scales: torch.Tensor,
                    group: int = GROUP) -> torch.Tensor:
    """Reference inverse of quantize_int4 -> float32 [N, K]."""
    n, k8 = packed.shape
    shifts = torch.arange(8, device=packed.device, dtype=torch.int64) * 4
    q = ((packed.to(torch.int64)[..., None] >> shifts) & 0xF).view(n, k8 * 8).float() - 8
    return (q.view(n, -1, group) * scales.t().float()[..., None]).view(n, k8 * 8)


_ZP: dict[torch.device, torch.Tensor] = {}


def int4_linear(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor,
                bias: torch.Tensor | None = None, group: int = GROUP) -> torch.Tensor:
    zp = _ZP.get(packed.device)
    if zp is None:
        zp = _ZP[packed.device] = torch.tensor([8], dtype=torch.int8, device=packed.device)
    x2 = x.reshape(-1, x.shape[-1])
    if x2.dtype != torch.bfloat16:
        x2 = x2.to(torch.bfloat16)
    out = torch.ops._xpu_C.int4_gemm_w4a16(x2, packed.t(), bias, scales, zp, group, None)
    return out.reshape(*x.shape[:-1], packed.shape[0]).to(x.dtype)


class XPUInt4WeightOnlyMethod:
    """Stands in for a layer's quant_method after conversion."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              bias: torch.Tensor | None = None) -> torch.Tensor:
        return int4_linear(x, layer.weight, layer.int4_scales, bias)


def convert_layer_int4(layer: torch.nn.Module) -> int:
    """Convert one layer in place; returns the bytes saved."""
    w = layer.weight.data
    q, s = quantize_int4(w)
    saved = w.numel() * w.element_size() - q.numel() * 4 - s.numel() * 2
    layer.weight = Parameter(q, requires_grad=False)
    layer.register_buffer("int4_scales", s, persistent=False)
    layer.quant_method = XPUInt4WeightOnlyMethod()
    return saved


def convert_unquantized_linears_int4(module: torch.nn.Module, what: str) -> None:
    """INT4 for every 16-bit linear whose input size fits the group; the rest FP8."""
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

    from .xpu_fp8_weight_only import convert_layer_fp8

    n4 = n8 = 0
    saved = 0
    for _, mod in module.named_modules():
        if not (isinstance(mod, LinearBase)
                and isinstance(mod.quant_method, UnquantizedLinearMethod)
                and getattr(mod, "weight", None) is not None and mod.weight.dim() == 2
                and mod.weight.dtype in (torch.float16, torch.bfloat16)):
            continue
        if mod.weight.shape[1] % GROUP == 0:
            saved += convert_layer_int4(mod)
            n4 += 1
        else:
            saved += convert_layer_fp8(mod)
            n8 += 1
    torch.xpu.empty_cache()
    logger.info("%s: %d linear layers converted to INT4 weight-only, %d to FP8 "
                "(%.0f MiB saved per rank)", what, n4, n8, saved / 2**20)
