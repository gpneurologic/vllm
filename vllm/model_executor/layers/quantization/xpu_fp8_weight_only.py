# SPDX-License-Identifier: Apache-2.0
"""Load-time FP8 weight-only conversion for 16-bit layers on XPU.

Turns an unquantized linear layer (or the lm_head) into per-output-row FP8 E4M3
weights with fp16/bf16 activations, run by oneDNN ``_xpu_C.fp8_gemm_w8a16``.
On a B70 that halves the bytes read at about the same bandwidth (~1.8-2x faster
for decode-sized batches). Used for:

  VLLM_MTP_FP8=1           the MTP draft layer (its outputs are only drafts, which
                           the target verifies, so the served text cannot change)
  VLLM_XPU_LM_HEAD_FP8=1   the target lm_head (changes logits slightly)
"""

import torch
from torch.nn import Parameter

from vllm.logger import init_logger

logger = init_logger(__name__)

FP8_MAX = 448.0


def quantize_rows_fp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, K] 16-bit -> (fp8 e4m3 [N, K], float32 scale [1, N]); one scale per row."""
    out = torch.empty(w.shape, dtype=torch.float8_e4m3fn, device=w.device)
    scale = torch.empty((w.shape[0], 1), dtype=torch.float32, device=w.device)
    step = 8192  # keep the float32 temporary small
    for i in range(0, w.shape[0], step):
        blk = w[i:i + step].float()
        s = blk.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / FP8_MAX
        out[i:i + step] = (blk / s).to(torch.float8_e4m3fn)
        scale[i:i + step] = s
    return out, scale.t().contiguous()


def fp8_linear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
               bias: torch.Tensor | None = None) -> torch.Tensor:
    x2 = x.reshape(-1, x.shape[-1])
    out = torch.ops._xpu_C.fp8_gemm_w8a16(x2, weight.t(), scale, bias)
    return out.reshape(*x.shape[:-1], out.shape[-1])


class XPUFp8WeightOnlyMethod:
    """Stands in for a layer's quant_method after conversion."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              bias: torch.Tensor | None = None) -> torch.Tensor:
        return fp8_linear(x, layer.weight, layer.weight_scale, bias)

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("an FP8-converted lm_head can't be used as an input embedding")


def convert_layer_fp8(layer: torch.nn.Module) -> int:
    """Convert one layer in place; returns the bytes saved."""
    w = layer.weight.data
    q, s = quantize_rows_fp8(w)
    saved = w.numel() * w.element_size() - q.numel() - s.numel() * 4
    layer.weight = Parameter(q, requires_grad=False)
    layer.weight_scale = Parameter(s, requires_grad=False)
    layer.quant_method = XPUFp8WeightOnlyMethod()
    return saved


def convert_unquantized_linears_fp8(module: torch.nn.Module, what: str) -> None:
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

    n, saved = 0, 0
    for _, mod in module.named_modules():
        if (isinstance(mod, LinearBase) and isinstance(mod.quant_method, UnquantizedLinearMethod)
                and getattr(mod, "weight", None) is not None and mod.weight.dim() == 2
                and mod.weight.dtype in (torch.float16, torch.bfloat16)):
            saved += convert_layer_fp8(mod)
            n += 1
    torch.xpu.empty_cache()
    logger.info("%s: %d linear layers converted to FP8 weight-only (%.0f MiB saved per rank)",
                what, n, saved / 2**20)


def convert_lm_head_fp8(model: torch.nn.Module) -> None:
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead, UnquantizedEmbeddingMethod)

    for name, mod in model.named_modules():
        if (isinstance(mod, ParallelLMHead) and isinstance(mod.quant_method, UnquantizedEmbeddingMethod)
                and mod.weight.dtype in (torch.float16, torch.bfloat16)):
            saved = convert_layer_fp8(mod)
            torch.xpu.empty_cache()
            logger.info("%s converted to FP8 weight-only (%.0f MiB saved per rank)", name, saved / 2**20)
