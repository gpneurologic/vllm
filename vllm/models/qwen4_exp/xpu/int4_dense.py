# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serve Qwen4Exp's block-FP8 dense linears as INT4 weight-only on XPU.

`VLLM_QWEN4_INT4_DENSE` is a comma list of groups to convert at load:
``linear`` (linear-attention in_proj_qkvz / out_proj), ``attn`` (QSA qkv_proj /
o_proj), ``shared`` (shared experts), or ``all``. Each converted layer is first
loaded as block FP8; its dequantized weight is then read back by running the
FP8 layer on an identity matrix (independent of how the FP8 method stores its
weights), quantized to symmetric INT4 per group of 128 input values, and run
by oneDNN ``int4_gemm_w4a16`` (``xpu_int4_weight_only``).
"""

import os
import re

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_GROUPS = {
    "linear": r"linear_attn\.(?:in_proj_qkvz|out_proj)",
    "attn": r"self_attn\.(?:qkv_proj|o_proj)",
    "shared": r"mlp\.shared_expert\.[a-z_]+",
}
_SELECTED = [
    g.strip() for g in os.environ.get("VLLM_QWEN4_INT4_DENSE", "").split(",") if g.strip()
]
if "all" in _SELECTED:
    _SELECTED = list(_GROUPS)
_RE = (
    re.compile(r"(?:^|\.)layers\.\d+\.(?:" + "|".join(_GROUPS[g] for g in _SELECTED) + r")$")
    if _SELECTED
    else None
)


def wants_int4(prefix: str) -> bool:
    return _RE is not None and _RE.search(prefix) is not None


def int4_from_fp8_method(fp8_method_cls, quant_config):
    """An Fp8LinearMethod that converts its layer to INT4 after loading."""

    class Int4FromFp8LinearMethod(fp8_method_cls):
        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            super().process_weights_after_loading(layer)
            from vllm.model_executor.layers.quantization.xpu_int4_weight_only import (
                GROUP,
                XPUInt4WeightOnlyMethod,
                quantize_int4,
            )

            k = layer.input_size_per_partition
            n = layer.output_size_per_partition
            if k % GROUP:
                return  # keep FP8
            dev = layer.weight.device
            w = torch.empty((n, k), dtype=torch.bfloat16, device=dev)
            step = 1024  # identity in slices keeps the temporary small
            for c0 in range(0, k, step):
                c1 = min(k, c0 + step)
                eye = torch.zeros((c1 - c0, k), dtype=torch.bfloat16, device=dev)
                eye[:, c0:c1] = torch.eye(c1 - c0, dtype=torch.bfloat16, device=dev)
                w[:, c0:c1] = super().apply(layer, eye).t()
            packed, scales = quantize_int4(w)
            for name in list(layer._parameters):
                if name != "bias":
                    delattr(layer, name)
            for name in ("weight_scale_inv", "weight_scale", "input_scale"):
                if hasattr(layer, name):
                    delattr(layer, name)
            layer.weight = torch.nn.Parameter(packed, requires_grad=False)
            layer.register_buffer("int4_scales", scales, persistent=False)
            layer.quant_method = XPUInt4WeightOnlyMethod()

    return Int4FromFp8LinearMethod(quant_config)


def log_selection() -> None:
    if _SELECTED:
        logger.info("Qwen4Exp dense linears served as INT4 weight-only: %s", _SELECTED)
