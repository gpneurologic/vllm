# SPDX-License-Identifier: Apache-2.0
"""MoE for the DeepSeek V4.1 DSpark draft layers on XPU.

The draft layers (mtp.0-2) have 128 MXFP4 routed experts (top 3) and a shared
expert. Unlike the target's 192 GGUF experts they fit on the cards, so they
use vLLM's resident FusedMoE (MXFP4 method, XPU backend), which also works in
captured graphs. Routing is DeepSeek's: sqrt(softplus) scores, top-k on scores
plus a correction bias, renormalized and scaled.

The XPU MXFP4 expert kernels support plain SiLU only, not DeepSeek's clamped
SwiGLU (swiglu_limit), so the routed experts run without the clamp. This only
affects how good the drafts are; the target model verifies every token.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear


def _mxfp4_experts(quant_config):
    """The engine's quant config with MXFP4 for routed experts.

    On XPU the DeepSeek checkpoint runs under vLLM's plain "fp8" config (the
    DeepSeek-specific config is only selected for model type deepseek_v4 and
    on CUDA / ROCm), which gives routed experts the FP8 method: the packed FP4
    checkpoint bytes would be read as FP8 values. The XPU MXFP4 experts take
    bf16 activations (activation_key None). Other layers keep the FP8 config.
    """
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod

    qc = copy.copy(quant_config)
    base = quant_config.get_quant_method

    def get_quant_method(layer, prefix):
        if isinstance(layer, RoutedExperts):
            return Mxfp4MoEMethod(layer.moe_config, activation_key=None)
        return base(layer, prefix)

    qc.get_quant_method = get_quant_method
    return qc


class XpuDSparkMoE(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        from .model import DeepseekV4MLP

        config = vllm_config.model_config.hf_config
        quant_config = _mxfp4_experts(vllm_config.quant_config)
        self.n_routed = getattr(config, "dspark_n_routed_experts", 0) or config.n_routed_experts
        self.topk = getattr(config, "dspark_num_experts_per_tok", 0) or config.num_experts_per_tok
        self.gate = GateLinear(
            input_size=config.hidden_size,
            output_size=self.n_routed,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(self.n_routed, dtype=torch.float32), requires_grad=False
        )
        self.gate.tid2eid = None
        self.gate.bias_vl = None
        self.shared_experts = DeepseekV4MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            hidden_act=config.hidden_act,
            swiglu_limit=config.swiglu_limit,
            quant_config=quant_config,
            reduce_results=False,
            prefix=f"{prefix}.shared_experts",
        )
        self.experts = FusedMoEFactory(
            reduce_results=True,
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=self.n_routed,
            top_k=self.topk,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            ckpt_names=("w1", "w2", "w3"),
            scoring_func=getattr(config, "scoring_func", "sqrtsoftplus"),
            routed_scaling_factor=config.routed_scaling_factor,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            swiglu_limit=None,  # not supported by the XPU MXFP4 experts (see module doc)
            router_logits_dtype=torch.float32,
        )

    def forward(self, x: torch.Tensor, input_ids=None) -> torch.Tensor:
        return self.experts(hidden_states=x, router_logits=x, input_ids=input_ids).view(x.shape)
