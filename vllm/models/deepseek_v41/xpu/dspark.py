# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark draft model for DeepSeek V4.1 on XPU (semi-autoregressive drafting).

Port of models/deepseek_v41/amd/dspark.py onto the XPU layers: three draft
decoder blocks (checkpoint mtp.0-2, sliding-window attention only) built from
xpu.model.DeepseekV4DecoderLayer, whose MoE for these layers is XpuDSparkMoE
(resident MXFP4 experts); a context combiner over the target's layer 37-39
hidden states; and the Markov and confidence heads. The embedding and LM head
are shared with the target.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import regex as re
import torch
import torch.nn as nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3_dspark import DSparkConfidenceHead, DSparkMarkovHead
from vllm.model_executor.models.utils import maybe_prefix

from .model import DeepseekV4DecoderLayer

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

logger = init_logger(__name__)
_DEBUG = __import__("os").environ.get("VLLM_DS41_DSPARK_DEBUG") == "1"
_dbg_n = [0]


def _dbg(msg, *args):
    if _DEBUG and _dbg_n[0] < 400:
        _dbg_n[0] += 1
        logger.info("DSPARKDBG " + msg, *args)

_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")


class _SharedPlaceholder(nn.Module):
    """Stands in for a module the speculator replaces with the target's."""

    def forward(self, *args, **kwargs):
        raise RuntimeError("DSpark XPU: the target's embedding / LM head was not shared")


class DSparkDeepseekV4Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        config = getattr(config, "text_config", config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.num_hidden_layers = config.num_hidden_layers
        self.target_layer_ids = tuple(config.dspark_target_layer_ids)
        self.num_dspark_layers = (
            getattr(config, "n_mtp_layers", None)
            or getattr(config, "num_nextn_predict_layers", None)
            or 3
        )
        # Replaced by the target's embedding when the speculator loads the
        # draft (has_own_embed_tokens = False): a placeholder, not a vocab-sized
        # copy (~0.4 GiB per GPU at TP=3, counted as model memory at load).
        self.embed_tokens = _SharedPlaceholder()
        self.main_proj = ReplicatedLinear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "main_proj"),
        )
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
        )
        current = get_current_vllm_config()
        self.layers = nn.ModuleList([
            DeepseekV4DecoderLayer(
                current,
                prefix=maybe_prefix(prefix, f"layers.{self.num_hidden_layers + i}"),
                topk_indices_buffer=self.topk_indices_buffer,
            )
            for i in range(self.num_dspark_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        draft_vocab_size = getattr(config, "draft_vocab_size", None) or config.vocab_size
        self.markov_head = DSparkMarkovHead(
            config.vocab_size, draft_vocab_size, config.dspark_markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
        )
        self.confidence_head: DSparkConfidenceHead | None = None
        if getattr(config, "enable_confidence_head", True):
            self.confidence_head = DSparkConfidenceHead(
                config.hidden_size + config.dspark_markov_rank,
                prefix=maybe_prefix(prefix, "confidence_head"),
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        """main_x = main_norm(main_proj(concat of the target's aux hidden states))."""
        out = self.main_norm(self.main_proj(aux_hidden_states))
        if _DEBUG:
            seg = aux_hidden_states.float().abs().view(aux_hidden_states.shape[0], -1, self.hidden_size)
            _dbg("combine: aux %s seg mean|x| %s -> main_x %s mean|x| %.4f",
                 tuple(aux_hidden_states.shape), [round(v, 4) for v in seg.mean((0, 2)).tolist()],
                 tuple(out.shape), out.float().abs().mean().item())
        return out

    @torch.inference_mode()
    def precompute_and_store_context_kv(self, main_x, context_positions,
                                        context_slot_mappings=None) -> None:
        """Each draft layer's context KV comes from the same projected target hidden
        `main_x` through that layer's own wkv + kv_norm + RoPE, written to its
        sliding-window cache (fp8_ds_mla). None slot mappings: profiling, no write."""
        from vllm.models.deepseek_v4.common.ops.cache_utils import quantize_and_insert_k_cache

        from .rope_xpu import gptj_rope_

        for i, layer in enumerate(self.layers):
            slot_mapping = None if context_slot_mappings is None else context_slot_mappings[i]
            attn = layer.attn
            qr_kv = attn.fused_wqa_wkv(main_x)
            if isinstance(qr_kv, tuple):
                qr_kv = qr_kv[0]
            kv = attn.kv_norm(qr_kv[..., attn.q_lora_rank:])
            if slot_mapping is None:
                continue
            if _DEBUG and i == 0:
                _dbg("context kv: T %d pos %s..%s slots %s kv mean|x| %.4f", kv.shape[0],
                     context_positions[:4].tolist(), context_positions[-2:].tolist(),
                     slot_mapping[:6].tolist(), kv.float().abs().mean().item())
            kv = kv.contiguous().clone()
            gptj_rope_(kv, context_positions, attn.rotary_emb.cos_sin_cache)
            cache = attn.swa_cache_layer.kv_cache
            quantize_and_insert_k_cache(
                kv, cache.view(cache.shape[0], -1), slot_mapping,
                block_size=attn.swa_cache_layer.block_size,
            )

    def forward(self, input_ids, positions, inputs_embeds=None) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states = inputs_embeds.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        if _DEBUG:
            _dbg("forward: T %d ids %s pos %s embeds mean|x| %.4f", positions.shape[0],
                 None if input_ids is None else input_ids[:12].tolist(), positions[:12].tolist(),
                 inputs_embeds.float().abs().mean().item())
        residual = post_mix = res_mix = pre_mix = None
        for li, layer in enumerate(self.layers):
            hidden_states, residual, post_mix, res_mix, pre_mix = layer(
                hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual,
            )
            if _DEBUG:
                _dbg("  layer %d: ffn out mean|x| %.4f residual mean|x| %.4f", li,
                     hidden_states.float().abs().mean().item(),
                     residual.float().abs().mean().item())
        hidden_states = self.layers[-1].mhc_post(hidden_states, residual, post_mix, res_mix)
        # Collapse the hc copies with the last layer's FFN pre-mix; return the
        # pre-norm head hidden (compute_logits applies self.norm).
        return torch.ops.vllm.hc_collapse_triton(hidden_states, pre_mix)


class DSparkDeepseekV4ForCausalLM(nn.Module):
    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        cfg = self.draft_model_config.hf_config
        self.config = getattr(cfg, "text_config", cfg)
        self.quant_config = vllm_config.quant_config
        self.model = DSparkDeepseekV4Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = _SharedPlaceholder()  # replaced by the target's LM head
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    # --- speculator hooks ---------------------------------------------------
    def embed_input_ids(self, input_ids):
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, aux_hidden_states):
        return self.model.combine_hidden_states(aux_hidden_states)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.attn.swa_cache_layer.prefix for layer in self.model.layers]

    def precompute_and_store_context_kv(self, context_states, context_positions,
                                        context_slot_mappings=None) -> None:
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mappings
        )

    def forward(self, input_ids, positions, inputs_embeds=None):
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states):
        logits = self.logits_processor(self.lm_head, self.model.norm(hidden_states))
        if _DEBUG and logits is not None:
            top = logits.float().topk(2, dim=-1)
            _dbg("logits: T %d top1 %s p1 %s", logits.shape[0], top.indices[:12, 0].tolist(),
                 [round(v, 2) for v in top.values[:12, 0].tolist()])
        return logits

    def compute_draft_logits(self, hidden_states):
        return self.compute_logits(hidden_states)

    def map_draft_to_target(self, draft_ids):
        return draft_ids

    def markov_embed(self, token_ids):
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed):
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    def compute_confidence(self, head_hidden, markov_embed):
        assert self.model.confidence_head is not None
        return torch.sigmoid(self.model.confidence_head(head_hidden, markov_embed))

    def sample_sequential(self, speculator: DSparkSpeculator, num_reqs: int,
                          head_hidden: torch.Tensor) -> bool:
        return False  # the speculator's own loop

    # --- weights ------------------------------------------------------------
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the mtp.{0,1,2}.* draft weights (from the original checkpoint)."""
        n_draft_experts = (
            getattr(self.config, "dspark_n_routed_experts", 0) or self.config.n_routed_experts
        )
        expert_mapping = fused_moe_make_expert_params_mapping(
            self, ckpt_gate_proj_name="w1", ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3", num_experts=n_draft_experts,
            routed_experts_prefix="routed_experts",
        )
        stacked = [
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
        ]
        params = dict(self.named_parameters())
        loaded: set[str] = set()

        def resolve(name):
            if name not in params:
                head, _, leaf = name.rpartition(".")
                if f"{head}.base_layer.{leaf}" in params:
                    return f"{head}.base_layer.{leaf}"
                if name.endswith("_inv") and name.removesuffix("_inv") in params:
                    return name.removesuffix("_inv")
                if f"{name}_inv" in params:
                    return f"{name}_inv"
            return name

        for name, w in weights:
            name = self._remap(name)
            if name is None:
                continue
            if name.endswith(".scale"):
                suffix = ".weight_scale" if _EXPERT_SCALE_RE.search(name) else ".weight_scale_inv"
                name = name.removesuffix(".scale") + suffix
            if ".shared_experts.w2" in name:
                name = name.replace(".shared_experts.w2", ".shared_experts.down_proj")
            if ".experts." in name and ".shared_experts." not in name:
                if "weight_scale" in name and w.dtype == torch.float8_e8m0fnu:
                    w = w.view(torch.uint8)
                for pname, wname, eid, sid in expert_mapping:
                    if wname not in name:
                        continue
                    n2 = name.replace(wname, pname)
                    if n2 not in params:
                        continue
                    param = params[n2]
                    if param.weight_loader(param, w, n2, shard_id=sid, expert_id=eid,
                                           return_success=True):
                        loaded.add(n2)
                        break
                continue
            is_layer = name.startswith("model.layers.")
            for pname, wname, sid in stacked:
                if not is_layer or wname not in name:
                    continue
                n2 = resolve(name.replace(wname, pname))
                params[n2].weight_loader(params[n2], w, sid)
                loaded.add(n2)
                break
            else:
                if name.endswith(".ffn.gate.bias"):
                    name = name.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")
                n2 = resolve(name)
                if n2 not in params:
                    logger.warning_once("DSpark XPU: no parameter for %s", name)
                    continue
                param = params[n2]
                getattr(param, "weight_loader", default_weight_loader)(param, w)
                loaded.add(n2)
        logger.info("DSpark XPU draft model loaded: %d params", len(loaded))
        return loaded

    def process_weights_after_loading(self) -> None:
        from .attention_xpu import DeepseekV41XPUAttention
        from .fp8_gemv import install_decode_fp8_linears

        for m in self.modules():
            if isinstance(m, DeepseekV41XPUAttention):
                m.prepare_decode_weights()
        n = install_decode_fp8_linears(self, skip=(".wo_a",))
        logger.info("DSpark XPU: decode FP8 GEMV on %d block-FP8 linear layers", n)
        # Memory by component (parameters and buffers on the device).
        groups: dict[str, int] = {}
        seen: set[int] = set()
        for name, t in list(self.named_parameters()) + list(self.named_buffers()):
            if t.device.type == "cpu" or t.data_ptr() in seen:
                continue
            seen.add(t.data_ptr())
            key = ("routed_experts" if ".experts." in name and "shared" not in name
                   else "shared_experts" if "shared_experts" in name
                   else "attention" if ".attn." in name
                   else name.split(".")[1] if name.startswith("model.") else name.split(".")[0])
            groups[key] = groups.get(key, 0) + t.numel() * t.element_size()
        big = sorted(((t.numel() * t.element_size(), n, tuple(t.shape), str(t.dtype))
                      for n, t in self.named_parameters()), reverse=True)[:6]
        logger.info("DSpark XPU largest tensors: %s",
                    "; ".join(f"{n} {sh} {dt} {b / 2**20:.0f} MiB" for b, n, sh, dt in big))
        logger.info("DSpark XPU memory by part (GiB): %s; allocated now %.2f GiB",
                    ", ".join(f"{k} {v / 2**30:.3f}" for k, v in sorted(groups.items(),
                                                                       key=lambda kv: -kv[1])),
                    torch.xpu.memory_allocated() / 2**30)

    def _remap(self, name: str) -> str | None:
        m = re.match(r"mtp\.(\d+)\.(.*)", name)
        if m is None:
            return None
        stage, rest = int(m.group(1)), m.group(2)
        if rest.startswith("confidence_head.") and self.model.confidence_head is None:
            return None
        if rest.startswith("markov_head.embed."):
            return "model.markov_head.markov_w1." + rest.removeprefix("markov_head.embed.")
        if rest.startswith("markov_head.head."):
            return "model.markov_head.markov_w2." + rest.removeprefix("markov_head.head.")
        if rest.startswith(("main_proj.", "main_norm.", "norm.", "markov_head.",
                            "confidence_head.")):
            return f"model.{rest}"
        return f"model.layers.{stage}.{rest}"


# --- VLLM_DS41_DSPARK_TIMING=1: host and GPU time of the DSpark speculator per step -------------
def _install_dspark_timing() -> None:
    """Wrap DSparkSpeculator.propose / _generate_draft with host timers and XPU timing events;
    log averages every 30 s (rank 0). Events are resolved 64 calls later (long finished)."""
    import os as _os
    import time as _time

    from vllm.v1.worker.gpu.spec_decode.dspark import speculator as _sp

    cls = _sp.DSparkSpeculator
    st: dict = {}
    last = [_time.time()]

    def wrap(name):
        orig = getattr(cls, name)

        def timed(self, *a, **k):
            t0 = _time.perf_counter()
            e0 = torch.xpu.Event(enable_timing=True)
            e0.record()
            r = orig(self, *a, **k)
            e1 = torch.xpu.Event(enable_timing=True)
            e1.record()
            s = st.setdefault(name, {"n": 0, "host": 0.0, "gpu": 0.0, "ng": 0, "q": []})
            s["n"] += 1
            s["host"] += _time.perf_counter() - t0
            s["q"].append((e0, e1))
            while len(s["q"]) > 64:
                a0, a1 = s["q"].pop(0)
                a1.synchronize()
                s["gpu"] += a0.elapsed_time(a1) / 1e3
                s["ng"] += 1
            if _time.time() - last[0] > 30:
                last[0] = _time.time()
                logger.info("DSpark timing: %s", ", ".join(
                    f"{k} {v['n']} calls host {1e3 * v['host'] / max(v['n'], 1):.2f} ms "
                    f"gpu {1e3 * v['gpu'] / max(v['ng'], 1):.2f} ms" for k, v in st.items()))
            return r

        setattr(cls, name, timed)

    for name in ("propose", "_generate_draft"):
        if hasattr(cls, name):
            wrap(name)


if __import__("os").environ.get("VLLM_DS41_DSPARK_TIMING") == "1":
    try:
        _install_dspark_timing()
    except Exception as _ex:  # noqa: BLE001
        logger.warning("DSpark timing not installed: %s", _ex)
