# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 text model for Intel XPU (first bring-up, eager only).

Derived from `deepseek_v41/amd/model.py` with the ROCm-only paths removed
(aiter, mono decode, mega MoE / mega mHC, mHC overlap, decoder replay, sequence
and prefill-context parallel). XPU-specific parts:
- attention: `DeepseekV41XPUAttention` (attention_xpu.py);
- routed experts stay on NVMe and are cached on the device on demand
  (`offload.ExpertCache`); attention, the shared expert, router, embeddings and
  mHC stay resident;
- the Engram tables are read from the checkpoint mmap (`offload.DiskEngramEmbedding`).
"""

import typing
from collections.abc import Iterable
from itertools import islice

import regex as re
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mhc import MHCPostOp, MHCPreDelayedOp
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import EagleModelMixin, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.models.deepseek_v4.xpu.model import DeepseekV4MLP
from vllm.models.deepseek_v41.common.engram import (
    Engram,
    EngramLayout,
    NgramHashState,
    gather_engram_hashes,
)
from vllm.models.deepseek_v41.common.mm_preprocess import image_sentinel_mask
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv

from .attention_xpu import DeepseekV41XPUAttention
from . import prof
from .gguf_loader import GgufEngramEmbedding, gguf_path, load_expert_slice, run_gguf_experts
from .offload import DiskEngramEmbedding, ExpertCache, run_expert

logger = init_logger(__name__)
_DEBUG = __import__("os").environ.get("VLLM_DS41_DEBUG") == "1"


def _avail_mib() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) // 1024
                if avail < 4000:
                    raise RuntimeError(f"DS41DBG host memory low: {avail} MiB")
                return avail
    return -1


class XpuDiskEngram(Engram):
    def __init__(self, config, quant_config, layout, layer_hash_index, prefix, folder):
        object.__setattr__(self, "_ckpt_folder", folder)
        object.__setattr__(self, "_layer_id", extract_layer_index(prefix))
        object.__setattr__(self, "_hf_config", config)
        super().__init__(
            config, quant_config, layout, layer_hash_index,
            use_sequence_parallel=False, prefix=prefix,
        )

    def _create_wkv(self, input_size, quant_config, prefix):
        # Output (hc_mult + 1) * hidden = 25600 does not split over 3 GPUs:
        # keep the projection whole on every GPU (about 157 MB in FP8).
        from vllm.model_executor.layers.linear import ReplicatedLinear

        return ReplicatedLinear(
            input_size,
            self.dim * (self.hc_mult + 1),
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=prefix,
        )

    def _create_embedding(self, layout: EngramLayout, layer_hash_index: int):
        n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        if gguf_path():
            return GgufEngramEmbedding(
                self._hf_config, self._layer_id, n_hash_cols, layout.head_dim
            )
        return DiskEngramEmbedding(
            self._ckpt_folder, self._layer_id, n_hash_cols, layout.head_dim
        )


class _Gate(nn.Module):
    def __init__(self, n_experts: int, hidden: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(n_experts, hidden, dtype=torch.bfloat16), requires_grad=False
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(n_experts, dtype=torch.float32), requires_grad=False
        )


class XpuOffloadMoE(nn.Module):
    """Router + shared expert resident; routed experts from the NVMe cache.

    With TP > 1 the routed experts are split over the TP ranks (expert
    parallel): each rank runs its own experts on all tokens and the partial
    sums are added by one all-reduce together with the shared expert's."""

    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.prefix_ckpt = prefix[prefix.index("layers."):].removesuffix(".ffn")
        self.folder = vllm_config.model_config.model
        self.n_routed = config.n_routed_experts
        self.topk = config.num_experts_per_tok
        self.route_scale = config.routed_scaling_factor
        self.swiglu_limit = float(config.swiglu_limit or 0.0)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.n_local = self.n_routed // self.tp_size
        self.e_start = get_tensor_model_parallel_rank() * self.n_local
        self.gate = _Gate(self.n_routed, config.hidden_size)
        self.shared_experts = DeepseekV4MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            hidden_act=config.hidden_act,
            swiglu_limit=config.swiglu_limit,
            quant_config=vllm_config.quant_config,
            reduce_results=False,
            prefix=f"{prefix}.shared_experts",
        )
        # GGUF builds: the local experts stay on the device in GGUF form.
        self.gguf = gguf_path() is not None
        self.w13 = self.w2 = None
        self.t13 = self.t2 = 0

    def load_gguf_experts(self, ck) -> int:
        layer = int(self.prefix_ckpt.split(".")[1])
        dev = self.gate.weight.device
        self.w13, self.w2, self.t13, self.t2 = load_expert_slice(
            ck, layer, self.e_start, self.n_local, dev
        )
        return self.w13.numel() + self.w2.numel()

    def forward(self, x: torch.Tensor, input_ids=None) -> torch.Tensor:
        with prof.section("router"):
            scores = F.linear(x.float(), self.gate.weight.float())
            scores = F.softplus(scores).sqrt()
            idx = (scores + self.gate.e_score_correction_bias).topk(self.topk, dim=-1)[1]
            w = scores.gather(1, idx)
            w = w / (w.sum(-1, keepdim=True) + 1e-20) * self.route_scale
            out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device=x.device)
        # The memory-profile / warm-up passes run without attention metadata:
        # skip routed experts there so the expert cache (allocated outside
        # vLLM's memory budget) is not counted as model memory.
        real_step = is_forward_context_available() and isinstance(
            get_forward_context().attn_metadata, dict
        )
        if self.gguf:
            if x.shape[0] > 0 and self.w13 is not None:
                with prof.section("routed_moe"):
                    run_gguf_experts(x, idx, w, out, self.w13, self.w2, self.t13, self.t2,
                                     self.e_start, self.n_local, self.swiglu_limit)
        elif x.shape[0] > 0 and real_step:
            local = (idx >= self.e_start) & (idx < self.e_start + self.n_local)
            experts = torch.unique(idx[local]).tolist()
            if experts:
                cache = ExpertCache.get(self.folder, x.device)
                cache.ensure(self.prefix_ckpt, experts)
                for e in experts:
                    rows, slot = torch.where(idx == e)
                    parts = cache.get_expert(self.prefix_ckpt, e)
                    out.index_add_(
                        0, rows,
                        run_expert(x[rows], parts, self.swiglu_limit, w[rows, slot, None]),
                    )
        with prof.section("shared_expert"):
            out = out + self.shared_experts(x).float()
        if self.tp_size > 1:
            with prof.section("allreduce"):
                out = tensor_model_parallel_all_reduce(out)
        return out.to(x.dtype)


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(self, vllm_config, prefix, topk_indices_buffer=None,
                 candidate_block_buffer=None, engram_layout=None):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size
        self.engram: Engram | None = None
        if engram_layout is not None:
            layer_id = extract_layer_index(prefix)
            if layer_id in engram_layout.layer_ids:
                self.engram = XpuDiskEngram(
                    config, vllm_config.quant_config, engram_layout,
                    engram_layout.layer_ids.index(layer_id),
                    prefix=f"{prefix}.engram",
                    folder=vllm_config.model_config.model,
                )
        self.rms_norm_eps = config.rms_norm_eps
        self.attn = DeepseekV41XPUAttention(
            vllm_config,
            prefix=f"{prefix}.attn",
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=None,
            candidate_block_buffer=candidate_block_buffer,
        )
        self.ffn = XpuOffloadMoE(vllm_config, prefix=f"{prefix}.ffn")
        self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size

        def p(*shape):
            return nn.Parameter(torch.empty(*shape, dtype=torch.float32), requires_grad=False)

        self.hc_attn_fn = p(mix_hc, hc_dim)
        self.hc_ffn_fn = p(mix_hc, hc_dim)
        self.hc_attn_base = p(mix_hc)
        self.hc_ffn_base = p(mix_hc)
        self.hc_attn_scale = p(3)
        self.hc_ffn_scale = p(3)
        self.hc_attn_fn_broadcast: torch.Tensor | None = None
        self.mhc_pre_delayed = MHCPreDelayedOp()
        self.mhc_post = MHCPostOp()

    @staticmethod
    def _mhc(op, *args, **kwargs):
        with prof.section("mhc"):
            return op(*args, **kwargs)

    def _pre_args(self, fn, scale, base):
        return (fn, scale, base, self.rms_norm_eps, self.hc_eps, self.hc_eps,
                self.hc_post_alpha, self.hc_sinkhorn_iters)

    def forward(self, x, positions, input_ids, pre_mix=None, post_mix=None,
                res_mix=None, residual=None, engram_hashes=None, engram_mask=None):
        mhc_pre = self.mhc_pre_delayed
        attn_args = self._pre_args(self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        if residual is None:
            if x.dim() == 2:
                assert self.hc_attn_fn_broadcast is not None
                residual = x.unsqueeze(1).expand(-1, self.hc_mult, -1).contiguous()
                residual, post_mix, res_mix, x, attn_pre = self._mhc(mhc_pre,
                    residual,
                    self.hc_attn_fn_broadcast,
                    *attn_args[1:],
                    x=x,
                )
            else:
                residual = x
                residual, post_mix, res_mix, x, attn_pre = self._mhc(mhc_pre,
                    residual, *attn_args, pre_mix=pre_mix,
                    norm_weight=None, norm_eps=self.attn_norm.variance_epsilon,
                    fn_packed=None,
                )
        elif self.engram is not None and engram_hashes is not None:
            with prof.section("mhc"):
                residual = self.mhc_post(x, residual, post_mix, res_mix)
            with prof.section("engram"):
                residual = self.engram(
                    residual, engram_hashes[:, self.engram.layer_hash_index], engram_mask
                )
            residual, post_mix, res_mix, x, attn_pre = self._mhc(mhc_pre,
                residual, *attn_args, pre_mix=pre_mix,
                norm_weight=None, norm_eps=self.attn_norm.variance_epsilon,
                fn_packed=None,
            )
        else:
            residual, post_mix, res_mix, x, attn_pre = self._mhc(mhc_pre,
                residual, *attn_args, pre_mix=pre_mix, sublayer_out=x,
                post_layer_mix=post_mix, comb_res_mix=res_mix,
                norm_weight=None, norm_eps=self.attn_norm.variance_epsilon,
                fn_packed=None,
            )
        with prof.section("attention"):
            x = self.attn_norm(x)
            x = self.attn(positions, x, None)
        if _DEBUG:
            torch.accelerator.synchronize()
            logger.info("DS41DBG   after attn, host avail %d MiB", _avail_mib())
        if self.attn.wo_b.reduce_results is False:
            with prof.section("allreduce"):
                x = tensor_model_parallel_all_reduce(x)
        ffn_args = self._pre_args(self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        residual, post_mix, res_mix, x, ffn_pre = self._mhc(mhc_pre,
            residual, *ffn_args, pre_mix=attn_pre, sublayer_out=x,
            post_layer_mix=post_mix, comb_res_mix=res_mix,
            norm_weight=None, norm_eps=self.ffn_norm.variance_epsilon, fn_packed=None,
        )
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        return x, residual, post_mix, res_mix, ffn_pre


class DeepseekV4Model(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.topk_indices_buffer = torch.empty(max_tokens, config.index_topk, dtype=torch.int32)
        cs = getattr(config, "candidate_source_layer_id", -1)
        cb = getattr(config, "candidate_topk_blocks", 0)
        self.candidate_block_buffer = (
            torch.empty(max_tokens, cb, dtype=torch.int32) if cs >= 0 and cb > 0 else None
        )
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size, config.hidden_size,
                quant_config=self.quant_config, prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.engram_layout = EngramLayout.from_config(config)
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV4DecoderLayer(
                vllm_config, prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
                candidate_block_buffer=self.candidate_block_buffer,
                engram_layout=self.engram_layout,
            ),
            prefix=f"{prefix}.layers",
        )
        self.engram_hash: NgramHashState | None = None
        self.engram_swa_prefix: str | None = None
        if self.engram_layout is not None and any(
            isinstance(layer, DeepseekV4DecoderLayer) and layer.engram is not None
            for layer in islice(self.layers, self.start_layer, self.end_layer)
        ):
            first = next(iter(islice(self.layers, self.start_layer, self.end_layer)))
            swa = first.attn.swa_cache_layer
            self.engram_hash = NgramHashState(vllm_config, self.engram_layout, swa)
            self.engram_swa_prefix = swa.prefix
        self.norm = (
            RMSNorm(config.hidden_size, self.rms_norm_eps)
            if get_pp_group().is_last_rank else PPMissingLayer()
        )
        self.mhc_post = MHCPostOp()
        self.decoder_replay_layers = None

    def embed_input_ids(self, input_ids):
        return self.embed_tokens(input_ids)

    def make_empty_intermediate_tensors(self, batch_size, dtype, device):
        return IntermediateTensors({
            "hidden_states": torch.zeros(
                (batch_size, self.hc_mult, self.config.hidden_size), dtype=dtype, device=device
            ),
            "pre_mix": torch.zeros((batch_size, self.hc_mult), dtype=torch.float32, device=device),
        })

    def forward(self, input_ids, positions, intermediate_tensors,
                inputs_embeds=None, lookback_token_ids=None):
        if _DEBUG:
            md = get_forward_context().attn_metadata if is_forward_context_available() else None
            logger.info("DS41DBG enter layers %d-%d md=%s ids=%s lookback=%s", self.start_layer,
                        self.end_layer - 1, type(md).__name__,
                        None if input_ids is None else tuple(input_ids.shape),
                        None if lookback_token_ids is None else tuple(lookback_token_ids.shape))
        prof.begin_step(positions.shape[0])
        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
            )
        else:
            hidden_states = intermediate_tensors["hidden_states"]

        engram_hashes = engram_mask = None
        _ps = prof.section("engram_hash")
        _ps.__enter__()
        if self.engram_hash is not None and input_ids is not None and is_forward_context_available():
            attn_metadata = get_forward_context().attn_metadata
            if isinstance(attn_metadata, dict) and self.engram_hash.ensure_cache():
                swa_md = attn_metadata[self.engram_swa_prefix]
                image_mask = image_sentinel_mask(input_ids)
                engram_mask = ~image_mask
                if lookback_token_ids is None:
                    raise NotImplementedError("engram needs `lookback_token_ids`")
                engram_hashes = self.engram_hash(
                    input_ids, positions, swa_md.query_start_loc, image_mask,
                    lookback_token_ids, image_sentinel_mask(lookback_token_ids),
                )
            else:
                engram_hashes, engram_mask = self.engram_hash.dummy_hashes(input_ids)
            if engram_hashes is not None:
                gathered = gather_engram_hashes(engram_hashes, dp_shared_memory=False)
                for layer in islice(self.layers, self.start_layer, self.end_layer):
                    if getattr(layer, "engram", None) is not None:
                        layer.engram.prepare_embeddings(
                            gathered[:, layer.engram.layer_hash_index]
                        )

        _ps.__exit__(None, None, None)
        if _DEBUG:
            logger.info("DS41DBG engram step done (hashes=%s)", engram_hashes is not None)
        residual = post_mix = res_mix = None
        pre_mix = None if get_pp_group().is_first_rank else intermediate_tensors["pre_mix"]
        ran = False
        dbg = _DEBUG
        if dbg:
            logger.info("DS41DBG forward layers %d-%d tokens %d", self.start_layer,
                        self.end_layer - 1, positions.shape[0])
        for idx in range(self.start_layer, self.end_layer):
            ran = True
            if dbg:
                torch.accelerator.synchronize()
                logger.info("DS41DBG layer %d start, host avail %d MiB", idx, _avail_mib())
            hidden_states, residual, post_mix, res_mix, pre_mix = self.layers[idx](
                hidden_states, positions, input_ids, pre_mix, post_mix, res_mix,
                residual, engram_hashes, engram_mask,
            )
        if ran:
            with prof.section("mhc"):
                hidden_states = self.mhc_post(hidden_states, residual, post_mix, res_mix)
        if dbg:
            torch.accelerator.synchronize()
            logger.info("DS41DBG layers done")
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "pre_mix": pre_mix})
        hidden_states = torch.ops.vllm.hc_collapse_triton(hidden_states, pre_mix)
        hidden_states = self.norm(hidden_states)
        prof.end_step()
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked = [
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
            ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
            ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
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
            if name.startswith(("vision.", "aligner.", "image_")):
                continue
            # Routed experts and Engram tables are read from disk at run time.
            if (".experts." in name and ".shared_experts." not in name) or (
                ".engram.embed_tokens." in name or ".engram.embed." in name
            ) or name.endswith(".bias_vl"):
                continue
            for pname, wname, sid in stacked:
                if wname not in name or ".experts." in name and ".shared_experts." not in name:
                    continue
                n2 = resolve(name.replace(wname, pname))
                if is_pp_missing_parameter(n2, self):
                    break
                params[n2].weight_loader(params[n2], w, sid)
                loaded.add(n2)
                break
            else:
                if is_pp_missing_parameter(name, self):
                    continue
                n2 = resolve(name)
                if n2 not in params:
                    logger.warning_once("DeepSeek V4.1 XPU: no parameter for %s", name)
                    continue
                param = params[n2]
                getattr(param, "weight_loader", default_weight_loader)(param, w)
                loaded.add(n2)
        return loaded

    def finalize_mhc_broadcast_weights(self) -> None:
        if not get_pp_group().is_first_rank or self.start_layer >= self.end_layer:
            return
        layer = self.layers[self.start_layer]
        layer.hc_attn_fn_broadcast = (
            layer.hc_attn_fn.detach().view(-1, layer.hc_mult, layer.hidden_size).sum(dim=1)
        )


def _linear_scale_param_name(vllm_config: VllmConfig, expert_dtype: str) -> str:
    # XPU serves the 32x32 block-FP8 linears with Fp8LinearMethod (weight_scale_inv).
    return "weight_scale_inv"


def _make_deepseek_v4_weights_mapper(expert_dtype: str,
                                     linear_scale_name: str = "weight_scale_inv") -> WeightsMapper:
    return WeightsMapper(
        orig_to_new_prefix={
            "layers.": "model.layers.",
            "embed.": "model.embed.",
            "norm.": "model.norm.",
            "mtp.": "model.mtp.",
        },
        orig_to_new_regex={
            re.compile(r"(\.experts\.\d+\.w[123])\.scale$"): r"\1.weight_scale",
            re.compile(r"(engram\.embed)\.scale$"): r"\1_tokens.weight_scale_inv",
            re.compile(r"\.scale$"): f".{linear_scale_name}",
        },
        orig_to_new_suffix={
            "head.weight": "lm_head.weight",
            "embed.weight": "embed_tokens.weight",
            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
        },
        orig_to_new_substr={
            ".shared_experts.w2": ".shared_experts.down_proj",
            "mtp.": None,
            "vision.": None,
            "aligner.": None,
            "image_": None,
        },
    )


class DeepseekV41LLMForCausalLM(nn.Module, SupportsPP):
    model_cls = DeepseekV4Model
    hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4")
    packed_modules_mapping = {
        "gate_up_proj": ["w1", "w3"],
        "fused_wqa_wkv": ["wq_a", "wkv"],
        "fused_wkv_wgate": ["wkv", "wgate"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.model = self.model_cls(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size, config.hidden_size, prefix=maybe_prefix(prefix, "lm_head")
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids):
        return self.model.embed_input_ids(input_ids)

    def compute_logits(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states)

    @staticmethod
    def get_model_state_cls():
        from ..nvidia.model_state import DeepseekV41ModelState

        return DeepseekV41ModelState

    @property
    def token_lookback_depth(self) -> int:
        eh = self.model.engram_hash
        return eh.lookback_depth if eh is not None else 0

    @property
    def decoder_replay_layers(self):
        return None

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, lookback_token_ids=None):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds,
                          lookback_token_ids=lookback_token_ids)

    def get_mtp_target_hidden_states(self):
        return None

    def load_weights(self, weights):
        return AutoWeightsLoader(self).load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def process_weights_after_loading(self) -> None:
        self.model.finalize_mhc_broadcast_weights()


__all__ = [
    "DeepseekV41LLMForCausalLM",
    "_linear_scale_param_name",
    "_make_deepseek_v4_weights_mapper",
    "cdiv",
]
