# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen4Exp MTP (Multi-Token Predictor) model.

The MTP draft model reuses the Qwen4Exp backbone (PLE/HC/MoE) but:
  - drops all multi-modal handling (text-only),
  - forces PLE off while keeping the main model's HC stream count,
  - fuses the backbone hidden and the new-token embedding via
    `residual_linear_shared` (fc_embedding + shared fc_hidden) instead of
    the `Linear(2H, H)` + repeat used by other MTP variants,
  - emits TWO hidden streams per step (scheme A): a single stream [T, H]
    (final-mixer collapsed, fed to the LM head) and a pre-final-mixer
    multi stream [T, hc_count*H] (fed to the next draft step).
"""

import os
from collections.abc import Iterable

import numpy as np
import regex as re
import torch
from torch import nn
from transformers import Qwen4ExpTextConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, replace, set_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.utils import configure_quant_config
from vllm.model_executor.models.interfaces import SupportsHCSP, SupportsPP
from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    get_draft_quant_config,
    get_fused_shared_expert_layers,
    make_empty_intermediate_tensors_factory,
    maybe_fuse_shared_experts,
    maybe_prefix,
)
from vllm.models.common.ops.sequence_parallel import sp_all_gather, sp_shard
from vllm.sequence import IntermediateTensors

from ..common.hyperconnection import is_hc_sequence_parallel_enabled
from .hyperconnection import GatedResidual, HyperConnectionConfig
from .low_latency_gemm import enable_qwen4_exp_low_latency_gemm
from .model import (
    _EXTRA_WEIGHTS_MAPPER,
    _dequantize_gptq_lm_head,
    _dequantize_hybrid_block_fp8,
    is_fp8_quant_config,
    _QWEN4_EXP_IGNORED_MISSING_SUFFIXES,
    Qwen4ExpDecoderLayer,
    Qwen4ExpMixtureOfExperts,
    Qwen4ExpSparseMoeBlock,
)

logger = init_logger(__name__)

# The bundled frequent-token list for the Qwen3.5+ tokenizer (see data/NOTICE).
_DEFAULT_DRAFT_VOCAB = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "model_executor",
    "models",
    "data",
    "qwen3_5_draft_vocab_65536.npy",
)


def _remap_ignored_layers(
    ignored_layers: list[str],
    mtp_start_layer_idx: int,
) -> list[str]:
    remapped: list[str] = []
    for name in ignored_layers:
        if name.startswith("mtp."):
            new_name = re.sub(
                r"(?<=\.layers\.)\d+",
                lambda m: str(mtp_start_layer_idx + int(m.group(0))),
                name,
            )
            remapped.append(new_name)
        else:
            remapped.append(name)
    return remapped


def _remap_quantized_layers(
    quantized_layers: dict[str, dict],
    mtp_start_layer_idx: int,
) -> dict[str, dict]:
    """Map checkpoint MTP layer indices to standalone draft indices."""
    return {
        _remap_ignored_layers([name], mtp_start_layer_idx)[0]: layer_info
        for name, layer_info in quantized_layers.items()
    }


def _remap_mtp_weight_name(name: str) -> str | None:
    """Map Qwen4Exp checkpoint paths into the standalone draft model."""
    for checkpoint_prefix in (
        "model.language_model.",
        "language_model.",
    ):
        if name.startswith(checkpoint_prefix):
            name = name.removeprefix(checkpoint_prefix)
            break

    if name.startswith("embed_tokens."):
        name = f"model.{name}"
    if name.startswith("model.mtp."):
        name = name.removeprefix("model.")
    if name.startswith("mtp.shared_head.head."):
        return name.replace("mtp.shared_head.head.", "lm_head.", 1)
    if name.startswith("model.shared_head.head."):
        return name.replace("model.shared_head.head.", "lm_head.", 1)
    if name.startswith("shared_head.head."):
        return name.replace("shared_head.head.", "lm_head.", 1)
    if name.startswith("model.lm_head."):
        return name.removeprefix("model.")
    if name.startswith("mtp."):
        return name.replace("mtp.", "model.", 1)
    if name.startswith("model.embed_tokens.") or name.startswith("lm_head."):
        return name
    return None


def _make_draft_vllm_config(
    vllm_config: VllmConfig,
    mtp_start_layer_idx: int,
    num_mtp_layers: int,
) -> VllmConfig:
    """Ensure that the draft model config is set in the vLLM config."""
    speculative_config = vllm_config.speculative_config
    if speculative_config is None or speculative_config.draft_model_config is None:
        raise ValueError("speculative_config.draft_model_config must be set")

    draft_quant_config = get_draft_quant_config(vllm_config)

    # inject packed and ignored modules to the quantization config of draft model
    if draft_quant_config is not None:
        configure_quant_config(draft_quant_config, Qwen4ExpMTP)
        ignored_layers = getattr(draft_quant_config, "ignored_layers", None)
        if ignored_layers:
            setattr(  # noqa: B010
                draft_quant_config,
                "ignored_layers",
                _remap_ignored_layers(ignored_layers, mtp_start_layer_idx),
            )
        exclude_modules = getattr(draft_quant_config, "exclude_modules", None)
        if exclude_modules:
            setattr(  # noqa: B010
                draft_quant_config,
                "exclude_modules",
                _remap_ignored_layers(exclude_modules, mtp_start_layer_idx),
            )
        quantized_layers = getattr(draft_quant_config, "quantized_layers", None)
        if quantized_layers:
            setattr(  # noqa: B010
                draft_quant_config,
                "quantized_layers",
                _remap_quantized_layers(quantized_layers, mtp_start_layer_idx),
            )
        # AutoGPTQ lists quantized modules by checkpoint name (`mtp.layers.0...`);
        # the standalone draft model numbers its layer after the main ones.
        gptq_modules = getattr(draft_quant_config, "modules_in_block_to_quantize", None)
        if gptq_modules and not getattr(draft_quant_config, "_mtp_remapped", False):
            setattr(  # noqa: B010
                draft_quant_config,
                "modules_in_block_to_quantize",
                _remap_ignored_layers(gptq_modules, mtp_start_layer_idx),
            )
            setattr(draft_quant_config, "_mtp_remapped", True)  # noqa: B010
        if draft_quant_config.get_name() == "quark":
            draft_quant_config.apply_vllm_mapper(
                WeightsMapper(
                    orig_to_new_prefix={
                        f"mtp.layers.{i}.": f"mtp.layers.{mtp_start_layer_idx + i}."
                        for i in range(num_mtp_layers)
                    }
                )
            )

    draft_vllm_config = replace(
        vllm_config,
        model_config=speculative_config.draft_model_config,
    )
    # VllmConfig post-init derives the target quant config, so restore the
    # independently resolved draft quant config after replacement.
    draft_vllm_config.quant_config = draft_quant_config
    return draft_vllm_config


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "hidden_states": 0,
    }
)
class Qwen4ExpMultiTokenPredictor(nn.Module):
    hf_to_vllm_mapper = Qwen3_5Model.hf_to_vllm_mapper | _EXTRA_WEIGHTS_MAPPER

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        model_config = vllm_config.model_config
        config: Qwen4ExpTextConfig = model_config.hf_text_config

        self.config = config
        # MoE SP also enables HC SP, while HC SP can be enabled independently.
        self.use_hc_sequence_parallel = is_hc_sequence_parallel_enabled(vllm_config)
        self.vocab_size = config.vocab_size

        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "mtp_num_hidden_layers", 1)

        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count

        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, self.hidden_size)
        draft_vllm_config = _make_draft_vllm_config(
            vllm_config,
            self.mtp_start_layer_idx,
            self.num_mtp_layers,
        )
        with set_current_vllm_config(draft_vllm_config, prefix=prefix):
            # residual_linear_shared fusion: fc_embedding projects the token
            # embedding, fc_hidden (shared across HC branches) projects the
            # backbone hidden; the embedding is added as a residual to every
            # branch (see mtp_residual_linear_shared.md).
            # Both gather their output, so every rank holds the full result
            # anyway: keep them whole on every rank when the hidden size does
            # not divide by the TP size (2560 at TP=3).
            from vllm.distributed import get_tensor_model_parallel_world_size

            fc_split = self.hidden_size % get_tensor_model_parallel_world_size() == 0
            self.fc_embedding = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                gather_output=fc_split,
                disable_tp=not fc_split,
                bias=False,
                return_bias=False,
                quant_config=draft_vllm_config.quant_config,
                prefix=f"{prefix}.fc_embedding",
            )
            self.fc_hidden = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                gather_output=fc_split,
                disable_tp=not fc_split,
                bias=False,
                return_bias=False,
                quant_config=draft_vllm_config.quant_config,
                prefix=f"{prefix}.fc_hidden",
            )
            self.layers = nn.ModuleList(
                Qwen4ExpDecoderLayer(
                    draft_vllm_config,
                    layer_type="qwen_sparse_attention",
                    prefix=f"{prefix}.layers.{self.mtp_start_layer_idx + idx}",
                )
                for idx in range(self.num_mtp_layers)
            )
        self.fused_shared_expert_layers = get_fused_shared_expert_layers(
            self,
            Qwen4ExpSparseMoeBlock,
            enable_eplb=vllm_config.parallel_config.enable_eplb,
        )

        self.pre_fc_norm_embedding = GemmaRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = GemmaRMSNorm(
            self.hidden_size * self.hc_count, eps=config.rms_norm_eps
        )
        # HC final mixer collapses the multi stream into [T, H] for the LM head.
        hc_config = HyperConnectionConfig(
            hc_count=config.hc_count,
            hidden_size=config.hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
            hc_per_branch_norm=True,
        )
        self.hyper_connection_mixer = GatedResidual(
            hc_config,
            use_combine=False,
            prefix=maybe_prefix(prefix, "hyper_connection_mixer"),
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.hidden_size * self.hc_count
        )

    def _iter_qsa_attentions(self):
        """Yield MTP attention modules that own a QSA indexer."""
        for layer in self.layers:
            attention = getattr(layer, "self_attn", None)
            if (
                attention is not None
                and getattr(attention, "indexer", None) is not None
            ):
                yield attention

    def set_skip_topk(self, skip: bool) -> None:
        """Select on MTP step 0 and reuse its QSA indices on later steps."""
        for attention in self._iter_qsa_attentions():
            attention.indexer.skip_topk = skip

    def compact_topk_indices(self, row_indices: torch.Tensor) -> None:
        """Keep each request's target-aligned step-0 sparse-index row."""
        num_rows = row_indices.numel()
        for attention in self._iter_qsa_attentions():
            buffer = attention.topk_indices_buffer
            selected = buffer.index_select(0, row_indices)
            buffer[:num_rows].copy_(selected)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | IntermediateTensors:
        hc_count = self.hc_count
        hidden_size = self.hidden_size

        assert hidden_states is not None
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed_input_ids(input_ids)
        # Embedding branch: pre-norm -> fc_embedding -> [T, H].
        inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
        inputs_embeds = self.fc_embedding(inputs_embeds)

        # Backbone hidden is multi-stream [T, hc_count*H] (scheme A:
        # the main model truly emits the pre-final-mixer multi stream
        # on the first step; subsequent steps reuse the prior draft
        # step's multi stream).
        num_tokens = hidden_states.shape[0]
        hidden_states = hidden_states.view(num_tokens, hc_count, hidden_size)
        hidden_states = self.pre_fc_norm_hidden(hidden_states.flatten(-2)).view(
            num_tokens, hc_count, hidden_size
        )
        hidden_states = self.fc_hidden(hidden_states)
        # Add the embedding residual to every branch, then fold back
        # to [T, hc_count*H] (HC outer, HS inner) for the HC decoder.
        hidden_states = inputs_embeds.unsqueeze(-2) + hidden_states
        hidden_states = hidden_states.flatten(-2)

        current_step_idx = spec_step_idx % self.num_mtp_layers
        full_num_tokens = positions.shape[-1]
        if self.use_hc_sequence_parallel:
            hidden_states = sp_shard(hidden_states)
        layer = self.layers[current_step_idx]
        hidden_states, block_output, injection = layer(
            hidden_states=hidden_states,
            prev_block_output=None,
            prev_injection=None,
            positions=positions,
            input_ids=None,
            query_start_loc=None,
            ngram_context=None,
        )

        # Finalize. Keep both:
        #   (A) sample_hidden_states [T, H]  -> single stream for the LM head
        #   (B) multi_hidden [T, hc_count*H] -> pre-final-mixer multi stream
        #       for the next draft step (zero extra compute, just kept).
        multi_hidden, sample_hidden_states, _ = (
            self.hyper_connection_mixer.combine_and_mix(
                hidden_states, block_output, injection
            )
        )
        if self.use_hc_sequence_parallel:
            # Gather both outputs together, then restore their contiguous layouts.
            packed_hidden_states = torch.cat(
                [sample_hidden_states, multi_hidden], dim=-1
            )
            packed_hidden_states = sp_all_gather(packed_hidden_states)[:full_num_tokens]
            sample_hidden_states, multi_hidden = packed_hidden_states.split(
                [hidden_size, hc_count * hidden_size], dim=-1
            )
            sample_hidden_states = sample_hidden_states.contiguous()
            multi_hidden = multi_hidden.contiguous()
        return sample_hidden_states, multi_hidden

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = maybe_fuse_shared_experts(
            weights,
            fused_shared_expert_layers=self.fused_shared_expert_layers,
            n_routed_experts=getattr(self.config, "num_experts", 0) or 0,
            n_shared_experts=1,
            ckpt_prefix="mlp.shared_expert",
        )
        mapper = self.hf_to_vllm_mapper | WeightsMapper(
            orig_to_new_substr={"hyper_connection_mixer.block_inject_weight": None}
        )
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        )
        return loader.load_weights(weights, mapper=mapper)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "hidden_states": 0,
    }
)
class Qwen4ExpMTP(nn.Module, SupportsHCSP, SupportsPP, Qwen4ExpMixtureOfExperts):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
        "input_mix_weight_down_block_inject": [
            "input_mix_weight_down",
            "block_inject_weight",
            "_input_mix_padding",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config: Qwen4ExpTextConfig = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.model = Qwen4ExpMultiTokenPredictor(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "mtp"),
        )

        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.set_moe_parameters(self.model.layers)
        enable_qwen4_exp_low_latency_gemm(self, vllm_config.model_config.dtype)
        self._draft_vocab_setup()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | IntermediateTensors:
        return self.model(
            input_ids,
            positions,
            hidden_states,
            intermediate_tensors,
            inputs_embeds,
            spec_step_idx=spec_step_idx,
        )

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor | None:
        if self._draft_weight is not None:
            return self._draft_logits(hidden_states)
        return self.logits_processor(self.lm_head, hidden_states)

    # ---- Reduced-vocabulary draft head (VLLM_MTP_DRAFT_VOCAB=default|<ids.npy>) ----
    # Ported from the dense branch's qwen3_5_mtp.py. The draft normally reads the
    # whole 248k-row lm_head for every drafted token (4 times per step with MTP 4).
    # With a list of frequent ids it scores only those rows; every other id gets
    # -inf. The target still verifies every draft, so the output is unchanged; only
    # the acceptance rate can move. The ids are split evenly across TP ranks (not by
    # vocab shard: frequent ids cluster at the start of the vocabulary), and each
    # rank keeps its rows, taken from the full lm_head.weight while it loads (this
    # checkpoint's GPTQ head is dequantized to BF16 on the way in).
    # VLLM_MTP_DRAFT_HEAD=fp8 keeps those rows in FP8 (per-row scale).
    def _draft_vocab_setup(self) -> None:
        self._draft_weight: torch.Tensor | None = None
        self._draft_scale: torch.Tensor | None = None
        path = os.environ.get("VLLM_MTP_DRAFT_VOCAB")
        if not path or self.config.tie_word_embeddings:
            return
        if path == "default":
            path = os.path.normpath(_DEFAULT_DRAFT_VOCAB)
        tp = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_rank()
        ids = np.unique(np.load(path).astype(np.int64))
        ids = ids[(ids >= 0) & (ids < self.config.vocab_size)]
        per = -(-len(ids) // tp)
        per = -(-per // 16) * 16
        ids = np.concatenate([ids, np.repeat(ids[-1], per * tp - len(ids))])
        self._draft_ids_np = ids[rank * per : (rank + 1) * per]
        self._draft_ids_all = torch.from_numpy(ids)
        self._draft_ids_local = torch.from_numpy(self._draft_ids_np)
        self._draft_tp = tp
        logger.info(
            "MTP draft vocabulary: %d ids from %s, %d rows per rank",
            len(np.unique(ids)),
            path,
            per,
        )

    def _draft_vocab_capture(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        if not hasattr(self, "_draft_ids_np"):
            yield from weights
            return
        dtype = self.vllm_config.model_config.dtype
        for name, weight in weights:
            if name == "lm_head.weight" and self._draft_weight is None:
                dev = next(self.model.parameters()).device
                rows = weight.index_select(0, self._draft_ids_local.to(weight.device))
                w = rows.to(device=dev, dtype=dtype).contiguous()
                if os.environ.get("VLLM_MTP_DRAFT_HEAD", "").lower() == "fp8":
                    from vllm.model_executor.layers.quantization.xpu_fp8_weight_only import (  # noqa: E501
                        quantize_rows_fp8,
                    )

                    w, self._draft_scale = quantize_rows_fp8(w)
                self._draft_weight = w
                self._draft_ids_all = self._draft_ids_all.to(dev)
                self._draft_ids_local = self._draft_ids_local.to(dev)
                logger.info(
                    "MTP draft head: %d rows x %d per rank (%s, %.0f MiB) instead of "
                    "the full lm_head",
                    w.shape[0],
                    w.shape[1],
                    "fp8" if self._draft_scale is not None else str(w.dtype),
                    w.numel() * w.element_size() / 2**20,
                )
            yield name, weight

    def _draft_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        w = self._draft_weight
        assert w is not None
        if self._draft_scale is not None:
            from vllm.model_executor.layers.quantization.xpu_fp8_weight_only import (
                fp8_linear,
            )

            return fp8_linear(hidden_states, w, self._draft_scale)
        return torch.nn.functional.linear(hidden_states.to(w.dtype), w)

    def _draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        local = self._draft_local(hidden_states)
        part = (
            tensor_model_parallel_all_gather(local, dim=-1)
            if self._draft_tp > 1
            else local
        )
        full = part.new_full((part.shape[0], self.config.vocab_size), float("-inf"))
        full.index_copy_(1, self._draft_ids_all, part)
        return full

    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Greedy draft token ids, used with use_local_argmax_reduction."""
        if self._draft_weight is None:
            logits = self.compute_logits(hidden_states)
            return logits.argmax(dim=-1)
        local = self._draft_local(hidden_states)
        vals, idx = local.max(dim=-1)
        gid = self._draft_ids_local[idx]
        if self._draft_tp == 1:
            return gid
        pair = torch.stack([vals.float(), gid.float()], dim=-1)  # [T, 2]
        allp = tensor_model_parallel_all_gather(pair, dim=-1)  # [T, 2*tp]
        allp = allp.view(allp.shape[0], self._draft_tp, 2)
        best = allp[..., 0].argmax(dim=-1, keepdim=True)
        return allp[..., 1].gather(1, best).squeeze(1).long()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def remap_weight_names():
            for name, weight in weights:
                remapped_name = _remap_mtp_weight_name(name)
                if remapped_name is not None:
                    yield remapped_name, weight

        mapper = WeightsMapper(
            orig_to_new_substr={"hyper_connection_mixer.block_inject_weight": None}
        )
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        )
        remapped: Iterable[tuple[str, torch.Tensor]] = remap_weight_names()
        # Hybrid AutoRound checkpoints: see the same handling in model.py.
        if not is_fp8_quant_config(self.quant_config):
            remapped = _dequantize_hybrid_block_fp8(remapped)
        lm_head = getattr(self, "lm_head", None)
        if not hasattr(lm_head, "qweight"):
            remapped = _dequantize_gptq_lm_head(
                remapped,
                group_size=getattr(self.quant_config, "group_size", 128),
                keep=lm_head is not None,
            )
        loaded = loader.load_weights(self._draft_vocab_capture(remapped), mapper=mapper)
        if os.environ.get("VLLM_MTP_FP8", "0") == "1":
            from vllm.model_executor.layers.quantization.xpu_fp8_weight_only import (
                convert_unquantized_linears_fp8,
            )

            convert_unquantized_linears_fp8(self.model, "MTP draft layer")
        return loaded


__all__ = ["Qwen4ExpMTP", "Qwen4ExpMultiTokenPredictor"]
