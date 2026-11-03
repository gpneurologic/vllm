# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3_5 MTP model."""

from collections.abc import Iterable

import torch
from torch import nn

from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import LocalArgmaxMixin
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5Model,
    Qwen3_5RMSNorm,
)
from vllm.model_executor.models.qwen3_next import (
    QwenNextMixtureOfExperts,
    _all_gather_hidden_and_residual,
    _is_shared_expert_fse_compatible,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5TextConfig
from vllm.transformers_utils.configs.qwen3_5_moe import Qwen3_5MoeTextConfig

from .interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    _require_is_multimodal,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    _merge_multimodal_embeddings,
    make_empty_intermediate_tensors_factory,
    maybe_fuse_shared_experts,
    maybe_prefix,
)

logger = init_logger(__name__)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "hidden_states": 0,
    }
)
class Qwen3_5MultiTokenPredictor(nn.Module):
    hf_to_vllm_mapper = Qwen3_5Model.hf_to_vllm_mapper

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        model_config = vllm_config.model_config
        quant_config = vllm_config.quant_config

        config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig = model_config.hf_text_config

        self.config = config

        self.vocab_size = config.vocab_size

        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "mtp_num_hidden_layers", 1)

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        # Workaround: mtp.fc is stored as BF16 in NVFP4 checkpoints but is
        # missing from hf_quant_config.json exclude_modules. Force unquantized.
        # Ref: https://github.com/vllm-project/vllm/pull/38650
        # Ref: https://github.com/NVIDIA/Model-Optimizer/pull/1124
        fc_quant = (
            None
            if (quant_config and quant_config.get_name() == "modelopt_fp4")
            else quant_config
        )
        if self.config.hidden_size % get_tensor_model_parallel_world_size() == 0:
            self.fc = ColumnParallelLinear(
                self.config.hidden_size * 2,
                self.config.hidden_size,
                gather_output=True,
                bias=False,
                return_bias=False,
                quant_config=fc_quant,
                prefix=f"{prefix}.fc",
            )
        else:
            # The output can't be split evenly (e.g. hidden 5120 at TP=3, with a
            # checkpoint padded for TP=3); the gathered result is the same, so
            # keep this one small layer whole on every card.
            self.fc = ReplicatedLinear(
                self.config.hidden_size * 2,
                self.config.hidden_size,
                bias=False,
                return_bias=False,
                quant_config=fc_quant,
                prefix=f"{prefix}.fc",
            )

        self.layers = torch.nn.ModuleList(
            Qwen3_5DecoderLayer(
                vllm_config,
                layer_type="full_attention",
                prefix=f"{prefix}.layers.{idx}",
            )
            for idx in range(self.num_mtp_layers)
        )

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                inputs_embeds = self.embed_input_ids(input_ids)
            assert hidden_states.shape[-1] == inputs_embeds.shape[-1]
            inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
            hidden_states = self.pre_fc_norm_hidden(hidden_states)
            hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
            hidden_states = self.fc(hidden_states)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[current_step_idx]
        hidden_states, residual = mtp_layer(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if mtp_layer.use_attn_reduce_scatter_for_moe:
            hidden_states, residual = _all_gather_hidden_and_residual(
                hidden_states,
                residual,
                positions.shape[-1],
                self.config.hidden_size,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = maybe_fuse_shared_experts(
            weights,
            enabled=rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
            and _is_shared_expert_fse_compatible(
                get_current_vllm_config().quant_config
            ),
            n_routed_experts=getattr(self.config, "num_experts", 0),
            n_shared_experts=1,
            ckpt_prefix="mlp.shared_expert",
        )
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "hidden_states": 0,
    }
)
class Qwen3_5MTP(LocalArgmaxMixin, nn.Module, SupportsMultiModal):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        cache_config = vllm_config.cache_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3_5MTP currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )

        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.model = Qwen3_5MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "mtp")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self._draft_vocab_setup()

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.model.embed_input_ids,
            is_multimodal=is_multimodal,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        is_multimodal = _require_is_multimodal(is_multimodal)

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, hidden_states, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        if self._draft_weight is not None:
            return self._draft_logits(hidden_states)
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def remap_weight_names(weights):
            for name, weight in weights:
                if name.startswith("mtp."):
                    name = name.replace("mtp.", "model.")
                elif any(key in name for key in ["embed_tokens", "lm_head"]):
                    if "embed_tokens" in name:
                        name = name.replace("language_model.", "")
                else:
                    continue
                yield name, weight

        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(self._draft_vocab_capture(remap_weight_names(weights)))
        import os

        if os.environ.get("VLLM_MTP_FP8", "0") == "1":
            from vllm.model_executor.layers.quantization.xpu_fp8_weight_only import (
                convert_unquantized_linears_fp8,
            )

            convert_unquantized_linears_fp8(self.model, "MTP draft layer")
        return loaded

    # ---- Private reduced-vocabulary draft head (VLLM_MTP_DRAFT_VOCAB=<ids.npy>) ----
    # The draft normally shares the target's lm_head, so every drafted token reads the
    # whole vocabulary (248k rows). With a list of frequent token ids, the draft scores
    # only those rows; every other id gets -inf. The target still verifies every drafted
    # token, so the output is unchanged; only the acceptance rate can move. The id list
    # is split evenly across the TP ranks (not by vocab shard, since frequent ids
    # cluster at the start of the vocabulary), and each rank keeps its rows, taken from
    # lm_head.weight as the checkpoint streams past.
    def _draft_vocab_setup(self) -> None:
        import os

        import numpy as np

        from vllm.distributed import get_tensor_model_parallel_rank

        self._draft_weight = None
        path = os.environ.get("VLLM_MTP_DRAFT_VOCAB")
        if not path or self.config.tie_word_embeddings:
            return
        if path == "default":  # the bundled list for the Qwen3.5+ tokenizer (see data/NOTICE)
            path = os.path.join(os.path.dirname(__file__), "data", "qwen3_5_draft_vocab_65536.npy")
        tp = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_rank()
        ids = np.unique(np.load(path).astype(np.int64))
        ids = ids[(ids >= 0) & (ids < self.config.vocab_size)]
        per = -(-len(ids) // tp)
        per = -(-per // 16) * 16
        ids = np.concatenate([ids, np.repeat(ids[-1], per * tp - len(ids))])
        self._draft_ids_np = ids[rank * per:(rank + 1) * per]
        self._draft_ids_all = torch.from_numpy(ids)
        self._draft_ids_local = torch.from_numpy(self._draft_ids_np)
        self._draft_tp = tp
        logger.info("MTP draft vocabulary: %d ids from %s, %d rows per rank",
                    len(np.unique(ids)), path, per)

    def _draft_vocab_capture(self, weights):
        if not hasattr(self, "_draft_ids_np"):
            yield from weights
            return
        dtype = self.vllm_config.model_config.dtype
        for name, weight in weights:
            if name == "lm_head.weight":
                dev = self.logits_processor_device()
                rows = weight.index_select(0, self._draft_ids_local.to(weight.device))
                self._draft_weight = rows.to(device=dev, dtype=dtype).contiguous()
                import os

                if os.environ.get("VLLM_MTP_DRAFT_HEAD", "").lower() == "fp8":
                    from vllm.model_executor.layers.quantization.xpu_fp8_weight_only import (
                        quantize_rows_fp8,
                    )

                    self._draft_weight, self._draft_scale = quantize_rows_fp8(self._draft_weight)
                self._draft_ids_all = self._draft_ids_all.to(dev)
                self._draft_ids_local = self._draft_ids_local.to(dev)
                logger.info("MTP draft head: %s rows x %d per rank (%.0f MiB) instead of the shared "
                            "lm_head", self._draft_weight.shape[0], self._draft_weight.shape[1],
                            self._draft_weight.numel() * self._draft_weight.element_size() / 2**20)
            yield name, weight

    def logits_processor_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _draft_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        w = self._draft_weight
        scale = getattr(self, "_draft_scale", None)
        if scale is not None:
            from vllm.model_executor.layers.quantization.xpu_fp8_weight_only import fp8_linear

            return fp8_linear(hidden_states, w, scale)
        return torch.nn.functional.linear(hidden_states.to(w.dtype), w)

    def _draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from vllm.distributed import tensor_model_parallel_all_gather

        local = self._draft_local(hidden_states)
        part = tensor_model_parallel_all_gather(local, dim=-1) if self._draft_tp > 1 else local
        full = part.new_full((part.shape[0], self.config.vocab_size), float("-inf"))
        full.index_copy_(1, self._draft_ids_all, part)
        return full

    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._draft_weight is None:
            return LocalArgmaxMixin.get_top_tokens(self, hidden_states)
        from vllm.distributed import tensor_model_parallel_all_gather

        local = self._draft_local(hidden_states)
        vals, idx = local.max(dim=-1)
        gid = self._draft_ids_local[idx]
        if self._draft_tp == 1:
            return gid
        pair = torch.stack([vals.float(), gid.float()], dim=-1)          # [T, 2]
        allp = tensor_model_parallel_all_gather(pair, dim=-1)            # [T, 2*tp]
        allp = allp.view(allp.shape[0], self._draft_tp, 2)
        best = allp[..., 0].argmax(dim=-1, keepdim=True)
        return allp[..., 1].gather(1, best).squeeze(1).long()


class Qwen3_5MoeMTP(Qwen3_5MTP, QwenNextMixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.set_moe_parameters()
