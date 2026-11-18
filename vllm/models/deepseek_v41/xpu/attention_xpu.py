# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU sparse-MLA attention for DeepSeek V4.1 (first bring-up version).

Built on the shared V4.1 `DeepseekV4Attention`. Differences from CUDA:
- KV records use the V4 fp8_ds_mla layout (448 fp8 NoPE + 64 bf16 RoPE), which is
  what V4.1 uses on every non-SM100 GPU, so the V4 XPU decode kernel reads them.
- Q gets GPT-J RoPE only (V4.1 norms qr before wq_b; there is no per-head Q norm,
  unlike V4, so the V4 XPU `qnorm_rope` kernel cannot be reused for Q).
- The output projection runs wo_a with a Triton FP8 GEMV in decode (fp8_gemv.py),
  and dequantizes it per call for larger batches.
"""

from typing import TYPE_CHECKING, ClassVar, cast

import torch

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.common.ops.cache_utils import (
    quantize_and_insert_k_cache,
)
from vllm.models.deepseek_v4.xpu.xpu_sparse_decode_fp8 import xpu_sparse_decode_fp8
from vllm.models.deepseek_v41.attention import DeepseekV4Attention
from vllm.models.deepseek_v41.common.ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
)
from vllm.models.deepseek_v41.sparse_mla import (
    DeepseekV4FlashMLAMetadata,
    DeepseekV4SparseMLABackend,
    DeepseekV4SparseMLAMetadataBuilder,
    DeepseekV41SparseSWAMetadataBuilder,
)
from vllm.utils.math_utils import round_up
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWABackend
from vllm.v1.attention.ops.xpu_mla_sparse import triton_bf16_mla_sparse_interface
from vllm.v1.worker.workspace import current_workspace_manager

from .fp8_gemv import fp8_grouped_gemv

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata


class DeepseekV41XPUMetadataBuilder(DeepseekV4SparseMLAMetadataBuilder):
    # Eager first: no graph support claimed.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER


class DeepseekV41XPUBackend(DeepseekV4SparseMLABackend):
    @staticmethod
    def get_name() -> str:
        return "XPU_SPARSE_DSV41"

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None):
        return [128]

    @staticmethod
    def get_builder_cls() -> type[DeepseekV41XPUMetadataBuilder]:
        return DeepseekV41XPUMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability) -> bool:
        return True


class DeepseekV41XPUSWAMetadataBuilder(DeepseekV41SparseSWAMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER


class DeepseekV41XPUSWABackend(DeepseekSparseSWABackend):
    @staticmethod
    def get_builder_cls() -> type[DeepseekV41XPUSWAMetadataBuilder]:
        return DeepseekV41XPUSWAMetadataBuilder


def _gptj_rope(x: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor,
               inverse: bool = False) -> None:
    """In-place GPT-J (interleaved pair) RoPE on the last `rope_dim` dims of
    x [T, ..., D]. cos_sin [max_pos, rope_dim]: first half cos, second half sin."""
    rope_dim = cos_sin.shape[-1]
    half = rope_dim // 2
    cs = cos_sin[positions].float()  # [T, rope_dim]
    cos, sin = cs[:, :half], cs[:, half:]
    shape = [x.shape[0]] + [1] * (x.dim() - 2) + [half]
    cos, sin = cos.view(shape), sin.view(shape)
    if inverse:
        sin = -sin
    r = x[..., -rope_dim:].float()
    even, odd = r[..., 0::2], r[..., 1::2]
    out = torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1)
    x[..., -rope_dim:] = out.flatten(-2).to(x.dtype)


def _act_fp8_roundtrip(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """Quantize to fp8 e4m3 with power-of-two (ue8m0) scales per `block` and
    back, as DeepSeek does to the attention output before wo_a."""
    xf = x.float().unflatten(-1, (-1, block))
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    m, e = torch.frexp(amax / 448.0)
    e = e - (m == 0.5).to(e.dtype)
    s = torch.ldexp(torch.ones_like(amax), e)
    q = (xf / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
    return (q * s).flatten(-2)


def _act_fp8_quant(x: torch.Tensor, block: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """x [..., K] -> (fp8 e4m3 [..., K], float32 power-of-two scales [..., K/block]);
    the same values as `_act_fp8_roundtrip`, kept quantized."""
    xf = x.float().unflatten(-1, (-1, block))
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    m, e = torch.frexp(amax / 448.0)
    e = e - (m == 0.5).to(e.dtype)
    s = torch.ldexp(torch.ones_like(amax), e)
    q = (xf / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.flatten(-2), s.squeeze(-1)


_WO_A_BMM = __import__("os").environ.get("VLLM_DS41_WO_A_BMM") == "1"
_WO_A_GEMV_MAX_TOKENS = int(__import__("os").environ.get("VLLM_DS41_WO_A_GEMV_MAX_TOKENS", "16"))


def _dequant_block_fp8(w: torch.Tensor, s: torch.Tensor, block: int = 32) -> torch.Tensor:
    n, k = w.shape
    sf = s.float() if s.dtype != torch.uint8 else torch.ldexp(
        torch.ones_like(s, dtype=torch.float32), s.to(torch.int32) - 127
    )
    sf = sf.repeat_interleave(block, 0)[:n].repeat_interleave(block, 1)[:, :k]
    return w.float() * sf


class DeepseekV41XPUAttention(DeepseekV4Attention):
    backend_cls = DeepseekV41XPUBackend
    swa_backend_cls = DeepseekV41XPUSWABackend
    use_fp8_ds_mla_layout = True

    def __init__(self, *args, **kwargs) -> None:
        # torch.cuda.Event() raises on XPU; the base class creates some.
        orig = torch.cuda.Event
        torch.cuda.Event = torch.xpu.Event  # type: ignore[misc]
        try:
            super().__init__(*args, **kwargs)
        finally:
            torch.cuda.Event = orig  # type: ignore[misc]
        self._wo_a_bf16: torch.Tensor | None = None
        self._wo_a_sf: torch.Tensor | None = None

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    # ---------------------------------------------------------------- KV insert
    def _fused_qnorm_rope_kv_insert(self, q, kv, positions, attn_metadata):
        cos_sin = self.rotary_emb.cos_sin_cache
        _gptj_rope(q, positions, cos_sin)
        if not isinstance(attn_metadata, dict):
            return q
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata", attn_metadata[self.swa_cache_layer.prefix]
        )
        kv_roped = kv.clone()
        _gptj_rope(kv_roped, positions, cos_sin)
        cache = self.swa_cache_layer.kv_cache
        quantize_and_insert_k_cache(
            kv_roped,
            cache.view(cache.shape[0], -1),
            swa_metadata.slot_mapping,
            block_size=swa_metadata.block_size,
        )
        return q

    # ---------------------------------------------------------------- output proj
    def _o_proj(self, attn_out: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        o = attn_out[:, : self.n_local_heads]
        _gptj_rope(o, positions, self.rotary_emb.cos_sin_cache, inverse=True)
        g = self.n_local_groups
        o = o.reshape(o.shape[0], g, -1)
        T = o.shape[0]
        if _WO_A_BMM and getattr(self.wo_a, "bmm_weight", None) is not None:
            # oneDNN FP8 grouped GEMM (opt-in: slower end to end in decode).
            q, qs = _act_fp8_quant(o)
            z = torch.ops.vllm.xpu_fp8_bmm(
                # the kernel needs a contiguous activation (a strided view gives
                # wrong results for more than one token)
                q.transpose(0, 1).contiguous(), self.wo_a.bmm_weight, torch.bfloat16,
                qs.transpose(0, 1).contiguous(), self.wo_a.bmm_scale, None,
            )  # [G, T, R]
            return self._wo_b_proj(z.transpose(0, 1).flatten(1))
        if T <= _WO_A_GEMV_MAX_TOKENS and self.wo_a.weight.dtype == torch.float8_e4m3fn:
            # Decode: Triton GEMV straight on the FP8 weight and its block scales.
            if self._wo_a_sf is None:
                s = getattr(self.wo_a, "weight_scale_inv", None)
                if s is None:
                    s = self.wo_a.weight_scale
                sf = s.data
                if sf.dtype == torch.float8_e8m0fnu:
                    sf = sf.view(torch.uint8).to(torch.int32).sub(127).float().exp2()
                elif sf.dtype == torch.uint8:
                    sf = sf.to(torch.int32).sub(127).float().exp2()
                self._wo_a_sf = sf.float().contiguous()
            ob = _act_fp8_roundtrip(o).to(torch.bfloat16)
            z = fp8_grouped_gemv(ob, self.wo_a.weight.data, self._wo_a_sf, g)
            return self._wo_b_proj(z.to(torch.bfloat16).flatten(1))
        o = _act_fp8_roundtrip(o).to(torch.bfloat16)
        # Fallback: dequantize wo_a per call (no bf16 copy kept).
        w = self.wo_a.weight
        s = getattr(self.wo_a, "weight_scale_inv", None)
        if s is None:
            s = self.wo_a.weight_scale
        wf = _dequant_block_fp8(w.data, s.data) if w.dtype != torch.bfloat16 else w.float()
        wo_a = wf.to(torch.bfloat16).view(g, self.o_lora_rank, -1)
        del wf
        z = torch.einsum("tgd,grd->tgr", o, wo_a)
        return self._wo_b_proj(z.flatten(1))

    # ---------------------------------------------------------------- attention
    def forward_mqa(self, q, kv, positions, output) -> None:
        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is None:
            swa_only = self.compress_ratio == 0
            N = 0 if swa_only else cdiv_(self.max_model_len, self.compress_ratio)
            M = N + self.window_size + self.max_num_batched_tokens
            top_k = 0 if swa_only else self.topk_indices_buffer.shape[-1]
            combined_topk = round_up(top_k + self.window_size, 128)
            current_workspace_manager().get_simultaneous(
                ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
                ((self.max_num_batched_tokens, combined_topk), torch.int32),
                ((self.max_num_batched_tokens,), torch.int32),
            )
            output.zero_()
            return
        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None,
            attn_metadata.get(self.compressed_cache_prefix)
            if self.compressed_cache_prefix is not None
            else None,
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata", attn_metadata[self.swa_cache_layer.prefix]
        )
        swa_only = self.compress_ratio == 0
        self_kv_cache = None if swa_only else self._compressed_kv_cache()
        nd = swa_metadata.num_decode_tokens
        if swa_metadata.num_prefill_tokens > 0:
            self._forward_prefill(
                q[nd:], self_kv_cache, self.swa_cache_layer.kv_cache, output[nd:],
                flashmla_metadata, swa_metadata,
            )
        if nd > 0:
            self._forward_decode(
                q[:nd], self_kv_cache, swa_metadata, flashmla_metadata, swa_only,
                output[:nd],
            )

    def _forward_decode(self, q, kv_cache, swa_metadata, attn_metadata, swa_only, output):
        num_decodes = swa_metadata.num_decodes
        nd = swa_metadata.num_decode_tokens
        topk_indices = topk_lens = None
        if not swa_only:
            assert attn_metadata is not None and self.topk_indices_buffer is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            global_indices, topk_lens = compute_global_topk_indices_and_lens(
                self.topk_indices_buffer[:nd],
                swa_metadata.token_to_req_indices,
                attn_metadata.block_table[:num_decodes],
                block_size,
                swa_metadata.is_valid_token[:nd],
            )
            topk_indices = global_indices.view(nd, 1, -1)
        xpu_sparse_decode_fp8(
            q=q,
            kv_cache=kv_cache,
            swa_kv_cache=self.swa_cache_layer.kv_cache,
            swa_only=swa_only,
            topk_indices=topk_indices,
            topk_lens=topk_lens,
            swa_indices=swa_metadata.decode_swa_indices,
            swa_lens=swa_metadata.decode_swa_lens,
            attn_sink=self.attn_sink,
            softmax_scale=self.scale,
            head_dim=self.head_dim,
            nope_head_dim=self.nope_head_dim,
            rope_head_dim=self.rope_head_dim,
            out=output,
        )

    def _forward_prefill(self, q, compressed_k_cache, swa_k_cache, output,
                         attn_metadata, swa_metadata):
        swa_only = self.compress_ratio == 0
        npt = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        nd = swa_metadata.num_decode_tokens
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        qsl_cpu = swa_metadata.query_start_loc_cpu
        qsl = swa_metadata.query_start_loc
        base = qsl_cpu[num_decodes]
        topk_indices = self.topk_indices_buffer[nd:][:npt]
        top_k = 0 if swa_only else topk_indices.shape[-1]
        plan = swa_metadata.get_prefill_chunk_plan(
            compress_ratio=self.compress_ratio,
            prefill_chunk_size=self.PREFILL_CHUNK_SIZE,
            has_compressed=not swa_only,
        )
        ws = current_workspace_manager()
        combined_topk = round_up(top_k + self.window_size, 128)
        for c0, c1, cN, cM in plan:
            n = c1 - c0
            kvbuf, idx_out, len_out = ws.get_simultaneous(
                ((n, cM, q.shape[-1]), torch.bfloat16),
                ((self.max_num_batched_tokens, combined_topk), torch.int32),
                ((self.max_num_batched_tokens,), torch.int32),
            )
            if not swa_only:
                bt = attn_metadata.block_table[num_decodes:]
                dequantize_and_gather_k_cache(
                    kvbuf[:n], compressed_k_cache,
                    seq_lens=seq_lens[c0:c1] // self.compress_ratio,
                    gather_lens=None,
                    block_table=bt[c0:c1],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                )
            sbt = swa_metadata.block_table[num_decodes:]
            dequantize_and_gather_k_cache(
                kvbuf[:n], swa_k_cache,
                seq_lens=seq_lens[c0:c1],
                gather_lens=gather_lens[c0:c1],
                block_table=sbt[c0:c1],
                block_size=swa_metadata.block_size,
                offset=cN,
            )
            qs = qsl_cpu[num_decodes + c0] - base
            qe = qsl_cpu[num_decodes + c1] - base
            ci, cl = combine_topk_swa_indices(
                topk_indices[qs:qe],
                qsl[num_decodes + c0 : num_decodes + c1 + 1],
                seq_lens[c0:c1],
                gather_lens[c0:c1],
                self.window_size,
                self.compress_ratio,
                top_k,
                cM,
                cN,
                out=(idx_out[: qe - qs], len_out[: qe - qs]),
            )
            # Pad positions past each row's length with -1 for the triton kernel.
            ar = torch.arange(ci.shape[-1], device=ci.device)
            ci = torch.where(ar[None, :] < cl[:, None], ci, -1)
            out, _, _ = triton_bf16_mla_sparse_interface(
                q=q[qs:qe],
                kv=kvbuf[:n].reshape(-1, 1, q.shape[-1]),
                indices=ci.unsqueeze(1),
                sm_scale=self.scale,
                d_v=q.shape[-1],
                block_dpe=0,
                attn_sink=self.attn_sink,
            )
            output[qs:qe] = out


def cdiv_(a: int, b: int) -> int:
    return (a + b - 1) // b
