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

import os
import time
import typing
from collections.abc import Iterable
from itertools import islice

import regex as re
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
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
from . import alloc_trim, expert_stats, prof
from .mhc_xpu import XpuMHCPreDelayed
from .gguf_loader import GgufEngramEmbedding, gguf_path, load_expert_slice, run_gguf_experts
from .offload import DiskEngramEmbedding, ExpertCache, run_expert
from . import stream as expert_stream

logger = init_logger(__name__)
_DEBUG = __import__("os").environ.get("VLLM_DS41_DEBUG") == "1"
# Timing experiments only (wrong outputs): skip parts of the model, comma list of
# moe, shared, attn, allreduce (VLLM_DS41_ABLATE).
_ABLATE = set(filter(None, __import__("os").environ.get("VLLM_DS41_ABLATE", "").split(",")))
_TORCH_ROUTER = __import__("os").environ.get("VLLM_DS41_TORCH_ROUTER") == "1"
_MHC_TRITON = __import__("os").environ.get("VLLM_DS41_MHC_TRITON", "1") == "1"
# wo_a in INT4 cost ~2.5% perplexity in tests, so it stays FP8 unless asked.
_INT4_WO_A = __import__("os").environ.get("VLLM_DS41_INT4_WO_A", "0") == "1"
# embedding / LM head: fp8 (32x32 blocks; default), int4 (~1.7% perplexity), bf16
_VOCAB_FORMAT = __import__("os").environ.get("VLLM_DS41_VOCAB_FORMAT", "fp8")
_ROUTER_F32_CACHE = __import__("os").environ.get("VLLM_DS41_ROUTER_F32_CACHE") == "1"
# Reduce the MoE output in bf16 instead of fp32 for steps of at least this
# many tokens (0 = always fp32; default 64: prefill only).
_AR_BF16_MIN_T = int(os.environ.get("VLLM_DS41_AR_BF16_MIN_T", "64"))
# Streamed experts (VLLM_DS41_STREAM_DIR): steps of at most this many tokens use
# the batched FP4 decode kernel; longer steps run expert by expert.
_STREAM_GEMV_MAX_T = int(os.environ.get("VLLM_DS41_STREAM_GEMV_MAX_T", "16"))
# Same-token lookahead for streamed experts: at layer l, layers l+1..l+N
# route the current MoE input and their predicted (not resident) experts are
# queued as low-priority reads, so they overlap this layer's wait and compute.
_LOOKAHEAD = int(os.environ.get("VLLM_DS41_LOOKAHEAD", "0"))
# Skip missing experts in decode (VLLM_DS41_SKIP_MISS=1, gated by _SKIP_MAX_FRAC below): a
# decode-step expert that is not resident is left out of that step (weight 0, no
# renormalization) instead of stalling every card at the next all-reduce while it
# is read; its read is queued in the background so it is resident soon after.
# Prefill stays exact. VLLM_DS41_SKIP_MISS=0 restores plain streaming (read every miss).
# Default off again (2026-11-26): on 500 GSM8K questions gated skip scored 430 vs 440 with
# plain streaming (22 lost / 12 gained, p=0.12; first half p=0.03), the same as full skip.
_SKIP_MISS = os.environ.get("VLLM_DS41_SKIP_MISS", "0") == "1"
# Test only: VLLM_DS41_DROP_PAIRS=p drops each routed (token, expert) pair of
# streamed prefill steps with probability p (seeded), so the perplexity suite can
# price the loss of a few percent of pairs. Not for serving.
_DROP_PAIRS = float(os.environ.get("VLLM_DS41_DROP_PAIRS", "0") or 0)
# With skipping on: skip a missing expert only if every pair routed to it carries less
# than this share of its token's routing weight (default 0.15; an even split is ~0.17);
# heavier misses are read as before. 0 = skip every miss (full skip: -2 points GSM8K/500).
_SKIP_MAX_FRAC = float(os.environ.get("VLLM_DS41_SKIP_MAX_FRAC", "0.15") or 0)
# Test only (plan Stage 0a): VLLM_DS41_NOWAIT=1 makes streamed decode launch its kernels on the
# slot table and return: no routing sync, no reads, no installs (cache frozen after prefill).
# Upper bound for removing the per-layer host waits; output quality is not meaningful.
_NOWAIT = os.environ.get("VLLM_DS41_NOWAIT", "0") == "1"


def _step_has_prefill() -> bool:
    """True if the current step has prefill tokens (prefill chunk or mixed step):
    missing experts must then be read, not skipped, or the skipped experts end up
    in the KV cache of the prompt."""
    md = get_forward_context().attn_metadata
    if isinstance(md, dict):
        for m in md.values():
            n = getattr(m, "num_prefill_tokens", None)
            if n is not None:
                return n > 0
    return False
_DROP_GEN: dict = {}
_LOOKAHEAD_EXTRA = int(os.environ.get("VLLM_DS41_LOOKAHEAD_EXTRA", "2"))
_ASYNC = os.environ.get("VLLM_DS41_STREAM_ASYNC", "1") == "1"
_PF_OVERLAP = os.environ.get("VLLM_DS41_PF_OVERLAP", "1") == "1"
_ASYNC_CHECK = os.environ.get("VLLM_DS41_STREAM_ASYNC_CHECK", "0") == "1"
_ASYNC_STATS = {"n": 0, "miss": 0, "max": 0.0, "bad": 0}
_STREAM_MOES: dict = {}  # layer id -> XpuOffloadMoE (main layers)


def _expert_range(n_routed: int, tp: int, rank: int, layer: int) -> tuple[int, int]:
    """(first expert, count) of this rank's routed experts in `layer`.

    VLLM_DS41_EXPERT_SPLIT="63,64,65;63,65,64" gives per-rank counts, one
    pattern per layer, cycled. It evens out free memory when a card carries
    extra fixed overhead (card 0 gets ~185 MiB more driver memory from every
    vLLM process). Unset or malformed: an even split."""
    spec = os.environ.get("VLLM_DS41_EXPERT_SPLIT", "").strip()
    if spec:
        pats = [[int(c) for c in pat.split(",")] for pat in spec.split(";") if pat.strip()]
        if pats and all(len(c) == tp and sum(c) == n_routed and min(c) > 0 for c in pats):
            counts = pats[layer % len(pats)]
            return sum(counts[:rank]), counts[rank]
        logger.warning_once("DeepSeek V4.1 XPU: ignoring VLLM_DS41_EXPERT_SPLIT=%r "
                            "(need %d counts summing to %d per pattern)", spec, tp, n_routed)
    n = n_routed // tp
    return rank * n, n


def _has_submodule(m: nn.Module, name: str) -> bool:
    try:
        m.get_submodule(name)
        return True
    except AttributeError:
        return False


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
        self.n_layers = config.num_hidden_layers
        self.layer_id = int(self.prefix_ckpt.split(".")[1])
        self.e_start, self.n_local = _expert_range(
            self.n_routed, self.tp_size, get_tensor_model_parallel_rank(),
            int(self.prefix_ckpt.split(".")[1]))
        self.gate = _Gate(self.n_routed, config.hidden_size)
        expert_stats.register_gate(int(self.prefix_ckpt.split(".")[1]), self.gate)
        if self.layer_id < self.n_layers:
            _STREAM_MOES[self.layer_id] = self
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
        self._gate_f32: torch.Tensor | None = None
        self.t13 = self.t2 = 0

    def _route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """sqrt(softplus) scores, top-k on scores + bias, renormalized x scale."""
        if _ROUTER_F32_CACHE:
            if self._gate_f32 is None:
                self._gate_f32 = self.gate.weight.float()
            gate_w = self._gate_f32
        else:
            gate_w = self.gate.weight.float()  # temporary: no persistent fp32 copy
        logits = F.linear(x.float(), gate_w)
        if _TORCH_ROUTER:
            scores = F.softplus(logits).sqrt()
            idx = (scores + self.gate.e_score_correction_bias).topk(self.topk, dim=-1)[1]
            w = scores.gather(1, idx)
            return w / (w.sum(-1, keepdim=True) + 1e-20) * self.route_scale, idx
        from vllm import _custom_ops as ops

        T = x.shape[0]
        w = torch.empty(T, self.topk, dtype=torch.float32, device=x.device)
        idx = torch.empty(T, self.topk, dtype=torch.int32, device=x.device)
        tei = torch.empty(T, self.topk, dtype=torch.int32, device=x.device)
        ops.topk_hash_softplus_sqrt(
            w, idx, tei, logits, renormalize=True,
            routed_scaling_factor=self.route_scale,
            e_score_correction_bias=self.gate.e_score_correction_bias,
        )
        return w, idx

    def _pred_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Biased routing scores of this layer's router for x (lookahead)."""
        gw = self.gate.weight
        logits = F.linear(x.to(gw.dtype), gw).float()
        return F.softplus(logits).sqrt() + self.gate.e_score_correction_bias

    def load_gguf_experts(self, ck) -> int:
        layer = int(self.prefix_ckpt.split(".")[1])
        dev = self.gate.weight.device
        self.w13, self.w2, self.t13, self.t2 = load_expert_slice(
            ck, layer, self.e_start, self.n_local, dev
        )
        return self.w13.numel() + self.w2.numel()

    def forward(self, x: torch.Tensor, input_ids=None) -> torch.Tensor:
        with prof.section("router"):
            w, idx = self._route(x)
            prof.note_experts(idx)
            expert_stats.add(int(self.prefix_ckpt.split(".")[1]), idx, self.n_routed)
            out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device=x.device)
        # The memory-profile / warm-up passes run without attention metadata:
        # skip routed experts there so the expert cache (allocated outside
        # vLLM's memory budget) is not counted as model memory.
        real_step = is_forward_context_available() and isinstance(
            get_forward_context().attn_metadata, dict
        )
        expert_stats.trace(int(self.prefix_ckpt.split(".")[1]), x, idx, real_step)
        if self.gguf:
            if x.shape[0] > 0 and self.w13 is not None and "moe" not in _ABLATE:
                with prof.section("routed_moe"):
                    run_gguf_experts(x, idx, w, out, self.w13, self.w2, self.t13, self.t2,
                                     self.e_start, self.n_local, self.swiglu_limit)
        elif x.shape[0] > 0 and expert_stream.enabled():
            # Eager segment between graph segments (breakable graphs): host
            # side cache work and NVMe reads; writes into `out` in place. The
            # real-step check runs inside, at replay time.
            _stream_moe(x, idx, w, out, self)
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
        if "shared" not in _ABLATE:
            with prof.section("shared_expert"):
                out = out + self.shared_experts(x).float()
        if self.tp_size > 1 and "allreduce" not in _ABLATE:
            with prof.section("allreduce"):
                if _AR_BF16_MIN_T and x.shape[0] >= _AR_BF16_MIN_T:
                    # Prefill: reduce the bf16 result (half the bytes); the
                    # sum is rounded to bf16 just below anyway.
                    return tensor_model_parallel_all_reduce(out.to(x.dtype))
                out = tensor_model_parallel_all_reduce(out)
        return out.to(x.dtype)


    def _forward_stream(self, x, idx, w, out) -> None:
        if not (is_forward_context_available()
                and isinstance(get_forward_context().attn_metadata, dict)):
            return  # memory profile / warm-up pass: no expert reads
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

        if BreakableCUDAGraphCapture.current() is not None:
            # Graph capture: the router ran inside a captured (not executed)
            # segment, so idx is garbage; the real work happens at replay.
            return
        tm = expert_stream.TIMING
        t0 = time.perf_counter() if tm else 0.0
        T = x.shape[0]
        nexts = []
        if _LOOKAHEAD and T <= _STREAM_GEMV_MAX_T:
            for d in range(1, _LOOKAHEAD + 1):
                nm = _STREAM_MOES.get(self.layer_id + d)
                if nm is None:
                    break
                nexts.append(nm)
        if _ASYNC and not nexts and T <= _STREAM_GEMV_MAX_T:
            self._forward_stream_async(x, idx, w, out, tm, t0)
            return
        if _DROP_PAIRS > 0:
            g = _DROP_GEN.get(x.device)
            if g is None:
                g = _DROP_GEN[x.device] = torch.Generator(device=x.device).manual_seed(1234)
            keep = torch.rand(w.shape, generator=g, device=w.device) >= _DROP_PAIRS
            w = w * keep.to(w.dtype)
        if nexts:
            # one host transfer for this layer's routing and the predictions
            kp = self.topk + _LOOKAHEAD_EXTRA
            preds = [nm._pred_scores(x).topk(kp, dim=-1)[1].reshape(-1) for nm in nexts]
            vals = torch.cat([idx.reshape(-1).to(torch.int64)] + preds).tolist()
            n0 = idx.numel()
            experts = sorted({v for v in vals[:n0]
                              if self.e_start <= v < self.e_start + self.n_local})
        else:
            local = (idx >= self.e_start) & (idx < self.e_start + self.n_local)
            experts = torch.unique(idx[local]).tolist()
        cache = expert_stream.StreamCache.get(x.device, self.n_layers)
        if nexts:
            keys, seen = [], set()
            for j, nm in enumerate(nexts):  # nearer layers first
                base = n0 + j * T * kp
                for t in range(T):
                    for r in range(kp):  # rank-major within a token
                        v = vals[base + t * kp + r]
                        if nm.e_start <= v < nm.e_start + nm.n_local and (nm.layer_id, v) not in seen:
                            seen.add((nm.layer_id, v))
                            keys.append((nm.layer_id, v))
            cache.prefetch(keys)
        if not experts:
            return
        t1 = time.perf_counter() if tm else 0.0
        gguf = cache.lay.get("format", "").startswith("ds41-gguf")
        if x.shape[0] > _STREAM_GEMV_MAX_T and _PF_OVERLAP:
            with prof.section("routed_moe_prefill"):
                self._prefill_experts(x, idx, w, out, cache, experts, gguf)
            return
        with prof.section("expert_fetch"):
            slot_row = cache.ensure(self.layer_id, experts)
        t2 = time.perf_counter() if tm else 0.0
        with prof.section("routed_moe"):
            if x.shape[0] <= _STREAM_GEMV_MAX_T:
                if gguf:
                    from .gguf_stream_moe import moe_decode_gguf as decode
                else:
                    from .fp4_moe import moe_decode_fp4 as decode

                decode(x, idx, w, out, cache.pool, slot_row, cache.lay, self.swiglu_limit)
                if tm:
                    t3 = time.perf_counter()
                    if tm == 2:
                        torch.xpu.synchronize()
                    cache.note_timing(x.shape[0], t1 - t0, t2 - t1, t3 - t2,
                                      time.perf_counter() - t3)
                return
            self._prefill_experts(x, idx, w, out, cache, experts, gguf)

    def _prefill_experts(self, x, idx, w, out, cache, experts, gguf) -> None:
        """Many tokens: run each local expert on its tokens as soon as it is
        resident (reads of the others continue meanwhile). The routing is
        sorted once (one host transfer for the counts) instead of a
        torch.where per expert, which made the host wait for the GPU each time."""
        from .gguf_stream_moe import run_expert_gguf

        k = idx.shape[1]
        flat = idx.reshape(-1)
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=self.e_start + self.n_local).tolist()
        starts = [0] * len(counts)
        acc = 0
        for i, c in enumerate(counts):
            starts[i] = acc
            acc += c
        wf = w.reshape(-1)
        hit_set = {e for e in experts if (self.layer_id, e) in cache.lru}
        done: set = set()

        def run(e):
            if e in done:
                return
            done.add(e)
            n = counts[e]
            if n == 0:
                return
            if e not in hit_set:
                cache.wait_copies()
            sel = order[starts[e]: starts[e] + n]
            rows = torch.div(sel, k, rounding_mode="floor")
            xs, ws = x[rows], wf[sel].unsqueeze(1)
            if gguf:
                y = run_expert_gguf(xs, cache.pool, cache.lru[(self.layer_id, e)], cache.lay,
                                    self.swiglu_limit, ws)
            else:
                y = run_expert(xs, cache.parts(self.layer_id, e), self.swiglu_limit, ws)
            out.index_add_(0, rows, y)

        for e in cache.ensure_iter(self.layer_id, experts):
            run(e)
        for e in experts:  # all resident now; none may be skipped
            run(e)

    def _check_async(self, x, idx, w, out, before, cache, decode, had_miss) -> None:
        """VLLM_DS41_STREAM_ASYNC_CHECK=1: redo the call the synchronous way
        (full slot row after ensure) and log the worst relative difference."""
        ref = torch.zeros_like(out)
        decode(x, idx, w, ref, cache.pool, cache.slot_dev[self.layer_id], cache.lay,
               self.swiglu_limit)
        d = ((out - before) - ref).abs().max().item()
        m = max(ref.abs().max().item(), 1e-6)
        st = _ASYNC_STATS
        st["n"] += 1
        st["miss"] += bool(had_miss)
        st["max"] = max(st["max"], d / m)
        st["bad"] += d / m > 1e-2
        if st["n"] % 2000 == 0:
            logger.info("async check: %d calls (%d with misses), max rel diff %.2e, %d over 1e-2",
                        st["n"], st["miss"], st["max"], st["bad"])

    def _forward_stream_async(self, x, idx, w, out, tm, t0) -> None:
        """Decode without waiting for the routing first: launch the expert
        kernels on the device slot table (experts not resident read slot -1 and
        add nothing), copy the routing to the host meanwhile, then fetch any
        missing experts and add their pairs in a top-up pass."""
        cache = expert_stream.StreamCache.get(x.device, self.n_layers)
        if cache.lay.get("format", "").startswith("ds41-gguf"):
            from .gguf_stream_moe import moe_decode_gguf as decode
        else:
            from .fp4_moe import moe_decode_fp4 as decode
        if _NOWAIT:
            decode(x, idx, w, out, cache.pool, cache.slot_dev[self.layer_id], cache.lay,
                   self.swiglu_limit)
            return
        if cache.step_mode and not _step_has_prefill():
            # Plan v2 Stage 1-lite: no host wait in decode; misses are skipped this step and
            # read in the background (stream.py "step mode").
            ta = time.perf_counter()
            cache.step_layer()
            tb = time.perf_counter()
            decode(x, idx, w, out, cache.pool, cache.slot_dev[self.layer_id], cache.lay,
                   self.swiglu_limit)
            tc = time.perf_counter()
            cache.step_record(self.layer_id, idx, w, self.e_start, self.n_local)
            st = cache.stats
            st["st_n"] = st.get("st_n", 0) + 1
            st["st_layer"] = st.get("st_layer", 0.0) + (tb - ta)
            st["st_launch"] = st.get("st_launch", 0.0) + (tc - tb)
            st["st_rec"] = st.get("st_rec", 0.0) + (time.perf_counter() - tc)
            return
        cache.sync_table()
        skip_ok = _SKIP_MISS and not _step_has_prefill()
        before = out.clone() if _ASYNC_CHECK else None
        hb = cache.idx_host(idx.numel(), idx.dtype)
        hb.copy_(idx.reshape(-1), non_blocking=True)
        wb = None
        if skip_ok and _SKIP_MAX_FRAC > 0:
            wb = cache.idx_host(w.numel(), torch.float32, "_wh")
            wf = w.float()
            wb.copy_((wf / wf.sum(-1, keepdim=True).clamp_min(1e-20)).reshape(-1), non_blocking=True)
        ev = torch.xpu.Event()
        ev.record()
        with prof.section("routed_moe"):
            decode(x, idx, w, out, cache.pool, cache.slot_dev[self.layer_id], cache.lay,
                   self.swiglu_limit)
        t1 = time.perf_counter() if tm else 0.0
        ev.synchronize()
        lo, hi = self.e_start, self.e_start + self.n_local
        experts = sorted({v for v in hb.tolist() if lo <= v < hi})
        t2 = time.perf_counter() if tm else 0.0
        miss: list = []
        if experts:
            miss = [e for e in experts if (self.layer_id, e) not in cache.lru]
            if _SKIP_MISS and miss and not skip_ok:
                cache.stats["skip_prefill_read"] = cache.stats.get("skip_prefill_read", 0) + len(miss)
            if skip_ok and miss:
                skip = miss
                if wb is not None:
                    ms = set(miss)
                    frac: dict = {}
                    for v, f in zip(hb.tolist(), wb.tolist()):
                        if v in ms and f > frac.get(v, 0.0):
                            frac[v] = f
                    skip = [e for e in miss if frac.get(e, 0.0) < _SKIP_MAX_FRAC]
                    cache.stats["skip_kept"] = cache.stats.get("skip_kept", 0) + len(miss) - len(skip)
                if skip:
                    cache.fetch_background([(self.layer_id, e) for e in skip])
                    experts = [e for e in experts if e not in skip]
                    miss = [e for e in miss if e not in skip]
            with prof.section("expert_fetch"):
                cache.in_decode = True
                cache.ensure(self.layer_id, experts)
                cache.in_decode = False
            if miss:
                with prof.section("routed_moe"):
                    decode(x, idx, w, out, cache.pool, cache.fix_row(self.layer_id, miss),
                           cache.lay, self.swiglu_limit)
        if _ASYNC_CHECK:
            self._check_async(x, idx, w, out, before, cache, decode, miss)
        if tm:
            # columns: routing wait, ensure + top-up, launch, 0
            cache.note_timing(x.shape[0], t2 - t1, time.perf_counter() - t2, t1 - t0, 0.0,
                              self.layer_id, len(miss))


@eager_break_during_capture
def _stream_moe(x, idx, w, out, moe) -> None:
    moe._forward_stream(x, idx, w, out)


def _time_graph_replay() -> None:
    """VLLM_DS41_STREAM_TIMING=3: host time per breakable-graph replay, split
    into graph-segment submits and eager segments by function, per graph
    (segment count tells target and draft apart), plus the host time between
    replays (sampling, drafting, scheduling). Logged every 500 replays."""
    from collections import defaultdict

    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture as C

    if getattr(C.replay, "_ds41_timed", False):
        return
    names: dict = {}
    st = {"n": 0, "gap": 0.0, "last": None}
    per: dict = defaultdict(lambda: defaultdict(float))  # nseg -> name -> s (+ "#")

    def name_of(r):
        k = id(r)
        if k not in names:
            nm = "?"
            for c in getattr(r, "__closure__", None) or ():
                f = c.cell_contents
                if callable(f) and hasattr(f, "__name__") and f.__name__ != "<lambda>":
                    nm = f.__name__
                    break
            names[k] = nm
        return names[k]

    def replay(self) -> None:
        t0 = time.perf_counter()
        if st["last"] is not None:
            st["gap"] += t0 - st["last"]
        d_ = per[len(self.segments)]
        for r in self.segments:
            a = time.perf_counter()
            r()
            d = time.perf_counter() - a
            if getattr(r, "__self__", None) is not None:
                d_["graph submit"] += d
            else:
                d_[name_of(r)] += d
        d_["#"] += 1
        d_["total"] += time.perf_counter() - t0
        st["last"] = time.perf_counter()
        st["n"] += 1
        if prof_at:
            _profile_tick(st, prof_at)
        if st["n"] % 500 == 0:
            n = st["n"]
            parts = []
            for nseg, d in sorted(per.items()):
                c = d["#"]
                parts.append("%d segs x%d: " % (nseg, c) + ", ".join(
                    "%s %.2f" % (k, v / c * 1e3) for k, v in sorted(d.items(), key=lambda kv: -kv[1])
                    if k != "#"))
            logger.info("graph replay (ms per replay): %s; between replays %.2f ms",
                        " | ".join(parts), st["gap"] / n * 1e3)
            per.clear()
            st["gap"] = 0.0

    prof_at = None
    if os.environ.get("VLLM_DS41_PROFILE"):
        a, _, b = os.environ["VLLM_DS41_PROFILE"].partition(",")
        prof_at = (int(a), int(b or 20))
    replay._ds41_timed = True
    C.replay = replay


def _profile_tick(st: dict, at: tuple) -> None:
    """VLLM_DS41_PROFILE=start,count: torch profiler (XPU kernels, rank 0) over replays
    start..start+count; logs kernels by XPU time and saves a Chrome trace in
    /bench/logs."""
    n = st["n"]
    if n == at[0]:
        from torch.profiler import ProfilerActivity, profile

        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return  # rank 0 only: the profiler's host memory adds up per worker
        torch.xpu.synchronize()
        p = profile(activities=[ProfilerActivity.XPU])
        p.__enter__()
        st["prof"], st["prof_t"] = p, time.perf_counter()
    elif n == at[0] + at[1] and st.get("prof") is not None:
        torch.xpu.synchronize()
        wall = time.perf_counter() - st["prof_t"]
        p = st.pop("prof")
        p.__exit__(None, None, None)
        ka = p.key_averages()
        tot = sum(e.self_device_time_total for e in ka)
        logger.info("profile: %d replays, wall %.2f ms per replay, XPU kernel time %.2f ms per "
                    "replay\n%s", at[1], wall / at[1] * 1e3, tot / at[1] / 1e3,
                    ka.table(sort_by="self_device_time_total", row_limit=45, max_name_column_width=70))
        del p, ka


if expert_stream.TIMING == 3:
    _time_graph_replay()

# VLLM_DS41_PROFILE_EAGER=start,count: the same torch profiler window over eager
# decode steps (the profiler hangs graph replays on XPU).
_EAGER_PROF_AT = None
if os.environ.get("VLLM_DS41_PROFILE_EAGER"):
    _a, _, _b = os.environ["VLLM_DS41_PROFILE_EAGER"].partition(",")
    _EAGER_PROF_AT = (int(_a), int(_b or 20))
_EAGER_PROF_ST: dict = {"n": 0}


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
        if extract_layer_index(prefix) >= config.num_hidden_layers:
            # DSpark draft layer (mtp.*): resident MXFP4 experts.
            from .dspark_moe import XpuDSparkMoE

            self.ffn = XpuDSparkMoE(vllm_config, prefix=f"{prefix}.ffn")
        else:
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
        self.mhc_pre_delayed = XpuMHCPreDelayed() if _MHC_TRITON else MHCPreDelayedOp()
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
            if "attn" not in _ABLATE:
                x = self.attn(positions, x, None)
        if _DEBUG:
            torch.accelerator.synchronize()
            logger.info("DS41DBG   after attn, host avail %d MiB", _avail_mib())
        if self.attn.wo_b.reduce_results is False and "allreduce" not in _ABLATE:
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
        alloc_trim.maybe_trim(positions.shape[0])
        expert_stats.step()
        expert_stats.trace_step()
        prof.begin_step(positions.shape[0])
        if _EAGER_PROF_AT and positions.shape[0] <= 4 and not torch.xpu.is_current_stream_capturing():
            _EAGER_PROF_ST["n"] += 1
            _profile_tick(_EAGER_PROF_ST, _EAGER_PROF_AT)
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
        aux_hidden_states: list[torch.Tensor] = []
        for idx in range(self.start_layer, self.end_layer):
            ran = True
            if dbg:
                torch.accelerator.synchronize()
                logger.info("DS41DBG layer %d start, host avail %d MiB", idx, _avail_mib())
            hidden_states, residual, post_mix, res_mix, pre_mix = self.layers[idx](
                hidden_states, positions, input_ids, pre_mix, post_mix, res_mix,
                residual, engram_hashes, engram_mask,
            )
            if idx + 1 in self.aux_hidden_state_layers:
                # Draft (DSpark) input: the layer's mHC-merged state, hc copies averaged.
                aux_hidden_states.append(
                    self.mhc_post(hidden_states, residual, post_mix, res_mix).mean(dim=1))
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
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
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
        for m in self.modules():
            if isinstance(m, DeepseekV41XPUAttention):
                m.prepare_decode_weights()
            elif (isinstance(m, XpuOffloadMoE) and m._gate_f32 is None
                  and _ROUTER_F32_CACHE):
                m._gate_f32 = m.gate.weight.float()
        from . import int4_linear

        if int4_linear.ENABLED:
            n4, b0, b1 = int4_linear.convert_dense_to_int4(self, skip=(".wo_a",))
            logger.info("DeepSeek V4.1 XPU: %d dense linears stored as INT4 (group %d): "
                        "%.2f GiB -> %.2f GiB on this GPU", n4, int4_linear.GROUP,
                        b0 / 2**30, b1 / 2**30)
            extra0 = extra1 = 0
            if _INT4_WO_A:
                for m in self.modules():
                    if (isinstance(m, DeepseekV41XPUAttention)
                            and m.wo_a.weight.dtype == torch.float8_e4m3fn):
                        a, b = int4_linear.convert_grouped_to_int4(m.wo_a, m.n_local_groups)
                        m._wo_a_sf = None
                        extra0, extra1 = extra0 + a, extra1 + b
            if _VOCAB_FORMAT in ("fp8", "int4"):
                conv = (int4_linear.convert_vocab_to_fp8 if _VOCAB_FORMAT == "fp8"
                        else int4_linear.convert_vocab_to_int4)
                for name in ("model.embed_tokens", "lm_head"):
                    if not _has_submodule(self, name):
                        continue
                    layer = self.get_submodule(name)
                    if layer.weight.dtype == torch.bfloat16:
                        a, b = conv(layer)
                        extra0, extra1 = extra0 + a, extra1 + b
            logger.info("DeepSeek V4.1 XPU: wo_a %s, embedding / LM head %s: "
                        "%.2f GiB -> %.2f GiB on this GPU", "int4" if _INT4_WO_A else "fp8",
                        _VOCAB_FORMAT, extra0 / 2**30, extra1 / 2**30)
        from .fp8_gemv import install_decode_fp8_linears

        n = install_decode_fp8_linears(self, skip=(".wo_a",))
        logger.info("DeepSeek V4.1 XPU: decode FP8 GEMV on %d block-FP8 linear layers", n)


__all__ = [
    "DeepseekV41LLMForCausalLM",
    "_linear_scale_param_name",
    "_make_deepseek_v4_weights_mapper",
    "cdiv",
]
