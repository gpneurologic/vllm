# SPDX-License-Identifier: Apache-2.0
"""Serve DeepSeek V4.1 GGUF builds (llama.cpp ``deepseek41``) with the XPU port.

``--load-format ds41_gguf`` with ``VLLM_DS41_GGUF=<folder of the split GGUF>``
(shard 6, the Engram tables, linked into that folder as llama.cpp expects).
The served model folder holds the original config (with ``n_routed_experts``
set to the GGUF's ``expert_count``) and tokenizer, and no weights.

What is loaded from where:
- dense linears (Q5_K / Q8_0 / BF16): dequantized and requantized to the FP8
  32x32 block format of the original checkpoint, then loaded by the normal
  loader (so TP padding and the FP8 linear kernels are unchanged);
- norms, router, mHC, sinks, compressor, indexer small weights: as is;
- routed experts: each rank keeps its contiguous slice of the GGUF expert
  tensors in their quantized form on the device (``XpuOffloadMoE.load_gguf_experts``)
  and runs them with the GGUF Triton fused MoE kernels;
- Engram tables (IQ4_NL): rows read from the GGUF on demand (``GgufEngramEmbedding``).

The Triton kernels come from vllm-gguf-plugin (``vllm_gguf_plugin.triton``),
found on ``sys.path``; only that subpackage is imported.
"""
from __future__ import annotations

import os
import re
import sys
import time
import types
from collections.abc import Generator, Iterator

import numpy as np
import torch
import torch.nn as nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

from .gguf_ckpt import GGUFCheckpoint, dequant_iq4_nl

logger = init_logger(__name__)
GGUF_ENV = "VLLM_DS41_GGUF"


def gguf_path() -> str | None:
    return os.environ.get(GGUF_ENV) or None


def checkpoint() -> GGUFCheckpoint:
    p = gguf_path()
    assert p, f"{GGUF_ENV} is not set"
    return GGUFCheckpoint.get(p)


# ------------------------------------------------------------ plugin kernels
_kern = None


def kernels():
    """(ggml_dequantize_triton, ggml_moe_a8_triton, get_triton_moe_block_m)."""
    global _kern
    if _kern is None:
        if "vllm_gguf_plugin" not in sys.modules:
            root = next((os.path.join(p, "vllm_gguf_plugin") for p in sys.path
                         if os.path.isdir(os.path.join(p, "vllm_gguf_plugin"))), None)
            if root is None:
                raise ImportError("vllm_gguf_plugin not found on sys.path (PYTHONPATH)")
            pkg = types.ModuleType("vllm_gguf_plugin")
            pkg.__path__ = [root]
            sys.modules["vllm_gguf_plugin"] = pkg
        from vllm_gguf_plugin.triton.dequantize.interface import ggml_dequantize_triton
        from vllm_gguf_plugin.triton.fused_moe.interface import ggml_moe_a8_triton
        from vllm_gguf_plugin.triton.fused_moe.utils import get_triton_moe_block_m

        _kern = (ggml_dequantize_triton, ggml_moe_a8_triton, get_triton_moe_block_m)
    return _kern


def _device() -> torch.device:
    return torch.device("xpu", torch.xpu.current_device())


def dequant(ck: GGUFCheckpoint, name: str, dev: torch.device) -> torch.Tensor:
    """Any 2D/1D GGUF tensor -> float32 on dev, torch shape."""
    t = ck.tensors[name]
    if t.type_name in ("F32", "F16", "BF16"):
        return ck.plain(name).to(dev).float()
    deq = kernels()[0]
    rows = int(np.prod(t.dims[1:])) if len(t.dims) > 1 else 1
    w = ck.raw(name).to(dev)
    return deq(w, t.type_id, rows, t.dims[0], torch.float32).reshape(t.shape)


def fp8_block_quant(w: torch.Tensor, b: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """float32 [R, C] -> (float8_e4m3fn [R, C], float8_e8m0fnu [R/b, C/b]) with
    power-of-two (ue8m0) block scales, the original checkpoint's format."""
    R, C = w.shape
    assert R % b == 0 and C % b == 0, (R, C)
    blk = w.view(R // b, b, C // b, b)
    amax = blk.abs().amax(dim=(1, 3)).clamp(min=1e-30)
    e = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    q = (blk / torch.exp2(e)[:, None, :, None]).clamp(-448, 448)
    q = q.to(torch.float8_e4m3fn).view(R, C)
    sc = (e + 127).to(torch.uint8).view(torch.float8_e8m0fnu)
    return q, sc


# GGUF block tensor suffix -> (original suffix, kind); kinds: fp8 (linear with
# .weight + .scale), bf16, f32, skip. A None original suffix = skip.
_BLK = {
    "attn_norm.weight": ("attn_norm.weight", "bf16"),
    "ffn_norm.weight": ("ffn_norm.weight", "bf16"),
    "attn_q_a.weight": ("attn.wq_a", "fp8"),
    "attn_q_a_norm.weight": ("attn.q_norm.weight", "bf16"),
    "attn_kv.weight": ("attn.wkv", "fp8"),
    "attn_kv_a_norm.weight": ("attn.kv_norm.weight", "bf16"),
    "attn_q_b.weight": ("attn.wq_b", "fp8"),
    "attn_output_a.weight": ("attn.wo_a", "fp8"),
    "attn_output_b.weight": ("attn.wo_b", "fp8"),
    "attn_sinks.weight": ("attn.attn_sink", "f32"),
    "exp_probs_b.bias": ("ffn.gate.bias", "f32"),
    "exp_probs_b_vl.bias": (None, "skip"),
    "ffn_gate_inp.weight": ("ffn.gate.weight", "bf16"),
    "ffn_gate_shexp.weight": ("ffn.shared_experts.w1", "fp8"),
    "ffn_up_shexp.weight": ("ffn.shared_experts.w3", "fp8"),
    "ffn_down_shexp.weight": ("ffn.shared_experts.w2", "fp8"),
    "ffn_gate_exps.weight": (None, "skip"),
    "ffn_up_exps.weight": (None, "skip"),
    "ffn_down_exps.weight": (None, "skip"),
    "hc_attn_base.weight": ("hc_attn_base", "f32"),
    "hc_attn_fn.weight": ("hc_attn_fn", "f32"),
    "hc_attn_scale.weight": ("hc_attn_scale", "f32"),
    "hc_ffn_base.weight": ("hc_ffn_base", "f32"),
    "hc_ffn_fn.weight": ("hc_ffn_fn", "f32"),
    "hc_ffn_scale.weight": ("hc_ffn_scale", "f32"),
    # engram_gate = q_weight * k_weight (the converter folds them); the gate
    # kernel only uses the product, so q = gate and k = 1 is the same model.
    "engram_gate.weight": ("engram.q_weight", "bf16"),
    "engram_kv.weight": ("engram.wkv", "fp8"),
    "attn_compressor_gate.weight": ("attn.compressor.wgate.weight", "bf16"),
    "attn_compressor_kv.weight": ("attn.compressor.wkv.weight", "bf16"),
    "attn_compressor_norm.weight": ("attn.compressor.norm.weight", "bf16"),
    "indexer.attn_k.weight": ("attn.indexer.wk.weight", "bf16"),
    "indexer.attn_q_b.weight": ("attn.indexer.wq_b", "fp8"),
    "indexer.k_norm.weight": ("attn.indexer.k_norm.weight", "bf16"),
    "indexer.proj.weight": ("attn.indexer.weights_proj.weight", "bf16"),
}
_GLOBAL = {
    "token_embd.weight": ("embed.weight", "bf16"),
    "output.weight": ("head.weight", "bf16"),
    "output_norm.weight": ("norm.weight", "bf16"),
    "per_layer_token_embd.weight": (None, "skip"),
}
_BLK_RE = re.compile(r"blk\.(\d+)\.(.+)$")


def translate(name: str) -> tuple[str | None, str]:
    """GGUF tensor name -> (original checkpoint name or None, kind)."""
    m = _BLK_RE.match(name)
    if m:
        if m.group(2) not in _BLK:
            raise KeyError(f"DeepSeek V4.1 GGUF: unknown tensor {name}")
        orig, kind = _BLK[m.group(2)]
        return (f"layers.{m.group(1)}.{orig}" if orig else None), kind
    if name not in _GLOBAL:
        raise KeyError(f"DeepSeek V4.1 GGUF: unknown tensor {name}")
    return _GLOBAL[name]


# Optional: take the dense linears (FP8, 32x32 block scales) and the bf16
# embedding / output head from the original checkpoint instead of the GGUF's
# Q5_K / Q8_0 copies (VLLM_DS41_ORIG_CKPT=<original model folder>).
_ORIG_KINDS = ("fp8",)
_ORIG_NAMES = ("embed.weight", "head.weight")


def _orig_checkpoint():
    folder = os.environ.get("VLLM_DS41_ORIG_CKPT")
    if not folder:
        return None
    from .offload import CheckpointFiles

    return CheckpointFiles.get(folder)


def gguf_weights(ck: GGUFCheckpoint, dev: torch.device) -> Iterator[tuple[str, torch.Tensor]]:
    """Original-checkpoint-named tensors made from the GGUF."""
    t0, n, nbytes = time.time(), 0, 0
    oc = _orig_checkpoint()
    n_orig = 0
    for gname, t in ck.tensors.items():
        orig, kind = translate(gname)
        if orig is None:
            continue
        if oc is not None and kind in _ORIG_KINDS and f"{orig}.weight" in oc.index:
            yield f"{orig}.weight", oc.tensor(f"{orig}.weight")
            yield f"{orig}.scale", oc.tensor(f"{orig}.scale").view(torch.float8_e8m0fnu)
            n_orig += 1
            continue
        if oc is not None and orig in _ORIG_NAMES and orig in oc.index:
            yield orig, oc.tensor(orig)
            n_orig += 1
            continue
        w = dequant(ck, gname, dev)
        nbytes += t.nbytes
        if kind == "fp8":
            q, sc = fp8_block_quant(w)
            yield f"{orig}.weight", q
            yield f"{orig}.scale", sc
        elif kind == "bf16":
            yield orig, w.to(torch.bfloat16)
        else:
            yield orig, w
        if orig.endswith("engram.q_weight"):
            yield orig.replace("q_weight", "k_weight"), torch.ones_like(w, dtype=torch.bfloat16)
        n += 1
        del w
    logger.info("DeepSeek V4.1 GGUF: %d dense tensors (%.2f GB quantized) converted, %d taken "
                "from the original checkpoint, in %.1f s", n, nbytes / 1e9, n_orig,
                time.time() - t0)


@register_model_loader("ds41_gguf")
class DeepseekV41GGUFLoader(DefaultModelLoader):
    """Default loader whose weights come from the GGUF (see module doc)."""

    def download_model(self, model_config) -> None:
        pass

    def _get_all_weights(self, model_config, model) -> Generator[tuple[str, torch.Tensor], None, None]:
        self.counter_before_loading_weights = time.perf_counter()
        if hasattr(model, "get_draft_kv_cache_layer_names"):
            # DSpark draft: its mtp.* weights come from the original checkpoint.
            oc = _orig_checkpoint()
            if oc is None:
                raise ValueError("the DSpark draft needs VLLM_DS41_ORIG_CKPT")
            for name in oc.index:
                if name.startswith("mtp."):
                    t = oc.tensor(name)
                    if name.endswith(".scale") and t.element_size() == 1:
                        t = t.view(torch.float8_e8m0fnu)
                    yield name, t
            return
        yield from gguf_weights(checkpoint(), _device())

    def load_weights(self, model: nn.Module, model_config) -> None:
        super().load_weights(model, model_config)
        from .model import XpuOffloadMoE

        ck = checkpoint()
        t0, total = time.time(), 0
        for m in model.modules():
            if isinstance(m, XpuOffloadMoE):
                total += m.load_gguf_experts(ck)
        torch.xpu.synchronize()
        logger.info("DeepSeek V4.1 GGUF: routed experts %.2f GiB on %s in %.1f s",
                    total / 2**30, _device(), time.time() - t0)


# ---------------------------------------------------------------- routed MoE
def gguf_expert_tensors(ck: GGUFCheckpoint, layer: int):
    return tuple(ck.tensors[f"blk.{layer}.ffn_{w}_exps.weight"] for w in ("gate", "up", "down"))


def load_expert_slice(ck: GGUFCheckpoint, layer: int, e_start: int, n: int,
                      dev: torch.device):
    """(w13 [n, 2I, rb], w2 [n, H, rb], type13, type2) for experts [e_start, e_start+n)."""
    g, u, d = gguf_expert_tensors(ck, layer)
    assert g.type_id == u.type_id, (layer, g.type_name, u.type_name)

    def part(t):
        rows = t.dims[1]
        eb = rows * t.row_bytes
        return ck.raw(t.name, e_start * eb, n * eb).reshape(n, rows, t.row_bytes)

    w13 = torch.cat([part(g), part(u)], dim=1).to(dev)
    w2 = part(d).to(dev)
    return w13, w2, g.type_id, d.type_id


# (token, expert) pairs up to which the decode GEMV path is used
_DECODE_MAX_PAIRS = int(os.environ.get("VLLM_DS41_DECODE_MAX_PAIRS", "64"))


# Prefill expert GEMM: "mmq" (tl.dot kernel in gguf_moe_prefill.py, checked
# against the plugin kernel once per type) or "plugin" (the plugin's kernel).
_MOE_PREFILL = os.environ.get("VLLM_DS41_MOE_PREFILL", "mmq")
# int8 activations x int8 weights on XMX for the IQ2_XXS expert matrices (~1.4x
# faster than the bf16 kernel; per-32 activation quantization as llama.cpp MMQ)
_MOE_I8 = os.environ.get("VLLM_DS41_MOE_I8", "1") == "1"


def run_gguf_experts(x, idx, w, out, w13, w2, t13, t2, e_start, n_local, limit):
    """Add the local experts' outputs to out (float32 [T, H]) in place."""
    from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

    if idx.numel() <= _DECODE_MAX_PAIRS:
        from .gguf_moe_decode import moe_decode

        kernels()  # registers the plugin package (IQ tables)
        moe_decode(x, idx, w, out, w13, w2, t13, t2, e_start, n_local, limit)
        return
    from . import prof
    from .gguf_moe_prefill import BLOCK_M as MMQ_BLOCK_M
    from .gguf_moe_prefill import (I8_TYPES, gguf_moe_mmq, gguf_moe_mmq_i8, mmq_ok,
                                   quantize_rows_i8)

    _, moe, block_m = kernels()
    _ps = prof.section("moe_sort")
    _ps.__enter__()
    local = (idx >= e_start) & (idx < e_start + n_local)
    tok, slot = local.nonzero(as_tuple=True)
    if tok.numel() == 0:
        _ps.__exit__(None, None, None)
        return
    eid = (idx[tok, slot] - e_start).to(torch.int32).view(-1, 1)
    xs = x[tok].contiguous()
    I2, H = w13.shape[1], w2.shape[1]
    P = xs.shape[0]
    mm13 = _MOE_PREFILL == "mmq" and mmq_ok(w13, t13)
    mm2 = _MOE_PREFILL == "mmq" and mmq_ok(w2, t2)
    i8_13 = mm13 and _MOE_I8 and t13 in I8_TYPES and mmq_ok(w13, t13, int8=True)
    i8_2 = mm2 and _MOE_I8 and t2 in I8_TYPES and mmq_ok(w2, t2, int8=True)
    bm = MMQ_BLOCK_M if mm13 else block_m(t13)
    s, e, n = moe_align_block_size(eid, bm, n_local)
    _ps.__exit__(None, None, None)
    with prof.section("moe_w13"):
        if mm13:
            if i8_13:
                # quantize each token once, then pick its rows for the pairs
                q, qs = quantize_rows_i8(x)
                h = gguf_moe_mmq_i8(xs, w13, t13, s, e, n, MMQ_BLOCK_M, P,
                                    xq=(q[tok].contiguous(), qs[tok].contiguous()))
            else:
                h = gguf_moe_mmq(xs, w13, t13, s, e, n, MMQ_BLOCK_M, P)
        else:
            h = moe(xs, w13, s, e, n, t13, I2, 1, P)
    _pa = prof.section("moe_act")
    _pa.__enter__()
    gate, up = h.float().chunk(2, dim=-1)
    if limit:
        gate = gate.clamp(max=limit)
        up = up.clamp(-limit, limit)
    a = (torch.nn.functional.silu(gate) * up).to(x.dtype)
    _pa.__exit__(None, None, None)
    bm2 = MMQ_BLOCK_M if mm2 else block_m(t2)
    if bm2 != bm:
        s, e, n = moe_align_block_size(eid, bm2, n_local)
    with prof.section("moe_w2"):
        if mm2:
            mm = gguf_moe_mmq_i8 if i8_2 else gguf_moe_mmq
            y = mm(a, w2, t2, s, e, n, MMQ_BLOCK_M, P)
        else:
            y = moe(a, w2, s, e, n, t2, H, 1, P)
    with prof.section("moe_combine"):
        out.index_add_(0, tok, y.float() * w[tok, slot, None].float())


# ------------------------------------------------------------------- Engram
# Engram rows are random 144-byte reads from a ~110 GB file that does not fit in
# RAM: one page fault (NVMe read) per row. A single thread waits on each read in
# turn (~1k rows/s); many threads keep the NVMe queue full (~60k rows/s with 64).
_ENGRAM_THREADS = int(os.environ.get("VLLM_DS41_ENGRAM_THREADS", "64"))
_engram_pool = None


def _engram_gather(table: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """table[idx] (rows), gathered by a thread pool when there are many rows."""
    global _engram_pool
    n = idx.shape[0]
    if _ENGRAM_THREADS <= 1 or n < 256:
        return table[idx]
    if _engram_pool is None:
        from concurrent.futures import ThreadPoolExecutor

        _engram_pool = ThreadPoolExecutor(_ENGRAM_THREADS, thread_name_prefix="ds41-engram")
    out = np.empty((n, table.shape[1]), dtype=table.dtype)
    bounds = np.linspace(0, n, min(n // 32, _ENGRAM_THREADS * 4) + 1, dtype=np.int64)

    def part(i):
        a, b = bounds[i], bounds[i + 1]
        np.take(table, idx[a:b], axis=0, out=out[a:b])  # releases the GIL

    list(_engram_pool.map(part, range(len(bounds) - 1)))
    return out


class GgufEngramEmbedding(nn.Module):
    """Stands in for ``ParallelEngramEmbedding``: hash columns split over TP
    ranks, IQ4_NL rows read from the GGUF Engram table on every lookup."""

    def __init__(self, config, layer_id: int, n_hash_cols: int, head_dim: int):
        super().__init__()
        self.ck = checkpoint()
        self.tname = "per_layer_token_embd.weight"
        t = self.ck.tensors[self.tname]
        assert t.dims[0] == head_dim and t.type_name == "IQ4_NL", (t.dims, t.type_name)
        ids = list(config.engram_layer_ids)
        sizes = list(config.engram_num_embeddings)
        k = ids.index(layer_id)
        self.base = int(sum(sizes[:k]))
        self.n_rows = int(sizes[k])
        assert sum(sizes) == t.dims[1], (sizes, t.dims)
        self.row_bytes = t.row_bytes
        self.dim = head_dim
        self.n_hash_cols = n_hash_cols
        self.tp_size = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_rank()
        self.part_n_hash_cols = (n_hash_cols + self.tp_size - 1) // self.tp_size
        self.head_start = min(rank * self.part_n_hash_cols, n_hash_cols)
        self.head_end = min(self.head_start + self.part_n_hash_cols, n_hash_cols)
        self.dp_size = 1
        self.cpu_offload = False
        self.packed_fp4 = False
        self._table = None
        self._warned = False

    def _rows(self) -> np.ndarray:
        if self._table is None:
            mv = self.ck.rows_map(self.tname)
            self._table = np.frombuffer(mv, dtype=np.uint8).reshape(-1, self.row_bytes)
        return self._table

    @eager_break_during_capture
    def lookup(self, hash_ids: torch.Tensor, rows: torch.Tensor, background=False):
        """Writes into `rows` (the caller's persistent staging buffer); runs
        eagerly between graph segments, since it reads the file on the host."""
        from vllm.forward_context import get_forward_context, is_forward_context_available

        if not (is_forward_context_available()
                and isinstance(get_forward_context().attn_metadata, dict)):
            rows.zero_()  # profile / warm-up pass: no disk reads
            return
        local = hash_ids[:, self.head_start:self.head_end]
        n, c = local.shape
        if n == 0:
            return
        if c < rows.shape[1]:
            rows[:, c:].zero_()
        if c == 0:
            return
        flat = local.reshape(-1).cpu().numpy().astype(np.int64)
        bad = (flat < 0) | (flat >= self.n_rows)
        if bad.any():
            # graph capture runs this once on dummy hashes; real hashes are
            # always in range
            if not self._warned:
                logger.warning("DeepSeek V4.1 GGUF Engram: %d out-of-range hash ids "
                               "(zero rows used); expected only during graph capture",
                               int(bad.sum()))
                self._warned = True
            flat = np.where(bad, 0, flat)
        raw = torch.from_numpy(_engram_gather(self._rows(), flat + self.base)).to(rows.device)
        v = dequant_iq4_nl(raw, self.dim)
        if bad.any():
            v[torch.from_numpy(bad).to(v.device)] = 0
        rows[:, :c].copy_(v.view(n, c, self.dim).to(rows.dtype))
