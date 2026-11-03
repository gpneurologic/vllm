# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pad Qwen3.5/3.8 and dense Gemma 4 models at load time so tensor parallelism
can split them.

Gemma 4 (e.g. 31B: 32 query heads, 16 KV heads on sliding layers, 4 KV heads
of size 512 on full-attention layers) gets one padded query-head count that
keeps whole KV groups for both layer types (TP=3: 48 query, 24 and 6 KV
heads). Its q/k norms are per head and its V norm has no weight, so zero
dummy heads stay zero. The Qwen layout follows.

Every dimension vLLM splits across GPUs must divide by the TP size. Qwen3.8-27B
has 4 KV heads, 16 linear-attention key heads and an MLP of 17408, so it can't
run at TP=3 as is. When the TP size doesn't divide the model, vLLM now appends
zero-weight dummy heads and MLP channels while the checkpoint is read, instead
of needing a padded copy of the model on disk:

  full attention    query heads H -> H', KV heads KV -> KV' (each KV head keeps
                    its group of H/KV query heads; the dummies form whole groups)
  linear attention  key heads LK -> LK', value heads LV -> LV' (LV/LK per key)
  MLP               intermediate I -> I' (a multiple of tp * 128)

The padding is the smallest that splits evenly, and it is decided from the
model's config and the TP size alone: nothing to configure. Models that
already divide (and other architectures) are left unchanged. Dummy heads have
zero projections in and out, so the model computes exactly what it did; the
layout is the same as host/tp3_pad/pad_qwen35_fp8.py (the FP8 ``-tp3pad``
folder), so outputs match that folder. Supported weights: unquantized; FP8
with block, per-channel or per-tensor scales (``fp8`` and compressed-tensors);
4-bit GPTQ (``gptq``) and AutoRound (``auto-round``, GPTQ or AWQ packing) and
AWQ. A 4-bit checkpoint packs 8 output channels per int32 word, so if the
linear-attention ``in_proj_a``/``in_proj_b`` layers (one channel per value
head) are quantized and their padded per-GPU size isn't a multiple of 8, they
are dequantized to plain weights while loading and served unquantized (they
are tiny: hidden x value heads).

The vision encoder is not padded. If its heads don't divide by the TP size
and no ``--mm-encoder-tp-mode`` was given, it runs whole on every GPU
(``data`` mode) instead; see pick_mm_encoder_tp_mode().
"""

import math
import re
from collections.abc import Iterable, Iterator
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

TAG = "TP padding"
PLAN_ATTR = "vllm_tp_pad_plan"
ALIGN = 128  # MLP / per-GPU alignment: FP8 blocks, GPTQ groups (<=128) and packing
# Text configs this padding knows the weight layout of (dense Qwen3.5/3.8).
MODEL_TYPES = ("qwen3_5_text", "gemma4_text")

_TEXT_KEYS = (
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "intermediate_size",
)

# The tensor-parallel size of the engine being configured. EngineArgs sets it
# before it builds the ModelConfigs (target and draft), which pad themselves
# right after reading the HF config, before anything derives sizes from it.
_tp_size: int | None = None


def set_tensor_parallel_size(tp: int | None) -> None:
    global _tp_size
    _tp_size = int(tp) if tp else None


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def _quant_dict(hf_config: Any) -> dict:
    q = getattr(hf_config, "quantization_config", None) or {}
    if not isinstance(q, dict):
        q = q.to_dict() if hasattr(q, "to_dict") else vars(q)
    return q


def _quant_info(hf_config: Any) -> dict:
    """What the loader needs to know about the checkpoint's weight format:
    ``int4`` (None, "gptq" or "awq": the packing of 4-bit weights), ``zero_offset``
    (GPTQ v1 stores zero points minus 1), ``block`` (FP8 block size, if any),
    ``group`` (4-bit group size) and ``unsupported`` (why it can't be padded)."""
    q = _quant_dict(hf_config)
    info = dict(int4=None, zero_offset=0, block=None, group=None, unsupported=None, method=None)
    method = q.get("quant_method")
    info["method"] = method
    if not method:
        return info
    if method == "fp8":
        bs = q.get("weight_block_size")
        if bs:
            if bs[0] != bs[1]:
                info["unsupported"] = f"fp8 with non-square weight blocks ({bs})"
            else:
                info["block"] = int(bs[0])
        return info  # without blocks: per-tensor (or per-channel) scales
    if method == "compressed-tensors":
        blocks = set()
        for g in (q.get("config_groups") or {}).values():
            w = g.get("weights") or {}
            strategy, bs = w.get("strategy"), w.get("block_structure")
            if w.get("num_bits") != 8 or w.get("type") not in (None, "float"):
                info["unsupported"] = (
                    f"compressed-tensors {w.get('num_bits')}-bit {w.get('type')} weights")
            elif strategy == "block":
                if not bs or bs[0] != bs[1]:
                    info["unsupported"] = f"compressed-tensors non-square blocks ({bs})"
                else:
                    blocks.add(int(bs[0]))
            elif strategy not in ("channel", "tensor"):
                info["unsupported"] = f"compressed-tensors {strategy} FP8 weights"
        if len(blocks) > 1:
            info["unsupported"] = f"compressed-tensors block sizes {sorted(blocks)}"
        elif blocks:
            info["block"] = blocks.pop()
        return info
    if method in ("gptq", "auto-round", "auto_round", "awq"):
        if q.get("bits") != 4:
            info["unsupported"] = f"{method} with {q.get('bits')}-bit weights"
        elif q.get("desc_act"):
            info["unsupported"] = f"{method} with act-order (desc_act)"
        else:
            if method == "awq":
                info["int4"] = "awq"
            elif method == "gptq":
                info["int4"] = "gptq"
            else:
                fmt = q.get("packing_format") or "auto_round:auto_gptq"
                info["int4"] = {"auto_round:auto_gptq": "gptq", "auto_round:auto_awq": "awq"}.get(fmt)
                if info["int4"] is None:
                    info["unsupported"] = f"auto-round packing {fmt}"
            # GPTQ v1 checkpoints (auto_gptq packing) store zero points minus 1
            if info["int4"] == "gptq" and q.get("checkpoint_format") != "gptq_v2":
                info["zero_offset"] = 1
            info["group"] = int(q.get("group_size") or 128)
            if info["group"] <= 0:
                info["unsupported"] = f"{method} without groups (group_size {q.get('group_size')})"
        return info
    info["unsupported"] = f"quant_method={method}"
    return info


_BA_PATTERN = r".*linear_attn\.in_proj_[ab].*"


def _serve_ba_unquantized(hf_config: Any, info: dict) -> None:
    """Tell vLLM's quant config to build linear_attn.in_proj_a/b (merged as
    in_proj_ba) unquantized; the loader dequantizes their weights."""
    q = getattr(hf_config, "quantization_config", None)
    if not isinstance(q, dict):
        raise ValueError(f"{TAG}: can't change the quantization config ({type(q).__name__})")
    method = info["method"]
    if method == "gptq":
        dyn = q.setdefault("dynamic", {})
        dyn.setdefault("-:" + _BA_PATTERN, {})
    elif method in ("auto-round", "auto_round"):
        extra = q.get("extra_config") or {}
        extra.setdefault(_BA_PATTERN, {"bits": 16, "data_type": "fp"})
        q["extra_config"] = extra
    elif method == "awq":
        skip = list(q.get("modules_to_not_convert") or [])
        for m in ("linear_attn.in_proj_a", "linear_attn.in_proj_b"):
            if m not in skip:
                skip.append(m)
        q["modules_to_not_convert"] = skip


def maybe_pad_config(hf_config: Any, hf_text_config: Any, tp: int | None = None) -> None:
    """If the TP size doesn't divide this Qwen3.5/3.8 model, enlarge its head
    counts and MLP size in ``hf_text_config`` (in place) to the smallest sizes
    that do, and remember the original sizes for the weight loader."""
    tp = tp or _tp_size
    if not tp or tp == 1 or getattr(hf_text_config, PLAN_ATTR, None) is not None:
        return
    if getattr(hf_text_config, "model_type", None) not in MODEL_TYPES:
        return
    if hf_text_config.model_type == "gemma4_text":
        return _pad_config_gemma4(hf_config, hf_text_config, tp)
    if any(getattr(hf_text_config, k, None) is None for k in _TEXT_KEYS):
        return
    c = {k: int(getattr(hf_text_config, k)) for k in _TEXT_KEYS}
    H, KV, D = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    LK, LV = c["linear_num_key_heads"], c["linear_num_value_heads"]
    DK, DV = c["linear_key_head_dim"], c["linear_value_head_dim"]
    inter = c["intermediate_size"]
    if H % KV or LV % LK:
        return  # not the layout this knows; let vLLM's own checks report it
    KV2 = _round_up(KV, tp)
    H2 = KV2 * (H // KV)
    LK2 = _round_up(LK, tp)
    LV2 = LK2 * (LV // LK)
    info = _quant_info(hf_config)
    block, group = info["block"], info["group"]
    unit = math.lcm(ALIGN, block or 1, group or 1)  # whole FP8 blocks / 4-bit groups per GPU
    inter2 = inter if inter % (tp * unit) == 0 else _round_up(inter, tp * unit)
    if (H2, KV2, LK2, LV2, inter2) == (H, KV, LK, LV, inter):
        return  # already splits evenly
    if info["unsupported"]:
        raise ValueError(
            f"{TAG}: this model doesn't split across {tp} GPUs and would need padding, "
            f"but its weight format isn't supported for that ({info['unsupported']}). "
            "Use a TP size that divides it, or a pre-padded checkpoint."
        )
    # 4-bit packing puts 8 output channels in a word: in_proj_a/b have one channel
    # per value head, which may not split into whole words per GPU
    dequant_ba = info["int4"] is not None and (LV2 // tp) % 8 != 0
    if dequant_ba:
        _serve_ba_unquantized(hf_config, info)
    gate = 2 if getattr(hf_text_config, "attn_output_gate", True) else 1
    for name, v in (
        ("q rows", H2 * gate * D // tp),
        ("kv rows", KV2 * D // tp),
        ("o_proj in", H2 * D // tp),
        ("linear k", LK2 * DK // tp),
        ("linear v", LV2 * DV // tp),
        ("mlp", inter2 // tp),
    ):
        if v % unit:
            raise ValueError(f"{TAG} for TP={tp}: {name} per GPU ({v}) is not a multiple of {unit}")
    plan = dict(
        arch="qwen3_5", tp=tp, H=H, H2=H2, KV=KV, KV2=KV2, D=D, gate=gate, LK=LK, LK2=LK2,
        LV=LV, LV2=LV2, DK=DK, DV=DV, I=inter, I2=inter2, block=block, group=group,
        int4=info["int4"], zero_offset=info["zero_offset"], dequant_ba=dequant_ba,
    )
    hf_text_config.num_attention_heads = H2
    hf_text_config.num_key_value_heads = KV2
    hf_text_config.linear_num_key_heads = LK2
    hf_text_config.linear_num_value_heads = LV2
    hf_text_config.intermediate_size = inter2
    setattr(hf_text_config, PLAN_ATTR, plan)
    logger.info(
        "%s: TP=%d doesn't divide the model, padding it while loading: query heads "
        "%d->%d, KV heads %d->%d, linear key heads %d->%d, value heads %d->%d, MLP %d->%d",
        TAG, tp, H, H2, KV, KV2, LK, LK2, LV, LV2, inter, inter2,
    )
    if dequant_ba:
        logger.info(
            "%s: linear_attn.in_proj_a/b (%d value heads per GPU) can't be split in whole "
            "4-bit words; serving them unquantized", TAG, LV2 // tp,
        )


def _pad_config_gemma4(hf_config: Any, hf_text_config: Any, tp: int) -> None:
    """Gemma 4 (dense): sliding layers have ``num_key_value_heads`` KV heads of
    ``head_dim``; full-attention layers have ``global_head_dim`` heads and, with
    ``attention_k_eq_v``, ``num_global_key_value_heads`` KV heads (K is also V).
    All layers share ``num_attention_heads`` query heads, so one padded query
    count must keep whole KV groups for both layer types."""
    t = hf_text_config
    if getattr(t, "enable_moe_block", False) or getattr(t, "num_experts", None):
        return  # MoE variants: not handled here; vLLM's own checks report it
    H = int(t.num_attention_heads)
    KVs = int(t.num_key_value_heads)
    D = int(t.head_dim)
    Dg = int(getattr(t, "global_head_dim", None) or D)
    k_eq_v = bool(getattr(t, "attention_k_eq_v", False))
    KVg = int(getattr(t, "num_global_key_value_heads", None) or KVs) if k_eq_v else KVs
    inter = int(t.intermediate_size)
    layer_types = list(getattr(t, "layer_types", None) or [])
    if not layer_types or H % KVs or H % KVg:
        return
    groups = {H // KVs, H // KVg}
    H2 = H
    while H2 % tp or any(H2 % g or (H2 // g) % tp for g in groups):
        H2 += 1
    KVs2, KVg2 = H2 // (H // KVs), H2 // (H // KVg)
    info = _quant_info(hf_config)
    block, group = info["block"], info["group"]
    unit = math.lcm(ALIGN, block or 1, group or 1)
    inter2 = inter if inter % (tp * unit) == 0 else _round_up(inter, tp * unit)
    if (H2, KVs2, KVg2, inter2) == (H, KVs, KVg, inter):
        return
    if info["unsupported"]:
        raise ValueError(
            f"{TAG}: this model doesn't split across {tp} GPUs and would need padding, "
            f"but its weight format isn't supported for that ({info['unsupported']}). "
            "Use a TP size that divides it, or a pre-padded checkpoint."
        )
    for name, v in (
        ("q rows", H2 * D // tp), ("global q rows", H2 * Dg // tp),
        ("kv rows", KVs2 * D // tp), ("global kv rows", KVg2 * Dg // tp),
        ("mlp", inter2 // tp),
    ):
        if v % unit:
            raise ValueError(f"{TAG} for TP={tp}: {name} per GPU ({v}) is not a multiple of {unit}")
    plan = dict(
        arch="gemma4", tp=tp, H=H, H2=H2, KVs=KVs, KVs2=KVs2, KVg=KVg, KVg2=KVg2, D=D, Dg=Dg,
        k_eq_v=k_eq_v, layer_types=layer_types, I=inter, I2=inter2, block=block, group=group,
        int4=info["int4"], zero_offset=info["zero_offset"], dequant_ba=False,
    )
    t.num_attention_heads = H2
    t.num_key_value_heads = KVs2
    if k_eq_v:
        t.num_global_key_value_heads = KVg2
    t.intermediate_size = inter2
    setattr(t, PLAN_ATTR, plan)
    logger.info(
        "%s: TP=%d doesn't divide the model, padding it while loading: query heads %d->%d, "
        "KV heads %d->%d, global KV heads %d->%d, MLP %d->%d",
        TAG, tp, H, H2, KVs, KVs2, KVg, KVg2, inter, inter2,
    )


def pick_mm_encoder_tp_mode(hf_config: Any, requested: str | None) -> str | None:
    """``data`` when the vision encoder's heads don't divide by the TP size and
    no mode was asked for explicitly; otherwise ``requested`` unchanged."""
    tp = _tp_size
    vision = getattr(hf_config, "vision_config", None)
    heads = getattr(vision, "num_heads", None) or getattr(vision, "num_attention_heads", None)
    if not tp or tp == 1 or not heads or heads % tp == 0:
        return requested
    if requested not in (None, "weights"):
        return requested  # e.g. already "data"
    # "weights" (the default) would split the heads across GPUs, which can't work
    logger.info(
        "%s: the vision encoder's %d heads don't divide by TP=%d; running it whole "
        "on every GPU (mm_encoder_tp_mode=data)", TAG, heads, tp,
    )
    return "data"


def get_plan(hf_text_config: Any) -> dict | None:
    return getattr(hf_text_config, PLAN_ATTR, None)


# --------------------------------------------------------------------------
# Weights

def _rules(p: dict) -> list[tuple[str, str, list[tuple[int, int]]]]:
    """(regex on the parameter name without its suffix, 'out' | 'in', segments).
    'out' pads output channels (rows of an nn.Linear weight), 'in' pads input
    channels. Each (old, new) segment is padded at its end."""
    if p.get("arch") == "gemma4":
        return _rules_gemma4(p)
    H, H2, KV, KV2, D, g = p["H"], p["H2"], p["KV"], p["KV2"], p["D"], p["gate"]
    LK, LK2, LV, LV2, DK, DV = p["LK"], p["LK2"], p["LV"], p["LV2"], p["DK"], p["DV"]
    lin_qkv = [(LK * DK, LK2 * DK), (LK * DK, LK2 * DK), (LV * DV, LV2 * DV)]
    return [
        (r"\.self_attn\.q_proj$", "out", [(H * g * D, H2 * g * D)]),
        (r"\.self_attn\.[kv]_proj$", "out", [(KV * D, KV2 * D)]),
        (r"\.self_attn\.o_proj$", "in", [(H * D, H2 * D)]),
        (r"\.linear_attn\.in_proj_qkv$", "out", lin_qkv),
        (r"\.linear_attn\.in_proj_z$", "out", [(LV * DV, LV2 * DV)]),
        (r"\.linear_attn\.in_proj_[ab]$", "out", [(LV, LV2)]),
        (r"\.linear_attn\.(A_log|dt_bias)$", "out", [(LV, LV2)]),
        (r"\.linear_attn\.conv1d$", "out", lin_qkv),
        (r"\.linear_attn\.out_proj$", "in", [(LV * DV, LV2 * DV)]),
        (r"\.mlp\.(gate|up)_proj$", "out", [(p["I"], p["I2"])]),
        (r"\.mlp\.down_proj$", "in", [(p["I"], p["I2"])]),
    ]


def _rules_gemma4(p: dict) -> list[tuple[str, str, list[tuple[int, int]]]]:
    """Per-layer rules: sliding and full-attention layers differ in head size and
    KV heads. Only the language model's layers (not the vision tower)."""
    H, H2 = p["H"], p["H2"]
    rules = []
    for i, lt in enumerate(p["layer_types"]):
        full = lt == "full_attention"
        d = p["Dg"] if full else p["D"]
        kv, kv2 = (p["KVg"], p["KVg2"]) if full and p["k_eq_v"] else (p["KVs"], p["KVs2"])
        pre = rf"language_model\.(?:model\.)?layers\.{i}\.self_attn\."
        rules += [
            (pre + r"q_proj$", "out", [(H * d, H2 * d)]),
            (pre + r"[kv]_proj$", "out", [(kv * d, kv2 * d)]),
            (pre + r"o_proj$", "in", [(H * d, H2 * d)]),
        ]
    if p["I2"] != p["I"]:
        rules += [
            (r"language_model\.(?:model\.)?layers\.\d+\.mlp\.(gate|up)_proj$", "out", [(p["I"], p["I2"])]),
            (r"language_model\.(?:model\.)?layers\.\d+\.mlp\.down_proj$", "in", [(p["I"], p["I2"])]),
        ]
    return rules


_SUFFIXES = (
    ".weight_scale_inv", ".weight_scale", ".input_scale", ".qweight", ".qzeros", ".scales",
    ".g_idx", ".weight", ".bias",
)

# AWQ packs output channels in this order within each int32 word
_AWQ_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)


def _unpack(packed: torch.Tensor, dim: int, order=None) -> torch.Tensor:
    """int32 words of 8 4-bit values -> int values, unpacked along ``dim``."""
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    x = packed.to(torch.int32).unsqueeze(-1)  # [..., 8]
    vals = (x >> shifts) & 0xF
    if order is not None:  # nibble i holds channel order[i]
        out = torch.empty_like(vals)
        out[..., list(order)] = vals
        vals = out
    if dim == 0:  # [rows, cols, 8] -> [rows*8, cols]
        return vals.permute(0, 2, 1).reshape(packed.shape[0] * 8, packed.shape[1])
    return vals.reshape(packed.shape[0], packed.shape[1] * 8)


def dequantize_int4(parts: dict, p: dict) -> torch.Tensor:
    """qweight/qzeros/scales of one 4-bit layer -> float weight [out, in]."""
    group, awq = p["group"], p.get("int4") == "awq"
    qw, qz, s = parts[".qweight"], parts[".qzeros"], parts[".scales"]
    order = _AWQ_ORDER if awq else None
    q = _unpack(qw, 1, order) if awq else _unpack(qw, 0)  # [in, out]
    z = _unpack(qz, 1, order) + p.get("zero_offset", 0)  # [in/group, out]
    rows = torch.arange(q.shape[0]) // group
    w = (q - z[rows]).to(torch.float32) * s.to(torch.float32)[rows]
    return w.t().contiguous().to(s.dtype)


def _split(name: str) -> tuple[str, str]:
    for s in _SUFFIXES:
        if name.endswith(s):
            return name[: -len(s)], s
    return name, ""  # bare parameters such as A_log, dt_bias


def _pad(t: torch.Tensor, dim: int, segs: list[tuple[int, int]], fill) -> torch.Tensor:
    if sum(o for o, _ in segs) != t.shape[dim]:
        raise ValueError(f"{TAG}: shape {tuple(t.shape)} dim {dim} != segments {segs}")
    fp8 = t.dtype == torch.float8_e4m3fn
    src = t.view(torch.uint8) if fp8 else t  # byte 0 == +0.0 in e4m3
    parts, start = [], 0
    for old, new in segs:
        parts.append(src.narrow(dim, start, old))
        if new > old:
            shape = list(src.shape)
            shape[dim] = new - old
            parts.append(torch.full(shape, 0 if fp8 else fill, dtype=src.dtype))
        start += old
    out = torch.cat(parts, dim=dim).contiguous()
    return out.view(torch.float8_e4m3fn) if fp8 else out


def _scaled(segs: list[tuple[int, int]], unit: int, what: str) -> list[tuple[int, int]]:
    for o, n in segs:
        if o % unit or n % unit:
            raise ValueError(f"{TAG}: segment {o}->{n} is not whole {what} ({unit})")
    return [(o // unit, n // unit) for o, n in segs]


def _pad_tensor(name: str, t: torch.Tensor, kind: str, segs, p: dict) -> torch.Tensor:
    base, suffix = _split(name)
    t = t.cpu() if t.device.type != "cpu" else t
    if suffix in ("", ".weight", ".bias"):
        # nn.Linear weight [out, in], conv1d [out, 1, k], 1-D params [out]
        dim = 0 if kind == "out" else 1
        return _pad(t, dim, segs, 0.0)
    if suffix in (".weight_scale_inv", ".weight_scale"):
        # FP8 weight scales (fp8: weight_scale_inv or weight_scale, compressed-tensors:
        # weight_scale). Padded rows/columns hold zero weights, so their scale
        # only needs to be finite.
        if t.numel() == 1:
            return t  # per-tensor
        if t.dim() == 1 or (t.dim() == 2 and t.shape[1] == 1):
            # per-channel [out] or [out, 1]: one scale per output row
            return _pad(t, 0, segs, 1.0) if kind == "out" else t
        b = p["block"]
        if b is None or t.dim() != 2:
            raise ValueError(f"{TAG}: {name} {tuple(t.shape)} is not an FP8 scale this knows")
        # block scales [out/B, in/B]
        return _pad(t, 0 if kind == "out" else 1, _scaled(segs, b, "FP8 blocks"), 1.0)
    if suffix == ".input_scale":
        return t  # static activation scale: one value
    group = p["group"]
    if group is None:
        raise ValueError(f"{TAG}: {name} but no quantization group size in the config")
    pack = 32 // 4  # 4-bit values per int32
    awq = p.get("int4") == "awq"
    if suffix == ".qweight":
        if awq:  # [in, out/pack]
            if kind == "out":
                return _pad(t, 1, _scaled(segs, pack, "packed words"), 0)
            return _pad(t, 0, segs, 0)
        # GPTQ [in/pack, out]
        if kind == "out":
            return _pad(t, 1, segs, 0)
        return _pad(t, 0, _scaled(segs, pack, "packed words"), 0)
    if suffix == ".scales":  # [in/group, out]
        if kind == "out":
            return _pad(t, 1, segs, 0.0)
        return _pad(t, 0, _scaled(segs, group, "groups"), 0.0)
    if suffix == ".qzeros":  # [in/group, out/pack] (GPTQ and AWQ); padded scales are 0
        fill = int(t.reshape(-1)[0])
        if kind == "out":
            return _pad(t, 1, _scaled(segs, pack, "packed words"), fill)
        return _pad(t, 0, _scaled(segs, group, "groups"), fill)
    if suffix == ".g_idx":  # [in]
        if kind == "out":
            return t
        old = t.shape[0]
        if not torch.equal(t, torch.arange(old, dtype=t.dtype) // group):
            raise ValueError(f"{TAG}: {name} uses act-order (desc_act); not supported")
        new = old + sum(n - o for o, n in segs)
        return torch.arange(new, dtype=t.dtype) // group
    raise ValueError(f"{TAG}: don't know how to pad {name}")


def pad_weights(
    weights: Iterable[tuple[str, torch.Tensor]], plan: dict
) -> Iterator[tuple[str, torch.Tensor]]:
    """Pad checkpoint tensors on the fly according to ``plan``."""
    rules = [(re.compile(pat), kind, segs) for pat, kind, segs in _rules(plan)]
    ba = re.compile(r"\.linear_attn\.in_proj_[ab]$")
    pending: dict[str, dict] = {}  # 4-bit in_proj_a/b parts waiting for the rest
    padded = dequantized = 0

    def pad(name, t):
        nonlocal padded
        base, _ = _split(name)
        for pat, kind, segs in rules:
            if pat.search(base):
                padded += 1
                return _pad_tensor(name, t, kind, segs, plan)
        return t

    for name, t in weights:
        if "visual" in name:
            yield name, t
            continue
        base, suffix = _split(name)
        if plan.get("dequant_ba") and ba.search(base) and suffix in (
            ".qweight", ".qzeros", ".scales", ".g_idx"
        ):
            parts = pending.setdefault(base, {})
            parts[suffix] = t.cpu()
            if all(k in parts for k in (".qweight", ".qzeros", ".scales")):
                g_idx = parts.get(".g_idx")
                if g_idx is not None and not torch.equal(
                    g_idx, torch.arange(g_idx.shape[0], dtype=g_idx.dtype) // plan["group"]
                ):
                    raise ValueError(f"{TAG}: {base}.g_idx uses act-order; not supported")
                w = dequantize_int4(parts, plan)
                parts["done"] = True
                dequantized += 1
                yield base + ".weight", pad(base + ".weight", w)
            continue  # g_idx arriving after the rest is dropped too
        yield name, pad(name, t)
    left = [b for b, parts in pending.items() if not parts.get("done")]
    if left:
        raise ValueError(f"{TAG}: incomplete 4-bit weights for {left[:3]}")
    logger.info("%s: padded %d tensors while loading", TAG, padded)
    if dequantized:
        logger.info("%s: dequantized %d in_proj_a/b layers", TAG, dequantized)
