# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pad Qwen3.5/3.8 models at load time so tensor parallelism can split them.

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
folder), so outputs match that folder. Supported weights: unquantized, block
FP8 (``weight_scale_inv``) and GPTQ-format INT4 (``qweight``/``scales``/
``qzeros``/``g_idx``, e.g. AutoRound).

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
MODEL_TYPES = ("qwen3_5_text",)

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


def _quant_info(hf_config: Any) -> tuple[int | None, int | None, str | None]:
    """(FP8 block size, GPTQ group size, why unsupported) from the checkpoint's
    quant config. Supported: none, block FP8 (``fp8`` with weight_block_size, or
    compressed-tensors with block weights) and GPTQ-format INT4 (``gptq``,
    which AutoRound uses)."""
    q = getattr(hf_config, "quantization_config", None) or {}
    if not isinstance(q, dict):
        q = q.to_dict() if hasattr(q, "to_dict") else vars(q)
    method = q.get("quant_method")
    if not method:
        return None, None, None
    if method == "fp8":
        bs = q.get("weight_block_size")
        if not bs or bs[0] != bs[1]:
            return None, None, f"fp8 without square weight blocks ({bs})"
        return int(bs[0]), None, None
    if method == "compressed-tensors":
        blocks = set()
        for g in (q.get("config_groups") or {}).values():
            w = g.get("weights") or {}
            bs = w.get("block_structure")
            if w.get("strategy") != "block" or not bs or bs[0] != bs[1] or w.get("num_bits") != 8:
                return None, None, f"compressed-tensors weights {w.get('strategy')} {bs}"
            blocks.add(int(bs[0]))
        if len(blocks) != 1:
            return None, None, f"compressed-tensors block sizes {sorted(blocks)}"
        return blocks.pop(), None, None
    if method == "gptq" and q.get("bits") == 4 and not q.get("desc_act"):
        return None, int(q["group_size"]), None
    return None, None, f"quant_method={method}"


def maybe_pad_config(hf_config: Any, hf_text_config: Any, tp: int | None = None) -> None:
    """If the TP size doesn't divide this Qwen3.5/3.8 model, enlarge its head
    counts and MLP size in ``hf_text_config`` (in place) to the smallest sizes
    that do, and remember the original sizes for the weight loader."""
    tp = tp or _tp_size
    if not tp or tp == 1 or getattr(hf_text_config, PLAN_ATTR, None) is not None:
        return
    if getattr(hf_text_config, "model_type", None) not in MODEL_TYPES:
        return
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
    block, group, unsupported = _quant_info(hf_config)
    unit = math.lcm(ALIGN, block or 1, group or 1)  # whole FP8 blocks / GPTQ groups per GPU
    inter2 = inter if inter % (tp * unit) == 0 else _round_up(inter, tp * unit)
    if (H2, KV2, LK2, LV2, inter2) == (H, KV, LK, LV, inter):
        return  # already splits evenly
    if unsupported:
        raise ValueError(
            f"{TAG}: this model doesn't split across {tp} GPUs and would need padding, "
            f"but its weight format isn't supported for that ({unsupported}). "
            "Use a TP size that divides it, or a pre-padded checkpoint."
        )
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
        tp=tp, H=H, H2=H2, KV=KV, KV2=KV2, D=D, gate=gate, LK=LK, LK2=LK2,
        LV=LV, LV2=LV2, DK=DK, DV=DV, I=inter, I2=inter2, block=block, group=group,
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


_SUFFIXES = (
    ".weight_scale_inv", ".weight_scale", ".qweight", ".qzeros", ".scales", ".g_idx",
    ".weight", ".bias",
)


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
        # block FP8 scales [out/B, in/B] (fp8: weight_scale_inv, compressed-tensors:
        # weight_scale); 1-D conv/norm-like params never carry these
        b = p["block"]
        if b is None or t.dim() != 2:
            raise ValueError(f"{TAG}: {name} {tuple(t.shape)} is not a block-FP8 scale")
        return _pad(t, 0 if kind == "out" else 1, _scaled(segs, b, "FP8 blocks"), 1.0)
    group = p["group"]
    if group is None:
        raise ValueError(f"{TAG}: {name} but no quantization group size in the config")
    pack = 32 // 4  # 4-bit values per int32
    if suffix == ".qweight":  # [in/pack, out]
        if kind == "out":
            return _pad(t, 1, segs, 0)
        return _pad(t, 0, _scaled(segs, pack, "packed words"), 0)
    if suffix == ".scales":  # [in/group, out]
        if kind == "out":
            return _pad(t, 1, segs, 0.0)
        return _pad(t, 0, _scaled(segs, group, "groups"), 0.0)
    if suffix == ".qzeros":  # [in/group, out/pack]; symmetric => one constant
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
    padded = 0
    for name, t in weights:
        if "visual" not in name:
            base, _ = _split(name)
            for pat, kind, segs in rules:
                if pat.search(base):
                    t = _pad_tensor(name, t, kind, segs, plan)
                    padded += 1
                    break
        yield name, t
    logger.info("%s: padded %d tensors while loading", TAG, padded)
