# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pad Qwen3.5/3.8 models at load time so tensor parallelism can split them.

Every dimension vLLM splits across GPUs must divide by the TP size. Qwen3.8-27B
has 4 KV heads, 16 linear-attention key heads and an MLP of 17408, so it can't
run at TP=3 as is. Setting ``VLLM_PAD_FOR_TP=<tp>`` appends zero-weight dummy
heads and MLP channels while the checkpoint is read, instead of writing a
padded copy of the model:

  full attention    query heads H -> H', KV heads KV -> KV' (each KV head keeps
                    its group of H/KV query heads; the dummies form whole groups)
  linear attention  key heads LK -> LK', value heads LV -> LV' (LV/LK per key)
  MLP               intermediate I -> I' (a multiple of tp * 128)

Dummy heads have zero projections in and out, so the model computes exactly
what it did. The padding is the same as host/tp3_pad/pad_qwen35_fp8.py (the
FP8 ``-tp3pad`` folder), so outputs match that folder. Supported weights:
unquantized, block FP8 (``weight_scale_inv``) and GPTQ-format INT4
(``qweight``/``scales``/``qzeros``/``g_idx``, e.g. AutoRound). The vision
encoder is not padded; run it with ``--mm-encoder-tp-mode data``.
"""

import math
import os
import re
from collections.abc import Iterable, Iterator
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ENV = "VLLM_PAD_FOR_TP"
PLAN_ATTR = "vllm_tp_pad_plan"
ALIGN = 128  # MLP / per-GPU alignment: FP8 blocks, GPTQ groups (<=128) and packing

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


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def requested_tp() -> int | None:
    v = os.environ.get(ENV, "").strip()
    if not v or v == "0":
        return None
    tp = int(v)
    if tp < 1:
        raise ValueError(f"{ENV} must be a positive integer, got {v!r}")
    return tp


def _quant_info(hf_config: Any) -> tuple[int | None, int | None]:
    """(fp8 block size, gptq group size) from the checkpoint's quant config."""
    q = getattr(hf_config, "quantization_config", None) or {}
    if not isinstance(q, dict):
        q = q.to_dict() if hasattr(q, "to_dict") else vars(q)
    block = None
    if q.get("quant_method") == "fp8" and q.get("weight_block_size"):
        bs = q["weight_block_size"]
        assert bs[0] == bs[1], f"non-square FP8 blocks {bs}"
        block = int(bs[0])
    group = int(q["group_size"]) if "group_size" in q and q.get("bits") else None
    return block, group


def maybe_pad_config(hf_config: Any, hf_text_config: Any) -> None:
    """If VLLM_PAD_FOR_TP is set, enlarge the head counts and MLP size in
    ``hf_text_config`` (in place) and remember the original sizes for the
    weight loader. Does nothing for models that already divide."""
    tp = requested_tp()
    if tp is None or getattr(hf_text_config, PLAN_ATTR, None) is not None:
        return
    missing = [k for k in _TEXT_KEYS if getattr(hf_text_config, k, None) is None]
    if missing:
        raise ValueError(
            f"{ENV} is set, but this model has no {missing} in its config; "
            "load-time padding supports Qwen3.5/3.8 (hybrid attention) models only."
        )
    c = {k: int(getattr(hf_text_config, k)) for k in _TEXT_KEYS}
    H, KV, D = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    LK, LV = c["linear_num_key_heads"], c["linear_num_value_heads"]
    DK, DV = c["linear_key_head_dim"], c["linear_value_head_dim"]
    inter = c["intermediate_size"]
    if H % KV or LV % LK:
        raise ValueError(f"{ENV}: unexpected head layout H={H} KV={KV} LK={LK} LV={LV}")
    KV2 = _round_up(KV, tp)
    H2 = KV2 * (H // KV)
    LK2 = _round_up(LK, tp)
    LV2 = LK2 * (LV // LK)
    inter2 = _round_up(inter, tp * ALIGN)
    if (H2, KV2, LK2, LV2, inter2) == (H, KV, LK, LV, inter):
        logger.info("%s=%d: model already divides by %d; no padding", ENV, tp, tp)
        return
    gate = 2 if getattr(hf_text_config, "attn_output_gate", True) else 1
    block, group = _quant_info(hf_config)
    unit = math.lcm(ALIGN, block or 1, group or 1)
    for name, v in (
        ("q rows", H2 * gate * D // tp),
        ("kv rows", KV2 * D // tp),
        ("o_proj in", H2 * D // tp),
        ("linear k", LK2 * DK // tp),
        ("linear v", LV2 * DV // tp),
        ("mlp", inter2 // tp),
    ):
        if v % unit:
            raise ValueError(f"{ENV}={tp}: {name} per GPU ({v}) is not a multiple of {unit}")
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
        "%s=%d: padding at load time: query heads %d->%d, KV heads %d->%d, "
        "linear key heads %d->%d, value heads %d->%d, MLP %d->%d",
        ENV, tp, H, H2, KV, KV2, LK, LK2, LV, LV2, inter, inter2,
    )


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


_SUFFIXES = (".weight_scale_inv", ".qweight", ".qzeros", ".scales", ".g_idx", ".weight", ".bias")


def _split(name: str) -> tuple[str, str]:
    for s in _SUFFIXES:
        if name.endswith(s):
            return name[: -len(s)], s
    return name, ""  # bare parameters such as A_log, dt_bias


def _pad(t: torch.Tensor, dim: int, segs: list[tuple[int, int]], fill) -> torch.Tensor:
    if sum(o for o, _ in segs) != t.shape[dim]:
        raise ValueError(f"{ENV}: shape {tuple(t.shape)} dim {dim} != segments {segs}")
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
            raise ValueError(f"{ENV}: segment {o}->{n} is not whole {what} ({unit})")
    return [(o // unit, n // unit) for o, n in segs]


def _pad_tensor(name: str, t: torch.Tensor, kind: str, segs, p: dict) -> torch.Tensor:
    base, suffix = _split(name)
    t = t.cpu() if t.device.type != "cpu" else t
    if suffix in ("", ".weight", ".bias"):
        # nn.Linear weight [out, in], conv1d [out, 1, k], 1-D params [out]
        dim = 0 if kind == "out" else 1
        return _pad(t, dim, segs, 0.0)
    if suffix == ".weight_scale_inv":  # block FP8 scales [out/B, in/B]
        b = p["block"]
        if b is None:
            raise ValueError(f"{ENV}: {name} but no FP8 block size in the config")
        return _pad(t, 0 if kind == "out" else 1, _scaled(segs, b, "FP8 blocks"), 1.0)
    group = p["group"]
    if group is None:
        raise ValueError(f"{ENV}: {name} but no quantization group size in the config")
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
            raise ValueError(f"{ENV}: {name} uses act-order (desc_act); not supported")
        new = old + sum(n - o for o, n in segs)
        return torch.arange(new, dtype=t.dtype) // group
    raise ValueError(f"{ENV}: don't know how to pad {name}")


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
    logger.info("%s: padded %d tensors while loading", ENV, padded)
