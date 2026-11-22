# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quality-only simulation of low-bit QSA K/V caches (no new kernels).

`VLLM_QWEN4_SIM_KV` rounds K and V through a low-bit format right before they
are written to the normal (bf16 or fp8) cache, so perplexity and recall show
what a real low-bit cache would cost before any kernel is written:

* ``off``: no rounding, but the unfused K/V write path (control run).
* ``int4g<N>``: symmetric INT4 per group of N dims, fp16 scale = amax / 7.
  The RoPE dims of K stay unrounded. ``int4g<N>all`` rounds them too.
* ``octave_k3v4`` / ``octave_k3v3`` / ``octave_k3v3_compact``: the Octave
  formats, through the PyTorch reference in ``rocm_octave``.

Setting it turns the fused QSA prepare off (that kernel writes the cache
itself). `VLLM_QWEN4_SIM_KV_ROPE` is the RoPE width of a K head (default 64).
"""

import os
import re

import torch

MODE = os.environ.get("VLLM_QWEN4_SIM_KV", "")
ENABLED = bool(MODE)
ROPE_DIM = int(os.environ.get("VLLM_QWEN4_SIM_KV_ROPE", "64"))


def _int4(x: torch.Tensor, group: int) -> torch.Tensor:
    shape = x.shape
    xf = x.float().reshape(*shape[:-1], -1, group)
    scale = (xf.abs().amax(-1, keepdim=True) / 7.0).clamp_min(1e-8)
    scale = scale.to(torch.float16).float()
    return ((xf / scale).round().clamp(-7, 7) * scale).reshape(shape)


def roundtrip(
    key: torch.Tensor, value: torch.Tensor, head_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Round (T, H, D) or (T, H*D) K/V through the simulated format."""
    if MODE in ("", "off") or key.numel() == 0:
        return key, value
    k = key.reshape(key.shape[0], -1, head_size)
    v = value.reshape(value.shape[0], -1, head_size)
    m = re.fullmatch(r"int4g(\d+)(all)?", MODE)
    if m:
        group = int(m.group(1))
        k2 = _int4(k, group)
        if not m.group(2):
            k2[..., :ROPE_DIM] = k[..., :ROPE_DIM].float()
        v2 = _int4(v, group)
    elif MODE.startswith("octave_"):
        from vllm.v1.attention.ops import rocm_octave as oc

        fmt = oc.OctaveFormat.from_cache_dtype(MODE, head_size, ROPE_DIM)
        k2, v2 = oc.reference_dequantize(
            oc.reference_quantize(k.float(), v.float(), fmt), fmt
        )
    else:
        raise ValueError(f"Unknown VLLM_QWEN4_SIM_KV={MODE!r}")
    return (
        k2.to(key.dtype).reshape(key.shape),
        v2.to(value.dtype).reshape(value.shape),
    )
