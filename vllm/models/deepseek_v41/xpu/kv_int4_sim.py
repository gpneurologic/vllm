"""Quality-only simulation of an INT4 KV cache for DeepSeek V4.1 on XPU.

VLLM_DS41_KV_INT4_SIM=<group> (32 or 64; unset/0 = off) rounds the 448 NoPE
values of every main-cache record (sliding-window and compressed caches) to
symmetric INT4 with one scale per `group` values (scale = amax / 7), just
before the normal fp8_ds_mla insert. The 64 RoPE values stay bf16, as they would
in a packed INT4 layout. Memory use does not change; this measures how much an
INT4 cache would cost in accuracy (needle test, perplexity) before building the
packed format and kernels. The indexer's 128-wide key cache is left alone.
"""

import os

import torch

GROUP = int(os.environ.get("VLLM_DS41_KV_INT4_SIM", "0") or 0)
NOPE = 448
LATENT = 512


def int4_roundtrip_(x: torch.Tensor) -> torch.Tensor:
    """In place: x[..., :448] -> INT4-representable values (per-group scales)."""
    if GROUP <= 0 or x.shape[-1] != LATENT or x.numel() == 0:
        return x
    v = x[..., :NOPE]
    g = v.float().reshape(*v.shape[:-1], NOPE // GROUP, GROUP)
    s = g.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 7.0
    q = torch.round(g / s).clamp_(-7, 7)
    v.copy_((q * s).reshape(v.shape).to(x.dtype))
    return x


def install() -> bool:
    """Wrap the compressor's main-cache insert; the SWA insert calls
    int4_roundtrip_ itself (attention_xpu.py)."""
    if GROUP <= 0:
        return False
    assert NOPE % GROUP == 0, f"VLLM_DS41_KV_INT4_SIM={GROUP} must divide {NOPE}"
    from vllm.models.deepseek_v41.compressor import DeepseekCompressor as _C

    if getattr(_C.insert_cache, "_int4_sim", False):
        return True
    orig = _C.insert_cache

    def insert_cache(self, latent, positions, rotary_emb):
        if latent is not None:
            latent = int4_roundtrip_(latent.clone())
        return orig(self, latent, positions, rotary_emb)

    insert_cache._int4_sim = True
    _C.insert_cache = insert_cache
    return True
