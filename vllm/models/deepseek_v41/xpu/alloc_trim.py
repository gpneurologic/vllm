# SPDX-License-Identifier: Apache-2.0
"""Return the caching allocator's unused blocks to the driver during long prefills.

A long prompt is prefilled in fixed-size chunks, and several per-chunk buffers
(indexer logits, gathered KV) grow with the context read so far. Each chunk asks
for slightly larger blocks than the last, so the cached blocks of the previous
chunk cannot be reused: on rank 0 the reserved pool grew ~0.2 GiB per 1024-token
chunk while allocated memory stayed flat, until the driver ran out at ~59k
tokens (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY, or GPU page faults when the
driver's own allocations failed). Expandable segments would avoid this but
crash oneCCL's transport.

At the start of each prefill-sized step, torch.xpu.empty_cache() releases the
unused blocks when they exceed 0.25 GiB and the driver reports less than
VLLM_DS41_TRIM_FREE_GB (default 1.0) free, or when they exceed
VLLM_DS41_TRIM_CACHE_GB (default 0 = no such limit). VLLM_DS41_TRIM=0 turns
it off. Trimming whenever more than 1 GiB was unused made the driver move
buffers to system memory on prompt-logprob requests (large temporary logits
freed and re-allocated every step) until the container ran out of host RAM.
Launchers should also set PYTORCH_ALLOC_CONF=roundup_power2_divisions:8,
which rounds allocation sizes so the growing buffers keep reusing blocks
(reserved memory then stays flat to ~70k tokens) and the trim rarely runs. Graph pools are private and not affected; the
check is skipped while capturing.
"""
from __future__ import annotations

import os

import torch

_ON = os.environ.get("VLLM_DS41_TRIM", "1") == "1"
_LIMIT = float(os.environ.get("VLLM_DS41_TRIM_CACHE_GB", "0")) * 2**30
_LOW_FREE = float(os.environ.get("VLLM_DS41_TRIM_FREE_GB", "1.0")) * 2**30
_MIN_UNUSED = 0.25 * 2**30
_MIN_TOKENS = 256


def maybe_trim(num_tokens: int) -> None:
    if not _ON or num_tokens < _MIN_TOKENS or torch.xpu.is_current_stream_capturing():
        return
    unused = torch.xpu.memory_reserved() - torch.xpu.memory_allocated()
    if (_LIMIT > 0 and unused > _LIMIT) or (
            unused > _MIN_UNUSED and torch.xpu.mem_get_info()[0] < _LOW_FREE):
        torch.xpu.empty_cache()
