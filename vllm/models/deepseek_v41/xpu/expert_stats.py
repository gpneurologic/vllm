# SPDX-License-Identifier: Apache-2.0
"""Routed-expert usage counts (VLLM_DS41_EXPERT_STATS=<file.pt>, rank 0).

Every MoE layer adds its routing ids to a [layers, experts] count; the counts
are saved to the file every 50 forward passes. Used to plan which experts go on
which card (load balance across the tensor-parallel ranks). Off: no cost."""
from __future__ import annotations

import os

import torch

PATH = os.environ.get("VLLM_DS41_EXPERT_STATS")
_counts: dict[int, torch.Tensor] = {}
_state = {"steps": 0}


def add(layer: int, idx: torch.Tensor, n_experts: int) -> None:
    if not PATH or torch.xpu.is_current_stream_capturing():
        return
    c = _counts.get(layer)
    if c is None:
        c = _counts[layer] = torch.zeros(n_experts, dtype=torch.int64, device=idx.device)
    c += torch.bincount(idx.reshape(-1).long(), minlength=n_experts)[:n_experts]


def step() -> None:
    if not PATH or not _counts or torch.xpu.is_current_stream_capturing():
        return
    _state["steps"] += 1
    if _state["steps"] % 50:
        return
    from vllm.distributed import get_tensor_model_parallel_rank

    if get_tensor_model_parallel_rank() == 0:
        torch.save({k: v.cpu() for k, v in _counts.items()}, PATH)
