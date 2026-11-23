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


# --- Per-token routing trace (VLLM_DS41_ROUTE_TRACE=<dir>, rank 0, eager only) ---
# Every real forward pass appends, per MoE layer, the routed ids [T, top_k] and
# the ids the next one and two layers' routers would pick from this layer's MoE
# input (look-ahead prediction for prefetching). Chunks of 8 forwards are
# saved as <dir>/trace_<n>.pt. Used offline (tools/ds41_route_stats.py) to
# size expert streaming: reuse between tokens, cache hit rates, bytes/token.
# Graph-captured steps are not traced: run with --enforce-eager. Off: no cost.
TRACE = os.environ.get("VLLM_DS41_ROUTE_TRACE")
_gates: dict[int, torch.nn.Module] = {}
_trace: dict = {"fwd": 0, "chunk": 0, "rows": [], "seen_layer": False}


def register_gate(layer: int, gate: torch.nn.Module) -> None:
    if TRACE:
        _gates[layer] = gate


def _predict(gate, x: torch.Tensor, k: int) -> torch.Tensor:
    s = torch.nn.functional.softplus(torch.nn.functional.linear(x.float(), gate.weight.float())).sqrt()
    return (s + gate.e_score_correction_bias).topk(k, dim=-1)[1]


def trace(layer: int, x: torch.Tensor, idx: torch.Tensor, real_step: bool) -> None:
    if not TRACE or not real_step or x.shape[0] == 0 or torch.xpu.is_current_stream_capturing():
        return
    from vllm.distributed import get_tensor_model_parallel_rank

    if get_tensor_model_parallel_rank() != 0:
        return
    k = idx.shape[-1]
    preds = [None, None]
    for j in (1, 2):
        g = _gates.get(layer + j)
        if g is not None:
            preds[j - 1] = _predict(g, x, k).to(torch.int16).cpu()
    _trace["rows"].append((_trace["fwd"], layer, idx.to(torch.int16).cpu(), preds[0], preds[1]))
    _trace["seen_layer"] = True


def trace_step() -> None:
    """Called once per forward pass (before the layers run)."""
    if not TRACE or not _trace["seen_layer"]:
        return
    _trace["seen_layer"] = False
    _trace["fwd"] += 1
    if _trace["fwd"] % 8 == 0:
        trace_flush()


def trace_flush() -> None:
    if not _trace["rows"]:
        return
    os.makedirs(TRACE, exist_ok=True)
    torch.save(_trace["rows"], os.path.join(TRACE, f"trace_{_trace['chunk']:05d}.pt"))
    _trace["chunk"] += 1
    _trace["rows"] = []


if TRACE:
    import atexit

    atexit.register(trace_flush)
