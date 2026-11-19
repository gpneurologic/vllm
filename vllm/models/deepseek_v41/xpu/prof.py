# SPDX-License-Identifier: Apache-2.0
"""Optional per-section timer for single-token decode steps (VLLM_DS41_PROF=1).

Each section synchronizes the device before and after, so the totals add up
to the step time (with the syncs' own cost included). Sections nest: a
nested section's time is also inside its parent's, and summary names show
the path ("attention/o_proj"). With VLLM_DS41_PROF_OPS=1 a torch dispatch
mode also counts the torch ops each section runs (innermost section).
Rank 0 logs a summary every VLLM_DS41_PROF_EVERY decode steps (default 50).
Eager mode only (the syncs cannot be captured in a graph)."""
from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import contextmanager

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm.logger import init_logger

logger = init_logger(__name__)
ENABLED = os.environ.get("VLLM_DS41_PROF") == "1"
_COUNT_OPS = ENABLED and os.environ.get("VLLM_DS41_PROF_OPS") == "1"
_EVERY = int(os.environ.get("VLLM_DS41_PROF_EVERY", "50"))
_tot: dict[str, float] = defaultdict(float)
_ops: dict[str, int] = defaultdict(int)
_stack: list[str] = []
_state = {"steps": 0, "active": False, "t0": 0.0, "step_total": 0.0, "mode": None}


class _OpCounter(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        _ops["/".join(_stack) or "(outside sections)"] += 1
        return func(*args, **(kwargs or {}))


def begin_step(num_tokens: int) -> None:
    """Called at the start of the model forward."""
    if not ENABLED:
        return
    _state["active"] = num_tokens == 1 and not torch.xpu.is_current_stream_capturing()
    if _state["active"]:
        torch.accelerator.synchronize()
        _state["t0"] = time.perf_counter()
        if _COUNT_OPS:
            _state["mode"] = _OpCounter()
            _state["mode"].__enter__()


def end_step() -> None:
    if not (ENABLED and _state["active"]):
        return
    if _state["mode"] is not None:
        _state["mode"].__exit__(None, None, None)
        _state["mode"] = None
    torch.accelerator.synchronize()
    _state["step_total"] += time.perf_counter() - _state["t0"]
    _state["steps"] += 1
    n = _state["steps"]
    if n % _EVERY == 0:
        from vllm.distributed import get_tensor_model_parallel_rank

        if get_tensor_model_parallel_rank() == 0:
            step = _state["step_total"] / n * 1e3
            top = {k: v for k, v in _tot.items() if "/" not in k}
            parts = ", ".join(f"{k} {v / n * 1e3:.1f}" for k, v in
                              sorted(_tot.items(), key=lambda kv: -kv[1]))
            covered = sum(top.values()) / n * 1e3
            logger.info("DS41PROF %d decode steps: %.1f ms/step; per step (ms): %s; "
                        "other %.1f", n, step, parts, step - covered)
            if _ops:
                tot_ops = sum(_ops.values()) / n
                parts = ", ".join(f"{k} {v / n:.0f}" for k, v in
                                  sorted(_ops.items(), key=lambda kv: -kv[1]))
                logger.info("DS41PROF torch ops per decode step: %.0f; by section: %s",
                            tot_ops, parts)


@contextmanager
def section(name: str):
    if not (ENABLED and _state["active"]):
        yield
        return
    torch.accelerator.synchronize()
    _stack.append(name)
    key = "/".join(_stack)
    t = time.perf_counter()
    try:
        yield
    finally:
        torch.accelerator.synchronize()
        _tot[key] += time.perf_counter() - t
        _stack.pop()


def wrap_method(obj, method: str, name: str) -> None:
    """Time obj.method as section `name` (when profiling is on)."""
    if not ENABLED:
        return
    fn = getattr(obj, method)

    def wrapped(*args, **kwargs):
        with section(name):
            return fn(*args, **kwargs)

    setattr(obj, method, wrapped)
