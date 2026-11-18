# SPDX-License-Identifier: Apache-2.0
"""Optional per-section timer for single-token decode steps (VLLM_DS41_PROF=1).

Each section synchronizes the device before and after, so the totals add up
to the step time (with the syncs' own cost included). Rank 0 logs a summary
every VLLM_DS41_PROF_EVERY decode steps (default 50)."""
from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import contextmanager

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)
ENABLED = os.environ.get("VLLM_DS41_PROF") == "1"
_EVERY = int(os.environ.get("VLLM_DS41_PROF_EVERY", "50"))
_tot: dict[str, float] = defaultdict(float)
_state = {"steps": 0, "active": False, "t0": 0.0, "step_total": 0.0}


def begin_step(num_tokens: int) -> None:
    """Called at the start of the model forward."""
    if not ENABLED:
        return
    _state["active"] = num_tokens == 1
    if _state["active"]:
        torch.accelerator.synchronize()
        _state["t0"] = time.perf_counter()


def end_step() -> None:
    if not (ENABLED and _state["active"]):
        return
    torch.accelerator.synchronize()
    _state["step_total"] += time.perf_counter() - _state["t0"]
    _state["steps"] += 1
    n = _state["steps"]
    if n % _EVERY == 0:
        from vllm.distributed import get_tensor_model_parallel_rank

        if get_tensor_model_parallel_rank() == 0:
            step = _state["step_total"] / n * 1e3
            parts = ", ".join(f"{k} {v / n * 1e3:.1f}" for k, v in
                              sorted(_tot.items(), key=lambda kv: -kv[1]))
            covered = sum(_tot.values()) / n * 1e3
            logger.info("DS41PROF %d decode steps: %.1f ms/step; per step (ms): %s; "
                        "other %.1f", n, step, parts, step - covered)


@contextmanager
def section(name: str):
    if not (ENABLED and _state["active"]):
        yield
        return
    torch.accelerator.synchronize()
    t = time.perf_counter()
    try:
        yield
    finally:
        torch.accelerator.synchronize()
        _tot[name] += time.perf_counter() - t
