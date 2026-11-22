# SPDX-License-Identifier: Apache-2.0
"""ctypes binding for _ds41_iq_mmvq.so (csrc/xpu/ds41_iq_mmvq.cpp): llama.cpp-style
decode GEMV for IQ1_M / IQ2_XXS / IQ2_XS experts with q8_1 activations.

Build: tools/build_ds41_iq_mmvq.sh (oneAPI icpx). The library is looked for
next to this module, or at VLLM_DS41_IQ_LIB. Kernels run on the current torch
XPU stream's sycl::queue, so they work under graph capture."""
from __future__ import annotations

import ctypes
import os

import torch

_ERR = 256
_lib = None
Q8_BLOCK_BYTES = 36  # half d, half s, int8 qs[32]


def _path() -> str:
    return os.environ.get("VLLM_DS41_IQ_LIB") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "_ds41_iq_mmvq.so")


def available() -> bool:
    return os.path.exists(_path())


def lib():
    global _lib
    if _lib is None:
        L = ctypes.CDLL(_path())
        vp, i, i64, cp = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_char_p
        L.ds41_q8_1_quantize.argtypes = [vp, vp, i, vp, i, i, cp, i]
        L.ds41_q8_1_quantize.restype = i
        L.ds41_moe_mmvq.argtypes = [vp, i, vp, i64, i64, vp, i, i, vp, i, vp, i, i, i, cp, i]
        L.ds41_moe_mmvq.restype = i
        if hasattr(L, "ds41_moe_mmvq_grouped"):
            L.ds41_moe_mmvq_grouped.argtypes = L.ds41_moe_mmvq.argtypes
            L.ds41_moe_mmvq_grouped.restype = i
        if hasattr(L, "ds41_moe_mmvq2"):
            L.ds41_moe_mmvq2.argtypes = [vp, i, vp, i64, i64, vp, i, i, vp, i, vp, i, i, i, i, i,
                                         cp, i]
            L.ds41_moe_mmvq2.restype = i
        _lib = L
    return _lib


def _queue(t: torch.Tensor) -> int:
    return torch.xpu.current_stream(t.device).sycl_queue


def quantize_q8_1(x: torch.Tensor) -> torch.Tensor:
    """x [R, K] bf16/fp32 contiguous -> uint8 [R, K/32 * 36] q8_1 blocks."""
    R, K = x.shape
    assert x.is_contiguous() and K % 32 == 0 and x.dtype in (torch.bfloat16, torch.float32)
    out = torch.empty(R, K // 32 * Q8_BLOCK_BYTES, dtype=torch.uint8, device=x.device)
    err = ctypes.create_string_buffer(_ERR)
    rc = lib().ds41_q8_1_quantize(_queue(x), x.data_ptr(), int(x.dtype == torch.bfloat16),
                                  out.data_ptr(), R, K, err, _ERR)
    if rc:
        raise RuntimeError(f"ds41_q8_1_quantize: {err.value.decode()}")
    return out


def has_grouped() -> bool:
    return available() and hasattr(lib(), "ds41_moe_mmvq_grouped")


def has_rows() -> bool:
    return available() and hasattr(lib(), "ds41_moe_mmvq2")


def moe_mmvq(xq: torch.Tensor, K: int, w: torch.Tensor, qtype: int, idx_flat: torch.Tensor,
             e_start: int, n_local: int, xdiv: int, grouped: bool = False) -> torch.Tensor:
    """y [P, N] float32: row p = W[idx[p] - e_start] @ x[p // xdiv] for local experts,
    zeros otherwise. w [E, N, row_bytes] uint8; idx_flat [P] int32. grouped: pairs
    routed to the same expert share one read of its weights (several tokens per
    step, e.g. speculative-decoding verification); same result."""
    E, N, rb = w.shape
    P = idx_flat.shape[0]
    assert idx_flat.dtype == torch.int32 and idx_flat.is_contiguous()
    y = torch.empty(P, N, dtype=torch.float32, device=w.device)
    err = ctypes.create_string_buffer(_ERR)
    fn = lib().ds41_moe_mmvq_grouped if grouped else lib().ds41_moe_mmvq
    rc = fn(_queue(w), qtype, w.data_ptr(), w.stride(0), rb, idx_flat.data_ptr(),
                             e_start, n_local, xq.data_ptr(), xdiv, y.data_ptr(), P, N, K,
                             err, _ERR)
    if rc:
        raise RuntimeError(f"ds41_moe_mmvq{'_grouped' if grouped else ''}: {err.value.decode()}")
    return y


def moe_mmvq2(xq: torch.Tensor, K: int, w: torch.Tensor, qtype: int, idx_flat: torch.Tensor,
              e_start: int, n_local: int, xdiv: int, rows: int = 4,
              grouped: bool = False) -> torch.Tensor:
    """moe_mmvq with `rows` output rows per sub-group (activation reads shared)."""
    E, N, rb = w.shape
    P = idx_flat.shape[0]
    assert idx_flat.dtype == torch.int32 and idx_flat.is_contiguous()
    y = torch.empty(P, N, dtype=torch.float32, device=w.device)
    err = ctypes.create_string_buffer(_ERR)
    rc = lib().ds41_moe_mmvq2(_queue(w), qtype, w.data_ptr(), w.stride(0), rb, idx_flat.data_ptr(),
                              e_start, n_local, xq.data_ptr(), xdiv, y.data_ptr(), P, N, K, rows,
                              int(grouped), err, _ERR)
    if rc:
        raise RuntimeError(f"ds41_moe_mmvq2: {err.value.decode()}")
    return y
