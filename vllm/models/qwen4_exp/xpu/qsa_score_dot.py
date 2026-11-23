# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tiled QSA indexer scoring on the XMX units (XPU).

Same output as `_qsa_mqa_paged_kernel` (score = sum over heads of
relu(q_h . k) / divisor, -inf where invisible), but one program scores
BLOCK_M query rows x BLOCK_N compressed keys with `tl.dot`, so each key tile
is loaded once for all rows and heads instead of once per row and head.
Rows of one request are contiguous in a batch; a tile that spans several
requests loops over them (usually one).
"""

import os

import torch

from vllm.triton_utils import tl, triton

ENABLED = os.environ.get("VLLM_QWEN4_QSA_SCORE_DOT", "1") == "1"
MIN_ROWS = int(os.environ.get("VLLM_QWEN4_QSA_SCORE_DOT_MIN_ROWS", "1"))


@triton.jit(do_not_specialize=["num_rows", "num_requests", "num_columns", "num_pages"])
def _qsa_score_dot_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    token_to_req_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_cache_block,
    stride_cache_token,
    stride_table_req,
    stride_logits_row,
    num_rows,
    num_columns,
    num_pages,
    num_requests,
    score_divisor,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
) -> None:
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    col_tile = tl.program_id(1)
    columns = col_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, HEAD_DIM)
    row_ok = rows < num_rows
    request = tl.load(token_to_req_ptr + rows, mask=row_ok, other=-1)
    req_ok = row_ok & (request >= 0) & (request < num_requests)
    safe_req = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    qpos = tl.load(query_positions_ptr + rows, mask=row_ok, other=0)
    seq_len = tl.load(sequence_lengths_ptr + safe_req, mask=req_ok, other=0)
    visible = tl.minimum((qpos + 1) // COMPRESS_RATIO, seq_len // COMPRESS_RATIO)
    visible = tl.where(req_ok, visible, 0)
    if col_tile == 0:
        tl.store(visible_blocks_ptr + rows, visible, mask=row_ok)

    score = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    valid = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    max_visible = tl.max(visible, axis=0)
    if col_tile * BLOCK_N < max_visible:
        req_lo = tl.min(tl.where(req_ok, request, num_requests), axis=0)
        req_hi = tl.max(tl.where(req_ok, request, -1), axis=0)
        logical_page = columns // PAGE_SIZE
        page_offset = columns % PAGE_SIZE
        q_base = q_ptr + rows[:, None] * stride_q_row + dims[None, :]
        for req in range(req_lo, req_hi + 1):
            col_ok = (columns < num_columns) & (logical_page < PAGE_TABLE_WIDTH)
            physical = tl.load(
                page_table_ptr
                + req * stride_table_req
                + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
                mask=col_ok,
                other=-1,
            )
            col_ok &= (physical >= 0) & (physical < num_pages)
            safe_page = tl.maximum(physical, 0).to(tl.int64)
            keys = tl.load(
                k_cache_ptr
                + safe_page[None, :] * stride_cache_block
                + page_offset[None, :] * stride_cache_token
                + dims[:, None],
                mask=col_ok[None, :],
                other=0.0,
            )
            in_req = req_ok & (request == req)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for head in tl.static_range(0, NUM_HEADS):
                query = tl.load(
                    q_base + head * stride_q_head, mask=in_req[:, None], other=0.0
                )
                acc += tl.maximum(tl.dot(query, keys), 0.0)
            here = in_req[:, None] & col_ok[None, :]
            score = tl.where(here, acc, score)
            valid = tl.where(here, 1, valid)
    keep = (valid != 0) & (columns[None, :] < visible[:, None])
    tl.store(
        logits_ptr + rows[:, None] * stride_logits_row + columns[None, :],
        tl.where(keep, score / score_divisor, -float("inf")),
        mask=row_ok[:, None] & (columns[None, :] < num_columns),
    )


PROFILE = None  # callable rows -> (block_m, block_n, warps), set by a sweep


def _default_profile(rows: int) -> tuple[int, int, int]:
    # B70 sweep (4 heads x 128, bf16 keys): 16 x 64 with 8 warps was best at
    # 5, 16 and 1024 rows; 1024 rows at 115k context 2.75 ms vs 10.6 ms for
    # the per-row kernel, 5 rows 0.031 vs 0.05 ms.
    return 16, 64, 8


def score(
    q, k_cache, page_table, token_to_req, query_positions, sequence_lengths,
    compress_ratio, columns, score_divisor, logits, visible_blocks,
) -> None:
    rows = q.shape[0]
    bm, bn, warps = (PROFILE or _default_profile)(rows)
    assert q.stride(2) == 1 and k_cache.stride(3) == 1 and page_table.stride(1) == 1
    _qsa_score_dot_kernel[(triton.cdiv(rows, bm), triton.cdiv(columns, bn))](
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        visible_blocks,
        logits,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        page_table.stride(0),
        logits.stride(0),
        rows,
        columns,
        k_cache.shape[0],
        page_table.shape[0],
        float(score_divisor),
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        BLOCK_M=bm,
        BLOCK_N=bn,
        COMPRESS_RATIO=compress_ratio,
        num_warps=warps,
    )


def usable(q: torch.Tensor, k_cache: torch.Tensor) -> bool:
    return (
        ENABLED
        and q.shape[0] >= MIN_ROWS
        and q.dtype == torch.bfloat16
        and k_cache.dtype == torch.bfloat16
        and q.shape[2] >= 16
        and q.shape[2] & (q.shape[2] - 1) == 0
        and q.stride(2) == 1
        and k_cache.stride(3) == 1
    )
