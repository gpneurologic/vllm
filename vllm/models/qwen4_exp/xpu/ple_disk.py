# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp n-gram (PLE) table served from safetensors files on XPU.

The table is far larger than a card (about 51 B values) while a token reads
only `ngram_heads` rows of it. `Qwen4ExpPLEDiskEmbedding` therefore allocates
no table: before each step the model state hashes the tokens of that step into
row ids, reads those rows on the host and copies them, dequantized, into a
fixed device buffer. The forward only reads that buffer, so graph replays work.

The table lives in its own directory (`VLLM_QWEN4_EXP_PLE_TABLE_DIR`), found by
tensor name, so it can come from a different checkpoint than the weights.
"""

import glob
import json
import os
import struct
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from vllm.logger import init_logger

from ..common.ngram_embedding import Qwen4ExpPLEDeviceEmbedding

logger = init_logger(__name__)

# Rows of a step are read by this many threads (os.preadv releases the GIL).
# A new prompt reads thousands of rows that are mostly not in the page cache,
# so serial reads wait on the disk one row at a time. 1 = serial reads.
_READ_THREADS = int(os.environ.get("VLLM_QWEN4_EXP_PLE_THREADS", "32"))
# Steps with fewer read runs than this stay serial (decode: a few dozen rows).
_PARALLEL_MIN_RUNS = int(os.environ.get("VLLM_QWEN4_EXP_PLE_PARALLEL_MIN", "64"))
# "random" disables kernel read-ahead on the table files (rows are scattered).
_FADVISE = os.environ.get("VLLM_QWEN4_EXP_PLE_FADVISE", "")
# How a step's rows reach the device buffer:
#   host  - decode FP8 -> bf16 on the CPU, copy bf16 (the original path)
#   gpu   - copy the raw FP8 bytes through a pinned buffer, decode on the device
#   split - like gpu, but each TP rank reads only its share of the n-gram heads
#           and the ranks all-gather the decoded rows (every rank used to read
#           every row, as DeepSeek's Engram reads did before its column split)
_STAGE_MODE = os.environ.get("VLLM_QWEN4_EXP_PLE_STAGE", "split")
# Compare the chosen mode with the host path for this many steps (raises on a
# mismatch). 0 = off.
_STAGE_CHECKS = int(os.environ.get("VLLM_QWEN4_EXP_PLE_CHECK", "0"))
# Quality-only: round every decoded row through symmetric INT4 per group of N
# values (fp16 scale = amax / 7), as an INT4 copy of the table would store it.
_SIM_INT4 = int(os.environ.get("VLLM_QWEN4_EXP_PLE_SIM_INT4", "0"))


def _sim_int4(rows: torch.Tensor) -> torch.Tensor:
    """float32 [n, D] -> the same rows after an INT4 round trip."""
    if not _SIM_INT4:
        return rows
    n, d = rows.shape
    g = rows.view(n, d // _SIM_INT4, _SIM_INT4)
    s = (g.abs().amax(-1, keepdim=True) / 7.0).clamp_min(1e-12)
    s = s.to(torch.float16).float()
    return ((g / s).round().clamp(-7, 7) * s).view(n, d)

_SAFETENSORS_DTYPES = {
    "F8_E4M3": torch.float8_e4m3fn,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
}


class PLETableFiles:
    """Row reader over `...ngram_embedding.shard_<i>.weight` tensors on disk."""

    def __init__(self, table_dir: str, row_dim: int) -> None:
        shards: dict[int, tuple[str, int, int, torch.dtype]] = {}
        scale: float | None = None
        paths = sorted(glob.glob(os.path.join(table_dir, "*.safetensors")))
        if not paths:
            raise ValueError(f"No safetensors files in PLE table dir {table_dir}")
        for path in paths:
            with open(path, "rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_len))
                base = 8 + header_len
                for name, info in header.items():
                    if name == "__metadata__" or ".ngram_embedding." not in name:
                        continue
                    leaf = name.split(".ngram_embedding.", 1)[1]
                    start, end = info["data_offsets"]
                    dtype = _SAFETENSORS_DTYPES[info["dtype"]]
                    if leaf == "weight_scale":
                        f.seek(base + start)
                        raw = bytearray(f.read(end - start))
                        scale = float(torch.frombuffer(raw, dtype=dtype).float()[0])
                    elif leaf.startswith("shard_") and leaf.endswith(".weight"):
                        rows, dim = info["shape"]
                        if dim != row_dim:
                            raise ValueError(
                                f"PLE shard {name} has row width {dim}, "
                                f"expected {row_dim}"
                            )
                        shards[int(leaf[6:-7])] = (path, base + start, rows, dtype)
        if not shards or sorted(shards) != list(range(len(shards))):
            raise ValueError(f"PLE table in {table_dir} has missing shards")
        self.dtype = shards[0][3]
        self.shard_size = shards[0][2]
        self.row_bytes = row_dim * self.dtype.itemsize
        self.num_rows = sum(s[2] for s in shards.values())
        fds: dict[str, int] = {}
        self._fds: list[int] = []
        self._bases: list[int] = []
        for index in range(len(shards)):
            path, offset, rows, dtype = shards[index]
            if dtype != self.dtype or (
                rows != self.shard_size and index != len(shards) - 1
            ):
                raise ValueError(f"PLE table in {table_dir} has uneven shards")
            if path not in fds:
                fds[path] = os.open(path, os.O_RDONLY)
                if _FADVISE == "random":
                    os.posix_fadvise(fds[path], 0, 0, os.POSIX_FADV_RANDOM)
            self._fds.append(fds[path])
            self._bases.append(offset)
        if self.dtype == torch.float8_e4m3fn and scale is None:
            raise ValueError(f"FP8 PLE table in {table_dir} has no weight_scale")
        self.scale = 1.0 if scale is None else scale
        self._pool: ThreadPoolExecutor | None = None
        logger.info(
            "PLE table on disk: %s, %d rows x %d, %s, %d shards, scale %.6g, "
            "%d read threads",
            table_dir,
            self.num_rows,
            row_dim,
            self.dtype,
            len(shards),
            self.scale,
            _READ_THREADS,
        )

    def read(self, ids: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """Return the raw bytes of the rows `ids` (CPU int64) as [n, row_bytes].

        Consecutive ids in one shard are read with a single call; when a step
        has many such runs they are spread over a thread pool. `out` (CPU
        uint8, at least [n, row_bytes], e.g. pinned) receives the rows."""
        n = ids.numel()
        if out is None:
            out = torch.empty(n, self.row_bytes, dtype=torch.uint8)
        else:
            out = out[:n]
        if n == 0:
            return out
        rows = out.numpy()
        idx = ids.numpy()
        shard = idx // self.shard_size
        local = idx - shard * self.shard_size
        new_run = np.ones(n, dtype=bool)
        new_run[1:] = (shard[1:] != shard[:-1]) | (local[1:] != local[:-1] + 1)
        starts = np.flatnonzero(new_run)
        ends = np.append(starts[1:], n)
        row_bytes = self.row_bytes
        fds, bases = self._fds, self._bases

        def read_runs(first: int, last: int) -> None:
            for r in range(first, last):
                a, b = int(starts[r]), int(ends[r])
                s = int(shard[a])
                pos = int(local[a])
                buf = rows[a:b].reshape(-1)
                if os.preadv(fds[s], [buf], bases[s] + pos * row_bytes) != buf.size:
                    raise ValueError(f"PLE table shard {s} is truncated at row {pos}")

        num_runs = len(starts)
        if _READ_THREADS <= 1 or num_runs < _PARALLEL_MIN_RUNS:
            read_runs(0, num_runs)
            return out
        if self._pool is None:
            self._pool = ThreadPoolExecutor(_READ_THREADS, thread_name_prefix="ple-read")
        parts = min(num_runs // 16, _READ_THREADS * 4)
        bounds = np.linspace(0, num_runs, parts + 1, dtype=np.int64)
        futures = [
            self._pool.submit(read_runs, int(bounds[i]), int(bounds[i + 1]))
            for i in range(parts)
        ]
        for f in futures:
            f.result()
        return out


class Qwen4ExpPLEDiskEmbedding(Qwen4ExpPLEDeviceEmbedding):
    """PLE rows staged per step from table files; no table is allocated."""

    def __init__(self, *args, table_dir: str, **kwargs) -> None:
        self._num_heads = int(kwargs.get("num_ngram_heads", 1))
        self._capacity = int(kwargs.get("max_total_tokens", 0))
        super().__init__(*args, **kwargs)
        self._table = PLETableFiles(table_dir, self.weight.shape[1])
        self._staging: torch.Tensor | None = None
        self._pinned: torch.Tensor | None = None
        self._checks_left = _STAGE_CHECKS
        # split: this rank reads heads [h0, h1); all ranks gather [T, tp*per, D]
        # and `index` picks the real heads back out of the padded layout.
        self._split: tuple[int, int, int, torch.Tensor] | None = None
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        tp = get_tensor_model_parallel_world_size()
        if tp > 1 and _STAGE_MODE == "split":
            heads = self._num_heads
            per = -(-heads // tp)
            rank = get_tensor_model_parallel_rank()
            h0, h1 = min(rank * per, heads), min((rank + 1) * per, heads)
            index = torch.tensor(
                [r * per + j for r in range(tp)
                 for j in range(min((r + 1) * per, heads) - min(r * per, heads))],
                dtype=torch.long,
                device=self.weight.device,
            )
            self._split = (h0, h1, per, index)
            logger.info("PLE rows: rank %d reads heads %d..%d of %d", rank, h0, h1 - 1, heads)

    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # Placeholder row: keeps the dtype, width and device of the parameter.
        return torch.zeros(1, embedding_dim, dtype=dtype)

    def stage(self, ngram_ids: torch.Tensor) -> None:
        """Read the rows of `ngram_ids` ([tokens, heads]) into the device buffer."""
        num_tokens = ngram_ids.shape[0]
        if self._staging is None:
            capacity = max(self._capacity, num_tokens)
            self._staging = torch.zeros(
                (capacity, self._num_heads, self.weight.shape[1]),
                dtype=self.weight.dtype,
                device=self.weight.device,
            )
        if num_tokens > self._staging.shape[0]:
            raise RuntimeError(
                f"PLE staging holds {self._staging.shape[0]} tokens, "
                f"step has {num_tokens}"
            )
        if num_tokens == 0:
            return
        ids = ngram_ids.detach().to("cpu")
        mode = _STAGE_MODE
        if mode == "split" and self._split is None:
            mode = "gpu"
        if mode == "host":
            self._staging[:num_tokens].copy_(self._decode_host(ids))
        elif mode == "gpu":
            self._staging[:num_tokens].copy_(self._decode_gpu(ids))
        else:
            h0, h1, per, index = self._split
            local = self._decode_gpu(ids[:, h0:h1].contiguous())
            if h1 - h0 < per:
                pad = local.new_zeros((num_tokens, per - (h1 - h0), local.shape[-1]))
                local = torch.cat((local, pad), dim=1)
            from vllm.distributed import tensor_model_parallel_all_gather

            full = tensor_model_parallel_all_gather(local, dim=1)
            torch.index_select(full, 1, index, out=self._staging[:num_tokens])
        if self._checks_left > 0:
            self._checks_left -= 1
            ref = self._decode_host(ids).to(self._staging.device)
            same = torch.equal(ref, self._staging[:num_tokens])
            logger.info("PLE stage check (%s, %d tokens): %s", mode, num_tokens,
                        "identical" if same else "MISMATCH")
            if not same:
                raise RuntimeError(f"PLE stage mode {mode} differs from the host path")

    def _unique(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        unique, inverse = ids.flatten().unique(return_inverse=True)
        if int(unique[0]) < 0 or int(unique[-1]) >= self._table.num_rows:
            raise IndexError(f"PLE id out of range for {self._table.num_rows} rows")
        return unique, inverse

    def _decode_host(self, ids: torch.Tensor) -> torch.Tensor:
        """Reference path: decode on the host. [T, H] ids -> [T, H, D] rows."""
        unique, inverse = self._unique(ids)
        raw = self._table.read(unique)
        rows = raw.view(self._table.dtype).to(torch.float32)
        if self._table.scale != 1.0:
            rows = rows * self._table.scale
        rows = _sim_int4(rows)
        out = rows.index_select(0, inverse).to(self.weight.dtype)
        return out.view(*ids.shape, -1)

    def _decode_gpu(self, ids: torch.Tensor) -> torch.Tensor:
        """Raw FP8 rows to the device through a pinned buffer, decoded there.
        Same float32 math as the host path, so the result is bit-identical."""
        unique, inverse = self._unique(ids)
        n = unique.numel()
        dev = self.weight.device
        if self._pinned is None or self._pinned.shape[0] < n:
            self._pinned = torch.empty(
                (max(n, 2 * (0 if self._pinned is None else self._pinned.shape[0])),
                 self._table.row_bytes),
                dtype=torch.uint8,
                pin_memory=True,
            )
        raw = self._table.read(unique, out=self._pinned)
        raw_d = raw.to(dev, non_blocking=True)
        inv_d = inverse.to(dev, non_blocking=True)
        rows = raw_d.view(self._table.dtype).to(torch.float32)
        if self._table.scale != 1.0:
            rows = rows * self._table.scale
        rows = _sim_int4(rows)
        out = rows.index_select(0, inv_d).to(self.weight.dtype)
        # The pinned buffer is reused next step: wait for the copy to finish.
        torch.accelerator.synchronize()
        return out.view(*ids.shape, -1)

    def forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        if self._staging is None:
            return self.weight.new_zeros((*ngram_ids.shape, self.weight.shape[1]))
        return self._staging[: ngram_ids.shape[0]]
