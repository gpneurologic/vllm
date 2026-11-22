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

    def read(self, ids: torch.Tensor) -> torch.Tensor:
        """Return the raw bytes of the rows `ids` (CPU int64) as [n, row_bytes].

        Consecutive ids in one shard are read with a single call; when a step
        has many such runs they are spread over a thread pool."""
        n = ids.numel()
        out = torch.empty(n, self.row_bytes, dtype=torch.uint8)
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
        ids = ngram_ids.detach().to("cpu").flatten()
        unique, inverse = ids.unique(return_inverse=True)
        if int(unique[0]) < 0 or int(unique[-1]) >= self._table.num_rows:
            raise IndexError(f"PLE id out of range for {self._table.num_rows} rows")
        raw = self._table.read(unique)
        rows = raw.view(self._table.dtype).to(torch.float32)
        if self._table.scale != 1.0:
            rows = rows * self._table.scale
        host = rows.index_select(0, inverse).to(self.weight.dtype)
        self._staging[:num_tokens].copy_(host.view(num_tokens, self._num_heads, -1))

    def forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        if self._staging is None:
            return self.weight.new_zeros((*ngram_ids.shape, self.weight.shape[1]))
        return self._staging[: ngram_ids.shape[0]]
