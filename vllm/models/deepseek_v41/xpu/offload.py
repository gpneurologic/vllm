# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 on XPU: weights read straight from the checkpoint on NVMe.

- `CheckpointFiles`: raw byte offsets of every tensor in the safetensors shards,
  mmapped read-only (nothing is held in host RAM beyond the page cache).
- `ExpertCache`: routed experts (FP4 + ue8m0 scales, ~18.8 MB each) copied to the
  device on first use and evicted LRU within a byte budget
  (VLLM_DS41_EXPERT_CACHE_GB, default 10).
- `DiskEngramEmbedding`: the Engram n-gram tables (FP8 rows + ue8m0 scales,
  ~189 GiB) read row by row from the shards.
"""

import json
import mmap
import os
import struct
import threading
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

_DT = {
    "BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.uint8, "I8": torch.uint8,
    "I32": torch.int32, "I64": torch.int64, "U8": torch.uint8,
}


_DROP_PAGE_CACHE = float(os.environ.get("VLLM_DS41_HOST_CACHE_GB", "0")) > 0
# Expert reads bypass the page cache (O_DIRECT): ~11.8 GB/s for 6 MB random
# reads on the T705 against ~3 GB/s buffered. VLLM_DS41_DIRECT_IO=0 turns it off.
_DIRECT_IO = os.environ.get("VLLM_DS41_DIRECT_IO", "1") == "1"


class CheckpointFiles:
    _instances: dict[str, "CheckpointFiles"] = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, folder: str) -> "CheckpointFiles":
        # The serving folder may hold symlinks to only part of the shards (the
        # Engram table shards are left out for the generic loader); read the
        # real checkpoint folder the config symlink points into.
        folder = os.path.dirname(os.path.realpath(os.path.join(folder, "config.json")))
        with cls._lock:
            if folder not in cls._instances:
                cls._instances[folder] = cls(folder)
            return cls._instances[folder]

    def __init__(self, folder: str):
        self.maps: dict[str, mmap.mmap] = {}
        self.fds: dict[str, int] = {}
        self.dfds: dict[str, int] = {}
        self.paths: dict[str, str] = {}
        self.index: dict[str, tuple] = {}
        for f in sorted(os.listdir(folder)):
            if not f.endswith(".safetensors"):
                continue
            with open(os.path.join(folder, f), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(n))
                mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            # Lookups touch scattered rows: no readahead (it would pull up to
            # 128 KiB into the page cache for every 512-byte Engram row).
            mm.madvise(mmap.MADV_RANDOM)
            self.maps[f] = mm
            self.fds[f] = os.open(os.path.join(folder, f), os.O_RDONLY)
            self.paths[f] = os.path.join(folder, f)
            base = 8 + n
            for k, v in header.items():
                if k == "__metadata__":
                    continue
                a, b = v["data_offsets"]
                self.index[k] = (f, base + a, b - a, v["dtype"], v["shape"])

    def tensor(self, name: str) -> torch.Tensor:
        f, off, nb, dt, shape = self.index[name]
        buf = np.frombuffer(self.maps[f], dtype=np.uint8, count=nb, offset=off).copy()
        return torch.from_numpy(buf).view(_DT[dt]).view(shape)

    def span(self, names: list[str]) -> tuple[str, int, int]:
        """(file, offset, length) covering `names` if they sit in one file."""
        locs = [self.index[n] for n in names]
        f = locs[0][0]
        assert all(loc[0] == f for loc in locs)
        start = min(loc[1] for loc in locs)
        end = max(loc[1] + loc[2] for loc in locs)
        return f, start, end - start

    def pread_tensors(self, names: list[str]) -> list[torch.Tensor]:
        """Read several tensors with large pread calls (one per contiguous run)
        instead of page-faulting through the mmap."""
        # Runs of tensors that are adjacent in one file (gaps under 64 KiB).
        runs: list[list[str]] = []
        for n in sorted(names, key=lambda n: (self.index[n][0], self.index[n][1])):
            f, off = self.index[n][0], self.index[n][1]
            if runs:
                pf, poff, pnb = self.index[runs[-1][-1]][:3]
                if pf == f and 0 <= off - (poff + pnb) < 65536:
                    runs[-1].append(n)
                    continue
            runs.append([n])
        out: dict[str, torch.Tensor] = {}
        for group in runs:
            f, start, length = self.span(group)
            if _DIRECT_IO:
                buf, base = self._read_direct(f, start, length)
            else:
                buf, base = self._read_buffered(f, start, length), start
            for n in group:
                _, off, nb, dt, shape = self.index[n]
                out[n] = buf[off - base : off - base + nb].view(_DT[dt]).view(shape)
        return [out[n] for n in names]

    def _read_buffered(self, f: str, start: int, length: int) -> torch.Tensor:
        buf = torch.empty(length, dtype=torch.uint8)
        mv = memoryview(buf.numpy())
        got = 0
        while got < length:
            r = os.preadv(self.fds[f], [mv[got:]], start + got)
            if r <= 0:
                raise OSError(f"short read from {f}")
            got += r
        if _DROP_PAGE_CACHE:
            # The host tier keeps its own copy: don't hold it twice in RAM.
            os.posix_fadvise(self.fds[f], start, length, os.POSIX_FADV_DONTNEED)
        return buf

    def _read_direct(self, f: str, start: int, length: int) -> tuple[torch.Tensor, int]:
        """O_DIRECT read (no page cache, no extra copy) of the 4 KiB-aligned
        range around [start, start + length) into a page-aligned buffer."""
        fd = self.dfds.get(f)
        if fd is None:
            fd = self.dfds[f] = os.open(self.paths[f], os.O_RDONLY | os.O_DIRECT)
        a0 = start & ~4095
        a1 = (start + length + 4095) & ~4095
        mm = mmap.mmap(-1, a1 - a0)  # anonymous, page aligned
        mv = memoryview(mm)
        got, want = 0, start + length - a0
        while got < want:
            r = os.preadv(fd, [mv[got:]], a0 + got)
            if r <= 0:
                raise OSError(f"short O_DIRECT read from {f}")
            got += r
        del mv
        return torch.frombuffer(mm, dtype=torch.uint8), a0

    def rows(self, name: str, row_ids: np.ndarray) -> torch.Tensor:
        f, off, nb, dt, shape = self.index[name]
        width = nb // shape[0]
        arr = np.frombuffer(self.maps[f], dtype=np.uint8, count=nb, offset=off)
        return torch.from_numpy(arr.reshape(shape[0], width)[row_ids])


def engram_table_shards(folder: str) -> set[str]:
    files = CheckpointFiles.get(folder)
    return {v[0] for k, v in files.index.items() if ".engram.embed." in k}


def engram_shard_small_weights(folder: str):
    """The non-table tensors that live in the Engram shards (q/k weights, wkv)."""
    files = CheckpointFiles.get(folder)
    skip = engram_table_shards(folder)
    if all(os.path.exists(os.path.join(folder, f)) for f in skip):
        return  # the generic loader reads these shards itself
    for name, v in files.index.items():
        if v[0] in skip and ".engram.embed." not in name:
            t = files.tensor(name)
            if v[3] == "F8_E8M0":
                t = t.view(torch.float8_e8m0fnu)
            yield name, t


# ----------------------------------------------------------------------- FP4 math
_FP4 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_FP4_TABLE = torch.tensor(_FP4 + [-v for v in _FP4], dtype=torch.float32)
_FP4_TABLE[8] = 0.0
_tables: dict[torch.device, torch.Tensor] = {}


def _e8m0(s: torch.Tensor) -> torch.Tensor:
    return torch.ldexp(torch.ones_like(s, dtype=torch.float32), s.to(torch.int32) - 127)


def dequant_fp4(w: torch.Tensor, s: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    """w [N, K/2] uint8 (low nibble first), s [N, K/32] ue8m0 -> [N, K]."""
    t = _tables.get(w.device)
    if t is None:
        t = _tables[w.device] = _FP4_TABLE.to(w.device)
    vals = torch.stack([t[(w & 0x0F).long()], t[(w >> 4).long()]], dim=-1).flatten(-2)
    return (vals.unflatten(-1, (-1, 32)) * _e8m0(s).unsqueeze(-1)).flatten(-2).to(dtype)


def act_fp8_roundtrip(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    xf = x.float().unflatten(-1, (-1, block))
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    m, e = torch.frexp(amax / 448.0)
    e = e - (m == 0.5).to(e.dtype)
    sc = torch.ldexp(torch.ones_like(amax), e)
    q = (xf / sc).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
    return (q * sc).flatten(-2).to(x.dtype)


# ----------------------------------------------------------------------- experts
class ExpertCache:
    _instance: "ExpertCache | None" = None

    @classmethod
    def get(cls, folder: str, device) -> "ExpertCache":
        if cls._instance is None:
            budget = float(os.environ.get("VLLM_DS41_EXPERT_CACHE_GB", "10"))
            cls._instance = cls(CheckpointFiles.get(folder), device, int(budget * 2**30))
            logger.info("DeepSeek V4.1 routed-expert cache: %.1f GiB on %s", budget, device)
        return cls._instance

    def __init__(self, files: CheckpointFiles, device, budget: int):
        self.files, self.device, self.budget = files, device, budget
        self.used = 0
        self.lru: OrderedDict = OrderedDict()
        self.hits = self.misses = self.read_bytes = 0
        self.read_time = 0.0
        self._last_log = time.time()
        self.prefix = "layers"
        # Second tier in host RAM: the CPU copies of experts read from disk are
        # kept (LRU) so an expert evicted from the GPU comes back with a host
        # copy instead of an NVMe read. Per worker process.
        self.host_budget = int(float(os.environ.get("VLLM_DS41_HOST_CACHE_GB", "0")) * 2**30)
        self.host_lru: OrderedDict = OrderedDict()
        self.host_used = 0
        self.host_hits = 0
        if self.host_budget:
            logger.info("DeepSeek V4.1 host-RAM expert tier: %.1f GiB",
                        self.host_budget / 2**30)

    _pool = None

    def _read(self, prefix: str, e: int):
        names = [
            f"{prefix}.ffn.experts.{e}.{w}.{k}"
            for w in ("w1", "w3", "w2")
            for k in ("weight", "scale")
        ]
        return self.files.pread_tensors(names)

    @staticmethod
    def _nbytes(parts) -> int:
        return sum(p.numel() * p.element_size() for p in parts)

    def _host_put(self, key, cpu_parts) -> None:
        if not self.host_budget:
            return
        if key in self.host_lru:
            self.host_lru.move_to_end(key)
            return
        nb = self._nbytes(cpu_parts)
        while self.host_used + nb > self.host_budget and self.host_lru:
            _, old = self.host_lru.popitem(last=False)
            self.host_used -= self._nbytes(old)
        self.host_lru[key] = cpu_parts
        self.host_used += nb

    def ensure(self, prefix: str, experts: list[int]) -> None:
        """Bring every expert in `experts` into the GPU cache: from the host
        tier if it is there, else from disk (in parallel)."""
        missing = [e for e in experts if (prefix, e) not in self.lru]
        self.hits += len(experts) - len(missing)
        for e in experts:
            if (prefix, e) in self.lru:
                self.lru.move_to_end((prefix, e))
        if not missing:
            return
        from_host = [e for e in missing if (prefix, e) in self.host_lru]
        from_disk = [e for e in missing if (prefix, e) not in self.host_lru]
        if ExpertCache._pool is None:
            from concurrent.futures import ThreadPoolExecutor

            ExpertCache._pool = ThreadPoolExecutor(
                int(os.environ.get("VLLM_DS41_READ_THREADS", "16"))
            )
        t = time.time()
        from concurrent.futures import as_completed

        futures = {ExpertCache._pool.submit(self._read, prefix, e): e for e in from_disk}
        # Host copies first (while the disk reads run), then each disk read as
        # soon as it lands, so GPU copies overlap the reads still in flight.
        import itertools

        sources = itertools.chain(
            ((e, None) for e in from_host),
            ((futures[fut], fut) for fut in as_completed(futures)),
        )
        for e, fut in sources:
            key = (prefix, e)
            if fut is None:
                cpu_parts = self.host_lru[key]
                self.host_lru.move_to_end(key)
                self.host_hits += 1
            else:
                cpu_parts = tuple(fut.result())
                self.read_bytes += self._nbytes(cpu_parts)
                self.misses += 1
                self._host_put(key, cpu_parts)
            parts = tuple(p.to(self.device, non_blocking=False) for p in cpu_parts)
            nb = self._nbytes(parts)
            while self.used + nb > self.budget and self.lru:
                _, old = self.lru.popitem(last=False)
                self.used -= self._nbytes(old)
            self.lru[key] = parts
            self.used += nb
        self.read_time += time.time() - t

    def get_expert(self, prefix: str, e: int):
        key = (prefix, e)
        parts = self.lru.get(key)
        if parts is None:
            self.ensure(prefix, [e])
            parts = self.lru[key]
        if time.time() - self._last_log > 30:
            self._last_log = time.time()
            logger.info(
                "expert cache: %d resident (%.1f GiB), GPU hits %d, host hits %d "
                "(%d in host, %.1f GiB), disk reads %d (%.1f GiB), fetch time %.1f s",
                len(self.lru), self.used / 2**30, self.hits, self.host_hits,
                len(self.host_lru), self.host_used / 2**30, self.misses,
                self.read_bytes / 2**30, self.read_time,
            )
        return parts


def run_expert(x: torch.Tensor, parts, swiglu_limit: float,
               weights: torch.Tensor) -> torch.Tensor:
    """One routed SwiGLU expert, as DeepSeek's reference computes it (fp8
    activations with ue8m0 scales per 32, fp4 weights). weights [n, 1] fp32.
    Returns fp32 [n, hidden]."""
    w1, s1, w3, s3, w2, s2 = parts
    xq = act_fp8_roundtrip(x)
    gate = (xq @ dequant_fp4(w1, s1).t()).float()
    up = (xq @ dequant_fp4(w3, s3).t()).float()
    if swiglu_limit > 0:
        up = up.clamp(-swiglu_limit, swiglu_limit)
        gate = gate.clamp(max=swiglu_limit)
    h = (torch.nn.functional.silu(gate) * up * weights).to(x.dtype)
    return (act_fp8_roundtrip(h) @ dequant_fp4(w2, s2).t()).float()


# ----------------------------------------------------------------------- engram
class DiskEngramEmbedding(nn.Module):
    """Stands in for `ParallelEngramEmbedding`: hash columns split over TP ranks,
    rows read from the checkpoint mmap on every lookup."""

    def __init__(self, folder: str, layer_id: int, n_hash_cols: int, head_dim: int):
        super().__init__()
        self.files = CheckpointFiles.get(folder)
        self.wname = f"layers.{layer_id}.engram.embed.weight"
        self.sname = f"layers.{layer_id}.engram.embed.scale"
        self.dim, self.block_size = head_dim, 32
        self.n_hash_cols = n_hash_cols
        self.tp_size = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_rank()
        self.part_n_hash_cols = (n_hash_cols + self.tp_size - 1) // self.tp_size
        self.head_start = min(rank * self.part_n_hash_cols, n_hash_cols)
        self.head_end = min(self.head_start + self.part_n_hash_cols, n_hash_cols)
        self.dp_size = 1
        self.cpu_offload = False
        self.packed_fp4 = False

    def lookup(self, hash_ids: torch.Tensor, rows: torch.Tensor, background=False):
        from vllm.forward_context import get_forward_context, is_forward_context_available

        if not (is_forward_context_available()
                and isinstance(get_forward_context().attn_metadata, dict)):
            rows.zero_()  # profile / warm-up pass: no disk reads
            return
        local = hash_ids[:, self.head_start : self.head_end]
        n, c = local.shape
        if n == 0:
            return
        if c < rows.shape[1]:
            rows[:, c:].zero_()
        if c == 0:
            return
        flat = local.reshape(-1).cpu().numpy()
        vals = self.files.rows(self.wname, flat).to(rows.device)
        sc = self.files.rows(self.sname, flat).to(rows.device)
        v = vals.view(torch.float8_e4m3fn).float().unflatten(-1, (-1, self.block_size))
        v = (v * _e8m0(sc).unsqueeze(-1)).flatten(-2)
        rows[:, :c].copy_(v.view(n, c, self.dim).to(rows.dtype))
