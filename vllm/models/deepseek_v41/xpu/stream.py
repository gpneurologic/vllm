# SPDX-License-Identifier: Apache-2.0
"""Streamed routed experts for DeepSeek V4.1 on XPU (full FP4 model).

The experts come from the repacked file (tools/ds41_repack.py): one 4 KiB
aligned block per expert, so a cache miss is one O_DIRECT read. Each TP rank
keeps a device pool of fixed-size slots (global LRU over all layers) and a
slot table [layers, experts] (int32, -1 = not resident) that the decode kernel
(fp4_moe.moe_decode_fp4) reads, so the kernel needs no host data.

Misses are read by a thread pool into pinned host staging buffers and copied to
their slots on a separate copy stream; the compute stream waits only for the
copies of the layer it is about to run. `prefetch()` queues low-priority reads
(demand reads go first; prefetches not yet started are dropped when a newer
prefetch round replaces them).

Settings:
  VLLM_DS41_STREAM_DIR      folder with layout.json + experts_fp4.bin (on = set)
  VLLM_DS41_STREAM_SLOTS    slots per layer on this rank (on average; the LRU
                            is global), total = slots * layers
                            (0 / unset: size from VLLM_DS41_STREAM_GB)
  VLLM_DS41_STREAM_GB       pool size in GiB per rank (default 16)
  VLLM_DS41_READ_THREADS    reader threads (default 16)
  VLLM_DS41_STAGING         pinned staging buffers (default 32, ~18.8 MB each)
"""
from __future__ import annotations

import heapq
import json
import mmap
import os
import threading
import time
from collections import OrderedDict

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

STREAM_DIR = os.environ.get("VLLM_DS41_STREAM_DIR")


def enabled() -> bool:
    return bool(STREAM_DIR)


class _Staging:
    """Pinned host buffers (4 KiB aligned for O_DIRECT), reused once the copy
    that read from them has finished."""

    def __init__(self, n: int, nbytes: int):
        self.nbytes = nbytes
        self.free: list[int] = list(range(n))
        self.bufs: list[torch.Tensor] = []
        self.views: list[memoryview] = []
        self.pending: dict[int, torch.Event] = {}
        self.pinned = True
        for _ in range(n):
            try:
                t = torch.empty(nbytes + 4096, dtype=torch.uint8, pin_memory=True)
            except Exception as e:  # noqa: BLE001
                if self.pinned:
                    logger.warning("DeepSeek V4.1 stream: pinned host memory failed (%s); "
                                   "using pageable buffers", e)
                self.pinned = False
                t = torch.frombuffer(mmap.mmap(-1, nbytes + 4096), dtype=torch.uint8)
            pad = (-t.data_ptr()) % 4096
            t = t[pad: pad + nbytes]
            self.bufs.append(t)
            self.views.append(memoryview(t.numpy()))
        self.cv = threading.Condition()

    def get(self) -> int:
        with self.cv:
            while not self.free:
                self._reclaim(block=True)
            return self.free.pop()

    def _reclaim(self, block: bool) -> None:
        done = [i for i, ev in self.pending.items() if ev.query()]
        if not done and block and self.pending:
            i, ev = next(iter(self.pending.items()))
            ev.synchronize()
            done = [i]
        for i in done:
            del self.pending[i]
            self.free.append(i)

    def release_after(self, i: int, ev: torch.Event) -> None:
        with self.cv:
            self.pending[i] = ev

    def release(self, i: int) -> None:
        with self.cv:
            self.free.append(i)


class StreamCache:
    _instance: "StreamCache | None" = None

    @classmethod
    def get(cls, device, n_layers: int) -> "StreamCache":
        if cls._instance is None:
            cls._instance = cls(STREAM_DIR, device, n_layers)
        return cls._instance

    def __init__(self, folder: str, device, n_layers: int):
        self.lay = json.load(open(os.path.join(folder, "layout.json")))
        self.E = self.lay["experts"]
        self.L = n_layers
        self.block = self.lay["block_bytes"]
        self.raw = self.lay["raw_bytes"]
        self.device = device
        path = os.path.join(folder, self.lay.get("file", "experts_fp4.bin"))
        self.fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        slots = int(os.environ.get("VLLM_DS41_STREAM_SLOTS", "0") or 0)
        if slots > 0:
            n_slots = slots * n_layers
        else:
            gb = float(os.environ.get("VLLM_DS41_STREAM_GB", "16"))
            n_slots = int(gb * 2**30) // self.raw
        self.n_slots = n_slots
        # Slots hold raw_bytes (the 4 KiB padding is not copied).
        self.pool = torch.empty(n_slots, self.raw, dtype=torch.uint8, device=device)
        self.slot_cpu = torch.full((n_layers, self.E), -1, dtype=torch.int32)
        self.slot_dev = torch.full((n_layers, self.E), -1, dtype=torch.int32, device=device)
        self.lru: OrderedDict = OrderedDict()   # (layer, e) -> slot
        self._dirty: set = set()                # layers whose slot row changed
        self.free_slots = list(range(n_slots - 1, -1, -1))
        self.inflight: dict = {}                # (layer, e) -> _Fetch
        self._zombies: list = []                # cancelled fetches still holding a buffer
        self.copy_stream = torch.xpu.Stream(device)
        self.staging = _Staging(int(os.environ.get("VLLM_DS41_STAGING", "32")), self.block)
        self.n_threads = int(os.environ.get("VLLM_DS41_READ_THREADS", "16"))
        n_pc = max(1, int(os.environ.get("VLLM_DS41_READ_PIECES", "8")))
        self.piece = ((self.raw + n_pc - 1) // n_pc + 4095) & ~4095
        self.pieces = [(p, min(self.piece, self.raw - p)) for p in range(0, self.raw, self.piece)]
        self.q: list = []                       # heap of (prio, seq, fetch, piece)
        self.qlock = threading.Condition()
        self.seq = 0
        self.pf_round = 0
        self.stats = dict(hits=0, misses=0, pf_issued=0, pf_used=0, pf_dropped=0,
                          read_bytes=0, wait_s=0.0, steps=0)
        self._last_log = time.time()
        for _ in range(self.n_threads):
            threading.Thread(target=self._reader, daemon=True).start()
        logger.info("DeepSeek V4.1 expert streaming: %s, %d slots (%.1f per layer, %.1f GiB) "
                    "on %s, %d reader threads, %d %s staging buffers", path, n_slots,
                    n_slots / n_layers, n_slots * self.raw / 2**30, device, self.n_threads,
                    len(self.staging.bufs), "pinned" if self.staging.pinned else "pageable")

    # ------------------------------------------------------------- reader side
    # A fetch is split into pieces (VLLM_DS41_READ_PIECES, default 8) read by
    # different threads: a lone miss then uses the drive's parallelism
    # instead of one queue-depth-1 read. Each piece is read with O_DIRECT into
    # the thread's anonymous buffer (O_DIRECT cannot target the runtime's
    # pinned memory: EFAULT) and copied into the fetch's pinned staging
    # buffer. The staging buffer is taken when the fetch is submitted (main
    # thread); ensure() never has more fetches in flight than buffers.
    class _Fetch:
        __slots__ = ("key", "prio", "round", "done", "buf", "err", "started", "cancel", "left",
                     "installed")

        def __init__(self, key, prio, rnd, n):
            self.key, self.prio, self.round = key, prio, rnd
            self.done = threading.Event()
            self.buf = None
            self.err = None
            self.started = False
            self.cancel = False
            self.left = n
            self.installed = False

    def _submit(self, key, prio: int):
        f = self._Fetch(key, prio, self.pf_round, len(self.pieces))
        f.buf = self.staging.get()
        with self.qlock:
            for k in range(len(self.pieces)):
                self.seq += 1
                heapq.heappush(self.q, (prio, self.seq, f, k))
            self.qlock.notify_all()
        self.inflight[key] = f
        return f

    def _reader(self) -> None:
        mm = mmap.mmap(-1, self.piece)
        mv = memoryview(mm)
        t = torch.frombuffer(mm, dtype=torch.uint8)
        while True:
            with self.qlock:
                while not self.q:
                    self.qlock.wait()
                _, _, f, k = heapq.heappop(self.q)
                if f.cancel:
                    f.left -= 1
                    if f.left == 0:
                        f.done.set()
                    continue
                f.started = True
            try:
                p0, n = self.pieces[k]
                l, e = f.key
                off = (l * self.E + e) * self.block + p0
                want = (n + 4095) & ~4095
                got = 0
                while got < want:
                    r = os.preadv(self.fd, [mv[got:want]], off + got)
                    if r <= 0:
                        raise OSError(f"short read at expert {f.key}")
                    got += r
                with torch.inference_mode():  # the buffers were made in inference mode
                    self.staging.bufs[f.buf][p0: p0 + n].copy_(t[:n])
            except Exception as ex:  # noqa: BLE001
                f.err = ex
            with self.qlock:
                f.left -= 1
                if f.left == 0:
                    f.done.set()

    # ------------------------------------------------------------ device side
    def _alloc_slot(self, protect: set) -> int:
        if self.free_slots:
            return self.free_slots.pop()
        for key in self.lru:
            if key not in protect:
                slot = self.lru.pop(key)
                self.slot_cpu[key[0], key[1]] = -1
                self._dirty.add(key[0])
                return slot
        raise RuntimeError("DeepSeek V4.1 stream: expert pool too small for one layer")

    def _install(self, f, protect: set) -> None:
        """Copy a finished read into a slot (copy stream)."""
        f.done.wait()
        if f.installed:
            return
        f.installed = True
        if self.inflight.get(f.key) is f:
            del self.inflight[f.key]
        if f.err is not None:
            raise f.err
        if f.cancel:
            self.staging.release(f.buf)
            return
        slot = self._alloc_slot(protect)
        src = self.staging.bufs[f.buf][: self.raw]
        with torch.xpu.stream(self.copy_stream):
            self.pool[slot].copy_(src, non_blocking=self.staging.pinned)
            ev = torch.xpu.Event()
            ev.record(self.copy_stream)
        if self.staging.pinned:
            self.staging.release_after(f.buf, ev)
        else:
            self.staging.release(f.buf)
        self.lru[f.key] = slot
        self.slot_cpu[f.key[0], f.key[1]] = slot
        self._dirty.add(f.key[0])
        self.stats["read_bytes"] += self.block

    def ensure(self, layer: int, experts: list[int]) -> torch.Tensor:
        """Make `experts` of `layer` resident; returns the layer's device slot
        row (valid on the current stream once this returns)."""
        t0 = time.time()
        keys = [(layer, e) for e in experts]
        protect = set(keys)
        need = []
        for k in keys:
            if k in self.lru:
                self.lru.move_to_end(k)
                self.stats["hits"] += 1
            else:
                need.append(k)
        if need:
            # At most `cap` fetches in flight (each holds a staging buffer):
            # long prefill steps miss up to ~128 experts per layer.
            cap = max(1, len(self.staging.bufs) - 4)
            todo = list(need)
            pending: list = []
            while todo or pending:
                while todo and len(pending) < cap:
                    k = todo.pop(0)
                    f = self.inflight.get(k)
                    if f is not None:
                        self.stats["pf_used"] += 1
                        promote = False
                        with self.qlock:  # not started yet: re-queue at demand priority
                            if f.prio > 0 and not f.started:
                                f.cancel = promote = True
                        if promote:
                            self._zombies.append(f)
                            f = self._submit(k, 0)
                    else:
                        self.stats["misses"] += 1
                        f = self._submit(k, 0)
                    pending.append(f)
                # Install in completion order.
                rest = [f for f in pending if not f.done.is_set()]
                for f in pending:
                    if f.done.is_set():
                        self._install(f, protect)
                if len(rest) == len(pending):
                    self._flush_prefetched(protect)
                    rest[0].done.wait(0.002)
                pending = rest
            torch.xpu.current_stream(self.device).wait_stream(self.copy_stream)
        self._flush_prefetched(protect)
        if layer in self._dirty:
            self.slot_dev[layer].copy_(self.slot_cpu[layer], non_blocking=False)
            self._dirty.discard(layer)
        self.stats["wait_s"] += time.time() - t0
        self._maybe_log()
        return self.slot_dev[layer]

    def _flush_prefetched(self, protect: set) -> None:
        """Install prefetches that have landed (no waiting); free the buffers
        of cancelled fetches whose pieces have all been dropped."""
        if self._zombies:
            for f in [z for z in self._zombies if z.done.is_set()]:
                self._zombies.remove(f)
                self.staging.release(f.buf)
        for k, f in list(self.inflight.items()):
            if f.prio > 0 and f.done.is_set():
                self._install(f, protect)

    def prefetch(self, keys: list[tuple[int, int]], prio: int = 1) -> None:
        """Queue low-priority reads (only into free staging buffers, keeping a
        margin for demand reads); drops older prefetches not yet started."""
        self.pf_round += 1
        with self.qlock:
            old = {id(f): f for _, _, f, _ in self.q if f.prio > 0}
            for f in old.values():
                if not f.started and not f.cancel:
                    f.cancel = True
                    self.stats["pf_dropped"] += 1
        for f in old.values():
            if f.cancel and self.inflight.get(f.key) is f:
                del self.inflight[f.key]
                self._zombies.append(f)
        for k in keys:
            if len(self.staging.free) <= 8:
                break
            if k in self.lru or k in self.inflight:
                continue
            self._submit(k, prio)
            self.stats["pf_issued"] += 1

    def parts(self, layer: int, e: int):
        from .fp4_moe import expert_parts

        return expert_parts(self.pool, self.lru[(layer, e)], self.lay)

    def _maybe_log(self) -> None:
        if time.time() - self._last_log < 30:
            return
        self._last_log = time.time()
        s = self.stats
        tot = s["hits"] + s["misses"] + s["pf_used"]
        logger.info("expert stream: %d resident, hit %.1f%% (%d hits, %d misses, %d from "
                    "prefetch; %d prefetches, %d dropped), %.1f GiB read, wait %.1f s",
                    len(self.lru), 100.0 * s["hits"] / max(tot, 1), s["hits"], s["misses"],
                    s["pf_used"], s["pf_issued"], s["pf_dropped"], s["read_bytes"] / 2**30,
                    s["wait_s"])
