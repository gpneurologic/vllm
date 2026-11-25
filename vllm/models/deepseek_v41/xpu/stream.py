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
  VLLM_DS41_STREAM_MIRRORS  comma list of "<folder>:<weight>" copies of the
                            expert file on other drives, e.g.
                            "/g/ds41-experts:1" with the main file weight
                            VLLM_DS41_STREAM_WEIGHT (default 3): experts are
                            spread over the files in proportion, piece by
                            piece within each expert (with READ_PIECES a
                            multiple of the weight sum), so all drives read
                            each miss at once.
"""
from __future__ import annotations

import ctypes
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
# VLLM_DS41_STREAM_TIMING=1: host time per streamed MoE call (routing sync,
# ensure, kernel launch); =2 also syncs after the kernel (GPU time; slower).
TIMING = int(os.environ.get("VLLM_DS41_STREAM_TIMING", "0") or 0)


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
        files = [(path, int(os.environ.get("VLLM_DS41_STREAM_WEIGHT", "3")))]
        for m in filter(None, os.environ.get("VLLM_DS41_STREAM_MIRRORS", "").split(",")):
            d, _, wt = m.rpartition(":")
            files.append((os.path.join(d, self.lay.get("file", "experts_fp4.bin")), int(wt)))
        self.fds = [os.open(p, os.O_RDONLY | os.O_DIRECT) for p, _ in files]
        # expert index -> file: smooth weighted round-robin over the files
        # (weights 3,1 -> 0,0,1,0), so neighbouring experts use different drives
        wts = [wt for _, wt in files]
        cur = [0] * len(files)
        self.pattern = []
        for _ in range(sum(wts)):
            cur = [c + w for c, w in zip(cur, wts)]
            j = max(range(len(files)), key=lambda i: cur[i])
            cur[j] -= sum(wts)
            self.pattern.append(j)
        self.fd = self.fds[0]
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
        self.done_cv = threading.Condition()     # readers notify on each finished fetch
        self.seq = 0
        self.pf_round = 0
        self.stats = dict(hits=0, misses=0, pf_issued=0, pf_used=0, pf_dropped=0,
                          pf_hit=0, pf_wasted=0, read_bytes=0, wait_s=0.0, steps=0)
        self.pf_keys: set = set()               # installed by prefetch, not yet demanded
        self._last_log = time.time()
        self.in_decode = False                  # set by the async decode path (fetch timing)
        # Demand misses of at most this many experts (VLLM_DS41_SYNC_READ_MAX,
        # default 0 = off: no decode gain measured) are read by the calling
        # thread in one preadv (no thread-pool handoff): a lone decode miss
        # then costs about read + 0.1 ms instead of read + ~1.5 ms.
        self.sync_max = int(os.environ.get("VLLM_DS41_SYNC_READ_MAX", "0"))
        self._sync_mm = mmap.mmap(-1, (self.block + 4095) & ~4095) if self.sync_max > 0 else None
        self._sync_t = torch.frombuffer(self._sync_mm, dtype=torch.uint8) if self._sync_mm else None
        # How the main-thread read reaches VRAM: "mt" = 4-thread memmove into
        # pinned staging, then an async copy (workers run torch with 1 thread,
        # so torch's own host copy of 10 MB takes ~1 ms); "pageable" = copy
        # straight from the read buffer (driver-staged, synchronous).
        self.sync_copy = os.environ.get("VLLM_DS41_SYNC_COPY", "mt")
        self.sync_verify = os.environ.get("VLLM_DS41_SYNC_VERIFY") == "1"
        self._rows = None
        self._row_i = 0
        if os.environ.get("VLLM_DS41_ASYNC_ROWS", "1") == "1":
            try:
                self._rows = [torch.empty(self.E, dtype=torch.int32, pin_memory=True) for _ in range(64)]
            except RuntimeError:
                self._rows = None
        self._sync_pool = None
        if self.sync_max > 0 and self.sync_copy == "mt":
            from concurrent.futures import ThreadPoolExecutor
            self._sync_pool = ThreadPoolExecutor(4)
            self._sync_addr = ctypes.addressof(ctypes.c_char.from_buffer(self._sync_mm))
            step = ((self.raw + 3) // 4 + 63) & ~63
            self._sync_jobs = [(o, min(step, self.raw - o)) for o in range(0, self.raw, step)]
        for _ in range(self.n_threads):
            threading.Thread(target=self._reader, daemon=True).start()
        logger.info("DeepSeek V4.1 expert streaming: %s, %d slots (%.1f per layer, %.1f GiB) "
                    "on %s, %d reader threads, %d %s staging buffers", ",".join(
                        f"{p}x{wt}" for p, wt in files), n_slots,
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
                     "installed", "t0", "t_start", "t_done")

        def __init__(self, key, prio, rnd, n):
            self.key, self.prio, self.round = key, prio, rnd
            self.done = threading.Event()
            self.buf = None
            self.err = None
            self.started = False
            self.cancel = False
            self.left = n
            self.installed = False
            self.t0 = time.perf_counter()
            self.t_start = self.t_done = 0.0

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
                if not f.started:
                    f.t_start = time.perf_counter()
                f.started = True
            try:
                p0, n = self.pieces[k]
                l, e = f.key
                gi = l * self.E + e
                # Stripe within the expert: its pieces are spread over the
                # files by weight (RAID-0 like), so one miss reads from all
                # drives at once and the slower drive never straggles.
                fd = self.fds[self.pattern[(k + gi) % len(self.pattern)]]
                off = gi * self.block + p0
                want = (n + 4095) & ~4095
                got = 0
                while got < want:
                    r = os.preadv(fd, [mv[got:want]], off + got)
                    if r <= 0:
                        raise OSError(f"short read at expert {f.key}")
                    got += r
                with torch.inference_mode():  # the buffers were made in inference mode
                    self.staging.bufs[f.buf][p0: p0 + n].copy_(t[:n])
            except Exception as ex:  # noqa: BLE001
                f.err = ex
            with self.qlock:
                f.left -= 1
                last = f.left == 0
            if last:
                f.t_done = time.perf_counter()
                f.done.set()
                with self.done_cv:
                    self.done_cv.notify_all()

    # ------------------------------------------------------------ device side
    def _alloc_slot(self, protect: set) -> int:
        if self.free_slots:
            return self.free_slots.pop()
        for key in self.lru:
            if key not in protect:
                slot = self.lru.pop(key)
                if key in self.pf_keys:
                    self.pf_keys.discard(key)
                    self.stats["pf_wasted"] += 1
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
        t_seen = time.perf_counter()
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
        if f.prio > 0:
            self.pf_keys.add(f.key)
        self.slot_cpu[f.key[0], f.key[1]] = slot
        self._dirty.add(f.key[0])
        self.stats["read_bytes"] += self.block
        if f.prio == 0 and f.t_done and self.in_decode:
            s = self.stats
            s["f_n"] = s.get("f_n", 0) + 1
            s["f_queue"] = s.get("f_queue", 0.0) + (f.t_start - f.t0)
            s["f_read"] = s.get("f_read", 0.0) + (f.t_done - f.t_start)
            s["f_seen"] = s.get("f_seen", 0.0) + (t_seen - f.t_done)
            s["f_inst"] = s.get("f_inst", 0.0) + (time.perf_counter() - t_seen)

    def ensure(self, layer: int, experts: list[int]) -> torch.Tensor:
        """Make `experts` of `layer` resident; returns the layer's device slot
        row (valid on the current stream once this returns)."""
        for _ in self.ensure_iter(layer, experts):
            pass
        return self.slot_dev[layer]

    def ensure_iter(self, layer: int, experts: list[int]):
        """As ensure(), but yields each expert id as soon as it is resident:
        hits at once (after the misses' reads are queued), misses in landing
        order. Before using a yielded miss, the consumer's stream must wait on
        copy_stream (wait_copies)."""
        t0 = time.time()
        keys = [(layer, e) for e in experts]
        protect = set(keys)
        need, hits = [], []
        given: set = set()  # expert ids already yielded
        for k in keys:
            if k in self.lru:
                self.lru.move_to_end(k)
                self.stats["hits"] += 1
                hits.append(k[1])
                given.add(k[1])
                if k in self.pf_keys:
                    self.pf_keys.discard(k)
                    self.stats["pf_hit"] += 1
            else:
                need.append(k)
        if (need and len(need) <= self.sync_max and len(self.fds) == 1
                and not any(k in self.inflight for k in need)):
            for k in need:
                self._read_sync(k, protect)
                hits.append(k[1])
                given.add(k[1])
            need = []
            torch.xpu.current_stream(self.device).wait_stream(self.copy_stream)
        if need:
            # At most `cap` fetches in flight (each holds a staging buffer):
            # long prefill steps miss up to ~128 experts per layer.
            cap = max(1, len(self.staging.bufs) - 4)
            todo = list(need)
            pending: list = []
            while todo or pending:
                while todo and len(pending) < cap:
                    if not self.staging.free and not self.staging.pending:
                        # every buffer is held by a read in flight (prefetches)
                        if pending:
                            break
                        self._make_room(protect)
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
                if hits:  # reads are queued: hand out the resident ones meanwhile
                    t1 = time.time()
                    yield from hits
                    hits = []
                    t0 += time.time() - t1
                # Install in completion order.
                # One snapshot: a read finishing between two checks must not be
                # both installed now and kept pending (it would be yielded twice).
                done_now, rest = [], []
                for f in pending:
                    (done_now if f.done.is_set() else rest).append(f)
                landed = []
                for f in done_now:
                    self._install(f, protect)
                    if f.key in self.lru and f.key[1] not in given:
                        given.add(f.key[1])
                        landed.append(f.key[1])
                if landed:
                    t1 = time.time()
                    yield from landed
                    t0 += time.time() - t1
                elif len(rest) == len(pending):
                    self._flush_prefetched(protect)
                    with self.done_cv:
                        if not any(f.done.is_set() for f in rest):
                            self.done_cv.wait(0.05)
                pending = rest
            torch.xpu.current_stream(self.device).wait_stream(self.copy_stream)
        yield from hits
        self._flush_prefetched(protect)
        if layer in self._dirty:
            self._row_to_dev(self.slot_dev[layer], self.slot_cpu[layer])
            self._dirty.discard(layer)
        self.stats["wait_s"] += time.time() - t0
        self._maybe_log()

    def _read_sync(self, key, protect: set) -> None:
        """Read one expert on this thread (one preadv into an aligned buffer,
        then pinned staging, then an async copy into its slot)."""
        t0 = time.perf_counter()
        l, e = key
        off = (l * self.E + e) * self.block
        mv = memoryview(self._sync_mm)
        want = (self.block + 4095) & ~4095
        got = 0
        while got < want:
            r = os.preadv(self.fds[0], [mv[got:want]], off + got)
            if r <= 0:
                raise OSError(f"short read at expert {key}")
            got += r
        t1 = time.perf_counter()
        slot = self._alloc_slot(protect)
        if self.sync_copy == "pageable":
            with torch.xpu.stream(self.copy_stream):
                self.pool[slot].copy_(self._sync_t[: self.raw])
            self.copy_stream.synchronize()  # the read buffer is reused next call
        else:
            buf = self.staging.get()
            src = self.staging.bufs[buf][: self.raw]
            if self._sync_pool is not None:
                # torch runs with 1 thread in the workers: split the host copy
                # over 4 threads (ctypes.memmove releases the GIL)
                d0 = src.data_ptr()
                list(self._sync_pool.map(
                    lambda j: ctypes.memmove(d0 + j[0], self._sync_addr + j[0], j[1]),
                    self._sync_jobs))
            else:
                src.copy_(self._sync_t[: self.raw])
            with torch.xpu.stream(self.copy_stream):
                self.pool[slot].copy_(src, non_blocking=self.staging.pinned)
                ev = torch.xpu.Event()
                ev.record(self.copy_stream)
            if self.staging.pinned:
                self.staging.release_after(buf, ev)
            else:
                self.staging.release(buf)
        if self.sync_verify:
            # test mode: the slot must hold exactly the bytes read from the drive
            self.copy_stream.synchronize()
            ok = torch.equal(self.pool[slot].cpu(), self._sync_t[: self.raw])
            self.stats["v_n"] = self.stats.get("v_n", 0) + 1
            self.stats["v_bad"] = self.stats.get("v_bad", 0) + (not ok)
            if not ok or self.stats["v_n"] % 500 == 0:
                logger.info("sync read verify: %d checked, %d wrong (last %s %s)",
                            self.stats["v_n"], self.stats["v_bad"], key, "ok" if ok else "WRONG")
        self.lru[key] = slot
        self.slot_cpu[l, e] = slot
        self._dirty.add(l)
        s = self.stats
        s["misses"] += 1
        s["read_bytes"] += self.block
        if self.in_decode:
            s["s_n"] = s.get("s_n", 0) + 1
            s["s_read"] = s.get("s_read", 0.0) + (t1 - t0)
            s["s_rest"] = s.get("s_rest", 0.0) + (time.perf_counter() - t1)

    def _row_to_dev(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        """Slot-table row to the device. A plain copy from pageable memory makes
        the host wait for the stream to drain; with VLLM_DS41_ASYNC_ROWS=1 (default) the row
        goes through a ring of pinned buffers instead (stream-ordered, no host
        wait). A ring entry is reused 64 rows later, long after its copy ran:
        every MoE call waits for its routing, which follows earlier copies."""
        if self._rows is None:
            dst.copy_(src)
            return
        b = self._rows[self._row_i % len(self._rows)]
        self._row_i += 1
        b.copy_(src)
        dst.copy_(b, non_blocking=True)

    def wait_copies(self) -> None:
        """Order the current stream after every slot copy issued so far."""
        torch.xpu.current_stream(self.device).wait_stream(self.copy_stream)

    def sync_table(self) -> None:
        """Bring every dirty device slot row up to date (stream-ordered), so
        kernels launched before the host sees the routing read valid slots."""
        if self._dirty:
            if len(self._dirty) > 4:
                self.slot_dev.copy_(self.slot_cpu)
            else:
                for layer in self._dirty:
                    self._row_to_dev(self.slot_dev[layer], self.slot_cpu[layer])
            self._dirty.clear()

    def idx_host(self, n: int, dtype) -> torch.Tensor:
        """Pinned host buffer for a layer's routing ids (reused; callers wait
        on the copy before the next use)."""
        b = getattr(self, "_ih", None)
        if b is None or b.numel() < n or b.dtype != dtype:
            try:
                b = torch.empty(max(n, 256), dtype=dtype, pin_memory=True)
            except RuntimeError:
                b = torch.empty(max(n, 256), dtype=dtype)
            self._ih = b
        return b[:n]

    def fix_row(self, layer: int, experts: list[int]) -> torch.Tensor:
        """Device slot row holding only `experts` (others -1): a top-up pass
        for the pairs whose experts were missing at launch."""
        r = torch.full((self.E,), -1, dtype=torch.int32)
        for e in experts:
            r[e] = int(self.slot_cpu[layer, e])
        if self._rows is None:
            return r.to(self.device)
        d = torch.empty(self.E, dtype=torch.int32, device=self.device)
        self._row_to_dev(d, r)
        return d

    def _make_room(self, protect: set) -> None:
        """Free a staging buffer for a demand read: install landed prefetches,
        cancel queued ones, wait for a started one to finish."""
        while not self.staging.free and not self.staging.pending:
            self._flush_prefetched(protect)
            if self.staging.free or self.staging.pending:
                return
            with self.qlock:
                for _, _, f, _ in self.q:
                    if f.prio > 0 and not f.started and not f.cancel:
                        f.cancel = True
                        self.stats["pf_dropped"] += 1
                        if self.inflight.get(f.key) is f:
                            del self.inflight[f.key]
                        self._zombies.append(f)
            with self.done_cv:
                self.done_cv.wait(0.01)

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
        want = set(keys)
        with self.qlock:
            old = {id(f): f for _, _, f, _ in self.q if f.prio > 0 and f.key not in want}
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

    _tm = None

    def note_timing(self, T: int, sync_s, ensure_s, launch_s, gpu_s, layer: int = -1,
                    nmiss: int = 0) -> None:
        """Per-call host times of the streamed MoE (decode steps only), split
        by whether the call had to read experts, plus misses per layer."""
        if T > 4:
            return
        if self._tm is None:
            self._tm = [0, 0.0, 0.0, 0.0, 0.0, time.time(), 0, 0.0, 0]
            self._tm_layer = [0] * self.L
        tm = self._tm
        tm[0] += 1; tm[1] += sync_s; tm[2] += ensure_s; tm[3] += launch_s; tm[4] += gpu_s
        if nmiss:
            tm[6] += 1; tm[7] += ensure_s; tm[8] += nmiss
            if 0 <= layer < self.L:
                self._tm_layer[layer] += nmiss
        if tm[0] % 4000 == 0:
            n = tm[0]
            nm = max(tm[6], 1)
            nh = max(n - tm[6], 1)
            logger.info("expert stream timing: %d calls, per call ms: routing sync %.3f, "
                        "ensure %.3f, kernel launch %.3f, kernel run (sync) %.3f; "
                        "%.1f calls/s", n, 1e3 * tm[1] / n, 1e3 * tm[2] / n, 1e3 * tm[3] / n,
                        1e3 * tm[4] / n, n / (time.time() - tm[5]))
            logger.info("expert stream misses: %.1f%% of calls read experts (%.2f reads each), "
                        "ensure ms: %.3f with reads, %.3f without; misses per layer: %s",
                        100 * tm[6] / n, tm[8] / nm, 1e3 * tm[7] / nm,
                        1e3 * (tm[2] - tm[7]) / nh, ",".join(map(str, self._tm_layer)))
            self._tm = None

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
                    "prefetch; %d prefetches, %d dropped, %d landed+used, %d evicted unused), "
                    "%.1f GiB read, wait %.1f s",
                    len(self.lru), 100.0 * s["hits"] / max(tot, 1), s["hits"], s["misses"],
                    s["pf_used"], s["pf_issued"], s["pf_dropped"], s["pf_hit"], s["pf_wasted"],
                    s["read_bytes"] / 2**30, s["wait_s"])
        if s.get("s_n"):
            n = s["s_n"]
            logger.info("expert sync read ms (decode, %d): read %.3f, staging + copy launch %.3f",
                        n, 1e3 * s["s_read"] / n, 1e3 * s["s_rest"] / n)
        if s.get("f_n"):
            n = s["f_n"]
            logger.info("expert fetch ms (demand, %d): queued %.3f, read+copy %.3f, landed->noticed "
                        "%.3f, install %.3f", n, 1e3 * s["f_queue"] / n, 1e3 * s["f_read"] / n,
                        1e3 * s["f_seen"] / n, 1e3 * s["f_inst"] / n)
