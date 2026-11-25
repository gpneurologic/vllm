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
            self.slot_dev[layer].copy_(self.slot_cpu[layer], non_blocking=False)
            self._dirty.discard(layer)
        self.stats["wait_s"] += time.time() - t0
        self._maybe_log()

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
                    self.slot_dev[layer].copy_(self.slot_cpu[layer])
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
        return r.to(self.device)

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

    def note_timing(self, T: int, sync_s, ensure_s, launch_s, gpu_s) -> None:
        """Per-call host times of the streamed MoE (decode steps only)."""
        if T > 4:
            return
        if self._tm is None:
            self._tm = [0, 0.0, 0.0, 0.0, 0.0, time.time()]
        tm = self._tm
        tm[0] += 1; tm[1] += sync_s; tm[2] += ensure_s; tm[3] += launch_s; tm[4] += gpu_s
        if tm[0] % 4000 == 0:
            n = tm[0]
            logger.info("expert stream timing: %d calls, per call ms: routing sync %.3f, "
                        "ensure %.3f, kernel launch %.3f, kernel run (sync) %.3f; "
                        "%.1f calls/s", n, 1e3 * tm[1] / n, 1e3 * tm[2] / n, 1e3 * tm[3] / n,
                        1e3 * tm[4] / n, n / (time.time() - tm[5]))
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
