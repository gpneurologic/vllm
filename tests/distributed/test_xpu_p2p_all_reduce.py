# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-process test for XpuP2PAllReduce. Needs 2+ Intel GPUs with P2P access
and the library built by tools/build_xpu_p2p_all_reduce.sh.

    python tests/distributed/test_xpu_p2p_all_reduce.py --world 3 [--bench]

Checks, per dtype and size, that every rank gets exactly the rank-order float
sum of all inputs (bit for bit), out-of-place and in place, with many calls
queued back to back, and when replayed from a recorded XPU graph.
"""

import argparse
import datetime
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

DTYPES = [torch.float16, torch.bfloat16, torch.float32]
HIDDEN = 5120


def _input(rank: int, step: int, numel: int, dtype: torch.dtype) -> torch.Tensor:
    g = torch.Generator().manual_seed(step * 1009 + rank * 7 + DTYPES.index(dtype))
    return (torch.randn(numel, generator=g) * 4).to(dtype)


def _expected(world: int, step: int, numel: int, dtype: torch.dtype) -> torch.Tensor:
    acc = torch.zeros(numel, dtype=torch.float32)
    for r in range(world):  # same order as the kernel, so results match exactly
        acc += _input(r, step, numel, dtype).float()
    return acc.to(dtype)


def _worker(rank: int, world: int, args, port: int, fails) -> None:
    from vllm.distributed.device_communicators.xpu_p2p_all_reduce import (
        XpuP2PAllReduce,
    )

    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world)
    os.environ.setdefault("CCL_ATL_TRANSPORT", "ofi")
    dev = torch.device(f"xpu:{rank}")
    torch.xpu.set_device(dev)
    dist.init_process_group(
        "xccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world,
        timeout=datetime.timedelta(seconds=120),
    )
    cpu_group = dist.new_group(backend="gloo")
    ar = XpuP2PAllReduce(cpu_group, dev, max_bytes=args.max_bytes)
    assert not ar.disabled, "P2P all-reduce failed to set up"
    errors = []

    def check(name, got, exp):
        if not torch.equal(got.cpu(), exp):
            bad = int((got.cpu() != exp).sum())
            errors.append(f"rank {rank}: {name}: {bad} wrong of {exp.numel()}")

    max_elems = ar.max_bytes // 4
    for dtype in DTYPES:
        esz = torch.tensor([], dtype=dtype).element_size()
        # 16-byte LL words hold three 4-byte words: cover all remainders mod 3
        numels = sorted(
            {2, 10, 512, HIDDEN, 3 * HIDDEN, 3 * HIDDEN + 4, 6 * HIDDEN}
            | {ar.max_bytes // esz, ar.max_bytes // esz - 4 // esz}
        )
        numels = [n for n in numels if n * esz <= ar.max_bytes and (n * esz) % 4 == 0]
        assert numels and max_elems > 0
        for numel in numels:
            step = 0
            # back to back: queue a batch of calls, check them all afterwards
            for _ in range(args.batches):
                outs, steps = [], []
                for k in range(args.batch):
                    step += 1
                    x = _input(rank, step, numel, dtype).to(dev)
                    inplace = k % 2 == 1
                    y = ar.all_reduce(x, inplace=inplace)
                    assert (y is x) == inplace
                    outs.append(y)
                    steps.append(step)
                torch.xpu.synchronize()
                for s, y in zip(steps, outs):
                    check(
                        f"{dtype} n={numel} step={s}",
                        y,
                        _expected(world, s, numel, dtype),
                    )
            if rank == 0:
                print(f"  eager ok: {dtype} numel={numel}", flush=True)

        # graph: record once, replay with new data each time
        numel = 3 * HIDDEN if 3 * HIDDEN * esz <= ar.max_bytes else max_elems
        static_in = torch.zeros(numel, dtype=dtype, device=dev)
        torch.xpu.synchronize()
        dist.barrier(group=cpu_group)
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            static_out = ar.all_reduce(static_in)
        torch.xpu.synchronize()
        dist.barrier(group=cpu_group)
        for s in range(10_000, 10_000 + args.replays):
            static_in.copy_(_input(rank, s, numel, dtype).to(dev))
            graph.replay()
            torch.xpu.synchronize()
            check(
                f"graph {dtype} step={s}", static_out, _expected(world, s, numel, dtype)
            )
        if rank == 0:
            print(
                f"  graph ok: {dtype} numel={numel} replays={args.replays}", flush=True
            )
        del graph

    timeouts = ar.timeouts()
    if timeouts:
        errors.append(f"rank {rank}: {timeouts} poll timeouts")

    if args.bench:
        for tokens in args.bench_tokens:
            n = tokens * HIDDEN
            if n * 2 > ar.max_bytes:
                continue
            x = torch.randn(n, dtype=torch.float16, device=dev)
            quantum = world * 64

            def padded(x=x):  # what XpuCommunicator does when n % quantum != 0
                p = x.new_zeros(n + (-n) % quantum)
                p[:n].copy_(x)
                dist.all_reduce(p)
                return p[:n]

            res = {}
            fns = [
                ("p2p", lambda x=x: ar.all_reduce(x, inplace=True)),
                ("oneCCL", lambda x=x: dist.all_reduce(x)),
            ]
            if n % quantum:
                fns.append(("padded", padded))
            for name, fn in fns:
                for _ in range(50):
                    fn()
                torch.xpu.synchronize()
                times = []
                for _ in range(5):
                    dist.barrier(group=cpu_group)
                    t0 = time.perf_counter()
                    for _ in range(500):
                        fn()
                    torch.xpu.synchronize()
                    times.append((time.perf_counter() - t0) / 500 * 1e6)
                res[name] = statistics.median(times)
            if rank == 0:
                print(
                    f"  bench fp16 {tokens:2d} tok ({n * 2:6d} B): "
                    f"p2p {res['p2p']:6.1f} us  oneCCL {res['oneCCL']:6.1f} us"
                    + (
                        f"  oneCCL+pad {res['padded']:6.1f} us"
                        if "padded" in res
                        else ""
                    ),
                    flush=True,
                )

    ar.close()
    dist.barrier(group=cpu_group)
    dist.destroy_process_group()
    if errors:
        fails.put("\n".join(errors[:20]))


def run(world: int, args) -> list[str]:
    ctx = mp.get_context("spawn")
    fails = ctx.SimpleQueue()
    port = 29000 + os.getpid() % 1000
    mp.spawn(_worker, args=(world, args, port, fails), nprocs=world, join=True)
    out = []
    while not fails.empty():
        out.append(fails.get())
    return out


try:
    import pytest
except ImportError:  # the script mode below doesn't need pytest
    pytest = None

if pytest is not None:

    @pytest.mark.skipif(
        not (
            hasattr(torch, "xpu")
            and torch.xpu.is_available()
            and torch.xpu.device_count() >= 2
        ),
        reason="needs 2+ Intel GPUs",
    )
    @pytest.mark.parametrize("world", [2, 3])
    def test_xpu_p2p_all_reduce(world):
        if world > torch.xpu.device_count():
            pytest.skip(f"needs {world} GPUs")
        args = argparse.Namespace(
            max_bytes=256 * 1024,
            batches=2,
            batch=50,
            replays=50,
            bench=False,
            bench_tokens=[],
        )
        assert run(world, args) == []


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--world", type=int, default=torch.xpu.device_count())
    p.add_argument("--max-bytes", type=int, default=256 * 1024)
    p.add_argument("--batches", type=int, default=4)
    p.add_argument("--batch", type=int, default=100)
    p.add_argument("--replays", type=int, default=200)
    p.add_argument("--bench", action="store_true")
    p.add_argument(
        "--bench-tokens",
        type=lambda s: [int(t) for t in s.split(",")],
        default=[1, 3, 6, 8],
        help="comma-separated token counts (x 5120 fp16) to benchmark",
    )
    a = p.parse_args()
    problems = run(a.world, a)
    print("FAIL\n" + "\n".join(problems) if problems else "PASS")
    sys.exit(1 if problems else 0)
