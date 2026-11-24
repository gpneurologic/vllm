#!/usr/bin/env python3
"""Repack DeepSeek V4.1 routed experts (FP4 + ue8m0 scales) into one file with
one contiguous, 4 KiB-aligned block per expert (one read per cache miss).

Block layout (bytes), N1 = moe_inter (2304), H = hidden (5120):
  w13 weight [2*N1, H/2]   (w1 rows then w3 rows)
  w2  weight [H, N1/2]
  w13 scale  [2*N1, H/32]
  w2  scale  [H, N1/32]
padded to 4 KiB. Expert (layer l, expert e) at offset (l * n_experts + e) * block.

usage: repack.py <checkpoint folder> <out dir> [--threads 8] [--layers 0-39] [--verify N]
"""
import argparse, json, mmap, os, struct, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--layers", default="")
ap.add_argument("--verify", type=int, default=0, help="only verify N random experts")
a = ap.parse_args()
src = os.path.dirname(os.path.realpath(os.path.join(a.src, "config.json")))
cfg = json.load(open(os.path.join(src, "config.json")))
cfg = cfg.get("text_config", cfg)
L, E, H, N1 = cfg["num_hidden_layers"], cfg["n_routed_experts"], cfg["hidden_size"], cfg["moe_intermediate_size"]

index = {}
for f in sorted(os.listdir(src)):
    if not f.endswith(".safetensors"):
        continue
    with open(os.path.join(src, f), "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    for k, v in hdr.items():
        if k != "__metadata__" and ".ffn.experts." in k and k.startswith("layers."):
            o0, o1 = v["data_offsets"]
            index[k] = (f, 8 + n + o0, o1 - o0, v["dtype"], v["shape"])

def name(l, e, w, k): return f"layers.{l}.ffn.experts.{e}.{w}.{k}"
t = index[name(0, 0, "w1", "weight")]; t2 = index[name(0, 0, "w2", "weight")]
s1 = index[name(0, 0, "w1", "scale")]; s2 = index[name(0, 0, "w2", "scale")]
assert t[4] == [N1, H // 2] and t2[4] == [H, N1 // 2], (t, t2)
assert s1[4] == [N1, H // 32] and s2[4] == [H, N1 // 32], (s1, s2)
parts = [("w1", "weight"), ("w3", "weight"), ("w2", "weight"), ("w1", "scale"), ("w3", "scale"), ("w2", "scale")]
offs, o = {}, 0
for p in parts:
    offs[p] = o; o += index[name(0, 0, *p)][2]
RAW = o
BLOCK = (RAW + 4095) & ~4095
layout = {
    "format": "ds41-fp4-experts-v1", "layers": L, "experts": E, "hidden": H, "inter": N1,
    "block_bytes": BLOCK, "raw_bytes": RAW,
    "offsets": {"w13_weight": offs[("w1", "weight")], "w2_weight": offs[("w2", "weight")],
                "w13_scale": offs[("w1", "scale")], "w2_scale": offs[("w2", "scale")]},
    "dtypes": {"weight": t[3], "scale": s1[3]}, "file": "experts_fp4.bin",
    "source": src,
}
os.makedirs(a.out, exist_ok=True)
outp = os.path.join(a.out, "experts_fp4.bin")
fds, lock = {}, threading.Lock()
def fd(f):
    with lock:
        if f not in fds:
            fds[f] = os.open(os.path.join(src, f), os.O_RDONLY | os.O_DIRECT)
        return fds[f]

def read_raw(f, off, nb):
    a0 = off & ~4095; a1 = (off + nb + 4095) & ~4095
    mm = mmap.mmap(-1, a1 - a0); mv = memoryview(mm); got = 0
    while got < a1 - a0:
        r = os.preadv(fd(f), [mv[got:]], a0 + got)
        if r <= 0:
            if got >= off + nb - a0: break
            raise OSError("short read")
        got += r
    return mv[off - a0: off - a0 + nb]

def build(l, e, buf):
    # group the expert's tensors by file and adjacency: few large reads
    names = [(p, index[name(l, e, *p)]) for p in parts]
    names.sort(key=lambda x: (x[1][0], x[1][1]))
    runs = []
    for p, ix in names:
        if runs and runs[-1][0] == ix[0] and 0 <= ix[1] - runs[-1][2] < 65536:
            runs[-1][2] = ix[1] + ix[2]; runs[-1][3].append((p, ix)); continue
        runs.append([ix[0], ix[1], ix[1] + ix[2], [(p, ix)]])
    for f, s, end, members in runs:
        mv = read_raw(f, s, end - s)
        for p, ix in members:
            buf[offs[p]: offs[p] + ix[2]] = mv[ix[1] - s: ix[1] - s + ix[2]]

if a.verify:
    import random
    vfd = os.open(outp, os.O_RDONLY)
    bad = 0
    for _ in range(a.verify):
        l, e = random.randrange(L), random.randrange(E)
        ref = bytearray(BLOCK); build(l, e, ref)
        got = os.pread(vfd, BLOCK, (l * E + e) * BLOCK)
        if bytes(ref) != got:
            bad += 1; print("MISMATCH", l, e)
    print(f"verify: {a.verify - bad}/{a.verify} ok"); sys.exit(1 if bad else 0)

layers = range(L)
if a.layers:
    lo, hi = a.layers.split("-"); layers = range(int(lo), int(hi) + 1)
total = L * E * BLOCK
wfd = os.open(outp, os.O_RDWR | os.O_CREAT | os.O_DIRECT, 0o644)
if not a.layers and os.fstat(wfd).st_size < total:
    os.posix_fallocate(wfd, 0, total)
json.dump(layout, open(os.path.join(a.out, "layout.json"), "w"), indent=1)
done = [0]; t0 = time.time()
local = threading.local()
def work(le):
    l, e = le
    if not hasattr(local, "mm"):
        local.mm = mmap.mmap(-1, BLOCK); local.mv = memoryview(local.mm)
    local.mv[RAW:BLOCK] = bytes(BLOCK - RAW)
    build(l, e, local.mv)
    off = (l * E + e) * BLOCK; got = 0
    while got < BLOCK:
        got += os.pwritev(wfd, [local.mv[got:]], off + got)
    with lock:
        done[0] += 1
        if done[0] % 500 == 0:
            dt = time.time() - t0
            print(f"{done[0]} experts, {done[0] * BLOCK / dt / 1e9:.2f} GB/s, {dt:.0f} s", flush=True)
with ThreadPoolExecutor(a.threads) as ex:
    list(ex.map(work, [(l, e) for l in layers for e in range(E)]))
os.fsync(wfd); os.close(wfd)
dt = time.time() - t0
print(f"done: {done[0]} experts, {done[0] * BLOCK / 1e9:.1f} GB in {dt:.0f} s ({done[0] * BLOCK / dt / 1e9:.2f} GB/s)")
