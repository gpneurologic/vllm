#!/usr/bin/env python3
"""Repack the routed experts of a full-expert DeepSeek V4.1 GGUF (all 384
experts, e.g. IQ2_XXS gate/up + Q2_K down) into the streaming layout: one
contiguous 4 KiB-aligned block per expert, read with one O_DIRECT read.

Block (bytes): w13 [2*I rows of gate then up, GGUF row bytes over K=H]
               w2  [H rows, GGUF row bytes over K=I]
Expert (layer l, expert e) at offset (l * n_experts + e) * block.

usage: repack_gguf.py <file.gguf> <out dir> [--threads 8] [--verify N]
Needs numpy only (gguf_ckpt.py is imported with a torch stub)."""
import argparse, json, mmap, os, random, sys, threading, time, types
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("gguf"); ap.add_argument("out")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--verify", type=int, default=0)
ap.add_argument("--ckpt-module", default="/home/neurologic/src/vllm-ds41-live/vllm/models/deepseek_v41/xpu")
a = ap.parse_args()
if "torch" not in sys.modules:
    sys.modules["torch"] = types.SimpleNamespace(float32=0, float16=0, bfloat16=0, Tensor=object,
                                                 tensor=lambda *x, **k: None)
sys.path.insert(0, a.ckpt_module)
import gguf_ckpt as G

meta, tens = G._parse(a.gguf, False)
T = {t.name: t for t in tens}
L = 1 + max(int(n.split(".")[1]) for n in T if n.startswith("blk.") and n.endswith("ffn_gate_exps.weight"))
g0, u0, d0 = (T[f"blk.0.ffn_{w}_exps.weight"] for w in ("gate", "up", "down"))
E, I, H = g0.shape
assert d0.shape == (E, H, I), d0.shape
for l in range(L):
    for w, ref in (("gate", g0), ("up", u0), ("down", d0)):
        t = T[f"blk.{l}.ffn_{w}_exps.weight"]
        assert t.type_id == ref.type_id and t.shape == ref.shape, (t.name, t.type_name, t.shape)
gb = I * g0.row_bytes          # gate (or up) bytes of one expert
db = H * d0.row_bytes          # down bytes of one expert
RAW = 2 * gb + db
BLOCK = (RAW + 4095) & ~4095
layout = {
    "format": "ds41-gguf-experts-v1", "layers": L, "experts": E, "hidden": H, "inter": I,
    "block_bytes": BLOCK, "raw_bytes": RAW,
    "offsets": {"w13_weight": 0, "w2_weight": 2 * gb},
    "qtypes": {"w13": g0.type_id, "w2": d0.type_id},
    "qtype_names": {"w13": g0.type_name, "w2": d0.type_name},
    "row_bytes": {"w13": g0.row_bytes, "w2": d0.row_bytes},
    "file": "experts_q2.bin", "source": os.path.realpath(a.gguf),
}
print(json.dumps(layout))
src = os.open(a.gguf, os.O_RDONLY | os.O_DIRECT)


def read(off, nb, mv):
    """O_DIRECT read of [off, off+nb) into mv[:nb] (mv is a page-aligned scratch)."""
    a0 = off & ~4095; a1 = (off + nb + 4095) & ~4095
    got = 0
    while got < a1 - a0:
        r = os.preadv(src, [mv[got:a1 - a0]], a0 + got)
        if r <= 0:
            raise OSError("short read")
        got += r
    return mv[off - a0: off - a0 + nb]


local = threading.local()


def build(l, e, out):
    if not hasattr(local, "mm"):
        local.mm = mmap.mmap(-1, max(gb, db) + 8192); local.mv = memoryview(local.mm)
    o = 0
    for w, nb in (("gate", gb), ("up", gb), ("down", db)):
        t = T[f"blk.{l}.ffn_{w}_exps.weight"]
        out[o:o + nb] = read(t.offset + e * nb, nb, local.mv)
        o += nb


os.makedirs(a.out, exist_ok=True)
outp = os.path.join(a.out, layout["file"])
if a.verify:
    vfd = os.open(outp, os.O_RDONLY)
    bad = 0
    for _ in range(a.verify):
        l, e = random.randrange(L), random.randrange(E)
        ref = bytearray(BLOCK); build(l, e, ref)
        if bytes(ref[:RAW]) != os.pread(vfd, RAW, (l * E + e) * BLOCK):
            bad += 1; print("MISMATCH", l, e)
    print(f"verify: {a.verify - bad}/{a.verify} ok"); sys.exit(1 if bad else 0)

wfd = os.open(outp, os.O_RDWR | os.O_CREAT | os.O_DIRECT, 0o644)
os.posix_fallocate(wfd, 0, L * E * BLOCK)
json.dump(layout, open(os.path.join(a.out, "layout.json"), "w"), indent=1)
done = [0]; lock = threading.Lock(); t0 = time.time()
wl = threading.local()


def work(le):
    l, e = le
    if not hasattr(wl, "mm"):
        wl.mm = mmap.mmap(-1, BLOCK); wl.mv = memoryview(wl.mm)
    wl.mv[RAW:BLOCK] = bytes(BLOCK - RAW)
    build(l, e, wl.mv)
    n = os.pwritev(wfd, [wl.mv], (l * E + e) * BLOCK)
    assert n == BLOCK
    with lock:
        done[0] += 1
        if done[0] % 1536 == 0:
            dt = time.time() - t0
            print(f"{done[0]}/{L * E} experts, {done[0] * BLOCK / dt / 1e9:.2f} GB/s", flush=True)


with ThreadPoolExecutor(a.threads) as ex:
    list(ex.map(work, [(l, e) for l in range(L) for e in range(E)]))
os.fsync(wfd)
print(f"REPACK DONE {L * E} experts, {L * E * BLOCK / 2**30:.1f} GiB in {time.time() - t0:.0f} s")
