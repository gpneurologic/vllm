"""Routing statistics for streaming DeepSeek V4.1 routed experts.

Reads a trace written with VLLM_DS41_ROUTE_TRACE=<dir> (see
vllm/models/deepseek_v41/xpu/expert_stats.py) and reports what an expert
streaming design needs to know:

  decode   - experts reused from the previous token, new experts per token,
             working set over N tokens, LRU hit rate vs cache size, bytes and
             bandwidth per generated token, per-card load (expert parallel)
  predict  - how well the next layer's (and the one after's) router, applied
             to this layer's MoE input, predicts the experts that layer picks
  prefill  - distinct experts per layer per prefill chunk
  batch    - distinct experts per step with B decode streams at once

Usage: python3 ds41_route_stats.py <trace dir> [--n-experts 384] [--top-k 6]
           [--expert-mb 18.80] [--cards 3] [--json out.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import OrderedDict, defaultdict

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--n-experts", type=int, default=384)
ap.add_argument("--expert-mb", type=float, default=18.80,
                help="bytes per expert in MB (1e6): full FP4 = 18.80, IQ2_XXS ~9.1, IQ1_M ~7.7")
ap.add_argument("--cards", type=int, default=3)
ap.add_argument("--json")
a = ap.parse_args()
E, MB = a.n_experts, a.expert_mb

# ---- load: fwd -> layer -> (idx [T,k], pred1, pred2)
steps: dict[int, dict[int, tuple]] = defaultdict(dict)
for f in sorted(glob.glob(os.path.join(a.trace, "trace_*.pt"))):
    for fwd, layer, idx, p1, p2 in torch.load(f):
        steps[fwd][layer] = (idx.long().numpy(), None if p1 is None else p1.long().numpy(),
                             None if p2 is None else p2.long().numpy())
fwds = sorted(steps)
layers = sorted({l for s in steps.values() for l in s})
L = len(layers)
K = next(iter(steps[fwds[0]].values()))[0].shape[1]
print(f"trace: {len(fwds)} forward passes, {L} MoE layers, top-{K} of {E} experts, "
      f"{MB:.2f} MB per expert")

# ---- split into requests: a forward with T > 1 starts a request (prefill)
reqs = []  # each: {"prefill": [fwd...], "decode": [fwd...]}
for f in fwds:
    T = steps[f][layers[0]][0].shape[0]
    if T > 1:
        if not reqs or reqs[-1]["decode"]:
            reqs.append({"prefill": [], "decode": []})
        reqs[-1]["prefill"].append(f)
    else:
        if not reqs:
            reqs.append({"prefill": [], "decode": []})
        reqs[-1]["decode"].append(f)
for i, r in enumerate(reqs):
    pt = sum(steps[f][layers[0]][0].shape[0] for f in r["prefill"])
    print(f"  request {i}: prompt {pt} tokens in {len(r['prefill'])} chunks, {len(r['decode'])} decode tokens")
out = {"n_experts": E, "top_k": K, "layers": L, "expert_mb": MB,
       "requests": [{"prompt": sum(steps[f][layers[0]][0].shape[0] for f in r["prefill"]),
                     "decode": len(r["decode"])} for r in reqs]}

# decode token sequences per request: list over tokens of {layer: set(experts)}
dec = [[{l: set(steps[f][l][0][0].tolist()) for l in layers} for f in r["decode"]]
       for r in reqs if len(r["decode"]) >= 16]
ntok = sum(len(s) for s in dec)
print(f"\n=== decode ({ntok} tokens in {len(dec)} requests)")

# 1. reuse from the previous W tokens
print("\nExperts of a token already used by one of the previous W tokens (same layer):")
reuse = {}
for W in (1, 2, 4, 8, 16, 32, 64, 128):
    hit = tot = 0
    for seq in dec:
        for t in range(W, len(seq)):
            for l in layers:
                prev = set().union(*(seq[t - j][l] for j in range(1, W + 1)))
                hit += len(seq[t][l] & prev)
                tot += K
    if not tot:
        continue
    reuse[W] = hit / tot
    new = (1 - reuse[W]) * K * L
    print(f"  W={W:4d}: {100 * reuse[W]:5.1f}% reused -> {new:6.1f} new experts/token "
          f"({new * MB / 1e3:.2f} GB/token)")
out["reuse_prev_W"] = reuse

# 2. working set: distinct experts per layer over N consecutive tokens
print("\nDistinct experts per layer over N consecutive tokens (mean over layers and windows):")
ws = {}
for N in (1, 4, 16, 64, 256):
    vals = []
    for seq in dec:
        for s in range(0, len(seq) - N + 1, max(1, N // 2)):
            for l in layers:
                vals.append(len(set().union(*(seq[t][l] for t in range(s, s + N)))))
    if vals:
        ws[N] = float(np.mean(vals))
        print(f"  N={N:4d}: {ws[N]:6.1f} of {E} ({100 * ws[N] / E:4.1f}%), "
              f"all layers {ws[N] * L * MB / 1e3:6.1f} GB")
out["working_set"] = ws


# 3. LRU per layer, capacity C experts per layer (cache starts warm: first
#    request's tokens only warm it up, counted from token 32 on)
def lru_rate(seqs, cap):
    miss = tot = 0
    for l in layers:
        c: OrderedDict = OrderedDict()
        n = 0
        for seq in seqs:
            for tok in seq:
                for e in tok[l]:
                    if e in c:
                        c.move_to_end(e)
                    else:
                        if n >= 32:
                            miss += 1
                        c[e] = 1
                        if len(c) > cap:
                            c.popitem(last=False)
                    if n >= 32:
                        tot += 1
                n += 1
    return miss / max(tot, 1)


print("\nLRU cache per layer (requests back to back, cache kept across requests):")
print("  per-layer slots   VRAM (all layers)   hit rate   experts read/token   GB/token")
lru = {}
for cap in (6, 12, 24, 48, 64, 96, 128, 192, 256, 320, E):
    if cap > E:
        continue
    m = lru_rate(dec, cap)
    lru[cap] = m
    print(f"  {cap:6d} ({100 * cap / E:4.0f}%)" + (" all" if cap == E else "")
          + f"   {cap * L * MB / 1e3:7.1f} GB          "
          f"{100 * (1 - m):5.1f}%     {m * K * L:8.1f}            {m * K * L * MB / 1e3:6.2f}")
out["lru_miss"] = lru

# 4. frequency skew: pin the most used experts (counted on the other half of
#    the requests, so it is not fitted to the tokens it is scored on)
print("\nPinned (static) experts per layer, chosen by use count in the other half of the requests:")
half = len(dec) // 2 or 1
pin = {}
for frac in (0.1, 0.25, 0.5, 0.75):
    n = int(E * frac)
    cov = []
    for train, test in ((dec[:half], dec[half:]), (dec[half:], dec[:half])):
        if not test:
            continue
        for l in layers:
            cnt = np.zeros(E)
            for seq in train:
                for tok in seq:
                    cnt[list(tok[l])] += 1
            top = set(np.argsort(-cnt)[:n].tolist())
            h = sum(len(tok[l] & top) for seq in test for tok in seq)
            cov.append(h / (K * sum(len(s) for s in test)))
    pin[frac] = float(np.mean(cov))
    print(f"  top {100 * frac:3.0f}% ({n} experts/layer, {n * L * MB / 1e3:6.1f} GB): "
          f"{100 * pin[frac]:5.1f}% of picks covered")
out["pinned_coverage"] = pin

# 5. per card (expert parallel, contiguous split): the slowest card sets the pace
per = E // a.cards
loads = []
for seq in dec:
    for t in range(1, len(seq)):
        new_per_card = np.zeros(a.cards)
        for l in layers:
            for e in seq[t][l] - seq[t - 1][l]:
                new_per_card[min(e // per, a.cards - 1)] += 1
        loads.append(new_per_card)
loads = np.array(loads)
print(f"\nNew experts per token per card (reuse only from the previous token), {a.cards} cards:")
print(f"  mean per card {loads.mean():.1f}, mean of the busiest card {loads.max(1).mean():.1f} "
      f"(x{loads.max(1).mean() / max(loads.mean(), 1e-9):.2f} imbalance)")
out["card_new_mean"] = float(loads.mean()); out["card_new_max_mean"] = float(loads.max(1).mean())

# 6. look-ahead prediction
print("\n=== look-ahead prediction (router of layer l+j on the MoE input of layer l)")
pred = {}
for j, slot in ((1, 1), (2, 2)):
    hit = tot = 0
    for f in fwds:
        for i, l in enumerate(layers):
            p = steps[f][l][slot]
            if p is None or i + j >= L:
                continue
            act = steps[f][layers[i + j]][0]
            for r in range(act.shape[0]):
                hit += len(set(p[r].tolist()) & set(act[r].tolist()))
                tot += K
    if tot:
        pred[j] = hit / tot
        print(f"  {j} layer(s) ahead: {100 * pred[j]:5.1f}% of the experts predicted correctly")
out["lookahead"] = pred
if 1 in pred:
    both = []
    for seq_r in [r["decode"] for r in reqs if len(r["decode"]) >= 16]:
        for t in range(1, len(seq_r)):
            h = tt = 0
            for i, l in enumerate(layers[:-1]):
                p = set(steps[seq_r[t]][l][1][0].tolist())
                prev = set(steps[seq_r[t - 1]][layers[i + 1]][0][0].tolist())
                act = set(steps[seq_r[t]][layers[i + 1]][0][0].tolist())
                h += len(act & (p | prev)); tt += K
            both.append(h / tt)
    if both:
        print(f"  previous token's experts + 1-layer-ahead prediction: {100 * np.mean(both):5.1f}% covered")
        out["lookahead_plus_prev"] = float(np.mean(both))

# 7. prefill: distinct experts per layer per chunk
print("\n=== prefill: distinct experts per layer in a chunk of C prompt tokens")
pf = {}
for r in reqs:
    rows = [np.concatenate([steps[f][l][0] for f in r["prefill"]]) for l in layers] if r["prefill"] else []
    if not rows or rows[0].shape[0] < 256:
        continue
    T = rows[0].shape[0]
    for C in (16, 64, 256, 512, 1024, 2048, 4096):
        if C > T:
            continue
        v = [len(np.unique(x[s:s + C])) for x in rows for s in range(0, T - C + 1, C)]
        pf.setdefault(C, []).extend(v)
for C, v in sorted(pf.items()):
    m = float(np.mean(v))
    pf[C] = m
    print(f"  C={C:5d}: {m:6.1f} of {E} ({100 * m / E:5.1f}%) -> streaming all of them: "
          f"{m * L * MB / 1e3:6.1f} GB per chunk, {m * L * MB / C:7.1f} MB per prompt token")
out["prefill_distinct"] = pf

# 8. batched decode: B streams side by side (different requests)
print("\n=== batched decode: distinct experts per layer per step with B streams")
bat = {}
for B in (1, 2, 4, 8):
    if len(dec) < B:
        continue
    n = min(len(s) for s in dec[:B])
    v = [len(set().union(*(dec[b][t][l] for b in range(B)))) for t in range(n) for l in layers]
    bat[B] = float(np.mean(v))
    print(f"  B={B}: {bat[B]:5.1f} distinct per layer per step ({bat[B] / B:4.1f} per stream)")
out["batch_distinct"] = bat

print("\n=== bandwidth needed (decode, streamed bytes only; MB = 1e6 bytes)")
for label, gbt in (("reuse previous token only", (1 - reuse[1]) * K * L * MB / 1e3),
                   *((f"LRU {c} slots/layer ({c * L * MB / 1e3:.0f} GB)", m * K * L * MB / 1e3)
                     for c, m in lru.items() if c in (48, 96, 128, 192, 256))):
    print(f"  {label:36s} {gbt:6.2f} GB/token -> " +
          ", ".join(f"{tps} tok/s: {gbt * tps:6.1f} GB/s" for tps in (5, 10, 20, 40)))
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
