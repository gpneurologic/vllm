"""Plan per-layer contiguous expert splits that balance routed load across 3 cards.

python3 split_plan.py <stats.pt> [current spec] [slack]
Load of a card in a layer = routed (token, expert) pairs that land on its experts.
Each layer waits for its slowest card, so the cost is sum over layers of the max
load. The plan keeps contiguous ranges (the loader's layout) and keeps each card's
total expert count within `slack` of the current split's (memory)."""
import sys

import torch

stats = torch.load(sys.argv[1])
spec = sys.argv[2] if len(sys.argv) > 2 else "63,64,65;63,65,64"
slack = int(sys.argv[3]) if len(sys.argv) > 3 else 8
pats = [[int(c) for c in p.split(",")] for p in spec.split(";")]
layers = sorted(stats)
E, TP = stats[layers[0]].numel(), 3


def loads(c, counts):
    out, s = [], 0
    for n in counts:
        out.append(int(c[s:s + n].sum()))
        s += n
    return out


cur_cost = mean_cost = 0
cur_tot = [0] * TP
rows = []
for L in layers:
    c = stats[L].double()
    counts = pats[L % len(pats)]
    ld = loads(c, counts)
    cur_cost += max(ld)
    mean_cost += sum(ld) / TP
    for r in range(TP):
        cur_tot[r] += counts[r]
    rows.append((L, c, counts, ld))
print(f"{len(layers)} MoE layers, {int(sum(sum(r[3]) for r in rows))} routed pairs")
print(f"current split: sum of per-layer max load / mean = {cur_cost / mean_cost:.3f} "
      f"(1.000 = perfect); experts per card {cur_tot}")
worst = sorted(rows, key=lambda r: -max(r[3]) / (sum(r[3]) / TP))[:5]
for L, _, counts, ld in worst:
    print(f"  layer {L}: counts {counts} loads {ld} -> max/mean {max(ld) / (sum(ld) / TP):.2f}")

# Per layer: all (a, b) boundaries; pick to minimize max load, then fix memory
# totals by a greedy pass that trades the cheapest layers.
cands = []
for L, c, counts, ld in rows:
    cs = torch.cat([torch.zeros(1, dtype=torch.float64), c.cumsum(0)])
    opts = []
    for a in range(48, 81):
        for b in range(a + 48, min(E - 48, a + 80) + 1):
            n = (a, b - a, E - b)
            lds = (cs[a].item(), (cs[b] - cs[a]).item(), (cs[E] - cs[b]).item())
            opts.append((max(lds), n))
    opts.sort()
    cands.append((L, opts))
# Exact DP over layers: state = (card 0, card 1) expert-count drift from the
# current split (card 2 follows); each layer may use its K best options.
K, D = 60, 3 * slack + 24
cur_counts = {L: counts for L, _, counts, _ in rows}
dp = {(0, 0): (0.0, [])}
for L, opts in cands:
    cc = cur_counts[L]
    nxt = {}
    for (d0, d1), (cost, path) in dp.items():
        for m, n in opts[:K]:
            e0, e1 = d0 + n[0] - cc[0], d1 + n[1] - cc[1]
            if abs(e0) > D or abs(e1) > D or abs(e0 + e1) > D:
                continue
            c = cost + m
            old = nxt.get((e0, e1))
            if old is None or c < old[0]:
                nxt[(e0, e1)] = (c, path + [(L, (m, n))])
    dp = nxt
ok = [(v[0], v[1]) for (d0, d1), v in dp.items()
      if abs(d0) <= slack and abs(d1) <= slack and abs(d0 + d1) <= slack]
if not ok:
    raise SystemExit("no split within the memory slack found")
best = min(ok, key=lambda x: x[0])
choice = {L: mn for L, mn in best[1]}


def totals():
    return [sum(choice[L][1][r] for L in choice) for r in range(TP)]


new_cost = sum(choice[L][0] for L in choice)
print(f"planned split: sum of max / mean = {new_cost / mean_cost:.3f}; experts per card "
      f"{totals()} (current {cur_tot})")
print(f"expected routed-expert time (load-bound) change: {(new_cost / cur_cost - 1) * 100:+.1f}%")
# one pattern per layer index 0..max (layers without MoE get the even split)
maxL = max(layers)
spec_out = ";".join(",".join(map(str, choice[L][1])) if L in choice else "64,64,64"
                    for L in range(maxL + 1))
print("SPEC " + spec_out)
