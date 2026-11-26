"""Device-side miss recording for streamed decode without per-layer host waits
(plan ONCE-PER-STEP-PLAN.md v2, Stage 1-lite).

record_misses(idx, w, slot_row, lo, n_local, rec, owner, key_base, bad): for every
routed pair (t, j) whose expert e = idx[t, j] is local (lo <= e < lo + n_local):
  rec[n_local + e - lo] = 1                       (used)
  if slot_row[e] < 0: rec[e - lo] = max(rec[e - lo], w[t, j] / sum_j w[t, :])   (miss share)
  else, if owner is given: bad += (owner[slot_row[e]] != key_base + e)          (race check)
`rec` is fp32 [2 * n_local] and must be zeroed by the caller. One launch, no host
sync; shares are >= 0, so the float atomic max is exact."""
import torch
import triton
import triton.language as tl


@triton.jit
def _record_kernel(idx_ptr, w_ptr, slot_ptr, rec_ptr, owner_ptr, bad_ptr, lo, n_local, key_base,
                   K: tl.constexpr, KP: tl.constexpr, CHECK: tl.constexpr):
    t = tl.program_id(0)
    j = tl.arange(0, KP)
    m = j < K
    e = tl.load(idx_ptr + t * K + j, mask=m, other=-1)
    w = tl.load(w_ptr + t * K + j, mask=m, other=0.0).to(tl.float32)
    share = w / tl.maximum(tl.sum(w, axis=0), 1e-20)
    loc = m & (e >= lo) & (e < lo + n_local)
    el = tl.where(loc, e - lo, 0)
    slot = tl.load(slot_ptr + tl.where(loc, e, 0), mask=loc, other=0)
    tl.atomic_max(rec_ptr + n_local + el, tl.full([KP], 1.0, tl.float32), mask=loc)
    tl.atomic_max(rec_ptr + el, share, mask=loc & (slot < 0))
    if CHECK:
        hit = loc & (slot >= 0)
        own = tl.load(owner_ptr + tl.where(hit, slot, 0), mask=hit, other=0)
        wrong = hit & (own != key_base + e)
        tl.atomic_add(bad_ptr + tl.zeros([KP], tl.int32), wrong.to(tl.int32), mask=hit)


def record_misses(idx, w, slot_row, lo: int, n_local: int, rec, owner=None, key_base: int = 0,
                  bad=None) -> None:
    T, K = idx.shape
    if T == 0:
        return
    chk = owner is not None
    _record_kernel[(T,)](idx, w, slot_row, rec, owner if chk else rec, bad if chk else rec,
                         lo, n_local, key_base, K=K, KP=triton.next_power_of_2(K), CHECK=chk)


def record_misses_ref(idx, w, slot_row, lo: int, n_local: int, owner=None, key_base: int = 0):
    """Python reference (host), for tests."""
    rec = torch.zeros(2 * n_local, dtype=torch.float32)
    bad = 0
    idx, w, slot_row = idx.cpu(), w.float().cpu(), slot_row.cpu()
    for t in range(idx.shape[0]):
        tot = max(float(w[t].sum()), 1e-20)
        for j in range(idx.shape[1]):
            e = int(idx[t, j])
            if lo <= e < lo + n_local:
                rec[n_local + e - lo] = 1.0
                s = int(slot_row[e])
                if s < 0:
                    rec[e - lo] = max(float(rec[e - lo]), float(w[t, j]) / tot)
                elif owner is not None and int(owner[s]) != key_base + e:
                    bad += 1
    return rec, bad


if __name__ == "__main__":  # unit test on one XPU
    torch.manual_seed(0)
    dev = torch.device("xpu")
    E, K, lo, nl, NS = 384, 6, 128, 128, 1000
    mism = 0
    for T in (1, 2, 4, 5, 8, 16):
        for trial in range(50):
            idx = torch.stack([torch.randperm(E)[:K] for _ in range(T)]).to(torch.int32)
            w = torch.rand(T, K) * 2.5
            slot_row = torch.where(torch.rand(E) < 0.3, -1, torch.randint(0, NS, (E,))).to(torch.int32)
            kb = 7 * E
            owner = torch.where(torch.rand(NS) < 0.1, -5, kb).to(torch.int32)  # mostly "matching" base
            # make owners match for the slots the row uses, except ~10% planted wrong
            for e2 in range(E):
                s2 = int(slot_row[e2])
                if s2 >= 0:
                    owner[s2] = kb + e2 if torch.rand(1).item() > 0.1 else -9
            rec = torch.zeros(2 * nl, dtype=torch.float32, device=dev)
            bad = torch.zeros(1, dtype=torch.int32, device=dev)
            record_misses(idx.to(dev), w.to(dev), slot_row.to(dev), lo, nl, rec, owner.to(dev), kb, bad)
            rr, rb = record_misses_ref(idx, w, slot_row, lo, nl, owner, kb)
            if not (torch.allclose(rec.cpu(), rr, atol=1e-6) and int(bad.item()) == rb):
                mism += 1
    torch.xpu.synchronize()
    print(f"STEP_MISS UNIT: {mism} mismatches in 300 cases")
