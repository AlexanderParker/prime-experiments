"""Per-gear fragile counts vs the closed-form prediction (mechanic round 2).

For each gear q, the round-1 constant-2 law predicts

    pred(q) = 2 * twins * ((q-1)/(q-2)) * S1(q) / pi_win

where S1(q) = lone-composite members owned by q (exactly one distinct gear
divisor, = q), pi_win = degree-0 prime members in (y, y^2], twins = degree-0
slots. Observed frag(q) = fragile slots whose composite member is owned by q.
Semiprime variant used throughout (S1/frag restricted to q*p and q^2 shapes);
loose aggregate reported for reference.

Special attention: the top-gear tail, where S1(q) -> O(1) (gear y owns only
its square y^2) and necessity events are rare. Poisson z = (obs-pred)/sqrt(pred)
per band; a healthy law has |z| ~ 1 and no systematic drift.

Refined (size-corrected) prediction also computed: the partner-prime
probability is ~ c/ln(m), so replace counts by 1/ln(m) sums:

    pred2(q) = 2 * twins * ((q-1)/(q-2)) * S1w(q) / piw,
    S1w(q) = sum over lone-q members of 1/ln(m),
    piw    = sum over degree-0 prime members of 1/ln(m).

If the mid-band deficit of pred1 vanishes under pred2, the deficit is pure
member-size geometry (large-q lone composites live only in (q*y, y^2)).

Usage: uv run python research/fragile_pergear.py [y]   (default 10007)
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fragile_census import primes_upto


def pergear(y, seg=4_000_000):
    gears = [q for q in primes_upto(y) if q >= 5]
    G = len(gears)
    garr = np.array(gears, dtype=np.int64)
    k_lo = -((-(y - 1)) // 6)
    k_hi = (y * y + 1) // 6
    y2 = y * y
    twins = 0
    primes_in = 0
    piw = 0.0
    frag_semi_q = np.zeros(G, dtype=np.int64)
    frag_loose_q = np.zeros(G, dtype=np.int64)
    s1_semi_q = np.zeros(G, dtype=np.int64)
    s1_loose_q = np.zeros(G, dtype=np.int64)
    s1w_semi_q = np.zeros(G, dtype=np.float64)
    uvals = [pow(6, -1, q) for q in gears]
    u2vals = [pow(6, -1, q * q) for q in gears]
    for a in range(k_lo, k_hi + 1, seg):
        b = min(k_hi + 1, a + seg)
        n = b - a
        cntL = np.zeros(n, np.int16)
        cntR = np.zeros(n, np.int16)
        ownL = np.zeros(n, np.int32)
        ownR = np.zeros(n, np.int32)
        sqL = np.zeros(n, bool)
        sqR = np.zeros(n, bool)
        top_member = 6 * (b - 1) + 1
        for q, u, u2 in zip(gears, uvals, u2vals):
            s = (u - a) % q
            cntL[s::q] += 1
            ownL[s::q] = q
            s = (-u - a) % q
            cntR[s::q] += 1
            ownR[s::q] = q
            q2 = q * q
            if q2 <= top_member:
                sqL[(u2 - a) % q2::q2] = True
                sqR[(-u2 - a) % q2::q2] = True
        kk = np.arange(a, b, dtype=np.int64)
        mL = 6 * kk - 1
        mR = 6 * kk + 1
        pL = cntL == 0
        pR = cntR == 0
        twins += int((pL & pR).sum())
        for pmask, m in ((pL & (mL <= y2), mL), (pR & (mR <= y2), mR)):
            primes_in += int(pmask.sum())
            piw += float((1.0 / np.log(m[pmask])).sum())
        for cnt, m, own, sq, partner_p in (
                (cntL, mL, ownL, sqL, pR), (cntR, mR, ownR, sqR, pL)):
            o = own.astype(np.int64)
            lone = (cnt == 1) & (m != o) & (m > y) & (m <= y2)
            lsem = lone & (~sq | (m == o * o))
            frag = lone & partner_p
            fsem = lsem & partner_p
            for mask, acc in ((lone, s1_loose_q), (lsem, s1_semi_q),
                              (frag, frag_loose_q), (fsem, frag_semi_q)):
                qs = own[mask].astype(np.int64)
                if len(qs):
                    acc += np.bincount(np.searchsorted(garr, qs), minlength=G)
            idx = np.searchsorted(garr, own[lsem].astype(np.int64))
            s1w_semi_q += np.bincount(
                idx, weights=1.0 / np.log(m[lsem]), minlength=G)
    return dict(gears=garr, twins=twins, primes_in=primes_in, piw=piw,
                frag_semi=frag_semi_q, frag_loose=frag_loose_q,
                s1_semi=s1_semi_q, s1_loose=s1_loose_q, s1w_semi=s1w_semi_q)


def report(r, y):
    g = r["gears"].astype(float)
    G = len(g)
    tw, pi_w = r["twins"], r["primes_in"]
    pred = 2.0 * tw * ((g - 1) / (g - 2)) * r["s1_semi"] / pi_w
    pred2 = 2.0 * tw * ((g - 1) / (g - 2)) * r["s1w_semi"] / r["piw"]
    obs = r["frag_semi"].astype(float)
    print(f"y={y}  twins={tw}  pi_win={pi_w}  gears={G}")
    print(f"total: obs {obs.sum():.0f}  pred {pred.sum():.1f} "
          f"(ratio {obs.sum()/pred.sum():.4f})  pred2 {pred2.sum():.1f} "
          f"(ratio {obs.sum()/pred2.sum():.4f})")
    print("\nsmallest gears individually (semi variant):")
    print(f"{'q':>7} {'S1(q)':>10} {'obs':>9} {'pred':>11} {'obs/pred':>9} {'z':>7}")
    for i in range(min(10, G)):
        z = (obs[i] - pred[i]) / pred[i] ** 0.5 if pred[i] > 0 else float('nan')
        print(f"{r['gears'][i]:>7} {r['s1_semi'][i]:>10} {obs[i]:>9.0f} "
              f"{pred[i]:>11.1f} {obs[i]/pred[i]:>9.4f} {z:>7.2f}")
    print("\nrank bands (semi variant; pred2 = size-corrected):")
    bands = [(0.0, 0.5), (0.5, 0.9), (0.9, 0.99), (0.99, 1.0)]
    print(f"{'band':>12} {'#gears':>7} {'S1':>11} {'obs':>10} {'obs/pred':>9} "
          f"{'z':>7} {'obs/pred2':>10} {'z2':>7}")
    for lo, hi in bands:
        i0, i1 = int(lo * G), max(int(hi * G), int(lo * G) + 1)
        o = obs[i0:i1].sum()
        p1, p2 = pred[i0:i1].sum(), pred2[i0:i1].sum()
        s1 = int(r["s1_semi"][i0:i1].sum())
        z1 = (o - p1) / p1 ** 0.5 if p1 > 0 else float('nan')
        z2 = (o - p2) / p2 ** 0.5 if p2 > 0 else float('nan')
        print(f"{f'{lo:.0%}-{hi:.0%}':>12} {i1-i0:>7} {s1:>11} {o:>10.0f} "
              f"{o/p1:>9.4f} {z1:>7.2f} {o/p2:>10.4f} {z2:>7.2f}")
    print("\ntop 10 gears individually (rare-event tail, semi variant):")
    print(f"{'q':>9} {'S1(q)':>7} {'obs':>5} {'pred':>8} {'pred2':>8} {'z2':>7}")
    for i in range(max(0, G - 10), G):
        p2 = pred2[i]
        z2 = (obs[i] - p2) / p2 ** 0.5 if p2 > 0 else float('nan')
        print(f"{r['gears'][i]:>9} {r['s1_semi'][i]:>7} {obs[i]:>5.0f} "
              f"{pred[i]:>8.3f} {p2:>8.3f} {z2:>7.2f}")
    predL = 2.0 * tw * ((g - 1) / (g - 2)) * r["s1_loose"] / pi_w
    oL = r["frag_loose"].sum()
    print(f"\nloose aggregate: obs {oL}  pred {predL.sum():.1f}  "
          f"ratio {oL/predL.sum():.4f}")


if __name__ == "__main__":
    y = int(sys.argv[1]) if len(sys.argv) > 1 else 10007
    t0 = time.time()
    r = pergear(y)
    report(r, y)
    print(f"\n{time.time()-t0:.1f}s")
