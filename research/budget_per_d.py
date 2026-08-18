"""Harvester round 17: the PER-d BUDGET ARITHMETIC - closing my own flagged limit.

Route hypothesis: incr <= alpha * q' at every consecutive gear step, in ADJACENT
(= halved) units - the unit in which the corpus's alpha lives (twins' slot ratio
0.811 at 31->37 is 0.811 x 3 = 2.432 adjacent, against budgets 2.5 and 3).
Here: F_d(y) = max gap of the gap-2e pattern in halved coordinates
(n survives iff n != 0, -e mod q for every gear q <= y), exact full-period scan.

Verifies for each even d and each step M -> M+q':   incr / q' <= alpha ?
"""
import numpy as np
from math import prod

YS = [11, 13, 17, 19, 23, 29]

def maxgap(gears, e, CH=10_000_000):
    P = prod(gears)
    first = last = None
    best = 0
    for lo in range(0, P, CH):
        hi = min(lo + CH, P)
        a = np.ones(hi - lo, bool)
        for q in gears:
            a[(-lo) % q::q] = False
            a[((-e) - lo) % q::q] = False
        idx = np.flatnonzero(a)
        if idx.size:
            idx = idx + lo
            if first is None:
                first = int(idx[0])
            if last is not None:
                best = max(best, int(idx[0]) - last)
            if idx.size > 1:
                best = max(best, int(np.diff(idx).max()))
            last = int(idx[-1])
    return max(best, first + P - last)

print("d    e   gcd(e,105) cap |  step        q'   F_old   F_new   incr  incr/q'  "
      "a=2.5 a=3")
worst = {}
for d in (2, 4, 6, 10, 12, 30, 210):
    e = d // 2
    g = np.gcd(e, 105)
    cap = {1: 6, 3: 6, 5: 6, 7: 6, 21: 6, 35: 6, 15: 10, 105: 12}[int(g)]
    Fs = {}
    for y in YS:
        gears = [q for q in (3, 5, 7, 11, 13, 17, 19, 23, 29) if q <= y]
        Fs[y] = maxgap(gears, e)
    mx = 0
    for a, b in zip(YS, YS[1:]):
        incr = Fs[b] - Fs[a]
        r = incr / b
        mx = max(mx, r)
        ok25 = "OK" if r <= 2.5 else "OVER"
        ok3 = "OK" if r <= 3.0 else "OVER"
        print(f"{d:>3} {e:>4}   {int(g):>3}      {cap:>2} |  {a:>2}->{b:<3} {b:>6} "
              f"{Fs[a]:>7} {Fs[b]:>7} {incr:>6}  {r:>6.3f}   {ok25:<5} {ok3}")
    worst[d] = mx
    print()
print("WORST per-step ratio by d (adjacent units):")
for d, m in worst.items():
    print(f"  d={d:>3}: max incr/q' = {m:.3f}   "
          f"alpha=2.5 {'OK' if m <= 2.5 else 'FAILS'} | "
          f"alpha=3 {'OK' if m <= 3.0 else 'FAILS'}")
