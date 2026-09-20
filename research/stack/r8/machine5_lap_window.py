"""Lap machine (owner 2026-09-21, follow-up to machine5_cycle.py): laps k index the copies
(30k-1, 30k+1) of machine 5's known opening. Gear g >= 7 strikes lap k iff k = +-30^{-1} mod g.
The lap machine q = gears 7..q acting on laps. Its window: laps k with 30k-1 > q and
30k+1 < q'^2 (q' the next prime above q): an open lap there is a twin by the square-root rule.
Report (1) the lap machine's record (longest run of struck laps over a period) for q <= 23, exact;
(2) for q up to 300, the open laps in the window, the first one, and the longest struck run of
laps inside the window (laps are columns 5k of the 6n+-1 machine, so runs are read on laps).
"""
from sympy import primerange, nextprime
from math import prod
ps = list(primerange(7, 400))
def struck(k, gears):
    return any((30 * k - 1) % g == 0 or (30 * k + 1) % g == 0 for g in gears)
def record(gears, P):
    best = run = 0
    for k in range(1, P + 1):
        run = run + 1 if struck(k, gears) else 0
        best = max(best, run)
    return best
print("lap-machine record (longest struck run of laps, one period), gears 7..q:")
for q in [7, 11, 13, 17, 19, 23]:
    gears = [g for g in ps if g <= q]
    print(f"  q={q:2d}: record {record(gears, prod(gears))} laps  (period {prod(gears)} laps)")
print("window on laps, (q/30, q'^2/30):")
for q in [g for g in ps if g <= 300]:
    gears = [g for g in ps if g <= q]; qn = nextprime(q)
    lo = q // 30 + 1; hi = (qn * qn - 2) // 30
    laps = list(range(lo, hi + 1))
    opens = [k for k in laps if not struck(k, gears)]
    best = run = 0
    for k in laps:
        run = run + 1 if struck(k, gears) else 0
        best = max(best, run)
    print(f"  q={q:3d}: window laps {lo}..{hi} ({len(laps):4d} laps), open laps {len(opens):3d}, first {opens[0] if opens else None} "
          f"(pair {30*opens[0]-1},{30*opens[0]+1})" if opens else f"  q={q}: NO OPEN LAP", f" longest struck run in window {best}")
