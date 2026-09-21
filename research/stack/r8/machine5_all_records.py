"""All record runs by period scan (node c.xii follow-up; the solver enumeration was too slow).
For q = 19, 23, 29 scan one lap period (7 x 11 x ... x q laps, numpy), find every run of record
length, and read what all record runs share: first- and last-lap strikers, the open laps on either
side, wasted strikes (coincidences), mirror pairing (k -> -k maps runs to runs).
"""
import numpy as np
from sympy import primerange
from math import prod
for q in [19, 23, 29]:
    gears = list(primerange(7, q + 1)); P = prod(gears)
    cls = {g: (pow(30, -1, g), (-pow(30, -1, g)) % g) for g in gears}
    struck = np.zeros(P + 64, dtype=bool)
    for g in gears:
        struck[cls[g][0]::g] = True; struck[cls[g][1]::g] = True
    d = np.diff(np.concatenate(([0], struck[:P + 64].astype(np.int8), [0])))
    st = np.flatnonzero(d == 1); en = np.flatnonzero(d == -1); runs = en - st
    L = int(runs.max()); k0s = sorted({int(k) for k, r in zip(st, runs) if r == L and k < P})
    def hits(k): return tuple(g for g in gears if k % g in cls[g])
    from collections import Counter
    first = Counter(hits(k) for k in k0s); last = Counter(hits(k + L - 1) for k in k0s)
    before = Counter(min(j for j in range(1, 64) if hits(k - j)) for k in k0s)
    after = Counter(min(j for j in range(1, 64) if hits(k + L - 1 + j)) for k in k0s)
    waste = Counter(sum(len(hits(k + i)) - 1 for i in range(L)) for k in k0s)
    mirror = all(((-k - L + 1) % P) in set(k0s) for k in k0s)
    sevens = Counter(tuple(i for i in range(L) if (k + i) % 7 in cls[7]) for k in k0s)
    print(f"q={q}: record {L}, record runs in the period: {len(k0s)}, mirror-closed: {mirror}", flush=True)
    print(f"   first-lap strikers {dict(first)}\n   last-lap strikers {dict(last)}\n   open laps before {dict(before)}, after {dict(after)}\n   wasted strikes {dict(waste)}\n   7's laps inside the run {dict(sevens)}", flush=True)
