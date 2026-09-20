"""Phases in the chains (node c.x follow-up). For the 36 two-hole chains at q = 23 and the 96 filled
holes at q = 19: the phase of each gear (h0 mod g), and the fill diagram (gear per position per
hole). PRE-REGISTERED: at q = 19 a filled hole is 11 at phase 3 or 6 (its double holes), 13 at a
position not taken by 11, and 17, 19 on the remaining two positions; the filled set is the union of
the CRT residue classes of these diagrams, 2 x 3 x 2 x 2 x 2 x 2 = 96 classes. At q = 23 the two-hole
chains use 11's windows "14|5" (phases 3) and "1|25" (phase 5) or its 2-windows, and 13's staircase
phases; the relation between 11's and 13's phases is what is read.
"""
from math import prod
from collections import Counter
import numpy as np
def strikers(h, gears):
    return [[g for g in gears if (30*(7*h+4+p)-1) % g == 0 or (30*(7*h+4+p)+1) % g == 0] for p in range(1, 6)]
gears = [11, 13, 17, 19]; P = prod(gears)
filled = [h for h in range(P) if all(strikers(h, gears))]
ph = Counter((h % 11, h % 13) for h in filled)
print("q=19 filled holes: (phase of 11, phase of 13) ->", dict(sorted(ph.items())))
print("  11 phases:", sorted({h % 11 for h in filled}), " 13 phases:", sorted({h % 13 for h in filled}),
      " 17 phases:", sorted({h % 17 for h in filled}), " 19 phases:", sorted({h % 19 for h in filled}))
diagrams = Counter(tuple(s[0] for s in strikers(h, gears)) for h in filled)
print("  diagrams (gear at positions 1..5):", len(diagrams), "distinct;", dict(diagrams))
gears = [11, 13, 17, 19, 23]; P = prod(gears)
struck = np.zeros(7 * P + 20, dtype=bool)
for g in [7] + gears:
    a = pow(30, -1, g); struck[a::g] = True; struck[(-a) % g::g] = True
f = struck[5:5 + 7 * P].reshape(P, 7)[:, :5].all(axis=1)
starts = np.flatnonzero(f[:-1] & f[1:])
print(f"q=23 two-hole chains: {len(starts)}")
for h in starts:
    D = [[tuple(s) for s in strikers(h + i, gears)] for i in range(2)]
    print(f"  h0={h:7d} phases 11:{h%11:2d} 13:{h%13:2d} 17:{h%17:2d} 19:{h%19:2d} 23:{h%23:2d}  diagram {D[0]} | {D[1]}")
