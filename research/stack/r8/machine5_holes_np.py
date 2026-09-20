"""Hole machine (numpy). PRE-REGISTERED: (a) in the hole machine at any fixed position p, gear g's
two classes sit at hole distance d_g * 7^{-1} mod g (d_g = 15^{-1} mod g the lap distance),
predicted nearer representatives 11:2, 13:1, 17:6, 19:2, 23:7, 29:8, 31:13; (b) the lap record
of gears 7..29 equals 7 * chain + 2 + ends with chain = the longest run of filled holes.
Test on q = 29 over the full hole period (30808063 holes, 2.16e8 laps).
"""
import numpy as np
from math import prod
gears = [11, 13, 17, 19, 23, 29]
for g in gears + [31]:
    d = pow(15, -1, g); hd = (d * pow(7, -1, g)) % g
    print(f"gear {g}: lap distance {min(d, g-d)}, hole distance {min(hd, g-hd)}")
P = prod(gears); L = 7 * P + 20
struck = np.zeros(L, dtype=bool)
for g in [7] + gears:
    a = pow(30, -1, g)
    struck[a::g] = True; struck[(-a) % g::g] = True
struck[0] = False
# lap record 7..29
runs = np.diff(np.flatnonzero(np.diff(np.concatenate(([0], struck.astype(np.int8), [0])))))[::2]
print("lap record gears 7..29:", runs.max())
# holes: laps 7h+5..7h+9, gears >= 11 only (7 never strikes those laps)
H = struck[5:5 + 7 * P].reshape(P, 7)[:, :5]
filled = H.all(axis=1)
f = np.concatenate(([0], filled.astype(np.int8), [0]))
edges = np.flatnonzero(np.diff(f)); chains = edges[1::2] - edges[::2]
print("longest chain of filled holes at q=29:", chains.max(), "(filled holes", int(filled.sum()), "check only)")
h0 = edges[::2][chains.argmax()]
for h in range(h0, h0 + chains.max()):
    print("  hole", h, "positions struck by", [[g for g in gears if (30*(7*h+4+p)-1) % g == 0 or (30*(7*h+4+p)+1) % g == 0] for p in range(1, 6)])
# check (a) on the data: for gear 13, holes hit at position 3 by the two classes
for g in [13, 11, 19]:
    hs = [h for h in range(g) if any((30*(7*h+7)-1) % g == 0 or (30*(7*h+7)+1) % g == 0 for _ in [0])]
    print(f"gear {g} position 3 holes mod {g}: {hs}")
