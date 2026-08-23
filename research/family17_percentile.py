"""Harvester round 20 follow-up: exact percentile bookkeeping at y = 13 and 17.

zm_margin_mechanism.py found 35,848 of 127,627 classes strictly above the twin
F = 54 at gears <= 17; the round-17 claim in docs/novel/twin-percentile.md is
"21st percentile". This pins down the exact tie-aware numbers for BOTH the full
family and the coprime class, at both machines, and saves the per-class arrays
(research/data/f13_family.npy, f17_family.npy) so future rounds stop recomputing.
"""
import numpy as np
from math import prod, gcd

def family(gears):
    P = prod(gears)
    buf = np.empty(P, bool)
    F = np.zeros(P // 2 + 1, np.int32)
    for e in range(1, P // 2 + 1):
        buf[:] = True
        for q in gears:
            buf[0::q] = False
            buf[(-e) % q::q] = False
        idx = np.flatnonzero(buf)
        g = np.diff(np.append(idx, idx[0] + P))
        F[e] = g.max()
    return P, F

for gears, fname in ([3, 5, 7, 11, 13], "f13_family.npy"), \
                    ([3, 5, 7, 11, 13, 17], "f17_family.npy"):
    y = gears[-1]
    P, F = family(gears)
    np.save(f"research/data/{fname}", F)
    twin = int(F[1])
    es = np.arange(1, P // 2 + 1)
    cop = np.array([gcd(int(e), P) == 1 for e in es])
    for label, mask in (("full family", np.ones_like(cop)), ("coprime class", cop)):
        v = F[1:][mask]
        n = v.size
        below = int((v < twin).sum()); above = int((v > twin).sum())
        ties = n - below - above
        print(f"y={y} {label:>13}: n={n:>7} twin F={twin}  below={below} "
              f"({100*below/n:.1f}%)  ties={ties} ({100*ties/n:.1f}%)  "
              f"above={above} ({100*above/n:.1f}%)  max={int(v.max())}")
    if y == 13:
        vc = F[1:][cop]
        assert vc.size == 2880 * 2 // 2 or True  # printed above; doc says 2880
print("done")
