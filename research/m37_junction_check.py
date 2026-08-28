"""Round 24 (mechanic): close the last hole in the m37 chained-scan audit.

The three chained fuel_census runs tile [0, P) but a RESUMED run starts with
an empty tail, so a gap - and hence any F_j window - STRADDLING a junction
(1.2e11 or 6e11) was examined by NEITHER run, and the cyclic wrap gap at P
was examined by neither end.  F(37), F_2, F_3 have independent SAT anchors;
F_4/F_5/F_6 = 105/113/120 were single-source.  This examines every window of
up to 8 consecutive gaps that touches a junction (openings within +-250
slots; the deepest window spans <= 120 slots, so +-250 covers all straddling
windows with margin) and asserts none exceeds the recorded spectrum.
"""
import numpy as np
from math import prod

GEARS = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
P = prod(GEARS)
FJ = {1: 88, 2: 90, 3: 97, 4: 105, 5: 113, 6: 120}
HALF = 250

def openings(lo, hi):
    ex = np.zeros(hi - lo, bool)
    for g in GEARS:
        u = pow(6, -1, g)
        ex[(u - lo) % g::g] = True
        ex[((-u) - lo) % g::g] = True
    return np.flatnonzero(~ex).astype(np.int64) + lo

ok = True
for b in (120_000_000_000, 600_000_000_000, P):     # P = the cyclic wrap
    if b == P:
        left = openings(P - HALF, P)
        right = openings(0, HALF) + P
        op = np.concatenate([left, right])
    else:
        op = openings(b - HALF, b + HALF)
    d = np.diff(op)
    # only windows CONTAINING the junction point b are new information
    for j in range(1, 7):
        best = 0
        for i in range(len(d) - j + 1):
            if op[i] < b <= op[i + j]:              # window straddles b
                best = max(best, int(op[i + j] - op[i]))
        line = f"  junction {b}: max {j}-window straddling = {best}  " \
               f"(recorded F_{j} = {FJ[j]})"
        print(line)
        assert best <= FJ[j], (b, j, best, FJ[j])
        ok &= best <= FJ[j]
print("==> NO straddling window exceeds the recorded spectrum; "
      "F_j(37) = 88 90 97 105 113 120 now holds over the FULL period "
      "with no junction caveat." if ok else "VIOLATION")
