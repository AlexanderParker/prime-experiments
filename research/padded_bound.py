"""Constructor round 14: the padding question from the tolerance side.

A padded link = two kills at the SAME tooth, so their separation is = 0 mod q'
and the interior gap is >= q' (smallest: exactly q'). Qualifying interior
values are V(q') = {v : v = 0, +2c or -2c mod q'}; literal letters are the two
smallest nonzero ones (2u', q'-2u'), padded letters are the multiples of q'.

ARITHMETIC OF PADDING (the shape question). A k-chain merges k+1 consecutive
gaps: merged = gL + (interiors) + gR. Tolerance needs merged <= F + 2.5q'/3
(k-frame). With p padded interiors, span >= p*q', so

    FS_max <= F + 2.5q'/3 - span <= F - (p - 5/6) q'.

p = 1 already forces FS < F - q'/6: a padded occurrence CANNOT carry a
near-maximal flank. p = 2 forces FS < F - 7q'/6, and p = 3 is impossible for
F < 13q'/6 - padding is self-limiting through the budget, not through a cap.

Equivalently the whole tier is the QUALIFYING SPECTRUM
    Q^qual_{k+1} = max sum of k+1 consecutive gaps whose middle k-1 are all
                   in V(q'), with >= 1 padded,
and tolerance = "Q^qual_{k+1} - F <= 2.5q'/3".

Computed per step: Q^qual by depth, the budget, the padded-gap census (how
COMMON the conditioning object is), and the minimum opening-distance from a
maximal gap to a padded gap (the non-clustering statement, measured).
"""
import numpy as np
import sys
sys.path.insert(0, "research")
from fuel_bound import gapword
from word_ceiling import FK, F2K, FNEW, STEPS


def analyse(y, q1):
    g = gapword(y).astype(np.int32)
    n = len(g)
    F = FK[y]
    c = pow(6, -1, q1)
    budget = 2.5 * q1 / 3
    qual = ((g % q1 == 0) | (g % q1 == (2 * c) % q1) |
            (g % q1 == (-2 * c) % q1))
    padded = (g % q1 == 0)
    npad = int(padded.sum())
    print(f"\n=== {y}->{q1}  F={F}  F2={F2K[y]}  actual={FNEW[q1]} "
          f"(incr {FNEW[q1]-F})  budget {budget:.2f}")
    print(f"  padded gaps (= 0 mod {q1}): {npad:,} of {n:,} "
          f"({100*npad/n:.3f}%)  sizes {sorted(set(g[padded].tolist()))[:6]}")
    rows = []
    for k in range(2, 6):                      # k kills -> k+1 gaps
        w = k + 1
        okmid = np.ones(n - w, bool)
        anyp = np.zeros(n - w, bool)
        for j in range(1, k):
            okmid &= qual[j:n - w + j]
            anyp |= padded[j:n - w + j]
        sel = np.flatnonzero(okmid & anyp)
        if len(sel) == 0:
            rows.append(f"k={k}: none")
            continue
        s = np.zeros(len(sel), np.int64)
        for j in range(w):
            s += g[sel + j]
        mx = int(s.max())
        rows.append(f"k={k}: Q^qual={mx} ({mx-F:+d}) "
                    f"[{'ok' if mx - F <= budget else 'OVER'}] n={len(sel):,}")
    print("  " + "  |  ".join(rows))
    # non-clustering: distance (in openings) from a maximal gap to a padded gap
    maxi = np.flatnonzero(g == F)
    padi = np.flatnonzero(padded)
    if len(maxi) and len(padi):
        d = np.abs(padi[np.searchsorted(padi, maxi).clip(0, len(padi) - 1)]
                   - maxi)
        d2 = np.abs(padi[(np.searchsorted(padi, maxi) - 1).clip(0)] - maxi)
        dm = int(np.minimum(d, d2).min())
        print(f"  max-gap instances {len(maxi)}; min opening-distance from a "
              f"maximal gap to a padded gap: {dm} "
              f"(need > 1 for the k=2 padded window)")


if __name__ == "__main__":
    for y, q1 in STEPS:
        analyse(y, q1)
