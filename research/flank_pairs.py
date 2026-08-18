"""Round 16: WHICH flank pairs attain FS_max, and part (D) at alpha = 3.

(D) at alpha = 3:  FS_max(w) <= F + q' - span(w)   for every compatible
qualifying word w. Equivalent (identity, r12): incr_k <= q'.

Reported per step and word: FS_max, the argmax pair (gL, gR), the largest
single flank, F, the alpha=3 requirement and margin, and the alpha=2.5 margin
for comparison. The question this settles: is the binding configuration
"both flanks maximal" (which tier A excludes) or mid-size pairs (which it
does not touch)?
"""
import numpy as np
import sys
sys.path.insert(0, "research")
from fuel_bound import gapword
from word_ceiling import words, valid_starts, FK, STEPS


def analyse(y, q1):
    g = gapword(y).astype(np.int32)
    n = len(g)
    F = FK[y]
    print(f"\n=== {y}->{q1}   F={F}   budget a=3: FS <= F + {q1} - span")
    for w in words(q1):
        if not valid_starts(w, q1):
            continue
        ell, span = len(w), sum(w)
        m = g[1:n - ell] == w[0]
        for j in range(1, ell):
            m &= g[1 + j:n - ell + j] == w[j]
        idx = np.flatnonzero(m) + 1
        if len(idx) == 0:
            continue
        lf, rf = g[idx - 1], g[idx + ell]
        fs = lf + rf
        j = int(np.argmax(fs))
        need3, need25 = F + q1 - span, F + 2.5 * q1 / 3 - span
        pair = (int(lf[j]), int(rf[j]))
        print(f"  w={str(w):14s} span={span:3d} FS_max={int(fs.max()):3d} "
              f"at (gL,gR)={str(pair):9s} maxflank={max(int(lf.max()), int(rf.max())):3d} "
              f"({max(int(lf.max()), int(rf.max()))/F:.2f}F)  "
              f"need3={need3:5.1f} margin3={need3-int(fs.max()):+6.1f}  "
              f"margin2.5={need25-int(fs.max()):+6.1f}"
              f"{'   <-- BINDING' if need3 - int(fs.max()) < 8 else ''}")


if __name__ == "__main__":
    for y, q1 in STEPS:
        analyse(y, q1)
