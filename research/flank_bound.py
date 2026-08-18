"""Round 12 follow-up: the flank-sum bound - the one missing input, measured.

The identity gives incr = max_w [span(w) + FS_max(w)] - F. Tolerance needs
FS_max(w) <= F + (2.5q'/3 - span(w)), i.e. the two gaps flanking a
compatible-word occurrence cannot BOTH be near-maximal. This is round 9-10's
adjacency question with a word in between: an (ell+2)-point correlation
instead of the 3-point A3.

Reported per step: per compatible word, max left flank, max right flank,
max sum, whether a TOP-STRATUM gap (= F) ever flanks an occurrence, and the
margin FS_max - (F + 2.5q'/3 - span).
"""
import numpy as np
import sys
sys.path.insert(0, "research")
from fuel_bound import literal_cap, gapword
from word_ceiling import words, valid_starts, FK, F2K, FNEW, STEPS


def analyse(y, q1):
    gaps = gapword(y).astype(np.int32)
    n = len(gaps)
    F = FK[y]
    print(f"\n=== {y}->{q1}  F={F}  F2={F2K[y]}  actual F(M+q')={FNEW[q1]}  "
          f"budget incr {2.5*q1/3:.1f}")
    for w in words(q1):
        if not valid_starts(w, q1):
            continue
        ell = len(w)
        m = gaps[1:n - ell] == w[0]
        for j in range(1, ell):
            m &= gaps[1 + j:n - ell + j] == w[j]
        idx = np.flatnonzero(m) + 1
        if len(idx) == 0:
            continue
        lf, rf = gaps[idx - 1], gaps[idx + ell]
        fs = lf + rf
        span = sum(w)
        allow = F + 2.5 * q1 / 3 - span      # FS may not exceed this
        topL = int((lf == F).sum())
        topR = int((rf == F).sum())
        print(f"  w={w}: occ={len(idx):,}  maxL={int(lf.max())} "
              f"maxR={int(rf.max())} FS_max={int(fs.max())}  "
              f"top-stratum flanks: L {topL}, R {topR}  "
              f"allowance {allow:.1f} -> margin {allow - int(fs.max()):+.1f}")


if __name__ == "__main__":
    for y, q1 in STEPS:
        analyse(y, q1)
