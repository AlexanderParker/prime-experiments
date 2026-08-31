"""Round 28 (mechanic): THE PEAK DEPTH OF THE QUALIFYING SPECTRUM.

Brief item (d).  Round 27 found Q_8(37;14) = 112 TURNING OVER from Q_7 = 114,
and Formalist had found the same shape at m31 (68, 85, 90, 91, 90, 88).  The
question is whether the peak depth is a measurable object across machines.

WHAT IS CHEAP.  Q_j(M; a) = max sum of j consecutive gaps whose j-2 MIDDLE gaps
are all >= a.  Over a machine's cyclic gap array this is a two-line numpy
computation: let R[i] be the number of consecutive gaps >= a starting at i and
S the prefix sums; then the window (i, j) qualifies iff R[i+1] >= j-2, and

    Q_j = max { S[i+j] - S[i]  :  R[i+1] >= j-2 }.

So EVERY depth of every machine up to 23 is exact in seconds - no transfer, no
solver, no seeding - and the full profile (not just the first few depths C13
printed) is available, including where it peaks, where it turns over and where
it goes vacuous.

Usage:  <venv>/python research/peak_depth_r28.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from onset_ladder_r28 import gaps_cyclic              # noqa: E402

# a = 2u'(q') with q' the NEXT gear - the floor C13 uses for each machine
NEXTGEAR = {11: 13, 13: 17, 17: 19, 19: 23, 23: 29}
JMAX = 14


def profile(y, a):
    g = gaps_cyclic(y).astype(np.int64)
    N = len(g)
    gg = np.concatenate([g, g[:JMAX + 2]])
    S = np.concatenate([[0], np.cumsum(gg)])
    big = gg >= a
    # R[i] = run of consecutive `big` starting at i (capped at JMAX)
    R = np.zeros(len(gg) + 1, np.int64)
    for i in range(len(gg) - 1, -1, -1):
        R[i] = R[i + 1] + 1 if big[i] else 0
    out = {}
    for j in range(2, JMAX + 1):
        need = max(0, j - 2)
        ok = R[1:N + 1] >= need
        if not ok.any():
            out[j] = 0
            continue
        idx = np.flatnonzero(ok)
        out[j] = int((S[idx + j] - S[idx]).max())
    return out


def main():
    print("EXACT QUALIFYING SPECTRA, ALL DEPTHS, from the cyclic period\n")
    print("  machine  a   " + "".join("%5d" % j for j in range(2, JMAX + 1)))
    rows = {}
    for y, qp in sorted(NEXTGEAR.items()):
        a = 2 * round(qp / 6)
        p = profile(y, a)
        rows[y] = p
        print("     m%-2d  %2d   " % (y, a)
              + "".join("%5d" % p[j] for j in range(2, JMAX + 1)))
    # GATE: C13's published rows, Q_3..Q_7 at each machine's own floor.  The
    # vehicle here (direct cyclic period) shares no code with
    # qualifying_spectrum.py, which produced them.
    C13 = {11: [16, 18, 20, 0, 0], 13: [18, 23, 0, 0, 0],
           17: [28, 31, 32, 34, 0], 19: [35, 37, 38, 0, 0],
           23: [43, 50, 55, 60, 0]}
    for y, row in C13.items():
        got = [rows[y][j] for j in range(3, 8)]
        assert got == row, ("C13 MISMATCH at m%d" % y, got, row)
    print("\n  GATE: every Q_3..Q_7 entry reproduces C13 exactly, by a vehicle "
          "sharing no code with it")

    print("\n  machine  peak value  peak depth  turns over?  vacuous from")
    for y in sorted(rows):
        p = rows[y]
        nz = [j for j in range(2, JMAX + 1) if p[j] > 0]
        peak = max(nz, key=lambda j: (p[j], -j))
        vac = min((j for j in range(2, JMAX + 1) if p[j] == 0), default=None)
        after = [p[j] for j in nz if j > peak]
        turns = bool(after) and max(after) < p[peak]
        print("     m%-2d      %4d        %3d        %-11s  %s"
              % (y, p[peak], peak, "YES" if turns else "no", vac))
    print("\n  KNOWN AT THE BIGGER MACHINES (C13 / C34 / Formalist):")
    print("     m31 a=12  Q_2..Q_8 = 68 85 90 91 90 88 0   peak 5, TURNS OVER")
    print("     m37 a=14  Q_2..Q_8 = 90 97 103 110 112 114 112  peak 7, "
          "TURNS OVER")


if __name__ == "__main__":
    main()
