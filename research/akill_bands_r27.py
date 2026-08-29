"""Round 27 (mechanic): THE REFUTED-SPAN BAND TABLE for machine 53.

Six completed scans, every one a set of range workers that TILES machine 23's
period exactly, each seeded at `lo` with span cap `hi` and each returning `lo`
(EMPTY) or a maximum.  A run seeded at lo with cap hi and reported maximum m
proves: NO window of the scanned family has span in (m, hi].  The family is
"word-legal for gear 59, depth J <= jmax" (floor-1 runs are the J = 2 case,
where the legality condition is vacuous).

A realised k-chain kill word at 53 -> 59 is k consecutive machine-53 openings
whose k-1 gaps are ALL legal letters, hence a word-legal window of J = k-1
gaps.  So any word whose (span, depth) lands in a refuted band is ZERO, with no
solver call.

    band                 jmax  source
    (152, 158]            2    r27 f2_53_top      (max 152)
    (159, 200]            2    r26 f2_53_*        (max 159 = F_2(53))
    (161, 184]            3    r27 f59_wlJ3       (max 161)
    (183, 194]            7    r27 sweep band 2   (EMPTY)
    (193, 204]            7    r27 sweep band 1   (EMPTY)
    (203, 260]            7    r27 f59_A          (EMPTY)

SCOPE, stated once and it is the whole soundness argument: a span cap
conditions claims about spans ABOVE it, never claims about spans INSIDE the
scanned interval.  So "F_2(53) <= 159" is cap-conditional while "no 2-window
has span in (159, 200]" is not, and only the second kind of statement is used.

Usage: python research/akill_bands_r27.py [--write]
"""
import os
import re
import sys
from itertools import product

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from akill_verify_r27 import (dead_gear, s_of, letters, window_valid,  # noqa
                              parse_ranges, check_tiling, NOPEN23)

DATA = os.path.join(HERE, "data")
LOG = os.path.join(DATA, "r27", "akill_53_59.log")
M, QP = 53, 59

#            logs                                        seed  cap  jmax  max
BANDS = [
    (["r27/f2_53_top_w0.log", "r27/f2_53_top_w1.log"],     145, 158, 2, 152),
    (["r26/f2_53_head.log", "r26/f2_53_mid.log",
      "r26/f2_53_w1.log"],                                 145, 200, 2, 159),
    (["r27/f59_wlJ3_152_184_w%d.log" % i for i in range(3)],
                                                           152, 184, 3, 161),
    (["r27/f59_b183_194_w%d.log" % i for i in range(7)],    183, 194, 7, 183),
    (["r27/f59_b193_204_w%d.log" % i for i in range(7)],    193, 204, 7, 193),
    (["r27/f59_A_w0.log", "r27/f59_A_w1.log"],             203, 260, 7, 203),
]


def verified_bands():
    out = []
    for logs, seed, cap, jmax, want in BANDS:
        paths = [os.path.join(DATA, p) for p in logs]
        rngs, maxes = parse_ranges(paths)
        check_tiling(rngs, NOPEN23, str(logs[0]))
        got = max(maxes)
        assert got == want, (logs[0], "reported max", got, "expected", want)
        print("  %d workers TILE [0,%d)  seed %3d cap %3d J<=%d  max %3d  "
              "=> NO window of depth <= %d has span in (%3d, %3d]"
              % (len(paths), NOPEN23, seed, cap, jmax, got, jmax, got, cap))
        out.append((got, cap, jmax))
    return out


def refuted(span, jdepth, bands):
    for lo, hi, jmax in bands:
        if jdepth <= jmax and lo < span <= hi:
            return (lo, hi, jmax)
    return None


def main():
    print(__doc__.splitlines()[0])
    print("\n=== THE BANDS, EACH RE-READ AND ITS TILING RE-ASSERTED ===")
    bands = verified_bands()

    vals = [v for v in range(1, 146)
            if v % QP in {0, s_of(QP), (-s_of(QP)) % QP}]
    lines = []
    n_by_k = {}
    for nlet in (2, 3, 4, 5):
        for w in product(vals, repeat=nlet):
            L = letters(w, QP)
            if L is None or not window_valid(L):
                continue
            if dead_gear(w):
                continue                     # the screen owns these
            b = refuted(sum(w), nlet, bands)
            if b is None:
                continue
            n_by_k[nlet + 1] = n_by_k.get(nlet + 1, 0) + 1
            lines.append("  RESULT m53 word (%s) span %d: ZERO (0 SAT calls - "
                         "scan band (%d,%d] at depth J = %d <= %d)"
                         % (", ".join(str(x) for x in w), sum(w),
                            b[0], b[1], nlet, b[2]))
    print("\n=== WORDS REFUTED BY BAND ALONE, by chain length k ===")
    for k in sorted(n_by_k):
        print("   k = %d: %4d words" % (k, n_by_k[k]))
    print("   TOTAL %d words, ZERO solver calls" % len(lines))
    if "--write" in sys.argv:
        with open(LOG, "a", encoding="utf-8") as fh:
            for ln in lines:
                fh.write(ln + "\n")
        print("appended to %s" % LOG)


if __name__ == "__main__":
    main()
