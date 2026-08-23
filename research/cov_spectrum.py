"""Round 20 (mechanic): THE COVERABILITY SPECTRUM COV(M).

My r17 hole census found which gap values are ABSENT from a machine's spectrum
but could only do it by scanning a whole period, which stops at machine 31.
The construct I named instead: a slot gap of v means 3v - 1 consecutive
ADJACENT-frame positions all blocked, with both ends spared, so

    v is a realizable gap at machine M  <=>  coverable(3v - 1, primes <= M)

which is CRT arithmetic over the gear set - no period scan - and therefore
reaches machines whose periods are out of range.  It also yields UPPER bounds
on F and on F_j, which every prefix row in my tables currently lacks.

This reuses the project's existing `coverable` (research/max_gap_search.py,
the pruned F(2,y) search), so where the two overlap they must agree: that is
the cross-check, run first and reported before any new claim.

Usage: uv run python research/cov_spectrum.py y [y ...] [--vmax N]
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from max_gap_search import coverable
from interval_avoidance import odd_primes_upto

# my exact full-period hole lists (r17, hole_structure.py) for the cross-check
MEASURED_HOLES = {11: [], 13: [9], 17: [17], 19: [19, 24], 23: [24],
                  29: [41, 42]}
MEASURED_F = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88}


def cov_spectrum(y, vmax):
    qs = odd_primes_upto(y)
    out = {}
    for v in range(1, vmax + 1):
        out[v] = coverable(3 * v - 1, qs)
    return out


def main():
    args = sys.argv[1:]
    vmax = None
    if "--vmax" in args:
        i = args.index("--vmax")
        vmax = int(args[i + 1])
        del args[i:i + 2]
    ys = [int(a) for a in args] or [11, 13, 17, 19]
    for y in ys:
        vm = vmax or (MEASURED_F.get(y, 60) + 2)
        t0 = time.time()
        cov = cov_spectrum(y, vm)
        holes = [v for v in range(1, vm + 1) if not cov[v]]
        Fup = min(holes) - 1 if holes else vm
        print(f"\n=== machine {y} (gears {odd_primes_upto(y)}), "
              f"v <= {vm}, {time.time()-t0:.1f}s")
        print(f"  non-coverable v (predicted holes + the top): {holes}")
        print(f"  smallest non-coverable v = {min(holes) if holes else '-'}"
              f"  => F(M) <= {Fup}   (UPPER bound, no period scan)")
        if y in MEASURED_F:
            print(f"  measured F (full period)        = {MEASURED_F[y]}"
                  f"   {'AGREES' if Fup == MEASURED_F[y] else 'MISMATCH'}")
        if y in MEASURED_HOLES:
            below = [v for v in holes if v < MEASURED_F[y]]
            m = MEASURED_HOLES[y]
            ok = below == m
            print(f"  predicted holes below F         = {below}")
            print(f"  measured holes below F          = {m}"
                  f"   {'AGREES' if ok else 'MISMATCH'}")


if __name__ == "__main__":
    main()
