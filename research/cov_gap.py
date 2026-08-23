"""Round 20 (mechanic): COV(M) with the ENDPOINT constraint - exact gap
realizability without a period scan.

The project's existing `coverable(L, qs)` asks only "can L consecutive
positions all be blocked".  That settles F (a maximal gap needs a maximal
blocked run) and my first pass reproduced F exactly at machines 11, 13, 17 -
but it does NOT settle which gap values OCCUR, because a gap of exactly v also
needs its two ENDPOINTS SPARED.  Adding that constraint is the difference
between "F <= 11" and "9 is a hole", and it is what my r17 hole census
measured by brute period scan.

Frame: slot gap v  <->  adjacent-frame positions 0 and 3v spared, positions
1..3v-1 all blocked; gear q blocks {o, o+1} mod q for one offset o per gear
(gear 3 included - it is what makes the spared positions the slots).

Usage: uv run python research/cov_gap.py y [y ...] [--vmax N]
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from interval_avoidance import odd_primes_upto

MEASURED_HOLES = {11: [], 13: [9], 17: [17], 19: [19, 24], 23: [24],
                  29: [41, 42]}
MEASURED_F = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88}


def realizable(v, qs):
    """Is a slot gap of exactly v realizable at gears qs?  Exact."""
    E = 3 * v                      # the two spared endpoints are 0 and E
    L = E - 1                      # interior positions 1..E-1 must be blocked
    if L <= 0:
        return True
    full = ((1 << L) - 1)          # bit i-1 <-> position i
    masks = {}
    for q in qs:
        ms = []
        bad = {0 % q, (q - 1) % q, E % q, (E - 1) % q}
        for o in range(q):
            if o in bad:
                continue           # this offset would block an endpoint
            m = 0
            for i in range(1, E):
                if i % q == o % q or i % q == (o + 1) % q:
                    m |= 1 << (i - 1)
            ms.append(m)
        if not ms:
            return False           # gear q cannot spare both endpoints
        masks[q] = ms
    best = {q: max(bin(m).count("1") for m in masks[q]) for q in qs}

    def search(covered, remaining):
        if covered == full:
            return True
        todo = L - bin(covered).count("1")
        if sum(best[q] for q in remaining) < todo:
            return False
        pos = (~covered & full)
        pos = (pos & -pos).bit_length()          # 1-based leftmost uncovered
        for idx, q in enumerate(remaining):
            rest = remaining[:idx] + remaining[idx + 1:]
            for m in masks[q]:
                if (m >> (pos - 1)) & 1:
                    if search(covered | m, rest):
                        return True
        return False

    return search(0, tuple(sorted(qs, reverse=True)))


def main():
    args = sys.argv[1:]
    vmax = None
    if "--vmax" in args:
        i = args.index("--vmax")
        vmax = int(args[i + 1])
        del args[i:i + 2]
    for y in [int(a) for a in args] or [11, 13, 17, 19]:
        qs = odd_primes_upto(y)
        vm = vmax or (MEASURED_F.get(y, 60) + 1)
        t0 = time.time()
        holes = [v for v in range(1, vm + 1) if not realizable(v, qs)]
        below = [v for v in holes if v < MEASURED_F.get(y, 10 ** 9)]
        Fup = min(holes) - 1 if holes else None
        print(f"\n=== machine {y} (gears {qs}), v <= {vm}, "
              f"{time.time()-t0:.1f}s")
        print(f"  non-realizable v: {holes}")
        if y in MEASURED_F:
            realF = max(v for v in range(1, vm + 1) if v not in holes)
            print(f"  largest realizable v = {realF}   measured F = "
                  f"{MEASURED_F[y]}   "
                  f"{'AGREES' if realF == MEASURED_F[y] else 'MISMATCH'}")
        if y in MEASURED_HOLES:
            print(f"  predicted holes below F = {below}")
            print(f"  measured  holes below F = {MEASURED_HOLES[y]}   "
                  f"{'AGREES' if below == MEASURED_HOLES[y] else 'MISMATCH'}")


if __name__ == "__main__":
    main()
