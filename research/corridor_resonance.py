"""Round 20 (mechanic): THE CORRIDOR RESONANCE - why the qualifying-gap
autocorrelation oscillates with lag-period ~ 35/mean_gap.

bool_lag_census found the big-gap indicator's autocorrelation is a barely
damped wave: trough at lag 3-4, peak at lag 6-8, second trough/peak a full
cycle later.  Across machines the period tracks 35/mean_gap (8.2 at m19,
7.5 at m23, 7.0 at m29 - matching measured peaks 8, 7-8, 6-7), i.e. a
FIXED SLOT-DISTANCE resonance at ~35 slots: the corridor mod 35.

This tool measures the mechanism directly, full period, exact counts:
  (1) left-endpoint residue mod 35 (and mod 5, 7) of gaps >= a vs all gaps
      - if big gaps are PINNED to few corridor classes, the wave follows;
  (2) the slot-separation autocorrelation of the big-gap-endpoint
      indicator at separations 1..105 - the 35-periodicity seen directly,
      no lag-frame conversion.

Usage: uv run python research/corridor_resonance.py y a [--limit N]
"""
import os
import sys
import time
import numpy as np
from math import prod

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from flank_envelope import primes_upto

DMAX = 105


def run(y, a, limit=None, seg=64_000_000):
    gears = [p for p in primes_upto(y) if p >= 5]
    P = prod(gears)
    K = P if limit is None else min(P, limit)
    uvals = [pow(6, -1, g) for g in gears]
    all35 = np.zeros(35, np.int64)
    big35 = np.zeros(35, np.int64)
    sepcor = np.zeros(DMAX + 1, np.int64)   # pairs of big endpoints at sep d
    nbig = 0
    nall = 0
    tailk = np.array([], dtype=np.int64)
    t0 = time.time()
    for lo in range(0, K, seg):
        hi = min(K, lo + seg)
        ex = np.zeros(hi - lo, bool)
        for g, u in zip(gears, uvals):
            ex[(u - lo) % g::g] = True
            ex[(-u - lo) % g::g] = True
        op = np.flatnonzero(~ex).astype(np.int64) + lo
        ops = np.concatenate([tailk, op])
        if len(ops) > 2:
            d = np.diff(ops)
            new = ops[1:] >= lo               # gap's right end is new
            starts = ops[:-1][new]
            gv = d[new]
            allm = starts % 35
            all35 += np.bincount(allm, minlength=35)
            nall += len(starts)
            bigs = starts[gv >= a]
            nbig += len(bigs)
            big35 += np.bincount(bigs % 35, minlength=35)
        tailk = ops[-2:].copy() if len(ops) >= 2 else ops.copy()
        # slot-separation autocorrelation needs the big-endpoint indicator
        # per slot; do it within the segment (boundary loss <= DMAX slots
        # per segment, negligible and identical for obs/expectation)
        ind = np.zeros(hi - lo, bool)
        segstarts = ops[:-1][(np.diff(ops) >= a) & (ops[:-1] >= lo)] - lo
        segstarts = segstarts[segstarts < hi - lo]
        ind[segstarts] = True
        w = np.flatnonzero(ind)
        for dd in range(1, DMAX + 1):
            # count pairs at separation exactly dd via sorted membership
            sepcor[dd] += np.count_nonzero(ind[:len(ind) - dd] &
                                           ind[dd:])
    secs = time.time() - t0
    print(f"=== machine {y}, floor a = {a}: {nall:,} gaps, {nbig:,} big "
          f"({100*nbig/nall:.2f}%), scanned {K:.4g}/{P:.4g} "
          f"({100*K/P:.1f}%), {secs:.0f}s")
    print("  (1) LEFT-ENDPOINT RESIDUES MOD 35 (share_big / share_all; "
          "1.00 = no pinning):")
    live = [(r, big35[r] / nbig / (all35[r] / nall))
            for r in range(35) if all35[r] > 0]
    live.sort(key=lambda t: -t[1])
    print("      exposed classes:", [r for r in range(35) if all35[r] > 0])
    print("      enrichment by class:",
          " ".join(f"{r}:{e:.2f}" for r, e in live))
    top = [r for r, e in live if e > 1.2]
    print(f"      classes with >1.2x enrichment: {top}")
    print("  (2) SLOT-SEPARATION autocorrelation of big-gap endpoints "
          "(obs/expected, expected = nbig^2/slots * corridor-admissible "
          "share):")
    dens = nbig / K
    print("      sep:  ratio vs flat density  (35 | sep marked *)")
    for dd in range(1, DMAX + 1):
        r = sepcor[dd] / (dens * dens * K)
        mark = " *" if dd % 35 == 0 else ("  " if dd % 5 and dd % 7
                                          else " .")
        if dd <= 50 or dd % 35 in (0, 1, 34) or r > 2:
            print(f"      {dd:3d}{mark}  {r:8.3f}")
    return big35, all35, sepcor, nbig, nall, K


if __name__ == "__main__":
    args = sys.argv[1:]
    limit = None
    if "--limit" in args:
        i = args.index("--limit")
        limit = int(float(args[i + 1]))
        del args[i:i + 2]
    run(int(args[0]), int(args[1]), limit)
