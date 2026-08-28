"""Gap-histogram probe sweep (mechanic r16).

Uses the r15 simplification: the padding supply of step M -> q' is exactly
  supply(M, q') = hist_M[q']
the number of gaps of machine M equal to exactly q'. So one gap histogram
per machine answers the padding-onset question for EVERY probe at once,
with no run classification. This script computes that histogram (full
period when affordable, otherwise a prefix) and reports:

  * F(M) on the scanned range;
  * hist[q'] at the next several primes above y - the padding supply;
  * every value below F that is MISSING from the spectrum (the holes that
    make the onset rule necessary-but-not-sufficient, r15).

INTERPRETATION OF A PREFIX RUN (stated because it decides claim strength):
a prefix scan gives a LOWER bound on each hist entry. hist[q'] > 0 found
in a prefix is DEFINITIVE (padding can exist at that step). hist[q'] = 0
in a prefix is INCONCLUSIVE for the full period - report as "not found in
X% of period", never as "missing".

Usage: uv run python research/hist_probe.py y [--limit SLOTS] [--top N]
Output: printed report + append research/data/gap_histograms.csv
"""
import os
import sys
import time
import numpy as np
from math import prod

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fragile_census import primes_upto
from prefix_census import is_prime


def histogram(y, limit=None, seg=64_000_000, cap=400):
    gears = [p for p in primes_upto(y) if p >= 5]
    P = prod(gears)
    K = P if limit is None else min(P, limit)
    uv = [pow(6, -1, g) for g in gears]
    hist = np.zeros(cap, dtype=np.int64)
    tail = np.array([], dtype=np.int64)
    t0 = time.time()
    for a in range(0, K, seg):
        b = min(K, a + seg)
        ex = np.zeros(b - a, bool)
        for g, u in zip(gears, uv):
            ex[(u - a) % g::g] = True
            ex[(-u - a) % g::g] = True
        ops = np.concatenate([tail,
                              np.flatnonzero(~ex).astype(np.int64) + a])
        if len(ops) > 1:
            d = np.diff(ops)
            d = d[ops[1:] >= a]
            d = d[d < cap]
            hist += np.bincount(d, minlength=cap)[:cap]
        tail = ops[-2:]
    # ROUND-25: cyclic close (see gap_pair_census.py / cyclic_close_r25.py).
    # A full period is a CIRCLE - N openings, N gaps - and the linear np.diff
    # dropped the wrap gap.  It equals the FIRST gap (slot 0 is always an
    # opening and the opening set is mirror-symmetric), which is small (3-7 at
    # every machine reached), so no PADDING SUPPLY number ever computed from
    # this tool moves: those probe q' >= 29.  Fixed anyway.
    if K == P:
        first = None
        k = 0
        while first is None:
            if all(k % g not in (u % g, (-u) % g) for g, u in zip(gears, uv)):
                if k:
                    first = k
            k += 1
            assert k < 10000, "first-gap window too small"
        if first < cap:
            hist[first] += 1
    return hist, P, K, time.time() - t0, gears


def main():
    args = sys.argv[1:]
    limit = None
    top = 5
    if "--limit" in args:
        i = args.index("--limit")
        limit = int(args[i + 1])
        del args[i:i + 2]
    if "--top" in args:
        i = args.index("--top")
        top = int(args[i + 1])
        del args[i:i + 2]
    y = int(args[0])
    hist, P, K, secs, gears = histogram(y, limit=limit)
    cov = K / P
    F = int(np.max(np.flatnonzero(hist)))
    full = limit is None
    print(f"machine y={y}: period {P:.4e}, scanned {K:.3e} "
          f"({100*cov:.2f}%), {secs:.0f}s")
    print(f"  F on scanned range = {F}" + ("" if full else " (LOWER bound)"))
    probes = []
    p = y + 2
    while len(probes) < top:
        if is_prime(p):
            probes.append(p)
        p += 2
    print(f"  PADDING SUPPLY hist[q'] at the next {top} primes:")
    for q in probes:
        v = int(hist[q]) if q < len(hist) else 0
        if v > 0:
            verdict = "CAN pad (definitive)"
        elif q > F:
            verdict = ("CANNOT pad (q' > F: theorem)" if full
                       else "not found; q' > F_scanned (likely cannot)")
        else:
            verdict = ("CANNOT pad (value absent from full spectrum)"
                       if full else
                       f"NOT FOUND in {100*cov:.2f}% of period "
                       f"(INCONCLUSIVE)")
        print(f"    q' = {q:<4} hist = {v:<12} {verdict}")
    holes = [v for v in range(1, F) if hist[v] == 0]
    label = "MISSING values below F" if full else \
            "values below F not seen in the scanned prefix"
    print(f"  {label}: {holes}")
    ddir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(ddir, exist_ok=True)
    path = os.path.join(ddir, "gap_histograms.csv")
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a") as f:
        if new:
            f.write("y,period,scanned,coverage,F,full_period,"
                    "probe_prime,hist_at_probe\n")
        for q in probes:
            v = int(hist[q]) if q < len(hist) else 0
            f.write(f"{y},{P},{K},{cov:.6f},{F},{int(full)},{q},{v}\n")
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
