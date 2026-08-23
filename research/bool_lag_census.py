"""Round 20 (mechanic): the 8-POINT BOOLEAN PATTERN CENSUS of the
qualifying-gap indicator - the exact object that decides what STATE a
transfer-matrix formulation of p_j needs.

transfer_spectrum.py showed the measured lag-2/3 deficit is NOT one-step
memory: the value-level Markov chain (state = last gap value) predicts
obs/indep = 1.00 at lags 2-5 while the census says 0.51-0.68.  So the
process has multi-step memory.  This tool measures it exactly:

  b_i = [d_i >= a]  (indicator of a qualifying-size gap), and the full
  joint distribution of (b_i, ..., b_{i+7}) - all 256 patterns - per
  floor a, over the whole period.

Every k-step-Markov hypothesis is then an EXACT conditional-independence
factorisation of these 256 counts (all factors are ratios of marginal
pattern counts from the same census - no fits), so the census decides the
Markov order and exhibits the worst-violating patterns as events.

Usage: uv run python research/bool_lag_census.py y [--limit N] [--seg N]
Writes research/data/bool_lag_{y}.csv (256 rows per floor).
"""
import os
import sys
import time
import numpy as np
from math import prod

HERE = os.path.dirname(os.path.abspath(__file__))
DDIR = os.path.join(HERE, "data")
sys.path.insert(0, HERE)
from flank_envelope import primes_upto

W = 8                       # window length in gaps (override with --W)
FLOORS = [4, 6, 8, 10, 12, 14]


def run(y, limit=None, seg=64_000_000, start=0):
    NPAT = 1 << W
    gears = [p for p in primes_upto(y) if p >= 5]
    P = prod(gears)
    K = P if limit is None else min(P, start + limit)
    uvals = [pow(6, -1, g) for g in gears]
    counts = {a: np.zeros(NPAT, np.int64) for a in FLOORS}
    tail = np.array([], dtype=np.int64)
    t0 = time.time()
    for lo in range(start, K, seg):
        hi = min(K, lo + seg)
        ex = np.zeros(hi - lo, bool)
        for g, u in zip(gears, uvals):
            ex[(u - lo) % g::g] = True
            ex[(-u - lo) % g::g] = True
        op = np.flatnonzero(~ex).astype(np.int64) + lo
        ops = np.concatenate([tail, op])
        if len(ops) > W + 2:
            d = np.diff(ops)
            n = len(d)
            # new windows: those whose LAST gap's right end is new
            for a in FLOORS:
                b = (d >= a).astype(np.int64)
                w = np.zeros(n - W + 1, np.int64)
                for t in range(W):
                    w += b[t:n - W + 1 + t] << t
                sel = ops[W:] >= lo          # right end of window is new
                counts[a] += np.bincount(w[sel], minlength=NPAT)
        tail = ops[-(W + 2):].copy() if len(ops) >= W + 2 else ops.copy()
    secs = time.time() - t0
    path = os.path.join(DDIR, f"bool_lag_{y}.csv" if W == 8 else f"bool_lag{W}_{y}.csv")
    with open(path, "w") as f:
        f.write("y,coverage,floor,pattern,count\n")
        for a in FLOORS:
            for pat in range(NPAT):
                if counts[a][pat]:
                    f.write(f"{y},{K/P:.6f},{a},{pat},{counts[a][pat]}\n")
    print(f"machine {y}: scanned {K:.4g} of {P:.4g} ({100*K/P:.2f}%), "
          f"{secs:.0f}s -> {path}", flush=True)
    return counts


if __name__ == "__main__":
    args = sys.argv[1:]
    limit = None
    if "--limit" in args:
        i = args.index("--limit")
        limit = int(float(args[i + 1]))
        del args[i:i + 2]
    seg = 64_000_000
    if "--seg" in args:
        i = args.index("--seg")
        seg = int(float(args[i + 1]))
        del args[i:i + 2]
    if "--start" in args:
        i = args.index("--start")
        start = int(float(args[i + 1]))
        del args[i:i + 2]
    else:
        start = 0
    if "--W" in args:
        i = args.index("--W")
        W = int(args[i + 1])
        del args[i:i + 2]
    run(int(args[0]), limit, seg, start)
