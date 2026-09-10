"""u45_prefix.py -- W32's first-hit law on the ENGINE at the range sizes it was validated on.

top_machine_2.md 3.7 measured W32 on three manifold wheels at N = 10^4 .. 10^10 and found the
prediction right to within one unit at 19 of 21 checkpoints.  The engine's own window is
N = 18 .. 570 columns, three to eight orders of magnitude below that.  This script measures the
same table for the engine, on the same phase-zero prefix [0, N) that the wheel table used, so that
the window's misses can be placed: is the law failing on the engine, or is the window simply far
below the regime in which the law was measured?

Usage: uv run python research/anchor235/r72/u45_prefix.py <y> [chunk]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
sys.path.insert(0, HERE)

from u45_census import c_of, gears_upto  # noqa: E402


def main():
    y = int(sys.argv[1])
    chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 50_000_000
    gears = gears_upto(y)
    P = 1
    for g in gears:
        P *= g
    checks = [10 ** k for k in range(2, 12) if 10 ** k <= P] + [P]
    checks = sorted(set(checks))
    t0 = time.time()
    # prefix records by a streaming sieve of [0, P)
    rec = {}
    prev = None
    best = 0
    ci = 0
    lo = 0
    while lo < P:
        hi = min(lo + chunk, P)
        n = hi - lo
        blocked = np.zeros(n, dtype=bool)
        for g in gears:
            u = pow(6, -1, g)
            blocked[(u - lo) % g::g] = True
            blocked[((-u) - lo) % g::g] = True
        op = np.flatnonzero(~blocked).astype(np.int64) + lo
        del blocked
        if op.size:
            if prev is not None:
                op = np.concatenate([[prev], op])
            d = np.diff(op)
            ends = op[1:]
            # running max of gaps whose RIGHT end is <= the checkpoint
            while ci < len(checks) and checks[ci] <= hi:
                m = ends <= checks[ci]
                v = int(d[m].max()) if m.any() else 0
                rec[checks[ci]] = max(best, v)
                ci += 1
            best = max(best, int(d.max()))
            prev = int(op[-1])
        lo = hi
    while ci < len(checks):
        rec[checks[ci]] = best
        ci += 1
    # the census and the first-hit ladder
    c = {}
    d = 1
    while True:
        v, _ = c_of(y, d)
        c[d] = v
        if v == 0:
            break
        d += 1
    pred = {}
    for N in checks:
        last = 0
        for dd in sorted(c):
            if c[dd] and c[dd] * N >= P:
                last = dd
        pred[N] = last - 1
    rows = [{"N": N, "record": rec[N], "pred": pred[N], "diff": rec[N] - pred[N]}
            for N in checks]
    print(f"m{y}: P = {P:,}, F = {max(k for k in c if c[k])}, {time.time()-t0:.1f}s", flush=True)
    for r in rows:
        print(f"  N = {r['N']:>14,}  record {r['record']:4d}  first hit {r['pred']:4d}  "
              f"diff {r['diff']:+3d}", flush=True)
    with open(os.path.join(OUT, f"prefix_m{y}.json"), "w") as f:
        json.dump({"y": y, "P": P, "rows": rows,
                   "c": {str(k): str(v) for k, v in c.items()}}, f)


if __name__ == "__main__":
    main()
