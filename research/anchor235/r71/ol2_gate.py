"""ol2_gate.py -- an INDEPENDENT exact check on the covering instrument at a new engine.

A bounded column range of the machine {5..y} is sieved directly, gear by gear, in the anchored
column coordinate (gear g strikes column k iff k = +-u_g mod g, u_g = 6^{-1} mod g).  Every window
that actually OCCURS in that range is a window the instrument must call realised.  One
disagreement is a false NO and condemns the instrument; the count is the gate.

The two sides share nothing: the scan is a sieve of a concrete stretch of columns of the machine,
the instrument is a covering problem over the gears' free phases with no column in it at all.

Nothing here needs a dictionary, and every question the gate asks has the answer YES, which is the
cheap direction for the solver (a witness is found; no exhaustive refutation is ever required).

Usage: uv run python research/anchor235/r71/ol2_gate.py <y> [columns] [x0] [procs]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ol2_core import OUT, decide_many, gears_upto, u_of  # noqa: E402

LIMITS = {1: None, 2: None, 3: 20000, 4: 6000}


def sieve(y, x0, X):
    struck = np.zeros(X, dtype=bool)
    for g in gears_upto(y):
        u = u_of(g)
        for t in (u % g, (-u) % g):
            struck[(t - x0) % g::g] = True
    return np.flatnonzero(~struck).astype(np.int64) + x0


def main():
    y = int(sys.argv[1])
    X = int(float(sys.argv[2])) if len(sys.argv) > 2 else 200_000_000
    x0 = int(float(sys.argv[3])) if len(sys.argv) > 3 else 0
    procs = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    gears = gears_upto(y)
    rng = np.random.default_rng(20260910)
    t0 = time.time()
    op = sieve(y, x0, X)
    gaps = np.diff(op)
    print(f"m{y}: sieved columns [{x0:,}, {x0+X:,}); {op.size:,} openings, {gaps.size:,} gaps, "
          f"density {op.size/X:.6f}, widest gap in the scan {gaps.max()}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    rep = {"y": y, "x0": x0, "columns": X, "openings": int(op.size), "gaps": int(gaps.size),
           "widest_gap_in_scan": int(gaps.max()), "gates": {}}
    total_checked = total_bad = 0
    for k in (1, 2, 3, 4):
        w = np.stack([gaps[i:gaps.size - (k - 1 - i)] for i in range(k)], axis=1) if k > 1 \
            else gaps[:, None]
        u = np.unique(w, axis=0)
        rows = [tuple(int(x) for x in r) for r in u]
        lim = LIMITS[k]
        take = rows if (lim is None or len(rows) <= lim) else \
            [rows[i] for i in rng.choice(len(rows), lim, replace=False)]
        print(f"gate D_{k}: {len(rows):,} distinct {k}-windows occur in the scan; "
              f"{len(take):,} put to the instrument", flush=True)
        v, st = decide_many(take, gears, procs=procs, log=False)
        bad = [list(t) for t in take if v[t] is not True]
        total_checked += len(take)
        total_bad += len(bad)
        print(f"gate D_{k}: {len(bad)} disagreement(s)  {bad[:10]}   "
              f"({st['wall']:.0f}s, {st['calls']:,} solver calls)", flush=True)
        rep["gates"][f"D_{k}"] = {"distinct_in_scan": len(rows), "tested": len(take),
                                  "disagreements": bad, "secs": round(st["wall"], 1),
                                  "widest_tested": max(map(sum, take))}
    rep["total_checked"] = total_checked
    rep["total_disagreements"] = total_bad
    rep["secs"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, f"gate_m{y}.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(f"\nGATE m{y}: {total_bad} disagreements in {total_checked:,} windows "
          f"[{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
