"""fl_family.py -- prediction T6 of fusion_lemma.md: on the counterfactual tooth family, is the
fusion lemma's non-root remainder R <= F(M) + q' a consequence of the budget?

Family at step y -> q': every gear g in {5..y} gets a tooth v_g in 1..(g-1)/2 (strikes
k = +-v_g mod g), and the new gear q' a tooth v' in 1..(q'-1)/2 (letters d' = 2v' mod q').
For each member: F(M), L, the maximal words, R = max_m [P(m) + |m| + S(m)], and F(M + q') by
a direct sieve of the bigger machine.  The real teeth (v_g = 6^{-1} mod g) are one member and
must reproduce the real numbers (gate).

Counts reported: budget holds / fails; remainder R <= budget holds / fails; the four cells.

Usage: uv run python research/anchor235/r73/fl_family.py 13 [workers]
       uv run python research/anchor235/r73/fl_family.py 17 [workers]
"""
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fl_core import (PRIMES, analyse, longest_legal_run, real_teeth, sieve,  # noqa: E402
                     u_of, windows_from_gaps)

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)


def one(args):
    gears, teeth, q, v = args
    tm = dict(zip(gears, teeth))
    P, gaps = sieve(tm)
    F = int(gaps.max())
    d = (2 * v) % q
    L = longest_legal_run(gaps, q, d)
    wL, _ = windows_from_gaps(gaps, L) if L > 0 else (None, None)
    wL2, _ = windows_from_gaps(gaps, L + 2)
    an = analyse(wL2, wL, q, d, L)
    tm2 = dict(tm)
    tm2[q] = v
    _, gaps2 = sieve(tm2)
    F2 = int(gaps2.max())
    budget = F + q
    return {"teeth": list(teeth), "v": v, "F": F, "L": L, "R": an["R"],
            "Qstar_Jmax": an["Qstar_Jmax"], "F_next": F2, "budget": budget,
            "budget_ok": F2 <= budget, "remainder_ok": an["R"] <= budget,
            "record_ge_Qstar": F2 >= an["Qstar_Jmax"], "fuse_fail": an["fuse_fail"],
            "n_words": len(an["words"])}


def main():
    y = int(sys.argv[1])
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    gears = [p for p in PRIMES if p <= y]
    q = PRIMES[PRIMES.index(y) + 1]
    ranges = [range(1, (g - 1) // 2 + 1) for g in gears]
    jobs = [(gears, teeth, q, v) for teeth in itertools.product(*ranges)
            for v in range(1, (q - 1) // 2 + 1)]
    print(f"family {y} -> {q}: {len(jobs):,} machines, {workers} workers", flush=True)
    t0 = time.time()
    with Pool(workers) as pool:
        rows = pool.map(one, jobs, chunksize=32)
    secs = time.time() - t0
    # the real member
    rt = real_teeth(gears)
    real_teeth_half = [min(rt[g], g - rt[g]) for g in gears]
    vreal = min(u_of(q), q - u_of(q))
    real = [r for r in rows if r["teeth"] == real_teeth_half and r["v"] == vreal][0]
    cells = {"budget_ok&rem_ok": 0, "budget_ok&rem_fail": 0, "budget_fail&rem_ok": 0,
             "budget_fail&rem_fail": 0}
    for r in rows:
        key = ("budget_ok" if r["budget_ok"] else "budget_fail") + "&" + \
              ("rem_ok" if r["remainder_ok"] else "rem_fail")
        cells[key] += 1
    n_fuse_fail = sum(r["fuse_fail"] for r in rows)
    n_rec = sum(1 for r in rows if not r["record_ge_Qstar"])
    Lmax = max(r["L"] for r in rows)
    worst = sorted([r for r in rows if r["budget_ok"] and not r["remainder_ok"]],
                   key=lambda r: -(r["R"] - r["budget"]))[:10]
    # the real member's percentile of the remainder's margin (budget - R) among budget-holding
    # members, and of the relaxation's excess R - F(M+q')
    ok = [r for r in rows if r["budget_ok"]]
    rm = real["budget"] - real["R"]
    pct_margin = 100.0 * sum(1 for r in ok if r["budget"] - r["R"] < rm) / len(ok)
    rx = real["R"] - real["F_next"]
    pct_excess = 100.0 * sum(1 for r in ok if r["R"] - r["F_next"] > rx) / len(ok)
    rep = {"step": [y, q], "machines": len(rows), "secs": round(secs, 1), "real": real,
           "real_margin_percentile_below": round(pct_margin, 1),
           "real_excess_percentile_above": round(pct_excess, 1),
           "rows": rows,
           "cells": cells, "fuse_fail_total": n_fuse_fail, "record_below_Qstar": n_rec,
           "L_max": Lmax, "worst_budget_ok_rem_fail": worst,
           "L_hist": {int(k): int(v) for k, v in
                      zip(*np.unique([r["L"] for r in rows], return_counts=True))}}
    print(json.dumps({k: v for k, v in rep.items() if k not in ("worst_budget_ok_rem_fail", "rows")},
                     indent=1), flush=True)
    print("worst budget-ok / remainder-fail members:", flush=True)
    for r in worst:
        print(f"  teeth {r['teeth']} v'={r['v']}: F={r['F']} L={r['L']} R={r['R']} "
              f"F_next={r['F_next']} budget={r['budget']}  excess {r['R']-r['budget']}",
              flush=True)
    with open(os.path.join(OUT, f"family_{y}_{q}.json"), "w") as f:
        json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
