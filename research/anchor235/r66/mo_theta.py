"""mo_theta.py -- the last rung of item 1: F(41) = 91 from the closure alone, with the
span-threshold prune of ladder_closure.md 4.4 (a window spanning less than theta can never be the
ancestor of a gap of size >= theta; theta = F(37) + 1 = 89 is free because F(M + q') >= F_2(M)).

This is research/anchor235/r61/lc_theta.py run into this branch's results directory, with the
closure step imported unchanged.

Usage: uv run python research/anchor235/r66/mo_theta.py [K0] [theta] [maxrung]
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mo_core import CORPUS_F, OUT, PRIMES, base_gaps, closure_step, dict_from_gaps, group_windows


def main():
    K0 = int(sys.argv[1]) if len(sys.argv) > 1 else 21
    theta = int(sys.argv[2]) if len(sys.argv) > 2 else 89
    maxrung = int(sys.argv[3]) if len(sys.argv) > 3 else 41
    base = 23
    P, g = base_gaps(base)
    t0 = time.time()
    win, mult = dict_from_gaps(g, K0)
    keep = np.flatnonzero(win.astype(np.int32).sum(axis=1) >= theta)
    win, mult = group_windows(np.ascontiguousarray(win[keep]), mult[keep])
    print(f"base m{base}: |D_{K0}| pruned to span >= {theta}: {win.shape[0]:,} rows, mass "
          f"{int(mult.sum()):,} of {g.size:,} [{time.time()-t0:.1f}s]", flush=True)
    log = []
    for k in range(PRIMES.index(base) + 1, len(PRIMES)):
        q = PRIMES[k]
        if q > maxrung:
            break
        K = win.shape[1]
        t0 = time.time()
        _, _, s1 = closure_step(win, mult, q, m=K, mode="fixed", tag_order=True, collect=False,
                                theta=theta)
        m = max(1, s1["mmin"])
        w2, m2, st = closure_step(win, mult, q, m=m, mode="fixed", theta=theta)
        specJ = s1["specJ"]
        spec = specJ.sum(axis=0)
        F = int(np.flatnonzero(spec).max()) if spec.any() else 0
        Qs = {int(j): int(np.flatnonzero(specJ[j]).max()) for j in range(1, specJ.shape[0])
              if specJ[j].any()}
        row = {"rung": q, "K_in": K, "depth_out": m, "F_above_theta": F, "Qstar": Qs,
               "dict_out": int(w2.shape[0]), "mass_out": int(m2.sum()), "over0": s1["over0"],
               "theta": theta, "corpus_F": CORPUS_F.get(q), "gate": F == CORPUS_F.get(q),
               "secs": round(time.time() - t0, 1)}
        log.append(row)
        print(f"m{q}: K={K}->{m} | F(>= {theta}) = {F} (corpus {CORPUS_F.get(q)}) | Q* = {Qs} | "
              f"|D| = {w2.shape[0]:,} mass {int(m2.sum()):,} | over0 = {s1['over0']} "
              f"[{row['secs']}s]", flush=True)
        with open(os.path.join(OUT, f"theta_K{K0}_t{theta}.json"), "w") as f:
            json.dump(log, f, indent=1)
        win, mult = w2, m2
        if win.shape[0] == 0 or m <= 1:
            break


if __name__ == "__main__":
    main()
