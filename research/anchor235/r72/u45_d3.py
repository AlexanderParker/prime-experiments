"""u45_d3.py -- build D_3(m37) exactly by the closure ladder, and save it.

research/proof/ladder_closure.md 2 and 3.1 record that a base dictionary of depth K_0 = 17 at m23
carries the ladder to m37 at depth 3 (|D_3(m37)| = 30,325 rows, F_j(37) = 88, 90, 97), through
m29 at depth 12 and m31 at depth 8.  The operator is r61/lc_core.closure_step, unchanged; this
driver only adds "write the m37 dictionary to disk", which the r61 run did not do.

D_3(m37) is what the J-run outer law needs at m37: the law at J = 3 is a maximum over the rows of
D_3, and at J = 4, 5 it is the exact prune that keeps the covering instrument's work finite.

Usage: uv run python research/anchor235/r72/u45_d3.py [K0]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "r61")))

from lc_core import base_gaps, dict_from_gaps, closure_step, fj  # noqa: E402

CORPUS_F = {29: 43, 31: 58, 37: 88}
CORPUS_FJ = {29: [43, 55, 65, 70, 85, 90], 31: [58, 68, 85, 90, 92, 97], 37: [88, 90, 97]}
CORPUS_N = {29: 214708725, 31: 6226553025, 37: 217929355875}


def main():
    K0 = int(sys.argv[1]) if len(sys.argv) > 1 else 17
    t0 = time.time()
    P, g = base_gaps(23)
    win, mult = dict_from_gaps(g, K0)
    print(f"base m23: |D_{K0}| = {win.shape[0]:,}  N = {int(mult.sum()):,}  "
          f"[{time.time()-t0:.1f}s]", flush=True)
    log = []
    for q in (29, 31, 37):
        K = win.shape[1]
        t1 = time.time()
        _, _, s1 = closure_step(win, mult, q, m=K, mode="fixed", tag_order=True, collect=False)
        m = max(1, s1["mmin"])
        w2, m2, st = closure_step(win, mult, q, m=m, mode="fixed", tag_order=False)
        spec = s1["specJ"].sum(axis=0)
        F = int(np.flatnonzero(spec).max())
        N = int(spec.sum())
        fjs = [fj(w2, m2, j) for j in range(1, min(m, 8) + 1)]
        row = {"rung": q, "K_in": K, "depth_out": m, "F": F, "Fj": fjs,
               "dict_out": int(w2.shape[0]), "N": N, "loss": st["loss"],
               "over0": s1["over0"], "kmax_at_K": s1["kmax"],
               "gate_F": F == CORPUS_F[q], "gate_N": N == CORPUS_N[q],
               "gate_Fj": fjs == CORPUS_FJ[q][:len(fjs)],
               "secs": round(time.time() - t1, 1)}
        log.append(row)
        print(f"m{q}: K_in={K} -> depth {m} | F={F} (gate {row['gate_F']}) Fj={fjs} "
              f"(gate {row['gate_Fj']}) | |D|={w2.shape[0]:,} N={N:,} (gate {row['gate_N']}) "
              f"| loss={st['loss']} over0={s1['over0']} | [{row['secs']}s]", flush=True)
        win, mult = w2, m2
        if q == 37:
            np.savez_compressed(os.path.join(OUT, "m37_D3.npz"), win=win, mult=mult)
            print(f"  saved D_{m}(m37): {win.shape} rows", flush=True)
    with open(os.path.join(OUT, "d3_ladder.json"), "w") as f:
        json.dump({"K0": K0, "rungs": log, "secs": round(time.time() - t0, 1)}, f, indent=1)
    print(f"total {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
