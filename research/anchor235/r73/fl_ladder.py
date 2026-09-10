"""fl_ladder.py -- the closure ladder m23 -> m29 -> m31, with the m29 and m31 dictionaries SAVED.

The r66 run recorded only summaries; the r70 run (ol_ladder.py) saved only m37's depth-2 table.
This is ol_ladder.py with the two intermediate dictionaries written to disk, nothing else
changed: the closure step is imported unchanged from r61/lc_core through r66/mo_core, so the
tables are the ones that computed F(29) = 43 and F(31) = 58.

Gates (must reproduce): F = 43, 58; |D_10(m29)| = 15,240,585; |D_6(m31)| = 2,678,901; loss = 0,
over0 = 0 at both rungs (monotone_functional.md 1, 2.2; order_law_37_41.md 2.1).

Outputs (results/, untracked): m29_dict.npz, m31_dict.npz, ladder.json.
Usage: uv run python research/anchor235/r73/fl_ladder.py
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "r66")))

from mo_core import CORPUS_F, base_gaps, closure_step, dict_from_gaps  # noqa: E402

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
EXPECT = {29: (43, 15_240_585), 31: (58, 2_678_901)}


def main():
    K0 = 15
    P, g = base_gaps(23)
    t0 = time.time()
    win, mult = dict_from_gaps(g, K0)
    print(f"base m23: |D_{K0}| = {win.shape[0]:,} rows, N = {int(mult.sum()):,} "
          f"[{time.time()-t0:.1f}s]", flush=True)
    log = []
    for q in (29, 31):
        K = win.shape[1]
        t0 = time.time()
        _, _, s1 = closure_step(win, mult, q, m=K, mode="fixed", tag_order=True, collect=False)
        m = max(1, s1["mmin"])
        win, mult, st = closure_step(win, mult, q, m=m, mode="fixed")
        specJ = s1["specJ"]
        spec = specJ.sum(axis=0)
        F = int(np.flatnonzero(spec).max())
        row = {"machine": q, "K_in": K, "depth_out": m, "F": F, "corpus_F": CORPUS_F[q],
               "dict_out": int(win.shape[0]), "N": int(spec.sum()), "sum_m": int(mult.sum()),
               "loss": st["loss"], "over0": s1["over0"],
               "gate": (F == EXPECT[q][0] and win.shape[0] == EXPECT[q][1]
                        and st["loss"] == 0 and s1["over0"] == 0),
               "secs": round(time.time() - t0, 1)}
        log.append(row)
        print(f"m{q}: K={K}->{m} F={F} (corpus {CORPUS_F[q]}) |D|={win.shape[0]:,} "
              f"(expected {EXPECT[q][1]:,}) loss={st['loss']} over0={s1['over0']} "
              f"gate={row['gate']} [{row['secs']}s]", flush=True)
        np.savez_compressed(os.path.join(OUT, f"m{q}_dict.npz"), win=win, mult=mult)
        with open(os.path.join(OUT, "ladder.json"), "w") as f:
            json.dump(log, f, indent=1)


if __name__ == "__main__":
    main()
