"""ol_ladder.py -- the closure ladder m23 -> m29 -> m31 -> m37, kept so that the m37 window
dictionary itself is written to disk (the r66 ladder recorded only its summary).

The closure step is imported unchanged from research/anchor235/r61/lc_core.py through r66/mo_core,
so the operator here is literally the one that computed F(37) = 88 in ladder_closure.md.

Outputs (results/, untracked):
    m37_dict.npz    win, mult of D_m(m37) at whatever depth the ladder reaches (expected m = 2)
    ladder.json     the gate row of every rung

Usage: uv run python research/anchor235/r70/ol_ladder.py [K0]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R66 = os.path.abspath(os.path.join(HERE, "..", "r66"))
sys.path.insert(0, R66)

from mo_core import (CORPUS_F, CORPUS_L, PRIMES, base_gaps, closure_step,  # noqa: E402
                     dict_from_gaps, fj_from_dict, next_gear, word_stats)

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)


def main():
    K0 = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    base = 23
    P, g = base_gaps(base)
    t0 = time.time()
    win, mult = dict_from_gaps(g, K0)
    print(f"base m{base}: |D_{K0}| = {win.shape[0]:,} rows, N = {int(mult.sum()):,} "
          f"[{time.time()-t0:.1f}s]", flush=True)
    log = []
    prevF = int(g.max())
    for q in (29, 31, 37):
        K = win.shape[1]
        t0 = time.time()
        _, _, s1 = closure_step(win, mult, q, m=K, mode="fixed", tag_order=True, collect=False)
        m = max(1, s1["mmin"])
        win, mult, st = closure_step(win, mult, q, m=m, mode="fixed")
        specJ = s1["specJ"]
        spec = specJ.sum(axis=0)
        F = int(np.flatnonzero(spec).max())
        nJ = {int(j): int(specJ[j].sum()) for j in range(1, specJ.shape[0]) if specJ[j].sum()}
        Qs = {int(j): int(np.flatnonzero(specJ[j]).max()) for j in nJ}
        N = int(spec.sum())
        Pnew = int((spec * np.arange(spec.size)).sum())
        nxt = next_gear(q)
        L, Lp, Wc, Zc, Lb, Lz = word_stats(win, mult, nxt)
        row = {"machine": q, "K_in": K, "depth_out": m, "P": Pnew, "N": N, "F": F,
               "corpus_F": CORPUS_F[q], "gate_F": F == CORPUS_F[q],
               "Fj": fj_from_dict(win, mult, min(m, 10)),
               "nJ": nJ, "Qstar": Qs, "loss": st["loss"], "over0": s1["over0"],
               "dict_out": int(win.shape[0]), "sum_m": int(mult.sum()),
               "sum_m_expected": (q - 2) * (N * q // q) if False else None,
               "sum_v_m": Pnew, "L_next": L, "Lbare_next": Lb, "Lpad_next": Lp,
               "corpus_L": CORPUS_L.get(q), "gate_L": L == CORPUS_L.get(q),
               "J_max_next": L + 2, "budget": prevF + q, "slack": prevF + q - F,
               "secs": round(time.time() - t0, 1)}
        log.append(row)
        print(f"m{q}: K={K}->{m} F={F} (corpus {CORPUS_F[q]}) | Q*={Qs} | |D|={win.shape[0]:,} "
              f"N={N:,} loss={st['loss']} over0={s1['over0']} | L={L}/{Lb}/{Lp} "
              f"(corpus {CORPUS_L.get(q)}) J_max={L+2} | [{row['secs']}s]", flush=True)
        prevF = F
        with open(os.path.join(OUT, "ladder.json"), "w") as f:
            json.dump(log, f, indent=1)
        if m <= 1:
            break
    np.savez_compressed(os.path.join(OUT, "m37_dict.npz"), win=win, mult=mult)
    print(f"saved D_{win.shape[1]}(m37): {win.shape[0]:,} rows, mass {int(mult.sum()):,}",
          flush=True)


if __name__ == "__main__":
    main()
