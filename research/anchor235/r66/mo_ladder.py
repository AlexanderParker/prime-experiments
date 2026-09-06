"""mo_ladder.py -- the operator iterated: T_29, T_31, T_37 from D_K^#(m23), recording everything
the functional tests need at each rung (spectrum with multiplicities, F_J, n_J, Q*_J, the word
statistics L / L_bare / L_pad, and the legal-word and all-pad densities W_r / N, Z_r / N).

This is lc_fixed.py of research/anchor235/r61 with the extra readouts; the closure step itself is
imported unchanged.

Usage: uv run python research/anchor235/r66/mo_ladder.py K0 maxrung [base]
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mo_core import (CORPUS_F, CORPUS_FJ, CORPUS_L, OUT, PRIMES, a_letter_floor, base_gaps,
                     closure_step, dict_from_gaps, fj_from_dict, fj_from_period, next_gear,
                     spectrum_of, summarise_spectrum, tail_excess, word_stats)

MEMCAP = 55_000_000
THRESH_ABS = [1, 2, 5, 10, 20, 30, 40]


def main():
    K0 = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    maxrung = int(sys.argv[2]) if len(sys.argv) > 2 else 37
    base = int(sys.argv[3]) if len(sys.argv) > 3 else 23
    P, g = base_gaps(base)
    t0 = time.time()
    win, mult = dict_from_gaps(g, K0)
    print(f"base m{base}: |D_{K0}| = {win.shape[0]:,}  N = {int(mult.sum()):,}  "
          f"[{time.time()-t0:.1f}s]", flush=True)
    log = []
    prevF = int(g.max())
    for k in range(PRIMES.index(base) + 1, len(PRIMES)):
        q = PRIMES[k]
        if q > maxrung:
            break
        K = win.shape[1]
        t0 = time.time()
        _, _, s1 = closure_step(win, mult, q, m=K, mode="fixed", tag_order=True, collect=False)
        m = max(1, s1["mmin"])
        try:
            w2, m2, st = closure_step(win, mult, q, m=m, mode="fixed", tag_order=False)
        except MemoryError as e:
            print(f"m{q}: STOP {e}", flush=True)
            break
        secs = time.time() - t0
        specJ = s1["specJ"]
        spec = specJ.sum(axis=0)
        F = int(np.flatnonzero(spec).max())
        nJ = {int(j): int(specJ[j].sum()) for j in range(1, specJ.shape[0]) if specJ[j].sum()}
        Qs = {int(j): int(np.flatnonzero(specJ[j]).max()) for j in nJ}
        N = int(spec.sum())
        Pnew = int((spec * np.arange(spec.size)).sum())
        fjs = fj_from_dict(w2, m2, min(m, 10))
        nxt = PRIMES[k + 1]
        L, Lp, Wc, Zc, Lb, Lz = word_stats(w2, m2, nxt)
        Cr = [nxt * N] + [Wc[r - 1] + Zc[r - 1] for r in range(1, min(len(Wc), 8))]
        tails = {}
        for x in THRESH_ABS + [max(1, F // 2), F]:
            e, c = tail_excess(spec, x)
            tails[str(x)] = {"E": e, "count": c}
        row = {"machine": q, "K_in": K, "depth_out": m, "P": Pnew, "N": N, "F": F,
               "mean_gap": Pnew / N,
               "spec": summarise_spectrum(spec),
               "spec_full": {int(v): int(spec[v]) for v in np.flatnonzero(spec)},
               "Fj": fjs, "tails": tails, "nJ": nJ, "Qstar": Qs,
               "loss": st["loss"], "over0": s1["over0"], "kmax_at_K": s1["kmax"],
               "dict_in": int(win.shape[0]), "dict_out": int(w2.shape[0]),
               "L_next": L, "Lbare_next": Lb, "Lpad_next": Lp, "Lallpad_next": Lz,
               "J_max_next": L + 2, "a_L_next": a_letter_floor(nxt),
               "W": Wc[:8], "Z": Zc[:8], "C_r_next": Cr,
               "budget": prevF + q, "slack": prevF + q - F,
               "corpus_F": CORPUS_F.get(q), "gate_F": F == CORPUS_F.get(q),
               "corpus_L": CORPUS_L.get(q), "gate_L": L == CORPUS_L.get(q),
               "sum_m": N, "secs": round(secs, 1)}
        if q in CORPUS_FJ:
            want = CORPUS_FJ[q]
            got = [x for x in fjs if x is not None]
            row["gate_Fj"] = got[:len(want)] == want[:len(got)]
        log.append(row)
        print(f"m{q}: K={K}->{m} F={F} (corpus {CORPUS_F.get(q)}) Fj={fjs} | nJ={nJ} Q*={Qs} | "
              f"|D|={w2.shape[0]:,} N={N:,} loss={st['loss']} over0={s1['over0']} | "
              f"L={L}/{Lb}/{Lp} (corpus {CORPUS_L.get(q)}) | slack={prevF + q - F} "
              f"[{secs:.1f}s]", flush=True)
        with open(os.path.join(OUT, f"ladder_K{K0}_base{base}.json"), "w") as f:
            json.dump(log, f, indent=1)
        win, mult, prevF = w2, m2, F
        if m <= 1 or win.shape[0] > MEMCAP:
            print("STOP: depth exhausted or dictionary too large", flush=True)
            break


if __name__ == "__main__":
    main()
