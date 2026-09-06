"""mo_rungs.py -- ITEM 1: the closure step written as an operator T_{q'} and run one rung at a
time, each rung starting from the machine below built directly as a period.

For every base machine M = {5..y} with y <= 23 this script
  * builds M's period exactly (direct sieve in the anchored column coordinate),
  * records M's own invariants: spectrum, F, F_J, mean gap, tail excesses,
  * forms D_K^#(M) (the multiset of realised K-windows of consecutive gap sizes),
  * applies T_{q'} once (lc_core.closure_step, the instrument of ladder_closure.md), and
  * reads off the WHOLE spectrum of M + q', n_J, Q*_J, and the record F(M + q').

The record of the rung is therefore computed from the machine one rung below and nothing else.
The rungs above 23 are the iterated ladder (mo_ladder.py) and the pruned ladder (mo_theta.py).

Usage: uv run python research/anchor235/r66/mo_rungs.py [K]
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mo_core import (CORPUS_F, CORPUS_FJ, CORPUS_L, OUT, PRIMES, a_letter_floor, base_gaps,
                     closure_step, dict_from_gaps, fj_from_period, next_gear, spectrum_of,
                     summarise_spectrum, tail_excess, word_stats)

BASES = [5, 7, 11, 13, 17, 19, 23]
THRESH_ABS = [1, 2, 5, 10, 20, 30]


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    log = []
    for y in BASES:
        q = next_gear(y)
        t0 = time.time()
        P, g = base_gaps(y)
        N = g.size
        spec = spectrum_of(g)
        F = int(g.max())
        fjs = fj_from_period(g, 12)
        mu = P / N
        Kb = min(N, K)
        win, mult = dict_from_gaps(g, Kb)
        tbuild = time.time() - t0

        # ---- the operator, one application -------------------------------------------------
        t1 = time.time()
        w2, m2, st = closure_step(win, mult, q, m=Kb, mode="fixed", tag_order=True, collect=False)
        specJ = st["specJ"]
        newspec = specJ.sum(axis=0)
        Fnew = int(np.flatnonzero(newspec).max())
        nJ = {int(j): int(specJ[j].sum()) for j in range(1, specJ.shape[0]) if specJ[j].sum()}
        Qs = {int(j): int(np.flatnonzero(specJ[j]).max()) for j in nJ}
        tstep = time.time() - t1

        L, Lp, Wc, Zc, Lb, Lz = word_stats(win, mult, q)
        Cr = [q * N] + [Wc[r - 1] + Zc[r - 1] for r in range(1, min(len(Wc), 8))]

        tails = {}
        for x in THRESH_ABS + [max(1, F // 2), F]:
            e, c = tail_excess(spec, x)
            tails[str(x)] = {"E": e, "count": c}

        row = {
            "machine": y, "P": P, "N": N, "F": F, "mean_gap": mu,
            "spec": summarise_spectrum(spec),
            "spec_full": {int(v): int(spec[v]) for v in np.flatnonzero(spec)},
            "Fj": fjs, "tails": tails,
            "rung": q, "K_in": Kb, "dictK": int(win.shape[0]),
            "F_new": Fnew, "corpus_F_new": CORPUS_F.get(q),
            "gate_F": (Fnew == CORPUS_F.get(q)),
            "nJ": nJ, "Qstar": Qs, "over0": st["over0"], "mmin": st["mmin"],
            "kmax": st["kmax"],
            "sum_m": int(newspec.sum()), "want_sum_m": (q - 2) * N,
            "sum_vm": int((newspec * np.arange(newspec.size)).sum()), "want_sum_vm": q * P,
            "new_spec_full": {int(v): int(newspec[v]) for v in np.flatnonzero(newspec)},
            "L": L, "L_bare": Lb, "L_pad": Lp, "L_allpad": Lz, "J_max": L + 2,
            "corpus_L": CORPUS_L.get(y),
            "W": Wc[:8], "Z": Zc[:8], "C_r": Cr,
            "a_L": a_letter_floor(q),
            "secs_build": round(tbuild, 1), "secs_step": round(tstep, 1),
        }
        gates = []
        gates.append("F " + ("OK" if row["gate_F"] else f"FAIL {Fnew} vs {CORPUS_F.get(q)}"))
        gates.append("sum_m " + ("OK" if row["sum_m"] == row["want_sum_m"] else "FAIL"))
        gates.append("sum_vm " + ("OK" if row["sum_vm"] == row["want_sum_vm"] else "FAIL"))
        if y in CORPUS_FJ:
            want = CORPUS_FJ[y]
            gates.append("Fj " + ("OK" if fjs[:len(want)] == want else f"FAIL {fjs[:len(want)]}"))
        if y in CORPUS_L:
            gates.append("L " + ("OK" if L == CORPUS_L[y] else f"FAIL {L} vs {CORPUS_L[y]}"))
        row["gates"] = gates
        log.append(row)
        print(f"m{y}: P={P:,} N={N:,} F={F} mu={mu:.4f} Fj={fjs[:8]} | "
              f"|D_{Kb}|={win.shape[0]:,} -> m{q}: F={Fnew} nJ={nJ} Q*={Qs} over0={st['over0']} "
              f"| L={L} Lbare={Lb} Lpad={Lp} a_L={row['a_L']} | {' | '.join(gates)} "
              f"[{tbuild:.1f}+{tstep:.1f}s]", flush=True)
        with open(os.path.join(OUT, "rungs.json"), "w") as f:
            json.dump(log, f, indent=1)


if __name__ == "__main__":
    main()
