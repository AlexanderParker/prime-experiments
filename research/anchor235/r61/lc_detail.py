"""lc_detail.py -- the same ladder, dumping the full spectrum and the record's composition.

At each rung: the whole gap spectrum with multiplicities, the chain counts C_r rebuilt from the
legal-word and all-pad counts, the order distribution and its variance, and a witness window for
each Q*_J (the flank + middles + flank that attains the per-depth extreme).

Usage: uv run python research/anchor235/r61/lc_detail.py K0 maxrung [base]
"""
import sys, os, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lc_core import (PRIMES, base_gaps, dict_from_gaps, closure_step, fj, word_stats,
                     witnesses, OUT)

NAMES = {0: "PAD", 1: "UP", 2: "DOWN", 3: "BAD"}


def main():
    K0 = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    maxrung = int(sys.argv[2]) if len(sys.argv) > 2 else 41
    base = int(sys.argv[3]) if len(sys.argv) > 3 else 23
    P, g = base_gaps(base)
    win, mult = dict_from_gaps(g, K0)
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
        specJ = s1["specJ"]
        spec = specJ.sum(axis=0)
        F = int(np.flatnonzero(spec).max())
        nJ = {int(j): int(specJ[j].sum()) for j in range(1, specJ.shape[0]) if specJ[j].sum()}
        Qs = {int(j): int(np.flatnonzero(specJ[j]).max()) for j in nJ}
        N = int(spec.sum())
        # the chain counts, from the incoming gear's legal-word / all-pad counts on D_K(M)
        Lm, Lpm, Wm, Zm, Lbm, Lzm = word_stats(win, mult, q)
        C = [q * int(mult.sum())] + [Wm[r - 1] + Zm[r - 1] for r in range(1, len(Wm) + 1)]
        nJ_chk = [C[J - 1] - 2 * C[J] + C[J + 1] for J in range(1, len(C) - 1)]
        S = sum(C[2:]) / int(mult.sum())
        var = 2 * ((q - 4) + S * (q - 2)) / (q - 2) ** 2
        wit = witnesses(win, mult, q, Qs)
        row = {"rung": q, "F": F, "N": N, "depth_out": m,
               "spectrum": {int(v): int(spec[v]) for v in np.flatnonzero(spec)},
               "nJ": nJ, "Qstar": Qs, "C": C[:8], "nJ_from_C": nJ_chk[:8],
               "L_in": Lm, "Lpad_in": Lpm, "Lbare_in": Lbm, "Lallpad_in": Lzm, "W_in": Wm[:8], "Z_in": Zm[:8],
               "S": S, "var_order": var,
               "witness": {str(J): [{"gaps": w["gaps"],
                                     "letters": [NAMES[x] for x in w["letters"]],
                                     "phase": w["z"], "mult": w["mult"]} for w in ws]
                           for J, ws in wit.items()},
               "budget": prevF + q, "slack": prevF + q - F, "secs": round(time.time() - t0, 1)}
        log.append(row)
        print(f"m{q}: F={F} Q*={Qs} C={C[:6]} nJ_from_C={nJ_chk[:5]} S={S:.6f} "
              f"var={var:.6f} [{row['secs']}s]", flush=True)
        for J, ws in row["witness"].items():
            for w in ws[:1]:
                print(f"    Q*_{J} = {Qs[int(J)]}: gaps {w['gaps']} letters {w['letters']} "
                      f"phase {w['phase']}", flush=True)
        with open(os.path.join(OUT, f"detail_K{K0}_base{base}.json"), "w") as f:
            json.dump(log, f, indent=1)
        w2, m2, st = closure_step(win, mult, q, m=m, mode="fixed")
        assert st["loss"] == 0
        win, mult, prevF = w2, m2, F
        if m <= 1:
            break


if __name__ == "__main__":
    main()
