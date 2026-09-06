"""lc_theta.py -- the ladder with the span-threshold prune, for the TOP of the spectrum.

The record is strictly increasing along the ladder, so a target rung r has `F(M_r) > F(M_{r-1})`
and a threshold `theta = F(M_{r-1}) + 1` is a valid lower bound for it.  A window of `M_i` whose
whole stored window spans less than `theta` can never be the ancestor of a gap of size `>= theta`
(a sub-run never spans more than the window), so it may be dropped -- at EVERY level, and without
losing a single window of span `>= theta`.  What comes out is exact for every value `>= theta`:
the multiset of windows of span `>= theta` at every rung, hence `F`, and `Q*_J` for the values
that matter.  Everything below theta is deliberately absent.

Usage: uv run python research/anchor235/r61/lc_theta.py K0 theta maxrung [base]
"""
import sys, os, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lc_core import PRIMES, base_gaps, dict_from_gaps, closure_step, group_windows, OUT

CORPUS_F = {29: 43, 31: 58, 37: 88, 41: 91, 43: 103, 47: 118, 53: 145, 59: 161}


def main():
    K0 = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    theta = int(sys.argv[2]) if len(sys.argv) > 2 else 89
    maxrung = int(sys.argv[3]) if len(sys.argv) > 3 else 41
    base = int(sys.argv[4]) if len(sys.argv) > 4 else 23
    P, g = base_gaps(base)
    t0 = time.time()
    win, mult = dict_from_gaps(g, K0)
    keep = np.flatnonzero(win.astype(np.int32).sum(axis=1) >= theta)
    win, mult = group_windows(np.ascontiguousarray(win[keep]), mult[keep])
    print(f"base m{base}: |D_{K0}| pruned to span >= {theta}: {win.shape[0]:,} rows, "
          f"mass {int(mult.sum()):,} of {g.size:,}  [{time.time()-t0:.1f}s]", flush=True)
    log = []
    for k in range(PRIMES.index(base) + 1, len(PRIMES)):
        q = PRIMES[k]
        if q > maxrung:
            break
        K = win.shape[1]
        t0 = time.time()
        _, _, s1 = closure_step(win, mult, q, m=K, mode="fixed", tag_order=True,
                                collect=False, theta=theta)
        m = max(1, s1["mmin"])
        w2, m2, st = closure_step(win, mult, q, m=m, mode="fixed", theta=theta)
        secs = time.time() - t0
        specJ = s1["specJ"]
        spec = specJ.sum(axis=0)
        F = int(np.flatnonzero(spec).max()) if spec.any() else 0
        Qs = {int(j): int(np.flatnonzero(specJ[j]).max())
              for j in range(1, specJ.shape[0]) if specJ[j].any()}
        gate = ""
        if q in CORPUS_F:
            gate = ("F OK" if F == CORPUS_F[q] else
                    f"F = {F} vs corpus {CORPUS_F[q]}")
        row = {"rung": q, "K_in": K, "depth_out": m, "F_above_theta": F, "Qstar": Qs,
               "dict_in": int(win.shape[0]), "dict_out": int(w2.shape[0]),
               "mass_out": int(m2.sum()), "loss": st["loss"], "over0": s1["over0"],
               "kmax": s1["kmax"], "theta": theta, "secs": round(secs, 1), "gate": gate}
        log.append(row)
        print(f"m{q}: K_in={K} -> depth {m} | F(>= {theta}) = {F} | Q* = {Qs} | "
              f"|D| = {w2.shape[0]:,} mass {int(m2.sum()):,} | loss={st['loss']} "
              f"over0={s1['over0']} kmax={s1['kmax']} | [{secs:.1f}s] {gate}", flush=True)
        with open(os.path.join(OUT, f"theta_K{K0}_t{theta}.json"), "w") as f:
            json.dump(log, f, indent=1)
        win, mult = w2, m2
        if win.shape[0] == 0:
            print("STOP: nothing above the threshold survives", flush=True)
            break
        if m <= 1:
            print("STOP: depth exhausted", flush=True)
            break


if __name__ == "__main__":
    main()
