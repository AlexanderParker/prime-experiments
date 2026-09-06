"""lc_fixed.py -- THE LADDER: the closure step iterated on fixed-depth dictionaries.

D_K^#(M) is the multiset of realised K-windows of consecutive gap sizes.  One rung takes
D_K^#(M) to D_m^#(M + q') exactly, with m the largest depth every (window, phase) pair can
complete inside the K old gaps (lc_core reports it as `mmin`; loss = 0 certifies the step).
Depth is spent, one or two units per rung, so the ladder's reach is set by how deep a dictionary
of the base machine is affordable.

The readouts at each rung are exact whichever depth survives:
  * the whole gap spectrum, hence F, |Spec| and the absent set,
  * n_J (the order distribution) and Q*_J (the largest span a J-fusion attains),
  * F_j for j <= the surviving depth,
  * L and L_pad for the NEXT gear, and the legal-word / all-pad densities by depth.

Usage: uv run python research/anchor235/r61/lc_fixed.py K0 maxrung [base]
"""
import sys, os, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lc_core import (PRIMES, base_gaps, dict_from_gaps, closure_step, group_windows,
                     fj, word_stats, OUT)

CORPUS_F = {7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88, 41: 91,
            43: 103, 47: 118, 53: 145, 59: 161}
CORPUS_FJ = {13: [11, 16, 23, 26, 28, 31], 17: [18, 25, 28, 33, 35, 40],
             19: [25, 31, 35, 38, 47, 50], 23: [34, 39, 50, 58, 65, 77],
             29: [43, 55, 65, 70, 85, 90], 31: [58, 68, 85, 90, 92, 97]}
CORPUS_L = {11: 1, 13: 1, 17: 1, 19: 2, 23: 1, 29: 3, 31: 3, 37: 2, 41: 2, 43: 2, 47: 4, 53: 3}
MEMCAP = 60_000_000


def main():
    K0 = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    maxrung = int(sys.argv[2]) if len(sys.argv) > 2 else 41
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
        # pass 1: measure the depth every pair can complete (no grouping)
        _, _, s1 = closure_step(win, mult, q, m=K, mode="fixed", tag_order=True, collect=False)
        m = max(1, s1["mmin"])
        t1 = time.time()
        # pass 2: build D_m^#(M + q')
        try:
            w2, m2, st = closure_step(win, mult, q, m=m, mode="fixed", tag_order=False)
        except MemoryError as e:
            print(f"m{q}: depth {m} would be exact but the dictionary does not fit: {e}",
                  flush=True)
            log.append({"rung": q, "K_in": K, "depth_out": m, "stopped": str(e)})
            with open(os.path.join(OUT, f"fixed_K{K0}_base{base}.json"), "w") as f:
                json.dump(log, f, indent=1)
            break
        secs = time.time() - t0
        if st["loss"]:
            print(f"m{q}: TRUNCATED, loss {st['loss']:,} at depth {m}", flush=True)
        specJ = s1["specJ"]
        spec = specJ.sum(axis=0)
        F = int(np.flatnonzero(spec).max())
        nJ = {int(j): int(specJ[j].sum()) for j in range(1, specJ.shape[0])
              if specJ[j].sum()}
        Qs = {int(j): int(np.flatnonzero(specJ[j]).max()) for j in nJ}
        N = int(spec.sum())
        fjs = [fj(w2, m2, j) for j in range(1, min(m, 8) + 1)]
        nxt = PRIMES[k + 1]
        L, Lp, Wc, Zc, Lb, Lz = word_stats(w2, m2, nxt)
        gate = ""
        if q in CORPUS_F:
            gate = "F OK" if F == CORPUS_F[q] else f"F FAIL (corpus {CORPUS_F[q]})"
        if q in CORPUS_FJ:
            want = CORPUS_FJ[q][:len(fjs)]
            got = fjs[:len(want)]
            gate += " | Fj " + ("OK" if got == want else f"FAIL got {got} want {want}")
        if q in CORPUS_L:
            gate += " | L " + ("OK" if L == CORPUS_L[q] else f"FAIL got {L} want {CORPUS_L[q]}")
        row = {"rung": q, "K_in": K, "depth_out": m, "F": F, "Fj": fjs, "N": N,
               "dict_in": int(win.shape[0]), "dict_out": int(w2.shape[0]),
               "loss": st["loss"], "over0": s1["over0"], "kmax_at_K": s1["kmax"],
               "nJ": nJ, "Qstar": Qs, "L_next": L, "Lpad_next": Lp, "Lbare_next": Lb,
               "Lallpad_next": Lz,
               "W": Wc[:9], "Z": Zc[:9], "secs": round(secs, 1),
               "budget": prevF + q, "slack": prevF + q - F,
               "spec_top": {int(v): int(spec[v]) for v in np.flatnonzero(spec)[-6:]},
               "absent": [int(v) for v in range(1, F + 1) if spec[v] == 0],
               "sum_m": N, "sum_vm": int((spec * np.arange(spec.size)).sum()),
               "gate": gate, "over0_at_step": s1["over0"]}
        log.append(row)
        print(f"m{q}: K_in={K} -> depth {m} | F={F} Fj={fjs} | |D|={w2.shape[0]:,} "
              f"N={N:,} | loss={st['loss']} over0={s1['over0']} kmax@K={s1['kmax']} | "
              f"nJ={nJ} Q*={Qs} | L={L} Lbare={Lb} Lpad={Lp} Lallpad={Lz} | budget={prevF + q} slack={prevF + q - F} "
              f"| [{secs:.1f}s, measure {t1-t0:.1f}s] {gate}", flush=True)
        with open(os.path.join(OUT, f"fixed_K{K0}_base{base}.json"), "w") as f:
            json.dump(log, f, indent=1)
        win, mult, prevF = w2, m2, F
        if m <= 1:
            print("STOP: depth exhausted (the next rung would have no window to stand on)",
                  flush=True)
            break
        if win.shape[0] > MEMCAP:
            print(f"STOP: |D_{m}(m{q})| = {win.shape[0]:,} rows exceeds the memory budget",
                  flush=True)
            break


if __name__ == "__main__":
    main()
