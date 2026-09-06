"""lc_ladder.py -- the ladder run on span-bounded dictionaries alone.

V_S(M) = the multiset of opening patterns of M inside [x, x + S], one entry per opening x,
written as the gap word truncated at span S.  Exactly closed under the rung step (lc_core).
From it, at every rung and with no period anywhere:

    F, F_2, F_3, ...            the top of the F_j ladder
    the whole gap spectrum      (values <= S)
    n_J and Q*_J                the order distribution and the per-depth extremes
    L, L_pad                    the longest realised legal / all-PAD word for the NEXT gear
    W_m/N, Z_m/N                legal-word and all-pad densities by depth

Usage: uv run python research/anchor235/r61/lc_ladder.py S maxrung [base]
"""
import sys, os, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lc_core import (PRIMES, base_gaps, dict_from_gaps, closure_step, group_windows,
                     spectrum, fj, word_stats, OUT)

CORPUS_F = {7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88, 41: 91,
            43: 103, 47: 118, 53: 145, 59: 161}
CORPUS_FJ = {13: [11, 16, 23, 26, 28, 31], 17: [18, 25, 28, 33, 35, 40],
             19: [25, 31, 35, 38, 47, 50], 23: [34, 39, 50, 58, 65, 77],
             29: [43, 55, 65, 70, 85, 90], 31: [58, 68, 85, 90, 92, 97]}


def width_for(gaps, S):
    N = gaps.size
    g = np.concatenate([gaps, gaps[:2 * S + 4]]).astype(np.int32)
    best = 0
    for lo in range(0, N, 4_000_000):
        hi = min(lo + 4_000_000, N)
        off = np.zeros(hi - lo, dtype=np.int32)
        cnt = np.zeros(hi - lo, dtype=np.int32)
        for i in range(2 * S + 4):
            off += g[lo + i:hi + i]
            live = off <= S
            if not live.any():
                break
            cnt += live
        best = max(best, int(cnt.max()))
    return best


def report(q, win, mult, st, S, secs, prevF, log):
    ng = win.shape[1] - 1                        # last column is the order tag
    val = win[:, 0].astype(np.int64)
    order = win[:, ng].astype(np.int64)
    sp = np.zeros(S + 2, dtype=np.int64)
    for v in np.unique(val):
        sp[v] = int(mult[val == v].sum())
    F = int(np.flatnonzero(sp).max())
    nJ, Qs = {}, {}
    for j in np.unique(order):
        if j == 0:
            continue
        sel = order == j
        nJ[int(j)] = int(mult[sel].sum())
        Qs[int(j)] = int(val[sel].max())
    core = np.ascontiguousarray(win[:, :ng])
    core, cmult = group_windows(core, mult)
    fjs = [fj(core, cmult, j) for j in range(1, 7)]
    nxt = PRIMES[PRIMES.index(q) + 1]
    L, Lp, Wc, Zc, Lb, Lz = word_stats(core, cmult, nxt)
    N = int(mult.sum())
    row = {"rung": q, "F": F, "Fj": fjs, "N": N, "dict": int(core.shape[0]),
           "width": int(core.shape[1]), "S": S, "secs": round(secs, 1),
           "loss": st["loss"], "over0": st["over0"], "kmax": st["kmax"],
           "nJ": nJ, "Qstar": Qs, "L_next": L, "Lpad_next": Lp,
           "W": Wc[:8], "Z": Zc[:8],
           "spec_top": {int(v): int(sp[v]) for v in np.flatnonzero(sp)[-8:]},
           "absent": [int(v) for v in range(1, F + 1) if sp[v] == 0],
           "sum_m": int(sp.sum()), "sum_vm": int((sp * np.arange(sp.size)).sum())}
    if prevF is not None:
        row["budget"] = prevF + q
        row["slack"] = prevF + q - F
    gate = ""
    if q in CORPUS_F:
        gate = "F GATE OK" if F == CORPUS_F[q] else f"F GATE FAIL (corpus {CORPUS_F[q]})"
    if q in CORPUS_FJ:
        want = CORPUS_FJ[q]
        got = fjs[:len(want)]
        gate += " | Fj GATE " + ("OK" if all(a == b for a, b in zip(got, want) if a) else
                                 f"FAIL got {got} want {want}")
    row["gate"] = gate
    log.append(row)
    print(f"m{q}: F={F} Fj={fjs} |V_S|={core.shape[0]:,} w={core.shape[1]} N={N:,} "
          f"loss={st['loss']} over0={st['over0']} L={L} Lpad={Lp} "
          f"nJ={nJ} Q*={Qs} slack={row.get('slack')} [{secs:.1f}s] {gate}", flush=True)
    return F


def main():
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 95
    maxrung = int(sys.argv[2]) if len(sys.argv) > 2 else 41
    base = int(sys.argv[3]) if len(sys.argv) > 3 else 23
    P, g = base_gaps(base)
    W = width_for(g, S) + 1
    t0 = time.time()
    win, mult = dict_from_gaps(g, W, span_cap=S)
    print(f"S={S} base m{base}: |V_S|={win.shape[0]:,} N={int(mult.sum()):,} width={W} "
          f"overfull={int((win[:, -1] != 0).sum())} [{time.time()-t0:.1f}s]", flush=True)
    log = []
    prevF = int(g.max())
    for k in range(PRIMES.index(base) + 1, len(PRIMES)):
        q = PRIMES[k]
        if q > maxrung:
            break
        t0 = time.time()
        w2, m2, st = closure_step(win, mult, q, m=win.shape[1], mode="span", tag_order=True)
        secs = time.time() - t0
        prevF = report(q, w2, m2, st, S, secs, prevF, log)
        ng = w2.shape[1] - 1
        win, mult = group_windows(np.ascontiguousarray(w2[:, :ng]), m2)
        del w2, m2
        with open(os.path.join(OUT, f"ladder_S{S}_base{base}.json"), "w") as f:
            json.dump(log, f, indent=1)
        if win.shape[0] > 55_000_000:
            print(f"STOP: |V_S(m{q})| = {win.shape[0]:,} rows exceeds the memory budget",
                  flush=True)
            break


if __name__ == "__main__":
    main()
