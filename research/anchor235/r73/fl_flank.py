"""fl_flank.py -- Theorem A of fusion_lemma.md at every step 5 -> 7 .. 37 -> 41: the maximal
words, their neighbourhoods P(m), S(m), N(m), the relaxed term R = max_m [P + |m| + S], the
exact term Q*_{J_max} = max_m [N + |m|] (no phase quantifier), the closed form
Phi = B_{L+1} = max(F(M+q'), R), and the fuse check on every flank.

Sources: m5..m23 by direct sieve of the full period; m29, m31 from the dictionaries saved by
fl_ladder.py (depth 10 and 6; the words here need depth L + 2 = 5); m37 from D_1, D_2
(r70/results/m37_dict.npz) plus the covering instrument (r70/ol_pattern.py) for the D_3 and D_4
memberships it needs.

Gates: the recorded relaxed terms 6, 10, 10, 21, 30, 35, 60, 55, 75, 98 (r66 order.json per_J
rows at level L+1, J = J_max; r70 b4.json), the recorded B_{L+1} 6, 10, 11, 21, 30, 35, 60, 58,
88, 98, the recorded Q*_{J_max} 5, 7, 8, 18, 25, 34, 43, 55, 68, 91, and the corpus F(M+q').

Usage: uv run python research/anchor235/r73/fl_flank.py [tops...]   (default: all ten)
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "r70")))

from fl_core import (PRIMES, analyse, consistent_starts, flank_struck, is_legal,  # noqa: E402
                     longest_legal_run, mirror_check, real_teeth, sieve, u_of,
                     windows_from_dict, windows_from_gaps)

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
R70 = os.path.join(HERE, "..", "r70", "results")

CORPUS_F = {5: 2, 7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88, 41: 91}
REC_RELAXED = {5: 6, 7: 10, 11: 10, 13: 21, 17: 30, 19: 35, 23: 60, 29: 55, 31: 75, 37: 98}
REC_B = {5: 6, 7: 10, 11: 11, 13: 21, 17: 30, 19: 35, 23: 60, 29: 58, 31: 88, 37: 98}
REC_Q = {5: 5, 7: 7, 11: 8, 13: 18, 17: 25, 19: 34, 23: 43, 29: 55, 31: 68, 37: 91}
REC_L = {5: 1, 7: 0, 11: 1, 13: 1, 17: 1, 19: 2, 23: 1, 29: 3, 31: 3, 37: 2}


def step_period(top, q, d):
    gears = [p for p in PRIMES if p <= top]
    P, gaps = sieve(real_teeth(gears))
    L = longest_legal_run(gaps, q, d)
    wL = windows_from_gaps(gaps, L)[0] if L > 0 else None
    wL2 = windows_from_gaps(gaps, L + 2)[0]
    return int(gaps.max()), L, wL, wL2, {"P": int(P), "N": int(gaps.size)}


def step_dict(top, q, d):
    z = np.load(os.path.join(OUT, f"m{top}_dict.npz"))
    win = z["win"]
    L = 0
    for K in range(1, win.shape[1] + 1):
        wK, _ = windows_from_dict(win, K)
        if any(is_legal(tuple(int(x) for x in r), q, d) for r in wK):
            L = K
        else:
            break
    if L + 2 > win.shape[1]:
        raise RuntimeError(f"m{top}: depth {win.shape[1]} < L + 2 = {L + 2}")
    wL = windows_from_dict(win, L)[0] if L > 0 else None
    wL2, short = windows_from_dict(win, L + 2)
    F = int(windows_from_dict(win, 1)[0].max())
    return F, L, wL, wL2, {"rows": int(win.shape[0]), "depth": int(win.shape[1]),
                            "short_rows": short}


def step_m37(q, d):
    """m37: D_1, D_2 from r70; D_3 / D_4 memberships by the covering instrument."""
    from ol_pattern import realised_word, realised_word_search
    gears = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    z = np.load(os.path.join(R70, "m37_dict.npz"))
    win = z["win"]
    vals = sorted(int(v) for v in np.unique(win[:, 0]))
    pairs = set((int(a), int(b)) for a, b in win[:, :2].astype(int))
    F = max(vals)
    memo = {}
    mpath = os.path.join(OUT, "m37_memo.json")
    if os.path.exists(mpath):
        memo = {tuple(int(x) for x in k.split(",")): v for k, v in json.load(open(mpath)).items()}
    calls = {"n": 0, "secs": 0.0}

    def real(w):
        w = tuple(int(x) for x in w)
        if w not in memo:
            t0 = time.time()
            try:
                v = bool(realised_word_search(w, gears))
            except RuntimeError:
                v = bool(realised_word(w, gears))
            memo[w] = v
            calls["n"] += 1
            calls["secs"] += time.time() - t0
        return memo[w]

    words = sorted(p for p in pairs if is_legal(p, q, d))
    L = 2
    assert all(len(w) == 2 for w in words)
    # no legal 3-word: every legal (p, x) with x a letter must be unrealised (corpus L = 2)
    table = []
    R = Q = None
    fuse_fail = 0
    for m in words:
        span = sum(m)
        starts = consistent_starts(m, q, d)
        preds = sorted((a for a in vals if (a, m[0]) in pairs), reverse=True)
        succs = sorted((c for c in vals if (m[1], c) in pairs), reverse=True)
        P = next(a for a in preds if real((a,) + m))
        S = next(c for c in succs if real(m + (c,)))
        # the theorem's flank check: every REALISED (a, m) in D_3 has its flank unstruck at the
        # phase of m.  A struck flank means (a, m) would be a legal 3-word, so only the letter
        # predecessors can fail; those are put to the instrument (they must be unrealised).
        struck_cands = []
        for a in preds:
            if any(l for (l, r) in flank_struck(a, 0, starts, q, d)):
                struck_cands.append(("pred", a, real((a,) + m)))
        for c in succs:
            if any(r for (l, r) in flank_struck(0, c, starts, q, d)):
                struck_cands.append(("succ", c, real(m + (c,))))
        fuse_fail += sum(1 for t in struck_cands if t[2])
        print(f"   m37 word {m}: struck-flank candidates (kind, gap, realised): {struck_cands}",
              flush=True)
        # joint: (a, m, c) in D_4, a <= P, c <= S, descending a + c
        cand = sorted(((a, c) for a in preds if a <= P for c in succs if c <= S),
                      key=lambda t: -(t[0] + t[1]))
        N = None
        Nw = None
        for a, c in cand:
            if N is not None and a + c < N:
                break
            if real((a,) + m) and real(m + (c,)) and real((a,) + m + (c,)):
                N, Nw = a + c, (a, c)
        best_c = max((c for c in succs if c <= S and real((P,) + m + (c,))), default=None)
        best_a = max((a for a in preds if a <= P and real((a,) + m + (S,))), default=None)
        row = {"m": list(m), "span": span, "P": P, "S": S, "N": N, "N_witness": list(Nw),
               "relaxed": P + span + S, "exact": N + span,
               "joint_realised": bool(real((P,) + m + (S,))),
               "best_c_given_P": best_c, "best_a_given_S": best_a, "starts": starts,
               "rows": None}
        table.append(row)
        R = row["relaxed"] if R is None else max(R, row["relaxed"])
        Q = row["exact"] if Q is None else max(Q, row["exact"])
        print(f"   m37 word {m}: P={P} S={S} N={N} {Nw} relaxed={row['relaxed']} "
              f"exact={row['exact']} [{calls['n']} covering problems, {calls['secs']:.0f}s]",
              flush=True)
    an = {"L": L, "J_max": 4, "words": words, "table": table, "R": R, "Qstar_Jmax": Q,
          "fuse_fail": fuse_fail}
    with open(os.path.join(OUT, "m37_memo.json"), "w") as f:
        json.dump({",".join(map(str, k)): v for k, v in memo.items()}, f)
    return F, an, {"D1": len(vals), "D2": len(pairs), "covering_problems": calls["n"],
                   "secs": round(calls["secs"], 1)}


def main():
    tops = [int(x) for x in sys.argv[1:]] or [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    path = os.path.join(OUT, "flank.json")
    rep = json.load(open(path)) if os.path.exists(path) else {}
    for top in tops:
        q = PRIMES[PRIMES.index(top) + 1]
        d = (2 * u_of(q)) % q
        t0 = time.time()
        print(f"\n=== m{top} -> {q}  (u={u_of(q)}, d={d}, a_L={min(d, q-d)})", flush=True)
        if top <= 23:
            F, L, wL, wL2, meta = step_period(top, q, d)
            an = analyse(wL2, wL, q, d, L)
        elif top in (29, 31):
            F, L, wL, wL2, meta = step_dict(top, q, d)
            an = analyse(wL2, wL, q, d, L)
        else:
            F, an, meta = step_m37(q, d)
            L = an["L"]
        Fn = CORPUS_F[q]
        budget = F + q
        Phi = max(Fn, an["R"])
        mir = mirror_check(an["table"])
        row = {"top": top, "q": q, "u": u_of(q), "d": d, "F": F, "F_next": Fn, "budget": budget,
               "L": L, "J_max": L + 2, "n_maximal_words": len(an["words"]),
               "R": an["R"], "Qstar_Jmax": an["Qstar_Jmax"], "Phi_closed_form": Phi,
               "overshoot": Phi - Fn, "margin": budget - Phi, "remainder_ok": an["R"] <= budget,
               "fuse_fail": an["fuse_fail"], "mirror_fail": mir,
               "gates": {"L": L == REC_L[top], "relaxed": an["R"] == REC_RELAXED[top],
                         "B": Phi == REC_B[top], "Qstar": an["Qstar_Jmax"] == REC_Q[top],
                         "Qstar_le_F_next": an["Qstar_Jmax"] <= Fn},
               "table": an["table"], "meta": meta, "secs": round(time.time() - t0, 1)}
        rep[str(top)] = row
        print(f"F={F} F_next={Fn} budget={budget} L={L} (rec {REC_L[top]}) J_max={L+2} "
              f"words={len(an['words'])}", flush=True)
        for r in an["table"]:
            print(f"  m={r['m']} |m|={r['span']}: P={r['P']} S={r['S']} N={r['N']} "
                  f"(N at {r['N_witness']})  relaxed P+|m|+S={r['relaxed']}  exact N+|m|="
                  f"{r['exact']}  joint(P,m,S) realised={r['joint_realised']}  "
                  f"c|P={r['best_c_given_P']} a|S={r['best_a_given_S']} starts={r['starts']}",
                  flush=True)
        print(f"R={an['R']} (rec {REC_RELAXED[top]})  Q*_Jmax={an['Qstar_Jmax']} (rec "
              f"{REC_Q[top]})  Phi=max(F_next,R)={Phi} (rec B_{{L+1}}={REC_B[top]})  "
              f"overshoot={Phi-Fn}  margin={budget-Phi}  fuse_fail={an['fuse_fail']}  "
              f"mirror_fail={mir}  gates={row['gates']}  [{row['secs']}s]", flush=True)
        with open(path, "w") as f:
            json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
