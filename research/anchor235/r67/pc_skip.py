"""pc_skip.py -- the skip law E2 and its cap table, the corridor test (shadow (c)), and the
skeletons of the corpus words at m41..m53.

The skeleton (pad_cap.md 0.1): the openings of a realised legal word sit on two arithmetic
progressions of difference q', x_0 + m q' (class 0) and x_0 + s + m q' (class 1), s in {a, b}.
The column x_0 + m q' is open in M iff for every gear g:  m != (+-u_g - x_0) q'^{-1} (mod g) -
two forbidden residues mod g at separation 2 u_g q'^{-1} mod g.  So each class's multiplier set
is a set of openings of a two-tooth machine on the multiplier line (the PULLBACK of M along the
AP; same gears, separations 2u_g q'^{-1}), and its size inside [0, T] is at most

    Omega(T + 1) := max over phases of the number of openings of the pullback in T + 1
                    consecutive multipliers,

so L + 1 = |S_0| + |S_1| <= 2 Omega(T + 1) with T = floor((F(M+q') - 2)/q') (docs/proofs/11's
span cap).  Omega depends on the separations, i.e. on q' mod the gears; taking the max over
every invertible q' gives a table uniform in the rung.  Gears g > n cannot be forced into a
window of n multipliers, so Omega over the gears <= n is exact for the small gears.

Usage: uv run python research/anchor235/r67/pc_skip.py
"""
import json
import os
import sys
from math import gcd

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pc_core import E35, OUT, exposed_set, letter_data, skeleton, word_kind, is_legal_word

NMAX = 40


def omega_table(gears, nmax=NMAX):
    """Omega(n) = max over invertible steps q' mod Q and starts x of #{m < n : x + m q' in E_Q},
    E_Q the residues open under the gears.  Also the per-step-class value."""
    Q, E = exposed_set(gears)
    isE = np.zeros(Q, dtype=bool)
    isE[E] = True
    steps = [s for s in range(1, Q) if gcd(s, Q) == 1]
    x = np.arange(Q)
    out = {}
    per_step = {}
    for n in range(1, nmax + 1):
        best = 0
        ps = {}
        for s in steps:
            m = np.arange(n)
            idx = (x[:, None] + m[None, :] * s) % Q
            c = int(isE[idx].sum(axis=1).max())
            ps[s] = c
            best = max(best, c)
        out[n] = best
        per_step[n] = ps
    return out, per_step, Q


def carrier(gaps, E=E35, Q=35):
    """Residues r mod Q such that every partial sum r, r+g1, ... is in E (docs/proofs/14 (c))."""
    Es = set(E)
    out = []
    for r in Es:
        o = r
        ok = True
        for g in gaps:
            o = (o + g) % Q
            if o not in Es:
                ok = False
                break
        if ok:
            out.append(r)
    return sorted(out)


def corrcap_span(q, F, G):
    """The longest legal word with letter values <= F (legal mod q'), total span <= G - 2, whose
    prefix-sum walk stays in E_35 from some start: a DP on (span used, residue mod 35, last
    nonzero class).  Exact; combines the corridor (file 14) with the span cap (file 11)."""
    Ld = letter_data(q)
    d = Ld["d"]
    letters = [(v, 0 if v % q == 0 else (1 if v % q == d else 2))
               for v in range(1, F + 1) if v % q in (0, d, (-d) % q)]
    Smax = G - 2
    Es = set(E35)
    best = {}  # (o, r, last) -> longest word length reaching this state
    frontier = {(0, r, 0): 0 for r in Es}
    best.update(frontier)
    L = 0
    witness = {(0, r, 0): [] for r in Es}
    while frontier:
        nxt = {}
        for (o, r, last), ln in frontier.items():
            for v, c in letters:
                if o + v > Smax or (c != 0 and c == last):
                    continue
                r2 = (r + v) % 35
                if r2 not in Es:
                    continue
                key = (o + v, r2, last if c == 0 else c)
                if best.get(key, -1) < ln + 1:
                    best[key] = ln + 1
                    nxt[key] = ln + 1
                    witness[key] = witness[(o, r, last)] + [v]
        frontier = nxt
        if frontier:
            L = max(L, max(frontier.values()))
    arg = max(best, key=lambda k: best[k])
    return L, witness[arg]


def main():
    res = {}
    # ---- the exact corridor + span cap per corpus rung
    corp = {19: (23, 25, 34), 23: (29, 34, 43), 29: (31, 43, 58), 31: (37, 58, 88), 37: (41, 88, 91),
            41: (43, 91, 103), 43: (47, 103, 118), 47: (53, 118, 145), 53: (59, 145, 161)}
    corpL = {19: 2, 23: 1, 29: 3, 31: 3, 37: 2, 41: 2, 43: 2, 47: 4, 53: 3}
    res["corrcap_span"] = {}
    print("exact corridor+span cap CC(q', F(M), F(M+q')):  rung  CC  witness  L")
    for y, (q, F, G) in corp.items():
        cc, w = corrcap_span(q, F, G)
        res["corrcap_span"][y] = {"q": q, "F": F, "G": G, "CC": cc, "witness": w, "L": corpL[y]}
        print(f"  m{y}: q'={q} F={F} G={G} CC={cc} witness={w} L={corpL[y]}")
    # ---- Omega tables
    for gs in [(5,), (5, 7), (5, 7, 11), (5, 7, 11, 13)]:
        om, ps, Q = omega_table(list(gs), NMAX if len(gs) < 4 else 24)
        res[f"omega_{'_'.join(map(str, gs))}"] = om
        print(f"Omega with gears {gs} (mod {Q}): {[om[n] for n in sorted(om)]}")
    om57 = res["omega_5_7"]
    om5711 = res["omega_5_7_11"]
    om4 = res["omega_5_7_11_13"]
    print("T : L_cap(E2, gears 5,7) = 2 Omega(T+1) - 1  vs  file 11's 2T + 1  vs  gears 5,7,11  vs 5,7,11,13")
    cap = {}
    for T in range(1, 23):
        cap[T] = {"E2_57": 2 * om57[T + 1] - 1, "file11": 2 * T + 1,
                  "E2_5711": 2 * om5711[T + 1] - 1,
                  "E2_571113": (2 * om4[T + 1] - 1) if (T + 1) in om4 else None}
        print(f"  T={T:2d}: {cap[T]['E2_57']:3d}  {cap[T]['file11']:3d}  {cap[T]['E2_5711']:3d}  {cap[T]['E2_571113']}")
    res["cap_by_T"] = cap
    # per-class (q' mod 35) Omega_{5,7}(n) for the corpus rungs
    _, ps57, _ = omega_table([5, 7], 12)
    corpus = {19: (23, 34), 23: (29, 43), 29: (31, 58), 31: (37, 88), 37: (41, 91), 41: (43, 103),
              43: (47, 118), 47: (53, 145), 53: (59, 161)}
    corpusL = {19: 2, 23: 1, 29: 3, 31: 3, 37: 2, 41: 2, 43: 2, 47: 4, 53: 3}
    rows = {}
    print("rung: q' G=F(M+q') T  Omega_57(T+1; q' mod 35)  cap_class  cap_uniform  file11  L")
    for y, (q, G) in corpus.items():
        T = (G - 2) // q
        oc = ps57[T + 1][q % 35]
        rows[y] = {"q": q, "G": G, "T": T, "omega_class": oc, "cap_class": 2 * oc - 1,
                   "cap_uniform": 2 * om57[T + 1] - 1, "file11": 2 * T + 1, "L": corpusL[y]}
        print(f"  m{y}: q'={q} G={G} T={T} Omega={oc} cap_class={2*oc-1} cap_uniform={2*om57[T+1]-1} "
              f"file11={2*T+1} L={corpusL[y]}")
    res["corpus_caps"] = rows
    # ---- corridor test (c): the m37 skip 2-words and the holes; m53's words
    tests = {"m37 (27,55)": [27, 55], "m37 (55,27)": [55, 27], "m37 (14,68)": [14, 68],
             "m37 (68,14)": [68, 14], "m37 (41,41)": [41, 41], "m37 (41,55)": [41, 55],
             "m37 gap 82": [82], "m29 gap 41": [41], "m31 gap 49": [49], "m37 gap 55": [55],
             "m37 gap 68": [68], "m53 (20,39,20)": [20, 39, 20], "m53 (20,98,20)": [20, 98, 20],
             "m53 (20,98)": [20, 98], "m53 (20,118)": [20, 118], "m53 (20,59,39)": [20, 59, 39],
             "m53 (39,59,20)": [39, 59, 20], "m47 (35,71,35)": [35, 71, 35],
             "m47 (18,35,53)": [18, 35, 53], "m47 (18,35,18,35)": [18, 35, 18, 35]}
    res["carriers"] = {}
    print("corridor carriers (mod 35):")
    for k, gs in tests.items():
        c = carrier(gs)
        res["carriers"][k] = c
        print(f"  {k}: carrier={c} ({'EMPTY' if not c else len(c)})")
    # ---- skeletons of the corpus words
    words = {41: [(43, 43)], 43: [(47, 47)],
             47: [(18, 35, 18, 35), (18, 35, 53), (18, 53, 35), (35, 18, 53), (35, 71, 35)],
             53: [(20, 39), (20, 59), (20, 98), (20, 118), (20, 98, 20)]}
    qs = {41: 43, 43: 47, 47: 53, 53: 59}
    res["corpus_skeletons"] = {}
    print("skeletons of the corpus words:")
    for y, ws in words.items():
        q = qs[y]
        for w in ws:
            s = skeleton(w, q)
            kind = word_kind(w, q)
            res["corpus_skeletons"][f"m{y} {w}"] = {"kind": kind, "S0": s["S0"], "S1": s["S1"],
                                                    "runs": s["runs0"] + s["runs1"],
                                                    "skips": s["skips0"] + s["skips1"],
                                                    "jumps": s["jumps"], "legal": is_legal_word(w, q)}
            print(f"  m{y} q'={q} {w}: {kind} S0={s['S0']} S1={s['S1']} skips={s['skips0']+s['skips1']} "
                  f"jumps={s['jumps']} legal={is_legal_word(w, q)}")
    with open(os.path.join(OUT, "skip.json"), "w") as f:
        json.dump(res, f, indent=1, default=int)


if __name__ == "__main__":
    main()
