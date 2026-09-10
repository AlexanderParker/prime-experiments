"""fl_next.py -- the remainder R(M; q') = max_m [P(m) + |m| + S(m)] at a step nothing has scanned,
by covering problems alone (fusion_lemma.md, Theorem A makes it a finite list).

At 41 -> 43: engine {5..41} (11 gears, period 5.3e13, never scanned), q' = 43, u = 36, d = 29,
letters DOWN = 14, 57; UP = 29, 72; PAD = 43, 86 (values <= F(m41) = 91).  Corpus: L(m41) = 2
with the word (43, 43); F(43) = 103; budget 91 + 43 = 134.

Steps: (1) every legal 2-word over the six letters is put to the instrument (28 candidates);
(2) the realised ones are the maximal words (gate: no legal 3-word over them is realised, the
corpus L = 2); (3) for each, P and S by descending covering problems from a = 91, and N by the
joint (a, m, c) in descending a + c; (4) R and Q*_4 = max N + |m|; gate Q*_4 <= F(43) = 103.

Usage: uv run python research/anchor235/r73/fl_next.py [top=41]
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "r70")))

from fl_core import PRIMES, consistent_starts, flank_struck, is_legal, letter_codes, u_of  # noqa
from ol_pattern import realised_word, realised_word_search  # noqa: E402

OUT = os.path.join(HERE, "results")
CORPUS_F = {37: 88, 41: 91, 43: 103, 47: 118, 53: 145, 59: 161}
CORPUS_L = {37: 2, 41: 2, 43: 2, 47: 4, 53: 3}


def main():
    top = int(sys.argv[1]) if len(sys.argv) > 1 else 41
    gears = [p for p in PRIMES if p <= top]
    q = PRIMES[PRIMES.index(top) + 1]
    d = (2 * u_of(q)) % q
    F = CORPUS_F[top]
    L = CORPUS_L[top]
    budget = F + q
    mpath = os.path.join(OUT, f"m{top}_memo.json")
    memo = {}
    if os.path.exists(mpath):
        memo = {tuple(int(x) for x in k.split(",")): v for k, v in json.load(open(mpath)).items()}
    calls = {"n": 0, "secs": 0.0, "fallback": 0}

    def real(w):
        w = tuple(int(x) for x in w)
        if w not in memo:
            t0 = time.time()
            try:
                v = bool(realised_word_search(w, gears, budget=3_000_000))
            except RuntimeError:
                calls["fallback"] += 1
                v = bool(realised_word(w, gears))
            memo[w] = v
            calls["n"] += 1
            calls["secs"] += time.time() - t0
            if calls["n"] % 25 == 0:
                with open(mpath, "w") as f:
                    json.dump({",".join(map(str, k)): v for k, v in memo.items()}, f)
        return memo[w]

    letters = [v for v in range(1, F + 1) if letter_codes([v], q, d)[0] != 3]
    print(f"m{top} -> {q}: u={u_of(q)} d={d} F={F} L={L} budget={budget}; letters {letters}",
          flush=True)
    # (1) the maximal words: every legal L-word over the letters, tested
    import itertools
    cands = [w for w in itertools.product(letters, repeat=L) if is_legal(w, q, d)]
    t0 = time.time()
    words = []
    for w in sorted(cands, key=lambda w: sum(w)):
        r = real(w)
        print(f"  legal {L}-word {w} span {sum(w)}: {'REALISED' if r else 'no'} "
              f"[{calls['n']} problems, {calls['secs']:.0f}s]", flush=True)
        if r:
            words.append(w)
    # (2) gate: no legal (L+1)-word over the realised ones
    ext = []
    for w in words:
        for v in letters:
            for e in ((v,) + w, w + (v,)):
                if is_legal(e, q, d) and e not in ext:
                    ext.append(e)
    ext_real = [e for e in ext if real(e)]
    print(f"maximal words: {words}; legal (L+1)-extensions tested {len(ext)}, realised "
          f"{ext_real} (corpus L = {L} needs none)", flush=True)
    # (3) neighbourhoods
    table = []
    R = Q = None
    for m in words:
        span = sum(m)
        starts = consistent_starts(m, q, d)
        P = next(a for a in range(F, 0, -1) if real((a,) + m))
        S = next(c for c in range(F, 0, -1) if real(m + (c,)))
        print(f"  m={m}: P={P} S={S} [{calls['n']} problems, {calls['secs']:.0f}s]", flush=True)
        # flanks: struck candidates among realised preds/succs must be none
        struck = []
        for a in range(1, P + 1):
            if any(l for (l, r) in flank_struck(a, 0, starts, q, d)) and real((a,) + m):
                struck.append(("pred", a))
        for c in range(1, S + 1):
            if any(r for (l, r) in flank_struck(0, c, starts, q, d)) and real(m + (c,)):
                struck.append(("succ", c))
        # joint N: descending a + c over realised preds and succs
        preds = [a for a in range(P, 0, -1) if real((a,) + m)]
        succs = [c for c in range(S, 0, -1) if real(m + (c,))]
        pairs = sorted(((a, c) for a in preds for c in succs), key=lambda t: -(t[0] + t[1]))
        N = None
        Nw = None
        for a, c in pairs:
            if N is not None and a + c < N:
                break
            if real((a,) + m + (c,)):
                N, Nw = a + c, (a, c)
        row = {"m": list(m), "span": span, "P": P, "S": S, "N": N, "N_witness": list(Nw),
               "relaxed": P + span + S, "exact": N + span, "struck_flanks_realised": struck,
               "preds": preds, "succs": succs, "starts": starts}
        table.append(row)
        R = row["relaxed"] if R is None else max(R, row["relaxed"])
        Q = row["exact"] if Q is None else max(Q, row["exact"])
        print(f"  m={m}: N={N} at {Nw}; relaxed {row['relaxed']}, exact {row['exact']}; "
              f"struck flanks realised: {struck} [{calls['n']} problems, {calls['secs']:.0f}s]",
              flush=True)
    rep = {"top": top, "q": q, "d": d, "F": F, "F_next": CORPUS_F[q], "budget": budget, "L": L,
           "maximal_words": [list(w) for w in words], "extensions_realised": ext_real,
           "table": table, "R": R, "Qstar_Jmax": Q, "Phi_closed_form": max(CORPUS_F[q], R),
           "remainder_ok": R <= budget, "gate_Qstar_le_F_next": Q <= CORPUS_F[q],
           "covering_problems": calls["n"], "fallbacks": calls["fallback"],
           "secs": round(time.time() - t0, 1)}
    print(json.dumps({k: v for k, v in rep.items() if k != "table"}, indent=1), flush=True)
    with open(os.path.join(OUT, f"next_{top}_{q}.json"), "w") as f:
        json.dump(rep, f, indent=1)
    with open(mpath, "w") as f:
        json.dump({",".join(map(str, k)): v for k, v in memo.items()}, f)


if __name__ == "__main__":
    main()
