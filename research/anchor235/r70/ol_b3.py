"""ol_b3.py -- B_3(m37; 41), the level-3 relaxation, by the free-phase pattern instrument.

The record law caps the order at J_max = L(m37) + 2 = 4, so

    B_3 = max over J <= 4 of the widest level-3 admissible J-word that fuses at some phase of 41.

For J <= 3 a level-3 admissible word IS a realised window, so those three terms are the exact
Q*_J(m37; 41).  Only the J = 4 term is relaxed, and its relaxation is one de Bruijn step: the word
(g1, g2, g3, g4) needs (g1, g2, g3) and (g2, g3, g4) in D_3(m37) and nothing more.

D_3(m37) is never built as a whole (m37 has period 1.24e12 and the closure ladder leaves it at
depth 2).  Instead every candidate word is put to ol_pattern.realised_word, which decides
membership in D_k(m37) exactly by the CRT free-phase covering problem.  Words are enumerated
completely -- every phase of 41, every realised gap value in each slot -- and tested in DESCENDING
span, so the first word whose triples are all realised is the maximum.

Usage: uv run python research/anchor235/r70/ol_b3.py
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "r66")))

from ol_pattern import realised_word, realised_word_search  # noqa: E402

OUT = os.path.join(HERE, "results")
GEARS = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
Q = 41
U = pow(6, -1, Q)
D = (2 * U) % Q            # 14
STRUCK = {0, D}
JMAX = 4
F37 = 88
BUDGET = F37 + Q           # 129

MEMO = {}
CALLS = {"n": 0, "secs": 0.0, "nodes": 0, "fallback": 0}


def is_realised(word):
    key = tuple(int(x) for x in word)
    if key not in MEMO:
        t0 = time.time()
        c = {}
        try:
            v = bool(realised_word_search(key, GEARS, counter=c))
        except RuntimeError:                      # search budget: fall back to the enumeration
            CALLS["fallback"] += 1
            v = bool(realised_word(key, GEARS))
        MEMO[key] = v
        CALLS["n"] += 1
        CALLS["nodes"] += c.get("nodes", 0)
        CALLS["secs"] += time.time() - t0
    return MEMO[key]


def letters(v):
    r = v % Q
    return 0 if r == 0 else (1 if r == D else (2 if r == (-D) % Q else 3))


def enumerate_words(J, vals, pairs):
    """Every J-word of realised gap values that fuses at some phase of 41, with its phase.
    Level-2 admissible (all adjacent pairs realised) -- a free exact filter."""
    out = {}
    for z in range(Q):
        if z in STRUCK:
            continue
        stack = [([], 0)]
        for i in range(J):
            nxt = []
            for w, off in stack:
                for g in vals:
                    if w and (w[-1], g) not in pairs:
                        continue
                    o = off + g
                    hit = (o + z) % Q in STRUCK
                    if i < J - 1:
                        if hit:
                            nxt.append((w + [g], o))
                    else:
                        if not hit:
                            nxt.append((w + [g], o))
            stack = nxt
        for w, off in stack:
            key = tuple(w)
            if key not in out:
                out[key] = z
    return out


def main():
    z = np.load(os.path.join(OUT, "m37_dict.npz"))
    win = z["win"]
    vals = sorted(int(v) for v in np.unique(win[:, 0]))
    pairs = set(map(tuple, win[:, :2].astype(int)))
    print(f"m37: |D_1| = {len(vals)} (max {max(vals)}), |D_2| = {len(pairs)}; "
          f"gear {Q}, u = {U}, d = {D}, budget {BUDGET}", flush=True)

    report = {"D1": len(vals), "D2": len(pairs), "budget": BUDGET, "q": Q, "d": D,
              "J_max": JMAX, "terms": {}}

    # ---- J = 1, 2 : exact off D_1, D_2 (level 3 does not relax them)
    for J in (1, 2):
        words = enumerate_words(J, vals, pairs)
        best = max(words, key=lambda w: sum(w))
        report["terms"][str(J)] = {"span": sum(best), "word": list(best),
                                   "phase": words[best], "exact": True,
                                   "candidates": len(words)}
        print(f"J = {J}: exact Q*_{J} = {sum(best)}  word {list(best)} at phase {words[best]} "
              f"({len(words):,} fusing words)", flush=True)

    # ---- J = 3 : exact, needs D_3 membership of the word itself
    for J in (3, 4):
        words = enumerate_words(J, vals, pairs)
        order = sorted(words, key=lambda w: -sum(w))
        print(f"J = {J}: {len(words):,} level-2 admissible fusing words, spans "
              f"{sum(order[0])} down to {sum(order[-1])}", flush=True)
        t0 = time.time()
        found = None
        ties = []
        tested = 0
        for w in order:
            if found is not None and sum(w) < sum(found):
                break
            if J == 3:
                need = [w]
            else:
                need = [w[:3], w[1:]]
            tested += 1
            if all(is_realised(t) for t in need):
                if found is None:
                    found = w
                ties.append(list(w))
                continue
            if tested % 500 == 0:
                print(f"  ... {tested:,} words tested, at span {sum(w)}, "
                      f"{CALLS['n']:,} membership calls, {time.time()-t0:.0f}s", flush=True)
        span = sum(found) if found else None
        report["terms"][str(J)] = {"span": span, "word": list(found) if found else None,
                                   "phase": words[found] if found else None,
                                   "exact": J <= 3, "candidates": len(words),
                                   "words_tested": tested, "ties": ties,
                                   "membership_calls": CALLS["n"],
                                   "secs": round(time.time() - t0, 1)}
        print(f"J = {J}: {'Q*_3' if J == 3 else 'level-3 term'} = {span}  word "
              f"{list(found) if found else None} at phase "
              f"{words[found] if found else None}  [{tested:,} words, {CALLS['n']:,} calls, "
              f"{time.time()-t0:.0f}s]", flush=True)
        with open(os.path.join(OUT, "b3.json"), "w") as f:
            json.dump(report, f, indent=1)

    B3 = max(report["terms"][k]["span"] for k in report["terms"]
             if report["terms"][k]["span"] is not None)
    arg = [k for k in report["terms"] if report["terms"][k]["span"] == B3]
    report["B_3"] = B3
    report["argmax_J"] = arg
    report["verdict"] = {"B_3": B3, "budget": BUDGET, "within": B3 <= BUDGET}
    report["memo"] = {"calls": CALLS["n"], "secs": round(CALLS["secs"], 1),
                      "search_nodes": CALLS["nodes"], "fallbacks": CALLS["fallback"],
                      "true": int(sum(MEMO.values())), "false": int(len(MEMO) - sum(MEMO.values()))}
    print(f"\nB_3(m37; 41) = {B3} against the budget {BUDGET}: "
          f"{'WITHIN' if B3 <= BUDGET else 'ABOVE'}; argmax J = {arg}", flush=True)
    with open(os.path.join(OUT, "b3.json"), "w") as f:
        json.dump(report, f, indent=1)
    with open(os.path.join(OUT, "memo.json"), "w") as f:
        json.dump({",".join(map(str, k)): v for k, v in MEMO.items()}, f)


if __name__ == "__main__":
    main()
