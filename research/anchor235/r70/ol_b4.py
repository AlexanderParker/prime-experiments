"""ol_b4.py -- the one relaxed term of B_3(m37; 41): J = J_max = 4 at level 3.

A level-3 admissible 4-word (g1, g2, g3, g4) is one whose two 3-subwindows (g1, g2, g3) and
(g2, g3, g4) both lie in D_3(m37); the fusion condition is that the three interior offsets are
struck by 41 at a common phase and the two ends are not.  Every such word is enumerated (all 41
phases, all 75 realised gap values in each slot, the adjacent pairs filtered exactly through
D_2(m37)), sorted by DESCENDING span, and its two triples put to the pattern instrument, so the
first word that passes is the maximum.  Membership verdicts are memoised on disk.

Usage: uv run python research/anchor235/r70/ol_b4.py
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
from ol_b3 import enumerate_words, GEARS, Q, BUDGET  # noqa: E402

OUT = os.path.join(HERE, "results")
MEMOF = os.path.join(OUT, "memo4.json")


def main():
    memo = {}
    if os.path.exists(MEMOF):
        memo = {tuple(int(x) for x in k.split(",")): v
                for k, v in json.load(open(MEMOF)).items()}
    stats = {"calls": 0, "nodes": 0, "fallback": 0, "secs": 0.0}

    def is_realised(t):
        t = tuple(int(x) for x in t)
        if t not in memo:
            t0 = time.time()
            c = {}
            try:
                v = bool(realised_word_search(t, GEARS, counter=c))
            except RuntimeError:
                stats["fallback"] += 1
                v = bool(realised_word(t, GEARS))
            memo[t] = v
            stats["calls"] += 1
            stats["nodes"] += c.get("nodes", 0)
            stats["secs"] += time.time() - t0
        return memo[t]

    z = np.load(os.path.join(OUT, "m37_dict.npz"))
    win = z["win"]
    vals = sorted(int(v) for v in np.unique(win[:, 0]))
    pairs = set(map(tuple, win[:, :2].astype(int)))
    words = enumerate_words(4, vals, pairs)
    order = sorted(words, key=lambda w: -sum(w))
    print(f"m37 -> {Q}: {len(words):,} level-2 admissible fusing 4-words, spans "
          f"{sum(order[0])} down to {sum(order[-1])}; budget {BUDGET}", flush=True)
    t0 = time.time()
    found, ties, tested = None, [], 0
    for w in order:
        if found is not None and sum(w) < sum(found):
            break
        tested += 1
        if is_realised(w[:3]) and is_realised(w[1:]):
            if found is None:
                found = w
                print(f"  HIT at span {sum(w)}: {list(w)} phase {words[w]} "
                      f"[{tested:,} words, {stats['calls']:,} calls, "
                      f"{time.time()-t0:.0f}s]", flush=True)
            ties.append(list(w))
            continue
        if tested % 200 == 0:
            print(f"  ... {tested:,} words tested, now at span {sum(w)}, "
                  f"{stats['calls']:,} membership calls, {time.time()-t0:.0f}s", flush=True)
            with open(MEMOF, "w") as f:
                json.dump({",".join(map(str, k)): v for k, v in memo.items()}, f)
    rep = {"term_J4_level3": sum(found) if found else None,
           "word": list(found) if found else None,
           "phase": words[found] if found else None,
           "ties": ties, "candidates": len(words), "words_tested": tested,
           "triples_tested": len(memo),
           "triples_realised": int(sum(1 for v in memo.values() if v)),
           "stats": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in stats.items()},
           "secs": round(time.time() - t0, 1)}
    print(json.dumps(rep, indent=1), flush=True)
    with open(os.path.join(OUT, "b4.json"), "w") as f:
        json.dump(rep, f, indent=1)
    with open(MEMOF, "w") as f:
        json.dump({",".join(map(str, k)): v for k, v in memo.items()}, f)


if __name__ == "__main__":
    main()
