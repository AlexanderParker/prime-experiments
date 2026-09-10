"""u45_j5band.py -- the J = 5 outer-law TEST at m37 (is anything realised with outer > 90?),
run as a banded scan with a low node budget so that one pathological word cannot stall it.

Every candidate 5-word with outer sum above the floor (all three 3-subwindows in D_3(m37), all
middles >= 6) is put to the covering solver with a small node budget.  A word that exceeds the
budget is UNDECIDED and is set aside; everything else is decided exactly.  The law's test is
settled if no word comes back realised and the undecided set is then cleared one by one.

Usage: uv run python research/anchor235/r72/u45_j5band.py [floor] [budget]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "r70")))

from ol_pattern import realised_search  # noqa: E402

GEARS = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]


def offs(word):
    o = [0]
    for g in word:
        o.append(o[-1] + g)
    return o


def main():
    floor0 = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 150_000
    z = np.load(os.path.join(OUT, "m37_D3.npz"))
    full = [tuple(int(x) for x in r) for r in z["win"]]
    nxt3 = {}
    for a, b, c in full:
        nxt3.setdefault((a, b), []).append(c)
    words = []
    for (a, u, v) in full:
        if u < 6 or v < 6:
            continue
        for w in nxt3.get((u, v), ()):
            if w < 6:
                continue
            for b in nxt3.get((v, w), ()):
                if a + b > floor0:
                    words.append((a, u, v, w, b))
    words = sorted(set(words), key=lambda t: -(t[0] + t[-1]))
    print(f"J = 5 at m37, outer > {floor0}: {len(words):,} candidate words, node budget {budget:,}",
          flush=True)
    memo = {}
    undecided = []
    realised = []
    t0 = time.time()
    for i, w in enumerate(words):
        for sub in (w[:4], w[1:], w):
            key = min(sub, sub[::-1])
            if key in memo:
                v = memo[key]
            else:
                try:
                    v = bool(realised_search(offs(key), GEARS, budget=budget))
                except RuntimeError:
                    v = None
                memo[key] = v
            if v is False:
                break
            if v is None:
                undecided.append(list(key))
                break
        else:
            realised.append(list(w))
            print(f"  REALISED {w} outer {w[0]+w[-1]} span {sum(w)}", flush=True)
        if (i + 1) % 5000 == 0:
            print(f"  ... {i+1:,}/{len(words):,} at outer {w[0]+w[-1]}, "
                  f"{len(memo):,} decided, {len(undecided):,} undecided, "
                  f"{time.time()-t0:.0f}s", flush=True)
    rep = {"floor": floor0, "budget": budget, "words": len(words),
           "decided": len(memo), "undecided": len(undecided),
           "undecided_words": undecided[:200], "realised": realised,
           "secs": round(time.time() - t0, 1)}
    print(json.dumps({k: rep[k] for k in
                      ("floor", "budget", "words", "decided", "undecided", "realised", "secs")},
                     indent=1), flush=True)
    with open(os.path.join(OUT, "j5band.json"), "w") as f:
        json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
