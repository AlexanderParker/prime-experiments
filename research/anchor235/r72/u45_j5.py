"""u45_j5.py -- the J = 5 term of the outer law at m37, alone (the J = 4 scan runs beside it).

Same descending scan as u45_outer37.py: candidates are the 5-words all of whose 3-subwindows lie
in D_3(m37) -- a complete superset of the realised 5-windows -- ordered by descending outer sum,
each decided by the covering instrument, first realised word is the exact maximum.

Usage: uv run python research/anchor235/r72/u45_j5.py [floor]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "r70")))

from ol_pattern import realised_word, realised_word_search  # noqa: E402

GEARS = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
MEMO = {}
CALLS = {"n": 0, "secs": 0.0, "fallback": 0}


def is_realised(word):
    key = min(word, word[::-1])
    if key not in MEMO:
        t0 = time.time()
        try:
            v = bool(realised_word_search(key, GEARS))
        except RuntimeError:
            CALLS["fallback"] += 1
            v = bool(realised_word(key, GEARS))
        MEMO[key] = v
        CALLS["n"] += 1
        CALLS["secs"] += time.time() - t0
    return MEMO[key]


def main():
    floor0 = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    global OUTNAME
    OUTNAME = sys.argv[2] if len(sys.argv) > 2 else "j5.json"
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
                words.append((a, u, v, w, b))
    words = sorted(set(words), key=lambda t: -(t[0] + t[-1]))
    print(f"J = 5 at m37: {len(words):,} candidates (every 3-subwindow in D_3, middles >= 6); "
          f"{sum(1 for x in words if x[0]+x[-1] > floor0):,} above {floor0}", flush=True)
    t0 = time.time()
    n = 0
    for w in words:
        s = w[0] + w[-1]
        if s <= floor0:
            break
        n += 1
        if n % 2000 == 0:
            print(f"  ... {n:,} words, at outer {s}, {CALLS['n']:,} solver calls, "
                  f"{time.time()-t0:.0f}s", flush=True)
        # the two 4-subwindows first (memoised, and each kills many 5-words)
        if not is_realised(w[:4]) or not is_realised(w[1:]):
            continue
        if is_realised(w):
            print(f"FIRST REALISED {w} outer {s} span {sum(w)} after {n:,} words, "
                  f"{time.time()-t0:.0f}s", flush=True)
            with open(os.path.join(OUT, OUTNAME), "w") as f:
                json.dump({"outer": s, "word": list(w), "span": sum(w), "words": n,
                           "calls": CALLS}, f, indent=1)
            return
    print(f"nothing realised above {floor0}: {n:,} words, {CALLS['n']:,} solver calls, "
          f"{time.time()-t0:.0f}s", flush=True)
    with open(os.path.join(OUT, OUTNAME), "w") as f:
        json.dump({"outer": None, "floor": floor0, "words": n, "calls": CALLS}, f, indent=1)


if __name__ == "__main__":
    main()
