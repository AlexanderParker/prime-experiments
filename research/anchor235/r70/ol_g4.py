"""ol_g4.py -- gate G4: Q*_4(m37; 41) from the pattern instrument alone.

The record law gives F(41) = max_{J <= 4} Q*_J(m37; 41), and the closure with the span-threshold
prune records Q*_J = 88, 90, 90, 91 (monotone_functional.md 2.2, r66 results/theta_K21_t89.json).
Q*_4 = 91 is therefore the span of a REALISED 4-window of m37 that fuses at some phase of 41.  The
instrument is asked for it with no m37 dictionary deeper than 2 in hand: every level-2 admissible
fusing 4-word in descending span, tested for membership in D_4(m37).  The first hit must be 91 --
which is at once a gate on the instrument and an independent recomputation of the record F(41).

Usage: uv run python research/anchor235/r70/ol_g4.py
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
from ol_b3 import enumerate_words, GEARS, Q  # noqa: E402

OUT = os.path.join(HERE, "results")


def main():
    z = np.load(os.path.join(OUT, "m37_dict.npz"))
    win = z["win"]
    vals = sorted(int(v) for v in np.unique(win[:, 0]))
    pairs = set(map(tuple, win[:, :2].astype(int)))
    words = enumerate_words(4, vals, pairs)
    order = sorted(words, key=lambda w: -sum(w))
    print(f"m37 -> {Q}: {len(words):,} level-2 admissible fusing 4-words, top span "
          f"{sum(order[0])}", flush=True)
    t0 = time.time()
    hit, tested, fb = None, 0, 0
    for w in order:
        tested += 1
        try:
            r = realised_word_search(w, GEARS)
        except RuntimeError:
            fb += 1
            r = realised_word(w, GEARS)
        if r:
            hit = w
            break
        if tested % 200 == 0:
            print(f"  ... {tested:,} words, at span {sum(w)}, {time.time()-t0:.0f}s", flush=True)
    rep = {"Qstar4": sum(hit), "expected": 91, "gate": sum(hit) == 91, "word": list(hit),
           "phase": words[hit], "words_tested": tested, "fallbacks": fb,
           "secs": round(time.time() - t0, 1)}
    print(json.dumps(rep, indent=1), flush=True)
    with open(os.path.join(OUT, "g4.json"), "w") as f:
        json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
