"""ol_j3.py -- the exact J = 3 term of B_3(m37; 41), and the slice of it that decides B_3.

At level 3 the terms J <= 3 of the record law are not relaxed at all, so the J = 3 term is the
exact Q*_3(m37; 41): the widest REALISED 3-window of m37 that fuses at a phase of 41.  Since the
J = 4 term is 98, all that B_3 needs from this term is that it does not exceed 98, so the script
first settles every fusing 3-word of span > 98 (the decisive slice) and then, if asked, carries on
down to the first realised one, which is Q*_3 itself.

Usage: uv run python research/anchor235/r70/ol_j3.py [floor]
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


def main():
    floor = int(sys.argv[1]) if len(sys.argv) > 1 else 98
    z = np.load(os.path.join(OUT, "m37_dict.npz"))
    win = z["win"]
    vals = sorted(int(v) for v in np.unique(win[:, 0]))
    pairs = set(map(tuple, win[:, :2].astype(int)))
    words = enumerate_words(3, vals, pairs)
    order = sorted(words, key=lambda w: -sum(w))
    above = [w for w in order if sum(w) > floor]
    print(f"m37 -> {Q}: {len(words):,} level-2 admissible fusing 3-words; {len(above):,} of span "
          f"> {floor} (top {sum(order[0])})", flush=True)
    t0 = time.time()
    hits, fb, nodes = [], 0, 0
    for i, w in enumerate(above):
        c = {}
        try:
            r = realised_word_search(w, GEARS, counter=c)
        except RuntimeError:
            fb += 1
            r = realised_word(w, GEARS)
        nodes += c.get("nodes", 0)
        if r:
            hits.append(list(w))
            print(f"  REALISED {list(w)} span {sum(w)} phase {words[w]}", flush=True)
        if (i + 1) % 100 == 0:
            print(f"  ... {i+1:,}/{len(above):,} at span {sum(w)}, {time.time()-t0:.0f}s",
                  flush=True)
    rep = {"floor": floor, "words": len(words), "words_above": len(above),
           "realised_above": hits, "fallbacks": fb, "nodes": nodes,
           "secs": round(time.time() - t0, 1),
           "Qstar3_at_most": floor if not hits else max(sum(h) for h in hits)}
    print(json.dumps(rep, indent=1), flush=True)
    with open(os.path.join(OUT, "j3.json"), "w") as f:
        json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
