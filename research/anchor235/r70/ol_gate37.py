"""ol_gate37.py -- gates for the pattern instrument AT m37 itself, where nothing can be scanned.

G3 (m29, the search solver): realised_word_search is put to the same complete test as the
enumeration solver -- every triple of D_3(m29) (from the closure, exact) must come back realised,
and a sample of the unrealised triples of the same grid must not.

G4 (m37, Q*_4): the record law says F(41) = max_{J <= 4} Q*_J(m37; 41) = 91, and the closure with
the span-threshold prune records Q*_J = 88, 90, 90, 91 (monotone_functional.md 2.2).  Q*_4 = 91 is
the span of a REALISED 4-window of m37 that fuses at a phase of 41.  The instrument is asked for
it directly, with no dictionary of m37 deeper than 2 in hand: enumerate every level-2 admissible
fusing 4-word in descending span and test the 4-window itself for membership in D_4(m37).  The
first hit must be 91.

Usage: uv run python research/anchor235/r70/ol_gate37.py [nsample]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "r66")))

from mo_core import base_gaps, closure_step, dict_from_gaps  # noqa: E402
from ol_pattern import realised_word_search  # noqa: E402
from ol_b3 import enumerate_words  # noqa: E402

OUT = os.path.join(HERE, "results")
GEARS37 = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]


def main():
    nsample = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    rng = np.random.default_rng(20260911)
    rep = {}

    # ---- G3: m29, the search solver against the closure's D_3(m29)
    P, g23 = base_gaps(23)
    win, mult = dict_from_gaps(g23, 9)
    w2, m2, st = closure_step(win, mult, 29, m=3, mode="fixed")
    D3 = set(map(tuple, w2[:, :3].astype(int)))
    v29 = sorted(set(int(v) for v in np.unique(w2[:, 0])))
    print(f"m29: |D_3| = {len(D3):,} (mass {int(m2.sum()):,}, loss {st['loss']})", flush=True)
    G29 = [5, 7, 11, 13, 17, 19, 23, 29]
    t0 = time.time()
    missed = [t for t in sorted(D3) if not realised_word_search(t, G29)]
    grid = [(a, b, c) for a in v29 for b in v29 for c in v29]
    non = [t for t in grid if t not in D3]
    idx = rng.choice(len(non), size=min(nsample, len(non)), replace=False)
    fp = [non[int(i)] for i in idx if realised_word_search(non[int(i)], G29)]
    print(f"G3 m29 (search solver): {len(D3):,} realised -> {len(missed)} missed; "
          f"{len(idx):,} unrealised sampled -> {len(fp)} false positives "
          f"[{time.time()-t0:.1f}s]", flush=True)
    rep["G3_m29_search"] = {"D3": len(D3), "grid": len(grid), "missed": len(missed),
                            "sampled": int(len(idx)), "false_positives": len(fp),
                            "secs": round(time.time() - t0, 1)}

    # ---- G4: Q*_4(m37; 41) by the instrument alone
    z = np.load(os.path.join(OUT, "m37_dict.npz"))
    w37 = z["win"]
    vals = sorted(int(v) for v in np.unique(w37[:, 0]))
    pairs = set(map(tuple, w37[:, :2].astype(int)))
    words = enumerate_words(4, vals, pairs)
    order = sorted(words, key=lambda w: -sum(w))
    print(f"m37: {len(words):,} level-2 admissible fusing 4-words, top span {sum(order[0])}",
          flush=True)
    t0 = time.time()
    hit, tested = None, 0
    for w in order:
        tested += 1
        if realised_word_search(w, GEARS37):
            hit = w
            break
    print(f"G4 m37: Q*_4 = {sum(hit)} (corpus/theta-ladder 91) word {list(hit)} at phase "
          f"{words[hit]}; {tested:,} words tested [{time.time()-t0:.1f}s]", flush=True)
    rep["G4_m37_Qstar4"] = {"Qstar4": sum(hit), "expected": 91, "gate": sum(hit) == 91,
                            "word": list(hit), "phase": words[hit], "words_tested": tested,
                            "secs": round(time.time() - t0, 1)}
    with open(os.path.join(OUT, "gate37.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1), flush=True)


if __name__ == "__main__":
    main()
