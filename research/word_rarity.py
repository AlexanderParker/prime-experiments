"""Round 17 (mechanic): the occurrence-count law for compatible words.

(D)'s residual is carried by RARE long words, so the object the route
actually needs is a bound on N(w) = the number of occurrences of a
compatible word.  This compares N(w) with the letter-independent
prediction  N * prod_i P(gap = w_i)  built from the machine's own gap
histogram, over every (machine, word) the envelope census has measured.

Usage: uv run python research/word_rarity.py
"""
import csv
import os
from collections import defaultdict

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def main():
    # NOTE: the gaphist CSV is appended once per (y, q') run, so a machine
    # probed n times carries n identical copies.  Deduplicate by (y, gap):
    # every copy is identical, so keep the first and ignore repeats.
    gh = defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(D,
                                              "flank_envelope_gaphist.csv"))):
        y, g, c = int(r["y"]), int(r["gap"]), int(r["count"])
        if g not in gh[y]:
            gh[y][g] = c
    N = {y: sum(v.values()) for y, v in gh.items()}

    print("machine  q'    word          ell   occ          indep-pred"
          "      obs/pred")
    rows = list(csv.DictReader(open(os.path.join(
        D, "flank_envelope_words.csv"))))
    for r in rows:
        y, occ = int(r["y"]), int(r["occ"])
        if y not in gh or not int(r["compat"]):
            continue
        w = [int(x) for x in r["word"].split("-")]
        p = 1.0
        for L in w:
            p *= gh[y].get(L, 0) / N[y]
        pred = N[y] * p
        if pred == 0:
            continue
        print(f"  {y:3d}   {r['qp']:>3s}  {r['word']:>12s}  {len(w)}  "
              f"{occ:>12,}  {pred:14.2f}  {occ/pred:10.3f}")


if __name__ == "__main__":
    main()
