"""The exact effect of an anchoring gear on the record: the rescaling law.

If a gear p leaves exactly ONE open residue class (p = 2 or p = 3), the open pairs all lie in
that class, so the machine on the remaining gears lives on the sub-lattice.  Predicted:
F(G + p) = p * F_p(G) + (p - 1), where F_p(G) is the record of G in the sub-lattice
coordinate, i.e. of the machine with teeth {0, -2 p^{-1}} mod g.  For p = 6 (gears 2 and 3
together) that coordinate is the bottom machine's column coordinate.
"""

import json
import sys
from math import prod

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])


def mask_with_teeth(gears, sep_inv):
    """Open mask of the machine with teeth {0, -2 * sep_inv} mod g."""
    W = prod(gears)
    m = np.ones(W, dtype=bool)
    idx = np.arange(W)
    for g in gears:
        t2 = (-2 * pow(sep_inv % g, -1, g)) % g if sep_inv % g else 0
        for t in {0, t2}:
            m &= idx % g != t
    return m


def record_of(mask):
    W = len(mask)
    struck = ~mask
    acc = np.ones(W, dtype=bool)
    j = 0
    while True:
        acc &= np.roll(struck, -j)
        if not acc.any():
            return j
        j += 1


def main():
    out = []
    BASE = [[5], [5, 7], [5, 7, 11], [5, 7, 11, 13], [5, 7, 11, 13, 17], [7, 11], [7, 11, 13],
            [11, 13], [11, 13, 17], [7, 11, 13, 17]]
    for G in BASE:
        row = {"G": G}
        # p = 2
        m2 = np.ones(prod([2] + G), dtype=bool)
        idx = np.arange(len(m2))
        for g in [2] + G:
            for t in {0, (-2) % g}:
                m2 &= idx % g != t
        F2 = record_of(m2)
        Fsub2 = record_of(mask_with_teeth(G, 2))
        row["F(2+G)"] = F2
        row["2*F_2+1"] = 2 * Fsub2 + 1
        row["F_2"] = Fsub2
        # p = 3
        m3 = np.ones(prod([3] + G), dtype=bool)
        idx = np.arange(len(m3))
        for g in [3] + G:
            for t in {0, (-2) % g}:
                m3 &= idx % g != t
        F3 = record_of(m3)
        Fsub3 = record_of(mask_with_teeth(G, 3))
        row["F(3+G)"] = F3
        row["3*F_3+2"] = 3 * Fsub3 + 2
        row["F_3"] = Fsub3
        # p = 6 (gears 2 and 3): the column coordinate
        W6 = prod([2, 3] + G)
        m6 = np.ones(W6, dtype=bool)
        idx = np.arange(W6)
        for g in [2, 3] + G:
            for t in {0, (-2) % g}:
                m6 &= idx % g != t
        F6 = record_of(m6)
        Fsub6 = record_of(mask_with_teeth(G, 6))
        row["F(2,3+G)"] = F6
        row["6*F_col+5"] = 6 * Fsub6 + 5
        row["F_col"] = Fsub6
        row["ok"] = (row["F(2+G)"] == row["2*F_2+1"] and row["F(3+G)"] == row["3*F_3+2"]
                     and row["F(2,3+G)"] == row["6*F_col+5"])
        out.append(row)
        print(json.dumps(row), flush=True)
    print("all ok:", all(r["ok"] for r in out))
    with open(__file__.rsplit("s7_")[0] + "results/s7_fold.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
