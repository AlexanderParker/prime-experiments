"""ol_b2.py -- B_1 and B_2 of m37 against the gear 41, off the depth-2 dictionary the ladder left.

The relaxation and its dynamic programme are r66/mo_order.py, imported unchanged, so the B_2
recomputed here is the same object as the 161 recorded in monotone_functional.md 4.2.

Usage: uv run python research/anchor235/r70/ol_b2.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "r66")))

from mo_order import dicts_from_windows, exact_J, relaxed_J  # noqa: E402

OUT = os.path.join(HERE, "results")

Q = 41
JMAX = 4          # L(m37) = 2, J_max = L + 2
F37 = 88
BUDGET = F37 + Q


def main():
    z = np.load(os.path.join(OUT, "m37_dict.npz"))
    win, mult = z["win"], z["mult"]
    print(f"D_{win.shape[1]}(m37): {win.shape[0]:,} rows, mass {int(mult.sum()):,}", flush=True)
    Ds = dicts_from_windows(win, 2)
    sizes = [int(D.shape[0]) for D in Ds]
    print(f"|D_1(m37)| = {sizes[0]}, |D_2(m37)| = {sizes[1]}", flush=True)
    out = {"dict_sizes": sizes, "budget": BUDGET, "J_max": JMAX,
           "D1": [int(v) for v in Ds[0][:, 0]]}
    for k in (1, 2):
        per = {}
        for J in range(1, JMAX + 1):
            if J <= k:
                v, arg = exact_J(Ds[J - 1], Q, J)
                per[J] = {"span": v, "word": arg, "exact": True}
            else:
                per[J] = {"span": relaxed_J(Ds[k - 1], Ds[k - 2] if k > 1 else None, Q, J, k),
                          "exact": False}
        B = max(p["span"] for p in per.values() if p["span"] is not None)
        out[f"B_{k}"] = B
        out[f"per_J_{k}"] = {str(J): per[J] for J in per}
        print(f"k = {k}: per-J {[per[J]['span'] for J in per]}  ->  B_{k} = {B} "
              f"(budget {BUDGET})", flush=True)
    with open(os.path.join(OUT, "b2.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
