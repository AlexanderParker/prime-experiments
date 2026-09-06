"""fc_violators.py -- pin the three recorded budget violators (theory_tree 2f.i) by sweeping the
incoming gear's tooth, then report where their slack profile goes negative.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r56"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fc_family import frontier, shape  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

CASES = [([5, 7, 11, 13, 17, 19], [1, 3, 4, 4, 4], 40, "m17 (1,3,4,4,4) -> +19, recorded F = 40"),
         ([5, 7, 11, 13, 17, 19], [2, 3, 3, 3, 3], 38, "m17 (2,3,3,3,3) -> +19, recorded F = 38"),
         ([5, 7, 11, 13], [1, 1, 5], 25, "m11 (1,1,5) -> +13, recorded F = 25")]


def main():
    lines = []
    W = lines.append
    W("=== the recorded budget violators: which incoming tooth reproduces them, and where "
      "the slack goes negative ===")
    for gears, vs0, Frec, tag in CASES:
        q = gears[-1]
        W(f"\n{tag}")
        for vq in range(1, (q - 1) // 2 + 1):
            o = frontier(gears, list(vs0) + [vq])
            sh = shape(o)
            mark = "  <== recorded" if o["F"] == Frec else ""
            W(f"  v_{q} = {vq}: F_old = {o['Fold']}, F = {o['F']}, budget = "
              f"{o['Fold']+q}, slack = {o['budget']}, s<0 at a = {o['viol']} "
              f"(a/F_old = {[round(a/o['Fold'],3) for a in o['viol']]}), "
              f"a* = {sh['amin']} ({sh['frac']:.3f}){mark}")
            if o["F"] == Frec:
                W("     slack profile: " +
                  " ".join(f"{a}:{o['s'][a]}" for a in o["realised"]))
    txt = "\n".join(lines)
    open(os.path.join(OUT, "fc_violators.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
