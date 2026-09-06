"""pl_letter_glue.py -- the one-gear glue at the LETTER only, with a large occurrence sample, and
the distribution of the achieved gap over occurrences.  Also the same for the sizes that break the
pinned bound, for contrast.

Answers: does some occurrence of the attaining (r(a_L), a_L) pair glue to within 3 of its span?
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "r56"))
from mf_core import build_levels, u_of                    # noqa: E402
from pl_glue import glue1, occurrences, letters           # noqa: E402

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)


def main():
    t0 = time.time()
    lines = []
    W = lines.append
    W("=== the one-gear glue at the letter, large samples ===")
    W("machine | v | a = r(v) | S | E = S-F | occurrences sampled | best Glue1 | loss | "
      "closing gear h | share of occurrences reaching the best")
    L = build_levels()
    res = {}
    CAP = 600
    for n, qn in ((2, 13), (3, 17), (4, 19), (5, 23), (6, 29)):
        lv = L[n]
        gears = list(lv.gears)
        us = [u_of(g) for g in gears]
        F = int(lv.size.max())
        aL, bL = letters(qn)
        lft, rgt = lv.size[:-1], lv.size[1:]
        # sizes to examine: the letter, the long letter, and the two worst offenders
        mult = np.bincount(lv.size.astype(np.int64))
        realised = np.flatnonzero(mult).tolist()
        r = {}
        for v in realised:
            s1 = set(np.unique(rgt[lft == v]).tolist()) | set(np.unique(lft[rgt == v]).tolist())
            r[v] = max(s1) if s1 else 0
        worst = sorted(realised, key=lambda v: -(v + r[v]))[:3]
        todo = [("a_L", aL), ("b_L", bL)] + [(f"worst{k}", v) for k, v in enumerate(worst)]
        for tag, v in todo:
            if v not in r or r[v] == 0:
                W(f"m{gears[-1]} | {tag} v={v} | NOT REALISED")
                continue
            a = r[v]
            S = a + v
            occ = occurrences(lv.size, a, v, CAP)
            vals = []
            bh = {}
            for i in occ:
                i = int(i)
                g, h, s = glue1(gears, us, int(lv.O[i]), int(lv.size[i]), int(lv.size[i + 1]), F)
                vals.append(g)
                if g == max(vals):
                    bh[g] = (h, s)
            best = max(vals)
            share = sum(1 for x in vals if x == best) / len(vals)
            W(f"m{gears[-1]} | {tag} v={v} | a={a} | S={S} | E={S-F:+d} | "
              f"{len(vals)} of {int(((lft==a)&(rgt==v)).sum() + ((lft==v)&(rgt==a)).sum())} | "
              f"{best} | {S-best} | h={bh[best][0]} s={bh[best][1]} | {share:.3f} | "
              f"F={F} tight={best==F}")
            res[f"{gears[-1]}:{tag}"] = dict(v=v, a=a, S=S, E=S - F, best=int(best),
                                             loss=int(S - best), F=F, h=bh[best][0],
                                             share=share, n=len(vals))
        W(f"  [{time.time()-t0:.1f}s]")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "pl_letter_glue.txt"), "w").write(txt)
    json.dump(res, open(os.path.join(OUT, "pl_letter_glue.json"), "w"))
    print(txt)


if __name__ == "__main__":
    main()
