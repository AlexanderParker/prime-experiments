"""pl_family.py -- the pinned letter on the tooth-counterfactual family.

Family member = the same gears with teeth at +-v_g, v_g uniform in 1..(g-1)/2 (alignment-rules
section 5); the SAME 20 members as r57/fc_family.py and r58/ag_family.py (rng seed 20260906), so
the tables compare member by member.

For each member at rungs 13->17, 17->19, 19->23:  F, F_2, the member's own letters a_L = 2 v_{q'},
r(a_L), E(a_L) = a_L + r(a_L) - F, and the excess profile E(v) over all realised sizes.

The question: is `0 <= E(a_L) <= 3` teeth-free (a glue-lemma-shaped law, which should survive any
tooth assignment) or real-teeth (an accident of 6^{-1})?

Outputs results/pl_family.txt / .json.
"""
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "r56"))
sys.path.insert(0, os.path.join(HERE, "..", "r58"))
from mf_core import build_levels, u_of                    # noqa: E402
from ag_gate import letters                               # noqa: E402

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)


def profile(size):
    mult = np.bincount(size.astype(np.int64))
    realised = np.flatnonzero(mult).tolist()
    F = max(realised)
    lft, rgt = size[:-1], size[1:]
    r = {}
    for v in realised:
        s1 = set(np.unique(rgt[lft == v]).tolist()) | set(np.unique(lft[rgt == v]).tolist())
        r[v] = max(s1) if s1 else 0
    F2 = max(v + r[v] for v in realised)
    return realised, F, F2, r


def main():
    t0 = time.time()
    lines = []
    W = lines.append
    W("=== THE PINNED LETTER ON THE TOOTH-COUNTERFACTUAL FAMILY ===")
    W("member | teeth v | F | F_2 | a_L | r(a_L) | E(a_L) | 0<=E<=3? | #sizes | #E>3 | max E")
    rng = random.Random(20260906)
    setups = [([5, 7, 11, 13, 17], "13->17"), ([5, 7, 11, 13, 17, 19], "17->19"),
              ([5, 7, 11, 13, 17, 19, 23], "19->23")]
    res = {}
    for gears, name in setups:
        real = [min(u_of(g), g - u_of(g)) for g in gears]
        W(f"\n--- rung {name} (gears {gears}); real teeth v = {real} ---")
        vsets = [("REAL", real)]
        seen = set()
        while len(vsets) < 21:
            vs = [rng.randrange(1, (g - 1) // 2 + 1) for g in gears]
            if tuple(vs) in seen or vs == real:
                continue
            seen.add(tuple(vs))
            vsets.append((f"m{len(vsets)}", vs))
        rows = []
        for tag, vs in vsets:
            L = build_levels(gears[:-1], vs[:-1])
            old = L[-1]
            q = gears[-1]
            aL, bL = letters(q, vs[-1])
            realised, F, F2, r = profile(old.size)
            ra = r.get(aL, 0)
            E = aL + ra - F if ra else None
            Es = {v: v + r[v] - F for v in realised}
            nbad = sum(1 for v in realised if Es[v] > 3)
            ok = (E is not None and 0 <= E <= 3)
            W(f"{tag} | {vs} | {F} | {F2} | {aL} | {ra} | "
              f"{'-' if E is None else f'{E:+d}'} | "
              f"{'yes' if ok else ('unrealised' if E is None else 'NO')} | {len(realised)} | "
              f"{nbad} | {max(Es.values()):+d}")
            rows.append(dict(tag=tag, vs=vs, F=F, F2=F2, aL=aL, bL=bL, r=ra, E=E,
                             ok=bool(ok), n=len(realised), nbad=nbad,
                             maxE=int(max(Es.values())),
                             E_all={int(k): int(v) for k, v in Es.items()}))
            del L, old
        fam = [x for x in rows if x["tag"] != "REAL"]
        good = [x for x in fam if x["ok"]]
        unre = [x for x in fam if x["E"] is None]
        hi = [x for x in fam if x["E"] is not None and x["E"] > 3]
        lo = [x for x in fam if x["E"] is not None and x["E"] < 0]
        W(f"  members with 0 <= E(a_L) <= 3: {len(good)} of 20 "
          f"(a_L unrealised at {len(unre)}); E > 3 at {len(hi)} "
          f"({[(x['tag'], x['aL'], x['E']) for x in hi]}); E < 0 at {len(lo)} "
          f"({[(x['tag'], x['aL'], x['E']) for x in lo]})")
        Ev = [x["E"] for x in fam if x["E"] is not None]
        if Ev:
            W(f"  family E(a_L): min {min(Ev):+d}, max {max(Ev):+d}, "
              f"values {sorted(Ev)}; REAL {rows[0]['E']:+d}")
        res[name] = rows
        W(f"  [{time.time()-t0:.1f}s]")

    # the constant across the whole family
    allE = [x["E"] for rs in res.values() for x in rs if x["E"] is not None]
    W(f"\n=== all 63 members (3 rungs x (20 + real)) ===")
    W(f"  E(a_L) over every member with a_L realised: n = {len(allE)}, "
      f"min {min(allE):+d}, max {max(allE):+d}, "
      f"0 <= E <= 3 at {sum(1 for e in allE if 0 <= e <= 3)} of {len(allE)}")
    hist = {}
    for e in allE:
        hist[e] = hist.get(e, 0) + 1
    W("  histogram of E(a_L): " + " ".join(f"{k:+d}:{hist[k]}" for k in sorted(hist)))
    json.dump(res, open(os.path.join(OUT, "pl_family.json"), "w"))
    txt = "\n".join(lines)
    open(os.path.join(OUT, "pl_family.txt"), "w").write(txt)
    print(txt[-2500:])


if __name__ == "__main__":
    main()
