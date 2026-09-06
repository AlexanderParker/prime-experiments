"""pa_family.py -- WHY the family breaks the pinned letter: coherence, and the repair experiment.

A family member gives gear g teeth at +-v_g with v_g uniform in 1..(g-1)/2 (the same 20 members
per rung as r62/pl_family.py, rng seed 20260906, so the tables compare member by member).  For the
member the letter is a_L = 2 v_{q'}, still the incoming gear's own tooth distance, so the CLOSER
law survives; what does not survive is the identity

    (I1)   a_L = (3 a_L) d_g (mod g)          d_g := 2 v_g,

which holds at gear g iff 3 d_g = 1 (mod g) iff 6 v_g = +-1 (mod g) iff v_g = min(6^{-1}, g-6^{-1}).
Call such a gear COHERENT.  The real machine has every gear coherent; that is the whole of the
real-teeth input.  Measured here:

  * the coherence count of every member, against whether it obeys 0 <= E(a_L) <= 3;
  * the tooth-unit vector mu_g(a_L) = a_L d_g^{-1} (mod g), which is the single integer 3 a_L for
    the real machine and a scattered vector for a member;
  * THE REPAIR EXPERIMENT: for every violating member, put ONE gear back on its real tooth and
    recompute -- which gear's misplacement is carrying the violation.

Outputs results/pa_family.txt / .json.
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


def unorm(g):
    w = u_of(g)
    return min(w, g - w)


def profile(size):
    mult = np.bincount(size.astype(np.int64))
    realised = np.flatnonzero(mult).tolist()
    F = max(realised)
    lft, rgt = size[:-1], size[1:]
    r = {}
    for v in realised:
        s = set(np.unique(rgt[lft == v]).tolist()) | set(np.unique(lft[rgt == v]).tolist())
        r[v] = max(s) if s else 0
    return realised, F, r


def measure(gears, vs):
    """(F, a_L, r(a_L), E) for one tooth assignment."""
    L = build_levels(gears[:-1], vs[:-1])
    old = L[-1]
    q = gears[-1]
    aL, bL = letters(q, vs[-1])
    realised, F, r = profile(old.size)
    ra = r.get(aL, 0)
    E = (aL + ra - F) if ra else None
    del L, old
    return F, aL, ra, E


def main():
    t0 = time.time()
    Lo = []
    W = Lo.append
    W("=== THE FAMILY: COHERENCE AND THE REPAIR EXPERIMENT ===")
    rng = random.Random(20260906)
    setups = [([5, 7, 11, 13, 17], "13->17"), ([5, 7, 11, 13, 17, 19], "17->19"),
              ([5, 7, 11, 13, 17, 19, 23], "19->23")]
    res = {}
    allrows = []
    for gears, name in setups:
        M = gears[:-1]
        real = [unorm(g) for g in gears]
        W(f"\n--- rung {name}: machine {M}, incoming {gears[-1]}; real teeth {real} ---")
        vsets = [("REAL", real)]
        seen = set()
        while len(vsets) < 21:
            vs = [rng.randrange(1, (g - 1) // 2 + 1) for g in gears]
            if tuple(vs) in seen or vs == real:
                continue
            seen.add(tuple(vs))
            vsets.append((f"m{len(vsets)}", vs))
        W("member | teeth | coherent gears of M | mu_g(a_L) over M | 3a_L | F | a_L | r | E | ok")
        rows = []
        for tag, vs in vsets:
            F, aL, ra, E = measure(gears, vs)
            coh = [g for g, v in zip(M, vs[:-1]) if v == unorm(g)]
            mu = []
            for g, v in zip(M, vs[:-1]):
                d = (2 * v) % g
                mu.append((aL * pow(d, -1, g)) % g)
            ok = (E is not None and 0 <= E <= 3)
            W(f"{tag:<6} | {vs} | {coh or '-'} ({len(coh)}) | {mu} | {3*aL} | {F} | {aL} | {ra} | "
              f"{'-' if E is None else f'{E:+d}'} | {'yes' if ok else ('unreal' if E is None else 'NO')}")
            rows.append(dict(tag=tag, vs=vs, coh=coh, ncoh=len(coh), mu=mu, F=F, aL=aL,
                             r=ra, E=E, ok=bool(ok)))
            allrows.append(dict(rung=name, **rows[-1]))
        res[name] = rows
        W(f"  [{time.time()-t0:.1f}s]")

    # ---------- coherence count vs obedience ----------
    W("\n--- coherence count vs the pinned bound, over all 60 members (real excluded) ---")
    W("coherent gears | members | 0<=E<=3 | E>3 | E<0 | a_L unrealised")
    fam = [r for r in allrows if r["tag"] != "REAL"]
    for k in range(0, 5):
        sub = [r for r in fam if r["ncoh"] == k]
        if not sub:
            continue
        W(f"{k:>14} | {len(sub):>7} | {sum(1 for r in sub if r['ok']):>7} | "
          f"{sum(1 for r in sub if r['E'] is not None and r['E'] > 3):>3} | "
          f"{sum(1 for r in sub if r['E'] is not None and r['E'] < 0):>3} | "
          f"{sum(1 for r in sub if r['E'] is None):>14}")
    W(f"  full coherence (every gear of M coherent): "
      f"{sum(1 for r in fam if r['ncoh'] == len(r['mu']))} of {len(fam)} members")
    W("  the tooth-unit vector mu_g(a_L) is the single integer 3a_L (mod g) at every gear for the "
      "REAL rows and scattered for every member")

    # ---------- the repair experiment ----------
    W("\n--- THE REPAIR EXPERIMENT: one gear back on its real tooth ---")
    W("rung | member | E before | gear repaired | teeth after | F | a_L | r | E after | repaired?")
    rep = []
    for gears, name in setups:
        M = gears[:-1]
        for row in res[name]:
            if row["tag"] == "REAL" or row["E"] is None or 0 <= row["E"] <= 3:
                continue
            fixes = []
            for i, g in enumerate(M):
                if row["vs"][i] == unorm(g):
                    continue
                vs2 = list(row["vs"])
                vs2[i] = unorm(g)
                F, aL, ra, E = measure(gears, vs2)
                good = (E is not None and 0 <= E <= 3)
                W(f"{name} | {row['tag']:<4} | {row['E']:+d} | {g:>2} | {vs2} | {F} | {aL} | "
                  f"{ra} | {'-' if E is None else f'{E:+d}'} | {'YES' if good else 'no'}")
                fixes.append(dict(gear=g, F=F, aL=aL, r=ra, E=E, good=bool(good)))
            rep.append(dict(rung=name, tag=row["tag"], E0=row["E"], vs=row["vs"], fixes=fixes))
    nviol = len(rep)
    nsingle = sum(1 for r in rep if any(f["good"] for f in r["fixes"]))
    W(f"\n  violating members (E outside [0,3]): {nviol}")
    W(f"  repaired by a SINGLE gear put back on its real tooth: {nsingle} of {nviol}")
    from collections import Counter
    cnt = Counter()
    for r in rep:
        for f in r["fixes"]:
            if f["good"]:
                cnt[f["gear"]] += 1
    W(f"  which gear repairs, counted over all successful single-gear repairs: {dict(sorted(cnt.items()))}")
    solo = Counter()
    for r in rep:
        gs = [f["gear"] for f in r["fixes"] if f["good"]]
        if len(gs) == 1:
            solo[gs[0]] += 1
    W(f"  members with exactly ONE repairing gear: {dict(sorted(solo.items()))}")

    json.dump(dict(rows=allrows, repair=rep), open(os.path.join(OUT, "pa_family.json"), "w"))
    txt = "\n".join(Lo)
    open(os.path.join(OUT, "pa_family.txt"), "w").write(txt)
    print(f"wrote {OUT}/pa_family.txt ({len(txt)} chars, {time.time()-t0:.1f}s)")
    print(txt[-3000:])


if __name__ == "__main__":
    main()
