"""pa_extra.py -- two follow-ups.

(A) THE WHOLE TOP OF THE ROW, not just the first cell above it.  For every rung and every
    a in (r(a_L), F]: is the cell killed by the gear-5 pair filter alone (a one-line arithmetic
    proof), or does it need the search?  And, for the cells the filter does not kill, the length
    of the shortest sub-interval of the run that already cannot be covered.

(B) THE REPAIR CONTROL.  The repair experiment moves one gear back to its REAL tooth and asks
    whether the family member's violation disappears.  The honest comparison is a control that
    moves the same gear to a DIFFERENT WRONG tooth: if a random re-tooth repairs as often, the
    repair says nothing about coherence.

Outputs results/pa_extra.txt / .json.
"""
import json
import os
import random
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "r56"))
sys.path.insert(0, os.path.join(HERE, "..", "r58"))
import pa_crt as C                                        # noqa: E402
import pa_crt2 as C2                                      # noqa: E402
from mf_core import build_levels, u_of                    # noqa: E402
from ag_gate import letters                               # noqa: E402

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

RUNGS = [(13, [5, 7, 11], 7, 3),
         (17, [5, 7, 11, 13], 11, 7),
         (19, [5, 7, 11, 13, 17], 18, 12),
         (23, [5, 7, 11, 13, 17, 19], 25, 20),
         (29, [5, 7, 11, 13, 17, 19, 23], 34, 25),
         (31, [5, 7, 11, 13, 17, 19, 23, 29], 43, 35),
         (37, [5, 7, 11, 13, 17, 19, 23, 29, 31], 58, 46)]


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
    L = build_levels(gears[:-1], vs[:-1])
    old = L[-1]
    aL, bL = letters(gears[-1], vs[-1])
    realised, F, r = profile(old.size)
    ra = r.get(aL, 0)
    E = (aL + ra - F) if ra else None
    del L, old
    return F, aL, ra, E


def partA(W):
    W("\n--- (A) the whole top of the row: what kills each cell a in (r(a_L), F] ---")
    W("rung | a_L | cells above the row | killed by the gear-5 pair filter | need the search |"
      " shortest uncoverable window over those cells")
    out = []
    for qn, gears, F, rk in RUNGS:
        aL = 2 * unorm(qn)
        cells = list(range(rk + 1, F + 1))
        filt, hard = [], []
        for a in cells:
            if C2.gear5_dead(gears, aL, a):
                filt.append(a)
            else:
                hard.append(a)
        wins = {}
        if qn <= 31:
            for a in hard:
                w = C2.min_uncoverable_window(gears, aL, a)
                wins[a] = w
        W(f"{qn:>4} | {aL:>3} | {len(cells):>19} | {str(filt) if filt else 'none':<34} | "
          f"{len(hard):>15} | "
          f"{ {a: (w[0] if w else None) for a, w in wins.items()} if wins else '(not computed at m31)'}")
        out.append(dict(q=qn, aL=aL, cells=cells, filt=filt, hard=hard,
                        wins={str(k): v for k, v in wins.items()}))
    W("  the pair filter is machine-independent arithmetic (parent 2.3, cited); the remaining")
    W("  cells are decided only by the covering search, and the shortest window that already")
    W("  fails is a measure of how local the obstruction is.")
    return out


def partB(W):
    W("\n--- (B) the repair control: real tooth vs a different WRONG tooth ---")
    rng = random.Random(20260906)
    setups = [([5, 7, 11, 13, 17], "13->17"), ([5, 7, 11, 13, 17, 19], "17->19"),
              ([5, 7, 11, 13, 17, 19, 23], "19->23")]
    ctrl = random.Random(4242)
    rows = []
    for gears, name in setups:
        M = gears[:-1]
        real = [unorm(g) for g in gears]
        vsets = []
        seen = set()
        while len(vsets) < 20:
            vs = [rng.randrange(1, (g - 1) // 2 + 1) for g in gears]
            if tuple(vs) in seen or vs == real:
                continue
            seen.add(tuple(vs))
            vsets.append((f"m{len(vsets)+1}", vs))
        for tag, vs in vsets:
            F, aL, ra, E = measure(gears, vs)
            if E is None or 0 <= E <= 3:
                continue
            for i, g in enumerate(M):
                if vs[i] == unorm(g):
                    continue
                vr = list(vs)
                vr[i] = unorm(g)
                _, _, _, Er = measure(gears, vr)
                alts = [t for t in range(1, (g - 1) // 2 + 1) if t not in (vs[i], unorm(g))]
                Ec = []
                for t in ctrl.sample(alts, min(3, len(alts))):
                    vc = list(vs)
                    vc[i] = t
                    _, _, _, e2 = measure(gears, vc)
                    Ec.append(e2)
                rows.append(dict(rung=name, tag=tag, gear=g, E0=E, Ereal=Er, Ectrl=Ec))
    nreal = sum(1 for r in rows if r["Ereal"] is not None and 0 <= r["Ereal"] <= 3)
    nc = sum(1 for r in rows for e in r["Ectrl"] if e is not None and 0 <= e <= 3)
    tc = sum(len(r["Ectrl"]) for r in rows)
    W(f"  single-gear moves tried: {len(rows)} to the REAL tooth, {tc} to a different wrong tooth")
    W(f"  landing in 0 <= E <= 3:  real tooth {nreal}/{len(rows)} = {nreal/len(rows):.2f};  "
      f"wrong tooth {nc}/{tc} = {nc/tc:.2f}")
    per = Counter()
    perN = Counter()
    for r in rows:
        perN[r["gear"]] += 1
        if r["Ereal"] is not None and 0 <= r["Ereal"] <= 3:
            per[r["gear"]] += 1
    W("  by gear (repairs / attempts, real tooth): " +
      "; ".join(f"{g}:{per[g]}/{perN[g]}" for g in sorted(perN)))
    # upper half only
    up = [r for r in rows if r["E0"] > 3]
    nru = sum(1 for r in up if r["Ereal"] is not None and r["Ereal"] <= 3)
    ncu = sum(1 for r in up for e in r["Ectrl"] if e is not None and e <= 3)
    tcu = sum(len(r["Ectrl"]) for r in up)
    W(f"  restricted to members violating the UPPER half (E > 3), landing at E <= 3: "
      f"real {nru}/{len(up)} = {nru/len(up):.2f}; wrong {ncu}/{tcu} = {ncu/tcu:.2f}")
    return rows


def main():
    t0 = time.time()
    L = []
    W = L.append
    W("=== FOLLOW-UPS: the whole top of the row, and the repair control ===")
    A = partA(W)
    W(f"[A done {time.time()-t0:.1f}s]")
    B = partB(W)
    W(f"[B done {time.time()-t0:.1f}s]")
    json.dump(dict(A=A, B=B), open(os.path.join(OUT, "pa_extra.json"), "w"))
    txt = "\n".join(L)
    open(os.path.join(OUT, "pa_extra.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
