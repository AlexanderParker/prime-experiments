"""pl_depth.py -- THE RE-PHASING DEPTH of a 2-run: the minimum number of gears whose phase must be
moved, starting from an occurrence of the attaining pair (r(v), v), to turn the pair into a single
gap of length >= S - c.

Exact formulation.  Fix an occurrence x_0 < x_1 < x_2 (gaps a = x_1 - x_0, v = x_2 - x_1,
S = a + v).  A CONFIGURATION assigns each gear g a shift s_g in Z_g; its teeth are then
{u_g + s_g, -u_g + s_g} mod g.  The observed run is s = 0 for every gear.  Every configuration
occurs somewhere in the period (CRT over coprime moduli), so if a configuration has an opening at
L, an opening at R and every column strictly between blocked, then F(M) >= R - L.

    depth(L, R) = min #{ g : s_g != 0 } over configurations making (L, R) a gap
    depth_c(v)  = min over targets L < x_1 < R with R - L >= S - c of depth(L, R)

depth = 1 is the one-gear glue of pl_glue.py.  Search: for k = 0, 1, 2, ... try every k-subset of
gears to move; the gears left alone contribute their observed strikes for free, and the holes they
leave must be covered by the moved gears (an exact cover branching on the lowest uncovered hole).

Outputs results/pl_depth.txt / .json.
"""
import json
import os
import sys
import time
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "r56"))
from mf_core import build_levels, u_of                    # noqa: E402
from pl_glue import occurrences, letters                  # noqa: E402

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
MAXK = 4


def opt_masks(g, u, L, R):
    """for gear g: list of (shift, covermask) over shifts leaving L and R open; plus the observed
    mask (shift 0) and whether the observed shift is itself valid."""
    out = []
    obs = None
    for s in range(g):
        T = set(((u + s) % g, (-u + s) % g))
        bad = any((L - t) % g == 0 or (R - t) % g == 0 for t in T)
        mask = 0
        for t in T:
            c = L + 1 + ((t - (L + 1)) % g)
            while c < R:
                mask |= 1 << (c - L - 1)
                c += g
        if s == 0:
            obs = (mask, not bad)
        if not bad:
            out.append((s, mask))
    return out, obs


def _cover_rest_multi(hole, opts, used):
    """cover the bitmask `hole` using at most one option from each unused list; branch on the
    lowest uncovered column."""
    if hole == 0:
        return []
    low = hole & -hole
    for j in range(len(opts)):
        if j in used:
            continue
        for s, m in opts[j]:
            if m & low:
                sub = _cover_rest_multi(hole & ~m, opts, used | {j})
                if sub is not None:
                    return [(j, s)] + sub
    return None


def depth_target(gears, us, L, R, maxk=MAXK):
    """min number of moved gears making (L, R) a gap of the machine, or None."""
    n = R - L - 1
    if n <= 0:
        return None, None
    full = (1 << n) - 1
    O, OBS, VAL = [], [], []
    for g, u in zip(gears, us):
        o, obs = opt_masks(g, u, L, R)
        if not o:
            return None, None                # this gear cannot avoid both ends
        O.append(o)
        OBS.append(obs[0])
        VAL.append(obs[1])
    forced = [i for i in range(len(gears)) if not VAL[i]]
    if len(forced) > maxk:
        return None, None
    idx = list(range(len(gears)))
    for k in range(len(forced), maxk + 1):
        for extra in combinations([i for i in idx if i not in forced], k - len(forced)):
            K = list(forced) + list(extra)
            covered = 0
            for i in idx:
                if i not in K:
                    covered |= OBS[i]
            hole = full & ~covered
            if hole == 0:
                return k, []
            sub = _cover_rest_multi(hole, [O[i] for i in K], set())
            if sub is not None:
                return k, [(gears[K[j]], s) for j, s in sub]
    return None, None


def depth_of_run(gears, us, x0, a, v, c=3, slack=3):
    """min re-phasing depth over targets L < x_1 < R with R - L >= S - c."""
    S = a + v
    x1, x2 = x0 + a, x0 + S
    cand = []
    for L in range(x0 - slack, x0 + slack + 1):
        for R in range(x2 - slack, x2 + slack + 1):
            if L < x1 < R and R - L >= S - c:
                cand.append((R - L, L, R))
    cand.sort(key=lambda t: -t[0])
    bestd, bestT = None, None
    for T, L, R in cand:
        if bestd is not None and bestd <= 1:
            break
        d, w = depth_target(gears, us, L, R, maxk=(MAXK if bestd is None else bestd - 1))
        if d is not None and (bestd is None or d < bestd):
            bestd, bestT = d, (L - x0, R - x0, T, w)
    return bestd, bestT


def main():
    t0 = time.time()
    lines = []
    W = lines.append
    W("=== THE RE-PHASING DEPTH: how many gears must move to glue the attaining 2-run ===")
    W("depth = min #{g : s_g != 0} over configurations with a gap of length >= S - 3 containing")
    W("the middle opening.  Every configuration occurs in the period, so F(M) >= that length.")
    W("A depth-k glue is a certificate E(v) <= 3 that moves k of the machine's gears.")
    L = build_levels()
    res = {}
    for n, qn in ((2, 13), (3, 17), (4, 19), (5, 23), (6, 29)):
        lv = L[n]
        gears = list(lv.gears)
        us = [u_of(g) for g in gears]
        F = int(lv.size.max())
        aL, bL = letters(qn)
        lft, rgt = lv.size[:-1], lv.size[1:]
        mult = np.bincount(lv.size.astype(np.int64))
        realised = np.flatnonzero(mult).tolist()
        r = {}
        for v in realised:
            s1 = set(np.unique(rgt[lft == v]).tolist()) | set(np.unique(lft[rgt == v]).tolist())
            r[v] = max(s1) if s1 else 0
        W(f"\n=== M = {gears}  q' = {qn}  F = {F}  a_L = {aL}  b_L = {bL} ===")
        W("  v | a=r(v) | S | E=S-F | depth | target (dL,dR,len) | gears moved | occ")
        rows = []
        for v in realised:
            if r[v] == 0:
                continue
            a = r[v]
            S = a + v
            occ = occurrences(lv.size, a, v, 8)
            bd, bt = None, None
            for i in occ:
                i = int(i)
                d, t = depth_of_run(gears, us, int(lv.O[i]), int(lv.size[i]), int(lv.size[i + 1]))
                if d is not None and (bd is None or d < bd):
                    bd, bt = d, t
                    if bd <= 1:
                        break
            rows.append(dict(v=v, a=a, S=S, E=S - F, depth=bd, target=bt, occ=int(occ.size)))
            W(f"  {v:>3} | {a:>3} | {S:>3} | {S-F:>+3} | {str(bd):>4} | "
              f"{bt[:3] if bt else '-'} | {bt[3] if bt else '-'} | {occ.size}")
        ds = [x["depth"] for x in rows if x["depth"] is not None]
        W("  depth distribution: " +
          " ".join(f"{k}:{sum(1 for d in ds if d == k)}" for k in sorted(set(ds))) +
          f"  (no glue within depth {MAXK}: {sum(1 for x in rows if x['depth'] is None)})")
        aLrow = next((x for x in rows if x["v"] == aL), None)
        if aLrow:
            W(f"  THE LETTER a_L = {aL}: E = {aLrow['E']:+d}, depth = {aLrow['depth']}, "
              f"target {aLrow['target']}")
        res[str(qn)] = dict(gears=gears, qn=qn, F=F, aL=aL, bL=bL, rows=rows)
        W(f"  [{time.time()-t0:.1f}s]")
    json.dump(res, open(os.path.join(OUT, "pl_depth.json"), "w"), default=str)
    txt = "\n".join(lines)
    open(os.path.join(OUT, "pl_depth.txt"), "w").write(txt)
    print(f"wrote {OUT}/pl_depth.txt ({len(txt)} chars, {time.time()-t0:.1f}s)")
    for k, e in res.items():
        ds = [x["depth"] for x in e["rows"] if x["depth"] is not None]
        aLrow = next((x for x in e["rows"] if x["v"] == e["aL"]), None)
        print(f"q'={k:>2} F={e['F']:>2} aL={e['aL']:>2} depth(aL)="
              f"{aLrow['depth'] if aLrow else '-'} | depths " +
              " ".join(f"{kk}:{sum(1 for d in ds if d == kk)}" for kk in sorted(set(ds))) +
              f" none:{sum(1 for x in e['rows'] if x['depth'] is None)}")


if __name__ == "__main__":
    main()
