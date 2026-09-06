"""pl_spare.py -- THE SPARE-GEAR LEMMA and its census.

LEMMA (proved in pinned_letter.md 4.1).  Let x_0 < x_1 < x_2 be three consecutive openings of M,
a = x_1 - x_0, v = x_2 - x_1, S = a + v.  Call a gear h

    OBSTRUCTED at the run   if neither tooth-placement of h on x_1 misses both x_0 and x_2,
                            i.e. if h | a, or h | v, or
                            (a = +d_h or v = -d_h) and (a = -d_h or v = +d_h)  (mod h),
    BUSY inside the run     if some column strictly between x_0 and x_2 is struck by h alone.

If some gear is neither obstructed nor busy then F(M) >= S, i.e. E(v) = S - F <= 0.

Contrapositive, which is what the excess means: a 2-run whose span exceeds the record has EVERY
gear busy or obstructed, at every one of its occurrences.

This script verifies the lemma on full periods (a violation would be an instrument failure) and
takes the census: how many gears are free / obstructed / busy at the attaining 2-run of each size,
at the letter in particular.

Outputs results/pl_spare.txt / .json.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "r56"))
from mf_core import build_levels, u_of                    # noqa: E402
from pl_glue import occurrences, letters                  # noqa: E402


OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)


def obstructed(h, u, a, v):
    d = (2 * u) % h
    if a % h == 0 or v % h == 0:
        return True
    c1 = (a % h == d) or (v % h == (-d) % h)          # teeth {x1, x1 - d} spoiled
    c2 = (a % h == (-d) % h) or (v % h == d)          # teeth {x1, x1 + d} spoiled
    return c1 and c2


def busy_and_free(gears, us, x0, a, v):
    """returns (free gears, obstructed gears, busy gears) at the run starting at x0."""
    S = a + v
    cols = np.arange(x0 + 1, x0 + S)
    strikes = np.zeros((len(gears), cols.size), dtype=bool)
    for i, (g, u) in enumerate(zip(gears, us)):
        rr = cols % g
        strikes[i] = (rr == u % g) | (rr == (-u) % g)
    cnt = strikes.sum(axis=0)
    interior_open = np.flatnonzero(cnt == 0)
    assert interior_open.size == 1 and cols[interior_open[0]] == x0 + a, "not a 2-run"
    sole = (cnt == 1)
    free, obs, busy = [], [], []
    for i, (g, u) in enumerate(zip(gears, us)):
        ob = obstructed(g, u, a, v)
        bu = bool((strikes[i] & sole).any())
        if ob:
            obs.append(g)
        if bu:
            busy.append(g)
        if not ob and not bu:
            free.append(g)
    return free, obs, busy


def main():
    t0 = time.time()
    lines = []
    W = lines.append
    W("=== THE SPARE-GEAR LEMMA: a free gear forces S <= F ===")
    W("free = neither obstructed at the middle opening nor a sole striker inside the run.")
    L = build_levels()
    res = {}
    viol = 0
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
        W(f"\n=== M = {gears}  q' = {qn}  F = {F}  a_L = {aL} ===")
        W("  v | a=r(v) | S | E | occ | max #free over occurrences | obstructed | busy | "
          "lemma consistent?")
        rows = []
        for v in realised:
            if r[v] == 0:
                continue
            a = r[v]
            S = a + v
            occ = occurrences(lv.size, a, v, 16)
            mf, det = 0, None
            for i in occ:
                i = int(i)
                fr, ob, bu = busy_and_free(gears, us, int(lv.O[i]), int(lv.size[i]),
                                           int(lv.size[i + 1]))
                if len(fr) >= mf:
                    mf, det = len(fr), (fr, ob, bu)
            okl = not (mf > 0 and S > F)
            if not okl:
                viol += 1
            rows.append(dict(v=v, a=a, S=S, E=S - F, occ=int(occ.size), free=mf,
                             free_gears=det[0], obstructed=det[1], busy=det[2], ok=bool(okl)))
            W(f"  {v:>3} | {a:>3} | {S:>3} | {S-F:>+3} | {occ.size:>3} | {mf} {det[0]} | "
              f"{det[1]} | {det[2]} | {'yes' if okl else 'LEMMA VIOLATED'}")
        nfree = sum(1 for x in rows if x["free"] > 0)
        npos = sum(1 for x in rows if x["E"] > 0)
        W(f"  sizes with a free gear at some occurrence: {nfree} of {len(rows)}; "
          f"sizes with E > 0: {npos}; overlap (must be 0): "
          f"{sum(1 for x in rows if x['free'] > 0 and x['E'] > 0)}")
        aLrow = next((x for x in rows if x["v"] == aL), None)
        if aLrow:
            W(f"  THE LETTER a_L = {aL}: E = {aLrow['E']:+d}, free gears {aLrow['free_gears']}, "
              f"obstructed {aLrow['obstructed']}, busy {aLrow['busy']}")
        res[str(qn)] = dict(gears=gears, F=F, aL=aL, rows=rows)
        W(f"  [{time.time()-t0:.1f}s]")
    W(f"\nlemma violations over every machine and size: {viol} (any violation is an instrument "
      f"failure or a false lemma)")
    json.dump(res, open(os.path.join(OUT, "pl_spare.json"), "w"))
    txt = "\n".join(lines)
    open(os.path.join(OUT, "pl_spare.txt"), "w").write(txt)
    print(f"wrote {OUT}/pl_spare.txt ({len(txt)} chars, {time.time()-t0:.1f}s); violations {viol}")
    for k, e in res.items():
        rows = e["rows"]
        aLrow = next((x for x in rows if x["v"] == e["aL"]), None)
        print(f"q'={k:>2} F={e['F']:>2} aL={e['aL']:>2} free(aL)="
              f"{aLrow['free'] if aLrow else '-'} | sizes with a free gear "
              f"{sum(1 for x in rows if x['free'] > 0)}/{len(rows)}; E>0 at "
              f"{sum(1 for x in rows if x['E'] > 0)}; overlap "
              f"{sum(1 for x in rows if x['free'] > 0 and x['E'] > 0)}")


if __name__ == "__main__":
    main()
