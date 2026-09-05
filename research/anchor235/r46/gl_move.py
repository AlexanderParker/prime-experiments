"""Item 3: the move lemma and what it predicts.

THE MOVE LEMMA.  Recolouring gear g from left to right translates the columns it inspects by
exactly +v: at target offset j the left base shows column x_0+1+j and the right base shows
x_0+1+j+v.  Hence g covers offset j after the move iff it strikes c+v where c = x_0+1+j, and
g strikes both c and c+v iff v = 0, +d_g or -d_g (mod g) -- the chain law's alphabet
(docs/proofs/05 (C)); d_g = 2u_g and the least positive representatives of +-d_g are the
letters a_g, b_g of gear g.  So

  * v = 0 (mod g)  (v padded at g): the move changes nothing at all; such a gear can never
    cover the right shadow x_2+v, because x_2+v = x_2 (mod g) and x_2 is an opening;
  * v = +-d_g (mod g)  (v a LETTER of g): exactly one of g's two teeth survives the move;
  * otherwise: not one strike of g survives -- its coverage is replaced by a disjoint set.

Prediction tested here: the glue succeeds on a HARD run (v < min(L,R)) essentially only when the
right shadow x_2+v is struck by a gear whose move is survivable, i.e. a gear of
Leg(v) = {g : v = +-d_g mod g} or a gear with no sole column to lose.
"""
import sys
import numpy as np
from gl_glue import (gears_of, us_of, sieve, gap_stats, cov_pair, solve_cover, strikers,
                     runs_with)
from gl_shadow import min_moves


def legality(gears, us, v):
    leg, pad = [], []
    for g, u in zip(gears, us):
        d = (2 * u) % g
        r = v % g
        if r == 0:
            pad.append(g)
        elif r in (d, (-d) % g):
            leg.append(g)
    return leg, pad


def analyse(out, top, thr_off=6):
    gears = gears_of(top)
    us = us_of(gears)
    P, blocked = sieve(gears, us)
    opens, gaps, F, F2, N = gap_stats(P, blocked)
    rows = []
    for (x0, L, v, R) in runs_with(opens, gaps, vmin=6, sum_gt=F - thr_off):
        if v >= min(L, R):
            continue
        x1, x2 = x0 + L, x0 + L + v
        good, bl, br = min_moves(gears, us, x0, L, v, R)
        KR = strikers(gears, us, x2 + v)
        KL = strikers(gears, us, x1 - v)
        leg, pad = legality(gears, us, v)
        rows.append((v, L, R, x0, good, KR, KL, leg, pad))
    tot = len(rows)
    ok = sum(r[4] for r in rows)
    # prediction P: success  =>  K_R meets Leg(v)
    p_ok = sum(1 for r in rows if r[4] and set(r[5]) & set(r[7]))
    p_fail_but = sum(1 for r in rows if (not r[4]) and set(r[5]) & set(r[7]))
    noleg = sum(1 for r in rows if not set(r[5]) & set(r[7]))
    noleg_ok = sum(1 for r in rows if (not set(r[5]) & set(r[7])) and r[4])
    out.write(f"\nm{top} (F={F}, F_2={F2}) HARD runs (v<min(L,R)) with L+R>{F-thr_off}: {tot}; "
              f"C2 ok {ok} ({100*ok/max(tot,1):.1f}%)\n")
    out.write(f"   of the {ok} successes, {p_ok} have a shadow-striker for which v is a LETTER "
              f"({100*p_ok/max(ok,1):.1f}%)\n")
    out.write(f"   runs with NO letter-gear striking the right shadow: {noleg}; of those "
              f"C2 succeeds {noleg_ok} ({100*noleg_ok/max(noleg,1):.1f}%)\n")
    out.write(f"   runs WITH a letter-gear striking the right shadow: {tot-noleg}; of those "
              f"C2 succeeds {ok-noleg_ok} ({100*(ok-noleg_ok)/max(tot-noleg,1):.1f}%)\n")
    byv = {}
    for r in rows:
        d = byv.setdefault(r[0], [0, 0, r[7], r[8]])
        d[0] += 1
        d[1] += r[4]
    out.write("   by middle size v:  v (runs, C2 ok, Leg(v), Pad(v))\n")
    for v in sorted(byv):
        d = byv[v]
        out.write(f"      v={v:3d}: {d[0]:5d} runs, {d[1]:5d} ok, Leg={d[2]}, Pad={d[3]}\n")
    out.flush()
    return rows


if __name__ == "__main__":
    dest = sys.argv[1] if len(sys.argv) > 1 else None
    o = open(dest, "w") if dest else sys.stdout
    for t in (17, 19, 23):
        analyse(o, t)
    if dest:
        o.close()
