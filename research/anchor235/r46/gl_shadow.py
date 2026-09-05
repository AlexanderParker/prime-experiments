"""The shadow lemma and the full sweep.

SHADOW LEMMA (claimed, verified here).  In the glue instance of a 3-run (L, v, R) the only
offsets of the target that are invisible to one side are:
  h = L-1        (the hole; x_1 under the left base, x_2 under the right base) -- invisible to
                 both, and that is part (i) of the glue lemma;
  h + v          (x_2 under the LEFT base, x_2 + v under the right base): only a RIGHT gear can
                 cover it, and only by striking x_2 + v; in the target iff v <= R-1;
  h - v          (x_1 under the RIGHT base, x_1 - v under the left base): only a LEFT gear can
                 cover it, and only by striking x_1 - v; in the target iff v <= L-1.
Every other offset has at least one candidate on each side.  Consequently, when v >= min(L, R)
one pinch is absent and the constant colouring glues -- which is the peel bound
(max(L,R) + v <= F_2, docs/proof-search/alignment-rules.md 736-790), cited not re-derived.
The covering problem is therefore only non-trivial when v < min(L, R).

Modes:
  shadow  : verify the lemma on every 3-run with v >= 6 at m13..m23 (exact, exhaustive).
  sweep   : C2 on EVERY 3-run with v >= 6 (and separately those with L+R > F), by machine, with
            the split v >= min(L,R) / v < min(L,R) and the loss of the fallback certificates.
  pinch   : for the C2 failures, the minimum-miss colourings and which gears the pinch needs.
"""
import sys
import numpy as np
from gl_glue import (gears_of, us_of, sieve, gap_stats, glue, glue_free, cov_pair,
                     solve_cover, shifted_best, strikers, runs_with)


def shadow_check(out, tops=(13, 17, 19, 23)):
    for top in tops:
        gears = gears_of(top)
        us = us_of(gears)
        P, blocked = sieve(gears, us)
        opens, gaps, F, F2, N = gap_stats(P, blocked)
        n = bad = 0
        onesided = {}
        for (x0, L, v, R) in runs_with(opens, gaps, vmin=6, sum_gt=F - 12):
            n += 1
            T, h, covL, covR = cov_pair(gears, us, x0, L, R, x0 + L + v)
            anyL = 0
            anyR = 0
            for i in range(len(gears)):
                anyL |= covL[i]
                anyR |= covR[i]
            full = ((1 << T) - 1)
            lonly = full & anyL & ~anyR          # only left gears can cover
            ronly = full & anyR & ~anyL
            none = full & ~anyL & ~anyR
            pred_l = (1 << (h - v)) if v <= L - 1 else 0
            pred_r = (1 << (h + v)) if v <= R - 1 else 0
            if lonly != pred_l or ronly != pred_r or none != (1 << h):
                bad += 1
                if bad <= 3:
                    out.write(f"   VIOLATION m{top} ({L},{v},{R}) x0={x0} lonly={bin(lonly)} "
                              f"ronly={bin(ronly)} none={bin(none)}\n")
            key = (v <= L - 1, v <= R - 1)
            onesided[key] = onesided.get(key, 0) + 1
        out.write(f"m{top}: shadow lemma checked on {n} 3-runs with v>=6 and L+R>F-12: "
                  f"{n - bad} confirmed, {bad} violations; "
                  f"(v<=L-1, v<=R-1) split {onesided}\n")
        out.flush()


def min_moves(gears, us, x0, L, v, R):
    """least number of gears that must be re-phased away from the run's own phase for the
    covering to succeed (all-left IS the run itself and misses exactly the right shadow).
    Returns (ok, moves_from_left, moves_from_right) with None where no solution."""
    T, h, covL, covR = cov_pair(gears, us, x0, L, R, x0 + L + v)
    k = len(gears)
    target = ((1 << T) - 1) ^ (1 << h)
    bl = br = None
    for mask in range(1 << k):
        acc = 0
        for i in range(k):
            acc |= covR[i] if (mask >> i) & 1 else covL[i]
        if acc & target == target:
            w = bin(mask).count('1')
            bl = w if bl is None else min(bl, w)
            br = (k - w) if br is None else min(br, k - w)
    return (bl is not None), bl, br


def sweep(out, tops=(13, 17, 19, 23)):
    for top in tops:
        gears = gears_of(top)
        us = us_of(gears)
        P, blocked = sieve(gears, us)
        opens, gaps, F, F2, N = gap_stats(P, blocked)
        for label, thr in (("L+R>F", F), ("L+R>F-6", F - 6)):
            tot = ok = 0
            triv = trivok = 0
            hard = hardok = 0
            fails = {}
            maxsum_fail = 0
            mv = {}
            for (x0, L, v, R) in runs_with(opens, gaps, vmin=6, sum_gt=thr):
                tot += 1
                good, bl, br = min_moves(gears, us, x0, L, v, R)
                ok += good
                if good:
                    m = min(bl, br)
                    mv[m] = mv.get(m, 0) + 1
                if v >= min(L, R):
                    triv += 1
                    trivok += good
                else:
                    hard += 1
                    hardok += good
                if not good:
                    fails[(L, v, R)] = fails.get((L, v, R), 0) + 1
                    maxsum_fail = max(maxsum_fail, L + R)
            out.write(f"m{top} [{label}]: 3-runs v>=6: {tot}; C2 ok {ok} "
                      f"({100*ok/max(tot,1):.2f}%);  trivial (v>=min(L,R)) {trivok}/{triv}; "
                      f"hard (v<min(L,R)) {hardok}/{hard} "
                      f"({100*hardok/max(hard,1):.2f}%)\n")
            out.write(f"      minimal gears re-phased (min over the two constant colourings): "
                      f"{ {m: mv[m] for m in sorted(mv)} }\n")
            if fails:
                srt = sorted(fails.items(), key=lambda kv: -(kv[0][0] + kv[0][2]))
                out.write(f"      distinct failing shapes {len(fails)}, "
                          f"total failures {sum(fails.values())}, max failing L+R "
                          f"{maxsum_fail} (F={F}, F_2={F2}); top shapes: "
                          f"{[(k, c) for k, c in srt[:8]]}\n")
            out.flush()


def pinch(out, tops=(17, 19, 23)):
    for top in tops:
        gears = gears_of(top)
        us = us_of(gears)
        P, blocked = sieve(gears, us)
        opens, gaps, F, F2, N = gap_stats(P, blocked)
        att = {v: s for v, s in N.items() if v >= 6}
        for (x0, L, v, R) in runs_with(opens, gaps, vmin=6, only_attaining=att):
            if glue(gears, us, x0, L, v, R) is not None:
                continue
            x1, x2 = x0 + L, x0 + L + v
            T, h, covL, covR = cov_pair(gears, us, x0, L, R, x2)
            k = len(gears)
            target = ((1 << T) - 1) ^ (1 << h)
            best = T + 1
            missed = {}
            for mask in range(1 << k):
                acc = 0
                for i in range(k):
                    acc |= covR[i] if (mask >> i) & 1 else covL[i]
                m = target & ~acc
                c = bin(m).count('1')
                if c < best:
                    best, missed = c, {}
                if c == best:
                    missed[m] = missed.get(m, 0) + 1
            offs = sorted({j for m in missed for j in range(T) if (m >> j) & 1})
            KL = strikers(gears, us, x1 - v)
            KR = strikers(gears, us, x2 + v)
            out.write(f"m{top} ({L},{v},{R}) x0={x0}: min miss {best}; uncovered offsets over "
                      f"all optimal colourings {offs} (h={h}, h-v={h-v}, h+v={h+v}); "
                      f"left shadow x1-v struck by {KL}, right shadow x2+v struck by {KR}; "
                      f"shadow gears disjoint: {not set(KL) & set(KR)}\n")
        out.flush()


if __name__ == "__main__":
    what = sys.argv[1]
    dest = sys.argv[2] if len(sys.argv) > 2 else None
    o = open(dest, "w") if dest else sys.stdout
    if what == "shadow":
        shadow_check(o)
    elif what == "sweep":
        sweep(o)
    elif what == "pinch":
        pinch(o)
    if dest:
        o.close()
