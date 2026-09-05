"""Item 1: the 20 residual runs.  For each: the strike structure of both flanks gear by gear,
the shared gears with their columns and residues, the exact reason no two-colouring works, and
then the three alternative constructions (C2+f free gears; the shifted/overlapped glue Cs; the
cross glue to a different opening's right flank), with the loss each achieves.
"""
import sys
import numpy as np
from gl_glue import (gears_of, us_of, sieve, gap_stats, glue, glue_free, cov_pair,
                     solve_cover, forced_sides, propagate, shifted_best, strikers)

RESID = {
    17: [(29055, 7, 6, 10), (56007, 10, 6, 7)],
    19: [(351300, 12, 6, 10), (724365, 12, 6, 10), (892222, 10, 6, 12), (1265287, 10, 6, 12),
         (118295, 10, 7, 18), (1498285, 18, 7, 10)],
    23: [(8083133, 12, 7, 23), (8480803, 10, 7, 25), (15578190, 25, 7, 10),
         (21603913, 10, 7, 25), (28701300, 25, 7, 10), (29098970, 23, 7, 12),
         (7052418, 9, 8, 23), (13636268, 9, 8, 23), (23545837, 23, 8, 9),
         (30129687, 23, 8, 9), (18159472, 15, 11, 12), (19022635, 12, 11, 15)],
}


def cross_glue(gears, us, x0, L, R, opens, gaps, rng, tries=4000):
    """glue this left flank to some other opening's right flank; best R' found."""
    cand = np.flatnonzero(gaps >= R)
    if cand.size == 0:
        return None
    order = rng.permutation(cand.size)[:tries]
    best = None
    for t in order.tolist():
        i = int(cand[t])
        y = int(opens[i]); Rp = int(gaps[i])
        T, h, covL, covR = cov_pair(gears, us, x0, L, Rp, y)
        if solve_cover(covL, covR, T, h) is not None:
            if best is None or Rp > best[1]:
                best = (y, Rp)
            if Rp >= R:
                return best
    return best


def report(out, top, runs, opens, gaps, F, F2):
    gears = gears_of(top)
    us = us_of(gears)
    rng = np.random.default_rng(20260905 + top)
    for (x0, L, v, R) in runs:
        x1, x2, x3 = x0 + L, x0 + L + v, x0 + L + v + R
        out.write(f"\n===== m{top}  (L,v,R)=({L},{v},{R})  sum={L+R} "
                  f"(F{L+R-F:+d}, F_2{L+R-F2:+d})  x0={x0}\n")
        out.write(f"      residues x0 mod g: "
                  f"{ {g: x0 % g for g in gears} }\n")
        out.write(f"      u_g: { {g: u for g, u in zip(gears, us)} }   "
                  f"v mod g: { {g: v % g for g in gears} }\n")
        SL, SR = {}, {}
        for c in range(x0 + 1, x1):
            for g in strikers(gears, us, c):
                SL.setdefault(g, []).append(c - x0)
        for c in range(x2 + 1, x3):
            for g in strikers(gears, us, c):
                SR.setdefault(g, []).append(c - x2)
        soleL = {g: [c - x0 for c in range(x0 + 1, x1) if strikers(gears, us, c) == [g]]
                 for g in gears}
        soleR = {g: [c - x2 for c in range(x2 + 1, x3) if strikers(gears, us, c) == [g]]
                 for g in gears}
        soleL = {g: o for g, o in soleL.items() if o}
        soleR = {g: o for g, o in soleR.items() if o}
        out.write(f"      left-flank strikers (gear: offsets from x0):  {SL}\n")
        out.write(f"      right-flank strikers (gear: offsets from x2): {SR}\n")
        out.write(f"      sole strikers left  {soleL}\n")
        out.write(f"      sole strikers right {soleR}\n")
        shared = sorted(set(SL) & set(SR))
        out.write(f"      gears striking BOTH flanks: {shared}\n")
        for g in shared:
            u = us[gears.index(g)]
            out.write(f"        gear {g}: left cols {[x0 + o for o in SL[g]]} "
                      f"(mod {g}: {[(x0 + o) % g for o in SL[g]]}), "
                      f"right cols {[x2 + o for o in SR[g]]} "
                      f"(mod {g}: {[(x2 + o) % g for o in SR[g]]}); teeth "
                      f"{ {u % g, (-u) % g} }\n")
        sharedsole = sorted(set(soleL) & set(soleR))
        out.write(f"      sole on BOTH flanks: {sharedsole} "
                  f"(these force left and right at once, {'CONFLICT' if sharedsole else 'none'})\n")
        # the covering instance
        T, h, covL, covR = cov_pair(gears, us, x0, L, R, x2)
        fs = forced_sides(covL, covR, T, h)
        fs_txt = {gears[i]: {('L' if s == 0 else 'R'): o for s, o in d.items()}
                  for i, d in fs.items()}
        out.write(f"      covering instance: T={T} offsets, hole at {h}; "
                  f"columns with a UNIQUE (gear,side) candidate force: {fs_txt}\n")
        conflict = [gears[i] for i, d in fs.items() if len(d) == 2]
        out.write(f"      gears forced to BOTH sides by unique columns: {conflict}\n")
        st, det = propagate(covL, covR, T, h)
        out.write(f"      unit propagation: {st}"
                  + (f"  (column {det[2]} has no candidate left)" if st == 'empty' else "")
                  + f"  assignment { {gears[i]: ('L' if s == 0 else 'R') for i, s in enumerate(det[0]) if s is not None} }\n")
        # which columns survive the best partial colouring
        best = None
        for mask in range(1 << len(gears)):
            acc = 0
            for i in range(len(gears)):
                acc |= covR[i] if (mask >> i) & 1 else covL[i]
            miss = bin((((1 << T) - 1) ^ (1 << h)) & ~acc).count('1')
            if best is None or miss < best[0]:
                best = (miss, mask, (((1 << T) - 1) ^ (1 << h)) & ~acc)
        missbits = [j for j in range(T) if (best[2] >> j) & 1]
        out.write(f"      best colouring leaves {best[0]} column(s) uncovered, at offsets "
                  f"{missbits} (side of the run: "
                  f"{['left' if j < h else 'right' for j in missbits]})\n")
        # ---- alternatives
        for f in (1, 2):
            sol = glue_free(gears, us, x0, L, v, R, f)
            if sol is not None:
                out.write(f"      C2+{f}: SUCCESS, loss 0 -- "
                          f"{ {g: t for g, t in zip(gears, sol)} }\n")
                break
        else:
            out.write("      C2+1, C2+2: no\n")
        sb = shifted_best(gears, us, x0, L, v, R, tmax=10)
        if sb:
            t, m = sb
            X = F + 1 + t
            out.write(f"      Cs (overlap t): t={t} certifies F >= {L+R-1-t}, i.e. "
                      f"L+R <= F+1+t = {X}; loss vs F_2 = {max(0, X - F2)}\n")
        else:
            out.write("      Cs: no overlap t <= 10 works\n")
        cg = cross_glue(gears, us, x0, L, R, opens, gaps, rng)
        if cg:
            y, Rp = cg
            out.write(f"      Cx (cross glue): left flank {L} + right flank {Rp} at y={y} "
                      f"certifies F_2 >= {L+Rp}; loss = {max(0, (L+R)-(L+Rp))}\n")
        else:
            out.write("      Cx: no partner found in the sample\n")
        out.flush()


def main(out):
    for top in (17, 19, 23):
        gears = gears_of(top)
        us = us_of(gears)
        P, blocked = sieve(gears, us)
        opens, gaps, F, F2, N = gap_stats(P, blocked)
        report(out, top, RESID[top], opens, gaps, F, F2)


if __name__ == "__main__":
    dest = sys.argv[1] if len(sys.argv) > 1 else None
    o = open(dest, "w") if dest else sys.stdout
    main(o)
    if dest:
        o.close()
