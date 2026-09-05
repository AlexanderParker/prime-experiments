"""Item 3: the cause in the teeth.

Exact condition.  Put the run's offsets relative to x0+1: the LEFT FLANK is the offset arc
A = [0, L-2] and the RIGHT FLANK is B = [L+v, L+v+R-2].  A gear with teeth {t, t+D} (mod g)
and phase alpha = (t - x0 - 1) mod g strikes the left flank iff alpha in A (mod g) or
alpha + D in A (mod g), and the right flank iff alpha in B or alpha + D in B.  So it strikes
BOTH iff

    ( [alpha]_A or [alpha+D]_A )  and  ( [alpha]_B or [alpha+D]_B ),      (*)

where [z]_S means z mod g lies in the arc S.  When the whole run fits inside one period of the
gear (L+v+R-1 <= g) the arcs A and B are disjoint and do not wrap, so (*) reduces to "one tooth
in each arc", which needs the tooth separation to lie in the window

    W = [v+2, L+v+R-2]   (width L+R-3, starting past the middle gap),

i.e. D in W or g-D in W.  A gear with D < v+2 and g-D > L+v+R-2 can then never strike both.
For the REAL teeth D = (g +- 1)/3 and g - D = (2g -+ 1)/3, so that exemption needs
g < 3(v+2) and g > 1.5(L+v+R-2) at once, i.e. v > L+R-6 -- the peel-bound region.

This script (a) verifies (*) against the direct strike test, (b) computes, for every gear and
every tooth separation D = 1..(g-1)/2, the exact fraction of phases at which the gear strikes
both flanks of the observed (L, v, R), and (c) ranks the real separation D_g = (g +- 1)/3
against all the others, per cell and in the aggregate E[ov] = sum_g P_both(g, D_g).
"""
import sys, os, json
from sp_core import gears_of, us_of, sieve, gap_stats, attaining_runs, sep_run

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def in_arc(z, lo, hi, g):
    """is z mod g inside the offset arc [lo, hi] (hi-lo+1 may exceed g)."""
    n = hi - lo + 1
    if n <= 0:
        return False
    if n >= g:
        return True
    return ((z - lo) % g) < n


def strikes_both(g, D, alpha, L, v, R):
    a1 = in_arc(alpha, 0, L - 2, g) or in_arc(alpha + D, 0, L - 2, g)
    b1 = in_arc(alpha, L + v, L + v + R - 2, g) or in_arc(alpha + D, L + v, L + v + R - 2, g)
    return a1 and b1


def p_both(g, D, L, v, R):
    return sum(strikes_both(g, D, a, L, v, R) for a in range(g)) / g


def real_D(g):
    u = pow(6, -1, g)
    d = (2 * u) % g
    return min(d, g - d)


def verify(gears, us, runs, out):
    """(a): the arc condition (*) reproduces the direct strike test."""
    bad = 0
    tot = 0
    for (x0, L, v, R) in runs:
        for g, u in zip(gears, us):
            direct_l = any((x0 + 1 + i) % g in (u % g, (-u) % g) for i in range(L - 1))
            direct_r = any((x0 + L + v + 1 + j) % g in (u % g, (-u) % g) for j in range(R - 1))
            t = (-u) % g          # teeth are {-u, -u + 2u} = {-u, u}
            D = (2 * u) % g
            alpha = (t - x0 - 1) % g
            pred = strikes_both(g, D, alpha, L, v, R)
            tot += 1
            if pred != (direct_l and direct_r):
                bad += 1
    out.write(f"  condition (*) against the direct strike test: {tot-bad}/{tot} agree\n")
    return bad


def main(out, tops=(13, 17, 19, 23)):
    dump = {}
    for top in tops:
        gears = gears_of(top)
        us = us_of(gears)
        P, blocked = sieve(gears, us)
        opens, gaps, F, F2, N = gap_stats(P, blocked)
        att = attaining_runs(opens, gaps, N)
        hard = [r for r in att if r[2] < min(r[1], r[3])]
        out.write(f"\n===== m{top}: {len(att)} attaining runs v>=6 ({len(hard)} hard)\n")
        verify(gears, us, att, out)
        out.write(f"  real separations D_g = {[real_D(g) for g in gears]} "
                  f"(= round(g/3): {[round(g/3) for g in gears]}); "
                  f"range of D is 1..(g-1)/2 = {[(g-1)//2 for g in gears]}\n")
        for nm, sub in (("HARD", hard), ("ALL", att)):
            if not sub:
                continue
            cells = 0
            argmin = 0
            below = 0          # real P_both strictly below the mean over D
            ties = 0
            ranks = []
            real_tot = 0.0
            mean_tot = 0.0
            minD_tot = 0.0
            for (x0, L, v, R) in sub:
                for g in gears:
                    Ds = list(range(1, (g + 1) // 2))
                    ps = [p_both(g, D, L, v, R) for D in Ds]
                    Dr = real_D(g)
                    pr = p_both(g, Dr, L, v, R)
                    cells += 1
                    mn = min(ps)
                    if pr <= mn + 1e-12:
                        argmin += 1
                    mean = sum(ps) / len(ps)
                    if pr < mean - 1e-12:
                        below += 1
                    elif abs(pr - mean) <= 1e-12:
                        ties += 1
                    ranks.append(sum(1 for p in ps if p < pr - 1e-12) / len(ps))
                    real_tot += pr
                    mean_tot += mean
                    minD_tot += mn
            n = len(sub)
            out.write(f"  {nm} ({n} runs, {cells} (run,gear) cells): "
                      f"real D is the argmin of P(strike both) at {argmin}/{cells}; "
                      f"strictly below the mean over D at {below}/{cells} "
                      f"(equal at {ties})\n")
            out.write(f"      mean rank of the real D among all separations "
                      f"(0 = smallest P_both): {sum(ranks)/len(ranks):.3f}\n")
            out.write(f"      E[ov] per run: real teeth {real_tot/n:.3f}, "
                      f"random separation {mean_tot/n:.3f}, "
                      f"best possible separation {minD_tot/n:.3f} "
                      f"(gears: {len(gears)})\n")
        dump[str(top)] = dict(D_real=[real_D(g) for g in gears], gears=gears)
        out.flush()
    with open(os.path.join(RES, 'sep_teeth.json'), 'w') as f:
        json.dump(dump, f)


if __name__ == "__main__":
    dest = sys.argv[1] if len(sys.argv) > 1 else None
    o = open(dest, 'w') if dest else sys.stdout
    main(o)
    if dest:
        o.close()
