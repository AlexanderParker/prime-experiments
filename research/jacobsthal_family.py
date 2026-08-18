"""Harvester round 19: the paired Jacobsthal function h_2 MEASURED across d.

Ziller-Morack (arXiv:1706.00317) define h_2(n) = j_2(p_n#) = the max over ALL even
differences of the maximal gap of {m : m and m+2e both coprime to p_n#}, and
conjecture (Conjecture 6) h_2(n) < p_n^2 - p_n for n >= 3. Per the corpus review
they compute NO values of h_2. This tool computes them exactly.

Frame: halved coordinates n (m = 2n+1); gear q blocks n = 0, -e mod q; gear 2 is
automatic. An integer-scale gap is TWICE the halved gap, so
    h_2 (integer scale) = 2 * max_e F_e (halved),
matching the corpus's maxgap printout "2F vs y^2-y-2".
Symmetry: e and P-e are reflections, so scanning e = 1..P/2 is exhaustive.
"""
import numpy as np
from math import prod, gcd

def F_of(gears, e, P):
    a = np.ones(P, bool)
    for q in gears:
        a[0::q] = False
        a[(-e) % q::q] = False
    idx = np.flatnonzero(a)
    if idx.size < 2:
        return 0
    g = np.diff(np.append(idx, idx[0] + P))
    return int(g.max())

SETS = [[3], [3, 5], [3, 5, 7], [3, 5, 7, 11], [3, 5, 7, 11, 13]]
print("gears          y      P    #e |   h_2 = 2*maxF   argmax e (d = 2e)     "
      "y^2-y   Conj.6")
rows = []
for gears in SETS:
    P = prod(gears)
    y = gears[-1]
    best, arg, worst, argw, prof = -1, [], 10**9, None, {}
    for e in range(1, P // 2 + 1):
        F = F_of(gears, e, P)
        if F > best:
            best, arg = F, [e]
        elif F == best:
            arg.append(e)
        if 0 < F < worst:
            worst, argw = F, e
        prof.setdefault(gcd(e, P), []).append(F)
    h2, bound = 2 * best, y * y - y
    rows.append((gears, y, P, best, arg, bound, prof, worst, argw))
    print(f"{str(gears):<14}{y:>3} {P:>6} {P//2:>6} | {h2:>13}   "
          f"e={arg[:3]} (d={[2*x for x in arg[:3]]})  {bound:>6}   "
          f"{'HOLDS' if h2 < bound else 'FAILS'}")

gears, y, P, best, arg, bound, prof, worst, argw = rows[-1]
print(f"\nd-PROFILE at gears {gears} (halved units; F_2 = twin case = {prof[1][0]}):")
print("  gcd(e,P)     n      Fmin    Fmean     Fmax   lambda   Fmax/lambda")
for g in sorted(prof):
    v = prof[g]
    dens = prod((q - (1 if g % q == 0 else 2)) / q for q in gears)
    lam = 1 / dens
    print(f"  {g:>7}  {len(v):>6}  {min(v):>6}  {np.mean(v):8.2f}  {max(v):>6}  "
          f"{lam:7.2f}   {max(v)/lam:9.2f}")
print(f"  overall max F = {best} at e = {arg[:5]}  |  min F = {worst} at e = {argw}")
