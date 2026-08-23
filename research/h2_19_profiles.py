"""Harvester r20c: a COMPLETE (not sampled) result at gears <= 19 on a named class.

Round-20 finding: the differences attaining the family maximum are EXACTLY those
carrying one or two delta-profiles (100% precision at every machine computed).
That converts the intractable y=19 scan into a complete statement about a
precisely-defined class: enumerate, EXHAUSTIVELY, every difference whose
gears-<=17 shape is one of the two gears-<=17 OPTIMAL shapes, plus the
maximal-spread shape, and evaluate all of them.

delta_q(e) = min(e mod q, q - e mod q); a profile fixes e mod q up to sign, so a
7-gear profile carries 2^7 = 128 residues, i.e. 64 differences up to reflection.
"""
import numpy as np
from math import prod
from itertools import product

G = [3, 5, 7, 11, 13, 17, 19]
P = prod(G)
buf = np.empty(P, bool)

def F_of(e):
    buf[:] = True
    for q in G:
        buf[0::q] = False
        buf[(-e) % q::q] = False
    idx = np.flatnonzero(buf)
    g = np.diff(np.append(idx, idx[0] + P))
    return int(g.max())

def crt(res):
    e, M = 0, 1
    for r, q in zip(res, G):
        while e % q != r % q:
            e += M
        M *= q
    return e

def diffs_of_profile(prof):
    """all differences (up to reflection) with this delta-profile"""
    out = set()
    for signs in product(*[(1, -1)] * len(G)):
        e = crt([s * d for s, d in zip(signs, prof)]) % P
        if e == 0:
            continue
        out.add(min(e, P - e))
    return sorted(out)

FAMILIES = {
    "E1 = extensions of gears<=17 winner (1,1,2,4,6,8)":
        [(1, 1, 2, 4, 6, 8, d) for d in range(1, 10)],
    "E2 = extensions of gears<=17 winner (1,1,2,3,4,3)":
        [(1, 1, 2, 3, 4, 3, d) for d in range(1, 10)],
    "M  = maximal spread on every gear above 5":
        [(1, 1, 3, 5, 6, 8, 9)],
    "T  = the twin difference itself":
        [(1, 1, 1, 1, 1, 1, 1)],
}
print(f"gears {G}, P = {P:,}; complete enumeration of each named class\n")
grand, grand_e, grand_p = 0, None, None
for name, profs in FAMILIES.items():
    best, barg, bprof, n = 0, None, None, 0
    for pr in profs:
        for e in diffs_of_profile(pr):
            n += 1
            f = F_of(e)
            if f > best:
                best, barg, bprof = f, e, pr
    print(f"{name}")
    print(f"   {n:>5} differences, ALL evaluated: max F = {best:>3} at e = {barg} "
          f"(profile {bprof})")
    if best > grand:
        grand, grand_e, grand_p = best, barg, bprof
print(f"\nCOMPLETE OVER THESE CLASSES: max F = {grand} at e = {grand_e}, "
      f"profile {grand_p}")
print(f"  => h_2(19) >= {2*grand};  bound 19^2-19 = {19*19-19};  "
      f"lower bound is {100*2*grand/342:.1f}% of the bound")
print(f"  Conjecture 6 at gears <= 19: "
      f"{'REFUTED' if 2*grand >= 342 else 'NOT refuted by this class'}")
