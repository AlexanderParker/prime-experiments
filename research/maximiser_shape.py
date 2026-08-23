"""Harvester r20b: is the delta-profile a SIGNATURE of the maximisers?

delta_q(e) = min(e mod q, q - e mod q): the separation of gear q's two blocked
residues (delta=1 clustered, delta ~ q/2 maximally spread; twins have delta=1
everywhere). Tests precision/recall: of the differences attaining max F, how many
share one profile; and of all differences with that profile, how many attain max F.
"""
import numpy as np
from math import prod, gcd
from collections import Counter

def F_all(gears):
    P = prod(gears); buf = np.empty(P, bool)
    F = np.zeros(P // 2 + 1, np.int32)
    for e in range(1, P // 2 + 1):
        buf[:] = True
        for q in gears:
            buf[0::q] = False
            buf[(-e) % q::q] = False
        idx = np.flatnonzero(buf)
        g = np.diff(np.append(idx, idx[0] + P))
        F[e] = int(g.max())
    return P, F

def prof(e, gears):
    return tuple(min(e % q, q - e % q) for q in gears)

for gears in ([3,5,7,11], [3,5,7,11,13], [3,5,7,11,13,17]):
    P, F = F_all(gears)
    mx = int(F.max())
    winners = [e for e in range(1, P//2+1) if F[e] == mx]
    pc = Counter(prof(e, gears) for e in winners)
    top_prof, top_n = pc.most_common(1)[0]
    same = [e for e in range(1, P//2+1) if prof(e, gears) == top_prof]
    hit = sum(1 for e in same if F[e] == mx)
    maxdelta = tuple((q - 1)//2 for q in gears)
    print(f"gears {gears}  P={P}  maxF={mx}  winners={len(winners)}")
    print(f"  distinct delta-profiles among winners: {len(pc)}; "
          f"most common {top_prof} covers {top_n}/{len(winners)}")
    print(f"  max possible delta per gear: {maxdelta}")
    print(f"  RECALL  : {100*top_n/len(winners):5.1f}% of winners have that profile")
    print(f"  PRECISION: {hit}/{len(same)} = {100*hit/len(same):5.1f}% of differences "
          f"with that profile are winners")
    print(f"  all winner profiles: {sorted(pc)[:6]}")
    print(f"  twin profile {prof(1, gears)} -> F={F[1]} (max is {mx})\n")
