"""Harvester round 20: WHY IS 13 EXTREMAL? Anatomy of the h_2 maximisers.

For each machine (gears 3..y) we compute F_e for every even difference and ask:
 - which e attain the max, and what is their SHAPE?
 - shape statistic: delta_q = min(e mod q, q - e mod q), the separation of gear q's
   two blocked residues. Twins (e=1) have delta_q = 1 for EVERY gear - maximally
   CLUSTERED blocking. Large delta_q = spread blocking.
 - does a maximiser at one machine stay extremal at the next?
"""
import numpy as np
from math import prod, gcd

def F_of(gears, e, P, buf):
    buf[:] = True
    for q in gears:
        buf[0::q] = False
        buf[(-e) % q::q] = False
    idx = np.flatnonzero(buf)
    g = np.diff(np.append(idx, idx[0] + P))
    return int(g.max())

def profile(e, gears):
    return [min(e % q, q - e % q) for q in gears]

SETS = {7: [3,5,7], 11: [3,5,7,11], 13: [3,5,7,11,13], 17: [3,5,7,11,13,17]}
res = {}
for y, gears in SETS.items():
    P = prod(gears)
    buf = np.empty(P, bool)
    F = np.zeros(P // 2 + 1, np.int32)
    for e in range(1, P // 2 + 1):
        F[e] = F_of(gears, e, P, buf)
    res[y] = (gears, P, F)
    top = np.argsort(-F[1:])[:6] + 1
    cop = np.array([e for e in range(1, P//2+1) if gcd(e, P) == 1])
    md = np.array([np.mean(profile(int(e), gears)) for e in cop])
    fc = F[cop]
    r = np.corrcoef(md, fc)[0, 1]
    print(f"y={y:>3} P={P:>7}  maxF={F.max():>3}  twin F_1={F[1]:>3}  "
          f"mean(coprime)={fc.mean():6.2f}")
    print(f"      top e: {[int(t) for t in top]}")
    for t in top[:3]:
        print(f"        e={int(t):>6}  F={F[t]:>3}  delta profile {profile(int(t), gears)}"
              f"  mean {np.mean(profile(int(t), gears)):.2f}")
    print(f"        e=     1  F={F[1]:>3}  delta profile {profile(1, gears)}  mean "
          f"{np.mean(profile(1, gears)):.2f}   <- TWINS (maximally clustered)")
    print(f"      corr(mean delta, F) over coprime class = {r:+.3f}")

print("\nDOES A MAXIMISER PERSIST? (F of one machine's champions at the next machine)")
for ysmall, ybig in ((11, 13), (13, 17)):
    gs, Ps, Fs = res[ysmall]
    gb, Pb, Fb = res[ybig]
    champs = (np.argsort(-Fs[1:])[:5] + 1).tolist()
    buf = np.empty(Pb, bool)
    print(f"  champions of y={ysmall} (F={Fs.max()}) evaluated at y={ybig} "
          f"(maxF there = {Fb.max()}):")
    for c in champs:
        fb = F_of(gb, int(c), Pb, buf)
        pct = 100 * (Fb[1:] < fb).mean()
        print(f"    e={int(c):>6}: F_{ysmall}={Fs[c]:>3} -> F_{ybig}={fb:>3} "
              f"({pct:5.1f}th percentile at y={ybig})")
