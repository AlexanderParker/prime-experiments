"""Harvester r20: targeted lower bound on h_2 at gears <= 19, and the twin trajectory.

Round-20 structure finding: h_2 maximisers have delta_q = min(e mod q, q - e mod q)
CLUSTERED (=1) on small gears and MAXIMALLY SPREAD (~q/2) on the top gears, and they
PERSIST across machines (y=13 champions land at the 99.3-99.8th percentile at y=17).
So instead of scanning all 2,424,922 differences at y=19 (infeasible), we LIFT the
y=17 elite: e = champion + t*255255, t = 0..18, choosing every residue mod 19.
This gives a rigorous LOWER bound on h_2(19), which is all that is needed to test
Ziller-Morack Conjecture 6 (h_2 < 19^2-19 = 342) from below.
"""
import numpy as np
from math import prod, gcd

G17 = [3, 5, 7, 11, 13, 17]
G19 = G17 + [19]
P17, P19 = prod(G17), prod(G19)

def F_of(gears, e, P, buf):
    buf[:] = True
    for q in gears:
        buf[0::q] = False
        buf[(-e) % q::q] = False
    idx = np.flatnonzero(buf)
    g = np.diff(np.append(idx, idx[0] + P))
    return int(g.max())

# twin trajectory: percentile of e=1 at each machine
print("TWIN TRAJECTORY (percentile of the twin difference within its own family):")
for y, gears in ((7,[3,5,7]), (11,[3,5,7,11]), (13,[3,5,7,11,13]), (17,G17)):
    P = prod(gears); buf = np.empty(P, bool)
    F = np.array([F_of(gears, e, P, buf) for e in range(1, P//2+1)])
    cop = np.array([F[e-1] for e in range(1, P//2+1) if gcd(e, P) == 1])
    t = F[0]
    print(f"  y={y:>2}: twin F={t:>3}  family max={F.max():>3}  "
          f"max/twin={F.max()/t:4.2f}  percentile(all)={100*(F<t).mean():5.1f}%  "
          f"percentile(coprime)={100*(cop<t).mean():5.1f}%")

# elite at y=17
buf17 = np.empty(P17, bool)
F17 = np.array([F_of(G17, e, P17, buf17) for e in range(1, P17//2+1)])
elite = (np.argsort(-F17)[:60] + 1).tolist()
print(f"\ny=17 elite: top-60 differences, F from {F17.max()} down to "
      f"{F17[np.argsort(-F17)[59]]}")

buf19 = np.empty(P19, bool)
best, arg = 0, None
for c in elite:
    for t in range(19):
        e = c + t * P17
        if e > P19 // 2:
            e = P19 - e
        f = F_of(G19, e, P19, buf19)
        if f > best:
            best, arg = f, e
print(f"\nLIFTED SEARCH at y=19 (P = {P19:,}; {len(elite)*19} targeted candidates "
      f"of {P19//2:,} total):")
print(f"  best F found = {best} at e = {arg}   =>   h_2(19) >= {2*best}")
print(f"  bound 19^2-19 = {19*19-19}")
print(f"  Conjecture 6 at n=8: {'REFUTED' if 2*best >= 342 else 'not refuted by this bound'}"
      f"  (lower bound is {100*2*best/342:.1f}% of the bound)")
