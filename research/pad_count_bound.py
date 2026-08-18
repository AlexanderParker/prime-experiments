"""Harvester r15 chunk 2: does the padded-link COUNT bound p <= ~F/q' transfer to general d?

Mechanism: each padded link consumes one M-gap = 0 mod q', so it contributes at
least c_d to the run's span, where (round-15 reconciliation, MEMBER units)
    c_d = 6q'  if 3 does not divide e      c_d = 2q'  if 3 | e.
A run of span <= F(M+q') therefore carries p <= F(M+q') / c_d padded links.
Measured here against actual legal runs.
"""
import numpy as np

def surv(gears, e, P):
    n = np.arange(P); a = np.ones(P, bool)
    for q in gears:
        a &= (n % q != 0) & (n % q != (-e) % q)
    return np.flatnonzero(a)

def maxpad(e, gears, q1, KMAX=9):
    P = 1
    for q in gears: P *= q
    o = surv(gears, e, P); m = len(o)
    gaps = np.diff(np.append(o, o[0] + P))
    t = []
    for g in gaps:
        r = int(g) % q1
        t.append('Z' if r == 0 else ('P' if r == e % q1 else ('N' if r == (-e) % q1 else None)))
    Pn = P * q1; on = surv(list(gears) + [q1], e, Pn)
    Fnew = int(np.diff(np.append(on, on[0] + Pn)).max())
    best_z = 0
    for i in range(m):
        last, z = None, 0
        for k in range(1, KMAX + 1):
            ty = t[(i + k - 1) % m]
            if ty is None: break
            if ty == 'Z': z += 1
            else:
                if last is not None and ty == last: break
                last = ty
            best_z = max(best_z, z)
    # member units: halved -> member is x2
    c_d = (2 * q1) if e % 3 == 0 else (6 * q1)
    return Fnew * 2, best_z, c_d, (Fnew * 2) / c_d

print("d   gears                 q'  F(M+q') members  max padded p  bound F/c_d  c_d")
for d, gears, q1 in [(2,[3,5,7,11,13],17), (2,[3,5,7,11,13,17],19),
                     (4,[3,5,7,11,13,17],19),
                     (6,[3,5,7,11,13],17), (6,[3,5,7,11,13,17],19),
                     (12,[3,5,7,11],13), (12,[3,5,7,11,13,17],19),
                     (30,[3,5,7,11,13],17)]:
    F, z, c, b = maxpad(d // 2, gears, q1)
    print(f"{d:>2}  {str(gears):<21}{q1:>3}  {F:>13}  {z:>12}  {b:>10.2f}  {c:>4}"
          f"   {'OK' if z <= b + 1e-9 else 'VIOLATED'}")
