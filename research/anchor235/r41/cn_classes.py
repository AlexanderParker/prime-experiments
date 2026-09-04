"""R2.a.i.a.1.a item 5 - the covering residue classes, counted by brute force.

A cover {g_1..g_K} with phases r_j forces q^2 = r_j (mod g_j); q^2 = r has exactly 2 roots mod g
for r a nonzero QR, so each (cover, phase vector) is realised by exactly 2^K classes of q mod
P = prod g_j.  Here that is checked by brute force: enumerate every q mod P and count those whose
real phases (r_g = q^2 mod g) make the gear set cover every island of [1, d).
"""
import sys
from math import prod

def islands(d):
    return [i for i in range(1, d) if i % 35 in (5, 10, 12, 17)]

def check(d, gears):
    isl = islands(d)
    m = len(isl)
    P = prod(gears)
    # per gear: for each q mod g, the mask of islands struck
    masks = []
    for g in gears:
        u = pow(6, -1, g)
        mk = [0] * g
        for x in range(g):
            r = (x * x) % g
            if r == 0:
                continue
            a = ((2 - r) * u) % g
            b = ((-r) * u) % g
            v = 0
            for e, i in enumerate(isl):
                if i % g == a or i % g == b:
                    v |= 1 << e
            mk[x] = v
        masks.append(mk)
    full = (1 << m) - 1
    # enumerate q mod P by CRT product (P small)
    good = 0
    phase_ok = set()
    for q in range(P):
        v = 0
        ph = []
        for g, mk in zip(gears, masks):
            x = q % g
            v |= mk[x]
            ph.append((x * x) % g)
        if v == full:
            good += 1
            phase_ok.add(tuple(ph))
    return P, m, good, len(phase_ok)

for d, gears in [(70, [11, 23, 37, 127]), (140, [11, 17, 19, 23, 37, 107])]:
    P, m, good, nph = check(d, gears)
    K = len(gears)
    print("d=%d gears=%s  P=%d  m=%d islands" % (d, gears, P, m))
    print("   q-classes mod P realising the cover: %d ; distinct phase vectors: %d ; 2^K x phases = %d ; match=%s"
          % (good, nph, nph * 2 ** K, good == nph * 2 ** K))
    print("   density %.4g   vs 1/P = %.4g" % (good / P, 1 / P))
