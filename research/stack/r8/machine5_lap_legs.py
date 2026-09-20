"""Lap machine: which gear pairs laps at which distance, and the shape of the record runs.
Lap k = copy (30k-1, 30k+1); gear g strikes lap k iff 30k = -+1 mod g. Leg rule on laps: gear g
strikes two laps at distance D iff g | D (same member) or g | 15D - 1 or g | 15D + 1 (opposite
members; 30(k+D) - 30k = 30D = +-2 mod g). Original machine: g | 3D -+ 1. Report the lap class
distance per gear, the gears pairing laps at each distance D, and the record runs of the lap
machine for q = 11..23 with the striking gear of each lap.
"""
from sympy import primerange, factorint
from math import prod
ps = list(primerange(7, 100))
print("lap class distance per gear (15^-1 mod g, nearer representative):")
print("  " + ", ".join(f"{g}:{min(pow(15,-1,g), g-pow(15,-1,g))}" for g in ps if g <= 61))
print("original class distance (3^-1 mod g, nearer representative):")
print("  " + ", ".join(f"{g}:{min(pow(3,-1,g), g-pow(3,-1,g))}" for g in ps if g <= 61))
print("gears pairing laps at distance D (prime factors >= 7 of 15D-1, 15D+1, and of D):")
for D in range(1, 13):
    legs = sorted({p for m in (15*D-1, 15*D+1, D) for p in factorint(m) if p >= 7})
    print(f"  D={D:2d}: {legs}   (15D-1={15*D-1}, 15D+1={15*D+1})")
# check the leg rule on laps directly
for g in ps[:8]:
    hits = [k for k in range(1, 3*g) if (30*k-1) % g == 0 or (30*k+1) % g == 0]
    ds = sorted({b-a for a in hits for b in hits if b > a and b-a < g})
    ok = all((15*D-1) % g == 0 or (15*D+1) % g == 0 for D in ds)
    assert ok, (g, ds)
print("leg rule on laps verified for gears 7..37")
def striker(k, gears):
    return [g for g in gears if (30*k-1) % g == 0 or (30*k+1) % g == 0]
for q in [11, 13, 17, 19, 23]:
    gears = [g for g in ps if g <= q]; P = prod(gears)
    best = (0, 0); run = 0
    for k in range(1, P + 1):
        run = run + 1 if striker(k, gears) else 0
        if run > best[0]: best = (run, k - run + 1)
    L, a = best
    print(f"q={q}: record run {L} laps at laps {a}..{a+L-1}:")
    for k in range(a, a + L):
        print(f"    lap {k:6d}  (pair {30*k-1}, {30*k+1})  struck by {striker(k, gears)}")
