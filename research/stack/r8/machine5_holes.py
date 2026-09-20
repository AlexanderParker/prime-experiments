"""Holes between gear 7's adjacent lap pairs. Gear 7 strikes laps 3, 4 mod 7; hole h is the five
laps 7h+5 .. 7h+9 (positions p = 1..5, lap = 7h + 4 + p). Gear g >= 11 (class a, i.e. laps
k = a mod g) strikes hole h at position p iff 7h + 4 + p = a mod g, i.e. h = 7^{-1}(a - 4 - p)
mod g: over g consecutive holes each class hits exactly five holes, at h0 - p * 7^{-1} (mod g)
with position p - an arithmetic progression of holes with difference 7^{-1} mod g, position
stepping by one. A hole is FILLED when positions 1..5 are all struck; a chain is a maximal set of
consecutive filled holes; the lap record is 7 * (chain) + 2 + partial ends.
Checks: (1) the progression law per gear; (2) filled holes and longest chain per period for
q = 11..23, and that 5 * chain + 2 * (chain + 1) <= record < that + 10.
"""
from sympy import primerange
from math import prod
ps = list(primerange(7, 60))
for g in [11, 13, 17, 19, 23]:
    inv7 = pow(7, -1, g)
    for a in (pow(30, -1, g), (-pow(30, -1, g)) % g):
        hits = {}
        for h in range(g):
            for p in range(1, 6):
                if (7 * h + 4 + p) % g == a: hits[h] = p
        # progression check: hole with position p+1 is the hole with position p minus 7^{-1}
        byp = {p: h for h, p in hits.items()}
        assert len(byp) == 5 and all((byp[p] - inv7) % g == byp[p + 1] for p in range(1, 5)), (g, a, byp)
        print(f"gear {g} class {a}: holes (h:pos) {dict(sorted(hits.items()))}, step 7^-1 = {inv7} mod {g}")
def hole_struck(h, gears):
    return [p for p in range(1, 6) if any((30 * (7*h+4+p) - 1) % g == 0 or (30 * (7*h+4+p) + 1) % g == 0 for g in gears)]
for q in [11, 13, 17, 19, 23]:
    gears = [g for g in ps if 11 <= g <= q]; P = prod(gears)  # hole period = product of gears >= 11
    filled = [h for h in range(P) if len(hole_struck(h, gears)) == 5]
    best = run = 0; where = None
    for h in range(P):
        run = run + 1 if len(hole_struck(h, gears)) == 5 else 0
        if run > best: best, where = run, h - run + 1
    print(f"q={q}: hole period {P}, filled holes {len(filled)} (check only), longest chain {best} at hole {where}")
    if where is not None:
        for h in range(where, where + best):
            print("   hole", h, "laps", 7*h+5, "..", 7*h+9, "positions struck by", [[g for g in gears if (30*(7*h+4+p)-1) % g == 0 or (30*(7*h+4+p)+1) % g == 0] for p in range(1, 6)])
