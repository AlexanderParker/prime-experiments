"""Waste in the record runs (node c.xiii follow-up). Waste = strikes landing inside a record run
beyond one per lap. Observed: 0, 1, 2 at q = 19, 23, 29, machines with 4, 5, 6 gears above 7.
PRE-REGISTERED prediction: waste = (number of gears 11..q) - 4, so 3 at q = 31 and 4 at q = 37.
Method (words only, no period scan): enumerate every maximal walk of record length; a gear left
unfixed by the walk contributes no waste iff it can avoid the whole run, which is possible iff the
run is shorter than the gear's larger class gap (g - d_g); otherwise it must be fixed. Waste of a
walk = (strikes of the fixed gears inside the run) - run length, minimised over the walks.
"""
import sys
from sympy import primerange
sys.setrecursionlimit(10000)

def records(gears, cap=None):
    cls = {g: (pow(30, -1, g), (-pow(30, -1, g)) % g) for g in gears}
    best = [0]
    sols = []

    def strikes(fixed, i):
        return any((r + i) % g in cls[g] for g, r in fixed.items())

    def walk(fixed, i):
        while strikes(fixed, i):
            i += 1
        if i > best[0]:
            best[0] = i
            sols.clear()
        if i == best[0] and (cap is None or len(sols) < cap):
            sols.append(dict(fixed))
        for g in [g for g in gears if g not in fixed]:
            for c in cls[g]:
                fixed[g] = (c - i) % g
                walk(fixed, i)
                del fixed[g]

    walk({}, 0)
    return best[0], sols, cls


for q in [19, 23, 29, 31]:
    gears = list(primerange(7, q + 1))
    L, sols, cls = records(gears)
    above7 = [g for g in gears if g > 7]
    best_waste = None
    for ph in sols:
        n = sum(1 for g in ph for i in range(L) if (ph[g] + i) % g in cls[g])
        free = [g for g in gears if g not in ph]
        # a free gear avoids the run iff L <= its larger class gap g - d_g; else it must add at least one strike
        forced = 0
        for g in free:
            d = pow(15, -1, g)
            gap = max(d, g - d)
            if L > gap:
                forced += 1
        w = n + forced - L
        if best_waste is None or w < best_waste:
            best_waste = w
    print(f"q={q:2d}: record {L:2d}, gears above 7: {len(above7)}, minimal waste {best_waste}, "
          f"predicted {len(above7) - 4}", flush=True)
