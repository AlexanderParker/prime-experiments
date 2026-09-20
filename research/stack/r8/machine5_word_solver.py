"""Lap record from the words alone (node c.xi follow-up). A run of laps k0, k0+1, ... is struck
when every lap has a gear whose class it lies in; fixing which class of gear g strikes lap i fixes
k0 mod g. Search: walk the laps; if a fixed gear already strikes lap i, advance; else branch over
the unfixed gears and their two classes (each fixes k0 mod g); a leaf is a lap no fixed gear
strikes with no gear left. The longest walk is the record, by CRT (every phase tuple occurs).
PRE-REGISTERED: returns 1, 2, 4, 5, 7, 12, 18, 25 for q = 5(none), 7, 11, 13, 17, 19, 23, 29,
matching the scans; then q = 31 (new exact value, prediction from the partial-period chain of 4:
at least 5 x 4 + 2 x 5 = 30 plus ends).
"""
import sys
sys.setrecursionlimit(10000)
from sympy import primerange
def record(gears):
    cls = {g: (pow(30, -1, g), (-pow(30, -1, g)) % g) for g in gears}
    best = [0, None]
    def strikes(fixed, i):
        return any((r + i) % g in cls[g] for g, r in fixed.items())
    def walk(fixed, i):
        while strikes(fixed, i): i += 1
        if i > best[0]: best[0], best[1] = i, dict(fixed)
        free = [g for g in gears if g not in fixed]
        for g in free:
            for c in cls[g]:
                fixed[g] = (c - i) % g
                walk(fixed, i)
                del fixed[g]
    walk({}, 0)
    return best
for q in [7, 11, 13, 17, 19, 23, 29, 31]:
    gears = list(primerange(7, q + 1))
    L, phases = record(gears)
    print(f"q={q}: lap record {L} from the words alone; phases k0 mod g = {phases}", flush=True)
