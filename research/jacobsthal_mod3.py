"""Harvester round 18: a complete mod-3 law for the paired-Jacobsthal family F_d.

F_d(y) = maximal gap of {n : n and n+e both coprime to the gears <= y} in halved
coordinates (the corpus's adjacent frame; F_2 is the corpus's F(2,y) = 21,33,54,...).

CLAIM (dichotomy):  3 | F_d(y) for every y  <=>  3 does not divide e.
Forward: gear 3 blocks n = 0 and n = -e mod 3, two DISTINCT classes when 3 does
not divide e, leaving ONE free class - so every survivor is congruent mod 3 and
every gap is divisible by 3. Converse: when 3 | e gear 3 blocks one class only,
two survive, and gaps of both non-zero residues occur (measured).
This generalises the kernel-checked twin case (Polignac.endpoint_run_mod_three).
"""
import numpy as np
from math import prod, gcd

def gaps_of(gears, e):
    P = prod(gears)
    n = np.arange(P)
    a = np.ones(P, bool)
    for q in gears:
        a[(-0) % q::q] = False
        a[(-e) % q::q] = False
    idx = np.flatnonzero(a)
    g = np.diff(np.append(idx, idx[0] + P))
    return g

print(" d    e  3|e |   F_d(y) for y = 11,13,17,19,23        all gaps div by 3?  3|F?")
for d in (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 30, 36, 42, 210):
    e = d // 2
    Fs, alldiv, Fdiv = [], True, True
    for y in (11, 13, 17, 19, 23):
        gears = [q for q in (3, 5, 7, 11, 13, 17, 19, 23) if q <= y]
        g = gaps_of(gears, e)
        F = int(g.max())
        Fs.append(F)
        alldiv &= bool(np.all(g % 3 == 0))
        Fdiv &= (F % 3 == 0)
    pred = (e % 3 != 0)
    ok = "OK" if (alldiv == pred and Fdiv == pred) else "MISMATCH"
    print(f"{d:>3} {e:>4}   {'Y' if e%3==0 else 'n'}  | {str(Fs):<34} "
          f"{str(alldiv):<6} {str(Fdiv):<6} {ok}")
