"""The third kind of field: for gear g, the composites made only of g and the lower gears.

Two readings, both computed:
  smooth field  C_g = { n in S composite : every prime factor of n is <= g }   (cumulative)
  layer field   L_g = { n in C_g : the largest prime factor of n is g } = C_g minus C_{prev(g)}
                    = g * (members of S with all factors <= g, at least 5)   (new at g)
In isolation, on [lo, hi): size, first member, last member, members below g^2 (echoes of
lower gears) and at/above g^2, class split, largest gap in S-steps, the exponent pattern
(how many members are g^a alone / g * one other prime / g * composite), and the count of
members of L_g per decade of the section.
Usage: uv run python smooth_fields.py lo hi
"""
import sys
from collections import Counter
from sympy import factorint, primerange


def main():
    lo, hi = int(sys.argv[1]), int(sys.argv[2])
    S = [n for n in range(lo, hi) if n % 6 in (1, 5)]
    pos = {n: i for i, n in enumerate(S)}
    fac = {n: factorint(n) for n in S}
    gears = list(primerange(5, int(hi ** 0.5) + 1))
    print(f"section [{lo}, {hi}): |S| = {len(S)}, gears {gears[0]}..{gears[-1]}")
    print("gear g | smooth field C_g size | layer L_g size | first of L_g | last of L_g | L_g below g^2 / at or above | left/right | largest gap of L_g (S-steps) | g^a alone / g*prime / g*composite")
    for g in gears[:10]:
        C = [n for n in S if len(fac[n]) and sum(fac[n].values()) >= 2 and max(fac[n]) <= g]
        L = [n for n in C if max(fac[n]) == g]
        if not L:
            print(f"{g} | {len(C)} | 0"); continue
        below = sum(1 for n in L if n < g * g)
        left = sum(1 for n in L if n % 6 == 5)
        gaps = [pos[b] - pos[a] for a, b in zip(L, L[1:])]
        kinds = Counter()
        for n in L:
            f = fac[n]; m = n // g
            if f == {g: f[g]}: kinds["g^a"] += 1
            elif sum(factorint(m).values()) == 1: kinds["g*prime"] += 1
            else: kinds["g*composite"] += 1
        print(f"{g} | {len(C)} | {len(L)} | {L[0]} = {dict(fac[L[0]])} | {L[-1]} | {below} / {len(L)-below} | {left}/{len(L)-left} | {max(gaps) if gaps else '-'} | {kinds['g^a']} / {kinds['g*prime']} / {kinds['g*composite']}")


if __name__ == "__main__":
    main()
