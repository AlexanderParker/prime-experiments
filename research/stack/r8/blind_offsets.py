"""Blind gears across offsets. Gear h is blind for offset i iff neither -(6i - 2) nor -6i is a
square mod h, i.e. the pair (-6i, -6i + 2) is a pair of non-residues at distance 2. As i runs
mod h, -6i runs over every class, so gear h is blind for exactly as many offset classes mod h
as there are non-residue pairs (x, x + 2) mod h. Reported: per gear h to hmax, the number of
blind offset classes mod h and its share; whether any gear is blind on every offset of a class
(i = c mod 35, the blind classes for 5 and 7): impossible when h is coprime to 35 since i mod h
then runs over everything, checked anyway; and the exact count against the formula from the
Legendre sums: pairs of non-residues at distance 2 = (h - 3 - (-1|h) - 2(2|h) ... ) / 4, given
here only as the measured count so the reader can fit it.
Usage: uv run python blind_offsets.py hmax
"""
import sys
from sympy import primerange, legendre_symbol


def main():
    hmax = int(sys.argv[1]); gears = list(primerange(5, hmax + 1))
    print("gear h | blind offset classes mod h | share | (-1|h) (2|h) (-2|h)")
    tot = 0
    for h in gears:
        nr = [x for x in range(1, h) if legendre_symbol(x, h) == -1]
        S = set(nr)
        blind = sum(1 for x in nr if (x + 2) % h in S)   # pair (x, x+2) both non-residues, x = -6i
        tot += blind / h
        if h <= 60 or h in (101, 211, 401, 1009, 2003):
            print(f"{h} | {blind} | {blind/h:.3f} | {legendre_symbol(h-1, h)} {legendre_symbol(2, h)} {legendre_symbol(h-2, h)}")
    print(f"mean share of blind offset classes over gears to {hmax}: {tot/len(gears):.4f} (a quarter of the offsets per gear)")
    # is any gear blind on every offset of a class mod 35?
    for c in (5, 10, 12, 17):
        always = []
        for h in gears:
            if h in (5, 7): continue
            ok = True
            for k in range(h):
                i = c + 35 * k
                a, b = (-(6 * i - 2)) % h, (-6 * i) % h
                if legendre_symbol(a, h) != -1 or legendre_symbol(b, h) != -1 or a == 0 or b == 0: ok = False; break
            if ok: always.append(h)
        print(f"gears blind on every offset i = {c} mod 35: {always}")


if __name__ == "__main__":
    main()
