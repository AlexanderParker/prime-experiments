"""Locations, not densities: every structure's kill set in the new range written as explicit
offsets from the origin, generated from formulas alone (no factoring), and checked against
the factoring map.

Range (g^2, g'^2), offsets i = 1 .. L from the square column a = (g^2 - 1)/6. Formulas:
  small and middle gear h:   i = c0_h + j h  and  i = c2_h + j h,   j >= 0,
        c0_h = (-g^2) 6^{-1} mod h,  c2_h = (2 - g^2) 6^{-1} mod h        (offset-strike law);
  band gear x > sqrt g, q = floor(g/x), r = g - q x:  i = (s x - r^2)/6 + j x,  j >= 0,
        for the s >= 1 with s x - r^2 > 0 and (g^2 + s x - r^2) in S (two classes of s mod 6);
        (the same set as the gear formula above; written this way it is the band parabola)
  the newest gear g: the band formula with q = 1, r = 0: i = column(g^2 + s g) - a + j g for the two classes of s (g (g+2) or g (g+4), then g (g+6) ...).
The twins are the offsets in no set. Usage: uv run python locations.py g
"""
import sys, math
from sympy import primerange, nextprime, isprime, mod_inverse


def gear_offsets(g, h, L):
    inv = int(mod_inverse(6, h))
    c0 = int((-g * g * inv) % h); c2 = int(((2 - g * g) * inv) % h)
    out = set()
    for c in (c0, c2):
        i = c if c > 0 else h
        while i <= L: out.add(i); i += h
    return out, c0, c2


def band_offsets(g, x, L):
    q = g // x; r = g - q * x; out = set(); first = None
    a = (g * g - 1) // 6
    s0 = (r * r) // x + 1
    for s in range(s0, s0 + 6):
        e = s * x - r * r; n = g * g + e
        if e > 0 and n % 6 in (1, 5):
            i = ((n + 1) // 6 if n % 6 == 5 else (n - 1) // 6) - a
            if first is None or i < first[1]: first = (s, i)
            while i <= L: out.add(i); i += x
    return out, q, r, first


def main():
    g = int(sys.argv[1]); gp = nextprime(g); a = (g * g - 1) // 6; L = (gp * gp - g * g - 2) // 6; sq = math.isqrt(g)
    print(f"g = {g}, origin column {a}, L = {L} offsets; g^2 mod 5, 7, 11, 13 = {g*g%5}, {g*g%7}, {g*g%11}, {g*g%13}")
    killed = set()
    print("small and middle gears (residues c0, c2 mod h, then the offsets):")
    for h in primerange(5, sq + 1):
        s, c0, c2 = gear_offsets(g, h, L); killed |= s
        print(f"  h={h}: i = {c0} + {h}j, {c2} + {h}j -> {sorted(s)}")
    print("band gears x in (sqrt g, g): q, r, first (s, offset), then the offsets (+ x each):")
    for x in primerange(sq + 1, g):
        s, q, r, first = band_offsets(g, x, L)
        if s: killed |= s; print(f"  x={x}: q={q}, r={r}, e = s*{x} - {r*r}: first s={first[0]} at offset {first[1]} -> {sorted(s)}")
    # newest gear
    s, _, _, _ = band_offsets(g, g, L)  # the newest gear: q = 1, r = 0, both teeth
    killed |= s
    print(f"the newest gear {g}: first at offset {min(s) if s else None} -> {sorted(s)}")
    twins_formula = [i for i in range(1, L + 1) if i not in killed]
    twins_true = [i for i in range(1, L + 1) if isprime(6 * (a + i) - 1) and isprime(6 * (a + i) + 1)]
    print(f"offsets in no set (the formulas' twins): {twins_formula}")
    print(f"twins by primality:                      {twins_true}   agree: {twins_formula == twins_true}")


if __name__ == "__main__":
    main()
