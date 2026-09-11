"""Caustics of the square origin: the bands of gears above sqrt(g) strike the new range along
parabolas in h with apexes at the harmonics g/(3q).

Gear h with quotient q = floor(g/h) and remainder r = g - q h has g^2 = r^2 (mod h); its first
strike above the square is at e = s h - r^2 for the least s >= 1 with e > 0 and g^2 + e in S;
as h runs through the primes of band q (g/(q+1) < h <= g/q), r = g - q h and e is a quadratic
in h with vertex near h = g/q where e = s g / q, i.e. offset s g / (6 q); for s = 2 the apex sits
at offset g/(3q). Near the apex the first strikes of the band pile up (density ~ 1/sqrt of the
distance below the apex): a caustic. Prediction: the twins of the new range are depleted just
below the offsets g/3, g/6, g/9 (the apexes for s = 2) relative to elsewhere.
This script, for primes g in [gmin, gmax]: (1) verifies the parabola (first strike of every
gear above sqrt(g) equals s h - r^2 with the least valid s); (2) histograms the first-strike
offsets of band-q gears in units of the apex (offset / (g/(3q))) for q = 1, 2, 3; (3) measures
the twin density in the offset window [apex - w, apex] against the whole range, for w = g/30,
per q, pooled over g.
Usage: uv run python caustics.py gmin gmax
"""
import sys, math
from collections import Counter
import numpy as np
from sympy import primerange, nextprime, isprime


def first_strike(g, h):
    q = g // h; r = g - q * h
    s = 1
    while True:
        e = s * h - r * r
        if e > 0 and (g * g + e) % 6 in (1, 5):
            return e, s
        s += 1


def main():
    gmin, gmax = int(sys.argv[1]), int(sys.argv[2])
    ok = bad = 0
    hist = {q: Counter() for q in (1, 2, 3)}
    dep = {q: [0, 0] for q in (1, 2, 3)}; base = [0, 0]
    for g in primerange(gmin, gmax + 1):
        gp = nextprime(g); a = (g * g - 1) // 6; L = (gp * gp - g * g) // 6
        tw = set(i for i in range(1, L + 1) if isprime(6 * (a + i) - 1) and isprime(6 * (a + i) + 1))
        base[0] += len(tw); base[1] += L
        for h in primerange(int(math.isqrt(g)) + 1, g):
            e, s = first_strike(g, h)
            # verify by brute force
            e2 = 2
            while not ((g * g + e2) % h == 0 and (g * g + e2) % 6 in (1, 5)): e2 += 2
            if e == e2: ok += 1
            else: bad += 1
            q = g // h
            if q in hist and s == 2:
                apex = g / (3 * q); off = e / 6
                hist[q][round(10 * off / apex)] += 1
        for q in (1, 2, 3):
            apex = g / (3 * q); w = max(2, int(g / 30))
            lo_i, hi_i = max(1, int(apex - w)), min(L, int(apex))
            if hi_i > lo_i:
                dep[q][0] += sum(1 for i in range(lo_i, hi_i + 1) if i in tw); dep[q][1] += hi_i - lo_i + 1
    print(f"primes g in [{gmin}, {gmax}]: parabola formula against brute force: {ok} agree, {bad} differ")
    for q in (1, 2, 3):
        tot = sum(hist[q].values())
        print(f"band q={q} (gears in (g/{q+1}, g/{q}]), s = 2 first strikes, offset in tenths of the apex g/(3q): " + ", ".join(f"{k/10:.1f}: {100*v/tot:.1f}%" for k, v in sorted(hist[q].items()) if v / tot >= 0.02))
    bd = base[0] / base[1]
    for q in (1, 2, 3):
        d = dep[q][0] / max(dep[q][1], 1)
        print(f"twin density in [g/(3q) - g/30, g/(3q)] for q={q}: {100*d:.3f}% against the whole range {100*bd:.3f}% (ratio {d/bd:.3f}, n = {dep[q][1]} columns)")


if __name__ == "__main__":
    main()
