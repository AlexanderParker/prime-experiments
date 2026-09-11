"""The band kills of a new range as the lattice points of the hyperbolic strip
g^2 < x y < g'^2, x a prime in (sqrt g, g], y in S: how they fall across the strip.

For primes g in [gmin, gmax], every member n of the range with least prime factor x > sqrt(g)
is a band kill; n = x y with y = n / x. Measured, pooled: (i) the kills' distribution over the
offsets in tenths of L (uniform?); (ii) the share of kills by band q = floor(g / x), and the
number of distinct x per band; (iii) the cofactor y: prime or not, and its size relative to
q g (y - q g in units of g); (iv) local structure at the twins: the band-kill rate at the
offsets i - 1 and i + 1 of a twin i, against the overall band-kill rate per offset; (v) the
number of kills per x against the strip width 2 g d / x (the mean ratio).
Usage: uv run python band_products.py gmin gmax
"""
import sys, math
from collections import Counter, defaultdict
import numpy as np
from sympy import primerange, nextprime, isprime, factorint


def main():
    gmin, gmax = int(sys.argv[1]), int(sys.argv[2])
    tenths = Counter(); byq = Counter(); xs_byq = defaultdict(set); yprime = [0, 0]; ypos = defaultdict(list)
    tw_nb = [0, 0]; all_off = [0, 0]; perx = []
    for g in primerange(gmin, gmax + 1):
        gp = nextprime(g); a = (g * g - 1) // 6; L = (gp * gp - g * g - 2) // 6; d = gp - g; sq = math.isqrt(g)
        band_at = [False] * (L + 2); tw = []
        cnt_x = Counter()
        for i in range(1, L + 1):
            l, r = 6 * (a + i) - 1, 6 * (a + i) + 1
            pl, pr = isprime(l), isprime(r)
            if pl and pr: tw.append(i)
            for n, pp in ((l, pl), (r, pr)):
                if pp: continue
                x = min(factorint(n))
                if x > sq:
                    band_at[i] = True; y = n // x; q = g // x
                    tenths[min(9, (i - 1) * 10 // L)] += 1; byq[min(q, 6)] += 1; xs_byq[min(q, 6)].add(x)
                    yprime[0] += isprime(y); yprime[1] += 1
                    ypos[min(q, 6)].append((y - q * g) / g)
                    cnt_x[x] += 1
        for x, c in cnt_x.items(): perx.append(c / (2 * g * d / x))
        all_off[0] += sum(band_at[1:L + 1]); all_off[1] += L
        for i in tw:
            for j in (i - 1, i + 1):
                if 1 <= j <= L: tw_nb[0] += band_at[j]; tw_nb[1] += 1
    tot = sum(tenths.values())
    print(f"primes g in [{gmin}, {gmax}]: {tot} band kills (least factor above sqrt g)")
    print("(i) kills by tenth of the range:", [f"{100*tenths[k]/tot:.1f}%" for k in range(10)])
    print("(ii) share of kills by band q (6 = 6 and above):", {q: f"{100*byq[q]/tot:.1f}% ({len(xs_byq[q])} distinct x)" for q in sorted(byq)})
    print(f"(iii) cofactor y prime: {100*yprime[0]/yprime[1]:.1f}%; mean (y - q g)/g by band: " + ", ".join(f"q={q}: {np.mean(v):+.3f} (sd {np.std(v):.3f})" for q, v in sorted(ypos.items())))
    print(f"(iv) band-kill rate at a twin's neighbour offsets {100*tw_nb[0]/tw_nb[1]:.1f}% against the overall band-kill rate per offset {100*all_off[0]/all_off[1]:.1f}%")
    print(f"(v) kills per x against the strip width 2 g d / x: mean ratio {np.mean(perx):.3f} (expected the survivor density of S free of primes below x, about 1/3 x prod (1 - 1/p))")


if __name__ == "__main__":
    main()
