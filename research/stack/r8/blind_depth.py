"""Blind set to all gears up to a cut c, as c grows toward g: density, twin enrichment, and
whether the first twin of the new range is blind to every gear up to some c(g).

For prime g with new range (g^2, g'^2), L = (g'^2 - g^2)/6 columns, offset i is blind to gear h
iff -6i and 2 - 6i are both non-squares mod h (never struck from any square origin). The blind
set to the cut c = the offsets blind to every gear 5 <= h <= c. Per cut c in a list: the share
of offsets below L that are blind to c (averaged over steps), the share of twins on them, the
enrichment, and the count of steps in which at least one twin is blind to c. Also, per step,
the largest cut c*(g) such that the FIRST twin is blind to every gear <= c* (the depth of the
first twin), summarised as a distribution.
Usage: uv run python blind_depth.py gmin gmax
"""
import sys
from collections import Counter
from sympy import primerange, nextprime, isprime


def blind_classes(h):
    sq = {(x * x) % h for x in range(1, h)}
    return frozenset(i for i in range(h) if ((-6 * i) % h not in sq) and ((2 - 6 * i) % h not in sq))


def main():
    gmin, gmax = int(sys.argv[1]), int(sys.argv[2])
    gears_all = list(primerange(5, gmax + 1))
    B = {h: blind_classes(h) for h in gears_all}
    cuts = [7, 13, 23, 37, 53, 79, 113, 163, 229]
    off_tot = 0; tw_tot = 0; blind_off = Counter(); blind_tw = Counter(); steps_with = Counter(); steps = 0
    depth = Counter()
    for g in primerange(gmin, gmax + 1):
        gp = nextprime(g); a = (g * g - 1) // 6; L = (gp * gp - g * g) // 6; steps += 1
        gears = [h for h in gears_all if h <= g]
        tw = [i for i in range(1, L + 1) if isprime(6 * (a + i) - 1) and isprime(6 * (a + i) + 1)]
        off_tot += L; tw_tot += len(tw)
        # blindness depth of each offset: the largest gear index such that blind to all gears up to it (prefix)
        def depth_of(i):
            d = 0
            for h in gears:
                if i % h in B[h]: d = h
                else: break
            return d
        for c in cuts:
            if c > g: continue
            cnt = sum(1 for i in range(1, L + 1) if depth_of(i) >= c)
            ctw = sum(1 for i in tw if depth_of(i) >= c)
            blind_off[c] += cnt; blind_tw[c] += ctw; steps_with[c] += (ctw > 0)
        if tw: depth[depth_of(tw[0])] += 1
    print(f"primes g in [{gmin}, {gmax}]: {steps} steps, {off_tot} offsets, {tw_tot} twins")
    print("cut c | share of offsets blind to every gear <= c | share of twins on them | enrichment | steps with a twin blind to c (of steps where c <= g)")
    for c in cuts:
        if blind_off[c] == 0: continue
        so = blind_off[c] / off_tot; st = blind_tw[c] / tw_tot
        n_steps = sum(1 for g in primerange(gmin, gmax + 1) if c <= g)
        print(f"{c} | {so:.4f} | {st:.4f} | {st/so:.2f} | {steps_with[c]} of {n_steps}")
    print("depth of the FIRST twin (largest gear it is blind to as a prefix from 5): " + ", ".join(f"{d}: {n}" for d, n in sorted(depth.items())))


if __name__ == "__main__":
    main()
