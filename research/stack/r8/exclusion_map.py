"""The exclusion map of a new range: every structure on record, applied in order, and the
twins as what no structure touches.

Order of structures (each an exact description of a set of offsets):
  1. the small wheel: the teeth of the gears 5 .. 13 at their squared phases
     (offset i struck by h iff g^2 + 6i = 0 or 2 mod h) - a periodic comb, period 5005 in i;
  2. the middle gears 17 .. sqrt(g): the same teeth, each striking about 2L/h offsets;
  3. the bands: the gears above sqrt(g), first strike at e = s h - r^2 then every h thereafter
     (each band gear strikes at most a few offsets in the range);
  4. the newest gear g itself (its strikes at g (g + 2) or g (g + 4), then every g).
For one range (g given) the script prints the map: each offset with the structure that kills
it (the smallest-numbered structure and the gear), and the twins as the untouched offsets.
Then, pooled over primes g in [gmin, gmax], the budget: the share of offsets removed by each
structure in order (marginal removal among the offsets the earlier structures left), and the
share of survivors of structures 1-3 that are twins.
Usage: uv run python exclusion_map.py g_example gmin gmax
"""
import sys, math
from sympy import primerange, nextprime, isprime


def killers(g, L):
    """for each offset 1..L: (structure number, gear) of the first structure that strikes it, or None"""
    gp = nextprime(g); a = (g * g - 1) // 6
    sq = int(math.isqrt(g))
    res = [None] * (L + 1)
    def strikes(h, i): return (g * g + 6 * i) % h in (0, 2)
    for i in range(1, L + 1):
        for h in (5, 7, 11, 13):
            if h <= g and strikes(h, i): res[i] = (1, h); break
        if res[i]: continue
        for h in primerange(17, sq + 1):
            if strikes(h, i): res[i] = (2, h); break
        if res[i]: continue
        for h in primerange(sq + 1, g):
            if strikes(h, i): res[i] = (3, h); break
        if res[i]: continue
        if strikes(g, i): res[i] = (4, g)
    return res


def main():
    gx, gmin, gmax = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    gp = nextprime(gx); a = (gx * gx - 1) // 6; L = (gp * gp - gx * gx - 2) // 6
    res = killers(gx, L)
    print(f"range after {gx}^2 = {gx*gx} up to {gp}^2, {L} offsets; killer per offset (structure:gear), '.' = twin:")
    line = []
    for i in range(1, L + 1):
        k = res[i]
        tw = isprime(6 * (a + i) - 1) and isprime(6 * (a + i) + 1)
        assert (k is None) == tw, (i, k, tw)
        line.append("." if k is None else f"{k[0]}:{k[1]}")
    for j in range(0, L, 12): print("  " + " ".join(f"{i+1:3d}={line[i]:>6s}" for i in range(j, min(L, j + 12))))
    print("  twins at offsets", [i for i in range(1, L + 1) if res[i] is None])
    # budget over ranges
    left = [0, 0, 0, 0, 0]; tot = 0; tw = 0
    for g in primerange(gmin, gmax + 1):
        gp = nextprime(g); L = (gp * gp - g * g - 2) // 6; r = killers(g, L); tot += L
        for i in range(1, L + 1):
            k = r[i]
            if k is None: tw += 1; left[4] += 1
            else: left[k[0] - 1] += 1
    print(f"budget over primes g in [{gmin}, {gmax}], {tot} offsets: removed by the small wheel (5..13) {100*left[0]/tot:.1f}%; by the middle gears (17..sqrt g) {100*left[1]/tot:.1f}%; by the bands (above sqrt g) {100*left[2]/tot:.1f}%; by the newest gear {100*left[3]/tot:.1f}%; twins {100*tw/tot:.1f}%")
    s1 = tot - left[0]; s2 = s1 - left[1]; s3 = s2 - left[2]
    print(f"  marginal: the middle gears remove {100*left[1]/s1:.1f}% of what the small wheel left; the bands remove {100*left[2]/s2:.1f}% of what remained; the newest gear {100*left[3]/s3:.1f}%; of the survivors of structures 1-3, {100*tw/s3:.1f}% are twins")


if __name__ == "__main__":
    main()
