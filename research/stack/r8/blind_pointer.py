"""The square origin's gift: offsets blind to the small gears, and the twins on them.

From a square origin g^2 (column a), gear h strikes offset i iff g^2 + 6i = 0 or 2 (mod h). Since
g^2 is a square mod h, the offsets i with BOTH -6i and 2 - 6i non-squares mod h are never
struck by h from any square origin: the blind classes of h. For the gears 5, 7 the blind
offsets are i = 5, 10, 12, 17 (mod 35); with 11 and 13 the classes thin further. This script,
for every prime g up to gmax with the new range (g^2, g'^2) of L = (g'^2 - g^2)/6 columns:
  the twin offsets in the range, how many lie in the {5,7}-blind classes, in the {5,7,11}-blind
  and {5,7,11,13}-blind classes; the number of blind offsets below L of each depth and how
  many of them are twins; the first twin's blindness depth; the minimum over g of the count
  of {5..13}-blind offsets below L (the candidate set is never empty?) and the minimum number
  of twins among them.
Usage: uv run python blind_pointer.py gmax
"""
import sys
import numpy as np
from sympy import primerange, nextprime, isprime


def blind_set(h):
    sq = {(x * x) % h for x in range(1, h)}
    return {i for i in range(h) if ((-6 * i) % h not in sq) and ((2 - 6 * i) % h not in sq)}


def main():
    gmax = int(sys.argv[1])
    depths = [[5, 7], [5, 7, 11], [5, 7, 11, 13]]
    blind = {h: blind_set(h) for h in (5, 7, 11, 13)}
    tot_tw = 0; in_depth = [0, 0, 0]; cand = [0, 0, 0]; cand_tw = [0, 0, 0]
    min_cand = [10**9] * 3; min_cand_tw = [10**9] * 3; first_depth = [0, 0, 0, 0]; steps = 0
    for g in primerange(17, gmax + 1):
        gp = nextprime(g); a = (g * g - 1) // 6; L = (gp * gp - g * g) // 6; steps += 1
        tw = [i for i in range(1, L + 1) if isprime(6 * (a + i) - 1) and isprime(6 * (a + i) + 1)]
        tot_tw += len(tw)
        first = tw[0] if tw else None
        fd = 0
        for d, gs in enumerate(depths):
            isb = lambda i: all(i % h in blind[h] for h in gs)
            c = [i for i in range(1, L + 1) if isb(i)]; ct = [i for i in tw if isb(i)]
            in_depth[d] += len(ct); cand[d] += len(c); cand_tw[d] += len(ct)
            min_cand[d] = min(min_cand[d], len(c)); min_cand_tw[d] = min(min_cand_tw[d], len(ct))
            if first is not None and isb(first): fd = d + 1
        first_depth[fd] += 1
    print(f"primes g in [17, {gmax}]: {steps} steps, {tot_tw} twins in the new ranges")
    for d, gs in enumerate(depths):
        print(f"  blind to {gs} (density {np.prod([len(blind[h])/h for h in gs]):.4f} of offsets): twins on these offsets {in_depth[d]} of {tot_tw} ({100*in_depth[d]/tot_tw:.1f}%); candidates below L: total {cand[d]}, twins among them {cand_tw[d]} ({100*cand_tw[d]/max(cand[d],1):.1f}%); min candidates per step {min_cand[d]}, min twins among candidates per step {min_cand_tw[d]}")
    print(f"  first twin of the range: blind to none of 5,7 in {first_depth[0]} steps; to 5,7 in {first_depth[1]}; to 5,7,11 in {first_depth[2]}; to 5,7,11,13 in {first_depth[3]}")


if __name__ == "__main__":
    main()
