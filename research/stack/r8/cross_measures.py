"""A battery of cross-part measurements on the new range (g^2, g'^2) of each prime g.

(a) the first twin offset L_1 and the twin density by the class of g modulo 35 (which fixes
    the phases of the gears 5 and 7 at the square) and modulo 6 (the class of g);
(b) the neighbourhood comb: the share of member-kills in the range by gears that divide
    g +- s for s <= S (phases s^2), for S = 4, 10, 30, 100, against the share of all gears;
(c) the mirror score: the fraction of twin offsets i whose mirror L + 1 - i is also a twin,
    against the same for a random set of the same size in 1..L (100 shuffles);
(d) the twin offsets relative to the new gear's own first strike (g (g + 2) or g (g + 4)):
    the share of twins below it, and the share of ranges whose first twin is below it.
Usage: uv run python cross_measures.py gmin gmax
"""
import sys, random
from collections import defaultdict
import numpy as np
from sympy import primerange, nextprime, isprime, factorint

random.seed(5)


def main():
    gmin, gmax = int(sys.argv[1]), int(sys.argv[2])
    by35 = defaultdict(list); by6 = defaultdict(list); dens35 = defaultdict(list)
    comb = {S: [0, 0] for S in (4, 10, 30, 100)}; tot_kills = 0
    mirror_real = []; mirror_rand = []
    below_new = [0, 0]; first_below = [0, 0]
    for g in primerange(gmin, gmax + 1):
        gp = nextprime(g); a = (g * g - 1) // 6; L = (gp * gp - g * g) // 6
        tw = []; kills = []  # kills: (offset, gear)
        for i in range(1, L + 1):
            l, r = 6 * (a + i) - 1, 6 * (a + i) + 1
            pl, pr = isprime(l), isprime(r)
            if pl and pr: tw.append(i)
            for n, pp in ((l, pl), (r, pr)):
                if not pp: kills.append((i, min(factorint(n))))
        if not tw: continue
        by35[g % 35].append(tw[0]); by6[g % 6].append(tw[0]); dens35[g % 35].append(len(tw) / L)
        tot_kills += len(kills)
        for S in comb:
            near = set()
            for s in range(1, S + 1):
                for m in (g - s, g + s):
                    if m > 1:
                        for q in factorint(m):
                            if q >= 5 and q <= g: near.add(q)
            comb[S][0] += sum(1 for _, h in kills if h in near); comb[S][1] += len(near)
        ts = set(tw); mirror_real.append(sum(1 for i in tw if (L + 1 - i) in ts) / len(tw))
        rr = []
        for _ in range(20):
            s = set(random.sample(range(1, L + 1), len(tw))); rr.append(sum(1 for i in s if (L + 1 - i) in s) / len(s))
        mirror_rand.append(np.mean(rr))
        e_new = 2 * g if (g + 2) % 6 in (1, 5) else 4 * g
        i_new = e_new // 6
        below_new[0] += sum(1 for i in tw if i < i_new); below_new[1] += len(tw)
        first_below[0] += tw[0] < i_new; first_below[1] += 1
    print(f"primes g in [{gmin}, {gmax}]")
    print("(a) first twin offset by g mod 6:", {c: f"n={len(v)}, mean L1={np.mean(v):.2f}" for c, v in sorted(by6.items())})
    rows = sorted(by35.items())
    print("(a) first twin offset by g mod 35 (class: n, mean L1, mean twin density x100):")
    print("    " + "; ".join(f"{c}: {len(v)}, {np.mean(v):.1f}, {100*np.mean(dens35[c]):.1f}" for c, v in rows))
    allL1 = [x for v in by35.values() for x in v]
    print(f"    overall mean L1 {np.mean(allL1):.2f}; spread of class means {np.std([np.mean(v) for v in by35.values()]):.2f} against the standard error of a class mean {np.std(allL1)/np.sqrt(np.mean([len(v) for v in by35.values()])):.2f}")
    print("(b) neighbourhood comb: share of member-kills by gears dividing g +- s, s <= S:", {S: f"{v[0]/tot_kills:.3f} (mean {v[1]/first_below[1]:.1f} such gears per step)" for S, v in comb.items()})
    print(f"(c) mirror score of the twin offsets about the range centre: real {np.mean(mirror_real):.4f}, random {np.mean(mirror_rand):.4f}")
    print(f"(d) twins below the new gear's first strike (offset g/3 or 2g/3): {below_new[0]} of {below_new[1]} ({100*below_new[0]/below_new[1]:.1f}%); ranges whose first twin is below it: {first_below[0]} of {first_below[1]}")


if __name__ == "__main__":
    main()
