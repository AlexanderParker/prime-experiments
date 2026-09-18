"""Round 86 / loop entry 92: the pair interaction alone.

field_pairs.py compared each pair's union against a lattice-share baseline, which mixes the
single-field deviation (large gears strike survivors below their lattice share, entry 90) with the
pair's own interaction.  This isolates the interaction: for each pair (g1, g2) the columns of the
survivors below g1 that BOTH strike, against the product of each gear's OWN measured kill rate on
those survivors.  A ratio of 1 means the two fields overlap exactly as if independent; below 1
means they avoid each other, which is the direction a block would need.
"""

import random
import sys

import numpy as np


def primes_to(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i :: i] = False
    return s


def main():
    for q in (1009, 2003):
        ps = primes_to(q * q + 10)
        gears = [int(h) for h in np.nonzero(ps)[0] if 5 <= h <= q]
        lo = q // 6 + 2
        hi = (q * q - 1) // 6
        n = hi - lo + 1
        hits = {}
        for g in gears:
            inv6 = pow(6, -1, g)
            m = np.zeros(n, dtype=bool)
            for r in (inv6 % g, (-inv6) % g):
                m[(r - lo) % g :: g] = True
            hits[g] = m
        surv = {}
        acc = np.zeros(n, dtype=bool)
        for g in gears:
            surv[g] = ~acc
            acc |= hits[g]
        rng = random.Random(2)
        allpairs = [(gears[i], gears[j]) for i in range(len(gears)) for j in range(i + 1, len(gears))]
        pairs = rng.sample(allpairs, min(6000, len(allpairs)))
        tot_both = 0.0
        tot_ind = 0.0
        bands = {"both <= q^0.5": [0, 0.0], "mixed": [0, 0.0], "both > q^0.5": [0, 0.0]}
        for g1, g2 in pairs:
            S = surv[g1]
            Sn = int(S.sum())
            a1 = int((hits[g1] & S).sum())
            a2 = int((hits[g2] & S).sum())
            both = int((hits[g1] & hits[g2] & S).sum())
            ind = a1 * a2 / Sn if Sn else 0.0
            tot_both += both
            tot_ind += ind
            key = "both <= q^0.5" if g2 <= q ** 0.5 else ("both > q^0.5" if g1 > q ** 0.5 else "mixed")
            bands[key][0] += both
            bands[key][1] += ind
        print("q = %d: pair overlap on the survivors below g1, against each gear's own measured rate" % q)
        print("   all %d sampled pairs: actual both-struck %d, independent %.1f, ratio %.3f"
              % (len(pairs), tot_both, tot_ind, tot_both / tot_ind))
        for k, (b, i) in bands.items():
            print("   %-14s  actual %6d  independent %8.1f  ratio %.3f" % (k, b, i, (b / i) if i else float("nan")))
        print()


if __name__ == "__main__":
    sys.exit(main())
