"""Round 86 / loop entry 92: pairs of fields - can any two combine into a blocking state?

The fields are the many-body property split into parts.  The next part is pairs: for two gears g1
and g2, the kills of their fields on the survivors of the gears below g1 either overlap as
independence says - union share 2/g1 + 2/g2 - 4/(g1 g2) - or they avoid each other, which is
what a block would need.  This measures every pair's union on the survivors against the
independence share, reports the pairs with the largest excess, and checks the one pair structure
that is exact by arithmetic: a gear's square is a lone killer of its column exactly when g^2 - 2 is
prime.
"""

import math
import random
import sys

import numpy as np


def primes_to(n):
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i :: i] = False
    return sieve


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
        # survivors of the gears below g1, cumulative
        surv_below = {}
        acc = np.zeros(n, dtype=bool)
        for g in gears:
            surv_below[g] = ~acc
            acc |= hits[g]
        # pairs: all with g1 among the first 12 gears, plus a random sample
        rng = random.Random(1)
        pairs = []
        for i, g1 in enumerate(gears[:12]):
            for g2 in gears[i + 1 :]:
                pairs.append((g1, g2))
        allpairs = [(gears[i], gears[j]) for i in range(len(gears)) for j in range(i + 1, len(gears))]
        pairs += rng.sample(allpairs, min(3000, len(allpairs)))
        rows = []
        for g1, g2 in pairs:
            S = surv_below[g1]
            Sn = int(S.sum())
            if Sn == 0:
                continue
            u = int(((hits[g1] | hits[g2]) & S).sum())
            both = int(((hits[g1] & hits[g2]) & S).sum())
            pred = (2.0 / g1 + 2.0 / g2 - 4.0 / (g1 * g2)) * Sn
            pred_both = 4.0 / (g1 * g2) * Sn
            rows.append((u / pred, u - pred, g1, g2, u, pred, both, pred_both))
        rows.sort(reverse=True)
        ratios = [r[0] for r in rows]
        print("q = %d: %d pairs, union of two fields' kills on the survivors below g1, against independence"
              % (q, len(rows)))
        print("   mean ratio %.4f, min %.4f, max %.4f" % (sum(ratios) / len(ratios), min(ratios), max(ratios)))
        print("   pairs with the largest excess over independence:")
        print("      g1    g2   union   predicted   ratio   both-struck   predicted both")
        for r in rows[:6]:
            print("   %5d %5d  %6d  %10.1f  %6.3f  %11d  %14.1f" % (r[2], r[3], r[4], r[5], r[0], r[6], r[7]))
        print("   pairs with the largest deficit (most overlap):")
        for r in rows[-4:]:
            print("   %5d %5d  %6d  %10.1f  %6.3f  %11d  %14.1f" % (r[2], r[3], r[4], r[5], r[0], r[6], r[7]))
        # the exact pair: squares as lone killers
        lone = 0
        tot = 0
        for g in gears:
            if g * g > q:
                tot += 1
                if ps[g * g - 2]:
                    lone += 1
        print("   squares: %d of %d in (sqrt q, q] are lone killers of their column (g^2 - 2 prime)" % (lone, tot))
        print()


if __name__ == "__main__":
    sys.exit(main())
