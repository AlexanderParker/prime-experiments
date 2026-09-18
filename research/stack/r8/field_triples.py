"""Round 87 / loop entry 93: triples of fields - can any three combine into a blocking state?

For gears g1 < g2 < g3, on the survivors of the gears below g1, the columns all three strike are
compared with two baselines:
  * independence of the three fields: the product of each gear's own measured kill rate;
  * the pairwise-consistent (Kirkwood) baseline built from the three measured pair overlaps, which
    removes what the pairs already explain and leaves the pure three-body term.
A ratio of 1 means no interaction at that level; below 1 means mutual avoidance, the direction a
block needs.  Triples are sampled at random and aggregated by how many of the three gears lie
above sqrt q.
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
        rng = random.Random(3)
        root = q ** 0.5
        small = [g for g in gears if g <= root]
        large = [g for g in gears if g > root]
        bands = {k: [0.0, 0.0, 0.0, 0] for k in (0, 1, 2, 3)}  # actual, indep, kirkwood, count
        N = 5000
        for k in (0, 1, 2, 3):
            if len(small) < 3 - k or len(large) < k:
                continue
            for _ in range(N):
                trip = sorted(rng.sample(small, 3 - k) + rng.sample(large, k))
                g1, g2, g3 = trip
                S = surv[g1]
                Sn = int(S.sum())
                if Sn == 0:
                    continue
                A, B, C = hits[g1] & S, hits[g2] & S, hits[g3] & S
                a, b, c = int(A.sum()), int(B.sum()), int(C.sum())
                ab, bc, ac = int((A & B).sum()), int((B & C).sum()), int((A & C).sum())
                abc = int((A & B & C).sum())
                indep = float(a) * float(b) * float(c) / (float(Sn) * float(Sn))
                kirk = float(Sn) * (float(ab) * float(bc) * float(ac)) / (float(a) * float(b) * float(c)) if (a and b and c) else 0.0
                bands[k][0] += abc
                bands[k][1] += indep
                bands[k][2] += kirk
                bands[k][3] += 1
        N = 4 * N
        print("q = %d: %d triples (stratified by gears above sqrt q), columns of the survivors below g1 struck by all three" % (q, N))
        print("   gears above sqrt q   triples   actual   independence   ratio   pairwise-consistent   ratio")
        for k in (0, 1, 2, 3):
            act, ind, kirk, cnt = bands[k]
            r1 = act / ind if ind else float("nan")
            r2 = act / kirk if kirk else float("nan")
            print("   %8d             %6d  %7d  %13.1f  %6.3f  %19.1f  %6.3f"
                  % (k, cnt, act, ind, r1, kirk, r2))
        print()


if __name__ == "__main__":
    sys.exit(main())
