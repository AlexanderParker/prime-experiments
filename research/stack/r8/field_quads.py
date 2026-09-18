"""Round 88 / loop entry 94: quadruples of fields - can any four combine into a blocking state?

Same construction as the pairs and triples: for gears g1 < g2 < g3 < g4, on the survivors of the
gears below g1, the columns all four strike, against
  * independence of the four fields (product of the four own measured rates), and
  * the triple-consistent baseline (the fourth-order superposition built from the four measured
    triple overlaps, the six pairs and the four singles), which removes everything the triples
    already explain and leaves the pure four-body term.
Stratified by how many of the four gears lie above sqrt q.
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
        rng = random.Random(4)
        root = q ** 0.5
        small = [g for g in gears if g <= root]
        large = [g for g in gears if g > root]
        bands = {k: [0.0, 0.0, 0.0, 0] for k in range(5)}
        N = 4000
        for k in range(5):
            if len(small) < 4 - k or len(large) < k:
                continue
            for _ in range(N):
                gs = sorted(rng.sample(small, 4 - k) + rng.sample(large, k))
                S = surv[gs[0]]
                Sn = float(S.sum())
                if Sn == 0:
                    continue
                M = [hits[g] & S for g in gs]
                s1 = [float(m.sum()) for m in M]
                p2 = {}
                for i in range(4):
                    for j in range(i + 1, 4):
                        p2[(i, j)] = float((M[i] & M[j]).sum())
                t3 = {}
                for i in range(4):
                    for j in range(i + 1, 4):
                        for l in range(j + 1, 4):
                            t3[(i, j, l)] = float((M[i] & M[j] & M[l]).sum())
                abcd = float((M[0] & M[1] & M[2] & M[3]).sum())
                indep = s1[0] * s1[1] * s1[2] * s1[3] / (Sn ** 3)
                prod_t = 1.0
                for v in t3.values():
                    prod_t *= v
                prod_p = 1.0
                for v in p2.values():
                    prod_p *= v
                prod_s = s1[0] * s1[1] * s1[2] * s1[3]
                kirk = (prod_t * prod_s) / (prod_p * Sn) if prod_p > 0 else 0.0
                bands[k][0] += abcd
                bands[k][1] += indep
                bands[k][2] += kirk
                bands[k][3] += 1
        print("q = %d: %d quadruples (stratified), columns of the survivors below g1 struck by all four" % (q, 5 * N))
        print("   gears above sqrt q   quads   actual   independence   ratio   triple-consistent   ratio")
        for k in range(5):
            act, ind, kirk, cnt = bands[k]
            r1 = act / ind if ind else float("nan")
            r2 = act / kirk if kirk else float("nan")
            print("   %8d             %5d  %8d  %13.1f  %6.3f  %17.1f  %6.3f" % (k, cnt, act, ind, r1, kirk, r2))
        print()


if __name__ == "__main__":
    sys.exit(main())
