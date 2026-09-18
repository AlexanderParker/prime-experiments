"""Round 84 / loop entry 90: what would have to emerge for a window to fail, and whether it can.

A machine q fails if every column of its window (q, q^2] is struck.  Split the gears at a cut P:
the gears up to P leave a set of SURVIVORS in the window (columns none of them strikes), and
failure requires the gears in (P, q] to strike every survivor.

Under independence - each large gear's two classes falling on survivors in the same proportion as
on all columns - the fraction of survivors the gears in (P, q] strike is

    1 - product over P < h <= q of (1 - 2/h),

which is well below 1 as soon as P is a power of q near 1 (for P = q/2 it is about 2 ln 2 / ln q).
So failure REQUIRES a departure from independence: the large gears' teeth must land on survivors
far more often than their share, by a factor that grows without bound as the cut approaches q.

This measures, on real machines, the actual fraction of survivors struck by the large gears
against the independence prediction, at several cuts.  A machine whose large gears were "on the
way" to failing would show the fraction pulling above the prediction; a machine with no such
tendency shows the required property is absent, with a number attached.
"""

import math
import sys

import numpy as np


def primes_to(n):
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i :: i] = False
    return np.nonzero(sieve)[0]


def struck_mask(gears, lo, hi):
    n = hi - lo + 1
    mask = np.zeros(n, dtype=bool)
    for h in gears:
        inv6 = pow(6, -1, int(h))
        for r in (inv6 % h, (-inv6) % h):
            start = (r - lo) % h
            mask[start::h] = True
    return mask


def main():
    print("survivors of the gears up to the cut P, and the share of them the gears in (P, q] strike")
    print("       q     cut P   survivors   struck by (P,q]   fraction   independence   ratio")
    for q in (1009, 2003, 5003):
        gears = [int(h) for h in primes_to(q) if h >= 5]
        lo = q // 6 + 2
        hi = (q * q - 1) // 6
        cuts = [("q/2", q // 2), ("q^0.75", int(q ** 0.75)), ("q^0.5", int(q ** 0.5)), ("q^0.25", int(q ** 0.25))]
        for name, P in cuts:
            small = [h for h in gears if h <= P]
            large = [h for h in gears if h > P]
            surv = ~struck_mask(small, lo, hi)
            big = struck_mask(large, lo, hi)
            S = int(surv.sum())
            hit = int((surv & big).sum())
            frac = hit / S if S else 0.0
            indep = 1.0
            for h in large:
                indep *= (1 - 2.0 / h)
            indep = 1 - indep
            print(
                "%8d  %8s  %10d  %16d  %9.4f  %13.4f  %6.3f"
                % (q, name, S, hit, frac, indep, frac / indep if indep else 0)
            )
        print()
    print("ratio = 1 means the large gears strike survivors exactly in proportion to their share;")
    print("failure would need the fraction to reach 1.0000, i.e. ratio 1/independence.")


if __name__ == "__main__":
    sys.exit(main())
