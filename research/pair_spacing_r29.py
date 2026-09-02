"""Twin-twin spacings on the new section - manager, round 29.

Human's question: do twin primes trend towards appearing at intervals of 210, 2310, 2520
(= 2310 + 210)? In slot units 35, 385, 420. The lower sieve's word already prefers these
spacings: the number of residues a mod g with a and a + D both open is c_g(D) = g-2 if
D = 0 mod g, g-3 if D = +-2u mod g, g-4 otherwise, so pairs of openings at spacing D are
weighted prod_g c_g(D). On the new section (p^2, q^2) count pairs of new twins at each slot
spacing D and compare with that weight (gears up to 50), normalised on the generic spacings.

Usage: python pair_spacing_r29.py [--qmin 1000] [--qmax 5000] [--dmax 450]
"""
import argparse
from math import prod

import numpy as np

from word_tree_r29 import spf_sieve


def c_g(g, D):
    u = pow(6, -1, g)
    r = D % g
    if r == 0:
        return g - 2
    if r in ((2 * u) % g, (-2 * u) % g):
        return g - 3
    return g - 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qmin", type=int, default=1000)
    ap.add_argument("--qmax", type=int, default=5000)
    ap.add_argument("--dmax", type=int, default=450)
    a = ap.parse_args()
    primes = [int(x) for x in np.flatnonzero(spf_sieve(a.qmax + 100) == np.arange(a.qmax + 101)) if x >= 5]
    spf = spf_sieve(a.qmax * a.qmax + 10).astype(np.int64)
    gears = [g for g in primes if g <= 50]
    weight = np.array([prod(c_g(g, D) for g in gears) for D in range(a.dmax + 1)], dtype=float)
    generic = np.array([all(D % g not in (0, (2 * pow(6, -1, g)) % g, (-2 * pow(6, -1, g)) % g) for g in gears) for D in range(a.dmax + 1)])
    counts = np.zeros(a.dmax + 1, dtype=np.int64)
    nsec = 0
    for i in range(1, len(primes)):
        p, q = primes[i - 1], primes[i]
        if q > a.qmax:
            break
        if q < a.qmin:
            continue
        k_lo, k_hi = p * p // 6 + 1, (q * q - 2) // 6
        ks = np.arange(k_lo, k_hi + 1)
        lo, hi = 6 * ks - 1, 6 * ks + 1
        tw = ks[(spf[lo] == lo) & (spf[hi] == hi)]
        s = set(tw.tolist())
        for k in tw:
            for D in range(1, min(a.dmax, int(k_hi) - int(k)) + 1):
                if k + D in s:
                    counts[D] += 1
        nsec += 1
    gen_mean = counts[generic].mean()
    w_gen = weight[generic].mean()
    print(f"sections {a.qmin} <= q <= {a.qmax}: {nsec}; pairs of new twins at slot spacing D (numbers 6D); "
          f"generic spacings (no gear <= 50 divides D or D +- 2u): mean count {gen_mean:.1f}")
    print(f"  D  numbers   count   count/generic   predicted prod c_g(D)/generic")
    for D in (1, 2, 5, 7, 35, 70, 77, 105, 140, 175, 210, 245, 280, 315, 350, 385, 420):
        if D <= a.dmax:
            print(f"  {D:>3}  {6 * D:>6}   {int(counts[D]):>6}   {counts[D] / gen_mean:>8.2f}        {weight[D] / w_gen:>8.2f}")
    ratio = counts[1:] / (weight[1:] * gen_mean / w_gen)
    print(f"  over all D = 1..{a.dmax}: observed / predicted mean {ratio.mean():.3f}, min {ratio.min():.3f} at D = {int(ratio.argmin()) + 1}, max {ratio.max():.3f} at D = {int(ratio.argmax()) + 1}")
    top = np.argsort(-counts[1:])[:8] + 1
    print("  most frequent spacings: " + ", ".join(f"D={int(D)} ({6 * int(D)}): {int(counts[D])}" for D in top))


if __name__ == "__main__":
    main()
