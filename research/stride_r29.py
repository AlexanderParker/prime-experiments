"""The new gear's stride over the lower sieve - manager, round 29 (pre-registration:
data/r29/stride_prereg.md).

The new period qP is q copies of the lower sieve's word (period P). Gear q's teeth hit slots
k = +-u mod q; in copy j (slots jP .. jP+P-1) a hit lands on an old opening (new block) or on
an already-blocked slot (nothing). Per copy: hits, new blocks, hits on the sieve; untouched
copies; the residue histogram n(r) of old openings mod q that generates the profile.

Usage: python stride_r29.py
"""
from math import prod, sqrt

import numpy as np

NGATE = 0
NFAIL = 0


def gate(cond, msg):
    global NGATE, NFAIL
    NGATE += 1
    NFAIL += (not cond)
    print(("  GATE ok:   " if cond else "  GATE FAIL: ") + msg)


def uinv(g):
    return pow(6, -1, g)


def open_word(gears, length):
    w = np.ones(length, dtype=bool)
    for g in gears:
        u = uinv(g)
        w[u % g::g] = False
        w[(-u) % g::g] = False
    return w


def main():
    primes = [5, 7, 11, 13, 17, 19, 23, 29]
    s1 = s2 = True
    untouched = {}
    spread_rel = {}
    spread_sqrt = {}
    for i in range(1, len(primes)):
        gears, q = primes[:i], primes[i]
        P = prod(gears)
        u = uinv(q)
        old = open_word(gears, P)
        idx = np.flatnonzero(old)
        N = len(idx)
        n = np.bincount(idx % q, minlength=q)
        js = np.arange(q)
        new = n[(u - js * P) % q] + n[(-u - js * P) % q]
        # total hits of gear q per copy, by direct count of k = +-u mod q in [jP, jP+P)
        hits = np.array([sum(1 for k in range(j * P, j * P + P) if k % q in (u % q, (-u) % q)) for j in js]) if P * q <= 200000 \
            else np.array([((np.arange(j * P, j * P + P) % q == u % q) | (np.arange(j * P, j * P + P) % q == (-u) % q)).sum() for j in js])
        print(f"=== lower sieve {gears} (P = {P}, N = {N} old openings), new gear {q} (teeth +-{u}), new period {q * P} ===")
        print(f"  n(r), old openings by residue r mod {q}: {n.tolist()}  (empty residues: {[int(r) for r in np.flatnonzero(n == 0)]})")
        print(f"  strides per period: {2 * P} hits, {2 * N} new blocks, {2 * P - 2 * N} land on the sieve (fraction on sieve {1 - N / P:.4f} = 1 - prod(1 - 2/g))")
        if q <= 17:
            # S1 by direct enumeration of the q x P word
            full_old = np.tile(old, q)
            full_new = open_word(gears + [q], q * P)
            direct = np.array([int((full_old[j * P:(j + 1) * P] & ~full_new[j * P:(j + 1) * P]).sum()) for j in js])
            s1 &= bool(np.array_equal(direct, new))
            print("  copy  j : hits  new-blocks  on-sieve   (new blocks = n(u - jP) + n(-u - jP))")
            for j in js:
                print(f"    {int(j):>3}   : {int(hits[j]):>4}  {int(new[j]):>4}        {int(hits[j] - new[j]):>4}" + ("   UNTOUCHED" if new[j] == 0 else ""))
        s1 &= int(new.sum()) == 2 * N
        s2 &= bool((np.abs(new - new[::-1]) <= 1).all())
        untouched[q] = [int(j) for j in js[new == 0]]
        mean = 2 * N / q
        spread_rel[q] = (new.max() - new.min()) / mean
        spread_sqrt[q] = (new.max() - new.min()) / sqrt(mean)
        print(f"  untouched copies: {untouched[q] or 'none'}; new blocks per copy min {int(new.min())} max {int(new.max())} mean {mean:.1f}; "
              f"spread/mean {spread_rel[q]:.4f}, spread/sqrt(mean) {spread_sqrt[q]:.2f}; mirror j <-> q-1-j max |diff| {int(np.abs(new - new[::-1]).max())}")
    print()
    gate(s1, "S1 new blocks in copy j = n(u - jP) + n(-u - jP), sum 2N (direct enumeration q <= 17)")
    gate(s2, "S2 copy profile palindromic j <-> q-1-j up to 1")
    gate(untouched[7] == [0, 6] and all(untouched[q] == [] for q in primes[2:]),
         f"S3 untouched copies only for {{5}} + 7 (copies 0 and 6): {untouched}")
    qs = [q for q in primes[3:]]
    gate(all(spread_rel[a] > spread_rel[b] for a, b in zip(qs, qs[1:])) and spread_rel[29] < 0.05,
         "S4 spread/mean decreasing from q = 13 and below 0.05 at q = 29: " + ", ".join(f"{q}: {spread_rel[q]:.4f}" for q in qs))
    gate(all(2 <= spread_sqrt[q] <= 6 for q in qs),
         "S5 spread/sqrt(mean) in [2, 6] for q = 13..29: " + ", ".join(f"{q}: {spread_sqrt[q]:.2f}" for q in qs))
    print(f"\n{NGATE - NFAIL}/{NGATE} gates passed")


if __name__ == "__main__":
    main()
