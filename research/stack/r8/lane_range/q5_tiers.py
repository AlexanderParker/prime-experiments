"""Q5: tiers of the range of q=13.  P=5005, members to 30031, T = primes 17..173.
Tier i covers the columns n with 6n+1 in (root_{i-1}^2, root_i^2]: tier 1 = (13, 169], tier 2 =
(169, 28561], tier 3 = (28561, 30031].  Deciding primes of tier i = T cap (q, root_i].
Exhaust cap: on tier i every strike by a prime above root_i on an opening of q is a home column
(p = 6n+-1) or an echo (cofactor has a prime factor in T below root_i).
Twins of tier i = openings of q in the tier struck by no deciding prime = openings of the machine
whose top gear is the largest prime <= root_i (13, 167, 173), restricted to the tier.
"""
import numpy as np
from math import isqrt
from lane_common import gears, period, c, openings, T_primes, struck_mask, primes_upto, factor


def main(q=13):
    P = period(q)
    O = openings(q)
    T = T_primes(q)
    top = 6 * P + 1
    print(f"q={q} P={P} members to {top}, sqrt={isqrt(top)}; T = {T}")
    roots = [q, q * q, isqrt(top)]              # tier tops in member terms: root^2 ; last tier ends at top
    bounds = []
    lo = 1
    for i, R in enumerate(roots):
        hi_member = min(R * R, top) if i < 2 else top
        hi = (hi_member - 1) // 6
        bounds.append((lo, hi, R))
        lo = hi + 1
    isprime = np.zeros(top + 1, dtype=bool)
    for p in primes_upto(top):
        isprime[p] = True
    all_twins_expected = set(int(n) for n in O if isprime[6 * int(n) - 1] and isprime[6 * int(n) + 1])
    twins_by_tier = []
    for (lo, hi, R) in bounds:
        deciding = [p for p in T if p <= R]
        On = O[(O >= lo) & (O <= hi)]
        acting = set()
        # exhaust-cap check: every prime factor p > R of a member of an opening is home or echo
        for n in On.tolist():
            for mem in (6 * n - 1, 6 * n + 1):
                fs = factor(mem)
                for p in set(fs):
                    if p <= R:
                        if p > q:
                            acting.add(p)
                        continue
                    if p == mem:
                        continue                                 # home column
                    cof = mem // p
                    small = [r for r in factor(cof) if r <= R]
                    assert small and all(r > q for r in small), (n, mem, p)   # echo of a T-prime below R
        # twins of the tier
        struck_dec = np.zeros(len(On), dtype=bool)
        for p in deciding:
            struck_dec |= ((On % p) == c(p)) | ((On % p) == p - c(p))
        twins = set(On[~struck_dec].tolist())
        # complement rule: openings of machine top_prime restricted to tier
        top_prime = max(p for p in primes_upto(R))
        mm = struck_mask(gears(top_prime), hi)
        pred = set(int(n) for n in range(lo, hi + 1) if not mm[n])
        assert twins == pred, "tier twins != openings of the tier's machine"
        assert twins == {n for n in all_twins_expected if lo <= n <= hi}, "tier twins != actual twin pairs"
        twins_by_tier.append(twins)
        print(f" tier (root {R}): columns {lo}..{hi} (members {6*lo-1}..{6*hi+1}); deciding primes of T: "
              f"{deciding[0] if deciding else '-'}..{deciding[-1] if deciding else '-'} ({len(deciding)}); "
              f"primes of T actually dividing an opening's member with p <= root: {len(acting)} of {len(deciding)}"
              f"{' (idle: ' + str(sorted(set(deciding) - acting)) + ')' if set(deciding) - acting else ''}")
        print(f"     exhaust cap PASS: every strike by a prime above {R} on an opening of q in this tier is home or echo")
        print(f"     twins = openings of machine {top_prime} on {lo}..{hi}: PASS; openings of q here: {len(On)} (check)")
        if R == q:
            print(f"     tier-1 twins (all openings of q in the tier): {sorted(twins)}")
        else:
            tl = sorted(twins)
            print(f"     first/last twins: {tl[:6]} ... {tl[-4:]}  (count {len(tl)}, check only)")
    # 173 acts only in tier 3: its strikes on tier-2 openings are all echoes
    On2 = O[(O >= bounds[1][0]) & (O <= bounds[1][1])]
    s173 = On2[((On2 % 173) == c(173)) | ((On2 % 173) == 173 - c(173))]
    echo = all(any(r in T and r < 173 for r in factor((6 * int(n) - 1) if (6 * int(n) - 1) % 173 == 0 else 6 * int(n) + 1))
               for n in s173)
    print(f" 173 on tier-2 openings: {len(s173)} strikes, all echoes of a smaller T-prime: {echo}")


if __name__ == "__main__":
    main()
