"""Item 2: COMPLETE KNOWLEDGE.  Enumerate the zone's open pairs from the rule, count them
exactly, and take the count apart into families.

  Z5  the rule-generated open-pair set equals the sieve's, 0 mismatches
  Z6  the exact admissible count  Psi(X, q) + sum_{s smooth <= X/p1} (pi(X/s) - pi(Q))
      and the three-part decomposition of the open-pair count
  Z4  the family census: every open pair (n, n+2) in the zone carries a label (s, s') of
      q-smooth cofactors with s'P' - sP = 2

usage: uv run python research/topmachine/r6/s2_count.py
"""

import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (admissible_by_rule, admissible_by_sieve, open_pairs,  # noqa: E402
                    primes_upto, smooth_numbers)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

OUT = []


def say(s=""):
    print(s)
    OUT.append(str(s))


def cofactors(idx, ps):
    """(smooth part, rough part) of every entry of idx, for the prime list ps."""
    v = idx.copy()
    cof = np.ones(len(idx), dtype=np.int64)
    for p in ps:
        while True:
            m = (v % p) == 0
            if not m.any():
                break
            v[m] //= p
            cof[m] *= p
    return cof, v


def main():
    say("# s2 COMPLETE KNOWLEDGE: enumeration, count, families")
    say()

    cases = [(5, 1000), (7, 1000), (11, 1000), (13, 1000), (17, 1000),
             (5, 3162), (11, 3162), (17, 3162),
             (5, 10000), (7, 10000), (11, 10000), (13, 10000), (17, 10000)]

    say("## Z5 / Z6  the rule against the sieve, and the exact count")
    say()
    say("| q | Q | X = Q^2 | open pairs in (Q, Q^2] (sieve) | from the rule | mismatches | "
        "admissible (sieve) | exact count formula |")
    say("|---|---|---|---|---|---|---|---|")

    lastQ, primes = None, None
    for q, Q in cases:
        X = Q * Q
        if Q != lastQ:
            primes = primes_upto(X)
            lastQ = Q
        prQ = primes[primes <= Q]
        p1 = int(primes[primes > Q][0])
        adm = admissible_by_sieve(q, Q, X, prQ)
        rule = admissible_by_rule(q, Q, X, primes)
        o1 = open_pairs(adm)
        o2 = open_pairs(rule)
        mism = int((o1[Q:] != o2[Q:]).sum())
        n1 = int(o1[Q:].sum())
        n2 = int(o2[Q:].sum())
        nadm = int(adm[1:].sum())
        # the exact count formula
        sm = smooth_numbers(q, X)
        piQ = int((primes <= Q).sum())
        pref = np.searchsorted(primes, np.arange(0), side="right")  # placeholder
        tot = len(sm)                                   # Psi(X, q), the P = 1 part
        for s in sm:
            if s * p1 > X:
                break
            tot += int(np.searchsorted(primes, X // s, side="right")) - piQ
        say(f"| {q} | {Q} | {X:,} | {n1:,} | {n2:,} | {mism} | {nadm:,} | {tot:,} |")
        del adm, rule, o1, o2

    # ------------------------------------------------------------- family census
    say()
    say("## Z4  the family census: which (s, s') pairs carry the zone's open pairs")
    say()
    for q, Q in [(5, 10000), (11, 10000), (5, 1000)]:
        X = Q * Q
        primes = primes_upto(X)
        prQ = primes[primes <= Q]
        ps = [int(p) for p in prQ if p <= q]
        adm = admissible_by_sieve(q, Q, X, prQ)
        op = open_pairs(adm)
        idx = np.flatnonzero(op[Q:]).astype(np.int64) + Q
        s1c, r1 = cofactors(idx, ps)
        s2c, r2 = cofactors(idx + 2, ps)
        both_big = (r1 > 1) & (r2 > 1)
        left_sm = (r1 == 1) & (r2 > 1)
        right_sm = (r1 > 1) & (r2 == 1)
        both_sm = (r1 == 1) & (r2 == 1)
        tot = len(idx)
        say(f"### q = {q}, Q = {Q:,}, zone ({Q:,}, {X:,}]: {tot:,} open pairs")
        say()
        say("| type | count | share |")
        say("|---|---|---|")
        for name, m in (("(large, large)", both_big), ("(smooth, large)", left_sm),
                        ("(large, smooth)", right_sm), ("(smooth, smooth) = Stormer", both_sm)):
            say(f"| {name} | {int(m.sum()):,} | {int(m.sum()) / tot:.6f} |")
        lab = Counter(zip(s1c[both_big].tolist(), s2c[both_big].tolist()))
        say()
        say(f"distinct (s, s') families used: **{len(lab)}**; "
            f"q-smooth numbers <= Q: {len(smooth_numbers(q, Q))}")
        say()
        say("| rank | (s, s') | count | share of (large,large) |")
        say("|---|---|---|---|")
        nb = int(both_big.sum())
        for i, ((a, b), c) in enumerate(lab.most_common(12), 1):
            say(f"| {i} | ({a}, {b}) | {c:,} | {c / nb:.4f} |")
        # gcd check
        from math import gcd
        badgcd = [k for k in lab if gcd(k[0], k[1]) not in (1, 2)]
        say()
        say(f"families with gcd(s, s') not in {{1, 2}}: **{len(badgcd)}**")
        # active-family bound: s <= n/p1
        p1 = int(primes[primes > Q][0])
        viol = int((s1c[both_big] * p1 > idx[both_big]).sum())
        say(f"open pairs violating the stratification bound s <= n / p1: **{viol}**")
        say()
        del adm, op, idx, s1c, s2c, r1, r2

    with open(os.path.join(RES, "s2_count.txt"), "w") as f:
        f.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
