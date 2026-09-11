"""The range rule per gear per field, validated, on one section.

On the survivors S (n = +-1 mod 6), field 1 = primes >= 5, field j = products of exactly j
primes >= 5. Gear g's strikes in field j are g * field_{j-1}; so gear g's OPEN RANGES in
field j are g times the gaps of field j-1 (an interval I is free of gear g's field-j strikes
iff I/g contains no member of field j-1). Combining gears within field j: I is free of field j
iff for every prime g <= sqrt(max I), I/g is free of field j-1 (recursion down to field 1 =
the primes). Combining fields: a column is a twin iff free of every field j >= 2.

This script checks the recursion exactly on the section [lo, hi) and lists the longest
field-2-free and field-3-free ranges (in numbers on S) with the scaled prime gaps that make
them, and the twin columns as the intersection.

Usage: uv run python range_rule.py lo hi
"""
import sys
import numpy as np
from sympy import primerange, factorint, isprime


def main():
    lo, hi = int(sys.argv[1]), int(sys.argv[2])
    S = [n for n in range(lo, hi) if n % 6 in (1, 5)]
    om = {}
    for n in S:
        f = factorint(n)
        om[n] = sum(f.values())  # all factors >= 5 since n in S
    fields = {j: sorted(n for n in S if om[n] == j) for j in range(1, 6)}
    primes_all = list(primerange(5, hi))
    # recursion check for field 2: n in field 2 iff exists prime g with n/g prime
    bad = 0
    for n in S:
        rec = any(n % g == 0 and isprime(n // g) for g in primes_all if g * g <= n)
        if rec != (om[n] == 2): bad += 1
    print(f"section [{lo}, {hi}): |S| = {len(S)}; field sizes " + ", ".join(f"{j}:{len(fields[j])}" for j in fields) + f"; recursion mismatches (field 2) = {bad}")
    # longest field-j-free ranges on S
    for j in (2, 3):
        hits = set(fields[j]); runs = []; start = None; cnt = 0
        for n in S:
            if n in hits:
                if start is not None: runs.append((cnt, start, prev))
                start = None; cnt = 0
            else:
                if start is None: start = n
                cnt += 1; prev = n
        if start is not None: runs.append((cnt, start, prev))
        runs.sort(reverse=True)
        print(f"field {j}: longest free ranges (members of S, start, end): {runs[:4]}")
        if j == 2:
            c, a, b = runs[0]
            # which scaled prime gaps make it: for each small g, the prime gap containing [a/g, b/g]
            for g in (5, 7, 11, 13):
                ps = [p for p in primes_all if p <= b // g + 50]
                below = max((p for p in ps if p < a / g), default=None); above = min((p for p in ps if p > b / g), default=None)
                print(f"   gear {g}: [a/g, b/g] = [{a/g:.1f}, {b/g:.1f}] sits in the prime gap ({below}, {above}), scaled by {g}: ({g*below if below else None}, {g*above if above else None})")
    twins = [n for n in fields[1] if (n + 2) in set(fields[1]) and n % 6 == 5]
    print(f"twin lowers in the section: {len(twins)}; first five {twins[:5]}")


if __name__ == "__main__":
    main()
