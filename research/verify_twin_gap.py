"""Self-contained verification of the twin blocked-slot algorithm.

Ground truth here is plain division by every candidate factor, written out in
full below. Nothing outside this repository is used: the trial divisors come
from the project's own single-residue blocking rule, and the comparison
baseline is direct division.
"""

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from algorithms.twin_gap import next_twin_gap, next_twin_pair, survivor_classes


def divides_any(m):
    """Return the smallest factor of m above 1, or None if m is prime."""
    if m < 2:
        return 1
    if m % 2 == 0:
        return 2 if m > 2 else None
    f = 3
    while f * f <= m:
        if m % f == 0:
            return f
        f += 2
    return None


def is_prime(m):
    return m > 1 and divides_any(m) is None


def brute_twin_pair(n):
    """First twin pair above n, by direct division only."""
    m = n + 1
    while True:
        if is_prime(m) and is_prime(m + 2):
            return m, m + 2
        m += 1


def main():
    random.seed(7)
    cases = list(range(1000, 1100))
    cases += [random.randrange(10**4, 10**6) for _ in range(60)]
    cases += [random.randrange(10**7, 10**8) for _ in range(20)]

    fails = 0
    for n in cases:
        got = next_twin_pair(n)
        expected = brute_twin_pair(n)
        if got != expected:
            fails += 1
            print(f"MISMATCH n={n} algorithm={got} direct={expected}")
    print(f"checked {len(cases)} values of n, {fails} mismatches")

    for e in (6, 8, 10, 12):
        n = 10**e
        g = next_twin_gap(n)
        print(f"n=10^{e}: twin gap={g}  log(n)^2={math.log(n) ** 2:.1f}")

    for y in (10, 30, 100):
        print(f"survivor classes for divisors <= {y}: {survivor_classes(y)}")


if __name__ == "__main__":
    main()
