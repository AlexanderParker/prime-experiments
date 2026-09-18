"""Round 83 / loop entry 89: the machine's own worst stretches, exactly.

For the gears 5..P with their REAL teeth (two classes each, fixed by arithmetic), the pattern of
struck columns repeats with period the product of those gears.  Within one period there is a
longest run of consecutive struck columns - the paired Jacobsthal length of that gear set, in
columns - and it recurs at every period.  These are the machine's own worst stretches: the places
where a window would have to begin for the first twin to be as far away as the small gears alone
can push it.  This finds them exactly for P up to 23, then puts a machine at the start of each
recurrence and measures the distance to the first twin.
"""

import math
import sys

from sympy import isprime, prevprime


def primes_to(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i :: i] = bytearray(len(sieve[i * i :: i]))
    return [i for i in range(2, n + 1) if sieve[i]]


def longest_run(gears):
    period = 1
    for h in gears:
        period *= h
    struck = bytearray(period)
    for h in gears:
        # column m is struck by h when h | 6m - 1 or h | 6m + 1
        inv6 = pow(6, -1, h)
        for r in ((inv6) % h, (-inv6) % h):
            struck[r::h] = b"\x01" * len(range(r, period, h))
    # longest run, allowing wrap-around
    best = 0
    best_start = 0
    run = 0
    start = 0
    for i in range(2 * period):
        if struck[i % period]:
            if run == 0:
                start = i
            run += 1
            if run > best:
                best, best_start = run, start % period
        else:
            run = 0
    return period, best, best_start


def first_twin_above(q):
    n = q + 1
    while not (isprime(n) and isprime(n + 2)):
        n += 1
    return n


def main():
    print("longest run of struck columns for the gears 5..P with their real teeth, per period")
    print("    P        period   longest run   run in numbers (x6)   window needed (run x 6)^0.5")
    runs = {}
    for P in (7, 11, 13, 17, 19, 23):
        gears = [h for h in primes_to(P) if h >= 5]
        period, best, start = longest_run(gears)
        runs[P] = (period, best, start)
        print("  %3d  %12d  %11d  %20d  %25.1f" % (P, period, best, 6 * best, math.sqrt(6 * best)))
    print()
    print("a machine placed at the start of the worst stretch: distance to its first twin")
    print("    P    k    q (prime just below the stretch)     D     D/(ln q)^2   run x 6")
    for P in (13, 17, 19, 23):
        period, best, start = runs[P]
        for k in (1, 2, 3):
            x = k * period + start  # first struck column of the run, in this recurrence
            n0 = 6 * x - 1
            q = prevprime(n0)
            t = first_twin_above(q)
            d = t - q
            print("  %3d  %3d  %30d  %6d  %10.2f  %8d" % (P, k, q, d, d / math.log(q) ** 2, 6 * best))


if __name__ == "__main__":
    sys.exit(main())
