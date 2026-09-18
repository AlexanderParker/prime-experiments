"""Round 82 / loop entry 88: the counterexample hunt with prejudice, part B - the closest calls.

A counterexample is a machine q whose window (q, q^2] has no twin.  The window is enormous, so the
first place to look is not the whole window but its bottom edge: the distance from q to the first
twin above it.  This scans every prime q up to a bound and records the worst cases - the largest
distance to the first twin, in absolute terms and against (ln q)^2 - and checks whether any
distance ever approaches the window's length.
"""

import math
import sys


def primes_to(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i :: i] = bytearray(len(sieve[i * i :: i]))
    return sieve


def main():
    LIMIT = 10 ** 7
    PAD = 20000
    sieve = primes_to(LIMIT + PAD)
    # twin lower members, ascending
    twins = [n for n in range(3, LIMIT + PAD - 2) if sieve[n] and sieve[n + 2]]
    print("machines to %d: distance from q to the first twin above q" % LIMIT)
    print()
    worst_abs = []  # (distance, q)
    worst_rel = []  # (distance / (ln q)^2, q, distance)
    j = 0
    q = 2
    count = 0
    maxratio_window = 0.0
    for n in range(11, LIMIT + 1):
        if not sieve[n]:
            continue
        q = n
        while twins[j] <= q:
            j += 1
        d = twins[j] - q
        count += 1
        worst_abs.append((d, q))
        worst_rel.append((d / (math.log(q) ** 2), q, d))
        r = d / (q * q - q)
        if r > maxratio_window:
            maxratio_window = r
        if len(worst_abs) > 20000:
            worst_abs.sort(reverse=True)
            worst_abs = worst_abs[:10]
            worst_rel.sort(reverse=True)
            worst_rel = worst_rel[:10]
    worst_abs.sort(reverse=True)
    worst_rel.sort(reverse=True)
    print("machines scanned: %d" % count)
    print()
    print("largest distance to the first twin above q:")
    print("       q      distance   distance / (ln q)^2   distance / window length")
    for d, q in worst_abs[:8]:
        print("  %8d  %10d  %19.2f  %24.3g" % (q, d, d / math.log(q) ** 2, d / (q * q - q)))
    print()
    print("largest distance relative to (ln q)^2:")
    print("       q      distance   distance / (ln q)^2")
    for r, q, d in worst_rel[:8]:
        print("  %8d  %10d  %19.2f" % (q, d, r))
    print()
    print("largest share of the window ever needed to reach the first twin: %.3g" % maxratio_window)


if __name__ == "__main__":
    sys.exit(main())
