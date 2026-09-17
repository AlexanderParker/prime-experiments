"""Round 62 / loop entry 67: the one-flip family is machine-independent, and how far it runs.

With the mirror {2, 3, 5} the stride is 360 and the candidate k has members 360 k - 7 and
360 k - 5.  By the teeth law a gear h strikes k at k = 7u and k = 5u modulo h with
u = 360^{-1} mod h - and none of that mentions the machine.  So there is ONE fixed pattern of
teeth on the k-line, computed once; growing the machine only brings more gears into play, and
never moves a tooth.  (Gear 7: 360 = 3 mod 7, u = 5, teeth at k = 0 and k = 4 mod 7, at every
machine, forever.)

The machine enters in two places only: which gears are present (h <= q), and where the window
starts (k > (q + 7) / 360).  So the open statement for machine q reads: some k in
((q + 7)/360, (q^2 + 5)/360] is missed by the fixed teeth of every gear up to q.

This measures how far along the k-line the first such k sits, across machines to 10^6, to pin
the period law the construction needs.
"""

import math
import sys


def primes_to(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i :: i] = bytearray(len(sieve[i * i :: i]))
    return [i for i in range(2, n + 1) if sieve[i]]


STRIDE = 360


def teeth(h):
    u = pow(STRIDE % h, -1, h)
    return (7 * u) % h, (5 * u) % h


def first_open(q, gears, K):
    """First open k at or above the window's start, and the open count within K periods."""
    kmin = (q + 7) // STRIDE + 1
    kmax = kmin + K - 1
    span = kmax - kmin + 1
    hits = bytearray(span)
    for h in gears:
        if STRIDE % h == 0:
            continue
        for r in set(teeth(h)):
            k = kmin + ((r - kmin) % h)
            while k <= kmax:
                hits[k - kmin] = 1
                k += h
    opens = [kmin + i for i, c in enumerate(hits) if c == 0]
    return kmin, (opens[0] if opens else None), len(opens)


def main():
    print("gear 7 teeth (fixed, machine-independent):", teeth(7))
    print("gear 11:", teeth(11), " gear 13:", teeth(13), " gear 17:", teeth(17))
    print()
    qs = [101, 251, 503, 1009, 2003, 5003, 10007, 20011, 50021, 100003, 200003, 500009, 1000003]
    allp = primes_to(qs[-1])
    print("     q      K=(ln q)^3   window starts at k   first open k   offset   open in K   offset/(ln q)^2")
    for q in qs:
        K = int(math.log(q) ** 3)
        gears = [h for h in allp if 5 <= h <= q]
        kmin, first, n = first_open(q, gears, K)
        if first is None:
            print("%7d  %8d  %14d  %14s  %7s  %9d" % (q, K, kmin, "none", "-", n))
        else:
            off = first - kmin
            print(
                "%7d  %8d  %14d  %14d  %7d  %9d   %.3f"
                % (q, K, kmin, first, off, n, off / (math.log(q) ** 2))
            )


if __name__ == "__main__":
    sys.exit(main())
