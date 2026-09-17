"""Round 72 / loop entry 77: Chen pairs in the window, measured.

A Chen pair is a prime p whose partner p + 2 has at most two prime factors counted with
multiplicity.  Every twin pair is one (`chenPair_of_twin`), so "a Chen pair in every window" is a
strictly weaker statement than the twin window statement - and it is the one the analytic
literature can support, since the parity barrier blocks 2 but not 2-almost-primes.

This measures how far into the window the first Chen pair sits and how many there are, to see
what a proof would have to establish and with what room.
"""

import sys

from sympy import factorint, isprime, primerange


def omega(n):
    return sum(factorint(n).values())


def main():
    print("first Chen pair past the window's start, and the count in the first stretch")
    print("      q   first Chen p   offset   first twin   offset   Chen pairs in (q, 20q]")
    for q in (11, 101, 1009, 10007, 100003, 1000003):
        first_chen = None
        first_twin = None
        count = 0
        for p in primerange(q + 1, 20 * q):
            if omega(p + 2) <= 2:
                count += 1
                if first_chen is None:
                    first_chen = p
            if first_twin is None and isprime(p + 2):
                first_twin = p
        print(
            "%7d  %12d  %7d  %11d  %7d  %22d"
            % (q, first_chen, first_chen - q, first_twin, first_twin - q, count)
        )

    print()
    print("the window is (q, q^2], so the stretch above is a vanishing part of it:")
    for q in (1009, 10007, 100003):
        print("  q = %7d: (q, 20q] is %.3g%% of the window" % (q, 100 * 19.0 * q / (q * q - q)))


if __name__ == "__main__":
    sys.exit(main())
