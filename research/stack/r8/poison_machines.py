"""Round 83 / loop entry 89: poison machines - the adversary's natural candidates.

If any machine were to fail, the natural suspects are those whose window begins inside a long run
of struck columns: q just below a multiple of a primorial, where the numbers k P# - j for small j
all carry small factors.  This measures the distance from such q to the first twin above it,
against the records over all machines (entry 88: record 1722 at q = 9923987).
"""

import math
import sys

from sympy import isprime, nextprime, prevprime


def first_twin_above(q):
    n = q + 1
    while not (isprime(n) and isprime(n + 2)):
        n += 1
    return n


def main():
    print("machines just below multiples of primorials: distance to the first twin above")
    print("   primorial P#     k       q = prime below k*P#      D = first twin - q   D/(ln q)^2")
    M = 1
    worst = (0, None)
    for P in (5, 7, 11, 13, 17, 19, 23):
        M *= P
        for k in range(1, 8):
            x = k * M
            if x > 10 ** 12:
                break
            q = prevprime(x)
            t = first_twin_above(q)
            d = t - q
            r = d / math.log(q) ** 2
            if d > worst[0]:
                worst = (d, q)
            print("  %13s  %4d  %22d  %20d  %10.2f" % ("%d#" % P, k, q, d, r))
    print()
    print("worst poison machine: D = %d at q = %d, against the all-machine record 1722 at q = 9923987" % worst)


if __name__ == "__main__":
    sys.exit(main())
