"""A closed-form expression for the distance from a slot to the next twin prime.

Written out explicitly, with no search, no iteration over candidates in the notation, and no
appeal to a primality oracle. Every ingredient is an elementary closed form.

    divisibility     [q divides K]  =  (1/q) sum_{c=0}^{q-1} cos(2 pi c K / q)
    primality        [q is prime]   =  floor( cos^2( pi ((q-1)! + 1) / q ) )      (Wilson)
    the pair at m    members 6m - 1 and 6m + 1, product 36 m^2 - 1

A gear threatens slot `m` exactly when it divides `36 m^2 - 1` (section 28d), so the indicator
that slot `m` is a twin slot is

    E(m) = prod_{q=5}^{floor(sqrt(6m+1))} ( 1 - [q prime] * [q divides 36 m^2 - 1] )

and the distance from `m0` to the next twin slot is

    J(m0) = sum_{J>=1} prod_{i=1}^{J} ( 1 - E(m0 + i) )

since each inner product is 1 while no twin has yet been passed and 0 afterwards, so the outer
sum counts steps until the first one. Equivalently, as a weighted form,

    J(m0) = sum_{J>=1} J * E(m0 + J) * prod_{i=1}^{J-1} ( 1 - E(m0 + i) )

The twin pair is then `(6(m0 + J) - 1, 6(m0 + J) + 1)`.

This module implements the formula literally - character sums for divisibility, Wilson floors for
primality - and checks it against direct computation. It is a closed form in exactly the sense
that Willans' formula for the nth prime is one: finite sums and products of elementary functions,
correct as written. What it does not do is separate the answer from the search; the indicators
carry all the work, and the formula gives no bound on `J`.
"""

import sys
from math import cos, factorial, floor, isqrt, pi
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def divides_char(q, K):
    """[q divides K] as the character sum, evaluated literally."""
    total = sum(cos(2 * pi * c * (K % q) / q) for c in range(q))
    return round(total / q)


def is_prime_wilson(q):
    """[q is prime] by Wilson's theorem, as a floor of a cosine square."""
    if q < 2:
        return 0
    x = (factorial(q - 1) + 1) % q  # reduce first; the formula is unchanged
    return floor(cos(pi * x / q) ** 2)


def exposure_closed(m):
    """E(m): 1 if slot m is a twin slot, by the closed-form product."""
    out = 1
    for q in range(5, isqrt(6 * m + 1) + 1):
        out *= 1 - is_prime_wilson(q) * divides_char(q, 36 * m * m - 1)
        if out == 0:
            break  # the product is already zero; value unchanged
    return out


def next_twin_distance(m0, cap=4096):
    """J(m0) by the closed-form sum of products."""
    total = 0
    running = 1
    for J in range(1, cap + 1):
        running *= 1 - exposure_closed(m0 + J)
        total += running
        if running == 0:
            break
    return total + 1 if False else total + 1  # J = (steps skipped) + 1


def next_twin_distance_weighted(m0, cap=4096):
    """The weighted form: sum over J of J * E * prod of (1 - E)."""
    total = 0
    running = 1
    for J in range(1, cap + 1):
        e = exposure_closed(m0 + J)
        total += J * e * running
        running *= 1 - e
        if running == 0:
            break
    return total


if __name__ == "__main__":
    def prime_direct(n):
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        d = 3
        while d * d <= n:
            if n % d == 0:
                return False
            d += 2
        return True

    print("the closed-form ingredients, checked")
    print(f"  {'q':>4} {'Wilson [q prime]':>17} {'direct':>8} {'agree':>7}")
    for q in range(2, 20):
        w = is_prime_wilson(q)
        print(f"  {q:>4} {w:>17} {int(prime_direct(q)):>8} "
              f"{str(bool(w) == prime_direct(q)):>7}")

    print("\n  character sum [q | K] against the remainder test")
    for q, K in ((5, 35), (5, 36), (7, 48), (7, 49), (11, 120), (13, 169)):
        print(f"    q={q:>3} K={K:>5}: character sum {divides_char(q, K)}, "
              f"K mod q == 0 is {K % q == 0}")

    print("\nJ(m0) from the closed form, against direct computation")
    print(f"  {'m0':>7} {'J closed form':>14} {'J weighted form':>16} {'J direct':>9} "
          f"{'next pair':>22} {'agree':>7}")
    for m0 in (1, 5, 20, 50, 100, 200, 400, 1000):
        j1 = next_twin_distance(m0)
        j2 = next_twin_distance_weighted(m0)
        j3 = next(J for J in range(1, 5000)
                  if prime_direct(6 * (m0 + J) - 1) and prime_direct(6 * (m0 + J) + 1))
        m = m0 + j3
        print(f"  {m0:>7} {j1:>14} {j2:>16} {j3:>9} "
              f"{f'({6 * m - 1}, {6 * m + 1})':>22} "
              f"{str(j1 == j2 == j3):>7}")
