"""Round 65 / loop entry 70: the certificate - how few landings settle every machine below X.

A landing t (a twin pair t - 1, t + 1) covers every machine q with sqrt(t) <= q < t - 1, and a
chain of landings covers everything provided each is below the square of the one before
(`chain_covers`, `chain_covers_upto`, proofs/MirrorWalkChain.lean).  Taking each link as LARGE as
the condition allows - the greatest twin centre below the previous link squared - makes the
chain as short as it can be: every link roughly squares the range covered, so the number of
landings needed to settle every machine below X grows like log log X.

This builds that chain and reports, for each link, how far below the square it sits (in steps of
6, the spacing of columns) and which mirror reaches it from home in one flip: a landing
t = 2 M k is reached by the mirror of product M, so the largest mirror available for a link is
the largest divisor of t / 2 made of distinct small gears.
"""

import sys

from sympy import isprime


def largest_mirror(t):
    """The product of the gears (primes) dividing t / 2 - the largest mirror reaching t."""
    half = t // 2
    m = 1
    d = 2
    x = half
    while d * d <= x:
        if x % d == 0:
            m *= d
            while x % d == 0:
                x //= d
        d += 1
    if x > 1:
        m *= x
    return m


def main():
    t = 12
    print("the greedy chain: each landing the largest twin centre below the square of the last")
    print("   n              landing t   digits   steps of 6 below the square   covers machines up to   mirror reaching it")
    n = 0
    links = [t]
    while t < 10 ** 200:
        lim = (t - 1) ** 2 - 1
        c = lim - (lim % 6)
        steps = 0
        while not (isprime(c - 1) and isprime(c + 1)):
            c -= 6
            steps += 1
        n += 1
        mirror = largest_mirror(c) if c < 10 ** 30 else None
        print(
            "  %2d  %21s  %6d  %27d  %21s   %s"
            % (n, "%.6g" % c, len(str(c)), steps, "%.6g" % (c - 1), ("%.6g" % mirror) if mirror else "(not factored)")
        )
        t = c
        links.append(t)

    print()
    ok = all(links[i + 1] + 1 < (links[i] - 1) ** 2 for i in range(len(links) - 1))
    print("chain condition holds at every link: %s" % ok)
    print(
        "%d landings settle every machine from %d to 10^%d"
        % (len(links), links[0] - 1, len(str(links[-1])) - 1)
    )


if __name__ == "__main__":
    sys.exit(main())
