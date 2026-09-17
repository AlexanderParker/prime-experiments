"""Round 66 / loop entry 71: the multiplicative chain - the next landing is a multiple of this one.

From a landing t (a twin centre), flip about the mirror whose product is t / 2.  Its gears are
exactly the gears dividing t / 2, all of them open at t, and none of them can strike the family
(`mirror_gear_never_strikes`).  The move is 2 (t/2) j = t j, so the candidates are the centres

    t * j,    j = 2, 3, 4, ...

and a landing t * j with j + 3 <= t automatically satisfies the chain condition
(t*j + 1 < (t - 1)^2), so the bands overlap by construction: `mult_chain_window`
[proofs/MirrorWalkChain.lean, round 66].

The whole open statement then reads, with no mention of machines or windows:

    for every twin centre t, some j <= t - 3 has t * j a twin centre.

This measures the smallest such j: over every twin centre to 20000, and by size of the landing.
"""

import sys

from sympy import isprime


def smallest_j(t, cap=4000):
    j = 2
    while j <= cap:
        if isprime(t * j - 1) and isprime(t * j + 1):
            return j
        j += 1
    return None


def twin_centres(lo, hi):
    c = lo - (lo % 6)
    out = []
    while c <= hi:
        if c >= 12 and isprime(c - 1) and isprime(c + 1):
            out.append(c)
        c += 6
    return out


def main():
    cents = twin_centres(12, 20000)
    js = [smallest_j(t) for t in cents]
    worst = max(zip(js, cents))
    print("every twin centre from 12 to 20000: %d landings" % len(cents))
    print(
        "  smallest j: min %d, max %d (at t = %d), mean %.2f; j <= 30 in %.1f%% of cases"
        % (
            min(js),
            worst[0],
            worst[1],
            sum(js) / len(js),
            100 * sum(1 for x in js if x <= 30) / len(js),
        )
    )
    print("  the allowance is j <= t - 3, so the largest j used is %.3g%% of what is allowed"
          % (100 * worst[0] / (worst[1] - 3)))
    print()
    print("by size of the landing (12 landings at each size):")
    print("   t near      j: min   max    mean       allowance t - 3")
    for mag in (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7, 10 ** 8):
        ts = twin_centres(mag, mag + 200000)[:12]
        js = [smallest_j(t) for t in ts]
        print(
            "  1e%-8d      %4d  %4d  %8.1f       %s"
            % (len(str(mag)) - 1, min(js), max(js), sum(js) / len(js), "{:,}".format(ts[0] - 3))
        )


if __name__ == "__main__":
    sys.exit(main())
