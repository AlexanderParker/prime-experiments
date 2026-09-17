"""Round 63 / loop entry 68c: the chain of primorial-mirror landings.

The flip from home about the mirror {2, 3, ...B} (product M = B#) at period k lands on the pair
2 M k +- 1.  A landing t = 2 M k that is a twin pair serves EVERY machine q with

    sqrt(t) <= q < t - 1,

because then the pair lies inside that machine's window (q, q^2].  So the machines are covered
by a CHAIN of landings whose bands overlap, and the overlap condition is just

    t(next) + 1 < (t - 1)^2,

each landing below the square of the one before (kernel: `chain_covers`,
proofs/MirrorWalkChain.lean).

This finds, for each primorial mirror, the first period whose landing is a twin, and checks the
chain condition between consecutive mirrors.
"""

import math
import sys

from sympy import isprime, primerange


def main():
    M = 6
    rows = []
    print("   B          stride 2M      k        landing t          band: q from        to")
    for B in primerange(5, 130):
        M *= B
        s = 2 * M
        k = None
        for kk in range(1, 400):
            t = s * kk
            if isprime(t - 1) and isprime(t + 1):
                k = kk
                break
        if k is None:
            print("  %3d   no twin landing within 400 periods" % B)
            continue
        t = s * k
        lo = math.isqrt(t) + 1
        rows.append((B, s, k, t, lo))
        print(
            "  %3d  %16s  %5d  %16s  %20d  %16s"
            % (B, "%.6g" % s, k, "%.8g" % t, lo, "%.6g" % (t - 1))
        )

    print()
    ok = 0
    for (B0, _, _, t0, _), (B1, _, _, t1, _) in zip(rows, rows[1:]):
        good = t1 + 1 < (t0 - 1) ** 2
        ok += good
        if not good:
            print("  chain condition FAILS between B = %d and B = %d" % (B0, B1))
    print("chain condition holds at %d of %d consecutive pairs" % (ok, len(rows) - 1))
    print(
        "the %d landings cover every machine from %d to %s"
        % (len(rows), rows[0][4], "%.6g" % (rows[-1][3] - 1))
    )
    print()
    print("slack: how many times larger the period k could have been and still chain")
    for (B0, _, _, t0, _), (B1, s1, k1, t1, _) in zip(rows, rows[1:]):
        kmax = ((t0 - 1) ** 2 - 1) // s1
        print("  B = %3d: k = %4d, allowed up to %s  (factor %.3g)" % (B1, k1, "%.6g" % kmax, kmax / k1))


if __name__ == "__main__":
    sys.exit(main())
