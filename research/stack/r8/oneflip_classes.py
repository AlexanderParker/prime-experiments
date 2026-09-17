"""Round 61 / loop entry 66: the strike classes of the one-flip family, and the choice of mirror.

The one-flip family from home is the set of columns c = -1 + 12 g k, whose pair members are

    6c - 1 = 72 g k - 7,     6c + 1 = 72 g k - 5.

So a gear h coprime to 72 g strikes the candidate k exactly when

    k == 7 u  (mod h)   or   k == 5 u  (mod h),      u = (72 g)^{-1} mod h.

Every gear's two teeth on the family are therefore the SAME shape - the fixed pair (7, 5) -
scaled by that gear's own unit u.  That is a mechanism, not a count: the family fixes the shape,
the gear fixes the scale, and nothing else enters.

This script
  1. verifies the class law directly (no assumption, division check against the law);
  2. measures, for every admissible mirror gear g, how many candidates of 1..K survive, to see
     whether the choice of g matters and whether g near sqrt q is special;
  3. measures the overlap: how many gears strike, how many candidates each strike lands on, and
     how many candidates are struck by exactly one gear (the ones a single gear is responsible
     for).
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


def classes(h, g):
    """The two classes of k modulo h at which gear h strikes the family of mirror g."""
    u = pow(72 * g % h, -1, h)
    return (7 * u) % h, (5 * u) % h


def verify_law(q, g, gears, K):
    """Check the class law against actual division, for every gear and every k in 1..K."""
    bad = 0
    for h in gears:
        if (72 * g) % h == 0:
            continue
        r1, r2 = classes(h, g)
        for k in range(1, K + 1):
            t = 72 * g * k
            law = (k % h == r1) or (k % h == r2)
            div = ((t - 7) % h == 0) or ((t - 5) % h == 0)
            if law != div:
                bad += 1
    return bad


def survey(q, g, gears, K):
    """Return (open_ks, first_open, n_strikers, struck_once) for the mirror g at machine q."""
    lo = q  # need 72 g k - 7 > q
    hi = q * q  # need 72 g k - 5 < q^2
    kmin = max(1, (lo + 7) // (72 * g) + 1)
    kmax = min(K, (hi + 5 - 1) // (72 * g))
    if kmax < kmin:
        return 0, None, 0, 0, 0
    span = kmax - kmin + 1
    hits = [0] * (span)
    strikers = 0
    for h in gears:
        if (72 * g) % h == 0:
            continue
        r1, r2 = classes(h, g)
        touched = False
        for r in ({r1, r2}):
            k = kmin + ((r - kmin) % h)
            while k <= kmax:
                hits[k - kmin] += 1
                touched = True
                k += h
        if touched:
            strikers += 1
    open_ks = [kmin + i for i, c in enumerate(hits) if c == 0]
    once = sum(1 for c in hits if c == 1)
    return span, (open_ks[0] if open_ks else None), strikers, once, len(open_ks)


def main():
    qs = [500, 1000, 2000, 5000]
    print("law check: classes (7u, 5u) mod h against division")
    for q in qs[:2]:
        gears = primes_to(q)
        g = next(p for p in gears if p * p > q)
        K = 60
        bad = verify_law(q, g, [h for h in gears if h >= 5], K)
        print("  q = %5d  g = %4d  K = %d   mismatches: %d" % (q, g, K, bad))

    print()
    print("open candidates by choice of mirror gear g (K = (ln q)^3, d = +1)")
    print("   q     g       role        span  open  first  strikers  struck-once")
    for q in qs:
        gears = [h for h in primes_to(q) if h >= 5]
        K = int(math.log(q) ** 3)
        gsqrt = next(p for p in gears if p * p > q)
        picks = []
        picks.append((gears[0], "smallest"))
        picks.append((gears[len(gears) // 8], "low"))
        picks.append((gsqrt, "first > sqrt q"))
        picks.append((gears[len(gears) // 2], "median"))
        picks.append((gears[-1], "largest (q)"))
        for g, role in picks:
            span, first, strikers, once, nopen = survey(q, g, gears, K)
            print(
                "%5d  %5d  %-14s  %5d  %4d  %5s  %8d  %11d"
                % (q, g, role, span, nopen, str(first), strikers, once)
            )
        # best and worst over every admissible mirror
        best = (-1, None)
        worst = (10 ** 9, None)
        tot = 0
        n = 0
        for g in gears:
            span, first, strikers, once, nopen = survey(q, g, gears, K)
            if span == 0:
                continue
            tot += nopen
            n += 1
            if nopen > best[0]:
                best = (nopen, g)
            if nopen < worst[0]:
                worst = (nopen, g)
        print(
            "       best g = %d with %d open; worst g = %d with %d open; mean %.2f over %d mirrors"
            % (best[1], best[0], worst[1], worst[0], tot / n, n)
        )
        print()


if __name__ == "__main__":
    sys.exit(main())
