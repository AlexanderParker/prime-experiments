"""Round 62 / loop entry 67: the family corrected, and its teeth law.

Entry 66 wrote the family's members as 72 g k - 7 and 72 g k - 5.  That is wrong: it treated the
landing -1 + 12 g k d as a COLUMN when it is a MEMBER.  The flip about the mirror {2, 3, ...g}
sends the home pair (-1, +1) to (2A - 1, 2A + 1) with the axis A = k * 6g, so the landing pair is

    (12 g k - 1,  12 g k + 1),     column m = 2 g k.

(Entry 64's measurement used this, correctly; entry 66's scripts did not.)

The teeth law on the corrected family: a gear h not dividing 12 g strikes the candidate k
exactly when

    12 g k == +1  or  -1   (mod h),   that is   k == +v  or  -v   (mod h),   v = (12 g)^{-1} mod h.

So each gear's two teeth are a SYMMETRIC pair, +v and -v, about k = 0 - which is home itself.
A gear dividing 12 g never strikes at all (it would have to divide 1).  This is the mirror's
carrying property read on the candidate line: the mirror's own gears are open by construction
along the whole family, and every other gear has its two teeth placed symmetrically about home.

Measures: the law against division, the mirror comparison redone, and the machines where the
family is empty.
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


def verify_law(g, gears, K):
    bad = 0
    stride = 12 * g
    for h in gears:
        if stride % h == 0:
            # must never strike
            for k in range(1, K + 1):
                if (stride * k - 1) % h == 0 or (stride * k + 1) % h == 0:
                    bad += 1
            continue
        v = pow(stride % h, -1, h)
        for k in range(1, K + 1):
            law = (k % h == v % h) or (k % h == (-v) % h)
            div = (stride * k - 1) % h == 0 or (stride * k + 1) % h == 0
            if law != div:
                bad += 1
    return bad


def survey(q, g, gears, K):
    """Open candidates of the family 12 g k +- 1 inside the window (q, q^2]."""
    stride = 12 * g
    kmin = max(1, (q + 1) // stride + 1)
    kwin = (q * q - 1) // stride
    kmax = min(K, kwin)
    if kmax < kmin:
        return 0, 0, None, max(0, kwin - kmin + 1)
    span = kmax - kmin + 1
    hits = bytearray(span)
    for h in gears:
        if stride % h == 0:
            continue
        v = pow(stride % h, -1, h)
        for r in {v % h, (-v) % h}:
            k = kmin + ((r - kmin) % h)
            while k <= kmax:
                hits[k - kmin] = 1
                k += h
    opens = [kmin + i for i, c in enumerate(hits) if c == 0]
    return span, len(opens), (opens[0] if opens else None), kwin - kmin + 1


def main():
    allp = primes_to(20011)
    print("law check: teeth at +-v, gears dividing 12g never strike")
    for q, g in ((1000, 5), (1000, 37), (5000, 71)):
        gears = [h for h in allp if 5 <= h <= q]
        print("  q = %5d  g = %3d  K = 80  mismatches: %d" % (q, g, verify_law(g, gears, 80)))

    print()
    print("open candidates by mirror, corrected family, K = (ln q)^3")
    print("    q   mirror  stride    K   span  open  first   window affords")
    for q in [101, 251, 503, 1009, 2003, 5003, 10007, 20011]:
        gears = [h for h in allp if 5 <= h <= q]
        K = int(math.log(q) ** 3)
        gsq = next(p for p in gears if p * p > q)
        for g in [1, 5, 7, 11, gsq]:
            span, nopen, first, afford = survey(q, g, gears, K)
            tag = "%d%s" % (g, "*" if g == gsq else "")
            print(
                "%6d  %-6s %6d %5d %6d %5d %6s   %s"
                % (q, tag, 12 * g, K, span, nopen, str(first), "{:,}".format(afford))
            )
        print()
    print("(* = first gear above sqrt q; mirror 1 = {2,3}, 5 = {2,3,5}, ...)")

    print()
    print("every machine 11..20011, mirror {2,3,5}, K = (ln q)^3:")
    zero = []
    mn = (10 ** 9, None)
    offs = []
    for q in [p for p in allp if p >= 11]:
        gears = [h for h in allp if 5 <= h <= q]
        K = int(math.log(q) ** 3)
        span, nopen, first, afford = survey(q, 5, gears, K)
        if nopen == 0:
            zero.append(q)
        else:
            offs.append(first - (max(1, (q + 1) // 60 + 1)))
        if nopen < mn[0]:
            mn = (nopen, q)
    print(
        "  machines with none: %s (count %d);  fewest open %d at q = %d;  first-open offset %d..%d, mean %.1f"
        % (zero[:10], len(zero), mn[0], mn[1], min(offs), max(offs), sum(offs) / len(offs))
    )


if __name__ == "__main__":
    sys.exit(main())
