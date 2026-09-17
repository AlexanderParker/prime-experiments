"""Round 61 / loop entry 66b: the one-flip family at the smallest mirrors, across machines.

oneflip_classes.py found that the best mirror is the smallest one: at every machine tested the
mirror {2, 3, 5} (stride 360) leaves more open candidates than any larger mirror, and mirrors
near sqrt q or above leave few or none.  The reason is the trade: a larger mirror carries more
gear phases but spaces the candidates further apart, so fewer of them fit the window.

This measures, for the mirrors g = 5, 7, 11 and for g = the first gear above sqrt q, across
machines to 20000:
  * the number of open candidates within K = (ln q)^3 periods,
  * the period of the first open one,
  * the number of periods the window itself affords (how much room is left unused).
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


def survey(q, g, gears, K):
    stride = 72 * g
    kmin = max(1, (q + 7) // stride + 1)
    kwin = (q * q + 5 - 1) // stride  # last k with 72 g k - 5 < q^2
    kmax = min(K, kwin)
    if kmax < kmin:
        return 0, 0, None, max(0, kwin - kmin + 1)
    span = kmax - kmin + 1
    hits = bytearray(span)
    for h in gears:
        if stride % h == 0:
            continue
        u = pow(stride % h, -1, h)
        for r in {(7 * u) % h, (5 * u) % h}:
            k = kmin + ((r - kmin) % h)
            while k <= kmax:
                hits[k - kmin] = 1
                k += h
    opens = [kmin + i for i, c in enumerate(hits) if c == 0]
    return span, len(opens), (opens[0] if opens else None), kwin - kmin + 1


def main():
    qs = [101, 251, 503, 1009, 2003, 5003, 10007, 20011]
    print("one-flip family, open candidates by mirror, K = (ln q)^3")
    print("    q  mirror   stride    K   span  open  first  periods the window affords")
    worst = {}
    for q in qs:
        gears = [h for h in primes_to(q) if h >= 5]
        K = int(math.log(q) ** 3)
        gsq = next(p for p in gears if p * p > q)
        for g in [5, 7, 11, gsq]:
            span, nopen, first, afford = survey(q, g, gears, K)
            tag = "%d" % g + (" (>sqrt q)" if g == gsq else "")
            print(
                "%5d  %-11s %6d %4d  %5d  %4d  %5s  %s"
                % (q, tag, 72 * g, K, span, nopen, str(first), "{:,}".format(afford))
            )
            worst.setdefault(g if g != gsq else "sqrt", []).append(nopen)
        print()

    print("minimum open candidates over these machines:")
    for k, v in worst.items():
        print("  mirror %-5s min %d  (values %s)" % (str(k), min(v), v))

    print()
    print("every machine 11..2000, mirror {2,3,5}, K = (ln q)^3: first open period")
    gears_all = [h for h in primes_to(2003) if h >= 5]
    mn = (10 ** 9, None)
    misses = 0
    firsts = []
    for q in [p for p in primes_to(2000) if p >= 11]:
        gears = [h for h in gears_all if h <= q]
        K = int(math.log(q) ** 3)
        span, nopen, first, afford = survey(q, 5, gears, K)
        if first is None:
            misses += 1
        else:
            firsts.append(first - max(1, (q + 7) // 360 + 1))
        if nopen < mn[0]:
            mn = (nopen, q)
    print(
        "  machines with no open candidate: %d;  fewest open: %d at q = %d;  first open at offset %d..%d (mean %.1f)"
        % (misses, mn[0], mn[1], min(firsts), max(firsts), sum(firsts) / len(firsts))
    )


if __name__ == "__main__":
    sys.exit(main())
