"""Round 63 / loop entry 68: which field types kill the one-flip family, and what a bigger
mirror buys.

The family from home with the mirror S (always containing 2 and 3) has members 2 M k +- 1 with
M the product of S, and by the teeth law the gears of S never strike it.  So the mirror is a
choice of which gears to switch off, paid for with the stride.

Part (a): the killer census.  For the first T candidates at or past the window's start, take
each composite member and name its killer by the fields explorer's ids
(research/tools/fields_twin.py):
    higher:g   composite whose smallest gear is g   (higher1:g if g divides exactly once)
    lower:g    composite whose largest gear is g    (lower1:g if g divides exactly once)
    products:j composite with exactly j prime factors counted with multiplicity
    squares    the composite is g^2
This says which gears a mirror would have to carry to remove those kills.

Part (b): the primorial mirrors.  For S = {2,3}, {2,3,5}, ... up to the primorial that still
fits the window, measure the open candidates and the first open period past the window's start.
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


def factor(n, primes):
    f = {}
    m = n
    for p in primes:
        if p * p > m:
            break
        while m % p == 0:
            f[p] = f.get(p, 0) + 1
            m //= p
    if m > 1:
        f[m] = f.get(m, 0) + 1
    return f


def census(q, M, T, gears, primes):
    """Killer census of the first T candidates at or past the window's start."""
    stride = 2 * M
    kmin = (q + 1) // stride + 1
    rows = []
    for k in range(kmin, kmin + T):
        lo, hi = stride * k - 1, stride * k + 1
        killers = []
        for x in (lo, hi):
            f = factor(x, primes)
            if sum(f.values()) == 1:
                continue  # prime member
            s = min(f)
            L = max(f)
            j = sum(f.values())
            kind_h = "higher1:%d" % s if f[s] == 1 else "higher:%d" % s
            kind_l = "lower1:%d" % L if f[L] == 1 else "lower:%d" % L
            sq = (len(f) == 1 and f[s] == 2)
            killers.append((x, s, L, j, kind_h, kind_l, sq))
        rows.append((k, killers))
    return kmin, rows


def survey(q, M, gears, kmax_periods):
    stride = 2 * M
    kmin = (q + 1) // stride + 1
    kwin = (q * q - 1) // stride
    kmax = min(kmin + kmax_periods - 1, kwin)
    if kmax < kmin:
        return 0, None, 0
    span = kmax - kmin + 1
    hits = bytearray(span)
    for h in gears:
        if M % h == 0:
            continue
        v = pow(stride % h, -1, h)
        for r in {v % h, (-v) % h}:
            k = kmin + ((r - kmin) % h)
            while k <= kmax:
                hits[k - kmin] = 1
                k += h
    opens = [kmin + i for i, c in enumerate(hits) if c == 0]
    return span, (opens[0] - kmin if opens else None), len(opens)


def main():
    allp = primes_to(200003)
    T = 60

    print("(a) killer census, mirror {2,3,5} (M = 30), first %d candidates past the window start" % T)
    for q in (1009, 5003, 20011, 100003):
        primes = [p for p in allp if p * p <= (2 * 30) * ((q + 1) // 60 + T + 2) + 2]
        gears = [h for h in allp if 5 <= h <= q]
        kmin, rows = census(q, 30, T, gears, primes)
        smallest = {}
        largest = {}
        jc = {}
        struck = 0
        both = 0
        for k, killers in rows:
            if killers:
                struck += 1
            if len(killers) == 2:
                both += 1
            for (x, s, L, j, kh, kl, sq) in killers:
                smallest[kh] = smallest.get(kh, 0) + 1
                largest[kl] = largest.get(kl, 0) + 1
                jc["products:%d" % j] = jc.get("products:%d" % j, 0) + 1
        top_s = sorted(smallest.items(), key=lambda t: -t[1])[:6]
        top_l = sorted(largest.items(), key=lambda t: -t[1])[:4]
        print("  q = %6d  window starts at k = %d;  struck %d of %d, both members %d" % (q, kmin, struck, T, both))
        print("     by smallest gear: " + ", ".join("%s x%d" % (a, b) for a, b in top_s))
        print("     by largest gear:  " + ", ".join("%s x%d" % (a, b) for a, b in top_l))
        print("     by factor count:  " + ", ".join("%s x%d" % (a, b) for a, b in sorted(jc.items())))
        # how many candidates would survive if the mirror carried every gear up to B
        for B in (5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            saved = 0
            for k, killers in rows:
                if killers and all(s <= B for (_, s, _, _, _, _, _) in killers):
                    saved += 1
            print("     gears up to %2d would clear %2d of the %2d struck candidates" % (B, saved, struck))
        print()

    print("(b) primorial mirrors: open candidates within 400 periods of the window start")
    print("     q     mirror up to   M        stride      span   first open offset   open")
    for q in (1009, 5003, 20011, 100003):
        gears = [h for h in allp if 5 <= h <= q]
        M = 6
        B = 3
        for p in [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]:
            if 2 * M * p > q * q:
                break
            M *= p
            B = p
            span, off, nopen = survey(q, M, gears, 400)
            print(
                "%7d  %10d  %12d  %12d  %6d  %17s  %5d"
                % (q, B, M, 2 * M, span, str(off), nopen)
            )
        print()


if __name__ == "__main__":
    sys.exit(main())
