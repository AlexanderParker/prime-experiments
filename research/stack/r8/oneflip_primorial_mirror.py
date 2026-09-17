"""Round 63 / loop entry 68b: how close to the window's start the primorial mirrors land.

With the mirror S = every gear up to B (plus 2 and 3), M = B# / 1, the family's members are
2 M k +- 1 and the gears of S never strike it (mirror_gear_never_strikes).  The bigger B is, the
fewer gears can strike - and the further apart the candidates sit.  This measures the offset of
the FIRST open candidate past the window's start, which is what the construction has to walk.
"""

import sys


def primes_to(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i :: i] = bytearray(len(sieve[i * i :: i]))
    return [i for i in range(2, n + 1) if sieve[i]]


def first_open(q, M, gears, T):
    stride = 2 * M
    kmin = (q + 1) // stride + 1
    kwin = (q * q - 1) // stride
    kmax = min(kmin + T - 1, kwin)
    if kmax < kmin:
        return None, 0, 0
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
    opens = [i for i, c in enumerate(hits) if c == 0]
    return (opens[0] if opens else None), len(opens), span


def main():
    QS = [1009, 5003, 20011, 50021, 100003, 200003, 500009, 1000003]
    BS = [5, 7, 11, 13, 17, 19, 23, 29]
    allp = primes_to(QS[-1])
    T = 200
    print("first open period past the window's start, by primorial mirror (T = %d periods tried)" % T)
    header = "      q  " + "".join("  B=%-6d" % b for b in BS)
    print(header)
    for q in QS:
        gears = [h for h in allp if 5 <= h <= q]
        row = "%7d  " % q
        M = 6
        for b in BS:
            M *= b
            if 2 * M > q * q:
                row += "  %-8s" % "-"
                continue
            off, nopen, span = first_open(q, M, gears, T)
            row += "  %-8s" % (str(off) if off is not None else "none/%d" % span)
        print(row)
    print()
    print("(- = the mirror does not fit the window; none/n = no open candidate in the n periods available)")
    print()
    print("open candidates of the first %d periods, same mirrors" % T)
    print(header)
    for q in QS:
        gears = [h for h in allp if 5 <= h <= q]
        row = "%7d  " % q
        M = 6
        for b in BS:
            M *= b
            if 2 * M > q * q:
                row += "  %-8s" % "-"
                continue
            off, nopen, span = first_open(q, M, gears, T)
            row += "  %-8s" % ("%d/%d" % (nopen, span))
        print(row)


if __name__ == "__main__":
    sys.exit(main())
