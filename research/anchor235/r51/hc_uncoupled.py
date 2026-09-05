"""hc_uncoupled.py -- what an uncoupled distance IS, in column coordinates.

Claim tested: for v < y^2/3, v is uncoupled in M = {5..y} (no gear of M divides v,
3v-1 or 3v+1) iff
   * every prime factor of v is <= 3 or > y, AND
   * (v even)  column v/2 is a TWIN column whose two members both exceed y;
   * (v odd)   both halves (3v-1)/2, (3v+1)/2 have all their prime factors <= 2 or > y,
               which for v < y^2/3 forces each odd part to be 1 or a prime > y.
Also lists the uncoupled sizes of every machine {5..y}, y prime <= 199.
Writes results/hc_uncoupled.txt.
"""
import os
from sympy import primerange, factorint, isprime

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)


def main():
    lines = []
    bad = []
    tested = 0
    for y in primerange(5, 200):
        gears = list(primerange(5, y + 1))
        lim = (y * y) // 3
        unc = []
        for v in range(2, lim + 1):
            if any(v % g == 0 for g in gears):
                continue
            if any((3 * v - 1) % g == 0 or (3 * v + 1) % g == 0 for g in gears):
                continue
            unc.append(v)
            tested += 1
            # the characterisation
            okv = all(p <= 3 or p > y for p in factorint(v))
            if v % 2 == 0:
                c = v // 2
                lo, hi = 6 * c - 1, 6 * c + 1
                oktw = isprime(lo) and isprime(hi) and lo > y and hi > y
                if not (okv and oktw):
                    bad.append((y, v, "even", okv, oktw))
            else:
                h1, h2 = (3 * v - 1) // 2, (3 * v + 1) // 2
                def odd_part(n):
                    while n % 2 == 0:
                        n //= 2
                    return n
                o1, o2 = odd_part(h1), odd_part(h2)
                okh = all(o == 1 or (isprime(o) and o > y) for o in (o1, o2))
                if not (okv and okh):
                    bad.append((y, v, "odd", okv, okh))
        if y <= 43:
            lines.append("y=%-4d gears %-2d  uncoupled v < y^2/3 = %-5d : %s"
                         % (y, len(gears), lim, unc))
    lines.append("")
    lines.append("characterisation tested at %d (machine, uncoupled size) cells, y prime 5..199, "
                 "v < y^2/3: exceptions = %d" % (tested, len(bad)))
    if bad:
        lines.append("  " + repr(bad[:10]))

    # the even uncoupled sizes are exactly 2 * (twin columns above y)
    lines.append("")
    lines.append("even uncoupled sizes as twice a twin column, per machine:")
    for y in [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]:
        gears = list(primerange(5, y + 1))
        lim = (y * y) // 3
        rows = []
        for v in range(2, lim + 1, 2):
            if any(v % g == 0 for g in gears):
                continue
            if any((3 * v - 1) % g == 0 or (3 * v + 1) % g == 0 for g in gears):
                continue
            c = v // 2
            rows.append("%d=2*%d (%d,%d)" % (v, c, 6 * c - 1, 6 * c + 1))
        lines.append("  y=%-3d : %s" % (y, ", ".join(rows)))
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(OUT, "hc_uncoupled.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
