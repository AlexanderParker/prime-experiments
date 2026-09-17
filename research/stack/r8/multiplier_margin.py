"""Round 68 / loop entry 73: the margin of the multiplicative step, and what the multipliers look like.

The open content of the construction is now: for every twin centre t, some j <= t - 3 has t * j a
twin centre.  This measures how much room that statement has - how many such j there are, not
just the first - and what those multipliers are made of: whether they use the gears the landing
already carries (which can never strike, `landing_gear_never_strikes`) or bring new ones.
"""

import sys

from sympy import factorint, isprime


def twin_centres(lo, hi):
    out = []
    c = lo - (lo % 6)
    while c <= hi:
        if c >= 12 and isprime(c - 1) and isprime(c + 1):
            out.append(c)
        c += 6
    return out


def main():
    print("all multipliers j <= t - 3 with t * j a twin centre")
    print("      t   allowance   working j   first   share of working j sharing a gear with t")
    rows = []
    for t in twin_centres(12, 3000):
        gears = set(factorint(t))
        works = []
        shared = 0
        for j in range(2, t - 2):
            if isprime(t * j - 1) and isprime(t * j + 1):
                works.append(j)
                if gears & set(factorint(j)):
                    shared += 1
        rows.append((t, t - 4, len(works), works[0] if works else None, shared))
        if t in (12, 18, 30, 42, 60, 72, 102, 108, 138, 150, 180, 192, 198, 228, 240, 270, 282,
                 462, 522, 570, 600, 618, 642, 660, 810, 822, 1020, 1032, 1050, 1062, 1092,
                 1152, 1230, 1278, 1290, 1302, 1320, 1428, 1452, 1482, 1488, 1608, 1620, 1668,
                 1698, 1722, 1788, 1872, 1932, 1950, 1998, 2028, 2088, 2130, 2142, 2238, 2262,
                 2270, 2310, 2340, 2382, 2550, 2688, 2730, 2802, 2970, 2999):
            print(
                "%7d  %10d  %10d  %6s  %28.1f%%"
                % (t, t - 4, len(works), str(works[0] if works else "-"),
                   (100 * shared / len(works)) if works else 0.0)
            )

    print()
    none = [r for r in rows if r[2] == 0]
    print("landings with no working multiplier: %d of %d" % (len(none), len(rows)))
    print("smallest margin: %d (at t = %d)" % (min(r[2] for r in rows), min(rows, key=lambda r: r[2])[0]))
    for lo, hi in ((12, 300), (300, 1000), (1000, 2000), (2000, 3000)):
        sel = [r for r in rows if lo <= r[0] < hi]
        if sel:
            print(
                "  t in [%4d, %4d): %3d landings, working multipliers %d to %d, mean %.1f, mean share of the allowance %.2f%%"
                % (lo, hi, len(sel), min(r[2] for r in sel), max(r[2] for r in sel),
                   sum(r[2] for r in sel) / len(sel),
                   100 * sum(r[2] / r[1] for r in sel) / len(sel))
            )


if __name__ == "__main__":
    sys.exit(main())
