"""hc_identity.py -- the exact identities of the half-column map (P1-P4).

P1: both letters of a gear point at its home column.
P2: for even v, Leg(v) = prime factors >= 5 of the two members of column v/2.
P3: for odd v, one of (3v-1)/2, (3v+1)/2 is a member of the quarter column.
P4: island coupling gears = Pad(delta) u Leg(delta) above 7.

Self-contained; no arguments.  Writes results/hc_identity.txt.
"""
import os
from sympy import primerange, factorint

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)


def u_of(g):
    """home column of gear g: the integer u with 6u = g -+ 1."""
    return (g + 1) // 6 if g % 6 == 5 else (g - 1) // 6


def members(c):
    return (6 * c - 1, 6 * c + 1)


def leg(v, gears):
    """{g in gears : g | 3v-1 or g | 3v+1} -- the letter gears of a middle gap v."""
    return {g for g in gears if (3 * v - 1) % g == 0 or (3 * v + 1) % g == 0}


def odd_part(n):
    while n % 2 == 0:
        n //= 2
    return n


def main():
    lines = []
    P = list(primerange(5, 10008))

    # ---- P1: both letters of a gear point at its home column -------------
    bad1 = []
    for g in P:
        u = u_of(g)
        if u != round(g / 6):
            bad1.append((g, "u != round(g/6)"))
        a = 2 * u
        b = g - 2 * u
        # short letter: half-column is exactly u, and g is a member of column u
        if a % 2 != 0 or a // 2 != u:
            bad1.append((g, "h(a) != u"))
        if g not in members(u):
            bad1.append((g, "g not a member of column u"))
        # long letter: b is odd, 3b = 2g +- 1, and one of 3b -+ 1 is 2g
        if b % 2 == 0:
            bad1.append((g, "b even"))
        if 3 * b not in (2 * g - 1, 2 * g + 1):
            bad1.append((g, "3b != 2g +- 1"))
        if 2 * g not in (3 * b - 1, 3 * b + 1):
            bad1.append((g, "2g not a member of half-column b/2"))
        # the odd part of that even member is g itself, in column u
        ev = 3 * b - 1 if (3 * b - 1) % 2 == 0 else 3 * b + 1
        # both 3b-1 and 3b+1 are even (b odd); the one equal to 2g:
        ev = 2 * g
        if odd_part(ev) != g:
            bad1.append((g, "odd part of 2g is not g"))
    lines.append("P1  gears 5..10007 (%d gears): exceptions = %d" % (len(P), len(bad1)))
    if bad1:
        lines.append("    " + repr(bad1[:10]))
    # the letters of the smallest gears, in column coordinates
    lines.append("    gear  u_g  a_g  h(a_g)  b_g  3b_g  members of column u_g")
    for g in P[:9]:
        u = u_of(g)
        lines.append("    %4d %4d %4d %6d %5d %5d   %s" %
                     (g, u, 2 * u, u, g - 2 * u, 3 * (g - 2 * u), str(members(u))))

    # ---- P2: the even map -------------------------------------------------
    gears = list(primerange(5, 2000))
    bad2 = []
    for v in range(2, 2001, 2):
        c = v // 2
        lo, hi = members(c)
        want = set()
        for m in (lo, hi):
            for p in factorint(m):
                if p >= 5 and p < 2000:
                    want.add(p)
        got = leg(v, gears)
        if want != got:
            bad2.append(v)
    lines.append("P2  even v = 2..2000 (%d values), gears 5..1999: exceptions = %d"
                 % (1000, len(bad2)))
    if bad2:
        lines.append("    " + repr(bad2[:10]))

    # ---- P3: the odd map --------------------------------------------------
    bad3 = []
    tab3 = []
    for v in range(1, 2000, 2):
        h1, h2 = (3 * v - 1) // 2, (3 * v + 1) // 2
        cop = [m for m in (h1, h2) if m % 6 in (1, 5)]
        if len(cop) != 1:
            bad3.append((v, "not exactly one member coprime to 6", h1, h2))
            continue
        m = cop[0]
        if v % 4 == 1:
            c = (v - 1) // 4
            ok = (m == 6 * c + 1) and (m == h1)
        else:
            c = (v + 1) // 4
            ok = (m == 6 * c - 1) and (m == h2)
        if not ok:
            bad3.append((v, "quarter column wrong", h1, h2, c))
        # the other half must be even
        other = h2 if m == h1 else h1
        if other % 2 != 0:
            bad3.append((v, "other half not even", h1, h2))
        # and Leg(v) = odd prime factors >= 5 of h1 and h2
        want = set()
        for x in (h1, h2):
            for p in factorint(x):
                if p >= 5 and p < 2000:
                    want.add(p)
        got = leg(v, gears)
        if want != got:
            bad3.append((v, "Leg mismatch", want, got))
    lines.append("P3  odd v = 1..1999 (%d values): exceptions = %d" % (1000, len(bad3)))
    if bad3:
        lines.append("    " + repr(bad3[:10]))
    lines.append("    odd v   (3v-1)/2  (3v+1)/2  quarter column c  member  Leg(v)")
    for v in (1, 3, 5, 7, 9, 11, 23, 25, 41, 57):
        h1, h2 = (3 * v - 1) // 2, (3 * v + 1) // 2
        c = (v - 1) // 4 if v % 4 == 1 else (v + 1) // 4
        m = h1 if v % 4 == 1 else h2
        lines.append("    %5d %9d %9d %14d %8d  %s" %
                     (v, h1, h2, c, m, sorted(leg(v, gears))))

    # ---- P4: the island coupling map --------------------------------------
    bad4 = []
    for delta in range(1, 1001):
        g11 = {g for g in gears if g > 7 and delta % g == 0}
        second = {g for g in gears if g > 7 and
                  ((3 * delta - 1) % g == 0 or (3 * delta + 1) % g == 0)}
        want = g11 | second
        got = {g for g in gears if g > 7 and
               (delta % g == 0 or (3 * delta - 1) % g == 0 or (3 * delta + 1) % g == 0)}
        # and the column reading of `second`
        if delta % 2 == 0:
            c = delta // 2
            col = set()
            for m in members(c):
                for p in factorint(m):
                    if 7 < p < 2000:
                        col.add(p)
            if col != second:
                bad4.append((delta, "column reading", col, second))
        if want != got:
            bad4.append((delta, "set mismatch"))
    lines.append("P4  delta = 1..1000: exceptions = %d" % len(bad4))
    if bad4:
        lines.append("    " + repr(bad4[:10]))

    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, "hc_identity.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
