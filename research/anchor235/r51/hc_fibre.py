"""hc_fibre.py -- the fibre of the half-column map is the alphabet of the column.

Claim: exactly three distances v have half/quarter column c, namely 2c, 4c-1, 4c+1;
and these are exactly the letters of the gears of column c:
   2c    = a_{6c-1} = a_{6c+1}   (the short letter, shared by both members)
   4c-1  = b_{6c-1}
   4c+1  = b_{6c+1}
Writes results/hc_fibre.txt.
"""
import os
from sympy import isprime

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)


def u_of(g):
    return (g + 1) // 6 if g % 6 == 5 else (g - 1) // 6


def col_index(v):
    if v % 2 == 0:
        return v // 2
    return (v - 1) // 4 if v % 4 == 1 else (v + 1) // 4


def main():
    lines = []
    CMAX = 2000
    VMAX = 4 * CMAX + 1
    fib = {}
    for v in range(1, VMAX + 1):
        fib.setdefault(col_index(v), []).append(v)
    bad = []
    for c in range(1, CMAX + 1):
        want = sorted([2 * c, 4 * c - 1, 4 * c + 1])
        got = sorted(fib.get(c, []))
        if got != want:
            bad.append((c, want, got))
    lines.append("fibre of the half-column map over columns 1..%d: exceptions = %d"
                 % (CMAX, len(bad)))
    if bad:
        lines.append("  " + repr(bad[:5]))

    bad2 = []
    n = 0
    for c in range(1, CMAX + 1):
        lo, hi = 6 * c - 1, 6 * c + 1
        for g in (lo, hi):
            if not isprime(g) or g < 5:
                continue
            n += 1
            a, b = 2 * u_of(g), g - 2 * u_of(g)
            if u_of(g) != c:
                bad2.append((g, "home column"))
            if a != 2 * c:
                bad2.append((g, "a != 2c"))
            if g == lo and b != 4 * c - 1:
                bad2.append((g, "b != 4c-1"))
            if g == hi and b != 4 * c + 1:
                bad2.append((g, "b != 4c+1"))
    lines.append("letters of the gears of columns 1..%d (%d gears): exceptions = %d"
                 % (CMAX, n, len(bad2)))
    if bad2:
        lines.append("  " + repr(bad2[:5]))
    lines.append("")
    lines.append(" column c   members        fibre {2c, 4c-1, 4c+1}    which letters")
    for c in range(1, 13):
        lo, hi = 6 * c - 1, 6 * c + 1
        tags = []
        tags.append("a of %s" % ("+".join(str(g) for g in (lo, hi) if isprime(g)) or "-"))
        tags.append("b of %d" % lo if isprime(lo) else "b of %d (composite)" % lo)
        tags.append("b of %d" % hi if isprime(hi) else "b of %d (composite)" % hi)
        lines.append(" %6d   (%4d,%4d)   %-22s  %s"
                     % (c, lo, hi, str([2 * c, 4 * c - 1, 4 * c + 1]), "; ".join(tags)))
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(OUT, "hc_fibre.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
