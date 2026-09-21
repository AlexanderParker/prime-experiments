"""Is the lap route the whole range statement for machine 5, or only part of it?

Machine 5 = base 2, 3 with gear 5. Gear 5 strikes column n iff 5 | 6n -+ 1, i.e. n = 1 or 4 mod 5.
So machine 5 leaves THREE open classes per period: n = 0, 2, 3 mod 5. The laps of nodes c..c.xv
follow n = 0 mod 5 only (the copies 30k -+ 1 of the known opening). The other two classes carry
machine 5 round its period just as well.

PRE-REGISTERED: (1) the openings of machine q are exactly the unstruck columns of the three
classes together, so the honest range statement for machine 5 as the cycle is about all three, and
the lap statement is a restriction to one third of them - strictly harder; (2) the record of the
lap machine against its window is therefore much tighter than the record of the full machine q
against the same window: compute both for q = 7..31 and compare.
"""
from sympy import primerange, nextprime


def record_columns(q):
    """Longest run of struck columns for the machine 5..q, over one period (run convention)."""
    gears = [g for g in primerange(5, q + 1)]
    P = 1
    for g in gears:
        P *= g
    cls = {g: (pow(6, -1, g), (-pow(6, -1, g)) % g) for g in gears}
    best = run = 0
    for n in range(P):
        if any(n % g in cls[g] for g in gears):
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best, P


def record_laps(q):
    """Longest run of struck laps for the lap machine 7..q, over one lap period."""
    gears = [g for g in primerange(7, q + 1)]
    P = 1
    for g in gears:
        P *= g
    cls = {g: (pow(30, -1, g), (-pow(30, -1, g)) % g) for g in gears}
    best = run = 0
    for k in range(P):
        if any(k % g in cls[g] for g in gears):
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best, P


print("open classes of machine 5 (columns n mod 5 not struck by gear 5):",
      [n for n in range(5) if n % 5 not in (pow(6, -1, 5), (-pow(6, -1, 5)) % 5)])
print()
print(" q | full machine 5..q          | lap machine 7..q (n = 0 mod 5 only)")
print("   | record  window   ratio     | record  window   ratio")
for q in [7, 11, 13, 17, 19, 23]:
    qn = nextprime(q)
    Fc, _ = record_columns(q)
    wc = qn * qn / 6
    Fl, _ = record_laps(q)
    wl = qn * qn / 30
    print(f"{q:3d}| {Fc:5d}  {wc:7.1f}  {Fc/wc:6.2f}     | {Fl:5d}  {wl:7.1f}  {Fl/wl:6.2f}")
# larger q from the recorded exact values (no scan)
KNOWN_F = {29: 42, 31: 57}
KNOWN_L = {29: 25, 31: 31}
for q in [29, 31]:
    qn = nextprime(q)
    wc = qn * qn / 6
    wl = qn * qn / 30
    print(f"{q:3d}| {KNOWN_F[q]:5d}  {wc:7.1f}  {KNOWN_F[q]/wc:6.2f}     | {KNOWN_L[q]:5d}  {wl:7.1f}  {KNOWN_L[q]/wl:6.2f}   (recorded values)")
