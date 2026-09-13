"""The paint just above q: the field at the landing zone's start (owner, 2026-09-13).

Origin q (the machine's top gear). Columns: the offsets d >= 1 above q, the numbers q + d.
Rows: the gears 5 .. q. Row h is painted at d iff h divides q + d, i.e. d = -q (mod h): one
class per row (the origin q fixes every row's phase, as the square did at the square origin).
The walk's landing zone starts here: its columns are (12k - 1, 12k + 1) with 12k - 1 > q, and
the run L(q) the rule steps past is the run of those columns with a painted member.
Exact facts read off: a gear h paints an offset d only if q + d is a multiple of h coprime to 6
other than h itself, so q + d >= 5h, i.e. h <= (q + d)/5: at the zone start only gears up to
about q/5 (and, for the first 2q, only gears whose small multiples 5h, 7h, 11h, ... land there)
can paint; gears above 2q/5 paint nothing below 2q; the number of rows that can reach the
first N offsets is the number of gears h with 5h <= q + N.
Per machine: the rows painting the run at the zone start (the gears striking the members of
the L(q) painted columns), the largest such gear, how many rows can reach that far at all,
and the first landing.
Usage: uv run python zone_start.py q [q ...]
"""
import sys
from sympy import primerange, isprime, factorint


def run(q):
    gears = list(primerange(5, q + 1))
    k0 = (q + 1) // 12 + 1
    k = k0; passed = []
    while not (isprime(12 * k - 1) and isprime(12 * k + 1)):
        rows = sorted(set(h for n in (12 * k - 1, 12 * k + 1) if not isprime(n) for h in factorint(n) if h >= 5))
        passed.append((k, 12 * k - 1 - q, rows)); k += 1
    d_land = 12 * k - 1 - q
    reach = [h for h in gears if 5 * h <= q + d_land + 2]
    used = sorted(set(h for _, _, rows in passed for h in rows))
    print(f"machine {q}: zone starts at k0 = {k0} (offset {12*k0-1-q} above q); painted run L = {len(passed)} columns; landing at offset {d_land} ({12*k-1}, {12*k+1})")
    print(f"   rows painting the run: {used} (largest {max(used) if used else None}); rows able to reach offsets up to the landing (5h <= q + d): {len(reach)} of {len(gears)} gears, largest {reach[-1] if reach else None}")
    for kk, d, rows in passed[:12]:
        print(f"   k = {kk}, offset {d}: ({12*kk-1}, {12*kk+1}) painted by rows {rows}")
    if len(passed) > 12: print(f"   ... {len(passed) - 12} more")


if __name__ == "__main__":
    for a in sys.argv[1:]: run(int(a))
