"""Phase: the coordinate vector of a slot, and what it determines about the step count.

The generating polynomial of section 32 depends only on the multiset of gears, so it is blind
to where the threats sit. Everything it discards is phase. The phase of slot `m` is the vector

    ph(m) = (m mod q)  over the gears q

and as `m` advances by one every coordinate advances by one, so the phase walks the diagonal of
the product of cyclic groups. By CRT that diagonal covers the whole product exactly once per
period, so the phase carries complete information - the entire difficulty of localisation lives
in it.

Two facts this module establishes.

1. **The small-gear phase gives an exact lower bound on the step count.** For a sub-machine `S`
   with period `P_S`, the offsets `J` that `S` leaves exposed depend only on `m mod P_S`. The
   least such `J > 0` is a lower bound on the true step count, since the remaining gears can
   only remove candidates, never add them. It is a table lookup: no stepping, no large gears.

2. **The bound is attained exactly when no larger gear threatens that candidate**, and when it
   is not attained the responsible gear is identifiable. So the step count decomposes as
   "phase-determined floor, plus interference", and the interference is what section 23 showed
   cannot be predicted from bounded depth.
"""

import sys
from bisect import bisect_left
from math import isqrt, prod
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from jump_distance import BulkGears
from twin_constructor import tooth

_GEARS = BulkGears()


class PhaseTable:
    """Exposed offsets of a sub-machine, indexed by the machine's phase."""

    MAX_PERIOD = 20_000_000  # the exposed set is enumerated, so the period must be walkable

    def __init__(self, gears):
        self.gears = [int(q) for q in gears]
        self.period = prod(self.gears)
        if self.period > self.MAX_PERIOD:
            raise ValueError(
                f"period {self.period} exceeds {self.MAX_PERIOD}; use fewer gears")
        self.exposed = sorted(
            m for m in range(self.period)
            if all(m % q not in (tooth(q), q - tooth(q)) for q in self.gears)
        )
        self.exposed_set = set(self.exposed)

    def phase(self, m):
        """The sub-machine's phase at slot m - a single residue by CRT."""
        return m % self.period

    def floor_offset(self, m):
        """Least J > 0 the sub-machine leaves exposed at m + J. Pure lookup."""
        r = (m + 1) % self.period
        i = bisect_left(self.exposed, r)
        if i == len(self.exposed):
            return self.period - (m % self.period) + self.exposed[0]
        return self.exposed[i] - r + 1

    def offsets(self, m, count):
        """The first `count` offsets the sub-machine leaves exposed."""
        out = []
        r = (m + 1) % self.period
        i = bisect_left(self.exposed, r)
        base = 0
        while len(out) < count:
            if i == len(self.exposed):
                i = 0
                base += self.period
            out.append(base + self.exposed[i] - r + 1)
            i += 1
        return out


def true_step(m, bound_extra=4096):
    """Exact step count to the next twin slot after m."""
    top = m + bound_extra
    bound = isqrt(6 * top + 1)
    gears = [int(q) for q in _GEARS.upto(bound)]
    J = 1
    while J <= bound_extra:
        k = m + J
        if all(k % q not in (tooth(q), q - tooth(q)) for q in gears):
            return J
        J += 1
    return None


def interference(m, table, limit=64):
    """Which candidate the phase floor proposes, and which gear kills each rejected one."""
    bound = isqrt(6 * (m + 4096) + 1)
    gears = [int(q) for q in _GEARS.upto(bound)]
    small = set(table.gears)
    rows = []
    for J in table.offsets(m, limit):
        k = m + J
        killer = next((q for q in gears
                       if q not in small and k % q in (tooth(q), q - tooth(q))), None)
        rows.append((J, killer))
        if killer is None:
            break
    return rows


if __name__ == "__main__":
    small = [5, 7, 11, 13]
    table = PhaseTable(small)
    print(f"sub-machine {small}, period {table.period}, "
          f"{len(table.exposed)} exposed phases")

    print("\nphase floor against the true step count")
    print(f"  {'m':>18} {'phase mod P':>12} {'floor from phase':>17} {'true step':>10} "
          f"{'attained':>9}")
    for m in (10**6 + 1, 10**7 + 3, 10**9 + 1, 10**10 + 7, 10**12 + 1, 10**12 + 7):
        f = table.floor_offset(m)
        t = true_step(m)
        print(f"  {m:>18} {table.phase(m):>12} {f:>17} {t:>10} {str(f == t):>9}")

    print("\ninterference: the candidates the phase proposes, and what kills them")
    for m in (10**12 + 1, 10**10 + 7):
        print(f"  m = {m}")
        for J, killer in interference(m, table):
            tag = "TWIN" if killer is None else f"killed by gear {killer}"
            print(f"     offset {J:>5}  {tag}")

    print("\nfloor quality as the sub-machine grows, at m = 10^12 + 1")
    m = 10**12 + 1
    truth = true_step(m)
    print(f"  {'sub-machine':>34} {'period':>10} {'floor':>7} {'true':>6}")
    for upto in (7, 13, 19, 29, 43):
        gs = [int(q) for q in _GEARS.upto(upto)]
        t = PhaseTable(gs)
        print(f"  {str(tuple(gs)):>34} {t.period:>10} {t.floor_offset(m):>7} {truth:>6}")
