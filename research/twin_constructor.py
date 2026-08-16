"""Construct the next twin prime from the openings of the gears.

Nothing here tests a number for primality, and nothing sieves an interval. The only inputs
are each gear's period, its two open arcs, and the rate at which the gears rotate against
one another. The gear list is grown by the machine's own rule, so no external prime source
is used either.

The three things the machine knows about a gear `q` and a twin slot `k`, from section 18:

    shield     k = 0 mod q                 q divides the midpoint 6k, so q cannot reach
                                           either member. Centre of the short arc.
    short arc  k in (q - u, u) mod q       the tight opening, of length 2u - 1, wrapped
                                           around the shield
    long arc   k in (u, q - u) mod q       the wide opening, of length q - 2u - 1

with `u = u_q = 6^{-1} mod q` in its small representative, `(q + 1)/6` or `(q - 1)/6`. A
twin slot is a `k` sitting in an opening of *every* gear up to the certifying bound
`sqrt(6k + 1)`; the two members are then `6k - 1` and `6k + 1`, both prime, with no
primality test performed.

How the construction advances. A sub-machine `S` of the smallest gears is combined once,
by the turn law: starting from the single open class `k = 0` of the empty machine, each new
gear `q` keeps `q - 2` of every `q` turns, and the survivors are computed from the closed
form for the struck turns. That yields the open residues modulo `P = prod S` - the only
phases at which the small gears expose slots 1 and 5 across a block boundary. The
constructor then steps from one such phase to the next, which is where the slip ratios
enter: consecutive open residues are spaced by the gaps of the combined pattern, so the
walk skips every phase the small gears have already closed. Each candidate is then put to
the remaining gears one at a time, and the first candidate that every gear leaves open is
the next twin.
"""

import sys
from bisect import bisect_left
from math import isqrt, prod
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SHIELD, SHORT, LONG = "shield", "short arc", "long arc"


def tooth(q):
    """`u_q = 6^{-1} mod q`, small representative. Closed form, no inversion needed."""
    return (q + 1) // 6 if q % 6 == 5 else (q - 1) // 6


class Gears:
    """The gear list, grown by the machine's own rule: no external prime source.

    A candidate is a gear when it is `+/-1 mod 6` and no smaller gear divides it. The
    divisors needed to settle a candidate `c` are the gears up to `sqrt(c)`, all of which
    are already in the list because they are smaller than `c`.
    """

    def __init__(self):
        self.list = [5, 7]

    def upto(self, limit):
        while self.list[-1] < limit:
            self._extend()
        i = bisect_left(self.list, limit + 1)
        return self.list[:i]

    def _extend(self):
        c = self.list[-1]
        while True:
            c += 2 if c % 6 == 5 else 4  # walk the 1 and 5 slots of the 6-cycle
            r = isqrt(c)
            if all(c % g for g in self.list if g <= r):
                self.list.append(c)
                return


def open_residues(gear_set):
    """Open classes of a sub-machine, built from surviving turns rather than strikes.

    Adding gear `q` to a machine of period `P`, an open class `k0` spawns `q` turns
    `k0 + t P`, of which `q` strikes exactly the two given by the closed form; the
    remaining `q - 2` survive.
    """
    reps = [0]
    P = 1
    for q in gear_set:
        u = tooth(q)
        inv = pow(P % q, -1, q) if P > 1 else 1
        nxt = []
        for k0 in reps:
            struck = {((u - k0) * inv) % q, ((-u - k0) * inv) % q}
            nxt.extend(k0 + t * P for t in range(q) if t not in struck)
        reps = nxt
        P *= q
    return sorted(reps), P


def arc_of(q, k):
    """Which opening of gear `q` slot `k` occupies, or None when `q` blocks it."""
    u = tooth(q)
    r = k % q
    if r == u or r == q - u:
        return None
    if r == 0:
        return SHIELD
    return SHORT if (r < u or r > q - u) else LONG


class TwinConstructor:
    """Next twin slot, driven by the openings of the gears."""

    def __init__(self, small=(5, 7, 11, 13)):
        self.gears = Gears()
        self.small = list(small)
        self.open_res, self.period = open_residues(self.small)
        self._cached_bound = None
        self._cached_needed = None

    def _needed(self, bound):
        """Gears to consult at this certifying bound, cached since it moves slowly."""
        if bound != self._cached_bound:
            small = set(self.small)
            self._cached_needed = [q for q in self.gears.upto(bound) if q not in small]
            self._cached_bound = bound
        return self._cached_needed

    def _phases_from(self, k):
        """Walk the open phases of the small machine, starting at or after `k`."""
        base, r = divmod(k, self.period)
        i = bisect_left(self.open_res, r)
        while True:
            if i == len(self.open_res):
                base += 1
                i = 0
            yield base * self.period + self.open_res[i]
            i += 1

    def next_twin(self, after):
        """The first twin slot strictly greater than `after`, with its certificate.

        The phase walk applies every gear of the sub-machine, which is only legitimate
        once all of them fall at or below the certifying bound - that is `6k + 1 >=
        max(small)^2`. Below that the small gears would self-block slots they have no
        authority over, so those `k` are scanned directly instead.
        """
        floor = (max(self.small) ** 2 - 1) // 6 + 1
        if after + 1 < floor:
            for k in range(after + 1, floor):
                if k == 0:
                    continue
                bound = isqrt(6 * k + 1)
                arcs = {}
                for q in self.gears.upto(bound):
                    a = arc_of(q, k)
                    if a is None:
                        break
                    arcs[q] = a
                else:
                    return k, arcs
            after = floor - 1
        for k in self._phases_from(after + 1):
            if k <= after or k == 0:
                continue
            bound = isqrt(6 * k + 1)
            needed = self._needed(bound)
            arcs = {}
            for q in needed:
                a = arc_of(q, k)
                if a is None:
                    break
                arcs[q] = a
            else:
                for q in self.small:
                    if q <= bound:
                        a = arc_of(q, k)
                        if a is None:
                            break
                        arcs[q] = a
                else:
                    return k, arcs
        raise RuntimeError("unreachable")

    def certificate(self, k, arcs):
        """Readable account of why every gear leaves `k` open."""
        rows = []
        for q in sorted(arcs):
            u = tooth(q)
            r = k % q
            room = min((r - u) % q, (q - u - r) % q)
            rows.append((q, r, u, arcs[q], room))
        return rows


if __name__ == "__main__":
    tc = TwinConstructor()
    print(f"small sub-machine {tc.small}, period {tc.period}, "
          f"{len(tc.open_res)} open phases per period "
          f"({len(tc.open_res)}/{tc.period} = one in "
          f"{tc.period / len(tc.open_res):.2f})")

    print("\nconsecutive twins constructed from openings only")
    k = 0
    print(f"  {'twin slot k':>12} {'pair':>26} {'gears checked':>14} {'shields':>18}")
    for _ in range(12):
        k, arcs = tc.next_twin(k)
        shields = [q for q, a in arcs.items() if a == SHIELD]
        print(f"  {k:>12} {f'({6 * k - 1}, {6 * k + 1})':>26} {len(arcs):>14} "
              f"{str(shields):>18}")

    print("\ncertificate for one twin: every gear's opening, and the room it leaves")
    k, arcs = tc.next_twin(100000)
    print(f"  twin slot {k}, pair ({6 * k - 1}, {6 * k + 1})")
    print(f"  {'gear q':>7} {'k mod q':>8} {'u_q':>5} {'opening':>10} "
          f"{'steps to nearest tooth':>23}")
    for q, r, u, a, room in tc.certificate(k, arcs):
        print(f"  {q:>7} {r:>8} {u:>5} {a:>10} {room:>23}")

    print("\njumping straight to a large slot")
    for start in (10**6, 10**7, 10**8):
        k, arcs = tc.next_twin(start)
        shields = [q for q, a in arcs.items() if a == SHIELD]
        print(f"  after k = {start:>10}: next twin slot {k:>10}  "
              f"pair ({6 * k - 1}, {6 * k + 1})  gears {len(arcs)}  shields {shields}")
