"""Exposure: each gear as a schedule of open intervals, and where they all overlap.

The blocking view asks which gear kills a slot. The exposure view asks the inverse: each
gear is open almost all the time, in intervals, and a twin is a point covered by an open
interval of *every* gear at once. Nothing here counts kills.

A gear `q` has teeth at `+/- u_q` and therefore two open arcs, alternating with period `q`:

    long arc   length  q - 2 u_q - 1
    short arc  length  2 u_q - 1        (centred on the shield at k = 0 mod q)

From a slot `k`, gear `q` is currently inside one of those arcs and will remain exposed for

    d_q = min( (u_q - k) mod q , (-u_q - k) mod q )

more steps. So the gear's forward schedule is fully known in closed form: exposed for `d_q`,
one step shut, exposed for the far arc, one step shut, and so on forever.

Two consequences the exposure view makes obvious, and the blocking view hides:

* **Most gears impose no constraint at all.** A gear only matters over the next `W` steps if
  its current exposure *ends* inside the window, that is `d_q <= W`. Every other gear is
  exposed across the whole window and can be ignored entirely.
* **Constraint is concentrated.** The gears whose exposure ends inside a window of `W` steps
  number about `2 W log log y`, a few hundred, against the full gear count.

The sweep. Because each gear's shut phase is a single step, one gear alone never advances the
candidate by more than one. Combined sub-machines are different: a group of gears has long
shut runs, so jumping to a *group's* next joint exposure can move far. The sweep therefore
works on groups: ask each group where its next joint exposure is at or after the candidate,
take the furthest answer, and repeat until every group agrees. Agreement is a twin.
"""

import sys
from bisect import bisect_left
from math import isqrt, prod
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from jump_distance import BulkGears
from twin_constructor import tooth

_GEARS = BulkGears()

LONG, SHORT, SHIELD = "long arc", "short arc", "shield"


def exposure(q, k):
    """Gear `q`'s current opening at slot `k`: which arc, how much room is left."""
    u = tooth(q)
    r = k % q
    if r == u or r == q - u:
        return {"gear": q, "arc": None, "room": 0, "shut": True}
    arc = SHIELD if r == 0 else (SHORT if (r < u or r > q - u) else LONG)
    return {"gear": q, "arc": arc, "room": min((u - r) % q, (-u - r) % q), "shut": False}


def schedule(q, k, steps):
    """Forward exposure schedule of gear `q`: alternating open runs and single shut steps."""
    out = []
    pos = k
    while pos - k < steps:
        e = exposure(q, pos)
        if e["shut"]:
            out.append(("shut", pos, 1))
            pos += 1
        else:
            out.append((e["arc"], pos, e["room"]))
            pos += e["room"] if e["room"] > 0 else 1
    return out


def constrainers(k, gears, window):
    """Gears whose current exposure ENDS inside the window - the only ones that matter."""
    out = []
    for q in gears:
        e = exposure(q, k)
        if e["room"] <= window:
            out.append((int(q), e["room"], e["arc"]))
    return out


class Group:
    """A sub-machine's joint exposure, as a lookup table over its own period."""

    def __init__(self, gear_set):
        self.gears = [int(q) for q in gear_set]
        self.period = prod(self.gears)
        alive = np.ones(self.period, dtype=bool)
        for q in self.gears:
            u = tooth(q)
            for t in (u, q - u):
                alive[t::q] = False
        self.open = np.flatnonzero(alive)
        self.openset = alive

    def next_exposure(self, k):
        """Least slot >= k at which every gear of this group is exposed."""
        base, r = divmod(k, self.period)
        i = bisect_left(self.open, r)
        if i == len(self.open):
            return (base + 1) * self.period + int(self.open[0])
        return base * self.period + int(self.open[i])


def sweep(k0, groups, singles):
    """Advance to the first slot every group and single gear exposes. Returns (slot, rounds)."""
    k = k0 + 1
    rounds = 0
    while True:
        rounds += 1
        moved = False
        for g in groups:
            nxt = g.next_exposure(k)
            if nxt > k:
                k = nxt
                moved = True
        for q in singles:
            if exposure(q, k)["shut"]:
                k += 1
                moved = True
        if not moved:
            return k, rounds


if __name__ == "__main__":
    k0 = 10**12 + 1
    bound = isqrt(6 * (k0 + 4096) + 1)
    gears = [int(q) for q in _GEARS.upto(bound)]
    print(f"k0 = {k0}, gears to {bound} ({len(gears)})")

    print("\nexposure of the first gears: which arc, and how much room before it shuts")
    print(f"  {'gear':>6} {'position':>9} {'arc':>10} {'room left':>10}")
    for q in gears[:12]:
        e = exposure(q, k0)
        print(f"  {q:>6} {k0 % q:>9} {str(e['arc']):>10} {e['room']:>10}")

    print("\nhow many gears actually constrain the next W steps (exposure ends inside)")
    print(f"  {'W':>6} {'constrainers':>13} {'exposed throughout':>19}")
    for W in (16, 64, 128, 512):
        c = constrainers(k0, gears, W)
        print(f"  {W:>6} {len(c):>13} {len(gears) - len(c):>19}")

    print("\nforward schedule of one gear, as alternating exposure and shut")
    for q in (5, 7, 4093):
        print(f"  gear {q}: {schedule(q, k0, 30)[:6]}")

    print("\nsweep on groups: jump to each group's next joint exposure, take the furthest")
    groups, singles, cur = [], [], []
    for q in gears:
        cur.append(q)
        if prod(cur) > 2_000_000:
            cur.pop()
            groups.append(Group(cur))
            cur = [q]
        if len(groups) >= 6:
            break
    singles = [q for q in gears if all(q not in g.gears for g in groups)]
    print(f"  {len(groups)} groups (periods "
          f"{[g.period for g in groups]}), {len(singles)} single gears")
    slot, rounds = sweep(k0, groups, singles)
    print(f"  swept to slot {slot} in {rounds} rounds; step count J = {slot - k0}")
