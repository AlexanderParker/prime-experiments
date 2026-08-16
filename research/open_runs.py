"""Openings rather than blockings: the 6-cycle walk, wasted teeth, shields, misses.

Coordinates are chosen so each gear blocks at the *end* of its own cycle: gear `q` blocks
`n = 0 mod q` and its open run is `n = 1 .. q - 1`, length `q - 1`. Every gear then starts
from the same offset, so the walks are directly comparable.

The 6-cycle. Gears 2 and 3 leave only `n = 1` and `n = 5` mod 6. A twin is the "5" of one
block together with the "1" of the next, straddling the multiple of 6. Write `k` for the
twin slot, so the pair is `(6k - 1, 6k + 1)` and the straddled multiple is `6k`.

The walk. Successive multiples of `q` land in successive slots of the 6-cycle, stepping by
`q mod 6`. Every prime gear is `1` or `5` mod 6, so the step is `+1` or `-1`: each gear's
tooth walks the six slots one at a time, in one of two directions, returning to slot 0
every sixth multiple. That is true of composite sub-machines too, since the units mod 6
are closed under multiplication - so **every gear and every combination of gears walks the
6-cycle at speed +/-1, never faster and never slower**.

The tooth budget. Over an interval of length `6q` a gear places 6 teeth, and their slots
are all six residues, once each:

    1 tooth  on slot 0        - the straddled multiple of 6: a SHIELD, and while it sits
                                there the gear cannot touch either member
    2 teeth  on slots 1 and 5 - the only teeth that can KILL a twin
    3 teeth  on slots 2, 3, 4 - MISSES, spent on positions gears 2 and 3 already killed

So a gear wastes three of every six teeth on ground already taken, spends one protecting a
twin, and only two do any work against twins. That is the `q - 2` survivor count of every
`q` twin slots, derived from the walk instead of from a residue count.

Roles. For a twin slot `k` and a gear `q`, exactly one of three cases holds:

    killer  k = +/- 6^{-1} mod q     (q divides 6k - 1 or 6k + 1)   2 classes
    shield  k = 0 mod q              (q divides the midpoint 6k)    1 class
    miss    everything else                                        q - 3 classes

The shield law in its short form: **gear q shields twin slot k exactly when q divides k.**

Lockstep. Advancing `k` by one advances every gear's phase by exactly one. The machine is
a bank of odometer wheels of circumference `q`, all turning at the same rate, each carrying
two teeth at `+/- 6^{-1}` and a shield mark at `0`. A twin is a `k` at which no wheel shows
a tooth.
"""

import sys
from math import prod
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slip_algebra import gears_upto, teeth, tooth

KILLER, SHIELD, MISS = "killer", "shield", "miss"


def role(q, k):
    """Which of the three things gear `q` does to twin slot `k`."""
    r = k % q
    if r in teeth(q):
        return KILLER
    if r == 0:
        return SHIELD
    return MISS


def walk(q, steps=7):
    """Slot of the 6-cycle occupied by the m-th multiple of `q`, for m = 1 .. steps."""
    return [(m, m * q, (m * q) % 6) for m in range(1, steps + 1)]


def tooth_budget(q):
    """Count how the 6 teeth of `q` per interval of 6q are spent across the 6 slots."""
    slots = [(m * q) % 6 for m in range(1, 7)]
    return {
        "slots_visited": sorted(slots),
        "shield": sum(1 for s in slots if s == 0),
        "killers": sum(1 for s in slots if s in (1, 5)),
        "wasted": sum(1 for s in slots if s in (2, 3, 4)),
    }


def machine_step(gear_set):
    """Slip of a sub-machine against the 6-cycle: its period mod 6."""
    return prod(gear_set) % 6


def shield_law_failures(limit, kmax):
    """Check `role(q, k) == SHIELD` iff `q | k`, over gears and slots."""
    bad = []
    for q in gears_upto(limit):
        for k in range(1, kmax):
            if (role(q, k) == SHIELD) != (k % q == 0):
                bad.append((q, k))
    return bad


def role_counts(q):
    """How many of the `q` classes fall into each role."""
    counts = {KILLER: 0, SHIELD: 0, MISS: 0}
    for k in range(q):
        counts[role(q, k)] += 1
    return counts


def open_run_position(q, k):
    """Where the pair sits inside `q`'s open run: distance from each member to a tooth.

    In n-space `q` blocks multiples of `q`. The pair members are `6k - 1` and `6k + 1`.
    Returns the distance from the midpoint `6k` to the nearest multiple of `q`, signed,
    plus whether either member is itself a multiple.
    """
    mid = 6 * k
    below = mid - mid % q
    above = below + q
    d = min(mid - below, above - mid)
    return {
        "midpoint": mid,
        "nearest_tooth": below if mid - below <= above - mid else above,
        "distance": d,
        "shielded": mid % q == 0,
        "lower_killed": (mid - 1) % q == 0,
        "upper_killed": (mid + 1) % q == 0,
    }


def profile(k, y=None):
    """Role of every gear up to `y` (default the certifying bound) against slot `k`."""
    if y is None:
        y = int((6 * k + 1) ** 0.5)
    out = []
    for q in gears_upto(y):
        out.append((q, role(q, k)))
    return out


if __name__ == "__main__":
    print("the 6-cycle walk: successive multiples of q step by q mod 6")
    for q in (5, 7, 11, 13):
        w = walk(q)
        step = q % 6
        arrows = " ".join(f"{s}" for _, _, s in w)
        print(f"  q = {q:>2}  (q mod 6 = {step}, so step {'+1' if step == 1 else '-1'})"
              f"   multiples {[n for _, n, _ in w]}")
        print(f"          slots visited: {arrows}")

    print("\ntooth budget per interval of 6q: where the six teeth go")
    print(f"  {'q':>4} {'slots hit':>22} {'shield':>7} {'killers':>8} {'wasted':>7}")
    for q in gears_upto(30):
        b = tooth_budget(q)
        print(f"  {q:>4} {str(b['slots_visited']):>22} {b['shield']:>7} "
              f"{b['killers']:>8} {b['wasted']:>7}")

    print("\nslip against the 6-cycle is always +/-1, for single gears and combinations")
    sets = [(5,), (7,), (5, 7), (5, 11), (7, 11), (5, 7, 11), (5, 7, 11, 13),
            (11, 13, 17, 19, 23)]
    for S in sets:
        m = machine_step(S)
        print(f"  {str(S):>22}  period {prod(S):>8}  period mod 6 = {m}  "
              f"step {'+1' if m == 1 else '-1'}")
    others = {machine_step(S) for r in range(1, 5)
              for S in __import__("itertools").combinations(gears_upto(60), r)}
    print(f"  all sub-machines of up to 4 gears from 5..59: periods mod 6 = {sorted(others)}")

    print("\nrole counts per gear: 2 killer classes, 1 shield class, q - 3 misses")
    print(f"  {'q':>4} {'killer':>7} {'shield':>7} {'miss':>6} {'q-3':>5} {'ok':>4}")
    for q in gears_upto(30):
        c = role_counts(q)
        ok = c[KILLER] == 2 and c[SHIELD] == 1 and c[MISS] == q - 3
        print(f"  {q:>4} {c[KILLER]:>7} {c[SHIELD]:>7} {c[MISS]:>6} {q - 3:>5} {str(ok):>4}")

    bad = shield_law_failures(200, 500)
    print(f"\nshield law, q shields k iff q | k: {len(bad)} failures over gears<=200, k<500")

    print("\nrole profile of the first few twin slots, gears up to the certifying bound")
    for k in (1, 2, 3, 5, 7, 10, 12, 17, 18, 23, 25, 30):
        p = profile(k)
        y = int((6 * k + 1) ** 0.5)
        shields = [q for q, r in p if r == SHIELD]
        killers = [q for q, r in p if r == KILLER]
        print(f"  k = {k:>3}  pair ({6 * k - 1:>4},{6 * k + 1:>4})  gears<={y:>3}  "
              f"shields {str(shields):>12}  killers {str(killers):>10}  "
              f"{'TWIN' if not killers else 'dead'}")
