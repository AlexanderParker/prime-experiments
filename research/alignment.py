"""Alignment of exposed runs across gears, derived from rotation index and slip.

Frame. In `n`-space a gear blocks **once per rotation** - at every multiple of `q` - and is
exposed for the other `q - 1`. One block, one run. The twin-relevant structure follows from
which rotation the block is on: rotation `j` puts the block at `n = jq`, in slot `jq mod 6`
of the 6-cycle, and over six rotations the block visits all six slots once each. So

    j = +/-1 mod 6   ->  block lands on slot 1 or 5, the only ones that can end a twin
    j = 0 mod 6      ->  block lands on the midpoint 6m: a shield
    otherwise        ->  block lands on slot 2, 3 or 4, ground gears 2 and 3 already took

Four rotations in every six are harmless. The threatening rotations are `j = 1, 5, 7, 11,
13, ...`, spaced alternately 4 and 2, so in `n` the threats are spaced `4q` then `2q`, and
the exposed runs between them, measured in twin slots, are the long and short runs of
section 24 - `4q/6` and `2q/6` to within one. That is where "two per `q` twin slots" comes
from: not a second block, but the two threatening rotations.

Alignment. In twin-slot space gear `q` therefore threatens `m = +/- 6^{-1} mod q`, periodic
with period `q`, and gear `q`'s threat pattern advances against gear `q'`'s frame by
`q mod q'` per turn - the machine slip. This module computes, exactly and without walking a
target range, how the exposed runs of a gear set are apportioned: how many runs there are,
their lengths, and how coincident threats (two gears spending a threat on the same slot)
merge runs together.
"""

import itertools
import sys
from math import prod
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slip_algebra import gears_upto, tooth


def threatening_rotations(q, count=6):
    """The rotations of gear `q` whose block can end a twin, with the slot each threatens."""
    out = []
    j = 1
    while len(out) < count:
        n = j * q
        s = n % 6
        if s in (1, 5):
            m = (n + 1) // 6 if s == 5 else (n - 1) // 6
            out.append((j, n, s, m))
        j += 1
    return out


def threat_mask(gear_set):
    """Boolean array over the period: True where some gear in the set threatens."""
    P = prod(gear_set)
    shut = np.zeros(P, dtype=bool)
    for q in gear_set:
        u = tooth(q)
        for t in (u, q - u):
            shut[t::P if q > P else q] = True
    return shut


def run_structure(gear_set):
    """Exposed runs of the union: how many, how long, and the exact totals."""
    P = prod(gear_set)
    shut = np.zeros(P, dtype=bool)
    threats = np.zeros(P, dtype=np.int16)
    for q in gear_set:
        u = tooth(q)
        for t in (u, q - u):
            shut[t::q] = True
            threats[t::q] += 1
    exposed = ~shut
    # maximal runs of exposed slots, treating the period as a circle
    idx = np.flatnonzero(exposed)
    runs = []
    if idx.size:
        breaks = np.flatnonzero(np.diff(idx) > 1)
        starts = np.concatenate([[0], breaks + 1])
        ends = np.concatenate([breaks, [idx.size - 1]])
        runs = [int(idx[e] - idx[s] + 1) for s, e in zip(starts, ends)]
        if exposed[0] and exposed[-1] and len(runs) > 1:
            runs[0] += runs.pop()  # wrap the first and last run together
    coincident = int((threats > 1).sum())
    return {
        "gears": tuple(int(q) for q in gear_set),
        "period": P,
        "exposed": int(exposed.sum()),
        "exposed_expected": prod(q - 2 for q in gear_set),
        "threat_slots": int(shut.sum()),
        "threat_events": int(threats.sum()),
        "coincident_slots": coincident,
        "runs": len(runs),
        "longest": max(runs) if runs else 0,
        "shortest": min(runs) if runs else 0,
        "mean": sum(runs) / len(runs) if runs else 0,
    }


if __name__ == "__main__":
    print("one block per rotation: which rotations of a gear can end a twin")
    for q in (5, 7, 11):
        rows = threatening_rotations(q)
        print(f"  gear {q:>2}: threatening rotations j = "
              f"{[j for j, *_ in rows]}, threatening twin slots {[m for *_, m in rows]}")

    print("\nexposed runs between threats match the long and short runs of section 24")
    print(f"  {'gear':>5} {'threatened slots':>32} {'run lengths':>13} {'4q/6, 2q/6':>12}")
    for q in gears_upto(30):
        ms = [m for *_, m in threatening_rotations(q, 7)]
        runs = sorted({ms[i + 1] - ms[i] - 1 for i in range(len(ms) - 1)}, reverse=True)
        print(f"  {q:>5} {str(ms):>32} {str(runs):>13} {str([4 * q // 6, 2 * q // 6]):>12}")

    print("\nrun structure of the union, exact over the period")
    print(f"  {'gears':>22} {'period':>8} {'exposed':>8} {'prod(q-2)':>10} {'ok':>5} "
          f"{'runs':>6} {'longest':>8} {'shortest':>9} {'mean':>7} {'coinc':>6}")
    for r in range(1, 5):
        for S in itertools.combinations(gears_upto(19), r):
            a = run_structure(list(S))
            ok = a["exposed"] == a["exposed_expected"]
            print(f"  {str(a['gears']):>22} {a['period']:>8} {a['exposed']:>8} "
                  f"{a['exposed_expected']:>10} {str(ok):>5} {a['runs']:>6} "
                  f"{a['longest']:>8} {a['shortest']:>9} {a['mean']:>7.2f} "
                  f"{a['coincident_slots']:>6}")
