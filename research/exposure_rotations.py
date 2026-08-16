"""Exposure measured in rotations: one block per rotation, threats only on j = +/-1 mod 6.

Frame, corrected. A gear blocks once per rotation - at `n = jq` for rotation `j` - and is
exposed for the rest of that rotation. The block lands in slot `jq mod 6` of the 6-cycle, and
over six rotations it visits all six slots once. So a gear's relationship to twins is decided
entirely by its rotation index:

    j = +/-1 mod 6   threatens - the block is on slot 1 or 5, a twin member
    j = 0 mod 6      shields   - the block is on the midpoint 6m, so neither member can be hit
    otherwise        harmless  - the block is on slot 2, 3 or 4, already dead ground

Four rotations in every six are harmless. The threatening rotations are `1, 5, 7, 11, 13, ...`
- spaced alternately 4 and 2 - and this alone produces the two exposure lengths, since a gap
of 4 rotations is `4q` numbers and a gap of 2 rotations is `2q`.

Everything needed is then rotation arithmetic, with no teeth and no modular inverse:

    current rotation of gear q at twin slot m   j0 = ceil(6m / q)
    next threatening rotation                  next j >= j0 with j = +/-1 mod 6
    twin slot it threatens                     (j q +/- 1) / 6, sign from j q mod 6
    exposure remaining, in twin slots           that slot minus m

This module builds the exposure schedule that way and checks it against the tooth form of
section 24, then generates twins from it.
"""

import sys
from math import isqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from jump_distance import BulkGears
from twin_constructor import tooth

_GEARS = BulkGears()

THREAT, SHIELD, HARMLESS = "threat", "shield", "harmless"


def rotation_role(q, j):
    """What rotation `j` of gear `q` does to the 6-cycle."""
    s = (j * q) % 6
    if s in (1, 5):
        return THREAT
    if s == 0:
        return SHIELD
    return HARMLESS


def threatened_slot(q, j):
    """The twin slot rotation `j` of gear `q` threatens, or None if it does not threaten."""
    n = j * q
    s = n % 6
    if s == 5:
        return (n + 1) // 6
    if s == 1:
        return (n - 1) // 6
    return None


def next_threat(q, m):
    """First twin slot at or after `m` that gear `q` threatens, by rotation arithmetic."""
    # the earliest rotation that can threaten slot `m` is the one landing on 6m - 1, so the
    # floor is (6m - 1) // q, not ceil(6m / q) - the latter skips a threat on m itself
    j0 = max(1, (6 * m - 1) // q)
    for j in range(j0, j0 + 13):         # a threatening rotation occurs within 6 steps
        slot = threatened_slot(q, j)
        if slot is not None and slot >= m:
            return slot, j
    raise RuntimeError("no threatening rotation found")


def exposure_in_slots(q, m):
    """How many twin slots gear `q` stays exposed for, counting from `m`."""
    slot, j = next_threat(q, m)
    return slot - m, j


def schedule_rotations(q, m, count=6):
    """The next few threatening rotations of gear `q` from slot `m`, with their slots."""
    out = []
    _, j = next_threat(q, m)
    while len(out) < count:
        slot = threatened_slot(q, j)
        if slot is not None:
            out.append((j, j * q, (j * q) % 6, slot))
        j += 1
    return out


def twins_by_rotation(m0, span):
    """Twin slots in `[m0, m0 + span)`, driven by rotation arithmetic alone."""
    bound = isqrt(6 * (m0 + span) + 1)
    exposed = bytearray(b"\x01") * span
    for q in _GEARS.upto(bound):
        q = int(q)
        slot, j = next_threat(q, m0)
        while slot < m0 + span:
            exposed[slot - m0] = 0
            j += 1
            while threatened_slot(q, j) is None:  # skip shield and harmless rotations
                j += 1
            slot = threatened_slot(q, j)
    return [m0 + i for i, e in enumerate(exposed) if e]


if __name__ == "__main__":
    print("rotation roles: six rotations of a gear cover the six slots once each")
    for q in (5, 7, 11):
        roles = [(j, (j * q) % 6, rotation_role(q, j)) for j in range(1, 7)]
        print(f"  gear {q:>2}: " + "  ".join(f"j={j} slot={s} {r}" for j, s, r in roles))

    m0 = 10**12 + 1
    print(f"\nexposure remaining at slot m = {m0}, by rotation arithmetic")
    print(f"  {'gear':>7} {'rotation j0':>12} {'next threat j':>14} {'threatens slot':>15} "
          f"{'exposure left':>14} {'tooth form':>11} {'agree':>6}")
    for q in [int(x) for x in _GEARS.upto(60)]:
        left, j = exposure_in_slots(q, m0)
        u = tooth(q)
        r = m0 % q
        tooth_left = min((u - r) % q, (-u - r) % q)
        print(f"  {q:>7} {-(-6 * m0 // q):>12} {j:>14} {m0 + left:>15} {left:>14} "
              f"{tooth_left:>11} {str(left == tooth_left):>6}")

    print("\nthreat spacing is 4 then 2 rotations, giving the two exposure lengths")
    for q in (5, 7, 29):
        rows = schedule_rotations(q, 1, 6)
        js = [j for j, *_ in rows]
        slots = [s for *_, s in rows]
        print(f"  gear {q:>2}: rotations {js}, slots {slots}, "
              f"slot gaps {[slots[i + 1] - slots[i] for i in range(len(slots) - 1)]}")

    print("\ntwins generated by rotation arithmetic only")
    for m, span in ((10**6 + 1, 20000), (10**9 + 1, 20000), (10**12 + 1, 20000)):
        got = twins_by_rotation(m, span)
        print(f"  from m = {m:>16}, span {span}: {len(got):>4} twins, first "
              f"{got[:3]}")
