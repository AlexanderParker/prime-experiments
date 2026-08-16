"""Navigating to the joint opening: exact step counts per sub-machine, and their increments.

For a start slot `k0` and a set `S` of gears, define

    N(S) = least J > 0 such that k0 + J is open for every gear in S

so `N(full set)` is the step count to the next twin. `N` is non-decreasing as gears are
added, and the whole question is whether the sequence of partial answers can be navigated
rather than searched.

Everything each gear contributes is known exactly and in closed form:

    position in its cycle      r_q = k0 mod q
    cycle length               q
    its two openings           the arcs either side of the teeth at +/- u_q
    step to its next bite      d_q = min( (u_q - r_q) mod q , (-u_q - r_q) mod q )
    slip against a machine     P mod q, where P is the machine's period

This module lays those out and reports, for each prefix of the gear list, the exact step
count and which gear forced the increment - the navigation path. What it is built to test is
whether the increments are predictable from the slip data, or whether each one is an
independent fact.
"""

import sys
from math import isqrt, prod
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from jump_distance import BulkGears
from twin_constructor import tooth

_GEARS = BulkGears()


def state(k0, gears):
    """The exact known state of every gear at `k0`: position, openings, bite distance."""
    rows = []
    for q in gears:
        u = tooth(q)
        r = k0 % q
        rows.append({
            "gear": q,
            "position": r,
            "teeth": (u, q - u),
            "bite": min((u - r) % q, (-u - r) % q),
            "arc": "short" if (r < u or r > q - u) else ("shield" if r == 0 else "long"),
        })
    return rows


def blocks(q, k0, J):
    """Does gear `q` block step `J` from `k0`?"""
    u = tooth(q)
    r = (k0 + J) % q
    return r == u or r == q - u


def step_count(k0, gears, cap=1 << 20):
    """`N(S)`: least J > 0 open for every gear in `gears`."""
    J = 1
    while J < cap:
        if not any(blocks(q, k0, J) for q in gears):
            return J
        J += 1
    return None


def navigation_path(k0, gears):
    """For each prefix of `gears`, the step count and the gear that forced the increase."""
    path = []
    prev = None
    for i in range(1, len(gears) + 1):
        S = gears[:i]
        n = step_count(k0, S)
        if prev is None or n > prev:
            # the newly added gear is what closed the old answer, but name the gear that
            # actually blocks the previous step so the path records a real reason
            blocker = None
            if prev is not None:
                blocker = next((q for q in S if blocks(q, k0, prev)), None)
            path.append({"added": S[-1], "N": n, "grew": prev is not None,
                         "closed_by": blocker, "period": prod(S)})
        prev = n
    return path


def full_answer(k0):
    """Step count to the next twin, with the gear bound it was certified against."""
    bound = isqrt(6 * (k0 + 4096) + 1)
    gears = list(_GEARS.upto(bound))
    return step_count(k0, gears), len(gears), bound


if __name__ == "__main__":
    k0 = 10**12 + 1
    gears = list(_GEARS.upto(200))

    print(f"state of the first gears at k0 = {k0}")
    print(f"  {'gear':>6} {'position':>9} {'teeth':>14} {'arc':>7} {'steps to bite':>14}")
    for row in state(k0, gears[:12]):
        print(f"  {row['gear']:>6} {row['position']:>9} {str(row['teeth']):>14} "
              f"{row['arc']:>7} {row['bite']:>14}")

    print(f"\nnavigation path: step count as gears are added, k0 = {k0}")
    print(f"  {'gear added':>11} {'N(S)':>7} {'grew?':>6} {'previous N closed by':>21} "
          f"{'machine period':>16}")
    for p in navigation_path(k0, gears):
        print(f"  {p['added']:>11} {p['N']:>7} {str(p['grew']):>6} "
              f"{str(p['closed_by']):>21} {p['period']:>16}")

    print("\ndoes the answer settle early? N for prefixes against the certified answer")
    ans, ngears, bound = full_answer(k0)
    print(f"  certified answer J* = {ans} using {ngears} gears (to {bound})")
    for z in (20, 50, 100, 200, 500, 1000, 10000, 100000, bound):
        S = list(_GEARS.upto(z))
        n = step_count(k0, S)
        print(f"  gears to {z:>9}: N = {n:>6}  {'MATCHES J*' if n == ans else ''}")
