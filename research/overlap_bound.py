"""Bounding the longest killable run by counting overlaps exactly, not capacity.

The capacity bound of section 18d charges every gear the most it could kill in a window of
`L` twin slots, as if no two gears ever killed the same slot. It dies at `y = 13`, where
`sum 2/q` crosses 1. The fix is to stop charging separately for gears whose overlaps are
computable.

Split the gears into a *group* and a *remainder*. For the group, scan its pattern over its
full period once and take the true maximum number of kills in any window of length `L` -
this is exact and every internal overlap is already accounted for. For the remainder,
fall back on per-gear capacity. So

    f(L) = maxgroupkills(group, L) + sum over remainder of maxkills(q, L)

and `f(L) >= L` is necessary for a run of `L` consecutive slots to be killable. Hence

    any single L with f(L) < L certifies F_k <= L

which holds for each `L` on its own, with no monotonicity assumed - so searching for a
small witnessing `L` by a geometric ladder and then refining is sound.

`maxkills` is closed form: a window of `L` consecutive integers meets residue `r` mod `q`
in `ceil` of `(L - offset) / q` places, so the maximum over window positions is a scan over
`q` starts with constant work each.
"""

import sys
from math import prod
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slip_algebra import gears_upto, tooth


def maxkills(q, L):
    """Most slots gear `q` can kill in a window of `L` consecutive twin slots."""
    u = tooth(q)
    best = 0
    for s in range(q):
        total = 0
        for r in (u, q - u):
            off = (r - s) % q
            if off < L:
                total += (L - off + q - 1) // q
        best = max(best, total)
    return best


def killed_pattern(group):
    """Boolean array over one period of the group: True where some gear kills."""
    P = prod(group)
    a = np.zeros(P, dtype=bool)
    for q in group:
        u = tooth(q)
        for t in (u, q - u):
            a[t::q] = True
    return a


class GroupScanner:
    """Exact maximum kills by a group in any circular window of a given length."""

    def __init__(self, group):
        self.group = group
        pat = killed_pattern(group)
        self.period = pat.size
        doubled = np.concatenate([pat, pat]).astype(np.int32)
        self.cum = np.concatenate([[0], np.cumsum(doubled)]).astype(np.int64)

    def max_in_window(self, L):
        if L >= self.period:
            return int(self.cum[self.period])  # whole period, cannot exceed its total
        starts = np.arange(self.period)
        return int((self.cum[starts + L] - self.cum[starts]).max())


def choose_group(y, cap):
    """Largest prefix of the gear list whose period fits inside `cap`."""
    G = gears_upto(y)
    n = 0
    while n < len(G) and prod(G[: n + 1]) <= cap:
        n += 1
    return G[:n], G[n:]


def bound(y, cap=6_000_000, ladder_max=1 << 14):
    """Smallest `L` found with f(L) < L, certifying F_k <= L. None if none found."""
    group, rest = choose_group(y, cap)
    scan = GroupScanner(group)

    def f(L):
        return scan.max_in_window(L) + sum(maxkills(q, L) for q in rest)

    lo, L = 1, 2
    while L <= ladder_max:
        if f(L) < L:
            break
        lo = L
        L *= 2
    else:
        return None, group, scan.period
    for cand in range(lo + 1, L + 1):  # refine to the smallest crossing in the bracket
        if f(cand) < cand:
            return cand, group, scan.period
    return L, group, scan.period


if __name__ == "__main__":
    # F_k values established earlier; the tabulated halved-coordinate F is exactly 3 F_k
    FK = {5: 2, 7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34,
          29: 43, 31: 58, 37: 88, 41: 91, 43: 103}
    print("exact-group overlap bound on the longest killable run")
    print(f"  {'y':>4} {'true F_k':>9} {'bound':>7} {'ratio':>7} {'group':>34} {'period':>9}")
    for y in sorted(FK):
        b, group, P = bound(y)
        ratio = f"{b / FK[y]:.2f}" if b else "-"
        print(f"  {y:>4} {FK[y]:>9} {str(b):>7} {ratio:>7} {str(group):>34} {P:>9}")
