"""Shared model of the top machine, pair view and single-number view.

Pair view   : gear g strikes pair n iff n % g in {0, g-2}.
Triple view : n starts a twin candidate iff n % g not in {0, g-1, g-2} for all g.

Everything here is exact over a full wheel period.
"""

from math import prod

import numpy as np


def open_mask_pair(gears):
    """Boolean array of length W = prod(gears); True where the pair n is open."""
    W = prod(gears)
    m = np.ones(W, dtype=bool)
    idx = np.arange(W)
    for g in gears:
        r = idx % g
        m &= (r != 0) & (r != g - 2)
    return m


def open_mask_triple(gears):
    """Boolean array of length W; True where n starts a run of 3 open integers."""
    W = prod(gears)
    m = np.ones(W, dtype=bool)
    idx = np.arange(W)
    for g in gears:
        r = idx % g
        m &= (r != 0) & (r != g - 1) & (r != g - 2)
    return m


def walk_lengths(mask):
    """L(x) = min{j >= 0 : mask[x+j]} on the cycle Z_W. Exact, O(W)."""
    W = len(mask)
    L = np.zeros(W, dtype=np.int64)
    # two passes round the cycle are enough (mask is non-empty)
    cur = 0
    for _ in range(2):
        for x in range(W - 1, -1, -1):
            if mask[x]:
                cur = 0
            else:
                cur = cur + 1
            L[x] = cur
        cur = L[0]  # wrap: L[W-1] continues into L[0]
    return L


def walk_lengths_fast(mask):
    """Vectorised version of walk_lengths."""
    W = len(mask)
    idx = np.arange(W)
    openpos = idx[mask]
    # next open position at or after x, cyclically
    nxt = np.searchsorted(openpos, idx, side="left")
    wrapped = nxt >= len(openpos)
    nxt[wrapped] = 0
    L = openpos[nxt] - idx
    L[wrapped] += W
    return L


def gaps_of(mask):
    """Gaps between consecutive open positions, cyclically. Returns array of gaps."""
    openpos = np.flatnonzero(mask)
    W = len(mask)
    d = np.diff(openpos)
    return np.concatenate([d, [openpos[0] + W - openpos[-1]]])


def all_struck_counts(mask, jmax):
    """C(j) = #{x : x, x+1, ..., x+j-1 all struck}, cyclically, for j = 0..jmax."""
    W = len(mask)
    struck = ~mask
    C = [W]
    acc = np.ones(W, dtype=bool)
    for j in range(1, jmax + 1):
        acc &= np.roll(struck, -(j - 1))
        C.append(int(acc.sum()))
    return C
