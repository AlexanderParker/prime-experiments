"""Shared model of the top machine with SMALL gears allowed (g = 2, 3, 5, ...).

Gear g strikes the pair n iff n = 0 or n = -2 (mod g).  For g = 2 the two teeth
coincide (one tooth); for g = 3 they are adjacent (a solid domino); for g >= 5 they
are a gapped distance-2 domino.  Nothing here assumes g >= 7.

Everything is exact over a full wheel period or over the stated range.
"""

from math import prod

import numpy as np


def teeth(g):
    """The distinct tooth residues of gear g."""
    return sorted({0 % g, (-2) % g})


def slots(g):
    """The open residues of gear g."""
    t = set(teeth(g))
    return [r for r in range(g) if r not in t]


def open_mask(gears):
    """Boolean array of length W = prod(gears); True where the pair n is open."""
    W = prod(gears)
    m = np.ones(W, dtype=bool)
    idx = np.arange(W)
    for g in gears:
        r = idx % g
        for t in teeth(g):
            m &= r != t
    return m


def open_mask_triple(gears):
    """True where n starts a run of three consecutive open integers (twin candidate)."""
    W = prod(gears)
    m = np.ones(W, dtype=bool)
    idx = np.arange(W)
    for g in gears:
        r = idx % g
        for t in (0 % g, (-1) % g, (-2) % g):
            m &= r != t
    return m


def col_teeth(g):
    """The tooth residues of gear g in the BOTTOM's column coordinate: +- 6^{-1} mod g.

    Defined only when gcd(6, g) = 1.
    """
    u = pow(6, -1, g)
    return sorted({u % g, (-u) % g})


def col_open_mask(gears):
    W = prod(gears)
    m = np.ones(W, dtype=bool)
    idx = np.arange(W)
    for g in gears:
        r = idx % g
        for t in col_teeth(g):
            m &= r != t
    return m


def walk_lengths(mask):
    """L(x) = min{j >= 0 : mask[x+j]} cyclically."""
    W = len(mask)
    idx = np.arange(W)
    openpos = np.flatnonzero(mask)
    nxt = np.searchsorted(openpos, idx, side="left")
    wrapped = nxt >= len(openpos)
    nxt = np.where(wrapped, 0, nxt)
    L = openpos[nxt] - idx
    L = np.where(wrapped, L + W, L)
    return L


def gaps_of(mask):
    openpos = np.flatnonzero(mask)
    W = len(mask)
    d = np.diff(openpos)
    return np.concatenate([d, [openpos[0] + W - openpos[-1]]])


def gap_census(mask, dmax=None):
    g = gaps_of(mask)
    if dmax is None:
        dmax = int(g.max())
    return {d: int((g == d).sum()) for d in range(1, dmax + 1)}


def all_struck_counts(mask, jmax):
    """C(j) = #{x : x, ..., x+j-1 all struck}, cyclically."""
    W = len(mask)
    struck = ~mask
    C = [W]
    acc = np.ones(W, dtype=bool)
    for j in range(1, jmax + 1):
        acc &= np.roll(struck, -(j - 1))
        C.append(int(acc.sum()))
        if C[-1] == 0:
            break
    while len(C) <= jmax:
        C.append(0)
    return C


def record_scan(mask):
    """F_top = longest run of consecutive struck pairs (cyclic)."""
    C = all_struck_counts(mask, min(len(mask) - 1, 4000))
    F = 0
    for j, c in enumerate(C):
        if c > 0:
            F = j
    return F


def run_starts(mask, L):
    """#starts of L consecutive open pairs."""
    acc = mask.copy()
    for j in range(1, L):
        acc &= np.roll(mask, -j)
    return int(acc.sum())


def chain_starts(mask, L):
    """#starts of a step-2 chain of L open pairs (n, n+2, ..., n+2(L-1))."""
    acc = mask.copy()
    for j in range(1, L):
        acc &= np.roll(mask, -2 * j)
    return int(acc.sum())


def longest_run(mask):
    L = 1
    while run_starts(mask, L + 1) > 0:
        L += 1
        if L > len(mask):
            break
    return L


def longest_chain(mask):
    L = 1
    while chain_starts(mask, L + 1) > 0:
        L += 1
        if L > len(mask):
            break
    return L


# ---------------------------------------------------------------- covering record


def gear_patterns(g, L):
    """All distinct nonempty strike patterns (bitmasks) of gear g in a window of L."""
    pats = set()
    tt = teeth(g)
    for s in range(g):
        mask = 0
        for x in range(L):
            if (x + s) % g in tt:
                mask |= 1 << x
        if mask:
            pats.add(mask)
    return sorted(pats, key=lambda m: -bin(m).count("1"))


def minpieces(R, L):
    """Fewest pool gears (each a singleton or a distance-2 domino) to cover bitmask R."""
    tot = 0
    for par in (0, 1):
        xs = [x for x in range(par, L, 2) if (R >> x) & 1]
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[j + 1] == xs[j] + 2:
                j += 1
            n = j - i + 1
            tot += (n + 1) // 2
            i = j + 1
    return tot


def feasible(gears, L, node_cap=3_000_000):
    """Can [0, L) be fully struck by some phase choice?  Exact (CRT realises any phase)."""
    if L == 0:
        return True
    full = (1 << L) - 1
    small = [g for g in gears if g <= L + 1]
    c0 = len(gears) - len(small)
    pats = {g: gear_patterns(g, L) for g in small}
    bypos = {}
    for g in small:
        d = {}
        for msk in pats[g]:
            for x in range(L):
                if (msk >> x) & 1:
                    d.setdefault(x, []).append(msk)
        bypos[g] = d
    nodes = [0]
    seen = set()

    def dfs(covered, unused, c):
        if covered == full:
            return True
        nodes[0] += 1
        if nodes[0] > node_cap:
            raise TimeoutError
        R = full & ~covered
        if not unused:
            return minpieces(R, L) <= c
        key = (covered, unused, c)
        if key in seen:
            return False
        seen.add(key)
        x = (R & -R).bit_length() - 1
        # option 1: a pool gear takes x with a domino {x, x+2} or a singleton {x}
        if c > 0:
            piece = 1 << x
            if x + 2 < L:
                if dfs(covered | piece | (1 << (x + 2)), unused, c - 1):
                    return True
            if dfs(covered | piece, unused, c - 1):
                return True
        # option 2: one of the unused small gears takes x
        for i, g in enumerate(unused):
            for msk in bypos[g].get(x, ()):
                if dfs(covered | msk, unused[:i] + unused[i + 1 :], c):
                    return True
        return False

    return dfs(0, tuple(small), c0)


def record_cover(gears, lo=0, cap=200):
    """F_top by the covering formulation: largest L with feasible(L)."""
    L = lo
    while L < cap:
        try:
            ok = feasible(gears, L + 1)
        except TimeoutError:
            return L, False
        if not ok:
            return L, True
        L += 1
    return L, False


# ---------------------------------------------------------------- mex forms


def mex(nums):
    s = set(nums)
    j = 0
    while j in s:
        j += 1
    return j


def mex_simple(gears, x, tooth_fn=teeth):
    return mex([(t - x) % g for g in gears for t in tooth_fn(g)])


def mex_progressions(gears, x, B, tooth_fn=teeth):
    """mex over the truncated arithmetic progressions {(t - x) mod g + k g <= B}."""
    vals = set()
    for g in gears:
        for t in tooth_fn(g):
            v = (t - x) % g
            while v <= B:
                vals.add(v)
                v += g
    j = 0
    while j in vals:
        j += 1
    return j


def prog_terms(gears, B, tooth_fn=teeth):
    return sum(1 + B // g for g in gears for _ in tooth_fn(g))
