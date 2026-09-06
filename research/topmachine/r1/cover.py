"""Exact record of the top machine as a covering problem.

A run of L consecutive struck pairs exists somewhere in the period iff [0, L) can be
covered by choosing, for each gear g, a phase s_g and taking
    S_g = {x in [0,L) : (x + s_g) mod g in {0, g-2}}.
CRT makes every phase vector realisable, so this is exact.

Structure used:
  * gear g > L+1 can cover at most a "domino" {x, x+2} inside the window (its two teeth
    are 2 apart; the other route between them is g-2 > L-1, outside).  All such gears are
    interchangeable: they form a POOL of size c.
  * gear g <= L+1 is SMALL: it can also use the long letter g-2, and if g <= L it repeats.

Feasibility: branch on the lowest uncovered position; either a pool piece {x, x+2} takes it,
or one of the unused small gears takes it (at most two phases put a given position on a tooth).
When no small gears are left the answer is exact in linear time:
    minpieces(R) = sum over maximal (step 2) chains of R of ceil(len/2)   <=  c.
"""

from math import prod


def small_patterns(g, L):
    """All distinct strike patterns (as bitmasks) of gear g in a window of L, indexed by
    the lowest position each contains, for quick lookup."""
    pats = set()
    for s in range(g):
        mask = 0
        for x in range(L):
            if (x + s) % g in (0, (g - 2) % g):
                mask |= 1 << x
        if mask:
            pats.add(mask)
    return sorted(pats, key=lambda m: -bin(m).count("1"))


def minpieces(R, L):
    """Fewest pool gears needed to cover the set R (bitmask) with pieces {x} or {x, x+2}."""
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


def feasible(gears, L, node_cap=4_000_000):
    full = (1 << L) - 1
    small = [g for g in gears if g <= L + 1]
    c0 = len(gears) - len(small)
    pats = {g: small_patterns(g, L) for g in small}
    # patterns of g containing position x
    bypos = {}
    for g in small:
        d = {}
        for m in pats[g]:
            for x in range(L):
                if (m >> x) & 1:
                    d.setdefault(x, []).append(m)
        bypos[g] = d
    maxsize = {g: max(bin(m).count("1") for m in pats[g]) for g in small}
    nodes = [0]
    seen = {}

    def dfs(covered, unused, c):
        if covered == full:
            return True
        nodes[0] += 1
        if nodes[0] > node_cap:
            raise TimeoutError
        R = full & ~covered
        # exact terminal
        if not unused:
            return minpieces(R, L) <= c
        # capacity prune
        cap = 2 * c + sum(maxsize[small[i]] for i in unused)
        if cap < bin(R).count("1"):
            return False
        # stronger prune: pool alone must be able to finish what small gears cannot help with
        key = (covered, unused, c)
        if key in seen:
            return seen[key]
        low = (R & -R).bit_length() - 1
        # option 1: a pool gear takes the lowest position (dominating piece {low, low+2})
        if c > 0:
            piece = 1 << low
            if low + 2 < L:
                piece |= 1 << (low + 2)
            if dfs(covered | piece, unused, c - 1):
                seen[key] = True
                return True
        # option 2: a small gear takes it
        for idx in list(unused):
            g = small[idx]
            for m in bypos[g].get(low, ()):
                if dfs(covered | m, unused - {idx}, c):
                    seen[key] = True
                    return True
        seen[key] = False
        return False

    try:
        return dfs(0, frozenset(range(len(small))), c0), nodes[0]
    except TimeoutError:
        return None, nodes[0]


def F_cover(gears, cap=200):
    """Largest L with a full cover.  Feasibility is monotone decreasing in L."""
    L = 1
    best = 0
    while L < cap:
        ok, _ = feasible(gears, L)
        if ok is None:
            return best, "timeout at L=%d" % L
        if ok:
            best = L
            L += 1
        else:
            return best, "exact"
    return best, "cap"


def witness(gears, L):
    """Return an assignment: for each gear, its strike positions in [0,L)."""
    full = (1 << L) - 1
    small = [g for g in gears if g <= L + 1]
    large = [g for g in gears if g > L + 1]
    pats = {g: small_patterns(g, L) for g in small}
    chosen = {}

    def dfs(covered, unused, pool):
        if covered == full:
            return True
        R = full & ~covered
        if not unused:
            return minpieces(R, L) <= pool
        low = (R & -R).bit_length() - 1
        if pool > 0:
            piece = 1 << low
            if low + 2 < L:
                piece |= 1 << (low + 2)
            if dfs(covered | piece, unused, pool - 1):
                chosen.setdefault("pool", []).append(piece)
                return True
        for idx in list(unused):
            g = small[idx]
            for m in pats[g]:
                if (m >> low) & 1:
                    if dfs(covered | m, unused - {idx}, pool):
                        chosen[g] = m
                        return True
        return False

    ok = dfs(0, frozenset(range(len(small))), len(large))
    if not ok:
        return None
    return chosen
