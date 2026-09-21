"""Independent check of the chain ladder (node R5.f.xxxv.c.xv). For each q the lane reports a
longest chain of filled holes; here the EXISTENCE half is verified directly by constructing a phase
tuple. A chain of m holes starting at hole h0 needs every cell (hole i, position p), i = 0..m-1,
p = 1..5, struck by some gear; gear g strikes cell (i, p) iff h0 + i = 7^{-1}(a - 4 - p) mod g for
one of its two classes a = +-30^{-1} mod g. Choosing h0 mod g for each gear is free (hole periods
coprime), so the question is a set cover: pick one residue per gear, cover all 5m cells.
Search: DFS over gears in decreasing capacity, pruned when the remaining gears cannot cover what is
left. Reports for each q the largest m found and an explicit phase tuple.
"""
import sys
from sympy import primerange

sys.setrecursionlimit(10000)


def cells(g, r, m):
    """Cells (i, p) struck by gear g when h0 = r mod g."""
    a = pow(30, -1, g)
    out = set()
    for i in range(m):
        h = (r + i) % g
        for p in range(1, 6):
            if (7 * h + 4 + p) % g in (a, (-a) % g):
                out.add((i, p))
    return out


def chain_exists(gears, m):
    opts = {g: [cells(g, r, m) for r in range(g)] for g in gears}
    caps = {g: max(len(c) for c in opts[g]) for g in gears}
    order = sorted(gears, key=lambda g: -caps[g])
    target = {(i, p) for i in range(m) for p in range(1, 6)}
    best = [None]

    def dfs(k, covered, chosen):
        if best[0] is not None:
            return
        if covered == target:
            best[0] = dict(chosen)
            return
        if k == len(order):
            return
        if sum(caps[g] for g in order[k:]) < len(target - covered):
            return
        g = order[k]
        seen = set()
        for r in range(g):
            gain = frozenset(opts[g][r] - covered)
            if gain in seen:
                continue
            seen.add(gain)
            chosen[g] = r
            dfs(k + 1, covered | opts[g][r], chosen)
            del chosen[g]
            if best[0] is not None:
                return
        dfs(k + 1, covered, chosen)

    dfs(0, frozenset(), {})
    return best[0]


LANE = {19: 1, 23: 2, 29: 3, 31: 4, 37: 5, 41: 6, 43: 8, 47: 9}
for q, claim in LANE.items():
    gears = [g for g in primerange(11, q + 1)]
    m = claim
    got = chain_exists(gears, m)
    print(f"q={q:2d}: chain of {m} holes exists: {got is not None}"
          + (f"  phases h0 mod g = {dict(sorted(got.items()))}" if got else ""), flush=True)
