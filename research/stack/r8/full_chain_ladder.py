"""Chain ladder on the FULL opening set (follow-up to full_blocks.py).

Block b = the three open columns 5b, 5b+2, 5b+3 (positions p = 1, 2, 3); gear g >= 7 strikes block
b at position p iff b = 5^{-1}(+-6^{-1} - e_p) mod g, (e_1, e_2, e_3) = (0, 2, 3). Only gear 7 can
strike two cells of one block (positions 1, 2, at b = 4 mod 7); every gear >= 11 gives at most one
cell per block and exactly 6 cells per period of g blocks.

A chain of m filled blocks exists for machine 5..q iff a residue h0 mod g can be chosen for each
gear so that all 3m cells are covered (block periods coprime, so every phase tuple occurs).

PRE-REGISTERED: (1) word bound - the largest m with sum over gears of the densest m-window >= 3m;
(2) the chain ladder by set cover, and the column record it implies (5m + ends) against the known
records F(q) = 4, 6, 10, 17, 24, 33, 42, 57 for q = 7..31; (3) the window in blocks is q'^2/30.
"""
import sys
from sympy import primerange, nextprime

sys.setrecursionlimit(10000)
E = (0, 2, 3)


def cells(g, r, m):
    c = pow(6, -1, g)
    inv5 = pow(5, -1, g)
    out = set()
    for i in range(m):
        b = (r + i) % g
        for p, e in enumerate(E, start=1):
            if (5 * b + e) % g in (c, (-c) % g):
                out.add((i, p))
    return out


def dense(g, m):
    return max(len(cells(g, r, m)) for r in range(g))


def chain_exists(gears, m):
    opts = {g: [cells(g, r, m) for r in range(g)] for g in gears}
    caps = {g: max(len(c) for c in opts[g]) for g in gears}
    order = sorted(gears, key=lambda g: -caps[g])
    target = {(i, p) for i in range(m) for p in range(1, 4)}
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


KNOWN_F = {7: 4, 11: 6, 13: 10, 17: 17, 19: 24, 23: 33, 29: 42, 31: 57}
print(" q | word bound | chain | columns implied | known F(q) | window in blocks")
for q in [7, 11, 13, 17, 19, 23, 29, 31]:
    gears = list(primerange(7, q + 1))
    m = 0
    while m < 30 and sum(dense(g, m + 1) for g in gears) >= 3 * (m + 1):
        m += 1
    chain = 0
    while chain < m and chain_exists(gears, chain + 1) is not None:
        chain += 1
    qn = nextprime(q)
    print(f"{q:3d}| {m:10d} | {chain:5d} | {5*chain:15d} | {KNOWN_F[q]:10d} | {qn*qn/30:8.1f}", flush=True)
