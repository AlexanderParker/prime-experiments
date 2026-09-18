"""The kill distance d(p) of a stretch: the least number of gears whose rigid tooth pair must be
re-phased (shifted, pair shape kept) away from the real phase so that every column of the
stretch (p^2, q^2) is struck.  A re-phased gear abandons its real teeth; columns it alone struck
become targets too.  Branch: the lowest uncovered target must be struck by a not-yet-re-phased
gear through one of its two teeth, which fixes that gear's shift.  Iterative deepening on d.

Also reports the capacity lower bound: re-phased gears g_1..g_d can newly strike at most
sum 2 ceil(L / g_i) columns, so d >= the least d with the d largest capacities summing to #twins.

usage: uv run python kill_distance.py [PMAX] [DMAX]
"""
import sys, math
import numpy as np

PMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 150
DMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 6

def primes_upto(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i*i::i] = False
    return np.nonzero(s)[0]

P = primes_upto(PMAX + 200)

def stretch(p):
    q = int(P[np.searchsorted(P, p) + 1])
    c0 = (p * p + 1) // 6 + 1; c1 = (q * q - 1) // 6 - 1
    L = c1 - c0 + 1
    gears = [int(h) for h in P if 5 <= h <= p]
    strikes = {}
    count = np.zeros(L, dtype=np.int32)
    for g in gears:
        u = pow(6, -1, g)
        m = np.zeros(L, dtype=bool)
        m[(u - c0) % g::g] = True; m[(g - u - c0) % g::g] = True
        strikes[g] = m; count += m
    return q, c0, L, gears, strikes, count

def solve(p):
    q, c0, L, gears, strikes, count = stretch(p)
    twins = set(np.nonzero(count == 0)[0].tolist())
    if not twins: return q, L, 0, 0, 0
    lone = {g: set(np.nonzero(strikes[g] & (count == 1))[0].tolist()) for g in gears}
    U = {g: pow(6, -1, g) for g in gears}

    def covered_by(g, s):
        u = U[g]; cls = {(s + u) % g, (s - u) % g}
        return {c for c in range(L) if (c + c0) % g in cls}

    best = None
    def rec(targets, rephased, depth, limit):
        nonlocal best
        if not targets: best = dict(rephased); return True
        if depth == limit: return False
        t = min(targets)
        for g in gears:
            if g in rephased: continue
            u = U[g]
            for tooth in (u, g - u):
                s = ((t + c0) - tooth) % g          # shift so that tooth lands on t
                cov = covered_by(g, s)
                new_targets = (targets - cov) | (lone[g] - cov)
                # a re-phased gear's abandoned lone kills also need covering
                rephased[g] = s
                if rec(new_targets, rephased, depth + 1, limit): return True
                del rephased[g]
        return False

    caps = sorted((2 * math.ceil(L / g) for g in gears), reverse=True)
    acc = 0; cap_bound = 0
    for c in caps:
        cap_bound += 1; acc += c
        if acc >= len(twins): break
    for limit in range(1, DMAX + 1):
        if rec(set(twins), {}, 0, limit):
            return q, L, len(twins), limit, cap_bound
    return q, L, len(twins), None, cap_bound

for p in [int(x) for x in P if 7 <= x <= PMAX]:
    q, L, nt, d, cb = solve(p)
    print(f"p={p:4d} q={q:4d} cols={L:5d} twins={nt:3d} kill distance d={d if d is not None else f'>{DMAX}'}  capacity bound={cb}", flush=True)
