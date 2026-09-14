"""Loop, iteration 2: the pick-up walk with backtracking.  Same invariant as iteration 1 (every
visited gear, the current gear and the base off their teeth after each step, column in
[0, q^2]); depth-first search over (periods 1..K, direction) with a node budget, to learn whether
ANY such walk exists per machine, and how the choices must go.  Not a blind rule: a feasibility
check of the principle.  usage: uv run python research/stack/r8/pickup_walk_search.py 600 3
"""
import sys
import numpy as np
from sympy import primerange
from pathlib import Path

def search(q, K, order, budget=200000):
    ps = list(primerange(2, q + 1)); base = []; P = 1
    for p in ps:
        if P * p <= q // 2: P *= p; base.append(p)
        else: break
    gears = [p for p in ps if p not in base]
    if order == 'desc': gears = gears[::-1]
    bg = [x for x in base if x >= 5]
    nodes = [0]
    def rec(i, n, visited, path):
        if nodes[0] > budget: return None
        if i == len(gears):
            return (n, path) if q < n else None
        g = gears[i]; vis = visited + [g]
        for k in range(1, K + 1):
            for d in (1, -1):
                nodes[0] += 1
                m = n + 2 * k * P * g * d
                if m < 0 or m > q * q - 2: continue
                if all(m % h not in (0, h - 2) for h in vis):
                    r = rec(i + 1, m, vis, path + [(g, k, d)])
                    if r: return r
        return None
    r = rec(0, -1, bg, [])
    return r, nodes[0], gears

def main():
    Q, K = int(sys.argv[1]), int(sys.argv[2]); qs = list(primerange(11, Q + 1))
    N = Q * Q + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    out = [__doc__.strip(), ""]
    for order in ('desc', 'asc'):
        found = 0; fails = []; ex = []
        for q in qs:
            r, nodes, gears = search(q, K, order)
            if r:
                n, path = r; assert sv[n] and sv[n + 2]; found += 1
                if q in (31, 101, 199, 499): ex.append((q, n, path[:6], '...' if len(path) > 6 else ''))
            else: fails.append((q, nodes))
        out.append(f"{order}, K = {K}: a pick-up walk exists at {found} of {len(qs)} machines; none found (budget exhausted or none) at {fails[:12]}{'...' if len(fails) > 12 else ''}")
        for e in ex: out.append(f"   {e}")
    Path("research/stack/r8/results_pickup_walk_search.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()
