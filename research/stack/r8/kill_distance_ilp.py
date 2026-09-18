"""Exact kill distance d(p) by integer programming (scipy milp / HiGHS).

Variables x[g, s] in {0,1}: gear g takes shift s (its rigid pair strikes the columns
c = s + u, s - u mod g, u = 6^-1 mod g; s = 0 is the real phase).  One shift per gear; every
column of the stretch (p^2, q^2) struck by at least one (gear, shift) chosen; minimise the number
of gears with s != 0.  d(p) = 0 would be a dead stretch.

usage: uv run python kill_distance_ilp.py [PMAX] [PMIN]
"""
import sys, math, time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

PMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 400
PMIN = int(sys.argv[2]) if len(sys.argv) > 2 else 7

def primes_upto(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i*i::i] = False
    return np.nonzero(s)[0]

P = primes_upto(PMAX + 200)

def solve(p):
    q = int(P[np.searchsorted(P, p) + 1])
    c0 = (p * p + 1) // 6 + 1; c1 = (q * q - 1) // 6 - 1
    L = c1 - c0 + 1
    gears = [int(h) for h in P if 5 <= h <= p]
    # variable index
    var = []; cost = []
    for g in gears:
        for s in range(g):
            var.append((g, s)); cost.append(0.0 if s == 0 else 1.0)
    nv = len(var)
    rows = []; cols = []
    # coverage rows: one per column
    for j, (g, s) in enumerate(var):
        u = pow(6, -1, g)
        for cls in ((s + u) % g, (s - u) % g):
            start = (cls - c0) % g
            cs = np.arange(start, L, g)
            rows.extend(cs.tolist()); cols.extend([j] * len(cs))
    A_cov = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(L, nv))
    # one-shift rows
    r2 = []; c2 = []
    for j, (g, s) in enumerate(var):
        r2.append(gears.index(g)); c2.append(j)
    A_one = coo_matrix((np.ones(len(r2)), (r2, c2)), shape=(len(gears), nv))
    # twins at the real phase
    real = [j for j, (g, s) in enumerate(var) if s == 0]
    struck = np.asarray(A_cov.tocsr()[:, real].sum(axis=1)).ravel()
    ntw = int((struck == 0).sum())
    cons = [LinearConstraint(A_cov, lb=1, ub=np.inf), LinearConstraint(A_one, lb=1, ub=1)]
    t = time.time()
    res = milp(c=np.array(cost), constraints=cons, integrality=np.ones(nv), bounds=Bounds(0, 1),
               options={"time_limit": 600})
    dt = time.time() - t
    d = int(round(res.fun)) if res.success else None
    moved = []
    if res.success:
        x = res.x
        moved = sorted({g for j, (g, s) in enumerate(var) if s != 0 and x[j] > 0.5})
    return q, L, ntw, d, dt, moved

print("   p     q   cols twins   d   d/twins  time  moved gears")
for p in [int(x) for x in P if PMIN <= x <= PMAX]:
    q, L, ntw, d, dt, moved = solve(p)
    ratio = f"{d/ntw:.2f}" if (d is not None and ntw) else "  -"
    print(f"{p:5d} {q:5d} {L:6d} {ntw:5d} {d if d is not None else '?':>4} {ratio:>8} {dt:5.1f}s  {moved if len(moved) <= 12 else str(moved[:12]) + '...'}", flush=True)
