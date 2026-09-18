"""Exact rigid record F({5..p}) by ILP feasibility with binary search between given bounds.
By CRT every shift vector of the rigid pairs is realised by one window position of the real
pattern, so the shift-rigid record equals the fixed record F(M) - and the ILP computes F(M)
without scanning the period.

usage: uv run python rigid_record_bisect.py p lo hi [time_limit_s]
   lo = a length known coverable (or 1), hi = a length known uncoverable (or a generous bound)
"""
import sys, time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

p, lo, hi = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
TL = float(sys.argv[4]) if len(sys.argv) > 4 else 1800.0

def primes_upto(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i*i::i] = False
    return np.nonzero(s)[0]

gears = [int(h) for h in primes_upto(p) if h >= 5]

def check(L):
    var = [(g, s) for g in gears for s in range(g)]
    nv = len(var)
    rows = []; cols = []
    for j, (g, s) in enumerate(var):
        u = pow(6, -1, g)
        for cls in ((s + u) % g, (s - u) % g):
            cs = np.arange(cls % g, L, g)
            rows.extend(cs.tolist()); cols.extend([j] * len(cs))
    A_cov = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(L, nv))
    r2 = [gears.index(g) for (g, s) in var]
    A_one = coo_matrix((np.ones(nv), (r2, list(range(nv)))), shape=(len(gears), nv))
    t = time.time()
    res = milp(c=np.zeros(nv), constraints=[LinearConstraint(A_cov, lb=1, ub=np.inf),
                                            LinearConstraint(A_one, lb=1, ub=1)],
               integrality=np.ones(nv), bounds=Bounds(0, 1), options={"time_limit": TL})
    dt = time.time() - t
    if res.status == 1 or (not res.success and 'time' in str(res.message).lower()):
        return None, dt
    return bool(res.success), dt

print(f"p={p} gears={gears}")
# invariant: lo coverable (assumed), hi uncoverable (assumed)
while hi - lo > 1:
    mid = (lo + hi) // 2
    ok, dt = check(mid)
    print(f"  L={mid}: {'coverable' if ok else 'UNCOVERABLE' if ok is False else 'TIMEOUT'}  ({dt:.0f}s)", flush=True)
    if ok is None:
        print("  stopping at timeout; bounds so far", lo, hi); break
    if ok: lo = mid
    else: hi = mid
print(f"F({{5..{p}}}) in [{lo}, {hi - 1}]" + ("  EXACT" if hi - lo == 1 else ""))
