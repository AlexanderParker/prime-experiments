"""Free two-class covering record j_2(P) (OEIS A072753) by ILP feasibility.

Variables x[p, r] in {0,1} for each gear p (prime 5..P) and residue r mod p; at most two residues
per gear; column n in [0, L) is covered iff some chosen (p, n mod p). j_2(P) = max L coverable.
Pre-registered (tree node R5.f.xxxii): a(73) = 436 reproduced (436 feasible, 437 infeasible);
a(79) in [460, 500]; increments stay in [20, 60] per prime.

usage: uv run python free_cover_ilp.py P lo hi [time_limit_s]
   lo = a length known coverable, hi = a length known (or believed) uncoverable
"""
import sys, time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

P, lo, hi = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
TL = float(sys.argv[4]) if len(sys.argv) > 4 else 1800.0

def primes_upto(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i*i::i] = False
    return [int(x) for x in np.nonzero(s)[0]]

gears = [p for p in primes_upto(P) if p >= 5]
off = {}; nv = 0
for p in gears:
    off[p] = nv; nv += p

def check(L):
    rows, cols, vals = [], [], []
    # coverage: for each column n, sum_p x[p, n mod p] >= 1
    for n in range(L):
        for p in gears:
            rows.append(n); cols.append(off[p] + n % p); vals.append(1.0)
    # cardinality: for each gear, sum_r x[p, r] <= 2
    for i, p in enumerate(gears):
        for r in range(p):
            rows.append(L + i); cols.append(off[p] + r); vals.append(1.0)
    A = coo_matrix((vals, (rows, cols)), shape=(L + len(gears), nv)).tocsr()
    lb = np.concatenate([np.ones(L), np.zeros(len(gears))])
    ub = np.concatenate([np.full(L, np.inf), np.full(len(gears), 2.0)])
    t0 = time.time()
    res = milp(c=np.zeros(nv), constraints=LinearConstraint(A, lb, ub),
               integrality=np.ones(nv), bounds=Bounds(0, 1),
               options={"time_limit": TL, "disp": False})
    dt = time.time() - t0
    if res.status == 0:
        return "coverable", dt, res.x
    if res.status == 2:
        return "UNCOVERABLE", dt, None
    return f"undecided({res.status})", dt, None

print(f"P={P} gears={gears} vars={nv}", flush=True)
while hi - lo > 1:
    mid = (lo + hi) // 2
    verdict, dt, x = check(mid)
    print(f"  L={mid}: {verdict}  ({dt:.0f}s)", flush=True)
    if verdict == "coverable": lo = mid
    elif verdict == "UNCOVERABLE": hi = mid
    else:
        print(f"  time limit at L={mid}; j_2({P}) in [{lo}, {hi - 1}]"); sys.exit(0)
print(f"j_2({P}) = {lo}  ({lo} coverable, {lo + 1} uncoverable)")
