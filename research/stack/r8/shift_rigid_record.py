"""The shift-rigid record F_shift(p): the longest run of columns [0, L) that SOME assignment of
shifts to the rigid tooth pairs of the gears 5..p strikes completely (pair shape kept: the two
teeth of gear g are the classes s + u and s - u modulo g, u = 6^-1 mod g).  Sits between the
fixed record F(M) (real phases, the machine) and the free two-class record h_2 / A072753 (both
classes free).  Computed by ILP feasibility, L increasing until infeasible.

usage: uv run python shift_rigid_record.py [PMAX]
"""
import sys, time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

PMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 37

def primes_upto(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i*i::i] = False
    return np.nonzero(s)[0]

P = primes_upto(PMAX + 10)

def feasible(gears, L):
    var = [(g, s) for g in gears for s in range(g)]
    nv = len(var)
    rows = []; cols = []
    for j, (g, s) in enumerate(var):
        u = pow(6, -1, g)
        for cls in ((s + u) % g, (s - u) % g):
            cs = np.arange(cls % g, L, g)
            rows.extend(cs.tolist()); cols.extend([j] * len(cs))
    A_cov = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(L, nv))
    r2 = [gears.index(g) for (g, s) in var]; c2 = list(range(nv))
    A_one = coo_matrix((np.ones(nv), (r2, c2)), shape=(len(gears), nv))
    res = milp(c=np.zeros(nv), constraints=[LinearConstraint(A_cov, lb=1, ub=np.inf),
                                            LinearConstraint(A_one, lb=1, ub=1)],
               integrality=np.ones(nv), bounds=Bounds(0, 1), options={"time_limit": 300})
    return res.success

def fixed_record(gears, period_cap=2_000_000):
    # longest struck run of the real pattern over one period (capped)
    per = 1
    for g in gears: per *= g
    per = min(per, period_cap)
    struck = np.zeros(per, dtype=bool)
    for g in gears:
        u = pow(6, -1, g)
        struck[u % g::g] = True; struck[(g - u) % g::g] = True
    best = 0; run = 0
    for v in struck:
        if v: run += 1; best = max(best, run)
        else: run = 0
    return best, per

print("   p  gears  F_fixed(real phases)  F_shift(rigid pairs, shifts free)   time")
L_prev = 1
for p in [int(x) for x in P if 5 <= x <= PMAX]:
    gears = [int(h) for h in P if 5 <= h <= p]
    Ffix, per = fixed_record(gears)
    t = time.time()
    L = max(L_prev, Ffix)
    while feasible(gears, L + 1):
        L += 1
    L_prev = L
    print(f"{p:4d}  {len(gears):3d}      {Ffix:5d}{'*' if per == 2_000_000 else ' '}               {L:5d}                    {time.time()-t:6.1f}s", flush=True)
print("* fixed record measured over the first 2,000,000 columns only")
