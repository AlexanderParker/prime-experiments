"""Square-class killers of a stretch: rigid-pair shift vectors restricted by the location law.

The stretch of p starts at the square's column c0 = (p^2 - 1)/6, so relative to c0 gear g's
shift is s_g = -c0 mod g with c0 = (a^2 - 1) 6^-1, a = p mod g - a SQUARE-derived residue.  A
window can be the stretch of SOME prime only if every gear's shift lies in the square class
S_g = { -(a^2 - 1) 6^-1 mod g : a in Z/g } (about half of the residues; a = 0 allowed, it is the
gear's own square).  This ILP asks: does a shift vector with every s_g in S_g strike every
column of a window of the stretch's length?  Compared with the unrestricted killers (entry 118:
from p = 37), it measures what the location law removes.

usage: uv run python square_class_killers.py [PMAX]
"""
import sys, time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

PMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 109

def primes_upto(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i*i::i] = False
    return np.nonzero(s)[0]

P = primes_upto(PMAX + 200)

def square_class(g):
    inv6 = pow(6, -1, g)
    return sorted({(-(a * a - 1) * inv6) % g for a in range(g)})

def solve(p, restrict):
    q = int(P[np.searchsorted(P, p) + 1])
    c0 = (p * p + 1) // 6 + 1; c1 = (q * q - 1) // 6 - 1
    L = c1 - c0 + 1
    gears = [int(h) for h in P if 5 <= h <= p]
    var = []
    for g in gears:
        allowed = square_class(g) if restrict else range(g)
        for s in allowed: var.append((g, s))
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
    res = milp(c=np.zeros(nv), constraints=[LinearConstraint(A_cov, lb=1, ub=np.inf),
                                            LinearConstraint(A_one, lb=1, ub=1)],
               integrality=np.ones(nv), bounds=Bounds(0, 1), options={"time_limit": 600})
    return q, L, res.success

print("   p    q  cols  killer(any shifts)  killer(square-class shifts)")
for p in [int(x) for x in P if 7 <= x <= PMAX]:
    t = time.time()
    q, L, anyk = solve(p, False)
    _, _, sqk = solve(p, True)
    print(f"{p:4d} {q:4d} {L:5d}     {'yes' if anyk else 'no ':>3}                {'yes' if sqk else 'no ':>3}      ({time.time()-t:.0f}s)", flush=True)
