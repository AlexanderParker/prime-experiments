"""Certificate for the coverable half of a rigid record: solve the ILP at length L for the gears
5..p, read off the shift vector, turn it into a window position x by CRT (x = -s_g mod g for
every g, RigidShift.window_realises_shift), and verify DIRECTLY that the real pattern of the
gears strikes every column x, x+1, ..., x+L-1.  Prints x and the run length at x.

usage: uv run python rigid_record_certificate.py p L
"""
import sys, time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix
from math import prod

p, L = int(sys.argv[1]), int(sys.argv[2])

def primes_upto(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i*i::i] = False
    return np.nonzero(s)[0]

gears = [int(h) for h in primes_upto(p) if h >= 5]
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
           integrality=np.ones(nv), bounds=Bounds(0, 1), options={"time_limit": 7200})
print(f"ILP: {'feasible' if res.success else 'not solved'} in {time.time()-t:.0f}s")
if not res.success:
    sys.exit(1)
shift = {g: s for j, (g, s) in enumerate(var) if res.x[j] > 0.5}
# CRT: x = -s_g mod g
P = prod(gears); x = 0
for g in gears:
    Pg = P // g
    x = (x + ((-shift[g]) % g) * Pg * pow(Pg, -1, g)) % P
# direct verification on the real pattern: column c struck iff c = +-u mod g for some gear
def struck(c):
    for g in gears:
        u = pow(6, -1, g)
        if c % g == u or c % g == (g - u) % g: return True
    return False
run = 0
while struck(x + run): run += 1
print(f"shifts: {shift}")
print(f"window start x = {x} (period {P}); real pattern struck run from x = {run} (needed >= {L}): {'CERTIFIED' if run >= L else 'FAILED'}")
print(f"members at x: 6x-1 = {6*x-1}, 6x+1 = {6*x+1}")
