"""Canonical twin ladder, second pass (tree node R5.f.vi claim C): nearest-centre rule with the
NEGATIVE offset first on ties (the lane's rule), and per rung: j_k, sign, 6|j_k|/(ln P_k)^2, the
index of the hit among the B-rough offsets (B = min(10^6, sqrt s); the base sieve to sqrt s is
infeasible beyond rung 3, so B = 10^6 is the proxy), the number of nearer offsets rejected, and
the phase check s_{k+1} = s_k^2 (mod g) for the gears g <= 100 (predicted to hold only when
g | 6 j_k, i.e. at rate about 1/g).

usage: uv run python canonical_ladder2.py [KMAX] [DIGMAX]
"""
import sys, math, time
import numpy as np
from sympy import isprime

KMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 9
DIGMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 420

def primes_upto(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i*i::i] = False
    return [int(g) for g in np.nonzero(s)[0] if g >= 5]

SMALL = primes_upto(100)
BASE = primes_upto(10**6)

def rough(m, gs):
    for g in gs:
        if m % g == 0: return False
    return True

s = 6
locks = 0; lock_expected = 0.0; lock_tot = 0
print("rung digits(P)    j_k   6|j|/(ln P)^2  base-index  rejected  locks(g<=100: obs/expected)  time")
for k in range(KMAX):
    P = s - 1
    if len(str(P)) > DIGMAX: break
    c = s // 6; s2 = s * s; jmax = 2 * c - 1
    B = math.isqrt(s) if math.isqrt(s) < 10**6 else 10**6
    gs = [g for g in BASE if g <= B]
    t = time.time(); found = None; base_idx = 0; rejected = 0
    for d in range(0, jmax + 1):
        for j in ((0,) if d == 0 else (-d, d)):
            lo = s2 + 6 * j - 1; hi = s2 + 6 * j + 1
            if not (rough(lo, gs) and rough(hi, gs)): rejected += 1; continue
            if isprime(lo) and isprime(hi): found = j; break
            base_idx += 1; rejected += 1
        if found is not None: break
    if found is None:
        print(f"rung {k}: no twin - break"); break
    s_next = s2 + 6 * found
    obs = sum(1 for g in SMALL if s_next % g == (s2 % g))
    exp = sum(1 for g in SMALL if (6 * found) % g == 0)
    locks += obs; lock_expected += exp; lock_tot += len(SMALL)
    lnP = math.log(P) if P > 1 else 1.0
    print(f"{k:3d}  {len(str(P)):7d}  {found:7d}   {6*abs(found)/lnP**2:9.3f}   {base_idx:8d}  {rejected:8d}   {obs:3d}/{exp:<3d}                {time.time()-t:6.1f}s", flush=True)
    s = s_next
print(f"phase locks over the path: observed {locks}, exactly the gears dividing 6j: {int(lock_expected)} (of {lock_tot} gear-rungs)")
print(f"final twin lower has {len(str(s-1))} digits")
