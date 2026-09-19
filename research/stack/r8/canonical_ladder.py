"""The canonical twin ladder from (5, 7): at each rung take the twin of the stretch nearest the
centre (smallest |j|; positive j first on ties).  s_0 = 6; s_{k+1} = s_k^2 + 6 j_k with
s_{k+1} -+ 1 both prime and |j_k| <= 2c_k - 1 (c_k = s_k / 6).  Prints each rung: digits of P,
j_k, 6|j_k| / (ln P)^2, and the phases s_k^2 mod g for g = 5, 7, 11, 13 (the squaring orbit).
Stops when the members exceed 10^DIGMAX digits or after KMAX rungs.

usage: uv run python canonical_ladder.py [KMAX] [DIGMAX]
"""
import sys, math, time
from sympy import isprime

KMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 9
DIGMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 400

s = 6
print("rung  digits(P)   j_k    6|j|/(ln P)^2   s^2 mod 5,7,11,13   time")
for k in range(KMAX):
    P = s - 1
    if len(str(P)) > DIGMAX: break
    c = s // 6; s2 = s * s; jmax = 2 * c - 1
    t = time.time(); found = None
    for d in range(0, jmax + 1):
        for j in ((d,) if d == 0 else (d, -d)):
            if isprime(s2 + 6 * j - 1) and isprime(s2 + 6 * j + 1):
                found = j; break
        if found is not None: break
    if found is None:
        print(f"rung {k}: NO TWIN in the stretch of P = {P} - the ladder breaks"); break
    lnP = math.log(P) if P > 1 else 1.0
    print(f"{k:3d}   {len(str(P)):6d}   {found:6d}   {6*abs(found)/lnP**2:9.3f}      {[s2 % g for g in (5, 7, 11, 13)]}   {time.time()-t:6.1f}s", flush=True)
    s = s2 + 6 * found
print(f"final P has {len(str(s - 1))} digits; P = {str(s-1)[:60]}...")
