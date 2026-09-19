"""The twin ladder (tree node R5.f.ii, lane claim 1-3, round 2).

For a twin (P, P+2), P = 6c-1: the stretch of P is the 4c-1 columns 6c^2 + j, |j| <= 2c-1, centred
on the column of P(P+2) = 36c^2 - 1 (lower member of column 6c^2); column 6c^2 + j has members
P(P+2) + 6j and P(P+2) + 6j + 2.  Gear P strikes the stretch only at the centre.

Claim 1 (ladder): every twin's own stretch contains a twin (a column with both members prime).
Claim 2 (short step): the nearest such column to the centre has 6|j| < 20 (ln P)^2.
Claim 3 (branching): for P >= 41 the stretch holds at least two twins.

usage: uv run python twin_ladder.py [PMAX]
"""
import sys, math
import numpy as np
from sympy import isprime

PMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 10**6

def primes_upto(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i*i::i] = False
    return np.nonzero(s)[0]

P = primes_upto(PMAX + 2)
ps = set(int(x) for x in P)
twin_lowers = [int(x) for x in P if x >= 5 and (int(x) + 2) in ps]

fail1 = []; fail3 = []; worst = (0, None, None); ratios = []
count_min = (10**9, None)
for Pm in twin_lowers:
    c = (Pm + 1) // 6
    A = Pm * (Pm + 2)
    jmax = 2 * c - 1
    found = None; ntw = 0
    # claim 1/2: walk outward from the centre
    for d in range(0, jmax + 1):
        for j in ((d,) if d == 0 else (d, -d)):
            if isprime(A + 6 * j) and isprime(A + 6 * j + 2):
                found = j; break
        if found is not None: break
    if found is None:
        fail1.append(Pm); continue
    r = 6 * abs(found) / (math.log(Pm) ** 2)
    ratios.append(r)
    if r > worst[0]: worst = (r, Pm, found)
    # claim 3: count twins in the stretch (full scan)
    if Pm >= 41:
        for j in range(-jmax, jmax + 1):
            if isprime(A + 6 * j) and isprime(A + 6 * j + 2): ntw += 1
        if ntw < 2: fail3.append((Pm, ntw))
        if ntw < count_min[0]: count_min = (ntw, Pm)

print(f"twin lowers tested: {len(twin_lowers)} (P <= {PMAX})")
print(f"Claim 1 (every twin's stretch holds a twin): failures {fail1}")
print(f"Claim 2 (6|j_min| / (ln P)^2 < 20): max ratio {worst[0]:.3f} at P = {worst[1]} (j = {worst[2]}); mean ratio {np.mean(ratios):.3f}; 99th pct {np.percentile(ratios, 99):.3f}")
print(f"Claim 3 (>= 2 twins in the stretch for P >= 41): failures {fail3}; minimum twin count {count_min}")
