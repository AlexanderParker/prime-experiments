"""The index laws at depth 5 of the (5, 7) tree, s ~ 10^24-10^26 (tree node R5.f.xviii).

Takes NS random depth-4 nodes (as depth4_sample.py), finds each one's nearest rung (a depth-5 node,
~50-digit members), then at that depth-5 node measures the nearest-rung offset ratio 6|j|/(ln P)^2
and the first-rung index among the 61-rough offsets against the mean law 0.0695 (ln s)^2 and the
cube bound 0.07 (ln s)^3.  (The sqrt(s) sieve is out of reach at this size.)

usage: uv run python depth5_sample.py [NS] [SEED]
"""
import sys, math, random
import numpy as np
from sympy import isprime

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 2
random.seed(SEED)

def primes_upto(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i*i::i] = False
    return np.nonzero(s)[0]

def children(s, small):
    lo, hi = (s - 1) ** 2 + 1, (s + 1) ** 2 - 1
    n = hi - lo + 1
    comp = np.zeros(n, dtype=bool)
    for p in small:
        p = int(p)
        if p * p > hi: break
        start = (-lo) % p
        if lo + start == p: start += p
        comp[start::p] = True
    prime = ~comp
    out = []
    first6 = lo + ((-lo) % 6)
    for sp in range(first6, hi, 6):
        if lo <= sp - 1 and sp + 1 <= hi and prime[sp - 1 - lo] and prime[sp + 1 - lo]:
            out.append(sp)
    return out

G61 = [int(g) for g in primes_upto(61) if g >= 5]

def nearest_rung(s):
    """nearest rung (negative first on ties), its 61-rough index, and the offset."""
    s2 = s * s; c = s // 6; i61 = 0
    for d in range(0, 2 * c):
        for j in ((0,) if d == 0 else (-d, d)):
            lo, hi = s2 + 6 * j - 1, s2 + 6 * j + 1
            if not all(lo % g and hi % g for g in G61): continue
            if isprime(lo) and isprime(hi): return j, i61
            i61 += 1
    return None, None

level = [6]
for d in range(1, 4):
    small = primes_upto(max(level) + 12)
    nxt = []
    for s in level: nxt.extend(children(s, small))
    level = sorted(nxt)
d3 = level
small4 = primes_upto(max(d3) + 12)

rows = []
for k in range(NS):
    s4 = random.choice(children(random.choice(d3), small4))       # depth-4 node
    j4, _ = nearest_rung(s4)
    s5 = s4 * s4 + 6 * j4                                          # depth-5 node
    j5, i61 = nearest_rung(s5)
    P = s5 - 1; lnP = math.log(P); lns = math.log(s5)
    rows.append((s5, j5, 6 * abs(j5) / lnP ** 2, i61))
    print(f"  depth-5 node s = {str(s5)[:12]}... ({len(str(s5))} digits): j_min = {j5}, 6|j|/(ln P)^2 = {6*abs(j5)/lnP**2:.3f}, "
          f"61-rough index {i61} (mean law {0.0695*lns**2:.0f}, cube {0.07*lns**3:.0f})", flush=True)
r = [x[2] for x in rows]; i = [x[3] for x in rows]
print(f"sample {NS} depth-5 nodes ({len(str(rows[0][0]))} digits): ratio mean {np.mean(r):.2f} max {np.max(r):.2f}; "
      f"61-rough index mean {np.mean(i):.1f} max {np.max(i)} (mean law ~ {0.0695*math.log(rows[0][0])**2:.0f}, cube ~ {0.07*math.log(rows[0][0])**3:.0f})")
