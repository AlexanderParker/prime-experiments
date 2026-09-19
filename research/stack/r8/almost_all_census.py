"""Empirical parameters of AlmostAll (tree node R5.f.xxii): for every twin centre s <= X, the rung
count T(s) against c s/(ln s)^2; the exceptional set for c = 0.66, 0.9, 1.0, 1.1, 1.2 (the mean law has
c = 1.3203); the largest exceptional s for each c.
usage: uv run python almost_all_census.py [X]
"""
import sys, math, numpy as np
X = int(sys.argv[1]) if len(sys.argv) > 1 else 10**5
def primes_upto(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i*i::i] = False
    return np.nonzero(s)[0]
P = primes_upto((X + 2) ** 1 + 10); ps = set(int(x) for x in P)
centres = [int(x) + 1 for x in P if 5 <= x <= X - 1 and (int(x) + 2) in ps]
big = primes_upto(X + 12)
def T(s):
    lo, hi = (s - 1) ** 2 + 1, (s + 1) ** 2 - 1; n = hi - lo + 1
    comp = np.zeros(n, dtype=bool)
    for p in big:
        p = int(p)
        if p * p > hi: break
        start = (-lo) % p
        if lo + start == p: start += p
        comp[start::p] = True
    prime = ~comp; first6 = lo + ((-lo) % 6)
    return sum(1 for sp in range(first6, hi, 6) if lo <= sp - 1 and sp + 1 <= hi and prime[sp - 1 - lo] and prime[sp + 1 - lo])
rows = [(s, T(s)) for s in centres]
print(f"twin centres {len(rows)} (s <= {X}); mean T/(1.3203 s/ln^2 s) = {np.mean([t/(1.3203*s/math.log(s)**2) for s,t in rows]):.4f}")
for c in (0.66, 0.9, 1.0, 1.1, 1.2):
    exc = [s for s, t in rows if t < c * s / math.log(s) ** 2]
    print(f"  c = {c}: exceptional {len(exc)} of {len(rows)}; largest exceptional s = {max(exc) if exc else None}; count above 10^4: {sum(1 for s in exc if s > 10**4)}")
