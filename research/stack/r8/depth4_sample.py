"""The index laws at depth 4 of the (5, 7) tree, s ~ 10^12-10^13 (tree node R5.f.xvi).

Takes NS random depth-4 nodes (a random child of a random depth-3 node) and measures at each:
the nearest rung's offset ratio 6|j|/(ln P)^2; the first-rung index among the 61-rough offsets
against the cube bound 0.07 (ln s)^3 and the mean law 0.0695 (ln s)^2; and the first-rung index
among the sqrt(s)-rough offsets (predicted mean ~4).  Extends the laws measured to 10^6 by six
orders of magnitude in s.

usage: uv run python depth4_sample.py [NS] [SEED]
"""
import sys, math, random
import numpy as np
from sympy import isprime

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1
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

# depths 0..3
level = [6]
for d in range(1, 4):
    top = max(level); small = primes_upto(top + 12)
    nxt = []
    for s in level: nxt.extend(children(s, small))
    level = sorted(nxt)
d3 = level
print(f"depth 3: {len(d3)} nodes")
small4 = primes_upto(int(math.isqrt((max(d3) + 1) ** 2)) + 10)
G61 = [int(g) for g in primes_upto(61) if g >= 5]

rows = []
for k in range(NS):
    s3 = random.choice(d3)
    ch = children(s3, small4)
    s = random.choice(ch)                      # a depth-4 node, s ~ 10^12-10^13
    s2 = s * s; c = s // 6; P = s - 1
    x = math.isqrt(s)
    base_x = [int(g) for g in primes_upto(x) if g >= 5]
    # walk outward from the centre
    j = 0; jmin = None; i61 = 0; ix = 0; hit61 = None; hitx = None
    order = [0]
    d = 1
    while jmin is None or hitx is None:
        for jj in ((0,) if d == 0 else (-d, d)):
            pass
        order = [d, -d]
        for jj in order:
            lo, hi = s2 + 6 * jj - 1, s2 + 6 * jj + 1
            r61 = all(lo % g and hi % g for g in G61)
            rx = r61 and all(lo % g and hi % g for g in base_x)
            tw = r61 and isprime(lo) and isprime(hi)
            if tw and jmin is None: jmin = jj
            if r61 and hit61 is None:
                if tw: hit61 = i61
                else: i61 += 1
            if rx and hitx is None:
                if tw: hitx = ix
                else: ix += 1
            if jmin is not None and hit61 is not None and hitx is not None: break
        d += 1
        if d > 2 * c - 1: break
    lnP = math.log(P); lns = math.log(s)
    rows.append((s, jmin, 6 * abs(jmin) / lnP ** 2 if jmin is not None else None, hit61, hitx))
    print(f"  s = {s} ({len(str(s))} digits): j_min = {jmin}, 6|j|/(ln P)^2 = {6*abs(jmin)/lnP**2:.3f}, "
          f"index among 61-rough {hit61} (mean law {0.0695*lns**2:.0f}, cube {0.07*lns**3:.0f}), index among sqrt(s)-rough {hitx}", flush=True)
r = [x[2] for x in rows]; i61s = [x[3] for x in rows]; ixs = [x[4] for x in rows]
print(f"sample {NS} depth-4 nodes: ratio 6|j|/(ln P)^2 mean {np.mean(r):.2f} max {np.max(r):.2f}; "
      f"61-rough index mean {np.mean(i61s):.1f} max {np.max(i61s)} (mean law ~ {0.0695*math.log(rows[0][0])**2:.0f}); "
      f"sqrt(s)-rough index mean {np.mean(ixs):.2f} max {np.max(ixs)} (predicted mean 4)")
