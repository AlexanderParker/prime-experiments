"""The index laws at depths 6 and 7 of the (5, 7) tree (tree node R5.f.xix), s ~ 10^50 and
10^100: from NS random depth-4 nodes descend by nearest rungs (negative first) to depth 6 and
depth 7, recording at each node the nearest-rung offset ratio 6|j|/(ln P)^2 and the first-rung
index among the 61-rough offsets against the mean law 0.0695 (ln s)^2 and the cube 0.07 (ln s)^3.

usage: uv run python depth_n_sample.py [NS] [DEPTH_MAX] [SEED]
"""
import sys, math, random
import numpy as np
from sympy import isprime

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
DMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 7
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 3
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

stats = {}
for k in range(NS):
    s = random.choice(children(random.choice(d3), small4))       # depth 4
    depth = 4
    while depth < DMAX:
        j, i61 = nearest_rung(s)
        s_next = s * s + 6 * j
        depth += 1
        lnP = math.log(s_next - 1); lns = math.log(s_next)
        # measure at the new node
        j2, i2 = nearest_rung(s_next) if depth < DMAX else (None, None)
        if depth >= 6:
            jj, ii = nearest_rung(s_next)
            stats.setdefault(depth, []).append((6 * abs(jj) / lnP ** 2, ii, lns))
            print(f"  depth {depth} node ({len(str(s_next))} digits): j_min = {jj}, ratio {6*abs(jj)/lnP**2:.3f}, 61-rough index {ii} (mean law {0.0695*lns**2:.0f}, cube {0.07*lns**3:.0f})", flush=True)
        s = s_next
for d in sorted(stats):
    r = [x[0] for x in stats[d]]; i = [x[1] for x in stats[d]]; lns = np.mean([x[2] for x in stats[d]])
    print(f"depth {d}: {len(r)} nodes; ratio mean {np.mean(r):.2f} max {np.max(r):.2f}; 61-rough index mean {np.mean(i):.0f} max {np.max(i)} (mean law ~ {0.0695*lns**2:.0f}, cube ~ {0.07*lns**3:.0f})")
