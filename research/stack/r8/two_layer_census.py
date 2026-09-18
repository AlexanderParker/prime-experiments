"""Two-layer census of every stretch (p^2, q^2): the base layer (gears <= B = q^(2/3)) leaves
base-open columns; the top layer (gears in (B, p]) kills some of them as straddling products
g x r (rough_member_form); the survivors are the twins.  Measures, per stretch, the plug run
K(p): the longest run of CONSECUTIVE base-open columns all killed by the top layer.  A dead
stretch is K(p) = #base-open.

Pre-registered (entry 115): killed fraction -> 5/9 (= 1 - (log B / log q)^2); K(p) <= 2 ln(#base-open) + 4
at every p; no dead stretch.

usage: uv run python two_layer_census.py [PMAX]
"""
import sys, math
import numpy as np

PMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 20000

def primes_upto(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i*i::i] = False
    return np.nonzero(s)[0]

P = primes_upto(PMAX + 200)
plist = [int(x) for x in P if x >= 7 and x <= PMAX]

def inv(a, m):
    return pow(a, -1, m)

rows = []
recK = 0
for idx, p in enumerate(plist):
    q = int(P[np.searchsorted(P, p) + 1])
    c0 = (p * p + 1) // 6 + 1          # first column with 6c-1 > p^2
    c1 = (q * q - 1) // 6 - 1          # last column with 6c+1 < q^2
    if c1 < c0:
        continue
    L = c1 - c0 + 1
    B = int(round(q ** (2.0 / 3.0)))
    base = np.zeros(L, dtype=bool)     # struck by a gear <= B
    top = np.zeros(L, dtype=bool)      # struck by a gear in (B, p]
    for h in P:
        h = int(h)
        if h < 5: continue
        if h > p: break
        u = inv(6, h)
        arr = base if h <= B else top
        for tooth in (u, h - u):       # 6c-1 = 0 mod h  <->  c = u ;  6c+1 = 0  <->  c = -u
            start = (tooth - c0) % h
            arr[start::h] = True
    bopen = ~base
    nb = int(bopen.sum())
    twins = bopen & ~top
    nt = int(twins.sum())
    # plug runs on the base-open subsequence
    killed_seq = top[bopen]
    K = 0; run = 0
    for k in killed_seq:
        if k: run += 1; K = max(K, run)
        else: run = 0
    frac = 1 - nt / nb if nb else float('nan')
    rows.append((p, q, L, nb, nt, K, frac))
    if K > recK:
        recK = K
        print(f"record K: p={p:6d} q={q:6d} cols={L:7d} base_open={nb:6d} twins={nt:5d} K={K:3d} "
              f"2ln(nb)+4={2*math.log(nb)+4:.1f} killed_frac={frac:.3f}", flush=True)

print("\nsummary by decade of p:")
import collections
byd = collections.defaultdict(list)
for r in rows:
    byd[int(math.log10(r[0]))].append(r)
for d in sorted(byd):
    rs = byd[d]
    fr = np.mean([r[6] for r in rs if not math.isnan(r[6])])
    Kmax = max(r[5] for r in rs)
    nbmin = min(r[3] for r in rs); nbmed = int(np.median([r[3] for r in rs]))
    dead = sum(1 for r in rs if r[4] == 0)
    print(f"  10^{d}: stretches={len(rs):5d} killed_frac_mean={fr:.3f} K_max={Kmax:3d} base_open min/med={nbmin}/{nbmed} dead={dead}")
