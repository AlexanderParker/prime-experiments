"""Twin centres between consecutive prime squares (p^2, q^2), p < q consecutive primes >= 5.

The stretch of a twin centre is the region with gap g = q - p = 2. Pre-registered (tree node
R5.f.xxvi): (1) no region up to q^2 <= 10^9 is empty; (2) the minimum count over regions of gap g
grows with g, so the g = 2 regions (the stretches) are the binding case of the machine's window
statement; (3) counts follow 1.3203 (q^2 - p^2)/ln^2(p^2) with mean ratio 1 within 1%.
"""
import numpy as np, time, sys
from collections import defaultdict

LIM = int(sys.argv[1]) if len(sys.argv) > 1 else 10**9
t0 = time.time()
N = LIM + 10**6
sieve = np.ones(N // 2, dtype=bool); sieve[0] = False
for p in range(3, int(N**0.5) + 1, 2):
    if sieve[p // 2]:
        sieve[p * p // 2::p] = False
primes = np.arange(1, N, 2)[sieve]
tw = primes[np.searchsorted(primes, primes + 2) < len(primes)]
tw = tw[primes[np.searchsorted(primes, tw + 2)] == tw + 2]
centres = tw + 1; centres = centres[centres % 6 == 0]
print(f"sieve to {N:,} in {time.time()-t0:.0f}s; twin centres {len(centres):,}")
small = primes[(primes >= 5) & (primes * primes <= LIM)]
rows = []
for p, q in zip(small[:-1], small[1:]):
    a, b = int(p) ** 2, int(q) ** 2
    cnt = np.searchsorted(centres, b) - np.searchsorted(centres, a + 1)
    law = 1.3203 * (b - a) / np.log(a) ** 2
    rows.append((int(p), int(q), int(q - p), int(cnt), law))
print(f"regions {len(rows)}; empty {sum(1 for r in rows if r[3] == 0)}")
print(f"mean ratio count/law {np.mean([r[3]/r[4] for r in rows]):.4f}")
bygap = defaultdict(list)
for r in rows: bygap[r[2]].append(r)
print("gap  regions  min_count(at p)  min_ratio(at p)  mean_ratio")
for g in sorted(bygap):
    rs = bygap[g]
    mn = min(rs, key=lambda r: r[3]); mr = min(rs, key=lambda r: r[3] / r[4])
    print(f"{g:3d}  {len(rs):5d}   {mn[3]:6d} ({mn[0]})   {mr[3]/mr[4]:.3f} ({mr[0]})   {np.mean([r[3]/r[4] for r in rs]):.3f}")
worst = sorted(rows, key=lambda r: r[3])[:8]
print("eight smallest counts (p, q, gap, count, law):")
for r in worst: print(f"  {r[0]} {r[1]} {r[2]} {r[3]} {r[4]:.1f}")
