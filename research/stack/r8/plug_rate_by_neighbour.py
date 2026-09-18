"""Does the top layer's plug rate on a base-open column depend on the base pattern around it?
Buckets every base-open column of every stretch (7 <= p <= PMAX) by the distance to the previous
base-open column (in columns) and by whether the base-open column's two members are struck-on-the-
left / right by the base pattern's neighbours, and reports the plug rate per bucket with counts.
Independence = the same rate in every bucket to within sampling error.

Also: the global extreme check - the longest plug run over the WHOLE sample against the
independent expectation ln(N_total)/ln(1/rate) (entry 115's clustering claim).

usage: uv run python plug_rate_by_neighbour.py [PMAX]
"""
import sys, math, collections
import numpy as np

PMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 20000

def primes_upto(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i*i::i] = False
    return np.nonzero(s)[0]

P = primes_upto(PMAX + 200)
plist = [int(x) for x in P if 7 <= x <= PMAX]

bucket_n = collections.Counter(); bucket_k = collections.Counter()
side_n = collections.Counter(); side_k = collections.Counter()
N_total = 0; K_total = 0; Kmax = 0; Kmax_p = 0; run = 0
for p in plist:
    q = int(P[np.searchsorted(P, p) + 1])
    c0 = (p * p + 1) // 6 + 1; c1 = (q * q - 1) // 6 - 1
    if c1 < c0: continue
    L = c1 - c0 + 1
    B = int(round(q ** (2.0 / 3.0)))
    base = np.zeros(L, dtype=bool); top = np.zeros(L, dtype=bool)
    baseL = np.zeros(L, dtype=bool); baseR = np.zeros(L, dtype=bool)
    for h in P:
        h = int(h)
        if h < 5: continue
        if h > p: break
        u = pow(6, -1, h)
        if h <= B:
            baseL[(u - c0) % h::h] = True        # 6c-1 struck
            baseR[(h - u - c0) % h::h] = True    # 6c+1 struck
        else:
            top[(u - c0) % h::h] = True; top[(h - u - c0) % h::h] = True
    base = baseL | baseR
    idx = np.nonzero(~base)[0]
    if len(idx) == 0: continue
    plugged = top[idx]
    gaps = np.diff(idx, prepend=idx[0] - 10**9)
    for i, (g, k) in enumerate(zip(gaps, plugged)):
        b = min(int(g), 12) if g < 10**8 else 0
        bucket_n[b] += 1; bucket_k[b] += int(k)
        # neighbours: is column c-1 struck on left/right, c+1 struck on left/right (base only)
        c = idx[i]
        if 0 < c < L - 1:
            key = (int(baseL[c-1]), int(baseR[c-1]), int(baseL[c+1]), int(baseR[c+1]))
            side_n[key] += 1; side_k[key] += int(k)
    N_total += len(idx); K_total += int(plugged.sum())
    run = 0
    for k in plugged:
        if k:
            run += 1
            if run > Kmax: Kmax, Kmax_p = run, p
        else: run = 0

rate = K_total / N_total
print(f"base-open columns {N_total}, plugged {K_total}, rate {rate:.4f}")
print(f"longest plug run over the whole sample: {Kmax} at p={Kmax_p}; independent expectation ln(N)/ln(1/rate) = {math.log(N_total)/math.log(1/rate):.1f}")
print("\nplug rate by distance to the previous base-open column (0 = first of stretch, 12 = 12 or more):")
for b in sorted(bucket_n):
    n, k = bucket_n[b], bucket_k[b]
    se = math.sqrt(rate * (1 - rate) / n)
    print(f"  d={b:2d}: n={n:8d} rate={k/n:.4f}  ({(k/n-rate)/se:+.1f} sigma)")
print("\nplug rate by the base strikes on the two neighbouring columns (L-1,R-1,L+1,R+1):")
for key in sorted(side_n):
    n, k = side_n[key], side_k[key]
    se = math.sqrt(rate * (1 - rate) / n)
    print(f"  {key}: n={n:8d} rate={k/n:.4f}  ({(k/n-rate)/se:+.1f} sigma)")
