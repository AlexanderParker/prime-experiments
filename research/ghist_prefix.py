"""Round 21 (mechanic, for Lateral): prefix gap histogram + C14 phase
arg H_5(1) at machines beyond full-scan reach. Decides pin-vs-drift:
Lateral's closed-form model predicts 125.5-125.9 deg at m41/43; a measured
126.0 +- 0.1 falsifies the drift. Usage: python ghist_prefix.py y nslots"""
import sys, cmath, math
import numpy as np

y, K = int(sys.argv[1]), int(float(sys.argv[2]))
def primes_upto(n):
    s = np.ones(n+1, bool); s[:2] = False
    for i in range(2, int(n**0.5)+1):
        if s[i]: s[i*i::i] = False
    return np.flatnonzero(s)
gears = [int(p) for p in primes_upto(y) if p >= 5]
hist = np.zeros(256, np.int64)
tail = np.array([], np.int64)
seg = 64_000_000
for lo in range(0, K, seg):
    hi = min(K, lo+seg)
    ex = np.zeros(hi-lo, bool)
    for q in gears:
        u = pow(6, -1, q)
        ex[(u-lo) % q::q] = True
        ex[(-u-lo) % q::q] = True
    op = np.flatnonzero(~ex).astype(np.int64) + lo
    ops = np.concatenate([tail, op])
    if len(ops) > 1:
        d = np.diff(ops)
        hist += np.bincount(np.minimum(d, 255), minlength=256)[:256]
    tail = ops[-1:].copy()
N = [0]*5
for v in range(1, 256):
    N[v % 5] += int(hist[v])
H0 = sum(N)
H1 = sum(N[r] * cmath.exp(2j*math.pi*r/5) for r in range(5))
print(f"machine {y}, prefix {K:.3g} slots, gaps {H0}")
print(f"mod-5 class counts {N}")
print(f"arg H_5(1) = {math.degrees(cmath.phase(H1)):+.2f} deg   |H_5(1)|/H0 = {abs(H1)/H0:.5f}")
print(f"mean gap = {K/H0:.4f}  1.015/mean = {1.015*H0/K:.5f}")
