"""Dispersion index of twin-centre counts in windows of length x^delta at height x.

Tree node R5.f.xxv follow-up: the inheritance lane measured index 0.82 (rungs) / 0.79 (control)
for windows of length 4 sqrt(x) at height x = s'^2. Here: random windows at heights in
[X, 2X], lengths x^delta for delta in {0.3, 0.4, 0.5, 0.6, 0.7}; index = var/mean of the
counts. Prediction written before running (Montgomery-Soundararajan shape): index falls with
delta, roughly 1 - c*delta.
"""
import numpy as np, sys, time

X = int(sys.argv[1]) if len(sys.argv) > 1 else 2 * 10**8
t0 = time.time()
N = 2 * X + 4 * 10**6  # windows at x <= 2X of length x^0.7 stay inside
sieve = np.ones(N // 2, dtype=bool)  # odd numbers: index i -> 2i+1
sieve[0] = False
r = int(N**0.5) + 1
for p in range(3, r, 2):
    if sieve[p // 2]:
        sieve[p * p // 2::p] = False
odd = np.arange(1, N, 2)
primes = odd[sieve]
# twin centres s = p+1 with p, p+2 prime, s multiple of 6
tw = primes[np.searchsorted(primes, primes + 2) < len(primes)]
tw = tw[primes[np.searchsorted(primes, tw + 2)] == tw + 2]
centres = tw + 1
centres = centres[centres % 6 == 0]
print(f"sieve to {N:,} in {time.time()-t0:.0f}s; twin centres {len(centres):,}")
rng = np.random.default_rng(20260920)
print("delta  windows  mean_count  ratio  index=mean((c-lam)^2/lam)  SE")
for delta in [0.3, 0.4, 0.5, 0.6, 0.7]:
    xs = rng.integers(X, 2 * X, size=4000)
    L = (xs.astype(float) ** delta).astype(np.int64)
    lo = np.searchsorted(centres, xs)
    hi = np.searchsorted(centres, xs + L)
    cnt = hi - lo
    lam = 1.3203 * L / np.log(xs.astype(float)) ** 2
    d = (cnt - lam) ** 2 / lam
    idx, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    print(f"{delta:.1f}    {len(cnt)}    {cnt.mean():9.2f}  mean(cnt/lam) {np.mean(cnt/lam):.3f}    {idx:6.3f}   {se:.3f}")
