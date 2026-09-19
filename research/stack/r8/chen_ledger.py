"""The Chen ledger of a stretch: T(s) = |R_x(s)| - N_semi(s) at x = s/3.

A column of the stretch of s whose members both have no prime factor <= s/3 is either a rung (both
members prime) or has a member that is a product of two primes in (s/3, 3s+6) (a member below
(s+1)^2 with all prime factors > s/3 has at most two of them). Pre-registered (R5.f.xxx):
(1) the decomposition is exact (every non-rung x-rough column has such a semiprime member);
(2) N_semi / |R_x| stays below 0.4 for every twin centre 100 <= s <= 5000 (Buchstab heuristic
gives about 0.2-0.3); (3) the ratio |R_x| / T(s) has mean near 1.25 and never exceeds 2 above s = 500.
"""
import numpy as np, sys, time
S_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
t0 = time.time()
N = (S_MAX + 1) ** 2 + 2
spf = np.zeros(N, dtype=np.int32)
for p in range(2, int(N ** 0.5) + 1):
    if spf[p] == 0:
        blk = spf[p * p::p]
        mask = blk == 0
        blk[mask] = p
        spf[p * p::p] = blk
idx = np.nonzero(spf == 0)[0]
spf[idx] = idx  # primes (and 0, 1)
spf[0] = spf[1] = 1
print(f"spf sieve to {N:,} in {time.time()-t0:.0f}s")
def is_prime(n): return n >= 2 and spf[n] == n
tw = [s for s in range(6, S_MAX + 1, 6) if is_prime(s - 1) and is_prime(s + 1)]
rows = []
for s in tw:
    if s < 100: continue
    c = s // 6; x = s // 3
    js = np.arange(-(2 * c - 1), 2 * c)
    lo = s * s + 6 * js - 1; hi = lo + 2
    rough = (spf[lo] > x) & (spf[hi] > x)
    R = int(rough.sum())
    T = int(((spf[lo] == lo) & (spf[hi] == hi)).sum())
    # exactness: every rough non-rung column has a member m = p*q with p = spf[m] > x and q = m/p prime
    ok = True
    for a, b in zip(lo[rough], hi[rough]):
        for m in (int(a), int(b)):
            p = int(spf[m])
            if p == m: continue
            q = m // p
            if not (q > x and is_prime(q) and spf[m] > x): ok = False
    rows.append((s, R, T, R - T, ok))
print(f"twin centres in [100, {S_MAX}]: {len(rows)}; decomposition exact at all: {all(r[4] for r in rows)}")
ratios = [r[3] / r[1] for r in rows]
print(f"N_semi/|R_x|: mean {np.mean(ratios):.3f}, max {max(ratios):.3f} at s = {rows[int(np.argmax(ratios))][0]}, min {min(ratios):.3f}")
rt = [r[1] / r[2] for r in rows if r[2] > 0]
print(f"|R_x|/T: mean {np.mean(rt):.3f}, max {max(rt):.3f} at s = {rows[int(np.argmax(rt))][0]}; rungless: {sum(1 for r in rows if r[2]==0)}")
big = [r for r in rows if r[0] > 500]
print(f"s > 500: N_semi/|R_x| max {max(r[3]/r[1] for r in big):.3f}, |R_x|/T max {max(r[1]/r[2] for r in big):.3f}")
print("s, |R_x|, T, N_semi for the last five:", rows[-5:])
