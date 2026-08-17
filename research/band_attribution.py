"""Band-by-band attribution of twin-candidate deaths.

Bands: B_h = (p_h^2, p_{h+1}^2] over consecutive primes. A candidate pair
(6k-1, 6k+1) lives in the band containing its midpoint 6k. Each composite
member is attributed to its root gear q = lpf(m) (unique, law L2 of
twin-prime-program.md), and q lives in its own band. Cofactor c = m/q prime
means the block is a fresh semiprime q*r - the machine re-entering its own
prime output; band(c) records where that prime was made.

Also computes the twin-deciding slice: pairs with exactly one composite
member ("one-away" pairs) and the band of the lone killer.

Findings at 6e6 (see docs/band-attribution.md): the matrix is the square-root
tower made visible; gears {5,7} carry 39.6%% of all kills; 50.5%% of all
blocks are fresh semiprime re-entry; and at the one-away margin, killers
from band <=13^2 upward act almost exclusively through semiprimes.
"""
import numpy as np
import bisect

LIMIT = 6 * 10**6 + 8

def spf_sieve(n):
    spf = np.zeros(n + 1, dtype=np.int64)
    for p in range(2, int(n**0.5) + 1):
        if spf[p] == 0:
            m = spf[p*p::p]
            m[m == 0] = p
            spf[p*p::p] = m
    return spf

spf = spf_sieve(LIMIT)

def is_prime(m):
    return m > 1 and spf[m] == 0

primes = [p for p in range(2, 4000) if all(p % d for d in range(2, int(p**0.5) + 1))]
sq = [p * p for p in primes]

def band(x):
    """index i: x in (primes[i-1]^2, primes[i]^2]"""
    return bisect.bisect_left(sq, x)

NB = band(LIMIT - 8)
labels = [f"<={primes[i]}^2" for i in range(NB + 1)]

cand = np.zeros(NB + 2, dtype=np.int64)
twin = np.zeros(NB + 2, dtype=np.int64)
dead = np.zeros(NB + 2, dtype=np.int64)
M = np.zeros((NB + 2, NB + 2), dtype=np.int64)   # death band x gear band
F = np.zeros((NB + 2, NB + 2), dtype=np.int64)   # fresh kills: gear band x cofactor band
semi = 0
deep = 0
lone = np.zeros(NB + 2, dtype=np.int64)          # one-away pairs: lone-killer band
lone_semi = np.zeros(NB + 2, dtype=np.int64)

for k in range(1, (LIMIT - 8) // 6 + 1):
    mid = 6 * k
    h = band(mid)
    cand[h] += 1
    a, b = mid - 1, mid + 1
    pa, pb = is_prime(a), is_prime(b)
    if pa and pb:
        twin[h] += 1
        continue
    dead[h] += 1
    if pa != pb:
        m = b if pa else a
        q = int(spf[m]); j = band(q)
        lone[j] += 1
        if is_prime(m // q):
            lone_semi[j] += 1
    for m in (a, b):
        if not is_prime(m):
            q = int(spf[m]); j = band(q)
            M[h][j] += 1
            c = m // q
            if is_prime(c):
                semi += 1
                F[j][band(c)] += 1
            else:
                deep += 1

show = min(band(2809), NB)
print("attribution matrix (rows: pair's band, cols: root gear's band):\n")
print("band(mid)      cand twin dead | " + " ".join(f"{labels[j]:>8}" for j in range(1, show + 1)))
for h in range(1, show + 1):
    row = " ".join(f"{M[h][j]:8d}" for j in range(1, show + 1))
    print(f"{labels[h]:>12} {cand[h]:6d} {twin[h]:4d} {dead[h]:4d} | {row}")

tot = M.sum()
print(f"\npairs {cand.sum()}, twins {twin.sum()}, dead {dead.sum()}")
print(f"members {tot}: fresh semiprime {semi} ({100*semi/tot:.1f}%), deeper {deep}")

gear_share = M.sum(axis=0)
print("\nkills by gear band:")
for j in np.argsort(gear_share)[::-1][:10]:
    if gear_share[j]:
        print(f"  {labels[j]:>8}: {gear_share[j]:8d} = {100*gear_share[j]/tot:.2f}%")

lt = lone.sum()
print(f"\none-away pairs: {lt}; lone-killer band | share | semiprime member")
for j in range(1, NB + 1):
    if lone[j]:
        print(f"  {labels[j]:>8} {lone[j]:8d} {100*lone[j]/lt:6.2f}%  {lone_semi[j]:8d}")
