"""E5 (origin structure): for consecutive primes q < q', the columns of the window of q' that machine
q leaves unstruck are exactly the twin columns of the window, plus the column (q'^2 - 1)/6 when
q'^2 - 2 is prime. Hence the next gear q' fills at most one hole of machine q inside the window of q'.
Verification for consecutive prime pairs up to q' = 400 (exact sieve).
"""
import numpy as np
LIM = 400 * 400 + 10
sieve = np.ones(LIM, dtype=bool); sieve[:2] = False
for i in range(2, int(LIM ** 0.5) + 1):
    if sieve[i]: sieve[i*i::i] = False
primes = [p for p in range(5, 401) if sieve[p]]
spf = np.zeros(LIM, dtype=np.int64)
for p in range(2, LIM):
    if sieve[p]:
        spf[p::p][spf[p::p] == 0] = p
bad = 0; squares_seen = 0; total_extra = 0
for q, q2 in zip(primes[:-1], primes[1:]):
    a = (q2 + 7) // 6; b = (q2 * q2 - 1) // 6
    n = np.arange(a, b + 1)
    lo = 6 * n - 1; hi = 6 * n + 1
    unstruck_q = (spf[lo] > q) & (spf[hi] > q)      # no gear <= q divides a member
    twin = sieve[lo] & sieve[hi]
    extra = unstruck_q & ~twin
    extra_cols = n[extra]
    sq_col = (q2 * q2 - 1) // 6
    expected = [sq_col] if sieve[q2 * q2 - 2] else []
    if list(extra_cols) != expected: bad += 1
    squares_seen += len(expected); total_extra += len(extra_cols)
print(f"consecutive pairs checked: {len(primes)-1}; violations: {bad}; square columns present: {squares_seen}; extra holes total: {total_extra}")
