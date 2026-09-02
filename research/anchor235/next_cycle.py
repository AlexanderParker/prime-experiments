from math import isqrt
OPEN30 = {1, 11, 13, 17, 19, 29}
def mset(q): return [m for m in range(1, 30) if (q * m) % 30 in OPEN30]
def forb(q): return {((q * m - 11) // 30) % q for m in mset(q)}
def primes_upto(n):
    s = bytearray([1]) * (n + 1); s[0:2] = b"\0\0"
    for i in range(2, isqrt(n) + 1):
        if s[i]: s[i*i::i] = bytearray(len(s[i*i::i]))
    return [p for p in range(7, n + 1) if s[p]]
FORB = {}
def next_open(j0, trace=False):
    j = j0 + 1; tested = 0; killer = {}
    while True:
        Q = isqrt(30 * j + 31)                  # gears needed at this cycle
        gs = primes_upto(Q)
        for q in gs:
            if q not in FORB: FORB[q] = forb(q)
            if (j % q) in FORB[q]:
                killer[q] = killer.get(q, 0) + 1; break
        else:
            return j, tested, killer
        tested += 1; j += 1
def is_prime(n): return n > 1 and all(n % p for p in range(2, isqrt(n) + 1))
for start in (0, 601, 3261, 1_000_000):
    j, tested, killer = next_open(start)
    nums = [30 * j + r for r in (11, 13, 17, 19, 29, 31)]
    top = sorted(killer.items(), key=lambda kv: -kv[1])[:6]
    print(f"after cycle {start}: next open cycle j = {j} (numbers {nums[0]}..{nums[-1]}), gears needed up to {isqrt(30*j+31)}; "
          f"cycles walked {j - start}; rejected by gear: {top}; all six prime: {all(is_prime(n) for n in nums)}")
print()
import bisect
P = primes_upto(200000)
for q in (37, 97, 499, 997, 4999, 10007, 100003):
    j0 = (q * q - 11) // 30            # cycle holding q^2 (gear q just switched on)
    j, tested, killer = next_open(j0)
    n0 = 30 * j + 11
    qn = P[bisect.bisect_right(P, q)]  # next prime after q
    secs = bisect.bisect_right(P, isqrt(n0)) - bisect.bisect_right(P, q)  # how many rungs past q the landing lies
    print(f"gear q={q:>6}: start at cycle {j0} (q^2={q*q}); next open cycle j={j} -> numbers {n0}..{n0+20}; "
          f"cycles walked {j-j0}; gears needed up to {isqrt(30*j+31)}; lands {secs} rung(s) past q (section of q ends at {qn}^2={qn*qn})")
