"""Sanity check for proofs/Polignac.lean statements before formalising.

1. survivorGap_iff_pair: in the window (y, y^2 - 2d], "no prime q <= y divides
   m or m + 2d" is exactly "m and m + 2d are both prime".
2. goldbach_of_survivor: for even N, an n with sqrt(N) < n, sqrt(N) < N - n and
   both parts free of prime factors <= sqrt(N) gives a Goldbach representation;
   conversely every representation with both parts above sqrt(N) is such an n.
3. slot_cap_gap: an odd prime q divides both m and m + 2d only when q | d.
"""

from sympy import isprime, primerange
from math import isqrt

def survivor_gap(d, y, m):
    return all(m % q != 0 and (m + 2 * d) % q != 0 for q in primerange(2, y + 1))

fails = 0
for d in [0, 1, 2, 3, 5, 6]:
    for y in [13, 23, 47]:
        for m in range(y + 1, y * y - 2 * d + 1):
            lhs = survivor_gap(d, y, m)
            rhs = isprime(m) and isprime(m + 2 * d)
            if lhs != rhs:
                fails += 1
                print("IFF FAIL", d, y, m)
print("windowed iff: d in {0,1,2,3,5,6}, y in {13,23,47}:", "OK" if fails == 0 else f"{fails} FAILS")

# Goldbach frame
gfails = 0
for N in range(6, 2000, 2):
    s = isqrt(N)
    for n in range(s + 1, N - s):
        lhs = all(n % q != 0 and (N - n) % q != 0 for q in primerange(2, s + 1))
        rhs = isprime(n) and isprime(N - n)
        if lhs != rhs:
            gfails += 1
print("goldbach frame N < 2000:", "OK" if gfails == 0 else f"{gfails} FAILS")

# slot cap
sfails = 0
for d in range(0, 20):
    for q in primerange(3, 100):
        hit = any(m % q == 0 and (m + 2 * d) % q == 0 for m in range(1, q * q))
        if hit != (d % q == 0):
            sfails += 1
print("slot cap:", "OK" if sfails == 0 else f"{sfails} FAILS")
