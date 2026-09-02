import sys; sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), ".."))
from math import isqrt
import numpy as np, bisect
from word_tree_r29 import spf_sieve
def primes_upto(n):
    s = bytearray([1]) * (n + 1); s[0:2] = b"\0\0"
    for i in range(2, isqrt(n) + 1):
        if s[i]: s[i*i::i] = bytearray(len(s[i*i::i]))
    return [p for p in range(7, n + 1) if s[p]]
U = {}
def teeth(q):
    if q not in U:
        u = pow(6, -1, q); U[q] = (u, q - u)
    return U[q]
def next_open_slot(k0):
    """walk slots k > k0: anchor-open (k mod 5 in {0,2,3}) and k mod q not in {+-u} for every gear q <= sqrt(6k+1)."""
    k = k0 + 1; killer = {}
    while True:
        if k % 5 in (0, 2, 3):
            for q in primes_upto(isqrt(6 * k + 1)):
                if k % q in teeth(q):
                    killer[q] = killer.get(q, 0) + 1; break
            else:
                return k, killer
        k += 1
def is_prime(n): return n > 1 and all(n % p for p in range(2, isqrt(n) + 1))
P = primes_upto(200000)
print("walk from q^2 by residue rule only (teeth +-u mod q of every gear <= sqrt):")
for q in (37, 97, 499, 997, 4999, 10007, 100003):
    k0 = q * q // 6
    k, killer = next_open_slot(k0)
    qn = P[bisect.bisect_right(P, q)]
    W = (qn * qn - q * q) // 6
    top = sorted(killer.items(), key=lambda kv: -kv[1])[:4]
    print(f"  q={q:>6}: first open slot k={k} -> twin {6*k-1} | {6*k+1}; slots walked {k-k0}; section {W} slots; position {(k-k0)/W:.3f} of section; "
          f"killed by {top}; both prime: {is_prime(6*k-1) and is_prime(6*k+1)}")
