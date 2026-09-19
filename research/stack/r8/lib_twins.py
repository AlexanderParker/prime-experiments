"""Shared helpers for the routes lane: exact twin-centre sieves.

Twin centre: s = 6c with s-1 and s+1 both prime.  All sieves are exact (no sampling).
"""
import numpy as np


def primes_upto(n):
    """All primes <= n (numpy int64), simple sieve."""
    n = int(n)
    if n < 2:
        return np.zeros(0, dtype=np.int64)
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(n ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = False
    return np.flatnonzero(sieve).astype(np.int64)


def twin_centres_upto(n):
    """All twin centres s <= n, exact.  Sieves only the numbers 6k-1, 6k+1 (k = 1..n/6),
    so memory is two bool arrays of n/6 entries.  Returns sorted int64 array of s."""
    n = int(n)
    K = n // 6
    if K < 1:
        return np.zeros(0, dtype=np.int64)
    lo = np.ones(K + 1, dtype=bool)   # lo[k]: 6k-1 prime (k >= 1)
    hi = np.ones(K + 1, dtype=bool)   # hi[k]: 6k+1 prime
    lo[0] = hi[0] = False
    ps = primes_upto(int((6 * K + 1) ** 0.5) + 1)
    for p in ps:
        p = int(p)
        if p < 5:
            continue
        inv6 = pow(6, -1, p)
        # 6k-1 = 0 mod p  <=>  k = inv6 mod p ; 6k+1 = 0 <=> k = -inv6 mod p
        k1 = inv6 % p
        k2 = (-inv6) % p
        # do not strike p itself (k with 6k-1 = p or 6k+1 = p)
        start1 = k1 if 6 * k1 - 1 != p else k1 + p
        start2 = k2 if 6 * k2 + 1 != p else k2 + p
        if start1 == 0:
            start1 = p
        if start2 == 0:
            start2 = p
        lo[start1::p] = False
        hi[start2::p] = False
    both = lo & hi
    ks = np.flatnonzero(both)
    return (6 * ks).astype(np.int64)


def count_in_windows(sorted_arr, los, his):
    """Number of elements of sorted_arr strictly inside (lo, hi) for each pair (vectorised)."""
    los = np.asarray(los, dtype=np.int64)
    his = np.asarray(his, dtype=np.int64)
    a = np.searchsorted(sorted_arr, los, side="right")
    b = np.searchsorted(sorted_arr, his, side="left")
    return b - a
