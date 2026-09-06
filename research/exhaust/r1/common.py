"""Shared machinery for the exhaust branch (R4.b.viii.a).

The stack on base q: tier 1 = primes <= q (period cut_1 = q#), tier 2 = primes in (q, q#]
(period cut_2), tier k+1 = primes in (cut_{k-1}, cut_k].

A tier is a machine of the usual construction: gear g strikes the pair n iff g | n or g | n + 2.
Everything here is exact.
"""

from math import isqrt, prod

import numpy as np


# ------------------------------------------------------------------ prime tables

def prime_mask(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for p in range(2, isqrt(n) + 1):
        if s[p]:
            s[p * p:: p] = False
    return s


def primes_upto(n):
    if n < 2:
        return np.array([], dtype=np.int64)
    return np.flatnonzero(prime_mask(n)).astype(np.int64)


def primes_in(a, b):
    """Primes p with a < p <= b (small b only)."""
    if b < 2:
        return []
    return [int(p) for p in primes_upto(b) if p > a]


def smooth_numbers(q, limit):
    """All n <= limit with every prime factor <= q, sorted; includes 1."""
    out = [1]
    for p in primes_in(1, q):
        new = []
        for v in out:
            w = v * p
            while w <= limit:
                new.append(w)
                w *= p
        out.extend(new)
    out.sort()
    return out


# ------------------------------------------------------------------ the machines

def cuts(q, kmax=2):
    """cut_0 = q, cut_1 = q#, cut_2 = prod of primes in (q, q#], ... (exact, big ints)."""
    c = [q, prod(primes_in(1, q))]
    for k in range(2, kmax + 1):
        c.append(prod(primes_in(c[k - 2], c[k - 1])))
    return c


def wheel_open(gears):
    """Boolean array over one full period W = prod(gears): open[n] iff no gear strikes pair n."""
    W = prod(gears)
    a = np.ones(W, dtype=bool)
    for g in gears:
        a[0:: g] = False
        a[(g - 2) % g:: g] = False
    return a


def range_open(gears, N):
    """Boolean array over [0, N]: open[n] iff no gear strikes pair n = (n, n+2)."""
    a = np.ones(N + 1, dtype=bool)
    for g in gears:
        g = int(g)
        a[0:: g] = False
        st = (g - 2) % g
        a[st:: g] = False
    return a


def admissible_range(gears, N):
    """Boolean array over [0, N]: a[n] iff no gear divides n (single-number view)."""
    a = np.ones(N + 1, dtype=bool)
    a[0] = False
    for g in gears:
        g = int(g)
        a[g:: g] = False
    return a


# ------------------------------------------------------------------ run statistics

def runs_of_true(a):
    """Lengths and start positions of maximal runs of True."""
    d = np.diff(np.concatenate(([0], a.view(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return starts, ends - starts


def longest_false_run(a):
    """(length, start) of the longest maximal run of False."""
    idx = np.flatnonzero(a)
    if idx.size == 0:
        return len(a), 0
    best, at = int(idx[0]), 0
    if idx.size > 1:
        d = np.diff(idx) - 1
        j = int(np.argmax(d))
        if int(d[j]) > best:
            best, at = int(d[j]), int(idx[j]) + 1
    tail = len(a) - 1 - int(idx[-1])
    if tail > best:
        best, at = tail, int(idx[-1]) + 1
    return best, at


def gap_census(a):
    """Counts of distances between consecutive open positions (cyclic not assumed)."""
    idx = np.flatnonzero(a)
    d = np.diff(idx)
    vals, cnt = np.unique(d, return_counts=True)
    return dict(zip(vals.tolist(), cnt.tolist()))
