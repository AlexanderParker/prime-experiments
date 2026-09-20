"""Shared objects for the range-analysis lane.

Columns n >= 1 hold (6n-1, 6n+1).  Gear g (prime >= 5) strikes column n iff
n = +-c_g (mod g), c_g = 6^{-1} mod g.  Tooth distance d_g = 3^{-1} mod g = 2 c_g mod g.
Machine q = gears 5..q, period P = product of gears.  Openings = unstruck columns.
"""
from math import isqrt, prod
import numpy as np


def primes_upto(N):
    s = np.ones(N + 1, dtype=bool)
    s[:2] = False
    for i in range(2, isqrt(N) + 1):
        if s[i]:
            s[i * i::i] = False
    return [int(p) for p in np.nonzero(s)[0]]


def gears(q):
    return [p for p in primes_upto(q) if p >= 5]


def period(q):
    return prod(gears(q))


def c(g):
    return pow(6, -1, g)


def d(g):
    return pow(3, -1, g)


def struck_mask(gs, N):
    """mask[n] for n in 0..N : True iff some gear of gs strikes column n."""
    m = np.zeros(N + 1, dtype=bool)
    for g in gs:
        cg = c(g)
        m[cg::g] = True
        m[(g - cg) % g::g] = True
    return m


def openings(q):
    """Openings of machine q in columns 1..P (sorted numpy array)."""
    P = period(q)
    m = struck_mask(gears(q), P)
    return np.nonzero(~m[1:])[0] + 1


def record(q):
    """F(q): longest run of consecutive struck columns (cyclic over one period)."""
    P = period(q)
    m = struck_mask(gears(q), P)[1:]           # columns 1..P
    mm = np.concatenate([m, m])
    best = run = 0
    # vectorised run lengths
    x = mm.astype(np.int8)
    # positions where runs start / end
    dx = np.diff(np.concatenate([[0], x, [0]]))
    starts = np.nonzero(dx == 1)[0]
    ends = np.nonzero(dx == -1)[0]
    if len(starts):
        best = int((ends - starts).max())
    return min(best, P)


def T_primes(q):
    P = period(q)
    return [p for p in primes_upto(isqrt(6 * P + 1)) if p > q]


def strikes(g, cols):
    """boolean mask: which columns of the array cols does gear g strike (residue sense)."""
    r = cols % g
    cg = c(g)
    return (r == cg) | (r == (g - cg) % g)


def factor(n):
    """prime factorisation (list with multiplicity) by trial division; n small."""
    out = []
    p = 2
    while p * p <= n:
        while n % p == 0:
            out.append(p)
            n //= p
        p += 1
    if n > 1:
        out.append(n)
    return out
