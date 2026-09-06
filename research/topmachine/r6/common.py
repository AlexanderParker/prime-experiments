"""Shared machinery for the zone of tranquillity (branch R4.b.vii).

The machine: gears G = {primes p : q < p <= Q}, pair coordinates.  A single integer n is
ADMISSIBLE iff no gear divides it (no prime factor in (q, Q]).  The pair n is open iff n and
n + 2 are both admissible.

Two independent constructions of the admissible set are provided and are meant to be compared:

  admissible_by_sieve   strike every multiple of every gear      (the machine itself)
  admissible_by_rule    generate s * P, s q-smooth, P = 1 or prime > Q   (the zone rule)

Everything is exact over the stated range.
"""

from math import isqrt

import numpy as np


# ----------------------------------------------------------------- basic tables

def prime_mask(n):
    """Boolean array of length n + 1, True at primes."""
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


def smooth_numbers(q, limit):
    """All n <= limit with every prime factor <= q, sorted.  Includes 1."""
    out = [1]
    for p in [int(p) for p in primes_upto(q)]:
        new = []
        for v in out:
            w = v * p
            while w <= limit:
                new.append(w)
                w *= p
        out.extend(new)
    out.sort()
    return out


def smooth_part(n, q):
    """The q-smooth part of n."""
    s = 1
    for p in [int(p) for p in primes_upto(q)]:
        while n % p == 0:
            n //= p
            s *= p
    return s


# ------------------------------------------------------- the machine, by sieving

def admissible_by_sieve(q, Q, X, pr=None):
    """Boolean array a[0..X]: a[n] iff no prime in (q, Q] divides n.

    a[0] is set False by hand (0 is divisible by every gear)."""
    if pr is None:
        pr = primes_upto(Q)
    a = np.ones(X + 1, dtype=bool)
    a[0] = False
    for g in pr:
        g = int(g)
        if g <= q:
            continue
        if g > Q:
            break
        a[g:: g] = False
    return a


# ---------------------------------------------------------- the zone rule, direct

def admissible_by_rule(q, Q, X, primes=None):
    """Boolean array r[0..X] built from the rule  n = s * P,  s q-smooth,  P = 1 or prime > Q.

    Correct (as a description of the admissible set) exactly on n <= Q^2; above Q^2 it
    UNDER-counts, because a product of two primes above Q is admissible but not of this form."""
    if primes is None:
        primes = primes_upto(X)
    r = np.zeros(X + 1, dtype=bool)
    sm = smooth_numbers(q, X)
    r[np.asarray(sm, dtype=np.int64)] = True          # P = 1
    r[0] = False
    big = primes[primes > Q]
    for s in sm:
        if s * (big[0] if len(big) else X + 1) > X:
            continue
        sel = big[big <= X // s]
        if len(sel):
            r[s * sel] = True
    return r


def open_pairs(adm):
    """Boolean array o[0..X-2]: the pair n is open."""
    return adm[:-2] & adm[2:]


# ------------------------------------------------------------------ gap analysis

def gap_report(openmask, lo, hi):
    """Gaps between consecutive open pairs whose BLOCK lies in [lo, hi).

    A block is the maximal run of struck pairs between two consecutive open pairs.
    Returns (positions, lengths) with position = first struck pair of the block,
    length = number of struck pairs.  Only blocks whose first struck pair is in
    [lo, hi) are reported."""
    idx = np.flatnonzero(openmask[lo:hi]).astype(np.int64) + lo
    if len(idx) < 2:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    d = np.diff(idx) - 1
    keep = d > 0
    return idx[:-1][keep] + 1, d[keep]


def largest_prime_gap(primes, lo, hi):
    """(gap, position) for consecutive primes p < p' with lo < p and p' <= hi."""
    sel = primes[(primes > lo) & (primes <= hi)]
    if len(sel) < 2:
        return 0, 0
    d = np.diff(sel)
    i = int(np.argmax(d))
    return int(d[i]), int(sel[i])
