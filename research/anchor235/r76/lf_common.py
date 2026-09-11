"""lf_common.py -- shared constructions for the length-face lane (research/proof/length_face.md).

Objects, by construction:
  column k = the pair of numbers (6k - 1, 6k + 1);
  gear g (a prime >= 5) strikes column k iff g divides 6k - 1 or 6k + 1, i.e. k = +-u_g (mod g)
    with u_g = 6^{-1} mod g;
  engine {5..p} = the gears 5 <= g <= p;
  a column is OPEN under the engine if no gear strikes it;
  a RUN of length L at x = the columns x .. x+L-1 all struck;
  F(p) = the largest distance between consecutive open columns = (longest run) + 1.
"""
import numpy as np

PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]


def gears_of(p):
    return [g for g in PRIMES if g <= p]


def u_of(g):
    return pow(6, -1, g)


def period(p):
    P = 1
    for g in gears_of(p):
        P *= g
    return P


def struck_segment(p, start, n):
    """bool array over the columns start .. start+n-1: struck by some gear of {5..p}?"""
    arr = np.zeros(n, dtype=bool)
    for g in gears_of(p):
        u = u_of(g)
        for t in (u % g, (-u) % g):
            i0 = (t - start) % g
            arr[i0::g] = True
    return arr


def strikers_segment(p, start, n):
    """int8 array: the number of gears striking each column (multiplicity)."""
    arr = np.zeros(n, dtype=np.int8)
    for g in gears_of(p):
        u = u_of(g)
        for t in (u % g, (-u) % g):
            i0 = (t - start) % g
            arr[i0::g] += 1
    return arr


def runs_from_struck(struck, start):
    """Maximal runs of struck columns inside the segment (interior runs only: those that begin
    after the first open column and end before the last). Returns list of (x, L)."""
    op = np.flatnonzero(~struck)
    if op.size < 2:
        return []
    lens = op[1:] - op[:-1] - 1
    starts = op[:-1] + 1 + start
    keep = lens > 0
    return list(zip(starts[keep].tolist(), lens[keep].tolist()))


def primes_upto(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return np.flatnonzero(s)
