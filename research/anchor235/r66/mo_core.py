"""mo_core.py -- shared helpers for node 4.i.b.ii (the monotone functional of the merge closure).

Everything here is exact integer arithmetic.  The closure step itself is NOT reimplemented: it is
imported from research/anchor235/r61/lc_core.py (the instrument of ladder_closure.md), so that the
operator T_{q'} tested in this branch is literally the one that computed F(37) = 88 and
F(41) = 91.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R61 = os.path.abspath(os.path.join(HERE, "..", "r61"))
if R61 not in sys.path:
    sys.path.insert(0, R61)

from lc_core import (PRIMES, base_gaps, dict_from_gaps, closure_step, group_windows,  # noqa: E402
                     letters_of, word_stats, u_of)

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

CORPUS_F = {5: 2, 7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88, 41: 91,
            43: 103, 47: 118, 53: 145, 59: 161}
# alignment-rules.md 3.7, reproduced in ladder_closure.md as gates
CORPUS_FJ = {13: [11, 16, 23, 26, 28, 31], 17: [18, 25, 28, 33, 35, 40],
             19: [25, 31, 35, 38, 47, 50], 23: [34, 39, 50, 58, 65, 77],
             29: [43, 55, 65, 70, 85, 90], 31: [58, 68, 85, 90, 92, 97]}
CORPUS_L = {11: 1, 13: 1, 17: 1, 19: 2, 23: 1, 29: 3, 31: 3, 37: 2, 41: 2, 43: 2, 47: 4, 53: 3}


def next_gear(y):
    return PRIMES[PRIMES.index(y) + 1]


def a_letter_floor(q):
    d = (2 * u_of(q)) % q
    return min(d, q - d)


def fj_from_period(gaps, jmax):
    """F_J = the widest stretch of the machine carrying J - 1 openings, i.e. the largest sum of J
    consecutive (cyclic) gaps.  Exact, for J = 1..jmax."""
    n = gaps.size
    g = np.concatenate([gaps.astype(np.int64), gaps.astype(np.int64)])
    pre = np.concatenate([[0], np.cumsum(g)])
    out = []
    for J in range(1, jmax + 1):
        if J > n:
            out.append(None)
            continue
        out.append(int((pre[J:J + n] - pre[:n]).max()))
    return out


def fj_from_dict(win, mult, jmax):
    """F_J off a dictionary of realised windows (rows may be short: a zero terminates)."""
    n, K = win.shape
    out = []
    for J in range(1, jmax + 1):
        if J > K:
            out.append(None)
            continue
        good = win[:, J - 1] != 0
        out.append(int(win[good, :J].astype(np.int64).sum(axis=1).max()) if good.any() else None)
    return out


def spectrum_of(gaps):
    return np.bincount(gaps.astype(np.int64))


def tail_excess(spec, x):
    """E_x = sum over gaps of (g - x)_+, per period; and the count of gaps >= x."""
    v = np.arange(spec.size)
    w = np.maximum(v - x, 0)
    return int((spec * w).sum()), int(spec[v >= x].sum())


def summarise_spectrum(spec):
    vals = np.flatnonzero(spec)
    return {"F": int(vals.max()), "n_values": int(vals.size),
            "absent": [int(v) for v in range(1, int(vals.max()) + 1) if spec[v] == 0],
            "top6": [int(v) for v in vals[-6:]],
            "N": int(spec.sum()), "P": int((spec * np.arange(spec.size)).sum())}
