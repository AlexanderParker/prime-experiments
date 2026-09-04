"""null_L_fast_r30.py -- the i.i.d. longest-legal-run expectation for HUGE N.

The matrix-power route in null_L_r30.py loses lambda^N to rounding once
N * eps ~ 1 (m43: 3.5e14 steps, m47: 1.6e16 steps).  Here P(no legal run of
length >= k in N steps) is taken as exp(-N / E[T_k]) with E[T_k] the mean
waiting time (mpmath, 50 digits) for the first legal k-run from the empty
state - the renewal / extreme-value form, whose relative error is O(1/E[T_k])
and is negligible exactly where the matrix power fails.  Both routes are
printed side by side at m29..m41 as the cross-check.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import null_L_r30 as Z


def mean_wait(pc, k, alternate):
    """E[T_k]: expected number of draws until a legal run of length k completes,
    starting from the empty state.  States (ell, c) as in Z, ell = run length,
    c = last nonzero class (0 none, 1 +, 2 -).  First-step analysis: every
    m(ell, c) with ell >= 1 is AFFINE in the four unknowns
    u = (m(0,0), m(1,0), m(1,1), m(1,2)) - a transition either lengthens the run
    (ell+1), restarts it at length 1 (a T3 violation), or breaks it (m(0,0)) -
    so a backward sweep from ell = k-1 yields four linear equations.  O(k),
    mpmath at 60 digits."""
    mp.mp.dps = 60
    p0, pp, pm, po = [mp.mpf(x) for x in pc]
    if k == 1:
        return 1 / (p0 + pp + pm)
    U = 5                                      # affine vectors: [const, m00, m10, m11, m12]
    def const(v):
        return [mp.mpf(v), 0, 0, 0, 0]
    def unit(i):
        e = [mp.mpf(0)] * U; e[i] = mp.mpf(1); return e
    def add(a, b, s=1):
        return [x + s * y for x, y in zip(a, b)]
    def scale(a, s):
        return [x * s for x in a]
    m00 = unit(1); m1 = {0: unit(2), 1: unit(3), 2: unit(4)}
    zero = const(0)
    # m(k, .) = 0 (run complete)
    nxt = {c: zero for c in (0, 1, 2)}         # m(ell+1, c) for ell = k-1
    cur = None
    for ell in range(k - 1, 0, -1):
        cur = {}
        for c in (0, 1, 2):
            v = const(1)
            v = add(v, scale(m00, po))
            v = add(v, scale(nxt[c], p0))
            for cls, p in ((1, pp), (2, pm)):
                if alternate and c == cls:
                    v = add(v, scale(m1[cls], p))
                else:
                    v = add(v, scale(nxt[cls], p))
            cur[c] = v
        nxt = cur
    # nxt == m(1, c) as affine in u; plus the equation for m(0,0):
    # m00 = 1 + po*m00 + p0*m10 + pp*m11 + pm*m12
    rows = []; rhs = []
    for c in (0, 1, 2):
        v = nxt[c]                              # m1[c] = v(u)  ->  v - e_{m1c} = 0
        r = [v[1], v[2], v[3], v[4]]; r[1 + c] -= 1
        rows.append(r); rhs.append(-v[0])
    rows.append([po - 1, p0, pp, pm]); rhs.append(mp.mpf(-1))
    A = mp.matrix(rows); b = mp.matrix(rhs)
    x = mp.lu_solve(A, b)
    return x[0]


def expected_longest_fast(pc, N, alternate, kmax=80):
    E = 0.0
    for k in range(1, kmax + 1):
        ET = mean_wait(pc, k, alternate)
        pk = float(-mp.expm1(-mp.mpf(N) / ET))          # P(L >= k)
        E += pk
        if pk < 1e-12:
            break
    return E


def main():
    ys = [int(a) for a in sys.argv[1:]] or [29, 31, 37, 41, 43, 47]
    print("y   q'   N                  method     I-eq     I-eqA   | I-act   I-actA  (actual histogram where on disk)")
    for y in ys:
        q = Z.next_prime(y); u, s = Z.classes(q)
        N = 1
        for g in Z.gears(y):
            N *= (g - 2)
        peq = (1 / q, 1 / q, 1 / q, 1 - 3 / q)
        fe = expected_longest_fast(peq, N, False); fea = expected_longest_fast(peq, N, True)
        line = f"{y:2d}  {q:2d}  {N:17d}  fast(exp)  {fe:6.3f}  {fea:6.3f}  |"
        if y <= 37:
            if y > 23:
                h = Z.hist_from_csv(y)
            else:
                gaps = Z.period_gaps(y)
                vals, cnts = np.unique(gaps, return_counts=True)
                h = {int(v): int(c) for v, c in zip(vals, cnts)}
            pc = [0.0] * 4
            for g, c in h.items():
                pc[Z.class_of(g, q, s)] += c / N
            fa = expected_longest_fast(tuple(pc), N, False); faa = expected_longest_fast(tuple(pc), N, True)
            line += f" {fa:6.3f}  {faa:6.3f}"
        print(line, flush=True)
        if y <= 41:
            me, _ = Z.expected_longest(peq, N, False); mea, _ = Z.expected_longest(peq, N, True)
            print(f"{'':2}  {'':2}  {'':17}  matpow     {me:6.3f}  {mea:6.3f}  |   (float64 matrix power; error ~ N*eps = {N*1.1e-16:.1e})", flush=True)
    print("null_L_fast_r30: done")


if __name__ == "__main__":
    main()
