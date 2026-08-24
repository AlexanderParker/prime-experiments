"""Exact rational LP core (Fractions) - shared by research/lp_dual_certs.py.

Two-phase tableau simplex with Bland's rule (guaranteed termination, no
cycling, no floating point anywhere).  Everything here is EXACT: inputs are
Fractions/ints, outputs are Fractions.

    solve_std(A, b, c)      max c.x  s.t.  A x = b, x >= 0, b >= 0
                            -> (status, value, x, y)
                            status in {'optimal', 'infeasible', 'unbounded'}
                            y = dual vector for the equalities (optimal only)

    feasible_eq(A, b)       is {x >= 0 : A x = b} nonempty?
                            -> (True, x) or (False, y) with y a Farkas
                            certificate: y.b < 0 and y.A <= 0 componentwise
                            (so no x >= 0 can satisfy A x = b).

House rule: floats are never used to DECIDE anything here.
"""
from fractions import Fraction

ZERO = Fraction(0)
ONE = Fraction(1)


def _pivot(T, basis, r, c):
    piv = T[r][c]
    inv = ONE / piv
    row = T[r]
    if piv != ONE:
        T[r] = [v * inv for v in row]
        row = T[r]
    for i, ri in enumerate(T):
        if i == r:
            continue
        f = ri[c]
        if f:
            T[i] = [a - f * b for a, b in zip(ri, row)]
    basis[r] = c


def _simplex_phase(T, basis, ncols):
    """T has one objective row at the end (row m), objective already in
    canonical form w.r.t. `basis`.  Maximise.  Bland's rule."""
    m = len(T) - 1
    it = 0
    stall, last = 0, None
    while True:
        it += 1
        obj = T[m]
        # entering: Dantzig (largest reduced cost) until progress stalls,
        # then Bland's rule (smallest index) which cannot cycle.
        enter = -1
        if stall < 60:
            best = ZERO
            for j in range(ncols):
                if obj[j] > best:
                    best, enter = obj[j], j
        else:
            for j in range(ncols):
                if obj[j] > 0:
                    enter = j
                    break
        if enter < 0:
            return 'optimal', it
        cur = obj[-1]
        if last is not None and cur == last:
            stall += 1
        else:
            stall = 0
        last = cur
        # ratio test, Bland tie-break on basis index
        leave, best = -1, None
        for i in range(m):
            a = T[i][enter]
            if a > 0:
                ratio = T[i][-1] / a
                if (best is None or ratio < best
                        or (ratio == best and basis[i] < basis[leave])):
                    best, leave = ratio, i
        if leave < 0:
            return 'unbounded', it
        _pivot(T, basis, leave, enter)


def solve_std(A, b, c):
    """max c.x s.t. A x = b (b >= 0), x >= 0.  Exact."""
    m, n = len(A), len(A[0])
    A = [[Fraction(v) for v in row] for row in A]
    b = [Fraction(v) for v in b]
    c = [Fraction(v) for v in c]
    for i in range(m):
        assert b[i] >= 0, "call with b >= 0 (negate rows first)"
    # ---- phase I: minimise sum of artificials == maximise -sum
    ncols = n + m
    T = [A[i] + [ONE if j == i else ZERO for j in range(m)] + [b[i]]
         for i in range(m)]
    basis = [n + i for i in range(m)]
    obj = [ZERO] * (ncols + 1)
    for i in range(m):                     # -sum artificials, canonicalised
        for j in range(ncols + 1):
            obj[j] += T[i][j]
    for i in range(m):
        obj[n + i] = ZERO
    T.append(obj)
    st, _ = _simplex_phase(T, basis, ncols)
    assert st == 'optimal'
    if T[m][-1] > 0:
        # infeasible; row m holds y.A - (dual) info; rebuild certificate
        y = [ONE + T[m][n + i] for i in range(m)]
        return 'infeasible', None, None, y
    # drive artificials out of the basis
    for i in range(m):
        if basis[i] >= n:
            piv = -1
            for j in range(n):
                if T[i][j] != 0:
                    piv = j
                    break
            if piv >= 0:
                _pivot(T, basis, i, piv)
    # ---- phase II
    T = [row[:n] + [row[-1]] for row in T[:m]]
    obj = [ZERO] * (n + 1)
    for j in range(n):
        obj[j] = c[j]
    for i in range(m):
        if basis[i] < n and obj[basis[i]] != 0:
            f = obj[basis[i]]
            for j in range(n + 1):
                obj[j] -= f * T[i][j]
    T.append(obj)
    st, _ = _simplex_phase(T, basis, n)
    if st == 'unbounded':
        return 'unbounded', None, None, None
    x = [ZERO] * n
    for i in range(m):
        if basis[i] < n:
            x[basis[i]] = T[i][-1]
    val = sum(c[j] * x[j] for j in range(n))
    return 'optimal', val, x, None


def feasible_eq(A, b):
    """Is {x >= 0 : A x = b} nonempty?  Rows with b < 0 are negated first.
    Returns (True, x) or (False, y) with y.b < 0 and (y.A)_j <= 0 for all j
    - an exact Farkas certificate of infeasibility."""
    A = [[Fraction(v) for v in row] for row in A]
    b = [Fraction(v) for v in b]
    m, n = len(A), len(A[0])
    sgn = [1] * m
    for i in range(m):
        if b[i] < 0:
            A[i] = [-v for v in A[i]]
            b[i] = -b[i]
            sgn[i] = -1
    ncols = n + m
    T = [A[i] + [ONE if j == i else ZERO for j in range(m)] + [b[i]]
         for i in range(m)]
    basis = [n + i for i in range(m)]
    obj = [ZERO] * (ncols + 1)
    for i in range(m):
        for j in range(ncols + 1):
            obj[j] += T[i][j]
    for i in range(m):
        obj[n + i] = ZERO
    T.append(obj)
    st, _ = _simplex_phase(T, basis, ncols)
    assert st == 'optimal'
    if T[m][-1] == 0:
        x = [ZERO] * n
        for i in range(m):
            if basis[i] < n:
                x[basis[i]] = T[i][-1]
        return True, x
    # y from the artificial columns of the final objective row:
    #   objective row = sum_i (row_i) - (canonicalisation), and the reduced
    #   costs of the artificials are 1 - y_i.  Extract y and repair signs.
    y = [(ONE + T[m][n + i]) * sgn[i] for i in range(m)]
    return False, y
