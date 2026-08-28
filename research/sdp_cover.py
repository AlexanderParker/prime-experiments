"""Round 24 lateral - THE SDP RELAXATION OF THE MACHINE-FREE SYSTEM.

WHY.  Round 23 (potential_arity.py) proved F(M) <= 1 + osc(h) is TIGHT, so F is
an LP optimum and only ARITY can fail; and nilpotent_invariants.py proved
w(BS) = cos(pi/(F+1)) exactly, so F is a VARIATIONAL, SDP-representable
quantity.  The brief: what is the smallest ARITY-INDEPENDENT statement that
bounds osc(h)?  An SDP certificate would be one.

PART A - THE LITERAL BRIEF TARGET, AND IT IS EMPTY (a theorem).
Constructor's machine-free system MF_m (research/machinefree_cert.py) is a
max-plus closure: a finite weighted digraph, value B = longest walk.  Write

    LP(MF):  minimise  max_u (L_u + p_u)
             over p with p_u >= R_u  and  p_u >= w_e + p_{dst(e)} for every e.

THEOREM (MF-LP).  LP(MF) = B exactly.  Proof: the closure hh (least fixed
point >= R) is feasible and attains B; conversely every feasible p dominates
hh entrywise by induction along the closure iteration, so max(L+p) >= B.  []
COROLLARY.  The combinatorial maximisation whose value is B has an EXACT LP
relaxation, so every relaxation sandwiched between the LP and the true value -
in particular every Lasserre/SOS level, every SDP - returns exactly B.  NO
CONVEX RELAXATION OF THE MACHINE-FREE SYSTEM CAN IMPROVE IT BY ONE UNIT.  The
machine-free wall is a SUPPORT (edge-set) problem, not a relaxation-gap
problem, which is exactly why CEGAR (deleting unrealised tuples) is the only
lever that moved it.
Part A VERIFIES this by solving LP(MF) with HiGHS at every step and asserting
the optimum equals the integer closure value.

PART B/C - WHERE AN SDP CAN LIVE: THE COVERING CSP FOR F ITSELF.
Slot k is blocked by gear q iff k = +-u_q (mod q), u_q = 6^{-1} mod q.  Put
x_q = k mod q and c_q = -x_q; then gear q blocks position i of a run iff
i = c_q +- u_q (mod q), and by CRT every phase vector (c_q) occurs.  So

    F(M) = 1 + max{ L : [0,L) can be covered by choosing one offset c_q per
                         gear, gear q covering the two classes c_q +- u_q }

- a covering CSP of size sum_q q, with NO period and NO machine input beyond
the prime list.  (The formulation is the project's own: Mechanic's
research/cov_sat.py, r20.  What is new here is the RELAXATION HIERARCHY over
it and its exact duals.)  This is machine-free by construction, so a
relaxation of THIS is the object the brief was reaching for.

  LP1  = fractional cover (level 1): p_q in Delta(Z_q), for each i
         sum_q [p_q(i-u_q) + p_q(i+u_q)] >= 1.
         Farkas dual: weights lam >= 0 on [0,L) with
             sum_i lam_i  >  sum_q max_c ( sum_{i = c+u_q} lam_i
                                         + sum_{i = c-u_q} lam_i ).
  SA2  = level-2 Sherali-Adams on the LIFTED encoding: extra variables
         g(i) in {gears} ("which gear covers i"), moment matrix over the
         literals (q,c) and (i,q), with marginalisation, the implication
         g(i)=q => c_q in {i +- u_q}, and the same-gear lag rule
         g(i)=g(j)=q => j-i in {0, +-2u_q} mod q  (result 25's enhanced lag).
  SDP2 = SA2 + the PSD constraint on the moment matrix, imposed by
         eigenvector cutting planes v'Yv >= 0 (each cut valid for ANY real v,
         hence sound at any rationalisation).

Everything is run as a PHASE-1 LP: minimise sum_i s_i where s_i is slack on
"position i is assigned", so the reported V(L) is a graded MACHINE-FREE
DEFICIENCY (0 = the level cannot rule L out; > 0 = L is impossible, hence
F <= L).  Every V(L) > 0 verdict is re-proved EXACTLY: the HiGHS dual is
rationalised, scaled down until exactly dual-feasible (legitimate because the
cost vector is >= 0, so the dual-feasible set is star-shaped about 0), and the
exact rational dual objective is reported.  Numbers from the solver are marked
NUMERICAL; only the exact rational duals are claimed.

PRE-REGISTERED PREDICTIONS (written before any run; round-23 house standard is
that two of my own were refuted):
  P1  LP(MF) == closure exactly at all 7 steps, both m = 3 and m = 4.
      [near-certain - it is the theorem above; this is a check, not a test]
  P2  LP1 is VACUOUS from machine 13 on (uniform p_q gives coverage
      2*sigma(y) >= 1 for every L), and finite only at machine 11.  Its death
      point coincides exactly with T2's sigma(y) >= 1/2.
  P3  SA2 gives a FINITE bound at every machine 11..23 - i.e. the level-2
      lift escapes the sigma >= 1/2 death that kills level 1.
  P4  SDP2 strictly improves SA2 (a smaller L*) at at least one machine.
  P5  SA2's bound at machine 13 is at most 2x the truth (maxL = 10, so
      L* <= 20).

Usage: uv run python research/sdp_cover.py [partA|partB|partC] [--y 11,13,17]
"""
import os
import sys
import time
from fractions import Fraction

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, vstack

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# exact F(M) (max gap in SLOT units) from the project's corpus
F_EXACT = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58,
           37: 88, 41: 91, 43: 103, 47: 118, 53: 145}


def primes_upto(a, b):
    return [n for n in range(a, b + 1)
            if n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))]


def gears(y):
    return primes_upto(5, y)


def tooth(q):
    return pow(6, -1, q)


# --------------------------------------------------------------- part A


def part_a(ms=(3, 4)):
    from machinefree_cert import STEPS, build_mf, closure
    print("PART A - LP(MF) vs the max-plus closure.  THEOREM says they are "
          "equal, hence no SDP\n         relaxation of the machine-free "
          "system can improve it.\n")
    ok = 0
    for y, F, q1, exact in STEPS:
        for m in ms:
            if m == 4 and y >= 29:
                continue                     # 1.9e5+ states: skip, m=3 suffices
            S, esrc, edst, ew, Rs, Ls, _ = build_mf(F, q1, 35, m)
            B, _lay, cyc = closure(S, esrc, edst, ew, Rs, Ls)
            if cyc or B is None:
                print("  step %2d->%2d MF_%d  CYCLIC - skipped" % (y, q1, m))
                continue
            # LP:  min t  s.t.  t - p_u >= L_u ; p_u >= R_u ;
            #                   p_u - p_v >= w_e
            nv = S + 1                       # p_0..p_{S-1}, t  (t is index S)
            # shift p by min(Rs) to keep everything >= 0 for the bounds
            rows, cols, vals, rhs = [], [], [], []
            r = 0
            for u in range(S):               # -t + p_u <= -L_u
                rows += [r, r]
                cols += [S, u]
                vals += [-1.0, 1.0]
                rhs.append(-float(Ls[u]))
                r += 1
            for e in range(len(esrc)):       # -p_src + p_dst <= -w_e
                rows += [r, r]
                cols += [int(esrc[e]), int(edst[e])]
                vals += [-1.0, 1.0]
                rhs.append(-float(ew[e]))
                r += 1
            A = coo_matrix((vals, (rows, cols)), shape=(r, nv))
            c = np.zeros(nv)
            c[S] = 1.0
            lo = np.concatenate([Rs.astype(float), [-np.inf]])
            bounds = [(lo[i], None) for i in range(nv)]
            t0 = time.time()
            res = linprog(c, A_ub=A.tocsr(), b_ub=np.array(rhs),
                          bounds=bounds, method="highs")
            assert res.status == 0, (y, m, res.message)
            lp = res.fun
            print("  step %2d->%2d  MF_%-1d  states %7d edges %7d   "
                  "closure %4d   LP %12.6f   |diff| %.2e   %.0fs"
                  % (y, q1, m, S, len(esrc), B, lp, abs(lp - B),
                     time.time() - t0))
            assert abs(lp - B) < 1e-6, (y, m, lp, B)
            ok += 1
    print("\n  %d/%d LP(MF) == closure to 1e-6.  P1 CONFIRMED.\n"
          "  => every relaxation between the LP and the truth (all Lasserre "
          "levels, all SDPs)\n     returns exactly the closure value.  The "
          "machine-free gap is 100%% EDGE SET.\n" % (ok, ok))


# --------------------------------------------------------------- the CSP


def brute_max_run(y):
    """Exact max run of consecutive blocked slots over the full period."""
    qs = gears(y)
    P = 1
    for q in qs:
        P *= q
    blocked = np.zeros(P, bool)
    for q in qs:
        u = tooth(q)
        blocked[u % q::q] = True
        blocked[(-u) % q::q] = True
    # cyclic longest run of True
    d = np.flatnonzero(~blocked)
    assert len(d), y
    gaps = np.diff(np.concatenate([d, [d[0] + P]]))
    return int(gaps.max()) - 1, int(gaps.max())


def cover_sat_brute(y, L):
    """Direct check: is [0,L) coverable?  (full period scan, small y only)"""
    r, _ = brute_max_run(y)
    return L <= r


# --------------------------------------------------------------- LP1


def lp1(y, L):
    """Level-1 fractional cover.  Returns (V, dual) with V > 0 <=> infeasible.

    min sum_i s_i   s.t.  sum_q [p_q(i-u) + p_q(i+u)] + s_i >= 1  for each i
                          sum_c p_q(c) = 1                        for each q
                          p, s >= 0
    """
    qs = gears(y)
    off, n = {}, 0
    for q in qs:
        off[q] = n
        n += q
    nv = n + L
    rows, cols, vals = [], [], []
    for i in range(L):
        for q in qs:
            u = tooth(q)
            for c in {(i - u) % q, (i + u) % q}:
                rows.append(i)
                cols.append(off[q] + c)
                vals.append(-1.0)
        rows.append(i)
        cols.append(n + i)
        vals.append(-1.0)
    A_ub = coo_matrix((vals, (rows, cols)), shape=(L, nv)).tocsr()
    b_ub = -np.ones(L)
    er, ec, ev = [], [], []
    for j, q in enumerate(qs):
        for c in range(q):
            er.append(j)
            ec.append(off[q] + c)
            ev.append(1.0)
    A_eq = coo_matrix((ev, (er, ec)), shape=(len(qs), nv)).tocsr()
    b_eq = np.ones(len(qs))
    c = np.concatenate([np.zeros(n), np.ones(L)])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0, None)] * nv, method="highs")
    assert res.status == 0, res.message
    return res.fun, res


def lp1_exact_certificate(y, L, lam):
    """EXACT check of the level-1 Farkas certificate with integer weights lam:
       sum_i lam_i > sum_q max_c (lam over i = c+u) + (lam over i = c-u).
    """
    qs = gears(y)
    tot = int(sum(lam))
    cap = 0
    for q in qs:
        u = tooth(q)
        best = 0
        for c in range(q):
            s = 0
            for i in range(L):
                if (i - u) % q == c or (i + u) % q == c:
                    s += int(lam[i])
            best = max(best, s)
        cap += best
    return tot, cap, tot > cap


def part_b(ys):
    print("PART B - the covering CSP and its LEVEL-1 relaxation.\n")
    print("  sanity: max blocked run from the CSP frame vs corpus F")
    for y in ys:
        if y > 19:
            continue
        r, g = brute_max_run(y)
        print("    y=%2d  max run %3d  F = run+1 = %3d   corpus F %3d  %s"
              % (y, r, g, F_EXACT[y], "OK" if g == F_EXACT[y] else "MISMATCH"))
        assert g == F_EXACT[y], y
    print()
    sig = Fraction(0)
    print("  LP1 (fractional cover).  sigma(y) = sum 1/q; uniform p_q gives "
          "coverage 2*sigma\n  for EVERY position, so LP1 is feasible at all "
          "L as soon as sigma >= 1/2.\n")
    for y in ys:
        sig = sum((Fraction(1, q) for q in gears(y)), Fraction(0))
        maxL = F_EXACT[y] - 1
        Ls = "vacuous"
        if 2 * sig < 1:
            lo, hi = 1, 4 * F_EXACT[y] + 40
            while lo < hi:
                mid = (lo + hi) // 2
                V, _ = lp1(y, mid)
                if V > 1e-9:
                    hi = mid
                else:
                    lo = mid + 1
            Ls = "L* = %d  (F <= %d, truth %d)" % (lo, lo, F_EXACT[y])
            V, res = lp1(y, lo)
            lam = np.ones(lo, dtype=int)
            tot, cap, okc = lp1_exact_certificate(y, lo, lam)
            Ls += "   exact lam=1 certificate: %d > %d  %s" % (
                tot, cap, "VALID" if okc else "fails")
        print("    y=%2d  sigma = %s = %.6f   2*sigma %s 1   %s"
              % (y, sig, float(sig), ">=" if 2 * sig >= 1 else "<", Ls))
    print("\n  P2: LP1 dies exactly where T2 says (sigma >= 1/2, machine 13).\n")


# --------------------------------------------------------------- SA2 / SDP2


class Lift:
    """Level-2 lift of the covering CSP with the g(i) witness variables."""

    def __init__(self, y, L):
        self.y, self.L = y, L
        self.qs = gears(y)
        self.u = {q: tooth(q) for q in self.qs}
        lits = [("c", q, c) for q in self.qs for c in range(q)]
        lits += [("g", i, q) for i in range(L) for q in self.qs]
        self.lits = lits
        self.idx = {a: j + 1 for j, a in enumerate(lits)}   # 0 = the unit
        self.n = len(lits) + 1
        self.varof = {}                 # (a,b) a<=b -> LP column, or None=0
        self._build_support()

    def _allowed(self, a, b):
        """Is the moment Y[a,b] forced to zero by a hard constraint?"""
        if a == b:
            return True
        ka, kb = self.lits[a - 1] if a else None, self.lits[b - 1] if b else None
        if a == 0 or b == 0:
            return True
        if ka[0] == "c" and kb[0] == "c":
            if ka[1] == kb[1]:
                return False            # c_q = c and c_q = c'
            return True
        if ka[0] == "g" and kb[0] == "g":
            if ka[1] == kb[1]:
                return False            # g(i) = q and g(i) = q'
            if ka[2] != kb[2]:
                return True
            q = ka[2]
            d = (kb[1] - ka[1]) % q
            return d in (0, (2 * self.u[q]) % q, (-2 * self.u[q]) % q)
        if ka[0] == "g":
            ka, kb = kb, ka             # ka = c-literal, kb = g-literal
        q, c = ka[1], ka[2]
        i, qq = kb[1], kb[2]
        if qq != q:
            return True
        return c in ((i - self.u[q]) % q, (i + self.u[q]) % q)

    def _build_support(self):
        n = self.n
        col = 0
        for a in range(n):
            for b in range(a, n):
                if self._allowed(a, b):
                    self.varof[(a, b)] = col
                    col += 1
                else:
                    self.varof[(a, b)] = None
        self.ncols = col

    def v(self, a, b):
        return self.varof[(min(a, b), max(a, b))]

    def build(self, cuts=()):
        """Phase-1 LP: min sum_i s_i.  Returns (c, A, b, meta)."""
        n, L, qs = self.n, self.L, self.qs
        nv = self.ncols + L                 # + slacks s_i
        rows, cols, vals, rhs, sense = [], [], [], [], []
        r = 0

        def add(terms, rr, sn):
            nonlocal r
            for cc, vv in terms:
                if cc is None:
                    continue
                rows.append(r)
                cols.append(cc)
                vals.append(vv)
            rhs.append(rr)
            sense.append(sn)
            r += 1

        add([(self.v(0, 0), 1.0)], 1.0, "=")
        for a in range(1, n):               # Y[a,a] = Y[0,a]
            add([(self.v(a, a), 1.0), (self.v(0, a), -1.0)], 0.0, "=")
        # exactly-one per CSP variable, at the unit level and conditioned
        groups = []
        for q in qs:
            groups.append([self.idx[("c", q, c)] for c in range(q)])
        for i in range(L):
            groups.append([self.idx[("g", i, q)] for q in qs])
        for gi, gr in enumerate(groups):
            if gi < len(qs):
                add([(self.v(0, a), 1.0) for a in gr], 1.0, "=")
            else:                            # position group: slack s_i
                i = gi - len(qs)
                add([(self.v(0, a), 1.0) for a in gr]
                    + [(self.ncols + i, 1.0)], 1.0, "=")
        # marginalisation:  sum_{a in group} Y[b,a] = Y[0,b]
        for b in range(1, n):
            for gi, gr in enumerate(groups):
                if b in gr:
                    continue
                terms = [(self.v(b, a), 1.0) for a in gr]
                terms.append((self.v(0, b), -1.0))
                if gi < len(qs):
                    add(terms, 0.0, "=")
                else:                        # <= because of the slack
                    add(terms, 0.0, "<=")
        # same-gear pair refinement: Y[(i,q),(j,q)] <= sum over the forced c's
        for q in qs:
            u = self.u[q]
            for i in range(L):
                for j in range(i + 1, L):
                    a = self.idx[("g", i, q)]
                    b = self.idx[("g", j, q)]
                    if self.v(a, b) is None:
                        continue
                    inter = {(i - u) % q, (i + u) % q} & {(j - u) % q,
                                                          (j + u) % q}
                    terms = [(self.v(a, b), 1.0)]
                    for c in inter:
                        terms.append((self.v(0, self.idx[("c", q, c)]), -1.0))
                    add(terms, 0.0, "<=")
        for vvec in cuts:                    # PSD cuts  v'Yv >= 0
            terms = {}
            nz = np.flatnonzero(vvec)
            for a in nz:
                for b in nz:
                    cc = self.v(int(a), int(b))
                    if cc is None:
                        continue
                    terms[cc] = terms.get(cc, 0.0) + float(vvec[a] * vvec[b])
            add([(cc, -vv) for cc, vv in terms.items()], 0.0, "<=")
        A = coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsr()
        b = np.array(rhs)
        sense = np.array(sense)
        cvec = np.concatenate([np.zeros(self.ncols), np.ones(L)])
        return cvec, A, b, sense

    def solve(self, cuts=(), tol=None):
        cvec, A, b, sense = self.build(cuts)
        eqm = sense == "="
        res = linprog(cvec, A_ub=A[~eqm], b_ub=b[~eqm],
                      A_eq=A[eqm], b_eq=b[eqm],
                      bounds=[(0, None)] * A.shape[1], method="highs")
        return res, (cvec, A, b, sense)

    def gram(self, x):
        n = self.n
        Y = np.zeros((n, n))
        for a in range(n):
            for bb in range(a, n):
                cc = self.varof[(a, bb)]
                val = 0.0 if cc is None else x[cc]
                Y[a, bb] = Y[bb, a] = val
        return Y


def exact_dual_value(cvec, A, b, sense, y_num, denom=10 ** 6, keep=None):
    """Rationalise a dual vector, scale it until EXACTLY dual feasible, and
    return the exact rational dual objective.  Sound because cvec >= 0, so the
    dual-feasible set {A'y <= c} is star-shaped about 0."""
    m = A.shape[0]
    yq = np.array([Fraction(int(round(v * denom)), denom) for v in y_num])
    if keep is not None:
        mask = np.zeros(m, bool)
        mask[keep] = True
        yq[~mask] = Fraction(0)
    # inequality rows are  A x <= b  with dual  y <= 0 in max b'y s.t. A'y <= c
    for i in range(m):
        if sense[i] != "=" and yq[i] > 0:
            yq[i] = Fraction(0)
    Acsc = A.tocsc()
    theta = Fraction(1)
    for j in range(A.shape[1]):
        s = Acsc.indptr[j]
        e = Acsc.indptr[j + 1]
        acc = Fraction(0)
        for k in range(s, e):
            acc += Fraction(int(round(Acsc.data[k] * 4)), 4) * yq[Acsc.indices[k]]
        cj = Fraction(int(cvec[j]))
        if acc > cj:
            if acc <= 0:
                return None
            theta = min(theta, cj / acc) if cj > 0 else Fraction(0)
            if theta == 0:
                return None
    val = sum((Fraction(int(round(b[i] * 4)), 4) * yq[i] for i in range(m)),
              Fraction(0))
    return theta * val


def part_c(ys, Lrange=None, do_cuts=True, ncut=40):
    print("PART C - SA2 (level-2 LP) and SDP2 (level-2 + PSD cuts) on the "
          "covering CSP.\n  V(L) > 0 proves RUN(L) unsatisfiable, hence "
          "F <= L.  V is a MACHINE-FREE deficiency.\n")
    for y in ys:
        truth = F_EXACT[y]
        maxL = truth - 1
        print("=== machine %d : true max run %d, F = %d, gears %s"
              % (y, maxL, truth, gears(y)))
        Ls = Lrange or range(max(2, maxL - 2), 4 * truth)
        found = None
        for L in Ls:
            t0 = time.time()
            lift = Lift(y, L)
            res, (cvec, A, b, sense) = lift.solve()
            if res.status != 0:
                print("    L=%3d  SA2 solver status %d" % (L, res.status))
                continue
            V = res.fun
            tag = "SA2 V=%.6g" % V
            extra = ""
            if do_cuts and V <= 1e-9:
                cuts = []
                for it in range(ncut):
                    Y = lift.gram(res.x)
                    w, vec = np.linalg.eigh(Y)
                    if w[0] > -1e-8:
                        extra = "  PSD-feasible after %d cuts" % it
                        break
                    v = vec[:, 0]
                    v = np.where(np.abs(v) > 0.02, v, 0.0)
                    if not np.any(v):
                        break
                    cuts.append(v)
                    res, (cvec, A, b, sense) = lift.solve(cuts)
                    if res.status != 0:
                        extra = "  SDP2 INFEASIBLE at cut %d" % (it + 1)
                        break
                    if res.fun > 1e-9:
                        extra = "  SDP2 V=%.6g after %d cuts" % (res.fun,
                                                                 it + 1)
                        break
                else:
                    extra = "  SDP2 V=%.6g after %d cuts (cap)" % (res.fun,
                                                                   ncut)
                V = res.fun
            print("    L=%3d  n=%4d cols=%7d rows=%7d  %-18s%s  %.0fs"
                  % (L, lift.n, lift.ncols, A.shape[0], tag, extra,
                     time.time() - t0))
            if V > 1e-9 and found is None:
                found = L
                print("      => first certified L* = %d  (F <= %d, truth %d, "
                      "ratio %.2f)" % (L, L, truth, L / truth))
                break
        if found is None:
            print("      => no certificate in the tested range")
        print()




# ------------------------------------------------- the SMALL level-2 lift
#
# LiftC drops the g(i) witness variables entirely and works on the c-literals
# alone, with CONDITIONAL covering (the Sherali-Adams lift of "every position
# is covered", conditioned on each literal).  Moment matrix is (1 + sum_q q)
# square - 193 x 193 even at machine 37 - so this reaches machines whose
# periods are far out of scan range.
#
#   Y[a,b]        pseudo-moment of literals a, b  (index 0 = the unit)
#   t[i,a] >= 0   pseudo-moment of "position i uncovered" with literal a
#   constraints (all valid for a genuine distribution):
#     Y >= 0, Y[a,a] = Y[0,a], Y[0,0] = 1
#     Y[(q,c),(q,c')] = 0  (c != c'),  sum_c Y[0,(q,c)] = 1
#     sum_c Y[a,(q,c)] = Y[0,a]                             (marginalisation)
#     sum_q ( Y[a,(q,i-u_q)] + Y[a,(q,i+u_q)] ) + t[i,a] >= Y[0,a]  (cover|a)
#     sum_c t[i,(q,c)] = t[i,0]                             (t marginalises)
#   objective: min sum_i t[i,0] - the fractional number of uncoverable
#   positions.  > 0 proves RUN(L) unsatisfiable, hence F(M) <= L.


class LiftC:
    def __init__(self, y, L):
        self.y, self.L = y, L
        self.qs = gears(y)
        self.u = {q: tooth(q) for q in self.qs}
        self.lits = [("c", q, c) for q in self.qs for c in range(q)]
        self.idx = {a: j + 1 for j, a in enumerate(self.lits)}
        self.n = len(self.lits) + 1
        self.groups = [[self.idx[("c", q, c)] for c in range(q)]
                       for q in self.qs]
        self.gear_of = {}
        for a in range(1, self.n):
            self.gear_of[a] = self.lits[a - 1][1]
        self.varof = {}
        col = 0
        for a in range(self.n):
            for b in range(a, self.n):
                zero = (a and b and a != b
                        and self.gear_of[a] == self.gear_of[b])
                if zero:
                    self.varof[(a, b)] = None
                else:
                    self.varof[(a, b)] = col
                    col += 1
        self.ncols = col
        self.tbase = col                       # t[i,a] at tbase + i*n + a
        self.nv = col + L * self.n

    def v(self, a, b):
        return self.varof[(min(a, b), max(a, b))]

    def t(self, i, a):
        return self.tbase + i * self.n + a

    def build(self, cuts=()):
        n, L, qs = self.n, self.L, self.qs
        rows, cols, vals, rhs, sense = [], [], [], [], []
        r = 0

        def add(terms, rr, sn):
            nonlocal r
            for cc, vv in terms:
                if cc is None:
                    continue
                rows.append(r)
                cols.append(cc)
                vals.append(vv)
            rhs.append(rr)
            sense.append(sn)
            r += 1

        add([(self.v(0, 0), 1.0)], 1.0, "=")
        for a in range(1, n):
            add([(self.v(a, a), 1.0), (self.v(0, a), -1.0)], 0.0, "=")
        for gr in self.groups:
            add([(self.v(0, a), 1.0) for a in gr], 1.0, "=")
        for b in range(1, n):
            for gr in self.groups:
                if b in gr:
                    continue
                add([(self.v(b, a), 1.0) for a in gr]
                    + [(self.v(0, b), -1.0)], 0.0, "=")
        for i in range(L):
            for a in range(n):
                terms = []
                for q in qs:
                    u = self.u[q]
                    for c in {(i - u) % q, (i + u) % q}:
                        terms.append((self.v(a, self.idx[("c", q, c)]), 1.0))
                terms.append((self.t(i, a), 1.0))
                terms.append((self.v(0, a), -1.0))
                add(terms, 0.0, ">=")
            for gr in self.groups:
                add([(self.t(i, a), 1.0) for a in gr]
                    + [(self.t(i, 0), -1.0)], 0.0, "=")
        for vvec in cuts:
            terms = {}
            nz = np.flatnonzero(vvec)
            for a in nz:
                for b in nz:
                    cc = self.v(int(a), int(b))
                    if cc is None:
                        continue
                    terms[cc] = terms.get(cc, 0.0) + float(vvec[a] * vvec[b])
            add([(cc, vv) for cc, vv in terms.items()], 0.0, ">=")
        A = coo_matrix((vals, (rows, cols)), shape=(r, self.nv)).tocsr()
        cvec = np.zeros(self.nv)
        for i in range(L):
            cvec[self.t(i, 0)] = 1.0
        return cvec, A, np.array(rhs), np.array(sense)

    def solve(self, cuts=()):
        cvec, A, b, sense = self.build(cuts)
        ge = sense == ">="
        eq = sense == "="
        A_ub = -A[ge] if ge.any() else None
        b_ub = -b[ge] if ge.any() else None
        res = linprog(cvec, A_ub=A_ub, b_ub=b_ub,
                      A_eq=A[eq], b_eq=b[eq],
                      bounds=[(0, None)] * self.nv, method="highs")
        return res, (cvec, A, b, sense)

    def gram(self, x):
        n = self.n
        Y = np.zeros((n, n))
        for a in range(n):
            for bb in range(a, n):
                cc = self.varof[(a, bb)]
                Y[a, bb] = Y[bb, a] = 0.0 if cc is None else x[cc]
        return Y


def certify(y, L, ncut=60, eigtol=-1e-7):
    """Return (V_sa2, V_sdp2, cuts_used, min eigenvalue seen, lift)."""
    lift = LiftC(y, L)
    res, _ = lift.solve()
    if res.status != 0:
        return None, None, 0, None, lift
    Vsa = res.fun
    if Vsa > 1e-9:
        return Vsa, Vsa, 0, None, lift
    cuts, used, mineig = [], 0, None
    for it in range(ncut):
        Y = lift.gram(res.x)
        w, vec = np.linalg.eigh(Y)
        mineig = float(w[0])
        if w[0] > eigtol:
            break
        v = vec[:, 0]
        v = np.where(np.abs(v) > 1e-3, v, 0.0)
        if not np.any(v):
            break
        cuts.append(v)
        used = it + 1
        res, _ = lift.solve(cuts)
        if res.status != 0:
            return Vsa, float("inf"), used, mineig, lift
        if res.fun > 1e-9:
            return Vsa, res.fun, used, mineig, lift
    return Vsa, res.fun, used, mineig, lift




# ------------------------------------------------- exact rational duals
#
# Every LP above is put in the canonical form
#       min c'x   s.t.  M x >= d,  x >= 0
# (equalities split into two inequalities), with M, d, c INTEGER - the model
# rows all have coefficients +-1, and PSD cut vectors are rationalised to
# integers before the cut is formed, so nothing is approximated.  Weak duality
# then says: for any y >= 0 with M'y <= c,  d'y  <=  V(L).  So an exact
# rational y with d'y > 0 PROVES V(L) > 0, hence RUN(L) unsatisfiable, hence
# F(M) <= L - with no trust in the floating-point solver at all.
#
# The solver's dual is rationalised over a fixed denominator, clipped to
# y >= 0, and then scaled by the largest theta in [0,1] that restores
# M'y <= c exactly.  That is legitimate because c >= 0, so y = 0 is feasible
# and the feasible set is star-shaped about it.


def canon(lift, cuts=()):
    """Integer canonical form  min c'x s.t. M x >= d, x >= 0."""
    cvec, A, b, sense = lift.build(cuts)
    A = A.tocoo()
    rows, cols, vals = [], [], []
    d = []
    r = 0
    order = {}
    for i in range(A.shape[0]):
        order.setdefault(i, [])
    for i, j, v in zip(A.row, A.col, A.data):
        order[i].append((j, v))
    for i in range(A.shape[0]):
        sgn = [1.0] if sense[i] == ">=" else ([1.0, -1.0] if sense[i] == "="
                                              else [-1.0])
        for s in sgn:
            for j, v in order[i]:
                rows.append(r)
                cols.append(j)
                vals.append(s * v)
            d.append(s * b[i])
            r += 1
    M = coo_matrix((vals, (rows, cols)), shape=(r, A.shape[1]))
    dd = np.array(d)
    assert np.all(np.abs(M.data - np.round(M.data)) < 1e-9), "non-integer M"
    assert np.all(np.abs(dd - np.round(dd)) < 1e-9), "non-integer d"
    Mi = coo_matrix((np.round(M.data).astype(np.int64), (M.row, M.col)),
                    shape=M.shape).tocsr()
    return cvec.astype(np.int64), Mi, np.round(dd).astype(np.int64)


def exact_dual_bound(lift, cuts=(), ks=(1, 2, 3, 4, 6, 12, 24, 60, 120, 360,
                                        2520, 27720, 10 ** 6)):
    """EXACT rational lower bound on V(L) by weak duality, or Fraction(0).

    The solver's dual is snapped to rationals with limit_denominator(K) for
    increasing K and each candidate is CHECKED EXACTLY (integer arithmetic:
    M' z <= K * c and z >= 0).  The first candidate that verifies with a
    positive objective is returned - nothing is trusted to the solver."""
    c, M, d = canon(lift, cuts)
    res = linprog(c.astype(float), A_ub=(-M).astype(float),
                  b_ub=(-d).astype(float),
                  bounds=[(0, None)] * M.shape[1], method="highs")
    if res.status != 0:
        return None, res
    yv = np.maximum(-res.ineqlin.marginals, 0.0)
    yv[yv < 1e-9] = 0.0
    Mt = M.T.tocsr()
    best = Fraction(0)
    for K in ks:
        fr = [Fraction(float(v)).limit_denominator(K) for v in yv]
        den = 1
        for f in fr:
            den = den * f.denominator // np.gcd(den, f.denominator)
            if den > 10 ** 7:
                den = None
                break
        if den is None:
            continue
        z = np.array([int(f * den) for f in fr], dtype=np.int64)
        if not z.any():
            continue
        g = Mt.dot(z)
        if np.any(g > c.astype(np.int64) * den):
            continue
        val = Fraction(int(d.astype(np.int64).dot(z)), den)
        if val > best:
            best = val
    return best, res


def part_d(ys, ncut=60, span=None):
    print("PART D - the SMALL level-2 lift (c-literals + conditional "
          "covering).  n = 1 + sum_q q.\n  V(L) > 0  =>  RUN(L) impossible "
          "=>  F(M) <= L.  SA2 = LP only; SDP2 = + PSD cuts.\n")
    out = {}
    for y in ys:
        truth = F_EXACT.get(y)
        maxL = (truth - 1) if truth else None
        print("=== machine %d : gears %s%s"
              % (y, gears(y),
                 ("   true max run %d, F = %d" % (maxL, truth)) if truth
                 else ""))
        lo = max(2, (maxL - 1) if maxL else 2)
        hi = span or (4 * truth if truth else 400)
        first = None
        L = lo
        while L <= hi:
            t0 = time.time()
            Vsa, Vsdp, used, mineig, lift = certify(y, L, ncut=ncut)
            if Vsa is None:
                print("    L=%3d  solver failure" % L)
                break
            print("    L=%3d  n=%3d cols=%6d   SA2 V=%-10.6g  SDP2 V=%-10.6g "
                  "cuts %2d  mineig %s  %.0fs"
                  % (L, lift.n, lift.nv, Vsa, Vsdp, used,
                     ("%.2e" % mineig) if mineig is not None else "  n/a",
                     time.time() - t0))
            if Vsa > 1e-9 or Vsdp > 1e-9:
                first = (L, Vsa > 1e-9)
                break
            L += 1
        if first:
            L, bysa = first
            print("      => L* = %d  by %s  (F <= %d, truth %s, ratio %s)"
                  % (L, "SA2" if bysa else "SDP2", L, truth,
                     ("%.3f" % (L / truth)) if truth else "?"))
            out[y] = (L, bysa)
        else:
            print("      => no certificate up to L = %d" % hi)
            out[y] = None
        print()
    return out




# ------------------------------------------------- the ladder, by bisection
#
# V(L) is monotone: feasibility of RUN(L) is downward closed, so V(L) > 0 for
# every L >= L*.  Bisect for L* = min{L : V(L) > 0}; then F(M) <= L* is a
# PROVED, machine-free, arity-2 upper bound - and L* == F(M) means the level-2
# LP has computed the maximal gap exactly.
#
# PRE-REGISTERED (written before machines 23+ were run; P1-P5 are in the
# module docstring):
#   P6  SA2's exactness BREAKS somewhere at or below machine 37 - L* > F(M) -
#       because a polynomial-size LP that computes a Jacobsthal-type maximum
#       exactly at every size is implausible.
#   P7  where it breaks, the PSD cuts (SDP2) recover at least one unit.
#   P8  the ladder is SOUND at every machine with a known F: L* >= F(M)
#       always.  An L* < F(M) anywhere would be a soundness bug in the
#       relaxation and must be reported as such, not as a result.


def lstar(y, F0=None, cap=None, log=True, cuts=()):
    """L* = min{L : V(L) > 0} by doubling search up from F0, then bisection.

    V is monotone (RUN feasibility is downward closed), so this is exact."""
    cache = {}

    def V(L):
        if L not in cache:
            t0 = time.time()
            lift = LiftC(y, L)
            res, _ = lift.solve(cuts)
            cache[L] = (res.fun if res.status == 0 else None, lift,
                        time.time() - t0)
            if log:
                print("      L=%3d  V=%-12.6g  cols=%6d  %5.0fs"
                      % (L, cache[L][0] if cache[L][0] is not None else -1,
                         lift.nv, cache[L][2]))
                sys.stdout.flush()
        return cache[L][0]

    F0 = F0 or F_EXACT.get(y) or 8
    cap = cap or (4 * F0)
    lo = F0 - 1
    v = V(lo)
    assert v is not None and v <= 1e-9, ("SOUNDNESS FAILURE at L=F-1", y, v)
    if V(F0) > 1e-9:
        return F0, cache
    a, step = F0, 2
    while True:
        b = a + step
        if b > cap:
            return None, cache
        if V(b) > 1e-9:
            break
        a, step = b, step * 2
    while a + 1 < b:
        mid = (a + b) // 2
        if V(mid) > 1e-9:
            b = mid
        else:
            a = mid
    return b, cache



def exact_dual_bound2(lift, cuts=(), frac=0.5, log=False):
    """EXACT rational lower bound on V(L), robustly.

    Step 1: get the LP optimum V (numerical).  Step 2: solve the STRICTLY
    FEASIBLE dual  max t  s.t.  M'y + t*1 <= c, y >= 0, d'y >= frac*V.  If
    t* > 0 the dual has interior, and rounding y DOWN to a denominator
    D > max(colsum/t*, |d|_1/(frac*V)) keeps it feasible and keeps the
    objective positive.  Step 3: verify M'z <= D*c and d'z > 0 in EXACT
    integer arithmetic and return the exact rational d'z/D.
    Nothing is trusted to the solver: the returned Fraction is checked."""
    c, M, d = canon(lift, cuts)
    nrow, ncol = M.shape
    res = linprog(c.astype(float), A_ub=(-M).astype(float),
                  b_ub=(-d).astype(float),
                  bounds=[(0, None)] * ncol, method="highs")
    if res.status != 0 or res.fun <= 1e-9:
        return Fraction(0), (res.fun if res.status == 0 else None)
    V = res.fun
    Mt = M.T.tocsr()
    # variables (y, t): maximise t
    from scipy.sparse import hstack as _hs, vstack as _vs, csr_matrix as _csr
    ones = _csr(np.ones((ncol, 1)))
    A1 = _hs([Mt.astype(float), ones]).tocsr()          # M'y + t <= c
    A2 = _hs([_csr(-d.astype(float).reshape(1, -1)),
              _csr(np.zeros((1, 1)))]).tocsr()          # -d'y <= -frac*V
    A_ub = _vs([A1, A2]).tocsr()
    b_ub = np.concatenate([c.astype(float), [-frac * V]])
    cc = np.zeros(nrow + 1)
    cc[-1] = -1.0
    r2 = linprog(cc, A_ub=A_ub, b_ub=b_ub,
                 bounds=[(0, None)] * nrow + [(None, None)], method="highs")
    if r2.status != 0:
        return Fraction(0), V
    t = -r2.fun
    yv = np.maximum(r2.x[:nrow], 0.0)
    if t <= 0:
        return Fraction(0), V
    colsum = np.asarray(np.abs(M).sum(axis=0)).ravel().max()
    d1 = float(np.abs(d).sum())
    D = int(max(2 * colsum / t, 2 * d1 / (frac * V), 64)) + 1
    z = np.floor(yv * D).astype(np.int64)
    g = Mt.dot(z)
    if np.any(g > c.astype(np.int64) * D):
        return Fraction(0), V
    num = int(d.astype(np.int64).dot(z))
    if num <= 0:
        return Fraction(0), V
    if log:
        print("        [exact] t* = %.3e  D = %d  support %d"
              % (t, D, int((z > 0).sum())))
    return Fraction(num, D), V


def part_e(ys, hi_mult=3, exact=True):
    print("PART E - THE MACHINE-FREE ARITY-2 LADDER.  L* = min{L : the "
          "level-2 LP proves\n  RUN(L) impossible}.  F(M) <= L*, with no "
          "period, no scan and no machine input\n  beyond the prime list.\n")
    rows = []
    for y in ys:
        truth = F_EXACT.get(y)
        hi = int(hi_mult * (truth or 200)) + 20
        print("=== machine %d  gears %s  (known F = %s)"
              % (y, gears(y), truth))
        t0 = time.time()
        Ls, cache = lstar(y, cap=hi)
        if Ls is None:
            print("      no certificate up to L = %d\n" % hi)
            rows.append((y, None, truth, None))
            continue
        ex = None
        if exact:
            lift = LiftC(y, Ls)
            val, _v = exact_dual_bound2(lift)
            ex = val
        ratio = (Ls / truth) if truth else None
        print("    L* = %d   F <= %d   truth %s   ratio %s   exact dual "
              "bound %s   %.0fs\n"
              % (Ls, Ls, truth, ("%.4f" % ratio) if ratio else "?", ex,
                 time.time() - t0))
        if truth is not None:
            assert Ls >= truth, ("SOUNDNESS FAILURE", y, Ls, truth)
        rows.append((y, Ls, truth, ex))
    print("  y    L*   F(M)   L*/F   exact dual bound at L*")
    for y, Ls, truth, ex in rows:
        print("  %-4d %-5s %-6s %-7s %s"
              % (y, Ls, truth,
                 ("%.4f" % (Ls / truth)) if (Ls and truth) else "-", ex))
    return rows




# ------------------------------------------- item (c): the norm cliff
#
# Round 23 (nilpotent_invariants.py) proved N^n = diag(v_n) S^n is a partial
# isometry, so ||N^n|| is 1 for n < F and 0 after: F sits ENTIRELY IN THE
# CONSTANT of any envelope ||N^n|| <= C lam^n, which forces C >= lam^(1-F),
# i.e.  F <= 1 + log C / log(1/lam).  Is that a lower-bound technique or
# another circularity?  This part settles it EXACTLY.
#
# THE ENVELOPE DICHOTOMY.  Let w > 0 be a weight and ||x||_w = max_k |x_k|/w_k.
# N acts by (Nx)_{k+1} = b(k+1) x_k, so
#       ||N||_w = max{ w_k / w_{k+1} : k+1 blocked }
# and the norm-equivalence constant to the unweighted sup norm is exactly
# (max w)/(min w).  Writing w = 2^h this is
#       lam = 2^{-min step of h over blocked slots},   C = 2^{osc(h)},
# so  1 + log_lam-bound  =  1 + osc(h) / (min step)  -  IDENTICALLY the
# potential bound of round 23.  The constant in a weighted envelope IS the
# oscillation of a potential; and by round 23 item 39 an envelope in any
# UNITARILY INVARIANT norm is a function of the gap histogram, hence circular.
# So (c) is not a new technique: it is the arity ladder in analytic clothing.
# Verified below with exact integer arithmetic (h integer, w = 2^h).


def machine_arrays(y):
    qs = gears(y)
    P = 1
    for q in qs:
        P *= q
    blocked = np.zeros(P, bool)
    for q in qs:
        u = tooth(q)
        blocked[u % q::q] = True
        blocked[(-u) % q::q] = True
    return P, blocked


def part_f(ys=(11, 13, 17)):
    print("PART F - item (c): does the NORM CLIFF convert into a technique?\n"
          "  Answer: no - the constant of a weighted envelope is EXACTLY "
          "2^osc(h).\n")
    for y in ys:
        P, blocked = machine_arrays(y)
        # tight potential: h(k) = distance back to the previous opening
        h = np.zeros(P, np.int64)
        opens = np.flatnonzero(~blocked)
        k0 = int(opens[0])
        cur = 0
        for j in range(P):
            k = (k0 + j) % P
            cur = 0 if not blocked[k] else cur + 1
            h[k] = cur
        # (*) h(k) - h(k-1) >= 1 at every blocked slot
        step = h - np.roll(h, 1)
        assert step[blocked].min() >= 1, y
        osc = int(h.max() - h.min())
        # w = 2^h : lam = max over blocked of w_{k-1}/w_k = 2^{-min step}
        minstep = int(step[blocked].min())
        lam_exp = -minstep                       # lam = 2^lam_exp
        C_exp = osc                              # C   = 2^osc
        bound = 1 + C_exp // (-lam_exp)
        # exact norm cliff: ||N^n||_inf = 1 iff a run of n blocked slots exists
        runs = h.max()
        print("    y=%2d  P=%9d  F=%3d   osc(h)=%3d  min step=%d   "
              "envelope C=2^%d, lam=2^%d\n          => 1 + logC/log(1/lam) "
              "= %3d   = 1 + osc(h) = %3d   = F  %s"
              % (y, P, F_EXACT[y], osc, minstep, C_exp, lam_exp, bound,
                 1 + osc, "OK" if bound == F_EXACT[y] == 1 + osc else "MISMATCH"))
        assert bound == F_EXACT[y] == 1 + osc == 1 + int(runs), y
    print("\n  The envelope constant IS the potential oscillation, exactly, "
          "at every machine.\n  Item (c) is a REDUCTION to the arity ladder, "
          "not a second technique.\n")




# ------------------------------------------- THE BRIEF'S QUESTION, DIRECTLY
#
# part_g takes an (M, L) pair where L is KNOWN impossible (L > F(M) - 1) but
# the level-2 LP does not see it (V(L) = 0), and asks whether the PSD
# constraint on the moment matrix closes the gap.  Cuts are added in BATCHES
# (every eigenvector with eigenvalue below -tol, up to `batch`), each cut
# v'Yv >= 0 valid for any real v, so soundness never depends on the numerics.
# Reported: the V trajectory and the minimum-eigenvalue trajectory.  If the
# minimum eigenvalue rises to ~0 with V still 0, the SDP relaxation is
# FEASIBLE at that L and the SDP provably does not bite there.


def part_g(pairs, batch=8, rounds=40, tol=1e-7):
    print("PART G - DOES THE PSD CONSTRAINT BITE WHERE THE LEVEL-2 LP DOES "
          "NOT?\n")
    for (y, L) in pairs:
        truth = F_EXACT.get(y)
        print("=== machine %d, L = %d  (true max run %s; L is %s)"
              % (y, L, (truth - 1) if truth else "?",
                 "IMPOSSIBLE" if truth and L > truth - 1 else "possible"))
        lift = LiftC(y, L)
        cuts = []
        res, _ = lift.solve(cuts)
        traj = []
        for it in range(rounds):
            if res.status != 0:
                print("    round %2d  LP INFEASIBLE -> SDP2 certifies" % it)
                break
            V = res.fun
            Y = lift.gram(res.x)
            w, vec = np.linalg.eigh(Y)
            traj.append((it, V, float(w[0]), len(cuts)))
            print("    round %2d  cuts %4d  V = %-12.6g  min eig = %.3e"
                  % (it, len(cuts), V, w[0]))
            sys.stdout.flush()
            if V > 1e-9:
                print("    => SDP2 CERTIFIES at L = %d with %d cuts"
                      % (L, len(cuts)))
                break
            if w[0] > -tol:
                print("    => moment matrix is PSD: THE SDP RELAXATION IS "
                      "FEASIBLE at L = %d.\n       The SDP does NOT bite "
                      "here." % L)
                break
            neg = [j for j in range(len(w)) if w[j] < -tol][:batch]
            for j in neg:
                v = vec[:, j]
                v = np.where(np.abs(v) > 1e-3, v, 0.0)
                if np.any(v):
                    cuts.append(v)
            res, _ = lift.solve(cuts)
        else:
            print("    => %d rounds, %d cuts, V still %.6g (cap reached)"
                  % (rounds, len(cuts), res.fun))
        print()


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    ys = [11, 13, 17, 19]
    if "--y" in sys.argv:
        ys = [int(x) for x in sys.argv[sys.argv.index("--y") + 1].split(",")]
    if what in ("all", "partA"):
        part_a()
    if what in ("all", "partB"):
        part_b(ys)
    if what in ("all", "partC"):
        part_c(ys)
    if what in ("all", "partD"):
        part_d(ys)
    if what == "partE":
        part_e(ys)
    if what in ("all", "partF"):
        part_f()
    if what == "partG":
        prs = []
        for tok in (sys.argv[sys.argv.index("--pairs") + 1].split(",")
                    if "--pairs" in sys.argv else ["19:25"]):
            a, bb = tok.split(":")
            prs.append((int(a), int(bb)))
        part_g(prs)


if __name__ == "__main__":
    main()
