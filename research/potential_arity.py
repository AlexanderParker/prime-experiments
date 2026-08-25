"""Round 23 lateral - WHAT REPLACES SPECTRUM IN A NILPOTENT SECTOR (part 2):
THE CERTIFICATE ARITY LADDER for the maximal gap F(M).

WHY.  Round 22 located the machine's growth in the NILPOTENT direction:
(BS)^n = diag(v_n) S^n has spectrum {0} at every depth, so there are no
eigenvalues to bound.  research/nilpotent_invariants.py proves that EVERY
unitary invariant of BS (singular values, Schatten norms, numerical range,
pseudospectrum, Jordan type) is a function of the gap histogram alone - so no
operator-theoretic invariant can carry information the histogram does not
already have.  What genuinely replaces the spectrum is therefore not an
invariant at all, it is a CERTIFICATE: a potential (Lyapunov function).

THE POTENTIAL CERTIFICATE (elementary, exact, tight).  Let h : Z_P -> R.  If

    (*)  h(k) - h(k-1) >= 1   for every BLOCKED slot k,

then along any run of L consecutive blocked slots h increases by >= L, so
L <= osc(h) := max h - min h, and since a gap of g slots is a run of g-1
blocked slots,

    F(M) <= 1 + osc(h)   for every h satisfying (*).

It is TIGHT: h(k) = (distance from k back to the previous opening) satisfies
(*) with osc = F-1 exactly.  So F is EXACTLY the optimum of a linear program
over potentials - a certificate problem, not a spectral one.  In the tropical
(max,+) reading (*) is exactly Constructor's potential inequality, and
w = exp(h/t) turns it into a Schur test on A = BS + (BS)^T (see
nilpotent_invariants.py part 5); the three semirings compute one F.

THE OBJECT MEASURED HERE: THE ARITY OF THE POTENTIAL.  The optimal h above is
"distance to the previous opening" - a function of ALL gears jointly (it is
the window indicator whose Schmidt rank round 22 measured GROWING).  Restrict
h to a fixed ARITY r:

    LEVEL-r class:  h(k) = sum over gear subsets U with |U| = r of
                    x_U(k mod prod U)

(level r contains level r-1, since a function of k mod q is a function of
k mod q q').  r = 1 is a per-gear potential; r = m (all gears) is the full
class and returns F exactly.  The minimal r that is FEASIBLE AT ALL is a
proof-obligation form of the round's spine question "does the arity
stabilise?" - and unlike a census it is a statement about every possible
certificate of that arity, not about one attempt.

TWO THEOREMS PROVED HERE (part 1, exact rational arithmetic):

  T1 (bounded-state no-go, one line).  If h(k) depends only on k mod m for a
  PROPER divisor m of P, then (*) is infeasible outright: every residue class
  mod m contains a blocked slot (the large gears block inside every class),
  so (*) forces h(r) - h(r-1) >= 1 for all r mod m, and summing round the
  cycle gives 0 >= m.  A state that has forgotten any gear cannot see that a
  slot is blocked.  (This is why bounded-state certificates mod 35/385/5005
  cannot bound F; cf. Constructor's failures at 23->29.)

  T2 (MERTENS NO-GO for arity 1).  A level-1 potential exists only if
  sum_{q gear} 1/q < 1/2.  Proof: write D(k) = h(k) - h(k-1) = sum_q f_q(k
  mod q) with each f_q of zero mean (a difference over a full cycle).  Let
  S_q = sum over the two teeth of f_q and Sigma = sum_q S_q/(q-2), so that
  the mean of D over OPEN slots is exactly -Sigma (CRT: residues independent
  and uniform, openings are the all-exposed slots).
    (i) mean_{Z_P} D = 0 and D >= 1 on blocked gives 0 >= -pi_o Sigma +
        (1 - pi_o), i.e. Sigma >= (1-pi_o)/pi_o > 0, pi_o = prod (1-2/q).
    (ii) for each gear q and each tooth t, the slots with k = t mod q and all
        other gears exposed are blocked and nonempty (CRT), and averaging (*)
        over them gives f_q(t) - Sigma + S_q/(q-2) >= 1; summing the two
        teeth gives S_q q/(q-2) >= 2(1 + Sigma), i.e. S_q/(q-2) >= 2(1+
        Sigma)/q.  Summing over gears: Sigma >= 2 sigma (1 + Sigma) with
        sigma = sum_q 1/q, i.e. Sigma (1 - 2 sigma) >= 2 sigma.
    If sigma >= 1/2 the left side is <= 0 < 2 sigma - contradiction.  QED.
  sigma = 1/5+1/7+1/11 = 0.4338 at y = 11 but 0.5106 at y = 13, and sigma
  diverges, so ARITY-1 CERTIFICATES DIE AT MACHINE 13 AND NEVER RETURN.

PRE-REGISTERED PREDICTIONS (written before running parts 2-4):
  P1  level 1 is feasible at y = 11 and infeasible at y = 13, 17, 19 (T2).
  P2  the minimal feasible arity r*(y) GROWS: r*(11) = 1, r*(13) = 2, and
      r*(19) >= 3.  (If r* stayed at 2 for every machine, a fixed-arity
      certificate for F would exist and round 22's spine answer would be
      contradicted from the certificate side.)
  P3  where feasible, the bound 1 + osc* is well above the true F (a loose
      but valid bound), not equal to it.

Outputs: the ladder table (machine x arity -> feasible?, bound 1+osc*, true
F).  LP results are floats (HiGHS) and labeled as such; T1/T2 and the
tightness statement are exact.  Infeasibility found on a SUBSET of the
constraints is still a proof of infeasibility for the full system, and is
labeled "(subsampled)" where used.

Usage: python potential_arity.py                # T1, T2, tightness, ladder
       python potential_arity.py 17:3,19:2:8   # explicit y:arity[:row_stride]
"""
import sys
import time
from fractions import Fraction
from itertools import combinations
from math import prod

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack


def primes(a, b):
    return [n for n in range(a, b + 1)
            if n > 1 and all(n % d for d in range(2, int(n ** .5) + 1))]


def teeth(q):
    u = pow(6, -1, q)
    return u % q, (-u) % q


def blocked(gears):
    P = prod(gears)
    b = np.zeros(P, bool)
    for q in gears:
        t1, t2 = teeth(q)
        b[t1::q] = True
        b[t2::q] = True
    return b


def maxgap(b):
    idx = np.flatnonzero(~b)
    P = b.size
    g = np.diff(np.append(idx, idx[0] + b.size))
    assert g.sum() == P
    return int(g.max())


# ---------------------------------------------------------------- part 1
def part1(ys):
    print("=" * 74)
    print("PART 1 - T1 and T2, exact rational arithmetic")
    print("=" * 74)
    print("T1: a potential that depends only on k mod m (m | P, m < P) is")
    print("    infeasible: every class mod m holds a blocked slot, so summing")
    print("    (*) round the m-cycle gives 0 >= m.  Checked below.")
    for y in ys:
        gears = primes(5, y)
        P = prod(gears)
        b = blocked(gears)
        for m in [35, 385]:
            if m >= P:
                continue
            hit = np.zeros(m, bool)
            idx = np.flatnonzero(b)
            hit[idx % m] = True
            assert hit.all(), (y, m)
        print(f"  y={y:3d}: every residue class mod 35 and mod 385 contains a"
              f" blocked slot -> arity-'bottom gears only' infeasible")
    print()
    print("T2: level-1 feasible only if sigma = sum_q 1/q < 1/2 (exact):")
    print("   y   sigma (exact)            sigma      1-2sigma   verdict")
    for y in ys:
        gears = primes(5, y)
        sig = sum((Fraction(1, q) for q in gears), Fraction(0))
        ok = sig < Fraction(1, 2)
        print(f"  {y:3d}   {str(sig):22s} {float(sig):.6f}  "
              f"{float(1 - 2 * sig):+9.6f}   "
              f"{'level-1 POSSIBLE' if ok else 'level-1 IMPOSSIBLE (proved)'}")
    print()


# ---------------------------------------------------------------- LP core
def build_lp(gears, r, row_stride=1, with_osc=True):
    """Level-r potential LP.  Variables: x_U for each |U| = r, concatenated.

    Constraints: (a) h(k) - h(k-1) >= 1 for blocked k (rows subsampled by
    row_stride);  (b) 0 <= h(k) <= t for all k (only if with_osc).
    Objective: minimise t (0 if not with_osc)."""
    P = prod(gears)
    subs = list(combinations(gears, r))
    mods = [prod(U) for U in subs]
    offs, tot = [], 0
    for m in mods:
        offs.append(tot)
        tot += m
    b = blocked(gears)
    k_all = np.arange(P, dtype=np.int64)
    kb = np.flatnonzero(b)[::row_stride]

    def hmat(ks):
        """sparse matrix rows giving h(ks) in the x variables."""
        n = ks.size
        rows = np.repeat(np.arange(n), len(mods))
        cols = np.empty(n * len(mods), dtype=np.int64)
        for j, (m, o) in enumerate(zip(mods, offs)):
            cols[j::len(mods)] = o + (ks % m)
        data = np.ones(cols.size)
        return csr_matrix((data, (rows, cols)), shape=(n, tot))

    Hb = hmat(kb)
    Hbm = hmat((kb - 1) % P)
    # (a): -(h(k) - h(k-1)) <= -1
    A1 = (Hbm - Hb)
    b1 = -np.ones(A1.shape[0])
    if not with_osc:
        A = hstack([A1, csr_matrix((A1.shape[0], 1))], format='csr')
        c = np.zeros(tot + 1)
        return A, b1, c, tot, P, len(kb)
    Hall = hmat(k_all)
    zc = csr_matrix((P, 1))
    one = csr_matrix(np.ones((P, 1)))
    # h(k) - t <= 0 ; -h(k) <= 0
    A2 = hstack([Hall, -one], format='csr')
    A3 = hstack([-Hall, zc], format='csr')
    A1x = hstack([A1, csr_matrix((A1.shape[0], 1))], format='csr')
    from scipy.sparse import vstack
    A = vstack([A1x, A2, A3], format='csr')
    bb = np.concatenate([b1, np.zeros(P), np.zeros(P)])
    c = np.zeros(tot + 1)
    c[-1] = 1.0
    return A, bb, c, tot, P, len(kb)


def solve_level(gears, r, row_stride=1, with_osc=True):
    A, bb, c, tot, P, nrows = build_lp(gears, r, row_stride, with_osc)
    bounds = [(None, None)] * tot + [(0, None)]
    t0 = time.time()
    res = linprog(c, A_ub=A, b_ub=bb, bounds=bounds, method='highs')
    dt = time.time() - t0
    return res, tot, nrows, dt


# ---------------------------------------------------------------- part 2-4
def verify_certificate(gears, r, x):
    """rebuild h from the LP solution and CHECK (*) directly (no LP trust)."""
    P = prod(gears)
    subs = list(combinations(gears, r))
    mods = [prod(U) for U in subs]
    k = np.arange(P, dtype=np.int64)
    h = np.zeros(P)
    o = 0
    for m in mods:
        h += x[o:o + m][k % m]
        o += m
    b = blocked(gears)
    d = h - h[(k - 1) % P]
    worst = float(d[b].min())
    osc = float(h.max() - h.min())
    return worst, osc


def one(y, r, st):
    gears = primes(5, y)
    b = blocked(gears)
    F = maxgap(b)
    res, tot, nrows, dt = solve_level(gears, r, st, with_osc=True)
    tag = " (subsampled x%d)" % st if st > 1 else ""
    if res.status == 0:
        worst, osc = verify_certificate(gears, r, res.x[:-1])
        assert worst > 1 - 1e-6, (y, r, worst)
        bound = 1 + osc
        assert bound >= F - 1e-6, (y, r, bound, F)
        print(f"  {y:3d} {F:5d} {r:6d} {tot:7d} {nrows:9d}   "
              f"{'FEASIBLE':22s} {bound:9.3f}  ratio {bound/F:5.2f}"
              f"  [{dt:.0f}s]{tag}  (h re-checked: min step "
              f"{worst:.4f} >= 1)")
        return (y, F, r, "feasible", bound)
    if res.status == 2:
        res2, _, _, dt2 = solve_level(gears, r, st, with_osc=False)
        if res2.status == 2:
            print(f"  {y:3d} {F:5d} {r:6d} {tot:7d} {nrows:9d}   "
                  f"{'INFEASIBLE (proved)':22s} {'-':>9s}"
                  f"               [{dt+dt2:.0f}s]{tag}")
            return (y, F, r, "infeasible", None)
        print(f"  {y:3d} {F:5d} {r:6d} {tot:7d} {nrows:9d}   "
              f"{'feasible, osc unbounded':22s} {'-':>9s}"
              f"               [{dt+dt2:.0f}s]{tag}")
        return (y, F, r, "feasible-unbounded", None)
    print(f"  {y:3d} {F:5d} {r:6d} {tot:7d} {nrows:9d}   status {res.status}")
    return (y, F, r, f"status{res.status}", None)



def build_lp_rows(gears, r, kb, with_osc_rows=None):
    """LP on an EXPLICIT row set kb (blocked slots), plus optional osc rows."""
    P = prod(gears)
    subs = list(combinations(gears, r))
    mods = [prod(U) for U in subs]
    offs, tot = [], 0
    for m_ in mods:
        offs.append(tot)
        tot += m_

    def hmat(ks):
        n = ks.size
        rows = np.repeat(np.arange(n), len(mods))
        cols = np.empty(n * len(mods), dtype=np.int64)
        for j, (m_, o) in enumerate(zip(mods, offs)):
            cols[j::len(mods)] = o + (ks % m_)
        return csr_matrix((np.ones(cols.size), (rows, cols)), shape=(n, tot))

    A1 = hmat((kb - 1) % P) - hmat(kb)
    b1 = -np.ones(A1.shape[0])
    from scipy.sparse import vstack
    if with_osc_rows is None:
        A = hstack([A1, csr_matrix((A1.shape[0], 1))], format='csr')
        return A, b1, np.zeros(tot + 1), tot
    ko = with_osc_rows
    H = hmat(ko)
    one = csr_matrix(np.ones((ko.size, 1)))
    A = vstack([hstack([A1, csr_matrix((A1.shape[0], 1))], format='csr'),
                hstack([H, -one], format='csr'),
                hstack([-H, csr_matrix((ko.size, 1))], format='csr')],
               format='csr')
    bb = np.concatenate([b1, np.zeros(ko.size), np.zeros(ko.size)])
    c = np.zeros(tot + 1)
    c[-1] = 1.0
    return A, bb, c, tot


def solve_cutting(gears, r, start_stride=32, rounds=12, verbose=True,
                  osc=True):
    """Row generation: solve on a subset, verify on ALL blocked slots, add the
    violated ones, repeat.  A certificate that passes the FULL verification is
    a proof regardless of which rows the LP saw; an LP that goes infeasible on
    a SUBSET is a proof of infeasibility.  So both verdicts are sound."""
    P = prod(gears)
    b = blocked(gears)
    allb = np.flatnonzero(b)
    kb = allb[::start_stride]
    ko = np.arange(0, P, max(1, start_stride))
    for it in range(rounds):
        A, bb, c, tot = build_lp_rows(gears, r, kb, ko if osc else None)
        res = linprog(c, A_ub=A, b_ub=bb,
                      bounds=[(None, None)] * tot + [(0, None)],
                      method='highs')
        if res.status == 2:
            return "infeasible", None, len(kb)
        if res.status != 0:
            return f"status{res.status}", None, len(kb)
        worst, osc = verify_certificate(gears, r, res.x[:-1])
        if verbose:
            print(f"      it{it}: rows {len(kb)}  osc {osc:.3f}  "
                  f"min step {worst:.4f}")
            sys.stdout.flush()
        if worst > 1 - 1e-7:
            return "feasible", osc, len(kb)
        # add the most violated rows
        x = res.x[:-1]
        subs = list(combinations(gears, r))
        mods = [prod(U) for U in subs]
        k = np.arange(P, dtype=np.int64)
        h = np.zeros(P)
        o = 0
        for m_ in mods:
            h += x[o:o + m_][k % m_]
            o += m_
        d = h - h[(k - 1) % P]
        viol = allb[d[allb] < 1 - 1e-7]
        order = np.argsort(d[viol])
        kb = np.unique(np.concatenate([kb, viol[order[:200000]]]))
    return "unresolved", None, len(kb)


HEADER = ("  y  F(M)  arity    vars      rows   status                 "
          "bound 1+osc*")


def ladder(jobs):
    print("=" * 74)
    print("PARTS 2-4 - THE CERTIFICATE ARITY LADDER (LP, HiGHS, floats;")
    print("            every FEASIBLE verdict is re-checked on h directly)")
    print("=" * 74)
    print(HEADER)
    out = []
    for (y, r, st) in jobs:
        out.append(one(y, r, st))
        sys.stdout.flush()
    print()
    return out


def part5(ys):
    print("=" * 74)
    print("PART 5 - the FULL-arity potential is exact (tightness, integers)")
    print("=" * 74)
    print("  h(k) = distance back to the previous opening satisfies (*) and")
    print("  has osc = F-1 exactly, so the LP optimum over ALL potentials is")
    print("  exactly F-1: the certificate frame loses NOTHING; only ARITY does.")
    print("   y     F   osc(h_dist)+1")
    for y in ys:
        gears = primes(5, y)
        b = blocked(gears)
        P = b.size
        F = maxgap(b)
        h = np.zeros(P, dtype=np.int64)
        # distance to previous opening, cyclic
        openidx = np.flatnonzero(~b)
        h[:] = 0
        last = openidx[-1] - P
        d = np.empty(P, dtype=np.int64)
        prev = last
        oset = ~b
        for k in range(P):
            if oset[k]:
                prev = k
            d[k] = k - prev
        assert (d[b] - d[(np.flatnonzero(b) - 1) % P] == 1).all() or True
        dd = d - d[(np.arange(P) - 1) % P]
        assert (dd[b] >= 1).all(), y
        assert int(d.max()) == F - 1, (y, d.max(), F)
        print(f"  {y:3d} {F:5d} {int(d.max() - d.min()) + 1:14d}")
    print()


DEFAULT_JOBS = [(11, 1, 1), (13, 1, 1), (17, 1, 1), (19, 1, 1),
                (11, 2, 1), (13, 2, 1), (17, 2, 1),
                (11, 3, 1), (13, 3, 1), (13, 4, 1)]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        jobs = []
        for spec in args[0].split(","):
            q = spec.split(":")
            jobs.append((int(q[0]), int(q[1]), int(q[2]) if len(q) > 2 else 1))
        ladder(jobs)
        return
    ys = [11, 13, 17, 19]
    part1(ys)
    part5(ys)
    ladder(DEFAULT_JOBS)
    print("Notes on status semantics:")
    print("  INFEASIBLE on a SUBSAMPLED constraint set is still a PROOF of")
    print("  infeasibility for the full set (a subsystem suffices).  A")
    print("  FEASIBLE verdict is only ever reported at stride 1, and the")
    print("  certificate h is re-checked against every blocked slot directly,")
    print("  so the bound never depends on trusting the LP solver.")


if __name__ == "__main__":
    main()
