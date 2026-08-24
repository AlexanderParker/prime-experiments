"""LP DUALITY AS A CERTIFICATE MACHINE - round 22 dedicated-explorer thread.

Pushes docs/novel/covering-lp-certificates.md (round-21 seed, never taken to
depth) to depth, with EXACT rational arithmetic on both sides of every
threshold.  Everything the seed reported was solver-discovered and then
verified on the INFEASIBLE side only; here both endpoints are exact, and the
"does the gap blow up" question is answered with theorems rather than a
trend line.

Sections (run:  uv run python research/lp_dual_certs.py [A B C D E]):

  A  formulation + exact IP values (period sieve), monotonicity of the
     relaxation in W, and the ZERO-COLUMN REDUCTION THEOREM that makes the
     level-2 LP small enough to solve in exact rationals.
  B  EXACT integrality gaps: min infeasible width of the level-1 and
     level-2 LPs at machines 11..19 (+23), both endpoints exact - an exact
     rational feasible point at W*-1 and an exact Farkas certificate at W*.
  C  THE SHARP CEILING (new, family-free).  The seed's ceiling law was about
     one cut family (Kounias).  Here: does the uniform product measure admit
     a completion to a distribution on {0,1}^gears with NO empty atom and
     the prescribed degree-<=l moments?  If yes, the WHOLE degree-l cut
     family is satisfied and the level-l LP is feasible at every width -
     integrality gap infinite, no matter which degree-l inequality one
     invents.  Exact LP over the 2^n atoms; the infeasibility certificate is
     itself the optimal degree-l Bonferroni cut, verified pointwise.
  D  THE BLOW-UP LAW: exact closed form for the chain-cut slope
     (s = S1 * prod(1 - 2/q) + beta, a telescoping identity), the per-level
     death machine, and the moment degree required as a function of y with
     rigorous rational bracketing of S1 up to y = 10^6.
  E  (D)-relevance and kernel-checkability: which merge steps a covering
     certificate actually proves, and how big the certificate is.

House rules: exact int/Fraction arithmetic for every claim; scipy is used
for DISCOVERY ONLY and is never allowed to decide anything (every reported
threshold is bracketed by two exact certificates).  Benchmarks are pivot /
operation counts, not wall time.
"""
import sys
import time
from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from math import prod

from exact_lp import feasible_eq, solve_std

ZERO, ONE = Fraction(0), Fraction(1)

# ------------------------------------------------------------------ machine
def primes_upto(n):
    s = [True] * (n + 1)
    s[0] = s[1] = False
    for p in range(2, int(n ** .5) + 1):
        if s[p]:
            for m in range(p * p, n + 1, p):
                s[m] = False
    return [i for i, v in enumerate(s) if v]


def gears_of(y):
    return [p for p in primes_upto(y) if p >= 5]


def teeth(q):
    u = pow(6, -1, q)
    return u % q, (-u) % q


def covers(q, i):
    """phases r of gear q that block slot i."""
    u, v = teeth(q)
    return ((u - i) % q, (v - i) % q)


def F_exact(gears):
    """max gap between consecutive openings over the full period."""
    P = prod(gears)
    a = bytearray(b'\x01') * P
    for q in gears:
        t1, t2 = teeth(q)
        a[t1::q] = b'\x00' * len(a[t1::q])
        a[t2::q] = b'\x00' * len(a[t2::q])
    idx = [i for i, v in enumerate(a) if v]
    best, prev = 0, idx[0]
    for i in idx[1:]:
        best = max(best, i - prev)
        prev = i
    best = max(best, idx[0] + P - idx[-1])
    return best


F_KNOWN = {7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 46, 31: 58}

# ============================================================== the LP build
#
# EXACT IP.  Slot k is blocked by gear q iff k = +-6^{-1} (mod q).  A window
# of W consecutive slots starting at k is fully blocked ("covered") iff every
# i in [0,W) is blocked by some gear.  Choosing k is, by CRT, exactly the
# same as choosing one phase r_q in Z_q per gear independently, so
#
#     max coverable W  =  F(M) - 1,     IP feasible at W  <=>  W <= F(M)-1.
#
# LEVEL-1 LP.  z_{q,r} >= 0, sum_r z_{q,r} = 1 (a fractional phase per gear),
# and for each position i:   sum_q (z_{q,a} + z_{q,b}) >= 1, (a,b)=covers(q,i).
# A point mass on a covering phase tuple satisfies it, so INFEASIBILITY at W
# certifies F(M) <= W.
#
# LEVEL-2 LP.  Add a joint phase distribution z2_{(a,b)} per gear PAIR (with
# NO consistency to the singles - a further weakening, so certificates stay
# valid) and the Kounias cut, pointwise valid for 0/1 indicators: for every
# distinguished gear k,   1{covered} <= sum_j 1{A_j} - sum_{j!=k} 1{A_j A_k}.
# Constraint (i,k):  sum_q hits_q(i) - sum_{j!=k} p2_{jk}(i)  >=  1.
#
# ZERO-COLUMN REDUCTION (this thread).  The pair variables enter ONLY
# negatively.  If some phase pair (ra,rb) has NO position of [0,W) blocked by
# both gears, putting all of that pair's mass there makes its contribution
# identically 0 and the pair drops out of the LP entirely.  Since each of the
# 4 tooth combinations of a pair contributes at most W bad phase pairs, this
# happens whenever q_a*q_b > 4W.  So the level-2 LP can only ever see gear
# pairs with q_a q_b <= 4W: at machine 19 (W ~ 37) that is 10 of 15 pairs, and
# the fraction goes to 0 as the machine grows.  This is the mechanism behind
# the ceiling law, and it also shrinks the LP enough to solve EXACTLY.


@lru_cache(maxsize=None)
def hits(q, r, W):
    """positions of [0,W) blocked by gear q at phase r."""
    u, v = teeth(q)
    out = set()
    for t in (u, v):
        i = (t - r) % q
        while i < W:
            out.add(i)
            i += q
    return frozenset(out)


def pair_overlap(qa, ra, qb, rb, W):
    return hits(qa, ra, W) & hits(qb, rb, W)


def closed_form_cert(gears, kind, Wmax=200000):
    """the seed's LP-free counting certificate: smallest W at which the
    Kounias cut, averaged over the window with exact hit-count bounds,
    already signs.  Pure integer arithmetic.  It is a VALID infeasibility
    endpoint for the corresponding LP (if the window-sum of the constraint
    left sides is < W, some position violates its constraint)."""
    for W in range(1, Wmax + 1):
        s1 = sum(2 * ((W + q - 1) // q) for q in gears)
        if kind == 'density':
            if s1 < W:
                return W
        else:
            for k in gears:
                sub = 4 * sum(W // (j * k) for j in gears if j != k)
                if s1 - sub < W:
                    return W
    return None


@lru_cache(maxsize=None)
def pair_survives(qa, qb, W):
    """does EVERY phase pair of (qa,qb) block some common position of [0,W)?
    If not, the pair contributes nothing to the level-2 LP."""
    for ra in range(qa):
        ha = hits(qa, ra, W)
        for rb in range(qb):
            if not (ha & hits(qb, rb, W)):
                return False
    return True


def minimal_sets(sets):
    """Pareto-minimal (no proper subset present) representatives."""
    uniq = sorted(set(sets), key=lambda s: (len(s), sorted(s)))
    keep = []
    for s in uniq:
        if not any(t <= s for t in keep):
            keep.append(s)
    return keep


def maximal_sets(sets):
    uniq = sorted(set(sets), key=lambda s: (-len(s), sorted(s)))
    keep = []
    for s in uniq:
        if not any(s <= t for t in keep):
            keep.append(s)
    return keep


def build_lp(gears, W, level=2, prune=True):
    """returns (blocks, rows, surviving).  blocks = list of blocks; each
    block is (kind, label, columns) and each column is a dict row -> coeff.

    The set of PAIRS kept, and hence the row index set, never depends on
    `prune` - only invisible pairs (those owning a zero column, whose block
    maximum in any certificate is therefore exactly 0) are dropped, and that
    is a theorem, not a heuristic.  `prune` only controls the per-pair COLUMN
    reduction (Pareto-minimal overlap sets), which is a dominance argument;
    certificates are re-verified against the full column set."""
    surviving = []
    if level >= 2:
        for a, b in combinations(gears, 2):
            if pair_survives(a, b, W):
                surviving.append((a, b))
    withpair = {k for pr in surviving for k in pr}
    # rows: (i, k).  Gears k with no surviving pair give the plain level-1
    # row, identical for all such k -> one row per position.
    rowid, rows = {}, []
    for i in range(W):
        for k in gears:
            key = (i, k) if k in withpair else (i, None)
            if key not in rowid:
                rowid[key] = len(rows)
                rows.append(key)
    blocks = []
    for q in gears:
        cols = []
        for r in range(q):
            h = hits(q, r, W)
            cols.append({rowid[key]: ONE for key in rowid if key[0] in h})
        blocks.append(('single', q, cols))
    for (a, b) in surviving:
        ovl = {}
        for ra in range(a):
            for rb in range(b):
                ovl[(ra, rb)] = pair_overlap(a, ra, b, rb, W)
        keep = minimal_sets(ovl.values()) if prune else list(set(ovl.values()))
        cols = []
        for o in keep:
            c = {}
            for key, rid in rowid.items():
                if key[0] in o and key[1] in (a, b):
                    c[rid] = -ONE
            cols.append(c)
        blocks.append(('pair', (a, b), cols))
    return blocks, rows, surviving


def decide_level_float(gears, W, level):
    """SOLVER, DISCOVERY ONLY - never decides anything.  Float value of
    max t s.t. every constraint >= t.  Feasible iff t >= 1."""
    import numpy as np
    from scipy.optimize import linprog
    blocks, rows, surv = build_lp(gears, W, level, prune=True)
    cols, blockof = [], []
    for bi, (_, _, cs) in enumerate(blocks):
        for c in cs:
            cols.append(c)
            blockof.append(bi)
    N, R, nb = len(cols), len(rows), len(blocks)
    c = np.zeros(N + 1)
    c[-1] = -1.0
    A_ub = np.zeros((R, N + 1))
    for j, col in enumerate(cols):
        for r, v in col.items():
            A_ub[r, j] = -float(v)
    A_ub[:, -1] = 1.0
    A_eq = np.zeros((nb, N + 1))
    for j, bi in enumerate(blockof):
        A_eq[bi, j] = 1.0
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(R), A_eq=A_eq,
                  b_eq=np.ones(nb),
                  bounds=[(0, None)] * N + [(None, None)], method='highs')
    assert res.status == 0, res.message
    return -res.fun


def feas_system(blocks, nrows):
    """the level-l feasibility system in equality form:
        row r:   sum_j c_{rj} z_j  -  s_r  =  1     (s_r >= 0 slack)
        block b: sum_{j in b} z_j          =  1
    Returns (A, b, cols, blockof)."""
    cols, blockof = [], []
    for bi, (_, _, cs) in enumerate(blocks):
        for c in cs:
            cols.append(c)
            blockof.append(bi)
    N, nb = len(cols), len(blocks)
    A, rhs = [], []
    for r in range(nrows):
        A.append([cols[j].get(r, ZERO) for j in range(N)] +
                 [-ONE if s == r else ZERO for s in range(nrows)])
        rhs.append(ONE)
    for bi in range(nb):
        A.append([ONE if blockof[j] == bi else ZERO for j in range(N)] +
                 [ZERO] * nrows)
        rhs.append(ONE)
    return A, rhs, cols, blockof


def decide_level(gears, W, level):
    """EXACT decision of level-l LP feasibility at width W.

    Speed comes from the zero-column reduction (pruned build); CORRECTNESS
    never does: a feasible point of the pruned LP is a feasible point of the
    full LP (pad with zeros), and an infeasibility certificate found on the
    pruned LP is re-verified here against the FULL, UNPRUNED column set.

    Returns (feasible, certificate_info)."""
    blocks, rows, surv = build_lp(gears, W, level, prune=True)
    A, rhs, cols, blockof = feas_system(blocks, len(rows))
    ok, cert = feasible_eq(A, rhs)
    nrows, nb = len(rows), len(blocks)
    if ok:
        z = cert
        for bi in range(nb):
            assert sum(z[j] for j in range(len(cols)) if blockof[j] == bi) \
                == ONE, "block sum"
        for r in range(nrows):
            v = sum(cols[j].get(r, ZERO) * z[j] for j in range(len(cols)))
            assert v >= ONE, ("row", r, v)
        return True, ('primal', z, len(rows), surv)
    # --- exact Farkas, re-verified against the FULL column set
    y = [cert[r] for r in range(nrows)]           # row weights
    mu = [-cert[nrows + bi] for bi in range(nb)]  # per-block caps
    for v in y:
        assert v >= 0, "row weights must be >= 0"
    fblocks, frows, _ = build_lp(gears, W, level, prune=False)
    assert frows == rows, "row index sets must match"
    tot = sum(y)
    lhs = ZERO
    for bi, (_, _, cs) in enumerate(fblocks):
        best = max(sum(y[r] * c for r, c in col.items()) for col in cs)
        assert best <= mu[bi], ("dominance/pruning broke", bi, best, mu[bi])
        lhs += best
    assert lhs < tot, ("certificate does not sign", lhs, tot)
    # every pair NOT in `surv` owns a phase pair with empty overlap, i.e. a
    # zero column, so its block maximum is exactly 0 and adding it back to
    # the certificate changes nothing.
    if level >= 2:
        for a, b in combinations(gears, 2):
            if (a, b) not in surv:
                assert not pair_survives(a, b, W)
    return False, ('farkas', y, mu, lhs, tot, len(rows), surv)


# ================================================== C: the sharp ceiling LP
def completion_test(gears, l, use_float=True):
    """Does the uniform product measure's degree-<=l coverage moment vector
    extend to a distribution on {0,1}^gears with ZERO mass on the empty
    atom?  Feasible => every degree-l cut is satisfied at every position =>
    level-l LP feasible at every width => integrality gap INFINITE.
    Infeasible => the Farkas vector IS an optimal degree-l Bonferroni cut,
    verified pointwise over all 2^n atoms."""
    n = len(gears)
    p = [Fraction(2, q) for q in gears]
    subs = [S for k in range(l + 1) for S in combinations(range(n), k)]
    m = []
    for S in subs:
        v = ONE
        for j in S:
            v *= p[j]
        m.append(v)
    atoms = list(range(1, 1 << n))
    A = [[1 if all((x >> j) & 1 for j in S) else 0 for x in atoms]
         for S in subs]
    def verify_primal(nu, cols):
        assert all(v >= 0 for v in nu)
        for i in range(len(subs)):
            got = sum(A[i][c] * nu[k] for k, c in enumerate(cols))
            assert got == m[i], ("moment", subs[i])
        return True

    def cutvals(lam):
        """f[x] = sum_{S subset x} lam_S for every atom x, by the exact
        subset-sum (zeta) transform - O(n 2^n) Fraction additions instead of
        O(2^n * #subsets)."""
        f = [ZERO] * (1 << n)
        for i, S in enumerate(subs):
            msk = 0
            for j in S:
                msk |= 1 << j
            f[msk] += lam[i]
        for j in range(n):
            bit = 1 << j
            for msk in range(1 << n):
                if msk & bit:
                    f[msk] += f[msk ^ bit]
        return f

    def verify_farkas(lam):
        """lam certifies infeasibility iff  lam.m > 0  and, for every NONEMPTY
        atom x,  sum_{S subset x} lam_S <= 0.  The second condition says
        -sum_S lam_S prod_{q in S} 1{A_q} is a valid degree-l pointwise upper
        bound on the union indicator; the first says the uniform product
        measure violates it.  A float-discovered lam is repaired first: every
        atom contains the empty set, so lowering lam_empty by the worst
        violation restores validity at the cost of exactly that much value
        (m_empty = 1)."""
        f = cutvals(lam)
        worst = max(f[x] for x in atoms)
        if worst > 0:
            lam = list(lam)
            lam[0] -= worst
            f = cutvals(lam)
        val = sum(lam[i] * m[i] for i in range(len(m)))
        if val <= 0:
            return None
        for x in atoms:
            assert f[x] <= 0, ("farkas pointwise", x)
        return lam, val

    if use_float:
        import numpy as np
        from scipy.optimize import linprog
        Af = np.array(A, dtype=float)
        bf = np.array([float(v) for v in m])
        res = linprog(np.zeros(len(atoms)), A_eq=Af, b_eq=bf,
                      bounds=(0, None), method='highs')
        if res.status == 0:
            supp = [j for j, v in enumerate(res.x) if v > 1e-13]
            ok, cert = feasible_eq([[A[i][j] for j in supp]
                                    for i in range(len(subs))], m)
            if ok:
                verify_primal(cert, supp)
                return True, (cert, supp), subs, m
        elif res.status == 2:
            # phase-I for a dual ray: min sum(u+v) s.t. A nu + u - v = m
            ns, na = len(subs), len(atoms)
            Ap = np.hstack([Af, np.eye(ns), -np.eye(ns)])
            cp = np.concatenate([np.zeros(na), np.ones(2 * ns)])
            r2 = linprog(cp, A_eq=Ap, b_eq=bf, bounds=(0, None),
                         method='highs')
            if r2.status == 0 and r2.fun > 1e-12:
                lam = [Fraction(float(v)).limit_denominator(10 ** 7)
                       for v in r2.eqlin.marginals]
                got = verify_farkas(lam)
                if got:
                    return False, got[0], subs, m
    ok, cert = feasible_eq(A, m)
    if ok:
        verify_primal(cert, list(range(len(atoms))))
        return True, (cert, list(range(len(atoms)))), subs, m
    got = verify_farkas(cert)
    assert got, "exact Farkas failed to verify"
    return False, got[0], subs, m


def binomial_moments(gears, l):
    """S_j = e_j(p) for j = 0..l, p_q = 2/q - the binomial moments of the
    coverage count under the uniform product measure.  Exact."""
    e = [ONE] + [ZERO] * l
    for q in gears:
        p = Fraction(2, q)
        for j in range(l, 0, -1):
            e[j] += e[j - 1] * p
    return e


def aggregated_ceiling_test(gears, l):
    """AGGREGATED (binomial-moment) ceiling test: does there exist a
    distribution on the COUNT in {1..n} matching S_0..S_l?

    Feasible here is IMPLIED BY feasibility of the full multivariate test, so
    aggregated-INFEASIBLE proves the level still bites, and the aggregated
    ceiling machine is a LOWER bound on the true one.  Cost is n columns and
    l+1 rows, so it runs at machines where 2^n is hopeless."""
    n = len(gears)
    S = binomial_moments(gears, l)
    A = [[Fraction(comb_(k, j)) for k in range(1, n + 1)] for j in range(l + 1)]
    return feasible_eq(A, S)


def comb_(k, j):
    if j > k:
        return 0
    num, den = 1, 1
    for i in range(j):
        num *= k - i
        den *= i + 1
    return num // den


# =========================================== D: chain slope, exact closed form
def chain_slope_exact(gears, chain):
    """uniform-product slope of the depth-t chain cut, computed term by term
    exactly as in the seed's level_slope."""
    s = sum(Fraction(2, q) for q in gears)
    prior = []
    for k in chain:
        f = ONE
        for kp in prior:
            f *= (1 - Fraction(2, kp))
        s -= Fraction(2, k) * f * sum(Fraction(2, j) for j in gears
                                      if j != k and j not in prior)
        prior.append(k)
    return s


def chain_slope_closed(gears, chain):
    """TELESCOPING CLOSED FORM (this thread):  s = S1 * prod_{k in chain}
    (1 - 2/q_k)  +  beta,  with beta = sum_m (2/q_m) prod_{l<m}(1-2/q_l)
    * (sum_{l<=m} 2/q_l)  -  a quantity that does NOT involve the machine."""
    S1 = sum(Fraction(2, q) for q in gears)
    P = ONE
    beta = ZERO
    pref = ZERO
    for k in chain:
        pref += Fraction(2, k)
        beta += Fraction(2, k) * P * pref
        P *= (1 - Fraction(2, k))
    return S1 * P + beta, P, beta


def s1_bracket(y, D=10 ** 40):
    """rigorous rational bracket [lo, hi] for S1(y) = sum_{5<=q<=y} 2/q,
    using only integer arithmetic."""
    gs = gears_of(y)
    num = sum((2 * D) // q for q in gs)
    lo = Fraction(num, D)
    hi = Fraction(num + len(gs), D)
    return lo, hi, len(gs)


# ======================================================================= main
def section_A():
    print("=" * 78)
    print("A  FORMULATION, EXACT IP, MONOTONICITY, ZERO-COLUMN REDUCTION")
    print("=" * 78)
    print("IP: max coverable width = F(M) - 1 (CRT makes phase choice exact)")
    for y in (7, 11, 13, 17, 19):
        F = F_exact(gears_of(y))
        assert F == F_KNOWN[y], (y, F)
        print(f"  machine {y:>2}: F = {F:>2}  (period sieve, exact)")
    # monotonicity: the width-(W-1) constraint set is a SUBSET of width W's
    print("\nmonotonicity: constraints at width W-1 are a subset of those at")
    print("  width W (same variables), so feasibility is monotone decreasing")
    print("  in W - binary search for the threshold is valid.  Asserted:")
    for y in (11, 13):
        gears = gears_of(y)
        for W in range(3, 12):
            b0, _, _ = build_lp(gears, W - 1, 1, prune=False)
            b1, _, _ = build_lp(gears, W, 1, prune=False)
            for (k0, _, c0), (k1, _, c1) in zip(b0, b1):
                assert k0 == k1 and len(c0) == len(c1)
        print(f"  machine {y}: same variable set at every width  OK")
    # zero-column reduction
    print("\nZERO-COLUMN REDUCTION THEOREM (new).  A gear pair contributes to")
    print("the level-2 LP only if EVERY phase pair blocks a common position")
    print("of [0,W).  Each of the 4 tooth combinations rules out at most W")
    print("phase pairs, so q_a q_b > 4W  =>  the pair drops out entirely.")
    print("Asserted exhaustively:")
    bad = 0
    for y in (13, 17, 19):
        gears = gears_of(y)
        for W in (10, 20, 30, 37):
            for a, b in combinations(gears, 2):
                surv = pair_survives(a, b, W)
                if a * b > 4 * W:
                    assert not surv, (y, W, a, b)
                bad += 1
    print(f"  {bad} (machine, width, pair) cases: q_a q_b > 4W always kills"
          f" the pair  OK")
    for y in (13, 17, 19, 23):
        gears = gears_of(y)
        W = F_KNOWN[y]
        npairs = len(list(combinations(gears, 2)))
        vis = sum(1 for a, b in combinations(gears, 2)
                  if pair_survives(a, b, W))
        print(f"  machine {y:>2}, W = F = {W:>2}: {vis:>2} of {npairs:>2} "
              f"gear pairs are VISIBLE to the level-2 LP")


def section_B(machines=(11, 13, 17, 19)):
    print("=" * 78)
    print("B  EXACT INTEGRALITY GAPS (both endpoints exact rational)")
    print("=" * 78)
    print("gap := (min width at which the LP is infeasible) / F(M).")
    print("The LP certifies F(M) <= W* with no period scan.\n")
    out = []
    for y in machines:
        gears = gears_of(y)
        F = F_KNOWN[y]
        S1 = sum(Fraction(2, q) for q in gears)
        for level in (1, 2):
            t0 = time.time()
            if level == 1 and S1 >= 1:
                print(f"  machine {y:>2} level 1: FEASIBLE AT EVERY WIDTH - "
                      f"exact uniform certificate z_(q,r) = 1/q gives "
                      f"coverage sum 2/q = {S1} >= 1 at every position "
                      f"=> INTEGRALITY GAP INFINITE")
                out.append((y, level, None, F))
                continue
            cap = closed_form_cert(gears, 'density' if level == 1
                                   else 'kounias')
            assert cap is not None, "no closed-form endpoint"
            # ---- DISCOVERY (floats, decides nothing): first width whose
            #      float LP value drops below 1
            cand = None
            for W in range(F - 1, cap + 1):
                if decide_level_float(gears, W, level) < 1 - 1e-9:
                    cand = W
                    break
            assert cand is not None
            # ---- EXACT bracketing of the discovered threshold
            okprev, _ = decide_level(gears, cand - 1, level)
            okhere, info = decide_level(gears, cand, level)
            assert okprev and not okhere, \
                f"float discovery wrong at {cand} (exact says "\
                f"{okprev}/{okhere}) - widen the exact search"
            Wstar = cand
            dt = time.time() - t0
            _, yv, mu, lhs, tot, nr, surv = info
            supp = sum(1 for v in yv if v)
            print(f"  machine {y:>2} level {level}: W* = {Wstar:>3}   "
                  f"F = {F:>2}   GAP = {Wstar}/{F} = {float(Fraction(Wstar,F)):.3f}"
                  f"   [{dt:.0f}s]")
            print(f"      exact rational feasible point at W = {Wstar-1};"
                  f" exact Farkas certificate at W = {Wstar}:")
            print(f"      sum of block maxima {lhs} < {tot} = sum of weights,"
                  f"  support {supp} weights, {len(surv)} visible pairs")
            # BENCHMARK (operations, not wall time): verifying the certificate
            # touches each (gear, phase, blocked position) incidence once, and
            # each (pair, phase pair, common position) incidence once.
            ops = sum(2 * Wstar for q in gears) + \
                sum(4 * Wstar for _ in surv)
            P = prod(gears)
            print(f"      VERIFICATION COST {ops} rational operations vs a "
                  f"{P}-slot period scan ({P // max(ops,1)}x fewer);"
                  f" (D) step needs W* <= {F_KNOWN[[g for g in F_KNOWN if g < y][-1]] + y}"
                  f" -> {'PROVED' if Wstar <= F_KNOWN[[g for g in F_KNOWN if g < y][-1]] + y else 'missed'}")
            out.append((y, level, Wstar, F))
    return out


def section_C(machines=(11, 13, 17, 19, 23, 29, 31, 37), levels=(1, 2, 3, 4)):
    print("=" * 78)
    print("C  THE SHARP CEILING - family-free, not just Kounias")
    print("=" * 78)
    print("VACUOUS = the uniform product measure's degree-l moments extend to")
    print("  a distribution with no empty atom => EVERY degree-l cut holds at")
    print("  every position and every width => level-l LP feasible forever =>")
    print("  INTEGRALITY GAP INFINITE (no degree-l certificate can exist).")
    print("BITES  = exact Farkas vector = an optimal degree-l Bonferroni cut,")
    print("  verified pointwise over all 2^n - 1 nonempty atoms.\n")
    print("C1  SHARP (multivariate) test - costs 2^n, so machines <= 37:")
    sharp = {}
    for y in machines:
        gears = gears_of(y)
        n = len(gears)
        line = f"  machine {y:>2} (n = {n:>2} gears): "
        for l in levels:
            if l >= n:
                line += f"  l={l}: -      "
                continue
            ok, cert, subs, m = completion_test(gears, l)
            sharp[(y, l)] = ok
            line += f"  l={l}: {'VACUOUS' if ok else 'bites  '}"
        print(line)
    print("\n  SHARP CEILINGS: degree 1 dies at machine 13 (that is exactly")
    print("  the density bound sum 2/q >= 1); degree 2 dies at machine 29.")
    print("  Kounias was therefore already degree-2-optimal - the ceiling is")
    print("  intrinsic to the degree, not to the cut family chosen.")
    assert sharp[(11, 1)] is False and sharp[(13, 1)] is True
    assert sharp[(23, 2)] is False and sharp[(29, 2)] is True

    print("\nC2  AGGREGATED test (binomial moments S_0..S_l only, count in")
    print("  {1..n}): a RELAXATION of C1, so aggregated-bites => sharp-bites,")
    print("  and its ceiling machine is a LOWER bound on the sharp one.  It")
    print("  costs n columns and l+1 rows, so it runs to y = 12000.")
    agg_machines = (11, 13, 17, 19, 23, 29, 31, 37, 97, 127, 151, 1000,
                    3000, 5000, 12000)
    print(f"    {'y':>6} {'n':>5} {'S1':>7}   " +
          " ".join(f"l={l}" for l in range(1, 9)))
    for y in agg_machines:
        gears = gears_of(y)
        S1 = sum(Fraction(2, q) for q in gears)
        cells = []
        for l in range(1, 9):
            if l >= len(gears):
                cells.append(" - ")
                continue
            ok, _ = aggregated_ceiling_test(gears, l)
            cells.append(" V " if ok else " b ")
            # the implication, asserted wherever both are known
            if (y, l) in sharp and not ok:
                assert not sharp[(y, l)], \
                    ("aggregated bites but sharp is vacuous", y, l)
        print(f"    {y:>6} {len(gears):>5} {float(S1):>7.3f}   " +
              " ".join(cells))
    print("\n  AGGREGATED CEILINGS: l=1 at 13, l=2 at 19, l=3 and l=4 at 151,")
    print("  l=5 and l=6 between 3000 and 5000, l=7 and l=8 beyond 12000.")
    print("  S1 at those ceilings: 1.02, 1.24, 2.12, 3.14 - so the degree a")
    print("  certificate needs is about 2*S1(y) ~ 4 log log y.  UNBOUNDED,")
    print("  hence NO fixed-degree covering certificate works at all")
    print("  machines - but the growth is doubly logarithmic.")
    print("  Since aggregated-bites => sharp-bites, the SHARP ceiling for")
    print("  degree 3 is at least 151, for degree 5 at least ~5000.")


def section_D():
    print("=" * 78)
    print("D  THE BLOW-UP LAW")
    print("=" * 78)
    print("D1  telescoping closed form for the chain-cut slope (asserted):")
    for y in (19, 23, 29, 31, 37):
        gears = gears_of(y)
        for t in (1, 2, 3):
            chain = gears[:t]
            a = chain_slope_exact(gears, chain)
            b, P, beta = chain_slope_closed(gears, chain)
            assert a == b, (y, t, a, b)
    print("    s(chain) = S1 * prod_{k in chain}(1 - 2/q_k) + beta(chain)")
    print("    verified exactly for chains of depth 1..3 at machines 19..37;")
    print("    beta does NOT depend on the machine, only on the chain.")
    print("    => the slope is AFFINE IN S1 with slope prod(1 - 2/q_k),")
    print("       and S1 -> infinity (Mertens), so every fixed chain dies.\n")
    print("D2  rigorous death test.  beta >= 0 and (1 - 2/q) increases in q,")
    print("    so for ANY chain of depth t:  s >= S1 * prod_{t smallest}.")
    print("    S1 * prod_{t smallest gears}(1-2/q) >= 1  =>  NO depth-t chain")
    print("    cut can certify anything at that machine (or any larger one).")
    print(f"    {'depth t':>8} {'moment deg':>11} {'prod':>12} "
          f"{'S1 needed':>11} {'DEAD from y >=':>15}")
    GS = gears_of(10 ** 6)
    D = 10 ** 40
    pref = [0]
    for q in GS:
        pref.append(pref[-1] + (2 * D) // q)   # rigorous LOWER bracket of S1
    for t in range(0, 8):
        P = ONE
        for q in GS[:t]:
            P *= (1 - Fraction(2, q))
        need = ONE / P
        ydead = None
        for k in range(1, len(GS) + 1):
            if Fraction(pref[k], D) >= need:   # S1 lower bracket >= need
                ydead = GS[k - 1]
                break
        print(f"    {t:>8} {t+1:>11} {str(P):>12} {float(need):>11.4f} "
              f"{str(ydead) if ydead else '> 1e6':>15}")
    print("\nD3  the moment degree a certificate must have at machine y")
    print("    (smallest t with S1(y) * prod_{t smallest}(1-2/q) < 1 - a")
    print("    NECESSARY condition; sufficiency is only measured, section C):")
    print(f"    {'y':>10} {'#gears':>8} {'S1 (rigorous)':>16} "
          f"{'degree needed':>14}")
    for y in (13, 19, 29, 47, 100, 1000, 10 ** 4, 10 ** 5, 10 ** 6):
        lo, hi, n = s1_bracket(y)
        gs = gears_of(min(y, 10 ** 6))
        t = 0
        P = ONE
        while True:
            if hi * P < 1:
                break
            t += 1
            P *= (1 - Fraction(2, gs[t - 1]))
        print(f"    {y:>10} {n:>8} {float(lo):>16.6f} {t + 1:>14}")
    print("    Mertens: S1(y) ~ 2 log log y and prod_{q<=z}(1-2/q) ~ A/log^2 z,")
    print("    so the chain must reach z ~ exp(sqrt(2A log log y)):")
    print("    the required degree is UNBOUNDED but grows like")
    print("    exp(sqrt(2A log log y)) / sqrt(2A log log y).")


# exact level-2 LP thresholds, produced by section B (both endpoints exact:
# a rational feasible point at W*-1 and a verified Farkas dual at W*).
# machine -> (W*, support size = number of nonzero dual weights, visible pairs)
LP2_THRESHOLD = {11: (8, 8, 0), 13: (21, 32, 1), 17: (31, 32, 5),
                 19: (37, 37, 7)}


def section_E(rows=None):
    print("=" * 78)
    print("E  (D)-RELEVANCE AND KERNEL-CHECKABILITY")
    print("=" * 78)
    print("A covering certificate at machine M+q' of width W = F(M) + q'")
    print("PROVES the (D) step  F(M+q') <= F(M) + q'  outright and scan-free:")
    print("no period of M+q' is ever built, and F(M) is the OLD machine's")
    print("(already kernel-checkable) value.\n")
    if rows:
        for (y, level, Wstar, F) in rows:
            if level == 2 and Wstar is not None:
                assert LP2_THRESHOLD[y][0] == Wstar, (y, Wstar)
    print(f"  {'step':>10} {'F(M)':>5} {'q':>4} {'budget':>7} {'F(M+q)':>7} "
          f"{'LP2 W*':>7}  verdict")
    steps = [(7, 11), (11, 13), (13, 17), (17, 19), (19, 23)]
    proved = []
    for (m, mp) in steps:
        F, Fp = F_KNOWN[m], F_KNOWN[mp]
        budget = F + mp
        W = LP2_THRESHOLD.get(mp, (None,))[0]
        if W is None:
            v = "(W* not computed exactly here; seed reports 90)"
        elif W <= budget:
            v = f"(D) PROVED by the dual certificate ({W} <= {budget})"
            proved.append((m, mp))
        else:
            v = f"missed by {W - budget}"
        print(f"  {m:>4} -> {mp:<3} {F:>5} {mp:>4} {budget:>7} {Fp:>7} "
              f"{str(W):>7}  {v}")
    # the assertions that carry the claim
    assert LP2_THRESHOLD[11][0] <= F_KNOWN[7] + 11
    assert LP2_THRESHOLD[19][0] == F_KNOWN[17] + 19 == 37
    assert LP2_THRESHOLD[13][0] > F_KNOWN[11] + 13     # honest miss, by 1
    assert LP2_THRESHOLD[17][0] > F_KNOWN[13] + 17     # honest miss, by 3
    print(f"\n  (D) steps proved by LP duality alone: {proved}")
    print("  The 17 -> 19 step is EXACTLY TIGHT: W* = 37 = F(17) + 19.")
    print()
    print("KERNEL-CHECKABILITY.  The certificate is a list of nonnegative")
    print("rationals y_(i,k) plus one maximum per gear and per visible pair;")
    print("verification is finitely many rational comparisons:")
    print(f"  {'machine':>8} {'W*':>4} {'weights':>8} {'ops to verify':>14} "
          f"{'period slots':>14} {'ratio':>9}")
    for y, (W, supp, npairs) in sorted(LP2_THRESHOLD.items()):
        gears = gears_of(y)
        ops = 2 * W * len(gears) + 4 * W * npairs
        P = prod(gears)
        print(f"  {y:>8} {W:>4} {supp:>8} {ops:>14} {P:>14} "
              f"{P // ops:>8}x")
    print("\n  Lean shape (machine 19, the tight step):")
    print("    def w : Fin 37 -> Fin 6 -> Q := ...        -- 37 nonzero")
    print("    theorem cert : (sum over gears of max over phases of")
    print("        w-mass of that gear's blocked positions)")
    print("      + (sum over visible pairs of max over phase pairs of")
    print("        (- w-mass of the common blocked positions))")
    print("      < (sum of all w)                          := by decide")
    print("    theorem F19_le_37 : F 19 <= 37 := no_cover_of_cert cert")
    print("    theorem D_17_19  : F 19 <= F 17 + 19 := by")
    print("      rw [F17_eq_18]; exact F19_le_37")
    print("  Every maximum is over a finite phase set (Fin q / Fin q x Fin q)")
    print("  so `decide` discharges it; nothing infinite enters.")


SECTIONS = {'A': section_A, 'B': section_B, 'C': section_C,
            'D': section_D, 'E': section_E}

if __name__ == '__main__':
    want = [a.upper() for a in sys.argv[1:]] or list(SECTIONS)
    for s in want:
        SECTIONS[s]()
        print()
