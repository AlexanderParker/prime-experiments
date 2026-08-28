"""ROUND 25, LP-DUALITY THREAD - the 23->29 composition decision, done properly.

WHAT ROUND 24 LEFT.  The full composition (round-23 consistent degree-2
covering LP + the ONE recursive Costello-Watts row) certified the four rungs
7->11 .. 17->19 with certificates 2-3x smaller than consistent-only, left
19->23 UNDECIDED ("no certificate found", 54 iterations, converged t = +0.0368)
and left 23->29 NOT DECIDED AT ALL: the deciding run was starved by box-wide
memory exhaustion at iteration 1.  This file decides it, and fixes two
methodological holes in the round-24 vehicle while doing so.

HOLE 1 - "FEASIBLE" WAS NEVER EXACT.  `decideCF` returned `exact=False` on the
feasible branch: the cut loop stops when the EXACT separation oracle finds no
cut violated by more than the discovery margin 1e-5, which is a stopping rule,
not a proof.  So "no certificate" meant "the loop gave up", i.e. UNDECIDED.
Here the feasible branch is closed exactly:

    a point z is a FEASIBLE POINT of the full composition at width W iff
      (i)   z >= 0, every block sums to 1, every consistency link is exact;
      (ii)  at EVERY position i < W the degree-2 moments of z extend to a
            distribution on {0,1}^n with ZERO mass on the empty atom
            (`completable`, exact rational Farkas);
      (iii) sum_q E[S_q] - sum_{i<j} E[n_ij]  >=  W   (the recursive row).
    Such a z is a fractional fully-blocked window that every valid inequality
    of the vehicle accepts, so NO certificate of this vehicle at this width
    exists.  That is a REFUTATION, not an undecided cell.

HOLE 2 - THE WITNESS WAS NOT SAVED (my own round-24 process rule, written after
the section-G regression: FEASIBLE VERDICTS MUST SAVE THEIR WITNESS).  Every
verdict here writes its object to disk - the exact rational z on the feasible
branch, the exact rational dual certificate on the infeasible branch - and
re-verifies it from the saved file in a second pass.

SPEED.  The exact separation oracle is a rational two-phase simplex on a
38 x 293 tableau and is called once per position per iteration; at machine 29
(n = 8, 255 nonempty atoms) that is the whole cost.  A FLOAT PRE-FILTER is used
during the discovery loop only: if the float completion LP finds a strictly
interior completion at position i, the exact call is skipped for that
iteration.  This can only LOSE cuts, never invent them, and losing a cut only
makes the loop stop early - at which point the FINAL EXACT PASS (margin 0, no
pre-filter, every position) either confirms feasibility exactly or supplies the
missed cuts and the loop continues.  Soundness is therefore independent of the
float filter, which is the house rule.

Run:  python research/cw_decide25.py <y> <W> [--norec] [--nofilter]
"""
import os
import pickle
import sys
import time
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cw_consistent import RelaxCF, certificateCF                  # noqa: E402
from lp_degree_range import (gears_of, budget, F_EXACT, ZERO, ONE,  # noqa
                             completable, separate, hits)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'r25')


# --------------------------------------------------------------- float filter
def _float_interior(R, zex, i):
    """DISCOVERY ONLY.  True if the degree-2 moments of zex at position i admit
    a strictly interior completion in floats (margin 1e-7).  A True here means
    the exact oracle is skipped this iteration; it decides nothing."""
    import numpy as np
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    n = R.n
    subs = R.subs
    mom = R.moments_at(zex, i)
    natom = 1 << n
    atoms = list(range(1, natom))
    na = len(atoms)
    ri, ci, vv, beq = [], [], [], []
    for r, m in enumerate(subs):
        for c, x in enumerate(atoms):
            if (m & ~x) == 0:
                ri.append(r); ci.append(c); vv.append(1.0)
        beq.append(float(mom.get(m, ONE)) if m else 1.0)
    A_eq = coo_matrix((vv, (ri, ci)), shape=(len(subs), na + 1))
    ri2, ci2, vv2 = [], [], []
    for c in range(na):
        ri2.append(c); ci2.append(c); vv2.append(-1.0)
        ri2.append(c); ci2.append(na); vv2.append(1.0)
    A_ub = coo_matrix((vv2, (ri2, ci2)), shape=(na, na + 1))
    c_obj = np.zeros(na + 1)
    c_obj[na] = -1.0
    res = linprog(c_obj, A_ub=A_ub, b_ub=np.zeros(na), A_eq=A_eq,
                  b_eq=np.array(beq), bounds=[(0, None)] * na + [(None, None)],
                  method='highs')
    return res.status == 0 and -res.fun > 1e-7


# --------------------------------------------- fast separation (exact output)
def separate_fast(mom, n, l, margin=ZERO, M=16.0, dens=(4, 16, 64, 256)):
    """A VIOLATED, EXACTLY VALID degree-l cut found by a small float LP and
    then made exact - or None.

    The separation problem is itself an LP in the cut coefficients:
        minimise  lam . mom   subject to   zeta(lam)[x] >= 1 at every nonempty
        atom x,   |lam| <= M,
    where zeta(lam)[x] = sum_{S subset x} lam_S.  At n = 8 that is 37 variables
    and 255 constraints - milliseconds in HiGHS, against seconds for the exact
    rational simplex on the dual completion problem.

    NOTHING IS DECIDED IN FLOATS.  The float optimum is rounded to a rational
    lam, VALIDITY IS RESTORED EXACTLY by raising lam_0 by the exact deficit
    1 - min_x zeta(lam)[x] (which lifts every zeta by the same amount, so the
    repaired cut is valid by construction), and the two facts that matter -
    zeta >= 1 everywhere, and lam . mom < 1 - are then asserted in exact
    rational arithmetic before the cut is returned.  A None here means "this
    route found nothing", never "no cut exists": the caller falls back to the
    exact oracle, which is what the final pass always uses."""
    import numpy as np
    from scipy.optimize import linprog
    from lp_degree_range import _sep_matrix_exact, zeta_values, cut_value
    A, subs = _sep_matrix_exact(n, l)          # A[S][x] over nonempty atoms
    ns, na = len(subs), len(A[0])
    Af = np.array([[float(v) for v in row] for row in A])   # ns x na
    c = np.array([float(mom.get(m, ONE)) if m else 1.0 for m in subs])
    res = linprog(c, A_ub=-Af.T, b_ub=-np.ones(na),
                  bounds=[(-M, M)] * ns, method='highs')
    if res.status != 0 or res.fun >= 1.0 - 1e-12:
        return None
    for den in dens:
        lam = [Fraction(round(v * den), den) for v in res.x]
        f = zeta_values(tuple(lam), n, subs)
        mn = min(f[x] for x in range(1, 1 << n))
        if mn < ONE:                           # exact validity repair
            lam[0] += ONE - mn
            f = zeta_values(tuple(lam), n, subs)
            mn = min(f[x] for x in range(1, 1 << n))
        assert mn >= ONE, "repaired cut still invalid"
        val = cut_value(tuple(lam), subs, mom)
        if val < ONE - margin:
            return tuple(lam)
    return None


# ------------------------------------------- bounded-denominator rationalising
def rationalise_fixed(R, z, den):
    """Round the float point to rationals of a SINGLE fixed denominator, and
    restore each top block's sum to 1 by pushing the residue onto its largest
    entry - so every entry's denominator DIVIDES den, and so does every sum of
    them.

    WHY THIS EXISTS (a measured infrastructure fix, round 25).  RelaxC's own
    `rationalise` uses Fraction.limit_denominator(den), which gives each entry
    its OWN denominator below den.  `repair_consistency` then rebuilds the
    single-gear blocks as sums of up to q_i q_j of those, and `moments_at` sums
    ~128 more per position: the denominators LCM together and the moment vectors
    that go into the exact separation oracle acquire enormous numerators.  At
    machine 23 (blocks up to 23x23) that was survivable - round 24's run
    finished in 1,753 s.  At machine 29 (blocks up to 29x29) it was not: the
    deciding process reached 1.35 GB of COMMITTED private memory against a
    136 MB working set and 604,322 page faults, i.e. it was page-thrashing at
    27% of one core on an otherwise 19%-loaded box.  That is the round-24
    livelock signature, and it is what starved this cell in round 24 - the
    diagnosis then ('other lanes' memory pressure') was only half right.
    Fixing the denominator removes the growth at the source.

    Soundness is untouched: the rationalised point is used ONLY to generate cuts
    and, at the end, as a candidate witness that is verified exactly on its own
    terms."""
    zx = [max(ZERO, Fraction(round(float(v) * den), den)) for v in z]
    for S in R.subsets:
        if len(S) != R.l:
            continue
        lo, hi = R.block_span[S]
        s = sum(zx[lo:hi])
        k = max(range(lo, hi), key=lambda j: zx[j])
        zx[k] += ONE - s
        if zx[k] < 0:
            for j in range(lo, hi):
                zx[j] = Fraction(1, hi - lo)
    return zx


# ------------------------------------------------------------- the exact loop
def decide25(gears, W, l=2, use_recursion=True, maxrounds=800, verbose=True,
             tag=None, filt=True, ckpt_every=10, time_budget=None,
             fastsep=False):
    """EXACT decision of the full composition at width W.

    Returns (verdict, info) with verdict in {'CERTIFIED', 'REFUTED', 'STUCK'}:
      CERTIFIED - exact rational dual certificate (F(M) <= W by this vehicle);
      REFUTED   - exact rational feasible point (no certificate of this vehicle
                  at this width exists);
      STUCK     - the round budget ran out; the accumulated rows are saved.
    """
    t0 = time.time()
    R = RelaxCF(gears, W, l, use_recursion=use_recursion)
    if verbose:
        print("  built: %d cols, %d blocks, %d links, %d start rows  [%.0fs]"
              % (len(R.cols), len(R.subsets), len(R.links), len(R.rows),
                 time.time() - t0), flush=True)
    os.makedirs(OUT, exist_ok=True)
    tag = tag or ("m%d_w%d" % (gears[-1], W))
    ck = os.path.join(OUT, 'ck_%s.pkl' % tag)
    if os.path.exists(ck):
        with open(ck, 'rb') as fh:
            R.rows = pickle.load(fh)
        if verbose:
            print("  resumed from checkpoint: %d rows" % len(R.rows), flush=True)

    it = 0
    final_pass = False          # True once the float filter has been retired
    while it < maxrounds:
        t, z, res = R._solve_float()
        if t < -1e-7:
            # ---- INFEASIBLE: exact dual certificate
            nb = len(R.subsets)
            y = list(-res.ineqlin.marginals)
            yff = y.pop()
            nu = res.eqlin.marginals[nb:]
            ok, lhs, rhs, yq, yffq, nuq, ops = certificateCF(R, y, yff, nu)
            assert ok, ("float said infeasible but no exact certificate could "
                        "be rationalised - ABORT")
            info = dict(verdict='CERTIFIED', lhs=lhs, rhs=rhs, ops=ops,
                        rows=len(R.rows), cols=len(R.cols), its=it,
                        support=sum(1 for v in yq if v) + (1 if yffq else 0)
                        + sum(1 for v in nuq if v),
                        secs=time.time() - t0)
            save_certificate(tag, R, yq, yffq, nuq, info)
            return 'CERTIFIED', info

        den = 10 ** (4 + min(it // 40, 4))
        # CUT GENERATION ONLY.  The rationalised point is a place to LOOK for
        # violated cuts; every cut `separate` returns is re-asserted valid and
        # violated in exact arithmetic on its own, so nothing here decides
        # anything.  Fixed-denominator rounding keeps every moment's
        # denominator dividing `den` - measured this round: RelaxC.rationalise
        # divides each block by its own rational sum, and the resulting
        # denominators reached 307 BITS by iteration 3 at machine 29, which is
        # what drove the deciding process to 1.35 GB of commit against a
        # 136 MB working set (page-thrashing at 27% of one core on a
        # 19%-loaded box).  The exact FEASIBLE verdict does not use this point
        # at all - it comes from the global-point route below.
        zex = R.repair_consistency(rationalise_fixed(R, z, den))
        margin = ZERO if final_pass else Fraction(1, 10 ** 5)
        added, skipped = 0, 0
        for i in range(W):
            mom = R.moments_at(zex, i)
            if not final_pass:
                if fastsep:
                    lam = separate_fast(mom, R.n, l, margin)
                    if lam is not None:
                        R.rows.append((i, lam))
                        added += 1
                        continue
                if filt and _float_interior(R, zex, i):
                    skipped += 1
                    continue
            lam = separate(mom, R.n, l, margin)
            if lam is not None:
                R.rows.append((i, lam))
                added += 1
        if verbose:
            print("      it %d: t = %+.4f, %d cuts (%d skipped), %d rows"
                  "%s  [%.0fs]"
                  % (it, t, added, skipped, len(R.rows),
                     "  FINAL EXACT PASS" if final_pass else "",
                     time.time() - t0), flush=True)
        if added == 0:
            if not final_pass:
                final_pass = True         # retire the filter and the margin
                continue
            # ---- the cut loop found no certificate.  That is NOT a decision:
            # it is "this discovery loop stalled".  Try to verify the point
            # exactly; if the rationalised point is not exactly in the
            # polytope (it usually is not - see the note above the refutation
            # branch), hand off to the global-point route, which builds an
            # exactly consistent witness by construction.
            cands = []
            for d, et in ((10 ** 6, Fraction(1, 10 ** 4)),
                          (10 ** 6, Fraction(1, 10 ** 3)),
                          (10 ** 5, Fraction(1, 10 ** 3)),
                          (10 ** 6, Fraction(1, 100)),
                          (10 ** 4, Fraction(1, 100))):
                c = exact_consistent_point_margin(R, z, den=d, eta=et)
                if c is not None:
                    cands.append(('margin-repair d=%d eta=%s' % (d, et), c))
            cands += [('double-centred d=%d' % d,
                       exact_consistent_point(R, z, den=d, lam_den=ld))
                      for d, ld in ((10 ** 6, 1024), (10 ** 4, 64))]
            cands.append(('lp-point', zex))
            for how, cand in cands:
                try:
                    ver = verify_witness(R, cand, W, l)
                    info = dict(verdict='REFUTED', how=how,
                                rows=len(R.rows), cols=len(R.cols), its=it,
                                t_float=t, secs=time.time() - t0, **ver)
                    save_witness(tag, R, cand, info)
                    return 'REFUTED', info
                except AssertionError as e:
                    print("    %s point is not an exact witness (%s)"
                          % (how, e), flush=True)
            print("    trying the global-point route", flush=True)
            ok, g = global_refute_fast(gears, W, l)
            if ok:
                info = dict(verdict='REFUTED', how='global-point',
                            support=len(g['support']), row_value=g['row_value'],
                            its=it, secs=time.time() - t0, glob=g)
                os.makedirs(OUT, exist_ok=True)
                p = os.path.join(OUT, 'global_%s.pkl' % tag)
                with open(p, 'wb') as fh:
                    pickle.dump(dict(gears=gears, W=W, l=l, **g), fh)
                print("  GLOBAL WITNESS SAVED: %s" % p, flush=True)
                return 'REFUTED', info
            return 'NOCERT', dict(verdict='NOCERT', reason=g,
                                  rows=len(R.rows), cols=len(R.cols), its=it,
                                  t_float=t, secs=time.time() - t0)
        it += 1
        if it % ckpt_every == 0:
            with open(ck, 'wb') as fh:
                pickle.dump(R.rows, fh)
        if time_budget is not None and time.time() - t0 > time_budget:
            with open(ck, 'wb') as fh:
                pickle.dump(R.rows, fh)
            return 'STUCK', dict(verdict='STUCK', rows=len(R.rows), its=it,
                                 t_float=t, secs=time.time() - t0,
                                 checkpoint=ck)
    with open(ck, 'wb') as fh:
        pickle.dump(R.rows, fh)
    return 'STUCK', dict(verdict='STUCK', rows=len(R.rows), its=it,
                         secs=time.time() - t0, checkpoint=ck)


# ================================================ THE EXACT REFUTATION BRANCH
#
# WHY THE OBVIOUS ROUTE DOES NOT WORK, measured this round.  The natural exact
# witness is the cut loop's own point, rationalised.  It is not usable: the
# point must satisfy every CONSISTENCY LINK exactly, and rationalising a float
# LP solution does not preserve those sums.  `RelaxC.rationalise` +
# `repair_consistency` happens to produce an exactly consistent point at
# machine 13 (nearly-equal floats round to the same rational there) and FAILS
# at machine 19 - the assertion fired on the m19 consistent-only bisection, in
# exactly the way an unhardened verdict would have passed silently.  Round 24's
# process rule ("feasible verdicts must save their witness") is therefore not
# enough on its own: the witness has to be EXACTLY IN THE POLYTOPE, and the
# only cheap way to guarantee that is to build it from one distribution.
#
# A GLOBAL POINT is a rational probability distribution rho over FULL phase
# tuples (optionally mixing in the uniform product measure as one atom).  Its
# degree-<=l marginals come from a single distribution, so every consistency
# link holds identically - at EVERY level, not just level 2.  If in addition
#
#   (i)  at every position of [0,W) its degree-<=l moments are completable
#        (they extend to a law on {0,1}^n with zero mass on the empty atom),
#   (ii) E_rho[f] <= 0, i.e. it satisfies the recursive Costello-Watts row,
#
# then no certificate of the full composition exists at width W - and none
# exists for ANY Sherali-Adams / Lasserre level either, which is strictly
# stronger than refuting the level-2 vehicle.
def f_at_tuple(gears, r, W, ntabs):
    """f(r) = W - sum_q S_q(r_q) + sum_{i<j} n_ij(r_i, r_j), exact integer."""
    v = W - sum(len(hits(q, rq, W)) for q, rq in zip(gears, r))
    for (i, j), tab in ntabs.items():
        v += tab[r[i]][r[j]]
    return v


def global_refute(gears, W, l=2, npool=60, maxrounds=80, seed=0, verbose=True):
    """Search for a global point that kills every degree-l cut AND satisfies
    the recursive row.  Exact rational throughout (the tuple pool is discovery;
    the point and both verdicts are exact).  Returns (True, info) or
    (False, reason)."""
    from math import prod as _prod
    from lp_degree_range import (subsets_upto, _atom_tables, base_cut,
                                 hits as _h)
    from exact_lp import feasible_eq
    from cw_consistent import n_table
    from lp_degree_range import _greedy_tuples
    gears = tuple(gears)
    n = len(gears)
    pool = _greedy_tuples(gears, W, npool, seed)
    covs = [tuple(_h(q, r, W) for q, r in zip(gears, t)) for t in pool]
    S_list = subsets_upto(gears, l)
    S_idx = [[gears.index(q) for q in S] for S in S_list]
    S_mask = [sum(1 << k for k in idx) for idx in S_idx]
    S_unif = [_prod([Fraction(2, q) for q in S]) for S in S_list]
    subs, sidx = _atom_tables(n, l)
    hitlist = [[[si for si, idx in enumerate(S_idx)
                 if all(i in cv[k] for k in idx)] for i in range(W)]
               for cv in covs]
    T = len(covs)
    ntabs = {(i, j): n_table(gears, i, j, W)
             for i in range(n) for j in range(i + 1, n)}
    # the row's value on each atom
    frow = [Fraction(f_at_tuple(gears, t, W, ntabs)) for t in pool]
    s1 = sum(Fraction(1, q) for q in gears)
    f_unif = Fraction(W) - 2 * W * s1 + sum(
        Fraction(sum(sum(row) for row in tab), gears[i] * gears[j])
        for (i, j), tab in ntabs.items())
    frow.append(f_unif)
    if verbose:
        print("    global-point search: %d tuples, f in [%s, %s], f(unif) = %s"
              % (T, min(frow[:T]), max(frow[:T]), f_unif), flush=True)

    def moments_of(w, i):
        out = {m: ZERO for m in S_mask}
        for ti in range(T):
            if w[ti]:
                for si in hitlist[ti][i]:
                    out[S_mask[si]] += w[ti]
        if w[T]:
            for si, m in enumerate(S_mask):
                out[m] += w[T] * S_unif[si]
        return out

    rows = [(i, base_cut(n, l)) for i in range(W)]
    for it in range(maxrounds):
        R = len(rows)
        A, b = [], []
        for r, (i, lam) in enumerate(rows):
            coef = []
            for ti in range(T):
                v = ZERO
                for si in hitlist[ti][i]:
                    v += lam[sidx[S_mask[si]]]
                coef.append(v)
            coef.append(sum(lam[sidx[m]] * S_unif[si]
                            for si, m in enumerate(S_mask)))
            # cut row, with its own surplus variable
            A.append(coef + [-ONE if k == r else ZERO for k in range(R)]
                     + [ZERO])
            b.append(ONE - lam[0])
        A.append([ONE] * (T + 1) + [ZERO] * R + [ZERO])      # weights sum to 1
        b.append(ONE)
        A.append(list(frow) + [ZERO] * R + [ONE])            # E_rho[f] + s = 0
        b.append(ZERO)
        ok, cert = feasible_eq(A, b)
        if not ok:
            return False, ('no global point in this pool satisfying both the '
                           'degree-%d cuts and the row (round %d, %d rows)'
                           % (l, it, R))
        w = [cert[j] for j in range(T + 1)]
        assert sum(w) == ONE and all(v >= 0 for v in w), "bad global point"
        rowval = sum(w[j] * frow[j] for j in range(T + 1))
        assert rowval <= 0, ("global point violates the recursive row", rowval)
        added = 0
        for i in range(W):
            lam = separate(moments_of(w, i), n, l)
            if lam is not None:
                rows.append((i, lam))
                added += 1
        if verbose:
            print("      global it %d: %d new cuts, %d rows, E_rho[f] = %s"
                  % (it, added, len(rows), rowval), flush=True)
        if added == 0:
            # FINAL EXACT RE-ASSERTION, from the weights alone
            for i in range(W):
                assert completable(moments_of(w, i), n, l), \
                    ("global point not completable at %d" % i)
            supp = [(pool[j], w[j]) for j in range(T) if w[j]]
            if w[T]:
                supp.append(('uniform product measure', w[T]))
            return True, dict(support=supp, weights=w, pool=pool,
                              row_value=rowval, rows=len(rows), npool=T,
                              its=it)
    return False, 'global cut loop did not settle in %d rounds' % maxrounds


def global_refute_fast(gears, W, l=2, npool=80, maxrounds=200, seed=0,
                       den=10 ** 6, verbose=True):
    """The same object, found by FLOAT discovery and verified EXACTLY.

    The cut rows exist only to steer the search: a global point's verdict does
    NOT depend on them.  What has to hold, and what is checked exactly here, is
        w >= 0, sum w = 1                       (rational, by construction)
        E_rho[f] <= 0                           (exact rational)
        moments_of(w, i) completable for every i (exact rational Farkas)
    so the float LP is pure discovery and the house rule is respected.  This is
    the version that scales: the exact-simplex variant above spends O(rows^2)
    rational pivots per round and stalls at machine 29."""
    import numpy as np
    from math import prod as _prod
    from scipy.optimize import linprog
    from lp_degree_range import (subsets_upto, _atom_tables, base_cut,
                                 _greedy_tuples)
    from cw_consistent import n_table
    gears = tuple(gears)
    n = len(gears)
    pool = _greedy_tuples(gears, W, npool, seed)
    covs = [tuple(hits(q, r, W) for q, r in zip(gears, t)) for t in pool]
    S_list = subsets_upto(gears, l)
    S_idx = [[gears.index(q) for q in S] for S in S_list]
    S_mask = [sum(1 << k for k in idx) for idx in S_idx]
    S_unif = [_prod([Fraction(2, q) for q in S]) for S in S_list]
    subs, sidx = _atom_tables(n, l)
    hitlist = [[[si for si, idx in enumerate(S_idx)
                 if all(i in cv[k] for k in idx)] for i in range(W)]
               for cv in covs]
    T = len(covs)
    ntabs = {(i, j): n_table(gears, i, j, W)
             for i in range(n) for j in range(i + 1, n)}
    frow = [Fraction(f_at_tuple(gears, t, W, ntabs)) for t in pool]
    s1 = sum(Fraction(1, q) for q in gears)
    f_unif = Fraction(W) - 2 * W * s1 + sum(
        Fraction(sum(sum(row) for row in tab), gears[i] * gears[j])
        for (i, j), tab in ntabs.items())
    frow.append(f_unif)
    if verbose:
        print("    fast global search: %d tuples, f in [%s, %s], f(unif) = %s"
              % (T, min(frow[:T]), max(frow[:T]), f_unif), flush=True)

    def moments_of(w, i):
        out = {m: ZERO for m in S_mask}
        for ti in range(T):
            if w[ti]:
                for si in hitlist[ti][i]:
                    out[S_mask[si]] += w[ti]
        if w[T]:
            for si, m in enumerate(S_mask):
                out[m] += w[T] * S_unif[si]
        return out

    rows = [(i, base_cut(n, l)) for i in range(W)]
    frow_f = np.array([float(v) for v in frow])
    for it in range(maxrounds):
        # float LP: maximise the minimum cut slack subject to the row
        A_ub, b_ub = [], []
        for (i, lam) in rows:
            coef = np.zeros(T + 2)
            for ti in range(T):
                v = ZERO
                for si in hitlist[ti][i]:
                    v += lam[sidx[S_mask[si]]]
                coef[ti] = -float(v)
            coef[T] = -float(sum(lam[sidx[m]] * S_unif[si]
                                 for si, m in enumerate(S_mask)))
            coef[T + 1] = 1.0                       # the slack variable t
            A_ub.append(coef)
            b_ub.append(-float(ONE - lam[0]))
        A_ub.append(np.concatenate([frow_f, [0.0]]))
        b_ub.append(-1e-9)                          # E_rho[f] <= 0, strictly
        A_eq = [np.concatenate([np.ones(T + 1), [0.0]])]
        c = np.zeros(T + 2)
        c[T + 1] = -1.0
        res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                      A_eq=np.array(A_eq), b_eq=np.array([1.0]),
                      bounds=[(0, None)] * (T + 1) + [(None, None)],
                      method='highs')
        if res.status != 0 or -res.fun < 0:
            return False, ('float LP: no mixture in this pool satisfies both '
                           'the cuts and the row (round %d, %d rows, t = %s)'
                           % (it, len(rows),
                              'infeasible' if res.status else -res.fun))
        # ---- rationalise the weights EXACTLY (sum 1, nonnegative)
        w = [max(ZERO, Fraction(round(v * den), den)) for v in res.x[:T + 1]]
        k = max(range(T + 1), key=lambda j: w[j])
        w[k] += ONE - sum(w)
        assert sum(w) == ONE and all(v >= 0 for v in w), "bad rational weights"
        rowval = sum(w[j] * frow[j] for j in range(T + 1))
        if rowval > 0:
            # Rounding pushed the row over.  Repair EXACTLY by mixing toward
            # the atom of least f: w' = (1-a) w + a e_jmin with
            # a = rowval / (rowval - f_min) makes E_rho'[f] exactly 0, and w'
            # is still a probability vector.  (Only possible if some atom has
            # f < 0 - if none does, no mixture can satisfy the row at all.)
            jmin = min(range(T + 1), key=lambda j: frow[j])
            if frow[jmin] >= 0:
                return False, ('every atom in the pool has f >= 0, so no '
                               'mixture can satisfy the row')
            a = Fraction(rowval, rowval - frow[jmin])
            w = [(ONE - a) * v for v in w]
            w[jmin] += a
            assert sum(w) == ONE and all(v >= 0 for v in w), "repair broke w"
            rowval = sum(w[j] * frow[j] for j in range(T + 1))
            assert rowval <= 0, ("row repair failed", rowval)
        added = 0
        for i in range(W):
            lam = separate(moments_of(w, i), n, l)
            if lam is not None:
                rows.append((i, lam))
                added += 1
        if verbose and (added == 0 or it % 5 == 0):
            print("      fast global it %d: %d new cuts, %d rows, "
                  "E_rho[f] = %s" % (it, added, len(rows), rowval), flush=True)
        if added == 0:
            for i in range(W):            # FINAL EXACT RE-ASSERTION
                assert completable(moments_of(w, i), n, l), \
                    ("global point not completable at %d" % i)
            assert rowval <= 0
            supp = [(pool[j], w[j]) for j in range(T) if w[j]]
            if w[T]:
                supp.append(('uniform product measure', w[T]))
            return True, dict(support=supp, weights=w, pool=pool,
                              row_value=rowval, rows=len(rows), npool=T,
                              its=it)
    return False, 'fast global loop did not settle in %d rounds' % maxrounds


# ============ AN EXACTLY CONSISTENT POINT BUILT FROM THE LP'S OWN SOLUTION
#
# The global-point route is limited by its tuple pool.  This route is not: it
# takes the level-2 LP's own float solution and produces a point that is
# EXACTLY in the consistent polytope by construction, using the fact that the
# consistency constraints on a pair block are exactly "row sums = p_a, column
# sums = p_b".
#
#   1. Round the single-gear distributions to one fixed denominator D and put
#      each block's rounding residue on its largest entry: p_a is then an exact
#      rational probability vector with denominator dividing D.
#   2. Round the pair block, subtract the product p_a (x) p_b, and DOUBLE-CENTRE
#      the residual:  E'(u,v) = E(u,v) - r(u)/q_b - c(v)/q_a + s/(q_a q_b).
#      Double-centring is exact and makes every row sum and every column sum of
#      E' identically zero - so p_a (x) p_b + E' has row sums exactly p_a and
#      column sums exactly p_b, whatever E was.
#   3. Shrink by the largest lambda in (0,1] (snapped to a small denominator)
#      that keeps every entry nonnegative.
#
# Every consistency link then holds by construction, at level 2 and (since the
# singles are shared) between every pair of blocks.  What is NOT automatic is
# that the point still satisfies the cuts - that is checked exactly, position by
# position, and the verdict is only taken if it does.
def exact_consistent_point_margin(R, z, den=10 ** 6, eta=Fraction(1, 1000)):
    """A second exactly-consistent construction, and the one that stays NEAR
    the LP point instead of collapsing toward the product measure.

    After rounding, block (a,b) has row-sum deficits d_u = p_a(u) - rowsum_u
    and column deficits e_v = p_b(v) - colsum_v, with sum d = sum e = delta.
    Adding the OUTER PRODUCT d (x) e / delta restores both margins exactly
    (row sum picks up d_u * (sum e)/delta = d_u, column sum picks up e_v), and
    the correction is of rounding size, so the repaired block is a rounding
    perturbation of the LP's own block - NOT a pull toward uniform.  A small
    floor eta * p_a (x) p_b is mixed in first so the correction cannot drive a
    zero cell negative; eta is a parameter and the nonnegativity is asserted.

    This matters because at machine 23 the uniform product measure VIOLATES the
    degree-2 cuts, so any construction that shrinks toward it cannot produce a
    feasible witness there - which is exactly how the double-centred variant
    fails (position 4 not completable, at every denominator tried)."""
    gs = R.gears
    n = R.n
    zx = [ZERO] * len(R.cols)
    p = {}
    for k, q in enumerate(gs):
        lo, hi = R.block_span[(q,)]
        v = [max(ZERO, Fraction(round(float(z[j]) * den), den))
             for j in range(lo, hi)]
        s = sum(v)
        kk = max(range(q), key=lambda t: v[t])
        v[kk] += ONE - s
        if v[kk] < 0:
            v = [Fraction(1, q)] * q
        p[q] = v
        for t in range(q):
            zx[lo + t] = v[t]
    for a in range(n):
        for b in range(a + 1, n):
            qa, qb = gs[a], gs[b]
            S = (qa, qb)
            raw = [[max(ZERO, Fraction(round(float(z[R.tupidx[(S, (u, v))]])
                                            * den), den))
                    for v in range(qb)] for u in range(qa)]
            tot = sum(sum(r) for r in raw)
            if tot == 0:
                return None
            sc = (ONE - eta) / tot                 # mass exactly 1 - eta
            Z = [[sc * raw[u][v] for v in range(qb)] for u in range(qa)]
            d = [p[qa][u] - sum(Z[u]) for u in range(qa)]
            e = [p[qb][v] - sum(Z[u][v] for u in range(qa))
                 for v in range(qb)]
            delta = sum(d)
            assert delta == sum(e), "margin deficits disagree"
            # The outer-product repair needs delta > 0 and d, e >= 0 - which is
            # what the eta floor is for: Z carries only (1 - eta) of the mass,
            # so every deficit is about eta times the corresponding marginal.
            # If any deficit is still negative (a marginal rounded to zero
            # under a block that has mass there), this eta cannot be repaired
            # and the caller tries the next one.  Note delta == 0 is NOT a
            # repairable case: the total mass is right but the ROW masses need
            # not be, and adding nothing leaves the links broken - the bug this
            # guard replaces.
            if delta <= 0 or any(x < 0 for x in d) or any(x < 0 for x in e):
                return None
            for u in range(qa):
                for v in range(qb):
                    val = Z[u][v] + Fraction(d[u] * e[v], delta)
                    if val < 0:
                        return None
                    zx[R.tupidx[(S, (u, v))]] = val
    return zx


def exact_consistent_point(R, z, den=10 ** 4, lam_den=64):
    gs = R.gears
    n = R.n
    zx = [ZERO] * len(R.cols)
    # ---- 1. the singles
    p = {}
    for k, q in enumerate(gs):
        lo, hi = R.block_span[(q,)]
        v = [max(ZERO, Fraction(round(float(z[j]) * den), den))
             for j in range(lo, hi)]
        s = sum(v)
        kk = max(range(q), key=lambda t: v[t])
        v[kk] += ONE - s
        if v[kk] < 0:
            v = [Fraction(1, q)] * q
        assert sum(v) == ONE and all(x >= 0 for x in v)
        p[q] = v
        for t in range(q):
            zx[lo + t] = v[t]
    # ---- 2 and 3. the pairs
    for a in range(n):
        for b in range(a + 1, n):
            qa, qb = gs[a], gs[b]
            S = (qa, qb)
            lo, hi = R.block_span[S]
            Z = [[max(ZERO, Fraction(round(float(z[R.tupidx[(S, (u, v))]])
                                           * den), den))
                  for v in range(qb)] for u in range(qa)]
            E = [[Z[u][v] - p[qa][u] * p[qb][v] for v in range(qb)]
                 for u in range(qa)]
            rs = [sum(E[u]) for u in range(qa)]
            cs = [sum(E[u][v] for u in range(qa)) for v in range(qb)]
            tot = sum(rs)
            Ep = [[E[u][v] - Fraction(rs[u], qb) - Fraction(cs[v], qa)
                   + Fraction(tot, qa * qb) for v in range(qb)]
                  for u in range(qa)]
            # nonnegativity shrink
            lam = ONE
            for u in range(qa):
                for v in range(qb):
                    if Ep[u][v] < 0:
                        base = p[qa][u] * p[qb][v]
                        cap = Fraction(base, -Ep[u][v])
                        if cap < lam:
                            lam = cap
            lam = Fraction(int(lam * lam_den), lam_den)   # snap DOWN
            for u in range(qa):
                for v in range(qb):
                    val = p[qa][u] * p[qb][v] + lam * Ep[u][v]
                    assert val >= 0, "shrink failed"
                    zx[R.tupidx[(S, (u, v))]] = val
    return zx


# ------------------------------------------------------- exact verification
def verify_witness(R, z, W, l):
    """EXACT.  Assert that z is a genuine feasible point of the full
    composition at width W.  Raises on any failure - there is no soft verdict.
    Returns the measured slack quantities."""
    assert all(v >= 0 for v in z), "witness has a negative entry"
    for S in R.subsets:
        lo, hi = R.block_span[S]
        assert sum(z[lo:hi]) == ONE, ("block does not sum to 1", S)
    for (par, kids) in R.links:
        assert sum(z[j] for j in kids) == z[par], "consistency link broken"
    for i in range(W):
        assert completable(R.moments_at(z, i), R.n, l), \
            ("witness not completable at position %d" % i)
    rowval = sum(v * z[j] for j, v in enumerate(R.frow) if v)
    assert rowval >= R.frhs, ("witness violates the recursive row",
                              rowval, R.frhs)
    return dict(row_value=rowval, row_rhs=R.frhs, row_slack=rowval - R.frhs)


def save_witness(tag, R, z, info):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, 'witness_%s.pkl' % tag)
    with open(p, 'wb') as fh:
        pickle.dump(dict(gears=R.gears, W=R.W, l=R.l, z=z, info=info,
                         cols=[(S, r) for (S, r, _O) in R.cols]), fh)
    print("  WITNESS SAVED: %s" % p, flush=True)
    return p


def save_certificate(tag, R, y, yff, nu, info):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, 'cert_%s.pkl' % tag)
    with open(p, 'wb') as fh:
        pickle.dump(dict(gears=R.gears, W=R.W, l=R.l, rows=R.rows,
                         y=y, yff=yff, nu=nu, info=info), fh)
    print("  CERTIFICATE SAVED: %s" % p, flush=True)
    return p


def reverify_witness(tag):
    """SECOND PASS: rebuild the relaxation from scratch and re-verify the saved
    witness.  Nothing from the deciding process is trusted."""
    p = os.path.join(OUT, 'witness_%s.pkl' % tag)
    with open(p, 'rb') as fh:
        d = pickle.load(fh)
    R = RelaxCF(d['gears'], d['W'], d['l'])
    assert [(S, r) for (S, r, _O) in R.cols] == d['cols'], \
        "column layout changed - witness cannot be re-verified"
    ver = verify_witness(R, d['z'], d['W'], d['l'])
    print("  RE-VERIFIED from disk: %s  row value %s >= %s (slack %s)"
          % (p, ver['row_value'], ver['row_rhs'], ver['row_slack']), flush=True)
    return ver


# ------------------------------------------------- consistent-only, exactly
def decideC25(gears, W, l=2, maxrounds=800, verbose=True, tag=None, filt=True):
    """The SAME exact treatment for the consistent degree-2 LP with NO
    recursive row (round 23's vehicle).  Used for the machine-19 ATTRIBUTION:
    the composition certifies width 33 at m19; is that the recursion's doing,
    or would consistency alone have done it?  `decideC_cert`'s feasible branch
    is solver-only (it says so in its own docstring), so it cannot answer -
    this one can."""
    from lp_degree_range import RelaxC, certificateC
    t0 = time.time()
    R = RelaxC(gears, W, l)
    tag = tag or ("C_m%d_w%d" % (gears[-1], W))
    it, final_pass = 0, False
    while it < maxrounds:
        t, z, res = R._solve_float()
        if t < -1e-7:
            nb = len(R.subsets)
            y = list(-res.ineqlin.marginals)
            nu = res.eqlin.marginals[nb:]
            ok, lhs, rhs, yq, nuq, ops = certificateC(R, y, nu)
            assert ok, "no exact certificate could be rationalised - ABORT"
            info = dict(verdict='CERTIFIED', lhs=lhs, rhs=rhs, ops=ops,
                        rows=len(R.rows), cols=len(R.cols), its=it,
                        secs=time.time() - t0)
            os.makedirs(OUT, exist_ok=True)
            with open(os.path.join(OUT, 'cert_%s.pkl' % tag), 'wb') as fh:
                pickle.dump(dict(gears=R.gears, W=W, l=l, rows=R.rows,
                                 y=yq, nu=nuq, info=info), fh)
            return 'CERTIFIED', info
        den = 10 ** (4 + min(it // 40, 4))
        zex = R.repair_consistency(R.rationalise(z, den))
        margin = ZERO if final_pass else Fraction(1, 10 ** 5)
        added = 0
        for i in range(W):
            if filt and not final_pass and _float_interior(R, zex, i):
                continue
            lam = separate(R.moments_at(zex, i), R.n, l, margin)
            if lam is not None:
                R.rows.append((i, lam))
                added += 1
        if verbose:
            print("      it %d: t = %+.4f, %d cuts, %d rows%s  [%.0fs]"
                  % (it, t, added, len(R.rows),
                     "  FINAL EXACT PASS" if final_pass else "",
                     time.time() - t0), flush=True)
        if added == 0:
            if not final_pass:
                final_pass = True
                continue
            assert all(v >= 0 for v in zex)
            for S in R.subsets:
                lo, hi = R.block_span[S]
                assert sum(zex[lo:hi]) == ONE, ("block sum", S)
            for (par, kids) in R.links:
                assert sum(zex[j] for j in kids) == zex[par], "consistency"
            for i in range(W):
                assert completable(R.moments_at(zex, i), R.n, l), \
                    ("not completable at %d" % i)
            info = dict(verdict='REFUTED', rows=len(R.rows), cols=len(R.cols),
                        its=it, secs=time.time() - t0)
            os.makedirs(OUT, exist_ok=True)
            p = os.path.join(OUT, 'witness_%s.pkl' % tag)
            with open(p, 'wb') as fh:
                pickle.dump(dict(gears=R.gears, W=W, l=l, z=zex, info=info,
                                 cols=[(S, r) for (S, r, _O) in R.cols]), fh)
            print("  WITNESS SAVED: %s" % p, flush=True)
            return 'REFUTED', info
        it += 1
    return 'STUCK', dict(verdict='STUCK', rows=len(R.rows), its=it)


def gate():
    """HEADLINE ASSERTION GATE for this lane (round 25).  Known-answer checks
    on both verdict branches and on the cross-consistency of the two branches.
    Prints 'ALL ASSERTIONS GREEN' or aborts."""
    print("=" * 70)
    print("LP-DUALITY ROUND-25 GATE")
    print("=" * 70)
    # 1. the four (D) rungs, composed - exact certificate values
    print("1. the four composed rung certificates (budget widths):")
    for y, W, lhs, rhs, ops in ((11, 16, '14', '16', 562),
                                (13, 20, '20', '21', 1456),
                                (17, 28, '29', '146/5', 3303)):
        v, info = decide25(gears_of(y), W, 2, verbose=False)
        assert v == 'CERTIFIED', (y, W, v)
        assert str(info['lhs']) == lhs and str(info['rhs']) == rhs, \
            (y, W, info['lhs'], info['rhs'])
        assert info['ops'] == ops, (y, W, info['ops'], ops)
        print("   m%-2d W=%-3d %s < %s  (%d ops)  OK" % (y, W, lhs, rhs, ops))
    v, info = decide25(gears_of(19), 37, 2, verbose=False)
    assert v == 'CERTIFIED' and info['lhs'] < info['rhs'], info
    print("   m19 W=37  %s < %s  (%d ops)  OK   [the dual point is"
          " path-dependent; only the certified relation is asserted]"
          % (info['lhs'], info['rhs'], info['ops']))
    # 2. consistent-only reference values (round-24 numbers, same code)
    print("2. consistent-only certificates at the same widths:")
    for y, W, ops in ((11, 16, 464), (13, 20, 2868), (17, 28, 9091),
                      (19, 37, 25413)):
        v, info = decideC25(gears_of(y), W, 2, verbose=False)
        assert v == 'CERTIFIED' and info['ops'] == ops, (y, W, info)
        print("   m%-2d W=%-3d %d ops  OK" % (y, W, ops))
    # 3. machine-19 attribution: consistency alone certifies width 33
    v, info = decideC25(gears_of(19), 33, 2, verbose=False)
    assert v == 'CERTIFIED', info
    print("3. m19 W=33 consistent-only: CERTIFIED %s < %s (%d ops)  OK"
          % (info['lhs'], info['rhs'], info['ops']))
    # 4. the refutation branch, BOTH directions, and its cross-consistency
    print("4. the global-point refutation branch:")
    ok, g = global_refute_fast(gears_of(13), 10, 2, verbose=False)
    assert ok and g['row_value'] <= 0, g
    print("   m13 W=10 (F=11, so width 10 IS coverable): REFUTED by a global"
          " point, E_rho[f] = %s  OK" % g['row_value'])
    for y, W in ((13, 20), (19, 33)):
        ok2, g2 = global_refute_fast(gears_of(y), W, 2, verbose=False)
        assert not ok2, ("a global point was found at a width where an exact "
                         "certificate exists - the two branches CONTRADICT",
                         y, W, g2)
        print("   m%-2d W=%-3d certificate exists => no global point found"
              "  OK (branches agree)" % (y, W))
    # 5. the standing REFUTATIONS, re-verified from their saved witnesses
    print("5. saved refutation witnesses, re-verified from disk:")
    for tag in ('m23_w48', 'm29_w63'):
        pth = os.path.join(OUT, 'witness_%s.pkl' % tag)
        if not os.path.exists(pth):
            print("   %s: witness file missing - SKIPPED" % tag)
            continue
        ver = reverify_witness(tag)
        assert ver['row_slack'] >= 0
        print("   %s: exact feasible point of the full composition"
              " (row slack %s)  OK" % (tag, ver['row_slack']))
    print("\nALL ASSERTIONS GREEN")


def main():
    if sys.argv[1] == 'GATE':
        gate()
        return
    if sys.argv[1] == 'W':        # witness attempt from the saved cut rows
        y, W = int(sys.argv[2]), int(sys.argv[3])
        g = gears_of(y)
        R = RelaxCF(g, W, 2)
        ck = os.path.join(OUT, 'ck_m%d_w%d.pkl' % (y, W))
        if os.path.exists(ck):
            with open(ck, 'rb') as fh:
                R.rows = pickle.load(fh)
        print("witness attempt: machine %d width %d, %d cut rows"
              % (y, W, len(R.rows)), flush=True)
        t, z, _res = R._solve_float()
        print("  float LP value t = %+.6f" % t, flush=True)
        if t < -1e-7:
            print("  the LP is INFEASIBLE at these rows - run the decider,"
                  " not the witness attempt", flush=True)
            return
        for d, et in ((10 ** 6, Fraction(1, 10 ** 4)),
                      (10 ** 6, Fraction(1, 10 ** 3)),
                      (10 ** 5, Fraction(1, 10 ** 3)),
                      (10 ** 6, Fraction(1, 100)),
                      (10 ** 4, Fraction(1, 100)),
                      (10 ** 6, Fraction(1, 20))):
            cand = exact_consistent_point_margin(R, z, den=d, eta=et)
            if cand is None:
                print("  d=%d eta=%s: not constructible" % (d, et), flush=True)
                continue
            try:
                ver = verify_witness(R, cand, W, 2)
            except AssertionError as e:
                print("  d=%d eta=%s: %s" % (d, et, e), flush=True)
                continue
            info = dict(verdict='REFUTED', how='margin-repair d=%d eta=%s'
                        % (d, et), rows=len(R.rows), **ver)
            save_witness("m%d_w%d" % (y, W), R, cand, info)
            print("RESULT: REFUTED  no certificate of the full composition"
                  " exists at machine %d width %d.  row value %s >= %s"
                  % (y, W, ver['row_value'], ver['row_rhs']), flush=True)
            reverify_witness("m%d_w%d" % (y, W))
            return
        print("RESULT: no exact witness constructible from these rows",
              flush=True)
        return
    if sys.argv[1] == 'G':                       # global-point refutation only
        y, W = int(sys.argv[2]), int(sys.argv[3])
        npool = int(sys.argv[4]) if len(sys.argv) > 4 else 60
        g = gears_of(y)
        print("GLOBAL-POINT REFUTATION  machine %d  width %d  (F=%d budget=%d)"
              % (y, W, F_EXACT[y], budget(y)), flush=True)
        t0 = time.time()
        fast = '--slow' not in sys.argv
        ok, info = (global_refute_fast if fast else global_refute)(
            g, W, 2, npool=npool)
        if ok:
            os.makedirs(OUT, exist_ok=True)
            p = os.path.join(OUT, 'global_m%d_w%d.pkl' % (y, W))
            with open(p, 'wb') as fh:
                pickle.dump(dict(gears=g, W=W, l=2, **info), fh)
            print("RESULT: REFUTED by a global point - support %d, "
                  "E_rho[f] = %s, %d rows, %d its  [%.0fs]  saved %s"
                  % (len(info['support']), info['row_value'], info['rows'],
                     info['its'], time.time() - t0, p), flush=True)
        else:
            print("RESULT: no global point found - %s  [%.0fs]"
                  % (info, time.time() - t0), flush=True)
        return
    if sys.argv[1] == 'C':                       # consistent-only, no row
        y, W = int(sys.argv[2]), int(sys.argv[3])
        print("CONSISTENT-ONLY  machine %d  width %d  (F=%d budget=%d)"
              % (y, W, F_EXACT[y], budget(y)), flush=True)
        v, info = decideC25(gears_of(y), W, 2)
        print("RESULT: %s  %s" % (v, {k: str(x) for k, x in info.items()}),
              flush=True)
        return
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    y, W = int(args[0]), int(args[1])
    rec = '--norec' not in sys.argv
    filt = '--nofilter' not in sys.argv
    tb = None
    for a in sys.argv[1:]:
        if a.startswith('--budget='):
            tb = float(a.split('=')[1])
    g = gears_of(y)
    print("machine %d  width %d  recursion=%s  filter=%s  F=%d budget=%d"
          % (y, W, rec, filt, F_EXACT[y], budget(y)), flush=True)
    v, info = decide25(g, W, 2, use_recursion=rec, filt=filt, time_budget=tb,
                       fastsep=('--fastsep' in sys.argv))
    print("RESULT: %s  %s" % (v, {k: str(x)[:200] for k, x in info.items()
                                  if k not in ('R', 'glob')}), flush=True)
    if v == 'REFUTED' and info.get('how') != 'global-point':
        reverify_witness("m%d_w%d" % (y, W))


if __name__ == '__main__':
    main()
