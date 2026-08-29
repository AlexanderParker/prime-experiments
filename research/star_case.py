"""ROUND 26, LP-DUALITY THREAD.

TWO NEW OBJECTS, BOTH BUILT OUT OF THE SAME OBSERVATION.

The composed vehicle (round 24; round 25's `cw_decide25.py`) is
    f(r) = W - sum_q S_q(r_q) + sum_{i<j} n_ij(r_i, r_j)  <=  open(r)
minimised over the pairwise-consistent polytope, PLUS the degree-2 covering
cuts at every position of [0, W).  Round 25 proved its frontier is a WIDTH:
E_u[f] = W Pi(y) - Delta(y, W), so the vehicle needs W >= W_u(y), and at
machine 41 the ladder's budget 129 falls six short of W_u(41) = 135.

THE OBSERVATION.  Both of round 25's named escapes are the same move -
RESTRICT THE POSITION SET AND KEEP THE VEHICLE.

  * STAR-k / CASE SPLIT.  Fix the phases of the k smallest gears (the "held"
    gears) at w.  Every position those gears block is already covered, so the
    obligation shrinks to U_w = [0,W) minus what they block, over the gear set
    gears[k:].  THE CONDITIONAL PROBLEM IS THE SAME VEHICLE ON A SMALLER
    POSITION SET, exactly:
        f_w(r) = |U_w| - sum_{q free} |hits(q,r_q) & U_w|
                       + sum_{i<j free} n^{U_w}_ij(r_i, r_j)   <=   open(U_w).
    A certificate in EVERY case is a CASE-SPLIT CERTIFICATE of the rung - a
    certificate species this project has not had.  It is STRICTLY STRONGER
    than the STAR-3 LP of `cw_consistent.Composed3`, which carries triple
    blocks (5, q_i, q_j) tied only to the singles: a STAR-3 point does not
    condition into a family of case points (its conditionals need not be
    pairwise consistent), while a family of case points always MIXES into a
    STAR-3 point.  (Mixing: completability is convex, and in a case where a
    held gear blocks the position the moment vector is trivially completable,
    so the mixture satisfies every cut; the row is an average of satisfied
    rows.)  Hence:  all cases infeasible  =>  STAR-3 LP infeasible, and
    conversely a single feasible case does NOT refute STAR-3 - which is why
    the case split is the object worth deciding.

  * WINDOWED / OPEN-POINT statements.  The (D) route does not only need
    "no fully blocked window of width W".  Constructor's R64 covering form of
    the two-gap statement is "no configuration with positions 0, a and W open
    and every other position of (0,W) blocked" - i.e. the SAME vehicle with
    three positions removed from the obligation AND every gear phase that
    blocks one of them removed from the domain.  Removing phases is a genuine
    tightening at the SAME width, so it is exactly the kind of narrower
    statement the frontier-is-a-width law points at.

Both are the class `RelaxStar` below: the composed relaxation over an
arbitrary position set with an arbitrary per-gear phase domain.

HOUSE RULES (this thread's, unchanged).  Exact rationals decide everything;
scipy is discovery only.  A CERTIFIED verdict carries an exact rational dual
certificate, re-checked from its own numbers.  A REFUTED verdict carries an
exact rational primal point that is EXACTLY IN THE POLYTOPE - built consistent
by construction, never by rationalising a float - saved to disk and
re-verified from a clean process.  Op counts, not wall time.

Run:
    python research/star_case.py GATE                # all assertions (~80 s)
    python research/star_case.py PRETEST [y] [k]     # the case-split pre-test
    python research/star_case.py CASE <y> [k]        # decide every case
    python research/star_case.py WINDOW <y> <W> [a..]# two-gap windows at span W
    python research/star_case.py F2 <y> <lo> <hi>    # the whole F_2 ladder
"""
import os
import pickle
import sys
import time
from fractions import Fraction
from itertools import combinations, product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lp_degree_range import (gears_of, budget, teeth, hits, F_EXACT,  # noqa
                             ZERO, ONE, subsets_upto, _atom_tables,
                             base_cut, separate, completable,
                             product_moments, zeta_values, cut_value,
                             _sep_matrix_exact)
from cw_decide25 import separate_fast                              # noqa
from row_decay import _max_cover, _coverable, _max_cover_masks     # noqa

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'r26')

OPS = [0]                 # exact-arithmetic operation counter


# ===================================================== coverage, with domains
def _hitset(q, r, W, pos):
    return frozenset(x for x in hits(q, r, W) if x in pos)


def max_cover_dom(P, lower, W, dom):
    """max over phase choices of the `lower` gears - each restricted to its
    domain dom[q] - of the number of positions of P covered.  EXACT where
    decidable, else the valid upper bound |P| (which only weakens n toward 0).
    Returns (value, exact?)."""
    m = len(P)
    if m == 0 or not lower:
        return 0, True
    Pset = frozenset(P)
    opts = {}
    for q in lower:
        s = set()
        for r in dom[q]:
            h = hits(q, r, W)
            s.add(frozenset(x for x in Pset if x in h))
        opts[q] = s
    # (1) complete backtracker on the first uncovered position

    def rec(covered, avail):
        if covered == Pset:
            return True
        p = min(Pset - covered)
        for k, q in enumerate(avail):
            for s in opts[q]:
                if p in s:
                    if rec(covered | s, avail[:k] + avail[k + 1:]):
                        return True
        return False

    if rec(frozenset(), tuple(lower)):
        return m, True
    # (2) reachable-mask sweep (exact, size bounded by prod |dom|)
    idx = {p: 1 << b for b, p in enumerate(sorted(Pset))}
    cur = {0}
    for q in lower:
        os_ = {sum(idx[p] for p in s) for s in opts[q]}
        cur = {a | o for a in cur for o in os_}
        if len(cur) > 400000:
            return m, False
    return max(bin(a).count('1') for a in cur), True


# ================================================ fast EXACT zeta / separation
#
# `lp_degree_range.zeta_values` computes f[x] = sum_{S subset x} lam_S by
# looping over the supersets of each subset: at n = 10 that is ~150,000 inner
# steps per call, and `separate_fast` calls it up to eight times per position
# per iteration - MEASURED: 98 s for one cut pass over 78 positions at machine
# 41.  The standard subset-sum (zeta) transform does the same job in
# n * 2^(n-1) = 5,120 exact additions.  Same arithmetic, same exact values -
# asserted equal to `zeta_values` in the gate.
def zeta_fast(lam, n, subs):
    f = [ZERO] * (1 << n)
    for m, v in zip(subs, lam):
        if v:
            f[m] += v
    for i in range(n):
        bit = 1 << i
        for x in range(1 << n):
            if x & bit:
                f[x] += f[x ^ bit]
    return f


SEPF_CACHE = {}


def _sepf_float(n, l):
    """cached float copy of the exact separation matrix."""
    key = (n, l)
    if key not in SEPF_CACHE:
        import numpy as np
        A, subs = _sep_matrix_exact(n, l)
        SEPF_CACHE[key] = (np.array([[float(v) for v in row] for row in A]),
                           subs, A)
    return SEPF_CACHE[key]


def sep_fast2(mom, n, l, margin=ZERO, M=16.0, dens=(4, 16, 64, 256)):
    """Same contract as `cw_decide25.separate_fast` - a VIOLATED, EXACTLY
    VALID degree-l cut or None - with the exact zeta transform above and a
    cached float matrix.  Nothing is decided in floats: validity is repaired
    exactly by raising lam_0, and both facts (zeta >= 1 everywhere,
    lam . mom < 1 - margin) are asserted in exact rationals."""
    import numpy as np
    from scipy.optimize import linprog
    Af, subs, _A = _sepf_float(n, l)
    ns, na = Af.shape
    c = np.array([float(mom.get(m, ONE)) if m else 1.0 for m in subs])
    res = linprog(c, A_ub=-Af.T, b_ub=-np.ones(na),
                  bounds=[(-M, M)] * ns, method='highs')
    if res.status != 0 or res.fun >= 1.0 - 1e-12:
        return None
    for den in dens:
        lam = [Fraction(round(v * den), den) for v in res.x]
        f = zeta_fast(lam, n, subs)
        mn = min(f[1:])
        if mn < ONE:
            lam[0] += ONE - mn
            f = zeta_fast(lam, n, subs)
            mn = min(f[1:])
        assert mn >= ONE, "repaired cut still invalid"
        val = cut_value(tuple(lam), subs, mom)
        if val < ONE - margin:
            return tuple(lam)
    return None


# ============================================ fast EXACT completion (round 26)
#
# WHY THIS EXISTS.  `lp_degree_range.separate` decides completability with an
# exact rational two-phase simplex on the (subsets x nonempty atoms) tableau.
# At n = 8 that is 37 x 293 and costs milliseconds; at n = 10 it is 56 x 1023
# and at n = 11 it is 67 x 2047, and the cost is quadratic in the tableau area
# times the pivot count - MEASURED this round: a single n = 11 call did not
# finish in 10 minutes, and the composed vehicle calls it once per position per
# iteration.  That is the whole reason the case-split at machine 41 looked
# unaffordable.
#
# THE FIX, AND IT IS SOUND.  Completability is an EXISTENCE question, so a
# claimed completion can be CHECKED far more cheaply than it can be searched
# for: nu >= 0 and A nu = b, both exact.  So: find the support in floats
# (discovery), then solve the small exact linear system on that support, then
# assert nonnegativity and the moment equations in exact rationals.  A True
# from this routine is a VERIFIED exact completion.  Anything else falls back
# to the exact oracle, so no verdict ever rests on the float step.
COMP_CACHE = {}


def _exact_solve_cols(A, b, cols):
    """Exact: solve A[:, cols] * v = b with Gaussian elimination, free
    variables set to 0.  Returns the full-length vector or None."""
    m = len(A)
    k = len(cols)
    M = [[A[r][c] for c in cols] + [b[r]] for r in range(m)]
    piv = []
    row = 0
    for c in range(k):
        pr = None
        for r in range(row, m):
            if M[r][c] != 0:
                pr = r
                break
        if pr is None:
            continue
        M[row], M[pr] = M[pr], M[row]
        pv = M[row][c]
        if pv != ONE:
            M[row] = [v / pv for v in M[row]]
        for r in range(m):
            if r != row and M[r][c]:
                f = M[r][c]
                M[r] = [a - f * bb for a, bb in zip(M[r], M[row])]
        piv.append((row, c))
        row += 1
        if row == m:
            break
    for r in range(row, m):
        if M[r][k] != 0:
            return None                     # inconsistent on these columns
    v = [ZERO] * k
    for (r, c) in piv:
        v[c] = M[r][k]
    return v


def completable_fast(mom, n, l, cap=4):
    """EXACT verdict on completability, with a float-discovered support.
    Returns True (an exact completion was built and checked), False (the exact
    oracle found a violated cut), or falls through to the exact oracle."""
    key = (n, l, tuple(sorted(mom.items())))
    if key in COMP_CACHE:
        return COMP_CACHE[key]
    A, subs = _sep_matrix_exact(n, l)
    ns, na = len(subs), len(A[0])
    b = [mom.get(m, ONE) if m else ONE for m in subs]
    try:
        import numpy as np
        from scipy.optimize import linprog
        Af = np.array([[float(v) for v in row] for row in A])
        bf = np.array([float(v) for v in b])
        res = linprog(np.zeros(na), A_eq=Af, b_eq=bf,
                      bounds=[(0, None)] * na, method='highs')
        if res.status == 0:
            order = list(np.argsort(-res.x))
            for width in (ns, 2 * ns, cap * ns):
                cols = order[:min(width, na)]
                v = _exact_solve_cols(A, b, cols)
                if v is None or any(x < 0 for x in v):
                    continue
                # EXACT re-assertion of the completion
                for r in range(ns):
                    s = ZERO
                    for idx, c in enumerate(cols):
                        if v[idx]:
                            s += A[r][c] * v[idx]
                    assert s == b[r], "completion does not match moments"
                COMP_CACHE[key] = True
                return True
        else:
            # the float completion LP says infeasible.  `separate_fast` turns
            # that into an EXACTLY VALID, EXACTLY VIOLATED degree-l cut, which
            # is an exact proof of NON-completability - so a cut here decides
            # False exactly, with no rational simplex.
            if sep_fast2(mom, n, l, ZERO) is not None:
                COMP_CACHE[key] = False
                return False
    except Exception:
        pass
    out = separate(mom, n, l) is None
    COMP_CACHE[key] = out
    return out


# ================================================== the restricted relaxation
class RelaxStar:
    """The composed level-l relaxation over
        * the FREE gears  gears[len(held):],
        * the POSITION SET pos = [0,W) minus what the held gears block at ws
          minus the required-open positions,
        * the PHASE DOMAIN dom[q] = phases of q that block no required-open
          position.
    Columns are genuine phase tuples over the domains; the recursion row and
    the degree-l cuts are both taken over `pos`.

    SOUNDNESS.  Every actual fully-blocked window whose held gears sit at ws
    and whose required-open positions are open induces a 0/1 point of this
    polytope: its phase tuple lies in the domains, every position of pos is
    covered by some free gear, and open(pos) = 0 gives
    sum_q S_q - sum n_ij >= |pos|.  So an infeasibility certificate excludes
    every such window.  Restricting the domains only makes n_ij LARGER (a min
    over fewer phases), and n_ij <= N_ij still holds at the actual phase
    tuple because that tuple is in the domain - so the row stays valid."""

    def __init__(self, gears, W, held=(), ws=(), openpts=(), l=2,
                 verbose=False):
        self.full = tuple(gears)
        self.held = tuple(held)
        self.ws = tuple(ws)
        assert self.held == self.full[:len(self.held)], "held must be a prefix"
        assert len(self.ws) == len(self.held)
        self.gears = self.full[len(self.held):]
        self.n = len(self.gears)
        self.W = W
        self.l = l
        self.openpts = tuple(sorted(openpts))
        self.dead = False                 # case is vacuous (held gear covers
        #                                   a required-open position)
        blocked = set()
        for q, w in zip(self.held, self.ws):
            h = hits(q, w, W)
            if any(p in h for p in self.openpts):
                self.dead = True
            blocked |= set(h)
        self.pos = tuple(sorted(set(range(W)) - blocked
                                - set(self.openpts)))
        self.dom = {}
        self.why = None
        for q in self.gears:
            self.dom[q] = tuple(r for r in range(q)
                                if not any(p in hits(q, r, W)
                                           for p in self.openpts))
            if not self.dom[q]:
                # EVERY phase of gear q blocks one of the required-open
                # positions: the configuration is impossible outright, and
                # that single gear IS the certificate.
                self.dead = True
                self.why = 'gear %d has no phase leaving all of %s open' \
                    % (q, list(self.openpts))
        if self.dead:
            return
        self.gidx = {q: i for i, q in enumerate(self.gears)}
        self.subsets = subsets_upto(self.gears, l)
        self.mask = {S: sum(1 << self.gidx[q] for q in S)
                     for S in self.subsets}
        posset = frozenset(self.pos)
        self.cols, self.block_span, self.tupidx = [], {}, {}
        for S in self.subsets:
            lo = len(self.cols)
            for r in product(*[self.dom[q] for q in S]):
                O = None
                for q, rq in zip(S, r):
                    h = _hitset(q, rq, W, posset)
                    O = h if O is None else (O & h)
                self.tupidx[(S, r)] = len(self.cols)
                self.cols.append((S, r, O))
            self.block_span[S] = (lo, len(self.cols))
        self.links = []
        for S in self.subsets:
            if len(S) < 2:
                continue
            for drop in range(len(S)):
                Sp = S[:drop] + S[drop + 1:]
                for rp in product(*[self.dom[q] for q in Sp]):
                    kids = tuple(self.tupidx[(S, rp[:drop] + (v,) + rp[drop:])]
                                 for v in self.dom[S[drop]])
                    self.links.append((self.tupidx[(Sp, rp)], kids))
        self.subs, self.sidx = _atom_tables(self.n, l)
        self.rows = [(i, base_cut(self.n, l)) for i in self.pos]
        self.bypos = {i: [] for i in self.pos}
        for j, (S, _r, O) in enumerate(self.cols):
            si = self.sidx[self.mask[S]]
            for i in O:
                self.bypos[i].append((j, si))
        # ---------------------------------------------- the recursion row
        self.inexact = 0
        self.frow = [ZERO] * len(self.cols)
        for j, (S, r, O) in enumerate(self.cols):
            if len(S) == 1:
                self.frow[j] = Fraction(len(O))
            elif len(S) == 2:
                a, b = self.gidx[S[0]], self.gidx[S[1]]
                lower = tuple(self.gears[:a])
                P = sorted(O)
                mc, ex = max_cover_dom(P, lower, W, self.dom)
                if not ex:
                    self.inexact += 1
                self.frow[j] = -Fraction(len(P) - mc)
        self.frhs = Fraction(len(self.pos))

    # ------------------------------------------------------------- helpers
    def moments_at(self, z, i):
        out = {self.mask[S]: ZERO for S in self.subsets}
        for j, si in self.bypos[i]:
            if z[j]:
                out[self.subs[si]] += z[j]
        return out

    def uniform_point(self):
        """the product measure that is uniform on every gear's DOMAIN."""
        z = [ZERO] * len(self.cols)
        for S in self.subsets:
            lo, hi = self.block_span[S]
            m = ONE
            for q in S:
                m *= Fraction(1, len(self.dom[q]))
            for j in range(lo, hi):
                z[j] = m
        return z

    def row_value(self, z):
        return sum(v * z[j] for j, v in enumerate(self.frow) if v)

    def _solve_float(self):
        import numpy as np
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
        N, R, B = len(self.cols), len(self.rows), len(self.subsets)
        ri, ci, vv = [], [], []
        bub = np.zeros(R + 1)
        for r, (i, lam) in enumerate(self.rows):
            for j, si in self.bypos[i]:
                v = lam[si]
                if v:
                    ri.append(r); ci.append(j); vv.append(-float(v))
            ri.append(r); ci.append(N); vv.append(1.0)
            bub[r] = -float(ONE - lam[0])
        for j, v in enumerate(self.frow):
            if v:
                ri.append(R); ci.append(j); vv.append(-float(v))
        ri.append(R); ci.append(N); vv.append(1.0)
        bub[R] = -float(self.frhs)
        A_ub = coo_matrix((vv, (ri, ci)), shape=(R + 1, N + 1))
        ri, ci, vv, beq = [], [], [], []
        for bi, S in enumerate(self.subsets):
            lo, hi = self.block_span[S]
            for j in range(lo, hi):
                ri.append(bi); ci.append(j); vv.append(1.0)
            beq.append(1.0)
        nr = B
        for (par, kids) in self.links:
            for j in kids:
                ri.append(nr); ci.append(j); vv.append(1.0)
            ri.append(nr); ci.append(par); vv.append(-1.0)
            beq.append(0.0)
            nr += 1
        A_eq = coo_matrix((vv, (ri, ci)), shape=(nr, N + 1))
        c = np.zeros(N + 1)
        c[-1] = -1.0
        res = linprog(c, A_ub=A_ub, b_ub=bub, A_eq=A_eq, b_eq=np.array(beq),
                      bounds=[(0, None)] * N + [(None, None)], method='highs')
        assert res.status == 0, res.message
        return -res.fun, res.x[:N], res

    def _solve_max(self):
        """THE NATURAL LP, and a round-26 reformulation of round 25's loop.

        Round 25 maximised a COMMON additive slack t over all rows.  That
        conflates two scales: the coverage cuts have right-hand side ~1 while
        the recursion row has right-hand side |pos| (78 at machine 41), so t is
        pinned by the recursion row and the coverage cuts never bind.
        MEASURED at machine 41 case (5 -> 0): t sat at exactly +0.221818 for
        six iterations while 78 cuts per pass were added, and the float LP time
        went 2.2 s -> 80.7 s as those inert rows accumulated.

        Here the objective is the quantity the certificate is about:
            maximise  sum_j frow_j z_j   subject to the cuts as HARD rows.
        A certificate exists exactly when the optimum is < frhs.  The duals of
        this LP are a certificate candidate with the recursion weight yff = 1,
        and are re-verified exactly by `certificate_star` either way."""
        import numpy as np
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
        N, Rn, B = len(self.cols), len(self.rows), len(self.subsets)
        ri, ci, vv = [], [], []
        bub = np.zeros(Rn)
        for r, (i, lam) in enumerate(self.rows):
            for j, si in self.bypos[i]:
                v = lam[si]
                if v:
                    ri.append(r); ci.append(j); vv.append(-float(v))
            bub[r] = -float(ONE - lam[0])
        A_ub = coo_matrix((vv, (ri, ci)), shape=(Rn, N))
        ri, ci, vv, beq = [], [], [], []
        for bi, S in enumerate(self.subsets):
            lo, hi = self.block_span[S]
            for j in range(lo, hi):
                ri.append(bi); ci.append(j); vv.append(1.0)
            beq.append(1.0)
        nr = B
        for (par, kids) in self.links:
            for j in kids:
                ri.append(nr); ci.append(j); vv.append(1.0)
            ri.append(nr); ci.append(par); vv.append(-1.0)
            beq.append(0.0)
            nr += 1
        A_eq = coo_matrix((vv, (ri, ci)), shape=(nr, N))
        c = np.array([-float(v) for v in self.frow])
        res = linprog(c, A_ub=A_ub, b_ub=bub, A_eq=A_eq, b_eq=np.array(beq),
                      bounds=[(0, None)] * N, method='highs')
        if res.status != 0:
            return None, None, res
        return -res.fun, res.x, res

    # -------------------------------------------------- exact verification
    def verify(self, z):
        """EXACT.  Assert z is a genuine feasible point.  Raises on failure."""
        assert all(v >= 0 for v in z), "negative entry"
        for S in self.subsets:
            lo, hi = self.block_span[S]
            assert sum(z[lo:hi]) == ONE, ("block does not sum to 1", S)
        for (par, kids) in self.links:
            assert sum(z[j] for j in kids) == z[par], "link broken"
        for i in self.pos:
            assert completable_fast(self.moments_at(z, i), self.n,
                                    self.l), \
                ("not completable at position %d" % i)
        rv = self.row_value(z)
        assert rv >= self.frhs, ("row violated", rv, self.frhs)
        return dict(row_value=rv, row_rhs=self.frhs, row_slack=rv - self.frhs)


# =============================================================== certificate
def certificate_star(R, yf, yff, nuf):
    """EXACT dual certificate for RelaxStar, same shape as `certificateCF`:
        a_j = sum_r y_r lam^r_{S(j)} [i_r in O_j] + yff * frow_j
              + sum_{links: j in kids} nu - sum_{links: par = j} nu
        certificate iff  sum_S max_{j in block S} a_j  <  sum_r y_r (1-lam^r_0)
                                                          + yff * frhs."""
    N = len(R.cols)
    scale = max(max((abs(v) for v in yf), default=0.0), abs(yff), 1e-12)
    grid = list(range(1, 65)) + [96, 128, 192, 256, 384, 512, 1024, 2048,
                                 4096, 8192, 16384, 65536,
                                 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7]
    for den, sgn in [(d, s) for d in grid for s in (1, -1)]:
        y = [max(ZERO, Fraction(round(v / scale * den), den)) for v in yf]
        yf2 = max(ZERO, Fraction(round(yff / scale * den), den))
        nu = [sgn * Fraction(round(v / scale * den), den) for v in nuf]
        if not any(y) and not yf2:
            continue
        a = [ZERO] * N
        ops = 0
        for r, (i, lam) in enumerate(R.rows):
            if not y[r]:
                continue
            for j, si in R.bypos[i]:
                v = lam[si]
                if v:
                    a[j] += y[r] * v
                    ops += 2
        if yf2:
            for j, v in enumerate(R.frow):
                if v:
                    a[j] += yf2 * v
                    ops += 2
        for k, (par, kids) in enumerate(R.links):
            if nu[k]:
                for j in kids:
                    a[j] += nu[k]
                a[par] -= nu[k]
                ops += len(kids) + 1
        lhs = ZERO
        for S in R.subsets:
            lo, hi = R.block_span[S]
            lhs += max(a[lo:hi])
            ops += hi - lo
        rhs = sum(y[r] * (ONE - lam[0])
                  for r, (i, lam) in enumerate(R.rows)) + yf2 * R.frhs
        ops += 2 * len(R.rows) + 2
        if lhs < rhs:
            return True, lhs, rhs, y, yf2, nu, ops
    return False, None, None, None, None, None, None


# ========================================================== the deciding loop
def decide_star(R, maxrounds=400, verbose=True, tag=None, fastsep=True,
                time_budget=None, filt=True):
    """EXACT decision of the restricted composition.  Verdicts:
       CERTIFIED - exact rational dual certificate;
       REFUTED   - exact rational feasible point, verified in-polytope;
       NOCERT    - the cut loop stalled and no exactly-consistent witness was
                   produced (an UNDECIDED cell, not a verdict);
       STUCK     - the time budget ran out."""
    t0 = time.time()
    l = R.l
    if R.dead:
        return 'CERTIFIED', dict(verdict='CERTIFIED', trivial=True, ops=0,
                                 secs=0.0, note=R.why or
                                 'held gear covers a required-open point')
    # the uniform-on-domain point is a free pre-test
    zu = R.uniform_point()
    mom = R.moments_at(zu, R.pos[0])
    same = all(R.moments_at(zu, i) == mom for i in R.pos)
    ok2 = completable_fast(mom, R.n, l) if same else None
    rv = R.row_value(zu)
    if same and ok2 and rv >= R.frhs:
        ver = R.verify(zu)
        return 'REFUTED', dict(verdict='REFUTED', how='uniform-on-domain',
                               its=0, secs=time.time() - t0, ops=0, **ver)
    it, final_pass = 0, False
    while it < maxrounds:
        val, z, res = R._solve_max()
        if val is None or val < float(R.frhs) - 1e-7:
            if val is None:
                # the generated cuts already make the polytope EMPTY; fall
                # back to the common-slack LP purely to get a dual vector.
                t, z2, res = R._solve_float()
                nb = len(R.subsets)
                y = list(-res.ineqlin.marginals)
                yff = y.pop()
                nu = res.eqlin.marginals[nb:]
            else:
                nb = len(R.subsets)
                y = list(-res.ineqlin.marginals)
                yff = 1.0
                nu = res.eqlin.marginals[nb:]
            ok, lhs, rhs, yq, yffq, nuq, ops = certificate_star(R, y, yff, nu)
            if not ok:
                # The float LP says a certificate exists but no rounding of
                # its dual closes the inequality EXACTLY.  That is a failure
                # to produce a certificate, not a certificate - so it is
                # reported as an undecided cell, never as CERTIFIED.  (Round
                # 25 asserted here and aborted; a whole sweep dying on one
                # awkward dual is worse than recording the cell.)
                return 'NODUAL', dict(verdict='NODUAL', its=it, lp_max=val,
                                      rows=len(R.rows), cols=len(R.cols),
                                      secs=time.time() - t0)
            info = dict(verdict='CERTIFIED', lhs=lhs, rhs=rhs, ops=ops,
                        rows=len(R.rows), cols=len(R.cols), its=it,
                        support=sum(1 for v in yq if v) + (1 if yffq else 0)
                        + sum(1 for v in nuq if v), secs=time.time() - t0)
            if tag:
                save_cert(tag, R, yq, yffq, nuq, info)
            return 'CERTIFIED', info
        den = 10 ** (4 + min(it // 40, 4))
        zex = repair_links(R, rationalise_star(R, z, den))
        margin = ZERO if final_pass else Fraction(1, 10 ** 5)
        added, skipped = 0, 0
        for i in R.pos:
            m = R.moments_at(zex, i)
            lam = sep_fast2(m, R.n, l, margin)
            if lam is not None:
                # exactly valid, exactly violated - a real cut
                R.rows.append((i, lam))
                added += 1
                continue
            if not final_pass:
                continue
            # FINAL EXACT PASS.  `separate_fast` found nothing, which decides
            # nothing.  `completable_fast` returns True only with an exact
            # completion it has verified; otherwise the exact oracle speaks.
            if completable_fast(m, R.n, l):
                continue
            lam = separate(m, R.n, l, ZERO)
            if lam is not None:
                R.rows.append((i, lam))
                added += 1
        if verbose:
            print("      it %d: max row = %.4f (need < %s), %d cuts,"
                  " %d rows%s  [%.0fs]"
                  % (it, val, R.frhs, added, len(R.rows),
                     "  FINAL EXACT PASS" if final_pass else "",
                     time.time() - t0), flush=True)
        if added == 0:
            if not final_pass:
                final_pass = True
                continue
            for how, cand in witness_candidates(R, z, zex):
                if cand is None:
                    continue
                try:
                    ver = R.verify(cand)
                    info = dict(verdict='REFUTED', how=how, its=it,
                                rows=len(R.rows), cols=len(R.cols),
                                lp_max=val, secs=time.time() - t0, **ver)
                    if tag:
                        save_wit(tag, R, cand, info)
                    return 'REFUTED', info
                except AssertionError as e:
                    if verbose:
                        print("    %s is not an exact witness (%s)" % (how, e),
                              flush=True)
            return 'NOCERT', dict(verdict='NOCERT', its=it, lp_max=val,
                                  rows=len(R.rows), secs=time.time() - t0)
        it += 1
        if time_budget is not None and time.time() - t0 > time_budget:
            return 'STUCK', dict(verdict='STUCK', its=it, rows=len(R.rows),
                                 lp_max=val, secs=time.time() - t0)
    return 'STUCK', dict(verdict='STUCK', its=it, rows=len(R.rows),
                         secs=time.time() - t0)


# --------------------------------------------------- rationalise and repair
def rationalise_star(R, z, den):
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


def repair_links(R, zx):
    for k in range(R.l - 1, 0, -1):
        for S in [s for s in R.subsets if len(s) == k]:
            par = next((T for T in R.subsets
                        if len(T) == k + 1 and set(S) <= set(T)), None)
            if par is None:
                continue
            drop = [i for i, q in enumerate(par) if q not in S][0]
            lo, hi = R.block_span[S]
            for j in range(lo, hi):
                rp = R.cols[j][1]
                zx[j] = sum(zx[R.tupidx[(par, rp[:drop] + (v,) + rp[drop:])]]
                            for v in R.dom[par[drop]])
    return zx


def witness_candidates(R, z, zex=None):
    """Exactly-consistent-by-construction candidates, in order of preference.

    ROUND-26 CORRECTION TO MY OWN ROUND-25 RULE.  Round 25 recorded that a
    rationalised LP point "is not exactly in the polytope" and built two
    special constructions to get one.  At level 2 that reading is too strong:
    `rationalise_star` normalises every SIZE-2 block to sum to exactly 1 with a
    single denominator, and `repair_links` then defines every single block as
    an exact marginal of a pair block - so the repaired point is exactly
    consistent BY CONSTRUCTION, and only its completability and its row value
    are in question.  Measured this round: at machine 19 span 28 split (2,26)
    the loop stalls with the repaired point satisfying every block sum, every
    link, the recursion row (26.75 >= 26) and completability at all 26
    positions - a perfectly good witness that round 25's candidate list would
    have thrown away, turning a REFUTED cell into an UNDECIDED one.  The rule
    that stands is the one that matters: the witness must be VERIFIED exactly
    in the polytope, whatever produced it."""
    out = []
    if zex is not None:
        out.append(('repaired-LP-point', zex))
    for den, eta in ((10 ** 6, Fraction(1, 10 ** 4)),
                     (10 ** 6, Fraction(1, 10 ** 3)),
                     (10 ** 5, Fraction(1, 10 ** 3)),
                     (10 ** 6, Fraction(1, 100)),
                     (10 ** 4, Fraction(1, 100))):
        out.append(('margin-repair d=%d eta=%s' % (den, eta),
                    margin_repair(R, z, den, eta)))
    for den in (10 ** 6, 10 ** 4, 720720, 100):
        out.append(('product-of-LP-singles d=%d' % den,
                    product_point(R, z, den)))
    out.append(('uniform-on-domain', R.uniform_point()))
    return out


def product_point(R, z, den):
    """EXACTLY CONSISTENT BY CONSTRUCTION, and the cheapest such point there
    is: round each single block to one fixed denominator and set every pair
    block to the OUTER PRODUCT of its two singles.  An outer product's
    marginals ARE its factors, so every consistency link holds identically and
    every block sums to 1.  It is the LP point's own first-order information
    with second-order information discarded - so it is a genuine witness
    candidate wherever the LP optimum is close to a product measure, which is
    exactly the regime the margin-repair construction cannot reach (that one
    needs strictly positive margin deficits)."""
    zx = [ZERO] * len(R.cols)
    p = {}
    for q in R.gears:
        lo, hi = R.block_span[(q,)]
        v = [max(ZERO, Fraction(round(float(z[j]) * den), den))
             for j in range(lo, hi)]
        s = sum(v)
        if s == 0:
            v = [Fraction(1, hi - lo)] * (hi - lo)
        else:
            k = max(range(hi - lo), key=lambda t: v[t])
            v[k] += ONE - s
            if v[k] < 0:
                v = [Fraction(1, hi - lo)] * (hi - lo)
        p[q] = v
        for t in range(hi - lo):
            zx[lo + t] = v[t]
    for a in range(R.n):
        for b in range(a + 1, R.n):
            qa, qb = R.gears[a], R.gears[b]
            S = (qa, qb)
            for iu, u in enumerate(R.dom[qa]):
                for iv, v2 in enumerate(R.dom[qb]):
                    zx[R.tupidx[(S, (u, v2))]] = p[qa][iu] * p[qb][iv]
    return zx


def margin_repair(R, z, den, eta):
    gs = R.gears
    zx = [ZERO] * len(R.cols)
    p = {}
    for q in gs:
        lo, hi = R.block_span[(q,)]
        v = [max(ZERO, Fraction(round(float(z[j]) * den), den))
             for j in range(lo, hi)]
        s = sum(v)
        kk = max(range(hi - lo), key=lambda t: v[t])
        v[kk] += ONE - s
        if v[kk] < 0:
            v = [Fraction(1, hi - lo)] * (hi - lo)
        p[q] = v
        for t in range(hi - lo):
            zx[lo + t] = v[t]
    for a in range(R.n):
        for b in range(a + 1, R.n):
            qa, qb = gs[a], gs[b]
            S = (qa, qb)
            Da, Db = R.dom[qa], R.dom[qb]
            raw = [[max(ZERO, Fraction(round(float(z[R.tupidx[(S, (u, v))]])
                                             * den), den))
                    for v in Db] for u in Da]
            tot = sum(sum(r) for r in raw)
            if tot == 0:
                return None
            sc = (ONE - eta) / tot
            Z = [[sc * raw[iu][iv] for iv in range(len(Db))]
                 for iu in range(len(Da))]
            d = [p[qa][iu] - sum(Z[iu]) for iu in range(len(Da))]
            e = [p[qb][iv] - sum(Z[iu][iv] for iu in range(len(Da)))
                 for iv in range(len(Db))]
            delta = sum(d)
            if delta != sum(e):
                return None
            if delta <= 0 or any(x < 0 for x in d) or any(x < 0 for x in e):
                return None
            for iu, u in enumerate(Da):
                for iv, v in enumerate(Db):
                    val = Z[iu][iv] + Fraction(d[iu] * e[iv], delta)
                    if val < 0:
                        return None
                    zx[R.tupidx[(S, (u, v))]] = val
    return zx


# ==================================================================== saving
def save_wit(tag, R, z, info):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, 'wit_%s.pkl' % tag)
    with open(p, 'wb') as fh:
        pickle.dump(dict(full=R.full, W=R.W, held=R.held, ws=R.ws,
                         openpts=R.openpts, l=R.l, z=z, info=info,
                         cols=[(S, r) for (S, r, _O) in R.cols]), fh)
    print("  WITNESS SAVED: %s" % p, flush=True)
    return p


def save_cert(tag, R, y, yff, nu, info):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, 'cert_%s.pkl' % tag)
    with open(p, 'wb') as fh:
        pickle.dump(dict(full=R.full, W=R.W, held=R.held, ws=R.ws,
                         openpts=R.openpts, l=R.l, rows=R.rows,
                         y=y, yff=yff, nu=nu, info=info), fh)
    print("  CERTIFICATE SAVED: %s" % p, flush=True)
    return p


def reverify(tag):
    """SECOND PASS from a clean process: rebuild and re-verify."""
    p = os.path.join(OUT, 'wit_%s.pkl' % tag)
    with open(p, 'rb') as fh:
        d = pickle.load(fh)
    R = RelaxStar(d['full'], d['W'], d['held'], d['ws'], d['openpts'], d['l'])
    assert [(S, r) for (S, r, _O) in R.cols] == d['cols'], \
        "column layout changed - witness cannot be re-verified"
    ver = R.verify(d['z'])
    print("  RE-VERIFIED %s: row %s >= %s (slack %s)"
          % (p, ver['row_value'], ver['row_rhs'], ver['row_slack']),
          flush=True)
    return ver


def case_margin(gears, W, held, ws):
    """EXACT E_u[f_ws] for the case, WITHOUT building the LP.  Uniform phases
    on the free gears; the held gears pinned at ws."""
    n = len(gears)
    k = len(held)
    U = set(range(W))
    for q, w in zip(held, ws):
        U -= hits(q, w, W)
    dom = {q: tuple(range(q)) for q in gears}
    tot = Fraction(len(U))
    for i in range(k, n):
        qi = gears[i]
        tot -= Fraction(sum(len(hits(qi, r, W) & U) for r in range(qi)), qi)
    inexact = 0
    for i in range(k, n):
        for j in range(i + 1, n):
            qi, qj = gears[i], gears[j]
            lower = tuple(gears[k:i])
            acc = 0
            for u in range(qi):
                for v in range(qj):
                    P = sorted(hits(qi, u, W) & hits(qj, v, W) & U)
                    mc, ex = max_cover_dom(P, lower, W, dom)
                    inexact += (0 if ex else 1)
                    acc += len(P) - mc
            tot += Fraction(acc, qi * qj)
    return tot, len(U), inexact


def pretest(machines=(41, 43, 47, 53), ks=(1, 2)):
    """THE CASE-SPLIT PRE-TEST.  In case ws the conditional uniform product
    measure is a point of the case polytope; every position of U_ws carries the
    SAME degree-<=2 moment vector (the product moments of p_q = 2/q over the
    FREE gears, by CRT), so ONE exact completion decides the degree-2 side of
    every case at once.  A case whose degree-2 side is satisfied AND whose row
    E_u[f_ws] <= 0 is REFUTED by that point - and one refuted case kills the
    whole case-split certificate."""
    print("=" * 78)
    print("PRETEST  the conditional uniform point, case by case")
    print("=" * 78, flush=True)
    for y in machines:
        g = gears_of(y)
        W = budget(y)
        for k in ks:
            held, free = g[:k], g[k:]
            t0 = time.time()
            p = [Fraction(2, q) for q in free]
            ok2 = completable_fast(product_moments(p, len(free), 2),
                                   len(free), 2)
            print("m%d W=%d  hold %s  free %s (n=%d)"
                  % (y, W, list(held), list(free), len(free)))
            print("   conditional deg-2 (identical at every position of every"
                  " case): %s   [%.0fs]"
                  % ("SATISFIED - no deg-2 cut bites" if ok2
                     else "VIOLATED - cuts bite", time.time() - t0),
                  flush=True)
            vals = []
            for ws in product(*[range(q) for q in held]):
                t1 = time.time()
                v, npos, inex = case_margin(g, W, held, ws)
                vals.append((ws, v))
                print("     case %-10s |U|=%3d  E_u[f] = %+12.6f %s [%.0fs]"
                      % (str(ws), npos, float(v),
                         "" if not inex else "(%d inexact cells)" % inex,
                         time.time() - t1), flush=True)
            bad = [ws for ws, v in vals if v <= 0]
            mean = sum(v for _, v in vals) / len(vals)
            print("   mean over cases %+.6f ; %d/%d cases with E_u[f] <= 0"
                  % (float(mean), len(bad), len(vals)))
            if bad and ok2:
                print("   => CASE-SPLIT REFUTED at this k: the conditional"
                      " uniform point is feasible in %d case(s)." % len(bad))
            elif not bad:
                print("   => necessary condition holds in EVERY case; the"
                      " cases must be decided by LP.")
            print(flush=True)


def run_cases(y, k=1, W=None, maxrounds=400, time_budget=None, cases=None):
    """DECIDE EVERY CASE of the k-gear case split at machine y, width W.
    A CERTIFIED verdict in every case is a CASE-SPLIT CERTIFICATE of
    F(machine y) <= W; one REFUTED case kills the species at that width."""
    g = gears_of(y)
    W = W or budget(y)
    held = g[:k]
    print("=" * 78)
    print("CASE SPLIT  machine %d, width %d, holding %s (%d cases)"
          % (y, W, list(held), 1 if not held else
             __import__('math').prod(held)))
    print("=" * 78, flush=True)
    out = []
    for ws in (cases if cases is not None
               else product(*[range(q) for q in held])):
        tag = "m%d_w%d_h%s" % (y, W, "_".join(map(str, ws)))
        t0 = time.time()
        R = RelaxStar(g, W, held, ws)
        print("  case %s: %d cols, %d links, |pos| = %d, %d inexact cells"
              " [built %.1fs]"
              % (str(ws), len(R.cols), len(R.links), len(R.pos), R.inexact,
                 time.time() - t0), flush=True)
        v, info = decide_star(R, maxrounds=maxrounds, tag=tag,
                              time_budget=time_budget)
        print("  case %s -> %s  %s\n" % (str(ws), v,
                                         {kk: vv for kk, vv in info.items()
                                          if kk not in ('R',)}), flush=True)
        out.append((ws, v, info))
        del R
    kinds = {}
    for _ws, v, _i in out:
        kinds[v] = kinds.get(v, 0) + 1
    print("  VERDICT SUMMARY: %s" % kinds)
    if kinds.get('CERTIFIED', 0) == len(out):
        print("  => CASE-SPLIT CERTIFICATE of F(m%d) <= %d" % (y, W))
    elif kinds.get('REFUTED', 0):
        print("  => NO case-split certificate at width %d: %d case(s) carry an"
              " exact feasible witness." % (W, kinds['REFUTED']))
    return out


def two_gap_geometry(W, a):
    """positions 0..W ; 0, a and W required OPEN; everything else blocked.
    That is exactly a two-gap configuration of total span W split (a, W-a)."""
    return W + 1, (0, a, W)


def run_windows(y, W, alist=None, maxrounds=400, time_budget=None,
                verbose=True):
    """THE WINDOWED (two-gap) STATEMENT.  For each split a, decide whether the
    vehicle excludes a two-gap configuration of machine y with gaps (a, W-a).
    Certifying every split is an LP proof that F_2(machine y) != W."""
    g = gears_of(y)
    A, _ = two_gap_geometry(W, 1)
    alist = alist or list(range(1, W))
    res = {}
    for a in alist:
        A, op = two_gap_geometry(W, a)
        t0 = time.time()
        R = RelaxStar(g, A, (), (), op)
        tag = "m%d_span%d_a%d" % (y, W, a)
        v, info = decide_star(R, maxrounds=maxrounds, tag=tag, verbose=False,
                              time_budget=time_budget)
        res[a] = (v, info)
        if verbose:
            print("  m%d span %d split (%d,%d): %-9s  |pos|=%d cols=%d"
                  " its=%s  [%.1fs]"
                  % (y, W, a, W - a, v, len(R.pos), len(R.cols),
                     info.get('its'), time.time() - t0), flush=True)
        del R
    nc = sum(1 for v, _ in res.values() if v == 'CERTIFIED')
    print("  m%d span %d: %d/%d splits CERTIFIED" % (y, W, nc, len(alist)))
    if nc == len(alist):
        print("  => LP PROOF that machine %d has NO two-gap window of span %d"
              % (y, W))
    return res


def f2_ladder(y, lo, hi, maxrounds=200, save=None, tb=25.0):
    """THE TWO-GAP LADDER.  For every span W in [lo, hi] and every split a,
    decide the windowed statement.  All splits certified at every span in the
    range, together with an upper bound Fcap on F(machine y) (so that
    F_2 <= 2 Fcap <= hi), is an LP-DUALITY PROOF that F_2(machine y) < lo -
    with no period scan anywhere."""
    g = gears_of(y)
    tot = dict(DEAD=0, CERTIFIED=0, REFUTED=0, NOCERT=0, STUCK=0)
    ops = 0
    rows = []
    for W in range(lo, hi + 1):
        kinds = dict(DEAD=0, CERTIFIED=0, REFUTED=0, NOCERT=0, STUCK=0)
        wops, t0 = 0, time.time()
        bad = []
        for a in range(1, W):
            A, op = two_gap_geometry(W, a)
            R = RelaxStar(g, A, (), (), op)
            if R.dead:
                kinds['DEAD'] += 1
                tot['DEAD'] += 1
                continue
            v, info = decide_star(R, maxrounds=maxrounds,
                                  verbose=False, time_budget=tb)
            kinds[v] = kinds.get(v, 0) + 1
            tot[v] = tot.get(v, 0) + 1
            wops += info.get('ops') or 0
            if v != 'CERTIFIED':
                bad.append((a, v))
            del R
        ops += wops
        rows.append((W, dict(kinds), wops, time.time() - t0))
        print("  m%d span %3d: %2d dead, %2d certified, %s  ops %d  [%.1fs]%s"
              % (y, W, kinds['DEAD'], kinds['CERTIFIED'],
                 {k: v for k, v in kinds.items()
                  if k not in ('DEAD', 'CERTIFIED') and v},
                 wops, time.time() - t0,
                 "" if not bad else "  NOT CERTIFIED: %s" % bad[:6]),
              flush=True)
    print("\n  m%d spans %d..%d: %s ; total certificate ops %d"
          % (y, lo, hi, tot, ops))
    allok = (tot['REFUTED'] == 0 and tot['NOCERT'] == 0 and tot['STUCK'] == 0)
    if allok:
        print("  => LP PROOF: machine %d has NO two-gap window of span in"
              " [%d, %d].  With F(m%d) <= %d (so F_2 <= %d), this is"
              " F_2(m%d) <= %d." % (y, lo, hi, y, hi // 2, hi, y, lo - 1))
    if save:
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, save), 'wb') as fh:
            pickle.dump(dict(y=y, lo=lo, hi=hi, rows=rows, tot=tot, ops=ops),
                        fh)
    return tot, ops, rows


def window_margin(y, W, a):
    """the uniform-on-domain row margin of the windowed problem - the analogue
    of E_u[f], and the cheap frontier test for the narrower statement."""
    g = gears_of(y)
    A, op = two_gap_geometry(W, a)
    R = RelaxStar(g, A, (), (), op)
    zu = R.uniform_point()
    return R.frhs - R.row_value(zu), R


def reverify_cert(tag):
    p = os.path.join(OUT, 'cert_%s.pkl' % tag)
    with open(p, 'rb') as fh:
        d = pickle.load(fh)
    R = RelaxStar(d['full'], d['W'], d['held'], d['ws'], d['openpts'], d['l'])
    R.rows = d['rows']
    y, yff, nu = d['y'], d['yff'], d['nu']
    N = len(R.cols)
    a = [ZERO] * N
    for r, (i, lam) in enumerate(R.rows):
        if not y[r]:
            continue
        for j, si in R.bypos[i]:
            if lam[si]:
                a[j] += y[r] * lam[si]
    if yff:
        for j, v in enumerate(R.frow):
            if v:
                a[j] += yff * v
    for k, (par, kids) in enumerate(R.links):
        if nu[k]:
            for j in kids:
                a[j] += nu[k]
            a[par] -= nu[k]
    lhs = sum(max(a[lo:hi]) for (lo, hi) in R.block_span.values())
    rhs = sum(y[r] * (ONE - lam[0])
              for r, (i, lam) in enumerate(R.rows)) + yff * R.frhs
    # every row must be an exactly VALID cut
    for (i, lam) in R.rows:
        f = zeta_values(tuple(lam), R.n, R.subs)
        assert min(f[x] for x in range(1, 1 << R.n)) >= ONE, "invalid cut row"
    assert all(v >= 0 for v in y) and yff >= 0, "negative dual weight"
    assert lhs < rhs, ("certificate does not close", lhs, rhs)
    print("  RE-VERIFIED %s: %s < %s" % (p, lhs, rhs), flush=True)
    return lhs, rhs


# ======================================================================= GATE
def gate():
    """Every structural claim of this file, asserted from scratch.
    Prints ALL ASSERTIONS GREEN or aborts."""
    import random
    from cw_consistent import RelaxCF
    from row_decay import _Ef, Ef_star
    t0 = time.time()

    # --- 1. the generalised relaxation reduces EXACTLY to round 25's vehicle
    for y, W in ((11, 16), (13, 20), (17, 28), (19, 33)):
        g = gears_of(y)
        A, B = RelaxStar(g, W), RelaxCF(g, W)
        assert [(S, r) for (S, r, _O) in A.cols] == \
               [(S, r) for (S, r, _O) in B.cols], ("columns differ", y)
        assert A.frhs == B.frhs and A.links == B.links, y
        assert A.frow == B.frow, ("recursion row differs", y)
    print("  1. RelaxStar(held=(), open=()) == RelaxCF at m11/13/17/19"
          "  (columns, links, recursion row, rhs)  GREEN", flush=True)

    # --- 2. the case decomposition reproduces round 25's two exact rows
    for y in (23, 29, 31):
        g, W = gears_of(y), budget(y)
        a, _n, _x = case_margin(g, W, (), ())
        b, _ex = _Ef(g, W)
        assert a == b, ("case_margin(held=()) != row_decay._Ef", y, a, b)
        vals = [case_margin(g, W, g[:1], (w,))[0] for w in range(g[0])]
        m = sum(vals) / len(vals)
        assert m == Ef_star(g, W, 1), ("case mean != Ef_star", y)
    print("  2. case_margin: held=() == row_decay._Ef and the case MEAN =="
          " Ef_star (STAR-3) at m23/29/31, exact rationals  GREEN", flush=True)

    # --- 3. the fast exact zeta transform
    random.seed(7)
    for n in (3, 5, 7, 9):
        subs, _ = _atom_tables(n, 2)
        for _ in range(3):
            lam = [Fraction(random.randint(-8, 8), random.choice([1, 2, 3, 4]))
                   for _ in subs]
            assert zeta_fast(lam, n, subs) == zeta_values(tuple(lam), n, subs)
    print("  3. zeta_fast == zeta_values, 12 random instances, n = 3,5,7,9"
          "  GREEN", flush=True)

    # --- 4. the fast exact completion oracle
    COMP_CACHE.clear()
    for y in (23, 29, 31):
        g = gears_of(y)
        mom = product_moments([Fraction(2, q) for q in g], len(g), 2)
        slow = completable(mom, len(g), 2)
        COMP_CACHE.clear()
        fast = completable_fast(mom, len(g), 2)
        assert slow == fast, ("completability verdicts differ", y, slow, fast)
    print("  4. completable_fast == completable at n = 7, 8, 9  GREEN",
          flush=True)

    # --- 5. the four composed rung certificates still land
    for y, W in ((11, 16), (13, 20), (17, 28)):
        v, info = decide_star(RelaxStar(gears_of(y), W), verbose=False)
        assert v == 'CERTIFIED', (y, v)
        assert info['lhs'] < info['rhs'], y
    print("  5. composed certificates reproduce at m11/13/17 (lhs < rhs,"
          " exact rationals)  GREEN", flush=True)

    # --- 6. TIGHTNESS: at span F_2 the vehicle must FAIL, and only on the
    #        true maximiser splits.  F_2(19) = 31.
    g = gears_of(19)
    bad = []
    for a in range(1, 31):
        Aw, op = two_gap_geometry(31, a)
        R = RelaxStar(g, Aw, (), (), op)
        if R.dead:
            continue
        v, info = decide_star(R, verbose=False, tag=("m19_span31_a%d" % a)
                              if True else None)
        if v != 'CERTIFIED':
            bad.append((a, v))
    assert [x[0] for x in bad] == [10, 21], ("tightness gate", bad)
    assert all(x[1] == 'REFUTED' for x in bad), bad
    for a in (10, 21):
        reverify("m19_span31_a%d" % a)
    print("  6. TIGHTNESS at m19: span 31 = F_2(19) fails on EXACTLY the two"
          " splits (10,21) and (21,10), each by an exact in-polytope witness"
          " re-verified from disk  GREEN", flush=True)

    # --- 7. one span above F_2: everything decided, a certificate re-verified
    n_c = 0
    for a in range(1, 32):
        Aw, op = two_gap_geometry(32, a)
        R = RelaxStar(g, Aw, (), (), op)
        if R.dead:
            continue
        v, info = decide_star(R, verbose=False, tag="m19_span32_a%d" % a)
        assert v == 'CERTIFIED', (a, v)
        n_c += 1
    assert n_c == 24, n_c
    reverify_cert("m19_span32_a2")
    print("  7. m19 span 32: 7 splits dead by gear 5, 24 CERTIFIED; one"
          " certificate re-verified from disk  GREEN", flush=True)

    # --- 8. THE RUNG.  19 -> 23 at the ladder's budget width 48 - the cell
    #        round 25 REFUTED for the level-2 vehicle with an exact witness.
    gg, WW = gears_of(23), budget(23)
    ops = 0
    for w in range(5):
        R = RelaxStar(gg, WW, (5,), (w,))
        v, info = decide_star(R, verbose=False, tag="gate_m23_w48_h%d" % w)
        assert v == 'CERTIFIED', ("rung 19->23 case %d: %s" % (w, v))
        assert info['lhs'] < info['rhs'] and info['its'] == 0, w
        ops += info['ops']
    for w in range(5):
        reverify_cert("gate_m23_w48_h%d" % w)
    assert ops == 38677, ("certificate op count changed", ops)
    print("  8. THE RUNG: 19->23 at budget width 48 CASE-SPLIT CERTIFIED - all"
          " five cases at iteration zero, %d exact ops, every certificate"
          " re-verified from disk  GREEN" % ops, flush=True)
    print("\n  ALL ASSERTIONS GREEN  [%.0fs]" % (time.time() - t0))


# ======================================================================= CLI
def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        return
    cmd = a[0].upper()
    if cmd == 'PRETEST':
        ms = tuple(int(x) for x in a[1:2]) or (41, 43, 47, 53)
        ks = tuple(int(x) for x in a[2:3]) or (1, 2)
        pretest(ms, ks)
    elif cmd == 'CASE':
        y = int(a[1]); k = int(a[2]) if len(a) > 2 else 1
        run_cases(y, k)
    elif cmd == 'WINDOW':
        y = int(a[1]); W = int(a[2])
        al = [int(x) for x in a[3:]] or None
        run_windows(y, W, al)
    elif cmd == 'F2':
        y, lo, hi = int(a[1]), int(a[2]), int(a[3])
        f2_ladder(y, lo, hi, save='f2_m%d_%d_%d.pkl' % (y, lo, hi))
    elif cmd == 'GATE':
        gate()
    else:
        print("unknown command %r" % cmd)


if __name__ == '__main__':
    main()
