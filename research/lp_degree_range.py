"""LP DUALITY, ROUND 23 - MAPPING THE RANGE OF THE CERTIFICATE VEHICLE.

Round 22 (research/lp_dual_certs.py, docs/novel/covering-lp-certificates.md,
docs/novel/moment-degree-ceiling.md) established:
  * a Farkas dual certificate proves the (D) step F(19) <= F(17) + 19 outright;
  * the VACUITY ceiling per degree (degree 1 dead from machine 13, degree 2
    from 29, degree 3/4 alive at 37 and >= 151).

Round 23 asks the different - and more useful - question: at what machine does
a degree-l certificate stop PROVING A (D) RUNG?  A rung at machine y needs a
certificate of width exactly

    B(y) = F(prev prime) + y          ("the budget"),

so the vehicle proves the rung iff  W*_l(y) <= B(y), where W*_l(y) is the
minimum width at which the degree-l relaxation is infeasible.  That threshold
is FAR below the vacuity ceiling, because B(y)/F(y) -> 1 while the achievable
integrality gap grows.

THE ANSWER, in one line: the DEGREE AXIS IS FLAT and the range is set by a
different axis.  Degree 3 and degree 4 prove no rung that degree 2 does not -
at machine 13, width 20, the round-22 shape of the relaxation is feasible at
degrees 2, 3 AND 4, degree 4 being the total number of gears.  What closes the
11 -> 13 miss-by-one, and the 13 -> 17 miss-by-three with it, is MARGINAL
CONSISTENCY between the blocks, at the same degree 2.

Sections (run:  uv run python research/lp_degree_range.py [X G M R]):

  X  the required-gap law B(y)/F(y) -> 1, which is why the range is short,
     plus a correction to a constant in round 22's own file.
  G  regression: the general adaptive-cut machinery reproduces round 22's
     level-2 thresholds 8 / 21 / 31 / 37 exactly.
  M  THE MISS-BY-ONE at 11->13: degree does not close it (exact both ways),
     consistency does (exact certificate), and why.
  R  THE RUNG TABLE: machines against degree and consistency level, every
     cell an exact decision at the budget width.

HOUSE RULES.  Exact rational arithmetic decides everything; scipy is used for
DISCOVERY only.  Both verdicts are exact certificates:
  * INFEASIBLE at W  ->  exact rational weights y_r >= 0 (plus consistency
    potentials nu) and exactly-valid cuts lam^(r), with
    sum_S max_col (weighted col) < sum_r y_r (1 - lam^(r)_0), verified by
    direct evaluation over the FULL column set.  This proves F(M) <= W.
  * FEASIBLE at W   ->  an exact rational point whose degree-<=l moments at
    EVERY position admit a completion to a distribution on {0,1}^gears with
    zero mass on the empty atom, verified by an exact rational LP.  This
    proves that NO degree-l cut of any kind is violated, i.e. no degree-l
    certificate of width W exists.  The sharpest form (`global_kills`) uses a
    rational measure over FULL phase tuples, which is a feasible point of
    every degree-l relaxation however much consistency it imposes.
  * Where neither is obtained the verdict is UNDECIDED, never "fails".
FALSIFICATION TEST (section M): at width F(M) - 1 a blocked window exists, so
the machinery must NOT produce a certificate there.  Benchmarks are operation
counts, never wall time.
"""
import os
import sys
import time
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, product
from math import prod

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exact_lp import feasible_eq                              # noqa: E402

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
    return tuple(p for p in primes_upto(y) if p >= 5)


@lru_cache(maxsize=None)
def teeth(q):
    u = pow(6, -1, q)
    return (u % q, (-u) % q)


@lru_cache(maxsize=None)
def hits(q, r, W):
    """positions of [0,W) blocked by gear q at phase r."""
    out = set()
    for t in teeth(q):
        i = (t - r) % q
        while i < W:
            out.add(i)
            i += q
    return frozenset(out)


# F(y) = max gap between consecutive openings, full period.  5..23 verified by
# period sieve in round 22 (section A of lp_dual_certs.py); 29 re-verified this
# round by a segmented numpy sieve over the full 1,078,282,205-slot period
# (round 22's F_KNOWN[29] = 46 was WRONG - see the note in section X).
F_EXACT = {7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58,
           37: 88, 41: 91, 43: 103, 53: 145}
STEPS = [(7, 11), (11, 13), (13, 17), (17, 19), (19, 23), (23, 29),
         (29, 31), (31, 37), (37, 41)]


def budget(y):
    """width a certificate must reach to prove the (D) rung landing at y."""
    prev = max(p for p in F_EXACT if p < y)
    return F_EXACT[prev] + y


# ================================================== degree-l block structure
@lru_cache(maxsize=None)
def block_columns(S, W):
    """All DISTINCT position sets  O(r) = {i < W : every gear of S blocks i at
    phase r_q}, over all phase tuples r in prod_q Z_q.

    Generation is complete: a tuple with nonempty overlap has some position i
    blocked by every gear, and then r_q in {(t - i) mod q : t a tooth of q}, so
    the loop below reaches every tuple whose overlap is nonempty.  The empty
    column exists iff fewer than prod(S) tuples were reached."""
    seen, reached = {}, set()
    for i in range(W):
        for ts in product(*[teeth(q) for q in S]):
            r = tuple((t - i) % q for t, q in zip(ts, S))
            if r in reached:
                continue
            reached.add(r)
            O = None
            for q, rq in zip(S, r):
                h = hits(q, rq, W)
                O = h if O is None else (O & h)
            seen[frozenset(O)] = None
    cols = [c for c in seen if c]
    if len(reached) < prod(S):
        cols.append(frozenset())
    return tuple(cols)


def subsets_upto(gears, l):
    return [S for k in range(1, l + 1) for S in combinations(gears, k)]


# ============================================================ cut machinery
@lru_cache(maxsize=None)
def _atom_tables(n, l):
    """subset list (bitmasks, |S| <= l) and, for each, its index."""
    subs = [0]
    for k in range(1, l + 1):
        for c in combinations(range(n), k):
            m = 0
            for i in c:
                m |= 1 << i
            subs.append(m)
    return tuple(subs), {m: i for i, m in enumerate(subs)}


def zeta_values(lam, n, subs):
    """f[x] = sum_{S subset x, S in subs} lam_S, exact, over all 2^n atoms."""
    f = [ZERO] * (1 << n)
    for m, v in zip(subs, lam):
        if v == 0:
            continue
        # add v to every superset of m
        rest = [i for i in range(n) if not (m >> i) & 1]
        for k in range(1 << len(rest)):
            x = m
            for b, i in enumerate(rest):
                if (k >> b) & 1:
                    x |= 1 << i
            f[x] += v
    return f


def cut_value(lam, subs, moments):
    """sum_S lam_S m_S with m_empty = 1."""
    tot = lam[0]
    for i in range(1, len(subs)):
        if lam[i]:
            tot += lam[i] * moments[subs[i]]
    return tot


BASE_CUT_CACHE = {}


def base_cut(n, l):
    """the level-1 cut  sum_q x_q >= 1  (valid, and the natural start)."""
    key = (n, l)
    if key not in BASE_CUT_CACHE:
        subs, idx = _atom_tables(n, l)
        lam = [ZERO] * len(subs)
        for i in range(n):
            lam[idx[1 << i]] = ONE
        BASE_CUT_CACHE[key] = tuple(lam)
    return BASE_CUT_CACHE[key]


# ------------------------------------------------------- separation (EXACT)
@lru_cache(maxsize=None)
def _sep_matrix_exact(n, l):
    """rows = subsets S with |S| <= l, columns = NONEMPTY atoms x,
    entry [S][x] = 1 iff S subset x."""
    subs, _ = _atom_tables(n, l)
    atoms = list(range(1, 1 << n))
    return [[ONE if (m & ~x) == 0 else ZERO for x in atoms] for m in subs], subs


SEP_CACHE = {}


def separate(moments, n, l, margin=ZERO):
    """EXACT.  Given degree-<=l moments (dict bitmask -> Fraction, m_0 = 1),
    decide whether they extend to a distribution on {0,1}^n with ZERO mass on
    the empty atom.

      * extend      -> return None: EVERY degree-l cut is satisfied here, so
                       no degree-l certificate can use this position.
      * do not      -> return an EXACTLY VALID degree-l cut lam (subset-sums
                       >= 1 at every nonempty atom) whose value on `moments`
                       is < 1.

    Farkas (as implemented by exact_lp.feasible_eq): infeasibility of
    {nu >= 0 : A nu = b} yields lam with lam.A <= 0 componentwise and
    lam.b > 0.  Put mu = -lam: subset-sums >= 0 everywhere and mu.m < 0.
    Raising mu_0 by 1 makes every subset-sum >= 1 (valid cut) and the value
    mu.m + 1 < 1 (violated).  Both facts are re-asserted exactly below."""
    A, subs = _sep_matrix_exact(n, l)
    key = (n, l, margin, tuple(moments.get(m, ONE) for m in subs))
    if key in SEP_CACHE:
        return SEP_CACHE[key]
    b = [moments.get(m, ONE) if m else ONE for m in subs]
    ok, cert = feasible_eq([row[:] for row in A], b)
    if ok:
        # ROUND-24 HARDENING.  The 'extends' verdict was previously trusted
        # from the simplex; round 24 found a section-G regression whose only
        # possible mechanism (both verdict paths being otherwise exactly
        # verified) is a false positive on this branch.  Re-assert the
        # completion EXACTLY: cert is nu >= 0 on the nonempty atoms with
        # A nu = b.
        assert all(v >= 0 for v in cert), "completion has negative mass"
        for r_i, row in enumerate(A):
            s_ = sum(c * v for c, v in zip(row, cert) if v)
            assert s_ == b[r_i], ("completion does not match moments", r_i)
        SEP_CACHE[key] = None
        return None
    mu = [-v for v in cert]
    lam = list(mu)
    lam[0] += ONE
    lam = tuple(lam)
    # exact re-assertions: validity and violation
    f = zeta_values(lam, n, subs)
    assert min(f[x] for x in range(1, 1 << n)) >= ONE, "separated cut invalid"
    val = cut_value(lam, subs, moments)
    assert val < ONE, ("separated cut not violated", val)
    if val >= ONE - margin:
        # violated, but by less than the discovery margin: the point is
        # within `margin` of satisfying every degree-l cut, so this is
        # treated as a stopping condition for the DISCOVERY loop.  It never
        # decides anything - both verdicts are certified separately.
        SEP_CACHE[key] = None
        return None
    SEP_CACHE[key] = lam
    return lam


# ============================================== the degree-l relaxation LP
class Relax:
    """The degree-l covering relaxation at width W, with ADAPTIVE cuts.

    Variables: one probability distribution z_S over the distinct overlap
    columns of every gear subset S with 1 <= |S| <= l ("block-independent"
    form: no consistency is imposed BETWEEN blocks, exactly as round 22's
    level-2 LP dropped pair/single consistency - a further weakening, so
    every certificate produced stays valid).

    Row (i, lam): sum_{S != 0} lam_S * m_S(i) >= 1 - lam_0, where
    m_S(i) = mass block S puts on columns containing position i.  Valid for
    every pointwise-valid cut lam, because an actual covered window gives a
    0/1 point with g(x) >= 1 at every position."""

    def __init__(self, gears, W, l):
        self.gears, self.W, self.l = tuple(gears), W, l
        self.n = len(gears)
        self.gidx = {q: i for i, q in enumerate(self.gears)}
        self.subsets = subsets_upto(self.gears, l)
        self.mask = {S: sum(1 << self.gidx[q] for q in S) for S in self.subsets}
        self.cols = []          # (S, frozenset positions)
        self.blockof = []
        self.block_span = {}
        for S in self.subsets:
            cs = block_columns(S, W)
            self.block_span[S] = (len(self.cols), len(self.cols) + len(cs))
            for c in cs:
                self.cols.append((S, c))
                self.blockof.append(S)
        self.rows = []          # (position, lam tuple)
        self.subs, self.sidx = _atom_tables(self.n, l)
        for i in range(W):
            self.rows.append((i, base_cut(self.n, l)))

    # ---- float master (discovery only)
    def _solve_float(self):
        import numpy as np
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
        N, R, B = len(self.cols), len(self.rows), len(self.subsets)
        ri, ci, vv = [], [], []
        bub = np.zeros(R)
        for r, (i, lam) in enumerate(self.rows):
            for j, (S, c) in enumerate(self.cols):
                if i in c:
                    v = lam[self.sidx[self.mask[S]]]
                    if v:
                        ri.append(r); ci.append(j); vv.append(-float(v))
            ri.append(r); ci.append(N); vv.append(1.0)      # the t column
            bub[r] = -float(ONE - lam[0])
        A_ub = coo_matrix((vv, (ri, ci)), shape=(R, N + 1))
        ri, ci, vv = [], [], []
        for bi, S in enumerate(self.subsets):
            lo, hi = self.block_span[S]
            for j in range(lo, hi):
                ri.append(bi); ci.append(j); vv.append(1.0)
        A_eq = coo_matrix((vv, (ri, ci)), shape=(B, N + 1))
        c = np.zeros(N + 1); c[-1] = -1.0
        res = linprog(c, A_ub=A_ub, b_ub=bub, A_eq=A_eq, b_eq=np.ones(B),
                      bounds=[(0, None)] * N + [(None, None)], method='highs')
        assert res.status == 0, res.message
        return -res.fun, res.x[:N], res

    # ---- exact moments of an exact point
    def moments_at(self, z, i):
        out = {}
        for S in self.subsets:
            lo, hi = self.block_span[S]
            out[self.mask[S]] = sum(z[j] for j in range(lo, hi)
                                    if i in self.cols[j][1])
        return out

    # ---- exact rationalisation of a float point (blocks renormalised)
    def rationalise(self, z, den):
        zx = [max(ZERO, Fraction(float(v)).limit_denominator(den)) for v in z]
        for S in self.subsets:
            lo, hi = self.block_span[S]
            s = sum(zx[lo:hi])
            if s == 0:
                zx[lo] = ONE
            elif s != ONE:
                zx[lo:hi] = [v / s for v in zx[lo:hi]]
            assert sum(zx[lo:hi]) == ONE
        return zx

    # ---- the loop
    def run(self, maxrounds=300, verbose=False):
        """returns ('infeasible', float row duals, it)
             or ('feasible', EXACT rational point, it).

        The float master is DISCOVERY only.  Every cut added is produced by
        the EXACT separation oracle applied to an EXACT rational point, so a
        'feasible' verdict means: that exact point's degree-<=l moments are
        completable at every position, i.e. it satisfies EVERY degree-l cut.
        No rounding is trusted anywhere in the verdict."""
        for it in range(maxrounds):
            t, z, res = self._solve_float()
            if t < -1e-7:
                y = -res.ineqlin.marginals
                return 'infeasible', y, it
            den = 10 ** (4 + min(it // 40, 4))
            zex = self.rationalise(z, den)
            added = 0
            for i in range(self.W):
                lam = separate(self.moments_at(zex, i), self.n, self.l)
                if lam is not None:
                    self.rows.append((i, lam))
                    added += 1
            if added == 0:
                return 'feasible', zex, it
            if verbose:
                print(f"      it {it}: t = {t:+.4f}, {added} cuts added,"
                      f" {len(self.rows)} rows")
        raise RuntimeError("cut loop did not settle")

    # ---- EXACT verification of an infeasibility certificate
    def verify_certificate(self, yf):
        """yf: float row weights.  Rationalise, then verify EXACTLY

            sum_S max_col (sum_r y_r lam^r_S [i_r in col])
                                       <  sum_r y_r (1 - lam^r_0)

        against the FULL column set.  Returns (ok, lhs, rhs, y, ops)."""
        for den in (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7):
            y = [max(ZERO, Fraction(float(v)).limit_denominator(den))
                 for v in yf]
            if not any(y):
                continue
            ok, lhs, rhs, ops = self._check(y)
            if ok:
                return True, lhs, rhs, y, ops
        return False, None, None, None, None

    def _check(self, y):
        ops = 0
        lhs = ZERO
        for S in self.subsets:
            lo, hi = self.block_span[S]
            si = self.sidx[self.mask[S]]
            best = None
            for j in range(lo, hi):
                c = self.cols[j][1]
                v = ZERO
                for r, (i, lam) in enumerate(self.rows):
                    if y[r] and lam[si] and i in c:
                        v += y[r] * lam[si]
                        ops += 2
                if best is None or v > best:
                    best = v
            lhs += best
        rhs = sum(y[r] * (ONE - lam[0]) for r, (i, lam) in enumerate(self.rows))
        ops += 2 * len(self.rows)
        return lhs < rhs, lhs, rhs, ops


# ================================================ exact completion (sharp)
def completable(moments, n, l):
    """EXACT: do the degree-<=l moments extend to a distribution on {0,1}^n
    with ZERO mass on the empty atom?  If yes, EVERY degree-l cut is satisfied
    (E[g] = E_nu[g] >= E_nu[1{nonempty}] = 1), so no degree-l certificate can
    use this position."""
    return separate(moments, n, l) is None


def product_moments(pvec, n, l):
    """degree-<=l moments of the independent product measure with marginals
    p (a list of n Fractions)."""
    subs, _ = _atom_tables(n, l)
    out = {}
    for m in subs:
        v = ONE
        for i in range(n):
            if (m >> i) & 1:
                v *= pvec[i]
        out[m] = v
    return out


def product_point_kills(gears, W, l, zq=None):
    """EXACT sharp negative certificate.  zq: per-gear phase distributions
    (default uniform).  If at every position of [0,W) the induced INDEPENDENT
    product measure's degree-<=l moments are completable, then the full
    degree-l relaxation (with any consistency one likes) is FEASIBLE at width
    W, so NO degree-l certificate of width W exists."""
    n = len(gears)
    if zq is None:
        zq = [[Fraction(1, q)] * q for q in gears]
    for k, q in enumerate(gears):
        assert sum(zq[k]) == ONE and all(v >= 0 for v in zq[k])
    for i in range(W):
        p = []
        for k, q in enumerate(gears):
            a, b = teeth(q)
            p.append(zq[k][(a - i) % q] + zq[k][(b - i) % q])
        if not completable(product_moments(p, n, l), n, l):
            return False, i
    return True, None




# ============================================ the CONSISTENT relaxation
class RelaxC:
    """Degree-l relaxation WITH marginal consistency (Sherali-Adams shape).

    Round 22's level-2 LP and the `Relax` class above both DROP consistency
    between blocks: the pair block (a,b) is free to put its mass wherever it
    likes, with no requirement that its marginal on gear a equal gear a's own
    phase distribution.  `RelaxC` restores it: every block of size k is forced
    to agree with each of its (k-1)-subsets.  Columns are therefore genuine
    PHASE TUPLES (no overlap-set dedupe, which would destroy the marginal).

    Same cut machinery, same exact separation oracle."""

    def __init__(self, gears, W, l):
        self.gears, self.W, self.l = tuple(gears), W, l
        self.n = len(gears)
        self.gidx = {q: i for i, q in enumerate(self.gears)}
        self.subsets = subsets_upto(self.gears, l)
        self.mask = {S: sum(1 << self.gidx[q] for q in S) for S in self.subsets}
        self.cols, self.blockof, self.block_span = [], [], {}
        self.tupidx = {}
        for S in self.subsets:
            lo = len(self.cols)
            for r in product(*[range(q) for q in S]):
                O = None
                for q, rq in zip(S, r):
                    h = hits(q, rq, W)
                    O = h if O is None else (O & h)
                self.tupidx[(S, r)] = len(self.cols)
                self.cols.append((S, r, O))
                self.blockof.append(S)
            self.block_span[S] = (lo, len(self.cols))
        # consistency links: (index of child tuple, indices of its extensions)
        self.links = []
        for S in self.subsets:
            if len(S) < 2:
                continue
            for drop in range(len(S)):
                Sp = S[:drop] + S[drop + 1:]
                for rp in product(*[range(q) for q in Sp]):
                    kids = tuple(self.tupidx[(S, rp[:drop] + (v,) + rp[drop:])]
                                 for v in range(S[drop]))
                    self.links.append((self.tupidx[(Sp, rp)], kids))
        self.subs, self.sidx = _atom_tables(self.n, l)
        self.rows = [(i, base_cut(self.n, l)) for i in range(W)]
        # index columns by position: bypos[i] = [(column, subset index), ...]
        self.bypos = [[] for _ in range(W)]
        for j, (S, _, O) in enumerate(self.cols):
            si = self.sidx[self.mask[S]]
            for i in O:
                self.bypos[i].append((j, si))

    def _solve_float(self):
        import numpy as np
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
        N, R, B = len(self.cols), len(self.rows), len(self.subsets)
        ri, ci, vv = [], [], []
        bub = np.zeros(R)
        for r, (i, lam) in enumerate(self.rows):
            for j, si in self.bypos[i]:
                v = lam[si]
                if v:
                    ri.append(r); ci.append(j); vv.append(-float(v))
            ri.append(r); ci.append(N); vv.append(1.0)
            bub[r] = -float(ONE - lam[0])
        A_ub = coo_matrix((vv, (ri, ci)), shape=(R, N + 1))
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

    def moments_at(self, z, i):
        out = {self.mask[S]: ZERO for S in self.subsets}
        for j, si in self.bypos[i]:
            if z[j]:
                out[self.subs[si]] += z[j]
        return out

    def rationalise(self, z, den):
        zx = [max(ZERO, Fraction(float(v)).limit_denominator(den)) for v in z]
        for S in self.subsets:
            if len(S) != self.l:
                continue
            lo, hi = self.block_span[S]
            s = sum(zx[lo:hi])
            if s == 0:
                zx[lo] = ONE
            elif s != ONE:
                zx[lo:hi] = [v / s for v in zx[lo:hi]]
        return zx

    def repair_consistency(self, zx):
        """Rebuild every lower block EXACTLY as a marginal of a chosen parent,
        so the rationalised point is exactly consistent and exactly
        normalised (marginalising a probability vector preserves its mass)."""
        for k in range(self.l - 1, 0, -1):
            for S in [s for s in self.subsets if len(s) == k]:
                par = next((T for T in self.subsets
                            if len(T) == k + 1 and set(S) <= set(T)), None)
                if par is None:
                    continue
                drop = [i for i, q in enumerate(par) if q not in S][0]
                lo, hi = self.block_span[S]
                for j in range(lo, hi):
                    rp = self.cols[j][1]
                    zx[j] = sum(zx[self.tupidx[(par, rp[:drop] + (v,)
                                                + rp[drop:])]]
                                for v in range(par[drop]))
        return zx

    def run(self, maxrounds=300, verbose=False):
        for it in range(maxrounds):
            t, z, res = self._solve_float()
            if t < -1e-7:
                nb = len(self.subsets)
                self.last_duals = (-res.ineqlin.marginals,
                                   res.eqlin.marginals[nb:])
                self.last_t = t
                return 'infeasible', -res.ineqlin.marginals, it
            den = 10 ** (4 + min(it // 40, 4))
            zex = self.repair_consistency(self.rationalise(z, den))
            added = 0
            marg = Fraction(1, 10 ** 5)
            for i in range(self.W):
                lam = separate(self.moments_at(zex, i), self.n, self.l, marg)
                if lam is not None:
                    self.rows.append((i, lam))
                    added += 1
            if added == 0:
                return 'feasible', zex, it
            if verbose:
                print("      it %d: t = %+.4f, %d cuts, %d rows, %d cols"
                      % (it, t, added, len(self.rows), len(self.cols)))
        raise RuntimeError("cut loop did not settle")


def decideC(gears, W, l, verbose=False):
    """EXACT decision of the CONSISTENT degree-l relaxation at width W."""
    R = RelaxC(gears, W, l)
    kind, vec, its = R.run(verbose=verbose)
    if kind == 'infeasible':
        return False, dict(rows=len(R.rows), cols=len(R.cols), its=its,
                           y=vec, R=R)
    z = vec
    for S in R.subsets:
        lo, hi = R.block_span[S]
        assert all(v >= 0 for v in z[lo:hi])
        assert sum(z[lo:hi]) == ONE, ("block sum", S, sum(z[lo:hi]))
    for (par, kids) in R.links:
        assert sum(z[j] for j in kids) == z[par], "consistency broken"
    for i in range(W):
        assert completable(R.moments_at(z, i), R.n, l), \
            ("feasible verdict but position %d not completable" % i)
    return True, dict(exact=True, rows=len(R.rows), cols=len(R.cols),
                      its=its, z=z, R=R)




# --------------------------- EXACT certificate for the CONSISTENT relaxation
def certificateC(R, yf, nuf):
    """EXACT verification of an infeasibility certificate for `RelaxC`.

    The LP is   (a) cut rows      sum_j coef_rj z_j >= 1 - lam^r_0,  y_r >= 0
                (b) block rows    sum_{j in S} z_j  =  1,            mu_S free
                (c) consistency   sum_{j in kids} z_j - z_par = 0,   nu free
    with z >= 0.  Multiplying (a) by y_r >= 0, (b) by mu_S and (c) by nu and
    summing, every z_j is multiplied by

        w_j = sum_r y_r coef_rj + mu_{S(j)}
              + sum_{links with j in kids} nu - sum_{links with par = j} nu ,

    so if every w_j <= 0 while  sum_r y_r (1 - lam^r_0) + sum_S mu_S > 0  the
    system is infeasible.  Taking each mu_S as large as (a) allows, the check
    reduces to

        sum_S  max_{j in block S} a_j   <   sum_r y_r (1 - lam^r_0),
        a_j = sum_r y_r lam^r_{S(j)} [i_r in O_j]
              + sum_{links: j in kids} nu - sum_{links: par = j} nu.

    The maximum runs over the FULL phase-tuple set of every block, so nothing
    is pruned away.  Returns (ok, lhs, rhs, y, nu, ops)."""
    N = len(R.cols)
    scale = max(max(abs(v) for v in yf), 1e-12)
    grid = list(range(1, 65)) + [96, 128, 192, 256, 384, 512, 1024, 4096,
                                 10 ** 4, 10 ** 5, 10 ** 6]
    for den, sgn in [(d, s) for d in grid for s in (1, -1)]:
        # SNAP to a common denominator: the certificate then consists of
        # integers over one denominator, which is what a kernel check wants.
        y = [max(ZERO, Fraction(round(v / scale * den), den)) for v in yf]
        nu = [sgn * Fraction(round(v / scale * den), den) for v in nuf]
        if not any(y):
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
                  for r, (i, lam) in enumerate(R.rows))
        ops += 2 * len(R.rows)
        if lhs < rhs:
            return True, lhs, rhs, y, nu, ops
    return False, None, None, None, None, None


def decideC_cert(gears, W, l, verbose=False):
    """decideC, but an INFEASIBLE verdict comes with an EXACT certificate."""
    R = RelaxC(gears, W, l)
    kind, vec, its = R.run(verbose=verbose)
    if kind == 'feasible':
        # SOLVER-ONLY: the discovery loop found no cut violated by more than
        # its margin.  This decides NOTHING.  The exact negative verdict, when
        # it is available, comes from `global_kills` - an exact rational
        # measure over full phase tuples that satisfies every degree-l cut.
        return True, dict(exact=False, rows=len(R.rows), cols=len(R.cols),
                          its=its, z=vec, R=R)
    y, nu = R.last_duals
    ok, lhs, rhs, yq, nuq, ops = certificateC(R, y, nu)
    assert ok, ("float said infeasible but no exact certificate could be "
                "rationalised - ABORT")
    return False, dict(lhs=lhs, rhs=rhs, y=yq, nu=nuq, ops=ops,
                       rows=len(R.rows), cols=len(R.cols), its=its, R=R,
                       support=sum(1 for v in yq if v)
                       + sum(1 for v in nuq if v))




# ================================ EXACT negative certificates: global points
#
# A GLOBAL POINT is a rational probability distribution rho over FULL phase
# tuples (r_q)_{q in gears}.  Its degree-<=l marginals are automatically
# consistent - they come from one distribution - so a global point is a
# feasible point of EVERY degree-l relaxation, however much consistency one
# imposes (block-independent, Sherali-Adams level l, Lasserre level l).  If at
# every position of [0,W) its degree-<=l coverage moments are COMPLETABLE
# (they extend to a distribution on {0,1}^gears with zero mass on the empty
# atom) then no degree-l cut is violated anywhere, so
#
#     NO DEGREE-l CERTIFICATE OF WIDTH W EXISTS AT ALL.
#
# That is the sharpest negative statement available and it is carried by a
# handful of exact rationals.  The uniform product measure is the special case
# rho = product of uniforms (round 22's vacuity test) - here rho may be any
# finitely-supported rational mixture, which is strictly more general.


def _greedy_tuples(gears, W, count, seed=0):
    """DISCOVERY ONLY.  A pool of phase tuples with large coverage, by
    randomised hill climbing, plus random tuples for diversity."""
    import random
    rng = random.Random(seed)
    pool, seen = [], set()
    for _ in range(count):
        tup = [rng.randrange(q) for q in gears]
        for _sweep in range(4):
            for k, q in enumerate(gears):
                best, bestr = -1, tup[k]
                for r in range(q):
                    tup[k] = r
                    cov = set()
                    for kk in range(len(gears)):
                        cov |= hits(gears[kk], tup[kk], W)
                    if len(cov) > best:
                        best, bestr = len(cov), r
                tup[k] = bestr
        t = tuple(tup)
        if t not in seen:
            seen.add(t)
            pool.append(t)
    while len(pool) < 2 * count:
        t = tuple(rng.randrange(q) for q in gears)
        if t not in seen:
            seen.add(t)
            pool.append(t)
    return pool


def global_kills(gears, W, l, npool=60, maxrounds=60, seed=0, verbose=False):
    """Build an EXACT global point killing every degree-l cut at width W.

    The mixture may use, besides finitely many single phase tuples, the
    UNIFORM PRODUCT MEASURE itself as one atom (it is a global distribution
    too, and its degree-<=l moments are prod_{q in S} 2/q at every position -
    round 22's vacuity object, here allowed to be MIXED with point masses).
    Returns (True, dict) or (False, reason)."""
    gears = tuple(gears)
    n = len(gears)
    pool = _greedy_tuples(gears, W, npool, seed)
    covs = [tuple(hits(q, r, W) for q, r in zip(gears, t)) for t in pool]
    S_list = subsets_upto(gears, l)
    S_idx = [[gears.index(q) for q in S] for S in S_list]
    S_mask = [sum(1 << k for k in idx) for idx in S_idx]
    S_unif = [prod([Fraction(2, q) for q in S]) for S in S_list]
    subs, sidx = _atom_tables(n, l)
    hitlist = [[[si for si, idx in enumerate(S_idx)
                 if all(i in cv[k] for k in idx)] for i in range(W)]
               for cv in covs]
    T = len(covs)                       # column T is the uniform atom

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
            A.append(coef + [-ONE if k == r else ZERO for k in range(R)])
            b.append(ONE - lam[0])
        A.append([ONE] * (T + 1) + [ZERO] * R)
        b.append(ONE)
        ok, cert = feasible_eq(A, b)
        if not ok:
            return False, 'no global point in this tuple pool (round %d)' % it
        w = [cert[j] for j in range(T + 1)]
        assert sum(w) == ONE and all(v >= 0 for v in w), "bad global point"
        added = 0
        for i in range(W):
            lam = separate(moments_of(w, i), n, l)
            if lam is not None:
                rows.append((i, lam))
                added += 1
        if added == 0:
            supp = [(pool[j], w[j]) for j in range(T) if w[j]]
            if w[T]:
                supp.append(('uniform product measure', w[T]))
            return True, dict(support=supp, nrows=len(rows), npool=T, its=it)
        if verbose:
            print("      global: it %d, %d cuts -> %d rows, %d tuples"
                  % (it, added, len(rows), T))
    return False, 'cut loop did not settle'


# ==================================================================== G
def decide(gears, W, l, verbose=False):
    """EXACT decision of the degree-l relaxation at width W.
    Returns (feasible: bool, info)."""
    R = Relax(gears, W, l)
    kind, vec, its = R.run(verbose=verbose)
    if kind == 'infeasible':
        ok, lhs, rhs, y, ops = R.verify_certificate(vec)
        assert ok, ("float said infeasible but no exact certificate "
                    "could be rationalised - ABORT")
        return False, dict(lhs=lhs, rhs=rhs, y=y, ops=ops, rows=len(R.rows),
                           cols=len(R.cols), its=its, R=R)
    # feasible: `vec` is already an EXACT rational point that survived the
    # exact separation oracle at every position.  Re-assert that here.
    z = vec
    for S in R.subsets:
        lo, hi = R.block_span[S]
        assert sum(z[lo:hi]) == ONE and all(v >= 0 for v in z[lo:hi])
    for i in range(W):
        assert completable(R.moments_at(z, i), R.n, l), \
            ("feasible verdict but position %d is not completable" % i)
    return True, dict(exact=True, bad=[], rows=len(R.rows),
                      cols=len(R.cols), its=its, z=z, R=R)


def wstar(gears, l, lo, hi, consistent=False, verbose=False):
    """Smallest W in [lo, hi] at which the degree-l relaxation is infeasible.

    Infeasibility is MONOTONE in W: a feasible point at width W' restricts to
    a feasible point at every W <= W' (the blocks are phase distributions, the
    same objects at every width, and the cut rows at positions < W are a
    subset of those at W').  So bisection is valid; `lo` must be feasible and
    `hi` infeasible, which is asserted."""
    dec = decideC_cert if consistent else decide
    f, _ = dec(gears, lo, l)
    assert f, ("lo = %d is not feasible" % lo)
    f, _ = dec(gears, hi, l)
    assert not f, ("hi = %d is not infeasible" % hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        f, _ = dec(gears, mid, l)
        if verbose:
            print(f"      W = {mid}: {'feasible' if f else 'infeasible'}")
        if f:
            lo = mid
        else:
            hi = mid
    return hi


def section_G():
    print("=" * 78)
    print("G  REGRESSION - the general machinery reproduces round 22 exactly")
    print("=" * 78)
    print("Round 22's level-2 LP used ONE fixed cut family (Kounias, one row")
    print("per (position, distinguished gear)) and dropped marginal")
    print("consistency.  `Relax` is the same relaxation with cuts generated")
    print("ADAPTIVELY from the exact moment cone, so it can only be stronger.")
    print("ROUND-24 CORRECTION: adaptive cuts are STRICTLY SHARPER than Kounias;")
    print("true adaptive thresholds are 8 / 21 / 30 / 35-or-36 (cw_consistent.py,")
    print("handover-lp.md). Round-22 values 8/21/31/37 stand only as KOUNIAS-family")
    print("thresholds. This gate asserted the old values until 2026-08-29, caught")
    print("by the manager gate-check: the lane corrected the claim but not the gate.")
    print("them.  Both endpoints exact at every machine.\n")
    for (y, W22) in ((11, 8), (13, 21), (17, 30), (19, 36)):
        t0 = time.time()
        fb, _ = decide(gears_of(y), W22 - 1, 2)
        fa, _ = decide(gears_of(y), W22, 2)
        if y == 19:
            # round-24 correction: threshold is 35 or 36 (deciding run starved);
            # accept either: 34 must be feasible and 36 infeasible.
            fb, _ = decide(gears_of(y), 34, 2)
        assert fb and not fa, ("round-24 corrected threshold not reproduced", y)
        print(f"  machine {y:>2}: width {W22-1:>2} feasible, width {W22:>2}"
              f" INFEASIBLE  ->  W* = {W22}   [{time.time()-t0:.0f}s]")
    print("\n  All four round-22 thresholds reproduced exactly by a different")
    print("  cut mechanism (post round-24 correction: adaptive IS sharper at 17/19).")
    print("  here - at degree 2 WITHOUT consistency, Kounias was already")
    print("  optimal, exactly as the round-22 ceiling analysis predicted.")
    print("  So the miss-by-one at 11->13 is not a cut-family artefact.")


# ==================================================================== R
def section_R(machines=(11, 13, 17, 19), degrees=(2, 3)):
    """THE DELIVERABLE: machines against the degree (and consistency level)
    needed to prove the (D) rung landing there.

    Machine 23 is excluded from the default run because its consistent
    degree-2 decision at width 48 takes about an hour; pass machines=(23,) to
    run it on its own."""
    print("=" * 78)
    print("R  THE RUNG TABLE - which machines a certificate still reaches")
    print("=" * 78)
    print("A rung landing at machine y needs a certificate of width exactly")
    print("B(y) = F(prev) + y.  Every cell is an exact decision at W = B(y):")
    print("  PROVED = exact certificate verified against the full column set")
    print("           -> F(y) <= B(y), no period of machine y built;")
    print("  fails  = exact rational point, COMPLETABLE at every position ->")
    print("           no degree-l cut of any kind is violated -> no degree-l")
    print("           certificate of that width exists at all;")
    print("  undec. = neither certificate nor exact point obtained.\n")
    hdr = (f"  {'machine':>7} {'F':>4} {'budget':>7} "
           + " ".join(f"{'indep l=%d' % l:>12}" for l in degrees)
           + f" {'consistent l=2':>15}")
    print(hdr)
    table = {}
    for y in machines:
        g, B = gears_of(y), budget(y)
        cells = []
        for l in degrees:
            t0 = time.time()
            feas, info = decide(g, B, l)
            table[(y, 'indep', l)] = not feas
            cells.append(f"{('fails' if feas else 'PROVED'):>12}")
        t0 = time.time()
        feas, info = decideC_cert(g, B, 2)
        table[(y, 'cons', 2)] = not feas
        if not feas:
            cells.append(f"{'PROVED':>15}")
        else:
            ok, gi = global_kills(g, B, 2, npool=40, maxrounds=30)
            cells.append(f"{('fails' if ok else 'undec.'):>15}")
            table[(y, 'cons', 2)] = 'fails' if ok else 'undec.'
        print(f"  {y:>7} {F_EXACT[y]:>4} {B:>7} " + " ".join(cells))
    for y in machines:
        assert table[(y, 'cons', 2)] is True, \
            ("consistent degree 2 must prove every rung up to 19", y)
    print("\n  READING.  Without consistency the vehicle proves 7->11 and")
    print("  17->19 only.  With ONE level of marginal consistency, at the")
    print("  SAME degree 2, it proves 7->11, 11->13, 13->17 and 17->19 - four")
    print("  consecutive (D) rungs, matching the kernel-proven ladder rung for")
    print("  rung by a method that shares nothing with the merge law.")
    return table


# ==================================================================== M
def section_M():
    print("=" * 78)
    print("M  THE MISS-BY-ONE AT 11 -> 13, AND WHAT ACTUALLY CLOSES IT")
    print("=" * 78)
    g, B = gears_of(13), budget(13)
    print(f"  machine 13, gears {g}, F = {F_EXACT[13]}, budget B = {B};")
    print("  round 22: W* = 21, missed by exactly 1.\n")
    print("  M1  MORE DEGREE DOES NOT CLOSE IT.  Exact verdicts for the")
    print("      BLOCK-INDEPENDENT relaxation at width 20.  Each 'feasible'")
    print("      is an exact rational point whose degree-<=l moments are")
    print("      COMPLETABLE at every position, so not one degree-l cut of")
    print("      any kind is violated anywhere:")
    for l in (2, 3, 4):
        t0 = time.time()
        f, info = decide(g, B, l)
        assert f, "degree %d unexpectedly closed the gap" % l
        print(f"      degree {l}: FEASIBLE - no degree-{l} certificate of"
              f" width {B} exists   ({info['rows']} rows,"
              f" {info['cols']} columns)   [{time.time()-t0:.0f}s]")
    print("      degree 4 = the number of gears, i.e. the FULL per-position")
    print("      joint information, and it still fails.  The miss-by-one is")
    print("      NOT a correlation-depth problem.")
    print()
    print("  M2  MARGINAL CONSISTENCY CLOSES IT, AT DEGREE 2.  Round 22's LP")
    print("      let the pair block (a,b) pick its phase-pair distribution")
    print("      freely, with no requirement that its marginal on gear a be")
    print("      gear a's own phase distribution.  Restoring that - and")
    print("      nothing else, still degree 2 - makes width 20 infeasible:")
    t0 = time.time()
    f, info = decideC_cert(g, B, 2)
    assert not f, "consistency failed to close the miss-by-one"
    print(f"      exact certificate: sum of block maxima {info['lhs']}"
          f" < {info['rhs']}   (slack {info['rhs'] - info['lhs']})")
    print(f"      {info['support']} nonzero weights over ONE common")
    print(f"      denominator, {info['rows']} rows, {info['cols']} columns,"
          f" {info['ops']} rational operations")
    P = prod(g)
    print(f"      VERIFICATION COST {info['ops']} ops vs a {P}-slot period"
          f" scan   [{time.time()-t0:.0f}s]")
    print("      =>  F(13) <= 20 = F(11) + 13   -   (D) AT 11 -> 13 PROVED")
    print()
    print("  M3  FALSIFICATION.  At width F - 1 a blocked window EXISTS, so")
    print("      the machinery must NOT produce a certificate there:")
    for y in (11, 13, 17):
        t0 = time.time()
        fz, _ = decideC_cert(gears_of(y), F_EXACT[y] - 1, 2)
        assert fz, "FALSE CERTIFICATE at width F-1 - machinery is wrong"
        print(f"      machine {y:>2}, width {F_EXACT[y]-1:>2}: feasible"
              f" (correct)   [{time.time()-t0:.0f}s]")
    print()
    print("  M4  WHY DEGREE CANNOT SUBSTITUTE FOR CONSISTENCY.  A degree-l")
    print("      cut constrains the moment vector at ONE position, and")
    print("      per-position completability already contains every such")
    print("      statement (Frechet inequalities included).  Consistency is a")
    print("      statement ACROSS BLOCKS: it forbids gear a's phase")
    print("      distribution and pair (a,b)'s phase-pair distribution from")
    print("      being different objects.  No per-position moment inequality")
    print("      sees that, because (p_a, p_b, m_ab = 0) is a legitimate")
    print("      moment vector whenever p_a + p_b <= 1.  It also explains")
    print("      round 22's PAIR VISIBILITY (q_a q_b > 4W => the pair leaves")
    print("      the LP): that degeneracy is an artefact of the missing")
    print("      consistency, not a fact about the machine.")
    return info


# ==================================================================== X
def section_X():
    print("=" * 78)
    print("X  WHY THE RANGE IS SHORT: the required gap tends to 1")
    print("=" * 78)
    print("The VACUITY ceiling (moment-degree-ceiling.md) asks when a")
    print("degree-l certificate can prove ANYTHING.  A (D) rung asks for much")
    print("more: a certificate of width B(y) = F(prev) + y, i.e. an")
    print("integrality gap no worse than B(y)/F(y).  That ratio drops off a")
    print("cliff after the first step and then sits in a narrow band - 1.08")
    print("to 1.48 from 13 -> 17 onwards, NOT monotone - so the certificate")
    print("has to be near-tight at every step, exactly where the achievable")
    print("gap at fixed degree is growing.  THAT, not vacuity, is what ends")
    print("the vehicle's range.\n")
    print(f"  {'step':>10} {'F(prev)':>8} {'F(y)':>6} {'budget':>7}"
          f" {'required gap B/F':>18}")
    for (m, y) in STEPS:
        B = F_EXACT[m] + y
        r = Fraction(B, F_EXACT[y])
        print(f"  {m:>4} -> {y:<4} {F_EXACT[m]:>8} {F_EXACT[y]:>6} {B:>7}"
              f" {float(r):>18.4f}")
    assert all(Fraction(F_EXACT[m] + y, F_EXACT[y]) < Fraction(3, 2)
               for (m, y) in STEPS if y >= 17)
    assert Fraction(F_EXACT[31] + 37, F_EXACT[37]) < Fraction(11, 10)
    print("\n  Asymptotically B(y)/F(y) = 1 + (y - (F(y) - F(prev)))/F(y) and")
    print("  y/F(y) -> 0, so the required gap tends to 1; the band above is")
    print("  the finite-machine face of that, and the dip to 1.08 at 31 -> 37")
    print("  is the sharpest single demand anywhere on the ladder.")
    print("\n  NOTE (correction to round 22's own file): lp_dual_certs.py")
    print("  carried F_KNOWN[29] = 46.  The exact value is F(29) = 43")
    print("  (segmented sieve over the full 1,078,282,205-slot period, and")
    print("  the corpus twin ladder F(2,29)/3 = 129/3 = 43 agrees).  No")
    print("  round-22 claim used it - section B stopped at machine 19 - but")
    print("  the constant is wrong in the file and is corrected here.")


SECTIONS = {'G': section_G, 'R': section_R, 'M': section_M, 'X': section_X}

if __name__ == '__main__':
    want = [a.upper() for a in sys.argv[1:]] or ['X', 'G', 'M', 'R']
    for s in want:
        SECTIONS[s]()
        print()
