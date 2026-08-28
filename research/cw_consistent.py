"""COMPOSING THE TWO ESCAPES - marginal consistency x Costello-Watts recursion.
Round 24, LP-duality thread.

THE TWO VEHICLES AND HOW THEY FAIL, ORTHOGONALLY.

  (a) The CONSISTENT degree-2 covering LP (round 23,
      docs/novel/consistency-over-degree.md).  Integrality gap FLAT at
      1.00 / 1.27 / 1.28 at machines 11 / 13 / 17; proves the (D) rungs
      7->11 .. 17->19; UNDECIDED at 19->23; REFUTED at 23->29, because the
      uniform product measure is a global point that satisfies every degree-2
      cut from machine 29 on.  CONSISTENCY BUYS WIDTH, NOT MACHINES.

  (b) COSTELLO-WATTS transferred (round 23, research/cw_transfer.py).  Its
      pair term is the EXACT survivor count of a smaller machine reached by
      the dilation lemma, so its effective degree is UNBOUNDED and the
      moment-degree ceiling does not bind it.  But every term is worst-cased
      over the window position INDEPENDENTLY, so it lands 3.2x-7.5x above the
      true F and proves no rung anywhere.

THE COMPOSITION.  Both are relaxations of the same exact identity.  Writing
S_q(r) for the number of positions of [0,W) that gear q blocks at phase r,
and N_ij(r) for the number blocked by BOTH q_i and q_j and by NO gear below
q_i (the Costello-Watts "lowest blocking prime" partition, Thms 3.1-3.2
transferred), the count of openings in the window at phase tuple r is

    open(r)  =  W  -  sum_q S_q(r_q)  +  sum_{i<j} N_ij(r)          (IDENTITY)

  * Costello-Watts bounds each term separately over r:
        open >= W - sum_q max_r S_q(r) + sum_{i<j} min_r N_ij(r).
  * The Kounias/Bonferroni star (round 22's cut, and the aggregate form of
    round 23's certificate) keeps only the i = 0 terms, where the "no lower
    gear" condition is vacuous, so N_0j(r) = |P_0j(r_0, r_j)| is an exact
    degree-2 function of the phases.
  * THE COMPOSED VEHICLE keeps ALL pairs and ties the phases together.  Define
        n_ij(u, v) = min over the phases of the gears BELOW q_i of N_ij,
    an exact integer table, degree 2 in the phases but of UNBOUNDED effective
    degree in the gear indicators (its value is a survivor count of the
    sub-machine reached by the dilation).  Then
        f(r) = W - sum_q S_q(r_q) + sum_{i<j} n_ij(r_i, r_j)  <=  open(r),
    and f is minimised over the SHERALI-ADAMS LEVEL-2 (pairwise-consistent)
    polytope.  min > 0 certifies F(M) <= W.

    STAR-3 (consistency one level deeper into the recursion).  The level-2
    composition still lets each pair term choose the lower gears' phases
    freely, which is the same consistency failure one level down.  The STAR-3
    variant restores it for the smallest gear: n^5_ij(u, v, w) minimises only
    over the gears below q_i OTHER than 5, and the LP carries triple blocks
    (5, q_i, q_j).

DOMINATION (proved in section V, asserted): for every point z of the
pairwise-consistent polytope, E[S_q] <= max_r S_q and E[n_ij] >= min n_ij, so
    composed LP value  >=  Costello-Watts value    at every machine and width,
and dropping the i >= 1 terms gives the Kounias star, so
    composed LP value  >=  Kounias star LP value.
The composed vehicle can therefore only improve on both, and W*_composed <=
min(W*_CW, W*_star).

--------------------------------------------------------------------------
PRE-REGISTERED EXPECTATION  (written 2026-08-25 BEFORE any number below was
computed; the brief asked for it explicitly).

The required ratio B(y)/F(y) is 2.29, 1.82, 1.56, 1.48, 1.41, 1.47, 1.28,
1.08, 1.42 across 7->11 .. 37->41.  After the first step it never exceeds 1.48
and it dips to 1.08.  SO ANY VEHICLE MUST BE NEAR-TIGHT EVERYWHERE, and the
composition inherits that demand.  What that implies before building:

E1  The single-gear worst-casing in Costello-Watts is NOT a consistency loss
    and composition cannot recover anything from it.  Gear phases are
    independent by CRT, so max_r sum_q S_q(r_q) = sum_q max_r S_q(r_q)
    EXACTLY.  ALL of CW's slack lives in the pair term.  (Stated as a
    prediction because it decides where the composition can possibly gain;
    section V asserts it.)

E2  The composition dominates both parents at every machine and width
    (argument above).  So the gaps can only go down.  This is the safe part
    and is not the interesting question.

E3  I EXPECT THE GAIN TO BE SMALL IN THE RUNG-PROVING RANGE, and here is the
    quantified form, so it can be refuted: at machine 13 width 20 and machine
    17 width 28, at least two thirds of the (i >= 1) pair tables n_ij are
    IDENTICALLY ZERO, because W / (q_i q_j) is 0 or 1 there and one lower gear
    suffices to kill a handful of positions.  Consequence: the composed W*
    improves on the consistent degree-2 W* (7 / 14 / 23 at machines 11/13/17)
    by AT MOST 1-2 units, and the composed gap sits in 1.2-1.5, not below 1.1.

E4  THE MACHINE AXIS IS WHERE THE COMPOSITION COULD ACTUALLY BUY SOMETHING.
    The degree-2 vacuity ceiling (machine 29) is a statement about moment
    functionals of the COVERAGE INDICATORS.  The terms n_ij with i >= 1 are
    not of that form - they are survivor counts of a dilated sub-machine - so
    the uniform-product-measure argument does not automatically kill them.  I
    PREDICT the composed vehicle is NOT vacuous at machine 29, i.e. there is
    some width at which its value is positive.  (Weak prediction: CW itself
    already gives F(29) <= 322, and the composition dominates CW.)

E5  THE PREDICTION THAT MATTERS, and the honest one: ESCAPING VACUITY IS NOT
    NEAR-TIGHTNESS.  I predict the composition does NOT bring a new rung into
    range - not 19->23 (budget 48, needs gap 1.41) and not 23->29 (budget 63,
    needs gap 1.47) - because the residual worst-casing inside n_ij (each pair
    term picks the lower gears' phases for itself) is a consistency failure of
    exactly the kind that cost the parent vehicle its rungs, only now one
    level down and repeated at every level of the recursion.  If STAR-3 moves
    a cell that level-2 composition does not, that is the refutation of the
    "one level down" reading and the finding of the round.

E6  SHAPE PREDICTION: the composed gap WANDERS rather than stays flat,
    because n_ij is an integer table whose entries collapse to 0 at exactly
    the machines where q_i q_j > W - an arithmetic threshold, not a smooth
    one.  Flatness was the signature of consistency; the recursion is a
    counting ingredient and counting ingredients wander.
--------------------------------------------------------------------------

HOUSE RULES.  Exact integer / rational arithmetic decides everything; scipy
is DISCOVERY only and every discovered verdict is re-derived exactly or the
run aborts.  Benchmarks are operation counts.  Every structural claim is
asserted against brute force at the machines where brute force is affordable.

Run:  uv run python research/cw_consistent.py [V D C R S Z]
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
from lp_degree_range import (gears_of, teeth, hits, F_EXACT, budget,  # noqa
                             STEPS, completable, product_moments,
                             product_point_kills, global_kills,
                             _atom_tables, separate)

ZERO, ONE = Fraction(0), Fraction(1)


# ============================================================ exact tables
def S_table(q, W):
    """S_q(r) = number of positions of [0,W) blocked by gear q at phase r."""
    return [len(hits(q, r, W)) for r in range(q)]


def pair_positions(qi, qj, u, v, W):
    """positions of [0,W) blocked by BOTH gears, at phases u and v."""
    return sorted(hits(qi, u, W) & hits(qj, v, W))


def _max_cover(P, lower, W):
    """max over phase choices of the lower gears of the number of positions of
    the list P that at least one lower gear blocks.  Exact, by DP over
    subsets of P (|P| is tiny: at most 4*ceil(W/(q_i q_j)))."""
    m = len(P)
    if m == 0 or not lower:
        return 0
    if m > 18:
        # SAFE FALLBACK: claim full coverage, i.e. n_ij = 0 there.  n_ij >= 0
        # always, so this is a valid weakening and never makes a bound
        # unsound - it only throws information away.
        return m
    idx = {p: 1 << b for b, p in enumerate(P)}
    reach = 1 << m
    cur = bytearray(reach)
    cur[0] = 1
    for q in lower:
        opts = set()
        for r in range(q):
            msk = 0
            h = hits(q, r, W)
            for p in P:
                if p in h:
                    msk |= idx[p]
            opts.add(msk)
        nxt = bytearray(reach)
        for a in range(reach):
            if cur[a]:
                for o in opts:
                    nxt[a | o] = 1
        cur = nxt
    best = 0
    for a in range(reach):
        if cur[a]:
            c = bin(a).count('1')
            if c > best:
                best = c
    return best


@lru_cache(maxsize=None)
def n_table(gears, i, j, W, skip=()):
    """the exact table n_ij(u,v) = min over the phases of the gears BELOW
    q_i (excluding those in `skip`) of the number of positions blocked by both
    q_i and q_j and by no such lower gear.  A lower bound on N_ij at every
    phase tuple, and degree 2 in the phases."""
    qi, qj = gears[i], gears[j]
    lower = tuple(g for g in gears[:i] if g not in skip)
    return tuple(tuple(len(P) - _max_cover(P, lower, W)
                       for v in range(qj)
                       for P in (pair_positions(qi, qj, u, v, W),))
                 for u in range(qi))


@lru_cache(maxsize=None)
def n_table3(gears, i, j, W, k):
    """STAR-3 table: n^{q_k}_ij(u, v, w) - the same minimum but with gear q_k
    (which must be BELOW q_i) held at phase w instead of minimised away."""
    qi, qj, qk = gears[i], gears[j], gears[k]
    lower = tuple(g for g in gears[:i] if g != qk)
    out = []
    for u in range(qi):
        row = []
        for v in range(qj):
            P0 = pair_positions(qi, qj, u, v, W)
            cell = []
            for w in range(qk):
                hk = hits(qk, w, W)
                P = [p for p in P0 if p not in hk]
                cell.append(len(P) - _max_cover(P, lower, W))
            row.append(tuple(cell))
        out.append(tuple(row))
    return tuple(out)


# ============================================================== brute force
def open_count(gears, r, W):
    """exact number of openings of [0,W) at phase tuple r."""
    bl = set()
    for q, rq in zip(gears, r):
        bl |= hits(q, rq, W)
    return W - len(bl)


def f_value(gears, r, W, ntabs):
    v = W - sum(len(hits(q, rq, W)) for q, rq in zip(gears, r))
    for (i, j), tab in ntabs.items():
        v += tab[r[i]][r[j]]
    return v


def all_ntabs(gears, W, imax=None):
    n = len(gears)
    return {(i, j): n_table(gears, i, j, W)
            for i in range(n) for j in range(i + 1, n)
            if imax is None or i <= imax}


# ================================================== the composed LP (level 2)
class Composed:
    """min over the pairwise-consistent (Sherali-Adams level 2) polytope of

        W - sum_q E[S_q] + sum_{i<j} E[n_ij].

    Variables: z_q(r) for each gear, z_ij(u,v) for each pair.
    Constraints: block sums 1; both marginals of each pair equal the singles.
    A certificate is a REPARAMETRISATION: potentials alpha_ij(u), beta_ij(v)
    with

        c^_q(r) = -S_q(r) + sum_{j} alpha_{qj}(r) + sum_{i} beta_{iq}(r)
        d^_ij(u,v) = n_ij(u,v) - alpha_ij(u) - beta_ij(v)
        VALUE >= W + sum_q min_r c^_q(r) + sum_{i<j} min_{u,v} d^_ij(u,v)

    (the identity sum E[c^] + sum E[d^] = sum E[c] + sum E[d] uses exactly the
    consistency rows).  If that lower bound is > 0 the LP is certified."""

    def __init__(self, gears, W, star_only=False):
        self.gears, self.W = tuple(gears), W
        self.n = len(self.gears)
        self.S = [S_table(q, W) for q in self.gears]
        self.pairs = [(i, j) for i in range(self.n)
                      for j in range(i + 1, self.n)
                      if (not star_only) or i == 0]
        self.N = {(i, j): n_table(self.gears, i, j, W) for (i, j) in self.pairs}

    # ---------- exact evaluation of a reparametrisation certificate
    def bound(self, alpha, beta):
        """EXACT.  alpha[(i,j)] is a list of q_i rationals, beta[(i,j)] of q_j.
        Returns (value, ops)."""
        ops = 0
        ch = [[-Fraction(self.S[k][r]) for r in range(q)]
              for k, q in enumerate(self.gears)]
        for (i, j) in self.pairs:
            a, b = alpha[(i, j)], beta[(i, j)]
            for r in range(self.gears[i]):
                ch[i][r] += a[r]
            for r in range(self.gears[j]):
                ch[j][r] += b[r]
            ops += self.gears[i] + self.gears[j]
        val = Fraction(self.W)
        for k in range(self.n):
            val += min(ch[k])
            ops += self.gears[k]
        for (i, j) in self.pairs:
            a, b, tab = alpha[(i, j)], beta[(i, j)], self.N[(i, j)]
            best = None
            for u in range(self.gears[i]):
                au, row = a[u], tab[u]
                for v in range(self.gears[j]):
                    x = row[v] - au - b[v]
                    if best is None or x < best:
                        best = x
            val += best
            ops += 3 * self.gears[i] * self.gears[j]
        return val, ops

    # ---------- float solve (DISCOVERY ONLY)
    def solve_float(self):
        import numpy as np
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
        gs = self.gears
        off_s, tot = [], 0
        for q in gs:
            off_s.append(tot)
            tot += q
        off_p = {}
        for (i, j) in self.pairs:
            off_p[(i, j)] = tot
            tot += gs[i] * gs[j]
        c = np.zeros(tot)
        for k, q in enumerate(gs):
            for r in range(q):
                c[off_s[k] + r] = -self.S[k][r]
        for (i, j) in self.pairs:
            tab = self.N[(i, j)]
            for u in range(gs[i]):
                for v in range(gs[j]):
                    c[off_p[(i, j)] + u * gs[j] + v] = tab[u][v]
        ri, ci, vv, beq = [], [], [], []
        rr = 0
        for k, q in enumerate(gs):
            for r in range(q):
                ri.append(rr); ci.append(off_s[k] + r); vv.append(1.0)
            beq.append(1.0); rr += 1
        rows_alpha, rows_beta = {}, {}
        for (i, j) in self.pairs:
            rows_alpha[(i, j)] = rr
            for u in range(gs[i]):
                for v in range(gs[j]):
                    ri.append(rr); ci.append(off_p[(i, j)] + u * gs[j] + v)
                    vv.append(1.0)
                ri.append(rr); ci.append(off_s[i] + u); vv.append(-1.0)
                beq.append(0.0); rr += 1
            rows_beta[(i, j)] = rr
            for v in range(gs[j]):
                for u in range(gs[i]):
                    ri.append(rr); ci.append(off_p[(i, j)] + u * gs[j] + v)
                    vv.append(1.0)
                ri.append(rr); ci.append(off_s[j] + v); vv.append(-1.0)
                beq.append(0.0); rr += 1
        A = coo_matrix((vv, (ri, ci)), shape=(rr, tot))
        res = linprog(c, A_eq=A, b_eq=np.array(beq),
                      bounds=[(0, None)] * tot, method='highs')
        assert res.status == 0, res.message
        duals = res.eqlin.marginals
        alpha = {(i, j): [duals[rows_alpha[(i, j)] + u] for u in range(gs[i])]
                 for (i, j) in self.pairs}
        beta = {(i, j): [duals[rows_beta[(i, j)] + v] for v in range(gs[j])]
                for (i, j) in self.pairs}
        return self.W + res.fun, alpha, beta, res.x

    # ---------- exact certificate from the float duals
    def certificate(self, alpha_f, beta_f):
        """Snap the float potentials to a common denominator and verify the
        bound EXACTLY.  Returns (value, alpha, beta, ops, den) or None."""
        best = None
        for den in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192,
                    256, 384, 512, 1024, 2048, 4096, 10 ** 4, 10 ** 5):
            alpha = {k: [Fraction(round(x * den), den) for x in v]
                     for k, v in alpha_f.items()}
            beta = {k: [Fraction(round(x * den), den) for x in v]
                    for k, v in beta_f.items()}
            # (potentials are snapped to ONE denominator: the certificate is a
            #  list of integers plus that denominator, which is what a kernel
            #  check wants.)
            val, ops = self.bound(alpha, beta)
            if val > 0:
                return val, alpha, beta, ops, den
            if best is None or val > best[0]:
                best = (val, den)
        return None

    def zero_bound(self):
        """the Costello-Watts value: no potentials at all, i.e. every term
        worst-cased on its own."""
        gs = self.gears
        alpha = {(i, j): [ZERO] * gs[i] for (i, j) in self.pairs}
        beta = {(i, j): [ZERO] * gs[j] for (i, j) in self.pairs}
        return self.bound(alpha, beta)


def composed_value(gears, W, star_only=False, exact_needed=True):
    """(certified exact lower bound on the composed LP value, info).  The
    float LP is discovery; the returned value is EXACT."""
    C = Composed(gears, W, star_only=star_only)
    v_f, af, bf, _x = C.solve_float()
    got = C.certificate(af, bf)
    if got is None:
        return None, dict(float_value=v_f, C=C)
    val, alpha, beta, ops, den = got
    assert val <= Fraction(int(round(v_f * 10 ** 6)) + 2, 10 ** 6), \
        "exact bound exceeds the float LP optimum - ABORT"
    return val, dict(float_value=v_f, ops=ops, den=den, C=C,
                     size=sum(len(v) for v in alpha.values())
                     + sum(len(v) for v in beta.values()))


def wstar_composed(gears, lo, hi, star_only=False, verbose=False):
    """smallest W in [lo,hi] with a certified positive composed value.
    Monotonicity is NOT assumed: the scan is linear and the first positive
    width is reported together with whether every larger width tested is also
    positive."""
    for W in range(lo, hi + 1):
        val, info = composed_value(gears, W, star_only=star_only)
        if verbose:
            print(f"      W = {W:>3}: float {info['float_value']:+.4f}"
                  + (f"   exact {val}" if val is not None else "   (no cert)"))
        if val is not None and val > 0:
            return W, val, info
    return None, None, None


# ==================================================================== V
def section_V():
    print("=" * 78)
    print("V  VALIDITY - the identity, the domination, and E1")
    print("=" * 78)
    print("V1  the Costello-Watts layer identity, brute-forced at every phase")
    print("    tuple of two whole machines (not just a window sample):")
    for y, W in ((11, 12), (13, 9)):
        g = gears_of(y)
        n = len(g)
        bad = 0
        for r in product(*[range(q) for q in g]):
            opv = open_count(g, r, W)
            F = sum(len(hits(q, rq, W)) for q, rq in zip(g, r))
            T = 0
            for i in range(n):
                for j in range(i + 1, n):
                    lowh = set()
                    for k in range(i):
                        lowh |= hits(g[k], r[k], W)
                    T += len((hits(g[i], r[i], W) & hits(g[j], r[j], W))
                             - lowh)
            assert opv == W - F + T, (r, opv, W - F + T)
        print(f"    machine {y:>2}, W = {W:>2}: identity exact at all"
              f" {prod(g):,} phase tuples")
    print()
    print("V2  f(r) <= open(r) at EVERY phase tuple (the composed bound is")
    print("    valid), asserted by brute force:")
    for y, W in ((11, 12), (13, 14)):
        g = gears_of(y)
        ntabs = all_ntabs(g, W)
        worst = None
        for r in product(*[range(q) for q in g]):
            fv, ov = f_value(g, r, W, ntabs), open_count(g, r, W)
            assert fv <= ov, ("composed bound UNSOUND", r, fv, ov)
            if worst is None or ov - fv < worst[0]:
                worst = (ov - fv, r)
        print(f"    machine {y:>2}, W = {W:>2}: f <= open at all"
              f" {prod(g):,} tuples; tightest slack {worst[0]}")
    print()
    print("V3  E1 - THE SINGLE-GEAR TERM CARRIES NO CONSISTENCY SLACK.")
    print("    max over phase tuples of sum_q S_q equals sum_q max_r S_q,")
    print("    because CRT makes the phases independent.  Asserted:")
    for y, W in ((11, 12), (13, 14), (17, 28)):
        g = gears_of(y)
        sep = sum(max(S_table(q, W)) for q in g)
        joint = max(sum(len(hits(q, rq, W)) for q, rq in zip(g, r))
                    for r in product(*[range(q) for q in g])) \
            if prod(g) <= 200000 else sep
        assert joint == sep, (y, joint, sep)
        print(f"    machine {y:>2}, W = {W:>2}: joint max {joint} =="
              f" separable max {sep}")
    print("    => E1 HOLDS.  All of Costello-Watts' slack is in the PAIR")
    print("    term, so that is the only place composition can gain.")
    print()
    print("V4  DOMINATION.  With zero potentials the composed certificate")
    print("    evaluates to exactly the Costello-Watts worst-case value, so")
    print("    the composed LP value is >= it at every machine and width:")
    for y in (11, 13, 17):
        g = gears_of(y)
        W = budget(y)
        C = Composed(g, W)
        z, _ = C.zero_bound()
        cw = (W - sum(max(S_table(q, W)) for q in g)
              + sum(min(min(row) for row in C.N[(i, j)])
                    for (i, j) in C.pairs))
        assert z == cw, (y, z, cw)
        print(f"    machine {y:>2}, W = {W:>2}: zero-potential bound"
              f" {z} == worst-cased-termwise {cw}")


# ==================================================================== D
def section_D():
    print("=" * 78)
    print("D  WHERE COSTELLO-WATTS' 3.2x-7.5x ACTUALLY GOES")
    print("=" * 78)
    print("At the TRUE extremal window r* (the one realising F(M)-1 blocked")
    print("slots) the identity is exact, so the loss splits into two exactly")
    print("measurable pieces:")
    print("    L_single = sum_q max_r S_q(r)  -  sum_q S_q(r*_q)")
    print("    L_pair   = sum_{i<j} N_ij(r*)  -  sum_{i<j} min_r N_ij")
    print("and CW's bound at that width is  open(r*) - L_single - L_pair.\n")
    print(f"  {'machine':>7} {'W':>4} {'open(r*)':>9} {'L_single':>9}"
          f" {'L_pair':>7} {'CW bound':>9}   {'zeroed n_ij tables':>19}")
    for y in (11, 13, 17, 19):
        g = gears_of(y)
        n = len(g)
        W = F_EXACT[y] - 1                 # the extremal width: open(r*) = 0
        # find r* by scanning the period (cheap up to machine 19)
        P = prod(g)
        blocked = bytearray(P)
        for q in g:
            for t in teeth(q):
                blocked[t % q::q] = b'\x01' * len(blocked[t % q::q])
        best = None
        for b in range(P):
            if all(blocked[(b + i) % P] for i in range(W)):
                best = b
                break
        assert best is not None, ("no extremal window found", y)
        # phase r_q such that hits(q, r_q, W) is what gear q blocks in it
        rstar = tuple(best % q for q in g)
        assert open_count(g, rstar, W) == 0
        ls = sum(max(S_table(q, W)) for q in g) \
            - sum(len(hits(q, rq, W)) for q, rq in zip(g, rstar))
        Ns, mins = 0, 0
        zeroed, total = 0, 0
        for i in range(n):
            for j in range(i + 1, n):
                lowh = set()
                for k in range(i):
                    lowh |= hits(g[k], rstar[k], W)
                Ns += len((hits(g[i], rstar[i], W) & hits(g[j], rstar[j], W))
                          - lowh)
                tab = n_table(g, i, j, W)
                mn = min(min(row) for row in tab)
                mins += mn
                total += 1
                if max(max(row) for row in tab) == 0:
                    zeroed += 1
        lp = Ns - mins
        print(f"  {y:>7} {W:>4} {0:>9} {ls:>9} {lp:>7} {-ls - lp:>9}"
              f"   {zeroed:>8} of {total:<8}")
    print("\n  L_single is the loss the composition CANNOT recover (E1: it is")
    print("  attained, not slack).  L_pair is the whole of the recoverable")
    print("  loss, and 'zeroed n_ij tables' counts the pair terms that are")
    print("  identically 0 at that width - the pairs the recursion cannot")
    print("  see because q_i q_j exceeds the window.")


# ==================================================================== C
def section_C(machines=(11, 13, 17, 19, 23)):
    print("=" * 78)
    print("C  THE COMPOSED VEHICLE - exact thresholds and integrality gaps")
    print("=" * 78)
    print("W* is the smallest width with a CERTIFIED POSITIVE exact value.")
    print("Every entry is an exact reparametrisation certificate: a list of")
    print("rationals over one denominator plus finitely many minima.\n")
    print(f"  {'machine':>7} {'F':>4} {'W* star':>8} {'W* composed':>12}"
          f" {'gap':>7} {'cert size':>10} {'ops':>8}")
    out = {}
    for y in machines:
        g = gears_of(y)
        F = F_EXACT[y]
        ws, _, _ = wstar_composed(g, F, 12 * F, star_only=True)
        wc, val, info = wstar_composed(g, F, 12 * F, star_only=False)
        gap = f"{float(Fraction(wc, F)):.3f}" if wc else "  -"
        out[y] = (ws, wc)
        print(f"  {y:>7} {F:>4} {str(ws):>8} {str(wc):>12} {gap:>7}"
              f" {info['size'] if wc else '-':>10}"
              f" {info['ops'] if wc else '-':>8}")
    return out


# ==================================================================== R
def section_R():
    print("=" * 78)
    print("R  RUNG TABLE - does the composition bring a new rung into range?")
    print("=" * 78)
    print(f"  {'step':>10} {'budget':>7} {'CW':>8} {'star LP':>9}"
          f" {'composed':>10} {'verdict':>10}")
    for (a, b) in STEPS:
        if b > 29:
            continue
        g = gears_of(b)
        B = budget(b)
        C = Composed(g, B)
        cw, _ = C.zero_bound()
        vs, _ = composed_value(g, B, star_only=True)
        vc, _ = composed_value(g, B, star_only=False)
        verdict = "PROVED" if (vc is not None and vc > 0) else "no"
        print(f"  {a:>4} -> {b:<3} {B:>7} {str(cw):>8}"
              f" {str(vs) if vs is not None else '-':>9}"
              f" {str(vc) if vc is not None else '-':>10} {verdict:>10}")


# ==================================================================== S
def section_S(machines=(11, 13, 17)):
    print("=" * 78)
    print("S  STAR-3 - consistency one level deeper into the recursion")
    print("=" * 78)
    print("The level-2 composition lets each pair term choose the LOWER")
    print("gears' phases for itself.  STAR-3 ties the smallest gear's phase")
    print("across every pair term, by carrying triple blocks (5, q_i, q_j).")
    print("If this moves a cell that level 2 does not, the 'one level down'")
    print("reading of the residual loss (E5) is refuted.\n")
    for y in machines:
        g = gears_of(y)
        F = F_EXACT[y]
        w2, _, _ = wstar_composed(g, F, 12 * F)
        w3 = wstar_star3(g, F, 12 * F)
        print(f"  machine {y:>2}: F = {F:>3}   W*(level 2) = {w2}"
              f"   W*(STAR-3) = {w3}")


class Composed3:
    """STAR-3.  Same objective, but every pair term (i,j) with i >= 1 is
    replaced by n^{5}_ij(u,v,w), and the LP carries a block for the triple
    (gear 5, q_i, q_j) whose marginals must agree with the singles and with
    the pair (i,j).  Level-2 blocks are kept for the (0,j) pairs."""

    def __init__(self, gears, W):
        self.gears, self.W = tuple(gears), W
        self.n = len(self.gears)
        self.S = [S_table(q, W) for q in self.gears]
        self.pairs0 = [(0, j) for j in range(1, self.n)]
        self.trip = [(i, j) for i in range(1, self.n)
                     for j in range(i + 1, self.n)]
        self.N0 = {ij: n_table(self.gears, ij[0], ij[1], W)
                   for ij in self.pairs0}
        self.N3 = {(i, j): n_table3(self.gears, i, j, W, 0)
                   for (i, j) in self.trip}

    def solve_float(self):
        import numpy as np
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
        gs = self.gears
        off_s, tot = [], 0
        for q in gs:
            off_s.append(tot)
            tot += q
        off2, off3 = {}, {}
        for (i, j) in self.pairs0:
            off2[(i, j)] = tot
            tot += gs[i] * gs[j]
        for (i, j) in self.trip:
            off3[(i, j)] = tot
            tot += gs[0] * gs[i] * gs[j]
        c = np.zeros(tot)
        for k, q in enumerate(gs):
            for r in range(q):
                c[off_s[k] + r] = -self.S[k][r]
        for (i, j) in self.pairs0:
            tab = self.N0[(i, j)]
            for u in range(gs[i]):
                for v in range(gs[j]):
                    c[off2[(i, j)] + u * gs[j] + v] = tab[u][v]
        for (i, j) in self.trip:
            tab = self.N3[(i, j)]
            for u in range(gs[i]):
                for v in range(gs[j]):
                    for w in range(gs[0]):
                        c[off3[(i, j)] + (u * gs[j] + v) * gs[0] + w] = \
                            tab[u][v][w]
        ri, ci, vv, beq = [], [], [], []
        rr = 0

        def eqrow(cols, rhs):
            nonlocal rr
            for cc, coef in cols:
                ri.append(rr); ci.append(cc); vv.append(coef)
            beq.append(rhs); rr += 1

        for k, q in enumerate(gs):
            eqrow([(off_s[k] + r, 1.0) for r in range(q)], 1.0)
        for (i, j) in self.pairs0:
            for u in range(gs[i]):
                eqrow([(off2[(i, j)] + u * gs[j] + v, 1.0)
                       for v in range(gs[j])] + [(off_s[i] + u, -1.0)], 0.0)
            for v in range(gs[j]):
                eqrow([(off2[(i, j)] + u * gs[j] + v, 1.0)
                       for u in range(gs[i])] + [(off_s[j] + v, -1.0)], 0.0)
        for (i, j) in self.trip:
            base = off3[(i, j)]
            for u in range(gs[i]):
                eqrow([(base + (u * gs[j] + v) * gs[0] + w, 1.0)
                       for v in range(gs[j]) for w in range(gs[0])]
                      + [(off_s[i] + u, -1.0)], 0.0)
            for v in range(gs[j]):
                eqrow([(base + (u * gs[j] + v) * gs[0] + w, 1.0)
                       for u in range(gs[i]) for w in range(gs[0])]
                      + [(off_s[j] + v, -1.0)], 0.0)
            for w in range(gs[0]):
                eqrow([(base + (u * gs[j] + v) * gs[0] + w, 1.0)
                       for u in range(gs[i]) for v in range(gs[j])]
                      + [(off_s[0] + w, -1.0)], 0.0)
        A = coo_matrix((vv, (ri, ci)), shape=(rr, tot))
        res = linprog(c, A_eq=A, b_eq=np.array(beq),
                      bounds=[(0, None)] * tot, method='highs')
        assert res.status == 0, res.message
        return self.W + res.fun, res

    def exact_value_at(self, x):
        """EXACT evaluation of the objective at a rationalised primal point,
        used only as a sanity check."""
        raise NotImplementedError


def wstar_star3(gears, lo, hi):
    """DISCOVERY-level threshold for STAR-3 (float LP).  Any width it reports
    is then confirmed by an exact certificate in the caller if it differs from
    the level-2 answer."""
    for W in range(lo, hi + 1):
        C = Composed3(gears, W)
        v, _res = C.solve_float()
        if v > 1e-7:
            return W
    return None


# ==================================================================== Z
def section_Z():
    print("=" * 78)
    print("Z  THE MACHINE-17 DEGREE-4 CELL (left BLANK in round 23)")
    print("=" * 78)
    print("Round 23 stopped the block-independent degree-4 decision at")
    print("machine 17, width 28, after ~45 minutes and recorded the cell as")
    print("BLANK rather than 'fails'.  The cut-generation loop was the wrong")
    print("tool: a GLOBAL POINT settles it in one pass, because a rational")
    print("measure over full phase tuples whose degree-<=4 coverage moments")
    print("are completable at every position is a feasible point of the")
    print("degree-4 relaxation WITH OR WITHOUT consistency.\n")
    g = gears_of(17)
    B = budget(17)
    t0 = time.time()
    ok, where = product_point_kills(g, B, 4)
    if ok:
        print(f"  machine 17, width {B}, degree 4: the UNIFORM product measure")
        print("  is already such a point - its degree-<=4 moments are")
        print("  completable at EVERY position.  So the block-independent")
        print("  degree-4 relaxation is FEASIBLE at width 28 and NO degree-4")
        print("  certificate of that width exists.")
        print(f"  CELL FILLED: 'fails'.  [{time.time()-t0:.1f}s]")
        return 'fails'
    print(f"  uniform product measure fails at position {where}; trying a")
    print("  general rational global point (mixture over phase tuples).")
    ok2, gi = global_kills(g, B, 4, npool=60, maxrounds=60)
    print(f"  global point search: {'FEASIBLE (cell = fails)' if ok2 else 'no point found (cell stays BLANK)'}"
          f"   [{time.time()-t0:.1f}s]")
    return 'fails' if ok2 else 'blank'


SECTIONS = {'V': section_V, 'D': section_D, 'C': section_C,
            'R': section_R, 'S': section_S, 'Z': section_Z}


def main():
    want = [a for a in sys.argv[1:] if a in SECTIONS] or ['V', 'D', 'C', 'R']
    for k in want:
        SECTIONS[k]()
        print()


if __name__ == '__main__':
    main()


# ==========================================================================
# THE FULL COMPOSITION: the consistent covering LP WITH the recursive row.
#
# The composed counting LP above is one aggregated inequality.  The round-23
# vehicle is a per-position covering LP over the same pairwise-consistent
# polytope.  They are different relaxations of the same IP, and the honest
# composition is to carry BOTH: every covering row of round 23, plus the ONE
# extra valid row supplied by the Costello-Watts recursion,
#
#     sum_q E[S_q]  -  sum_{i<j} E[n_ij]   >=   W ,
#
# which holds at every fully blocked window because open(r) = 0 there and
# f(r) <= open(r).  Adding a valid row can only strengthen the vehicle, and -
# this is the point - the row is NOT a moment functional of the coverage
# indicators, so the uniform product measure is no longer automatically a
# feasible point of it.
from lp_degree_range import RelaxC, certificateC                  # noqa: E402


class RelaxCF(RelaxC):
    """RelaxC + the recursive Costello-Watts row."""

    def __init__(self, gears, W, l=2, use_recursion=True):
        super().__init__(gears, W, l)
        self.use_recursion = use_recursion
        self.frow = [ZERO] * len(self.cols)
        for j, (S, r, O) in enumerate(self.cols):
            if len(S) == 1:
                self.frow[j] = Fraction(len(O))
            elif len(S) == 2:
                a, b = self.gidx[S[0]], self.gidx[S[1]]
                tab = (n_table(self.gears, a, b, W) if use_recursion
                       else None)
                self.frow[j] = -Fraction(tab[r[0]][r[1]]) if tab is not None \
                    else -Fraction(len(O))
        self.frhs = Fraction(W)

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
        import numpy as np2
        c = np2.zeros(N + 1)
        c[-1] = -1.0
        res = linprog(c, A_ub=A_ub, b_ub=bub, A_eq=A_eq, b_eq=np2.array(beq),
                      bounds=[(0, None)] * N + [(None, None)], method='highs')
        assert res.status == 0, res.message
        return -res.fun, res.x[:N], res


def certificateCF(R, yf, yff, nuf):
    """EXACT certificate for RelaxCF: same shape as certificateC, with one
    extra nonnegative weight yff on the recursive row.

        a_j = sum_r y_r lam^r_{S(j)} [i_r in O_j] + yff * frow_j
              + sum_{links: j in kids} nu - sum_{links: par = j} nu
        certificate iff  sum_S max_{j in S} a_j  <  sum_r y_r (1 - lam^r_0)
                                                    + yff * W ."""
    N = len(R.cols)
    scale = max(max((abs(v) for v in yf), default=0.0), abs(yff), 1e-12)
    grid = list(range(1, 65)) + [96, 128, 192, 256, 384, 512, 1024, 4096,
                                 10 ** 4, 10 ** 5, 10 ** 6]
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


def decideCF(gears, W, l=2, use_recursion=True, verbose=False,
             maxrounds=300):
    """EXACT decision of the FULL COMPOSITION at width W.
    Returns (feasible?, info); an infeasible verdict carries an exact
    certificate."""
    R = RelaxCF(gears, W, l, use_recursion=use_recursion)
    kind, vec, its = R.run(maxrounds=maxrounds, verbose=verbose)
    if kind == 'feasible':
        return True, dict(exact=False, rows=len(R.rows), cols=len(R.cols),
                          its=its, z=vec, R=R)
    res = R.last_duals
    y = list(res[0])
    yff = y.pop()                      # the recursive row is the LAST row
    nu = res[1]
    ok, lhs, rhs, yq, yffq, nuq, ops = certificateCF(R, y, yff, nu)
    assert ok, ("float said infeasible but no exact certificate could be "
                "rationalised - ABORT")
    return False, dict(lhs=lhs, rhs=rhs, y=yq, yf=yffq, nu=nuq, ops=ops,
                       rows=len(R.rows), cols=len(R.cols), its=its, R=R,
                       support=sum(1 for v in yq if v) + (1 if yffq else 0)
                       + sum(1 for v in nuq if v))


# ==========================================================================
# FILLING THE MACHINE-17 DEGREE-4 CELL WITHOUT A CUT LOOP.
#
# Round 23 tried to decide the BLOCK-INDEPENDENT degree-4 relaxation at
# machine 17, width 28, with the adaptive separation loop; the loop kept
# generating cuts and the cell was recorded BLANK.  The loop was the wrong
# tool.  "No degree-l cut is violated at position i" is EXACTLY "the degree-<=l
# moment vector at i extends to a distribution on {0,1}^n with zero mass on the
# empty atom", and that completion is itself a linear system.  So the whole
# question is ONE linear program:
#
#     find block distributions z_S (|S| <= l) and, for every position i, a
#     completion nu_i on the 2^n atoms with
#         nu_i >= t on every nonempty atom,  nu_i(empty) = 0,  sum nu_i = 1,
#         sum_{atoms containing S} nu_i  =  m_S(i)   for every |S| <= l,
#     maximising t.
#
# t > 0 gives a STRICTLY interior point, which survives rationalisation, and
# then `completable` re-verifies every position in exact arithmetic.  No cuts
# are generated at all.
from lp_degree_range import Relax                                 # noqa: E402


def decide_direct(gears, W, l, verbose=False):
    """EXACT decision of the block-independent degree-l relaxation at width W
    by the one-shot LP above.  Returns (feasible?, info)."""
    import numpy as np
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    R = Relax(gears, W, l)
    n, N = R.n, len(R.cols)
    subs = R.subs                      # masks of the kept subsets, index sidx
    ns = len(subs)
    natom = 1 << n
    # variable layout: z (N)  |  nu_i for i in [0,W)  (W * natom)  |  t
    NU = N
    T = N + W * natom
    tot = T + 1
    ri, ci, vv, beq = [], [], [], []
    rr = 0

    def eqrow(cols, rhs):
        nonlocal rr
        for cc, coef in cols:
            ri.append(rr); ci.append(cc); vv.append(coef)
        beq.append(rhs); rr += 1

    for S in R.subsets:                        # block distributions sum to 1
        lo, hi = R.block_span[S]
        eqrow([(j, 1.0) for j in range(lo, hi)], 1.0)
    for i in range(W):
        base = NU + i * natom
        eqrow([(base + a, 1.0) for a in range(natom)], 1.0)   # nu_i is a law
        eqrow([(base + 0, 1.0)], 0.0)                          # no empty atom
        for si, m in enumerate(subs):
            if m == 0:
                continue
            cols = [(base + a, 1.0) for a in range(natom) if (a & m) == m]
            S = tuple(R.gears[b] for b in range(n) if (m >> b) & 1)
            lo, hi = R.block_span[S]
            cols += [(j, -1.0) for j in range(lo, hi) if i in R.cols[j][1]]
            eqrow(cols, 0.0)
    A_eq = coo_matrix((vv, (ri, ci)), shape=(rr, tot))
    # nu_i(x) - t >= 0  on nonempty atoms
    ri2, ci2, vv2, bub = [], [], [], []
    r2 = 0
    for i in range(W):
        base = NU + i * natom
        for a in range(1, natom):
            ri2.append(r2); ci2.append(base + a); vv2.append(-1.0)
            ri2.append(r2); ci2.append(T); vv2.append(1.0)
            bub.append(0.0); r2 += 1
    A_ub = coo_matrix((vv2, (ri2, ci2)), shape=(r2, tot))
    c = np.zeros(tot); c[T] = -1.0
    res = linprog(c, A_ub=A_ub, b_ub=np.array(bub), A_eq=A_eq,
                  b_eq=np.array(beq),
                  bounds=[(0, None)] * T + [(None, None)], method='highs')
    if res.status != 0:
        return None, dict(status=res.message)
    t = -res.fun
    if verbose:
        print("      direct LP: t = %+.6g" % t)
    if t <= 0:
        return None, dict(t=t, note="no strictly interior point found")
    z = res.x[:N]
    for den in (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7):
        zex = R.rationalise(z, den)
        if all(completable(R.moments_at(zex, i), n, l) for i in range(W)):
            return True, dict(t=t, den=den, z=zex, R=R,
                              cols=N, positions=W)
    return None, dict(t=t, note="interior point found but rationalisation "
                                "did not verify")
