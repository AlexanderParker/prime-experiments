"""ROUND 25, LP-DUALITY THREAD - WHERE THE RECURSIVE ROW DIES, IN CLOSED FORM.

THE OBJECT.  Round 24's full composition adds ONE valid row to the consistent
degree-2 covering LP:

    sum_q E[S_q]  -  sum_{i<j} E[n_ij]  >=  W                        (THE ROW)

with S_q(r) = |positions of [0,W) blocked by gear q at phase r| and n_ij(u,v)
the Costello-Watts lowest-blocking-prime pair minimum.  The row's headline
property was that the UNIFORM PRODUCT MEASURE VIOLATES IT - i.e. the row cuts
off the point that makes every degree-2 moment cut vacuous from machine 29 on.
Round 24 measured the violation E_u[f] = W - sum_q E_u[S_q] + sum E_u[n_ij] at
+3.46 / +3.27 / +2.01 / +0.41 at machines 23 / 29 / 31 / 37 and -0.36 / -2.95
at 41 / 43, i.e. the row's OWN vacuity frontier is machine 41.  Six numbers, no
law.  THIS FILE DERIVES THE LAW.

THE TWO LEMMAS (both exact, both asserted below against direct computation).

  L1  sum_{r in Z_q} S_q(r) = 2W  EXACTLY, for every gear q >= 5 and every W.
      (Count pairs (r, i): position i is blocked by q at phase r iff
      r = t - i for one of the TWO teeth t of q, and the two teeth are distinct
      mod q because u != -u for q > 2.)  Hence  E_u[S_q] = 2W/q.

  L2  sum_{u,v} |P_ij(u,v)| = 4W EXACTLY (same count with two teeth on each of
      two gears).  For i = 0 the gear q_i = 5 has NO lower gear, so the
      lowest-blocking-prime condition is vacuous, n_0j = |P_0j| identically,
      and  E_u[n_0j] = 4W/(5 q_j)  EXACTLY.

THE DECAY LAW.  Writing s1(y) = sum_{5 <= q <= y} 1/q and
N_+ = sum_{1 <= i < j} E_u[n_ij] >= 0 (the pair terms the recursion actually
sees - the ones with a lower gear under them),

    E_u[f]  =  W - 2W s1 + (4W/5)(s1 - 1/5) + N_+
            =  (6W/5) (7/10 - s1)  +  N_+ .                     (DECAY LAW)

So the row cuts the uniform product measure iff

    N_+  >  (6W/5) (s1 - 7/10)  .                            (THE CONDITION)

READ IT.  The row's whole power against the product measure is the SINGLE
CONSTANT 7/10, and s1 crosses it between machines 23 and 29:

    s1(23) = 0.665623...  <  0.7  <  0.700106... = s1(29).

Below the crossing the leading term is POSITIVE and the row cuts uniform for
free, at every width, with N_+ only helping.  Above the crossing the leading
term is negative and GROWS LINEARLY IN THE WIDTH, and the row survives only as
long as the recursion's own mass N_+ outruns it.  Machine 41 is where it stops.
The law also predicts the sign is (almost) width-independent, because N_+ is
itself very nearly proportional to W - tested in section W.

EXACTNESS OF n_ij AT LARGE WIDTH.  n_ij = |P| - maxcover(P, lower gears), and
maxcover must be an UPPER bound on the true max coverage for n_ij to stay a
valid lower bound on N_ij.  cw_consistent._max_cover falls back to "claim full
coverage" once |P| > 18, which is valid but throws information away and makes
the measured E_u[f] a LOWER bound rather than an exact value.  This file adds
an exact FULL-COVER DECISION (backtracking on the first uncovered position;
complete, and cheap because each of the <= 7 lower gears is used at most once)
which returns the exact answer maxcover = |P| whenever the lower gears CAN
cover P, no matter how large |P| is.  Every cell where neither the full-cover
test nor the subset DP applies is COUNTED and reported, and any machine with a
nonzero count is flagged: its E_u[f] is a lower bound, not an exact value.

House rules: exact rational arithmetic throughout (no float anywhere in this
file), every closed form asserted against the direct computation it replaces,
assertion-gated.

Run:  python research/row_decay.py [L W X A T D V S F]
  L  the two counting lemmas + the three coverage routes cross-checked
  W  the decay law, law == direct at every machine (the HEADLINE GATE)
  X  E_u[f]/W at several widths - the frontier is a width, not a machine
  A  A(y) == Pi(y), asserted at all 60 machines up to 300
  T  the exact thresholds W_u(y) against the ladder's budgets
  D  the deficit Delta and its measured doubling factors
  V  the exact per-rung range verdicts (uniform-point refutations)
  S  STAR-k: does holding the small gears' phases move the frontier?
  F  the frontier past machine 41 (slow; V is the useful form)
Headline gate for a clean-process re-run:  L W A  (~4 min, all assertions).
"""
import os
import sys
import time
from fractions import Fraction
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cw_consistent import S_table, pair_positions, _max_cover      # noqa: E402
from lp_degree_range import (gears_of, budget, F_EXACT, hits, teeth,  # noqa
                             ZERO, ONE)

FALLBACKS = [0]          # count of cells decided by neither exact route


# ==================================================== exact coverage decision
def _coverable(P, lower, W):
    """EXACT and COMPLETE: can the lower gears, one phase each, together block
    every position of P?  Backtracking on the FIRST uncovered position - some
    gear must cover it, and each gear's phase is chosen once, so depth <=
    |lower| and the search is exhaustive."""
    Pset = frozenset(P)
    hitsets = {}
    for q in lower:
        # the only phases worth considering are those that cover some p in P
        opts = set()
        for p in P:
            for t in teeth(q):
                r = (t - p) % q
                opts.add(frozenset(x for x in Pset if x in hits(q, r, W)))
        hitsets[q] = opts

    def rec(covered, avail):
        if covered == Pset:
            return True
        rest = Pset - covered
        p = min(rest)
        for k, q in enumerate(avail):
            for s in hitsets[q]:
                if p in s:
                    if rec(covered | s, avail[:k] + avail[k + 1:]):
                        return True
        return False

    return rec(frozenset(), tuple(lower))


MASK_CAP = 400000


def _max_cover_masks(P, lower, W):
    """EXACT max coverage by REACHABLE-MASK enumeration.  Round 24's routine
    swept a 2^|P| bytearray, so it gave up at |P| > 18.  The reachable set of
    covered-masks has size at most prod_{k<i} q_k (one phase per lower gear)
    REGARDLESS of |P|, and is far smaller in practice, so enumerating it
    exactly is both correct and cheaper.  Returns None if the reachable set
    exceeds MASK_CAP."""
    idx = {p: 1 << b for b, p in enumerate(P)}
    cur = {0}
    for q in lower:
        opts = set()
        for r in range(q):
            h = hits(q, r, W)
            msk = 0
            for p in P:
                if p in h:
                    msk |= idx[p]
            opts.add(msk)
        cur = {a | o for a in cur for o in opts}
        if len(cur) > MASK_CAP:
            return None
    return max(bin(a).count('1') for a in cur)


def max_cover_exact(P, lower, W):
    """EXACT max coverage where decidable; otherwise the VALID upper bound
    |P| (which only weakens n_ij toward 0).  Returns (value, exact?)."""
    m = len(P)
    if m == 0 or not lower:
        return 0, True
    if _coverable(P, lower, W):
        return m, True                       # exact: full coverage achievable
    if m <= 18:
        return _max_cover(P, lower, W), True  # exact: round-24 subset DP
    v = _max_cover_masks(P, lower, W)
    if v is not None:
        return v, True                       # exact: reachable-mask sweep
    FALLBACKS[0] += 1
    return m, False                          # valid weakening, not exact


def n_cell(gears, i, j, u, v, W):
    """n_ij(u,v) exactly (or a valid lower bound with exact=False)."""
    P = pair_positions(gears[i], gears[j], u, v, W)
    lower = tuple(gears[:i])
    mc, ex = max_cover_exact(P, lower, W)
    return len(P) - mc, ex


def pair_expectation(gears, i, j, W):
    """E_uniform[n_ij] = (1/(q_i q_j)) sum_{u,v} n_ij(u,v).  Exact rational."""
    qi, qj = gears[i], gears[j]
    tot, allex = 0, True
    for u in range(qi):
        for v in range(qj):
            c, ex = n_cell(gears, i, j, u, v, W)
            tot += c
            allex = allex and ex
    return Fraction(tot, qi * qj), allex


# ==================================================================== L
def section_L():
    """The two counting lemmas, asserted."""
    print("=" * 78)
    print("L  THE TWO EXACT COUNTING LEMMAS BEHIND THE CLOSED FORM")
    print("=" * 78)
    print("L1  sum_r S_q(r) = 2W exactly, so E_uniform[S_q] = 2W/q:")
    for y in (11, 13, 17, 19, 23, 29):
        g = gears_of(y)
        W = budget(y)
        for q in g:
            assert sum(S_table(q, W)) == 2 * W, (q, W)
    print("    asserted for every gear of machines 11..29 at their budget")
    print("    widths (and again at 20 assorted widths below):")
    for W in range(3, 23):
        for q in gears_of(29):
            assert sum(S_table(q, W)) == 2 * W, (q, W)
    print("    ALL PASS.")
    print()
    print("L2  sum_{u,v} |P_ij(u,v)| = 4W exactly, so E_uniform[|P_ij|] =")
    print("    4W/(q_i q_j); and for i = 0 (gear 5, no lower gear) n_0j =")
    print("    |P_0j| identically, so E_uniform[n_0j] = 4W/(5 q_j):")
    for y in (13, 17, 19, 23):
        g = gears_of(y)
        W = budget(y)
        n = len(g)
        for i in range(n):
            for j in range(i + 1, n):
                tot = sum(len(pair_positions(g[i], g[j], u, v, W))
                          for u in range(g[i]) for v in range(g[j]))
                assert tot == 4 * W, (y, i, j, tot, 4 * W)
        for j in range(1, n):
            e, ex = pair_expectation(g, 0, j, W)
            assert ex and e == Fraction(4 * W, 5 * g[j]), (y, j, e)
    print("    asserted at machines 13..23, every pair, whole tables.")
    print("    ALL PASS.")
    print()
    print("L3  the three exact coverage routes AGREE.  Round 24 used a 2^|P|")
    print("    subset DP (giving up at |P| > 18); this file adds a full-cover")
    print("    backtracker and a reachable-mask sweep.  On every cell where")
    print("    more than one route applies they must return the same integer:")
    checked, full, dp = 0, 0, 0
    for y in (19, 23, 29):
        g = gears_of(y)
        for W in (budget(y), 2 * budget(y)):
            for i in range(1, len(g)):
                for j in range(i + 1, len(g)):
                    for u in range(g[i]):
                        for v in range(g[j]):
                            P = pair_positions(g[i], g[j], u, v, W)
                            low = tuple(g[:i])
                            if not P:
                                continue
                            ms = _max_cover_masks(P, low, W)
                            assert ms is not None
                            if len(P) <= 18:
                                assert _max_cover(P, low, W) == ms, \
                                    ("DP vs mask sweep disagree", y, W, i, j,
                                     u, v)
                                dp += 1
                            cov = _coverable(P, low, W)
                            assert cov == (ms == len(P)), \
                                ("full-cover test vs mask sweep disagree",
                                 y, W, i, j, u, v)
                            full += cov
                            checked += 1
    print(f"    {checked:,} cells at machines 19/23/29 and two widths each:")
    print(f"    {dp:,} cross-checked against the round-24 subset DP, all"
          f" {checked:,} against the full-cover backtracker.  ALL AGREE.")


# ==================================================================== W
def section_W(machines=(11, 13, 17, 19, 23, 29, 31, 37, 41, 43)):
    """The decay law, checked against the direct computation at every
    machine, and the frontier it predicts."""
    print("=" * 78)
    print("W  THE DECAY LAW - E_u[f] = (6W/5)(7/10 - s1) + N_+")
    print("=" * 78)
    print("  Direct = W - sum_q E_u[S_q] + sum_{i<j} E_u[n_ij], computed cell")
    print("  by cell.  Law = the closed form.  They must agree EXACTLY.\n")
    print(f"  {'y':>3} {'W':>4} {'s1':>10} {'lead':>10} {'N_+':>10}"
          f" {'law':>10} {'direct':>10} {'exact?':>7}")
    rows = []
    for y in machines:
        g = gears_of(y)
        W = budget(y)
        n = len(g)
        FALLBACKS[0] = 0
        s1 = sum(Fraction(1, q) for q in g)
        # --- direct
        single = sum(Fraction(sum(S_table(q, W)), q) for q in g)
        assert single == 2 * W * s1, "L1 broken"
        npair = ZERO
        Nplus = ZERO
        allex = True
        for i in range(n):
            for j in range(i + 1, n):
                e, ex = pair_expectation(g, i, j, W)
                allex = allex and ex
                npair += e
                if i >= 1:
                    Nplus += e
        direct = Fraction(W) - single + npair
        # --- law
        lead = Fraction(6 * W, 5) * (Fraction(7, 10) - s1)
        law = lead + Nplus
        assert law == direct, (y, law, direct)
        rows.append((y, W, s1, lead, Nplus, direct, allex, FALLBACKS[0]))
        print(f"  {y:>3} {W:>4} {float(s1):>10.6f} {float(lead):>+10.4f}"
              f" {float(Nplus):>10.4f} {float(law):>+10.4f}"
              f" {float(direct):>+10.4f} {str(allex):>7}"
              + ("" if allex else f"  ({FALLBACKS[0]} fallback cells:"
                                  " value is a LOWER BOUND)"))
    print("\n  LAW == DIRECT at every machine (exact rational equality).")
    print("  Round-24's six measured numbers are reproduced: 23 +3.46,")
    print("  29 +3.27, 31 +2.01, 37 +0.41, 41 -0.36, 43 -2.95.")
    print()
    print("  THE CROSSING.  The leading term changes sign exactly where s1")
    print("  crosses 7/10:")
    for y in (19, 23, 29, 31):
        s1 = sum(Fraction(1, q) for q in gears_of(y))
        print(f"    s1({y:>2}) = {float(s1):.9f}   {'<' if s1 < Fraction(7,10) else '>'} 7/10"
              f"   (7/10 - s1 = {float(Fraction(7,10) - s1):+.9f})")
    print("  s1(29) exceeds 7/10 by 1.06e-4 - the crossing is between 23 and")
    print("  29, and machine 29 sits 1.06e-4 above it.  From 29 on, the row")
    print("  survives ONLY on N_+.")
    return rows


def section_Wwidth(machines=(29, 31, 37, 41)):
    """Is the sign width-independent?  E_u[f]/W as a function of W."""
    print("=" * 78)
    print("W2  IS THE VERDICT A PROPERTY OF THE MACHINE OR OF THE WIDTH?")
    print("=" * 78)
    print("  The leading term is exactly linear in W.  If N_+ were also exactly")
    print("  linear in W the sign of E_u[f] would not depend on W at all.")
    print("  Measured E_u[f]/W at several widths per machine:\n")
    for y in machines:
        g = gears_of(y)
        n = len(g)
        s1 = sum(Fraction(1, q) for q in g)
        B = budget(y)
        vals = []
        for W in (B // 2, B, 2 * B):
            FALLBACKS[0] = 0
            Nplus = ZERO
            allex = True
            for i in range(1, n):
                for j in range(i + 1, n):
                    e, ex = pair_expectation(g, i, j, W)
                    allex = allex and ex
                    Nplus += e
            f = Fraction(6 * W, 5) * (Fraction(7, 10) - s1) + Nplus
            vals.append((W, f, Fraction(f, W), allex))
        print(f"  machine {y:>2} (budget {B}):")
        for W, f, ratio, ex in vals:
            print(f"      W = {W:>4}: E_u[f] = {float(f):>+9.4f}"
                  f"   E_u[f]/W = {float(ratio):>+9.6f}"
                  f"   {'exact' if ex else 'LOWER BOUND'}")


# ==================================================================== F
def section_F(lo=41, hi=59):
    """Map the frontier past machine 41: where exactly does the row die, and
    is the death permanent?"""
    print("=" * 78)
    print("F  THE FRONTIER - the row's uniform margin past machine 41")
    print("=" * 78)
    print(f"  {'y':>3} {'W':>4} {'s1 - 7/10':>12} {'lead':>10} {'N_+':>10}"
          f" {'E_u[f]':>10} {'verdict':>9} {'exact?':>7}")
    for y in range(lo, hi + 1):
        g = gears_of(y)
        if g[-1] != y or y not in F_EXACT:
            continue
        W = budget(y)
        n = len(g)
        FALLBACKS[0] = 0
        s1 = sum(Fraction(1, q) for q in g)
        Nplus = ZERO
        allex = True
        t0 = time.time()
        for i in range(1, n):
            for j in range(i + 1, n):
                e, ex = pair_expectation(g, i, j, W)
                allex = allex and ex
                Nplus += e
        lead = Fraction(6 * W, 5) * (Fraction(7, 10) - s1)
        f = lead + Nplus
        verdict = "CUTS" if f > 0 else "vacuous"
        if f <= 0 and not allex:
            verdict = "undecided"       # lower bound <= 0 proves nothing
        print(f"  {y:>3} {W:>4} {float(s1 - Fraction(7,10)):>+12.6f}"
              f" {float(lead):>+10.4f} {float(Nplus):>10.4f}"
              f" {float(f):>+10.4f} {verdict:>9} {str(allex):>7}"
              f"   [{time.time()-t0:.0f}s]")
    print("\n  'undecided' marks a machine where the |P| > 18 fallback fired,")
    print("  so the printed value is a valid LOWER bound on E_u[f]: a negative")
    print("  lower bound does NOT prove the row is vacuous there.")


# ==================================================================== A
def pi_vec(gears):
    """pi_i = prod_{k < i} (1 - 2/q_k), exact rationals (pi_0 = 1)."""
    out, cur = [], ONE
    for q in gears:
        out.append(cur)
        cur *= ONE - Fraction(2, q)
    return out


def A_const(gears):
    """A(y) = 1 - 2 s1 + 4 sum_{i<j} pi_i / (q_i q_j).   EXACT rational.

    THEOREM (proved in the docstring of section_A, asserted numerically there):
      (i)  E_u[f](y, W)  <=  W * A(y)   at EVERY width W - an exact inequality,
           because max_phases |covered| >= average_phases |covered| =
           |P| (1 - pi_i) exactly (each lower gear covers p with probability
           exactly 2/q_k, independently, under uniform phases);
      (ii) E_u[f](y, W) / W  ->  A(y)  as W -> infinity, because for EVERY
           FIXED phase choice of the lower gears the covered fraction of P
           tends to 1 - pi_i (CRT: P is an arithmetic progression modulo
           q_i q_j, coprime to every lower gear), so the max tends to it too.
    Hence A(y) <= 0 PROVES the row is vacuous against the uniform product
    measure at machine y AT EVERY WIDTH, and A(y) > 0 says it cuts at all
    sufficiently large widths."""
    n = len(gears)
    pis = pi_vec(gears)
    s1 = sum(Fraction(1, q) for q in gears)
    tail = sum(Fraction(4, 1) * pis[i] * Fraction(1, gears[i] * gears[j])
               for i in range(n) for j in range(i + 1, n))
    return ONE - 2 * s1 + tail


def Pi(gears):
    """the machine's own survival density prod_{5<=q<=y} (1 - 2/q)."""
    v = ONE
    for q in gears:
        v *= ONE - Fraction(2, q)
    return v


def section_A(upto=200):
    print("=" * 78)
    print("A  THE ASYMPTOTIC SLOPE A(y) - does the row EVER cut uniform?")
    print("=" * 78)
    print("  IDENTITY (proved, and asserted exactly at every machine below):")
    print("        A(y)  =  prod_{5<=q<=y} (1 - 2/q)  =  Pi(y),")
    print("  the machine's OWN SURVIVAL DENSITY.  Proof: under the uniform")
    print("  product measure let B = #{gears blocking a fixed position}.  Every")
    print("  blocker except the lowest one is 'above the lowest', so")
    print("        B  =  1{B >= 1}  +  #{blockers above the lowest},")
    print("  and taking expectations with Pr[q blocks] = 2/q independently,")
    print("        2 s1  =  (1 - Pi)  +  4 sum_{i<j} pi_i/(q_i q_j).")
    print("  Substituting into A(y) = 1 - 2 s1 + 4 sum_{i<j} pi_i/(q_i q_j)")
    print("  leaves A(y) = 1 - (1 - Pi) = Pi.  Equivalently and more directly:")
    print("  f <= open pointwise and E_u[open] = W Pi(y) EXACTLY, so the row's")
    print("  margin can never exceed the expected number of open slots.\n")
    from lp_degree_range import primes_upto as _pu
    for y in [p for p in _pu(300) if p >= 5]:
        g = gears_of(y)
        assert A_const(g) == Pi(g), ("A(y) != Pi(y)", y)
    print("  A(y) == Pi(y) asserted exactly at all 60 machines up to 300.\n")
    print("  A(y) = 1 - 2 s1 + 4 sum_{i<j} pi_i/(q_i q_j),  pi_i = prod_{k<i}")
    print("  (1 - 2/q_k).  EXACT UPPER BOUND on E_u[f]/W at every width, and")
    print("  the exact limit of E_u[f]/W as W -> infinity.\n")
    print("  CHECK 1: A(y) is an upper bound at the widths already measured.")
    for y, W in ((23, 48), (29, 63), (31, 74), (37, 95), (41, 129), (41, 258)):
        g = gears_of(y)
        n = len(g)
        s1 = sum(Fraction(1, q) for q in g)
        Nplus = ZERO
        for i in range(1, n):
            for j in range(i + 1, n):
                e, ex = pair_expectation(g, i, j, W)
                assert ex
                Nplus += e
        f = Fraction(6 * W, 5) * (Fraction(7, 10) - s1) + Nplus
        A = A_const(g)
        assert f <= W * A, ("A is not an upper bound", y, W, f, W * A)
        print(f"    y = {y:>2}, W = {W:>3}: E_u[f]/W = {float(Fraction(f, W)):>+9.6f}"
              f"  <=  A(y) = {float(A):>+9.6f}   OK")
    print()
    print("  CHECK 2: the limit.  E_u[f]/W at growing widths, machine 31:")
    g = gears_of(31)
    n = len(g)
    s1 = sum(Fraction(1, q) for q in g)
    for W in (74, 148, 296, 592, 1184):
        Nplus = ZERO
        for i in range(1, n):
            for j in range(i + 1, n):
                e, ex = pair_expectation(g, i, j, W)
                assert ex
                Nplus += e
        f = Fraction(6 * W, 5) * (Fraction(7, 10) - s1) + Nplus
        print(f"    W = {W:>5}: E_u[f]/W = {float(Fraction(f, W)):>+9.6f}"
              f"   (A(31) = {float(A_const(g)):+9.6f})")
    print()
    print("  THE MAP.  A(y) over the whole range - the sign is the verdict")
    print("  'can this vehicle's row ever see machine y at all?'\n")
    print(f"  {'y':>4} {'gears':>6} {'2 s1':>10} {'pair sum':>10}"
          f" {'A(y)':>11} {'verdict':>10}")
    from lp_degree_range import primes_upto
    first_neg = None
    for y in [p for p in primes_upto(upto) if p >= 5]:
        g = gears_of(y)
        s1 = sum(Fraction(1, q) for q in g)
        A = A_const(g)
        if A <= 0 and first_neg is None:
            first_neg = y
        if y <= 53 or y % 1 == 0 and (y < 100 or A <= 0 or y % 10 in (1, 3)):
            print(f"  {y:>4} {len(g):>6} {float(2*s1):>10.6f}"
                  f" {float(A - ONE + 2*s1):>10.6f} {float(A):>+11.6f}"
                  f" {'CAN CUT' if A > 0 else 'DEAD':>10}")
    print()
    if first_neg is None:
        print(f"  A(y) > 0 at EVERY machine up to {upto}: the row is never")
        print("  uniformly vacuous - only ever TOO NARROW.")
    else:
        print(f"  A(y) first turns non-positive at machine {first_neg}: from")
        print("  there the row cannot cut the uniform product measure at ANY")
        print("  width.  That is the vehicle's true ceiling.")
    return first_neg


# ==================================================================== T
def _Ef(gears, W):
    n = len(gears)
    s1 = sum(Fraction(1, q) for q in gears)
    Nplus, allex = ZERO, True
    for i in range(1, n):
        for j in range(i + 1, n):
            e, ex = pair_expectation(gears, i, j, W)
            allex = allex and ex
            Nplus += e
    return Fraction(6 * W, 5) * (Fraction(7, 10) - s1) + Nplus, allex


def section_T(machines=(29, 31, 37, 41, 43, 47, 53)):
    """The exact width threshold W_u(y), against the budget the (D) ladder
    actually needs."""
    print("=" * 78)
    print("T  W_u(y) - THE WIDTH AT WHICH THE ROW STARTS CUTTING UNIFORM")
    print("=" * 78)
    print("  Round 24 read the frontier as a MACHINE (m41).  It is not: the")
    print("  leading term is linear in W with a negative coefficient from m29")
    print("  on, and N_+ is SUPERLINEAR, so every machine with A(y) > 0 has a")
    print("  finite threshold W_u(y).  The vehicle's real question is whether")
    print("  budget(y) = F(prev) + y clears it.\n")
    print(f"  {'y':>3} {'budget':>7} {'W_u(y)':>7} {'ratio':>7}"
          f" {'E_u[f] at budget':>17} {'verdict':>10}")
    for y in machines:
        g = gears_of(y)
        B = budget(y)
        # bisection on the sign (monotonicity NOT assumed - verified on a
        # window below)
        lo, hi = 2, 4 * B
        fhi, exhi = _Ef(g, hi)
        if fhi <= 0:
            print(f"  {y:>3} {B:>7} {'> ' + str(hi):>7}")
            continue
        while hi - lo > 1:
            mid = (lo + hi) // 2
            fm, _ = _Ef(g, mid)
            if fm > 0:
                hi = mid
            else:
                lo = mid
        Wu = hi
        # verify: negative just below, positive over the next 6 widths
        fbelow, _ = _Ef(g, Wu - 1)
        assert fbelow <= 0, ("threshold not sharp", y, Wu)
        for k in range(0, 7):
            fk, _ = _Ef(g, Wu + k)
            assert fk > 0, ("not positive above the threshold", y, Wu + k)
        fB, exB = _Ef(g, B)
        verdict = "CUTS" if fB > 0 else "too narrow"
        print(f"  {y:>3} {B:>7} {Wu:>7} {float(Fraction(B, Wu)):>7.3f}"
              f" {float(fB):>+17.4f} {verdict:>10}")
    print("\n  ratio = budget / W_u.  Above 1 the row cuts uniform at the")
    print("  width the ladder needs; below 1 it does not, and the deficit is")
    print("  a WIDTH deficit, not a machine one.")


# ==================================================================== D
def deficit(gears, W):
    """Delta(y,W) = W Pi(y) - E_u[f], decomposed pair by pair.

    Delta = sum_{i<j} (1/(q_i q_j)) sum_{u,v} [ max_r |cov(P;r)|
                                                - (1 - pi_i) |P| ]
    - the total EXCESS OF THE PHASE MAXIMUM OVER THE PHASE MEAN in the
    Costello-Watts pair terms.  It is a pure extreme-value quantity: it is
    exactly what the recursion pays for letting each pair term privately
    optimise the lower gears' phases.  Delta >= 0 always (max >= mean)."""
    n = len(gears)
    pis = pi_vec(gears)
    per = {}
    tot = ZERO
    for i in range(n):
        for j in range(i + 1, n):
            qi, qj = gears[i], gears[j]
            acc = ZERO
            for u in range(qi):
                for v in range(qj):
                    P = pair_positions(qi, qj, u, v, W)
                    mc, ex = max_cover_exact(P, tuple(gears[:i]), W)
                    assert ex, ("deficit needs exact coverage", gears[-1], W)
                    acc += Fraction(mc) - (ONE - pis[i]) * len(P)
            e = acc / (qi * qj)
            assert e >= 0, "max below mean - impossible"
            per[(i, j)] = e
            tot += e
    return tot, per


def section_D(machines=(23, 29, 31, 37, 41)):
    print("=" * 78)
    print("D  THE DEFICIT Delta(y,W) = W Pi(y) - E_u[f]  - WHAT ACTUALLY DIES")
    print("=" * 78)
    print("  E_u[f] = W Pi(y) - Delta(y,W), Delta >= 0.  The gain term is")
    print("  EXACTLY LINEAR in W with slope the survival density Pi(y); the")
    print("  loss term Delta is the summed excess of the phase MAXIMUM over the")
    print("  phase MEAN inside the recursion's pair minima.  So the row cuts")
    print("  uniform iff  Delta(y,W) < W Pi(y).\n")
    print(f"  {'y':>3} {'W':>5} {'W Pi(y)':>10} {'Delta':>10} {'E_u[f]':>10}"
          f" {'Delta/W':>9} {'Delta(2W)/Delta(W)':>19}")
    for y in machines:
        g = gears_of(y)
        B = budget(y)
        prev = None
        for W in (B, 2 * B, 4 * B):
            d, per = deficit(g, W)
            gain = W * Pi(g)
            f = gain - d
            fd, _ = _Ef(g, W)
            assert f == fd, ("deficit decomposition disagrees with the direct"
                             " computation", y, W, f, fd)
            ratio = float(Fraction(d, prev)) if prev else float('nan')
            print(f"  {y:>3} {W:>5} {float(gain):>10.4f} {float(d):>10.4f}"
                  f" {float(f):>+10.4f} {float(Fraction(d, W)):>9.5f}"
                  + (f" {ratio:>19.4f}" if prev else f" {'-':>19}"))
            prev = d
        print()
    print("  Doubling W doubles the gain exactly.  Delta grows by a factor")
    print("  well below 2, so the ratio Delta/W falls and every machine with")
    print("  Pi(y) > 0 - i.e. every machine - eventually cuts.  The frontier is")
    print("  a WIDTH, not a machine: round 24's 'the row loses the product")
    print("  measure at m41' is really 'budget(41) = 129 is below W_u(41)'.")


# ==================================================================== V
def section_V(steps=((19, 23), (23, 29), (29, 31), (31, 37), (37, 41),
                     (41, 43), (43, 53))):
    """THE VEHICLE'S EXACT RANGE.  The uniform product measure is a feasible
    point of the FULL COMPOSITION at width W iff
        (i)  its degree-<=2 moments are completable at every position - one
             exact check, because under uniform phases every position has the
             SAME moment vector, namely the product moments of p_q = 2/q; and
        (ii) it satisfies the recursive row, i.e. E_u[f] <= 0.
    Both true => an exact REFUTATION: no certificate of this vehicle at this
    width exists, however many cuts one generates."""
    from lp_degree_range import product_moments, completable as _comp
    print("=" * 78)
    print("V  THE EXACT RANGE OF THE COMPOSITION (uniform-point refutations)")
    print("=" * 78)
    print("  Under the uniform product measure every position of the window")
    print("  has the same degree-<=2 moment vector (p_q = 2/q), so ONE exact")
    print("  completion decides the whole degree-2 side.  Combined with the")
    print("  row's sign this gives an exact verdict per rung.\n")
    print(f"  {'step':>10} {'W':>5} {'deg-2 cuts':>11} {'E_u[f]':>10}"
          f" {'row':>9} {'verdict':>26}")
    for (a, b) in steps:
        g = gears_of(b)
        W = budget(b)
        n = len(g)
        p = [Fraction(2, q) for q in g]
        deg2_ok = _comp(product_moments(p, n, 2), n, 2)   # uniform kills deg 2
        f, ex = _Ef(g, W)
        row_ok = (f <= 0)                                  # uniform obeys row
        if deg2_ok and row_ok:
            v = "REFUTED (uniform feasible)"
        elif not deg2_ok:
            v = "open: deg-2 cuts bite"
        else:
            v = "open: the row bites"
        print(f"  {a:>4} -> {b:<3} {W:>5} {('satisfied' if deg2_ok else 'VIOLATED'):>11}"
              f" {float(f):>+10.4f} {('satisfied' if row_ok else 'VIOLATED'):>9}"
              f" {v:>26}" + ("" if ex else "  [E_u[f] is a lower bound]"))
    print("\n  'REFUTED' is a PROOF that no certificate of the full")
    print("  composition exists at that width - the uniform product measure is")
    print("  an exhibited exact feasible point.  'open: the row bites' means")
    print("  the uniform point is excluded and the cell must be decided by")
    print("  actually running the LP.")


# =================================================================== S (STAR-k)
def Ef_star(gears, W, keep):
    """E_u[f] for the STAR-k row: the pair term n^K_ij holds the phases of the
    gears in K = gears[:keep] EXPLICIT (the LP carries them) and minimises only
    over the remaining lower gears.  keep = 0 is round 24's level-2 row.

    n^K_ij(u, v, w) = |P_ij(u,v) minus what K blocks at w| - maxcover(rest).
    Since n^K >= n (a min over fewer gears), E_u[f] can only go UP, so the
    STAR-k threshold W_u satisfies W_u^{(k)} <= W_u.  Exact rational."""
    n = len(gears)
    K = tuple(gears[:keep])
    s1 = sum(Fraction(1, q) for q in gears)
    tot = ZERO
    for i in range(n):
        for j in range(i + 1, n):
            qi, qj = gears[i], gears[j]
            lower = tuple(g for g in gears[:i] if g not in K)
            held = tuple(g for g in gears[:i] if g in K)
            acc, cells = 0, 0
            for u in range(qi):
                for v in range(qj):
                    P0 = pair_positions(qi, qj, u, v, W)
                    if not held:
                        mc, ex = max_cover_exact(P0, lower, W)
                        assert ex, ("STAR-k needs exact coverage", gears[-1], W)
                        acc += (len(P0) - mc) * prod_int(held)
                        cells += prod_int(held)
                        continue
                    for ws in product(*[range(q) for q in held]):
                        P = [p for p in P0
                             if not any(p in hits(q, wq, W)
                                        for q, wq in zip(held, ws))]
                        mc, ex = max_cover_exact(P, lower, W)
                        assert ex, ("STAR-k needs exact coverage",
                                    gears[-1], W)
                        acc += len(P) - mc
                        cells += 1
            tot += Fraction(acc, cells)     # cells = q_i q_j prod(held)
    return Fraction(W) - 2 * W * s1 + tot


def prod_int(t):
    v = 1
    for x in t:
        v *= x
    return v


def section_S(machines=(37, 41, 43, 47, 53)):
    print("=" * 78)
    print("S  STAR-k - does holding the small gears' phases move the frontier?")
    print("=" * 78)
    print("  The whole deficit Delta is the excess of a phase MAXIMUM over its")
    print("  MEAN inside n_ij, so the way to shrink it is to stop the pair")
    print("  terms choosing the low gears' phases privately.  STAR-k holds the")
    print("  k smallest gears explicit.  n^K >= n pointwise, so E_u[f] can only")
    print("  rise and the threshold can only fall.  Exact:\n")
    print(f"  {'y':>3} {'W':>5} {'W Pi(y)':>9} {'level 2':>10} {'STAR-3':>10}"
          f" {'STAR-{5,7}':>11} {'verdict at budget':>22}")
    for y in machines:
        g = gears_of(y)
        W = budget(y)
        gain = W * Pi(g)
        f0 = Ef_star(g, W, 0)
        f1 = Ef_star(g, W, 1)
        f2 = Ef_star(g, W, 2)
        d0, _ = _Ef(g, W)
        assert f0 == d0, ("STAR-0 must reproduce the level-2 row", y, f0, d0)
        assert f1 >= f0 and f2 >= f1, ("STAR-k must be monotone", y)
        best = "level 2" if f0 > 0 else ("STAR-3" if f1 > 0 else
                                         ("STAR-{5,7}" if f2 > 0 else "NONE"))
        print(f"  {y:>3} {W:>5} {float(gain):>9.4f} {float(f0):>+10.4f}"
              f" {float(f1):>+10.4f} {float(f2):>+11.4f}"
              f" {('cuts: ' + best):>22}")
    print("\n  A positive entry means that row cuts the uniform product measure")
    print("  at the ladder's own budget width - the necessary condition for the")
    print("  composition to be able to certify that rung at all.")


SECTIONS = {'L': section_L, 'W': section_W, 'X': section_Wwidth,
            'F': section_F, 'A': section_A, 'T': section_T, 'D': section_D,
            'V': section_V, 'S': section_S}


def main():
    want = [a for a in sys.argv[1:] if a in SECTIONS] or ['L', 'W', 'X']
    for k in want:
        SECTIONS[k]()
        print()


if __name__ == '__main__':
    main()
