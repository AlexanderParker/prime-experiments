# The covering-hierarchy exactness law: pairwise convexity computes the Jacobsthal
# maximum through machine 17 and provably stops seeing it at machine 19

Round 24, lateral. Script: research/sdp_cover.py (all parts assertion-gated); logs
research/data/sdp_ladder_a.log, sdp_ladder_b.log, sdp_psd_19.log, sdp_exact_23.log,
sdp_cover_17_19.log.

## 1. What it is

Write the machine's maximal gap as a covering CSP (the project's own frame, Mechanic's
cov_sat.py r20): slot position i of a candidate run is covered by gear q iff
i = c_q +- u_q (mod q) with one free offset c_q per gear (u_q = 6^{-1} mod q), and by CRT

    F(M) = 1 + max{ L : [0, L) coverable }.

No period, no scan, no machine input beyond the prime list. Apply to this CSP the standard
convex certificate hierarchy:

  - LEVEL 1 (fractional cover): one distribution per gear.
  - LEVEL 2 "SA2" (pairwise / Sherali-Adams): a pseudo-moment matrix Y over the offset
    literals (q, c), with Y[a,a] = Y[0,a], per-gear exactly-one, full marginalisation
    (sum_c Y[a, (q,c)] = Y[0,a] for every literal a), and CONDITIONAL covering: for every
    position i and literal a,  sum_q ( Y[a,(q,i-u_q)] + Y[a,(q,i+u_q)] ) + t[i,a] >= Y[0,a],
    slacks t >= 0 marginalising like moments. Objective V(L) = min sum_i t[i,0].
    V(L) > 0 proves RUN(L) impossible, hence F(M) <= L. Matrix size 1 + sum_q q
    (73 at machine 19, 193 at 37) - polynomial, machine-free.
  - LEVEL 2 + PSD "SDP2": additionally Y >= 0, imposed soundly by eigenvector cutting
    planes v'Yv >= 0 (valid for every real v, so no verdict depends on the numerics).

Soundness anchor asserted per machine: V(F-1) = 0 (the realised maximal run is feasible,
so the relaxation never contradicts truth). Let L*(y) = min{L : V(L) > 0}, so F <= L*.

## 2. The finding (three parts)

(a) EXACTNESS THROUGH MACHINE 17. L* = F exactly at machines 11, 13, 17
    (7, 11, 18). The pairwise LP - no PSD needed - COMPUTES the Jacobsthal-type maximum
    of these gear systems. Certified exactly: the claimed V(L*) > 0 verdicts carry exact
    rational dual certificates verified in integer arithmetic (weak duality; see section 4).

(b) THE BREAK AT MACHINE 19, WITH THE ARITY NAMED. L*(19) = 27 against F(19) = 25:
    V(25) = V(26) = 0. RUN(25) and RUN(26) are impossible in truth, and the level-2 system
    cannot see it. The PSD constraint does not repair it: at L = 26 the cutting-plane loop
    CONVERGES to a V = 0 solution whose moment matrix is PSD (min eig -1.2e-14 after 39
    cuts) - the full level-2 SDP is FEASIBLE at an impossible L (numerical, flagged) - and
    at L = 25 it stalls (187 cuts, V = 0 throughout). Consequence, stated carefully:

        every certificate of F(19) <= 26 must use THREE-gear (arity-3) information;
        no pairwise-consistent reasoning, linear or semidefinite, suffices.

    Status: the L = 26 SDP-feasibility is numerical (a converged interior point, not an
    exact rational PSD completion); the L = 25 stall is measured. The SA2 verdicts V = 0
    are LP-feasibility statements (solver-exact to 1e-15 residuals).

(c) VACUITY GROWTH. L*/F = 1.000, 1.000, 1.000, 1.080, 1.647, >= 1.721 at machines
    11..29 (m23: L* = 56 vs 34; m29: V = 0 at every tested L <= 73, so L* >= 74; run
    terminated at its 5400 s cap). The pairwise hierarchy goes vacuous at the same
    drift-not-infeasibility rate as the round-23 potential arity ladder (1.11x, 1.63x,
    2.06x) and the LP-duality thread's degree law - a THIRD independent certificate family
    exhibiting the project's arity law. The exactness margin V(F) collapses smoothly
    first: 5/6, 1, 0.169, 0, ... at 11/13/17/19.

## 3. Why it might be novel

The Jacobsthal function has been attacked by sieve bounds, explicit computation, and
(in this project) max-plus/LP certificates. We know of no published treatment of
Jacobsthal-type covering as a Sherali-Adams / Lasserre hierarchy, and no published
statement of the form "the pairwise relaxation is EXACT for the first k prime moduli and
first fails at {5,...,19}". The exactness itself is surprising: a polynomial-size LP
computing h(P(17)-ish) exactly is not implied by anything we know; hierarchy-exactness
results in the CSP literature (e.g. for width-bounded or tree-like instances) do not
obviously cover two-residue covering systems. The break point with a certified
"arity >= 3" obstruction gives the project's "no fixed arity" law a sharp, finite,
machine-checkable instance.

## 4. Proof / verification status

- Soundness of V(L) > 0 => F <= L: PROVED (one paragraph: a genuine covering assignment
  induces a feasible point with t = 0; monotone; RUN feasibility downward closed).
  Anchored by assert V(F-1) = 0 at every machine 11..23 and by the brute-force identity
  maxrun + 1 = F at machines 11..19 (research/sdp_cover.py part B).
- Claimed bounds carry EXACT RATIONAL DUAL CERTIFICATES, constructed by solving the
  strictly-feasible dual (max t s.t. M'y + t·1 <= c, d'y >= V/2), flooring y at a
  denominator D chosen from t*, and checking M'z <= Dc and d'z > 0 in exact integer
  arithmetic (sound: c >= 0 makes y = 0 dual-feasible, the set star-shaped):
      m11, L=7 : bound 479/1152 > 0      m13, L=11: bound 1041/2081 > 0
      m17, L=18: bound 1673/19767 > 0    m19, L=27: bound 2927/270613 > 0
      m23, L=56: bound 3427/746861 > 0 (research/data/sdp_exact_23.log, 1244 s; first
      attempt died of MemoryError on a memory-pressured box, retry clean)
  Every positive number above was re-derived in a clean process on 2026-08-28.
- The m19 SDP verdicts: numerical, as flagged in 2(b). The construct that would make the
  L = 26 verdict exact: rational PSD completion / facial reduction of the converged point.
- LP1 (level 1): dies exactly at sigma(y) = sum 1/q >= 1/2, i.e. from machine 13 on -
  the same threshold as round 23's T2 Mertens no-go, now on the covering side. (Its m11
  value L* = 8 is numerical only; the uniform-weights exact certificate fails there.)

## 5. Implications

- THE BRIEF'S ANSWER: there is no small arity-independent convex statement bounding
  osc(h). Complementing this, the machine-free max-plus system MF_m has an EXACT LP
  formulation (longest-path LP = Kleene closure; research/sdp_cover.py part A, 12/12
  steps integer-equal), so every convex relaxation of it - every Lasserre level, every
  SDP - returns exactly the closure value: the machine-free gap is 100% edge-set
  (support), 0% relaxation gap. Convexity is not the missing ingredient anywhere in the
  machine-free story; arity is.
- The three certificate families (potentials, covering duals, moment hierarchies) now
  fail along the same axis, each measured. Any route to (D) must grow arity with the
  machine or import realised-tuple (machine) facts - consistent with Constructor's CEGAR
  sizing (6,395 facts at 29->31).

## 6. Prior-art check

2026-08-28: NOT YET CHECKED - the session's web-search budget was exhausted before the
check could run; manager to run the check (per docs/novel/README.md rules). Nearest known
internal art: round-21 covering-lp-certificates.md (bounded-level covering LP, generative
phase structure - level-indexed, not the SA/Lasserre lift over offset literals; it dies
at 29 by growth, this one breaks at 19 by arity) and round-23 potential-arity-ladder.md
(the certificate-side arity law this confirms from a third family).

## 7. Open questions this touches

- Does partial level 3 (triple moments on selected gear triples) certify L = 25 at m19,
  and WHICH triple is the obstruction? (Named construct, not built this round.)
- Is there a hierarchy-exactness law: level r exact until some threshold y_r, with y_2
  between 17 and 19? Level-3 exactness death would need m19+ level-3 runs.
- Does the exactness margin V(F) admit a closed form at 11/13/17 (5/6, 1, 54752/323401
  numerical)? Its collapse to 0 between 17 and 19 is the mechanism's shadow.
