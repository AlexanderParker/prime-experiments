# covering-lp-certificates - scan-free upper bounds on F(M) by LP duality over the covering IP

Status: SCRIPT-VERIFIED (every claim asserted in `research/matrix_shapes.py`,
section SHAPE 2; exact integer/Fraction arithmetic for all claims - the LP
solver is used for DISCOVERY only and every reported bound is re-verified by
an exact rational Farkas certificate). Established round 21 (human matrix
directive: "try different matrix shapes and operations"). Prior-art check:
NOT YET CHECKED (section 6).

## 1. WHAT IT IS

Plain language. "How wide can a run of blocked slots be?" is a covering
question: a window of W consecutive slots is fully covered iff every slot is
hit by some gear's tooth. Because CRT realises every combination of per-gear
phases, choosing the window's position IS choosing one phase per gear - so
F(M) - 1 = the largest W for which the phase-covering integer program is
feasible, EXACTLY (not a model). Relaxing that IP to a linear program and
taking DUALS produces certificates: finite lists of rational weights whose
verification is pure linear arithmetic and which prove "no window of width W
is fully covered", i.e. F(M) <= W - with no period scan, at machines whose
periods are far beyond scanning.

The exact IP. For gears q with tooth sets T_q = {u_q, -u_q}: width W is
coverable iff there exist phases r_q in Z_q with, for every i in [0, W),
some q having (r_q + i) mod q in T_q. Coverable(W) iff W <= F(M) - 1.

Level-1 LP. Fractional phases z_{q,r} >= 0, sum_r z_{q,r} = 1, coverage
constraint sum_q [z_{q,(u-i) mod q} + z_{q,(-u-i) mod q}] >= 1 per position.

Level-2 LP. Add one joint distribution per gear PAIR and the KOUNIAS CUT,
which is a POINTWISE identity of 0/1 indicators (brute-verified over all
patterns on <= 6 events): for every distinguished gear k,

    1{covered} <= sum_j 1{A_j}  -  sum_{j != k} 1{A_j and A_k}.

Results (all exact unless labeled):

1. LEVEL-1 = THE DENSITY BOUND, AND ITS INTEGRALITY GAP IS INFINITE FROM
   MACHINE 13 ON. Uniform fractional phases give coverage exactly
   sum_q 2/q at every position of every width; sum 2/q >= 1 from machine 13
   (5112/5005) onward, so the level-1 LP is feasible at EVERY width while
   true F stays finite. At machine 11 (sum = 334/385 < 1) the level-1 LP
   threshold is finite and small: min infeasible width 8, exact dual
   certificate sum_q max_r (y-mass) = 23/24 < 1 - so F(11) <= 8 by LP
   duality alone (true 7).

2. LEVEL-2 CERTIFICATES, EXACT-VERIFIED (the finding). Min infeasible
   widths of the level-2 LP, each with a rational Farkas certificate
   verified in Fraction arithmetic (certificate = a list of nonnegative
   cut weights y_{i,k}; verification = per-gear and per-pair maxima of
   y-weighted coverage summing below sum y):

       machine 11: F <= 8    (true 7,  certificate support 8 weights)
       machine 13: F <= 21   (true 11, support 25)
       machine 17: F <= 31   (true 18, support 45)
       machine 19: F <= 37   (true 25, support 44)
       machine 23: F <= 90   (true 34, support 93)  <- SCAN-FREE: period
                                    3.7e7 exceeds the exhaustive-scan cap

3. CLOSED-FORM COROLLARY (no LP at all). Averaging the Kounias cut over
   the window with exact hit counts (per gear <= 2*ceil(W/q) per phase,
   per pair >= 4*floor(W/(q q')) per phase pair, both asserted):

       if for some gear k:
           sum_q 2*ceil(W/q) - 4*sum_{j != k} floor(W/(q_j q_k)) < W
       then F(M) <= W.

   Values: F(13) <= 35, F(17) <= 65, F(19) <= 110, F(23) <= 285 - pure
   integer arithmetic a Lean kernel could decide in one pass.

4. THE CEILING LAW (exact Fractions). The level-2 mechanism DIES at
   machine 29: the uniform product measure has per-position slope
   sum_q 2/q - (4/q_k) sum_{j != k} 1/q_j >= 1 for every k from y = 29 on
   (1.0001 at 29 with k = 5), so it satisfies every Kounias cut at every
   width and the level-2 LP is feasible forever. The chain-cut extension
   (a depth-t pointwise Bonferroni family, also brute-verified) revives
   the mechanism: chain depth 2 (triple moments, moment degree 3) has
   slope < 1 at machines 29..43. The moment degree needed grows with the
   machine - the hierarchy meets the escape-distance wall as a QUANTIFIED
   ceiling, level by level.

## 2. WHY IT MIGHT BE NOVEL

- The project has exact F values (scans, COV/coverable search, nilpotency
  powering) but NO certifying upper bounds: COV's `coverable` is a search
  whose negative answer is exhaustion, not an artifact. A Farkas
  certificate is a checkable OBJECT - dozens of rationals, verified by
  finitely many comparisons - the first scan-free, polynomial-size,
  kernel-checkable route to F(M) <= W statements.
- The identification is sharp on both ends: level-1 LP = exactly the
  first-moment density bound (nothing more), and the level-l hierarchy's
  feasibility thresholds are the Bonferroni truncation signs - the same
  2^|Y| inclusion-exclusion ladder Constructor's renewal ladder (R38)
  climbs for counts, now appearing as the DUAL side of an optimization,
  with a computable per-level ceiling (level 2 spans machines 13..23,
  dies at 29).
- Honest caveats: the bounds are weak (1.14x to 2.65x above true F, and
  the ratio degrades toward each level's ceiling); the ingredients
  (Kounias/Bonferroni inequalities, LP duality, set-cover relaxations)
  are all standard - the candidate novelty is only their application to
  Jacobsthal-type maximal-gap certification. Kanold/Stevens/Iwaniec-style
  upper bounds on the Jacobsthal function may contain the closed-form
  corollary or better; the LP-dual certificate form is the part not
  expected to appear there. UNCONFIRMED until section 6 is done.

## 3. PROOF

SCRIPT-VERIFIED: `research/matrix_shapes.py` (run:
`uv run python research/matrix_shapes.py 2`). Asserted: pointwise validity
of the Kounias and chain cuts (exhaustive over all 0/1 patterns, <= 6
events); the exact hit-count bounds; IP width = F - 1 at machines 11..19
(period sieve); the uniform level-1 certificate (Fractions); every reported
level-2 bound via exact rational Farkas verification; F - 1 < closed-form
W* wherever F is known; the level-2 death at 29+ (exact slope Fractions).
SOLVER-labeled: only the discovery of minimal infeasible widths (scipy
HiGGS); each endpoint is then exact-verified.

Validity chain for a bound: exact IP (CRT) => any valid relaxation's
infeasibility at W => no covered window of width W => F(M) <= W. The
level-2 relaxation drops pair-marginal consistency (a further weakening,
so certificates remain valid).

Kernel-checkable candidates: the closed-form corollary at a fixed machine
(one line of integer arithmetic per W); a fixed Farkas certificate (finite
list of rationals + finitely many linear comparisons), e.g. F(23) <= 90
without any period object.

## 4. IMPLICATIONS

- Fills a named gap from the round-19 summary: "the UPPER bounds on F_j
  that every prefix row currently lacks" - this is an upper-bound
  MECHANISM for F (extension to F_j windows with interior conditions
  untested; the qualifying-interior condition is a disjunction and will
  meet the same hierarchy wall).
- Gives the formalist a new certificate species: rational Farkas lists,
  much smaller than period scans (machine 19: 44 weights vs a 1,616,615
  slot scan; machine 23: 93 weights vs an unscanned 3.7e7 period).
- Quantifies HOW MUCH correlation information each moment level carries:
  degree 2 certifies finiteness of F only up to machine 23; from 29 the
  pair table provably cannot certify any width bound (matches, from the
  optimization side, Constructor R37's "the depth cap is a >= 3-point
  phenomenon from machine 19" and R36's non-Markov deficit).

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

- Jacobsthal upper bounds: the mechanism produces certified h-type upper
  bounds for arbitrary gear sets (nothing here is twin-specific; the
  general-difference family of paired-jacobsthal-values.md is in scope).
- (D): a certificate for F(M+q') <= F + q' would need the ratio to reach
  1 + q'/F - far beyond level-2 strength (2.65x at 23). The hierarchy
  formalises what (D) demands: moment degree growing with the machine,
  i.e. exactly the anti-correlation depth Constructor measures.
- Level-3: chain cuts with triple moments have slope < 1 through at least
  machine 43 (exact) - the LP is larger but polynomial; its bounds and its
  own ceiling are unmeasured.

## 6. PRIOR-ART CHECK

Not yet checked (round-21 subagent had the check out of scope). Suggested
searches: "Jacobsthal function upper bound linear programming"; "covering
congruences LP relaxation certificate"; "Bonferroni inequality sieve
maximal gap"; "Kounias bound covering system"; Kanold 1967, Stevens,
Iwaniec on Jacobsthal upper bounds (do any give per-machine finite
certificates of this counting form?); Erdos covering systems LP literature.
Status: UNCONFIRMED.
