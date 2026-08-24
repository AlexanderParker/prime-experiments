# covering-lp-certificates - scan-free upper bounds on F(M) by LP duality over the covering IP

Status: SCRIPT-VERIFIED, and since round 22 EXACT ON BOTH SIDES of every
reported threshold (`research/lp_dual_certs.py` + the exact rational LP core
`research/exact_lp.py`; the round-21 origin is `research/matrix_shapes.py`
SHAPE 2).  Round 21 verified only the INFEASIBLE endpoint; round 22 added an
exact rational feasible point at W*-1, so the thresholds below are now
bracketed by two exact certificates and are not solver claims.  Established
round 21 (human matrix directive: "try different matrix shapes and
operations"), pushed to depth round 22 (LP-duality dedicated explorer).
Prior-art check: DONE 2026-08-24 - PARTIAL OVERLAP, section 6.

Companion entry: `moment-degree-ceiling.md` is the obstruction half - how far
this vehicle can possibly go.  Read both.

## 1. WHAT IT IS

Plain language. "How wide can a run of blocked slots be?" is a covering
question: a window of W consecutive slots is fully covered iff every slot is
hit by some gear's tooth. Because CRT realises every combination of per-gear
phases, choosing the window's position IS choosing one phase per gear - so
F(M) - 1 = the largest W for which the phase-covering integer program is
feasible, EXACTLY (not a model). Relaxing that IP to a linear program and
taking DUALS produces certificates: finite lists of rational weights whose
verification is pure linear arithmetic and which prove "no window of width W
is fully covered", i.e. F(M) <= W - with no period scan.

The exact IP. For gears q with tooth sets T_q = {u_q, -u_q}: width W is
coverable iff there exist phases r_q in Z_q with, for every i in [0, W),
some q having (r_q + i) mod q in T_q. Coverable(W) iff W <= F(M) - 1.

Level-1 LP. Fractional phases z_{q,r} >= 0, sum_r z_{q,r} = 1, coverage
constraint sum_q [z_{q,(u-i) mod q} + z_{q,(-u-i) mod q}] >= 1 per position.

Level-2 LP. Add one joint distribution per gear PAIR and the KOUNIAS CUT,
which is a POINTWISE identity of 0/1 indicators (brute-verified over all
patterns on <= 6 events): for every distinguished gear k,

    1{covered} <= sum_j 1{A_j}  -  sum_{j != k} 1{A_j and A_k}.

### Results

1. LEVEL-1 = THE DENSITY BOUND, AND ITS INTEGRALITY GAP IS INFINITE FROM
   MACHINE 13 ON. Uniform fractional phases give coverage exactly
   sum_q 2/q at every position of every width; sum 2/q >= 1 from machine 13
   (5112/5005) onward, so the level-1 LP is feasible at EVERY width while
   true F stays finite. At machine 11 (sum = 334/385 < 1) the threshold is
   finite: W* = 8 exactly, so F(11) <= 8 by LP duality alone (true 7).

2. EXACT LEVEL-2 THRESHOLDS AND INTEGRALITY GAPS (round 22; both endpoints
   exact - an exact rational feasible point at W*-1 and an exact Farkas
   certificate at W*, the latter re-verified against the UNPRUNED column
   set):

       machine 11: W* =  8,  F =  7,  gap 8/7  = 1.143   (8 dual weights)
       machine 13: W* = 21,  F = 11,  gap 21/11 = 1.909  (32 weights)
       machine 17: W* = 31,  F = 18,  gap 31/18 = 1.722  (32 weights)
       machine 19: W* = 37,  F = 25,  gap 37/25 = 1.480  (37 weights)
       machine 23: W* = 90,  F = 34,  gap        = 2.647  (SOLVER-discovered
                     + exact Farkas only; the feasible endpoint at 89 was not
                     recomputed exactly in round 22)

   The round-21 solver values 8 / 21 / 31 / 37 are CONFIRMED EXACTLY.

3. A (D) STEP PROVED BY A DUAL CERTIFICATE (round 22, the finding).  A
   certificate at machine M+q' of width W = F(M) + q' proves the (D) step
   F(M+q') <= F(M) + q' outright, with no period of M+q' built:

       step    budget F(M)+q'   LP2 W*   verdict
        7->11        16            8     (D) PROVED
       11->13        20           21     missed by 1
       13->17        28           31     missed by 3
       17->19        37           37     (D) PROVED, EXACTLY TIGHT
       19->23        48           90     missed

   F(19) <= 37 = F(17) + 19 from 37 nonnegative rationals and finitely many
   comparisons - a second, fully independent proof vehicle for (D) at one
   step, unrelated to the merge law / flatness / qualifying-spectrum route.
   Verification cost 1,480 rational operations against a 1,616,615-slot
   period scan (1,092x fewer operations).

4. ZERO-COLUMN REDUCTION / PAIR VISIBILITY (round 22).  Pair variables enter
   the level-2 LP only NEGATIVELY, so if some phase pair of (q_a, q_b) blocks
   no common position of [0,W), putting all that pair's mass there makes its
   contribution identically zero and the pair drops out of the LP entirely.
   Each of the 4 tooth combinations rules out at most W phase pairs, hence

       q_a q_b > 4W   =>   the pair is INVISIBLE to the level-2 LP.

   Asserted exhaustively.  Consequences: at W = F the level-2 LP sees 0 of 6
   pairs at machine 13, 1 of 10 at 17, 3 of 15 at 19, 6 of 21 at 23 - the
   visible fraction goes to 0.  This is both the mechanism behind the ceiling
   law AND the reduction that makes exact rational solution affordable (the
   certificates above use 1, 5 and 7 visible pairs).

5. CLOSED-FORM COROLLARY (no LP at all). Averaging the Kounias cut over
   the window with exact hit counts (per gear <= 2*ceil(W/q) per phase,
   per pair >= 4*floor(W/(q q')) per phase pair, both asserted):

       if for some gear k:
           sum_q 2*ceil(W/q) - 4*sum_{j != k} floor(W/(q_j q_k)) < W
       then F(M) <= W.

   Values: F(13) <= 35, F(17) <= 65, F(19) <= 110, F(23) <= 285 - pure
   integer arithmetic a Lean kernel could decide in one pass.  (Section 6:
   this corollary has a published stronger relative.)

6. THE CEILING, NOW FAMILY-FREE (round 22; details in
   `moment-degree-ceiling.md`).  Round 21 reported that the level-2
   mechanism dies at machine 29 because the uniform product measure
   satisfies every Kounias cut there.  Round 22 shows this is not a
   limitation of the Kounias family: the SHARP degree-2 test (does the
   product measure's degree-<=2 moment vector extend to a distribution with
   no empty atom?) is also feasible from machine 29 - so NO degree-2
   inequality of any kind can certify anything from machine 29 on, and the
   integrality gap of every degree-2 relaxation is infinite there.  Degree 1
   dies at 13, degree 2 at 29, degree 3 at >= 151.  The degree a certificate
   must carry grows like 2*S1(y) ~ 4 log log y.

## 2. WHY IT MIGHT BE NOVEL

- The project has exact F values (scans, COV/coverable search, nilpotency
  powering) but its other upper bounds are searches whose negative answer is
  exhaustion. A Farkas certificate is a checkable OBJECT - dozens of
  rationals, verified by finitely many comparisons - a scan-free,
  polynomial-size, kernel-checkable route to F(M) <= W statements, and at
  machine 19 it lands a full (D) step exactly.
- The identification is sharp on both ends: level-1 LP = exactly the
  first-moment density bound (nothing more), and the level-l hierarchy's
  feasibility thresholds are the Bonferroni truncation signs - the same
  inclusion-exclusion ladder Constructor's renewal ladder (R38) climbs for
  counts, now appearing as the DUAL side of an optimization, with a
  computable per-degree ceiling.
- The pair-visibility bound q_a q_b > 4W is specific to the two-teeth
  structure and appears to be new; it is what turns an intractable LP into a
  handful of columns.
- Honest caveats: the bounds are weak (1.14x to 2.65x above true F); the
  ingredients (Kounias/Bonferroni inequalities, LP duality, set-cover
  relaxations, the Boole-Bonferroni LP) are all standard; and the closed-form
  corollary has a published stronger relative (Costello-Watts).  The
  candidate novelty is the dual-certificate FORM for Jacobsthal-type maximal
  gaps, the visibility reduction, and the (D) application.

## 3. PROOF

SCRIPT-VERIFIED.  Round 22: `research/lp_dual_certs.py` (run
`uv run python research/lp_dual_certs.py A B C D E`), exact LP core
`research/exact_lp.py` (two-phase rational simplex with Bland's rule; its
Farkas extraction is itself validated on 400 random systems, every
certificate verified exactly).  Round 21 origin: `research/matrix_shapes.py`
section 2.

Asserted: pointwise validity of the Kounias and chain cuts (exhaustive over
all 0/1 patterns, <= 6 events); the exact hit-count bounds; IP width = F - 1
at machines 7..19 (period sieve); the uniform level-1 certificate
(Fractions); the pair-visibility theorem exhaustively over
(machine, width, pair); the exact feasibility of the primal at W*-1 and the
exact Farkas certificate at W* for every threshold in results 1-3; F - 1 < W*
wherever F is known; the (D) verdict at every step.

SOLVER-labeled: the DISCOVERY of candidate thresholds (scipy HiGHS).  Every
discovered endpoint is then bracketed by two exact rational certificates, and
the run ABORTS if the exact answer disagrees with the float discovery.

Validity chain for a bound: exact IP (CRT) => any valid relaxation's
infeasibility at W => no covered window of width W => F(M) <= W.  The
level-2 relaxation drops pair-marginal consistency (a further weakening, so
certificates remain valid).  The pruning used for speed is separately safe:
a feasible point of the pruned LP pads with zeros to a feasible point of the
full LP, and every certificate is re-verified against the full column set.

Kernel-checkable candidates: the closed-form corollary at a fixed machine
(one line of integer arithmetic per W); a fixed Farkas certificate (finite
list of rationals + finitely many maxima over finite phase sets, all
`decide`-able), specifically `F 19 <= 37` and hence `D_17_19 : F 19 <= F 17 + 19`.

## 4. IMPLICATIONS

- Fills a named gap from the round-19 summary: "the UPPER bounds on F_j that
  every prefix row currently lacks" - this is an upper-bound MECHANISM for F.
- Gives the formalist a new certificate species: rational Farkas lists.  At
  machine 19 the object is 37 rationals and ~1,480 comparisons versus a
  1,616,615-slot scan, and it proves a (D) step end to end given only
  F(17) = 18.
- Quantifies HOW MUCH correlation information each moment degree carries, and
  (via `moment-degree-ceiling.md`) shows the required degree grows - which is
  the LP-side answer to round 22's arity question.

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

- Jacobsthal upper bounds: the mechanism produces certified h-type upper
  bounds for arbitrary gear sets (nothing here is twin-specific).
- (D): proved at 7->11 and 17->19 by certificate; 11->13 misses by ONE, which
  makes "is there a degree-2 certificate of width 20 at machine 13?" a sharp,
  finite, answerable question (the answer is no - W* = 21 is exact - so any
  improvement must come from degree 3, which the ceiling says is available
  until machine ~151).
- Level-3 certificates: the ceiling says degree 3 survives to at least
  machine 151, but no level-3 LP THRESHOLD has been computed.  That is the
  named next construct: it is the only version of this vehicle that could
  reach the 19->23 and 23->29 steps.

## 6. PRIOR-ART CHECK

Done 2026-08-24.  Ten searches, listed in full in
`moment-degree-ceiling.md` section 6 (they cover both entries), plus the full
text of Costello-Watts arXiv:1208.5342 fetched and read.

Nearest published results.
- Costello-Watts, "A computational upper bound on Jacobsthal's function"
  (arXiv:1208.5342, 2012): computes upper bounds on h(k) from a recursive
  counting bound with a pairwise correction term
  (sum sum phi_min(floor(r/(p_i p_j)), i-1)) and an E-term for residue
  co-occurrence.  This is the SAME SPECIES as the closed-form corollary
  (result 5) and is STRONGER, because it recurses.  Result 5 must therefore
  be presented as a rediscovery of a weaker case, not as new.  Costello-Watts
  contains no LP, no dual object, and no ceiling analysis.
- Kanold, Stevens, Iwaniec: asymptotic upper bounds on the Jacobsthal
  function; none is a per-machine certificate.
- Hagedorn; Ziller-Morack; Holt-Rudd: exact computation of h(n); searches,
  not certificates.
- Kounias (1968), Hunter (1976), Worsley: the degree-2 cut family, standard.
  For weights 4/(q_i q_j) the maximum spanning tree is the star at the
  smallest gear, so Hunter-Worsley collapses to Kounias with k = 5 here.
- Prekopa, Boros-Prekopa, "Boole-Bonferroni Inequalities and Linear
  Programming" (Oper. Res. 36, 1988): sharp bounds on P(union) from binomial
  moments as an LP with dual feasible bases - the machinery of the ceiling
  test.  Classical.
- Set-cover LP relaxation and integrality-gap theory (Lovasz, Chvatal,
  Vazirani): the general frame.  Classical.
- Hough; Balister-Bollobas-Morris-Sahasrabudhe-Tiba "distortion method" for
  Erdos covering systems: density-of-uncovered-set arguments over a product
  measure - degree-1 arguments in this language.  Different problem, same
  first-moment core.

VERDICT: PARTIAL OVERLAP.  The closed-form corollary (result 5) is a weaker
case of Costello-Watts and is NOT new.  The LP-dual certificate form for
Jacobsthal-type maximal gaps, the pair-visibility reduction q_a q_b > 4W, and
the use of a certificate to prove a (D) merge step were not found in any
source searched.  NOT independently confirmed.
