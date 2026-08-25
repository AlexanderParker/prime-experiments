# covering-lp-certificates - scan-free upper bounds on F(M) by LP duality over the covering IP

Status: **KERNEL-CHECKED at machine 19 since round 23** (`proofs/CoveringCert.lean`,
`CoveringCert.F19_le_37` and `CoveringCert.D_17_19_lp`: `F(19) <= 37 = F(17) + 19`,
axiom footprint the standard three, `cert_signs` on NO axioms), and
SCRIPT-VERIFIED elsewhere, EXACT ON BOTH SIDES of every
reported threshold (`research/lp_dual_certs.py` + the exact rational LP core
`research/exact_lp.py`; the round-21 origin is `research/matrix_shapes.py`
SHAPE 2).  Round 21 verified only the INFEASIBLE endpoint; round 22 added an
exact rational feasible point at W*-1, so the thresholds below are now
bracketed by two exact certificates and are not solver claims.  Established
round 21 (human matrix directive: "try different matrix shapes and
operations"), pushed to depth round 22 (LP-duality dedicated explorer).
Prior-art check: DONE 2026-08-24 - PARTIAL OVERLAP, section 6.

Companion entries: `moment-degree-ceiling.md` is the obstruction half (how far
this vehicle can possibly go), and `consistency-over-degree.md` is the round-23
finding that changes which axis the vehicle should be pushed along.  Read all
three; round 23 amends results 3, 4, 5 and 6 below.

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

3. (D) STEPS PROVED BY A DUAL CERTIFICATE (round 22, the finding; round 23,
   the range).  A certificate at machine M+q' of width W = F(M) + q' proves
   the (D) step F(M+q') <= F(M) + q' outright, with no period of M+q' built:

       step    budget F(M)+q'   LP2 W*   round-22 verdict
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

   ROUND 23 - THE TWO MISSES ARE BOTH CLOSED, AND NOT BY A SHARPER CUT.  The
   relaxation above (round 22's, and the classical Bonferroni/Kounias shape)
   drops MARGINAL CONSISTENCY: the pair block (a,b) may choose its phase-pair
   distribution freely, with no requirement that its marginal on gear a equal
   gear a's own phase distribution.  Restoring that, at the SAME degree 2,
   turns both misses into certificates:

       step    budget   consistent degree-2 certificate
        7->11     16    9 < 10                            (D) PROVED
       11->13     20    660/37 < 664/37                   (D) PROVED - new
       13->17     28    2533/96 < 5081/192                (D) PROVED - new
       17->19     37    258513/8192 < 64637/2048          (D) PROVED

   Four consecutive (D) rungs, the same four the kernel-proven ladder has, by
   a method that shares nothing with the merge law.  More DEGREE does not do
   this: at machine 13, width 20, the block-independent relaxation is feasible
   at degree 2, 3 and 4 (degree 4 = all the gears), each verdict an exact
   point completable at every position.  Full account, mechanism and rung
   table in `consistency-over-degree.md`.

4. ZERO-COLUMN REDUCTION / PAIR VISIBILITY (round 22 - and see the round-23
   caveat at the end of this item, which changes what it MEANS).
   Pair variables enter
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

   ROUND-23 CAVEAT (self-correction).  The theorem is true exactly as stated,
   but the READING round 22 gave it was wrong.  Pair invisibility is an
   ARTEFACT OF THE MISSING MARGINAL CONSISTENCY, not a property of the
   machine: a pair can only send all its mass to a zero-overlap phase pair
   because nothing ties its marginals to the single blocks.  Under
   consistency no pair can leave the LP, and that is precisely what closes
   the 11->13 and 13->17 rungs.  Read result 4 as a statement about round
   22's relaxation, never as "pair correlations at this scale are not in the
   LP's field of view".

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

**KERNEL CHECK LANDED (round 23, formalist), `proofs/CoveringCert.lean`.** The
machine-19 certificate is in the Lean kernel, zero sorries, no `native_decide`.
Three facts about the object emerged from formalising it:

1. THE CERTIFICATE IS SUPPORTED ON A SINGLE DISTINGUISHED GEAR. All 37 nonzero
   dual weights sit on rows `(i, 5)` - the Kounias cut is used with `k = 5` at
   every position, none other. That collapses the pair blocks to the five pairs
   `(5, q)`, so the whole certificate is 37 weights, 6 single-gear maxima and 5
   phase-pair minima. (The LP's row set has 222 rows and 7 surviving pairs; the
   optimum uses 37 rows and 5 pairs. `(7,11)` and `(7,13)` survive the
   visibility test but get weight 0.)
2. IT IS A PALINDROME - `y_i = y_{36-i}` exactly - which is the machine's
   mirror symmetry `k -> -k` appearing in the dual.
3. SCALED TO INTEGERS the certificate is denominator 1101 and reads
   `sum_q max_r S_q(r) = 12489  <  9757 + 2749 = sum y + sum_j min P_(5,j)`,
   margin 17 out of 12489 (0.14%). `cert_signs` depends on NO AXIOMS.

**ROUND 23, SECOND KERNEL CHECK: THE CONSISTENT FORM AT 11->13 AND 13->17**
(`proofs/CoveringCert2.lean`, `D_11_13_lp` / `D_13_17_lp` / `lp_ladder`). The
consistency the round-23 analysis identified as missing does NOT need dual
multipliers to be checked. In the aggregated form used here the inconsistency is
visible in one line: bounding `max_r S_5(r)` and `min_(r5,rj) P_j(r5,rj)`
separately lets gear 5 use two different phases. Keeping the phases under ONE
quantifier - the quantity is literally `sum_i y_i * Kounias_i` - gives

    sum y  <  max over PHASE TUPLES of [ S_5(r5) + sum_j (S_j(rj) - P_j(r5,rj)) ]

which is the `k = 5` STAR case of marginal consistency, available precisely
because the optimum is supported on one distinguished gear (fact 1 above). It is
strictly weaker than full marginal consistency and still closes both rungs, with
certificates an order of magnitude smaller than the full system's dual:

    rung      width  weights                    sum   max over tuples  margin
    11 -> 13    20   20 integers, EIGHTEEN 1s    22          21           1
    13 -> 17    28   28 integers, all in [2,5]   94          92           2
    17 -> 19    37   37 integers (round 22)    9757        9740          17

(against 106 integers over denominator 37 and 2,868 rational operations for the
full consistent dual at 11->13). Both new vectors are palindromes again. So the
vehicle now proves THREE CONSECUTIVE (D) RUNGS in the kernel, sharing nothing
with the merge law. Being weaker than full consistency, the star form inherits
19->23's undecided status and does not extend the range.

Kernel cost: the 72 single-phase and 335 phase-pair evaluations of a 37-term
sum are 11 `decide +kernel` declarations that elaborate in seconds - against a
1,616,615-slot period scan that costs hours. That ratio, not the strength of
the bound, is the finding: `F(19) <= 37` is weaker than the scan's exact
`F(19) = 25`, but it is the FIRST upper bound on a Jacobsthal-type maximal gap
in this development that a kernel can check without enumerating the period.

ROUND-23 UPDATE TO THE KERNEL SHAPE.  The consistent certificates are better
shaped than round 22's: their weights snap to ONE COMMON DENOMINATOR, so the
Lean object is a list of INTEGERS plus a denominator, and the check has the
same form with one extra ingredient - a consistency potential nu, one rational
per (block, sub-tuple), which shifts each column's weight before the per-block
maximum is taken:

    a_j = sum_r y_r lam^r_{S(j)} [i_r in O_j]
          + sum_{links with j among the extensions} nu
          - sum_{links with j the restricted tuple} nu
    theorem cert : sum_S (max over the phase tuples of S of a_j)
                 < sum_r y_r * (1 - lam^r_0)          := by decide

Smallest instance: machine 11, 26 integers, 464 rational operations, proving
F(11) <= 16 = F(7) + 11.  Machine 13: 106 integers over denominator 37, 2,868
operations, proving the rung the (D) ladder records as TIGHT (margin 0).

BUT NOTE THE COST, because it changes which certificate to formalise.  The
consistent LP's columns are full phase tuples, so its certificates are much
larger: 464 / 2,868 / 9,091 / 25,413 operations at machines 11 / 13 / 17 / 19,
against period scans of 385 / 5,005 / 85,085 / 1,616,615 slots - ratios
0.8x, 1.7x, 9.4x, 63.6x.  Round 22's consistency-free certificate at machine
19 is 1,480 operations (1,092x), i.e. SEVENTEEN TIMES SMALLER at the one
machine where both forms work.  Formalise the round-22 object at 17 -> 19 and
the consistent object only at 11 -> 13 and 13 -> 17, where nothing cheaper
exists.

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
- (D): proved at 7->11, 11->13, 13->17 and 17->19 (round 23; the last three
  need the consistent form).  19->23 is the first rung where neither a
  certificate nor an exact negative was obtained.
- SUPERSEDED, and worth recording as a wrong prediction: round 22 named
  "build the sharp degree-3 cut" as the next construct, "the only version of
  this vehicle that could reach 19->23 and 23->29".  Round 23 measured degree
  3 AND degree 4 at machine 13 and both fail where consistency succeeds, so
  degree was the wrong axis.  The live next construct is MORE CONSISTENCY
  (Sherali-Adams level 2), which costs columns, not moment degree - and,
  separately, the recursive pair term of Costello-Watts (section 6), which
  is the only ingredient found so far that escapes the degree ceiling
  altogether.

## 6. PRIOR-ART CHECK

Done 2026-08-24, RE-RUN AND EXTENDED 2026-08-25 (prior-art checks expire).
Ten searches listed in `moment-degree-ceiling.md` section 6, three more listed
in `consistency-over-degree.md` section 6, and - new in round 23 - the FULL
LaTeX SOURCE of Costello-Watts arXiv:1208.5342 downloaded and read line by
line, not the abstract-level content round 22 used.

### COSTELLO-WATTS, READ IN FULL (round 23)

What the paper contains, in their notation (p_i the i-th prime, P_k their
product, phi(b,m,k) the count of integers in (b, b+m] coprime to P_k,
phi_min(m,k) its minimum over b, h(k) the least m with phi_min > 0):

  Thm 3.1  phi(b,m,k) = m - sum_i F_{b,m}(p_i) + sum_{a: w_k(a)>0}(w_k(a)-1).
           Bookkeeping: the first-moment count undercounts by the
           multiplicity excess.  An IDENTITY.
  Thm 3.2  Partition the blocked a by their LOWEST blocking prime p_x; then
           sum over that class of (w_k(a)-1) = sum_{i>x} F_S(p_i p_x).
           Purely combinatorial, no arithmetic.
  Thm 2.1  THE DILATION LEMMA, and the real engine.  For coprime squarefree
           d and n, the arithmetic progression b+d, b+2d, ... has the SAME
           gcd-with-n pattern as a run of CONSECUTIVE integers cb+1, cb+2,
           ... where cd = 1 (mod n).
  Thm 3.3/3.4  Combining: an EXACT recursion
           phi(b,m,k) = m - sum_i F(p_i) + sum_{j>=2} F(2 p_j)
                        + sum_{2<=i<j<=k} phi(c_b(p_i p_j), F(p_i p_j), i-1).
           The pair term is the SAME FUNCTION at a smaller machine - not a
           truncated second moment.
  Thm 4.2/4.3/4.4  The computable version: worst-case every term over b, plus
           E, an integer correction counting the primes for which the two
           worst cases F(p) = ceil(r/p) and F(2p) = floor(r/2p) CANNOT
           CO-OCCUR.  E is at most k-1 and is the paper's only genuinely
           arithmetic (as opposed to combinatorial) ingredient.
  Algorithms 1-3  recursion bottoming at k <= 6 with exact phi_min tables for
           all m < P_6, plus the stopping rules r < 2 p_{k-1} and (optionally)
           Hagedorn's h(k) for k <= 49.  Results: b(k) < 3 h(k) for k <= 49;
           b(k) <= 0.27749612254 k^2 log k for k <= 10^4.

WHAT IT GIVES US THAT WE DID NOT HAVE.
1. AN ESCAPE FROM OUR OWN CEILING.  `moment-degree-ceiling.md` proves that any
   relaxation keeping only l-gear joint information dies at a computable
   machine.  Costello-Watts is not of that form: its effective degree is
   unbounded (its leaves are exact 6-prime values and the nesting composes
   them), so the ceiling does not bind it.  That identifies the shape of the
   escape hatch: SELF-SIMILARITY, not higher moments.
2. A SELF-SIMILARITY LAW FOR OUR MACHINE.  Transferred (derivation and brute-
   force assertions in `research/cw_transfer.py`): the slots blocked by BOTH
   gear q_i and gear q_j are four arithmetic progressions mod d = q_i q_j, and
   under t |-> (a - c)/d each of them, seen by the gears below q_i, is again a
   TWO-TEETH MACHINE - gear q keeps a symmetric tooth pair {s_q +- v_q} with
   half-width v_q = (6 q_i q_j)^{-1} mod q determined and centre s_q free.  So
   the twin machine is self-similar under "restrict to a pair modulus", at the
   cost of a different tooth separation.  Asserted term by term.
3. THE E-TERM IDEA: two worst cases that cannot be attained simultaneously,
   worth one unit each.  That is exactly the species of argument our tolerance
   route keeps needing.

WHAT IT DOES NOT GIVE US.  Measured, not asserted (`research/cw_transfer.py`,
end-to-end soundness checked against brute force over whole periods): the
transferred bound proves F(13) <= 35, F(17) <= 65, F(19) <= 110, F(23) <= 230,
F(29) <= 322 - equal to result 5 at machines 13/17/19 and better at 23
(230 vs 285), but 3.2x to 7.5x above the true F.  Since a (D) rung needs a
ratio tending to 1 (section 7), the Costello-Watts family CANNOT prove a merge
step at any machine, while the dual certificate proves four.  So:
  * result 5 (the closed-form corollary) IS a weaker case of Costello-Watts -
    precisely, it is their double sum truncated at recursion depth 0 and
    restricted to the pairs incident to one gear.  The round-22 downgrade
    STANDS and is now derived rather than inferred from an abstract.
  * the DUAL CERTIFICATE is not superseded by them: it is a different object
    (a checkable list of rationals, not a computation) and it is an order of
    magnitude sharper on this problem.
  * their correction term does not strengthen our certificate directly.  It
    does name the next construct: replace the LP's crude pair term by the
    recursive exact one - column generation by self-similarity.

Other nearest published results.
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
