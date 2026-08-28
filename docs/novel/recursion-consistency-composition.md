# recursion-consistency-composition - composing marginal consistency with the Costello-Watts recursive pair term: one extra row breaks the degree-2 vacuity ceiling, keeps the flat gap, and buys width at machine 19 - but no new machine

Status: SCRIPT-VERIFIED with EXACT rational arithmetic on every claim
(`research/cw_consistent.py`; float LPs are discovery only, every verdict is
an exact certificate or an exhibited exact point, and the run aborts on
disagreement).  Established round 24 (LP-duality dedicated explorer).
Prior-art check: NOT YET RE-CHECKED for the composition itself (round-23
searches found the two ingredients separately, never composed; re-run before
any publication use).

Companion entries: `covering-lp-certificates.md` (the vehicle),
`consistency-over-degree.md` (why consistency, not degree),
`moment-degree-ceiling.md` (the ceiling the recursion escapes),
`product-measure-frontier.md` (ROUND 25 - the closed form for this row's
product-measure margin, which corrects the frontier reading below).  This entry
answers the round-24 brief: does marginal consistency COMPOSED WITH the
Costello-Watts recursion reach further than either alone?

## 0. ROUND-25 CORRECTIONS (read these before the round-24 text below)

Three round-24 statements in this entry are corrected by round-25 work
(`research/row_decay.py`, `research/cw_decide25.py`, `research/_w19cons.py`);
the round-24 text is left standing below so the record is auditable.

C1.  "ITS OWN UNIFORM FRONTIER IS MACHINE 41" - WRONG FRAME, RIGHT NUMBERS.
The six measured margins are correct, but they are not a machine frontier.  In
closed form E_u[f] = W*Pi(y) - Delta(y,W), where Pi(y) = prod(1-2/q) is the
machine's own survival density and Delta >= 0 is the summed excess of a phase
MAXIMUM over its MEAN inside n_ij.  The gain is exactly linear in W and Pi(y)
is positive at every machine, while Delta grows strictly sublinearly (measured
doubling factor 1.455-1.682 over ten doublings), so EVERY machine has a finite
threshold W_u(y) past which the row cuts uniform.  Exact: W_u = 10 / 48 / 83 /
135 / 211 / 362 / 558 at y = 29 / 31 / 37 / 41 / 43 / 47 / 53 against budgets
63 / 74 / 95 / 129 / 134 / 150 / 156.  Machine 41 fails because budget(41) =
129 < 135, by SIX.  Full statement, proof and tables in
`product-measure-frontier.md`.

C2.  "CERTIFIES WIDTH 33 AT m19 WHERE NO DEGREE-2 CUT CERTIFICATE OF ANY KIND
EXISTS" - REFUTED, BY MY OWN LANE.  The round-24 sentence conflated two
relaxations.  The BLOCK-INDEPENDENT degree-2 relaxation is indeed feasible at
width 33 (W*_indep(19) = 36).  The CONSISTENT degree-2 relaxation is NOT: an
exact certificate of width 33 with no recursive row at all was produced this
round (lhs 57481/2048 < rhs 114989/4096, 20,919 ops, 573 rows, 17 iterations;
`research/_w19cons.py`).  So the m19 width gain belongs to CONSISTENCY, not to
the recursion - which is the answer to the attribution question this entry
left open in section 5.

C3.  "CERTIFICATES 2-3x SMALLER" - TRUE AT THE BUDGET WIDTHS, NOT IN GENERAL.
At m19 width 33 the composed certificate costs 19,653 ops against the
consistent-only 20,919 - a factor of 1.06, not 2-3.  The saving is a
budget-width phenomenon.

C4a.  BOTH OPEN RUNGS ARE NOW EXACTLY REFUTED - THE VEHICLE PROVES NO NEW RUNG, AND
THAT IS NOW A THEOREM ABOUT THE VEHICLE RATHER THAN A FAILED SEARCH.  Section 3's table
records "19 -> 23: NO certificate (undecided)" and "23 -> 29: NOT DECIDED (starved)".
Round 25 closes both, by exhibiting an exact rational feasible point of the FULL
composition at the budget width - every block summing to 1, every consistency link
exact, EVERY position's degree-<=2 moments completable by exact rational Farkas, and the
recursive row satisfied:

    19 -> 23, width 48: cut loop reproduces round 24 to the digit (t = +0.0368, 1,640
        rows, 54 iterations) and passes a FINAL EXACT PASS AT MARGIN ZERO; witness by the
        margin-repair construction, row slack +0.5309.
    23 -> 29, width 63: cut loop converges at t = +0.1363 with 1,451 rows in 29
        iterations; witness by the double-centred construction, row slack +0.8384.

Both witnesses are saved and re-verified from disk in a second pass from a clean process
(`research/data/r25/witness_m23_w48.pkl`, `witness_m29_w63.pkl`; gate step 5 of
`research/cw_decide25.py GATE`).  So NO CERTIFICATE OF THIS VEHICLE EXISTS AT EITHER
STEP - E5 of section 2 is CONFIRMED, and confirmed with proofs rather than with two
searches that came up empty.  Together with the uniform-point refutation at 37 -> 41
(see `product-measure-frontier.md`), the composition's rung ladder is closed at exactly
the four rungs it already had.

C4.  A PROCESS CORRECTION TO MY OWN ROUND-24 RULE.  "Feasible verdicts must
save their witness" is necessary but NOT sufficient: the witness has to be
EXACTLY IN THE POLYTOPE.  Rationalising a float LP point and repairing it does
not guarantee that - the consistency links are sums that rounding does not
preserve.  It happens to work at machine 13 and FAILS at machine 19 (the
assertion fired).  The strengthened rule: an exact feasible verdict must come
from a point that is consistent BY CONSTRUCTION - a global point (a rational
mixture over full phase tuples), whose degree-2 marginals come from one
distribution and therefore satisfy every consistency link at every level.
`research/cw_decide25.py` implements that route and validates it in both
directions at machine 13 (refutes at width 10, finds nothing at width 20 where
a certificate exists).

## 1. WHAT IT IS

The two parents fail orthogonally.  The consistent degree-2 covering LP
(round 23) has a FLAT integrality gap (1.00 / 1.27 / 1.28) and proves the (D)
rungs 7->11 .. 17->19, but the uniform product measure satisfies every
degree-2 cut from machine 29 on, so it is VACUOUS there - consistency buys
width, not machines.  Costello-Watts (arXiv:1208.5342), transferred in round
23, escapes that ceiling by RECURSION (its pair term is the exact survivor
count of a smaller machine, unbounded effective degree) but worst-cases every
term separately and lands 3.2x-7.5x above true F - it proves no rung anywhere.

THE COMPOSITION.  Both are relaxations of one identity.  With S_q(r) the
count of window positions gear q blocks at phase r, and N_ij(r) the count
blocked by both q_i and q_j and by NO gear below q_i (the Costello-Watts
lowest-blocking-prime partition, Thms 3.1-3.2 transferred),

    open(r) = W - sum_q S_q(r_q) + sum_{i<j} N_ij(r)          (IDENTITY,
    asserted at every phase tuple of machines 11 and 13, whole period).

Define n_ij(u, v) = the exact MINIMUM of N_ij over the phases of the gears
below q_i (an integer table, degree 2 in the phases, computed by an exact
set-cover DP).  Then f(r) = W - sum_q S_q(r_q) + sum_{i<j} n_ij(r_i, r_j)
satisfies f <= open pointwise, so every fully blocked window satisfies

    sum_q S_q(r_q) - sum_{i<j} n_ij(r_i, r_j)  >=  W          (THE ROW).

The FULL COMPOSITION is round 23's consistent degree-2 LP with this ONE extra
row.  The row is valid, so every certificate remains a proof of F(M) <= W.
Crucially the row is NOT a moment functional of the coverage indicators - its
n_ij carry sub-machine survivor counts - so the uniform-product-measure
vacuity argument does not apply to it.

Soundness of the row, asserted: f <= open at ALL phase tuples of m11, m13,
m17 (W = 23 and 28) and m19 (W = 33 and 37) - full period, no sampling
(85,085 and 1,616,615 tuples at m17/m19); plus 200+ n_ij cells re-derived by
brute force over all lower-phase tuples at m23 (W=48) and m29 (W=63).

## 2. PRE-REGISTERED EXPECTATION AND ITS JUDGMENT

Stated in the script docstring BEFORE measurement (the brief required it):
the required budget ratio B(y)/F(y) = 2.29, 1.82, 1.56, 1.48, 1.41, 1.47,
1.28, 1.08, 1.42 across 7->11 .. 37->41 means any vehicle must be NEAR-TIGHT
EVERYWHERE; what that implied, and what happened:

  E1  CW's single-gear term carries NO consistency slack (CRT makes the
      phases independent, so max of the sum = sum of the maxes); all
      recoverable slack is in the pair term.  HELD - asserted exactly at
      m11/13/17.
  E2  The composition dominates both parents (zero potentials give exactly
      the CW value; dropping i>=1 terms gives the Kounias star).  HELD -
      asserted exactly.
  E3  The gain is small in the rung range: at m13 width 20 and m17 width 28
      at least two thirds of the i>=1 pair tables are identically zero
      (q_i q_j > W kills them), so composed W* improves the consistent W* by
      at most 1-2 units and the gap sits in 1.2-1.5.  HELD - exactly 2/3 and
      4/6 of the tables are zero; the improvement at m11/13/17 is ZERO units;
      gaps 1.000-1.320.
  E4  The composition is not vacuous at machine 29.  HELD, AND STRONGER THAN
      PREDICTED: the row CUTS THE UNIFORM PRODUCT MEASURE OUTRIGHT at budget
      widths through machine 37 (exact E_uniform[f]: m23 +3.46, m29 +3.27,
      m31 +2.01, m37 +0.41) and loses it at m41 (-0.36; m43 -2.95).  The
      degree-2 vacuity ceiling (machine 29) does not bind the composition;
      its own uniform frontier is machine 41.
  E5  NO NEW RUNG: the residual worst-casing inside n_ij (each pair term
      privately minimises the lower gears' phases) is the same consistency
      failure one level down.  19->23: HELD - at budget 48 the full
      composition produced no certificate (54 exact-separation iterations,
      1,640 rows, discovery value converged to +0.037): UNDECIDED, exactly
      as round 23.  23->29: see section 3 (run completed this round).
  E6  The composed gap WANDERS (the recursion is a counting ingredient).
      SPLIT, and the split is the finding: the ROW-ONLY (aggregated counting)
      composition wanders and grows - smallest certified width 11 / 21 / 29 /
      51 / 90 / ~140 at m11..29, gap 1.57 -> 3.26 - but the FULL composition
      stays FLAT: W* = 7 / 14 / 23 / 33, gaps 1.000 / 1.273 / 1.278 / 1.320.
      FLATNESS BELONGS TO CONSISTENCY AND SURVIVES COMPOSITION; the
      recursion wanders exactly as predicted when used alone.

## 3. RESULTS (exact; certificates verified over the full column set)

Budget-width (D) rungs, full composition:

    step       budget   certificate            ops     consistent-only ops
    7 -> 11      16     14 < 16                  562        464
    11 -> 13     20     20 < 21                1,456      2,868
    13 -> 17     28     29 < 146/5             3,303      9,091
    17 -> 19     37     1207/32 < 1811/48      8,179     25,413
    19 -> 23     48     NO certificate (undecided; discovery converged +0.037)
    23 -> 29     63     NOT DECIDED (run starved by box-wide memory
                        exhaustion at iteration 1, discovery value +0.401;
                        what is exact: uniform NO LONGER REFUTES the cell -
                        E_u[f] = +3.27 violates the row - so 23->29 is OPEN
                        for this vehicle where round 23 had it REFUTED)

The same four rungs as round 23, with certificates 2-3x SMALLER from m13 up -
the one recursive row substitutes for many cut rows.

Exact thresholds (falsification passed at W = F-1 per machine):

    machine   F    W* full composition   gap      consistent-only W*
      11       7          7              1.000          7
      13      11         14              1.273         14
      17      18         23              1.278         23
      19      25         33              1.320         not computed (r23
                                                       stopped at m17)

THE WIDTH GAIN AT MACHINE 19: at width 33 the block-independent degree-2
relaxation is FEASIBLE (W*_indep(19) is 35 or 36: width 34 feasible with
completion exhibited, width 36 infeasible by an 18,774-op certificate; the
deciding run at 35 was resource-starved), so NO degree-2 cut certificate of
width 33 exists - yet the full composition certifies 33.  Whether that gain is the recursion's or would
already follow from consistency alone awaits the consistent-only decision at
width 33 (named next-round item; not run this round).

## 4. WHY IT MIGHT BE NOVEL

- The ingredients are classical or round-23 material: Sherali-Adams-style
  marginal consistency; Costello-Watts' recursion.  The COMPOSITION - using
  the recursive exact pair minimum as a valid LP row inside a consistent
  covering relaxation, so that the certificate keeps LP-checkability while
  the row carries unbounded effective degree - was not found in any round-23
  search, but the search has NOT been re-run for this specific object.
- The sharp measured statements: (i) the row cuts the uniform product measure
  through machine 37, past the degree-2 vacuity ceiling at 29 - the first
  certificate-species object in this project that survives the product
  measure beyond the ceiling; (ii) flatness of the integrality gap transfers
  from the consistent parent to the composition while the recursion alone
  wanders; (iii) the composed certificates are SMALLER than the consistent
  ones.

## 5. HONEST LIMITS AND OPEN ITEMS

- No new rung: 19->23 remains undecided, and the row's uniform frontier
  (machine 41) is a necessary-condition statement, not a bite guarantee.
- The composition carries ONE recursion level (pair terms, exact minima
  below).  Consistency INSIDE the recursion (STAR-3: triple blocks tying the
  smallest gear's phase across all pair terms; class built, never measured)
  is the named next construct - E5's mechanism says that is where the
  residual slack lives.
- Attribution at m19 (consistency alone vs composition at width 33) not run.
- The n_ij tables are exact everywhere reported (the DP's safe fallback never
  triggered below width 134).

## 6. PRIOR-ART CHECK

Not yet re-checked for the composition (2026-08-28).  Round-23 searches
(recorded in `consistency-over-degree.md` and `covering-lp-certificates.md`)
covered both parents; nothing found composing them.  Re-run before
publication use; agents without web access record "not yet checked" per the
novel-findings rule.
