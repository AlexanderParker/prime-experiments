# consistency-over-degree - in the covering LP for F(M), marginal consistency is worth more than any amount of extra moment degree, and it is what proves the (D) rungs

Status: SCRIPT-VERIFIED with EXACT rational arithmetic on both sides
(`research/lp_degree_range.py`, sections G/M/R; exact LP core
`research/exact_lp.py`).  Every "proved" verdict is an exact Farkas-type
certificate verified by direct evaluation against the FULL column set; every
"no certificate exists" verdict is an exact rational point whose degree-<=l
moments are completable at every position, so not one degree-l cut of any kind
is violated.  Established round 23 (LP-duality dedicated explorer).
Prior-art check: PARTIAL OVERLAP, done 2026-08-25, section 6.

Companion entries: `covering-lp-certificates.md` (the vehicle) and
`moment-degree-ceiling.md` (its vacuity ceiling).  This entry says which of
the two limits is the operative one - and it is neither of the ones round 22
expected.

## 1. WHAT IT IS

Plain language.  F(M) - 1 is the widest window of slots that can be fully
blocked, and "is a window of width W blockable?" is an integer program whose
LP relaxations give scan-free certificates.  A relaxation of DEGREE l keeps,
per l-subset of gears, a joint distribution over those gears' phases.  Round
22 built the degree-2 version, and it missed the (D) rung at 11 -> 13 by
EXACTLY ONE unit (it proved F(13) <= 21 where 20 was needed).  A one-unit miss
invites the obvious question: does a sharper cut - more degree - close it?

THE ANSWER IS NO, AND THE REASON IS THE FINDING.  At machine 13 the
block-independent relaxation is feasible at width 20 at degree 2, degree 3
AND degree 4 - and degree 4 is the total number of gears, i.e. the complete
per-position joint information.  No cut of any degree closes a one-unit gap.

What closes it is a constraint that carries no extra degree at all: MARGINAL
CONSISTENCY.  Round 22's LP (and the standard Bonferroni/Kounias setup it
came from) lets the pair block (a,b) choose its distribution over phase pairs
FREELY, with no requirement that its marginal on gear a equal gear a's own
phase distribution.  Restoring that requirement - and changing nothing else,
staying at degree 2 - makes width 20 infeasible, with an exact certificate.

    F(13) <= 20 = F(11) + 13      (D) AT 11 -> 13, PROVED

Certificate: 106 nonzero weights over a single denominator (37), verified by
2,868 rational operations.

WHAT CONSISTENCY COSTS, stated plainly because it is not small.  The
consistent LP's columns are full phase tuples, so its certificates are much
bigger than round 22's:

    machine   certificate ops   period slots   ratio
      11             464             385        0.8x   (worse than scanning)
      13           2,868           5,005        1.7x
      17           9,091          85,085        9.4x
      19          25,413       1,616,615       63.6x
      19 (round 22, no consistency)
                   1,480       1,616,615     1,092x

So at 17 -> 19, where BOTH forms work, round 22's consistency-free certificate
is the better object by a factor of 17 and should be the one formalised.  The
consistent form is what to use exactly where the cheap one fails - 11 -> 13
and 13 -> 17 - and there its advantage over a period scan is 1.7x and 9.4x, not
three orders of magnitude.  The real argument for it is not speed: it is that
it needs no period of the new machine at all, which is what makes the vehicle
scale in principle.

### The precise statements

Gears q in [5, y], gear q blocks slot k iff k = +- 6^{-1} (mod q).  By CRT,
choosing a window position IS choosing one phase per gear independently, so
max blockable width = F(M) - 1 EXACTLY.

BLOCK-INDEPENDENT degree-l relaxation (round 22's shape, generalised).  One
distribution z_S per gear subset S, |S| <= l, over the distinct sets of window
positions that all of S block simultaneously.  Row per (position i, cut lam):
sum_S lam_S m_S(i) >= 1 - lam_0, for every polynomial cut lam of degree <= l
with lam(x) >= 1 at every nonempty atom x.

CONSISTENT degree-l relaxation.  The same, but the columns are genuine PHASE
TUPLES and every block of size k is forced to agree with each of its
(k-1)-subsets:  sum over extensions of z_S(r) = z_{S'}(r') .

Both are valid relaxations (a blocked window gives an integral feasible point
of each), so infeasibility at width W certifies F(M) <= W in both.

### The exact integrality gaps, and how much consistency is worth

W* is the smallest width at which the relaxation is infeasible; the
integrality gap is W*/F.  Bisection is legitimate because infeasibility is
monotone in W (a feasible point at W' restricts to one at every W <= W').
Both endpoints exact at every entry.

    machine   F    W* block-independent   W* consistent   gap: indep -> cons
      11       7            8                    7         1.143 -> 1.000
      13      11           21                   14         1.909 -> 1.273
      17      18           30 (r24; was 31)     23         1.667 -> 1.278
      19      25           36 (r24 gate-check)  <= 33      1.440 -> <= 1.320

ROUND-25 ADDITION (the machine-19 cell, blank since round 23).  The consistent
degree-2 relaxation is INFEASIBLE at width 33 at machine 19, by an exact
certificate with NO recursive row: lhs 57481/2048 < rhs 114989/4096, 20,919
ops, 573 rows, 17 cut iterations (`research/_w19cons.py`, using the exact
decider in `research/cw_decide25.py`).  So W*_cons(19) <= 33 while
W*_indep(19) = 36 (settled exactly by the round-24 manager gate-check): the
consistency gain at machine 19 is at least 3 units of width, and the flat
consistent gap extends to a fourth machine at <= 1.320.  The exact value of
W*_cons(19) is NOT pinned - a bisection below 33 needs an exact FEASIBLE
verdict at the trial width, and that turned out to be harder than it looks:
rationalising the LP's own point does not reliably land exactly in the
consistent polytope (the consistency links are sums that rounding does not
preserve - it works at machine 13 and the assertion FIRED at machine 19).  The
sound route is a global point, consistent by construction; see the round-25
correction C4 in `recursion-consistency-composition.md`.  Recorded as an open
cell rather than guessed.

ROUND-24 CORRECTION (the assertion gate caught a round-23 error).  Round 23's
section-G regression claim - "the adaptive machinery reproduces round 22's
Kounias thresholds 8/21/31/37; Kounias was already family-optimal at degree
2" - FAILED its own assert on re-run.  The true sharp block-independent
degree-2 threshold at machine 17 is W* = 30, not 31: an exact certificate at
width 30 (independently re-verified by from-scratch code - fresh zeta-
transform validity of all 138 used cuts, fresh lhs < rhs over the full column
set) and an exact feasible point at 29 whose completion is exhibited.  So the
adaptive degree-2 cuts ARE strictly sharper than the Kounias family at
machine 17, and round 22's 8/21/31/37 are correct only as KOUNIAS-FAMILY
thresholds.  The wrong round-23 "feasible at 30" verdict could not be
reproduced; the one verification asymmetry that could admit it (the
completion-LP "extends" verdict was trusted without re-asserting the
completion) is now hardened in research/lp_degree_range.py, and every
standing feasible verdict in this entry has been re-run through the hardened
gate and survives with its completion exhibited.  Nothing in the rung table
or the headline changes: the budget-width verdicts below were all
re-verified.

At machine 11 the consistent degree-2 relaxation is EXACT - W* = F = 7, gap
1.  And the consistency-free gap wanders (1.14, 1.91, 1.67) while the
consistent gap is flat at 1.27-1.28.  That flatness is the whole story: the
budget ratio B(y)/F(y) a rung needs is 2.29, 1.82, 1.56, 1.48 at the steps
landing on 11, 13, 17, 19, so a gap pinned near 1.28 clears all four while a
gap that jumps to 1.91 does not.  (The required ratio never exceeds 1.48 again
further up the ladder and dips to 1.08 at 31 -> 37 - see
`moment-degree-ceiling.md` - so the flat gap is exactly the property that
would have to survive.)

### Measured, exact

    machine 13, width 20 (the budget for the 11 -> 13 rung):
      block-independent degree 2   FEASIBLE   (no certificate exists)
      block-independent degree 3   FEASIBLE
      block-independent degree 4   FEASIBLE   (= all gears; full per-position
                                               joint information)
      CONSISTENT       degree 2    INFEASIBLE - exact certificate
                                   sum of block maxima 660/37 < 664/37

    machine 17, width 28 (the budget for the 13 -> 17 rung):
      block-independent degree 2   FEASIBLE
      block-independent degree 3   FEASIBLE
      CONSISTENT       degree 2    INFEASIBLE - exact certificate
                                   2533/96 < 5081/192

The FEASIBLE verdicts are not solver reports: each is an exact rational
point whose degree-<=l moment vector at EVERY position of [0,W) extends to a
distribution on {0,1}^gears with zero mass on the empty atom.  That is the
sharp condition for "no degree-l cut is violated here", so it rules out every
degree-l inequality at once, not merely the ones the LP happened to generate.

Degree ON TOP of consistency also works and is not needed: the consistent
degree-3 LP proves the same rungs (machine 11: 9 < 10; machine 13:
1305/128 < 1309/128), at a larger cost.  Degree is a fine thing to have once
the relaxation is consistent; it is worthless without.

### Why degree cannot substitute for consistency

The two constraints live in different directions.

A degree-l cut is a statement about ONE POSITION: it constrains the moment
vector (m_S(i))_{|S| <= l} at that position, and per-position completability
already includes every such statement - Frechet inequalities
m_{ab} >= m_a + m_b - 1 among them.

Marginal consistency is a statement ACROSS BLOCKS: it forbids gear a's own
phase distribution and the pair (a,b)'s phase-pair distribution from being
different objects.  Without it the LP can have gear a spread its phase to
cover well while the pair block simultaneously pretends a and b are arranged
to overlap as little as possible - and no per-position moment inequality
detects that, because the moment vector (p_a, p_b, m_ab = 0) is a perfectly
legitimate moment vector of a real distribution whenever p_a + p_b <= 1.

THE OTHER TWO CANDIDATE FIXES, for the record.  A ROUNDING / INTEGRALITY
argument cannot help: W* is already an integer and the thresholds are exact,
so there is no fractional value to round down - the LP at width 20 is
genuinely feasible, with an exhibited exact point.  The only version of
"integrality" that could bite is a lift of the covering IP that cuts that
point off, and marginal consistency IS such a lift, which is what the finding
says.  A TARGETED CUT at that machine cannot help either, for the same reason
the degree ladder cannot: the exhibited point violates NO degree-l cut, for
l up to the number of gears.

This also explains round 22's PAIR-VISIBILITY phenomenon (q_a q_b > 4W => the
pair drops out of the LP entirely, "0 of 6 pairs visible at machine 13").
That degeneracy was never a fact about the machine; it was an artefact of the
missing consistency.  Under consistency no pair can leave the LP, because it
cannot choose a zero-overlap phase pair unless its marginals allow it.

### What it buys: the rung table

A rung landing at machine y needs a certificate of width exactly
B(y) = F(prev prime) + y.  Round 22 proved two rungs; consistency at the SAME
degree 2 proves the first four outright:

      step       budget B   round-22 W*   consistent degree 2     verdict
      7 -> 11       16            8        9 < 10                 PROVED (both)
      11 -> 13      20           21        660/37 < 664/37        PROVED (new)
      13 -> 17      28           31        2533/96 < 5081/192     PROVED (new)
      17 -> 19      37           37        258513/8192 < 64637/2048   PROVED (both)
      19 -> 23      48           90        no certificate         UNDECIDED
      23 -> 29      63       vacuous       none can exist         REFUTED

  19 -> 23: the consistent degree-2 LP at width 48 was driven to 1,607 cut
  rows over 3,836 columns without producing a certificate, and the discovery
  loop converged to a point violating no degree-2 cut by more than 1e-5; but
  no EXACT global point was built either, so the honest verdict is UNDECIDED,
  not "fails".  What is certain is that the vehicle does not prove it.

  23 -> 29: REFUTED outright, and consistency cannot help.  At machine 29 the
  uniform product measure's degree-2 moments are completable
  (`moment-degree-ceiling.md`), so every degree-2 cut is satisfied at every
  position and every width.  The uniform product measure is a GLOBAL
  distribution, hence a feasible point of the consistent relaxation too - so
  CONSISTENCY BUYS WIDTH, NOT MACHINES, and the round-22 vacuity ceilings all
  stand unchanged for the consistent hierarchy.

  THE DEGREE AXIS, for the record, since the brief asked for it as a table of
  machines against degree.  In the consistency-free shape the answer is that
  the axis is FLAT: degree 3 and degree 4 prove no rung that degree 2 does
  not, and at machine 13 no degree at all reaches the budget width.  So there
  is no "machine at which degree 3 stops": it never started.  The table that
  does have content is degree x consistency, and it has exactly two columns.

## 2. WHY IT MIGHT BE NOVEL

- The mechanism (marginal consistency between the blocks of a lifted LP) is
  utterly standard - it is the defining feature of the Sherali-Adams and
  Lasserre hierarchies, and its absence is the defining feature of the
  classical Bonferroni/Kounias/Hunter-Worsley bounds.  NOTHING in the
  machinery is new.
- What appears to be new is the MEASUREMENT and its direction on this
  problem: for the Jacobsthal-type covering IP, one level of marginal
  consistency at degree 2 strictly dominates two extra degrees WITHOUT it -
  demonstrated by exact certificates on both sides at the same machine and
  the same width - and it is exactly the difference between proving and not
  proving three merge steps.
- The classical bounds in this area (Costello-Watts included) are all
  consistency-free counting bounds.  The observation that the missing unit is
  a consistency unit, not a moment unit, is the reading this entry adds.

## 3. PROOF

SCRIPT-VERIFIED, exact: `research/lp_degree_range.py`
(`uv run python research/lp_degree_range.py X G M R`).

Validity chain.
(a) IP exactness: CRT realises every phase tuple, so max blockable width is
    F(M) - 1.  (Asserted against period sieves in round 22.)
(b) Relaxation validity: a blocked window gives an integral feasible point of
    both relaxations, so LP infeasibility at W implies F(M) <= W.  For the
    consistent LP the integral point is a single phase tuple, whose block
    marginals are trivially consistent.
(c) The INFEASIBLE certificate.  With y_r >= 0 on cut rows, mu_S on block
    rows and nu on consistency rows, every column j gets the weight
    a_j = sum_r y_r lam^r_{S(j)} [i_r in O_j]
          + sum_{links with j among the extensions} nu
          - sum_{links with j the restricted tuple} nu,
    so if  sum_S max_{j in S} a_j  <  sum_r y_r (1 - lam^r_0)  the system is
    infeasible.  Both sides are computed EXACTLY over the full phase-tuple
    column set, and the weights are snapped to a single common denominator
    (machine 13: denominator 37).
(d) Every cut used is asserted EXACTLY valid: its subset-sums over all
    2^n - 1 nonempty atoms are >= 1, by an exact zeta transform.
(e) The FEASIBLE verdicts are exact points plus an exact completion LP at
    every position (`exact_lp.feasible_eq` over the 2^n atoms).
(f) FALSIFICATION TEST, run every time: at width F(M) - 1 a blocked window
    exists, so no certificate may be produced.  Machines 11, 13, 17, 19 all
    return "feasible", as they must.
(g) Floats are used for DISCOVERY only (the master LP and the cut-generation
    loop); nothing float decides anything, and the run aborts if a float
    verdict cannot be reproduced exactly.

Honest limits.  The consistent relaxation's columns are full phase tuples, so
its size grows like the product of the gears in each block: degree 2 is
affordable to machine 29 (6,620 columns), degree 3 only to machine 13-17.
Where no certificate is found and no exact global point is built either, the
entry says UNDECIDED rather than "fails".  The block-independent degree-4
decision at machine 17 was STOPPED after about 45 minutes without settling -
its exact-separation loop kept generating cuts - so that cell was blank, not
"fails".  ROUND 24 FILLED IT: the cut loop was the wrong tool; posing the
whole question as ONE completion LP (find block distributions plus, per
position, a strictly interior completion - decide_direct in
research/cw_consistent.py) settles it in 13 seconds: machine 17, width 28,
degree 4 is FEASIBLE, completion exhibited and re-asserted at every position,
so NO degree-4 certificate of the budget width exists.  The degree row is now
decided at machine 13 (degrees 2, 3, 4) AND machine 17 (degrees 2, 3, 4), and
no degree tested anywhere reached a budget width.

## 4. IMPLICATIONS

- (D) now has FOUR consecutive rungs proved by a second, wholly independent
  vehicle (7->11, 11->13, 13->17, 17->19), matching the kernel-proven ladder
  rung for rung by a method that shares nothing with the merge law.
- It redirects the vehicle's development: round 22's named next construct was
  "build the sharp degree-3 cut".  That is now measured to be the WRONG
  direction - degree 3 and degree 4 both fail where consistency succeeds.
  The right direction is more consistency (Sherali-Adams level 2 and up),
  which costs columns rather than moment degree.
- It gives the formalist a smaller certificate species than round 22's: the
  weights snap to one common denominator, so the Lean object is a list of
  INTEGERS plus one denominator.

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

- How far up the ladder does one level of consistency reach?  (Section 3 of
  the script settles 19 -> 23 and 23 -> 29 exactly where it can.)
- Is there a machine at which consistency at degree 2 stops but consistency
  at degree 3 continues?  The columns make this expensive above machine 19.
- The vacuity ceiling of `moment-degree-ceiling.md` was computed for
  consistency-free degree-l families.  What is the ceiling for the CONSISTENT
  hierarchy?  The uniform product measure is a global point, so it still
  kills every degree-l cut from the same machine on - i.e. THE CEILING IS
  UNCHANGED by consistency.  Consistency buys width, not machines.

## 6. PRIOR-ART CHECK

Done 2026-08-25 (the round-22 checks are re-run, since prior-art checks
expire).  Searches:
1. "Sherali-Adams lift-and-project relaxation Jacobsthal function covering
    residue classes certificate"
2. "Jacobsthal function upper bound 2026 linear programming certificate
    maximal gap coprime"
3. "covering integer program marginal consistency versus higher moments
    integrality gap product measure Bonferroni certificate number theory"
4. full LaTeX source of Costello-Watts arXiv:1208.5342, read in full (round
   22 read the abstract-level content only).

Nearest published results.
- Sherali & Adams (1990); Lovasz-Schrijver; Lasserre: the hierarchies whose
  level-1 consistency is exactly what is used here.  Classical, and cited as
  the machinery.
- Kounias (1968), Hunter (1976), Worsley (1982), Prekopa / Boros-Prekopa
  (Oper. Res. 36, 1988): the Bonferroni bound families and their LP form.
  All are consistency-free - they bound P(union) from binomial moments alone,
  which is precisely the weakness this entry measures.
- Costello & Watts, arXiv:1208.5342: a consistency-free recursive counting
  bound (see `covering-lp-certificates.md` section 6 for the full reading).
- No source found that measures consistency against degree on a
  Jacobsthal-type covering problem, or uses either to certify a merge step.

VERDICT: PARTIAL OVERLAP.  All machinery classical; the measured direction
(one consistency level beats two degrees, on this problem, at the same width)
and the (D) application were not found.  NOT independently confirmed.
