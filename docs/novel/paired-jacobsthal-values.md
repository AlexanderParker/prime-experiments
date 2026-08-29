# paired-jacobsthal-values - exact values of the paired Jacobsthal function h_2 and the per-difference family F_d(y)

Status: COMPUTED / SCRIPT-VERIFIED (finite computations, exhaustive at each y).
Prior-art verdict: PARTIAL OVERLAP - the headline h_2 values are KNOWN (Ziller-Morack
2017 computed them, and further, in a companion note the project had not found);
the per-difference family, the fixed-twin ladder, and the maximiser structure are
novel as far as searched. See section 6, checked 2026-08-23.

## 1. What it is

Plain language. Take the primes up to y and ask: how long can a run of consecutive
positions be in which no position carries a pair of numbers, two apart times some
fixed even difference, both coprime to every prime in the set? The classical
Jacobsthal function asks this for a single number; here it is asked for a PAIR at a
fixed even difference - the two-residue analogue. The project computed, for every
even difference separately and exhaustively, the maximal such run, and hence the
first values of the maximum over all differences - the paired Jacobsthal function
h_2 named by Ziller and Morack - together with which differences attain the maximum.

Precise form. Following Ziller-Morack (arXiv:1706.00317, Definitions 2.1-2.2): for
n in N, j_2(n) is the smallest m such that every paired progression <a,b>_m =
{(a+i, b+i) : i = 1..m} with 2 | (b-a) contains a pair (x,y) with gcd(x,n) =
gcd(y,n) = 1; and h_2(n) = j_2(p_n#), the primorial case. The project works in
halved coordinates (slots 6k+-1 for the twin difference; the general even
difference 2d has reduced difference e): for each e, F_e(y) is the maximal window
length avoiding a slot whose both members are coprime to the primorial of primes
<= y, and h_2 = 2 * max_e F_e (equivalently, in Ziller-Morack's condensed form,
h_2(n) = 6*omega_2(n) + 6; so max_e F_e = 3*omega_2 + 3 - the two conventions
agree at every computed point).

The computed table (exhaustive over every even difference at each y;
research/jacobsthal_family.py, jacobsthal_h2_17.py):

    y    P (odd)   #diffs    h_2   p^2-p   Conj.6   margin
    3        3         1       6      6   EXCLUDED    0.0%   <- n = 2, equality
    5       15         7      18     20    HOLDS    10.0%
    7      105        52      30     42    HOLDS    28.6%
    11    1155       577      66    110    HOLDS    40.0%
    13   15015      7507     150    156    HOLDS     3.8%   <- the dip
    17  255255    127627     192    272    HOLDS    29.4%

ROUND-23 REFEREE CORRECTION to the first row (research/j2_referee.py section R1).
The table previously read "y = 3 ... h_2 = 0 ... holds". That was a code artefact:
research/jacobsthal_family.py returns 0 whenever a period carries fewer than two
survivors, and at gears = {3}, e = 1 the survivor set mod 3 is the single class
{1}, whose CYCLIC gap is 3, so h_2 = 6. OEIS A288815 confirms h_2 = 6 at p_n = 3.
The correct reading is sharper than the old one: h_2 = p_n^2 - p_n EXACTLY at
n = 2 (and also at n = 1: h_2(2) = 2 = 2^2 - 2), so Conjecture 6 fails by
EQUALITY at both excluded indices - which is exactly why Ziller-Morack state it
for n >= 3, and the "n >= 3" hypothesis is sharp, not conservative.

The Ziller-Morack Conjecture 6 bound (exact wording, arXiv:1706.00317, Conjecture 6:
"Let n in N >= 3. Then h_2(n) < p_n^2 - p_n") holds at all five admissible points, but the
margin is non-monotone with a one-off dip to 3.8% at y = 13 (vs 10.0, 28.6, 40.0,
29.4). Round 17 resolved the dip: it belongs to the STEP 11->13, not to any
difference class - it needs both a twin prime step (bound grows only x1.42) and a
clean extension of the extremal delta-profile (h_2 gains fully, x2.27); at 17 the
profile must compromise (x1.28) while the bound grows x1.74.

Attached structure (same computations):

- Maximisers, COMPLETE lists (round-23 correction: earlier versions of this line
  printed the first five / first six entries of research/jacobsthal_family.py's
  argmax slice `arg[:5]` and read as if complete; the true counts are 8, 16 and 64
  at y = 11, 13, 17, matching the round-22 delta-space ladder exactly):
    y = 11, F = 33, 8 maximisers:
      e = 41, 146, 239, 316, 344, 349, 421, 454
    y = 13, F = 75, 16 maximisers:
      e = 344, 734, 839, 916, 2164, 2269, 2659, 3919, 4166, 4271, 4661, 5921,
          6091, 7169, 7274, 7351
    y = 17, F = 96, 64 maximisers (first six: 2791, 3176, 5584, 5794, 6361, 6571)
  All coprime to P, none small, none structured.
- Delta-profile law (delta_q(e) = min(e mod q, q - e mod q)): maximisers are
  exactly the carriers of specific profiles - (1,1,1,3) at gears <= 11,
  (1,1,1,3,6) at <= 13 (16 of 7507, recall and precision 100%), (1,1,2,4,6,8) and
  (1,1,2,3,4,3) at <= 17 (precision 100%, recall 50/50). Every winning profile
  begins delta_3 = delta_5 = 1.
- Fixed twin difference (d = 2) ladder, halved coordinates: F(2,37) = 264,
  F(2,41) = 273, F(2,43) = 309, F(2,53) >= 426 (pruned search still running;
  needs <= 486 for the tolerance constant; quadratic-law prediction ~441).
- y = 19: exhaustive scan out of reach in-project (2,424,922 differences);
  lifting the gears <= 17 elite gave h_2(19) >= 222 vs bound 342. (Settled by the
  literature - see sections 4 and 6: h_2(19) = 258.)
- A failed prediction, recorded: from the first four points the project predicted
  extrapolation (~330) would BREACH the bound at y = 17; refuted the same round
  by exact computation (192, holds).

## 2. Why it might be novel

h_2 is a named function with a literature: Ziller-Morack define it, conjecture the
bound p_n^2 - p_n, and prove (their Theorem 4.1) that the bound implies Goldbach
AND the infinitude of prime pairs at every fixed even difference. The project
believed, from arXiv:1706.00317 alone, that the literature computes no exact
values ("ZM compute none"). That belief was WRONG - see section 6 - but two layers
remain that the literature does not have:

- the PER-DIFFERENCE resolution: F_e(y) computed and recorded for every even
  difference separately (7,507 differences at y = 13; 127,627 at y = 17), where
  the literature computes only the maximum over all differences;
- the fixed-twin-difference ladder F(2,y) up to y = 43 (and the y = 53 bound),
  which no published table or OEIS sequence contains;
- the maximiser identification and the delta-profile law (which differences are
  extremal and why), which the literature's condensed formulation (free residue
  pairs, difference forgotten) cannot even express without unwinding the CRT.

This is not a restatement of a classical sieve bound: the exact values are finite
combinatorial facts about two-residue coverings, and the classical shadow
(Hardy-Littlewood singular series) ranks densities, not covering maxima - the
project measured that density does not determine the extreme (F_max/lambda ranges
2.88-7.52 over the 31 gcd classes at y = 13).

## 3. Proof / verification

Status: COMPUTED / SCRIPT-VERIFIED (finite). No kernel-checked theorem claims the
table itself; each value is an exhaustive finite computation.

- research/jacobsthal_family.py, research/jacobsthal_h2_17.py - the h_2 table,
  exhaustive over every even difference, and the percentile data.
- research/why13.py, research/maximiser_shape.py, research/h2_19_lift.py - the
  dip analysis, maximiser shapes, and the y = 19 lift.
- research/zm_margin_mechanism.py (round 20) - slack quantisation, step law,
  jump-ratio uniqueness, persistence events, argmax-trajectory verifications
  against ZM's full table (section 4a); research/family17_percentile.py - exact
  tie-aware percentile bookkeeping, per-class arrays saved to
  research/data/f13_family.npy and f17_family.npy.
- rust2/src/bin/maxgap_pruned.rs (+ log research/data/maxgap53_pruned.log) - the
  pruned fixed-twin search; verified identical to the unpruned original on
  F(2,y) for y = 11..37 before being trusted further.
- Kernel-checked adjacent structure (proofs/Polignac.lean, standard axioms or
  fewer): endpoint_run_mod_three (F(2,y) = 0 mod 3, justifying the pruned
  search's mod-3 skip; all known exact values comply) and the mod-3 dichotomy
  for the family (3 | F_d(y) for every gear set iff d != 0 mod 6).
- External cross-check found during the prior-art sweep (section 6): all five
  h_2 values agree exactly with Ziller-Morack's independently computed table
  (different algorithms: their sequential/ILP searches vs the project's
  exhaustive per-difference scan). The project's computation is therefore an
  independent replication at the overlap points.

## 4. Implications

Inside the project:
- Conjecture 6 is the localised form of the route's target family; the margin
  table prices how much room the bound has at each scale, and the dip at 13 is
  the sharp "why is 13 extremal?" question that round 17 answered.
- The tolerance-constant computation still needs F(2,53) <= 486 - a
  per-difference value the literature does NOT supply (it only bounds
  max_e F_e(53) = 711 from h_2(16) = 1422).
- The prior-art sweep itself settles a listed open question: Ziller-Morack's
  table has h_2(19) = 258 < 342, so Conjecture 6 HOLDS at y = 19 (margin 24.6%);
  the project's lift bound (>= 222) was consistent, and its round-17 "compromise"
  scenario (~250) was the right prediction, not the "clean extension" (~288).
- Computing margins from their full table (p_n = 19..73: 24.6, 27.7, 44.6, 38.7,
  46.8, 45.5, 42.2, 40.6, 48.4, 51.6, 48.0, 50.5, 50.5, 50.1 percent), the 3.8%
  dip at 13 remains the unique extreme through p_n = 73, and the margin drifts
  toward ~50% - i.e. h_2 ~ (p_n^2 - p_n)/2 empirically. The dip observation
  survives contact with the full published table; the "one-off" qualifier is now
  verified 19 points deep instead of 5.

Outside the project: the per-difference family is the finer object - Ziller-
Morack's reduction (their Propositions 3.2/3.5) uses one uniform bound for every
even difference, while the family shows the quantity being bounded varies by more
than 2x across differences at fixed y (see twin-percentile.md). A per-difference
Conjecture 6 would be a strictly sharper, previously unformulated statement.

## 4a. Round-20 mechanism: why 13 is extremal, resolved against the full ZM table

(research/zm_margin_mechanism.py, all assertions pass; output at
research/data/zm_margin_mechanism.out. Every item below is an exact event, no fits.)

SLACK QUANTISATION. h_2 = 0 mod 6 always, and the bound B = p^2 - p has fixed
residue mod 6 (0 for p = 1 mod 6, 2 for p = 5 mod 6), so the conjecture's slack
B - h_2 is quantised with minimum admissible value 6 resp. 2. EVENT: the minimum is
attained exactly TWICE through p = 73 - at p = 5 (slack 2) and p = 13 (slack 6) -
and never again. "3.8% margin at 13" is really "one quantum": in ZM's condensed
units omega_2(6) = 24 = cap 25 minus one. Both exact points are twin-step landings.

THE STEP LAW (18 steps of ZM's table). Relative margin FALLS at all 6 twin steps
landing at p >= 13 (11->13, 17->19, 29->31, 41->43, 59->61, 71->73), RISES at all
5 gap-6 steps, and is genuinely mixed at gap-4 steps (3 up, 2 down). Absolute slack
falls ONLY at twin steps: ->13 (-38), ->31 (-2), ->61 (-8). Mechanism of the sign:
margin is stable when h_2 grows like the bound; d(B)/B ~ 2g/p while d(h_2)/h_2 ~
2r/p (h_2 ~ p^2/2, r = per-step jump in halved units per q'), so the crossover sits
at gap g ~ mean r ~ 1.9-2.5 - twin steps always lose margin, gap-6 always gains,
gap-4 is the knife edge. A DIP therefore needs r >> g: a huge jump landing on a
twin step.

THE UNIQUE JUMP. r = Delta(maxF)/q' at 11->13 is 42/13 = 3.231, the unique value
above 2.6 in all 18 steps (runner-up 2.553 at ->47, then 2.348 at ->23). The dip at
13 is exactly this outlier landing on a twin step.

WHY THE JUMP: THE LAST CLEAN-EXTENSION STEP. Exhaustive family scans at y = 5..17:
winners at 13 restrict mod 1155 to winners at 11, every one (16/16 have F_11 = 33,
F_13 = 75 - the SAME fixed difference gains 3.231 q' in one step); winners at 11
likewise extend winners at 7; but winners at 17 restrict to F_13 in 42..51 (family
max 75) - NOT winners - and the best 17-extension of any 13-winner reaches only
F = 87 vs the true max 96. So maximiser persistence holds up to 11->13 and DIES at
13->17; the merge-law round showed it stays dead (the 19-argmax restricts to the
twin's own value 54 at 17, with 35,848 classes strictly above it - verified here
from the full 17-scan). 11->13 is the LAST step where the family maximum can grow
by full profile extension, and it happens to land on a twin step of the bound. Both
coincidences - and the mod-6 quantum - meet only at 13.

THE BUDGET EVENTS (route-relevant). Per-difference single-step increments measured:
e = 344 gains 3.231 q' at 11->13; the 19-argmax e = 1,532,627 gains 75 = 3.947 q'
at 17->19 (54 -> 129, verified by direct construction); the 23-argmax
e = 107,207,699 gains 102 = 4.435 q' at 19->23 (81 -> 183, verified). The round-14
budget audit's structured-d worst was 1.846 q' and twins' own worst 2.432 q'
(31->37). CONSEQUENCE: no uniform increment budget alpha <= 3 (in q' units) holds
over the full even-difference family - the tolerance-route constant is
structured-d-specific, and the known family-argmax jumps 3.23, 3.95, 4.43 are
non-decreasing.

EMPIRICAL SHARE. The margin drift toward ~50% (section 4) says h_2 ~ (p^2 - p)/2.
Neither ZM paper states any growth observation (both re-read in full this round;
the computation note has no asymptotic commentary and no remark on the 13 case), so
the refined statement "h_2(n) = (1/2 + o(1)) p_n^2, with the p = 13 point the
unique quantum-exact approach to the bound" has no counterpart in print. Recorded
here as an OBSERVATION with a candidate conjecture attached, not a claim.

## 4b. Round-21 mechanism: why clean extension dies at 17 - the exact 9 explained

(research/ext_death.py + ext_death2.py, all assertions green; outputs
research/data/ext_death{,2}.out. Round-20 open item closed.)

THE SHALLOW-EXTENSION CAP (proved for the observed configurations, exact). A
family maximiser's record window is a maximal GAP of its machine - it has no
interior openings. Lifting e to the next gear q', the window can only grow by
fusing ADJACENT gaps, whose shared endpoints become interiors; interiors must lie
in the two tooth classes {0, -e} mod q', and THREE interiors would need three
distinct residues inside a 2-element set - impossible (the endpoint triples have
3 distinct residues mod q' in every observed case; asserted). At most two
interiors ever fuse, and any 2-interior configuration needs ONE separation
congruence mod q', which the lift choice (e mod q' is free) can always grant. So

    best extension of a record  =  F_old + max(g_L + g_R, g_R1 + g_R2, g_L2 + g_L1)

- the record plus the best adjacent TWO-GAP sum. The deep-fusion winners, by
contrast, sit on mediocre bases whose openings align mod q' (4-5 old gaps fused,
3-4 interiors filling both tooth classes - anatomies in ext_death.out).

THE EXACT NUMBERS. All 16 13-winners have identical local context
(..6, 3, 6, [75], 6, 3, 6..): adjacent 2-sums {6+6, 6+3} -> cap 75 + 12 = 87,
attained (e = +-7 mod 17 kills both 75-endpoints since 75 = 7 mod 17); the
exhaustive extension value set over all 16 x 17 lifts is exactly
{81, 84, 87} = {75+6, 75+6+3, 6+75+6}. The true max 96 is a 4-5 gap deep fusion
on bases with F_13 in {42, 51}. THE 9 = 96 - (6+75+6): the difference between
the best deep fusion and the record's shallow cap.

THE LADDER (death is permanent and the deficit doubles). Best 19-extension of
all 64 17-winners (1216 lifts, direct sieves): 111 = 96 + (6+9) vs true max 129 -
deficit 18. Best 23-extension of the 19-argmax: 147 = 129 + (6+12) vs 183 -
deficit 36 (lineage-only caveat: all 19-winners are not known; the 18 is over all
17-winners). Deficits 9, 18, 36; the records' best adjacent 2-sums grow by 3
(12, 15, 18). Anatomies: the 111-window is [96, 6, 9] and the 147-window is
[129, 6, 12] - one-sided two-gap chains, exactly at the cap.

CONSEQUENCE for the family's structure: from 17 on, the argmax trajectory is
forced to abandon its ancestors - a record window is self-limiting (no interior
structure to exploit, small flanks by the anti-correlation of record windows),
while each new gear's winner is a fresh deep resonance. This is the mechanism
behind maximiser non-persistence (4a item 4), now with the cap law and the exact
deficit accounting.

## 4c. Round-22: the DELTA REDUCTION, the complete 19-winner set, and the end of
## the deficit doubling

(research/delta_frame.py, family_scan.py, family_scan_fast.py, ext_deficit19.py,
family_scan23.py, ext_deficit23.py; all assertion-gated, outputs in research/data/.)

THE DELTA REDUCTION (proved; verified against the round-19 definition at y = 11, 13,
17 and by full reproduction of the y=13 winner set). For every even difference with
3 not dividing e, the halved-coordinate max-gap depends on e ONLY through

    delta = e * 3^{-1} mod Q,        Q = prod_{5<=q<=y} q,

and equals 3 * G(delta), where G(delta) is the maximal cyclic gap of
S_delta = {k in Z_Q : k != 0, -delta mod q for every gear q}.  Reason: with 3 not
dividing e, gear 3 kills n = 0 and n = -e mod 3, so every survivor lies in the one
remaining class c mod 3; writing n = 3k + c turns the gear-q condition into
k != -c/3, (-e-c)/3 mod q, which is the tooth pair {0, -delta} translated by the
single integer -c*3^{-1} mod Q.  Translation does not move the gap multiset, and
gaps in n are 3x gaps in k.  This collapses the y=19 family from 2,424,922
differences (round 17: "exhaustive scan out of reach") to 1,616,615 deltas, and the
y=23 family from 55,773,217 to 37,182,145.

THE HELD-OUT-GEAR PREFILTER (exact, not heuristic).  A gap G >= Gmin needs a run of
L = Gmin - 1 consecutive killed positions.  Hold out the top gear qt: every survivor
of the smaller gears inside such a window must be killed by qt, i.e. must lie in
{0, -delta} mod qt.  The window's absolute position mod qt is free by CRT, so the
condition on the window is exactly

    |{ j mod qt : j an offset of a surviving position in the window }|  <=  2,

and two distinct residues r1 != r2 force delta = +-(r1-r2) mod qt.  Nothing that
could carry a run of length L is discarded, so the output is COMPLETE.  Validated
against brute force in delta space at y = 13 (16 winners) and y = 17 (64 winners),
and the three-level fast version against the two-level one at both scales.

RESULTS.

1. h_2(19) = 258 INDEPENDENTLY REPLICATED by exhaustive family scan.  The prefilter
   keeps 64 of 1,616,615 deltas (0.0040%); all 64 reach G = 43 and none exceeds it,
   so max_e F_e(19) = 129 and h_2(19) = 258 - matching Ziller-Morack's computation
   note by an entirely different method (they compute the maximum; this enumerates
   the whole argmax set).  Round 17 had this scan as out of reach.

2. THE COMPLETE 19-WINNER SET: exactly 64 deltas.  The winner-count ladder is
   8 (y=11), 16 (y=13), 64 (y=17), 64 (y=19) - it does NOT keep quadrupling, and the
   19-winners are not lifts of the 17-winners (the best 19-extension of any
   17-winner is 111 < 129).

3. THE 3 | e BRANCH, settled exhaustively.  For 3 | e the survivors occupy two
   classes mod 3, so a gap of 3G needs killed runs of length >= G-1 in BOTH
   sub-lattices, and both sub-lattices are translates of the same S_delta - hence
   such a delta must already be in the G >= 43 list.  Checking those 64 directly:
   the best 3|e difference at y = 19 reaches F = 44 against 129.  The family
   maximiser is never divisible by 3, at y = 19, by exhaustion rather than by
   observation.

4. THE DEFICIT LADDER OVER COMPLETE WINNER SETS.  Recomputed with independent code:

       step       #winners   best extension F   true max F   deficit
       13 -> 17         16                 87           96         9
       17 -> 19         64                111          129        18
       19 -> 23         64                147          183        36

   The 19->23 value 36 was lineage-only in round 21 and is now over the complete
   64-winner set: round 21's three numbers are confirmed exactly.  A best-extension
   anatomy at 19->23 (delta_19 = 27996, lift r = 1): the fused old-machine gap word
   is [129, 3, 15] in F units, sum 147 - a one-sided two-gap chain exactly at the
   cap law's ceiling, and a second realisation of the round-21 anatomy [129, 6, 12].

5. RECONCILED WITH ZILLER-MORACK'S OWN EXHAUSTIVE DATA - and it matches exactly.
   ZM's computation note DOES publish exhaustive maximiser data, in a different
   representation the project had not looked at: full_details.pdf Table 1 carries a
   column nseq, "number of sequences of maximum length", and the ancillary files
   remainders_2.txt / permutations_2.txt / moduli_2.txt list them.  Their nseq runs
   1, 6, 1, 1, 4, 2, 2, 14, 8, 4, 1, 8, 2, 16, ... for p_n = 5, 7, 11, 13, 17, 19,
   23, 29, ... - which is NOT the winning-difference count.  Converting: for each
   winning delta take every record window and record which gear (smallest) kills each
   position; that covering pattern is ZM's sequence.  Then

       y        winning deltas   record windows   distinct patterns   ZM nseq
       11             8                 8                 1              1
       13            16                16                 1              1
       17            64               128                 4              4
       19            64               128                 2              2

   EXACT MATCH at all four (research/zm_seq_reconcile.py, assertion-gated), with
   reverses counted separately exactly as ZM state, and the y=11, y=13 singletons
   self-symmetric - which is ZM's own remark that the single sequences at n = 5, 6
   are self-symmetric by default.  So: many differences, very few patterns (8, 16, 64,
   64 differences carry 1, 1, 4, 2 patterns), the two data sets are the same object in
   two representations, and each is now an independent check on the other.  HONEST
   CONSEQUENCE FOR NOVELTY: the winner sets are recoverable from ZM's published
   ancillary lists, so what is new here is NOT the maximiser data.  Nor, on a closer
   read, is the reduction: ZM's Proposition 1.5(2) already states the equivalent
   covering form "for every prime there exist TWO NON-ZERO residue classes covering
   {1,...,m}", which drops the pair (a,b) entirely - the delta frame used here is the
   same normalisation with the two classes written as {0, -delta} mod q after a global
   translation.  And their algorithm suite (BSA2/DSA2/GPA2/ILP2/CRPDSA2/RPA2/BPA2)
   reaches p_n = 73 where the scan here reaches 23, so the held-out-gear prefilter is
   not a competitive method either.  WHAT IS ACTUALLY NEW HERE: the independent
   replication, the exhaustive settlement of the 3 | e branch, and the CROSS-GEAR
   EXTENSION LADDER below - a question ZM never ask.

6. THE DOUBLING IS REFUTED - by arithmetic, before any computation.  A deficit can
   never exceed the increment F(new) - F(old), because the best extension is at
   least F(old).  From OEIS A288815 (Ziller-Morack; pulled in full 2026-08-24)
   F = h_2/2 runs 75, 96, 129, 183, 225, 285, 354, ... at y = 13, 17, 19, 23, 29,
   31, 37, so the increments are 21, 33, 54, 42, 60, 69 - the 23->29 increment
   COLLAPSES to 42, and 42 < 72.  So 9, 18, 36 cannot continue to 72: the doubling
   was a coincidence of three consecutive increments, not a law.  What survives is
   the ACCOUNTING IDENTITY behind it,

       deficit  =  increment  -  (best adjacent 2-gap sum of a record window),

   with the measured 2-gap sums 12, 15, 18 (+3 per rung).  If that arithmetic
   progression continues at 23->29, the predicted deficit is 42 - 21 = 21 (best
   extension 204).  PRE-REGISTERED PREDICTION, tested by the exhaustive y=23 scan
   plus the 29-lift computation (research/family_scan23.py, ext_deficit23.py);
   result in section 4d.  ZM's nseq(23) = 2 gives a second pre-registered check on
   that scan: it must produce exactly 2 distinct covering patterns.

## 4d. Round-22: the y=23 rung - THE DEFICIT COLLAPSES TO ZERO AND CLEAN EXTENSION
## RESUMES (both round-21 conclusions falsified, and the cap law confirmed)

(research/family_scan23.py, ext_deficit23.py, ext23_witness.py, zm_seq_reconcile.py;
outputs research/data/{family_scan23,ext_deficit23,ext23_witness,zm_seq_reconcile}.out.)

THE EXHAUSTIVE y=23 SCAN (1,616,615 prefilter classes, four shards, ~3 h). The
prefilter keeps 128 of 37,182,145 deltas (0.00034%); all 128 reach G = 61 and none
exceeds it, so max_e F_e(23) = 183 and h_2(23) = 366 - Ziller-Morack's value
REPLICATED EXHAUSTIVELY by an independent method, the second such replication this
round. The complete 23-winner set is 128 deltas, extending the winner-count ladder to

    y             11   13   17   19   23
    winning delta  8   16   64   64  128
    ZM nseq        1    1    4    2    2          (patterns; matched exactly, 4c item 6)

PRE-REGISTERED CHECK 1 PASSED: the 23 scan produces exactly 2 distinct covering
patterns, = ZM's nseq(23) = 2. Five machines now agree in both representations.

PRE-REGISTERED PREDICTION 2 REFUTED, AND SO IS ROUND 21'S NARRATIVE. Section 4c
predicted deficit 21 at 23->29 (increment 42 minus a 2-gap sum continuing the sequence
12, 15, 18 to 21). THE MEASURED DEFICIT IS ZERO: 23-winners lift to the FULL y = 29
family maximum G = 75, F = 225, h_2 = 450. This is certified, not merely computed -
ext23_witness.py locates the run and checks it from the definitions with no shared
code: e.g. delta_29 = 743,911,918 (from delta_23 = 269,018, lift r = 3 mod 29) has
k = 134,406,257 .. 134,406,330 all killed (74 consecutive positions, the killing gear
listed for each) with both flanking positions open, hence G = 75 exactly. Three more
witnesses at delta_23 = 1,110,243 / 1,185,318 / 1,334,082, all at r = 3 mod 29.

AND IT IS NOT ONE LUCKY WINNER. Over the complete 128 winners x 29 lifts, EVERY one of
the 128 reaches G = 75, and each does so at exactly the same four lift residues
r in {3, 12, 17, 26} mod 29 = {+-3, +-12} - 512 (winner, lift) pairs, and no other r
works for any winner. Those two residue pairs are precisely the two interior
separations available in the fused word: the openings around a record sit at
0, 2, 14, 75, 77, 79, so the 75-gap is either 0 -> 75 (killing the openings at 2 and
14, separation 12, forcing delta = -+12 mod 29) or its mirror 2 -> 77 (killing 4 and
65, separation 61 = 3 mod 29, forcing delta = -+3). The cap law therefore does not
merely BOUND the extension at this rung - it PREDICTS the admissible lifts exactly.
(Side effect: h_2(29) = 450 now has an independent explicit lower-bound WITNESS as
well, so three consecutive ZM values are confirmed here by three different routes.)

WHY - AND THE CAP LAW IS CONFIRMED, NOT BROKEN. The 23-machine's gap word around a
record window is [2, 12, 61, 2, 2] in slot units (61 = the record G_23). The lift
fuses the record with its two neighbours on one side, 12 and 2: 61 + 12 + 2 = 75,
i.e. TWO interior openings killed - exactly the shallow-extension cap law's maximum
(4b), attained. In F units, best extension = 183 + (36 + 6) = 225. So the accounting
identity of 4c survives intact,

    deficit  =  increment  -  (record's best adjacent 2-gap sum),

and what was wrong was the guess that the 2-gap sums continue 12, 15, 18, 21: they
run 12, 15, 18, 42, and at 23->29 the 2-sum EQUALS the increment, so the deficit
vanishes.

CONSEQUENCES, stated as corrections to round 21:
- "The deficit doubles (9, 18, 36)" - REFUTED. The ladder is 9, 18, 36, 0.
- "From 17 on, the argmax trajectory is forced to abandon its ancestors; a record
  window is self-limiting; each new gear's winner is a fresh deep resonance"
  (4b's closing paragraph) - REFUTED at 23->29: a 23-winner IS a 29-winner's ancestor,
  by explicit certificate. Maximiser persistence is not monotone in y; it fails at
  17, 19, 23 and returns at 29.
- What survives from 4b: the shallow-extension CAP LAW itself (at most two interiors
  fuse), which is exactly what the zero-deficit case attains, and the accounting
  identity. The mechanism was right; the extrapolation from three points was not.

HONEST LIMIT: this is one more rung, not a law. The 2-gap sum next to a record window
is an arithmetic accident of that record's neighbourhood; predicting the deficit at
29->31 needs the 29-winner set, which is a 1.08e9-delta scan - out of reach for the
prefilter as implemented (the y=23 scan already cost ~3 CPU-hours x 4).

## 5. Unsolved questions or conjectures it touches

- Ziller-Morack Conjecture 6 (open; now known verified to p_n = 73 by their
  computation). By their Theorem 4.1 it implies Goldbach and prime pairs at every
  fixed even difference - so any exact value is data on a live conjecture.
- Goldbach's conjecture and the prime pairs (fixed-difference Polignac)
  conjecture, via that reduction.
- The project's route: lemma (D) and the tolerance constant (needs F(2,53) <= 486).
- OEIS: A288815 exists (h_2 at primorials; full data pulled 2026-08-24: 2, 6, 18,
  30, 66, 150, 192, 258, 366, 450, 570, 708, 894, 1044, 1284, 1422, 1656, 1902,
  2190, 2460, 2622 for p_n = 2..73). The per-difference family, the fixed-twin
  ladder F(2,y), and the FAMILY WINNER COUNTS (8, 16, 64, 64 at y = 11, 13, 17, 19)
  are candidate new sequences.

## 6. Prior-art check (2026-08-23)

Searches run:

- WebSearch: `Ziller Morack "Jacobsthal function" computation arXiv` -> found
  arXiv:1706.03668 (the decisive hit, see below), arXiv:1611.03310, OEIS
  A288815/A072753, Hagedorn's paper.
- Fetched and read in full: arXiv:1706.00317 (Ziller-Morack, "Divisibility in
  paired progressions, Goldbach's conjecture, and the infinitude of prime
  pairs", 2017) - definitions of j_2/h_2, Conjecture 6 exact wording, Theorem
  4.1; NO numerical h_2 values in this paper.
- Fetched and read in full: arXiv:1706.03668 (Ziller-Morack, "A short note on
  the computation of the generalised Jacobsthal function for paired
  progressions", June 2017), including its 32-page ancillary full_details.pdf.
- OEIS text-interface lookups: id:A288815, id:A072753, id:A072752, id:A048670;
  sequence searches 18,30,66,150,192,258,366 (hit: A288815 only);
  21,33,54,75,102 (no results); 33,54,75,102,129 (no results); 16,28,39,57,65
  (no results); 42,66,108,150,204 (no results); 264,273,309 (no results);
  keyword search "paired Jacobsthal" (A288815 only).
- WebSearch: `"paired Jacobsthal" OR "generalised Jacobsthal function for paired
  progressions" -Ziller` -> no third-party follow-ups computing further values.
- WebSearch: `Hagedorn "Jacobsthal" function computation "prime pairs" OR twin`
  -> Hagedorn, "Computation of Jacobsthal's function h(n) for n < 50", Math.
  Comp. 78 (2009): ordinary (single-residue) function only, no paired analogue.
- Fetched: arXiv:2007.01808 (Ziller 2020, "On differences between consecutive
  numbers coprime to primorials"): the unpaired coprime-set gap spectrum -
  Ziller's own later work stayed on the single-residue side.

Nearest prior art, and the correction:

1. KNOWN - the h_2 values. Ziller-Morack's companion note arXiv:1706.03668
   (Table 1) computes h_2(n) exactly for ALL n <= 21 (p_n <= 73): 2, 6, 18, 30,
   66, 150, 192, 258, 366, 450, 570, 708, 894, 1044, 1284, 1422, 1656, 1902,
   2190, 2460, 2622 - agreeing exactly with the project's five values at
   p_n = 5..17, and extending twelve points beyond them. OEIS A288815 (Ziller,
   June 2017) carries the same values; the condensed equivalent A072753
   ("maximum gap in two-stage prime-sieves", h_2 = 6*A072753 + 6) is Ziller's
   own sequence dating to 2002, with values accreted by Ziller, Morack, and
   Giovanni Resta through 2017. The harvester claim "first exact values in the
   literature (ZM compute none)" is FALSE - the project had read arXiv:1706.00317
   (which indeed contains no values) and missed the companion computation note
   posted eleven days later. Their computation was also exhaustive over extremal
   sequences (their ancillary files list every maximal sequence in three
   representations).
2. NOT in the literature - everything per-difference. The Ziller-Morack condensed
   formulation (omega_2: two free residues per prime, the difference eliminated)
   computes only max over differences; neither paper nor ancillary files nor any
   OEIS sequence records per-difference values F_d(y), the fixed-twin ladder
   F(2,y) (21, 33, 54, 75, 102, 129, 264, 273, 309, >= 426 - all six OEIS
   sequence probes empty), which differences attain h_2, or the delta-profile
   law. No follow-up literature (2017-2026) computes further h_2 values or any
   per-difference refinement.
3. Context: ordinary-Jacobsthal computations (Hagedorn 2009 to n = 49;
   Ziller-Morack 1611.03310 to p = 251; A048670 to n = 64) are the one-residue
   analogue; Costello-Watts bounds (arXiv:1306.1064 for the explicit
   g(n) <= 2 e^gamma k^{5+5 loglog k}, k > 120; their arXiv:1208.5342 is a
   SEPARATE range-restricted computational bound for 50 <= k <= 10000 and must
   not be quoted for the first - corrected round 26 by research/j2_citesweep.py),
   Ford-Green-Konyagin-Maynard-Tao asymptotics - none paired.

VERDICT: PARTIAL OVERLAP. The five headline h_2 values and the Conjecture 6
verification are KNOWN (Ziller-Morack 2017, arXiv:1706.03668 + OEIS A288815;
their table extends to p_n = 73 and settles the project's open y = 19 case:
h_2(19) = 258 < 342). The delta that stands as novel as far as searched: the
exhaustive per-difference family F_d(y), the fixed-twin ladder F(2,y) including
F(2,37) = 264, F(2,41) = 273, F(2,43) = 309, F(2,53) >= 426, the maximiser
identification and delta-profile law, and the margin-dip analysis (the dip is
visible in their published table but nowhere commented on or explained; the
project's round-17 step-11->13 explanation has no counterpart in print). The
project's values are an independent replication of theirs at the five overlap
points, by a different method - which is corroboration, not priority.
