# restricted-covering-certificates - the composed covering LP applied to a RESTRICTED POSITION SET with a RESTRICTED PHASE DOMAIN: one construct that (i) gives a CASE-SPLIT certificate species strictly stronger than the level-2 vehicle, which certifies EVERY (D) rung through 37 -> 41 hypothesis-free - including two rungs and one whole cell that vehicle was PROVED unable to certify - and (ii) decides ADJACENT-GAP-PAIR realisability by LP duality, giving scan-free exact F_2 at machines 19 and 23

Status: SCRIPT-VERIFIED, exact rational arithmetic on every verdict
(`research/star_case.py`, gate `python research/star_case.py GATE`;
comparison against a full-period scan in `research/window_dict.py`).
Every CERTIFIED verdict carries an exact rational dual certificate saved to
disk and re-verified from a clean rebuild; every REFUTED verdict carries an
exact rational primal point verified to be IN the polytope.
Established round 26 (LP-duality dedicated explorer).
Prior-art check: NOT YET CHECKED (agent has no web access this round).

ROUND-27 ADDENDUM (Formalist): **THE SPECIES IS NOW KERNEL-CHECKED AT THE
19->23 RUNG** - `proofs/CaseSplit.lean` + `proofs/CaseCert23B/C0..C4/.lean`,
`CaseCert23.D_19_23_case (n) : Machine23.g23 n <= 25 + 23`, axiom footprint
the standard three, no `native_decide`, no census hypothesis and no period
anywhere in the derivation. Two things had to be settled first, and both are
findings about the vehicle rather than about Lean:
 * EVERY CERTIFIED CASE OF THE THREE LADDER RUNGS ON DISK USES **ONLY THE
   BASE CUT** (`rows_all_base_cut`, asserted for 5 + 35 + 35 cases in
   `research/lp_cert_lean.py`): the cut loop separated nothing at these
   widths, so the coverage rows are just "this position is blocked by some
   free gear". The whole certificate is therefore the RECURSION ROW plus the
   consistency links.
 * THE RECURSION ROW IS THE **LOWEST-BLOCKER IDENTITY**, and that is what the
   kernel proof discharges pointwise: at a real configuration each position
   contributes `d - 1` to `sum_{a<b} n_ab` because only the lowest blocker can
   be the `a` of a counted pair. Formalised as `CaseSplit.lowest6/7` (a
   64-/128-case Boolean decision) - so `sum_a |A_a| >= |pos| + sum_ab n_ab`
   with no combinatorial machinery at all.
 * A KERNEL-SIZING FACT worth having: `n_ab` is ZERO for **96.4%** of the
   gear-index-1 columns at 29->31 (1,972 of 54,145 over the 35 cases), because
   the single gear below can cover the whole overlap. Listing the exceptions
   lets the kernel skip the max-cover evaluation everywhere else and cut the
   per-case check from 9m01 to 4m10 solo (the whole 35-case rung then built in
   47 minutes wall at two workers). The vehicle's recursion row is,
   numerically, almost entirely a Kounias row at the SMALLEST FREE GEAR.
See `covering-lp-certificates.md` section 8 for the Lean side in full.

Companion entries: `product-measure-frontier.md` (the width frontier this
entry escapes and the STAR-k question it left open),
`recursion-consistency-composition.md` (the composed vehicle),
`covering-lp-certificates.md`, `consistency-over-degree.md`,
`scanfree-certificate.md` (Constructor's CRT set-cover oracle - the same
dictionary decided by a different mechanism), `old-machine-spectrum.md`
(Mechanic's Q*_J, the depth-J generalisation of the windowed statement).

## 1. WHAT IT IS

Plain language.  The project's strongest LP certificate object is the
COMPOSED VEHICLE: minimise the Costello-Watts lower bound

    f(r) = W - sum_q S_q(r_q) + sum_{i<j} n_ij(r_i, r_j)   <=   open(r)

over the pairwise-consistent (Sherali-Adams level-2) polytope, together with
the degree-2 covering cuts at every position of [0, W).  If the minimum is
positive, no fully blocked window of width W exists, i.e. F(M) <= W.  Round 25
proved this vehicle's frontier is a WIDTH (`product-measure-frontier.md`) and
closed its ladder at four rungs: 19 -> 23 and 23 -> 29 were REFUTED by
exhibited exact feasible points, and 37 -> 41 by the uniform product measure.

This entry says the vehicle was being asked the wrong question.  Both of the
ways forward round 25 named turn out to be the SAME move - RESTRICT THE
POSITION SET AND KEEP THE VEHICLE - and both of them work.

  DEFINITION (the restricted vehicle).  Fix a machine M with gears
  q_0 = 5 < ... < q_{n-1} = y, a width W, a set K of HELD gears (a prefix of
  the gear list) with a phase w_q for each, and a set OP of positions required
  to be OPEN.  Put

     pos  =  [0, W)  minus  {positions the held gears block at w}
                     minus  OP,
     dom(q)  =  {phases r of q : q at phase r blocks no position of OP},
     n^{pos,dom}_ij(u,v) = |P_ij(u,v) & pos|
                           - max over the phases IN THEIR DOMAINS of the free
                             gears below q_i of what they cover of that set.

  THE RESTRICTED LP is the composed vehicle with [0,W) replaced by pos, the
  gear list replaced by the free gears, each gear's phase block indexed by
  dom(q), and the degree-2 cuts taken at the positions of pos only.

  SOUNDNESS (proved, elementary).  Any actual configuration of M in which the
  held gears sit at w, every position of OP is open, and every position of pos
  is blocked, induces a 0/1 point of this polytope: its phases lie in the
  domains; every position of pos is covered; and open(pos) = 0 gives the
  recursion row.  Restricting the lower gears to their domains only RAISES
  n_ij (a minimum over fewer phases) and n_ij <= N_ij still holds at the
  actual tuple, because that tuple's phases are in the domains.  So an exact
  dual certificate of infeasibility EXCLUDES every such configuration.

Two instances, and they are the two objects this entry is about.

  (A) THE CASE SPLIT (OP empty, K = the k smallest gears).  Case w is exactly
  the composed vehicle on the sub-window U_w = [0,W) minus what the held gears
  block.  A certificate in EVERY case is a CASE-SPLIT CERTIFICATE of
  F(M) <= W, because every real window puts the held gears at some phase.

  (B) THE WINDOWED STATEMENT (K empty, OP = {0, a, W} in ambient width W+1).
  Then pos = {1..W-1} minus {a}, and a certificate says: MACHINE M HAS NO
  ADJACENT GAP PAIR (a, W-a).  That is exactly membership in the level-2 gap
  dictionary the chain (`scanfree-certificate.md`) and the merge law consume -
  decided here by LP duality with an exact rational certificate, no CSP
  search and no period scan.

  RELATION TO STAR-3 (proved).  The case split is STRICTLY STRONGER than the
  STAR-3 LP of `cw_consistent.Composed3`, which carries triple blocks
  (5, q_i, q_j) tied only to the singles.  A family of feasible case points
  always MIXES into a feasible STAR-3 point (completability is convex, and in
  a case where a held gear blocks the position the moment vector is trivially
  completable; the row is an average of satisfied rows), but a STAR-3 point
  does NOT condition into a family of case points, because its conditionals
  need not be pairwise consistent.  So "all cases infeasible" implies "STAR-3
  infeasible" and not conversely.

## 2. WHAT IT ESTABLISHES (the numbers, all exact)

RESULT 1 - THE CASE SPLIT RECOVERS A RUNG THE LEVEL-2 VEHICLE PROVABLY CANNOT.
At 19 -> 23 (machine 23, budget width 48) round 25 REFUTED the composed level-2
vehicle with an exhibited exact rational feasible point, saved and re-verified
(`research/data/r25/witness_m23_w48.pkl`; row value
434038501259968447/6799020800000000 ... at 23 -> 29, and the analogous m23
witness at 19 -> 23).  Holding gear 5's phase, ALL FIVE CASES CERTIFY - and at
ITERATION ZERO, i.e. with only the level-1 coverage rows, no cut generation at
all:

    case w      0        1        2        3        4
    |pos|      29       29       29       29       28
    cols     3381     3381     3381     3381     3381
    cert ops 7,525    7,687    8,095    7,873    7,497      total 38,677

Each certificate is an exact rational dual vector; case 0 re-verified from
disk against a fresh rebuild of the relaxation, 202/7 < 607/21.  Every one of
the five carries the RECURSION ROW with strictly positive weight (yff = 8/21,
8/15, 9/16, 5/13, 1), so this is the composed vehicle and not the
consistent-degree-2 LP in disguise; and the margins are thin (451/15 < 452/15,
382/13 < 383/13, 282/7 < 286/7), so width 48 is only just enough.

THE LADDER, WITH THE NUMBER OF HELD GEARS AS ITS PARAMETER.  Every entry below
is a CASE-SPLIT CERTIFICATE at the (D) ladder's own budget width, hypothesis-
free (the input is the list of primes), with an exact rational dual certificate
per case saved to disk and one per rung re-verified against a fresh rebuild:

  rung      W    held        cases   exact certificate ops   re-verified
  19->23   48    (5)             5                  38,677   202/7 < 607/21
  23->29   63    (5,7)          35                 362,049   25 < 26
  29->31   74    (5,7)          35                 576,472   92/3 < 94/3
  31->37   95    (5,7,11)      385               8,388,426   32 < 34
  37->41  129    (5,7,11)      385              12,778,058   95/2 < 97/2

With round 24's four rungs (7->11 .. 17->19, which the level-2 member already
had) that is EVERY (D) STEP THE PROJECT HAS, certified by LP duality with no
census hypothesis anywhere - including 37 -> 41, which round 25 recorded as an
EXACT REFUTATION for the level-2 member.

The parameter is real, not decoration: 23->29 does NOT certify with one held
gear (the LP maximum of the recursion row stalls at 38.316 against the 38 it
must beat after 33 cut passes, 0.83% short), 31->37 does not certify with two
(40.994 against 40, 2.5%), and 37->41 does not certify with one (86.756 against
78, 11.2%) or two (57.281 against 55, 4.1%).  Each extra held gear multiplies
the case count by that gear and buys roughly a halving of the residual.

AND THE VEHICLE BECOMES TIGHT ON F ITSELF.  With two held gears it certifies
F(M) <= F(M), the exact value, at three machines: F(19) <= 25 (107,188 ops),
F(23) <= 34 (202,959 ops), F(29) <= 43 (373,775 ops).  The level-2 member
needed width 33 at machine 19 and was REFUTED at width 48 at machine 23.

AND THE SECOND RUNG FALLS TO TWO HELD GEARS.  23 -> 29 (machine 29, budget
width 63) was the other cell round 25 refuted by an exhibited exact witness.
Holding gear 5 alone does not close it - the LP maximum of the recursion row
falls 39.0662 -> 38.32 over 28 cut passes against the 38 it must beat, a
residual of 0.32, i.e. 0.8% - but holding 5 AND 7 (35 cases) CERTIFIES ALL 35,
362,049 exact certificate ops, 27.5 s, case (0,0) re-verified from disk
(25 < 26).  The same two-gear split then certifies 29 -> 31 as well (budget
width 74, all 35 cases, 576,472 ops, 75.9 s, re-verified 92/3 < 94/3) - a rung
whose only other proof in this project is hypothesis-explicit on a census.
SO THE CASE-SPLIT VEHICLE IS STRICTLY STRONGER THAN THE ONE ROUND
25 CLOSED, the refutations round 25 recorded are refutations of the LEVEL-2
MEMBER only, and the family has a ladder parameter (the number of held gears)
that the level-2 member does not.

RESULT 2 - THE CONDITIONAL PRE-TEST, AND A SECOND INGREDIENT ROUND 25 MISSED.
The conditional uniform product measure decides both sides of a case for free.
(i) THE ROW.  E_u[f_w] is exact and its mean over the cases is exactly round
25's STAR-k number (asserted in the gate).  At every budget width m41..m53 and
every case, k = 1, 2 and 3:

     y    W    cases  min E_u[f_w]   max      mean (= round 25's STAR-k)
    41   129     5      +8.8304    +8.9661     +8.8853
    41   129    35     +13.6538   +14.7904    +14.2963
    41   129   385     +15.9739   +19.1761    +17.6640
    43   134     5      +6.6359    +6.7442     +6.6797
    43   134    35     +12.4104   +13.1870    +12.7830
    47   150     5      +5.0773    +5.1046     +5.0991
    47   150    35     +11.8010   +12.6057    +12.2560
    53   156     5      +3.0700    +3.1439     +3.1065
    53   156    35     +10.4790   +11.2061    +10.8054

  NOT ONE CASE has E_u[f_w] <= 0, so the necessary condition holds case by
  case, not merely on average - which is what a case-split certificate needs
  and what an average could not have shown.
(ii) THE DEGREE-2 SIDE, AND THIS IS NEW.  At machine 41 the UNCONDITIONAL
  uniform product measure IS completable (n = 11) - that is precisely what
  makes round 25's 37 -> 41 refutation work.  The CONDITIONAL one is NOT:
  dropping gear 5 (n = 10) and dropping 5 and 7 (n = 9) both give moment
  vectors with an exactly-verified violated degree-2 cut.  So holding one gear
  revives BOTH of the vehicle's ingredients, and ROUND 25's EXACT REFUTATION OF
  37 -> 41 DOES NOT TRANSFER TO THE CASE SPLIT: that cell is open again for the
  stronger species.

RESULT 3 - THE RESIDUAL IS A LADDER IN k, NOT A WALL, AND MACHINE 41 FALLS AT
k = 3.  The quantity that must drop below |pos| is the LP maximum of the
recursion row.  At machine 41, width 129:

  held      |pos|   LP maximum, first pass -> last measured pass    residual
  (5)         78    87.0713 -> 86.7556  (10 passes, 855 rows)     8.76  (11.2%)
  (5,7)       55    57.6287 -> 57.2808  (12 passes, 692 rows)     2.28  ( 4.1%)
  (5,7,11)    46    CERTIFIED AT ITERATION ZERO, 34,774 ops, 2.1 s

Each held gear roughly halves the relative residual while multiplying the case
count by that gear; the third one crosses.  The same shape continues above the
ladder: at k = 3 machine 43's first case certifies at iteration zero (49,699
ops) while machines 47 and 53 stall at 1.7% and 11.6%, and at k = 4 both of
those first cases certify at iteration zero (54,757 and 81,337 ops).  A random
sample of eight k = 4 cases certifies 8/8 at machine 47 and 6/8 at machine 53,
so 47 -> 53 looks like a k = 5 problem.  Full sweeps above 37 -> 41 were not run
(see section 6).

RESULT 4 - THE WINDOWED VEHICLE IS TIGHT ON F_2 AND SOUND ON THE LEVEL-2
DICTIONARY, WITH A SMALL STRUCTURED INTEGRALITY GAP BELOW F_2.
  (a) THE F_2 LADDER.  Every span W in [32, 66] and every split a in [1, W-1]:
      1,680 cells, of which 413 are killed outright (gear 5 has no phase
      leaving all three required-open positions open) and 1,267 carry an exact
      dual certificate.  ZERO refuted, ZERO undecided; 2,760,053 certificate
      ops in total.  Since every adjacent gap pair has both gaps at most
      F(19), and round 25's composed certificate gives F(19) <= 33, hence
      F_2 <= 66, this is a COMPLETE SCAN-FREE LP PROOF THAT F_2(19) <= 31 -
      and the true value is 31.  TIGHT.  (The case split now certifies
      F(19) <= 25 outright, so the bound F_2 <= 50 is also self-contained.)
      AND THE SAME AT MACHINE 23: with gear 5 held, spans [40,68] give 1,537
      splits, 368 fully vacuous and 1,151 certified, with 18 left over (13 at
      span 40, five at span 42); those 18 all certify with gears 5 AND 7 held
      (1,009,730 ops, 28 s).  Since the case split also certifies F(23) <= 34,
      F_2 <= 68 and the sweep is complete: F_2(23) <= 39, the true value.
  (b) TIGHTNESS AND SOUNDNESS AT THE MAXIMUM.  At span 31 = F_2(19) the
      vehicle fails on EXACTLY two splits - (10, 21) and (21, 10) - each by an
      exact in-polytope witness, and certifies the other 22 live splits.  A
      full-period scan of machine 19 (1,616,615 slots) says the realised
      adjacent pairs of sum 31 are exactly {(10,21), (21,10)}.  The vehicle
      does not merely bound F_2; IT LOCATES THE MAXIMISER.
  (c) AGAINST THE WHOLE DICTIONARY, AND THE HONEST LIMIT.
      `research/window_dict.py` compares every (span, split) cell in a range
      against the 221 realised adjacent gap pairs of machine 19 computed by
      full-period scan.  Over spans 20..31: NO UNSOUND CELL ANYWHERE - every
      CERTIFIED cell and every DEAD cell is genuinely unrealised - with 109
      certified, 94 correctly refuted and 72 killed by gear 5.  But the
      vehicle is NOT exact: NINE unrealised cells are not certified, all at
      spans 28 and 30 - (2,26) and (26,2) at span 28, and (2,28), (4,26),
      (7,23), (15,15), (23,7), (26,4), (28,2) at span 30.  FOUR of them carry
      EXACT in-polytope witnesses - (2,26) and (26,2) at row value 107/4 >= 26
      with slack 3/4, and (4,26) and (26,4) at row value 28 >= 28 with slack 0,
      all saved and re-verified from disk - so they are genuine INTEGRALITY
      GAPS of the relaxation, not budget artefacts; the other five stall.
      Holding gear 5 as well CLOSES THREE OF THE NINE ((2,28), (15,15),
      (28,2)) and leaves the other six.  Note (15,15) among them: the
      self-mirror split, which is exactly the palindromic case Lateral's
      mirror parity law singles out.  So the correct statement is: sound
      everywhere, tight at and above F_2, with a small and structured
      integrality gap below it.

RESULT 5 - THE SAME CONSTRUCT SEES SPECTRUM HOLES.  With OP = {0, W} (both
endpoints open, no interior opening) a certificate says "machine M has no gap
of size exactly W".  At machine 19, whose exact gap-value set is
{1..18, 20, 21, 22, 23, 25} (period scan), the vehicle certifies 24, 27 and 29
- all genuine HOLES, two of them BELOW F(19) = 25 - and refuses at 22, 23, 25,
which are attained.  So the object is finer than F: it is a certificate
species for the gap SPECTRUM, not just its maximum.

RESULT 6 - THE TWO RESTRICTIONS COMPOSE, AND THE COMPOSITION IS STRICTLY
STRONGER THAN EITHER.  They are the same construct with different arguments,
so they can be applied at once: prescribe the open positions AND hold a gear's
phase.  The first test was decisive.  At machine 23, span 40 (one above
F_2(23) = 39), the plain windowed vehicle cannot certify the split (2, 38):
after 61 cut passes and 91 s its LP maximum is 39.7689 against the 38 it must
beat, a 4.7% residual.  With gear 5 held, three of the five cases are VACUOUS
(that phase of gear 5 blocks a required-open position, so the configuration is
impossible outright) and the remaining two CERTIFY - in ONE SECOND.  The
vacuous cases are the point: prescribing open positions does not merely shrink
the obligation, it deletes whole branches of the case split for free.

## 2A. ROUND-27 ADDITIONS (three, all with the same construct)

RESULT 7 - THE NINTH RUNG, 41 -> 43, CASE-SPLIT CERTIFIED.  Round 26 left this
one a PARTIAL SWEEP (163 of 385 cases at k = 3, six stalls at a 45 s/case
budget) and said so.  Round 27 finished it: 228 further cases on ten striped
workers at 240 s/case, and ALL 385 CASES CERTIFY.

  rung      W    held        cases   exact certificate ops   re-verified
  41->43  134    (5,7,11)      385              18,649,193   all 385 from disk

Case (0,0,0) closes 3523/128 < 1763/64; the SMALLEST margin over all 385 cases
is 19/100000, so the budget width 134 is only just enough - as at every other
rung.  Iteration histogram over the 385 cases: 371 certify at ITERATION ZERO
(level-1 coverage rows plus the recursion row, no cut generation at all) and 14
need between 2 and 7 cut passes.  The gate
(`research/gate_rung_41_43_r27.py`) rebuilds each relaxation from the primes,
re-checks every cut row's validity by the exact zeta transform, and re-closes
lhs < rhs in exact rationals, for every one of the 385.

This rung matters beyond arithmetic: it is the step the project's other
scan-free route (Constructor's CRT closure) reported as NOT CERTIFIED in round
26, oracle-bound at arity 4.  So the ladder now stands at TEN rungs - 7->11,
11->13, 13->17, 17->19, 19->23, 23->29, 29->31, 31->37, 37->41, 41->43 - every
(D) step the project has, plus one it did not.

RESULT 8 - THE VEHICLE REACHES THE INCREMENT WIDTH, SO THE MANAGER'S INCREMENT
LAW GETS ITS LITERAL-STEP BASE CASES BY CERTIFICATE.  The round-26 derivation
block conjectured

    F(M + q') - F_2(M)  <=  s_min(q') = min(2u' mod q', (-2u') mod q')

at every literal step (u' = 6^{-1} mod q', the tooth).  The half a DUAL
certificate can carry is the upper half, and it is exactly this vehicle run at
the INCREMENT WIDTH  W_inc = F_2(M) + s_min(q')  instead of at the ladder's
budget width F(M) + q'.  W_inc is strictly smaller at every step, so this is a
STRICTLY HARDER obligation than the corresponding (D) rung and is not implied
by it.  All six literal steps the vehicle reaches CERTIFY:

  step     s_min  F_2(M)  W_inc   budget  k   cases   exact ops   secs
  11->13     4      11      15       20    1     5        4,416      2
  13->17     6      16      22       28    1     5       10,620     <1
  17->19     6      25      31       37    1     5       22,409      1
  19->23     8      31      39       48    2    35      203,921      5
  23->29    10      39      49       63    2    35      365,473     23
  29->31    10      55      65       74    2    35      574,172     55

THE TIGHTER WIDTH COSTS EXACTLY ONE HELD GEAR, AND ONLY WHERE THERE WAS ROOM
FOR IT TO.  19 -> 23 certifies at k = 1 at the budget width 48 (five cases at
iteration zero) and does NOT at W_inc = 39 - case w = 0 stalls - certifying only
at k = 2.  23 -> 29 and 29 -> 31 already needed k = 2 at their budget widths and
still need exactly k = 2 at W_inc, so the extra difficulty is absorbed.  The
ladder parameter is measuring difficulty rather than serving as a knob.

The OTHER half of the increment law - F_2(M) >= W_inc - s_min, i.e. that the
two-gap record really is that large - is a REALISABILITY statement and no dual
certificate can carry it.  It is discharged here by exhibited configurations
instead: `increment_cert_r27.witness_f2` builds an explicit phase vector by an
exact-cover backtrack over the gears (NO PERIOD SCAN), and `check_witness`
re-checks it by CRT arithmetic on [0, s].  Witnesses at machines 11, 13, 17,
19, 23, 29 realise spans 11, 16, 25, 31, 39, 55 - the recorded F_2 values, so
each is tight.  The machine-19 witness has split (10, 21), which is exactly the
maximiser this vehicle located from the dual side in round 26 (RESULT 4); the
machine-29 witness independently reproduces the project's F_2(29) = 55 with no
scan.  So at every literal step through 29->31 the increment law holds by
CERTIFICATE + WITNESS, with no period scan anywhere in the chain.

NOT REACHED: 41 -> 43 at W_inc = 117 (from F_2(41) = 103, s_min(43) = 14).  The
pre-test is passed - E_u[f] is positive in every sampled case at k = 1, 2, 3
(min +5.62, +10.80, +14.01 at W = 117), so nothing REFUTES the species there -
but the LP does not converge: at k = 3, case (0,0,0), the LP maximum of the
recursion row falls 44.2578 -> 43.4856 over fifteen cut passes (654 rows, 377 s)
against the 43 it must beat, about 0.05 per pass and decelerating, and a single
case did not decide in 35 minutes against 10-40 s per case at the budget width
134.  So the vehicle's cost is not a smooth function of the width: it explodes
as W approaches the value being proved, while the product-measure necessary
condition stays healthy.  There are two frontiers, and only the width one has a
closed form (see `product-measure-frontier.md` section 5).  Priced, not
attempted.

RESULT 8b - THE VEHICLE IS TIGHT ON F AT A FOURTH MACHINE, AS PRE-REGISTERED.
Round 26 predicted (E3) that F(31) <= 58 - the exact value, which FAILS at
k = 2 (19 of 35 cases) - would certify at k = 3.  It does: 385/385 cases,
5,294,517 exact ops, ~180 s on four workers, zero failures.  So the tightness
list is F(19) <= 25, F(23) <= 34, F(29) <= 43 (k = 2) and F(31) <= 58 (k = 3),
each the exact value, each scan-free and hypothesis-free.  The monotonicity
prediction (E2) also held where tested: the 29 -> 31 rung at budget width 74,
certified at k = 2, re-certifies at k = 3 (385/385, 5,220,357 ops) - a real test
because the k = 3 case problems are different LPs, not refinements of the k = 2
ones.

RESULT 9 - THE CERTIFICATES CARRY NO DEGREE-2 CUT AT ALL AT THE TWO RUNGS
EMITTED FOR THE KERNEL.  At 19->23 (5 cases) and 29->31 (35 cases) every case
certifies at iteration zero, so every row of every certificate is the BASE CUT
`sum_i x_i >= 1`.  Two consequences: cut validity is valid by inspection rather
than by a 2^n subset-sum check, and since a pair column's mask is not a
singleton, the cut rows contribute NOTHING to pair columns - `a_j = yff*frow_j`
plus link terms there, and only single columns see `y`.  The degree-2 structure
of the vehicle enters these certificates ONLY through the recursion row's
n_ij coefficients.  The machine-readable emission (`research/data/r27/`,
`research/emit_certs_r27.py`, integers only, gated by re-verification from
disk) records this as `rows_all_base_cut`.

## 3. WHY IT MIGHT BE NOVEL

- The move itself.  Covering/packing LP relaxations are normally strengthened
  by adding variables (higher Sherali-Adams or Lasserre levels) or cuts.  Here
  the relaxation is strengthened by SHRINKING THE GROUND SET AND THE VARIABLE
  DOMAINS in a way that is exactly a case split of the original question, and
  the gain is enormous: at 19 -> 23 the level-2 vehicle has an exhibited
  feasible point at width 48 and the five conditional vehicles all certify
  with NO CUTS AT ALL.  The reason is structural and worth stating: in the
  unconditional LP gear 5's blocking is a fractional variable that the
  adversary spreads; conditioning removes those positions from the obligation
  outright.
- The certificate species.  A CASE-SPLIT CERTIFICATE - a finite family of
  exact rational dual vectors, one per phase of a held gear, together
  exhausting the phase space - is a new certificate shape for this project and
  is exactly the finite object a formal verifier can consume (five dual
  vectors and one exhaustive case list, no search transcript).
- The windowed statement decides a DICTIONARY by DUALITY.  The realisability
  of a prescribed gap pattern is normally decided by search (a CRT set-cover
  CSP, `scanfree-certificate.md`, or a period scan).  Here the NEGATIVE
  direction - "this pattern is not realised" - has a short dual certificate,
  and on machine 19 the LP relaxation has NO INTEGRALITY GAP on that family
  over the range tested.  Whether that exactness is a machine-19 accident or a
  law is the entry's main open question.
- The direction of strength is the opposite of the usual one.  For the
  SINGLE-gap statement the composed vehicle is far from tight (it needs width
  33 at machine 19 where F = 25, and cannot reach machine 41's budget at all).
  For the TWO-gap statement, which is a strictly harder statement about a more
  constrained object, the same vehicle is EXACT.  Constraining the
  configuration helps the certificate more than it helps the adversary.

## 4. PROOF

Status: SCRIPT-VERIFIED (finite, exact rational) for every verdict; the
soundness lemma of section 1 and the mixing lemma of section 1 are proved
above in full and are elementary.

Gate: `python research/star_case.py GATE`, which asserts, from scratch:
 1. `RelaxStar(held=(), open=())` is IDENTICAL to round 25's `RelaxCF` at
    machines 11/13/17/19 - same columns, same links, same recursion row, same
    right-hand side.  So the generalisation is a strict extension, not a
    reimplementation.
 2. `case_margin` with no held gear equals `row_decay._Ef` exactly, and the
    MEAN of the conditional margins over the cases equals `row_decay.Ef_star`
    exactly, at machines 23/29/31 - so the case decomposition reproduces round
    25's two independent rows as exact rationals.
 3. `zeta_fast` == `zeta_values` on random instances at n = 3, 5, 7, 9.
 4. `completable_fast` == `completable` at n = 7, 8, 9.
 5. The composed rung certificates reproduce at m11/13/17.
 6. TIGHTNESS at machine 19: span 31 fails on exactly the splits (10,21) and
    (21,10), each by an exact in-polytope witness re-verified from disk.
 7. Machine 19 span 32: 7 splits dead by gear 5, 24 certified, one certificate
    re-verified from disk.

Soundness is also checked against ground truth that the LP never sees: the
full-period scans of machines 17 and 19 (gap-value sets and the 221 realised
adjacent gap pairs) in `research/window_dict.py`, and the requirement that the
case split must NOT certify a width below F - at machine 23 (F = 34) width 33
is not certified in any case.

Nothing rests on a float.  scipy is used for DISCOVERY only: to find a
candidate cut (whose validity is then repaired and asserted exactly), to find
the support of a completion (which is then solved and asserted exactly), and
to point at a candidate dual vector (which is then rounded and the certificate
inequality asserted in exact rationals).  Two infrastructure pieces were
needed to make the exact side affordable and are themselves gated:
  * `zeta_fast` - the subset-sum transform in n 2^{n-1} exact additions in
    place of the superset loop's ~150,000 steps at n = 10;
  * `completable_fast` - completability decided by VERIFYING a float-discovered
    completion exactly (nu >= 0 and A nu = b on a small support) instead of
    running the exact rational simplex on the (subsets x atoms) tableau.
    Measured: n = 8, 1.86 s -> 0.02 s; n = 9, 46.45 s -> 0.27 s; n = 11, a
    call that did not finish in ten minutes -> 16.7 s.  A "completable"
    verdict is an exactly checked completion; anything else falls back to the
    exact oracle, so no verdict rests on the float step.
And one modelling fix: round 25's loop maximised a COMMON additive slack over
all rows, which conflates the coverage rows (right-hand side ~1) with the
recursion row (right-hand side |pos|); measured at m41 the common slack sat at
exactly +0.221818 for six passes while inert rows accumulated and the float LP
went from 2.2 s to 80.7 s.  The loop here maximises the recursion row itself,
which is the quantity the certificate is about.

## 5. IMPLICATIONS

Inside the project.

- The composed vehicle's ladder is NOT closed at four rungs.  19 -> 23 is
  certified by the case split at the ladder's own budget width, with 38,677
  exact operations and no cut generation.  Round 25's refutations remain
  correct as statements about the LEVEL-2 member; they do not bound the family.
- THE CERTIFICATES ARE KERNEL-SHAPED.  A case-split certificate is a finite
  list of exact rationals: for each case, one nonnegative weight per cut row,
  one weight on the recursion row, one signed weight per consistency link, and
  the claim `sum_S max_{j in S} a_j < sum_r y_r (1 - lam^r_0) + yff * |pos|`.
  Checking it needs only rational arithmetic plus, for each cut row, the check
  that its subset-sums are >= 1 over the 2^n atoms - and n <= 9 in every rung
  certified here, so <= 512 atoms per row.  A (D) rung with NO census
  hypothesis at all is therefore plausibly in reach for the Lean lane, where
  the rungs from 29 -> 31 on are currently hypothesis-explicit on censuses.
- A scan-free, dual-certified route to F_2, which is exactly what the chain
  and the merge law consume.  Mechanic computes F_2 by period scan (F_2(47) =
  134 cost a 529 s transfer scan, superseding a standing range [119,141]);
  Constructor decides pattern realisability by CRT set-cover search.  This
  entry gives the same answers as short dual certificates, and at machine 19
  it gives ALL of them, exactly.
- It is the depth-2 member of Mechanic's Q*_J family and generalises to it
  verbatim: OP = {0, a_1, a_1+a_2, ..., W} is a depth-J qualifying window, and
  Mechanic's word-legality condition (middle gaps legal kill letters mod q')
  simply removes cells from the sweep.  Formalist's round-25 correction - that
  the binding obligation is the FAMILY of qualifying-window bounds, not the
  two-gap member - therefore lands on an object this vehicle can address at
  every depth.
- It sharpens what "the frontier is a width" meant.  The width frontier
  W_u(y) governs the vehicle applied to a FULL window.  Restricting the
  obligation changes the frontier, and the two restrictions measured here move
  it in opposite ways: holding a gear raises the row's margin a lot and still
  is not enough at m41; prescribing openings makes the vehicle exact at m19.

Outside it.  If the exactness of Result 4 survives at larger machines, the
statement is that the Sherali-Adams level-2 relaxation of a covering problem
becomes integral once the ground set is punctured at a prescribed pattern -
which is a statement about when a covering LP's integrality gap collapses
under conditioning, and that is not a shape the covering-LP literature is
organised around as far as this agent knows.

## 6. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

- Conjecture (D) / the twin-prime route: the case split certifies EVERY rung
  the project has, through 37 -> 41, hypothesis-free.  Open: 41 -> 43 (three of
  the first 110 k = 3 cases needed more than a 45 s budget; two of those three
  certify at 73 s and 204 s and the third stalls 0.17% short), 43 -> 47 and
  47 -> 53.  (41 -> 43 at k = 3 was run to 163 of 385 cases and STOPPED: 157
  certified, six stalled at a 45 s/case budget, and re-running three of those
  six at 150-200 s certified two and left the third 0.17% short.  A partial
  sweep, not a rung.)  The cost is a primorial in the number of held gears, so k = 4
  (5,005 cases) is affordable and k = 6 (1,616,615) is not - a NEW KIND OF
  LIMIT for this project: not a degree ceiling and not a width frontier.
- IS THE WINDOWED VEHICLE EXACT ON THE LEVEL-2 DICTIONARY IN GENERAL?
  Measured exact at machine 19 over spans 20..66.  Machine 23 and above are
  open.  A proof would say the composed relaxation is integral on
  punctured-window instances.
- Q*_max = F(M + q') (Mechanic's registered conjecture, exact at two anchors):
  the windowed vehicle computes exactly the quantities that conjecture is
  about, from the primes alone.
- Open, and now sharply posed: WHY does the residual halve with each held
  gear, and does it always cross?  If the halving is a law, the number of held
  gears needed at machine y is O(log of the level-2 residual), and the cost is
  the primorial of that many primes - which would turn "does the vehicle reach
  machine y" into an explicit arithmetic question.
- Open: the vehicle is TIGHT ON F at machines 19, 23 and 29 with two held gears
  and not at 31.  Is F(M) <= F(M) certifiable at every machine for k large
  enough?  That would be a certificate-based computation of F itself.

## 7. PRIOR-ART CHECK

NOT YET CHECKED (2026-08-29; this agent has no web access this round).
Terms to search: conditioning / case-split strengthening of Sherali-Adams
relaxations; integrality of covering LPs on punctured ground sets; LP-duality
certificates for forbidden gap patterns in sieve/Jacobsthal problems;
certificates of non-realisability for admissible tuples; "case-split
certificate" in the LP-proof-system literature (this is close to, but not the
same as, a Res(LP)/CP branching proof - the branching here is over the phase
of a single modulus and each branch is a Farkas certificate).
