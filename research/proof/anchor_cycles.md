# Cycles as the unit: the dead-cycle record of the anchor-30 machine (branch 7a)

Prover 7a, 2026-09-05. Scripts in research/anchor235/r34/ (cycle_record.py, cycle_mechanism.py,
sandwich_check.py), results in research/anchor235/r34/results/ (cycle_record_<q>.json,
mechanism_<q>.txt, sandwich_check.txt, run29.log, run31.log). Vocabulary:
docs/proof-search/alignment-rules.md section 0; cycle conventions: docs/proof-search/anchor-235.md
header. Nothing here is committed.

Column k = the pair (6k-1, 6k+1). Gear g >= 5 strikes k iff k = +-u_g (mod g), 6u_g = g -+ 1.
Anchor 30 = {2,3,5}: cycle j = the numbers 30j+11, 13, 17, 19, 29, 31 = the three twin slots at
columns 5j+2, 5j+3, 5j+5; anchor-open columns are k mod 5 in {0,2,3}. Machine M = {5..q}.
Opening = a column no gear strikes. F(M) = the record in the max-gap convention (distance
between consecutive openings; blocked count F_bc = F - 1). The budget inequality
F(M+q') <= F(M) + q' is a target, never a law.

The one-line answer, for the reader in a hurry: the dead-cycle record is the column record
divided by five, exactly and provably - F_c(M) = floor((F(M) - 2)/5) at every machine (proof in
Definitions item 3, exact at eight full periods, {5,7} .. {5..31}) - so the
cycle frame carries no size information the column frame does not have, and no increment bound
with c < 1/5 can come out of it that is not the budget inequality divided by five.

## Pre-registered (written 2026-09-05 before any script was run; verdicts filled after)

### What was reasoned out before computing, and what the computation is for

The predictions below are not guesses; most of them follow from one elementary argument
(the cycle sandwich, section "Definitions" item 3) applied to the corpus record ladder
F = 5, 7, 11, 18, 25, 34, 43, 58, 88, 91, 103, 118, 145, 161 at {5,7} .. {5..59} (max-gap
convention, docs/proof-search/alignment-rules.md 3.2 and mechanic.md 389-394). The scorecard
therefore tests (i) the sandwich argument itself against exact full-period computation,
(ii) the one machine the sandwich [as I first wrote it] does not decide ({5..23}, F = 34 = 4 mod 5),
(iii) the things the argument says nothing about: where the dead-cycle record sits, which gears
kill which slots there, what the new gear's kills look like in cycle units, and whether the
class q' mod 30 shows at all.

Definitions used in the predictions. F_c(M) = the longest run of consecutive DEAD cycles
(all three twin slots blocked), counted in cycles (a blocked count; the corresponding
max-gap form between live cycles is F_c + 1). H_1(M) = the longest run of consecutive cycles
with at most one open slot ("wall-free run"; a cycle with two or more open slots is a wall
that no single gear >= 11 can kill, since such a gear kills at most one number per cycle).

### Predictions and scorecard

P1 (the sandwich, exact values). F_c(M) = floor((F(M) - 2)/5) at every machine whose record
F(M) is not 4 mod 5; when F = 4 mod 5, F_c = floor((F-2)/5) if the machine has a gap of size
F-1 or F-2 and floor((F-2)/5) - 1 otherwise. Predicted values, exact full periods:
{5}: 0; {5,7}: 0; {5..11}: 1; {5..13}: 1; {5..17}: 3; {5..19}: 4; {5..23}: 6 (F = 34 = 4 mod 5;
the record needs a gap of 33 or 32, and anchor-235.md section 9 records two-kill chains of
blocked count 32, i.e. a gap 33, at rung 23, so 6); {5..29}: 8 (computed exactly this round;
215,656,441 cycles). From the corpus ladder without a scan: {5..31}: 11; {5..37}: 17;
{5..41}: 17; {5..43}: 20; {5..47}: 23; {5..53}: 28; {5..59}: 31.
  VERDICT: VALUES CONFIRMED at every full period, 0, 1, 1, 3, 4, 6, 8 at {5,7} .. {5..29}, and
  11 at {5..31} (6.7e9 cycles; see Results). The CONDITIONAL CLAUSE WAS WRONG: for a record gap
  starting at a column 3 mod 5 the number of whole cycles inside is floor((F-3)/5), not
  floor((F-5)/5) as I had it, so F = 4 mod 5 needs no gap of size F-1 or F-2 - the record gap
  itself holds floor((F-2)/5) cycles (at {5..23} all four gaps of 34 start at 3 mod 5 and each
  holds 6 dead cycles). The sandwich is unconditional: F_c = floor((F-2)/5) always. The slip
  was found when the {5..23} output showed a 6-run inside the gap of 34 at column 12694428.

P2 (where the record sits). At every machine with F not 4 mod 5 the dead-cycle record is
attained by the column record stretch or its mirror image (the one of the pair whose first
opening is 0 mod 5, i.e. sits on slot 29|31), so it IS the slot record coarsened. At {5..23}
it is attained by a gap of 33 (or 32), NOT by the record gap 34, whose phase (start 3 mod 5,
forced by 34 = 4 mod 5) contains only 5 whole cycles. Other stretches may tie.
  VERDICT: FIRST HALF CONFIRMED (sandwich_check.txt section 2: at {5..11}, {5..17}, {5..29} the
  record gaps come in mirror pairs at phases {0,3}, {0,2}, {0,2}, and only the phase-0 member
  holds floor((F-2)/5) cycles; at {5..13} and {5..19} every record gap holds it). SECOND HALF
  REFUTED: the gap of 34 at phase 3 holds 6 whole cycles; the nine 6-runs at {5..23} are the
  four record gaps of 34 (phase 3) plus the gap of 33 at phase 0 plus four gaps of 32 at
  phase 0. Ties with shorter gaps do occur (as predicted): 5 of 9 at {5..23}, 0 of 1 at {5..29}.

P3 (increment against q'/5). F_c(M+q') - F_c(M) <= q'/5 at every rung 7->11 .. 23->29
(exact) and at every corpus rung to 53->59. Predicted increments: +1, 0, +2, +1, +2, +2 at
7->11 .. 23->29; then +3, +6, 0, +3, +3, +5, +3 at 29->31 .. 53->59. Mechanism, stated now:
the sandwich gives F_c(M+q') - F_c(M) <= (F(M+q') - F(M) + 3)/5 + 1, so the cycle increment is
the column increment divided by 5 with a rounding term, and the column increment is <= q' on
every rung of the record. There is no cycle mechanism in it.
  VERDICT: CONFIRMED, all 13 rungs, increments exactly as predicted (Results table 2).

P4 (increment against q'/15). FAILS, at 7->11 (1 > 0.73), 13->17 (2 > 1.13), 19->23
(2 > 1.53), 23->29 (2 > 1.93), 29->31 (3 > 2.07), 31->37 (6 > 2.47), 41->43 (3 > 2.87),
47->53 (5 > 3.53). The "q' x delta/30 cycles between a gear's hits" spacing (delta the gap
between consecutive open multipliers m) does not bound the increment because the new gear's
kills in a record chain are not consecutive hits of the gear: they are at column distances
+-2u' or q' apart (the literal and padded letters), i.e. multiplier differences 2, 4 or 6,
and any two such kills are legal as long as the column gap is 0 or +-2u' mod q'.
  VERDICT: CONFIRMED - fails at exactly those eight rungs and holds at the other five.

P5 (the best c on the record). The smallest c with F_c(M+q') <= F_c(M) + c q' on every
computed rung is set by 31->37: c = 6/37 = 0.162 < 1/5. This is numerically true and
mechanically empty: it is the column increment 30 = 0.81 x 37 at that rung divided by 5.
Any c < 1/5 as a law would be the statement F(M+q') - F(M) <= 5c q' + O(1), a sharpened
budget inequality, which the column frame does not have either.
  VERDICT: CONFIRMED (0.162 at 31->37; next 0.118 at 13->17).

P6 (class q' mod 30). No dependence of the increment on the class of q' (+-1, +-7, +-11,
+-13) beyond what F(M+q') - F(M) already carries. Mechanism: the six kill residues of gear q
on the cycle index have the class-free closed form R_q = -30^{-1} x {11,13,17,19,29,31}
(mod q) (derived in Definitions item 2); the class only permutes which multiplier m lands on
which residue. With 13 rungs the class means are not distinguishable; the largest increment
(31->37, class +7) is matched by ordinary increments at 23 and 53 (class -7).
  VERDICT: CONFIRMED as far as 13 rungs can test it (class +-7: 0.087, 0.162, 0.094; +-1: 0.069,
  0.097, 0.051; +-11: 0.091, 0.053, 0.000; +-13: 0.000, 0.118, 0.070, 0.064); the closed form
  checked at all 300 primes 7..2000 with 0 mismatches.

P7 (the new gear's kills at the dead-cycle record of M+q'). At each rung 7->11 .. 23->29:
(i) q' kills exactly one slot in every cycle it touches (theorem for q' >= 11: two multiples
of q' in one 30-number cycle would be <= 20 apart); gear 7 is the one gear that double-kills,
at j = 2 mod 7 (77 = 7x11 and 91 = 7x13 in cycle 2), so at 5->7 a dead cycle can have two
killing gears, from 11 on every dead cycle has at least three distinct killing gears.
(ii) The slots q' kills alone in the record run (the interior M-openings the chain kills) are
1, 1, 2, 1, 3, 2 in number at +11, +13, +17, +19, +23, +29 when the dead-cycle record is the
column record coarsened (anchor-235.md section 9 kills column), and their consecutive column
distances are +-2u' mod q' (literal), never a multiple of q', at these six rungs.
(iii) In cycle units two q'-kills at numbers q'm_1, q'm_2 with in-cycle offsets e_1, e_2 are
(q'(m_2 - m_1) + e_1 - e_2)/30 cycles apart, an identity, so "on the lattice" cannot fail;
the content is that the multiplier difference m_2 - m_1 of consecutive chain kills is 2 or 4
(literal) or 6 (padded) at these rungs.
  VERDICT: (i) one-kill-per-cycle CONFIRMED (sandwich_check section 7: only q = 7 has a residue
  with two kills, j = 2 mod 7, offsets 17 and 31); the clause "from 11 on every dead cycle has
  at least three distinct gears" is REFUTED - the gear-7 double never goes away, so cycles
  j = 2 mod 7 die under two gears at every machine (9 of 30 record-run cycles at {5..17}, 2 of 54
  at {5..23}, 0 of 8 at {5..29}). (ii) CONFIRMED on the runs the new gear makes: 1, -, 2, 2, 3, 2
  sole kills at +11, +13, +17, +19, +23, +29 with all 79 consecutive-kill column distances in
  {+2u', -2u'} and none padded; at +13 the record F_c = 1 is not made by 13 at all (F_c stays 1;
  20 of the 76 single dead cycles were already dead under {5..11}), and at +19, 27 of 76 record
  runs carry one kill of 19 rather than two. (iii) CONFIRMED (79/79 identities); multiplier
  differences are 2 (for +2u') and 4 (for -2u') only, and the two alternate (T3).

P8 (the wall bound is dead as a route). H_1(M) (longest run of cycles with <= 1 open slot)
is an upper bound on F_c(M+q') for EVERY single added gear q' >= 11, with no q' in it, but it
exceeds F_c(M) + q'/5 at every machine from {5..11} on, by a growing margin, so it certifies
no rung. Its column analogue is the T4 spacing bound (consecutive kills >= 2u' apart), which
is stronger since 2u' >= 4 > 3 and does not need the two openings to share a cycle.
  VERDICT: CONFIRMED. H_1 = 1, 4, 6, 8, 17, 35, 48 at {5,7} .. {5..29} against F_c + q'/5 =
  2.2, 3.6, 4.4, 6.8, 8.6, 11.8, 14.2; certifies 7->11 only.

P9 (coarse-graining loses exactly this). (i) Size: F_c = c pins F only to the ten values
5c+2 .. 5c+11. (ii) The chain law: in columns, two consecutive kills of q' differ by one of
3 residues mod q' (0, +-2u'); in cycle index alone (slot and member forgotten) they differ by
one of the 21 residues (E - E)/30 mod q', E = {11,13,17,19,29,31}, so the cycle-only law is
7 times weaker for q' >= 23. (iii) The gap spectrum: gaps of one column size g fold into
cycle runs of two lengths depending on phase.
  VERDICT: CONFIRMED, and sharpened by the unconditional sandwich: (i) F_c = c pins F to
  5c+2 .. 5c+6 (five values, not ten); (ii) 21 residues of q' for q' >= 23, and ALL residues
  for q' = 11, 13, 17, 19 (the cycle-only chain law says nothing at all there); (iii) as stated.

Scorecard: 9 predictions; 7 confirmed as written (P3, P4, P5, P6, P8, P9 sharpened, P7 in its
two computational clauses), 2 with a refuted clause (P1's conditional case, P2's {5..23} clause,
both the same arithmetic slip in the phase-3 count; P7(i)'s "three gears from 11 on").

## Definitions

1. Frames. Cycle j <-> columns 5j+2, 5j+3, 5j+5 (numbers 30j + {11,13}, {17,19}, {29,31});
   the columns 5j+1, 5j+4 are the anchor's own kills (k = +-1 mod 5, gear 5's teeth). The
   period of {5..q} is P = 5 x prod_{7<=g<=q} g columns = prod g cycles: 7, 77, 1001, 17017,
   323323, 7436429, 215656441, 6685349671 cycles at {5,7} .. {5..31}. (The brief's "37182145 for
   {5..23}" is the column period 5 x 7436429.) A slot is open iff no gear divides either
   member; a cycle is dead iff all three slots are blocked; F_c(M) = the longest run of
   consecutive dead cycles; H_1(M) = the longest run of cycles with at most one open slot.
   Column 0 (the pair (-1, 1)) is slot 29|31 of cycle P_c - 1 and is open at every machine, so
   no dead run wraps around the period.

2. Closed form of the kill residues (derived; sandwich_check.txt section 5). Gear g kills the
   number 30j + e iff 30j + e = 0 (mod g) iff j = -e x 30^{-1} (mod g). Hence, with
   E = {11, 13, 17, 19, 29, 31},

       R_g = { -e x 30^{-1} mod g : e in E }           (six residues; five for g = 7),

   per slot S_{g,t} = -30^{-1} x {e_t, e_t + 2} with e_t = 11, 17, 29, and in column terms
   j = (+-u_g - o) x 5^{-1} (mod g) for the slot at column offset o in {2,3,5} (the script
   asserts the two forms equal for every gear). The corpus form ((g x m - 11) div 30) mod g
   over the six anchor-open multipliers m of the class g mod 30 is the same set (0 mismatches
   at the 300 primes 7..2000): g x m = 30c + e gives c = -e x 30^{-1} (mod g), so the class
   fixes only WHICH multiplier m lands on WHICH offset e; the residue set is class-free.
   Instances (offset -> residue): g = 7: 11->5, 13->4, 17->2, 19->1, 29->3, 31->2 (R_7 =
   {1,2,3,4,5}; offsets 17 and 31 share the residue 2 - the one double kill, 77 = 7x11 and
   91 = 7x13 in cycle 2); g = 11: 11->0, 13->8, 17->2, 19->10, 29->6, 31->3; g = 23: 11->5,
   13->8, 17->14, 19->17, 29->9, 31->12; g = 29: 11->18, 13->16, 17->12, 19->10, 29->0, 31->27;
   g = 31: 11->11, 13->13, 17->17, 19->19, 29->29, 31->0. These reproduce anchor-235.md
   section 5's table. For g >= 11 the six residues are distinct (two offsets e, e' share a
   residue iff g | e - e', and |e - e'| <= 20), so a gear >= 11 kills at most one number per
   cycle; gear 7 kills two in cycles j = 2 mod 7 and one in j = 1, 3, 4, 5 mod 7.

3. The cycle sandwich (theorem, elementary). For every machine M containing gear 5,

       F_c(M) = floor( (F(M) - 2) / 5 ).

   Upper bound. A run of c dead cycles j_0 .. j_0+c-1 blocks the columns 5j_0+2 .. 5j_0+5c;
   the columns 5j_0+1 and 5j_0+5c+1 are 1 mod 5, blocked by gear 5; so the openings on either
   side are at most 5j_0 and at least 5j_0+5c+2, a gap >= 5c + 2, whence F >= 5F_c + 2.
   Lower bound. Take a record gap from an opening x to x + F. Both ends are openings, so
   x mod 5 and (x + F) mod 5 lie in {0, 2, 3}. The blocked columns are x+1 .. x+F-1 and the
   whole cycles inside number floor((F-1)/5), floor((F-4)/5), floor((F-3)/5) for
   x = 0, 2, 3 (mod 5) (cycle j needs 5j+2 >= x+1 and 5j+5 <= x+F-1). Write F = 5a + f.
     f = 0: x in {0,2,3}, counts a-1, a-1, a-1 = floor((F-2)/5).
     f = 1: x = 2 only (x+F must avoid 1, 4), count a-1 = floor((F-2)/5).
     f = 2: x in {0, 3}, counts a, a-1; floor((F-2)/5) = a.
     f = 3: x in {0, 2}, counts a, a-1; floor((F-2)/5) = a.
     f = 4: x = 3 only, count a = floor((F-2)/5).
   In the cases f = 2, 3 the mirror k -> -k (mod P) - the opening set is symmetric because
   the teeth are +-u_g, and 5 | P - sends the gap (x, x+F) to the gap (-x-F, -x), whose start
   is -(x+F) = -(3+2) = 0 or -(2+3) = 0 (mod 5): every record gap starting at phase 3 (f = 2)
   or 2 (f = 3) has a mirror record gap starting at phase 0, which holds a cycles. So some
   record gap holds floor((F-2)/5) whole dead cycles, F_c >= floor((F-2)/5), and the upper
   bound gives equality. Checked exactly (sandwich_check.txt sections 1-2): every record gap
   of every machine {5,7} .. {5..31} holds exactly the phase count above, every record gap's
   mirror is a record gap, and F_c equals the formula at all eight machines.
   Corollary (the increment). F_c(M+q') - F_c(M) <= (F(M+q') - F(M) + 3)/5 + 1 in general and
   = floor((F'-2)/5) - floor((F-2)/5) exactly; the budget F' - F <= q' gives the cycle
   increment <= floor((q'+3)/5) + 1, and any c < 1/5 with F_c' <= F_c + c q' at a rung is the
   statement F' - F <= 5c q' + 6 at that rung, i.e. the budget with slack.

4. The chain law in cycle units (identity, not a law). Two kills of q' at numbers
   q'm_1 = 30j_1 + e_1 and q'm_2 = 30j_2 + e_2 satisfy 30(j_2 - j_1) + (e_2 - e_1) = q'(m_2 - m_1)
   identically; the column chain law (kernel, AnchorChain.chain_law) says consecutive kills in a
   chain have column difference 0 or +-2u' (mod q'), which is multiplier difference 0, 2, 4
   (mod 6) with the member parity fixed. Forgetting slot and member, the admissible cycle
   spacings j_2 - j_1 (mod q') are (E - E) x 30^{-1}: E - E = {0, +-2, ..., +-20}, 21 values,
   giving 21 residues of q' for q' >= 23 and every residue for q' = 11, 13, 17, 19.

## Results

Table 1 - exact full periods (cycle_record_<q>.json; dead density against the CRT
inclusion-exclusion 1 - 3 pi_1 + sum pi_2 - pi_3, exact to the cycle at every machine):

  machine   cycles/period   dead cycles    density   F   F mod 5  F_c  floor((F-2)/5)  runs of F_c  record gaps: phase(count)   H_1   seconds
  {5,7}               7             0      0.0000    5      0      0        0             -           2(1) 3(1)                  1     0
  {5..11}            77             2      0.0260    7      2      1        1             2           0(2) 3(2)                  4     0
  {5..13}          1001            76      0.0759   11      1      1        1            76           2(12)                      6     0
  {5..17}         17017          2162      0.1270   18      3      3        3            10           0(10) 2(10)                8     0
  {5..19}        323323         57488      0.1778   25      0      4        4            76           0(20)                     17     0
  {5..23}       7436429       1648234      0.2216   34      4      6        6             9           3(4)                      35     2
  {5..29}     215656441      55387556      0.2568   43      3      8        8             1           0(1) 2(1)                 48    24
  {5..31}    6685349671    1936500146     0.2897   58      3     11       11             2           0(2) 2(2)                 61   787

  Whole dead cycles inside each record gap, by start phase: {5..11} (0,1) (3,0); {5..13} (2,1);
  {5..17} (0,3) (2,2); {5..19} (0,4); {5..23} (3,6); {5..29} (0,8) (2,7); {5..31} (0,11) (2,10) - exactly the
  formula floor((F-1)/5), floor((F-4)/5), floor((F-3)/5) at phases 0, 2, 3. Which record gaps
  hold the full count: all of them when F = 0, 1, 4 (mod 5); the phase-0 half of the mirror
  pairs when F = 2, 3 (mod 5). Ties from shorter gaps: {5..23} has 5 of its 9 six-runs inside
  gaps of 32 and 33 at phase 0; {5..29}'s single 8-run sits in the phase-0 record gap at column
  200906185 (cycles 40181237..40181244, numbers 1205437121..1205437351).

Table 2 - the increment (exact where a full period was run, corpus ladder elsewhere):

  rung      F->F'    F'-F   F_c->F_c'  dF_c   q'/5   q'/15   <= q'/5   <= q'/15   dF_c/q'   class   source
   7->11     5->7      2     0->1       1     2.20   0.73    yes       no         0.091     +11     exact
  11->13     7->11     4     1->1       0     2.60   0.87    yes       yes        0.000     +13     exact
  13->17    11->18     7     1->3       2     3.40   1.13    yes       no         0.118     -13     exact
  17->19    18->25     7     3->4       1     3.80   1.27    yes       yes        0.053     -11     exact
  19->23    25->34     9     4->6       2     4.60   1.53    yes       no         0.087      -7     exact
  23->29    34->43     9     6->8       2     5.80   1.93    yes       no         0.069      -1     exact
  29->31    43->58    15     8->11      3     6.20   2.07    yes       no         0.097      +1     exact
  31->37    58->88    30    11->17      6     7.40   2.47    yes       no         0.162      +7     ladder
  37->41    88->91     3    17->17      0     8.20   2.73    yes       yes        0.000     +11     ladder
  41->43    91->103   12    17->20      3     8.60   2.87    yes       no         0.070     +13     ladder
  43->47   103->118   15    20->23      3     9.40   3.13    yes       yes        0.064     -13     ladder
  47->53   118->145   27    23->28      5    10.60   3.53    yes       no         0.094      -7     ladder
  53->59   145->161   16    28->31      3    11.80   3.93    yes       yes        0.051      -1     ladder

  The q'/5 line holds at all 13 rungs; it is the budget inequality divided by five (Definitions
  item 3, corollary). The q'/15 line fails at 8 of 13. The largest dF_c/q' is 0.162 at 31->37,
  which is 30/37 = 0.81 of a column budget divided by five. The "ladder" rows are exact
  consequences of the sandwich and the corpus F; they need no scan.

Table 3 - the wall bound H_1(M) against F_c(M) + q'/5: 1 vs 2.2 (certifies 7->11), 4 vs 3.6,
6 vs 4.4, 8 vs 6.8, 17 vs 8.6, 35 vs 11.8, 48 vs 14.2, 61 vs 18.4 - fails from 11->13 on and the
ratio grows (H_1/F_c = 4, 6, 2.7, 4.3, 5.8, 6.0, 5.5). Its longest runs: {5..19} at cycle 154791 (17),
{5..23} at 1943115 (35), {5..29} at 16017702 (48).

## Mechanism at the record (the machine's own terms first)

The {5..29} record run (mechanism_29.txt; the only run of 8; cycles 40181237..40181244,
numbers 1205437121..1205437351, columns 200906187..200906225):

  j          j mod 7   30j+11        +13         +17         +19         +29         +31        gears        open under {5..23}
  40181237      5      7/19          11          -           13          17          -          7,11,13,17,19    none
  40181238      6      -             23          -           19          -           29         19,23,29         slot 29|31 (29 kills 31-offset)
  40181239      0      13            -           -           11          23          -          11,13,23         none
  40181240      1      11            -           -           7           29          -          7,11,29          slot 29|31 (29 kills 29-offset)
  40181241      2      17            -           7           -           13          7          7,13,17          none
  40181242      3      -             19          11          -           7           23         7,11,19,23       none
  40181243      4      -             7           -           17          -           11         7,11,17          none
  40181244      5      7             -           13/23       -           19          -          7,13,19,23       none

What the machine does here. Under {5..23} the eight cycles read dead, live, dead, live, dead,
dead, dead, dead, and each live cycle has exactly ONE open slot, both times the slot 29|31.
Gear 29 kills the 31-member of the first (1205437171 = 29 x 41566799) and the 29-member of the
second (1205437229 = 29 x 41566801). The multipliers 41566799 = 29 (mod 30) and 41566801 = 1
(mod 30) are CONSECUTIVE entries of gear 29's open-multiplier list for its class (-1: m = 1, 11,
13, 17, 19, 29; the wrap 29 -> 31 is the delta = 2 step of anchor-235.md section 2), so the two
glue kills are two consecutive hits of the gear, 2 x 29 = 58 numbers = 10 columns apart, two
cycles apart. What forces it: the kills must lie on the two teeth alternately at column distance
2u' = 10 (chain law, kernel) and every cycle between them must already be dead under {5..23}
(cycle 40181239 is, by 13, 11, 23); a kill at the next open multiplier but one (delta = 6,
padded, 29 columns) would need an old gap of 29 columns inside the run, and F({5..23}) = 34
permits it in principle but no such configuration is realised at the record. Every dead cycle
in the run is killed by 3, 4 or 5 distinct gears (3: five cycles, 4: two, 5: one); none uses
gear 7's double kill (that needs j = 2 mod 7, which is cycle 40181241 - there 7 kills the
17-offset and the 31-offset, and the cycle still needs 13 and 17 for its other slot).

The {5..23} record runs (9 runs of 6; mechanism_23.txt). First run, cycles 2538886..2538891
(numbers 76166591..76166761): three glue kills by 23 at offsets 13, 29, 31 of cycles 2538886,
2538887, 2538890, multipliers 3311591, 3311593, 3311597 (class -7 list m = 7, 11, 13, 17, 19,
23: the multipliers are 11, 13, 17 mod 30 - again consecutive entries of the gear's list, steps
2 then 4), columns 12694432 -> 12694440 -> 12694455, distances 8 = +2u' and 15 = -2u'; the
cycles 2538888 (gear 7's double, 76166657 and 76166671, with 11 on the 11-offset), 2538889 and
2538891 were already dead under {5..19}. Over the nine runs: sole kills 3, 3, 3, 3, 3, 2, 3, 3,
3; the 17 consecutive-kill distances are +2u' (8) and -2u' (9), alternating inside each run,
never 23; distinct gears per dead cycle 2 (two cycles, both j = 2 mod 7), 3 (34), 4 (18).

The {5..19} record runs (76 runs of 4): the first is cycles 22..25, numbers 671..781, right
after 19^2: 19 kills 703 = 19 x 37 (cycle 23, offset 13) and 779 = 19 x 41 (cycle 25, offset 29),
multipliers 37, 41 = 7, 11 (mod 30) - consecutive in the class -11 list m = 1, 7, 11, 19, 23, 29 -
columns 117 -> 130, distance 13 = +2u' (2u' = 32 = 13 mod 19); cycles 22 and 24 dead under
{5..17} by 11, 7, 13 and 17, 11, 7. Across the 76 runs: 49 have two kills of 19, 27 have one
(the run is M-dead cycles plus one glue); the 51 kill distances are +2u' (28) and -2u' (23).
At {5..17} (10 runs of 3): two kills of 17 in every run, 6 = +2u' (six runs, one cycle apart)
or 11 = -2u' (four runs, two cycles apart). At {5..13}: F_c = 1 = F_c({5..11}), 56 of the 76
single dead cycles made by one kill of 13, 20 already dead under {5..11}. At {5..11}: two
dead cycles, 30 and 44, both j = 2 mod 7 (gear 7's double) plus one kill of 11.

What repeats at every rung: the dead-cycle record of M + q' is a run of M-dead cycles glued by
one-open cycles whose single open slot sits on a tooth of q'; the glue kills are consecutive
entries of q''s open-multiplier list (multiplier steps 2 or 4, i.e. numbers 2q' or 4q' apart,
columns +-2u' apart, alternating), never a skipped entry (padded); and the M-dead cycles need
three distinct gears each unless j = 2 mod 7, where gear 7 kills two slots (17- and 31-offset:
7 x (m, m+2) with m = 11 mod 30) and one more gear suffices. The gear-7 double is the only
two-gear death and it persists at every machine (it is a residue class of j).

## What is new, what is known in cycle coordinates, and what use each is to the route

Checked against docs/novel/README.md (no entry mentions dead cycles, a cycle record, or a
coarse-grained record; the nearest entries are mirror-parity-laws, whose k -> -k symmetry the
sandwich uses, and paired-hlb-cycles, which counts survivors per cycle and not runs).

N1. THE RECORD-GAP PHASE LAW (no located prior art in the project; elementary). The start
   column x of any record gap satisfies x = 0, 2, 3 (mod 5) AND x + F = 0, 2, 3 (mod 5), so
   F mod 5 fixes where a record gap can start: F = 1 (mod 5) forces slot 11|13 (x = 2), F = 4
   forces slot 17|19 (x = 3), F = 2 allows {0, 3} and F = 3 allows {0, 2} as mirror pairs, F = 0
   allows all three. Observed exactly (Table 1): {5..13} all twelve at 2; {5..23} all four at 3;
   {5..11}, {5..17}, {5..29}, {5..31} mirror pairs; and at {5..19} (F = 25, all phases allowed)
   all twenty record gaps start at 0 - slot 29|31 at both ends - which the phase law does not
   force. Use to the route: POSITION only (which slot a record gap sits on), subject to the
   escape-distance-1 verdict on bounded-modulus arithmetic; no size content.

N2. THE CYCLE SANDWICH F_c(M) = floor((F(M) - 2)/5) (no located prior art; elementary; exact at
   eight full periods, {5,7} .. {5..31}). Its use to the route is NEGATIVE and final for this
   branch: the dead-cycle record carries no information beyond F, so no cycle-frame increment
   bound can be sharper than the budget inequality divided by five, and the largest chunk the
   anchor offers (the cycle) is a lossy relabelling of the column.

N3. The three-gear structure and the gear-7 double (no located prior art as a statement about
   dead cycles; the residue arithmetic is the tooth rule). A dead cycle needs kills from three
   distinct gears unless j = 2 mod 7, where two suffice; at the {5..29} record 3, 4, 5 gears per
   cycle. Use: none found - the counting bound it yields, c(3 - 6 sum_{7<=g<=q} 1/g) <= 6 pi_7(q),
   is the column counting bound rescaled and is vacuous from {5..29} (6 sum 1/g = 3.005 > 3).

K1. The glue kills at consecutive open multipliers, steps 2 or 4, alternating: this is the chain
   law (AnchorChain.chain_law, kernel) and T3 in multiplier coordinates (2u' columns = 2q'
   numbers = multiplier step 2; q' - 2u' columns = 4q' numbers = step 4; q' columns = step 6,
   padded). KNOWN in cycle coordinates - stopped after the record check (79/79 pairs literal).

K2. The class-free closed form R_g = -30^{-1} E (mod g) of anchor-235.md section 5's residue
   table (300 primes, 0 mismatches). A derivation of a table already on record; the one
   addition is that the class g mod 30 permutes multipliers onto offsets and does not change
   the set, which is why no class effect on the increment exists (Table 2).

K3. The lattice statement 30 dj + de = q' dm is an identity; "on the lattice without exception"
   cannot fail and decides nothing.

## Verdict

New: N1 (record gaps can only start on the slot F mod 5 dictates; at {5..19} they start only on
slot 29|31 although the law allows all three) and N2 (the dead-cycle record is exactly
floor((F - 2)/5), exact at {5,7} .. {5..31}). Both are elementary, both are position facts, and
N2 is the reason the branch closes: the cycle frame is the column frame divided by five with
nothing left over, so F_c(M+q') <= F_c(M) + c q' with c < 1/5 is true on the record (best
c = 0.162 at 31->37) only because F' - F = 0.81 q' there, and any such c as a law would be a
sharpened budget inequality, which the column frame does not have either. F_c is F/5 in
disguise, plainly and provably.

Proved (script-verified where finite): the sandwich (Definitions item 3; exact at eight
machines); the phase law (same argument); one kill per cycle for gears >= 11 and the gear-7
double at j = 2 mod 7; the class-free residue set (300 primes).

Refuted: q'/15 as an increment bound (8 of 13 rungs); the wall bound H_1 as a certificate
(fails from 11->13, ratio to F_c growing to 6); any class q' mod 30 effect; two of my own
pre-registered clauses (the conditional F = 4 mod 5 case - an arithmetic slip in the phase-3
count - and "three gears from 11 on").

For the tree: branch 7a DEAD as a route to the budget inequality; N1 and N2 kept as exact
position/identity facts; the cycle frame recorded as a lossy relabelling of the column frame.
No gap: the {5..31} period (6,685,349,671 cycles, 787 s) completed and agrees.

## Dead ends

- D1. The dead-cycle record as an object with content of its own: DEAD by N2 (exact at
  {5,7} .. {5..31}; no refuting instance exists).
- D2. F_c(M+q') <= F_c(M) + c q' with c < 1/5 as a law: DEAD - numerically c = 0.162, mechanically
  the column increment at 31->37 divided by five (corollary in Definitions item 3).
- D3. The q'/15 lattice bound (a gear's consecutive hits q' x delta/30 cycles apart): DEAD - fails
  at 7->11, 13->17, 19->23, 23->29, 29->31, 31->37, 41->43, 47->53; the glue kills ARE consecutive
  hits (delta = 2 or 4) but a record run holds several of them plus the M-dead cycles between.
- D4. The wall bound H_1(M) (a two-open cycle cannot be killed by one gear >= 11): DEAD -
  H_1 = 4, 6, 8, 17, 35, 48, 61 at {5..11} .. {5..31} against F_c + q'/5 = 3.6 .. 18.4; the column
  form (kills >= 2u' >= 4 apart, T4) is on record and stronger.
- D5. Three-gear counting: DEAD - the column counting bound rescaled, c <= 178 against 6 at
  {5..23}, vacuous from {5..29}.
- D6. Class dependence of the increment on q' mod 30: DEAD - residue set class-free, no signal
  in 13 rungs.
- D7. The cycle-only chain law as a residue constraint: DEAD - vacuous for q' <= 19 (every
  residue admissible), 21 of q' residues beyond, against the column law's 3.
