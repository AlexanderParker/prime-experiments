# Branch 5g - the coverage profile and the hinge

Prover, round 36 (2026-09-05). Parent: node 5 ("made at the top"), through the two observations
its children made at the top of the tree:

- 5d.i's **coverage-maximality split** (research/proof/record_frame.md section 6): in every record
  stretch of every machine m13..m31 gear 5 sits at the phase where it covers the most columns of
  the stretch, gears 7 and 11 do from m19 on, and the top one or two gears never do.
- 5d.ii's **holder profile in the window** (research/proof/deletion_profile.md R2, R4): gear 5 is
  the largest-drop holder of the window's longest blocked stretch at 151 of 165 rungs and a holder
  at 164; the stretch is held up by a chosen fifth of the gears, ordered by column position; at
  rung 997 the smallest initial segment that blocks it is {5..877}, forced by the single column
  (851567, 851569) whose only striker is gear 877.

Two theories, one document. Theory A asks whether the coverage split is a **profile** - a fixed
shape in the gear's rank - and whether such a shape says anything about the length of a stretch.
Theory B asks about the columns that a single gear holds alone (**hinges**) - how big that gear
is, where the column sits, and whether the stretch's length is tied to the hinge gear's size by a
rule with no exception.

Scripts research/anchor235/r36/ (self-contained, numpy only, `uv run python <script>` from the
repository root); results in research/anchor235/r36/results/ (untracked). Nothing committed.

Vocabulary as docs/proof-search/alignment-rules.md section 0 fixes it. Column k = (6k-1, 6k+1);
gear g strikes k iff k = +-u_g (mod g) with u_g = 6^{-1} mod g; machine M = {5..q}; period
P = prod g; an opening is a column no gear strikes; F(M) = the longest gap between consecutive
openings (max-gap convention, wrap included), so the record STRETCH is the F-1 blocked columns
between them. The window at rung q is the columns (q/6, W], W = (q'^2-1)/6 with q' the next prime;
a stretch is a sliding run; the window is the certified range and never a sliding run.

## Pre-registered (written before any script of this round was run)

### Definitions fixed before testing

For a stretch of L consecutive columns starting at column s:

- **coverage** c_g(s, L) = #{ j in [0, L) : s + j = +-u_g (mod g) }, the columns of the stretch
  that gear g strikes, counted once per gear per column.
- **maximal coverage** m_g(L) = max over the g phases r of c_g(r, L). Note before testing: the two
  teeth of every gear split its circle into arcs of about g/3 and 2g/3 (u_g = 6^{-1}, so the teeth
  sit at +-u and 2u = 3^{-1}·2 is near g/3 - the known 1:2 tooth split), so m_g(L) = 1 for every
  gear with g/3 > L, m_g(L) = 2 while L is between the two arcs, and so on. This is stated first
  because it decides how much content the ratio can carry at the top of a machine.
- **ratio** r_g = c_g/m_g, in [0, 1]. A gear at r_g = 1 is "at coverage maximum".
- **sole column** (= **hinge**): a column of the stretch struck by exactly one gear of M. Its
  striker is the **hinge gear**. **min-strikers** of a stretch = the smallest striker count over
  its columns; a stretch has a hinge iff its min-strikers is 1.
- **waste** w_g = the number of gear g's strikes inside the stretch that land on columns with two
  or more strikers.

### Theory A - greedy from the bottom

At an extremal stretch the machine allocates its coverage by size: the small gears run at their
coverage maximum because they supply bulk, the large gears run below it because what the stretch
needs from them is the one or two columns nobody else can take. If that allocation is a fixed
profile rho(rank) - the same shape at every machine - then the shape itself is a statement about
how long a blocked stretch can be.

- **A1 (the shape at a period record).** At every record stretch of m13..m31, r_g = 1 for gears
  5, 7 and 11 from m19 on, and r_g < 1 for the top gear at every machine from m13 on. Stated
  doubt, from the parent's own table before testing: the strictly-decreasing form of the shape is
  expected to FAIL, because 5d.i's (b2) row already shows gear 13 at m31 in no record at its
  maximum while gear 29 is at its maximum in every record. So the pre-registered claim is the
  weak form (bottom three at 1, top one below 1) and the strong form (r_g non-increasing in rank)
  is pre-registered as expected-refuted. REFUTED (weak form) by one record of m19..m31 with
  r_5 < 1, r_7 < 1 or r_11 < 1, or with r_top = 1.
- **A2 (the profile is a function of g/top, not of the machine).** Plotting r_g against g/q, the
  values at different machines fall on one curve: r_g >= 0.9 for g/q <= 0.4 and r_g <= 0.8 for
  g/q >= 0.9, at every machine m19..m31. REFUTED by a machine where a gear with g/q <= 0.4 has
  r_g < 0.7, or where the top gear has r_g = 1.
- **A3 (the same at the runner-ups).** The same profile holds at the stretches of length F-2 and
  F-3 (the runner-up gaps) at m13..m31: gear 5 at r = 1 in at least 80 % of them. REFUTED if gear
  5 is at maximum in fewer than half of the runner-ups, which would make the split a property of
  the record alone and not of "extremal" at all.
- **A4 (the window is the same object).** The r_g profile of the window's longest blocked stretch
  at rungs 23..997 has the same shape: the bottom gears at 1, the top gears below 1. Stated doubt
  before testing, from the arc arithmetic above: at rung 997 the stretch is 242 columns and every
  gear above 726 has m_g = 1, so every large gear that strikes at all is trivially at its maximum
  and the ratio carries no information there. Pre-registered prediction: at rung 997 more than
  80 % of the gears that strike the stretch have m_g = 1, so A4 is expected to be REFUTED in the
  direction "the window profile is degenerate at the top, not merely different".
- **A5 (what a profile would force, and whether it is the dead counting bound).** Any blocked
  stretch of length L satisfies L <= sum_g c_g <= sum_g m_g. The right-hand inequality is the
  capacity count, which is on the project's dead list. A fixed profile would sharpen it to
  L <= sum_g rho_g m_g. Pre-registered test of whether that sharpening can matter: measure
  sum_g m_g / L at the records of m13..m31. Prediction: the ratio GROWS with the machine (1.2 at
  m13 to above 1.5 at m31), so the capacity bound is already loose by more than any profile can
  recover, and this half of the branch is to be stopped in one line. REFUTED (i.e. the half stays
  alive) if the ratio is flat or falling and below 1.2 at m31.

### Theory B - the hinge

- **B1 (a hinge always exists).** At every prime rung q = 23..1999 the window's longest blocked
  stretch has min-strikers = 1. Prior art stated before testing, so that no credit is claimed for
  it: a column of the window struck by exactly one gear g is a column (p, g^a m) with p and m
  prime and m > q - a **pseudo-twin**, and "a gear is needed iff it owns a pseudo-twin in the
  window" is already on record (alignment-rules 4.1); 5d.ii already measured sole-column densities
  of 0.22-1.00 in these stretches. So B1 is expected TRUE and expected to be a restatement; it is
  run as the gate for B2-B5 and will be reported in one line. REFUTED by one rung with
  min-strikers >= 2.
- **B2 (the hinge gear is large).** Let g_h be the largest hinge gear of the window's longest
  stretch. Prediction: g_h > q/2 at at least 80 % of the rungs 23..1999, and g_h/q has median
  above 0.6. REFUTED if g_h > q/2 at fewer than half the rungs.
- **B3 (the hinge is central).** The relative position of the hinge column of g_h inside the
  stretch, (c - s)/(L-1), lies in [0.25, 0.75] at at least 70 % of the rungs. Stated doubt: with
  several hinges per stretch the positions may simply be uniform, in which case the central band
  gets 50 % by construction; the test therefore also reports the distribution over ALL hinge
  columns and the position of the LARGEST-DROP hinge. REFUTED if the central band holds fewer
  than 55 % of the rungs, i.e. no better than uniform.
- **B4 (the length rule).** Stated before testing, three candidates in decreasing strength:
  (i) L <= g_h/3 (the smaller arc of the hinge gear); (ii) L <= 2 g_h/3 (the larger arc, i.e. the
  distance between the hinge gear's two teeth the long way round); (iii) L <= g_h. Prediction:
  (i) FAILS, (ii) HOLDS at every rung 23..1999 with no exception, (iii) holds trivially. The
  branch's claim is (ii): **the window's longest blocked stretch is never longer than the long arc
  of its largest hinge gear**, equivalently g_h >= 3L/2. REFUTED by one rung with 3L > 2 g_h.
- **B5 (the second and third stretches).** The same statements hold for the window's second- and
  third-longest blocked stretches at every rung, with the same rule (ii). REFUTED by one rung
  where the second- or third-longest stretch has min-strikers >= 2 or violates (ii).
- **B6 (what decides g_h).** If (ii) holds, ask what makes it hold: is g_h large because L is
  large (a correlation), or is g_h's size independent of L and the rule a coincidence of scale?
  Pre-registered discriminator: the ratio 2g_h/(3L) - if the rule is structural the ratio should
  sit just above 1 with a floor; if it is a coincidence of scale the ratio should be broadly
  distributed with its minimum drifting toward 1 as the rungs grow. Prediction: the ratio's
  minimum over the rungs 23..1999 is above 1.0 and BELOW 1.3, i.e. the rule is tight somewhere.
  REFUTED as a structural rule if the minimum over the range is above 3 (the rule is then just
  "large gears exist").

### Scorecard

| # | prediction | verdict | evidence |
|---|---|---|---|
| A1 | bottom three at r = 1, top gear below, in every record m19..m31 | SPLIT | gears 5, 7, 11 at maximum in every record of m19..m31 (held); the top gear is at maximum in EVERY record at m13 and m17 and in half of m31's, so "top gear below" is refuted (a1 A1/A2 tables) |
| A1s | the strong form: r_g non-increasing in rank (pre-registered as expected-refuted) | REFUTED, as pre-registered | rank inversions per record 0-4 at m13..m23 and 5-7 (of 36) at every m31 record |
| A2 | one curve in g/q: r >= 0.9 below 0.4q, r <= 0.8 above 0.9q | SPLIT | pooled band means 1.000, 1.000, 0.963, 0.981, 0.821, 0.889 at g/q bands [0,.2) .. [.95,1]; the low half holds, the top band is 0.889 not <= 0.8, and the curve is not monotone |
| A3 | gear 5 at maximum in >= 80 % of runner-up stretches | HELD at 100 %, and it is a THEOREM | gear 5 at its maximum in ALL 16 (machine, length) cells of m13..m31, records and both runner-up lengths, against phase baselines as low as 0.20 (a1 A3, c1 (A)); proved in d1 |
| A4 | the window stretch has the same profile | REFUTED both ways | the window's top band is r = 1.00 (the record's is 0.00-0.50); classified exactly, the window's longest stretch has 63.0 % FREE deficits against the record's 2.3 % (e2); the pre-registered "80 % of striking gears have m_g = 1 at rung 997" is also refuted (15.7 %) |
| A5 | sum m_g / L grows past 1.5 by m31, so the profile cannot beat capacity | HELD | 1.200, 1.294, 1.417, 1.455, 1.500, 1.544 at m13..m31; half stopped |
| B1 | min-strikers = 1 at every rung 23..1999 | HELD, and it is prior art | 295 of 295 for the longest, second- and third-longest stretches; the hinge is alignment-rules 4.1's pseudo-twin |
| B2 | g_h > q/2 at >= 80 % of rungs; median g_h/q > 0.6 | REFUTED | 56.9 % above q/2; median 0.575, mean 0.553, min 0.225 |
| B3 | the g_h hinge is central at >= 70 % of rungs | REFUTED, in the opposite direction | central band at 31.2 % of rungs, BELOW the 50 % a uniform position gives; all hinges pooled 55.7 % |
| B4 | L <= 2 g_h/3 at every rung, and L <= g_h/3 fails | REFUTED | (ii) fails at 52 of 295 rungs, min 2g_h/(3L) = 0.472 at q = 37; even (iii) L <= g_h fails at 8 rungs; (i) holds at only 63 |
| B5 | the same for the 2nd and 3rd longest stretches | REFUTED | (ii) fails at 36 and 43 of 295 |
| B6 | min 2g_h/(3L) in (1.0, 1.3] | moot, rule refuted; null measured | over 1,754,131 blocked stretches of all windows, 3L > 2 g_h at 20.9 % and rank corr(L, g_h) = +0.516 - a scale correlation, not a rule |

### What this branch could find that is not already known

Known and not to be re-derived: the coverage-maximality split itself (5d.i section 6); the holder
and drop profiles of the window stretch (5d.ii R2); the square gate (7d); the nested-decreasing
holder law (5d.ii R3); the pseudo-twin characterisation of a needed gear (alignment-rules 4.1);
that F_W is the largest twin gap in (q, q'^2) (7d's kernel identity); the capacity and overlap
counting dead end; corridor law and corridor resonance. What is not on the record is (a) the
coverage RATIO gear by gear rather than the binary at-maximum flag, and what its shape is as a
function of g/q, at records, runner-ups and the window's stretch; (b) any statement tying the
LENGTH of a blocked stretch to the SIZE of the gear that holds one of its columns alone.

Filed here in advance, so that it is a pre-registration and not a post-hoc explanation: if a
universal profile exists, the only inequality it yields is L <= sum_g rho_g m_g, which is the
capacity count with weights below one. If A5's ratio grows, the profile half is the dead capacity
bound in phase-lock clothing and is stopped there.

## Setup (exact ranges)

Five scripts, all exact, no sampling anywhere.

**a1_coverage.py** - full periods of {5..13} .. {5..31}. m13..m23 by one array; m29
(P = 1,078,282,205) and m31 (P = 33,426,748,355) by a chunked pass keeping every gap of length
>= F-3 (3.9 s and 107.5 s). Every collected stretch is re-gated as exactly blocked with both
flanks open. Record counts 12, 20, 20, 4, 2, 4 at m13..m31, reproducing F = 11, 18, 25, 34, 43, 58
and 5d.i's record starts. The runner-up lengths are the next two DISTINCT gap lengths: 10 and 8 at
m13, 16 and 15 at m17, 23 and 22 at m19, 33 and 32 at m23, and only 40 at m29 and 55 at m31 (the
gap spectrum has no 41, 42 at m29 and no 56, 57 at m31 - a fact the scan produces in passing).
Per stretch: c_g, m_g, r_g, sole counts, overlap.

**b1_hinge.py** - every prime rung q = 23..1999 (295 rungs), window columns
lo = q//6+1 .. hi = (q'^2-1)/6. Per column the striker count and the striker sum, so a column with
count 1 names its gear. The three longest gaps by DISTINCT length; per stretch the min-strikers,
every hinge column with its gear and offset, the largest hinge gear g_h, and the coverage profile.
m_g(L) is computed in closed form (the maximum count of a periodic 2-point set in a window of L is
attained at a window starting on a point, so m_g = max(c_g at phase u, c_g at phase g-u)) and the
closed form is gated against brute force at every gear 5..199 and twelve lengths.

**c1_controls.py** - the phase baseline share_max(g, L) = the share of the g phases attaining
m_g(L), without which "at maximum in every record" can be free; the same measurement on the window
side at all 295 rungs; hinge position, striker count and hinge-gear size by decile of the stretch,
pooled; and the null for the length rule taken over EVERY blocked stretch of every window with
L >= 10 (1,754,131 stretches).

**d1_gear5_lock.py** - the gear-5 lock: brute force over every gear 5..97 and every length
L = 1..600, gear 5 exhaustively to L = 2000, the five-case argument, the corollary against node
5e's slot rule, and gates against all 62 record stretches of m13..m31 and against every maximal
blocked stretch of every window at all 295 rungs (asserted column by column, not sampled).

**e1_deficit.py / e2_window_deficit.py** - for each gear of a stretch, the classification
AT MAX / FORCED DEFICIT (no phase attains m_g while still striking the columns only g strikes) /
FREE DEFICIT (such a phase exists, so the gear could have had both). Run over every record stretch
of m13..m31 and over the window's longest stretch at every rung.

## Results

### R1. The coverage ratio at the period records (A1, A2)

Each cell is c_g/m_g at the machine's record stretches; "at max" is the share of the machine's
records with that gear at its coverage maximum (a1).

| machine | L | records | 5 | 7 | 11 | 13 | 17 | 19 | 23 | 29 | 31 | sum m_g | sum m_g/L |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m13 | 10 | 12 | 1.00 | 0.67 | 1.00 | 1.00 | | | | | | 12 | 1.200 |
| m17 | 17 | 20 | 1.00 | 0.40 | 0.40 | 0.80 | 1.00 | | | | | 22 | 1.294 |
| m19 | 24 | 20 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.80 | | | | 34 | 1.417 |
| m23 | 33 | 4 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 0.00 | | | 48 | 1.455 |
| m29 | 42 | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 0.00 | | 63 | 1.500 |
| m31 | 57 | 4 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 0.00 | 1.00 | 0.50 | 88 | 1.544 |

Read down the columns and the "greedy from the bottom" picture does not survive. Gear 5 is at its
maximum everywhere; but the gears NOT at maximum are one or two per machine and they are not the
top ones: gear 17 at m19 and m29, gears 13 and 23 at m23 and m31, gears 7, 11 and 13 at m17. The
top gear is at its maximum in EVERY record at m13 and at m17, in 80 % at m19 and in half of m31's.
The parent's wording "the top one or two gears never do" is a description of m23 and m29 only; the
underlying numbers in 5d.i's (b2) row and mine agree, the reading did not.

Pooled over the records of m19..m31, r_g against g/q gives band means 1.000, 1.000, 0.963, 0.981,
0.821, 0.889 on the bands [0,.2), [.2,.4), [.4,.6), [.6,.8), [.8,.95), [.95,1]. The low half of the
curve holds as pre-registered; the top does not fall to 0.8 and the curve is not monotone. Rank
inversions per record are 0-4 at m13..m23 and 5-7 (of a possible 36) at every m31 record: the
strong form is refuted, as pre-registered.

The deficits are tiny in total. sum c_g against sum m_g is 11/12, 22/22, 33/34, 46/48, 61/63,
85/88: **the record uses within 3 of the machine's whole coverage capacity at every machine**, and
the overlap sum c_g - L is 1, 5, 9, 13, 19, 28.

### R2. Why the below-maximum gears are below maximum (the mechanism, e1/e2)

For each gear of a record stretch, ask whether any phase of that gear both attains m_g and still
strikes every column only that gear strikes. Over all 62 record stretches of m13..m31, 348
gear-cells:

| machine | stretches | gear-cells | at max | forced deficit | FREE deficit |
|---|---|---|---|---|---|
| m13 | 12 | 48 | 44 | 4 | 0 |
| m17 | 20 | 100 | 72 | 24 | 4 |
| m19 | 20 | 120 | 96 | 20 | 4 |
| m23 | 4 | 28 | 22 | 6 | 0 |
| m29 | 2 | 16 | 12 | 4 | 0 |
| m31 | 4 | 36 | 26 | 10 | 0 |

So the law is not "the small gears run at maximum and the big ones do not". It is: **every gear of
a record is at its coverage maximum subject to keeping the columns only it strikes**, and the gears
below maximum (none to three per record, 1.1 on average) are exactly those for which the two
requirements are incompatible. That
holds at 340 of 348 gear-cells and at every cell of m13, m23, m29 and m31. The eight exceptions
are gear 13 at four m17 records and gear 19 at four m19 records - the only cells where a gear
could have had both and the record does not use the extra strike.

Worked at m31's record (start 1468940243, L = 57): c_g/m_g = 23/23, 17/17, 11/11, 8/10, 7/7, 6/6,
5/6, 4/4, 4/4 with sole columns 9, 6, 3, 4, 4, 2, 3, 2, 2 and waste 14, 11, 8, 4, 3, 4, 2, 2, 2.
Gear 13 is two strikes below its maximum and gear 23 one, and in each case no maximal phase of
that gear keeps its own sole columns.

### R3. THE GEAR-5 LOCK (A3, and the branch's one theorem)

**Statement.** Let a stretch be a MAXIMAL blocked run of any machine containing gear 5: columns
s .. s+L-1 with s-1 and s+L openings. Then c_5(s, L) = m_5(L). Gear 5 is at its coverage-maximal
phase automatically, at every length, in every machine, at every position - record, runner-up,
window stretch, anywhere.

**Proof.** Gear 5's teeth are T = {1, 4} = {+-1} mod 5 (u_5 = 6^{-1} = 1) and its open residues
are {0, 2, 3}. Write L = 5t + e. Then c_5(s, L) = 2t + n_e(s) with n_e(s) = #{ j < e : s+j in T },
so m_5(L) = 2t + max_s n_e. The flanking openings are not struck by gear 5, which gives s-1 not in
T (so s in {1,3,4}) and s+e not in T (so s not in T-e). In each of the five cases the surviving
phases are contained in argmax n_e:

| e = L mod 5 | argmax n_e | phases the flanks allow | contained |
|---|---|---|---|
| 0 | {0,1,2,3,4} (n = 0) | {3} | yes |
| 1 | {1,4} | {1,4} | yes |
| 2 | {0,1,3,4} | {1,3} | yes |
| 3 | {4} | {4} | yes |
| 4 | {1,3,4} | {1,3,4} | yes |

Checked exhaustively for every L to 2000 (0 violations), gated at all 62 record stretches of
m13..m31 and asserted column by column at EVERY maximal blocked stretch of every window at all 295
rungs (over 1.7 million stretches), no exception.

**It is specific to gear 5.** Running the same flank test gear by gear over every length to 600,
the minimum share of flank-allowed phases that attain m_g is 1.000 at gear 5 and 0.333, 0.143,
0.111, 0.077, 0.067, 0.053, ... at gears 7, 11, 13, 17, 19, 23, ..., falling like 2/(g-2), and
already below 1 at L = 1 for every gear above 5. The reason is arithmetic: gear 5 has only three
open residues, so the two flank conditions remove enough of the five phases to pin the rest inside
the argmax; no larger gear has that little room.

**It is node 5e, proved.** The opening that opens a gap of length F sits at x = s-1 and its residue
mod 5 names the twin slot (k = 0 mod 5 gives 29|31, k = 2 gives 11|13, k = 3 gives 17|19; k = 1, 4
are struck by gear 5). With e = (F-1) mod 5 the table above reads:

| F mod 5 | start openings x mod 5 | slots | node 5e (measured at eight full periods) |
|---|---|---|---|
| 0 | {0, 2, 3} | any | any |
| 1 | {2} | 11\|13 | 11\|13 |
| 2 | {0, 3} | 29\|31, 17\|19 | mirror pair {29\|31, 17\|19} |
| 3 | {0, 2} | 29\|31, 11\|13 | mirror pair {29\|31, 11\|13} |
| 4 | {3} | 17\|19 | 17\|19 |

Exact agreement in all five rows. Node 5e was a FACT measured at the eight full periods to m31 and
stated for record gaps; it is now a theorem for every machine, every length and every maximal
blocked stretch, and it is the same statement as gear 5's coverage maximality.

**What the lock forces, exactly.** Since c_5 = m_5, the number of columns of the stretch that gear
5 does NOT strike is L - m_5(L) = floor(3L/5), for every L, with no dependence on the machine or
the position. So every maximal blocked stretch of length L obliges the gears {7..q} to cover
exactly floor(3L/5) columns, arranged in gear 5's fixed alternation of blocks of two and one.
Nearest prior art: node 7a's cycle identity F_c(M) = floor((F(M)-2)/5), which is the record-level
statement of the same arithmetic; the branch does not re-derive it, and the natural next step -
asking the same question of the residual after gear 5 - IS the cycle frame, which is DEAD (7a).

Against the phase baselines this is what makes the parent's observation non-trivial and also what
explains it away: share_max(5, L) at the record and runner-up lengths of m13..m31 is 1.00, 0.60,
0.80 / 0.80, 1.00, 0.60 / 0.60, 0.80, 0.40 / 0.20, 0.80, 0.40 / 0.80, 0.60 / 0.80, 0.60 - as low
as one phase in five at m23 - and gear 5 is at maximum in all 16 cells. It is not luck; it is the
flanks.

### R4. The window's stretch is a different object (A4)

At every prime rung 23..1999 the window's longest blocked stretch (b1, c1, e2). Note for honesty:
the 295 rungs carry only ELEVEN distinct stretches, because F_W is inherited until a longer twin
gap appears - starts 111, 398, 981, 2234, 3091, 4071, 10384, 31319, 114743, 141726, 478161 with
L = 24, 27, 34, 46, 61, 82, 104, 153, 167, 241, 251. Per-rung counts are therefore not independent
samples; the gear-5 gate above is over every stretch of every window and does not depend on it.

| q | L | start | strikers min/mean/max | hinges | g_h | g_h/q | pos of g_h | 2g_h/3L | gears striking | with m_g = 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 59 | 27 | 398 | 1 / 1.81 / 4 | 14 | 41 | 0.695 | 0.73 | 1.012 | 15 | 0 |
| 173 | 82 | 4071 | 1 / 2.20 / 5 | 29 | 137 | 0.792 | 0.77 | 1.114 | 38 | 0 |
| 499 | 153 | 31319 | 1 / 2.54 / 7 | 38 | 311 | 0.623 | 0.95 | 1.355 | 85 | 5 |
| 997 | 241 | 141726 | 1 / 2.74 / 7 | 50 | 709 | 0.711 | 0.80 | 1.961 | 140 | 22 |
| 1999 | 251 | 478161 | 1 / 2.88 / 8 | 49 | 1201 | 0.601 | 0.15 | 3.190 | 183 | 67 |

The coverage ratio by band g/q at these rungs is U-shaped, not falling: at q = 997 it is 0.88,
0.77, 0.57, 0.67, 1.00 on the bands [0,.2) .. [.8,1]. The top band's 1.00 is vacuous - those gears
have m_g = 1, so any strike at all is "maximal" - and the share of striking gears with m_g = 1
grows 0.000, 0.000, 0.059, 0.157, 0.366 at q = 59, 173, 499, 997, 1999. Restricted to gears with
m_g >= 2 the mean ratio settles at 0.725, 0.734, 0.732 at the three largest rungs. So the window's
profile is the opposite of the record's at the top: in a record the top gear is at maximum in 0 to
50 % of cases, in a window every gear above 3L is at maximum for free.

The decisive comparison is the classification of R2 applied to the window, over the 295 rungs and
28,154 gear-cells with m_g >= 2:

| frame | at max | forced deficit | FREE deficit | rungs / stretches with a free deficit |
|---|---|---|---|---|
| period record, m13..m31 | 272 of 348 (78.2 %) | 68 (19.5 %) | 8 (2.3 %) | 8 of 62 stretches |
| window's longest stretch | 8,400 (29.8 %) | 2,004 (7.1 %) | 17,750 (63.0 %) | 292 of 295 rungs |

At q = 997: 36 gears at maximum, 10 forced deficits, 80 free deficits. The window's longest stretch
is simply a stretch: two thirds of its gears could have covered more of it and do not. **The answer
to "one object or two laws" is: two laws, plus one theorem they share.** The theorem is the gear-5
lock, which holds at both because it holds at every maximal blocked run. Everything above gear 5 is
a property of extremality - and the window has no extremality, because it is not chosen.

The window-side baseline table makes the same point per gear: gear 5 is at maximum on the window's
longest stretch at 295 of 295 rungs against a mean phase baseline of 0.436 (lift 2.29, and it is
the theorem), while gear 7 manages 0.627 against 0.472 (lift 1.33), gear 11 0.807 against 0.602,
gear 13 0.586 against 0.479, gear 17 0.949 against 0.537, gear 23 0.414 against 0.301, and gear 19
0.105 against 0.324 - below its own baseline. There is no profile above gear 5.

### R5. Capacity (A5)

sum m_g / L at the records is 1.200, 1.294, 1.417, 1.455, 1.500, 1.544 at m13..m31, rising at every
rung. **One line, as pre-registered: a coverage profile can only give L <= sum_g rho_g m_g, the
capacity count with weights below one; capacity is already 54 % loose at m31 and loosening, and the
missing quantity is a LOWER bound on the overlap sum c_g - L (1, 5, 9, 13, 19, 28), which is the
overlap-counting dead end. That half of the branch is stopped here.**

### R6. The hinge (B1-B6)

**B1.** Min-strikers is 1 on the longest, second-longest and third-longest window stretch at all
295 rungs. This is the pseudo-twin condition of alignment-rules 4.1 and is reported in one line:
a hinge column is (p, g^a m) with p and m prime and m > q. Worked: at q = 997 the largest hinge is
gear 709 at column 141918 = (851507, 851509) with 851507 prime and 851509 = 709 x 1201; the parent's
column 141928 = (851567, 851569) has 851567 = 877 x 971 and 851569 prime; at q = 1999 the largest
is 1201 at column 478198 = (2869187, 2869189) with 2869189 = 1201 x 2389. A hinge column is a twin
pair missed by exactly one prime, and the longest window stretch carries between 12 and 53 of them
(median 38).

**B2.** g_h/q has min 0.225, median 0.575, mean 0.553, max 0.967; g_h exceeds q/2 at 56.9 % of
rungs, q/4 at 96.6 %, 3q/4 at 15.3 %. The pre-registered 80 % and median 0.6 are both refuted; the
honest statement is a weak tendency, and its cause is that g_h is a maximum over 12-53 hinge gears,
so it inherits the size distribution of the largest prime factor rather than any structure.

**B3.** REFUTED in the direction opposite to the prediction. The g_h hinge lies in the central half
of the stretch at 31.2 % of rungs, against 50 % for a uniform position; its mean position is 0.486,
so it is symmetric but pushed to the ends. Pooled over all 11,726 hinge columns the central half
holds 55.7 %, and the two facts are consistent: the rank correlation of hinge gear size with
|position - 0.5| is +0.100, and the median hinge gear is 11 in the central tenth of the stretch, 19
in the next band and 23 in the outer fifth. Per decile of the stretch the count of hinges is
1333, 906, 1085, 1154, 1374, 1413, 1188, 1259, 1121, 893 and the mean striker count is flat at
2.47-2.76, so the effect is in WHICH gear holds the column, not in how many hold it.

**B4, B5, B6. All three length rules are refuted.**

| rule | holds at | first failures (q, L, g_h) |
|---|---|---|
| L <= g_h/3 | 63 of 295 | (23, 24, 19), (29, 24, 19), (31, 24, 19), (37, 24, 17) |
| L <= 2 g_h/3 (the branch's claim) | 243 of 295 | the same, then (61, 27, 37), (67, 27, 31), (193, 82, 103), (397, 104, 149), (421, 104, 97), (643, 153, 229) ... (827, 153, 193) |
| L <= g_h | 287 of 295 | (23, 24, 19) .. (43, 24, 17), (421, 104, 97), (431, 104, 97) |

The minimum of 2g_h/(3L) is 0.472 at q = 37, 41, 43; the median is 1.355. On the second-longest
stretch the middle rule fails at 36 of 295 rungs and on the third at 43. And the null kills the
idea outright: over the 1,754,131 blocked stretches of all the windows with L >= 10, 3L > 2 g_h at
20.9 % of them and L > g_h at 14.7 %, with rank correlation of L with g_h equal to +0.516 and
median g_h/L rising 4.08, 6.47, 6.88, 5.95 over the length bands [10,25), [25,50), [50,100),
[100,300). g_h grows with L because both grow with the scale; there is no rule in it.

## Mechanism

**Gear 5.** The lock is the whole positive content of Theory A and it is elementary. A blocked
stretch is bounded by two openings, and an opening is a non-tooth of every gear. Gear 5 has two
teeth and three open residues, so "the column before is open" removes two of its five phases and
"the column after is open" removes two more; what survives is always inside the set of phases that
maximise gear 5's count over the stretch. Nothing the machine does can prevent it: the argument
uses only 6u = 1 mod 5 and the definition of a maximal stretch. It fixes gear 5's phase relative to
the stretch, hence the exact multiset of columns gear 5 leaves to the others - floor(3L/5) of them,
in the fixed pattern of blocks of two and one - and hence, at a record, the twin slot the gap can
open on (node 5e). It fixes nothing about L: the residual covering problem it hands to {7..q} is
the cycle frame, and the cycle frame is F/5 in disguise (7a), already dead.

**The rest of a record.** Above gear 5 the allocation is not by size at all. Each gear takes its
coverage maximum unless doing so would cost it a column that only it strikes; then it takes one or
two strikes fewer. That is L4's sole-striker requirement priced in coverage units, and it is
exactly right at 340 of 348 gear-cells over all 62 record stretches of m13..m31, with the eight
exceptions at gear 13 (m17) and gear 19 (m19). The gears that pay are decided by their sole columns
and not by their rank: gear 17 pays at m19 and m29, gears 13 and 23 at m23 and m31, gears 7, 11 and
13 at m17, and the top gear pays nothing at m13 and m17. The total price is small - the record's total coverage is within
3 of the machine's whole capacity at every machine - so the length of a record is set by the
OVERLAP (1, 5, 9, 13, 19, 28 at m13..m31), not by the capacity, and bounding overlap from below is
the dead end.

**The window.** The window's longest stretch obeys none of that, because it is not chosen: it is
whatever stretch the primes leave, and two thirds of its gears sit below a maximum they could have
had while keeping their sole columns. Its top is degenerate in the other direction - every gear
above 3L can strike at most once, so it is "at maximum" for free, and the share of such gears grows
0.000 -> 0.366 from q = 59 to q = 1999.

**Why no rule can tie a stretch's length to its hinge gear.** For a FIXED stretch, the set of gears
owning a column alone can only shrink as the machine grows (5d.ii's nested-decreasing holder law: a
sole column can gain a striker, never lose one). So g_h is non-increasing in q at fixed L - measured
at all eleven distinct window stretches with no exception, e.g. the 241-column stretch at column
141726 has g_h = 877, 859, 853, 709, 563, 409 as q runs 919 -> 1669, and the 153-column stretch at
31319 has 311, 307, 293, 229, 193 as q runs 433 -> 827. Any rule of the form L <= f(g_h) with f
increasing must therefore break for any stretch that stays the window record over enough rungs, and
it does: the 153-stretch violates L <= 2g_h/3 from q = 821 on, exactly when g_h falls to 193 < 229.5.
The machine does this routinely, so the whole family of such rules is closed, not just the three
tested. In the limit the surviving hinge gears are only those owning a column (p, g^a) - a prime
power - since a column (p, g m) loses its hinge as soon as q reaches m; that is alignment-rules
4.1's "droppability is transient", from the other side.

## What is new

1. **The gear-5 lock, with a proof:** every maximal blocked stretch of every machine has gear 5 at
   its coverage-maximal phase, for every length, because the two flanking openings remove four of
   gear 5's five phases into its argmax. Five cases, no machine in it. 5d.i's coverage-maximality
   observation for gear 5 was measured at the records of m13..m31; it is a theorem, and it holds
   at every stretch anywhere, gated at over 1.7 million of them.
2. **Node 5e is the same theorem.** The slot rule "F = 1 mod 5 starts on 11|13, F = 4 on 17|19,
   F = 2 and 3 on mirror pairs, F = 0 on any", a FACT measured at eight full periods, is the
   x = s-1 reading of the lock's five cases, exact in all five rows, now proved uniformly.
3. **The exact residual:** every maximal blocked stretch of length L leaves the gears {7..q}
   exactly floor(3L/5) columns to cover, in gear 5's fixed two-and-one pattern - independent of the
   machine and of the position.
4. **The record's allocation law, and the correction it forces:** every gear of a record is at its
   coverage maximum SUBJECT TO keeping the columns only it strikes (340 of 348 gear-cells over all
   62 records of m13..m31; the eight exceptions named). The gears below maximum are none to three
   per record and are NOT the top ones - 5d.i's "the top one or two gears never do" is a reading of
   m23 and m29 only; the top gear is at its maximum in every record at m13 and m17.
5. **Two laws, not one:** classified identically, the period record has 78.2 % of its gear-cells at
   maximum, 19.5 % forced deficits and 2.3 % free; the window's longest stretch has 29.8 %,
   7.1 % and 63.0 %. The only
   shared statement is the gear-5 lock.
6. **The hinge gear shrinks as the machine grows** (non-increasing in q at all eleven distinct
   window stretches), which closes every rule of the form "a stretch is no longer than a function
   of its largest hinge gear" - not only the three pre-registered, which fail at 232, 52 and 8 of
   295 rungs.
7. **The hinge geometry, measured:** 12-53 hinge columns per window record (median 38); the largest
   hinge gear sits in the central half at 31.2 % of rungs, below uniform, and hinge gear size
   correlates with distance from the centre at +0.100 over 11,726 columns (median gear 11 in the
   central tenth against 23 in the outer fifth).
8. **A ledger fact from the runner-up scan:** the gap spectrum of m29 has no gaps of length 41 or
   42 below the record 43, and m31 none of 56 or 57 below 58 - the record is isolated by 3 at both
   of the largest full periods.

Toward the root: (1)-(3) are a forced object, and that is what the root asks for - but it is a
POSITION object. It says where gear 5 must be relative to any blocked stretch, and hence where a
record gap can open; it says nothing about how long the stretch can be, and its only length
consequence, L <= m_5(L) + sum_{g>=7} m_g, is the capacity count unchanged. (4)-(8) are
measurements of the margin and two corrections to the parent's readings.

## Verdict

Theory A: **SPLIT - one theorem, and the rest is the dead counting bound.** The gear-5 half is
proved and is more general than the observation that spawned it; above gear 5 there is no profile
in rank or in g/q, the true rule is "maximal subject to the sole columns", and the length
consequence of any profile is the capacity count, which is 1.20 -> 1.54 loose and worsening. That
half is stopped, as pre-registered.

Theory B: **DEAD.** The hinge exists always (prior art: the pseudo-twin), it is not reliably large
(56.9 % above q/2), it is not central (31.2 %, below uniform), and no rule ties the stretch's
length to the hinge gear's size - the pre-registered rule fails at 52 of 295 rungs and even
L <= g_h fails at 8, with a structural reason: the hinge gear is non-increasing in the machine
while the stretch is not, so the machine breaks any such rule as a matter of course.

Node 5g's status for the tree: **FACT (forced, position only)** for the gear-5 lock and its
identification with 5e; **DEAD** for the hinge as a length lever; the branch does not move the
root, which needs a bound on the length of a blocked stretch, and nothing here bounds one.

## Dead ends (with the refuting instance)

- "The coverage profile is a fixed shape in the gear's rank" - m31, where gear 13 is at 8/10 and
  gear 29 at 4/4 in every record; rank inversions 5-7 of a possible 36 in each.
- "The top gear is never at its coverage maximum" - m13 and m17, where the top gear is at its
  maximum in every one of the 12 and 20 records.
- "The window's stretch has the record's profile" - q = 997, where the top band g/q in [0.8, 1] has
  mean ratio 1.00 (against 0.00-0.50 for a record's top gear) because m_g = 1 there, and 80 of the 90
  deficient gears could have been at maximum without losing a sole column.
- "A coverage profile bounds the length of a stretch better than capacity" - sum m_g/L = 1.544 at
  m31 and rising; the missing quantity is a lower bound on the overlap, the dead overlap count.
- "The window's longest stretch is never longer than the long arc of its largest hinge gear" -
  q = 37, L = 24, g_h = 17: 2 g_h/3 = 11.3, and 51 further rungs, the largest being
  q = 821, L = 153, g_h = 193.
- "L <= g_h" - q = 421 and 431, L = 104, g_h = 97.
- "The largest hinge gear sits centrally" - the central half holds it at 31.2 % of rungs, below the
  50 % a uniform position gives.
- The residual after gear 5 (the floor(3L/5) columns it leaves) is node 7a's cycle frame, DEAD; not
  pursued, one line.

## Prior art

Nearest located, each in one line and none re-derived. The pseudo-twin characterisation of a needed
gear is alignment-rules 4.1 and is exactly what a hinge column is. The nested-decreasing holder law
is 5d.ii R3 and is the reason no hinge-size length rule can exist. The sole-striker requirement in
any above-record stretch is L4 (research/proof/pair_statement.md); R2's classification is L4 priced
in coverage units. The 1:2 tooth split (teeth at +-u with 2u = 3^{-1}·2) is the known kernel fact
behind m_g's step structure. Node 7a's F_c = floor((F-2)/5) is the record-level form of the
floor(3L/5) residual. Node 5e is the slot rule the gear-5 lock proves. The capacity and overlap
counts are the project's dead ends and were not opened. Outside the project, "a maximal gap in a
sieved set is phase-locked to the smallest modulus" has no located published form for a two-class
sieve; the statement here is elementary and is offered as elementary.
