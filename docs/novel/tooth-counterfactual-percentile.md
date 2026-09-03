# The twin machine is a low-F outlier among its own counterfactuals

Lateral, round 27 (2026-08-29). Status: SCRIPT-VERIFIED (exhaustive, exact) for
machines 11, 13, 17, 19. Mechanism: NOT explained (one candidate pre-registered
and refuted in the sign).

## 1. WHAT IT IS

The machine has two kinds of input: WHICH gears (the primes 5..y) and WHERE each
gear's teeth sit. The gears are the problem; the tooth positions are FORCED by
the twin constellation, since gear q blocks slot k exactly when 6k = +-1 mod q,
i.e. at k = +-v_q with v_q = 6^{-1} mod q. Nothing in the project had ever asked
what happens if the teeth move.

Define the counterfactual family: keep the gears, keep the mirror symmetry
(teeth at +-v_q, which every twin-type constellation has), and let the half-width
v_q range freely:

    V(y) = prod_{q <= y} {1, 2, ..., (q-1)/2},   |V| = 30, 180, 1440, 12960
                                                 at y = 11, 13, 17, 19.

Every member of V(y) has the SAME period P, the SAME number of openings
prod (q-2) (the sharing law: phases move where survivors are, never how many),
and the same per-gear kill density. Only the POSITIONS differ. F is invariant
under k -> +-k + b but NOT under k -> ck (scaling is not an isometry of Z_P), so
F genuinely varies over V(y). The family is small enough to ENUMERATE
EXHAUSTIVELY.

RESULT (exact, all of V(y), research/tooth_counterfactual.py):

    y    |V|     F(twin)  min   median   max   twin's percentile in V(y)
    11   30      7        6     8        11    20.0%
    13   180     11       10    13       25    18.1%
    17   1440    18       14    19       32    26.4%
    19   12960   25       20    28       43    17.1%

THE TWIN MACHINE'S RECORD GAP IS IN THE BOTTOM FIFTH TO QUARTER OF ITS OWN
COUNTERFACTUAL DISTRIBUTION AT EVERY MACHINE TESTED, and roughly 10-15% below
the median - but it is never the minimum. The maximum is 1.6-1.9x the twin value
(43 vs 25 at m19), so the family is wide and the twin's position inside it is a
real fact, not a narrow band.

## 2. WHY IT MIGHT BE NOVEL

Jacobsthal-type quantities are always studied for the actual reduced residue
system (or for a fixed admissible constellation). The COUNTERFACTUAL
DISTRIBUTION of the maximal gap over all symmetric two-tooth sievings with the
same gears and the same survivor count appears not to be a studied object, and
the twin machine's position inside it is a new kind of statement about the twin
problem: it says the twin constellation is not a generic sieve, and that the
non-genericity is in the DIRECTION OF SMALLER GAPS.

It is also the first quantity this project has found on which the real phase
vector IS distinguished. Round 2's enumeration (lateral Refuted 3) scored the
real phase vector on WASTE metrics and found it in the top 10-25% with "no
variational handle". This is the same parameter space with F itself as the
objective, and it separates: the real vector sits low, consistently.

## 3. PROOF / STATUS

SCRIPT-VERIFIED, exhaustive and exact: research/tooth_counterfactual.py builds
every one of the 30 / 180 / 1440 / 12960 sievings by direct full-period sieve
(P up to 1,616,615), computes the exact cyclic maximal gap, and asserts (a) the
true tooth vector is a member of the family, (b) every member has exactly
prod (q-2) openings. Log research/data/tooth_counterfactual.log, 10 gates, exit 0.

NOT PROVED, and not extrapolated: four machines is four machines. m23 would be
|V| = 12960 * 11 = 142,560 sievings over P = 37,182,145 - about an hour of
single-core work and the next honest rung.

INDEPENDENCE CAVEAT, stated because it matters: the four rows are NOT four
independent observations. The twin tooth vector at m19 is the m17 vector with one
coordinate appended, and so on down, so the four percentiles are nested and a
naive "0.264^4 = 0.005" significance calculation is WRONG. What the data support
is "consistently below the median at four nested machines with the deficit
neither growing nor shrinking", not a p-value.

## 4. IMPLICATIONS

(i) It is a POSITIVE fact of the kind the project keeps failing to find: the
arithmetic of the twin constellation makes F smaller than a generic same-density
sieve, which is the right direction for the conjecture. Every upper-bound
argument that treats the machine as "some sieve with these densities" is
therefore leaving something on the table, and the measurement says how much: the
median counterfactual F is 10-15% above the truth, and the counterfactual maximum
is 60-90% above it.

(ii) It gives a NEW FALSIFIABLE OBJECT for the extreme-value question: instead of
asking how F grows, ask how the twin's PERCENTILE moves. If the percentile is
stable (~20%) the twin machine is a fixed distance into the tail of its own
family; if it drifts to 50% the twin's advantage is a small-machine effect.

(iii) It reframes "arithmetic selection" concretely. The project's standing
verdict on erratic quantities is "arithmetic luck, not structure". Here the
arithmetic luck has a sign and a size.

## 5. UNSOLVED QUESTIONS IT TOUCHES

Jacobsthal's function and its two-teeth analogue; the extremal problem "which
symmetric tooth vector maximises / minimises the maximal gap" (a covering-design
question with a Jacobsthal flavour, and the minimum is attained away from the
twin vector at every machine tested); the general question of how much of the
twin problem's difficulty is generic-sieve difficulty.

MECHANISM: OPEN, and one candidate is already dead. Pre-registered P11 (round 27)
predicted that the explanation is ANGULAR COHERENCE - the twin vector has
v_q/q ~ 1/6 at every gear, the smallest angular dispersion in the family, and
coherent teeth should pack better. REFUTED, and refuted in the SIGN: Spearman
correlation between F and angular dispersion is -0.14 / -0.20 / -0.11 at
m13/m17/m19 (higher dispersion goes with slightly LOWER F), and the twin sits in
the LOWEST-dispersion quartile, which is the quartile with the HIGHEST mean F
(28.56 vs 27.69 at m19). Within that quartile alone the twin is at the 15.6% /
20.8% / 10.5% percentile. So the twin vector is a low-F outlier INSIDE the
high-F coherence class - the effect is real and its cause is not coherence.

SECOND CANDIDATE, ALSO DEAD. By CRT every symmetric tooth vector is
v_q = m^{-1} mod q for some integer m, and the twin machine is m = 6. P13
predicted the feature is "m is small". REFUTED (research/tooth_msweep.py, log
research/data/tooth_msweep.log): over m = 1..60 coprime to the gears at m19 the
F values have median 28.0 - EXACTLY the full family's median - with m = 1 giving
33, m = 2 giving 34 and m = 4 giving 32, while the sweep's minimum F = 20 is at
m = 12, not at the twin's m = 6 (F = 25). Small m is not low-F, and 6 is not
distinguished among small m.

So two natural mechanisms are refuted and the effect stands unexplained. That is
the honest state: a real, exactly-measured, consistently-signed anomaly with no
mechanism, which by the project's own measurement directive is a target rather
than a wall.

## 5A. ROUND-28 EXTENSION: THE OTHER STATISTICS, AND THE THIRD DEAD MECHANISM

Lateral, round 28. Status: SCRIPT-VERIFIED, exhaustive and exact.
`research/tooth_stats_r28.py --upto 19` (19 gates, log
`research/data/r28/tooth_stats.log`); `research/tooth_mech_r28.py --upto 19`
(4 gates, log `research/data/r28/tooth_mech.log`).

Round 27 placed the twin in ONE statistic, `F`. The live route does not use `F`
alone: it uses `F_2`, the INCREMENT `F(M+q') - F_2(M)` (the increment law says
this is `<= s_min(q') = min(2v_q', q' - 2v_q')`), and the budget slack
`F(M+q') - F(M) - q'`. Every one of these is defined for every member of the
family, because the family fixes the gears, the period and the survivor count -
so each is a null model, and each favourable placement is measured evidence that
the route's inequality has room the worst case does not use.

### 5A.1 THE TWIN'S PERCENTILE IN EACH STATISTIC

    machine  |V|      F        F_2      F_3      #gap values
    m7       6        66.7%    41.7%    91.7%    33.3%
    m11      30       20.0%    46.7%    75.0%    43.3%
    m13      180      18.1%    34.2%    61.1%     8.3%
    m17      1440     26.4%    47.6%    15.2%    38.3%
    m19      12960    17.1%    12.3%     6.3%    10.5%

`F_2` is below the median at EVERY machine, but only marginally at m11/m17 - so
on the small machines the effect is real for `F` and weak for `F_2`. **At m19 it
reverses and STRENGTHENS WITH DEPTH: 17.1% for `F`, 12.3% for `F_2`, 6.3% for
`F_3`.** Since the route consumes `F_2` and not `F`, that is the favourable
direction, and m23 is the test.

### 5A.1b THE m23 RUNG - THE PLATEAU HOLDS AND THE DEPTH TREND IS CONFIRMED

SCOPE FIRST. The full family `V(23)` is 142,560 sievings (~6 core-hours) and it
did NOT complete: the box ran the round at 96% of its commit limit with six
lanes active, which killed two worker pools. What is delivered is the
EXHAUSTIVE, EXACT **pinned family (B)** - all 12,960 m19 tooth vectors with
`v_23` fixed at the twin's own value 4, which is exactly the (B) column reported
at every other step below. **The full family (A) at m23 is NOT measured.**

    m23, pinned family, 12,960 members, exhaustive and exact:
      F(m23)       twin 34   min 27   median 37   max 57   percentile 11.9%
      F_2(m23)     twin 39   min 35   median 45   max 65   percentile  3.1%
      increment    twin  3   min  0   median  2   max 24   percentile 56.0%
      budget slack twin -14  min -22  median -14  max  9   percentile 49.3%

**The ~20% plateau holds** (11.9% against 20.0 / 18.1 / 26.4 / 17.1 at m11..m19 -
five machines, no drift toward the median), **and the depth trend is confirmed:
`F_2` sits at the 3.1 percentile, far below `F`'s own 11.9%.** The two largest
machines now both say the twin's advantage GROWS WITH DEPTH while m13/m17 said
the opposite - and depth is where the route lives. The increment (56.0%) and the
budget slack (49.3%) are undistinguished, matching 5A.2(iv).

FOURTH INDEPENDENT AGREEMENT with Constructor's R68 increment table: the twin's
19->23 increment is 3 against a cap of 8, their fourth entry against their
fourth cap, from a completely different vehicle.

### 5A.2 THE STEP STATISTICS

For a step `M -> M + q'` the family is `V(y')`, which factors exactly as
`V(y) x {1..(q'-1)/2}`, so both the old machine's teeth and the new gear's tooth
vary. Column (A) is that full family; column (B) pins `v_q'` to the twin's own
value (the cleaner null model for "given the new gear, is the OLD machine's
arithmetic favourable?").

    step        F(M+q')      increment      budget slack   law margin
                 (A)/(B)       (A)/(B)        (A)/(B)      s_min - inc (A)
    5->7      66.7 / 75.0   66.7 / 75.0    83.3 / 75.0       41.7%
    7->11     20.0 / 25.0   25.0 / 25.0    15.0 / 25.0       83.3%
    11->13    18.1 / 15.0   23.6 / 21.7    32.5 / 28.3       78.9%
    13->17    26.4 / 28.3   61.5 / 60.8    59.0 / 58.6       66.8%
    17->19    17.1 / 17.9   14.9 / 13.9    37.2 / 38.7       82.2%

Readings, in order of how much they matter:

(i) **THE LAW MARGIN IS THE FAVOURABLE ONE, CONSISTENTLY.** `s_min - inc` is the
    slack the increment law actually has at a member. The twin sits at the
    66.8-83.3 percentile of it at the four non-degenerate steps - i.e. THE TWIN
    MACHINE USES LESS OF THE INCREMENT LAW'S BUDGET THAN TWO THIRDS TO FOUR
    FIFTHS OF ITS OWN COUNTERFACTUALS. That is the statement the route wants,
    and it is stronger and steadier than the raw-increment placement.

(ii) **THE INCREMENT LAW IS NOT GENERIC.** Over the full family it is VIOLATED
     by 0 / 13.3 / 13.9 / 14.5 / 21.7 percent of members at the five steps, and
     the rate GROWS with the machine. So no argument that uses only "same gears,
     same density, symmetric teeth" can prove it: the law needs the arithmetic.

(iii) **AND MOST OF WHAT IT NEEDS IS THE NEW GEAR'S TOOTH.** Pinning `v_q'` to
      `round(q'/6)` and letting the old machine's teeth range freely drops the
      violation rate to 0 / 0 / 0 / 1.1 / 6.5 percent. The new gear's tooth
      position carries most of the law, the old machine's arithmetic the rest -
      a decomposition of the law's difficulty that the counterfactual frame can
      state and no scan of the real machine can.

(iv) **THE BUDGET SLACK IS THE UNFAVOURABLE ONE - the honest negative.** At the
     two largest steps the twin sits at 59.0% and 37.2%, i.e. essentially
     undistinguished. The twin machine's advantage does NOT show up in
     `F(M+q') - F(M) - q'`. (Reported here because it was measured as a free
     byproduct of the same sieves; the budget-slack null model is the manager's
     item U13 and this row is offered as an independent replication, not as a
     claim of that item.)

(v) Pinning `v_q'` (column B) never moves any placement by more than about 2
    percentile points except at the degenerate 2-gear step.

### 5A.3 THE THIRD MECHANISM IS DEAD TOO - AND IT DIED THE SAME WAY

Round 27 killed angular coherence (refuted in the sign) and "the teeth are the
reciprocal of a small integer". U12(ii) named the next candidate: gears 5 and 7
decide every `<= 5`-point shape (the completeness lemma), so the effect should be
localised in `(v_5, v_7)`, of which the twin's `(1,1)` is one of six classes.

**REFUTED, and in the same direction as the first candidate.** One-way variance
decomposition over the exhaustive family:

  * The gear whose tooth explains the most variance in `F` is **gear 7** at
    m13/m17 (`eta^2` = 0.092 / 0.091) and **gear 11** at m19 (0.066). It is
    NEVER gear 5, and `eta^2` is NOT monotone in `q`. No single gear explains
    more than 9% of the variance at any machine.
  * The twin's own `v_q` is the argmin of the marginal `F` profile for 0 of 4,
    0 of 5 and 1 of 6 gears at m13/m17/m19. On gears 5 and 7 it is the ARGMAX.
  * The twin's class `(v_5, v_7) = (1,1)` has the HIGHEST mean `F` of all six
    classes at m13 (14.57 vs family 12.94) and m17 (22.12 vs 19.65), and is
    joint-highest at m19 (28.48 vs 27.90).
  * **Inside that worst class the twin is at the 1.7 / 6.9 / 4.6 percentile**,
    far more extreme than its overall 18.1 / 26.4 / 17.1.

Conditioning ladder at m19 (pin the twin's own value on a growing prefix of the
gears and re-rank inside the survivors): 17.1% -> 11.4% -> 4.6% -> 7.1% ->
26.4% -> 22.2% -> 50%. The percentile DEEPENS while the pinned set is small and
then dilutes only as the sub-family collapses to 72, 9 and 1 members, where the
number is no longer informative. Pinning the LARGE gears instead keeps it at
17.9-33.3%.

**So the pattern of round 27 repeats with a sharper conditioning variable: the
twin vector is a low-F outlier INSIDE the high-F class on every axis anyone has
proposed. The effect is not a main effect of any gear's tooth; it is an
interaction spread over the whole vector.** Three candidate mechanisms are now
dead and the anomaly stands unexplained - which by the project's measurement
directive is the target, not the wall.

## 5B. ROUND-29 EXTENSION: WHAT THE INCREMENT LAW'S RESIDUAL VIOLATORS ACTUALLY ARE

Lateral, round 29 (2026-09-03).  Status: SCRIPT-VERIFIED, exhaustive and exact.
`research/tooth_resid_r29.py --steps small` (21 gates) and `--steps 19_23`
(9 gates); logs `research/data/r29/tooth_resid_{small,1923}.log`; tables
`research/lateral_r29_results.txt` block A.

Section 5A(iii) established that pinning the incoming gear's tooth to
`v_q' = round(q'/6)` drops the increment law's violation rate over the family
from 13-22% to 0-6.5%.  This section asks what the RESIDUAL violators are, and
tests the shape Constructor proposed for them.

### 5B.1 FIRST, A STRUCTURAL RESULT THE FAMILY DELIVERS FOR FREE

> **THE RECORD LAW IS FAMILY-WIDE.**  For every member of the counterfactual
> family, at every step,
>
>     max( F_2(M),  max_{J >= 3} Q*_J(M; q') )  =  F(M + q')
>
> exactly.  Asserted at 30 + 180 + 1440 + 12960 + 12960 = **27,570
> counterfactual machines**, zero exceptions.

`Q*_J` is the maximal span of a word-legal `J`-window (middles `0` or `+-2v_q'`
mod `q'`, nonzero classes strictly alternating, padded middles transparent).
Constructor's attainment theorem (R68) proves this for the TWIN machine from CRT
plus the two-tooth structure; both ingredients survive moving the teeth, so the
theorem should be family-wide - and it is.  **That is the sharp localisation of
where arithmetic enters the route:** the RECORD LAW is structural, and only the
SIZE of `Q*_J` is arithmetic.  (D) and the increment law are the arithmetic half;
the identity that computes `F(M+q')` from the old machine is not.

### 5B.2 CONSTRUCTOR'S CONGRUENCE SHAPE IS REFUTED AS A CHARACTERISATION

Round 28's violator anatomy was "a palindrome whose central letter is the old
record", so the natural predicate is
`Pcong := F(M) mod q' in {0, A, B}` with `A = 2v_q' mod q'`, `B = q' - A`
("the old record is congruent to a tooth difference").  Over the pinned family:

    step     violators   Pcong sensitivity  PPV    specificity  best "F mod q' in S"
    13->17    2 / 180      0.0%             0.0%   88.8%        88.5% (2 violators)
    17->19   94 / 1440    34.0%             9.8%   78.2%        64.6%
    19->23  745 / 12960    5.6%             6.5%   95.0%        57.9%

At the largest step **94.4% of the residual violators have `F(M)` NOT congruent
to a legal letter**, and the depth-3 attaining middle IS the old record in
**0.0%** of them.  The best predictor of the form "`F(M) mod q'` in `S`", chosen
optimally over all `S` by greedy search, reaches only 57.9% balanced accuracy -
barely above chance.  So the answer to the question "is the residual set one
congruence condition on `F(M)`?" is **no**, and decisively so at the machine
where it matters most.

### 5B.3 WHAT IT IS INSTEAD: A DEPTH-4 WORD-LEGAL WINDOW

`P3 := Q*_3 > F_2 + s_min` (a depth-3 word-legal window beating the budget) is
SOUND - it implies violation at all 27,570 members, zero false positives - but
INCOMPLETE, and increasingly so:

    step     agreement of P3   violators needing J >= 4   attaining-depth split
    13->17   100.000%          0                          J=3:2
    17->19    97.569%          35 of 94   (37%)           J=3:47 J=4:43 J=5:4
    19->23    95.949%          525 of 745 (70%)           J=3:176 J=4:425
                                                          J=5:136 J=6:8

**Depth 4 is the MODE at 19->23, and depth 6 is populated.**  Counterfactual
machines therefore exist whose kill arity exceeds the real m19's `A_kill`, which
is exactly what one wants of a null family: the real machine's shallow `J_max`
is itself arithmetic, not structural.

### 5B.4 THE CORRECT ELEMENTARY NECESSARY CONDITION IS THE PEEL BOUND

`F_2 >= g_L + w` and `F_2 >= w + g_R`, so a depth-3 window's span is at most
`F_2 + min(g_L, g_R)`.  Hence

> `Q*_3 > F_2 + s_min` forces **MIN FLANK `> s_min`** at the attaining window.

Asserted at all 27,570 members.  It is a condition on the FLANKS, not on the
middle: 41-100% of depth-3 violators have their middle equal to the MINIMAL
legal letter `s_min`.  (I pre-registered the opposite - that the middle must
exceed `s_min`, "because `g_L + g_R <= F_2`" - which is false: `g_L` and `g_R`
are at lag 2, not adjacent.  Scored REFUTED, by my own gate.)

### 5B.5 HOW MUCH OF THE LAW IS SPECTRAL AND HOW MUCH IS ARITHMETIC

Constructor's spectrum-plus-depth certificate uses no congruence at all:
`SPEC_J := max(F_2 .. F_J) <= F_2 + s_min`.  Over the pinned family:

    step     SPEC_3 holds  unsound at  SPEC_4  unsound at  SPEC_5  unsound at
    13->17    85.0%          0          22.8%      0        0.6%       0
    17->19    79.5%         30          20.6%      0        1.1%       0
    19->23    87.7%        437          14.5%      5        0.3%       0

`SPEC_5` is sound at every step tested and certifies **0.3-1.2%** of the family;
word-legality certifies 96-100%.  **The arithmetic (word-legality) is worth
roughly a hundredfold in coverage over the purely spectral certificate** - a
number the counterfactual frame can state and no scan of the real machine can.

### 5B.6 CROSS-CHECKS AND SCOPE

The 19->23 rung reuses round 28's gated `F(m23)` table
(`research/data/r28/tooth_m23_pinned.npy`); this round re-sieves all 12,960
`m19` members independently and asserts `F(m19)` and `F_2(m19)` agree cell for
cell, and re-derives `F(m23)` from scratch at 400 randomly sampled members
(400/400 agree).  The pinned violation rates 0 / 0 / 1.11 / 6.53 / 5.75 percent
reproduce round 28's 0 / 0 / 1.1 / 6.5 / 5.7 by a second vehicle that also
carries `Q*_J`.  The FULL (unpinned) family at 19->23 is still not measured.

## 6. PRIOR-ART CHECK

Not yet checked. Terms to run: "Jacobsthal function admissible tuple dependence";
"maximal gap reduced residue system varying residue classes extremal"; "covering
systems two residues per prime largest uncovered run"; "Jacobsthal function
g(n,k) / Erdos-Rankin tooth placement extremal". NOTE: the closest classical
object is the extremal question "choose one (or two) residue classes per prime to
maximise the longest uncovered run", which IS studied (Erdos-Rankin,
Ford-Green-Konyagin-Tao style constructions). The DELTA to check is the
DISTRIBUTION over all choices and the LOCATION of the arithmetically-forced
choice inside it, which is a different question from the extremum.
