# The small-K theorems - working record

PROVER-WRITER lane, round 53. Parent: `research/proof/dead_branches_reopened_2.md`,
"Reading the file as a whole", item 1: *"Prove the small cases: the adversarial lemma to
K = 10 is within reach as a written theorem; A(K)'s exact values to K = 5 or 6 by the
dictionary mechanism. New mathematics, bounded, and the base a later induction would need."*

Scripts in `research/anchor235/r53/` (prefix `sk_`); result outputs, untracked, in
`research/anchor235/r53/results/`. The finished statements and proofs are
`docs/proofs/20-adversarial-lemma-small-K.md`; this file is the working: what was
pre-registered, what was verified, what failed.

---

## 0. Pre-registration (written before any computation this round)

### 0.1 The objects, fixed

Column `k` is the pair `(6k-1, 6k+1)`. A **gear** is a prime `g >= 5`; it **strikes** column
`k` iff `k = +-u_g (mod g)` with `u_g = 6^{-1} (mod g)`. The two teeth sit at separation
`d_g = 2 u_g = 3^{-1} (mod g)`; the **short arc** is `a_g = min(d_g, g - d_g)`, the **long
arc** is `g - a_g`, and `3 a_g = g -+ 1`.

A set `S` of gears **covers** `L` if there are phases (equivalently, integer translates of the
run) making every one of `L` consecutive columns struck by some member of `S`. `F(S)` is the
least `L` that `S` cannot cover; `A(K) = max{F(S) : |S| = K, S a set of K distinct primes >= 5}`.
Recorded exact values (`research/proof/arc_multiset.md` R1):

    K      1   2   3   4   5   6   7   8   9  10  11  12
    A(K)   2   5   7  16  22  28  37  45  68  88 101 115

`W(K) = (p_{K+1}^2 - 1)/6` with `p_{K+1}` the `(K+1)`-th prime above 3, i.e. `W = 8, 20, 28,
48, 60, 88, 140, 160, 228, 280` at `K = 1..10`. The open lemma is `A(K) < W(K)` for all `K`.

**Convention note, pre-registered because the brief uses the other one.** `A(K)` is the least
UNCOVERABLE length, not the longest coverable one. So `A(4) = 16` says "some four primes cover
15 consecutive columns and no four primes cover 16", and the upper-bound half of `A(4) = 16` is
"no 4-set covers 16", not "no 4-set covers 17". The brief's "no K-set covers 17 or 23" is the
other convention; the theorems below are stated in the project's own.

### 0.2 Theory

Two statements are within reach as written theorems, and the second is the base of the first.

* **Theorem A.** For `K <= 10`, no `K` primes above 3, each striking two residue classes at its
  own separation `3^{-1} (mod g)` with any phase, cover `W(K)` consecutive columns.
* **Theorem B.** `A(K) = 2, 5, 7, 16, 22, 28` at `K = 1..6`, with the mechanism (hole-distance
  dictionary of the small part + the domino supply of the big part) written out.

The distortion lane (`research/proof/distortion_method.md` R7, and `the_wall.md` 5f) records
"the localised budget proves the adversarial lemma for `K <= 10`". This round's first job is to
audit that claim line by line.

### 0.3 Predictions, and what refutes each

* **P1 (the audit).** The localised-distortion argument is NOT a proof of Theorem A: it applies
  a localised form of BBMST Theorem 3.1 that is not proved anywhere, and the lane's own text
  says so conditionally ("if a localised Theorem 3.1 is valid"). REFUTED if the localisation can
  be derived from the paper's Theorem 3.1 or re-proved directly on an interval.
* **P2 (the envelope).** The specific quantity `eta_max` the lane computes is not a proved upper
  bound on any localised second moment: the step `alpha <= 2/m` (a gear puts at most two teeth
  in a fibre) is false once the fibre holds more than `g` columns, and the code repairs that
  regime by taking `max(4/g^2, ...)`, i.e. by substituting the Cauchy-Schwarz LOWER bound.
  REFUTED if both branches are valid upper bounds on `E[alpha^2]`.
* **P3 (the worst set).** The lane asserts without proof that the `K` smallest gears are the
  worst `K`-set for the budget. Predicted TRUE numerically for `K <= 12` over prime pools to 200
  (so the arithmetic of the table stands even though the argument does not). REFUTED by any
  `K`-set with a larger threshold `L*`.
* **P4 (the replacement route).** The type lemma makes the pool finite at each level, so the
  statement "no `K` primes cover `L` columns" is a finite check for each `(K, L)`; predicted
  that an independent, self-contained implementation reproduces `A(K) = 2, 5, 7, 16, 22, 28` by
  exhaustive search and `37, 45, 68, 88` by exact integer programming. REFUTED by any
  disagreement with the recorded ladder.
* **P5 (Theorem A).** `A(K) < W(K)` at `K = 1..10`, certified. Since coverability is monotone
  decreasing in `L`, infeasibility at `L = A(K) <= W(K)` gives Theorem A. REFUTED by a cover of
  `W(K)` columns by `K` primes.
* **P6 (A(3) by hand).** `A(3) = 7` has a complete hand proof: at `L = 7` only 5 and 7 are
  small, so the case split is over the four subsets of `{5,7}`, and each closes on the parity of
  the hole distance plus the arc supply. Predicted: 4 cases, no computer needed.
* **P7 (A(4), A(5) by hand).** Predicted PARTLY: the big-part cases close by the arc supply and
  parity as at `K = 3`, but the all-small cases (`{5,7,11,13,17,19,23}` at `L = 16`, and the
  larger pool at `L = 22`) need a finite enumeration over phases. Predicted that the enumeration
  is small (product of the small gears' periods) and that its completeness is the type lemma.
* **P8 (the brief's optimal sets).** The brief names `{5,7,11,17}` and `{5,7,11,13,17}` as the
  optima at `K = 4, 5`. Predicted: the first is right and the second is WRONG - `{5..17}` covers
  only 17 columns (`F = 18`), and the `K = 5` optimum is `{5,7,11,23,29}` (`arc_multiset.md` R2).
  REFUTED if `{5,7,11,13,17}` covers 21 columns.

### 0.4 Scorecard

| # | prediction | verdict | evidence |
|---|---|---|---|
| P1 | distortion route is not a proof | **CONFIRMED, and stronger** | the localised inequality is not merely unproved, it is FALSE: eight covering sets with `eta < 1` on the interval they cover (section 2, part D) |
| P2 | `eta_max` is not a proved upper bound | **CONFIRMED** | five per-gear instances where the exact localised second moment exceeds the envelope (section 2, part C) |
| P3 | `K` smallest gears are worst for the budget | **CONFIRMED numerically, `K <= 6`** | `sk_distortion.py` part B, exhaustive over the 14 smallest gears; not tested above `K = 6` |
| P4 | independent code reproduces the ladder | **CONFIRMED** | `sk_gate.py` (F ladder to m31, `A(1..6)`), `sk_theoremA.py` (`A(1..10)`) |
| P5 | Theorem A certified at `K <= 10` | **CONFIRMED** | `sk_theoremA.py --window`, infeasible at `L = W(K)` itself |
| P6 | `A(3) = 7` by hand | **CONFIRMED** | 2 cases, 4 hole sets, 12 pairings; written out in `docs/proofs/20` and gated by `sk_head.py` |
| P7 | `A(4)`, `A(5)` partly by hand | **CONFIRMED, and better than predicted** | the split by `g < L` (not by the type lemma) leaves 5 and 18 cases, not the ~20 and ~73 the type-lemma split gives; at `K = 4` all five close on one number each |
| P8 | the brief's `K = 5` optimum is wrong | **CONFIRMED** | `F({5,7,11,13,17}) = 18`, not 22 (`sk_gate.py` G3) |

(The verdict column is filled in below from the runs; the predictions above were written first.)

---

## 1. What was verified

### 1.1 Gates (`sk_gate.py`)

Both engines are written from scratch in `sk_core.py` (standard library only; no import from
r48, r50 or r51), so agreement with the record is a genuine independent check.

* **G1.** Engine (1) - a direct exhaustive search over the phases of an explicit prime set -
  reproduces the certified record ladder `F({5..q}) = 5, 7, 11, 18, 25, 34, 43, 58` at
  `q = 7, 11, 13, 17, 19, 23, 29, 31`. (The `q = 31` run takes 194 s; the rest are seconds.)
* **G2.** Engine (2) - the type-reduced search over ALL primes - reproduces
  `A(1..6) = 2, 5, 7, 16, 22, 28`.
* **G3.** The optimal sets of `arc_multiset.md` R2 have the recorded records:
  `F({5,7,11,17}) = F({5,7,11,19}) = 16`, `F({5,7,11,23,29}) = F({5,7,11,23,31}) = 22`,
  `F({5,7,11,17,23,37}) = F({5,7,11,13,19,47}) = 28`.
* **G4.** Every type-reduced cover found at `K <= 5`, `L < A(K)` (47 of them) is realisable by
  an explicit set of distinct primes, checked by engine (1). So the type reduction is not
  inventing covers.

### 1.2 Theorem A (`sk_theoremA.py`)

The 0/1 feasibility program over the type-reduced item list is **infeasible at `L = W(K)` for
every `K = 1..10`** - the literal statement, with no appeal to monotonicity:

    K            1     2     3     4     5     6     7     8     9    10
    W(K)         8    20    28    48    60    88   140   160   228   280
    binaries    53   274   513  1358  1916  3696  9272 12162 23342 34099
    seconds    0.0   0.0   0.0   0.0   0.3   1.3   2.6   3.3   7.2  12.5

and infeasible at `L = A(K) = 2, 5, 7, 16, 22, 28, 37, 45, 68, 88` as well (2 to 3,696
binaries, at most 3.8 s), while explicit covers of `A(K) - 1` columns exist:

    K   L=A(K)-1   gears at their phases
    1     1        5@0
    2     4        5@0, 7@3
    3     6        5@0, 7@3, 11@0
    4    15        7@0, 5@1, 11@9, 17@4
    5    21        5@0, 11@8, 29@3, 7@4, 23@6
    6    27        5@3, 23@1, 11@2, 37@16, 7@0, 17@5
    7    36        5@0, 17@12, 31@13, 11@4, 19@12, 7@2, 13@11
    8    44        5@3, 7@3, 13@6, 29@4, 19@7, 83@9, 31@21, 11@1
    9    67        5@3, 23@1, 37@2, 11@0, 17@0, 7@0, 13@7, 31@10, 47@36
   10    87        5@3, 17@12, 7@2, 11@0, 13@6, 19@4, 23@16, 37@2, 79@57, 29@24

(phase = the column of one tooth; the other tooth is `3^{-1} (mod g)` further on). So
`A(K) < W(K)` at `K = 1..10`, ratio `0.25, 0.25, 0.25, 0.33, 0.37, 0.32, 0.26, 0.28, 0.30,
0.31`. P4 and P5 confirmed.

### 1.3 Theorem B, and the split that makes it human (`sk_cases.py`, `sk_head.py`)

The round's own contribution to the proof is the SPLIT. The recorded split is by the type
lemma - small means `g - a_g <= L-1`, i.e. `g <= 1.5 L` - which at `L = 28` puts eleven gears in
the small pool and leaves 462 subsets to check. The right split is by `g < L`: a gear `g >= L`
meets each of its classes at most once in the run and so strikes at most TWO columns. That
gives `T(16) = {5,7,11,13}`, `T(22) = {5,..,19}`, `T(28) = {5,..,23}`, and with the counting
filter

    sum_{g in S} maxstrike(g, L) + 2 (K - |S|) >= L

the number of surviving cases collapses:

    K   L    |T(L)|   cases surviving the counting filter   phase vectors   search nodes
    2    5      0        0                                        0              0
    3    7      1        1                                        5              6
    4   16      4        5                                    6,595          1,005
    5   22      6       18                                  756,870         15,475
    6   28      7       53                               29,418,815        266,643

(the node column is what the search actually visits once the prune "the gears still to place
cannot bring the hole count down to `2(K-|S|)`" is applied; the phase-vector column is the
nominal product of the periods)

and every case is closed. The verdicts, in the shape the written proof uses:

* At `K = 4` **all five cases close on one number each**: the largest number of columns of a
  16-run the small part can strike, over all its phase vectors, is 11, 13, 13, 11, 14 for
  `{5,7}`, `{5,7,11}`, `{5,7,13}`, `{5,11,13}`, `{5,7,11,13}`, leaving 5, 3, 3, 5, 2 holes
  against the 4, 2, 2, 2, 0 that the auxiliary gears could take. No arc argument is needed.
* At `K = 5`, 14 of 18 cases close on the hole count; 4 need the matching filter, and they are
  the mechanism: `{5,7,11,13}` can get down to holes `[10,15]` (distance 5), `{5,7,11,17}` to
  `[15,17]` (distance 2), `{5,7,11,19}` to `[10,15]` (distance 5), and by the span lemma a
  distance of 5 is spannable only by `(3*5 -+ 1)/2 = 7, 8` and a distance of 2 only by
  `3*2 -+ 1 = 5, 7` - gears 5 and 7, which are inside `S` already. `{5,7,11}` reaches four
  holes `[4,10,15,17]` whose six distances 6, 11, 13, 5, 7, 2 are spannable only by 17/19, 17,
  19, 7, 11, 5/7 respectively - every one of those primes is below 22, so none of them is an
  auxiliary gear, and no pair forms at all.
* At `K = 6`, 51 of 53 close on the hole count; two need the matching filter
  (`{5,7,11,17}` and `{5,7,11,17,23}`).

**The head collision** (new, and it is what closes the tight case at every `K` from 4):
`maxstrike(5,L) + maxstrike(7,L)` is 12, 16, 20 at `L = 16, 22, 28` but the two gears can
strike together at most 11, 15, 18 columns - a deficit of 1, 1, 2. The counting filter's tight
case is always `S = {5,7}`, and the deficit kills it.

**The span lemma** (new in this form): a pair of columns at distance `t` inside a run shorter
than the gear is struck by `g` only if `t = a_g` (t even, `g = 3t -+ 1`) or `t = g - a_g`
(t odd, `g = (3t -+ 1)/2`). At most two primes span any distance, and the distance names them.
This is what turns `arc_multiset.md` R7's measured hole-distance dictionary into a rule, and it
is what makes the `A(3) = 7` proof complete rather than sketched.

### 1.4 An independent second route at `K <= 5` (`sk_theoremB.py`)

For every `K`-subset `S` of the type lemma's small pool `Sm(L)`, an exhaustive search whose
concrete gears are restricted to `S` (dominoes and singles still available, budget `K` gears).
Any `K`-set's small part sits inside one such `S`, so the enumeration is complete. At
`L = A(K)`: 1, 1, 1, 35, 126 subsets at `K = 1..5`, **none covering**; and at `L = A(K) - 1` a
cover is found, as it must be. This uses no solver and a different decomposition from
`sk_cases.py`, and agrees.

---

## 2. The audit of the distortion route

`distortion_method.md` R7 and `the_wall.md` 5f record: *"the localised budget proves the open
lemma `A(K) < (p_{K+1}^2-1)/6` for every `K <= 10`, and fails from `K = 11`"*, and 5f lists it
as the one positive of the covering round. The chain of steps is:

* **S1.** BBMST Theorem 3.1: `eta = sum_i min{M1_i, M2_i/(4 d_i(1-d_i))} < 1` implies the system
  does not cover `Z`, the moments taken over the fibres of
  `Z_{Q_i} = Z_{Q_{i-1}} x Z_{p_i}` under the method's own reweighted measures.
* **S2.** Localise: replace `Z_Q` by an interval `I` of `L` columns with the uniform measure and
  the fibres by (class mod `Q_{i-1}`) intersect `I`, and assert the same conclusion for `I`.
* **S3.** Bound the localised `eta` above by
  `eta_max = sum_g max(4/g^2, min(1, 2/m_g) min(1, 2/g + 2/L))`, `m_g = L/Q_{<g}`, and let
  `L*(S)` be the least `L` with `eta_max < 1`.
* **S4.** Take the worst `K`-set to be the `K` smallest gears, so `A(K) <= L*max(K)`.
* **S5.** Read off `L*max(K) < W(K)` for `K <= 10`.

**S5 and the arithmetic reproduce exactly** (`sk_distortion.py` part A): `L*max` =
2.00, 5.03, 10.32, 16.07, 24.98, 40.34, 76.55, 106.5, 158.9, 254.6, 632.9, 1025 at
`K = 1..12`, below `W` through `K = 10` and above from `K = 11`. So the numbers on the record
are right.

**S4 holds numerically** (part B): over every `K`-subset of the fourteen smallest gears,
`K = 1..6`, the maximum of `L*` is attained at the `K` smallest gears, with no tie elsewhere.
P3 confirmed - but the lane asserts it, it does not prove it, and it is needed for the
conclusion.

**S3 is false as a bound** (part C). The envelope is compared with the exactly computed
`max_phase E[alpha_g^2]` over the fibres of the interval under the uniform measure - the same
object `dm_budget.py` tabulates. The envelope is BELOW the truth at, among others,

    g = 5,  L = 88   (K=6 window):   true 0.16736   envelope 0.16000
    g = 5,  L = 308  (K=11 window):  true 0.16208   envelope 0.16000
    g = 7,  L = 60   (K=5 window):   true 0.09167   envelope 0.08163
    g = 7,  L = 610  (real m59):     true 0.08236   envelope 0.08163
    g = 11, L = 610  (real m59):     true 0.03571   envelope 0.03306

The reason is in the code: `sup alpha <= 2/m` ("a gear puts at most two teeth in a fibre") is
false as soon as a fibre holds more than `g` columns - a fibre of `m` columns meets each class
of `g` about `m/g` times, so it holds about `2m/g` teeth - and the regime is repaired by taking
`max(4/g^2, ...)`, i.e. by substituting the Cauchy-Schwarz LOWER bound as if it were an upper
one. P2 confirmed, with five refuting instances.

**S2 is false, not merely unproved** (part D). Take a gear set that DOES cover an interval and
evaluate the localised budget on that very interval:

    K   covering set                          covers L   exact localised eta
    2   {5,7}                                     4            0.7500
    3   {5,7,11}                                  6            0.9167
    4   {5,7,11,17}                              15            0.6933
    5   {5,7,11,23,29}                           21            0.6980
    6   {5,7,11,17,23,37}                        27            0.8302
    7   {5,7,11,13,17,19,31}                     36            0.9628
    8   {5,7,11,13,19,29,31,83}                  44            0.9325
    9   {5,7,11,13,17,23,31,37,47}               67            0.9233
   10   {5,7,11,13,17,19,23,29,37,79}            87            1.0081

Eight of the nine have `eta < 1` on an interval they cover. So "the localised `eta < 1` implies
no cover of `I`" is FALSE, and no repair of the estimates can rescue it: the hypothesis simply
does not imply the conclusion on an interval. P1 confirmed, and more sharply than
pre-registered - the pre-registration said "unproved", the measurement says "false".

**What `eta_max` actually is** (part E). Decomposed at `L = W(K)`, every gear except 5 and 7 is
collapsed, and its term is `2/g + 2/L`, i.e. its own capacity; summed over all gears that is
the union bound `sum_g 2 ceil(L/g) / L`, a genuine theorem. The union bound is 1.083, 1.200,
1.296, 1.371, 1.438, 1.491, 1.550 at `K = 4..10`: vacuous everywhere. The head is gears 5 and 7
and contributes 0.2416 where their capacity is 0.6857. **So the entire margin that puts
`eta_max` below 1 for `K <= 10` is the replacement of two gears' capacity by `4/g^2`, worth
0.44, and that replacement is the step with no proof and (by part D) no truth behind it.**

The record entry is therefore withdrawn, and Theorem A stands in its place.

---

## 3. What failed, and what was corrected

* **The distortion "positive" is withdrawn.** Not repaired: the localised inequality is false
  (section 2, part D). This is a correction to `the_wall.md` 5f and `distortion_method.md` R7
  and section 6 item 3.
* **The brief's `K = 5` optimum is wrong.** The brief names `{5,7,11,13,17}` as the optimal
  5-set. `F({5,7,11,13,17}) = 18`, not 22; the optimum is `{5,7,11,23,29}` or `{5,7,11,23,31}`
  (`arc_multiset.md` R2), which covers 21. P8 confirmed.
* **The brief's convention is off by one.** It asks for "no `K`-set covers 17 or 23" at
  `K = 4, 5`. In the project's convention `A(K)` is the least uncoverable length, so the
  statements proved are "no 4-set covers 16" and "no 5-set covers 22". Both were proved; the
  brief's version follows from them by monotonicity, but it is weaker.
* **The recorded one-paragraph proof of `A(3) = 7` is incomplete.** `arc_multiset.md` R7 argues
  "at `L = 7` every gear except 5 and 7 is big, `{5,7}` leaves at best two holes at distance 1,
  2 or 5, and a big gear can join two holes only at an even distance some prime realises - which
  leaves distance 2 alone, offered by 5 and 7 and by no other prime, and both are already
  spent." Three gaps: (i) it assumes the 3-set contains both 5 and 7, and the case where 7 is an
  auxiliary gear (small part `{5}` alone, four holes, two pairs) is not treated; (ii) "a big
  gear can join two holes only at an even distance" is true but the reason - the long arc is
  odd - is not given, and an ODD distance is joinable, by the gear `(3t -+ 1)/2`, which is why
  distance 5 has to be excluded by name (it is gear 7's long arc); (iii) the counting that
  rules out the 3-sets missing 5 or 7 is not stated. The proof written in `docs/proofs/20` fills
  all three.
* **The exhaustive DFS is not the tool at `K >= 7`.** Confirmed again: `A_exact(6)` takes 44 s
  and `K = 7` does not finish. The 0/1 program settles `K = 10` at `L = 280` in 12.5 s. The
  cost of the honest certificate is therefore concentrated in a solver whose infeasibility
  proof is not independently checkable; the four corroborations are listed in `docs/proofs/20`
  Status.
* **What is still not proved.** No induction step. `A(K) < W(K)` for `K >= 11` is untouched -
  `A(11) = 101 < 308` and `A(12) = 115 < 368` are on the record as computations, not proofs
  here, and nothing above extends to general `K`. The residual is unchanged: a lower bound on
  the tiler function `h_S(L)`.

---

## 4. Verdict

Two theorems written and proved:

* **Theorem A** (`K <= 10`, no `K` primes at fixed separation cover `W(K)` columns) - proved,
  the case work by reasoning, the ten instances by exact finite certificate.
* **Theorem B** (`A(K) = 2, 5, 7, 16, 22, 28`) - proved; `K <= 3` entirely by reasoning plus a
  twelve-line table, `K = 4, 5, 6` by reasoning down to 5, 18 and 53 stated cases each settled
  by exhaustive enumeration.

Two lemmas that are new tools rather than bookkeeping: the **span lemma** (a distance names the
at most two primes that can span it) and the **head collision** (gears 5 and 7 cannot both be
maximal and disjoint). One record entry withdrawn: the localised distortion budget does not
prove the adversarial lemma, and the inequality it uses is false on an interval.
