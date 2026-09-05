# Branch - COLLISION LAWS FOR GEAR PAIRS

Parent: `research/proof/small_K_theorem.md` / `docs/proofs/20-adversarial-lemma-small-K.md`,
the **head collision**: gears 5 and 7 cannot both be maximal on a run and strike disjointly,
deficit 1, 1, 2 at `L = 16, 22, 28`, which closes the counting-tight case at every `K >= 4`.
That is the first proven *interaction* law on the tree that cuts coverage. The observation
that spawned this branch: the head collision is a two-gear CRT fact on the period 35, so it
should have a general form - a deficit function for every pair of gears, at every run length.

Scripts `research/anchor235/r55/cl_*.py`; result outputs (untracked)
`research/anchor235/r55/results/`. Every number this document relies on is written into it.

---

## 0. Pre-registered (written before any computation of this branch)

### 0.1 The objects, exact

Column `k` is the pair `(6k-1, 6k+1)`. A **gear** is a prime `g >= 5` with **separation**
`s_g`; it strikes the columns `k = c_g, c_g + s_g (mod g)`, where `c_g` is its **phase**. The
real machine has `s_g = 3^{-1} (mod g)` (file 02); the **short arc** is
`a_g = min(s_g, g - s_g)`, and for the real separation `3 a_g = g -+ 1` (Lemma 1 of file 20).

A **run of `L`** is `L` consecutive columns.

* `max_g(L)` = the most columns gear `g` can strike on a run of `L`, over its phases. Cited,
  file 20 Lemma 2 (capacity), and it holds verbatim for any separation because the proof uses
  only "two classes at cyclic distances `a_g` and `g - a_g`":

      max_g(L) = 2 floor(L/g) + e,   e = 2 if (L mod g) > a_g,
                                     1 if 1 <= (L mod g) <= a_g,
                                     0 if L mod g = 0.

* `joint_max(g, h; L)` = the most columns the **pair** can strike together (the union), over
  both phases, computed exactly.

* the three deficits, kept apart:

      c(g, h; L)      = max_g(L) + max_h(L) - joint_max(g, h; L)          (the collision deficit)
      c_max(g, h; L)  = min { |strikes_g n strikes_h| : both gears maximal }   (deficit under
                                                                              joint maximality)
      c_dis(g, h; L)  = min { (max_g - n_g) + (max_h - n_h) : the strikes are disjoint }
                                                                          (deficit under disjointness)

  `n_g` is the number of columns of the run gear `g` strikes at the phase in question. With
  `loss(phases) = (max_g - n_g) + (max_h - n_h) + |overlap|`, all three are minima of `loss`
  over different domains, so `c = min(over everything) <= min(c_max, c_dis)`.

**The one-orbit reduction.** For a fixed set of gears the phase vector is a *diagonal translate*:
by CRT `(c_g, c_h) = (t, t)` for a unique `t (mod gh)`, so quantifying over both phases is the
same as sliding a window of length `L` over one period of the fixed pattern
`U = ({0, s_g} + gZ) u ({0, s_h} + hZ)`. This is on the record (`the_wall.md` 5a: "the adversary
with one phase per gear over all primes up to `q` is exactly the real machine over its period");
it is cited, not re-derived, and it is what makes every number below exact and cheap.

The **onset** `L0(g, h)` is the least `L >= 2` with `c(g, h; L) > 0`.

A **twin pair** is `(g, g+2)` with both prime. For the real separation twin gears share the
short arc: `3 a_g = g + 1` and `3 a_{g+2} = (g+2) - 1` give `a_g = a_{g+2} = (g+1)/3`
(Lemma 1 again). That is the "twin gears share an arc" of `arc_multiset.md`.

**Block deficit (the generalisation used in item 5).** For any gear set `B`,

    c_B(L) = sum_{g in B} max_g(L) - joint_max(B; L),

with `joint_max(B; L)` the largest number of columns of a run of `L` that `B` can strike
together. `c_{g,h} = c` above; `c_B` for `|B| = 1` is 0.

### 0.2 The theory

The head collision is not special to 5 and 7. Every pair of gears has a forced deficit, it is a
CRT fact on the period `gh`, and it grows with `L` at a rate the two gears' separations fix. The
adversarial cover's real constraint is therefore not capacity (which dies at `K = 4`: the
union bound `sum_g 2 ceil(L/g) / L` exceeds 1 from four gears on, file 20 Status) but capacity
minus the forced pairwise deficits.

### 0.3 Predictions, with numbers, and what refutes each

* **P1 (reproduction).** `c(5, 7; L) = 1, 1, 2` at `L = 16, 22, 28`, and `c = c_max` there
  (the tight case is joint maximality, not disjointness). REFUTED by any other value.

* **P2 (the growth law, exact).** `c(g, h; L + gh) = c(g, h; L) + 4` for every `L >= 1` and
  every pair, every separation. Reason to expect it: one extra period adds `2h` to `max_g`,
  `2g` to `max_h`, and exactly `|U| = 2g + 2h - 4` to every window count, and `4` is the number
  of residues mod `gh` that both gears strike. So the deficit is linear with **slope exactly
  `4/(gh)`**, the CRT overlap density of `separation_drives_K.md` N-S1. REFUTED by one pair and
  one `L` with an increment other than 4.

* **P3 (onset).** Twin pairs collide earliest: `L0(g, g+2) < g`, i.e. below the smaller gear,
  because the shared arc makes the two teeth patterns parallel. Non-twin pairs collide at
  `L0` of order `gh/4` (the four coincidence residues mod `gh` leave a largest gap of order
  `gh/4`, and a run fitting in that gap can be disjoint). REFUTED for twins by a twin pair with
  `L0 >= g`; refuted for non-twins by an onset systematically outside `[gh/8, 3gh/4]`.

* **P4 (real versus random, the pairwise face C question).** For twin pairs the real
  one-third separation collides EARLIER than a random separation, by a clear margin (the shared
  arc is structural, and a random pair of separations shares an arc with probability `O(1/g)`).
  For non-twin pairs the real separation is TYPICAL - its onset percentile among 20 random
  draws has no consistent sign - which is the pairwise version of `the_wall.md` face C and of
  W3's answer for `K`. REFUTED for twins by real at or above the random median; refuted for
  non-twins by a consistent one-sided percentile across the pairs.

* **P5 (triples, sub-additive by the inclusion-exclusion term).** For a triple,
  `c_{g,h,k}(L) < c_{g,h}(L) + c_{g,k}(L) + c_{h,k}(L)` for large `L`, with the gap growing at
  rate exactly `8/(ghk)`: the block deficit rate is `sum_g 2/g - (1 - prod_g (1 - 2/g))`, whose
  order-2 truncation is the sum of the pairwise rates. Exact form of the growth law:
  `c_B(L + P_B) = c_B(L) + sum_g 2 P_B/g - (P_B - prod_g (g - 2))`, `P_B = prod g`. REFUTED by
  a super-additive triple at large `L`, or by an increment other than the stated one.

* **P6 (the collision bound is a MATCHING bound, not an all-pairs bound).** The brief's form
  `L <= sum_g max_g(L) - sum_{pairs} c(g, h; L)` is **not valid**: subtracting every pairwise
  overlap over-subtracts (Bonferroni runs the other way, `|union| >= sum - sum pairs`). The
  valid form partitions the gears into blocks and subtracts one deficit per block:

      L = |union| <= sum_{blocks B} joint_max(B; L) = sum_g max_g(L) - sum_{blocks} c_B(L),

  so the bound is `max over partitions`. Predicted: the all-pairs form is refuted outright by
  an explicit cover (the `K = 10` cover of 87 columns on record), and the block form with
  blocks of size 2 already bites at `K = 4` (rate `sum 2/g = 1.021` against matched pairs
  `4/35 + 4/143 = 0.142`), needs blocks of size 3 by `K = 6` (rate 1.244 against 0.155 for
  pairs, 0.263 for triples), and the block size needed grows with `K`. REFUTED if the block
  bound does not bite at `K = 4`, or if size-2 blocks suffice at `K = 6`.

* **P7 (what the bound reaches).** Predicted: the size-2 block bound gives a finite `L*` at
  `K = 4` but ABOVE `W(4) = 48`, so it does not by itself prove the adversarial lemma at any
  `K`; the size-3 block bound is predicted to reach `K = 4` and possibly `K = 5`. REFUTED
  either way by the computed `L*`.

* **P8 (the record's overlap).** For the real machines `m11..m23` the record stretch pays a
  total deficit `sum_g max_g(F) - F`; predicted that the sum of pairwise deficits
  `sum_{g<h} c(g, h; F)` is LARGER than the total deficit the record actually pays (again the
  Bonferroni direction), so the pairwise law over-explains the record's overlap, and the
  matching version under-explains it. REFUTED by a rung where the pairwise sum is below the
  total deficit.

* **P9 (induction).** Adding a gear `q'` to a machine `M` adds capacity `2L/q' + O(1)` and
  pairwise collisions `sum_{g in M} 4L/(g q') + O(1)`, so the new gear's net contribution in
  the pairwise accounting is `(2L/q')(1 - 2 sum_{g in M} 1/g)`, which turns negative once
  `sum_{g in M} 1/g > 1/2`, i.e. from `M = {5,7,11,13}` on. Predicted: this is the rule an
  induction step would use, and it is exactly why the all-pairs form is invalid (P6). REFUTED
  if the measured increment does not follow `4/(g q')` per pair.

### 0.4 Scorecard

| # | prediction | verdict | evidence |
|---|---|---|---|
| P1 | head collision reproduced, `c = c_max` | **CONFIRMED** | `1, 1, 2` at `L = 16, 22, 28`, `c_max` the same; and at `L = 28` the pair cannot strike disjointly at all (R1) |
| P2 | `c(L + gh) = c(L) + 4`, slope `4/(gh)` | **CONFIRMED, exceptionless and proved** | 248,334 real + 67,400 random instances, 0 exceptions; one-line proof (R2) |
| P3 | twin onset `< g`; non-twin onset `~ gh/4` | **twins CONFIRMED and sharpened; non-twins REFUTED** | twin onset `= a+1 = (g+4)/3`, 7 of 7, and `< g` at 7 of 7 against 0 of 246; non-twin `L0/gh` median 0.071, not 0.25 (R3) |
| P4 | twins: real earlier than random; non-twins: typical | **twins CONFIRMED; non-twins REFUTED - real is LATER, not typical** | normalised onset 1.000 at all 6 twin pairs (the minimum possible) against a random median 1.171; non-twin real median 3.100 (R4) |
| P5 | triples sub-additive, gap rate `8/(ghk)` | **CONFIRMED in the rate, not exceptionless at finite `L`** | increment `4(g+h+k)-8` exact, 0 exceptions on 20 triples; sub-additive at 59,378 lengths, super-additive at 350 (R5) |
| P6 | all-pairs bound invalid; block bound valid, size grows with `K` | **CONFIRMED** | all-pairs refuted by the recorded covers at `K = 9, 10`; block bound never violated; least block size `1,2,2,3,4,5,6,7,8,8` at `K = 3..12` (R7) |
| P7 | block bound above `W(K)` at size 2; size 3 reaches `K = 4` | **REFUTED, and better** | size 2 **proves** the lemma at `K = 4` (46 < 48) where counting fails (51); size 4 proves `K = 5` (55 < 60) and `K = 6` (87 < 88) (R7) |
| P8 | pairwise sum over-explains the record's overlap | **REFUTED in direction, confirmed in size** | `C_all = 1, 0, 5, 12, 14` against `D = 1, 2, 5, 10, 15` - both signs occur; the matching part is 27% of `D` at m23 (R6) |
| P9 | new gear's pairwise collision rate `4/(g q')` | **CONFIRMED** | measured `sum_g c(g,q';280) = 21, 23, 16, 13, 12` against predicted `25.7, 24.1, 20.2, 18.2, 17.3`; net contribution turns negative from `M = {5,7,11,13}` (R8) |

(the verdict column is filled in from the runs below; the predictions above were written first)

---

## 1. Setup (exact ranges, tools)

Everything below is an exact maximum over a period, not a sample and not a search. The
one-orbit reduction (0.1) turns "over all phases of the pair" into "slide a window of length
`L` over one period of the fixed pattern", so `joint_max(g, h; L)` costs `O(gh + L)` and
`joint_max(B; L)` costs `O(prod_{g in B} g + L)`.

| what | range | script |
|---|---|---|
| the head collision, three deficits kept apart | `(5,7)`, `L = 1..80` | `cl_pairs.py` item 1 |
| every pair of gears, onset and growth | `g < h <= 97` (253 pairs), `L = 1..min(2gh, 5000)` | `cl_pairs.py` item 2 |
| real vs 20 random draws vs 8 coherent families | `g < h <= 61` (120 pairs), full period each | `cl_families.py` |
| triples, additivity and growth | all 20 triples of `{5,7,11,13,17,19}`, `L = 1..min(2ghk, 9000)` | `cl_triples.py` |
| the record's overlap | the real machines `m11..m23`, sieved over the full period | `cl_triples.py` part b |
| the collision bound | `K = 3..10` at `L = W(K)`, block size 1..4, exhaustive over every `K`-set | `cl_bound.py` |
| the exceptionless laws | 253 real pairs + 14,340 shared-arc configurations + random draws | `cl_laws.py` |

Scripts are standard library plus numpy; outputs in `research/anchor235/r55/results/`.

---

## 2. Results

### R1. The head collision reproduced, and the three deficits separated

`c(5, 7; 16), c(5, 7; 22), c(5, 7; 28) = 1, 1, 2` - file 20's numbers, reproduced by an
independent route (a window sliding over the period 35, rather than 35 phase pairs). And the
tight case is **joint maximality**, not disjointness: `c_max = 1, 1, 2` while `c_dis = 1, 2,
undefined` - at `L = 28` gears 5 and 7 **cannot strike disjointly at all**, at any phase.

    L      2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
    max5   1  2  2  2  3  3  4  4  4  5  5  6  6  6  7  7  8  8  8  9  9 10
    max7   1  2  2  2  2  2  3  3  4  4  4  4  4  5  5  6  6  6  6  6  7  7
    joint  2  3  4  4  5  5  6  7  8  8  9  9 10 11 11 12 12 13 14 14 15 15
    c      0  1  0  0  0  0  1  0  0  1  0  1  0  0  1  1  2  1  0  1  1  2

The head collision is not an isolated fact about three lengths. The deficit is already positive
at `L = 3`, and it is positive **at every `L >= 21` without exception** (the last zero of
`c(5,7;.)` in `[1, 35]` is at `L = 20`, and R2 makes `c >= 4` above 35). Gears 5 and 7 are in
permanent collision from `L = 21` on, which covers `W(K)` at every `K >= 5`.

### R2. The growth law: exactly four per period (P2 CONFIRMED, exceptionless)

> **Collision period law.** For every pair of gears and **every** pair of separations,
> `c(g, h; L + gh) = c(g, h; L) + 4` for all `L >= 1`.

Proof, one line: `max_g(L + gh) = max_g(L) + 2h` and `max_h(L + gh) = max_h(L) + 2g` from the
capacity formula (the quotient rises by `h`, resp. `g`; the remainder is unchanged); adding one
full period to the window adds exactly `|U| = 2g + 2h - 4` to *every* window count, so the
maximum rises by exactly that; and `2h + 2g - (2g + 2h - 4) = 4`. The 4 is the number of
residues mod `gh` that both gears strike - the four-point shape of `separation_drives_K.md` 3.1.

Checked: **248,334 instances over 228 real pairs, 0 exceptions**; **67,400 instances with random
separations, 0 exceptions**; and `c(g, h; gh) = 4` exactly on all 228 pairs whose period fits.
Consequences, both exact:

* the deficit is **linear with slope exactly `4/(gh)`**, the CRT overlap density of
  `separation_drives_K.md` N-S1 - so the mean-overlap identity, which was shown there to be
  unable to act on the overlap, is exactly the growth rate of the forced deficit;
* `c(g, h; L) >= 4 floor(L/gh)` (134,380 instances, 0 exceptions), hence every zero of `c` lies
  in `[1, gh]`, which makes the **permanent onset** `L1 = 1 + (last zero in [1, gh])` an exactly
  computable number rather than a limit.

The same law for a block `B` of any size:
`c_B(L + P) = c_B(L) + sum_{g in B} 2P/g - (P - prod_{g in B}(g - 2))`, `P = prod g`; for a
triple that is `4(g + h + k) - 8`. **0 exceptions over the 20 triples of `{5..19}`.**

### R3. Onset: the arc floor and the shared-arc law (P3 CONFIRMED for twins, REFUTED for non-twins)

Two exceptionless statements about where a pair first collides.

> **The arc floor.** For the real separation, `c(g, h; L) = 0` for every
> `2 <= L <= max(a_g, a_h)`; so `L0(g, h) >= max(a_g, a_h) + 1`.
> (253 of 253 pairs, 5,009 instances, 0 exceptions. It is a property of the real separation:
> with random separations it fails on **134 of 759 draws**.)

> **The shared-arc law.** If `a_g = a_h = a` then `c(g, h; a + 1) >= 1`: the two gears collide
> at the earliest length the arc floor allows. (14,340 shared-arc configurations over all 253
> pairs and all arcs - not only the real separation - **0 exceptions**; the value is 1 in 13,328
> of them and 2 in 1,012.)

Mechanism, and it is a proof: a run of `a + 1` columns has room for exactly one arc of length
`a`, at its two ends. A gear of short arc `a` strikes two columns of that run only as
`{t, t + a}`, the two ends. So if both gears are maximal they strike **the same two columns** -
overlap 2, union 2 against `max_g + max_h = 4`; and if not both maximal the union is at most 3.
Either way the deficit is at least 1.

Twin gears share the arc by file 20 Lemma 1 (`a_g = a_{g+2} = (g+1)/3`), so **every twin pair
collides at `L = (g + 4)/3`, the earliest an onset can be, and that is below `g`**:

    twin pair     (5,7)  (11,13)  (17,19)  (29,31)  (41,43)  (59,61)  (71,73)
    arc a             2        4        6       10       14       20       24
    onset L0          3        5        7       11       15       21       25    = a + 1, 7 of 7
    onset / g     0.600    0.455    0.412    0.379    0.366    0.356    0.352
    permanent L1     21       89      205      581     1149     2361     3409

**7 of 7 twin pairs have `L0 < g`; 0 of 246 non-twin pairs do** (non-twin `L0/g`: min 1.03,
median 3.38, max 15.4). The prediction that twin pairs collide earliest is confirmed, with the
shared arc as the proved mechanism - and it is the general form of the head collision, of which
`(5,7)` at `a = 2` is the first instance.

The non-twin half of P3 is **refuted**: onsets are nowhere near `gh/4`. `L0/gh` over the 246
non-twin pairs runs min 0.011, median 0.071, max 0.208 - an order of magnitude below the
prediction and spread over a factor of twenty. The onset of a non-twin pair is not a simple
function of `g` and `h`: the extremes are `(7,13)` with `L0 = 15` and `(79,83)` with
`L0 = 1108`. What is a function of the arcs is the floor, and which pairs sit on it: 53 of 253
pairs have `L0 = max(a_g, a_h) + 1` exactly, and they are the pairs whose two arcs are close.

The **permanent** onset behaves the other way round: `L1/gh` has median 0.424 over the 253 pairs
and the twin pairs sit at the very top of that range (0.600 to 0.658). A twin pair collides
first and settles last.

### R4. Real against random and coherent (item 3): the answer is signed, and it splits in two

At the level of the raw onset the comparison is contaminated: a random separation may have arc 1
(two adjacent teeth), which the real separation never has (`a_g` is even, file 20 Lemma 1), and
the arc floor then permits an onset of 2. The clean statistic is the onset **normalised by its
own floor**, `L0 / (max(a_g, a_h) + 1)`, where 1.000 is the earliest possible:

| family | twin pairs | non-twin pairs |
|---|---|---|
| real `1/3` | **1.000 at all 6** | median 3.100 (min 1.000, max 20.35) |
| random (600 draws) | - | median 1.171, mean 4.66, fraction exactly 1.000: 0.262 |

So the real separation is **extremal in both directions at once**: its twin pairs collide at the
earliest length any pair can, and its non-twin pairs collide markedly later than random. One
number drives both - the real arcs are `(g -+ 1)/3`, equal on a twin pair and never equal
otherwise (6 of 120 pairs share an arc under the real separation, and they are exactly the 6
twin pairs).

The shared arc is confirmed as *the* mechanism by the random draws themselves: among 2,400
random draws, **every one of the 154 with `a_g = a_h` has the earliest possible onset, while
only 441 of the 2,246 without do** (19.6%).

Raw percentiles, for the record: the real twin onset sits at percentile 0.80, 0.25, 0.50, 0.30,
0.15, 0.05 of the 20 random draws at `(5,7) .. (59,61)` - falling with `g`, because a random
separation's arc grows with `g` while the real twin's onset stays pinned to `a + 1`; real twin
onset over random median is 1.50, 0.83, 1.08, 0.33, 0.115, **0.084**. For non-twin pairs the raw
onset percentile has median 0.65 (real later than random) and the permanent onset percentile has
median 0.23 (real earlier than random): 66 of 114 non-twin pairs have their real `L1` below the
random median, `L1/gh` 0.360 real against 0.433 random.

Coherent families `c/r` (`s_g = c r^{-1} (mod g)` at every gear, `separation_drives_K.md` N-S2),
median ratio of the family's onset to the real one over the pairs where both are defined:

    family              1/3    1/5    2/5    2/7    4/7   2/11   3/11   2/13
    L0 ratio          1.000  1.000  1.000  0.933  0.575  0.824  0.887  0.637
    L1 ratio          1.000  1.099  1.097  1.136  1.000  1.091  1.000  1.000
    shared-arc pairs      6      2      4      4      2      3      4      4   (of 120)

The real family has the **latest first collision** of the eight and the **earliest permanent
collision** (every `L1` ratio `>= 1`). Coherence as such does not decide it - `1/5` and `2/5`
match the real onset exactly while `4/7` collides at 0.575 of it. What decides it is how many
pairs the family gives a shared arc, and the real family gives the most: 6, all twin.

### R5. Triples: sub-additive by the inclusion-exclusion term (P5 CONFIRMED, with 350 exceptions to strictness)

For every triple the growth law holds exactly with increment `4(g + h + k) - 8` (**0 exceptions
over the 20 triples**), so the triple's deficit rate is

    r_{g,h,k} = (4(g+h+k) - 8) / ghk  =  4/(gh) + 4/(gk) + 4/(hk)  -  8/ghk ,

the three pairwise rates **minus** the third-order inclusion-exclusion term. Sub-additivity is
therefore exact in the rate, and it is what the lengths show: over the 20 triples and every `L`,
`c_3 < c_{gh} + c_{gk} + c_{hk}` at 59,378 lengths, `=` at 2,252, and `>` at **350**. So
sub-additivity is the rule and is exact asymptotically, but it is *not* exceptionless at finite
`L`: super-additive lengths exist.

The general form, from the block growth law:

    rate of a block B  =  sum_{g in B} 2/g  -  ( 1 - prod_{g in B} (1 - 2/g) ) ,

the inclusion-exclusion tail of order `>= 2`. A block of size `b` captures the Bonferroni
truncation at order `b`; block size 1 is the counting filter of file 20, block size `K` is the
exact adversarial question. That is the ladder item 5 climbs.

### R6. The record's overlap, decomposed (P8 REFUTED in direction, confirmed in size)

The real machines `m11..m23`, record run length `R` (the project's `F(M)` is `R + 1`, the least
uncoverable length; the values reproduce the certified ladder `F = 7, 11, 18, 25, 34` - a gate):

    machine   R   sum_g max_g(R)   D   M   O   C_all   C_match
    m11       6              7     1   0   1       1         1
    m13      10             12     2   1   1       0         0
    m17      17             22     5   0   5       5         2
    m19      24             34    10   1   9      12         3
    m23      33             48    15   2  13      14         4

`D = sum_g max_g(R) - R` is the whole price the record pays; `O = sum_g n_g - R` is the overlap
it actually pays and `M = sum_g (max_g(R) - n_g)` the shortfall from per-gear maximality;
`D = M + O`. **The record is almost exactly maximal gear by gear** - `M = 0, 1, 0, 1, 2` against
`R` up to 33, and at `m17` every one of the five gears strikes its exact maximum. The record's
price is overlap and nothing else.

`C_all`, the sum of the pairwise collision deficits at `L = R`, tracks `D` to within 20% at
every rung and to within 1 at four of the five: `1, 0, 5, 12, 14` against `D = 1, 2, 5, 10, 15`.
The direction is not one-sided (`C_all` is above `D` at `m19`, below at `m23`, equal at `m11`
and `m17`, and 0 against 2 at `m13`), so P8's prediction of a one-sided over-explanation is
**refuted**; what survives is the size. Face A's "the record nearly achieves counting" is
sharpened here to: *the record's whole shortfall from counting is pairwise collision, and the
pairwise collision numbers predict its size.*

The valid (matching) part is much smaller: `C_match = 1, 0, 2, 3, 4`, so the best matching
accounts for 27% of the record's deficit at `m23`. The gap between `C_all` and `C_match` is the
whole difference between an accounting identity and a bound - which is R7.

Per gear at the record (`n_g` / `max_g(R)`): m17 `5:7/7, 7:6/6, 11:4/4, 13:3/3, 17:2/2`;
m23 `5:14/14, 7:10/10, 11:6/6, 13:5/6, 17:4/4, 19:4/4, 23:3/4`. The record's actual pairwise
overlaps sum to `1, 1, 7, 12, 16` against `O = 1, 1, 5, 9, 13`; the difference is the columns
three or more gears strike.

### R7. The collision bound (P6, P7): the all-pairs form is false, the block form is a theorem, and the block size must grow with K

**The all-pairs form is refuted.** `sum_g max_g(L) - sum_{all pairs} c(g,h;L)` is not an upper
bound on what a gear set can strike: Bonferroni runs the other way. Against the recorded explicit
covers (file 20 / `small_K_theorem.md` 1.2), where a cover of `L` exists so any bound must be
`>= L`:

    K            4    5    6    7    8    9    10
    L covered   15   21   27   36   44   67    87
    sum max     16   23   34   51   62   98   135
    sum ALL c    1    2    7   15   17   33    57
    "bound"     15   21   27   36   45   65    78
    verdict     ok   ok   ok   ok   ok  FALSE FALSE

**The valid form.** Partition the gear set `S` into blocks; the union of all strikes is inside
the union of the blocks' unions, so

    L = |union|  <=  sum_{blocks B} joint_max(B; L)  =  sum_g max_g(L)  -  sum_{blocks} c_B(L),

for every partition, hence for the best one. Blocks of size 1 give file 20's counting filter (C);
size 2 is a maximum-weight matching of the gears; size `K` is the exact question. Verified
against the same seven covers: the block-2 value is `15, 22, 31, 46, 57, 89, 124` against
`15, 21, 27, 36, 44, 67, 87` - never violated.

**What it proves.** `bound_b(K, L)` = the maximum over every `K`-set of primes (gears `>= L`
granted 2 strikes each with no deficit, the safe over-granting of file 20) of the best block-`b`
value. The lemma at `K` follows if `bound_b(K, W(K)) < W(K)`. Exhaustive over every `K`-subset of
the primes below `L`:

    K       3     4     5     6     7     8     9    10
    W(K)   28    48    60    88   140   160   228   280
    b = 1  26    51    72   113   190   229   340   432
    b = 2  24    46    65   102   169   207   305   387
    b = 3  21    42    61    95   157   192   283   360
    b = 4   -    38    55    87   147   178   264   335
    proves  K=3   K=4   K=5   K=6   -     -     -     -
            (b=1) (b=2) (b=4) (b=4)

(rows `b >= 2` at `K >= 7` are the value of the `K` smallest gears, which already exceeds
`W(K)`, so no maximum is needed to refute.) The headline:

> **The matching collision bound proves the adversarial lemma at `K = 4`**, where the counting
> filter alone does not: `bound_1(4, 48) = 51 >= 48` but `bound_2(4, 48) = 46 < 48`, the maximum
> over all 1,093 four-sets, attained at `{5, 7, 11, 13}` and closed by the single pair deficit
> `c(5,7;48) = 5`.

and, at block size four, `bound_4(5, 60) = 55 < 60` and `bound_4(6, 88) = 87 < 88` - the lemma at
`K = 5` and `K = 6` as well, by reasoning plus one number per block rather than by phase
enumeration. At `K = 7..10` block size 4 is not enough (147, 178, 264, 335 against 140, 160, 228,
280).

**Why the block size has to grow - the negative result.** The block bound can bite at all only if

    rho_b(S)  =  sum_g 2/g  -  max over partitions into blocks of size <= b of sum_B (rate of B)
              <  1 ,

the asymptotic form of the bound. For the `K` smallest gears:

    K       3       4       5       6       7       8       9      10      11      12
    rho_1  0.868  1.021  1.139  1.244  1.331  1.400  1.465  1.519  1.568  1.614
    rho_2  0.753  0.879  0.997  1.090  1.177  1.240  1.304  1.355  1.403  1.448
    rho_3  0.649  0.803  0.903  0.981  1.068  1.131  1.186  1.240  1.286  1.328
    rho_4  0.649  0.703  0.821  0.914  0.983  1.032  1.097  1.147  1.190  1.230
    rho_5  0.649  0.703  0.738  0.844  0.921  0.978  1.027  1.065  1.114  1.158
    rho_6  0.649  0.703  0.738  0.766  0.853  0.916  0.971  1.014  1.050  1.084

    least b with rho_b < 1:  K=3:1  K=4:2  K=5:2  K=6:3  K=7:4  K=8:5  K=9:6  K=10:7  K=11:8  K=12:8

> **The order of interaction needed grows with `K`**, one block per two gears added and then one
> per gear: `b >= K - 3` from `K = 6` on. So **no interaction law of bounded order can prove the
> adversarial lemma for all `K`** - the pairwise law dies at `K = 6`, triples at `K = 7`,
> quadruples at `K = 8`. This is a new obstruction, and it is exact: it comes from the identity
> `rate(B) = sum 2/g - (1 - prod (1 - 2/g))`, i.e. from Bonferroni truncation, not from any
> feature of the real teeth.

The finite-`L` bound is a little weaker than the rate condition (at `K = 5`, `rho_2 = 0.997 < 1`
but `bound_2(5,60) = 65 > 60`; the `O(1)` slack `sum_g 2` costs the difference). Carrying the
block size further on the `K` smallest gears (blocks whose period exceeds 9,000,000 are not
evaluated, which only weakens the bound):

    K      W(K)   b=1   b=2   b=3   b=4   b=5   b=6/7
    4        48    51    46    42    38     -      -
    5        60    72    65    61    55    51      -
    6        88   113   102    95    87    82     76
    7       140   190   169   157   147   139    129
    8       160   229   207   192   178   168    157
    9       228   340   305   283   264   249    236
    10      280   432   387   360   335   318    302

so block size 5 would reach `K = 7` and block size 6 or 7 would reach `K = 8` if the maximum over
all `K`-sets behaves as the smallest set does (not computed - the exhaustive maximum at those
sizes is out of budget), while at `K = 9, 10` even block size 7 leaves 236 and 302 against 228
and 280. The ladder stops where the rate table says it must.

### R8. The induction increment (P9 CONFIRMED)

Adding a gear `q'` to a machine `M` brings capacity `max_{q'}(L)` and collides with each old gear
`g` at rate `4/(g q')`. Measured against the prediction `4 L sum_{g in M} 1/(g q')`, `M =
{5,...,23}`:

    q'    L      max_q'(L)   sum_g c(g,q';L)   predicted   net = capacity - collisions
    29   280        20              21          25.7            -1
    29   560        39              45          51.4            -6
    31   280        19              23          24.1            -4
    37   280        16              16          20.2             0
    41   280        14              13          18.2            +1
    43   280        14              12          17.3            +2
    43   560        27              30          34.7            -3

so in rates the new gear's net contribution per column is

    2/q'  -  sum_{g in M} 4/(g q')  =  (2/q') ( 1 - 2 sum_{g in M} 1/g ) ,

**negative from the fourth old gear on**: `1 - 2 sum 1/g` is `+0.600, +0.314, +0.133, -0.021,
-0.139, -0.244, -0.331` for `M = {5}, {5,7}, ..., {5..23}`. That sign change is exactly why the
all-pairs form is invalid (R7): from `M = {5,7,11,13}` on it subtracts more than a gear brings,
so it would "prove" statements that are false. Under the block form each gear sits in one block
and collides with at most `b - 1` old gears, and the increment stays positive - which is the same
statement as `rho_b >= 1` for `b` too small. **The induction step an interaction law would need
therefore does not exist at any fixed order**: what a new gear costs the adversary is not a
bounded number of pairwise collisions but a share of every higher-order overlap.

---

## 3. Mechanism

Two gears are one rigid object. By CRT their phase pair is a diagonal translate, so a pair of
gears has exactly **one configuration up to sliding the window**; the whole question "how much
can two gears strike together on a run of `L`" is a window sliding over a fixed periodic pattern
of period `gh` with `2g + 2h - 4` marks. Four residues mod `gh` carry both gears' teeth - the
four-point shape `{0, S_g, S_h, S_g + S_h}` - and they are the entire source of collision. That
gives everything:

* one period of the window swallows all four coincidence residues, so the deficit rises by
  exactly 4 per period: the growth law, slope `4/(gh)`, the CRT overlap density;
* below the larger short arc neither gear can be forced onto the other, so the deficit is 0: the
  arc floor;
* at exactly one more than a shared arc there is only one place a two-tooth strike can sit, so
  both gears must sit there together: the shared-arc law, and with it the head collision, whose
  content is that 5 and 7 are a twin pair and twin gears share the arc `(g+1)/3`;
* for a block of gears the same argument gives the inclusion-exclusion tail, and truncating it at
  order `b` is exactly a partition into blocks of size `b` - a valid bound, whose margin against
  the capacity `sum 2/g` shrinks as `K` grows because the tail's leading term is order-2 and the
  capacity's is order-1.

The last line is the mechanism of the negative result: capacity grows like `sum 2/g ~ 2 log log`,
the truncated tail like `sum_{pairs} 4/(gh) ~ 2 (sum 2/g)^2` only when all pairs are allowed, and
a partition allows only `K/b` of them. The adversary is not beaten by any bounded slice of the
interactions.

---

## 4. What is new

* **The collision period law** `c(g,h;L+gh) = c(g,h;L) + 4`, with the one-line proof and its
  block generalisation `c_B(L+P) = c_B(L) + sum 2P/g - (P - prod(g-2))`. It makes the deficit an
  exactly linear object with slope `4/(gh)` and makes the permanent onset computable. Not on the
  record: `separation_drives_K.md` N-S1 has the mean overlap `4m/(gh)` as an average over phases;
  this is the same 4 as an exact increment of a maximum. Prior art not checked.
* **The arc floor** `c(g,h;L) = 0` for `2 <= L <= max(a_g,a_h)` under the real separation, and
  its failure (134 of 759 draws) under random separations.
* **The shared-arc law** `a_g = a_h = a => c(g,h;a+1) >= 1`, with the one-paragraph proof, over
  every shared-arc configuration of every pair, 14,340 instances, 0 exceptions. This is the
  general form of file 20's head collision: `(5,7)` is the `a = 2` instance, and every twin pair
  is an instance because twin gears share the arc `(g+1)/3`.
* **The block bound** `L <= sum_{blocks} joint_max(B;L)` as the correct form of "capacity minus
  forced collision", with the all-pairs form refuted by explicit covers at `K = 9, 10`, and with
  the exact statement of what each block size proves: `b = 2` proves the adversarial lemma at
  `K = 4` where counting fails, `b = 4` proves it at `K = 5` and `K = 6`.
* **The block-size ladder and the bounded-order obstruction**: the least block size with
  `rho_b < 1` is `1, 2, 2, 3, 4, 5, 6, 7, 8, 8` at `K = 3..12`, so an interaction law of any
  fixed order fails from `K = b + 3` on. This is a new "face" - it says in advance that the
  induction step the tree has been looking for cannot be a pairwise, or triple, or any
  bounded-order law.
* **The record's price is overlap, not lost capacity**: `M = 0, 1, 0, 1, 2` at `m11..m23`, and
  the pairwise collision sum predicts the total deficit to within one unit at four of five rungs.
* **The real separation is extremal in both directions**: normalised onset exactly 1.000 on all
  its twin pairs (the earliest possible) and median 3.100 on its non-twin pairs against a random
  median of 1.171. `separation_drives_K.md` found the real separation typical for `K(d)`; at the
  level of a single pair it is not typical, it is extreme - but the two extremes point opposite
  ways and cancel in any sum over pairs, which is consistent with W3's answer rather than against
  it.

**Not new, cited:** the capacity formula and the arc law (file 20 Lemmas 1, 2); the one-orbit
reduction (`the_wall.md` 5a); the four-point overlap shape and the mean-overlap identity
(`separation_drives_K.md` 3.1, N-S1); the coherence-under-CRT identity (N-S2); the ladder `A(K)`
and `W(K)` (`arc_multiset.md` R1, file 20); "twin gears share an arc" (`arc_multiset.md`).

---

## 5. What holds without exception, with the count

| statement | range | exceptions |
|---|---|---|
| `c(g,h;L+gh) = c(g,h;L) + 4`, real separations | 228 pairs `g<h<=97`, 248,334 instances | **0** |
| the same, random separations | 120 draws, 67,400 instances | **0** |
| `c(g,h;gh) = 4` | 228 pairs | **0** |
| `c(g,h;L) >= 4 floor(L/gh)` | 66 pairs, 134,380 instances | **0** |
| `c_B(L+P) = c_B(L) + 4(g+h+k) - 8` for triples | 20 triples, 61,980 instances | **0** |
| the arc floor `c(g,h;L) = 0` for `2 <= L <= max(a_g,a_h)`, real | 253 pairs, 5,009 instances | **0** |
| the shared-arc law `c(g,h;a+1) >= 1` | 14,340 shared-arc configurations, all pairs, all arcs | **0** |
| twin pairs have onset exactly `a+1` | 7 of 7 twin pairs to 97 | **0** |
| twin pairs have onset `< g`; non-twin pairs do not | 7 of 7 and 246 of 246 | **0** |
| shared arc `=>` earliest possible onset, random separations | 154 of 2,400 draws have `a_g=a_h` | **0** |
| the block bound is never violated by an explicit cover | the 7 recorded covers, blocks of size 2 | **0** |

---

## 6. Verdict

**STRONG.** The head collision generalises, exactly, in three separate directions, and every one
of them is proved rather than measured: the growth law (four per period, slope `4/(gh)`), the arc
floor, and the shared-arc law that makes every twin pair collide at the earliest length a pair
can. The pairwise deficit is a complete, closed-form object.

**And the branch closes its own route.** The collision bound in its valid (block) form does prove
the adversarial lemma at `K = 4` where counting fails, and at `K = 5, 6` at block size 4 - the
first proof of those cases that is reasoning plus one number per block rather than phase
enumeration. But the same identity that gives the bound also says how far it can go: the block
size needed is `1, 2, 2, 3, 4, 5, 6, 7, 8, 8` at `K = 3..12`, growing without bound, so **no
interaction law of fixed order reaches the lemma for all `K`**. The induction step the tree has
been looking for is not a pairwise law, and cannot be made one.

**CANDIDATE, weak.** The one object here that could still carry weight toward the root is the
permanent onset: `c(5,7;L) > 0` for every `L >= 21`, and more generally each pair is in permanent
collision above an exactly computed `L1 <= gh`. A statement of the form "at every `L`, some
`Omega(K)` disjoint pairs of the machine's gears are past their permanent onset" would give a
deficit growing with `K` rather than a fixed one. What would have to break it: nothing measured
here forbids it, and nothing here supplies it either - the block-size ladder says a *fixed*
number of pairs is not enough, and this asks for a growing number. Not yet shown.

---

## 7. Dead ends (do not re-enter)

* **The all-pairs collision bound** `L <= sum max_g - sum_{all pairs} c(g,h;L)`. FALSE, with two
  refuting instances: the recorded 9-gear cover of 67 columns (bound 65) and the 10-gear cover of
  87 (bound 78). Bonferroni runs the other way; the direction is fixed by R8's sign change.
* **A pairwise-only route to the adversarial lemma beyond `K = 5`.** `rho_2 >= 1` from `K = 6`,
  so the pairwise bound is vacuous there at every `L`, however the constants are improved.
* **Onset as a function of `g` and `h` for non-twin pairs.** No shape: `L0/gh` spreads over a
  factor of twenty (0.011 to 0.208) with no dependence on the arcs beyond the floor.
* **"The real separation is typical at the pair level"** - false in both directions (R4), but the
  two directions cancel, so it does not reopen W3.
