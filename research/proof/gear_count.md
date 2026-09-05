# Branch 5d.ii.i - COUNT GEARS, NOT COLUMNS

Parent: node 5d.ii (`research/proof/deletion_profile.md`), whose closing observation was that the
period record needs EVERY gear (minimum cover = the whole machine at m7..m23) while the window's
record needs about a fifth of them. The wall's thin place 1 (`research/proof/the_wall.md`
section 5c, `dead_branches_reopened.md` "Reading the file as a whole" item 1, 2d Idea 2, 5d.ii
Idea 1) says face A kills every argument that counts columns or strikes but says nothing about
counting GEARS, and names two proven gear-counting facts: the umbrella bound (a blocked stretch of
span `S` contains a tooth of every gear whose long arc is below `S + 2`) and L4's sole-striker
corollary (in an above-record stretch every gear is a sole striker somewhere, so every gear is
needed). The question is whether the number of gears a stretch of span `S` NEEDS grows with `S` by
a rule, and whether that rule inverted bounds `S` by the gear count `pi(q) - 3`.

Scripts in `research/anchor235/r48/`; result outputs (untracked) in
`research/anchor235/r48/results/`. Every number this document relies on is written here.

---

## 0. Pre-registered (written before the lattice computation of this branch)

### 0.1 Objects, stated exactly

Machine `M = {5..q}`, gears the primes `5 <= g <= q`. `u_g = 6^{-1} mod g`; gear `g` strikes
column `k` iff `k = +-u_g (mod g)`. The two teeth are separated by `d_g = 2 u_g = 3^{-1} (mod g)`;
the **short arc** is `a_g = min(d_g, g - d_g)` (so `3 a_g = g -+ 1`) and the **long arc** is
`g - a_g`. `n(M)` = the number of gears of `M` = `pi(q) - 2`. The brief's `pi(q) - 3` counts the
gears ABOVE 7; both counts are given wherever it matters.

* A **stretch of span `S`** is a set of `S - 1` consecutive columns, all struck. A maximal gap of
  span `S` is such a stretch whose two bounding columns are open; the endpoints of a maximal
  blocked run are open, so the two notions agree in every minimum taken below.
* A **cover** of a stretch is a set of gears whose strikes include every column of it.
  `cov(stretch)` = the size of a minimum cover; `hold(stretch)` = the number of gears striking at
  least one of its columns (`cov <= hold`).
* **`h_M(S)`** = the minimum, over every stretch of span `S` in the period of `M`, of
  `cov(stretch)`. Equivalently the least number of gears of `M` that can block `S - 1`
  consecutive columns.
* **`f_M(S)`** = `#{g in M : g - a_g < S + 2}`, the FORCED strikers of a span-`S` stretch
  (umbrella bound, proved, `flank_walk.md` F4/F13).
* **`S_max^M(K)`** = the longest span any `K`-gear sub-machine of `M` can block =
  `max{ F(A) : A a subset of M's gears, |A| = K }`, where `F(A)` is the record of the gear set `A`
  taken over ITS OWN full period.
* `W(q) = (q'^2 - 1)/6` is the window's top column, `q'` the next prime after `q`.
* `F_W(q)` = the longest blocked stretch lying inside the window at rung `q` (5d.ii's object).

**The inversion identity (stated, not predicted).** `h_M` and `S_max^M` are exact inverses:
`h_M(S) = min{K : S_max^M(K) >= S}`, because a span-`S` stretch covered by `K` gears is exactly a
`K`-gear sub-machine blocking `S - 1` consecutive columns.

**The tool.** `F(A)` is computed with no period scan: over the full period of `A` every
combination of phases occurs exactly once (CRT), so `A` blocks `L` consecutive columns iff one
translate of each gear's tooth pair can cover `{0..L-1}`; `F(A)` is the least uncoverable `L`.
Gate: the tool must reproduce the recorded ladder `F = 5, 7, 11, 18, 25, 34, 43, 58` at
`{5,7} .. {5..31}`.

### 0.2 The theory

Face A forbids counting columns and strikes. The two gear counts it leaves open are the FORCED
count `f` (proved) and the NEEDED count `h` (a set-cover quantity). The theory is that they are
not the same quantity and do not saturate at the same place: `f` is a statement about the arcs
alone and must reach the whole machine as soon as the span passes the top gear's long arc, i.e. at
span about `2q/3`, whereas `h` keeps growing to the record `F ~ q^2/24`. If that is so, the proven
gear count is spent a factor of order `q` below the window, and the only gear count that reaches
the window is `h`, which is the F ladder inverted, i.e. the root itself. The branch is therefore
designed to decide, with numbers, WHICH of the two it is; and to test the one part of the
inversion that is not a restatement, namely whether an ADVERSARIAL `K`-gear sub-machine (any `K`
gears of `M`, not the initial segment) can block more than `F({5..p_K})`.

### 0.3 Predictions, with numbers, and what refutes each

**From the brief.**

* **G1 (the brief's shape).** `h_M(S)` grows at least like `f_M(S)` and faster, at every machine.
  REFUTED by any span at which `f > h`.
* **G2 (the whole machine holds the record).** `h_M(F(M)) = n(M)` at m11..m31: the minimum cover
  of the record is the whole machine. REFUTED by one machine whose record span is blocked by a
  proper subset of its gears.
* **G3 (the margin one below the record).** `h_M(F - 1) <= n(M) - 1` at every machine: a stretch
  one shorter than the record can spare one gear; the branch is to say which gear. REFUTED if
  `h_M(F - 1) = n(M)` at any machine.
* **G4 (the inversion is quadratic).** `S_max^M(K) <= c K^2` with a constant `c` not depending on
  the machine. Pre-registered `c` from the record law `F ~ q^2/24` with `pi(q) ~ q/ln q`: the
  initial-segment value is `F({5..p_K}) ~ p_K^2/24 ~ (K ln K)^2/24`, which is NOT `O(K^2)`, so G4
  as literally stated is expected to FAIL by the `(ln K)^2`; the measurable form is the ratio
  `S_max(K)/(K ln K)^2` and whether it settles near `1/24`.
* **G5 (the window is a sub-machine record).** `F_W(q) <= S_max(pi(sqrt(6k+1)) - 3)` at the window
  record's own column `k`, at every rung 23..997. Pre-registered as HOLDING but VACUOUS: the
  effective machine at column `k` has about `pi(q)` gears whose own record is of order `q^2/24`,
  far above `F_W ~ (log)^2`; the informative quantity is instead `h(F_W(q))`, the number of gears
  a span-`F_W` stretch NEEDS, against the number available.
* **G6 (the gear-count inequality).** `h(W(q)) > n(M)`: a stretch of span `W` needs more gears than
  the machine has. Pre-registered TRUE (it is `F < W`) with margin, in gear count, of a factor
  `pi(sqrt(24 W))/pi(q) -> pi(2q)/pi(q) ~ 2`.

**The branch's own.**

* **G7 (initial segments are optimal).** `S_max^M(K) = F({5..p_K})` for every `K` and every
  machine to m31: the best `K`-gear sub-machine is the `K` SMALLEST gears. If G7 holds, the
  inversion is the F ladder read backwards and item 3 of the brief is a restatement, to be said in
  one line. REFUTED by one `K`-subset containing a gap (a skipped prime) that beats the initial
  segment. Prior evidence that it will hold: at m23 the deletion lattice gives
  `S_max(6) = 34 - 9 = 25 = F({5..19})` and `S_max(5) = 34 - 16 = 18 = F({5..17})`
  (`deletion_profile.md` R1); untested at m29, m31 and at small `K`.
* **G8 (the umbrella saturates at `2q/3`).** `f_M(S) = n(M)` for every `S >= S_sat`, where
  `S_sat = (q - a_q) - 1` is one below the top gear's long arc, i.e. `S_sat ~ 2q/3`; and
  `S_sat / F(M) -> 0` like `48/q` while `S_sat / W(q) -> 0` like `4/q`. Concretely: `S_sat = 20`
  at m31 against `F = 58` and `W = 228`; `S_sat = 664` at `q = 997` against `W = 169,680`, a
  factor of 255. REFUTED if `f` fails to reach `n(M)` at that span, or if the ratio does not fall.
* **G9 (forced is not needed).** `f_M(S) > h_M(S)` for a range of spans at every machine from m17
  up, so the umbrella's forced set is NOT a lower bound for the cover count and cannot be used as
  one. Predicted crossover near `S = 11` (where `1.5 S` passes `sqrt(24 S)`). REFUTED if
  `f <= h` at every span at every machine.
* **G10 (the window stretch is redundantly covered).** The actual window record stretch's minimum
  cover (5d.ii's numbers: 6, 10, 14, 20, 22, 27, 34, 32 gears at the eight largest distinct
  stretches) exceeds the free minimum `h(F_W)` for a stretch of the same span, by a factor above
  1.3 at the large rungs. REFUTED if the two agree to within one gear.

### 0.4 Scorecard

| # | prediction | verdict | evidence |
|---|---|---|---|
| G1 | `h` grows at least like `f` | **REFUTED** | `f > h` at every machine from m17; gap 4 at m31 (`f = 9`, `h = 5` at `S = 20`) |
| G2 | `h(F) = n(M)` | **CONFIRMED** at m7..m31 | `S_max(n-1) < F` at all eight machines (R2) |
| G3 | one gear sparable at `F - 1` | **REFUTED** at every machine | the first sparable span is `F - 2, 4, 2, 4, 9, 8, 13` below the record (R2) |
| G4 | `S_max(K) <= c K^2` | **as pre-registered, fails the literal form** | measured `A(K)/(K ln K)^2` = 0.52, 0.34, 0.24 at K = 4, 5, 6, still falling, against `1/24 = 0.042` (R3) |
| G5 | window record is a sub-machine record | **CONFIRMED and VACUOUS** as pre-registered; the informative form is R5 |
| G6 | `h(W) > n(M)` | **CONFIRMED** where `A` is exact (K = 4, 5, 6: `A = 16, 22, 28` against `W = 48, 60, 88`), margin factor 3.0, 2.7, 3.1 in span (R6) |
| G7 | initial segments optimal | **REFUTED from K = 4** | `{5,7,11,17}` blocks span 16, `{5,7,11,13}` only 11 (R3) |
| G8 | umbrella saturates at `2q/3` | **CONFIRMED exactly** | `S_sat = 4, 6, 8, 10, 12, 14, 18, 20` at m7..m31, `W/S_sat` rising 5.0 -> 11.4 -> 255 at q = 997 (R1) |
| G9 | forced is not needed | **CONFIRMED**, crossover at `S = 10` at every machine from m17 | (R1) |
| G10 | window cover is redundant | **CONFIRMED**, factor 1.0 -> 2.1 rising with the rung | (R5) |

---

## 1. Setup (exact ranges)

Five scripts in `research/anchor235/r48/`, numpy-free, `uv run python <script>` from the
repository root.

| script | what it computes | range | cost |
|---|---|---|---|
| `cover_core.py` | `F(A)` for any gear set by phase covering, exhaustive | gate: the ladder to m31 | 97 s at m31 |
| `lattice.py` | `F(A)` for ALL 511 non-empty subsets of `{5,7,...,31}`; `S_max^M(K)`, `h_M`, `f_M` per machine | m7..m31, every subset | 104 s, 4 cores |
| `adversary.py`, `adversary2.py` | `A(K)`, the free adversarial ladder, exhaustive over the primes 5..101 and 5..149 | `K <= 6` exact | 56 s |
| `adversary3.py` | certified lower bounds `A(K) >= L` by exhibited covers, hill climb | `K = 7..12` | 20 min |
| `win.py` | `F_W(q)`, the exact minimum cover of every distinct window record stretch, `hold`, `f` | every prime rung 7..997, columns to 169,680 | 4 min |
| `mech.py` | the arc table, the one-gear sweep, witness covers with their waste | gears to 199 | 30 s |
| `summary.py` | the tables quoted below | - | 1 s |

**Gates passed.** (i) `cover_core.py` reproduces the recorded F ladder exactly:
`F = 2, 5, 7, 11, 18, 25, 34, 43, 58` at `{5} .. {5..31}`. (ii) `win.py` reproduces node
5d.ii's exact minimum covers of the window record stretches: `2, 6, 10, 14, 20, 22, 27, 34, 32`.
(iii) `S_max^M(n-1) = F(M) - min_g drop(g)` agrees with 5d.ii's deletion profile at m23
(`25 = 34 - 9`).

---

## 2. Results

### R1. The two gear counts, and where each stops growing

`h_M(S)` (needed) and `f_M(S)` (forced by the umbrella), at the spans where either changes.
Machine `{5..31}`, `n = 9` gears, `F = 58`:

    S     2   3   4   6   8  10  12  14  17  18  20  23  28  38  46  58
    h     1   2   2   3   4   4   4   4   5   5   5   6   7   8   9   9
    f     1   1   2   3   4   5   6   7   7   8   9   9   9   9   9   9

Machine `{5..23}`, `n = 7`, `F = 34`:

    S     2   3   4   6   8  10  12  14  17  22  26  34
    h     1   2   2   3   4   4   4   4   5   6   7   7
    f     1   1   2   3   4   5   6   7   7   7   7   7

**The umbrella count saturates and stops.** `f_M(S) = n(M)` for every span at or above
`S_sat = (q - a_q) - 1`, one below the top gear's long arc, because past that every gear of the
machine has its long arc below `S + 2`. Exactly:

    q            7    11    13    17    19    23    29    31        997
    n            2     3     4     5     6     7     8     9        166
    F            5     7    11    18    25    34    43    58          -
    S_sat        4     6     8    10    12    14    18    20        664
    F / S_sat  1.25  1.17  1.38  1.80  2.08  2.43  2.39  2.90         -
    W / S_sat  5.00  4.67  6.00  6.00  7.33 10.00  8.89 11.40     255.5

`S_sat / q -> 2/3` exactly (the long arc of `q` is `q - (q -+ 1)/3`), so the proven gear count
reaches the whole machine at span `2q/3` while the record is at `q^2/24` and the window at
`q^2/6`. **The umbrella's count is spent a factor `q/4` below the window**, and the factor grows
without bound: 11.4 at m31, 255 at rung 997.

**Forced is not needed.** From m17 up, `f > h` from span 10 onward: at m31, span 20 is blocked by
5 gears (`h = 5`) while all 9 are forced to strike it. The maximum excess `f - h` is
`0, 0, 1, 2, 3, 3, 4` at m11..m31 and grows. So the forced set is not contained in a minimum
cover and **the umbrella bound cannot be used as a lower bound on the cover count**: the two gear
counts point in opposite directions.

### R2. The record needs the whole machine, and the margin is never one

`S_max^M(K)`, the longest span the best `K`-gear sub-machine of `M` can block, exact for every
subset:

    machine     K=1  2  3   4   5   6   7   8   9    F   n
    {5..7}        2  5                                5   2
    {5..11}       2  5  7                             7   3
    {5..13}       2  5  7  11                        11   4
    {5..17}       2  5  7  16  18                    18   5
    {5..19}       2  5  7  16  21  25                25   6
    {5..23}       2  5  7  16  21  25  34            34   7
    {5..29}       2  5  7  16  22  26  35  43        43   8
    {5..31}       2  5  7  16  22  27  37  45  58    58   9

`S_max^M(n) = F(M)` by definition and `S_max^M(n-1) < F(M)` at all eight machines, so
`h_M(F) = n(M)`: **the record of the period needs every gear** (G2). The mechanism is L4's
sole-striker corollary, not a new fact; what is new is the size of the margin.

`F - S_max^M(n-1)`, the amount by which a stretch must fall below the record before one gear can
be spared, is `3, 2, 4, 2, 4, 9, 8, 13` at m7..m31 - never 1, and growing. (It equals
`min_g drop(g)` of 5d.ii's deletion profile, and reproduces its m23 value 9.) G3 refuted: a
stretch one shorter than the record spares no gear at any machine; at m31 you must come 13
columns down before any of the nine gears becomes dispensable, and the gear that then goes is
gear 23 (`{5,7,11,13,17,19,29,31}` blocks 45).

### R3. The inversion is NOT the F ladder read backwards - the central result

The brief asked whether `S_max(K)` is just the record ladder inverted, in which case the branch
would be a restatement of the root. **It is not.** The best `K`-gear sub-machine is not the `K`
smallest gears, from `K = 4` onward:

    K     best K gears (free choice)   A(K)    {5..p_K}       F ladder   A / F
    1     {5}                             2    {5}                   2   1.00
    2     {5,7}                           5    {5,7}                 5   1.00
    3     {5,7,11} (22 sets tie)          7    {5,7,11}              7   1.00
    4     {5,7,11,17}, {5,7,11,19}       16    {5,7,11,13}          11   1.45
    5     {5,7,11,23,29}, {5,7,11,23,31} 22    {5,7,11,13,17}       18   1.22
    6     {5,7,11,17,23,37}              28    {5,7,11,13,17,19}    25   1.12
    7     >= {5,7,11,13,17,19,31}        >=37  {5..23}              34  >=1.09
    8     >= {5,7,11,13,17,19,29,31}     >=45  {5..29}              43  >=1.05

`A(K)` is EXHAUSTIVE and exact for `K <= 6` over the pool of all primes 5..149 (1.1 million
6-subsets at `K = 6`); the `K >= 7` rows are certified lower bounds, taken from the exact
`{5..31}` lattice, each proved by an exhibited cover. Extending the pool from 31 to 101 to 149
does not raise `A(4)`, `A(5)` or `A(6)`, so the optimum is a small-gear object and the exact
values are stable. A node-capped hill climb over the primes 5..71 independently reached 37 at
`K = 7` with a DIFFERENT set, `{5,7,11,13,17,31,37}`, and found nothing better at `K = 8, 9`
inside its budget (43, 57 against the lattice's 45, 58), so the `K >= 7` rows should be read as
lower bounds and nothing more.

Two consequences.

1. **The dead-ends list's proposed identity is false.** `K_columns(d)` = "the number of gears of
   the smallest machine with `F >= d`" (`dead_branches_reopened.md`, fixed-depth counting, Idea 2)
   fails at `d = 16`: the identity says 5 gears (`{5..17}`, `F = 18`), the truth is 4
   (`{5,7,11,17}`).
2. **The wall's section 5a needs a correction.** It records that "the adversary with one phase per
   gear over all primes up to `q` is exactly the real machine over its period, so
   `K_columns(W(q)) > pi(q) - 3` IS `F(y) < y^2/6`". That is true only for the FIXED gear set
   `{5..q}`. `K_columns` quantifies over gear SETS as well as phases, and the two differ: the
   covering statement is STRICTLY STRONGER than the root, by a factor that is 1.45 at `K = 4` and
   measured falling (1.22, 1.12, >=1.09, >=1.05).

The growth. `A(K)/(K ln K)^2` = 0.52, 0.34, 0.24 at `K` = 4, 5, 6 against the record law's
`1/24 = 0.042`; the sequence is still falling and the pre-registered quadratic form `A(K) <= cK^2`
has no constant that holds at these `K` (`A/K^2` = 1.00, 0.88, 0.78, still falling). Nothing here
decides the asymptotic constant; what it decides is the SIGN of the gap between the two ladders,
which is positive.

### R4. The mechanism: a gear enters a cover by its ARC, not by its size

Why is `{5,7,11,17}` better than `{5,7,11,13}`? The one-gear sweep answers it exactly. With
`{5,7,11}` fixed, `F({5,7,11,g})` over every prime `g` from 13 to 103:

    g          13  17  19  23  29  31  37  41  43  47 ... 101 103
    a_g         4   6   6   8  10  10  12  14  14  16 ...  34  34
    F           11  16  16  11  11  11  11  11  11  11 ...  11  11

The fourth gear is worth 5 extra columns if and only if its short arc is 6, and its size is
irrelevant: 17 and 19 give the same value, and so do 29 and 31, and 41 and 43. Likewise
`F({5,7,11,17,g})` is 18 for every `g` except `g = 19` (arc 6, gives 21) and `g = 47` (arc 16,
gives 20).

The reason is elementary and exact, and it is the complement of the umbrella bound.

> **The dichotomy (proved here, elementary).** Let a blocked run have `L` columns, span
> `S = L + 1`. Every gear is in exactly one of two states.
> (i) `g - a_g < S + 2`: the gear MUST strike the run (the umbrella bound, on record).
> (ii) `g - a_g >= S + 2`: then `g > g - a_g > L`, so each of the gear's two residue classes
> meets the run at most once, and the long arc cannot fit inside it; the gear contributes at
> most a PAIR of columns at distance exactly `a_g`, or a single column. It is a bare domino of
> span `a_g`, and its size is invisible.
>
> Consequently two gears in state (ii) with the same short arc are interchangeable in any cover.

And

> **two gears have the same short arc iff they are a twin prime pair.**
> `a_g = (g -+ 1)/3`, so `a_g = a_h = a` with `g < h` forces `{g, h} = {3a-1, 3a+1}`.

So the arc map on the gears of `{5..q}` is injective except on twin pairs, and the number of
distinct arcs is `n(M) - pi_2(q)`:

    q             23   31   47   61  101  199
    gears          7    9   13   16   24   44
    distinct arcs  4    5    8   10   17   30
    twin pairs     3    4    5    6    7   14

The initial-segment machine is forced to spend two gears on one arc at every twin pair it
contains; the free adversary is not. That is the whole of the `K = 4` gap: `{5,7,11,13}` carries
the arcs `2,2,4,4` and `{5,7,11,17}` the arcs `2,2,4,6`. Read the witness cover of the 15-column
run and the gap is one number: gears 5, 7 and 11 at their best phases leave two holes 6 apart
(columns 4 and 10), so the fourth gear must supply **a domino of span exactly 6**. Gear 17 has
arcs `(6, 11)` and supplies it; gear 13 has arcs `(4, 9)` and cannot; and no gear anywhere with
`a_g != 6` can, whatever its size. The fourth gear is chosen by an arc the lower gears' holes ask
for, and the primes offer that arc only at `3a -+ 1`.

The covers themselves show it as waste. For the optimal cover of the longest run each set can
block (`mech.py` part C, strikes counted inside the run):

    gear set                 arcs        L    strikes  waste  sum 2/g
    {5,7,11,13}              2,2,4,4    10      11       1     1.021
    {5,7,11,17}              2,2,4,6    15      16       1     0.985
    {5,7,11,13,17}           2,2,4,4,6  17      22       5     1.139
    {5,7,11,23,29}           2,2,4,8,10 21      23       2     1.023
    {5,7,11,13,17,19}        2,2,4,4,6,6 24     32       8     1.244
    {5,7,11,17,23,37}        2,2,4,6,8,12 27    32       5     1.126

The winners are the near-perfect tilings (waste 1 and 2) and they have LESS counting capacity
than the initial segments they beat (`sum 2/g` = 0.985 against 1.021 at `K = 4`, 1.023 against
1.139 at `K = 5`). Capacity is not what decides; the arcs are. This is a which-residues property,
which face A permits, and it is the same object the wall's section 4 identified as what makes
covering harder than counting - the fixed separation - now measured as a CHOICE the adversary
makes and the real machine cannot.

A caution, recorded because it refutes the obvious generalisation: "all arcs distinct" is not the
rule. At `K = 7` and above inside `{5..31}` the best sets do contain twin pairs
(`{5,7,11,13,17,19,31}`), because once `g < L` a gear is no longer a bare domino - its
periodicity matters and two gears of the same arc are no longer the same object. The rule holds
where the gears exceed the run.

### R5. The window: the real stretch is covered about twice over

Every prime rung 7..997 (166 rungs, columns to 169,680; the openings there are the twin-prime
columns). There are 13 distinct window record stretches. For each, at the first rung holding it:
`hold` gears strike it, `cov` is its exact minimum cover, `f` is the umbrella's forced count, and
`h_A` is the free minimum for a stretch of that span: exact where the `A` ladder is exact
(`K <= 6`), and above that an upper bound, since `A(K) >= F({5..p_K})` makes the certified F
ladder an upper bound for `h_A`. The last row's `h_A` is an estimate: the F ladder stops at 161
and 242 is past it.

     x        F_W  first q  gears  hold  cov   f   h_A          cov/h_A   gate z
        12      5      7       2     2    2    2   2 (exact)      1.00       10
        52      6     17       5     5    3    2   3 (exact)      1.00       18
        58     12     19       6     6    6    5   4 (exact)      1.50       20
       110     25     23       7     7    6    7   6 (exact)      1.00       28
       397     28     47      13    13   10   11   6 (exact)      1.67       50
       980     35     73      19    19   11   14   7 (exact)      1.57       78
      2233     47    113      28    28   14   18   <= 9         >= 1.56     116
      3090     62    137      31    30   17   22   <= 10        >= 1.70     137
      4070     83    157      35    35   20   28   <= 10        >= 2.00     157
     10383    105    241      51    47   22   35   <= 13        >= 1.69     250
     31318    154    433      82    77   27   48   <= 15        >= 1.80     434
    114742    168    829     143   117   34   52   <= 16        >= 2.12     830
    141725    242    919     155   133   32   70   ~ 19 (est)    ~ 1.68     922

Three readings.

* **`cov` is far below `hold` and below `f`.** At rung 919, where the 242-column stretch first
  becomes the window record, it is struck by 133 of the machine's 155 gears, 70 of them forced by
  the umbrella, and 32 suffice. The umbrella is informative in the window (it forces 70 gears to
  strike) and still useless as a lower bound on the cover.
* **The real phases cost a factor of about two.** `cov/h_A` rises from 1.0 at the small stretches
  to 1.7-2.1 at the large ones. The gears the window's stretch uses are available to the free
  adversary too; the only difference is that the window's gears sit at the phases the primes give
  them. So the measured redundancy is the price of fixed phases, and it is the gear-count image of
  the span-side factor `F/W = 0.25` (a factor 2 in gears is a factor 4 in span).
* **G5 as literally stated holds and says nothing.** The effective machine at the record stretch's
  own column `k` is `{5..z}` with `z = sqrt(6k+1)` (the square gate, node 7d): `z = 922` at rung
  997, a machine whose own record is of order `z^2/24 ~ 35,000` against `F_W = 242`. The window's
  stretch is a sub-machine record of a machine 145 times too big; the honest statement is the
  `h_A` column, which says the window's longest stretch is what a machine of about 19 gears could
  block, sitting inside a machine of 166.

### R6. The gear-count inequality, and what it would take

The needed statement, in the branch's language: **a stretch of span `W(q)` needs more than `n(q)`
gears**, `h(W(q)) > n(q)`, equivalently `A(n(q)) < W(q)`. Where `A` is exact:

    K = n(q)   q     q'    W(q)     A(K)   A/W    F({5..q})  F/W
       4      13     17      48      16    0.333      11     0.229
       5      17     19      60      22    0.367      18     0.300
       6      19     23      88      28    0.318      25     0.284

So the covering form is TRUE at the machines where it can be decided, with a margin of a factor
3.0, 2.7, 3.1 in span - against the real machine's factor 4.4, 3.3, 3.5. The adversary eats about
a quarter of the slack and the remainder is still large. In gear count the margin is: to reach
`W = 48` an adversary needs `K` with `A(K) >= 48`, and `A(9) >= 58` shows `K <= 9` against the 4
gears the machine has, so the machine is short by a factor of at most 2.25 and (since
`A(6) = 28 < 48` exactly) at least 1.75 - the pre-registered factor of about 2.

**The smallest lemma that would close it.** `A(K) < W(p_K')`, i.e.

    A(K) < (p_{K+1}^2 - 1)/6   for every K,

with `p_K` the `K`-th gear (`p_1 = 5`). The parts and their status:

* PROVEN: the umbrella bound (every gear with long arc below `S + 2` strikes a span-`S` stretch)
  - and R1 shows it is spent at `S = 2q/3`, a factor `q/4` below `W`, so it cannot supply the
  lemma at any `q`;
* PROVEN: L4's sole-striker corollary - it gives `h(F) = n` (R2), which is the boundary case of
  the lemma and no more;
* PROVEN and ELEMENTARY (this branch): the arc law `a_g = (g -+ 1)/3`, hence the injectivity of
  the arc map off the twin pairs, hence "a gear above the run is a domino of span `a_g`";
* MEASURED: `A(K)` exactly at `K <= 6`, lower bounds above; the ratio `A/F ladder` = 1.45, 1.22,
  1.12, and falling;
* MISSING, and it is the whole lemma: an upper bound on `A(K)` of the form `A(K) = O((K ln K)^2)`
  with a constant below `1/6`. No proven ingredient on the tree bounds `A(K)` above at all. The
  counting bound `sum 2/g >= 1` is passed by 10 gears (wall section 4) and the umbrella saturates
  at `2q/3`; both are exhausted long before `W`.

### R7. What the dichotomy gives as an inequality, and why it is not enough

The dichotomy of R4 splits the machine at every span into `f(S)` forced strikers and `n - f(S)`
bare dominoes, and each domino contributes at most two columns. So, PROVEN:

    S - 1  <=  sum_{g : g - a_g < S+2} 2 ceil(S/g)  +  2 (n - f(S)).

This is a genuine gear-counting inequality: the second term counts GEARS, not strikes. It is also
not binding anywhere measured. At the m31 record (`S = 58`, `f = 9`, `n - f = 0`) the right side
is 94 against 57. At rung 919 on the window's record stretch (`S = 242`, `f = 70`, `n - f = 85`)
it is about 490 against 241, and at `S = W = 169,680` the first term alone is about twice `S`.
The reason is that the forced term is the ordinary counting bound and it already has a factor of
two in hand (`sum 2/g > 1` from ten gears, wall section 4). The dichotomy therefore adds a gear
count to a bound that a strike count already loses, which is face A's ruling reached by a new
road.

### R8. Exceptionless, with the count

Every statement below was checked on the whole of its stated range, with no exception.

| statement | range | count | status |
|---|---|---|---|
| `f_M(S) = n(M)` for every `S >= (q - a_q) - 1`, and `f_M(S) < n(M)` below it | m7..m31, every span `2..F` | 8 machines, 201 spans | exact, and a one-line consequence of the arc law |
| `h_M(F(M)) = n(M)`: the record needs every gear | m7..m31 | 8 machines, all 1,013 non-empty subsets | exact (lattice) |
| `h_M(S) <= f_M(S)` fails, i.e. `f > h` somewhere, at every machine from m17 | m17..m31 | 5 machines; first at `S = 10` at each | exact |
| `S_max^M(K) >= F({5..p_K})` with equality only at `K <= 3` and `K = n` | m17..m31 | 5 machines, `K = 1..n` | exact |
| `A(K)` unchanged when the pool grows 31 -> 101 -> 149 | `K = 4, 5, 6` | 3 values, 1.4 million subsets | exact |
| a gear with `g - a_g >= S + 2` strikes at most two columns of the run, at distance exactly `a_g` | all | proof, not a check | proved (R4) |
| `cov(window record stretch) >= h_A(F_W)` | 13 distinct stretches, rungs 7..997 | 13 | exact where `h_A` is exact (6 of 13), a bound elsewhere |
| `S_max^M(n-1) = F(M) - min_g drop(g)` (the lattice agrees with 5d.ii's deletion profile) | m7..m31 | 8 | exact, cross-check |


---

## 3. Mechanism, in one paragraph

A blocked run of `L` columns is covered by two kinds of gear and the kinds are decided by one
number, the gear's long arc against the run. Gears with long arc below `S + 2` cannot miss the
run (umbrella); gears above it can only lay a single domino whose span is the gear's short arc
`a_g = (g -+ 1)/3`, their size being invisible inside the run. Since `a_g = a_h` forces
`{g, h} = {3a-1, 3a+1}`, the arcs available to an adversary are indexed by `a` with `3a -+ 1`
prime, and each arc is offered once except at twin pairs, where it is offered twice. Covering a
run is therefore choosing a multiset of arcs that fits the holes the smaller gears leave: at
`K = 4` the holes left by 5, 7 and 11 are 6 apart, so the fourth gear must have arc 6, which 17
and 19 have and 13 does not - and that single fact is the whole 11 -> 16 jump. The machine
`{5..q}` is one particular multiset of arcs, and not the best one: it is forced to buy both
members of every twin pair, i.e. two copies of one arc. That is why the best `K`-gear sub-machine
is not the `K` smallest gears, and why the covering form of the root is strictly stronger than
the root. Against that, the umbrella count - the only gear count already proved - is a spent
force: it reaches the whole machine at span `2q/3` and cannot say anything about a span of
`q^2/6`, a factor `q/4` away.

---

## 4. What is new (no prior art located)

1. **The adversarial sub-machine ladder `A(K)` and the refutation of the initial-segment
   identity.** `A(4) = 16 > 11 = F({5,7,11,13})`, exhaustive over all 4-subsets of the primes to
   149; likewise `A(5) = 22 > 18` and `A(6) = 28 > 25`. The dead-ends list's proposed identity
   `K_columns(d) =` "gears of the smallest machine with `F >= d`" is false, and the wall's section
   5a claim that the covering form IS the root is false as stated: the covering form quantifies
   over gear SETS as well as phases and is strictly stronger, by a measured factor 1.45, 1.22,
   1.12 at `K = 4, 5, 6`.
2. **The dichotomy and the domino law.** Every gear at a given span is either an umbrella gear
   (must strike) or a bare domino of span `a_g` with its size invisible; proved, elementary, and
   the exact complement of the umbrella bound. The arc-collision corollary - two gears share a
   short arc iff they are a twin prime pair - is one line from the recorded kill-spacing law
   `3 a_g = g -+ 1` (`docs/proofs/05`, `alignment-rules` 1.2) and is claimed new only in its USE:
   it is the adversary's currency, and it makes the machine `{5..q}` carry only
   `n(q) - pi_2(q)` distinct arcs (4, 5, 8, 10, 17, 30 at `q` = 23, 31, 47, 61, 101, 199).
3. **The umbrella count saturates at `2q/3`.** `f_M(S) = n(M)` for every `S >= (q - a_q) - 1`;
   `W/S_sat` = 5.0, 4.7, 6.0, 6.0, 7.3, 10.0, 8.9, 11.4 at m7..m31 and 255 at rung 997. The one
   proven gear count is exhausted a factor `q/4` below the window.
4. **Forced is not needed.** `f > h` from span 10 at every machine from m17 (`f - h` reaching 4 at
   m31), so the umbrella's forced set is not contained in a minimum cover and cannot lower-bound
   one. The two proven gear-counting facts point opposite ways.
5. **The record's sparing margin.** `F - S_max(n-1) = 3, 2, 4, 2, 4, 9, 8, 13` at m7..m31: a
   stretch must fall 13 columns below the m31 record before any of the nine gears can be spared,
   and the gear that then goes is 23, not the top gear.
6. **The window's record is covered about twice over.** Its exact minimum cover is 1.5 to 2.1
   times the free minimum for a stretch of the same span; the gears are the same, only the phases
   are fixed, so the factor is the price of the real phases and is the gear-count image of
   `F_W/W`.

Named once, not re-derived: the umbrella bound and the kill-spacing law (`flank_walk.md`, F4/F13);
L4's sole-striker corollary (`pair_statement.md`); the square gate and the nested-decreasing
holder law (`deletion_profile.md`, node 7d); `F(A)` by phase covering is the project's phase
reduction (`docs/proofs/09`) and the existing `research/max_gap_search.py`, re-implemented in the
column frame for arbitrary gear sets.

---

## 5. Verdict

**The branch is NOT a restatement of the root, and it does not reach it either.**

* The inversion is a real object and it is not the F ladder read backwards (R3). That is the
  branch's finding, it corrects a recorded claim in `the_wall.md` 5a and kills a proposed identity
  in the dead-ends list, and it makes the covering formulation strictly stronger than the root
  rather than equal to it. Verdict on node: **FACT**, exact, with the mechanism (R4) visible.
* The route the thin place proposed - bound the span by a gear count - is **DEAD in its proven
  form**. The only gear count that is proved is the umbrella's, and it saturates at span `2q/3`
  (R1), a factor `q/4` below the window; worse, it is not a lower bound on the cover count at all
  (R1, `f > h`). The dichotomy turns it into a genuine gear-count inequality (R7) which reduces to
  the counting bound and loses by a factor of two, which is face A again.
* The gear count that DOES reach the window is `h`, and `h` is `A` inverted; bounding `A` above is
  the whole content of the root in covering form, with nothing proven on the tree that touches it
  (R6). So the branch converts "count gears, not columns" into one clean open lemma,
  `A(K) < (p_{K+1}^2 - 1)/6`, measured true with a margin of a factor 3 in span at `K = 4, 5, 6`,
  and identifies the only new handle it found: the arc structure, which is a which-residues
  property (face A permits it) and is where the real machine is measurably WORSE than the
  adversary.

The one thing the branch offers the root: the real machine is not the best blocker of its own
size, and the amount by which it is not is measurable and appears to be shrinking (1.45, 1.22,
1.12). If that ratio is bounded - and it must be if the covering form is true - the bound has to
come from the arc multiset, because size and capacity are both measured not to decide it.

---

## 6. Dead ends, with the refuting instance

* **The umbrella bound as a route to a length bound.** DEAD. It saturates at
  `S_sat = (q - a_q) - 1 ~ 2q/3` while the window is `q^2/6`; at rung 997 the ratio is 255. No
  argument that uses only "which gears are forced to strike" can see a span above `2q/3`.
* **The forced set as a lower bound on the cover count.** DEAD, refuted at m17: span 10 is
  forced-struck by all 5 gears and covered by 4. At m31 the excess is 4 gears.
* **"A stretch one shorter than the record can spare a gear".** DEAD at every machine; the first
  sparable span is 13 below the record at m31.
* **"Initial segments are optimal", i.e. the inversion is the F ladder.** DEAD, refuted by
  `{5,7,11,17}` blocking span 16 against `{5,7,11,13}`'s 11.
* **"The optimal set has all short arcs distinct".** DEAD past the domino regime: at `K = 7, 8, 9`
  inside `{5..31}` the best sets contain twin pairs (`{5,7,11,13,17,19,31}`), because a gear below
  the run length is no longer a bare domino.
* **The dichotomy inequality of R7 as a bound.** DEAD: its forced term is the counting bound,
  which is loose by a factor of two at every span measured.

---

## 7. Children this branch opens

* **5d.ii.i.a - the arc multiset as the object.** `A(K)` is a function of the available arcs, not
  of the primes; the arcs are `{a : 3a - 1 or 3a + 1 prime}` with multiplicity 2 at twin pairs.
  Ask: what is `A(K)` for an ADVERSARY WITH ARBITRARY ARCS (every integer arc available, one gear
  per arc)? That is a purely combinatorial covering question with the primes removed, and it upper
  bounds `A(K)`; if its growth is `(K ln K)^2/6` or worse, the covering form of the root is false
  and the framing dies cleanly. Cheap: the same search with the arc list replaced.
* **5d.ii.i.b - the ratio `A(K)/F({5..p_K})`.** Measured 1.45, 1.22, 1.12 and falling. Is it
  bounded, and by what? If it tends to 1, the covering form and the root are asymptotically the
  same statement and the wall's 5a is right in the limit though wrong in the finite range.
