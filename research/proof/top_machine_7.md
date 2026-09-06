# Closing the wheels' open laws (branch R7)

Parent: `top_machine_4.md` (branch R4.b.iii.a), whose L53 states the core/tail rule for the wheel
record as an *observation* - "0 mismatches on 13 known records, 89 sets decided" - with a
sketch and no proof; and `top_machine_2.md`, whose L25 leaves `M_k(d) = 0` for `k < r(d)` as an
OPEN claim verified to `d = 16`, and whose closing section names "L31 as a formula" and "the
vanishing moments" as the two things the next pass should take up.

**The object.** The wheels: the top machine `G` as a wheel of pairwise coprime odd gears acting
on the pair coordinate, its record `F_top(G)` (the longest run of consecutive struck pairs per
period), its gap census `N_d(G)`, and the covering problem of L16 that both reduce to. Free
wheels = every gear above `2m`; loaded wheels = small gears present; core gears = those at most
`F + 1`; tail gears = the rest. Nothing on a range, nothing about the motor, no clutch.

Numbering of laws continues from **L67** (documents 1-6 reached L21, L38, L45, L56, L59, L66 in
order; the highest number used anywhere in the six is L66, in `top_machine_6.md`).

---

## 1. Pre-registered predictions and scorecard

Written before any computation of this branch.

### Section 1. The loaded record rule, proved

**P1 (THE RULE, exact and with no side hypothesis).** For a length `L`, put
`core(L) = {g in G : g <= L + 1}` and `t(L) = #{g in G : g > L + 1}`. A **core phase vector**
`U` assigns each core gear a residue and leaves the uncovered set
`U = [0, L) \ (union of the core gears' traces)`. The **domino cost** `D(U)` is the sum, over
the maximal step-2 runs of `U` inside each of the two parity classes, of `ceil(run/2)`.
Predicted, as an *iff* with a two-sided proof:

        [0, L) is coverable  <=>  min over core phase vectors of D(U) <= t(L) ,

and hence `F_top(G) = max { L : min_U D_L(U) <= t(L) }`. Refuted by one gear set where the
formula and an independent full-period scan disagree.

**P2 (the matching lemma).** Predicted: a set `U` of cells is the union of `k` pieces each of
which is a distance-2 domino or a single cell **iff** `k >= D(U)`; dominoes never cross parity,
the distance-2 graph on `U` is a disjoint union of paths, and a path on `j` vertices needs
exactly `j - floor(j/2) = ceil(j/2)` pieces. Predicted exact on all `2^L` subsets of `[0, L)`
for `L <= 14` against a brute-force minimum: **0 exceptions**.

**P3 (the tail hypothesis is exactly `g > L + 1`, and it is sharp).** Predicted: a gear
`g > L + 1` shows inside `[0, L)` either the domino `{c, c + 2}` or one cell, never anything
else, *because* its two teeth are at distance 2 and `L - 1 < g - 2`; while `g = L + 1` shows the
end pair `{0, L - 1}` and `g = L` the wrap pair `{L - 2, 0}`, both of which **cross parity** and
are not dominoes. Predicted therefore: moving the boundary gear `g = L + 1` into the tail
**breaks** the rule, and the refuting instances are exactly the sets with `q' = F + 1` named in
`top_machine_5.md`'s dead ends (`{9,11,13,17}`, `{13,17,19,23,29,31}`,
`{17,19,23,29,31,37,41,43}`). Predicted at least 3 mismatches under the wrong boundary and 0
under the right one.

**P4 (verification).** Predicted **0 mismatches**: (a) against a full-period scan on every
pairwise-coprime odd gear set whose period is under `2 x 10^8`; (b) against the 13 independently
known records of `top_machine_4.md` 3.7; (c) against the exhaustive triple and quadruple tables
of `top_machine_2.md` L30 - all 1,540 triples and 7,315 quadruples of odd primes 7..97, where
the rule must give `F = 5` (6 with 7 a gear) and `8` (9 with 7 a gear).

### Section 2. The free/loaded boundary as a corollary

**P5 (the empty-core cost in closed form).** With an empty core, `U = [0, L)`, the two parity
classes are single step-2 runs of `ceil(L/2)` and `floor(L/2)` cells, so

        D(L) = ceil(ceil(L/2)/2) + ceil(floor(L/2)/2) = 2 floor(L/4) + min(L mod 4, 2) .

Predicted values `D(L)` for `L = 0..12`: `0, 1, 2, 2, 2, 3, 4, 4, 4, 5, 6, 6, 6`.

**P6 (the parity law falls out).** Predicted: `max { L : D(L) <= m } = 2m - (m mod 2)` for every
`m`, so the free-wheel record is the parity law, derived and not assumed. Predicted exact for
`m = 1..200`.

**P7 (the sharp threshold, and why odd `m` needs one more).** Predicted: the parity law holds
iff no gear is a core gear at `L = F + 1`, *except* that at the boundary the single smallest
gear may sit in the core and still fail to buy the extra cell. Predicted mechanism: at
`q' = 2m + 1` the one core gear offers a single parity-crossing piece; removing it from `[0, L)`
leaves two step-2 runs whose lengths are `m` and `m - 1` at even `m` (cost `m > t = m - 1`,
fails) and `m - 1` and `m - 1` at odd `m` (cost `m - 1 = t`, succeeds). Hence

        q' >= 2m + 1 (m even) ,      q' >= 2m + 3 (m odd) ,

reproducing `top_machine_5.md` L55's 7 boundary pairs with **0 exceptions**, now as a corollary
of P1 rather than a measurement.

### Section 3. The moment vanishing (L25)

**P8 (THE VANISHING, proved).** In the universal regime (`g > d + 2` for every gear),
`e(S) = |A(S)|` with `A(S) = {0, 2, d, d + 2} union {j, j + 2 : j in S}` and
`c_e(d) = sum over S with e(S) = e of (-1)^{|S|}`, so
`M_k(d) = sum over S subset of [1, d-1] of (-1)^{|S|} e(S)^k`. Predicted identity: this is
`(-1)^{d-1}` times the coefficient of the full monomial `x_1 ... x_{d-1}` in the multilinear
form of `f(x)^k`, where `f = |B union (dominoes chosen by x)|`; and that coefficient is zero
unless `k` pieces from the list `{J_p = {p - 2, p} ∩ [1, d-1] : p in [1, d+1], p != 2, p != d}`
cover `[1, d-1]`. Predicted consequence: **`M_k(d) = 0` for every `k < r(d)`**, with `r(d)` the
minimum number of such pieces - which is exactly the free-boundary domino cost `D(d - 1)`,
because the two excluded pieces are singletons and singletons never help.

**P9 (`d = 4` is not an exception but the extreme case).** Predicted: for `d = 4` the only
pieces touching position 2 are `J_2` and `J_4`, both excluded, so position 2 is uncoverable,
`r(4) = infinity`, and `M_k(4) = 0` for **every** `k` - which is `N_4 = 0`, i.e. L4. Predicted:
this removes L26's "unique exception at `d = 4`" as an exception.

**P10 (the numbers).** Predicted `r(d)` for `d = 1..24`:
`0, 1, 2, (inf), 2, 3, 4, 4, 4, 5, 6, 6, 6, 7, 8, 8, 8, 9, 10, 10, 10, 11, 12, 12`.
Predicted `M_k(d) = 0` for all `k < r(d)` and `M_{r(d)}(d) != 0`, exact to `d = 24`
(`2^23` subsets, computed by exact integer inclusion-exclusion, not sampled). Predicted
`|M_{r(d)}(d)|` at the gear-independent lengths, from L25's published record multiplicities:
`d = 6 -> 18`, `d = 7 -> 96`, `d = 8 -> 24`, `d = 9 -> 24`, `d = 10 -> 480`, `d = 11 -> 6480`,
`d = 12 -> 1440`, `d = 13 -> 720`.

### Section 4. The kernel shape of L22

**P11.** Predicted: L22 splits into exactly three lemmas - (a) a local characterisation of a
consecutive-open pair at distance `d`, (b) inclusion-exclusion over the interior positions,
(c) a CRT product per subset - each of which is a `Finset` statement with no analysis in it, and
each verifiable numerically on its own. Predicted **0 mismatches** on 5 wheels for each of the
three separately.

### Section 5. The ledger

**P12.** Predicted: the six wheel documents leave between 8 and 16 structural open items;
predicted that this branch closes at least 3 of them outright, that at least 2 are measurements
with no structural content, that at least 1 is the twin conjecture in disguise (L65), and that
the remainder are genuinely open on the wheels alone.

### Scorecard

| # | Prediction | Result |
|---|---|---|
| P1 | the loaded record rule as an iff, proved both directions | |
| P2 | matching lemma: min pieces `= D(U)`, exhaustive to `L = 14` | |
| P3 | the tail hypothesis `g > L + 1` is sharp; the wrong boundary breaks the rule | |
| P4 | 0 mismatches: scan, 13 known records, 1,540 triples, 7,315 quadruples | |
| P5 | `D(L) = 2 floor(L/4) + min(L mod 4, 2)`; `0,1,2,2,2,3,4,4,4,5,6,6,6` | |
| P6 | `max{L : D(L) <= m} = 2m - (m mod 2)`, `m = 1..200` | |
| P7 | threshold `2m + 1` / `2m + 3` derived, 7 boundary pairs, 0 exceptions | |
| P8 | `M_k(d) = 0` for `k < r(d)`, proved by the covering argument | |
| P9 | `r(4) = infinity`, `M_k(4) = 0` for all `k`, `d = 4` is not an exception | |
| P10 | `r(d)` table to `d = 24`; `M_{r(d)} != 0`; `18, 96, 24, 24, 480, 6480, 1440, 720` | |
| P11 | L22 as three lemmas, each verified on 5 wheels | |
| P12 | the ledger: >= 3 closed, >= 2 measurements, >= 1 the conjecture, rest open | |

---

## 2. Setup as computed

Scripts in `research/topmachine/r7/`, results (untracked) in `.../results/`. Every count is
exact - full-period scans, exhaustive subset enumerations and exact integer inclusion-exclusion;
nothing is sampled and nothing is fitted.

| script | what it computes |
|---|---|
| `rule.py` | the matching lemma exhaustively; the rule against a full-period scan on every scannable gear set; against the 13 known records; against the exhaustive triple/quadruple tables; the wrong-boundary control |
| `sharp.py` | the wrong boundary over the whole scannable family; the capacity bound |
| `check.py` | how many of those sets could possibly break (`F + 1` a gear), against how many do |
| `boundary.py` | the empty-core cost in closed form; the parity law from it; the sharp threshold and the mechanism at the boundary |
| `moments.py` | `c_e(d)` and `M_k(d)` exactly to `d = 26` by integer inclusion-exclusion over all `2^(d-1)` subsets; `r(d)` independently by exact set cover; the `d = 4` degeneracy |
| `kernel.py` | the three lemmas of L22, each separately, on 5 wheels |

Ranges. Matching lemma: all `2^L` subsets of `[0, L)` for `L = 1..14` (32,766 sets). Wheel
records: **6,659 gear sets** - every pairwise-coprime subset of
`{5,7,9,11,13,17,19,23,25,29,31,37,41,43,47,49}` of size 2 to 6 with period at most 24,000,000 -
each decided twice, once by a full-period cyclic scan and once by the rule; **5,006 of the 6,659
are loaded** (nonempty core), which is the regime the parity law cannot reach. Plus the 13
independently known records and all 1,540 triples and 7,315 quadruples of odd primes 7..97.
Moments: `d = 2..26`, up to 33,554,432 subsets per length, exact integers. Census lemmas:
5 wheels (`{7,11,13}`, `{11,13,17}`, `{5,7,11,13}`, `{7,11,13,17}`, `{9,11,13,17}`), gap lengths
1..10, 5,115 subset terms. Composite gears (9, 25, 49) are included everywhere: nothing in this
branch uses primality, only pairwise coprimality and oddness.

---

## 3. Results

### 3.1 The loaded record rule, proved and verified

The rule is a theorem with two directions (proof in section 4, L69). Verification:

| test | sets | mismatches |
|---|---|---|
| matching lemma `min #pieces = D(U)`, exhaustive to `L = 14` | 32,766 subsets | **0** |
| the rule against a full-period cyclic scan | 6,659 gear sets (5,006 loaded) | **0** |
| the rule against the 13 independently known records | 13 | **0** |
| the rule against the exhaustive triple and quadruple tables (L30) | 8,855 | **0** |
| `D(L)` against the closed form `2 floor(L/4) + min(L mod 4, 2)` | `L = 0..399` | **0** |

The triple and quadruple counts come out of the rule exactly as L30 measured them: `m = 3` gives
`F = 5` at 1,330 sets and `6` at the 210 containing 7; `m = 4` gives `8` at 5,985 and `9` at the
1,330 containing 7 - and the rule now *explains* the split, because 7 is the only gear among
7..97 that is at most `L + 1` at those record lengths.

The core/tail split at the record, on the 13 known sets:

| gears | `m` | `F_top` | core (`g <= F + 1`) | `t` |
|---|---|---|---|---|
| 7,11,13 | 3 | 6 | {7} | 2 |
| 11,13,17 | 3 | 5 | empty | 3 |
| 7,11,13,17 | 4 | 9 | {7} | 3 |
| 11,13,17,19,23 | 5 | 10 | {11} | 4 |
| 7,11,13,17,19,23,29,31 | 8 | 32 | all eight | 0 |
| 13,17,19,23,29,31,37,41 | 8 | 18 | {13,17,19} | 5 |
| 19,23,29,31,37,41,43,47 | 8 | 16 | empty | 8 |

### 3.2 The tail hypothesis is exactly `g > L + 1`, and it is sharp

A gear `g > L + 1` shows inside `[0, L)` either the domino `{c, c + 2}` or a single cell, and
nothing else. A gear at either boundary size shows something else:

| gear size | what it can show inside `[0, L)` |
|---|---|
| `g > L + 1` | `{c, c+2}`, or one cell when the partner falls outside |
| `g = L + 1` | a domino, **or the end pair `{0, L-1}`**, or one cell |
| `g = L` | a domino, **or a wrap pair `{0, L-2}` or `{1, L-1}`** |
| `g <= L - 1` | a `g`-periodic trace of up to `2 ceil(L/g)` cells |

The two boundary pieces **join the two ends of the window**: their separations are `L - 1` and
`L - 2`, not 2, so at the corresponding parity of `L` they cross parity - the only pieces in the
whole machine that can. Counting `g = L + 1` as a tail gear therefore breaks the rule. Measured
over the whole scannable family:

| test | sets | differences |
|---|---|---|
| rule with core `{g <= L + 1}` against the scan | 6,659 | **0** |
| rule with core `{g <= L}` against the scan | 6,659 | **605**, every one exactly 1 short |
| sets where a break is even possible (`F + 1` is a gear) | 6,659 | 1,422 |

A break *requires* `F + 1` to be a gear, because at length `L` the only gear whose classification
changes is `g = L + 1`; that holds at **1,422** of the 6,659 sets, and the record actually drops
at **605** of those 1,422 - the ones whose record cover genuinely needs the ends-joining piece.
The breakers are sets like `{7,9,11}`, `{7,11,13}`, `{7,9,13}`, all with `F = 6` and the gear 7
at `L + 1`; the failure is always downward and always by exactly one cell, the one cell the
ends-joining piece buys. At the other 817 the boundary gear is present at `L + 1` but the cover
does not need it: notably at `q' = F + 1` (`{9,11,13,17}`, `{13,17,19,23,29,31}`,
`{17,19,23,29,31,37,41,43}`) the record is already a free tiling, which is why those three -
pre-registered as the expected refuting instances - do **not** break.

### 3.3 The free/loaded boundary, as a corollary

With an empty core the uncovered set is the whole window, its two parity classes are single
step-2 runs of `ceil(L/2)` and `floor(L/2)` cells, and

        D(L) = 2 floor(L/4) + min(L mod 4, 2)  =  2k, 2k+1, 2k+2, 2k+2  for L = 4k + 0,1,2,3 .

Solving `D(L) <= m`: at `m = 2k` the largest such `L` is `4k = 2m`; at `m = 2k + 1` it is
`4k + 1 = 2m - 1`. That is the parity law `F_top = 2m - (m mod 2)`, **derived**, 0 mismatches for
`m = 1..200`. The defect `-(m mod 2)` is the one cell with no partner inside its own parity
class.

The threshold, measured with the rule over gear sets built by taking the next coprime odd number
each time (odd composites included, so `q' = 2m + 1` is reachable at every `m`):

| `m` | `P = 2m - (m mod 2)` | `F` at `q' = 2m-1` | at `q' = 2m+1` | at `q' = 2m+3` | smallest `q'` that works | predicted |
|---|---|---|---|---|---|---|
| 2 | 4 | - | **4** | 4 | 5 = 2m+1 | 5 |
| 3 | 5 | 9 | 6 | **5** | 9 = 2m+3 | 9 |
| 4 | 8 | 12 | **8** | 8 | 9 = 2m+1 | 9 |
| 5 | 9 | 13 | 10 | **9** | 13 = 2m+3 | 13 |
| 6 | 12 | 16 | **12** | 12 | 13 = 2m+1 | 13 |
| 7 | 13 | 17 | 14 | **13** | 17 = 2m+3 | 17 |
| 8 | 16 | 20 | **16** | 16 | 17 = 2m+1 | 17 |
| 9 | 17 | 21 | 18 | **17** | 21 = 2m+3 | 21 |

**0 mismatches** against `top_machine_5.md` L55 - the threshold is now a corollary, not a
measurement. And the mechanism, computed cell by cell at `q' = 2m + 1`, `L = P + 1`, where
exactly one gear is a core gear:

| `m` | `L = P+1` | `q'` | `q'` vs `L` | its best piece | cells left | runs left | `D(left)` | `t = m-1` | coverable |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 5 | 5 | `= L` | {0,3} | 3 | 2, 1 | 2 | 1 | no - parity law holds |
| 3 | 6 | 7 | `= L+1` | {0,5} | 4 | 2, 2 | 2 | 2 | **yes - parity law fails** |
| 4 | 9 | 9 | `= L` | {0,7} | 7 | 4, 3 | 4 | 3 | no |
| 5 | 10 | 11 | `= L+1` | {0,9} | 8 | 4, 4 | 4 | 4 | **yes** |
| 6 | 13 | 13 | `= L` | {0,11} | 11 | 6, 5 | 6 | 5 | no |
| 7 | 14 | 15 | `= L+1` | {0,13} | 12 | 6, 6 | 6 | 6 | **yes** |
| 8 | 17 | 17 | `= L` | {0,15} | 15 | 8, 7 | 8 | 7 | no |
| 9 | 18 | 19 | `= L+1` | {0,17} | 16 | 8, 8 | 8 | 8 | **yes** |

**Even `m`.** `L = 2m + 1` is odd, so the core gear sits at `g = L`, its ends-joining piece is
the wrap pair `{0, L-2}`, and what remains is two step-2 runs of `m` and `m - 1` cells: cost
`m/2 + m/2 = m`, one more than the `t = m - 1` tail gears. The cover fails; the parity law
survives at `q' = 2m + 1`.

**Odd `m`.** `L = 2m` is even, so the core gear sits at `g = L + 1`, its ends-joining piece is
the end pair `{0, L-1}`, and what remains is two runs of `m - 1` cells **each** - and `m - 1` is
even, so each tiles exactly: cost `(m-1)/2 + (m-1)/2 = m - 1 = t`. The cover closes; the parity
law dies, and `q'` must go up to `2m + 3`. That single parity bit - whether `m - 1` is even - is
the whole of the `+2`.

A second corollary falls out on the way, and it settles a refuted guess of
`top_machine_5.md`. The record cover is a free-domino tiling iff the empty-core cost of `[0, F)`
is affordable, `D(F) <= m`, i.e. (since `D` is non-decreasing) iff `F <= 2m - (m mod 2)`; and
`F >= 2m - (m mod 2)` always (attainment, which needs no hypothesis at all). So **the cover is a
free tiling exactly when the parity law holds** - at odd `m` a tiling plus the one edge singleton
of L43 -
L55's equivalence (b) = (c), derived - and `q' = F + 1` is no obstacle to a free tiling, which is
why `{9,11,13,17}`, `{13,17,19,23,29,31}` and `{17,19,23,29,31,37,41,43}` refuted the guess
"free tiling iff `q' > F + 1`".

### 3.4 The capacity bound: a closed-form upper bound with no enumeration

A cover of `[0, L)` must cover `L` cells; a tail gear covers at most 2 and a core gear at most
`2 ceil(L/g)`. So `F_top(G) <= Lcap(G) = max { L : L <= 2 t(L) + sum over core of 2 ceil(L/g) }`,
computable in a line. Measured on 37 gear sets: **0 violations**, exact (slack 0) at 10 of them -
`{11,13,17,19}`, `{13,17,19,23}`, `{17,19,23,29}`, `{19..47}`, `{17,19,23,29,31,37}` and the
other free wheels - and progressively loose on loaded wheels with small gears: slack 6 at
`{13..41}` (18 against 24), 9 at `{7,11,13,17}` (9 against 18), 123 at `{7..31}` (32 against
155). The bound is tight exactly where the core is empty, which is where it reduces to
`L <= 2m`.

### 3.5 The moment vanishing, proved, and `r(d)` to `d = 26`

`M_k(d) = 0` for every `k < r(d)`: exact over `d = 2..26`, up to 33,554,432 subsets per length,
integer arithmetic, **0 failures**; and `M_{r(d)}(d) != 0` at every such `d`, **0 failures**.
`r(d)` computed three ways - the closed form `D(d - 1)`, an independent exact set cover over the
available pieces (to `d = 20`), and the observed first non-vanishing `k` - agrees everywhere,
**0 mismatches**:

| `d` | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `r(d)` | 1 | 2 | **inf** | 2 | 3 | 4 | 4 | 4 | 5 | 6 | 6 | 6 |
| `M_{r(d)}(d)` | -2 | 2 | 0 | 2 | -18 | 96 | 24 | 24 | -480 | 6480 | 1440 | 720 |

| `d` | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `r(d)` | 7 | 8 | 8 | 8 | 9 | 10 | 10 | 10 | 11 | 12 | 12 | 12 | 13 |
| `M_{r(d)}(d)` | -25200 | 645120 | 120960 | 40320 | -2177280 | 90720000 | 14515200 | 3628800 | -279417600 | 17244057600 | 2395008000 | 479001600 | -49816166400 |

The eight lengths where L18/L25 published a record multiplicity are reproduced exactly:
`|M_{r(d)}(d)| = 18, 96, 24, 24, 480, 6480, 1440, 720` at `d = 6..13`, **0 mismatches of 8**.
The universal signatures agree with the published ones: `c_e(3) = c_e(5) = (1, -2, 1)` at
`e = 4, 5, 6` - L24's gap-3 / gap-5 coincidence, now an identity of integer vectors and not just
of the two polynomials - and `c_e(6) = (1, -1, -3, 5, -2)` at `e = 4..8`, the signature that
refuted the hand-derived `N_6`.

**`d = 4` is not an exception.** The pieces available on the ground set `[1, d-1] = [1, 3]` are
`J_1 = {1}`, `J_3 = {1,3}` and `J_5 = {3}`; the only two that would contain position 2 are `J_2`
and `J_4`, and those are exactly the two the census law excludes (they are the positions already
in the boundary set `B`). Position 2 is in no piece, so no `k` covers it, `r(4) = infinity`, and
`M_k(4) = 0` for **every** `k` - measured 0 at `k = 0..8`. That is `N_4 = 0` identically:
**L4, the forbidden gap of 4, is the `d = 4` case of the same covering statement**, not a
boundary condition sitting beside it, and L26's "unique exception at `d = 4`" disappears.

### 3.6 The three lemmas of L22, separately

| wheel | `W` | K1 (local) | K2 (inclusion-exclusion) | K3 (CRT product) | assembled L22 vs scan |
|---|---|---|---|---|---|
| 7,11,13 | 1,001 | 0 | 0 | 0 | 0 |
| 11,13,17 | 2,431 | 0 | 0 | 0 | 0 |
| 5,7,11,13 | 5,005 | 0 | 0 | 0 | 0 |
| 7,11,13,17 | 17,017 | 0 | 0 | 0 | 0 |
| 9,11,13,17 | 21,879 | 0 | 0 | 0 | 0 |

**0 mismatches** at every stage, gap lengths 1..10, 5,115 subset terms. The censuses, for the
record: `{11,13,17}` has 52 gaps of length 3 and 52 of length 5 and **none of length 4**;
`{9,11,13,17}` has 642 and 642; `{7,11,13}` has 32 and 34 - unequal exactly because 7 is a gear.
The composite gear 9 behaves like any other gear at every stage; no lemma uses primality.

---

## 4. Laws

Numbered from **L67**. `G` is a finite set of pairwise coprime odd integers `g >= 3` (gears),
acting on the pair coordinate by striking `n` when `n = 0` or `n = -2` mod `g`; `m = |G|`,
`W = prod G`, `q' = min G`; `F_top(G)` is the longest run of consecutive struck pairs per period.
For a window length `L`, `core(L) = {g in G : g <= L + 1}` and `t(L) = #{g in G : g > L + 1}`.

**L67 (THE PIECE LAW - what a gear can show in a window, and where the tail begins).**
Fix `L >= 1` and a phase for gear `g`, i.e. a residue `a = (-x) mod g`; the gear's **trace** in
`[0, L)` is `{c in [0, L) : c = a or a - 2 mod g}`.

* if `g > L + 1`, the trace is `{c, c + 2}` for some `c` with `c + 2 < L`, or a single cell, or
  empty - **always a subset of a distance-2 domino, and always inside one parity class**;
* if `g = L + 1`, the trace is a domino, or the **end pair** `{0, L - 1}`, or a single cell;
* if `g = L`, the trace is a domino, or a **wrap pair** `{0, L - 2}` or `{1, L - 1}`;
* if `g <= L - 1`, the trace has `2 ceil(L/g)` cells up to boundary effects and is `g`-periodic.

*Proof.* The two teeth are the residues `a` and `a - 2` mod `g`. Each residue class meets a
window of length `L <= g` in at most one cell, so for `g >= L` the trace has at most two cells,
at positions `c1 = c2 + 2` or `c1 = c2 + 2 - g`. The second case needs `2 - g >= -(L - 1)`, i.e.
`g <= L + 1`; so for `g > L + 1` only the first occurs and the two cells differ by exactly 2. At
`g = L + 1` the window misses exactly one residue and the wrapped case gives `c2 = 0`,
`c1 = L - 1`; at `g = L` every residue occurs once and the wrapped case gives `{0, L-2}` or
`{1, L-1}`. For `g <= L - 1` each of the two classes meets the window `ceil(L/g)` or
`floor(L/g)` times. QED

*Reading.* **A gear can join the two ends of the window if and only if `g <= L + 1`.** That, and
not size as such, is what makes a gear a core gear. Since the ends-joining pieces have
separations `L - 1` and `L - 2`, they cross parity exactly when `L` is even and odd
respectively; every other piece in the machine lives inside one parity class.

**L68 (THE MATCHING LEMMA - the domino cost is the exact price of a set).** Let `U` be a set of
cells and let `D(U)` be the sum, over the maximal step-2 runs of `U` inside each parity class, of
`ceil(run/2)`. Then `U` is contained in the union of `k` pieces, each a distance-2 pair or a
single cell, **iff** `k >= D(U)`.

*Proof.* A piece meets `U` in at most two cells and those two are at distance 2, hence in the
same parity class and adjacent in that class's step-2 order. So a covering by `k` pieces is a
partition of `U` into `k` parts each of size at most 2 with the two elements at distance 2; the
minimum number of parts is `|U|` minus the maximum matching of the graph on `U` whose edges are
the distance-2 pairs. That graph is a disjoint union of paths - one per maximal step-2 run - and
a path on `j` vertices has maximum matching `floor(j/2)`, so the minimum is
`sum_runs (j - floor(j/2)) = sum_runs ceil(j/2) = D(U)`. Conversely `D(U)` pieces suffice: pair
along each run and leave at most one cell per run as a singleton. QED

*Evidence.* All `2^L` subsets of `[0, L)` for `L = 1..14` against a brute-force minimum:
32,766 sets, **0 mismatches**.

**L69 (THE LOADED RECORD RULE - L31 as a formula, proved).** For every `L >= 1`,

        [0, L) is coverable  <=>  min over core phase vectors of D(U) <= t(L) ,

where `U = [0, L) \ (union of the core gears' traces)`; and consequently

        F_top(G) = max { L : min over core phase vectors of D_L(U) <= t(L) } .

No hypothesis beyond pairwise coprimality and `g >= 3` is needed; in particular the "tail
hypothesis" is exactly `g > L + 1`, which is the definition of the tail.

*Proof.* **Necessity.** Suppose `[0, L)` is covered by the traces of all gears at some phase
vector. Restrict to the core gears: they leave an uncovered set `U`, which is the `U` of that
core phase vector, and the tail gears must cover `U`. By L67 each tail gear's trace is contained
in a distance-2 domino, so by L68 covering `U` needs at least `D(U)` of them; hence
`D(U) <= t(L)` and the minimum over core phase vectors is at most `t(L)`.

**Sufficiency.** Take a core phase vector attaining the minimum, with uncovered set `U` and
`D(U) <= t(L)`. By L68 `U` is the union of `D(U)` pieces, each a pair `{c, c + 2}` inside
`[0, L)` or a single cell `{c}`. Assign the pieces to distinct tail gears, one each, and give
tail gear `g` the phase `a = (-c) mod g` for its piece's left cell `c`. Since `g > L + 1` and
`0 <= c < L`, the residues `c` and `c + 2` are their own representatives modulo `g`, so that
gear's trace inside `[0, L)` is exactly `{c, c + 2}` (or `{c}` if `c + 2 >= L`), which contains
the piece. Give the remaining `t(L) - D(U)` tail gears any phase. The gears are pairwise coprime,
so CRT produces a single `x` realising every chosen phase simultaneously, and `[0, L)` is
covered. QED

*The record.* Coverability is monotone in `L` - restricting a cover of `[0, L)` to `[0, L')` for
`L' < L` leaves the same phases covering every cell - so the coverable lengths are an initial
segment and `F_top` is their maximum. (`wheelrec.py` assumed this monotonicity as an observation;
it is a one-line consequence of restriction.)

*Evidence.* **0 mismatches** against a full-period cyclic scan on **6,659** pairwise-coprime odd
gear sets with period up to 24,000,000, of which **5,006 are loaded**; **0 of 13** against the
independently known records; **0 of 8,855** against the exhaustive triple and quadruple tables of
L30. *This is `top_machine_2.md` L31 turned from a tabulated dependence into a formula, and
`top_machine_4.md` L53 turned from an observation into a theorem.*

**L70 (THE CAPACITY BOUND - a closed-form upper bound with no search).**

        F_top(G) <= Lcap(G) = max { L : L <= 2 t(L) + sum over g in core(L) of 2 ceil(L/g) } .

*Proof.* A cover of `[0, L)` covers `L` cells; by L67 a tail gear's trace has at most 2 cells and
a core gear's at most `2 ceil(L/g)`. QED

*Evidence.* 37 gear sets, **0 violations**; equality at 10 of them, all with an empty core, where
the bound reduces to `L <= 2m`. Slack grows with the core: 6 at `{13..41}`, 9 at `{7,11,13,17}`,
123 at `{7..31}`. *The bound is free of the exponential minimisation and is the natural companion
to L69's exact but exponential decision.*

**L71 (THE PARITY LAW, DERIVED).** With an empty core, `D([0, L)) = 2 floor(L/4) + min(L mod 4,
2)`, and `max { L : D(L) <= m } = 2m - (m mod 2)`. Hence for a free wheel
`F_top = 2m - (m mod 2)` - `top_machine_1.md` L17, now a corollary of L69 rather than a separate
theorem, with the defect `-(m mod 2)` identified as the single cell that has no partner in its
own parity class.

*Evidence.* `m = 1..200`, **0 mismatches**; the closed form for `D(L)` checked at `L = 0..399`,
**0 mismatches**.

**L72 (THE SHARP THRESHOLD, DERIVED, AND WHY ODD `m` NEEDS ONE MORE).** The parity law holds iff
`[0, P + 1)` is not coverable, `P = 2m - (m mod 2)`. At `q' = 2m + 1` exactly one gear is a core
gear at that length, and it offers exactly one ends-joining piece:

* `m` even: `L = P + 1 = 2m + 1` is odd, `q' = L`, the piece is the wrap pair `{0, L-2}`, what
  remains is two step-2 runs of `m` and `m - 1` cells, cost `m/2 + m/2 = m > t = m - 1`. Not
  coverable: the parity law holds at `q' = 2m + 1`.
* `m` odd: `L = P + 1 = 2m` is even, `q' = L + 1`, the piece is the end pair `{0, L-1}`, what
  remains is two runs of `m - 1` cells each, and `m - 1` is **even**, so the cost is
  `(m-1)/2 + (m-1)/2 = m - 1 = t`. Coverable: the parity law fails, and `q'` must reach `2m + 3`.

Hence the threshold `q' >= 2m + 1` (`m` even), `q' >= 2m + 3` (`m` odd) - `top_machine_5.md` L55,
now derived. *Evidence.* 8 values of `m`, three values of `q'` each, **0 mismatches**; the
mechanism table computed cell by cell (3.3).

**L73 (THE MOMENT VANISHING - `M_k(d) = 0` for `k < r(d)`, PROVED).** In the universal regime
(`g > d + 2` for every gear) put `A(S) = {0, 2, d, d+2} u {j, j+2 : j in S}` for
`S subset of [1, d-1]`, `e(S) = |A(S)|`, `c_e(d) = sum over S with e(S) = e of (-1)^{|S|}` and
`M_k(d) = sum_e c_e(d) e^k`. Then `M_k(d) = 0` for every `k` less than

        r(d) = the least number of pieces J_p = {p-2, p} n [1, d-1] ,
               p in [1, d+1] , p != 2 , p != d ,   whose union is [1, d-1] ,

with `r(d) = infinity` when no such family exists.

*Proof.* Identify `S` with `x in {0,1}^{d-1}` and write `f(x) = e(S)`. For a position
`p in [1, d+1]` outside `B = {0, 2, d, d+2}` let `u_p(x) = 1 - prod_{j in J_p} (1 - x_j)`, the
indicator that `p` is covered by one of the chosen dominoes; then

        f(x) = 4 + sum over p not in B of u_p(x) ,

because `|B| = 4` for `d >= 3` and every element of `A(S)` outside `B` is such a `p`. The
positions `p = 2` and `p = d` lie in `B` and so contribute no `u_p`: those are exactly the two
excluded pieces. Now `M_k(d) = sum over x in {0,1}^{d-1} of (-1)^{|x|} f(x)^k`, and for any
function `h` on `{0,1}^n`, `sum_x (-1)^{|x|} h(x) = (-1)^n` times the coefficient of the full
monomial `x_1 ... x_n` in the (unique) multilinear form of `h`; a monomial of the multilinear
form of a function that does not depend on some variable `x_j` cannot be the full monomial.
Expand `f^k = (4 + sum_p u_p)^k`: every term is `4^{k-i}` times a product of `i <= k` of the
`u_p`, and such a product depends only on the variables in the union of the corresponding `J_p`.
So if fewer than `r(d)` pieces are available in any term - which is the case whenever `k < r(d)`,
since `i <= k` - no term depends on all `d - 1` variables, the full monomial has coefficient
zero, and `M_k(d) = 0`. QED

**L74 (`r(d)` IS THE FREE-BOUNDARY DOMINO COST: `r(d) = D(d - 1)`, AND `d = 4` IS THE
DEGENERATE CASE).** The pieces of L73 are: every step-2 domino `{p-2, p}` inside `[1, d-1]`,
together with the two singletons `{1}` (from `p = 1`) and `{d-1}` (from `p = d+1`). By L68
singletons never reduce the count, so the minimum is the domino cost of the interval `[1, d-1]`,
i.e. `D(d - 1) = ceil(ceil((d-1)/2)/2) + ceil(floor((d-1)/2)/2)`. The two excluded pieces `J_2`
and `J_d` are also singletons (`{2}` and `{d-2}`), so excluding them changes nothing - **except
at `d = 4`**, where they are the only pieces containing position 2 at all; there the covering is
impossible, `r(4) = infinity`, and `M_k(4) = 0` for every `k`, i.e. `N_4 = 0` identically.

*Evidence.* `r(d)` by exact set cover against `D(d-1)`: `d = 2..20`, **0 mismatches** (`d = 4`
returning IMPOSSIBLE); the first non-vanishing `k` of `M_k(d)` equals `r(d)` at every `d = 2..26`,
**0 exceptions**; `M_k(4) = 0` measured at `k = 0..8`. *This closes L26 and removes its
exception: `r(d)` is the record problem's own cost function `D` evaluated at `d - 1`, so the gap
census's covering number and the record's covering cost are the same object, and L4 is the case
where that object is infinite.*

**L75 (THE DEGREE LAW'S CONVERSE, measured).** `M_{r(d)}(d) != 0` at every `d = 2..26` except
`d = 4`: **0 exceptions of 24**. With L73 this makes `deg N_d = m - r(d)` exact, and the
gear-independent case `r(d) = m` gives the record multiplicities
`18, 96, 24, 24, 480, 6480, 1440, 720` at `d = 6..13`, **0 mismatches of 8** against L18/L25.
*Not proved:* the top coefficient at `k = r(d)` is `r(d)!` times an alternating sum over the
minimal covers, and no argument here rules out cancellation. This is the one arithmetic step
between L73 and a full proof of L25.

---

## 5. Kernel shape of L22

What the Formalist needs, in the shape the existing `TopMachine` namespace already uses
(`StrikesR`, `OpenN`, `card_filter_crt`). `G : Finset N` with `hcop` pairwise coprime and
`hg : forall g in G, 3 <= g`; `W = prod G`; `d : N`, `1 <= d`.

**Definitions.**

```lean
-- the offsets whose negatives n must avoid: the four boundary ones, plus i and i+2 for i in S
def Offs (d : ℕ) (S : Finset ℕ) : Finset ℕ :=
  ({0, 2, d, d + 2} : Finset ℕ) ∪ S ∪ S.image (· + 2)

-- E_g(S), the forbidden residues of n modulo g
def E (d : ℕ) (S : Finset ℕ) (g : ℕ) : Finset ℕ :=
  (Offs d S).image (fun x => (g - x % g) % g)

def ConsecOpen (G : Finset ℕ) (d n : ℕ) : Prop :=
  OpenN G n ∧ OpenN G (n + d) ∧ ∀ i, 0 < i → i < d → ¬ OpenN G (n + i)

def Ncount (G : Finset ℕ) (d : ℕ) : ℕ :=
  ((Finset.range (∏ g in G, g)).filter (ConsecOpen G d)).card
```

**K1 (the local characterisation).** `ConsecOpen G d n` holds **iff**

* for every `g in G`, `n mod g` avoids `{0, -2, -d, -d-2}` (four residues, distinct when
  `g > d + 2`, fewer otherwise), **and**
* for every `i` with `0 < i < d`, there exists `g in G` with `n mod g in {-i, -(i+2)}`.

*Proof in one step from the definitions*: `OpenN G n` is `forall g, n != 0, -2 mod g`, and
`OpenN G (n+d)` is `forall g, n != -d, -d-2 mod g`; `not OpenN G (n+i)` is
`exists g, n = -i or -(i+2) mod g`. No arithmetic beyond `mod` rewriting. Verified: 5 wheels,
`d = 1..10`, **0 mismatches** against a direct cyclic census.

**K2 (inclusion-exclusion over the interior positions).** Let
`A i = {n : forall g in G, n != -i, -(i+2) mod g}` be the event "position `n + i` is *not*
struck". Then, over `n in range W`,

        Ncount G d = sum over S in (Finset.range (d-1)).image (·+1) |>.powerset of
                     (-1)^{S.card} * card {n : (K1 first clause) and forall i in S, n in A i} .

*Proof*: K1's second clause is `forall i, n notin A i`; apply
`Finset.prod_sub` / the standard `Finset.inclusion_exclusion` over the `d - 1` events `A i`
intersected with the fixed set of the first clause. This is the only lemma that needs a general
inclusion-exclusion over a `Finset.powerset`; no property of the machine enters. Verified:
5 wheels, `d = 1..10`, **0 mismatches** (each of the 5,115 subset terms counted by scan and
summed).

**K3 (the CRT product for each subset).** For fixed `S`, the predicate
"`n mod g` avoids `E d S g` for every `g in G`" is a conjunction of per-gear residue
conditions, so by `card_filter_crt` (already in the kernel, the engine behind `wheel_count`)

        card {n in range W : ...} = prod over g in G of (g - (E d S g).card) .

*Proof*: exactly the existing `wheel_count` argument with the per-gear allowed set
`Finset.range g \ E d S g` in place of `{0, g-2}`. Verified: 5 wheels, every one of the 5,115
subsets, **0 mismatches** between the scan count and the product.

**L22 is then K1, K2, K3 composed**, with no further step:

        N_d(G) = sum over S subset of [1, d-1] of (-1)^{|S|} prod over g in G of
                 (g - |E_g(S)|) .

Verified assembled: 5 wheels, `d = 1..10`, **0 mismatches**. Carrying L22 into the kernel carries
L4, L9, L15, L18 and (with L73 and L74, both of which are pure combinatorics on
`Finset (Fin (d-1))` and need no CRT) L25 and L26 with it.

---

## 6. The wheels' open items - the ledger entry

Every structural item left open or filed as a dead end by the six wheel documents, classified:
**(a)** closed by this branch, **(b)** a measurement or a compute cutoff with no structural
content, **(c)** the twin conjecture in disguise and not the wheels' business, **(d)** genuinely
open on the wheels alone.

### (a) Closed by this branch - 6

| item | where it was left | status now |
|---|---|---|
| **L31 as a formula** ("the function itself is tabulated, not derived", `top_machine_2.md` s.10) | open | **closed**: L69, an iff with both directions proved, 0 mismatches on 6,659 + 13 + 8,855 sets |
| **The vanishing moments** `M_k(d) = 0` for `k < r(d)` (`top_machine_2.md` s.10) | verified to `d = 16`, unproved | **closed**: L73, proved; verified to `d = 26` |
| **`r(d)` in closed form** (named as a likely by-product of the above) | open | **closed**: L74, `r(d) = D(d-1)`, the record problem's own cost function |
| **L26's `d = 4` exception** | "the unique exception" | **closed**: not an exception - `r(4) = infinity` because the two pieces covering position 2 are the two the census excludes, so `N_4 = 0`; L4 is the degenerate case of the same statement |
| **L18 universal record multiplicity** (`top_machine_1.md`, measured) | measured | **closed** modulo L75: derived from L73 + L74, values reproduced 8 of 8 |
| **"free-domino tiling iff `q' > F + 1`"** (`top_machine_5.md` dead ends, refuted with three counterexamples) | refuted, unexplained | **closed**: the cover is a free tiling iff `D(F) <= m` iff `F = 2m - (m mod 2)` iff the parity law - so `q' = F + 1` is no obstacle (3.3) |

Two more are closed as *derivations* rather than new results: the parity law (L71) and its sharp
threshold (L72) are now corollaries of L69 instead of independent theorems, and the mechanism of
the odd-`m` `+2` is exhibited cell by cell.

### (b) Measurements or compute cutoffs, no structural content - 5

| item | why it is not structural |
|---|---|
| **The census beyond `d = 20`** (`top_machine_2.md` s.10) | a cost, not a law: both routes are `2^(d-1)`. This branch pushes the *universal* signature to `d = 26` and, via L73, cuts `N_d` to its top `m - r(d) + 1` moments; a specific wheel with small gears still needs the full sum |
| **The largest ladder rungs are lower bounds** (`top_machine_1.md` dead ends) | a search cutoff. L69 decides any single length exactly and L70 caps the ladder in closed form |
| **L32, L33 and L21 at small `q'`** (`top_machine_5.md` dead ends, "the only rows with no measurement") | whole-period scans of wheels containing 2 and 3; a budget item, and in any case range statements, not wheel ones |
| **L18 below `q' = 7`** (`top_machine_5.md` dead ends) | not a law there, and L25 already says why: the gear-independence hypothesis `g > d + 2` is violated. Nothing left to find |
| **A bijection behind W1** (`top_machine_1.md`, `top_machine_2.md` dead ends) | answered no - every shift and reflection refuted - and L24 plus this branch's `c_e(3) = c_e(5) = (1,-2,1)` give the reason: an identity of integer signature *vectors*, not of sets |

### (c) The conjecture in disguise - 2, and NOT the wheels' business

| item | what it really is |
|---|---|
| **An upper bound on the zone record** (`top_machine_6.md` L65, dead ends) | by L59 the zone's open pairs are `P' - P = 2` in primes above `Q`; an upper bound on the record is a lower bound for a linear twin-prime problem. **This is the range, not the wheels** |
| **An upper bound on the walk above `Q`** (`top_machine_4.md` s.5, "what is not proved") | a two-dimensional sieve lower bound in a short interval. Again the range, not the wheels |

The wheels' record `F_top(G)` is a finite covering problem on `[0, L)` and is now decided
exactly; neither of these two items is a statement about it, and nothing in this branch is
offered against the twin conjecture.

### (d) Genuinely open on the wheels alone - 4

1. **The cost of `min_U D_L(U)`.** L69 is exact but the minimisation runs over `prod(core)`
   phase vectors, exponential in the core. *Exact statement:* is there an algorithm polynomial in
   `L` and `|core|` (or a closed form) for `min over core phase vectors of D_L(U)`? *Attack:* a
   transfer matrix along the window with state "the vector of core phases reduced to their action
   on the last two cells", or a DP over the core wheel's period with the domino cost as an
   additive weight; the run structure of `U` is local, so the cost is a sum of local terms and
   the obstruction is only the phases' global periodicity. This is the natural child branch.
2. **`M_{r(d)}(d) != 0` (L75).** Measured 0 exceptions to `d = 26`, unproved. *Exact statement:*
   the coefficient of the full monomial in `f^{r(d)}` is `r(d)!` times a signed sum over the
   minimal covers of `[1, d-1]`; show the signs do not cancel. *Attack:* the minimal covers are
   enumerable in closed form (each parity class is a run tiled by `ceil(k/2)` pieces, with at most
   one overlap or one end singleton), so the sum is a product of two one-dimensional sums;
   evaluate them.
3. **L22 in the kernel.** The shape is now written (section 5) with all three lemmas verified
   separately; only K2's general `Finset.powerset` inclusion-exclusion is new machinery. *Attack:*
   `Finset.prod_one_sub` / the existing `card_filter_crt`; K1 and K3 are mechanical.
4. **A per-gear parity-refined bound** (`top_machine_2.md` s.10's suggestion). L70 is the
   unrefined version. *Exact statement:* sharpen `2 ceil(L/g)` to a bound that accounts for how
   many of a core gear's cells can lie in one parity class, and ask whether the refined `Lcap`
   equals `F_top`. *Attack:* a core gear's trace splits between the classes as `L/g` up to one,
   and the parity split is forced by `g` odd; the refined count is computable, and the question is
   whether it is ever tight on a loaded wheel.

**Ledger totals: (a) 6 closed, (b) 5 measurements, (c) 2 the conjecture (not the wheels'), (d) 4
genuinely open.**

---

## 7. What is new

**The wheel record has a formula, and it is proved.** `F_top(G) = max { L : min over core phase
vectors of D_L(U) <= t(L) }`, with core `= {g <= L + 1}` and `D` the domino cost. Both directions
proved from two lemmas that are themselves proved: the piece law (L67 - a gear can join the two
ends of the window iff `g <= L + 1`) and the matching lemma (L68 - `D(U)` is exactly the minimum
number of pieces). 0 mismatches against full-period scans on 6,659 gear sets, 5,006 of them
loaded. `top_machine_2.md`'s L31 said `F_top` is a function of `m` and the gears below `F + 1`
and left the function tabulated; this is the function.

**The tail begins at `g > L + 1`, and that is not a convention.** Counting the boundary gear as a
tail gear breaks the rule at 605 of the 6,659 sets, always by exactly one cell - the cell the
ends-joining piece buys. The ends-joining pieces (the end pair `{0, L-1}` at `g = L + 1` and the
wrap pairs at `g = L`) are the only pieces in the machine that cross parity.

**The parity law and its sharp threshold are corollaries, with the `+2` explained.** With an
empty core the cost of `[0, L)` is `2 floor(L/4) + min(L mod 4, 2)`, so the largest affordable
length is `2m - (m mod 2)`. At `q' = 2m + 1` one gear becomes a core gear and offers one
ends-joining piece: at even `m` it leaves runs of `m` and `m - 1` and the cost is `m > m - 1`; at
odd `m` it leaves two runs of `m - 1`, which is even, and the cost is exactly `m - 1`. The whole
of `top_machine_5.md`'s `2m + 3` at odd `m` is that one parity bit. And the record cover is a
free-domino tiling exactly when the parity law holds, which settles the refuted guess "free
tiling iff `q' > F + 1`".

**The moment vanishing is proved, and it is the same covering problem.** `M_k(d)` is
`(-1)^{d-1}` times the top multilinear coefficient of `f^k`, and a term of `f^k` depends on all
`d - 1` variables only if `k` of the pieces `{p-2, p}` cover `[1, d-1]`; so `M_k(d) = 0` below
the covering number. The covering number is `D(d-1)` - **the record's own cost function** - so
the gap census's `r(d)` and the record's domino cost are one object. `top_machine_2.md`'s two
open items were "the vanishing moments" and "L31 as a formula"; they turn out to be the same
combinatorics twice.

**L4 stops being a boundary condition.** The two pieces that could cover interior position 2 at
`d = 4` are exactly the two the census law excludes (they sit in the boundary set `B`), so
`r(4) = infinity` and `N_4 = 0` identically. L26's "unique exception at `d = 4`" is the case
where the covering problem has no solution at all - not a different problem.

**A closed-form upper bound on the record.** `F_top <= max { L : L <= 2 t(L) + sum_{core} 2
ceil(L/g) }`, no search; 0 violations on 37 sets, exact on all 10 free wheels tested.

**Prior art, in a line.** `F_top` is the two-class Jacobsthal function of the gear set
(Jacobsthal 1961); the covering formulation and the parity law were already in this project's
documents 1 and 5 as measurements with a partial proof. The inclusion-exclusion identity behind
L73 is the standard "alternating sum over a Boolean cube kills everything but the top Fourier
coefficient" (Möbius inversion on the Boolean lattice); what is new is the identification of that
top coefficient with a covering requirement on the *same* piece set that defines the record, and
hence the closed form `r(d) = D(d-1)`. Nothing asymptotic is used anywhere.

---

## 8. Verdict

**The loaded record rule is proved.** It is an iff, with no side hypothesis beyond pairwise
coprimality: a window of length `L` is coverable exactly when some phasing of the gears at most
`L + 1` leaves a residue whose domino cost is at most the number of gears above `L + 1`. The two
halves each rest on a proved lemma - what a gear can show in a window (L67) and what a set of
free dominoes costs (L68) - and the verification is 0 mismatches over 6,659 scanned gear sets
(5,006 loaded), the 13 known records, and all 8,855 triples and quadruples. The wheel record is
therefore no longer a table: it is a formula, and the only thing left about it is the *cost* of
evaluating the minimisation.

**The free/loaded boundary is a corollary.** The parity law is the empty-core case of the same
inequality, and the sharp threshold `2m + 1` / `2m + 3` is what happens when exactly one gear
crosses into the core: the single ends-joining piece it offers leaves two step-2 runs, and
whether they tile is decided by the parity of `m - 1`. Odd `m` needs one more gear-size because
`m - 1` is even there.

**The moment vanishing is proved.** `M_k(d) = 0` for `k < r(d)` because `M_k` is the top
coefficient of a `k`-fold product of piece indicators, and below the covering number no product
touches every variable. `r(d)` is the free-boundary domino cost `D(d-1)`, so the census's
covering number and the record's cost function are the same function evaluated at different
arguments; `d = 4` is the case where the covering is impossible, which is L4. Verified exactly to
`d = 26`. The one arithmetic step still missing is that the top coefficient does not cancel at
`k = r(d)` (measured, 0 exceptions of 24).

**The wheels' ledger.** Six items closed, five are measurements or compute cutoffs, two are the
twin conjecture wearing a range coordinate and are not the wheels' business, and four are
genuinely open - the complexity of the minimisation (the natural child branch), the
non-cancellation at `k = r(d)`, L22 in the kernel (shape now supplied, three lemmas verified
separately), and a parity-refined capacity bound.

No interpretation against the twin conjecture is offered; no clutch, no motor.

---

## 9. Scorecard, filled

| # | Prediction | Result |
|---|---|---|
| P1 | the loaded record rule as an iff, proved both directions | **held**: L69, proof in section 4 |
| P2 | matching lemma: min pieces `= D(U)`, exhaustive to `L = 14` | **held**: L68 proved; 32,766 subsets, 0 mismatches |
| P3 | the tail hypothesis `g > L + 1` is sharp; the wrong boundary breaks the rule (>= 3 instances) | **held in substance, the named instances refuted**: 605 of 6,659 sets break, every one by exactly 1 - but the three sets named in advance (`{9,11,13,17}`, `{13,17,19,23,29,31}`, `{17,19,23,29,31,37,41,43}`) are **not** among them. There `q' = F + 1`, so the boundary gear is the *smallest*, and the record is already reachable by a free tiling; the sets that break are those where the boundary gear is a small gear that the cover actually needs, e.g. `{7,9,11}`, `{7,11,13}`, all with `F = 6` and the gear 7 at `L + 1` |
| P4 | 0 mismatches: scan, 13 known records, 1,540 triples, 7,315 quadruples | **held**: 0 of 6,659; 0 of 13; 0 of 8,855. The scan cap was lowered from the pre-registered `2 x 10^8` to `2.4 x 10^7` so that all 6,659 sets could be scanned inside the lane's budget |
| P5 | `D(L) = 2 floor(L/4) + min(L mod 4, 2)`; `0,1,2,2,2,3,4,4,4,5,6,6,6` | **held** exactly, `L = 0..399` |
| P6 | `max{L : D(L) <= m} = 2m - (m mod 2)`, `m = 1..200` | **held**, 0 mismatches |
| P7 | threshold `2m + 1` / `2m + 3` derived, 0 exceptions | **held**, 8 values of `m`, with the mechanism computed cell by cell |
| P8 | `M_k(d) = 0` for `k < r(d)`, proved by the covering argument | **held**: L73 proved; 0 failures `d = 2..26` |
| P9 | `r(4) = infinity`, `M_k(4) = 0` for all `k`, `d = 4` not an exception | **held**: 0 at `k = 0..8`, and the reason is that `J_2`, `J_d` are the excluded pieces |
| P10 | `r(d)` table to `d = 24`; `M_{r(d)} != 0`; `18, 96, 24, 24, 480, 6480, 1440, 720` | **held**, and extended to `d = 26`; all eight multiplicities reproduced |
| P11 | L22 as three lemmas, each verified on 5 wheels | **held**: K1, K2, K3 and the assembled law, 0 mismatches each |
| P12 | ledger: >= 3 closed, >= 2 measurements, >= 1 conjecture, rest open | **held**: 6 / 5 / 2 / 4 |

---

## 10. Holds without exception (the count)

| statement | count | exceptions |
|---|---|---|
| the matching lemma `min #pieces = D(U)` | 32,766 subsets, `L = 1..14` | **0** |
| the loaded record rule against a full-period scan | 6,659 gear sets (5,006 loaded) | **0** |
| the rule against the 13 independently known records | 13 | **0** |
| the rule against the exhaustive triple/quadruple tables | 8,855 | **0** |
| the wrong boundary (`core = {g <= L}`) fails, always by exactly 1 | 605 of 6,659 (of the 1,422 where `F + 1` is a gear) | **0** other sizes of failure |
| `D(L)` closed form | `L = 0..399` | **0** |
| the parity law from the empty-core cost | `m = 1..200` | **0** |
| the sharp threshold `2m+1` / `2m+3` | 8 values of `m`, 3 gear sets each | **0** |
| the capacity bound `F_top <= Lcap` | 37 gear sets | **0** (exact at 10) |
| `M_k(d) = 0` for `k < r(d)` | `d = 2..26` | **0** |
| `M_{r(d)}(d) != 0` | `d = 2..26`, `d != 4` | **0** |
| `r(d) = D(d-1)` against exact set cover | `d = 2..20`, `d != 4` | **0** |
| the eight published record multiplicities from `M_{r(d)}` | 8 | **0** |
| K1, K2, K3 and the assembled L22 | 5 wheels x 10 lengths, 5,115 subset terms | **0** each |

---

## 11. Dead ends

- **The additive form `F_core + 2t - defect`.** Already refuted in `top_machine_4.md` (L54);
  this branch says why in one line: a core gear inside a longer window contributes
  `2 ceil(L/g)` cells, and the additive form prices it at 2. Not re-entered.
- **Making the minimisation cheap by pruning alone.** The exact engine here memoises on
  `(gear index, covered mask)` and prunes with the capacity bound, which is enough for cores of
  eight gears at `L = 32` but is still exponential; no polynomial route was found in this branch.
  What survived is the closed-form upper bound L70 and the exact statement of the open question
  (ledger (d) item 1).
- **Proving `M_{r(d)} != 0` by the same top-coefficient argument.** The argument that kills
  `k < r(d)` says nothing at `k = r(d)`: there the terms are `r(d)!` times a signed sum over
  minimal covers, and signs can in principle cancel. Measured non-zero at every `d <= 26`;
  left as a stated open item rather than asserted.
- **Reading `r(d)` off the census by a transfer matrix.** Not re-entered - `top_machine_2.md`
  already showed the DP has `2^(d-1)` states. What this branch does instead is compute `r(d)` in
  closed form (L74) and use it to say how many of `N_d`'s moment terms are nonzero, which is a
  statement about the census's *shape*, not a cheaper evaluation of it.
