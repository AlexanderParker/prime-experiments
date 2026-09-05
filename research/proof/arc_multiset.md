# Branch 5d.ii.i.a - THE ARC MULTISET

Parent: node R2.b (`research/proof/gear_count.md`), whose closing observation was that the only
new handle it found is the ARC: a gear beyond its umbrella is a bare domino of size
`a_g = 2 u_g = (g -+ 1)/3`, its size invisible, so covering is choosing a multiset of arcs; and
`3 a_g = g -+ 1` makes twin gears share an arc, so the real machine `{5..q}` carries only
`pi(q) - 2 - pi_2(q)` distinct arcs and (r48's reading) is a worse coverer than an adversary of
the same gear count because of it. The wall's section 5d records the arc multiset as the one
which-residues handle where the real machine sits on the good side, and leaves one open lemma:
`A(K) < (p_{K+1}^2 - 1)/6`.

Scripts in `research/anchor235/r50/` (prefix `arc_`); result outputs, untracked, in
`research/anchor235/r50/results/`. Every number this document relies on is written here.

---

## 0. Pre-registered (written before this round's computation, gates excepted)

### 0.1 Objects, stated exactly

Machine `M = {5..q}`; gears the primes `5 <= g <= q`; `u_g = 6^{-1} (mod g)`; gear `g` strikes
column `k` iff `k = +-u_g (mod g)`. Separation `d_g = 2 u_g = 3^{-1} (mod g)`; **short arc**
`a_g = min(d_g, g - d_g)`, so `3 a_g = g -+ 1`; **long arc** `g - a_g`. A **run of length `L`**
is `L` consecutive struck columns; its **span** is `S = L + 1` (the two bounding columns of a
maximal run are open). `F(A)` = the smallest `L` no phase assignment of the gear set `A` can
cover = the record span of `A` over its own full period (r48 `cover_core.py`, by CRT). `A(K)` =
`max{F(A) : |A| = K}` over all sets of `K` primes `>= 5`.

Two elementary facts used throughout, both one line from `3 a_g = g -+ 1`:

* **every short arc is even** (`g -+ 1 = 3 a_g` and `g` odd force `6 | 3 a_g`), and
* **the long arc is `2 a_g -+ 1`**, so writing `g = 3a + e` with `e = +-1`, a gear's complete
  arc content is the pair `(a, 2a + e)`. Twin gears share `a` and differ in `e`, hence their
  LONG arcs differ by 2. (r48 says twin gears are interchangeable; that is true only where the
  long arc is invisible.)

**The split at level `L`.** A gear is **big at `L`** if `g - a_g >= L` and **small at `L`**
otherwise. This round's tool is:

> **Type lemma (proved, elementary; r48's domino lemma made exact).** If `g - a_g >= L` then
> `g > L - 1`, each residue class of `g` meets the run at most once, and consecutive teeth of `g`
> are at distance `a_g` on one side and `g - a_g >= L` on the other. So the family of subsets of
> `{0..L-1}` the gear can realise is exactly
> `{} u {{i, i+a} : i + a <= L-1} u {{i} : i < a or i >= L - a}` with `a = a_g` - a function of
> `a_g` alone.

Hence at level `L` the infinite prime pool collapses to a finite item list: every prime `p` with
`p - a_p <= L-1` (concrete, multiplicity 1); for each even `a < L` the type `domino(a)` with
multiplicity the number of primes among `3a-1, 3a+1` that are prime and big at `L` (0, 1 or 2);
and one type `single` (`arc >= L`) with multiplicity `K`. **An `A(K)` computed over that item
list is exhaustive over ALL primes, not over a truncated pool** - strictly stronger than r48's
"exhaustive over the pool 5..149".

**The de-twinned machine.** The brief's rule - walk the gears upward and replace any gear whose
arc is already used (the larger member of a twin pair) by the next prime with a new arc - carried
out on `{5..q}` gives exactly `D(K)` = one gear per arc, the smallest prime realising each of the
`K` smallest arcs: `5, 11, 17, 23, 29, 37, 41, 47, 53, 59, ...` (arcs `2, 4, 6, ...`).

### 0.2 The theory

The adversary's currency is the arc, not the prime. If that is the whole story then (i) the
optimal `K`-set is decided by its arc multiset alone, (ii) the optimal arc multiset is the
smallest distinct arcs whenever the gears are big, and (iii) the real machine's loss against the
adversary is exactly the arcs it duplicates - one wasted domino per twin pair - so the loss grows
with `pi_2(q)`. The counter-theory, already visible in r48's own caution, is that arcs decide
only in the domino regime: the small gears are periodic tilers, not dominoes, and the arc picture
applies only above a threshold.

### 0.3 Predictions and what refutes each

* **P1.** `A(7) = 37` and `A(8) = 45` exactly (r48's certified lower bounds are tight). REFUTED
  by any exact value above those.
* **P2.** For `K = 4..8` the optimal set has all short arcs distinct. (Pre-registered as expected
  to fail from `K = 7`, r48 having seen twin pairs in the best subsets of `{5..31}`.)
* **P3.** The optimal arc multiset is the `K` smallest realisable arcs `2, 4, ..., 2K`.
  Pre-registered FALSE (`{5,7,11,23,29}` skips arc 6); the branch's job is to say what replaces
  it.
* **P4.** The matching identity: `G` covers `L` iff some phase assignment of `G`'s small part
  leaves holes the big part can match, one big gear per hole or per hole-pair at distance exactly
  its arc. Pre-registered TRUE; the brief's caveat (a big gear using `g - a_g` instead)
  pre-registered VOID, since `g - a_g >= L > L-1` is the definition of big.
* **P5.** The twin tax grows with `pi_2(q)`: `F_real/A(K)` falls and `F_real/F_D` falls as
  `pi_2` grows. REFUTED if either is flat, or if `F_D < F_real` (de-twinning HURTS).
* **P6.** The refined matching bound is tight by construction; relaxing "which arcs" to "how many
  holes" (the level-1 form) is pre-registered NOT enough to reach `A(K)`, because a count is what
  face A forbids.
* **P7.** The record's letters are the arcs of the new gear, at every rung, by the record law.
  Cited, not re-derived.
* **P8.** The increment `F(q') - F(q)` at a twin rung is systematically smaller relative to `q'`
  than at a non-twin rung. REFUTED if the two sets of `increment/q'` interleave.
* **P9.** At fixed `pi(q)`, `F` decreases as `pi_2(q)` increases (every duplicated arc is a
  wasted domino). REFUTED by one family where duplicating an arc raises `F`.

### 0.4 Scorecard

| # | prediction | verdict | evidence |
|---|---|---|---|
| P1 | `A(7) = 37`, `A(8) = 45` | **CONFIRMED, both exact** | MILP infeasibility over all primes; r48's lower bounds were tight (R1) |
| P2 | optimal arcs distinct | **REFUTED at every `K >= 2`** | every optimum contains the twin pair `5, 7`; the `K = 7` optimum `{5,7,11,13,17,19,31}` has three duplicated arcs (R2) |
| P3 | optimal arcs `2, 4, ..., 2K` | **REFUTED**, as pre-registered | the `K` smallest arcs are `D(K)`, whose record is 1.1-1.7 times SHORTER than the real machine's (R2, R4) |
| P4 | matching identity | **CONFIRMED exactly**, caveat void | 223 (gear set, level) instances, 0 disagreements, every one with a non-empty big part (R3) |
| P5 | twin tax grows with `pi_2` | **REFUTED and inverted** | `F_real/A(K)` RISES 0.69 -> 1.00; de-twinning LOWERS the record at every rung, `F_real/F_D` = 1.10 to 1.70 (R4) |
| P6 | level-1 (count only) insufficient | **REFUTED** | the count-only relaxation `B1` equals `A(K)` at `K = 4, 7` and exceeds it by at most 5 (11%) to `K = 8`; parity and prime-realisability of arcs are worth 0-2 columns (R7) |
| P7 | letters are the new gear's arc | **KNOWN RESULT**, cited in one line; what is measured is WHICH letter (R8) |
| P8 | twin rungs differ | **REFUTED** | twin `inc/q'` in [0.279, 0.484] sits strictly INSIDE non-twin [0.073, 0.811]; means 0.374 vs 0.364 (R8) |
| P9 | `F` decreasing in duplicated arcs | **REFUTED and inverted** | mean `F` is monotone INCREASING in the number of duplicated arcs at `K = 3..7`, no exception (R5) |

---

## 1. Setup (exact ranges, tools, gates)

Scripts in `research/anchor235/r50/`, `uv run python <script>` from the repository root,
numpy/scipy only where stated, at most 3 worker processes. (`arc_adv.py` and `arc_a7.py` are the
superseded DFS drivers, kept because they produced the `K <= 6` gate.)

| script | what it computes | range | cost |
|---|---|---|---|
| `arc_core.py` | the type reduction; item lists; an exhaustive DFS cover search over item multisets | any `L, K` | gate `A(K)`, `K <= 6` |
| `arc_milp.py` | `A(K)` by exact MILP feasibility (HiGHS via `scipy.optimize.milp`) over the same item list | `K = 4..13` | seconds to a minute per level |
| `arc_match.py` | the domino-matching identity against the direct search | 15 gear sets, every level `1..F` | 2 min |
| `arc_tax.py` | the ladder: real machine, `A(K)`, de-twinned `D(K)` | `K = 4..9` | 1 min |
| `arc_famtax.py`, `arc_fam7.py` | `F` at fixed gear count by number of duplicated arcs, exhaustive | `K = 3..7`, pools to 61 | 2 min |
| `arc_signature.py` | is `F` a function of the short-arc multiset? exhaustive | `K = 3..5`, pools to 61 | 1 min |
| `arc_relax.py` | the relaxation ladder `B1 >= B2 >= B3 >= A` by MILP | `K = 3..8` | 30 s |
| `arc_dict.py` | the level-2 hole-distance dictionary of the small part | 4 small parts, `L` to 22 | 5 min |
| `arc_record.py` | the record word and letters at each rung; the increment table | rungs `5 -> 7` .. `29 -> 31` | 10 s |
| `arc_sixth.py` | the dictionary of `{5..17}` to `L = 32` and the sixth-gear sweep it predicts | primes 23..89 | 3 min |
| `arc_crosscheck.py` | independent re-run of a MILP infeasibility by brute force with the r48 tool | `K = 7`, `L = 37`, primes to 79 | 4 min |

**Gates passed.** (i) `arc_core` reproduces the r48 F ladder `2, 5, 7, 11, 18, 25, 34, 43, 58`
at `{5} .. {5..31}` through the unchanged r48 `cover_core`. (ii) The type-reduced DFS reproduces
r48's exhaustive `A(1..6) = 2, 5, 7, 16, 22, 28`. (iii) The MILP reproduces the same
`A(4), A(5), A(6) = 16, 22, 28` in under a second each, agreeing with the DFS on both the cover
and the no-cover side. (iv) The `A(9)` witness was re-tested with the r48 tool independently:
`coverable(67, {5,7,11,13,17,23,31,37,47}) = True`, `coverable(68, ...) = False`, so that set's
own record is exactly 68. (v) The record words extracted at m29 and m31 reproduce the wall's
recorded decompositions B3 (`10 + 10 + 23` and `23 + 10 + 25`) exactly. (vi) The MILP's
infeasibility claim at `K = 7`, `L = 37` was re-run independently by brute force with the r48
tool over every 7-subset of the primes 5..79 (77,520 subsets, 212 s): **0 sets cover 37
columns**, in agreement. The same brute force at `K = 8`, `L = 45` over the primes 5..89
(319,770 subsets) was launched and stopped incomplete after 65 minutes; it is redundant given
(vi) and (vii) and is not relied on anywhere here. (vii) The `A(7)` and `A(8)` optima were
re-tested with the r48 tool:
`{5,7,11,13,17,19,31}` covers 36 columns and not 37, `{5,7,11,13,19,29,31,83}` covers 44 and not
45 - their own records are exactly 37 and 45.

---

## 2. Results

### R1. The adversarial ladder, exact and far above the old lower bounds

`A(K)` is now exact and exhaustive over ALL primes `>= 5` (type reduction + MILP infeasibility),
where r48 had exact values only to `K = 6` and hill-climb lower bounds above.

    K            1     2     3     4     5     6     7     8     9    10    11    12
    A(K)         2     5     7    16    22    28    37    45    68    88   101   115
    r48          2     5     7    16    22    28  >=37  >=45  >=58     -     -     -
    F({5..p_K})  2     5     7    11    18    25    34    43    58    88    91   103
    A/F       1.00  1.00  1.00  1.45  1.22  1.12  1.09  1.05  1.17  1.00  1.11  1.12
    p_{K+1}      7    11    13    17    19    23    29    31    37    41    43    47
    W            8    20    28    48    60    88   140   160   228   280   308   368
    A/W       0.25  0.25  0.25 0.333 0.367 0.318 0.264 0.281 0.298 0.314 0.328 0.313

`W = (p_{K+1}^2 - 1)/6` is the window the open lemma asks `A(K)` to stay below. At `K = 13` the
search was still finding covers when the lane closed and was stopped: `A(13) >= 137` by exhibited
covers (each a positive certificate), against `F({5..47}) = 118` and `W(53) = 468`, so
`A/F >= 1.16` and `A/W >= 0.29` there too.

Three readings.

* **The open lemma `A(K) < (p_{K+1}^2 - 1)/6` holds at every `K <= 12` with a margin of a factor
  2.7 to 3.8, and the ratio `A/W` is FLAT** (0.264 to 0.367 over `K = 4..12`, no trend). The real
  machine's own `F/W` is 0.23-0.31 over the same range (margin 3.2 to 4.4). So the adversarial
  statement is strictly stronger than the root, but only by the difference between a factor 3.2
  and a factor 2.7 of slack - it is not a harder statement by any large factor.
* **r48's ratio `A/F` was NOT falling to 1.** It reads 1.45, 1.22, 1.12, 1.09, 1.05, then rises
  to 1.17 at `K = 9`, is exactly 1.00 at `K = 10`, then 1.11 and 1.12. r48 read the first
  three terms as a falling sequence and conjectured the covering form converges to the root;
  measured further, it oscillates. **At `K = 10` the real machine `{5..37}` is an optimal 10-gear
  blocker**: no ten primes anywhere block more than 87 consecutive columns, and `{5..37}` blocks
  exactly 87.
* **`A(9) = 68` is 10 above r48's certified lower bound 58**, and the optimal set is
  `{5, 7, 11, 13, 17, 23, 31, 37, 47}` - it drops 19 and 29 from the real machine and reaches up
  to 37 and 47. Verified twice: MILP infeasibility at `L = 68`, and the r48 tool run on that set
  alone (`F = 68` exactly).

### R2. The optimal arc multisets - the arcs are never distinct, and never the smallest

The optimal set at each `K` (one representative; ties exist at `K <= 6`), with its short-arc
multiset. `dom(a)` means the item is a bare domino of arc `a`, realised by the prime `3a -+ 1`
that is big at that level.

    K   optimal set                                arcs                        duplicated arcs
    4   {5,7,11,17} or {5,7,11,19}                 2,2,4,6                          1
    5   {5,7,11,23,29} or {5,7,11,23,31}           2,2,4,8,10                       1
    6   {5,7,11,17,23,37}                          2,2,4,6,8,12                     1
        {5,7,11,13,19,47} (ties)                   2,2,4,4,6,16                     2
    7   {5,7,11,13,17,19,31}                       2,2,4,4,6,6,10                   3
    8   {5,7,11,13,19,29,31} + dom(28) = 83        2,2,4,4,6,10,10,28               3
    9   {5,7,11,13,17,23,31,37,47}                 2,2,4,4,6,8,10,12,16             2
   10   {5,7,11,13,17,19,23,29,37,79}              2,2,4,4,6,6,8,10,12,26           3
        {5..37} (ties, = the real machine)         2,2,4,4,6,6,8,10,10,12           4

* **The optimal arcs are never distinct.** Every optimum from `K = 2` up contains the twin pair
  `5, 7`, i.e. the arc 2 twice; from `K = 6` up the optima carry two or three duplicated arcs.
  P2 refuted at every `K`.
* **The optimal arcs are never the smallest available.** The `K` smallest arcs are exactly
  `D(K)`, whose record is far SHORTER (R4). The optimum takes the smallest GEARS, and the
  smallest gears are exactly the ones that duplicate arcs, because a duplicated arc is a twin
  pair `3a-1, 3a+1` and a twin pair is the cheapest way to buy two gears.
* **Which gear realises each arc.** Where a twin pair exists the optimum usually takes both
  members (`5,7`; `11,13`; `17,19`; `29,31`); where only one member is prime the choice is
  forced (`23`, `37`, `47`, `79`, `83`). Where a single member of a twin pair is taken, the
  choice matters and is measured in R6.

### R3. The domino-matching identity: exact, with the caveat void

For a gear set `G` at level `L`, split `small = {g : g - a_g <= L-1}`, `big = {g : g - a_g >= L}`.

> **Matching form.** `G` covers `L` columns iff some phase assignment of `small` leaves a hole
> set `H` with `maxpairs(H, arcs(big)) >= |H| - |big|`, where `maxpairs` is the largest number of
> disjoint hole pairs whose distances form a sub-multiset of the big gears' arcs.
>
> A big gear can ALWAYS take one hole on its own: a gear of arc `a` covers hole `i` by the legal
> singleton `{i}` when `i < a`, and otherwise (`i >= a`) by the pair `{i-a, i}`, whose left
> column is already covered. So only the PAIRS are constrained, and they are constrained by the
> arc multiset alone.

Checked against the direct exhaustive search (r48 `cover_core`) on 15 gear sets - the optimal
adversarial sets at `K = 4, 5, 6`, the real machines m11..m19, the initial segment, the
de-twinned machines, and two big-gear-heavy sets - at **every** level `L = 1..F(G)` at which the
big part is non-empty:

    223 (gear set, level) instances, 0 disagreements.

The brief's caveat - a big gear placing its two strikes at distance `g - a_g` if that fits - is
**void by definition**: `big` means `g - a_g >= L > L - 1`, and `L - 1` is the largest distance
inside the run. Checked: 0 instances out of 223 where a big gear's long arc fits.

This is the exact statement of r48's domino lemma, and it is what makes the type reduction of
section 0.1, and therefore the exhaustive-over-all-primes `A(K)`, legitimate.

### R4. The twin tax, quantified - and it has the opposite sign

    K    q   pi_2  distinct arcs  F_real   A(K)  F_real/A   D(K)                    F_D   F_real/F_D
    4   13     2        2           11      16     0.688   {5,11,17,23}             10      1.100
    5   17     2        3           18      22     0.818   {5,11,17,23,29}          14      1.286
    6   19     3        3           25      28     0.893   + 37                     17      1.471
    7   23     3        4           34      37     0.919   + 41                     20      1.700
    8   29     3        5           43      45     0.956   + 47                     29      1.483
    9   31     4        5           58      68     0.853   + 53                     35      1.657
   10   37     4        6           88      88     1.000
   11   41     4        7           91     101     0.901

`F_D` is exact (the r48 cover search on the de-twinned set) for `K = 4..9`; at `K = 10, 11` the
de-twinned sets `{5,11,...,59}` and `{5,11,...,67}` did not settle inside the lane's memory
budget and are left blank - the trend over the six exact rows is already monotone.

* **`F_real/A(K)` rises, it does not fall**: 0.69, 0.82, 0.89, 0.92, 0.96, 0.85, 1.00, 0.90. It
  is not a fixed fraction and it does not track `pi_2(q)` (`pi_2 = 2` gives 0.69 and 0.82;
  `pi_2 = 4` gives 0.85, 1.00, 0.90). P5's first half refuted.
* **De-twinning LOWERS the record at every rung**, by a factor rising from 1.10 to 1.70. Removing
  the duplicated arcs - which is exactly what the brief's de-twinning rule does - makes the
  machine a strictly worse blocker, because the gears it removes (`7, 13, 19, 31, ...`) are the
  small tilers and the gears it puts in their place (`17, 23, 29, 37, ...`) are larger. P5's
  second half refuted, with the sign reversed.

So **the real machine's deficit against the adversary is not the twin tax**. The duplicated arcs
are not what costs it; being confined to an initial segment of primes is. At `K = 9` the
adversary's gain comes entirely from dropping 19 and 29 and reaching up to 37 and 47 - a change
that raises, not lowers, the number of distinct arcs used, and lowers the number of duplicates
from 3 to 2.

### R5. Duplicated arcs HELP - exhaustive, no exception

`F` over every `K`-subset of a prime pool, grouped by the number of duplicated arcs (= the number
of twin pairs the set contains).

    K=3, primes 5..61 (560 subsets)        K=4, primes 5..61 (1820 subsets)
      dup  sets   max F  mean F              dup  sets   max F  mean F
        0   476       6    4.46                0  1289      10    6.27
        1    84       7    4.64                1   516      16    6.97
                                               2    15      15    7.80

    K=5, primes 5..47 (1287 subsets)       K=6, primes 5..43 (924 subsets)
      dup  sets   max F  mean F              dup  sets   max F  mean F
        0   552      14    9.24                0   144      18   14.08
        1   645      22   10.38                1   520      28   15.10
        2    90      21   11.46                2   250      28   16.28
                                               3    10      25   17.40

    K=7, primes 5..43 (792 subsets)
      dup  sets   max F  mean F   argmax
        0    32      20   19.75   {7,11,19,23,31,37,43}
        1   320      35   21.22   {5,7,11,17,23,31,37}
        2   380      37   22.23   {5,7,11,13,17,31,37}
        3    60      37   23.20   {5,7,11,13,17,19,31}

**The mean record is monotone increasing in the number of duplicated arcs at every `K`, with no
exception** (5 gear counts, 16 duplication classes, 5,383 gear sets). The maximum is attained at 1
to 3 duplicates at every `K` and never at 0; at `K = 7` the global optimum `A(7) = 37` is reached
only at 2 and 3 duplicates, and a set with NO duplicated arc reaches only 20 against 37.

Mechanism, in one line: a duplicated arc is a twin pair `3a-1, 3a+1`, and a twin pair is the
smallest possible pair of gears carrying that arc, so a set that duplicates arcs is a set of
small gears, and small gears are periodic tilers whose strikes are dense. The duplication is not
a waste; it is a symptom of buying cheap.

This inverts the wall's 5d reading. Arc duplication is on the WRONG side for the conjecture: it
lengthens the record. What keeps the real machine below the adversary is something else.

### R6. The short arc is not the whole of a gear - the sign is a covering coordinate

Group every `K`-subset by its short-arc MULTISET and measure the spread of `F` inside each group.

    K   pool   subsets  arc multisets  realised >1 way  F constant on them  max spread
    3   5..61      560            174              146           146/146             0
    4   5..61     1820            441              389           271/389             3
    5   5..47     1287            291              256           146/256             5

* At `K = 3` the short-arc multiset determines `F` exactly - 146 out of 146 multi-realised
  multisets, no exception. That is the domino regime: at `K = 3` the record is at most 6 columns
  and every gear above 11 is big, so only `a_g` is visible.
* From `K = 4` it does not. The extreme case at `K = 5`: the arc multiset `(2,4,6,10,16)` is
  realised by `{7,11,17,29,47}` with `F = 9` and by `{5,13,19,31,47}` with `F = 14` - a spread of
  5 on a record of 14, 36% of the value.
* **The two extremes are sign-complementary.** In all 16 largest-spread rows reported (8 at
  `K = 4`, 8 at `K = 5`) the minimising and maximising sets are obtained from one another by
  swapping each gear for the other member of its twin pair wherever that partner is prime
  (`7<->5`, `11<->13`, `17<->19`, `29<->31`, `41<->43`, `59<->61`); a gear whose partner is not
  prime (`23`, `37`, `47`) is the same in both. Since `g = 3a + e` with
  `e = +-1` and the long arc is `2a + e`, the swap changes only the LONG arcs, each by 2.

So the covering coordinates of a gear are `(a, e)`: the short arc `a`, which is all that survives
in the domino regime, and the sign `e`, which decides the long arc and is worth up to 36% of the
record once the gear is a tiler. r48's "two gears with the same short arc are interchangeable"
holds exactly in the regime it was proved in, and fails everywhere else.

### R7. The refined bound: which arcs is worth nothing, how many is worth everything

Relax what the adversary may buy for a big gear, keeping the small gears real. `B1`: any two
columns of the run, or one (hole COUNT only). `B2`: any two columns at an even distance (every
arc is even). `B3`: any two columns at a distance `a` with `3a -+ 1` prime, unlimited
multiplicity. `A`: the truth (arcs of primes actually big at that level, multiplicity 1 or 2).
`B1 >= B2 >= B3 >= A`, all exact, all by the same MILP.

    K     A(K)    B3    B2    B1    B3/A   B2/A   B1/A
    3        7    10    10    11    1.43   1.43   1.57
    4       16    16    16    16    1.00   1.00   1.00
    5       22    23    23    23    1.05   1.05   1.05
    6       28    28    28    30    1.00   1.00   1.07
    7       37    37    37    37    1.00   1.00   1.00
    8       45    50    50    50    1.11   1.11   1.11

`B2 = B3` at every `K` and `B1 = B2` at four of six. **Telling the adversary which distances his
dominoes may span - the entire arc structure, parity included - is worth 0 to 2 columns.** What
binds is the NUMBER of dominoes and how many copies of one arc the primes offer (the `B3 > A`
rows at `K = 3, 5, 8` are multiplicity, not arc identity).

Since `B1` is within 11% of `A` at every `K` from 4 up (1.00, 1.05, 1.07, 1.00, 1.11), and `B1`
has no arc information in it at all, the whole of `A(K)` is decided by the SMALL part - by how
few columns `k` tiler gears can leave unstruck in a run of `L`. The arc multiset is not the
lever. (At `K = 3`, where every gear above 11 is big, the relaxation does bite: `B1 = 11`
against `A = 7`, the one regime in which the arcs decide.)

**The level-2 dictionary of the small part** (`arc_dict.py`), for the initial segments, exact by
enumeration of all phase assignments: at each `L`, the minimum number of holes and, when that
minimum is 2, the distances those two holes can be at.

    small part {5,7}          L:  5..6 -> 1 hole;  7 -> {1,2,5};  8 -> {1,2};  9 -> {1,2};
                             10 -> {1};  11+ -> 3 holes
    small part {5,7,11}       L:  7..10 -> 1 hole; 11 -> {1,2,3,5,6}; 12 -> {1,3,5,6};
                             13 -> {1,3,6}; 14 -> {1,6}; 15 -> {6}; 16+ -> 3 holes
    small part {5,7,11,13}    L: 11..15 -> 1 hole; 16 -> {1,2,5,6,7,8,11}; 17 -> same;
                             18, 19, 20 -> {5}
    small part {5,7,11,13,17} L: to 17 -> 0 holes; 18..24 -> 1 hole;
                             25 -> {2,5,7,10,12,18}; 26 -> {5,7,18}; 27 -> {5,7,18};
                             28+ -> 3 holes

Two exact mechanisms fall out, and they are which-residues statements, not counts.

* **The dictionary narrows to a single distance as `L` rises.** `{5,7,11}` at `L = 15` can leave
  its two holes at distance 6 and at no other distance (2 phase assignments realise it). This is
  the sharp form of r48's `K = 4` mechanism: with `{5,7,11}` at a hole-minimal phasing the fourth
  gear must supply distance 6, and among all primes only 17 and 19 have an arc equal to 6 - which
  is why r48 measured `F({5,7,11,g}) = 16` at `g = 17, 19` and 11 at every other `g`, and why the
  gear's size is irrelevant.
* **The parity obstruction.** Every short arc is even, so a hole pair at an ODD distance can
  never be taken by one big gear. `{5,7}` at `L = 10` can only leave its holes at distance 1, and
  `{5,7,11,13}` at `L = 18, 19, 20` only at distance 5. In both cases the next gear cannot be a
  domino at all - it must be small enough to tile. **`A(3) = 7` is a complete proof of this
  shape**: at `L = 7` every gear except 5 and 7 is big (`11 - 4 = 7 > 6`), `{5,7}` leaves at best
  two holes, at distance 1, 2 or 5 only, and a big gear can join two holes only at an even
  distance that some prime realises - which leaves distance 2 alone, offered by 5 and 7 and by no
  other prime, and both are already spent. So no three primes block 7 columns.
* **The dictionary PREDICTS the next gear, one level up.** At `L = 27` the small part `{5..17}`
  can leave its two holes only at distances 5, 7 or 18; 5 and 7 are odd, so the sixth gear must
  span 18, and the only prime with arc 18 is 53 (`3*18 - 1 = 53`; `3*18 + 1 = 55` is not prime).
  Tested by sweeping the sixth gear over every prime 23 to 89:

        g     23  29  31  37  41  43  47  53  59  61  67  71  73  79  83  89
        a_g    8  10  10  12  14  14  16  18  20  20  22  24  24  26  28  30
        F     25  26  26  26  25  25  25  28  25  25  25  25  25  25  25  25

  `F({5,7,11,13,17,53}) = 28 = A(6)`, so `{5..17} u {53}` is an optimal six-gear machine, and the
  gear that makes it is picked out by an arc the dictionary named in advance. This is r48's
  `K = 4` mechanism repeated one level up, and it is the one place in the branch where the arc
  multiset does real work.

### R8. The record's letters, and the twin rungs

The record run of `M' = {5..q'}` at each rung, extracted from a witness cover: the columns of the
run that no gear of `M = {5..q}` strikes are the interior `M`-openings `q'` must strike, the word
is the `M`-gaps around them, and the letters are the differences of those interior openings
(`docs/proofs/05` (F): the letters of `g` are `a = 2u_g` and `b = g - 2u_g`).

    q'   a_q'  b_q'   F    word (M-gaps of the record run)   letters
     7      2     5    5   2 + 2 + 1                          2      = a
    11      4     7    7   5 + 2                              -
    13      4     9   11   5 + 6                              -
    17      6    11   18   7 + 6 + 5                          6      = a
    19      6    13   25   7 + 18                             -
    23      8    15   34   7 + 15 + 8 + 4                     15, 8  = b, a
    29     10    19   43   10 + 10 + 23                       10     = a
    31     10    21   58   25 + 10 + 23                       10     = a

The m29 and m31 words reproduce the wall's B3 decompositions exactly (`10 + 10 + 23`,
`23 + 10 + 25`), which gates the extraction. Every letter at every rung is `a_{q'}` or `b_{q'}` -
the word grammar, a recorded theorem, cited not re-derived. What is measured and new: **the
record uses the SHORT arc at every rung that has a letter at all, and the long arc appears only
where the word has three interior openings (m23), where the grammar's alternation forces it.**

The twin rung 29 -> 31: `a_29 = a_31 = 10`, and both records carry a single letter 10. The m31
word `25 + 10 + 23` is the m29 word `10 + 10 + 23` with the leading 10 replaced by 25 - the same
letter, the same right flank, a longer left flank. So the twin rung does gain a gear without
gaining an arc, and the record does reuse the letter-sized piece. But that does not make the
increment special:

    q -> q'   twin   F(q)  F(q')  inc   inc/q'    q -> q'   twin  F(q) F(q')  inc  inc/q'
     5 ->  7   TWIN     2      5    3    0.429     7 -> 11          5     7     2   0.182
    11 -> 13   TWIN     7     11    4    0.308    13 -> 17         11    18     7   0.412
    17 -> 19   TWIN    18     25    7    0.368    19 -> 23         25    34     9   0.391
    29 -> 31   TWIN    43     58   15    0.484    23 -> 29         34    43     9   0.310
    41 -> 43   TWIN    91    103   12    0.279    31 -> 37         58    88    30   0.811
                                                  37 -> 41         88    91     3   0.073
                                                  43 -> 47        103   118    15   0.319
                                                  47 -> 53        118   145    27   0.509
                                                  53 -> 59        145   161    16   0.271

Twin rungs: `inc/q'` in [0.279, 0.484], mean 0.374. Non-twin: [0.073, 0.811], mean 0.364.
**The twin range sits strictly inside the non-twin range** - the two are not separated, the means
are indistinguishable, and P8 is refuted. The one real difference is spread: the twin rungs are
the REGULAR ones (range 0.205), and every extreme of the ladder - the 0.811 at `31 -> 37` and the
0.073 at `37 -> 41` - is a non-twin rung (range 0.738). 5 twin rungs, 9 non-twin, to m59.

### R9. Parts, status, and the exact residual

The object is "the longest run `K` gears can block". Its parts, each with its status:

| part | statement | status |
|---|---|---|
| the arc law | `3 a_g = g -+ 1`, so `a_g` is even and the long arc is `2 a_g -+ 1` | PROVED (recorded; the even/long-arc form written here) |
| the umbrella bound | every gear with long arc `< S + 2` strikes a span-`S` stretch | PROVED (recorded, `flank_walk.md`) |
| the domino / type lemma | a gear with `g - a_g >= L` realises exactly `{}`, `{i,i+a}`, `{i}` (`i<a` or `i>=L-a`) | PROVED here, elementary; the complement of the umbrella bound |
| the type reduction | at each `L` the infinite prime pool has finitely many item types | PROVED here, from the type lemma |
| the matching identity | cover = small-part phase assignment + a degree-constrained matching of its holes | PROVED here (the "one big gear always takes one hole" step), VERIFIED 223/223 |
| `A(K)` itself | 2, 5, 7, 16, 22, 28, 37, 45, 68, 88, 101, 115 | MEASURED exact, `K <= 12`, over all primes; cross-checked |
| the tiler function `h_S(L)` | the least holes `k` prime gears leave in a run of `L` | MEASURED for initial segments to `L = 22`; NO bound proved |
| `A(K) < (p_{K+1}^2-1)/6` | the open lemma | MEASURED true, `K <= 12`, margin 2.7-3.8 |

Interactions already proved and used: umbrella + domino (the dichotomy, r48 R4); domino + arc law
(two gears share a short arc iff they are twins, r48 R4); type lemma + matching (this round).
**The lowest-order interaction NOT proved on the way from the parts to the shape is the tiler
function**: nothing on the tree bounds `h_S(L)` below for a general prime set `S`, and R7 shows
that `h_S(L)` alone - with every arc fact thrown away - already determines `A(K)` to 11%. That is
the next child, and it is not an arc question.

For the two formulations:

* **Adversary (`A(K)`).** The arc multiset gives the exact computation (type reduction) and
  therefore the ladder, and gives the mechanism at small `K` (`A(3) = 7` is one
  line from the hole-distance dictionary plus the arc supply, and `A(4) = 16` is r48's arc
  explanation made sharp). It gives NO upper bound of
  any kind at large `K`, and R7 shows it cannot, because the arcs are worth at most 11% of the
  answer.
* **Real machine (`F(M)`).** The arc multiset gives the ratio `F_real/A(K)` = 0.69 to 1.00 - the
  real machine is close to optimal and exactly optimal at `K = 10` - and it kills the twin-tax
  explanation of the gap. It gives no bound on `F(M)`.

**The exact residual**: a lower bound on `h_S(L)`, i.e. "no `k` primes leave fewer than
`2(K-k)+1` holes in a run of `(p_{K+1}^2-1)/6` columns". That is a capacity statement with a gear
count attached, and the capacity part is loose by a factor of two (face A, A2).

---

## 3. Mechanism, in one paragraph

A gear enters a run in one of two roles, decided by its long arc. Above the run it is a bare
domino whose only content is its short arc `a_g`, and that is a theorem (the type lemma), sharp
enough to collapse the infinite prime pool at any level to two dozen item types and so to make
`A(K)` exactly computable over ALL primes. But the record is not made by dominoes. Relaxing every
domino to "any two columns anywhere" - throwing away parity, arc identity and prime realisability
together - lengthens the best `K`-gear run by between 0 and 5 columns out of 45. The record is
made by the small gears, the tilers, whose long arc fits inside the run and whose second
coordinate `e = +-1` (the sign in `g = 3a + e`, which decides the long arc `2a + e`) is worth up
to 36% of the record. That is why the arc multiset behaves opposite to r48's reading: a
duplicated arc is a twin pair, a twin pair is the cheapest pair of gears carrying that arc, so
duplicating arcs means buying small gears, and the mean record rises monotonically with the
number of duplicated arcs at every gear count tested. De-twinning the real machine - replacing
`7, 13, 19, 31` by `37, 41, 47, 53` - is not removing waste, it is selling small tilers to buy
large ones, whose strikes are sparser,
and it costs between a tenth and two fifths of the record (`F_real/F_D` = 1.10 to 1.70). The real machine's deficit against the free adversary is
therefore not a twin tax; it is the initial-segment constraint - 31% at K = 4, then 4% to 15% from
`K = 7` up, and zero at `K = 10`, where `{5..37}` is an optimal ten-gear blocker and no ten primes
block more than 87 columns.

---

## 4. What is new (no prior art located)

1. **`A(7) = 37`, `A(8) = 45`, `A(9) = 68`, `A(10) = 88`, `A(11) = 101`, `A(12) = 115`, exact and exhaustive
   over ALL primes**, where r48 had exact values only to `K = 6` over a truncated pool and
   hill-climb lower bounds above. `A(9)` is 10 above the recorded lower bound. Two new tools make
   it possible: the type lemma (a gear above the run is described by its short arc alone, so the
   pool is finite at each level - a rigorous replacement for r48's "pool 5..149") and the MILP
   feasibility model, whose infeasibility certificate is the proof.
2. **`A(K)/W(p_{K+1})` is flat at 0.26-0.37 over `K = 4..12`.** The open lemma
   `A(K) < (p_{K+1}^2 - 1)/6` holds with a factor 2.7 to 3.8, and the adversarial form of the root has
   nearly the same slack as the root itself (whose own margin is 3.2 to 4.4). r48's reading that
   `A/F` falls to 1 is wrong:
   the sequence is 1.45, 1.22, 1.12, 1.09, 1.05, 1.17, 1.00, 1.11, 1.12.
3. **At `K = 10` the real machine is optimal.** `{5..37}` blocks 87 columns and no ten primes
   block more. The covering form and the root COINCIDE at that gear count.
4. **Arc duplication helps, exceptionlessly.** Mean `F` is monotone increasing in the number of
   duplicated arcs at `K = 3..7` (5,383 gear sets, 16 duplication classes, no exception),
   and the maximum is never at zero duplicates. De-twinning the real machine lowers its record by
   a factor 1.10 to 1.70 at every rung `m13..m31`. The wall's 5d claim that the real machine is a
   worse coverer BECAUSE it duplicates arcs is refuted; it is a worse coverer, but for another
   reason.
5. **The gear's second coordinate.** Writing `g = 3a + e`, `e = +-1`, the long arc is `2a + e`;
   the short-arc multiset determines `F` exactly at `K = 3` (146/146 multi-realised multisets)
   and fails from `K = 4` (spread up to 5 on a record of 14 at `K = 5`), and the extreme
   realisations of one arc multiset are exact sign-complements. So twin gears are interchangeable
   in the domino regime and nowhere else.
6. **The matching identity, verified.** 223 (gear set, level) instances, 0 disagreements, with
   the "one big gear always takes one hole" observation that makes the condition depend on the
   arc multiset only through the PAIRS.
7. **The relaxation ladder `B1 >= B2 >= B3 >= A`.** Which distances the dominoes may span -
   parity and prime realisability together - is worth 0 to 2 columns; the count and the
   multiplicity are worth everything. The arc multiset is measured NOT to be the lever.
8. **The hole-distance dictionary and the parity obstruction.** `{5,7,11}` at `L = 15` can leave
   its two holes only at distance 6 (the sharp form of r48's `K = 4` mechanism); `{5,7}` at
   `L = 10` and `{5,7,11,13}` at `L = 18..20` can leave them only at an ODD distance, which no
   arc can span, so the next gear must be a tiler. `A(3) = 7` is proved by this in one line. And
   the dictionary makes a prediction that holds: at `L = 27` the only even distance `{5..17}` can
   leave is 18, the only prime with arc 18 is 53, and `F({5,7,11,13,17,53}) = 28 = A(6)` while
   every other sixth gear from 23 to 89 gives 25 or 26.
9. **The record letters at every rung to m31**, extracted from witness covers and gated against
   the wall's B3: the record uses the short arc `a_{q'}` at every rung with a letter, and the
   long arc only at m23 where the alternation forces it. The m31 word is the m29 word with the
   left flank changed (`25 + 10 + 23` against `10 + 10 + 23`).

Screened against `docs/novel/README.md` before opening (the standing rule): the index carries
merge-law, record-law, walk-path, reachability, island-witness and flank-walk entries, and no
entry on an adversarial covering ladder, on dominoes, or on the arc multiset; the only
"short arc" occurrence there is the walk-path entry's one-tooth-per-run exception, a different
object. The parent branch `gear_count.md` is where `A(K)` and the domino lemma were introduced.

Named once, not re-derived: the umbrella bound and the kill-spacing law `3 a_g = g -+ 1`
(`flank_walk.md` F4/F13, `docs/proofs/05`); the word grammar and legality (`docs/proofs/05` (F));
the record law by phase reduction (`docs/proofs/09`); L4's sole-striker corollary
(`pair_statement.md`); r48's domino lemma and `A(K)` for `K <= 6` (`gear_count.md`); the m29/m31
record decompositions (`the_wall.md` B3); the m31 record class as the resistant m29 run
(`separability.md`).

---

## 5. Verdict

**The arc multiset is a real object, now measured to the bottom, and it is NOT the handle the
wall hoped for. What survives is the opposite reading and one sharper open lemma.**

* **The object is fully understood.** The type lemma says exactly what a gear above the run is
  (its short arc, nothing else), the matching identity says exactly how those gears combine with
  the tilers, and both are proved and verified. Verdict on the object: **FACT**, exact.
* **The handle is dead.** The wall's 5d recorded the arc multiset as the one which-residues
  property in which the real machine is measurably on the good side. Measured: (i) relaxing all
  arc information to a bare hole count changes `A(K)` by at most 11%, so the arc identities carry
  almost none of the content; (ii) arc duplication RAISES the record, exceptionlessly, so the
  real machine's twin pairs are not what holds it back; (iii) de-twinning the real machine makes
  it a worse blocker at every rung. The `A(K)` deficit of the real machine is the
  initial-segment constraint and it is zero at `K = 10`. Verdict on the route: **DEAD**, with the
  refuting instances recorded in section 6.
* **What survives, and it is better than what went in.** The open lemma is now
  `A(K) < (p_{K+1}^2 - 1)/6` with `A(K)` exact to `K = 12` and the ratio `A/W` FLAT at 0.26-0.37
  rather than climbing, and with the real machine attaining the adversarial optimum at `K = 10`.
  So the covering form is strictly stronger than the root but not by much; it is the root
  with about the same slack, and it is the formulation face A does not forbid. The smallest
  lemma that would close it is now stated about the TILERS, not the arcs:

  > **The tiler lemma (the residual).** For a set `S` of `k` primes let `h_S(L)` be the least
  > number of columns of a run of `L` consecutive columns that `S` leaves unstruck, minimised
  > over phases. Then `A(K) <= max{ L : some k-subset `S` of the primes has `h_S(L) <= 2(K-k)`,
  > for some k <= K }`, and this is tight to 11% with the right-hand side taken with NO arc
  > information at all. What is needed is a lower bound on `h_S(L)` - "k prime gears cannot leave
  > fewer than `2(K-k)+1` holes in a run of `(p_{K+1}^2-1)/6` columns" - for every `k`-set `S`.

  Is that the spectrum-plus-depth bound (node 2e) in adversarial form? **No.** Spectrum-plus-depth
  bounds a record by the gap spectrum and the depth of the machine below, and it dies on legality
  (29->31, 47->53). The tiler lemma is a hole-count statement about one run and it is a capacity
  statement - face A's A2 - with a gear count attached, which is exactly r48's R7 inequality
  reached from the other side. So the honest classification is: the arc multiset route reduces to
  the counting bound once the domino part is priced, and the counting bound is loose by a factor
  of two. Does the adversarial form have slack the real one lacks? **Little, and it shrinks**:
  `A/F` is 1.45 at `K = 4` and then 1.00 to 1.22 for every `K = 5..12`, exactly 1.00 at
  `K = 10`. The adversary's freedom to choose the small part is worth at most 22% above `K = 4`
  and sometimes nothing, so an adversarial proof would not be much stronger than the root - and
  correspondingly not much harder.
* **The self-referential reading is refuted, and with the sign reversed.** "Twin primes below `q`
  duplicate arcs and weaken the machine as a coverer, leaving more openings above `q`" is false:
  duplicated arcs strengthen it. `F` at fixed gear count is an INCREASING function of the number
  of twin pairs in the set, monotone in the mean at every `K` tested. So the feedback, if it
  exists, pushes the wrong way, and it cannot be made into the inequality the brief asked for.
  What it would need is a mechanism by which twin pairs raise the record only boundedly - which
  is a restatement of the root, not a route to it.

---

## 6. Dead ends, with the refuting instance

* **"The optimal set has distinct arcs" / "the smallest distinct arcs".** DEAD at every `K`. Every
  optimum from `K = 2` contains `5, 7`; the `K = 7` optimum `{5,7,11,13,17,19,31}` has three
  duplicated arcs; the `K` smallest distinct arcs are `D(K)`, whose record is 1.1-1.7 times
  shorter than the real machine's.
* **The twin tax as the cause of the real machine's deficit.** DEAD. `F_real/A(K)` rises with `K` and does not track
  `pi_2` at all, and de-twinning lowers the record at every rung (`F_real/F_D` = 1.10 to 1.70).
* **Arc duplication as a weakening.** DEAD, and inverted: mean `F` rises monotonically with the
  number of duplicated arcs at `K = 3..7`, without exception; at `K = 7` a set with no duplicated arc reaches 20 against the optimum 37.
* **The arc multiset as the lever on `A(K)`.** DEAD. The count-only relaxation `B1` equals `A(K)`
  at `K = 4, 7` and exceeds it by at most 5 columns to `K = 8`; parity and prime-realisability of
  the arcs together are worth 0-2 columns.
* **"Twin rungs have a different increment law".** DEAD: the twin `inc/q'` range [0.279, 0.484]
  lies strictly inside the non-twin range [0.073, 0.811], and the means are 0.374 against 0.364.
* **"`A/F` falls to 1, so the covering form and the root converge".** DEAD as stated (r48's
  reading of three terms): the sequence oscillates - 1.45, 1.22, 1.12, 1.09, 1.05, 1.17, 1.00,
  1.11, 1.12 - and touches 1 at `K = 10` without settling.
* **The exhaustive DFS over item multisets as the tool for `K >= 7`.** DEAD on cost: `K = 6`,
  `L = 28` took 6.9 million nodes and 22 s; `K = 7`, `L = 37` had not finished in 40 minutes and
  exceeded 1 GB. Replaced by the MILP, which settles the same question in 0.5 s.

---

## 7. Exceptionless, with the count

| statement | range | count | status |
|---|---|---|---|
| a gear with `g - a_g >= L` realises exactly the subsets `{}`, `{i, i+a}` (`i+a <= L-1`), `{i}` (`i < a` or `i >= L-a`) | all | proof, not a check | proved (type lemma) |
| every short arc is even; the long arc is `2 a_g -+ 1` | all | proof | one line from `3 a_g = g -+ 1` |
| the matching form and the direct search agree | 15 gear sets, every level with a non-empty big part | 223 instances | exact, 0 disagreements |
| no big gear's long arc fits inside its run | same | 223 | exact, the caveat is void |
| mean `F` increases with the number of duplicated arcs | `K = 3..7`, pools to 61 | 5,383 sets, 16 classes | exact, 0 exceptions |
| the short-arc multiset determines `F` at `K = 3` | primes 5..61 | 146 multi-realised multisets | exact, 0 exceptions |
| every record letter is `a_{q'}` or `b_{q'}` | rungs `5->7` .. `29->31` | 8 rungs, 11 letters | exact (the word grammar) |
| `F(q') - F(q) <= q'` (the budget) | rungs to m59 | 14 | exact, from the recorded ladder |
| `A(K) < (p_{K+1}^2 - 1)/6` | `K = 1..12` | 12 | exact, margin factor 2.7-3.8 |
| `A(K) >= F({5..p_K})` | `K = 1..12` | 12 | exact, equality at `K <= 3` and `K = 10` |
| no 7 primes below 80 block 37 columns (the MILP's `A(7) = 37`, re-run with the r48 tool) | primes 5..79 | 77,520 subsets | exact, 0 covers |
| `F({5..17, g}) = 28` iff `a_g = 18` (i.e. `g = 53`), else 25 or 26 | primes 23..89 | 16 gears | exact, 1 hit |

---

## 8. Children this branch opens

* **The tiler function `h_S(L)`** (R9's residual). How few columns can `k` prime gears leave
  unstruck in a run of `L`? Measured only for initial segments to `L = 22`. The object is the
  small part alone, with every arc fact discarded (R7 licenses that), and a lower bound on it is
  the whole of `A(K) < W`. First questions: is `h_{5..p_k}(L)` the minimum over all `k`-sets (the
  initial segment being the densest striker), and does `h` grow linearly in `L` once `L` passes
  the top gear?
* **The sign coordinate `e`.** `g = 3a + e`, long arc `2a + e`; R6 shows `e` is worth up to 36%
  of the record at fixed short-arc multiset, and that the best and worst realisations of one arc
  multiset are exact sign-complements. Which sign pattern maximises `F`, and does the real
  machine (which takes BOTH signs wherever a twin pair exists) sit at the maximum or the middle?
* **`A(K) = F({5..p_K})` at `K = 10`.** The only gear count at which the real machine is an
  optimal blocker. Is `K = 10` an accident of the prime list, or does equality recur? Cheap to
  test at `K = 13..16` with the MILP.
