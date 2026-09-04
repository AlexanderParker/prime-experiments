# R2.a - The machine feeds on itself

Branch of R2 (the whole-window formulation). Parent observation: the gears of a machine are
primes, so a twin gear pair of a large machine IS an opening of a small machine inside that
small machine's window; each level's openings are the next level's gears. Observation-first
branch: section 0 was written before any computation and has not been edited since.

Scripts: `research/anchor235/r37/sf_frame.py`, `sf_chain.py`, `sf_walks.py`, `sf_birth.py`,
`sf_birth2.py`. Result outputs (untracked): `research/anchor235/r37/results/sf_*.txt`. Every
number this document relies on is printed here.

---

## 0. Pre-registered (written before computing)

### 0.1 The correspondence, stated exactly

Let `(g, g+2)` be a twin prime pair with `g >= 5`. Write `u_g = (g+1)/6` (an integer, since
`g = 5 mod 6` for every twin pair start `g >= 5`). Then:

* `u_g` is a **column**: its members are `6 u_g - 1 = g` and `6 u_g + 1 = g + 2`.
* `u_g` is the **shared first tooth** of the two gears `g` and `g+2` (tooth-sharing-pinning (a)).
* For a prime `y`, column `u_g` lies in the window of the machine `{5..y}` iff `y < g` and
  `g + 2 <= y^2`, i.e. iff `sqrt(g+2) <= y < g`.
* Column `u_g` is an **opening** of `{5..y}` for **every** prime `y < g`: a gear `q <= y` strikes
  column `u_g` iff `q | g` or `q | g+2`, and `g`, `g+2` are primes greater than `y`.

So, with `L(g)` = the set of primes `y` with `sqrt(g+2) <= y < g`:

> **(C) The correspondence.** For every prime `y`, the openings of the machine `{5..y}` lying in
> its window `(y, y^2]` are exactly the columns `u_g` of the twin prime pairs `(g, g+2)` with
> `y < g` and `g + 2 <= y^2`. Equivalently: *the machine `{5..q}` has a twin gear pair
> `(g, g+2)` with `y < g <= y^2 - 2` for every prime `y <= sqrt(q)`* is the same statement as
> *an opening lands in the window of every machine `{5..y}` with `y^2 <= q`*.

The level map used below: `lev(g)` = the smallest prime `y` with `y^2 >= g + 2` (the lowest
level whose window holds the pair); the pair is a gear pair of every machine `{5..q}` with
`q >= g + 2`. "Level n to level n+1" means `y -> Y` with `Y` the largest prime `<= y^2` (the
level whose gear set first contains all of level `n`'s window openings as gears).

Frame consequence, stated once and not re-derived: in column units the window of level `n` is
`(y/6, y^2/6]` and the gears of level `n+1` occupy the columns `(g +- 1)/6 <= Y/6 = y^2/6`, so
**level n's window is the top part of level n+1's pre-window**, and a machine has no opening
below its own window (every column `k <= q/6` has a member `<= q+1`, hence a member that is
either a gear or has a prime factor below `sqrt(q)`).

### 0.2 What would count as a rule

A rule here must be (i) a statement about **positions or residues**, not a rate; (ii) checkable
with an exact exception count over a stated range; (iii) **uniform across levels** - it must say
the same thing at level `n`, `n+1`, `n+2`. A density, a fitted curve, or an average is not a
rule (project standing direction). A rule that only restates the definition of an opening, the
tooth rule `u_q = 6^{-1} mod q`, the self-blocking law, the neighbour-of-hit law, the CRT
double-kill classes of tooth-sharing-pinning, the Hardy-Littlewood count, or the `6ab +- a +- b`
form of twin columns is **not** a finding; it is noted in one line and the sub-question stops.

### 0.3 Objects and predictions, one block per object

**O1. Twin-gear content of a machine, and interaction of the coincidences.**
For each machine `{5..q}` list its twin gear pairs, their shared tooth column `u_g` and the four
pinned double-kill classes `+-u_g, +-u_g(g+1)` mod `g(g+2)`.

* P1.1 (predicted TRUE): of the four classes, exactly one representative - `u_g` itself - ever
  lies in the window of a machine below `g`; the other three classes' least representatives are
  `>= g(g+2) - u_g`, whose members exceed `y^2` for every `y < g`. Refuted by one twin pair with
  a second class representative inside a lower window.
* P1.2 (predicted TRUE): no two twin gear pairs of the same machine share a double-kill column
  inside a *lower* window. Refuted by one such column.
* P1.3 (predicted FALSE as a rule): inside the machine's own window, double-kill columns of
  different pairs do collide, and the collision set is a CRT class - i.e. there is no rule, only
  the CRT. **Stop line: the moment the answer is "the CRT classes intersect", the sub-question
  stops.**
* P1.4 (open, no prediction): the pair's strike lattice relative to its own birth column. Where
  are the columns struck by `g` and by `g+2` in terms of `k = u_g`? Is there structure beyond
  the single shared tooth?

**O2. The chain of landings.**
`g_0 = q`; `g_{n+1}` = the landing of the layered walk from the column holding `g_n^2` under the
machine `{5..g_n}`, which is the column of the first twin pair at or above `g_n^2`. Chain from
every prime `q <= 200`, iterated while `g_n^2` stays inside 12 digits.

* P2.1 (predicted TRUE, trivial but recorded): every landing is a twin pair, hence a shared
  first tooth column; the chain is "iterate the tooth of the next pair".
* P2.2 (predicted TRUE): the walk's **first** blocked column is the one holding `g_n^2`, and its
  smallest striker is `g_n` itself exactly when `g_n^2 - 2` is prime (the square gate), and
  `lpf(g_n^2 - 2)` otherwise. Exception count over all primes `q <= 5000`.
* P2.3 (predicted FALSE - nothing expected): the residues of `g_{n+1}` modulo the gears that made
  the hops at level `n` are unstructured. What would surprise me: a residue class avoided at
  every level of every chain (e.g. `g_{n+1} != +-u_T (mod T)` for the top hop layer `T`), or a
  forced residue.
* P2.4 (predicted FALSE - nothing expected): no relation between the top hop layer `T_n` (the
  largest gear whose hit moved the landing) at level `n` and at level `n+1`. What would
  surprise me: `T_{n+1}` a function of `T_n`, or `T_n` bounded uniformly, or `T_n | ` anything.
* P2.5 (predicted FALSE as a rule): walk lengths along a chain follow no rule. The
  Hardy-Littlewood heuristic says the mean gap between twins near `X` is `~ (log X)^2 / (2 C_2)`
  and `log g_{n+1} = 2 log g_n`, so a *rate* would quadruple per level; that is a rate, not a
  rule, and is recorded only to say what the null is. What would surprise me: monotonicity of
  the walk length along every chain (predicted to fail), or a bound like `W_n <= g_n` failing.
* P2.6 (predicted TRUE): chains never merge in range (merging needs equal `g_n`).

**O3. The window as the next machine's birthplace.**
For machine `{5..q}` and each opening `k` of its section `(p^2, q^2]`, the **flanking strikers**
are the gears striking `k-1` and `k+1`.

* P3.1 (predicted TRUE, and if true it closes O3 as a route): a gear pair's entire action is a
  function of its birth column alone - the teeth of `6k-1` and `6k+1` are `+-k`, so nothing but
  `k` crosses from level `n` to level `n+1`. Hence any relation between the flanking-striker
  pattern and what the pair later strikes is a relation between the factorisations of
  `6k +- 5, 6k +- 7` and `k` itself. Refuted by exhibiting any level-crossing datum other than
  `k`.
* P3.2 (predicted TRUE): the new pair never strikes `k-1` or `k+1` (teeth are never adjacent -
  the known neighbour-of-hit law); so flanking strikers are always old gears. One line, stop.
* P3.3 (open, no prediction): the distribution of the smallest flanking striker over the
  section, and whether the openings with an unusually large smallest flanking striker are
  distinguished in any positional way at the next level.

**O4. Anything else that repeats across levels.** Recorded with an exact instance count only.

### 0.4 Scorecard

| # | Prediction | Verdict |
|---|---|---|
| P1.1 | one visible class only | **CONFIRMED**, 0 exceptions in 2,159 pairs (`g <= 200000`) |
| P1.2 | no cross-pair coincidence in a lower window | **CONFIRMED**, proved, 0 exceptions |
| P1.3 | in-machine collisions are CRT, no rule | **CONFIRMED and STOPPED** at the stop line |
| P1.4 | strike lattice - open | **ANSWERED**: the straddle ladder, exact (rule N5) |
| P2.1 | landings are shared teeth | **CONFIRMED**, 667/667 walks, 97/97 chain levels |
| P2.2 | first hop = square gate | **CONFIRMED**, 0 exceptions in 667 walks; grew into N2 |
| P2.3 | no residue rule | **CONFIRMED** (only non-divisibility, i.e. the definition) |
| P2.4 | no top-layer rule | **CONFIRMED** across levels; but N2 fixes `T_n` at one level |
| P2.5 | no walk-length rule | **CONFIRMED** (33 of 44 chains monotone, not all) |
| P2.6 | no chain merges | **CONFIRMED**: the only joins are chain inclusion (9 cases) |
| P3.1 | only `k` crosses levels | **CONFIRMED**, and it is what makes N3 level-free |
| P3.2 | pair never strikes `k +- 1` | **CONFIRMED** (the known neighbour-of-hit law) |
| P3.3 | flanking strikers - open | **ANSWERED**: gear 5 owns a flank of every opening (N6); the
free flank's smallest striker has median 11, max 907 |

---

## 1. Setup

| object | range computed | script |
|---|---|---|
| machines run directly, window openings vs twin columns | `y = 5..199`, all columns to `(y^2-1)/6` | `sf_frame.py` A |
| `N(y)` = openings in the window | every prime `y = 5..4999` (twins to `2.5e7`, 130,511 of them) | `sf_frame.py` A |
| the four pinned double-kill classes | every twin pair `g <= 200000` (2,159 pairs) | `sf_frame.py` B |
| strike lattice, protected multiples | every twin pair `g <= 600` (26 pairs), columns `<= 400000` | `sf_frame.py` C |
| the walk from `q^2` under `{5..q}` | every prime gear `q = 5..4999` (667 walks, 18,743 hops) | `sf_walks.py` |
| chains `g_{n+1}` = first twin `>= g_n^2` | every prime `q = 5..199`, iterated while `g_n < 10^6`; 97 levels, landings to 12 digits | `sf_chain.py` |
| the transfer rule | 341 twin pairs `g <= 20000`, offsets `j = -8..8`, `i = 0..40`: 832,915 checks | `sf_birth.py` B1 |
| flanking strikers on the section | sections of machines `7..1009`, 8,309 openings | `sf_birth2.py` |

Landings are certified twins by the window lemma (open under `{5..g}` and below
`nextprime(g)^2`) and, at 12 digits, re-checked by a deterministic Miller-Rabin.

## 2. Observations

### 2.1 The correspondence and the frame (O1, setup)

Running the machines directly for `y = 5..199` and comparing with the twin pairs: **0
mismatches**. `N(y)` = 2, 4, 7, 9, 15, 17, 21, 28, 30, 41 at `y = 5..37`; `N(101) = 201`,
`N(1009) = 8278`, `N(4999) = 130343`. The minimum over all primes `y <= 5000` is `N(5) = 2`.

Read on one machine: `{5..q}` has a twin gear pair `(g, g+2)` with `sqrt(q) < g` for every prime
`q >= 7`; the only zero is the degenerate `q = 5` (one gear, no pair). That count is `N` of the
level below, so it is the same statement, not a second one.

### 2.2 The pinned classes (O1, P1.1-P1.3)

For all 2,159 twin pairs with `g <= 200000`: the shared tooth column `u_g` has members exactly
`(g, g+2)` (0 exceptions), and none of the other three pinned classes `-u_g`, `+-u_g(g+1)` has a
representative at or below `(g^2-1)/6`, the top of *every* lower machine's window (0 exceptions).
The least of the three is the twin-product column `c_2 = 6u_g^2 = (g+1)^2/6`; `c_3 = g(g+2) - u_g`
and `c_4 = g(g+2) - c_2` are larger.

P1.3 hit its stop line: two pairs' classes do intersect inside a *large* machine's own window
(moduli `g_1(g_1+2)` and `g_2(g_2+2)` are coprime, so the intersection is a CRT class, non-empty
once the window exceeds `g_1^2 g_2^2` columns). That is the CRT and nothing else; stopped.

### 2.3 The strike lattice of a pair about its own birth column (O1, P1.4)

Let the pair be born at column `k` (`g = 6k-1`, `g+2 = 6k+1`, teeth `+-k` in both gears).
Enumerating every strike of both members over columns `<= 400000` for all 26 pairs with
`g <= 600`: **0 mismatches** against

```
  t = 6m+1 :  g strikes t*k - m,      g+2 strikes t*k + m        (separation 2m)
  t = 6m+5 :  g strikes t*k - (m+1),  g+2 strikes t*k + (m+1)    (separation 2m+2)
```

`t = 1, m = 0` is the shared first tooth. `t = 6k+1` gives the twin-product column `6k^2`.
Consequence, also 0 failures: for `2 <= t <= g-2` **neither member strikes `t*k`** - a pair
leaves the multiples of its own birth column alone. How open those columns are is entirely the
shield of whichever gear divides `t` (`t=5`: ratio 1.62 against `5/3 = 1.667`; `t=7`: 1.35
against `7/5 = 1.4`; `t = 2,3,4,6`: 0.91-1.00). A rate, and an old one; recorded, not pursued.

### 2.4 The walk from `q^2` (O2, and the new rules)

667 walks, `q = 5..4999`: length min 1, median 19, max 265 (at `q = 4637`); every landing a twin
pair. Hops by layer: gear 5 makes 7,564 of 18,743 (40.4%), gear 7 makes 3,273 (17.5%), gears
above `sqrt(q)` make 2,827 (15.1%). Distinct layers used per walk: min 1, median 9, max 44 -
a median of 3.2% of the machine's gears.

| statement | count | exceptions |
|---|---|---|
| `k0 = -c (mod q)` where `6c = 1 (mod q)`: the walk starts ON a tooth of the top gear | 667 | 0 |
| the next strike of `q` is `d = 2c mod q` columns higher (`= 2u_q` if `q = 5 mod 6`, `= q - 2u_q` if `q = 1 mod 6`) | 667 | 0 |
| `L < d`, i.e. the top gear strikes its own walk interval **once**, at the first column | 667 | **1** (`q = 53`: `L = 27`, `d = 18`) |
| `T = q` (the top gear is the deepest hop layer) iff `q^2 - 2` is prime | 667 (153 with the gate open) | **0** |

Above `q = 53` the tightest walk is `q = 137` with `L = 24` against `d = 46` (ratio 0.5217);
median `L/d = 0.0231`, 90th percentile 0.1086.

### 2.5 The chain (O2)

97 levels from 46 starting primes; chain length 2 to 5 (e.g. `5 -> 29 -> 857 -> 734471 ->
539447650517`). `g_{n+1} - g_n^2 = 6 L_n - 2` exactly at every level (0 exceptions), which is
just the landing column read as a number.

* P2.3: over 827 (level, hop gear) pairs the landing's only residue constraint modulo a hop gear
  is non-divisibility - the definition of an opening. No further structure.
* P2.4: `T_n` takes 53 consecutive-pair values with 23 conflicting successors; `T_{n+1}` is not a
  function of `T_n`. `T_n > sqrt(g_n)` at 85 of 97 levels; `T_n = g_n` at 23 (the square-gate
  share, 23.7% against 22.9% of primes with `q^2-2` prime).
* P2.5: walk lengths increase along 33 of 44 multi-level chains, not all; `L_n <= g_n` at every
  level. No rule.
* P2.6: nine `g` values are reached by two chains, and every one is chain inclusion (`29` is
  `g_1(5)`, `59` is `g_1(7)`, `137` is `g_1(11)`, `179` is `g_1(13)`), so the chain forest has no
  confluence from unrelated starts.

### 2.6 Birth against later action (O3)

* Gear 5 strikes at least one neighbour of **every** opening, and both exactly at the openings on
  its shield: 8,309 openings over the sections of machines 7..1009, split `k = 0/2/3 mod 5` as
  2742/2778/2789, **0 exceptions** to either half.
* The flank gear 5 leaves free: smallest striker min 7, median 11, max 907; struck by no gear of
  the machine at 340 of 5,567 free flanks - those are the 170 prime quadruplets in the range,
  each counted from both ends.
* The transfer rule (below) checked at 832,915 (pair, `j`, striker, `i`) quadruples: **0
  mismatches**. Its `i = 0` case censused on real machines: of 50,906 (opening, `j = +-1`,
  striker) triples, 3,093 carried over to the pair's own walk-start column, and the gears that
  did so were **exactly** `{7: 1663, 17: 1149, 31: 281}` - the predicted level-free set.

## 3. Candidate rules, with exception counts

**N1 (the one-strike rule).** The walk from `q^2` under `{5..q}` begins on a tooth of the top
gear - `6 k_0 = q^2 - 1 = -1 (mod q)`, so `k_0 = -c` with `c = 6^{-1} mod q` - and the top gear's
next strike is exactly `d = 2c mod q` columns higher (`2u_q` when `q = 5 mod 6`, `q - 2u_q` when
`q = 1 mod 6`). Hence the top gear strikes the whole walk interval **exactly once, at its first
column**, whenever `L < d`. Exceptions over every prime gear `5..4999`: **one**, `q = 53`
(`L = 27 >= d = 18`). Above `q = 53` the worst ratio `L/d` is 0.5217 at `q = 137`.

**N2 (the square-gate top layer).** In that walk, the largest layer that hops is the top gear
itself if and only if `q^2 - 2` is prime. Exceptions over the same 667 walks: **zero** in both
directions (153 walks with the gate open, all with `T = q`; 514 with it shut, none with `T = q`).

**N3 (the transfer rule).** Let a pair be born at column `k` (`g = 6k-1`) and let
`k_0' = k(g-1) = 6k^2 - 2k` be the first column of that pair's own walk one level up. A gear `h`
that strikes column `k + j` at the birth level strikes column `k_0' + i` if and only if

```
   h | (6j)^2   + 6i - 2   or   h | (6j)^2   + 6i        (h hit the LOWER member at k+j)
   h | (6j+2)^2 + 6i - 2   or   h | (6j+2)^2 + 6i        (h hit the UPPER member at k+j)
```

Neither `k` nor `g` appears: **the two offsets alone decide which gears can carry from a birth
column to that pair's own working region.** Exceptions in 832,915 checks (341 pairs, `|j| <= 8`,
`i <= 40`): **zero**. At `i = 0` the conditions read `h | (6j)^2 - 2` and `h | (6j+2)^2 - 2` -
the square-gate numbers of the offsets themselves - so at `j = +-1` the entire admissible set is
`{7, 17, 31}`, and the census on real machines produced exactly those three gears and no other
(3,093 carry-overs of 50,906).

**N4 (the pair's own frame).** The next level's walk starts at `k_0' = 6k^2 - 2k`, which is
`2k` columns below the pair's twin-product column `6k^2`; both are strikes of the pair, and they
are consecutive strikes of each member. So the two newest gears each strike the next level's
walk interval **once**, at `k_0'` (gear `g`) and at nothing until `k_0' + 2k` (both gears), i.e.
`(g+1)/3` columns on. Checked at all 667 landings (identity and spacing), with the whole `2k`
gap brute-forced at the 30 smallest: **0 failures**. At the 17 landings whose own walk was also
computed, `L(g) < 2k` in every case.

**N5 (the straddle ladder).** Section 2.3, 0 mismatches at 26 pairs over 400,000 columns; and
the pair strikes no multiple `t*k`, `2 <= t <= g-2`, 0 failures.

**N6 (gear 5 owns a flank).** Every opening has at least one neighbour struck by gear 5, and both
exactly when `k = 0 mod 5`. 8,309 openings, 0 exceptions. This is the gear-5 lock (node 5g) read
at a single column and should be filed there, not as a new law.

## 4. Mechanism

**N1/N4.** Gear `q`'s teeth are `+-c`, `c = 6^{-1} mod q`, and the two arcs between them have
lengths `2c mod q` and `q - (2c mod q)` (the two-teeth kill-spacing law, kernel
`TwoTeeth.kill_spacing`). What is new is *where the walk starts*: the column holding `q^2` has
`6k_0 + 1 = q^2`, so `6 k_0 = -1 (mod q)` and `k_0 = -c` - the walk begins standing on one tooth.
It therefore has a whole tooth arc of room before the top gear can interfere again, and the arc
is `d = 2c mod q`, one third or two thirds of `q` depending on `q mod 6`. The same computation
at the pair `(6k-1, 6k+1)` gives `d = 2k` for both members, and `2k = (g+1)/3`.

**N2.** The first column of the walk carries the members `q^2 - 2` and `q^2`. Gear `q` is its
smallest striker iff nothing below `q` divides `q^2 - 2`, i.e. iff `q^2 - 2` is prime (a composite
`q^2-2` has a prime factor below `q`, since `q^2 - 2 < q^2`). For `q` to be the smallest striker
of a *later* traversed column `x`, one member must be `q m` with `m` free of primes below `q`,
hence `m >= q` and, `m = q` being the start, `m >= q + 2`: that column is at least `d` further
on. So the only way to break N2 is a walk with `L >= d`, i.e. N1's single exception; and at
`q = 53` the second `q`-strike happens to carry a smaller striker, so N2 survives it.

**N3.** If `h` strikes `k+j` on the lower member then `6(k+j) - 1 = 0 (mod h)`, so
`g = 6k - 1 = -6j (mod h)` and `g^2 = (6j)^2 (mod h)`. The members of column `k_0' + i` are
`g^2 - 2 + 6i` and `g^2 + 6i`. Substituting removes `k` and `g` entirely. The upper-member case
gives `g = -(6j+2)` instead. So a gear's position relative to a *birth* column and its position
relative to that pair's *own square* are locked together by a congruence in which the level does
not appear - which is exactly the sense in which this recursion has no memory: the only datum
that crosses a level is the birth column `k`, and even that enters the next level only through
`g = 6k-1` as a modulus.

**N5.** `k` is invertible modulo both `6k-1` and `6k+1` (`6k = 1` and `6k = -1` respectively), so
`t k = +-k` forces `t = +-1` modulo the gear: the ladder and the protected multiples are the two
readings of that one inversion.

**N6.** Gear 5's teeth are `+-1 mod 5`, so an opening is at `k = 0, 2, 3 mod 5`; each of those
puts at least one of `k+-1` on `+-1 mod 5`, and `k = 0` puts both.

## 5. What is new

Located prior art for each piece, checked against `docs/novel/README.md`,
`docs/proof-search/alignment-rules.md` and `docs/proof-search/anchor-235.md`:

* The correspondence (C) is the route (`docs/proofs/01`) plus tooth-sharing-pinning (a) read in
  one direction; it is the branch's frame, not a finding. The observation that a machine has no
  opening below its own window, and that level `n`'s window is the top of level `n+1`'s
  pre-window, is a restatement of the horizon lemma; recorded in one line only.
* N5 and the protected multiples are elementary and follow from `gcd(k, 6k+-1) = 1`; the ladder
  form (separations `2m` and `2m+2` about `t k`) is not on record anywhere in the project, but it
  is one line of algebra and should be presented as such.
* N6 is node 5g (the gear-5 lock) at length 1.
* **N1, N2 and N4 are new as far as the project's record goes.** Section 9c/9d of anchor-235
  gives the walk, the hit law and the chain law; nothing there says the walk *starts on a tooth of
  the top gear*, that the top gear is therefore inert for a whole tooth arc, or that the deepest
  hop layer is decided by a single primality test on `q^2 - 2`. The square gate itself is on
  record (alignment-rules 4.1, "a gear is needed iff it owns a pseudo-twin in the window"), but
  as a *necessity* criterion for the gear, not as the identity of the walk's top layer.
* **N3 is new** and is the branch's answer to "is there a relation between where an opening is
  born and what it does as a gear". The relation exists, it is exact, and it is level-free - which
  is precisely why it carries no information forward: the admissible carriers depend on the
  offsets only.

Standard results these resemble, named after the mechanism as the directions require: N1 and N4
are the two-teeth kill-spacing law evaluated at a distinguished starting phase; N2 is the
least-prime-factor (horizon) lemma applied to `q^2 - 2`; N3 is elementary congruence
substitution; the chain's absence of structure is what one expects of consecutive
prime-counting problems and matches the Hardy-Littlewood null.

## 6. Interpretation against the root

The root needs an opening to land inside `(y, y^2]`. This branch says three things about how the
machine's own growth bears on that.

1. **The newest gears are almost inert where they are needed.** The gear that creates the window
   strikes the first `d ~ q/3` columns of it exactly once, at the column holding `q^2` (N1), and
   the pair born at the landing does the same one level up (N4). So the question "does the walk
   from `q^2` land soon" is decided by the *old* gears: 40% of hops are gear 5, 58% gears 5 and 7,
   and a median of 3.2% of the machine's gears do all the work of any one walk.
2. **The recursion carries one number per level and no more.** A pair's whole action is `+-k`
   modulo `6k -+ 1`; N3 makes the failure of information transfer exact - the gears that can act
   in both places are fixed by the offsets, not by the level.
3. **None of this bounds a length.** N1 says the top gear cannot push the first opening past its
   own tooth arc *by itself*; it says nothing about the other `pi(q)` gears, and the walk is
   exactly what they do. N2 fixes which layer is deepest, not how far the walk goes. So the
   branch produces forced position objects, in the same family as the gear-5 lock, the corridor
   law and the slot rule - and, like them, no size lever.

What would have to happen for N1 to fail: a walk longer than one tooth arc of the top gear. That
happens once in range (`q = 53`) and is not forbidden by anything - it is a statement of the same
kind as the root itself, only at scale `q/3` instead of `q^2/6`, and proving it would prove
twin-Bertrand. N2 would fail if a walk reached `q` times a prime `>= q + 2` and that column had no
smaller striker; that needs `L >= d` again. Both rules therefore rest on the same unproved size
statement, which is why they are FACTs and not a route.

## 7. Verdict

**FACT, not a route.** Four exact level-uniform rules (N1, N2, N3, N4) with the exception counts
above, plus three restatements filed where they belong (N5 elementary, N6 to node 5g, (C) to the
route). The chain of landings has no rule, as pre-registered: 10 of 13 predictions confirmed as
stated, two open ones answered, none refuted.

The self-similarity the human asked about is real and is now stated exactly: an opening of level
`n` is a gear pair of level `n+1` whose teeth are its own birth column, the window of level `n`
is the top of the pre-window of level `n+1`, and the pair's only coincidence visible at its birth
level is the birth column itself. What the recursion does **not** do is carry structure: N3 shows
the transfer between a birth neighbourhood and the pair's own working region is governed by a
congruence with no level in it.

## 8. Dead ends (do not re-enter)

* Cross-pair coincidences inside a machine's own window: the CRT and nothing else (stopped at the
  pre-registered stop line).
* Openness of the protected multiples `t*k`: entirely the shield of the gear dividing `t`; a rate.
* Chain structure - residues of `g_{n+1}` modulo the hop gears, `T_n` against `T_{n+1}`, walk
  lengths along a chain, chain confluence: all null, as pre-registered.
* "Where an opening is born predicts what it does as a gear": closed by N3 - the relation is exact
  and level-free, hence carries nothing.
