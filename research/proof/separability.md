# Separability of flanks by gear (branch 2g.i.a.i, prover, 2026-09-05)

Parent: node **2g.i.a** (the glue as a covering statement, `research/proof/glue_covering.md`).
What spawned this branch: that branch's one live puzzle and the wall's new thin place 6
(`research/proof/the_wall.md` section 5c). In every measure tried before (symmetry, spacing,
coherence, squareness of phases, cover numbers) the real machine is typical of its family; in one
it is not. Its extremal 3-runs are GLUEABLE at 62.5% against a pooled family rate of 9.4%, the
99.6th percentile of 223 comparable m19 members, and the obvious cause (the count of letter gears)
is exactly the family mean. Gluability is a which-residues property of how the two flanks of a
record-class run are struck. This branch asks the property directly, as a graded quantity: **how
separable are the strikers of the two flanks?**

What this branch can find that is not already known: whether "the flanks are struck by separable
sets of gears" is a real quantity of the machine (it is defined here for the first time), what its
distribution is on the real machine and on the family, which gears are the ones that cannot be
separated, whether they are exactly the gears the move lemma allows to move, whether the real
teeth's one-third separation is what makes the sets separable, and whether the record's three
top gears are exactly its unseparable gears.

**Verdict in one line: gluability is NOT separability of the flanks. Not one of the 106 hard
attaining 3-runs at m13..m31 is separable, and neither is any of 2832 family ones, for a
counting reason no machine below `{5..109}` can escape (`sum 2/g < 2`); the gears that must
serve both flanks are the bottom of the gear set (gear 5 or 7 at 106 of 106) and are never the
movable ones (0 of 106); the one-third separation MAXIMISES sharing rather than minimising it
(256 of 256 hard cells); and the record's top gears are exactly the gears that are NOT shared
(10 of 10 record stretches), so "made at the top" and "separable at the top" are the same fact
with the sign reversed. Measured at the cell it actually occupies, the real machine's glue rate
is 10 of 14 against a family 30.0% at m19, not against 9.4%: face C's exception survives in
direction but shrinks from a factor 6.6 at the 99.6th percentile to about 2.4 on seven
independent mirror classes. Kept: the divisor form `Leg_real(v) = {g : g | 3v-1 or g | 3v+1}`
(400 of 400) and ten exceptionless counts. Verdict details in section 6.**

## 0. Pre-registered (written before any computation of this branch)

### Definitions fixed here

- Machine `M = {5..q}`, gears `G`, `u_g = 6^{-1} mod g`, teeth `T_g = {u_g, -u_g} mod g`; gear `g`
  **strikes** column `k` iff `k mod g in T_g`.
- **Separation.** `d_g := 2 u_g mod g`, the distance between the two teeth. Because
  `3 d_g = 6 u_g = 1 (mod g)`, the real separation satisfies `3 d_g = 1 (mod g)` at EVERY gear:
  `d_g = (g+1)/3` when `g = 2 (mod 3)` and `(2g+1)/3` when `g = 1 (mod 3)`. Writing
  `D_g = min(d_g, g - d_g)` for the short arc, `D_g = (g +- 1)/3` always: **the real teeth divide
  every gear's period in the ratio 1 : 2.** A random symmetric-tooth member has teeth `{w, -w}`
  with `w` uniform in `1..(g-1)/2`, i.e. `D` uniform-ish in `1..(g-1)/2`.
- **3-run** `(x_0, L, v, R)`: four consecutive openings `x_0 < x_1 < x_2 < x_3` with `L = x_1 - x_0`,
  `v = x_2 - x_1`, `R = x_3 - x_2`. Its **left flank** is the column set
  `Lam = [x_0+1, x_1-1]` (`L-1` blocked columns) and its **right flank** is
  `Rho = [x_2+1, x_3-1]` (`R-1` blocked columns). The middle gap's interior `(x_1, x_2)` is NOT
  part of either flank.
- **Blocking assignment.** A choice, for each column of `Lam`, of a gear that strikes it, and for
  each column of `Rho`, of a gear that strikes it. `A` = the gears used on `Lam`, `B` = the gears
  used on `Rho`. Equivalently: a label in `{left, right, both, unused}` for each gear such that
  `left+both` covers `Lam` and `right+both` covers `Rho`.
- **The shared number** `s(run) := min |A n B|` over all blocking assignments -- the minimum number
  of gears that must strike BOTH flanks. A gear that happens to strike both but is not needed on
  one side does not count. `s = 0` is **separable**.
- **The used number** `u(run) := min |A u B|` among the assignments attaining `s`.
- **The separation index** `sigma(run) := s / u`.
- **The raw overlap** `ov(run)` := the number of gears that strike both flanks (`ov >= s` always).
- **The separation certificate.** If `s = 0`, colour `A` left, `B` right, the unused gears left.
  By CRT pick `z` with `z = x_0 (mod g)` for left gears and `z = x_2 - L (mod g)` for right gears.
  Then `z+1..z+L-1` are blocked (each by its `A` gear at the left phase), `z+L` is an opening (it
  is `x_1` modulo every left gear and `x_2` modulo every right gear, both openings), and
  `z+L+1..z+L+R-1` are blocked by `B`. Hence `F_2(M) >= L + R`. So **separability implies the glue**
  and is a strictly local, search-free version of it.
- **The graded certificate and its loss.** For any colouring `sigma : G -> {left, right}` (no
  `both`, no `unused`), let `a` = the largest `a` with `x_1 - 1, .., x_1 - a` all struck by left
  gears and `b` = the largest `b` with `x_2 + 1, .., x_2 + b` all struck by right gears. The same
  CRT point certifies `F_2(M) >= a + b + 2`. The **loss** of the run is
  `c(run) := (L + R) - max_sigma (a + b + 2) = (L-1-a) + (R-1-b)`. `c = 0` iff `s = 0`.
- **`J`-run version.** For `J` consecutive gaps `g_1 .. g_J` the flanks are the interior of `g_1`
  and the interior of `g_J`; everything above applies verbatim, and `s = 0` certifies
  `F_2 >= g_1 + g_J`.

Known results cited, never re-derived: the peel bound and middle-sum lemma (alignment-rules
736-790), L4 (docs/proofs/19), the chain law and merge law (docs/proofs/05, 09), the shadow lemma
and the move lemma (2g.i.a), the junction theorem (R3.h.i), `N(v) <= F_2` (2g.i).

### Theory

T. **The real machine's extremal 3-runs are glueable because their flanks are struck by nearly
separable sets of gears, and what cannot be separated is exactly what the move lemma allows to
move.** The cause is the one-third separation: with the two teeth dividing every gear's period in
the ratio 1 : 2 and the middle gap `v` pushing the two flanks into different arcs, a gear is forced
onto both flanks less often than for a random separation, so `s` is smaller for the real teeth than
for the family at the same `(L, v, R)`.

### Predictions, with the number that would refute each

- **P1 (instrument).** `s = 0` implies C2 succeeds, at every run tested; and `s = 0` implies
  `c = 0`. REFUTED by one separable run at which C2 fails.
- **P2 (the real machine is separable).** On the HARD attaining 3-runs with `v >= 6`
  (`v < min(L,R)`, the ones with content) at m13..m31, the real machine has `s = 0` at 40% or more,
  and `s <= 1` at 90% or more. REFUTED if `s = 0` at fewer than 20%, or if the median `s` is 2 or
  more.
- **P3 (the family is not).** On 200 random symmetric-tooth members at m13, m17, m19 the pooled
  `s = 0` rate on hard attaining runs is below 15%, and the real machine sits above the 95th
  percentile of the members at each machine. REFUTED if the real machine is inside the family's
  interquartile range.
- **P4 (separability explains gluability).** Restricted to the hard attaining runs, the C2 success
  rate given `s = 0` is 100% and given `s >= 2` is below 20%; and the correlation carries the
  family gap, i.e. the real machine's 62.5% at m19 is reproduced to within 10 points by
  "`s <= 1`". REFUTED if the C2 rate given `s >= 2` exceeds 50%, or if the `s` distribution does
  not separate the real machine from the family.
- **P5 (the shared gears are movable).** Whenever `s >= 1` in the real machine, every gear in some
  minimum shared set lies in `Leg(v) u Pad(v)` (the move lemma's movable gears: `v = +-d_g` or
  `v = 0 (mod g)`). REFUTED by more than 10% exceptions.
- **P6 (which band).** The shared gears of the real machine's runs are the TOP band (the largest
  three gears) at 60% or more of the runs with `s >= 1`; the family's are spread evenly over the
  bands. REFUTED if the real machine's shared gears are predominantly the small gears (5, 7).
- **P7 (one third minimises sharing).** At the observed `(L, v, R)` of the real machine's hard
  attaining runs, the expected number of shared gears `E[ov]` computed over all phases as a
  function of the separation `D` is minimised at, or within one of the minimum at, `D = (g +- 1)/3`
  for a majority of (gear, run) cells; and the real machine's `E[s]` is below the family mean at
  every machine. REFUTED if the real `D` ranks above the median separation at more than half the
  cells.
- **P8 (the record's top gears are its shared gears).** For each record stretch of m19..m31, taken
  as the 3-run or `J`-run of the machine below whose pieces R3.h names, the minimum shared set is
  exactly the top gears that R3.h says do the gluing. REFUTED by one record whose shared set
  contains a gear outside the top three, or misses one of them.
- **P9 (bounded loss).** At every attaining 3-run with `v >= 6` at m13..m31 the loss `c` is at most
  4, and at the resistant m29 run `(18, 10, 30)` it is at most 4. REFUTED by a loss above 8.

Stop rules: any sub-question that reduces to L4, the chain law, the peel bound, the middle-sum
lemma, the shadow lemma or the move lemma is stopped in one line and cited.

The scorecard for P1-P9 is filled in section 7.


## 1. Setup (exact ranges)

All numbers are exact, on full periods, in integer arithmetic. m13..m23 were sieved whole in
RAM. m29 (1,078,282,205 columns) and m31 (33,426,748,355 columns) were passed in 3e7-column
chunks by four processes with a 4096-column margin each side, reusing `r46/gl_deep.run`
verbatim (7.9 s and 235.8 s at 4 cores). The separability test itself needs only `x_0 mod g`
for each gear, so no period is ever held for it.

Instrument gate before any new claim: the attaining-3-run counts with `v >= 6` come out
**48, 90, 124, 188, 148, 264 at m13..m31** -- exactly 2g.i.a's counts -- and the trivial/hard
split (`v >= min(L,R)` against `v < min(L,R)`) comes out **48/0, 86/4, 108/16, 168/20, 126/22,
220/44**, exactly 2g.i.a's table. The C2 verdict is recomputed here with 2g.i.a's own solver, so
every "C2" column below is that branch's object, not a re-implementation.

Scripts, `research/anchor235/r47/`: `sp_core.py` (the shared number, the used number, the loss;
minimal-cover enumeration), `sp_measure.py` (m13..m23), `sp_deep.py` (m29, m31),
`sp_family.py` (200 members at m13, m17, m19), `sp_teeth.py` (the arc condition and the
separation sweep), `sp_record.py` (the records layer by layer), `sp_letters.py` (the letter
gears and the divisor identity), `sp_floor.py` (the counting floor), `sp_loss.py` (SEP-c and
GLUE-c), `sp_confound.py` (the matched-cell comparison). Outputs in
`research/anchor235/r47/results/` (gitignored); every number relied on is written here.

Exactness of the minimum: `s` is computed by enumerating the MINIMAL covers of each flank
(shrinking a cover can only shrink the intersection, so the minimum is attained at a pair of
minimal covers) and taking `min |A n B|` over all pairs. This is an exact minimum, not a
heuristic, and `all_min_shared` returns every shared set attaining it.

## 2. Results

### 2.1 The shared number on the real machine (item 1)

Attaining 3-runs with `v >= 6`, split at `v = min(L, R)`:

| machine | trivial | `s` on the trivial runs | separable (`s=0`) | hard | `s` on the hard runs | separable |
|---|---|---|---|---|---|---|
| m13 | 48 | 0:8, 1:8, 2:24, 3:8 | 8 | 0 | - | - |
| m17 | 86 | 0:12, 1:16, 2:30, 3:28 | 12 | 4 | **3:4** | **0** |
| m19 | 108 | 0:44, 1:14, 2:20, 3:26, 5:4 | 44 | 16 | **4:16** | **0** |
| m23 | 168 | 0:24, 1:40, 2:34, 3:18, 4:32, 5:16, 6:4 | 24 | 20 | **4:2, 5:12, 6:4, 7:2** | **0** |
| m29 | 126 | 0:26, 1:16, 2:16, 3:44, 4:16, 5:6, 6:2 | 26 | 22 | **4:6, 5:6, 6:8, 7:2** | **0** |
| m31 | 220 | 0:32, 1:22, 2:36, 3:32, 4:20, 5:30, 6:34, 7:14 | 32 | 44 | **4:4, 5:22, 6:10, 7:8** | **0** |
| total | 756 | | **146** | 106 | | **0** |

Separation index `sigma = s/u` on the hard runs, min/median/max: 0.600/0.600/0.600 (m17),
0.667/0.667/0.667 (m19), 0.571/0.714/1.000 (m23), 0.500/0.625/0.875 (m29), 0.444/0.625/0.778
(m31). The used number `u` is the whole gear set at almost every hard run (`u = 6` at 16 of 16
at m19, `u = 8` at 20 of 22 at m29, `u = 9` at 38 of 44 at m31): an extremal run's two flanks
use every gear the machine has.

**Not one of the 106 hard attaining runs is separable.** Separability lives entirely on the
trivial runs, which the peel bound already discharges.

### 2.2 The family, scored the same way (item 1)

200 random symmetric-tooth members per machine, full periods, the real member scored as one of
them (`results/sep_family.txt`):

| | m13 (179 members) | m17 (200) | m19 (200) |
|---|---|---|---|
| REAL trivial: mean `s` | 1.67 | 1.86 | 1.41 |
| family trivial: pooled mean `s` | 1.04 (4271 runs) | 1.34 (10409) | 1.68 (17162) |
| real percentile in mean `s` (trivial) | 91.1 | 92.5 | 22.5 |
| REAL **hard**: mean `s` | none (no hard runs) | 3.00 (4 runs) | 4.00 (16 runs) |
| family **hard**: pooled mean `s` | none | 3.34 (698 runs, 148 members) | 4.20 (2134 runs, 198) |
| real percentile in mean `s` (hard) | none | **11.5** | **22.2** |
| real percentile in mean `sigma` (hard) | none | **10.8** | **21.7** |
| separable hard runs | none | family **0/698**, real 0/4 | family **0/2134**, real 0/16 |

The real machine is mildly more separable than a random member on the hard runs (11th and 22nd
percentile) and mildly LESS separable on the trivial runs at m13 and m17 (91st, 92nd
percentile). Nowhere near the 99.6th percentile that gluability sits at. **Separability does not
carry face C's exception.**

### 2.3 Separability does not explain gluability (item 1, the decisive cross-tab)

Hard attaining runs, `(s, C2 succeeds) -> count`:

| machine | cross-tab |
|---|---|
| m17 | (3, no): 2, (3, yes): 2 |
| m19 | (4, no): 6, (4, yes): 10 |
| m23 | (4, no): 2, (5, no): 10, (5, yes): 2, (6, yes): 4, (7, yes): 2 |
| m29 | (4, no): 6, (5, no): 6, (6, no): 8, (7, no): 2 |
| m31 | (4, no): 2, (4, yes): 2, (5, no): 16, (5, yes): 6, (6, no): 10, (7, no): 6, (7, yes): 2 |

At m17 and m19 the shared number is CONSTANT across the hard runs (3 and 4) while C2 succeeds at
half and at 10 of 16: zero discriminating power. At m23 the association runs the wrong way --
C2 succeeds at 0 of 2 when `s = 4`, 2 of 12 when `s = 5`, 4 of 4 when `s = 6` and 2 of 2 when
`s = 7`. The implication `s = 0 => C2` holds (146 of 146 separable runs glue) but is vacuous
where it matters, because `s = 0` never happens on a hard run.

### 2.4 Why `s = 0` never happens: the counting floor (item 2's mechanism)

Let `a_g`, `b_g` be the number of left- and right-flank columns gear `g` strikes. Separation
needs a partition `G = A u B` with `sum_A a_g >= L-1` and `sum_B b_g >= R-1`, because each flank
column must be taken by a gear on its own side. That is a knapsack test on COUNTS alone; if no
partition passes it, `s = 0` is impossible for a counting reason and no arrangement of teeth
could rescue it (`results/sep_floor.txt`):

| machine | hard runs | counting alone forbids separation | actually separable |
|---|---|---|---|
| m17 | 4 | **4** | 0 |
| m19 | 16 | **16** | 0 |
| m23 | 20 | **20** | 0 |
| m29 | 22 | **22** | 0 |
| m31 | 44 | **42** | 0 |
| family m17 (100 members) | 294 | **294** | 0 |
| family m19 (100 members) | 1099 | **1099** | 0 |

**104 of 106 real hard runs and 1393 of 1393 family hard runs are counting-infeasible**; the two
m31 exceptions pass the count and are still not separable. The reason is one number: covering a
flank of length `n` needs gears of total capacity `sum 2/g >= 1`, and covering both flanks from
disjoint sets needs 2, while the machine's whole capacity is

`S(M) = sum_{g in M} 2/g = 1.0214, 1.1390, 1.2443, 1.3312, 1.4002, 1.4647` at m13..m31,

always below 2 and rising only like `log log y`. So separability of two long flanks is
impossible in every machine of this size, real or counterfactual. It is a face-A obstruction,
not a face-C one, which is exactly why the quantity cannot be where the real teeth are atypical.

### 2.5 Which gears are shared, and they are NOT the movable ones (item 2)

For each run every minimum shared set was enumerated and intersected. On the hard attaining runs:

| machine | gears in EVERY minimum shared set of EVERY hard run | per-gear forced count (out of `n` runs) |
|---|---|---|
| m17 (4) | **5, 7, 11** | 5:4, 7:4, 11:4 |
| m19 (16) | **5, 17** | 5:16, 7:14, 11:10, 13:4, 17:16, 19:0 |
| m23 (20) | **5** | 5:20, 7:14, 11:8, 13:18, 17:18, 19:18, 23:10 |
| m29 (22) | **5, 7** | 5:22, 7:22, 11:10, 13:18, 17:14, 19:6, 23:12, 29:10 |
| m31 (44) | **7** | 5:42, 7:44, 11:42, 13:30, 17:36, 19:18, 23:6, 29:14, 31:2 |

**At every one of the 106 hard attaining runs at m17..m31, at least one of gears 5 and 7 is
forced into every minimum shared set (106 of 106).** The band split, forced slots per
(run, gear) cell: the three smallest gears 1.000, 0.833, 0.700, 0.818, 0.970 at m17..m31 against
the three largest 0.333, 0.417, 0.767, 0.424, 0.167. At m31 the top gear is forced at 2 of 44
runs while gear 7 is forced at 44 of 44; m23 is the one machine where the top band is forced
more often than the bottom.

**P5 is refuted exceptionlessly in the opposite direction: at 0 of 106 hard attaining runs does
any minimum shared set consist only of movable gears** (letters `v = +-d_g` or pads `g | v`).
The shared gears are precisely the gears the move lemma does NOT allow to move. On the trivial
runs an all-movable minimum shared set exists at 258 of 708 runs. So the real machine's runs are
not "separable because the shared gears are movable": the two properties are disjoint.

### 2.6 The teeth: the one-third separation MAXIMISES sharing (item 3)

*The exact condition, verified 2702 of 2702 (run, gear) cells at m13..m23.* Take offsets from
`x_0+1`; the left flank is the arc `A = [0, L-2]` and the right flank the arc
`B = [L+v, L+v+R-2]`, both read mod `g`. With teeth `{t, t+D}` and phase
`alpha = (t - x_0 - 1) mod g`, gear `g` strikes both flanks iff

`( [alpha]_A or [alpha+D]_A )  and  ( [alpha]_B or [alpha+D]_B )`.

When the whole run fits in one gear period (`L+v+R-1 <= g`) the two arcs are disjoint and this
says: one tooth in each arc, so the separation must lie in the window `W = [v+2, L+v+R-2]`, and
a gear with `D < v+2` and `g-D > L+v+R-2` can never strike both. **That exemption is empty for
the real machine on record-class runs**: with `D = (g +- 1)/3` and `g - D = (2g -+ 1)/3` it needs
`g < 3(v+2)` and `g > 1.5(L+v+R-2)` at once, i.e. `v > L+R-6`, which is the peel-bound region;
and at m29/m31 the top gear (29, 31) is smaller than `L+v+R` (about 55) at every extremal run,
so every gear wraps and the "different arcs" picture does not apply at all.

*The sweep.* For each gear and each separation `D = 1..(g-1)/2`, the exact fraction of phases at
which the gear strikes both flanks of the observed `(L, v, R)`. Hard attaining runs:

| machine | gear | `D` real (range) | P(strike both), real | same, averaged over `D` | mid-rank of real `D` |
|---|---|---|---|---|---|
| m19 | 11 | 4 (1..5) | 0.943 | 0.880 | 0.662 |
| m19 | 13 | 4 (1..6) | 0.856 | 0.800 | 0.646 |
| m19 | 17 | 6 (1..8) | 0.772 | 0.670 | **0.766** |
| m19 | 19 | 6 (1..9) | 0.678 | 0.606 | 0.715 |
| m23 | 13 | 4 (1..6) | 0.985 | 0.956 | 0.583 |
| m23 | 17 | 6 (1..8) | 0.941 | 0.864 | 0.713 |
| m23 | 19 | 6 (1..9) | 0.884 | 0.805 | 0.678 |
| m23 | 23 | 8 (1..11) | 0.817 | 0.722 | 0.709 |

and in the aggregate, expected shared gears per run `E[ov] = sum_g P(both)`:

| machine | runs | real teeth | random separation | best possible separation |
|---|---|---|---|---|
| m13 all | 48 | 2.154 | 2.006 | 1.676 |
| m17 all / hard | 90 / 4 | 2.465 / 4.325 | 2.290 / 4.075 | 1.853 / 3.437 |
| m19 all / hard | 124 / 16 | 2.450 / 5.249 | 2.275 / 4.955 | 1.855 / 4.237 |
| m23 all / hard | 188 / 20 | 4.144 / 6.627 | 3.861 / 6.332 | 3.153 / 5.597 |

**The real separation is above the family average at every gear of every machine, and the real
machine's expected sharing exceeds a random member's at every machine and on both run classes;
on the hard runs the real value is at or above the mean over separations at 256 of 256
(run, gear) cells.** The reason is elementary once seen: the admissible separations are
`1..(g-1)/2`, the real one is `(g +- 1)/3`, so the real teeth sit at two thirds of the way to
the widest possible separation, and wide teeth are MORE likely to put one tooth in each flank.
The one-third separation is a LARGE separation, not a small one. Item 3's hypothesis is refuted
in the opposite direction.

### 2.7 The record across rungs: the top gears are the ones that are NOT shared (item 4)

Each record stretch read as a `J`-run of every lower machine (`results/sep_record.txt`;
positions from R3.h, re-verified here -- ends open, interior blocked). Top layer of each record
(where the record is a 3-run of the machine below):

| machine | record `x` | layer | pieces | flanks `(g_1, g_J)` | `s` | minimum shared set | SEP-c loss |
|---|---|---|---|---|---|---|---|
| m19 | 110 | 17 | 7, 18 | (7, 18) | 4 | 5, 7, 11, 13 | 6 |
| m19 | 26045 | 17 | 7, 13, 5 | (7, 5) | 2 | 5, 11 | 4 |
| m23 | 12694428 | 19 | 4, 8, 15, 7 | (4, 7) | 2 | **5, 17** | 3 |
| m23 | 18165208 | 19 | 4, 8, 15, 7 | (4, 7) | 1 | **13** | 2 |
| m29 | 200906185 | 23 | 10, 10, 23 | (10, 23) | 4 | **5, 13, 17, 19** | 9 |
| m29 | 877375977 | 23 | 23, 10, 10 | (23, 10) | 4 | **5, 13, 17, 19** | 9 |
| m31 | 1468940242 | 29 | 23, 10, 25 | (23, 25) | 4 | **5, 7, 13, 17** | 22 |
| m31 | 21844264615 | 29 | 18, 10, 30 | (18, 30) | 4 | **5, 7, 13, 17** | 17 |

**P8 is refuted, and its converse holds: at the top layer of every record stretch computed
(10 of 10 at m17..m31), the layer's own top gear is NOT in the minimum shared set.** At m29 the
shared set is `{5, 13, 17, 19}` and gear 23 is free; at m31 it is `{5, 7, 13, 17}` and gears 19,
23, 29 are all free. Read the other way: the gears R3.h says do the gluing (19+23+29 at m29,
23+29+31 at m31) are exactly the gears that carry no sharing obligation, so they are the ones
available to serve one flank alone or to be spent on a junction. "Made at the top" and
"separable at the top" are the same fact, with the sign opposite to the branch's guess: the
record is separable in its top band and inseparable in its bottom band, and it is inseparable at
the bottom because of the counting floor of 2.4.

Down the layers the shared number is 0 or 1 at the bottom and reaches 4 at the top at every
record: `s = 0, 0, 0, 2, 2, 4, 4` at layers 5, 7, 11, 13, 17, 19, 23 for m29's first record, and
`s = 0, 1, 0, 0, 2, 2, 4, 4` at layers 5..29 for m31's. **A record's pieces are separable while
they are short and become inseparable exactly when they become long** -- the counting floor
again. (At m19's first record the top layer has only two pieces, so its "3-run" is a 2-run with
`v = 0`: the pair-statement configuration.)

One exact connection, new: **the m31 record class at `x = 21,844,264,615` has layer-29 word
`(18, 10, 30)`, and `21844264615 mod 1078282205 = 278620515`** -- so the 3-run of `{5..29}` that
resists every glue certificate in 2g.i.a is precisely the decomposition of an m31 record. The
run that no local certificate can prove is the run the next gear fuses into the record.

### 2.8 The letter gears: same count, different size (item 2, following the clue)

For the real teeth `3 d_g = 6 u_g = 1 (mod g)`, so `v = +- d_g (mod g)` iff `3v = -+ 1 (mod g)`:

**Leg_real(v) = { g : g | 3v-1 or g | 3v+1 }, minus the pads.** Verified for every `v = 1..399`
against gears 5..199: **400 of 400.**

So the real machine's letter gears of a middle `v` are the prime factors of two specific
integers just above `3v`; for `v = 6..13` those integers are `17, 19 | 20, 22 | 23, 25 | 26, 28 |
29, 31 | 32, 34 | 35, 37 | 38, 40`, and `Leg(v)` is `{17,19}, {5,11}, {5,23}, {7,13}, {29,31},
{17}, {5,7}, {5,19}`. A random member has `v` as a letter of `g` with probability about `2/g`,
so its letter gears are typically SMALL. Same expected count (2g.i.a's D5, cited, not
re-derived); different size distribution.

The move cost of the letter set, `W(v) = sum_{g in Leg(v)} 2/g` (small = the movable gears hold
few flank columns, so moving one is cheap), real against 5000 random-separation draws at m31:

| `v` | `3v-1`, `3v+1` | `Leg_real` | `W` real | `W` family mean | `P(fam < real)` |
|---|---|---|---|---|---|
| 6 | 17, 19 | 17, 19 | 0.223 | 0.399 | 0.29 |
| 7 | 20, 22 | 5, 11 | 0.582 | 0.304 | 0.86 |
| 8 | 23, 25 | 5, 23 | 0.487 | 0.402 | 0.62 |
| 9 | 26, 28 | 7, 13 | 0.440 | 0.396 | 0.57 |
| 10 | 29, 31 | 29, 31 | 0.134 | 0.201 | 0.41 |
| 11 | 32, 34 | 17 | 0.118 | 0.362 | 0.24 |
| 12 | 35, 37 | 5, 7 | 0.686 | 0.403 | 0.80 |
| 13 | 38, 40 | 5, 19 | 0.505 | 0.375 | 0.69 |

The direction is not consistent: `v = 6, 10, 11` (where `3v +- 1` is prime and large) give the
real machine a cheap, top-band letter set, and `v = 7, 12` (where `3v +- 1` factors into small
primes) give it an expensive one. Against 2g.i.a's per-`v` glue rates on hard runs at m23
(`v=6`: 16 of 40, `v=7`: 6 of 88, `v=8`: 2 of 40) the ordering matches -- cheap letters glue,
expensive letters do not -- but `W` is above the family mean as often as below, so it cannot be
the source of a 99.6th percentile either.

### 2.9 The two graded certificates and their loss (item 5)

SEP-c (this branch's certificate, on the real flanks) and GLUE-c (2g.i.a's C2, graded by the
longest covered block around the hole), on the attaining 3-runs (`results/sep_loss.txt`):

| machine | trivial: GLUE-c loss | hard: SEP-c loss | hard: GLUE-c loss | hard: max, mean GLUE-c |
|---|---|---|---|---|
| m13 | 0 at 48 of 48 | none | none | none |
| m17 | 0 at 86 of 86 | 5-6 | 0:2, 1:2 | 1, 0.50 |
| m19 | 0 at 108 of 108 | 6-9 | 0:10, 2:2, 4:4 | 4, 1.25 |
| m23 | 0 at 168 of 168 | 8-13 | 0:8, 1:6, 3:4, 5:2 | 5, 1.40 |
| m29 | 0 at 126 of 126 | 11-25 | 2:2, 4:2, 5:2, 8:2, 9:4, 10:2, 13:2, 14:2, 17:2, 18:2 | **18**, 9.91 |
| m31 | 0 at 220 of 220 | 6-30 | 0:10, 1:4, 2:2, 3:2, 6:8, 9:6, 10:2, 11:6, 19:2, 24:2 | **24**, 6.55 |

`SEP-c loss = 0` exactly when `s = 0`: **862 of 862 attaining runs** (an instrument check on the
definitions). **The loss is not bounded: it grows with the machine** -- max GLUE-c loss
1, 4, 5, 18, 24 at m17..m31, a third of the run at m29. P9 refuted.

At the resistant run, m29 `(18, 10, 30)` at `x_0 = 278,620,515`: `s = 4`, `u = 8`,
`sigma = 0.500`, `ov = 8` (every gear strikes both flanks), minimum shared set `{5, 7, 13, 17}`
with move-classes pad, stuck, stuck, stuck; **SEP-c loss 17** (certifies `F_2 >= 31`) and
**GLUE-c loss 8** (certifies `F_2 >= 40`). Both fall short of `F = 43`, let alone of the
`L + R = 48` the run needs. A bounded-loss glue lemma would in any case buy nothing: a
certificate of loss `c` gives `N(v) <= F_2 + c` and hence `Q*_3 <= F_2 + c + b`, so the chain
budget needs `F_2 + c - F <= a`, and `F_2 - F - a` is already `0, -1, +1, -2, -5, +2, -2` at
m11..m31 (2g.i.a section 5). Any `c > 0` makes two more rungs fail. **A bounded-loss glue lemma
is both false and useless.**

### 2.10 The face-C exception itself, re-measured at matched cells (the deflation)

Following the clue means checking what the 99.6th percentile is a percentile OF. 2g.i.a pools
the family's hard attaining runs over ALL their `(v, L+R)` cells and compares that pool with the
real machine's own cells. Listing the real machine's hard attaining runs shows how few cells
there are (`results/sep_confound.txt`, and the enumeration with mirrors):

| machine | hard attaining runs | mirror pairs | distinct `(L, v, R)` up to mirror | C2 |
|---|---|---|---|---|
| m17 | 4 | 2 | **1**: (10,6,7) | ok at 2, FAIL at 2 -- the SAME shape at different positions |
| m19 | 16 | 8 | **3**: (7,6,15), (12,6,10), (10,7,18) | ok at all 8 runs of (7,6,15); FAIL at all 8 of the other two |
| m23 | 20 | 10 | **5**: (14,6,17), (12,7,23), (10,7,25), (9,8,23), (15,11,12) | ok at all 8 runs of (14,6,17), FAIL at the other 12 |

So the real machine's "62.5% at m19" is *one shape out of three glues*, carried by 8 runs that
are 4 mirror pairs of a single `(L, v, R)`; and "40% at m23" is again one shape out of five.
These are not 16 or 20 independent trials.

Matched comparison, 200 random members per machine, restricted to the cell the real machine
actually occupies (`results/sep_confound.txt`):

| machine | cell `(v, slack = F_2 - (L+R))` | family C2 at that cell | real C2 at that cell | family pooled over all cells |
|---|---|---|---|---|
| m17 | `(6, 8)` | 2 of 8 = 25.0% | 2 of 4 | 32 of 201 = 15.9% |
| m19 | `(6, 9)` | 6 of 20 = 30.0% | 10 of 14 | 24 of 216 = 11.1% |
| m19 | `(7, 3)` | 0 of 10 = 0.0% | 0 of 2 | |

**At its own cell the real machine glues at 10 of 14 against a family 30.0%, not against 9.4%.**
Counting mirror pairs as one trial that is 5 of 7 against `p = 0.30`, i.e. `P = 0.029` -- around
the 97th percentile on seven trials, not the 99.6th on 223 members. The family's low pooled rate
is largely made of cells the real machine does not occupy: at m19 the family's per-slack rates
run 0%, 0%, 0%, 10%, 27%, 0%, 0%, **29%**, 29%, 18%, 0%, 0%, 0% at slack 0..15, and the real
machine sits at slack 9, one of the three highest.

So face C's "first exception" is, on this measurement, a factor of about 2.4 at one cell with
seven independent trials, not a factor of 6.6 at the 99.6th percentile. It is not refuted -- the
real machine is still above its matched family cell at both machines -- but it is much smaller
than the number on record, and it has no separability mechanism behind it.

Sample-size caveat, stated plainly: the matched family cell at m19 holds 20 runs (10 mirror
classes) from 200 members, and the real machine contributes 14 runs (7 mirror classes). A wider
sweep (1500-3000 members) was attempted and abandoned on the lane's compute budget; the cell rate
is therefore known to about +-10 points. What is NOT sample-limited is the cell structure itself:
the real machine has 1, 3 and 5 distinct hard shapes at m17, m19, m23, and that alone shows the
recorded percentile is not comparing like with like.

## 3. Mechanism, stated once

**A flank is covered by the bottom of the gear set; only the top of the gear set is free to be
assigned.** Gear `g` strikes about `2n/g` of the `n` columns of a flank, so a set of gears can
cover one flank only if `sum 2/g >= 1` over that set, and two disjoint covers need 2. The whole
machine has

`S(M) = sum_{5 <= g <= y} 2/g = 1.02, 1.14, 1.24, 1.33, 1.40, 1.46` at m13..m31,

and `S` first exceeds 2 at **y = 109** (`S = 2.0152`). So for every machine we can compute -- and
for every machine up to `{5..107}`, real or counterfactual -- the strikers of two long flanks
**cannot** be separated: `s >= 1` always, and in fact 3 to 7 on the extremal runs. The knapsack
form of this test already forbids separation at 104 of 106 real hard runs and 1393 of 1393
family hard runs.

Three consequences follow, and they are the whole of this branch:

1. **Separability is not a which-residues property at this size; it is a capacity property.**
   Its value is pinned by `S(M)`, which is the same for every member of the family. That is why
   the real machine sits at the 11th-22nd percentile in it and not in the tail: there is nothing
   for the teeth to be atypical about.
2. **What can be assigned is the top band.** Gear 5 alone must serve both flanks of every
   extremal run (forced at 104 of 106); gears 5 and 7 between them are forced at 106 of 106. The
   top gears strike a flank once or twice and are free. So the two flanks separate at the top and
   are welded at the bottom -- and the top gears are exactly the mortar R3.h names. "Made at the
   top" and "separable at the top" ARE the same fact, with the sign reversed from the branch's
   guess: the record's gluing gears are free of sharing precisely because they are too big to be
   needed twice.
3. **The one-third separation works the wrong way for sharing.** Admissible separations run
   `1..(g-1)/2`; the real one is `(g +- 1)/3`, two thirds of the way to the maximum. Wide teeth
   straddle the two flanks more often, so the real teeth share MORE than a random member's, at
   every gear of every machine and at 256 of 256 (run, gear) cells on the hard runs. The picture
   in which "a stretch shorter than `g/3` meets at most one tooth" needs the run to fit inside one
   gear period; at the extremal runs `L+v+R` is about 55 while the top gear is 29 or 31, so every
   gear wraps and that picture does not apply.

And the reason gluability is not separability: the glue never asks the left flank to be blocked
by left gears. Offset `j` below the hole is covered by a left gear through the column `x_0+1+j`
OR by a right gear through the column `x_0+1+j+v`, which is a different column. Its content is
one column (the shadow lemma) and its currency is the move lemma -- both cited from 2g.i.a, not
re-derived. Separability is the strictly stronger demand, and being stronger is what puts it
below the counting floor.

## 4. What is new

1. **The shared number, the used number and the separation index**, defined and computed exactly
   (minimal-cover enumeration, not a heuristic) for all 862 attaining 3-runs with `v >= 6` at
   m13..m31, for 10 record stretches at every layer, and for 34,674 attaining runs of 579 family
   members. Not on record anywhere.
2. **`s = 0` never happens on an extremal run**: 0 of 106 real hard attaining runs, 0 of 2832
   family hard attaining runs at m17 and m19. Separability lives only on the trivial runs (146 of
   756), which the peel bound already discharges.
3. **The counting floor, with its crossing point.** Two long flanks can be separated only if
   `sum_g 2/g >= 2`; that sum is 1.02 to 1.46 at m13..m31 and first exceeds 2 at `y = 109`. The
   knapsack version of the test forbids separation at 104 of 106 real hard runs and 1393 of 1393
   family ones. So the quantity is a face-A object, not a face-C one -- which is why it cannot
   carry the exception it was opened to explain.
4. **At least one of gears 5 and 7 is in EVERY minimum shared set of EVERY hard attaining run:
   106 of 106** at m17..m31; and the top gear is forced at 0 of 16 (m19) and 2 of 44 (m31). The
   flanks separate at the top and weld at the bottom.
5. **The shared gears are never the movable ones: 0 of 106** hard attaining runs has a minimum
   shared set consisting only of letter or padded gears. The move lemma's alphabet and the
   sharing obstruction are disjoint phenomena.
6. **The exact arc condition** for a gear to strike both flanks (verified 2702 of 2702 cells),
   with the window `W = [v+2, L+v+R-2]` in which the tooth separation must lie when the run fits
   inside one gear period, and the proof that the real teeth get no exemption from it on
   record-class runs (it would need `v > L+R-6`, the peel-bound region).
7. **The one-third separation maximises, not minimises, expected sharing** -- above the mean over
   separations at every gear of m13..m23 and at 256 of 256 (run, gear) cells on the hard runs;
   `E[ov]` per run 2.15 vs 2.01 (m13), 2.47 vs 2.29 (m17), 2.45 vs 2.28 (m19), 4.14 vs 3.86
   (m23). The one-third separation is a WIDE separation (two thirds of the maximum admissible).
8. **The divisor form of the letter set: `Leg_real(v) = {g : g | 3v-1 or g | 3v+1}`**, exact
   (from `3 d_g = 1 mod g`), verified 400 of 400 for `v = 1..399` over gears 5..199. The real
   machine's letter gears of a small middle are the prime factors of two specific integers just
   above `3v`, hence large and few when `3v +- 1` is prime (`v = 6: {17,19}`, `v = 10: {29,31}`)
   and small when it is not (`v = 7: {5,11}`, `v = 12: {5,7}`) -- the same expected count as the
   family (2g.i.a's D5) with a different size distribution. The per-`v` glue rates of 2g.i.a
   follow that split, but `W(v) = sum_{Leg} 2/g` is above the family mean as often as below.
9. **The record's shared set excludes its own top gear at 10 of 10 record stretches** at
   m17..m31, and the shared number rises monotonically with the layer (0, 0, 0, 2, 2, 4, 4 at
   m29; 0, 1, 0, 0, 2, 2, 4, 4 at m31). The record is separable while its pieces are short and
   inseparable exactly when they become long.
10. **The resistant run is a record.** `21,844,264,615 mod 1,078,282,205 = 278,620,515`: the m31
    record class whose layer-29 word is `(18, 10, 30)` is, read in `{5..29}`, exactly the 3-run
    at `x_0 = 278,620,515` that resists every certificate in 2g.i.a. The one run no local
    certificate can prove is the one the next gear fuses into the record above.
11. **The graded certificates and the unbounded loss.** SEP-c loss `= 0` iff `s = 0` (862 of
    862); GLUE-c loss on hard attaining runs has max 1, 4, 5, **18**, **24** at m17..m31 and mean
    9.91 at m29 -- the loss grows with the machine, so no bounded-loss glue lemma exists; and
    even if it did, loss `c` gives `N(v) <= F_2 + c` and the chain budget needs `F_2 + c - F <= a`,
    already false at two rungs with `c = 0`.
12. **The face-C exception re-measured at matched cells.** The real machine's hard attaining runs
    occupy 1, 3 and 5 distinct `(L, v, R)` shapes up to mirror at m17, m19, m23, and its "62.5%"
    is one shape of three. At the matched `(v, slack)` cell the family glues at 30.0% (m19) and
    25.0% (m17) rather than 9.4%; the real machine's 5 of 7 independent mirror classes against
    `p = 0.30` is `P = 0.029`. The exception survives in direction and shrinks from a factor 6.6
    to a factor 2.4.

### 4a. Exceptionless statements, with counts

- **X1.** `s = 0` implies C2 succeeds and SEP-c loss `= 0`. **146 of 146** separable attaining
  runs, m13..m31. (Proof in section 0; the verification is an instrument check.)
- **X2.** SEP-c loss `= 0` if and only if `s = 0`. **862 of 862** attaining 3-runs with
  `v >= 6`, m13..m31.
- **X3.** No hard attaining run is separable. **0 of 106** (real, m17..m31) and **0 of 2832**
  (family, m17 and m19, 400 members drawn).
- **X4.** At least one of gears 5, 7 lies in every minimum shared set of every hard attaining
  run. **106 of 106**, m17..m31.
- **X5.** No hard attaining run has a minimum shared set made only of movable gears (letters or
  pads). **0 of 106**, m17..m31.
- **X6.** The arc condition `([alpha]_A or [alpha+D]_A) and ([alpha]_B or [alpha+D]_B)` decides
  whether a gear strikes both flanks. **2702 of 2702** (run, gear) cells, m13..m23.
- **X7.** On hard attaining runs, `P(gear strikes both flanks)` at the real separation is at
  least the mean over all separations. **256 of 256** (run, gear) cells, m17..m23.
- **X8.** `Leg_real(v) = {g : g | 3v-1 or g | 3v+1}` minus the pads. **400 of 400** values
  `v = 1..399` over gears 5..199. (Proved: `3 d_g = 6 u_g = 1 (mod g)`.)
- **X9.** At the top layer of a record stretch, the layer's own top gear is not in the minimum
  shared set. **10 of 10** record stretches, m17..m31.
- **X10.** Every trivial attaining run has GLUE-c loss 0. **756 of 756** -- the peel bound,
  cited, not re-derived.

## 5. Toward the root

The branch was opened on the hope that the real machine's extremal runs are separable except for
a bounded number of movable gears, which would give a search-free proof of `N(v) <= F_2` and a
lever on the chain statement. Every part of that hope is measured false:

- separability never holds on an extremal run, for a counting reason that no machine below
  `{5..109}` can escape (section 3);
- the gears that must be shared are the bottom of the gear set and are never the movable ones
  (0 of 106);
- the graded version has unbounded loss (max 18 at m29, 24 at m31), and a bounded-loss glue
  lemma would not close the chain statement even if it existed, because loss `c` turns
  `F_2 - F <= a` into `F_2 + c - F <= a` and the margin `F_2 - F - a` is already
  `0, -1, +1, -2, -5, +2, -2` at m11..m31.

What the branch does leave for the root is a sharper statement of where the real machine's
freedom is. On an extremal run the bottom gears are spent on both flanks and the top gears on at
most one: measured, the three smallest gears are forced into every minimum shared set at 0.97 of
the (run, gear) cells at m31 while the three largest are forced at 0.17, and at m31 gears 23, 29,
31 are forced at 6, 14 and 2 of 44 runs (m23 is the one machine where this reverses). At the two
deep rungs the top band is the part of the gear set with assignment freedom, and it is the band
R3.h finds doing the gluing at a record's junctions and the band the move lemma can move when
`3v +- 1` is prime. The object to take
forward is therefore not the flank pair but **how many gears carry no sharing obligation**, which
is a question about how many gears strike a flank of length `F` at most once -- a counting
question about the gear set, i.e. the wall's open thin place 1 (count gears, not columns), not
this branch.

## 6. Verdict

- **The branch's theory T is DEAD, and the premise it was built on is refuted.** Gluability is
  not separability of the flanks: separability fails at every extremal run of every machine, real
  and counterfactual, for a capacity reason (`sum 2/g < 2` below `y = 109`), and the shared
  number has no discriminating power for C2 (constant across the hard runs at m17 and m19, and
  running the wrong way at m23).
- **Face C's first exception is smaller than recorded.** Measured at the cell the real machine
  actually occupies, its glue rate is 10 of 14 against a family 30.0% at m19 and 2 of 4 against
  25.0% at m17, on 3 and 1 distinct run shapes; the recorded 99.6th percentile compares one
  shape's rate with a pool over cells the real machine never visits. The exception survives in
  direction, at about a factor 2.4 on seven independent mirror classes.
- **Status of the node:** DEAD as a route. FACT for X1-X10 and for the divisor form of the
  letter set, which is the one piece of new arithmetic here and is worth carrying: it says which
  gears the chain law makes movable, in closed form, for the real teeth only.
- **What survives and where it goes.** The identity `Leg_real(v) = {g : g | 3v +- 1}` and the
  observation that the top band is the only part of the gear set with assignment freedom both
  point at the wall's thin place 1. The finding that the resistant m29 run is an m31 record
  closes a small loop: 2g.i.a's residual case is not an accident of that machine, it is the shape
  the next gear needs.

## 7. Scorecard, filled

| # | prediction | result |
|---|---|---|
| P1 | `s = 0` implies C2 and `c = 0` | **HELD but vacuous** -- 146 of 146 separable runs glue and have loss 0; every one of them is a trivial run the peel bound already covers |
| P2 | real: `s = 0` at >= 40%, `s <= 1` at >= 90% of hard runs | **REFUTED** -- `s = 0` at 0 of 106, `s <= 1` at 0 of 106; the hard-run values are 3 to 7 |
| P3 | family pooled `s = 0` below 15%, real above the 95th percentile | **REFUTED both ways** -- family `s = 0` at 0 of 2832 hard runs, and the real machine is at the 11.5th and 22.2nd percentile in mean `s`, i.e. inside the family, not in the tail |
| P4 | separability explains gluability | **REFUTED** -- `s` is constant (3, 4) across all hard runs at m17 and m19 while C2 varies; at m23 C2 succeeds at 0/2 for `s=4` and 2/2 for `s=7`, the wrong way round |
| P5 | shared gears are movable | **REFUTED exceptionlessly, in the opposite direction** -- 0 of 106 hard runs has an all-movable minimum shared set; the shared gears are the stuck ones |
| P6 | shared gears are the top band at >= 60% | **REFUTED** -- they are the bottom band: gear 5 or 7 forced at 106 of 106, the top gear forced at 0 of 16 (m19) and 2 of 44 (m31) |
| P7 | one third minimises expected sharing | **REFUTED in the opposite direction** -- the real separation is at or above the mean at 256 of 256 hard (run, gear) cells and above it at every gear of every machine; `E[ov]` real exceeds random at m13..m23 |
| P8 | the record's shared gears are its three top gears | **REFUTED, converse holds** -- the top gear is absent from the minimum shared set at 10 of 10 record stretches; the shared set is `{5,13,17,19}` at m29 and `{5,7,13,17}` at m31 |
| P9 | loss `c <= 4` everywhere | **REFUTED** -- max GLUE-c loss 1, 4, 5, 18, 24 at m17..m31; 8 at the resistant m29 run `(18,10,30)`, certifying only `F_2 >= 40` against `L+R = 48` |

## 8. Dead ends, each with its refuting instance

- **D1. Separability of the flanks as the cause of gluability.** Refuted at every level: 0 of 106
  hard runs separable; `s` constant while C2 varies at m17 and m19; the real machine at the 11th
  and 22nd percentile of the family in `s`. Refuting instance: m19, all 16 hard attaining runs
  have `s = 4` and C2 succeeds at 10.
- **D2. The graded quantity `sigma = s/u` as a discriminator.** It tracks `s` exactly (`u` is the
  whole gear set at almost every hard run), so it adds nothing: real percentiles 10.8 and 21.7
  against 11.5 and 22.2.
- **D3. "The shared gears are the movable ones."** 0 of 106. Refuting instance: m29 `(18,10,30)`,
  shared set `{5,7,13,17}` with classes pad, stuck, stuck, stuck.
- **D4. The one-third separation as a sharing-minimiser.** Refuted at 256 of 256 hard cells and at
  every gear of m13..m23; the real `D = (g +- 1)/3` is two thirds of the way to the maximum
  admissible separation, so it maximises straddling. Refuting instance: m23, gear 23,
  `D = 8` of `1..11`, `P(both) = 0.817` against a mean of 0.722 over separations.
- **D5. The record's three top gears as its shared gears.** Refuted at 10 of 10; they are its
  FREE gears.
- **D6. A bounded-loss glue lemma restricted to the real teeth.** The loss is unbounded in the
  machine (18 at m29, 24 at m31) and would not close the chain statement at any positive `c`.
  Refuting instance: m29 `(18,10,30)`, GLUE-c loss 8, SEP-c loss 17.
- **D7. `W(v) = sum_{Leg(v)} 2/g` (the move cost of the letter set) as the cause.** Above the
  family mean at `v = 7, 8, 9, 12, 13` and below at `v = 6, 10, 11`; no consistent direction.
