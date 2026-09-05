# The glue as a covering statement (branch 2g.i.a, prover, 2026-09-05)

Parent: node **2g.i** (the neighbour-sum profile, `research/proof/neighbour_profile.md`). What
spawned this branch: that branch's closing line -- the glue lemma is proved, the two-colouring
succeeds at 426 of 446 attaining 3-runs with `v >= 6` at m13..m23, and the named unproven
interaction is *for every 3-run with middle `v >= v_0` there is a two-colouring of the gears whose
CRT re-phasing blocks the glued target*. That condition is finite per machine, covering-theoretic,
uses which residues, sits at a modulus that grows with the machine (the product of the gears) and
quantifies over the machine's own 3-runs with no transfer -- the shape `research/proof/the_wall.md`
asks for.

What this branch can find that is not already known: the exact obstruction at the 20 residual runs
(named gear by gear, column by column), whether a wider certificate class closes them and at what
loss, the success rate over ALL 3-runs rather than the attaining ones, the rate at m29 and m31, the
family rate for the *glue* (the family rate for the *law* is on record, 94-98%), and whether the
same construction reaches depth 4.

**Verdict in one line: the covering statement is FALSE as a lemma, and the 95.5% that made it look
alive is 92% peel bound. Split off the runs the peel bound already discharges and the glue's own
success rate is 28% and falling with the machine (50, 62, 40, 0, 23% at m17..m31).**

## 0. Pre-registered (written before any new computation)

### Definitions fixed here

- Machine `M = {5..y}` with gear set `G`, `u_g = 6^{-1} mod g`, teeth `T_g = {u_g, -u_g} mod g`;
  gear `g` strikes column `k` iff `k mod g in T_g`. Period `P = prod_{g in G} g`.
- A **3-run** is four consecutive openings `x_0 < x_1 < x_2 < x_3` with
  `L = x_1 - x_0`, `v = x_2 - x_1`, `R = x_3 - x_2`. Its **outer sum** is `L + R`.
- **The glue (version B of 2g.i, restated as a covering condition).** Fix the 3-run. Put
  `T = L + R - 1` target offsets `j = 0 .. T-1` and the **hole** `h = L - 1`. For a colouring
  `sigma : G -> {left, right}` define the base `b_g = x_0 + 1` if `sigma(g) = left` and
  `b_g = x_0 + 1 + v` if `sigma(g) = right`. Offset `j` is **covered** iff
  `(b_g + j) mod g in T_g` for some `g`. The colouring **glues** iff every `j != h` is covered.
  (The hole is never covered: under the left base `h` is the column `x_1` and under the right base
  it is `x_2`, both openings. This is part (i) of the glue lemma.)
- **Why it is a certificate.** By CRT there is `z` with `z = b_g (mod g)` for every gear; then
  column `z + j` is blocked in `M` exactly when offset `j` is covered. So a glueing colouring
  proves `M` contains `L-1` blocked columns, an opening, `R-1` blocked columns, hence
  `F_2(M) >= L + R`. The condition uses only the residues `x_0 mod g` and needs no sieve.
- **Loss.** A certificate that instead exhibits an adjacent pair of total `L + R - c` proves
  `L + R <= F_2 + c`; `c` is its loss. `c = 0` is the target.
- **Certificate classes, in increasing strength.**
  - **C2** the two-colouring above (the glue).
  - **C2+f** two colours plus `f` **free** gears, each given an arbitrary residue `t_g in Z_g`
    (search over all `g` values), subject to not covering the hole. `f = |G|` is vacuous (it is
    the statement `F_2 >= L + R` itself), so only small `f` is a certificate.
  - **Cs** the shifted glue: right base `x_0 + 1 + s` for `s != v`. `s = v + t` (`t >= 1`)
    overlaps the flanks by `t` and, when it succeeds, certifies `F(M) >= L + R - 1 - t`, i.e.
    `L + R <= F + 1 + t`, which is stronger than the `F_2` form whenever `t < F_2 - F - 1`.
  - **Cx** the cross glue: the left flank of this run glued to the right flank of a DIFFERENT
    opening of `M` whose right gap is `R' >= R`; success certifies `F_2 >= L + R' >= L + R`.
- **The J-run outer law.** For `J` consecutive gaps `g_1 .. g_J` the same construction with right
  base `x_0 + 1 + (x_{J-1} - x_1)` puts the hole at `h = g_1 - 1` again, so a glueing colouring
  certifies `F_2 >= g_1 + g_J` for a pair of gaps separated by `J - 2` intervening gaps. `J = 3`
  is `N(v) <= F_2`.

### Theory

T. **The glue is a covering statement that holds for every 3-run with `v >= 6`, up to a bounded
loss.** The obstruction at a residual run is local and nameable: a gear that is the unique
available coverer of a column on each side and therefore is asked for both phases at once. Where
C2 fails, one of C2+1, Cs or Cx succeeds with loss 0, so the lemma "every 3-run with `v >= 6`
admits a certificate of loss `<= c`" is true with a small `c`.

### Predictions, with the number that would refute each

(Written before computing; results in section 8.)

- **G1 (instrument).** The covering formulation reproduces 2g.i exactly: 426 of 446 attaining
  3-runs with `v >= 6` at m13..m23, failures 0/2/6/12 at m13/m17/m19/m23 and all at
  `v in {6,7,8,11}`. REFUTED by any other count.
- **G2 (local obstruction).** At all 20 residual runs, unit propagation on the covering instance
  ends in a CONFLICT: one gear forced left and right. REFUTED if any residual run has no
  propagation conflict.
- **G3 (one free gear).** C2+1 rescues at least 15 of the 20 with loss 0. REFUTED below 10.
- **G4 (overlap).** `Cs` with `s = v + t` certifies `L + R <= F + 1 + t` at `t <= 3` for at least
  half the residual runs. REFUTED if the best `t` exceeds 5 at more than half.
- **G5 (all runs).** Over EVERY 3-run with `v >= 6` and `L + R > F` at m13..m23, C2 succeeds at
  `>= 90%`, and the best certificate has loss `0` everywhere. REFUTED below 75%, or by a run whose
  best loss exceeds 8.
- **G6 (m29 and m31).** On the attaining 3-runs with `v >= 6` at m29 and m31, C2 succeeds at
  `>= 90%`. REFUTED below 75%.
- **G7 (the family).** On 200 random symmetric-tooth members at m13, m17, m19 the C2 rate is
  within 5 points of the real machine's. REFUTED if the real machine is more than 10 points above
  the family median.
- **G8 (depth 4 and the outer law).** The `J`-run glue succeeds at a rate comparable to `J = 3`
  for `J = 4, 5`; the LAW `g_1 + g_J <= F_2` first fails at `J >= 5` at m17..m23. REFUTED if it
  fails at `J = 4`, or never up to `J = 8`.

Stop rules: any sub-question that reduces to L4 (docs/proofs/19), the chain law
(docs/proofs/05 (C)), the peel bound or the middle-sum lemma (alignment-rules 736-790) is stopped
in one line and cited, not re-derived.

## 1. Setup

All numbers are exact, on full periods, in integer arithmetic. Machines m13..m23 were sieved
whole in RAM; m29 (1,078,282,205 columns) and m31 (33,426,748,355 columns) in 3e7-column chunks by
four processes, each owning a contiguous range plus a 4096-column margin on both sides so that
every gap is counted exactly once (attributed by its left endpoint) and every 3-run is complete;
m29 took 12.1 s and m31 384.5 s at 4 cores. **The covering test itself needs no period at all**:
it reads only `x_0 mod g` for each gear, which is why the deep rungs are as cheap as the shallow
ones.

Scripts, `research/anchor235/r46/`: `gl_glue.py` (the covering core: masks, the exact solver, the
free-gear and shifted variants), `gl_gate.py` (G1 and the machine cross-check), `gl_resid.py`
(the 20 residuals, gear by gear, and the three alternatives), `gl_shadow.py` (the shadow lemma,
the sweep, the pinch), `gl_split.py` (the trivial/hard split), `gl_deep.py` (m29, m31),
`gl_move.py` (the move lemma), `gl_family.py`, `gl_fam2.py`, `gl_legfam.py` (the family),
`gl_sep.py`, `gl_jrun.py`, `gl_verify.py` (the outer law), `gl_m29.py`, `gl_cx.py` (deep and
cross-glue cases). Outputs in `research/anchor235/r46/results/` (gitignored).

Instrument check before any new claim: the covering formulation was cross-checked against a
direct machine lookup at the CRT point on 40 successes per machine -- for every one of them the
sieve at `z` shows exactly `L-1` blocked columns, an opening, `R-1` blocked columns. The two
definitions are the same object; the covering form is not an approximation of the machine test.
It returns `F = 11, 18, 25, 34, 43, 58` and `F_2 = 16, 25, 31, 39, 55, 68` at m13..m31, the
recorded ladder, and reproduces 2g.i's residual set exactly (section 2.1).

## 2. Results

### 2.1 The gate, and the four-run discrepancy

| machine | attaining 3-runs, `v >= 6` | C2 succeeds | C2 fails | failing `v` |
|---|---|---|---|---|
| m13 | 48 | 48 (100%) | 0 | - |
| m17 | 90 | 88 (97.8%) | 2 | `v = 6` |
| m19 | 124 | 118 (95.2%) | 6 | `v = 6` (4), `v = 7` (2) |
| m23 | 188 | 176 (93.6%) | 12 | `v = 7` (6), `v = 8` (4), `v = 11` (2) |
| total | 450 | 430 (95.6%) | 20 | |

2g.i reported 446/426 with the SAME 20 failures and the same per-`v` counts; the four extra
attaining runs are at m23 and are successes, so the residual set -- the object of this branch --
is reproduced exactly. G1 is a pass with a four-run bookkeeping difference noted.

### 2.2 The headline: 92% of that 95.6% is the peel bound

The covering condition has content only when `v < min(L, R)`. When `v >= R` the constant colouring
"all gears left" already glues (its pattern IS the run: blocked `x_0+1..x_1-1`, the opening `x_1`,
then `x_1+1..x_1+R-1`, all inside the middle gap); when `v >= L` "all gears right" glues. Both are
the peel bound `max(L,R) + v <= F_2` (alignment-rules 736-790, Theorem D) restated -- if
`v >= min(L,R)` then `L + R <= max(L,R) + v <= F_2` with no construction at all. **Cited, not
re-derived.** So split the attaining runs at that line:

| machine | attaining `v>=6` | trivial `v >= min(L,R)` | C2 there | **HARD `v < min(L,R)`** | **C2 there** |
|---|---|---|---|---|---|
| m13 | 48 | 48 | 48/48 | 0 | - |
| m17 | 90 | 86 | 86/86 | 4 | **2 (50.0%)** |
| m19 | 124 | 108 | 108/108 | 16 | **10 (62.5%)** |
| m23 | 188 | 168 | 168/168 | 20 | **8 (40.0%)** |
| m29 | 148 | 126 | 126/126 | 22 | **0 (0.0%)** |
| m31 | 264 | 220 | 220/220 | 44 | **10 (22.7%)** |
| total | 862 | 756 | 756/756 | 106 | **30 (28.3%)** |

Every trivial run succeeds -- necessarily, since the constant colouring works there. Every one of
the 20 residuals of 2g.i is hard. On the hard runs, which are the only ones carrying content, the
glue is a coin flip at m17-m23 and worse at the two deep rungs. At m29 it fails on **all 22**.

Widen from the attaining runs to every 3-run with a large outer sum and it gets worse, because the
proportion of hard runs rises with the machine:

| machine | condition | 3-runs `v>=6` | trivial | C2 on trivial | HARD | **C2 on hard** |
|---|---|---|---|---|---|---|
| m13 | `L+R > F = 11` | 12 | 12 | 12/12 | 0 | - |
| m17 | `L+R > F = 18` | 14 | 8 | 8/8 | 6 | 4 (66.7%) |
| m19 | `L+R > F = 25` | 8 | 0 | - | 8 | 2 (25.0%) |
| m23 | `L+R > F = 34` | 10 | 4 | 4/4 | 6 | **0 (0.0%)** |
| m29 | `L+R >= 35` | 4328 | 908 | 908/908 | 3420 | **264 (7.7%)** |
| m31 | `L+R >= 58 = F` | 220 | 18 | 18/18 | 202 | **8 (4.0%)** |
| m17 | `L+R > F-6` | 356 | 330 | 330/330 | 26 | 22 (84.6%) |
| m19 | `L+R > F-6` | 482 | 406 | 406/406 | 76 | 46 (60.5%) |
| m23 | `L+R > F-6` | 260 | 92 | 92/92 | 168 | **24 (14.3%)** |

At m23 the six 3-runs of the whole period with `L + R > F` and `v < min(L,R)` are ALL C2 failures:
on the runs whose outer sum already exceeds the record -- the only ones where the `F_2` cap is
saying anything the record does not -- the certificate delivers nothing.

**How much does the colouring ever do?** Recording the least number of gears re-phased away from
the constant colouring: over the 3-runs above the count is `0` (trivial) at 756 of 786 successes
and `1` at almost all the rest; `2` occurs twice at m17, `2` and `3` occur 74 times in m23's
`L+R>=26` sweep and 16 times in m29's. **The glue is not a covering argument with room in it: it
is "the run itself, plus at most one gear moved".**

### 2.3 The shadow lemma: what the covering problem actually is

*Lemma (proved, and verified exhaustively).* In the glue instance of a 3-run the target offsets
that only ONE side can cover are exactly two:

- `h + v`: under the left base this offset is the column `x_2`, an opening, so no left gear can
  ever cover it; under the right base it is `x_2 + v`. It lies in the target iff `v <= R - 1`.
- `h - v`: under the right base it is `x_1`, an opening; under the left base it is `x_1 - v`. It
  lies in the target iff `v <= L - 1`.

*Proof.* Offset `j` is invisible to the left iff `x_0 + 1 + j` is an opening. The openings in
`[x_0+1, x_0+T] = [x_0+1, x_1+R-1]` are `x_1` and `x_2` only (`x_3 = x_1 + v + R > x_1 + R - 1`
since `v >= 1`), giving offsets `h` and `h+v`. Symmetrically the right base `x_2 - L + 1` sees the
openings `x_2` (offset `h`) and `x_1` (offset `h - v`, present iff `x_2 - L + 1 <= x_1`, i.e.
`v <= L-1`); `x_0` would sit at a negative offset. Every other offset has at least one candidate on
each side. QED

Verified with **0 violations on 23,880 3-runs** (188 at m13, 2754 at m17, 10,288 at m19, 10,650 at
m23; every 3-run with `v >= 6` and `L + R > F - 12`).

Two consequences, both exact:

1. When `v >= min(L,R)` one shadow is outside the target and the constant colouring glues -- the
   trivial column of the table above. So the covering problem is exactly the problem of covering
   the TWO shadows at once, plus not losing anything else.
2. Since the all-left colouring covers everything except `h` and `h + v`, **at every C2 failure the
   minimum number of uncovered columns is 1** (checked: 178 of 178 failures at m17/m19/m23 with
   `L+R > F-6`), and at every one of those 178 failures some optimal colouring leaves a SHADOW
   uncovered (178 of 178). The glue's whole task is to buy one column.

### 2.4 The move lemma: why buying that column is hard

*Lemma (proved).* Recolouring one gear from left to right translates the columns it inspects by
exactly `+v`: at offset `j` the left base shows `c = x_0+1+j` and the right base shows `c + v`. So
`g`'s coverage of `j` survives the move iff `g` strikes both `c` and `c+v`, which (teeth `+-u_g`,
`d_g = 2u_g`) happens iff `v = 0, +d_g` or `-d_g (mod g)` -- **the chain law's alphabet**
(docs/proofs/05 (C)), whose least positive representatives are the letters `a_g, b_g` of gear `g`.
Hence

- `v = 0 (mod g)` (`v` padded at `g`): the move changes nothing at all -- and such a gear can never
  cover the right shadow, because `x_2 + v = x_2 (mod g)` and `x_2` is an opening;
- `v = +-d_g (mod g)` (`v` a LETTER of `g`): exactly one of the two teeth survives the move;
- otherwise: **not one strike of `g` survives** -- its coverage is replaced by a disjoint set.

*Corollary (where L4 bites).* If `L + v > F(M)` then by L4's sole-striker corollary (docs/proofs/19)
every gear is the sole striker of some column of `(x_0, x_2) \ {x_1}`, i.e. of some target offset
under the left base. Moving such a gear destroys that column unless the move is survivable, i.e.
unless `v` is a letter or a pad of that gear. So at an extremal run every move must be paid for,
and the payment cascades.

Measured, on the hard runs with `L+R > F-6` (`Leg(v) = {g : v = +-d_g mod g}`):

| machine | hard runs | C2 ok | of the successes, a shadow-striker in `Leg(v)` | runs WITH such a gear: C2 rate | runs WITHOUT: C2 rate |
|---|---|---|---|---|---|
| m17 | 26 | 22 | 13 (59%) | 13/15 (86.7%) | 9/11 (81.8%) |
| m19 | 76 | 46 | 39 (85%) | 39/52 (75.0%) | 7/24 (29.2%) |
| m23 | 168 | 24 | 18 (75%) | 18/85 (21.2%) | 6/83 (7.2%) |

Per middle size, with the legality that governs it:

| machine | `v` | hard runs | C2 ok | `Leg(v)` | `Pad(v)` |
|---|---|---|---|---|---|
| m17 | 6 | 16 | 14 | `{17}` | - |
| m17 | 7 | 10 | 8 | `{5, 11}` | `{7}` |
| m19 | 6 | 24 | 16 | `{17, 19}` | - |
| m19 | 7 | 48 | 30 | `{5, 11}` | `{7}` |
| m19 | 8 | 4 | 0 | `{5}` | - |
| m23 | 6 | 40 | 16 | `{17, 19}` | - |
| m23 | 7 | 88 | 6 | `{5, 11}` | `{7}` |
| m23 | 8 | 40 | 2 | `{5, 23}` | - |

So the letter condition is a real discriminator (a factor 2.6 at m19, 2.9 at m23) and nowhere near
sufficient. `v = 7` at m23 is the worst cell: `Leg(7) = {5, 11}` and gear 7 is padded, and only 6
of 88 hard runs glue.

### 2.5 The 20 residual runs, one by one

Every one of the 20 has `v < min(L, R)`; every one has several gears that are sole strikers on both
flanks (the hypothesis 2g.i refuted as D4 -- it is present at successes too, and the covering
instance shows why: a left-flank column solely struck by `g` can still be covered by a DIFFERENT
gear coloured right, which strikes the column `v` further on); every one has a unique uncovered
column, always a shadow. Full gear-by-gear tables in `results/resid.txt`; the summary:

| machine | `(L,v,R)` | `x_0` | left shadow `x_1-v` struck by | right shadow `x_2+v` struck by | C2+`f` | best `Cs` | `Cx` |
|---|---|---|---|---|---|---|---|
| m17 | (7,6,10) | 29055 | 5,7,17 | 5 | `f=1` | `t=5`, loss 0 | loss 0 |
| m17 | (10,6,7) | 56007 | 5 | 5,7,17 | `f=1` | `t=5`, loss 0 | loss 0 |
| m19 | (12,6,10) | 351300 | 5 | 5,7 | `f=1` | `t=5`, loss 0 | loss 0 |
| m19 | (12,6,10) | 724365 | 5,13 | 5,7 | `f=1` | `t=5`, loss 0 | loss 0 |
| m19 | (10,6,12) | 892222 | 5,7 | 5,13 | `f=1` | `t=5`, loss 0 | loss 0 |
| m19 | (10,6,12) | 1265287 | 5,7 | 5 | `f=1` | `t=5`, loss 0 | loss 0 |
| m19 | (10,7,18) | 118295 | 13 | 5 | `f=2` | `t=8`, loss 3 | loss 0 |
| m19 | (18,7,10) | 1498285 | 5 | 13 | `f=2` | `t=8`, loss 3 | loss 0 |
| m23 | (12,7,23) | 8083133 | 13 | 5 | **none** | **none** `t<=10` | loss 0 |
| m23 | (10,7,25) | 8480803 | 5 | 17 | **none** | `t=10`, loss 6 | loss 0 |
| m23 | (25,7,10) | 15578190 | 17 | 5 | **none** | `t=10`, loss 6 | mirror |
| m23 | (10,7,25) | 21603913 | 5 | 17 | **none** | `t=10`, loss 6 | loss 0 |
| m23 | (25,7,10) | 28701300 | 17 | 5 | **none** | `t=10`, loss 6 | loss 0 |
| m23 | (23,7,12) | 29098970 | 5 | 13 | **none** | **none** `t<=10` | loss 0 |
| m23 | (9,8,23) | 7052418 | 5,17 | 7,11 | `f=2` | `t=6`, loss 2 | loss 0 |
| m23 | (9,8,23) | 13636268 | 5,11 | 7 | `f=2` | `t=4`, loss 0 | loss 0 |
| m23 | (23,8,9) | 23545837 | 7 | 5,11 | `f=2` | `t=4`, loss 0 | loss 0 |
| m23 | (23,8,9) | 30129687 | 7,11 | 5,17 | `f=2` | `t=6`, loss 2 | loss 0 |
| m23 | (15,11,12) | 18159472 | 5,7 | 5 | `f=1` | `t=6`, loss 2 | loss 0 |
| m23 | (12,11,15) | 19022635 | 5 | 5,7 | `f=1` | `t=6`, loss 2 | loss 0 |

Read: **C2+1 closes 8 of 20 and C2+2 closes 6 more (14 of 20), all at loss 0; the six that resist
both are exactly m23's `v = 7` group.** The overlapped glue `Cs` gives loss 0 at 8 of 20, loss `<=3`
at 10, loss 6 at 4, and nothing at `t <= 10` at 2. **The cross glue `Cx` closes 19 of 20 outright at
loss 0** (a partner opening `y` with right gap `>= R` at which the same two-colouring works); the
twentieth, `(25,7,10)` at `x_0 = 15578190`, was not found in 60,000 sampled partners, but the bound
it needs (`F_2 >= 35`) is proved by its mirror image, and the opening set is closed under
`k -> -k` (file 03 (c)), so the bound is certified for all 20.

So on the shallow rungs the residual is closed at loss 0 -- by a certificate that searches the
period, which is a different kind of object from a certificate read off the run.

### 2.6 What resists at the deep rungs

`Cx` is out of reach at m29/m31 (the period cannot be scanned), so the certificate classes readable
off the run are all there is. Six deep residuals, chosen at the extremes:

| machine | `(L,v,R)` | `L+R` | `Leg(v)` | `Pad(v)` | right shadow struck by | movable shadow gear | C2 | C2+1..3 | `Cs` |
|---|---|---|---|---|---|---|---|---|---|
| m29 | (25,7,30) | 55 = `F_2` | `{5,11}` | `{7}` | `{11}` | `{11}` | FAIL | **`f=1` OK** | none `t<=12` |
| m29 | (18,10,30) | 48 = `F+5` | `{29}` | `{5}` | `{7,11}` | none | FAIL | **none** | **none** |
| m29 | (28,9,26) | 54 = `F_2-1` | `{7,13}` | - | `{5}` | none | FAIL | **none** | **none** |
| m29 | (32,6,15) | 47 | `{17,19}` | - | `{5}` | none | FAIL | **none** | **none** |
| m31 | (31,7,35) | 66 = `F_2-2` | `{5,11}` | `{7}` | `{19}` | none | FAIL | **none** | **none** |
| m31 | (47,6,12) | 59 = `F+1` | `{17,19}` | - | `{5}` | none | FAIL | **none** | `t=12`, loss 3 |

At all six, L4 is visible directly: 5 to 9 of the 8 gears (m29) or 9 gears (m31) are sole strikers
on the left flank, 5 to 8 on the right, and 4 to 7 on both. **The exact residual case that resists everything is
m29's `(18, 10, 30)` at `x_0 = 278620515`** -- the run that killed 2g.i's `F + 1` law
(`N(10) = 48 = F + 5`). Its middle `v = 10` is padded at gear 5 (so gear 5 cannot cover the right
shadow at all, by the move lemma) and is a letter only of gear 29 (which does not strike the
shadow); the shadow `x_2 + 10` is struck by 7 and 11 alone, and moving either destroys a column it
solely holds and starts a cascade that does not close. No colouring, no colouring with up to three
free gears, and no overlap `t <= 12` certifies it.

### 2.7 The family: the real teeth are NOT typical here

200 random symmetric-tooth members (teeth at `+-v_g`, `v_g` uniform in `1..(g-1)/2`; the
alignment-rules section 5 family), each on its full period, scored the same way as the real member:

| | m13 (180 members, whole family) | m17 (200) | m19 (200) |
|---|---|---|---|
| REAL: trivial attaining runs | 48/48 | 86/86 | 108/108 |
| REAL: **HARD attaining runs** | 0 of 0 | **2 of 4 (50%)** | **10 of 16 (62.5%)** |
| family pooled trivial | 4319/4319 (100%) | 10409/10409 (100%) | 17162/17162 (100%) |
| family pooled **HARD** | 4/29 (13.8%) | **100/698 (14.3%)** | **176/2134 (8.2%)** |
| members with a hard run | 13/180 | 148/200 | 198/200 |
| members below the real rate | 0/13 | 128/148 | 196/198 |
| members with a law exception (`N(v) > F_2`, `v>=6`) | 3/180 | 11/200 | 12/200 |

A fairer comparison at m19, restricting to the 223 of 400 members with at least 10 hard attaining
runs (so the rates are comparable): pooled hard rate 340/3608 = 9.4%, and **exactly 1 of the 223
members matches or beats the real machine's 62.5% -- the real machine sits at the 99.6th
percentile.** The trivial part is 100% everywhere, as it must be.

The obvious explanation was tested and REFUTED: the number of gears for which a small `v` is a
letter, `|Leg(v)|`, is entirely typical for the real machine (2000 random members per cell:
`P(|Leg| >= real)` is 0.80, 0.24, 0.83, 1.00 at m17 for `v = 6,7,8,10`, and 0.51, 0.40, 0.52, 0.32
at m31 -- the real value sits at the family mean). The small gears' teeth explain only a sliver:
grouping the m19 members by `v_5` gives 10% (`v_5 = 1`, the real value) against 7%, and by `v_7`
gives 11% (`v_7 = 1`, real) against 6-8%. So the real machine's advantage on this certificate is
measured and its cause is not found.

### 2.8 The outer law: what the construction really needs

The glue never uses that `x_1` and `x_2` are CONSECUTIVE openings, only that the columns strictly
between them are blocked. Two measurements pull in opposite directions.

**(a) Drop the middle entirely and the law dies at once.** Define
`N*(S) = max over openings p with p + S an opening of (leftgap(p) + rightgap(p+S))`, so
`N*(0) = F_2` and `N*(v) >= N(v)`. Least `S >= 6` with `N*(S) > F_2`: **12** at m13, **7** at m17,
**6** at m19, **6** at m23; and `max_{S>=6} N*(S) = 22, 30, 41, 54` against `F_2 = 16, 25, 31, 39`.
The C2 test fails at exactly those extremal pairs, as it must. So the `F_2` cap is not a statement
about two openings at a distance; it is a statement about a GAP between them.

**(b) Keep the middles but make them all wide and the law survives to depth 8.** For `J`
consecutive gaps with every one of the `J-2` middles `>= 6`, `max(g_1 + g_J)`:

| machine | `F_2` | `J=3` | `J=4` | `J=5` | `J=6` | `J=7` | `J=8` | runs tested |
|---|---|---|---|---|---|---|---|---|
| m13 | 16 | 12 (188) | 10 (24) | - | - | - | - | 212 |
| m17 | 25 | 21 (4126) | 15 (858) | 11 (102) | 6 (8) | - | - | 5094 |
| m19 | 31 | 28 (91264) | 21 (23966) | 15 (4272) | 11 (380) | 7 (6) | - | 119888 |
| m23 | 39 | 35 (2285006) | 34 (688546) | 34 (153472) | 31 (23898) | 25 (2360) | 18 (496) | 3153778 |

**0 exceptions in 3,278,972 `J`-runs**, and the maximum falls with `J`. Without the middle
condition it fails already at `J = 3` (max outer 18 at m13 against `F_2 = 16`, 46 at m23 against
39). G8 is therefore refuted in both directions: the law does not fail at `J >= 5`, it holds to
`J = 8`; what fails is the version with no condition on the middles, and it fails at `J = 3`.
The C2 certificate at the extremal `J`-run witnesses is OK at 14 of the 17 non-empty cells and
FAILS at m19 `J=3`, m23 `J=4` and m23 `J=5` -- so the level-4 and level-5 colourings fail at
exactly the two deepest witnesses the largest machine has.

## 3. Mechanism, stated once

The covering problem is not "two flanks that need different gears". It is one column.

1. The all-left colouring reproduces the run itself and covers every target offset except the hole
   `h` and the right shadow `h + v` (the column `x_2` under the left base). The all-right colouring
   is its mirror. So a glue is exactly a way of buying the shadow column without selling anything.
2. Recolouring a gear translates its strike pattern by `+v` (the move lemma). By the chain law a
   strike survives that translation only if `v` is `0` or a letter of that gear. A padded gear
   (`g | v`) moves for free but is useless -- it cannot strike `x_2 + v` because it does not strike
   `x_2`. A gear for which `v` is a letter keeps one tooth. Any other gear loses everything it held.
3. By L4, once `L + v > F(M)` every gear holds at least one column alone, so every move must be
   compensated, and the compensation is another move, which must itself be compensated. That
   cascade is what fails: at m23's `v = 7` cell only 6 of 88 hard runs close it, and at m29 none of
   the 22 hard attaining runs does.
4. Which is why the certificate never needs more than one move when it works at all (756 of 786
   successes use zero, i.e. are the peel bound; almost all the rest use one). There is no reservoir
   of colourings to draw on: the search space is `2^{|G|}` but the feasible region is a point.

The residue-level content, in one line: **the glue works iff the middle gap `v` is a letter of a
gear that strikes the column `x_2 + v`** -- and that is a coincidence between two residues
(`v mod g` in the chain-law alphabet, and `x_2 + v mod g` on a tooth), with no reason to hold and
measured probability falling with the machine.

## 4. The lemma and its residual

**What can be proved, exactly.**

- **(P1) The shadow lemma.** Proved above; the only single-sided offsets are `h +- v`. Exhaustively
  confirmed, 23,880/23,880.
- **(P2) The trivial case.** If `v >= min(L, R)` then `L + R <= F_2`. This is the peel bound
  (alignment-rules 736-790), and the constant colouring is its covering-language proof. It covers
  756 of the 862 attaining 3-runs with `v >= 6` at m13..m31.
- **(P3) The move lemma.** Proved above from the chain law; it classifies every recolouring into
  free-and-useless (`g | v`), half-surviving (`v` a letter of `g`) and total-loss.
- **(P4) The single-move criterion.** A glue with one move exists iff some gear `g` striking
  `x_2 + v` re-covers, at its right phase, every target column it alone held at its left phase.
  With L4 this says: at any run with `L + v > F(M)`, every gear holds a column alone, so the moved
  gear must satisfy `v = +-d_g (mod g)` and have all its sole columns on the surviving tooth.

**The lemma as it would have to be stated.** *For every 3-run of `M` with `v >= 6` and
`v < min(L, R)` there is a colouring `sigma` of the gears such that every column of the glued target
is covered.* **This is false.** Refuting instances at every rung from m19 up; the cleanest is m29's
`(18, 10, 30)` at `x_0 = 278620515` (section 2.6), which also resists three free gears and every
overlap to `t = 12`. The weakened form *"...admits a certificate of loss at most `c`"* is true with
`c = 0` at m13..m23 only because `Cx` is allowed, and `Cx` searches the period -- it is not a
certificate read off the run, so it does not reduce anything.

**The exact residual that resists.** A hard 3-run in which (i) the right shadow `x_2 + v` is struck
only by gears for which `v` is neither `0` nor a letter, and (ii) every such gear holds a target
column alone (guaranteed by L4 once `L + v > F`). Instance: m29 `(18, 10, 30)`, `Leg(10) = {29}`,
`Pad(10) = {5}`, shadow strikers `{7, 11}`. There is no mechanism on the tree that forbids this
configuration, and its frequency rises with the machine.

## 5. Toward the root: what the glue can and cannot buy

**Depth 3.** Where it works, the glue gives `N(v) <= F_2(M)` for `v >= 6`, hence
`Q*_3^literal = max(N(a) + a, N(b) + b) <= F_2 + b`. The chain budget is `F + q' = F + a + b`, so
the glue closes depth 3 only if `F_2 - F <= a`. Measured `F_2 - F = 4, 5, 7, 6, 5, 12, 10` against
`a = 4, 6, 6, 8, 10, 10, 12` at m11..m31, i.e. `F_2 - F - a = 0, -1, +1, -2, -5, +2, -2`: **it fails
at m17 and m29.** The loss of one letter enters because the middle can be as large as `b = q' - a`,
so only `q' - a` of the budget's `q'` is spent on the middle and the leftover `a` must absorb the
whole of `F_2 - F`.

**Combined with the pair statement.** The pair statement is `F_2 <= F + q'`. Substituted, it gives
`Q*_3 <= F + q' + b`, which overshoots the budget by `b`. What the glue actually needs is the
STRENGTHENED pair statement `F_2 <= F + a`, stronger than the pair statement by exactly
`b = q' - a`, and false at two of seven rungs. So the glue plus the pair statement does not close
the chain statement; the gap is a whole letter and it is not a slack that can be traded.

**Depth 4 (the level-4 glue).** The same construction applies verbatim to a `J`-run, with the right
base shifted by the middle span: the hole is again at `h = g_1 - 1`, and a glueing colouring
certifies `F_2 >= g_1 + g_J`. For a literal word-legal 4-run the two middles alternate classes
(the middle-sum lemma, alignment-rules 736-790), so `g_2 + g_3 = a + b = q'` exactly, and the glue
gives **`Q*_4 <= F_2 + q'`**. The budget is `F + q'`, so depth 4 needs `F_2 <= F`, which is false at
EVERY rung (`F_2 - F = 4, 5, 7, 6, 5, 12, 10`). Depth 4 is therefore strictly worse than depth 3,
losing the entire letter `a` rather than the excess `F_2 - F - a`, and the reason is structural and
depth-independent: part (i) of the glue lemma forces one open column in the glued object at every
depth, so every glue bound is an `F_2` bound, never an `F` bound. **The level-4 version gives the
deeper cases no cap of their own.** Measured on top of that, the level-4 colouring fails at the
extremal 4-run witness at m23 (section 2.8).

**Against the wall.** The object has the shape the wall asks for (which residues, growing modulus,
no transfer, no counting) and it is still false, which is information: the failure is not a
limitation of the method used to study it but of the object. What it does deliver is a proved
reduction of the whole `N(v) <= F_2` question to a two-column residue coincidence (section 3), and
the news that this coincidence is where the real teeth are, for once, atypical (section 2.7).

## 6. What is new

1. **The 95.5% was 92% peel bound.** Split at `v = min(L, R)`, the glue's own success rate on the
   attaining 3-runs is 30 of 106 hard runs (28.3%) over m13..m31, against 756 of 756 on the runs the
   peel bound already discharges. Not on record anywhere; 2g.i's headline number is arithmetic on
   the union of the two.
2. **The rate falls with the machine and vanishes at m29:** 50%, 62.5%, 40%, **0%**, 22.7% at
   m17..m31 on hard attaining runs; 66.7%, 25%, **0%**, 7.7%, 4.0% on all 3-runs with `L+R > F`.
3. **The shadow lemma** (proved, 23,880/23,880): the covering instance has exactly two single-sided
   columns, `x_1 - v` and `x_2 + v`, and the constant colouring already covers everything but one of
   them -- so the glue's entire content is buying one column.
4. **The move lemma** (proved from the chain law): recolouring translates a gear's strikes by `+v`,
   so a strike survives iff `v` is `0` or a letter of that gear; padded gears move free but cannot
   cover the shadow; all other gears lose everything. This is the chain law's alphabet appearing as
   the governing condition of a covering problem, which is new.
5. **The certificate never uses more than one move.** 756 of 786 successes use zero gears, almost
   all the rest one, at most three ever.
6. **The `J`-run outer law**, new and exceptionless: for `J` consecutive gaps with every middle
   `>= 6`, `g_1 + g_J <= F_2(M)` -- 0 exceptions in 3,278,972 `J`-runs at `J = 3..8`, m13..m23, with
   the maximum falling as `J` grows. Its companion negative: dropping the middle condition (the
   separation profile `N*(S)`) breaks the cap at `S = 12, 7, 6, 6` at m13..m23, with
   `max N* = 22, 30, 41, 54` against `F_2 = 16, 25, 31, 39`. So the cap is a statement about a gap,
   not about a distance.
7. **The real teeth are atypical for this certificate**, at the 99.6th percentile of 223 comparable
   m19 family members (62.5% against a pooled 9.4%) -- the first quantity on the tree where the real
   machine sits in the upper tail of the symmetric-tooth family rather than in the middle (contrast
   the wall's face C: `F` at the 14th-22nd percentile, phase vectors at parity). The obvious cause
   (`|Leg(v)|`, the count of gears for which `v` is a letter) is REFUTED -- it is exactly typical.
8. **The residual 20 of 2g.i, closed at loss 0 by the cross glue** (19 directly, the 20th by the
   mirror), and 14 of 20 by two colours plus one or two free gears; with the six that resist both
   named as m23's `v = 7` cell.
9. **Depth 4 is worse, not better**: `Q*_4 <= F_2 + q'` needs `F_2 <= F`, false at every rung; the
   deficit `F_2 - F` is depth-independent because the glue's hole is forced at every depth.

### 6a. Exceptionless statements, with counts

- **E1.** The shadow lemma: the only single-sided target offsets are `h - v` and `h + v`.
  **0 violations in 23,880 3-runs** (every 3-run with `v >= 6` and `L + R > F - 12` at m13..m23).
  Proved, and the verification is an instrument check on the proof.
- **E2.** At every C2 failure the minimum number of uncovered columns is 1: **178 of 178** failures
  among hard runs with `L + R > F - 6` at m17, m19, m23. (Forced by the constant colouring; the
  content is that it is never 2.)
- **E3.** At every C2 failure some optimal colouring's single uncovered column is a shadow:
  **178 of 178**.
- **E4.** The `J`-run outer law `g_1 + g_J <= F_2(M)` when every one of the `J-2` middles is
  `>= 6`: **0 exceptions in 3,278,972 `J`-runs**, `J = 3..8`, m13..m23.
- **E5.** `N(v) <= F_2(M)` for `v >= 6` (2g.i's law) re-confirmed independently at m29 and m31 on
  full periods by this branch's chunked pass: **0 exceptions**, tight once
  (`N(7) = 55 = F_2` at m29; `N(7) = 66` against `F_2 = 68` at m31).
- **E6.** Every trivial run (`v >= min(L, R)`) glues: **756 of 756** attaining runs at m13..m31,
  as the peel bound forces.

Prior art checked in `docs/novel/README.md` and the tree: the peel bound and middle-sum lemma
(alignment-rules 736-790), L4 (docs/proofs/19), the chain and merge laws (docs/proofs/05), the
junction theorem (R3.h.i) and the `N(v) <= F_2` law itself (2g.i) are all cited and none is
re-derived. The shadow and move lemmas, the trivial/hard split, the `J`-run outer law and the
family percentile are not on record.

## 7. Verdict

- **The branch's theory T is DEAD.** The covering statement "every 3-run with `v >= 6` admits a
  glueing colouring" is false, and false in the region that matters: 0 of 22 hard attaining runs at
  m29, 0 of 6 runs with `L + R > F` at m23. Refuting instance for the whole line, resistant to every
  wider certificate readable off the run: m29 `(18, 10, 30)` at `x_0 = 278620515`.
- **What survives.** Three proved lemmas (shadow, move, and the constant-colouring proof of the
  peel bound in covering language), and one new exceptionless law (the `J`-run outer law, 3.28
  million runs). The `N(v) <= F_2` law of 2g.i is untouched -- it is still exceptionless to m31 --
  but it now has no constructive route: the construction that was supposed to prove it works on a
  minority of the runs that carry it.
- **Status of the node:** DEAD as a route; FACT for the three lemmas and the outer law.
- **Where the survivors go.** The `J`-run outer law is the object to take forward: it is the same
  cap holding to depth 8 with a wide-middle condition, which says the `F_2` cap is really a
  statement about runs of wide gaps, not about pairs of openings. The family percentile is the one
  live puzzle: the real teeth do something for this certificate that random symmetric teeth do not,
  and it is not the letter count.

## 8. Scorecard, filled

| # | Prediction | Result |
|---|---|---|
| G1 | covering formulation reproduces 426/446 | **HELD** (430/450 here; the same 20 failures at the same `v`; four extra attaining runs at m23, all successes) |
| G2 | all 20 residuals are unit-propagation conflicts | **REFUTED** -- 0 of 20 have a gear forced to both sides by a unique column; the obstruction is one uncovered SHADOW column (20/20), which is a different and sharper statement |
| G3 | C2+1 rescues >= 15 of 20 | **REFUTED** -- C2+1 rescues 8, C2+2 rescues 6 more (14 of 20); m23's `v = 7` cell (6 runs) resists both |
| G4 | overlap `t <= 3` at half the residuals | **REFUTED** -- best `t` is 4 to 10, and 2 of 20 have no `t <= 10`; but `Cs` still gives loss 0 at 8 of 20 |
| G5 | all runs with `L+R > F`: C2 >= 90%, best loss 0 | **REFUTED hard** -- 85.7%, 25%, 40% at m17/m19/m23 overall and 66.7%, 25%, **0%** on the hard part; 7.7% at m29 and 4.0% at m31 |
| G6 | m29, m31 attaining runs: C2 >= 90% | **REFUTED** -- 85.1% and 87.1% overall, but **0/22** and **10/44** on the hard part |
| G7 | family rate within 5 points of the real | **REFUTED, in the opposite direction** -- family pooled hard rate 14.3% (m17) and 8.2-9.4% (m19) against the real 50% and 62.5%; the real machine is at the 99.6th percentile |
| G8 | outer law first fails at `J >= 5` | **REFUTED both ways** -- with middles `>= 6` it never fails to `J = 8` (0 of 3,278,972); without that condition it fails at `J = 3` |

## 9. Dead ends, each with its refuting instance

- **D1.** The covering statement as a lemma. m29 `(18,10,30)` at `x_0 = 278620515`: no colouring, no
  colouring with up to 3 free gears, no overlap `t <= 12`.
- **D2.** Reading 2g.i's 95.5% as evidence for the construction. 756 of the 786 successes are the
  peel bound, which needs no construction; the construction's own rate is 28.3%.
- **D3.** "A shared sole striker explains the failures" (2g.i's D4, re-examined). It is not even the
  right notion: a left-flank column solely struck by `g` can be covered by another gear coloured
  right, because the right phase inspects the column `v` further on. 0 of 20 residuals have a gear
  forced to both sides by a unique column.
- **D4.** Unit propagation as the diagnostic. It reaches `empty` at 16 of 20 and `ok` at 4, so it
  does not separate; the separating statement is the shadow (20/20).
- **D5.** `|Leg(v)|`, the number of gears for which `v` is a letter, as the reason the real teeth do
  better. Refuted: `P(|Leg| >= real)` = 0.24 to 1.00 across 20 cells; the real value is the family
  mean.
- **D6.** The separation form of the law (`N*(S) <= F_2` for `S >= 6`). Refuted at `S = 6` at m19 and
  m23; `max_{S>=6} N* = 54` against `F_2 = 39` at m23.
- **D7.** The level-4 glue as a cap for the deeper chain cases. `Q*_4 <= F_2 + q'` needs
  `F_2 <= F`; `F_2 - F = 4, 5, 7, 6, 5, 12, 10` at m11..m31.
- **D8.** `Cx` (the cross glue) as a proof route. It closes all 20 shallow residuals at loss 0, but
  it searches the machine's period for a partner flank, so it is not a certificate read off the run
  and it cannot be evaluated at m29 or m31 at all.
