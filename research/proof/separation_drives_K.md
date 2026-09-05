# Branch R2.a.i.a.1.a.i - IS THE REAL SEPARATION WHAT DRIVES K(d)?

Parent: node R2.a.i.a.1.a (the cover number, `research/proof/cover_number.md`). Weak point **W3** of
`research/proof/the_wall.md`, with W1 and W2 in view.

The observation that spawned this branch: the parent isolated the growth of `K(d)` to two causes -
one phase per gear, and a separation the gear cannot choose - and measured the second at a factor
1.5 (`K/K_free = 1.33..1.56`). It never asked whether the *value* of that separation matters. Branch
6 asked the same question of `F`, the record of one machine, and found the real one-third spacing at
the 14th-22nd percentile of random symmetric spacings, and closed as DEAD *as a driver of F*.
Nobody has asked it of `K`, the adversarial cover, where the wall says the answer decides whether
the target statement is about **the machine's teeth** (W1: prove the overlap bound for the one-third
separation specifically) or about **the whole one-phase family** (W2: the constant to prove is the
family's).

Scripts: `research/anchor235/r43/sep_*.py`. Result outputs (untracked):
`research/anchor235/r43/results/`. Every number this document relies on is written into the
document.

---

## 0. Pre-registered (written before any computation of this branch)

### 0.1 The object, exact - one parameterisation for every family

Islands of `[1, d)`: offsets `i` with `1 <= i < d` and `i mod 35 in {5, 10, 12, 17}`; `m = m(d)`
their number. A **gear** is a prime `g > 7`.

The parent's real gear is: classes `(2 - r) u_g` and `(-r) u_g` mod `g`, with `u_g = 6^{-1} (mod g)`
and `r = q^2 mod g` any nonzero quadratic residue. The two classes sit at separation
`s_g = 2 u_g = 3^{-1} (mod g)` and the phase `a_g = -r u_g` runs over the coset `-u_g * QR(g)`,
of size `(g-1)/2`. **The whole family is obtained by replacing `u_g = 6^{-1}` with an arbitrary
nonzero `u_g` mod `g`, changing nothing else**: separation `s_g = 2 u_g`, phase
`a_g = -r u_g` over `r` a nonzero QR, classes `{a_g, a_g + s_g}`, one phase per gear. So every
family below has *identical* structure - same one-phase rule, same reachable-phase coset of size
`(g-1)/2`, same island set, same gear pool - and differs only in `u_g`:

| family | `u_g` | separation `s_g` |
|---|---|---|
| **real** | `6^{-1} (mod g)` | `3^{-1}`, the same rational one third at every gear |
| **coherent(c, r)** | `c r^{-1} 2^{-1} (mod g)` | `c r^{-1}`, the same rational `c/r` at every gear |
| **random** | uniform on `1..g-1`, independent per gear | uniform nonzero |
| **free** | - | the gear picks any two classes (parent's `K_free`) |

`K_X(d)` = the least number of gears of family `X`, each used once at one reachable phase, whose
strikes cover every island of `[1, d)`.

**Gear pool.** As in r41: gears `11 <= g <= 3d + 2`, plus one generic singleton per island. For the
real separation this pool is *complete* (a gear covers a pair `{i, j}` only if `g | 3(i-j) -+ 1`, so
only if `g <= 3d`; a singleton is available at infinitely many gears above that). For a family with
an arbitrary separation the pool is a **restriction** - a large gear whose random `s_g` happens to
land below `d` can pair two islands - and a restriction can only *raise* `K`. That is the
conservative direction against this branch's headline (it makes `K_random` bigger, so `K_real`
looks *lower* in the random distribution), so it is used as the default and checked separately
with a pool four times deeper.

### 0.2 What would count as a rule

An exact statement about positions, counts or residues with an exception count over a stated range,
uniform in `d`. A fitted curve, a density, or a restatement of the doubling law, of N-R5 (large
gears strike islands at `2/g`), or of the counting requirement is **not** a finding. Any
sub-question that reduces to CRT, to Mertens, to the Erdos-Rankin construction or to the Jacobsthal
function `j_2 = h_2` is named in one line as classical and stopped.

### 0.3 Predictions, with numbers, and what refutes each

**P1 - the correctness gate.** The parameterised solver at `u_g = 6^{-1}` reproduces the parent's
exact ladder: `K_real(d) = 6, 9, 14, 17, 20, 22` at `d = 140, 280, 560, 840, 1120, 1330`, each
HiGHS-certified optimal. REFUTED (and the branch stops) if any value differs.

**P2 - the central question: where does `K_real` sit among random separations?**

* **P2a.** `K_real(d) >= median(K_random(d))` at every one of the five/six `d` (5 of 5, or 6 of 6).
  REFUTED by two `d` with `K_real` below the median.
* **P2b.** The percentile of `K_real` in the random distribution is between **0.55 and 0.95** at
  every `d`, mean near **0.78** - the mirror of branch 6's 0.14-0.22 on `F` (a *short* record and a
  *large* cover number are the same statement, "these teeth cover badly"). REFUTED if the percentile
  is below 0.50 at two or more `d`, and equally refuted (in the other direction) if `K_real` is the
  strict maximum of the draw at every `d`.
* **P2c.** The random distribution is narrow: `max(K_random) - min(K_random) <= 3` at every
  `d <= 1120`. REFUTED by a spread of 5 or more.
* **P2d (the deciding form).** If P2b holds, the verdict is W1: the real teeth drive the
  adversarial cover and the overlap lower bound must be proved *for the one-third separation*. If
  the percentile is at or below 0.5, the verdict is W2: the one-phase rule alone drives the growth
  and the constant to prove is the family's.

**P3 - coherence versus the particular rational.** For `s_g = 2 r^{-1}` with `r = 5, 7, 11, 13` and
for a few `s_g = c r^{-1}`:

* **P3a.** Every coherent family lies within `+-2` of `K_real` at every `d`. REFUTED by a coherent
  family differing by 3 or more at two `d`.
* **P3b.** Every coherent family lies at or above the random median. REFUTED by two coherent
  families below it at the same `d`. (If P3a and P3b both hold, what matters is *coherence itself* -
  the same rational at every gear - and not the fact that the rational is one third. If instead the
  real family stands out among the coherent ones, the one third is special.)

**P4 - the free row, reproduced.** `K_free(d) = 4, 6, 9` at `d = 140, 280, 560`, budget exactly `m`
(a perfect partition) at `d = 140, 280`. REFUTED by any disagreement with the parent.

**P5 - the mechanism: pairwise overlap.**

* **P5a (derived before computing, predicted exact, 0 exceptions).** For gears `g != h` and *all*
  `gh` phase pairs (QR restriction dropped), the mean overlap
  `|S_g ∩ S_h ∩ Islands(d)|` equals **exactly `4 m / (g h)`**, whatever `s_g` and `s_h` are: for
  each of the four class combinations `(α, β)` the CRT point `x_{αβ}` runs over every residue mod
  `gh` exactly once as the two phases run over their full ranges, so the total is `4 m` regardless
  of the separations. **Coherence therefore cannot act through the mean.** REFUTED by one pair whose
  measured full-phase mean differs from `4m/(gh)`.
* **P5b.** Consequently, if coherence acts at all it acts on the *minimum* over reachable phases.
  Prediction: at `d = 560`, the fraction of gear pairs `(g, h)` for which some reachable phase pair
  achieves overlap 0 is **smaller for the real separation than for random by at least 3 percentage
  points**. REFUTED if the two fractions agree within 1 pp (which would say coherence does nothing
  pairwise), or if real is the larger.
* **P5c (the opposing effect, pre-registered so it is not read as a surprise).** The one-third
  separation has an exact integer form: `3 s_g = 1 (mod g)` forces `s_g = (g+1)/3` or `(2g+1)/3`, so
  `min(s_g, g - s_g) = (g +- 1)/3` for **every** gear. Hence every gear `g < 3d` is **pair-capable**
  (it can strike two islands of `[1, d)`) and no gear above `3d` is - a sharp threshold. For a
  random separation the same gear is pair-capable with probability `min(1, 2d/g)`, so the fraction
  of pair-capable gears in `(d, 3d)` is predicted **1.000 for real and about 0.90 for random**
  (`0.5 + ln 1.5 = 0.905`). This effect *helps* the real adversary. REFUTED if the real fraction is
  below 1 or the random fraction is outside `[0.85, 0.95]`.
* **P5d (the diagonal identity, derived before computing).** Write `S_g` for the CRT lift of
  `(s_g, 0)` and `S_h` for that of `(0, s_h)` modulo `gh`; the four struck residues are
  `x, x + S_g, x + S_h, x + S_g + S_h` for a free translate `x`. For the real separation
  `3(S_g + S_h) = 1 (mod g)` and `(mod h)`, hence `S_g + S_h = 3^{-1} (mod gh)` - **the one third
  reproduces itself at the composite modulus**, and as an integer it is `(gh +- 1)/3` or
  `(2gh +- 1)/3`. Consequence predicted exact: for `gh > 3d` the two *diagonal* strikes of any pair
  of real gears can never both be islands of `[1, d)`, while for random separations they can, with
  probability about `2d/(gh)`. REFUTED by one real pair with both diagonal points in `[1, d)` at
  `gh > 3d`.

**P6 - anatomy of the optimal covers.** Gear by gear, for each family: islands per gear, islands
struck twice, the pairs that overlap most. Prediction: the budget `sum |S_j| / m` is **larger** for
the real separation than the random median at every `d` (that is the same coherence claim as P2, in
the currency the mechanism uses). REFUTED if the real budget ratio is below the random median at two
`d`.

**P7 - toward the root: the constant, worked out.** The window statement needs
`K(W(q)) > pi(q) - 3` with `W ~ q^2/6`, i.e. `K(d) > pi(sqrt(6 d)) - 3`. Pre-registered arithmetic:
at `d = 1120` that is `K > 19` against the measured 20, and at `d = 1330` `K > 20` against 22, so
the island cover number is **already above the requirement at every arc where it is known exactly** -
predicted 6 of 6 at the branch's `d`, 0 exceptions. The pairwise overlap statement that would give
it is stated in section 7 with the required `X` computed, not guessed.

**P8.** Everything that holds without exception over the computed range, with counts.

### 0.4 Scorecard

| # | prediction | verdict and evidence |
|---|---|---|
| P1 | `K_real = 6, 9, 14, 17, 20, 22` reproduced | **CONFIRMED**, all six, every one HiGHS-certified optimal (2.1) |
| P2a | `K_real >= median(K_random)` at every `d` | **CONFIRMED** 5 of 5 - but with equality at four of them (2.2) |
| P2b | percentile 0.55-0.95, mean ~0.78 | **REFUTED**: 0.500, 0.463, 0.750, 0.483, 0.500 over 189 draws; `K_real` is the MODE of the random distribution at every arc (2.2) |
| P2c | random spread `<= 3` | **CONFIRMED and exceeded**: the spread is 2, 2, 1, 2, 2 (2.1) |
| P2d | which of W1 / W2 | **W2** (4.4) |
| P3a | coherent families within `+-2` of real | **CONFIRMED**: `+-1` for `r = 3, 5, 7`; `+1..+2` for `r = 11, 13`, and that is the missing gear, not the separation (2.3) |
| P3b | coherent families at or above the random median | **REFUTED**: `coh:1/5` below at `d = 140, 280`; `1/5, 2/5, 4/7` below at `d = 560` (2.3) |
| P4 | `K_free = 4, 6, 9`; perfect partition at 140, 280 | **CONFIRMED on `K`** (4, 6, 9); the budget of the optimum returned is `1.062, 1.000, 1.062 m`, i.e. the perfect partition is one of several optima at `d = 140` and the ILP printed another (2.1) |
| P5a | full-phase mean overlap `= 4m/(gh)` exactly | **CONFIRMED, 0 exceptions in 72 brute-force checks**, and it is a one-line identity - so no separation can act through the mean (3.2) |
| P5b | real achieves overlap 0 on 3 pp fewer pairs | **REFUTED**: the reachable-phase correlation puts real at percentile 0.83, 0.37, 0.97 at `d = 280, 560, 1120` - no consistent sign, 5% spread (3.2) |
| P5c | pair-capable fraction in `(d, 3d)`: real 1.000, random ~0.90 | **REFUTED in the numbers, CONFIRMED in the direction**: a tooth distance must also be an island difference mod 35, so the fractions are real 0.14-0.34 and random 0.15-0.45; but real is below EVERY random draw in the `(1.5d, 3d)` band at all three arcs (3.4) |
| P5d | real diagonal `S_g + S_h = 3^{-1} mod gh`; no real pair with both diagonal points inside | **CONFIRMED and generalised**: `r(S_g+S_h) = c (mod gh)` for every coherent family, 0 exceptions in 32,490 pair-checks; `\|diag\|/gh >= 0.3316` for real, 0 exceptions in 4,095 (3.3) |
| P6 | real budget ratio above the random median | **REFUTED**: below it at 4 of 5 arcs (1.062/1.125, 1.156/1.203, 1.167/1.240, 1.227/1.266) (2.4) |
| P7 | `K(d) > pi(sqrt(6d)) - 3` at every exact arc | **SPLIT**: holds by exactly one gear at `d = 560, 840, 1120, 1330`; **fails with equality** at `d = 140, 280` (4.1) |
| P8 | exception-free statements with counts | thirteen, listed in section 5 |

---

## 1. Setup (exact ranges)

No sampling except where a row says so. Scripts in `research/anchor235/r43/`; outputs (untracked)
in `research/anchor235/r43/results/`.

| object | range | script |
|---|---|---|
| `K_real(d)` exact, ILP with the one-phase-per-gear rule, HiGHS dual bound meeting the incumbent | `d = 140, 280, 560, 840, 1120, 1330` | `sep_cover.py` |
| `K_rand(d)`, separations `s_g` uniform on the nonzero residues, independent per gear | 40 draws at `d = 140, 280, 560`; 30 at `d = 840`; 35 at `d = 1120`; 4 at `d = 1330` (189 in all) | `sep_cover.py --rand` |
| `K_coh(d)` for `s_g = c r^{-1}`: `(c,r) = (2,5), (2,7), (2,11), (2,13), (1,5), (4,7), (3,11), (5,13)` | the same six arcs | `sep_cover.py --fams coh:c/r` |
| `K_free(d)`, separation free (parent's row) | `d = 140, 280, 560` | `sep_cover.py --fams free` |
| gear pool robustness: the same at pool `11 .. 12d + 2` instead of `11 .. 3d + 2` | `d = 280`, real + 30 random | `sep_cover.py --gmul 12` |
| pairwise overlap: the full-phase mean, the reachable-phase correlation `C(g,h)`, pair-capability of tail gears | `d = 280, 560, 1120`, all pairs of gears `11..199`, 30 random families | `sep_overlap.py` |
| tail-gear tooth distances and the island-pair counts at those distances | `d = 140, 280, 560, 840, 1120, 1330`, every gear in `(d, 3d]`, 30 random families each | `sep_tail.py` |
| the CRT-closure identity `r (S_g + S_h) = c (mod gh)` | every pair of gears `11..499` (4,095 pairs) x 8 rationals | inline check, section 5.3 |
| anatomy of the optimal covers (sizes, multiplicities, pairwise overlaps, the chain sum) | `d = 280, 560`, 7 coherent families + 8 random draws | `sep_anatomy.py` |
| the frontier the root needs: budget and MAX COVERAGE of the `pi(sqrt(6d)) - 3` cheapest gears | `d = 140 .. 1330`, real + 6 random each | `sep_frontier.py` |

**One parameterisation.** Every family is the same object with a different `u_g`: classes
`(2 - r) u_g` and `(-r) u_g` mod `g`, `r` a nonzero quadratic residue, separation `s_g = 2 u_g`.
`u_g = 6^{-1}` is the machine. So the comparison changes the separation and nothing else - not the
one-phase rule, not the reachable-phase coset (always of size `(g-1)/2`), not the island set, not
the gear pool.

**The gear pool.** `11 <= g <= 3d + 2` plus one generic singleton per island, r41's complete play
list for the real separation. For an arbitrary separation the pool is a restriction, and a
restriction can only raise `K` - the conservative direction for this branch's question. The
robustness row settles that it does not matter: at `d = 280` with the pool taken out to `12d + 2`
(four times deeper, 367 candidates for real against 128) the real answer is unchanged at 9 (as it
must be, the bound being proved) and the random distribution moves from `{8:2, 9:33, 10:5}` to
`{8:1, 9:25, 10:4}` - the same median 9 and the same percentile for real (0.450 against 0.463).

## 2. Results - `K(d)` under every separation (items 1-4)

**239 ILP rows, every one HiGHS-certified optimal** (dual bound equal to the incumbent); 0 rows
returned as a bound only.

### 2.1 The correctness gate, and the answer in one table

| `d` | `m` | **`K_real`** | `K_free` | random `K`: min / median / max | the random distribution | **percentile of real** | coherent `s_g = c r^{-1}` |
|---|---|---|---|---|---|---|---|
| 140 | 16 | **6** | 4 | 5 / 6.0 / 7 (n = 40) | `{5:2, 6:36, 7:2}` | **0.500** | 1/5: 5, 2/5: 6, 2/7: 6, 4/7: 6, 2/11: 6, 3/11: 7, 2/13: 7, 5/13: 6 |
| 280 | 32 | **9** | 6 | 8 / 9.0 / 10 (n = 40) | `{8:2, 9:33, 10:5}` | **0.463** | 1/5: 8, 2/5: 9, 2/7: 9, 4/7: 9, 2/11: 10, 3/11: 10, 2/13: 10, 5/13: 10 |
| 560 | 64 | **14** | 9 | 13 / 13.5 / 14 (n = 40) | `{13:20, 14:20}` | **0.750** | 1/5: 13, 2/5: 13, 2/7: 14, 4/7: 13, 2/11: 15, 3/11: 15, 2/13: 15, 5/13: 15 |
| 840 | 96 | **17** | - | 16 / 17.0 / 18 (n = 30) | `{16:1, 17:27, 18:2}` | **0.483** | 1/5: 17, 2/5: 17, 2/7: 17, 4/7: 17, 2/11: 19, 3/11: 18, 2/13: 19, 5/13: 18 |
| 1120 | 128 | **20** | - | 19 / 20.0 / 21 (n = 35) | `{19:1, 20:33, 21:1}` | **0.500** | 1/5: 20, 2/5: 20, 2/7: 20, 4/7: 20 |
| 1330 | 152 | **22** | - | 21 / 22.0 / 22 (n = 4) | `{21:1, 22:3}` | **0.625** | - |

**P1 confirmed.** `K_real = 6, 9, 14, 17, 20, 22` at `d = 140, 280, 560, 840, 1120, 1330`, the
parent's ladder reproduced exactly by the parameterised solver. **P4 confirmed**: `K_free = 4, 6, 9`
at `d = 140, 280, 560`, the parent's row exactly (the budget of the `K_free` optimum returned here
is `1.062, 1.000, 1.062 m`; the parent reports a perfect partition at `d = 140` too - both are size-4
optima, and the ILP returns whichever it finds first, so nothing is in dispute except which optimum
is printed).

### 2.2 The deciding number: `K_real` sits at the MODE of the random distribution

> **N-S5 (the real separation is typical for `K`).** At every arc computed, `K_real(d)` is exactly
> the **mode** of the distribution of `K` over 30-40 independent random separations, and its
> mid-rank percentile is **0.500, 0.463, 0.750, 0.483, 0.500** at `d = 140, 280, 560, 840, 1120`
> (and 0.625 on the four draws affordable at `d = 1330`). It is never above the 80th percentile and
> never below the 46th. The pre-registered discriminator (P2b: percentile in 0.55-0.95 at every `d`,
> mean 0.78) is **REFUTED**.

The distribution is extremely tight: 36 of 40 draws give the same value as the real machine at
`d = 140`, 33 of 40 at `d = 280`, 27 of 30 at `d = 840`, 33 of 35 at `d = 1120`, 3 of 4 at
`d = 1330`. The full spread is **2, 2, 1, 2, 2, 1** gears at the six arcs (P2c confirmed: never
above 3), against a `K` running from 6 to 22 - so the separation moves the adversarial cover number
by at most one gear in either direction, on a quantity that has more than trebled over the same
range. P2a is confirmed (`K_real >= median` at 6 of 6, with equality at five of them).

**Robustness.** With the gear pool taken out four times further (`11 .. 12d+2`, `d = 280`) the
random distribution is `{8:1, 9:25, 10:4}` and `K_real` is unchanged at 9: percentile 0.450 against
0.463. The pool restriction is not what makes the real machine look ordinary.

### 2.3 Coherence: the particular rational does not matter either, once one artefact is removed

The coherent families split cleanly into two groups, and the split is **not** about the separation.

* `r = 3` (the machine), `r = 5`, `r = 7`: the separation `c r^{-1}` is defined at every gear
  `g > 7`. These give `K = 5..6, 8..9, 13..14, 17, 20` - **identical to real and to the random mode
  at every arc**, and at `d = 1120` all five families (real, 1/5, 2/5, 2/7, 4/7) return exactly 20.
* `r = 11` or `r = 13`: the rational `c/r` **has no value at the gear `g = r`**, so those families
  simply lose gear 11 or gear 13. That is the parent's N-C5 effect, not a separation effect. The
  control, run with the real separation and the gear deleted:

  | `d` | real, all gears | real, no gear 11 | real, no gear 13 | `coh:2/11` | `coh:3/11` | `coh:2/13` | `coh:5/13` |
  |---|---|---|---|---|---|---|---|
  | 280 | 9 | **11** | **10** | 10 | 10 | 10 | 10 |
  | 560 | 14 | **15** | **14** | 15 | 15 | 15 | 15 |
  | 840 | 17 | **19** | **19** | 19 | 18 | 19 | 18 |

  Every `r = 11, 13` family lands **at or below** its own missing-gear baseline. The whole excess is
  the lost gear.

**P3a confirmed** (all coherent families within `+-2` of real, and within `+-1` once the missing
gear is accounted for). **P3b refuted**: `coh:1/5` is below the random median at `d = 140, 280` and
`coh:1/5, 2/5, 4/7` are below it at `d = 560`. There is no "coherent separations are harder to
cover" effect: coherent, incoherent and real are the same object as far as `K` can see.

### 2.4 The budget at the optimum

| `d` | real | random median | random range | coherent range |
|---|---|---|---|---|
| 140 | 1.188 | 1.062 | 1.000-1.188 | 1.000-1.188 |
| 280 | 1.062 | 1.125 | 1.000-1.250 | 1.062-1.250 |
| 560 | 1.156 | 1.203 | 1.094-1.312 | 1.094-1.234 |
| 840 | 1.167 | 1.240 | 1.167-1.302 | 1.135-1.271 |
| 1120 | 1.227 | 1.266 | 1.188-1.320 | 1.250-1.297 |

**P6 is refuted**: the real budget ratio is *below* the random median at four of the five arcs
(1.062 vs 1.125, 1.156 vs 1.203, 1.167 vs 1.240, 1.227 vs 1.266) and above it only at `d = 140`. The
real separation makes the optimum slightly *tidier*, not wastier - the opposite of the coherence
prediction, and by an amount (3-6% of `m`) that is inside the random spread.

The optimal real gear sets are the parent's shape at every arc - the compulsory prefix `11..31`
present from `d = 385` on, then a tail:

```
  d =  140   11, 13, 17, 23, 37, 127
  d =  280   11, 13, 19, 23, 29, 47, 53, 101, 199
  d =  560   11 .. 43, 71, 193, 211, 337
  d =  840   11 .. 47, 59, 103, 113, 149, 367, 2099
  d = 1120   11 .. 47, 59, 61, 73, 101, 109, 211, 443, 757, 1979
  d = 1330   11 .. 61, 73, 79, 139, 181, 251, 383, 419, 509
```

## 3. Mechanism (item 5): what the separation can and cannot do to the overlap

### 3.1 The overlap of two gears, written down

Gear `g` at phase `a_g` strikes the classes `a_g` and `a_g + s_g` mod `g`; gear `h` at `a_h` strikes
`a_h` and `a_h + s_h` mod `h`. By CRT the four combinations are four residues mod `gh`, and they are
a **translate of one fixed four-point shape**:

```
    x,   x + S_g,   x + S_h,   x + S_g + S_h      (mod gh),
    S_g = CRT(s_g mod g, 0 mod h),   S_h = CRT(0 mod g, s_h mod h),
```

with `x = CRT(a_g, a_h)` free. So

```
    |S_g ∩ S_h ∩ Islands(d)|  =  N(x) + N(x + S_g) + N(x + S_h) + N(x + S_g + S_h),
    N(y) = #{ i in [1, d) : i = y (mod gh),  i mod 35 in {5, 10, 12, 17} } .
```

**Only the shape `{0, S_g, S_h, S_g + S_h}` depends on the separations**; the two phases move the
whole shape rigidly. That splits the question exactly in two: what the separations do to the mean
(nothing - 3.2) and what they do to the shape (3.3, 3.4).

### 3.2 The separation cannot change the mean overlap - exact, and it kills the obvious mechanism

> **N-S1 (the mean-overlap identity).** For any two gears `g != h` and **any** separations
> `s_g, s_h`, the mean of `|S_g ∩ S_h ∩ Islands(d)|` over all `g h` phase pairs is exactly
> `4 m / (g h)`.

Proof in one line: island `i` lies in exactly two of the `g` classes of gear `g` (`a_g = i` and
`a_g = i - s_g`) and in exactly two of the `h` classes of gear `h`, so it is counted in exactly four
of the `gh` phase pairs; summing over islands gives `4m`, whatever the separations are.
Brute-force verified over all phase pairs for `(g,h) = (11,13), (11,17), (13,19), (17,23)` and six
separation families at `d = 280, 560, 1120`: **72 checks, 0 exceptions** (P5a confirmed).

So a "coherence raises the overlap" mechanism cannot exist at the level of the mean. Whatever the
separation does, it does through the reachable-phase restriction or through the minimum.

**The reachable-phase version, measured.** Only `(g-1)/2` phases of each gear are reachable, and
island `i` is struck by `n_g(i)` of them, `n_g(i) in {0,1,2}` (`n_g(i) = 2 chi_g(i)`, the parent's
doubling law). The reachable-phase mean overlap is `(4 / ((g-1)(h-1))) sum_i n_g(i) n_h(i)`, i.e.
the full-phase mean times the correlation `C(g,h) = (1/m) sum_i n_g(i) n_h(i)`, which is 1 under no
correlation. Over all 861 pairs of gears `11..199`:

| `d` | real `C` | random `C`: min / median / max (30 draws) | percentile of real | coherent `C` (1/5, 2/5, 2/7, 4/7, 2/11, 3/11, 2/13) |
|---|---|---|---|---|
| 280 | 0.9755 | 0.8975 / 0.9599 / 0.9968 | 0.833 | 0.985, 0.982, 0.983, 0.970, 1.003, 0.944, 1.007 |
| 560 | 0.9534 | 0.9289 / 0.9585 / 0.9930 | **0.367** | 0.974, 0.989, 0.965, 0.977, 0.980, 0.977, 0.967 |
| 1120 | 0.9740 | 0.9461 / 0.9632 / 0.9861 | 0.967 | 0.971, 0.975, 0.967, 0.956, 0.959, 0.969, 0.953 |

The real percentile is 0.83, 0.37, 0.97 at the three arcs - **no consistent sign** - and the whole
spread of the statistic is 5%. **P5b is refuted**: the one-third separation does not make pairwise
overlap among the small gears systematically larger (or smaller) than a random separation.

### 3.3 Where coherence IS exact: it reproduces itself under CRT

> **N-S2 (coherence is closed under CRT).** If every gear takes the same rational separation
> `s_g = c r^{-1} (mod g)`, then for every pair of gears the shape's diagonal satisfies
> `r (S_g + S_h) = c (mod g h)` - the same rational at the composite modulus. Proof: `S_g + S_h` is
> `s_g` mod `g` and `s_h` mod `h`, so `r(S_g + S_h)` is `c` mod `g` and mod `h`, hence mod `gh`.
> For the machine (`c/r = 1/3`) the diagonal is `3^{-1} (mod gh)` exactly.

Checked over **every pair of gears `11..499` (4,095 pairs) for eight rationals: 32,490 checks, 0
exceptions**. A random separation has no such property: `S_g + S_h` is a uniform residue mod `gh`.

The consequence is a positional fact, not a rate:

> **N-S3 (the machine's diagonal is never short).** For the real separation
> `min(S_g + S_h, gh - (S_g + S_h)) = (gh ± 1)/3` at every pair, so it is never below `gh/3`: the
> measured minimum of `|diag| / gh` over all 4,095 pairs is **0.331551**, 0 exceptions. Hence for
> `gh > 3d` the two *diagonal* strikes of a pair of real gears can never both land in `[1, d)`. For
> random separations the diagonal falls below `d = 1120` on 0.165, 0.170, 0.165 of the same 4,095
> pairs (three draws), against 0.067 for the real separation (273 of 4,095 - and every one of those
> has `gh < 3d`).

That is a genuine, exception-free coherence effect of the machine's teeth. Section 3.4 measures what
it buys: it acts on the **tail gears**, and it acts against the adversary. Section 2 shows it does
not move `K`.

### 3.4 Where coherence bites: the tail gears, and it costs the adversary

A gear `g > d` covers at most two islands, and two only if the two islands differ by exactly one of
its tooth distances `{s_g, g - s_g}` inside `(0, d)`. Two filters follow, and they are the whole
story for the tail: the distance must be an island difference mod 35 (i.e. in
`{0, 2, 5, 7, 12, 23, 28, 30, 33}`, 9 of 35 residues), and the number of island pairs at a distance
`delta` falls off linearly in `delta`.

For the real separation `3 s_g = 1 (mod g)` forces `s_g = (g+1)/3` or `(2g+1)/3` **as integers**, so
the smaller tooth distance is `(g ± 1)/3` at every gear - always about a third of the modulus, never
small. A random separation puts it uniformly in `(0, g/2)`. Over every gear in `(d, 3d]`, 30 random
families per arc:

| `d` | tail gears | mean `delta / d`: real | random mean [min, max] | island pairs at the best `delta`: real | random mean [min, max] |
|---|---|---|---|---|---|
| 140 | 48 | **0.688** | 0.496 [0.444, 0.565] | **0.42** | 1.10 [0.54, 1.79] |
| 280 | 87 | **0.694** | 0.500 [0.441, 0.555] | **1.40** | 1.93 [1.02, 3.41] |
| 560 | 161 | **0.695** | 0.503 [0.473, 0.546] | **2.19** | 3.79 [2.34, 5.76] |
| 840 | 223 | **0.694** | 0.502 [0.473, 0.542] | **2.95** | 5.62 [3.84, 6.79] |
| 1120 | 287 | **0.697** | 0.502 [0.467, 0.526] | **3.92** | 7.92 [5.43, 10.01] |
| 1330 | 333 | **0.696** | 0.502 [0.464, 0.525] | **5.26** | 9.32 [7.57, 11.27] |

> **N-S4 (the machine's tail teeth are too far apart, without exception).** The real separation puts
> a tail gear's tooth distance at **0.688-0.697 of the arc** at every one of the six arcs, against
> 0.496-0.503 for a random separation - **outside the entire random range at all six arcs (180
> draws, 0 exceptions)**. Because the number of island pairs at a given distance falls linearly in
> that distance, the real tail gear reaches about **half** as many island pairs as a random one
> (0.42, 1.40, 2.19, 2.95, 3.92, 5.26 against random means 1.10, 1.93, 3.79, 5.62, 7.92, 9.32),
> below the whole random range at 5 of the 6 arcs.

Read directly on pair-capability (can one reachable phase strike two islands at all?), by band, 10
random families:

| `d` | band | real | random [min, max] | percentile of real |
|---|---|---|---|---|
| 280 | `(1.5d, 3d)` | **0.169** | [0.185, 0.262] | 0.00 |
| 560 | `(1.5d, 3d)` | **0.137** | [0.145, 0.248] | 0.00 |
| 1120 | `(d, 1.5d)` | **0.250** | [0.303, 0.447] | 0.00 |
| 1120 | `(1.5d, 3d)` | **0.162** | [0.214, 0.310] | 0.00 |

Below every random draw in all four rows. **P5c is refuted in its numbers and confirmed in its
direction**: the prediction "real is pair-capable at 1.000 throughout `(d, 3d)`" ignored the mod-35
filter and is wrong by a factor of five; what is true, and was not predicted, is the sign and its
uniformity - the real tail gear is the worst tail gear in the comparison, at every arc and in every
band.

### 3.5 Anatomy of the optimal covers, family by family

`d = 280` and `d = 560`, one optimal cover per family (7 coherent families, 8 random draws, real):
`T` is the total pairwise overlap `sum_{j<k} |S_j ∩ S_k|`, `T_crt = sum_{j<k} 4m/(g_j g_k)` the
CRT-independent value for the same gear set, `E = sum_i (mult_i - 1)` the wasted strikes.

| `d` | family | `K` | budget/`m` | islands struck 1x/2x/3+ | `T` | `T_crt` | `T / T_crt` | `E` |
|---|---|---|---|---|---|---|---|---|
| 280 | **real** | 9 | 1.062 | 30 / 2 / 0 | 2 | 6.7 | **0.300** | 2 |
| 280 | random (8 draws) | 9-10 | 1.094-1.250 | 25-29 / 3-6 / 0-1 | 3-9 | 5.2-11.4 | 0.380-0.924 | 3-8 |
| 280 | coherent (6) | 8-10 | 1.062-1.250 | 24-30 / 2-8 / 0-2 | 2-8 | 6.6-8.7 | 0.230-1.028 | 2-8 |
| 560 | **real** | 14 | 1.156 | 55 / 8 / 1 | 11 | 27.5 | **0.400** | 10 |
| 560 | random (8 draws) | 13-14 | 1.141-1.281 | 48-55 / 7-14 / 0-3 | 9-20 | 22.5-29.5 | 0.396-0.717 | 9-18 |
| 560 | coherent (6) | 13-15 | 1.094-1.234 | 51-58 / 6-11 / 0-2 | 6-17 | 19.9-28.1 | 0.302-0.606 | 6-15 |

Two readings. First, the shape of an optimal cover is the parent's shape in every family: a
near-partition, budget 1.06-1.28 `m`, triple coverage rare, the small gears compulsory and a tail of
opportunistic large ones. Nothing about the separation changes the *anatomy*.

Second, the real separation's optimum has the **lowest** `T/T_crt` of the comparison at both arcs
(0.300 against a random range 0.380-0.924 at `d = 280`; 0.400 against 0.396-0.717 at `d = 560`) -
the real adversary beats CRT-independence by a factor 2.5-3.3 where a random one beats it by 1.4-2.6.
That is the *opposite* sign to the tail effect of 3.4, and it is confounded (the covers being
compared have different `K` and different gear sets). Taken with 3.2 and section 2 it says the same
thing three ways: **the separation moves the pairwise overlap by tens of percent in both directions
and does not move `K`.**

## 4. Toward the root (item 6)

### 4.1 The constant actually required, worked out

The window statement follows from `K(W(q)) > pi(q) - 3` with `W ~ q^2/6`, i.e.

```
    K(d)  >  pi(sqrt(6 d)) - 3  =:  Kneed(d) ,
```

which is asymptotically `K(d) > 2 sqrt(6) sqrt(d) / ln(6 d) = 4.899 sqrt(d) / ln(6 d)`. Written the
way the wall writes it - `K(d) ~ pi(sqrt(c d))` - the constant `c` must exceed **6**. Against the
exact ladder:

| `d` | 140 | 280 | 560 | 840 | 1120 | 1330 |
|---|---|---|---|---|---|---|
| `m` | 16 | 32 | 64 | 96 | 128 | 152 |
| `Kneed = pi(sqrt(6d)) - 3` | 6 | 9 | 13 | 16 | 19 | 21 |
| **`K_real(d)`** | **6** | **9** | **14** | **17** | **20** | **22** |
| margin `K - Kneed` | 0 | 0 | **+1** | **+1** | **+1** | **+1** |
| the `c` with `K = pi(sqrt(c d)) - 3` | 3.78 | 4.89 | **6.22** | **6.00** | **6.15** | **7.07** |

**The requirement is met from `d = 560` on and fails, with equality, at the two smallest arcs** (4 of
6; P7's "6 of 6" is refuted). The measured `c` crosses 6 at `d = 560` and reaches 7.07 at
`d = 1330`; the margin is exactly one gear at every arc where it holds - the island version of the
statement is true with no room to spare, which is the wall's own reading of W2 (the plain-columns
version keeps the factor four; the island version spends it).

### 4.2 The direct frontier: what the cheapest `Kneed` gears can actually do

The requirement asks that **no** `Kneed` gears cover. The cheapest `Kneed` gears have the largest
budget, so they are the natural place to look. For the gear set `11 .. p_Kneed`, one reachable phase
per gear, HiGHS-certified **maximum coverage** (`sep_frontier.py`):

| `d` | `m` | `Kneed` | gears | budget `B` | `B/m` | max coverage | **islands left open** | `T` there | `T_crt` | max mult |
|---|---|---|---|---|---|---|---|---|---|---|
| 140 | 16 | 6 | 11..29 | 19 | 1.188 | 16 | **0** | 4 | 3.3 | 3 |
| 280 | 32 | 9 | 11..41 | 39 | 1.219 | 30 | **2** | 8 | 10.8 | 3 |
| 560 | 64 | 13 | 11..59 | 83 | 1.297 | 61 | **3** | 21 | 31.3 | 3 |
| 840 | 96 | 16 | 11..71 | 129 | 1.344 | 93 | **3** | 31 | 56.2 | 3 |
| 1120 | 128 | 19 | 11..83 | 179 | 1.398 | 124 | **4** | 49 | 86.4 | 4 |
| 1330 | 152 | 21 | 11..97 | 218 | 1.434 | 148 | **4** | 60 | 110.6 | 4 |

Random separations on the same gear sets leave 0-2, 0-1, 2-3, 2-4, 3-4 and 3-4 islands open (5-6
draws each) - the real separation is at the top of that range at `d = 280` and inside it everywhere
else. Same answer as section 2, from a different instrument.

### 4.3 The smallest overlap statement, and the constant it needs

A cover by gears `g_1 .. g_K` at one phase each, with strike sets `S_j` and multiplicities
`mult_i`, satisfies the identities

```
    m  =  |union S_j|  =  sum_j |S_j|  -  E ,        E = sum_i (mult_i - 1) ,
    T  =  sum_{j<k} |S_j ∩ S_k|  =  sum_i C(mult_i, 2)  ,      E  >=  2 T / max_i mult_i .
```

So **no `K` gears can cover if `E > B - m` at every phase choice**, where
`B = sum_j max_phase |S_j|` is the budget of the `K` cheapest gears. In the machine's terms:

> **(S-W3) For every `Kneed = pi(sqrt(6d)) - 3` gears above 7, each used once at the fixed separation
> `2 x 6^{-1} (mod g)` and any reachable phase, the strikes wasted on already-struck islands exceed
> the budget surplus: `sum_i (mult_i - 1) > sum_j |S_j| - m`.**

Its pairwise form, which is the form the brief asks for, follows from `E >= 2T / max mult`:

> **(S-W3-pair) ... any two of those gears strike at least `X` islands of `[1, d)` in common, with
> `X > (max mult / 2) * 2 (B - m) / (K (K - 1))`.**

The constants, computed exactly on the frontier gear sets above, against the CRT-independent mean
pairwise overlap (`4m/(g_j g_k)` averaged over the `K(K-1)/2` pairs):

| `d` | 140 | 280 | 560 | 840 | 1120 | 1330 |
|---|---|---|---|---|---|---|
| `X` needed if multiplicities were `<= 2` | 0.200 | 0.194 | 0.244 | 0.275 | 0.298 | **0.314** |
| CRT-independent mean pairwise overlap | 0.222 | 0.300 | 0.401 | 0.469 | 0.505 | **0.527** |
| ratio: the fraction of CRT the proof needs | 0.90 | 0.65 | 0.61 | 0.59 | 0.59 | **0.60** |
| what the adversary achieves, `T / T_crt` | 1.21 | 0.74 | 0.67 | 0.55 | 0.57 | **0.54** |

Three readings, and they are this branch's contribution to the root.

1. **The ratio the proof needs settles at about 0.59-0.60 while the adversary reaches 0.54-0.57.**
   The pairwise statement is therefore *true* at every arc computed and has **no slack**: it must
   hold with a constant essentially equal to the best the adversary can do. That is Face A2's "no
   margin" reappearing on the adversary, and it is the first time it has been given a number for
   `K`.
2. **The elementary conversion from pairwise overlap to wasted strikes is lossy by `max mult / 2`,
   and the loss is fatal.** The measured maximum multiplicity is 3 or 4, so the rigorous requirement
   is `X > 0.45..0.63` at `d >= 1120` - at or above the CRT-independent mean itself. That asks the
   adversary to be unable to reach even independence, and it beats independence by a factor 1.75.
   **A pairwise-overlap lower bound cannot close the root statement through this conversion.** The
   live object is a lower bound on the wasted strikes `E` directly.
3. `X` and the CRT mean both **grow** with `d` (0.20 -> 0.31 and 0.22 -> 0.53), so no *constant*
   pairwise overlap statement is the right object at all; the object is the ratio, and the ratio is
   flat.

### 4.4 Which weak point this branch decides

The brief's rule: `K_real` above the 80th percentile of the random distribution at every `d` means
W1 (prove the overlap bound for the one-third separation specifically); `K_real` typical means W2
(the one-phase rule alone drives the growth and the constant to prove is the family's).

`K_real` is **typical**: it is the mode of the random distribution at all six arcs where the
distribution was computed, mid-rank percentile 0.500, 0.463, 0.750, 0.483, 0.500 - never above 0.80,
never below 0.46. **The verdict is W2.**

That does not make the machine's coherence a fiction - N-S2, N-S3 and N-S4 are exact and
exception-free, and they act in the direction that *hurts* the adversary. It makes it **invisible to
`K`**: the tail gears carry 3-7 of the 20 gears of an optimal cover and about two islands each, so
halving their reach is worth a fraction of one gear, below the resolution of `K`. The growth of
`K(d)` is bought by the one-phase-per-gear rule (the parent's factor: 20 against 12 at `d = 1120`)
and by *having* a fixed separation at all (the parent's factor 1.5 against `K_free`), and by nothing
about **which** separation it is.

## 5. What holds without exception, with counts (item 7)

| statement | range | exceptions |
|---|---|---|
| the mean overlap of two gears over all `gh` phase pairs is exactly `4m/(gh)`, whatever the separations | 4 gear pairs x 6 separation families x 3 arcs, brute force over every phase pair | **0** of 72 |
| `r (S_g + S_h) = c (mod gh)` for a coherent family `s_g = c r^{-1}` - coherence is closed under CRT | every pair of gears `11..499` (4,095 pairs) x 8 rationals | **0** of 32,490 |
| for the real separation `min(S_g + S_h, gh - (S_g+S_h)) / gh >= 0.3315` (the diagonal is never short) | the same 4,095 pairs | **0** |
| the smaller tooth distance of a real gear is exactly `(g +- 1)/3`, so no gear above `3d` can take an island pair | every gear in `(d, 4d]` at 6 arcs | **0** |
| the real separation's mean tail tooth distance `delta/d` lies outside the whole random range (real 0.688-0.697, random max 0.565) | 6 arcs x 30 random families = 180 draws | **0** |
| the real tail gear reaches fewer island pairs than the random mean | 6 arcs | **0** (below the whole random range at 5 of 6) |
| real pair-capability of tail gears is below every random draw in band `(1.5d, 3d)` | 3 arcs x 10 draws | **0** of 30 |
| `K_real(d)` equals the MODE of the random-separation distribution | 6 arcs, 189 random draws | **0** |
| the random-separation spread `max - min` is at most 2 gears | 6 arcs | **0** |
| every ILP row certified optimal (dual bound = incumbent) | 239 rows | **0** |
| `K_real(d) > pi(sqrt(6d)) - 3` | `d = 560, 840, 1120, 1330` | **0** (it FAILS, with equality, at `d = 140` and `d = 280`) |
| the `pi(sqrt(6d)) - 3` cheapest gears leave at least one island open, real separation | `d = 280 .. 1330` | **0** (they cover everything at `d = 140`) |
| a coherent family with denominator `r in {11, 13}` needs at most as many gears as the real machine deprived of gear `r` | 3 arcs x 4 families | **0** of 12 |

## 6. What is new

Screened line by line against `docs/novel/README.md` - in particular `reachability-landscape`,
`island-witness-integers`, `tooth-counterfactual-percentile` (branch 6, the `F` version of this
question), `cover-half-counter-ladder`, `covering-lp-certificates`, `restricted-covering-certificates`,
`jk-family`, `j2-lower-ladder`, `walk-path-transforms` - and against `cover_number.md` and
`the_wall.md`. The register carries `K(d) = 3 .. 20` and the parent's structural findings; it carries
**nothing** about `K` under any separation but the real one, and branch 6's percentile work is about
`F`, a different object (one machine's record, not the adversarial cover).

**Prior art, named once and stopped.** The four-point CRT shape of two gears' joint strike set and
the `4m/(gh)` mean are the Chinese Remainder Theorem; the free-separation row is the project's own
`jk-family` restatement of the two-class Jacobsthal function. Neither is developed here.

* **N-S5 (the deciding result: the real separation is TYPICAL for `K`).** `K_real(d)` is exactly the
  **mode** of the distribution of `K` over independent random separations at all six arcs where the
  distribution was computed (189 draws), mid-rank percentile 0.500, 0.463, 0.750, 0.483, 0.500,
  never above 0.80. The whole distribution has a spread of at most 2 gears on a `K` that runs 6 to
  22. New, and it decides W3.
* **N-S6 (coherence does not matter either, and the apparent exception is a missing gear).** The
  coherent families with `r = 3, 5, 7` return the same `K` as the real machine and as the random
  mode at every arc (at `d = 1120`: real, 1/5, 2/5, 2/7, 4/7 all return **20**). The families with
  `r = 11, 13` return 1-2 more, and the control shows that is exactly the cost of losing gear 11 or
  13 - which the parent already measured as the one thing the bound `B` changes (N-C5). New.
* **N-S1 (the mean-overlap identity).** For any two gears and **any** separations, the mean overlap
  over all `gh` phase pairs is exactly `4m/(gh)`. One line to prove, 0 exceptions in 72 brute-force
  checks - and it forbids in advance any mechanism in which "coherence raises the overlap". New as a
  statement (the arithmetic is CRT).
* **N-S2 / N-S3 (coherence is closed under CRT, and the machine's diagonal is never short).** If
  every gear takes the same rational `s_g = c r^{-1}`, then `r (S_g + S_h) = c (mod gh)` at every
  pair of gears: the same rational reappears at the composite modulus (32,490 checks, 0 exceptions).
  For the machine that makes the diagonal of the four-point shape `3^{-1} (mod gh)`, an integer never
  below `0.3315 gh` - so two real gears with `gh > 3d` can never place both diagonal strikes inside
  the arc, while random separations do so on 16.5-17.0% of pairs. This is the exact form of "the
  same rational one third at every gear" that W3 asked about. New.
* **N-S4 (where the machine's coherence actually bites: its tail teeth are too far apart).** The real
  separation forces the smaller tooth distance of every gear to be `(g +- 1)/3`, so a tail gear's
  distance is 0.688-0.697 of the arc against 0.502 for a random separation - **outside the entire
  random range at all six arcs, 180 draws, 0 exceptions** - and because the number of island pairs at
  a distance falls linearly in the distance, the real tail gear reaches about half as many island
  pairs, and is pair-capable on 0.14-0.25 of the tail band against 0.20-0.38 for random, below every
  random draw. So the machine's coherence is real, exact, and works *against* the adversary; it is
  simply too small to move `K`, because the tail carries 3-7 gears at two islands each. New.
* **N-S7 (the frontier the root needs, with its constant).** For the `pi(sqrt(6d)) - 3` cheapest
  gears the maximum coverage is exactly certified: they leave **0, 2, 3, 3, 4, 4** islands open at
  `d = 140, 280, 560, 840, 1120, 1330`. The pairwise-overlap statement that would prove the root
  needs the average pairwise overlap to be at least `0.20, 0.19, 0.24, 0.28, 0.30, 0.31` islands
  against a CRT-independent average of `0.22, 0.30, 0.40, 0.47, 0.51, 0.53` - a ratio that settles at
  **0.59**, while the adversary actually achieves 0.54-0.57 of the CRT value. So the statement is
  true at every arc and has **no slack**; and the elementary conversion from pairwise overlap to
  wasted strikes loses a factor `max mult / 2 = 1.5..2`, which is exactly enough to make that route
  vacuous. New as the first quantitative frontier for the adversarial cover.
* Filed, not claimed: the real optimum's total pairwise overlap is the lowest in the comparison
  (`T/T_crt = 0.300` at `d = 280` and `0.400` at `d = 560`, against random ranges 0.380-0.924 and
  0.396-0.717) and its budget ratio is below the random median at four of five arcs - two more
  measurements in which the real teeth are *better* than random for the adversary, not worse.

## 7. Verdict

**W2, not W1: the one-phase rule drives `K`, and the constant to prove is the family's.**

`K(d)` under the real separation is `6, 9, 14, 17, 20, 22` at `d = 140 .. 1330`. Under 189
independent random separations, with the one-phase rule and the reachable-phase coset held fixed, it
is the same number - the mode of the distribution at every arc, mid-rank percentile between 0.46 and
0.75, spread at most two gears. Under coherent separations `c r^{-1}` with `r = 3, 5, 7` it is the
same number again (all five families return 20 at `d = 1120`). The two coherent families that differ
do so because the rational has no value at the gear `g = r`, and the control shows their whole excess
is the cost of the missing gear. **Nothing about the value of the separation is visible in `K`.**

That does not mean the machine's coherence is absent - three exact, exception-free coherence
statements come out of this branch (the CRT closure `r(S_g+S_h) = c mod gh`, the diagonal never
shorter than `gh/3`, and tail tooth distances pinned at `(g±1)/3` and hence at 0.69 of the arc
against 0.50 at random, outside the entire random range at 180 draws). It means those statements act
where `K` cannot see them: on the tail gears, which carry three to seven of an optimal cover's twenty
gears and about two islands each. Halving their reach is worth a fraction of a gear.

So the target statement is about the **family** - every prime above 7 taking two classes at *some*
fixed separation, one phase each - and the constant to prove is the family's constant, not the
machine's. The machine only has to be one member. The branch also puts a number on what that
statement costs: the `pi(sqrt(6d)) - 3` cheapest gears leave 2, 3, 3, 4, 4 islands open at
`d = 280 .. 1330` (certified), the requirement `K(d) > pi(sqrt(6d)) - 3` is met by exactly one gear
from `d = 560` on and fails at `d = 140, 280`, and the pairwise-overlap route to it needs the
adversary held at 0.59 of CRT-independence while it reaches 0.54-0.57 - true, with no slack, and
unreachable through the elementary multiplicity conversion, which loses the factor that matters.

## 8. Dead ends (do not re-enter)

* **"The real one-third separation drives `K`."** Refuted at six arcs with 189 random draws:
  `K_real` is the mode of the random distribution every time, percentile never above 0.75. Do not
  re-open the question of whether the machine's teeth are special *for the adversarial cover
  number*.
* **"Coherence (the same rational at every gear) raises the pairwise overlap."** Refuted twice: the
  mean overlap over all phase pairs is exactly `4m/(gh)` for every separation (a one-line identity,
  so no separation can move it), and the reachable-phase correlation `C(g,h)` puts the real
  separation at percentile 0.83, 0.37, 0.97 at `d = 280, 560, 1120` - no consistent sign, 5% spread.
* **"The real separation makes the optimal cover wastier."** Refuted: its budget ratio is below the
  random median at four of five arcs and its optimum's total pairwise overlap is the *lowest* in the
  comparison.
* **A coherent family with denominator `r` equal to a gear.** `c r^{-1}` is undefined at `g = r`, so
  `coh:c/11` and `coh:c/13` are the real problem minus gear 11 or 13; their `+1` and `+2` in `K` is
  the parent's N-C5, not a separation effect. Any future family comparison must use `r` outside the
  gear range or delete the same gear from every family.
* **The pairwise-overlap lower bound as a route to `K(d) > pi(sqrt(6d)) - 3`, via
  `E >= 2T/max mult`.** The conversion loses `max mult / 2`, measured at 1.5-2.0, and the rigorous
  requirement then exceeds the CRT-independent pairwise mean itself - i.e. it asks the adversary to
  be unable to reach independence, which it beats by a factor 1.75. The live object is a lower bound
  on the **wasted strikes** `E = sum_i (mult_i - 1)` directly, not on `T`.
* **`P5c` as stated** ("every gear below `3d` is pair-capable for the real separation"): false by a
  factor of five, because a tooth distance must also be an island difference mod 35 (9 of 35
  residues). The correct exact statement is the converse half: no gear *above* `3d` is pair-capable
  for the real separation, and the pair-capable fraction below it is 0.14-0.34.
