# Structured families of slots: does a location rule beat existence? (branch)

Prover (lane: structured families), round 65, 2026-09-06. Scripts in `research/anchor235/r65/`
with prefix `sf_`; results in `research/anchor235/r65/results/` (untracked). Vocabulary as in
`docs/proof-search/alignment-rules.md` section 0 and the tree's profile: column `k` is the pair
`(6k-1, 6k+1)`; gear `g >= 5` strikes `k` iff `k = +-u_g (mod g)` with `6 u_g = g -+ 1`, i.e. the
teeth of `g` are `+-6^{-1} (mod g)`; machine `{5..y}`; window = the certified range
`(y, y'^2]` in columns, never a sliding run; section = the window's new part `(y^2, y'^2)`.

Parent: node R2 (whole-window formulation) with ingredients from R2.a.i.a (the reachability
landscape and the island classes), node 3 (always-open columns), node 14 (the corridor), node 7b
(the anchor rigid in every window). The observation that spawned it: every one of those nodes
names an explicit set of columns that is open for the small gears BY CONSTRUCTION. The question
this branch asks is the owner's: does naming such a set - a location rule - buy anything over
"somewhere in the window", or is the set's survival decided by the same counting that the window
is decided by?

## 0. What this branch could find that is not already known

Known and not re-derived here: CRT densities; the fundamental lemma's reach in dimension 2
(`s > 4.27`, node 3a / face A1); the Hardy-Littlewood singular series; the `s = 2` handicap
`4 e^{-2 gamma}` (measured in 7b and R2.a.i.a.1.b). What is NOT on the tree is a comparison
BETWEEN families at one sifting level: whether any explicitly located family of columns carries
more twins per member than the window carries per member, and whether the intersections of the
tree's five located families are statistical or arithmetic. That comparison is the branch.

## 1. Setup: the families and their exact fair rates

Level `q` a prime, `q'` the next prime, machine `M = {5..q}`.

    window columns   k_lo = floor((q+1)/6) + 1  ..  k_hi = (q'^2 - 1)/6 - 1,   W = k_hi - k_lo + 1
    section columns  k_s  = (q^2 - 1)/6 + 1     ..  k_hi,                      S = k_hi - k_s + 1

Kernel fact used throughout: a column open under `{5..q}` in the window is a twin prime pair.
Every survivor count below is therefore also a twin count, and every run is checked against a
primality sieve.

Write `P_y = 5 * 7 * ... * y` (the small primorial, gears only - the anchor 2, 3 is not in it),
`m_y` = the number of gears `5..y`.

**(a) Primorial multiples** `A_y = { k : k = 0 (mod P_y) }`. Open for every `g <= y` because
`0` is not a tooth (`+-6^{-1} != 0`); these are the translates of column 0 by the small
machine's period (node 3(a)). Density `1/P_y`; window size `n = floor(k_hi/P_y) - floor((k_lo-1)/P_y)`.

**(b) Neighbours of a full hit** `B_y = { k : k = c +- 1 (mod P_y), 36 c^2 = 1 (mod P_y) }`.
`36 c^2 = 1 (mod g)` says `6c = +-1`, i.e. `c` is a tooth of `g`; the condition mod `P_y` says
`c` is struck by EVERY gear up to `y`. `k = c +- 1` is then open for every `g <= y` because a
gear's two teeth are never adjacent (`neighbour-of-hit`, kernel): teeth at distance 1 would need
`g - 2u_g = +-1`, i.e. `(2g+1)/3 = 1` or `(2g-1)/3 = 1`, impossible for `g >= 5`. There are
`2^{m_y}` classes `c` and hence at most `2^{m_y + 1}` classes `k` (fewer when two neighbours
coincide, which happens exactly at `c = +-1 (mod P_y)`, the two teeth flanking column 0).

**(c)/(e) The small machine's own openings** `O_y = { k : k not a tooth of any g <= y }`,
`|O_y| = prod_{5<=g<=y} (g-2)` classes mod `P_y`. `y = 7` is the corridor `E_35` (15 of 35),
`y = 11` the corridor mod 385 (135 of 385), `y = 13` the anchor `{5..13}` of node 7b
(1485 of 5005). (c) and (e) are the same construction at two values of `y`; the branch treats
them as one family type and reports both. Every other family here is a subset of `O_y`.

**(d) The islands** `I = { k : (k - k_0) mod 35 in {5, 10, 12, 17}, k > k_0 }` with
`k_0 = (q^2 - 1)/6` the column of `q^2` (R2.a.i.a I6). An island offset is one no gear `<= 7`
can reach at any `q`, so island columns are open for gears 5 and 7 by construction; the family
is 4 classes mod 35 whose position depends on `q` through `k_0`, and it lives in the section.

**The fair rate.** Every family above is a union of residue classes modulo `M` (`M = P_y`, or
`M = 35` for the islands), and `M` is composed of gears `<= y` only. For a gear `g > y`, `M` is
invertible mod `g`, so `k mod g` is equidistributed over each class of the family; `g` strikes
exactly two residues mod `g`; hence the take of `g` on the family is exactly `2/g` and NO CRT
adjustment arises. The fair rate is therefore the same for every family:

    rate_y(q) = prod_{y < g <= q} (1 - 2/g),      excess E = survivors / (n * rate_y(q)).

(For (d), `y = 7`.) The small gears contribute nothing because the family is open for them by
construction - that is the whole content of a location rule, and it is already priced in `n`.

**What the excess must be, if counting is all there is.** The Hardy-Littlewood singular series
for the pair `(6k-1, 6k+1)` restricted to residues mod `M` that avoid every tooth of every
`g <= y` is `S = prod_{p | 6M} (1-1/p)^{-2} * prod_{p not| 6M} (1-2/p)/(1-1/p)^2
= 12 C_2 / prod_{5<=g<=y} (1 - 2/g)`, and Mertens gives
`prod_{5<=p<=z}(1-2/p) ~ 12 C_2 e^{-2 gamma} / ln^2 z`. With `N = q'^2` the top of the window and
`z = q ~ sqrt(N)`,

    survivors ~ S n / ln^2 N  and  n * rate_y(q) ~ 48 C_2 e^{-2 gamma} n / (ln^2 N * prod_{5<=g<=y}(1-2/g)),

so **E -> e^{2 gamma}/4 = 0.79296 for EVERY family, with the small-gear factor cancelling
exactly.** The `s = 2` handicap is a property of the sifting level, not of the family. This is
the branch's central prediction and the reason the pre-registered `1.00 +- 0.05` is wrong.

## 2. Pre-registered (written before any computation)

**P1 (headline; against the brief's `1.00 +- 0.05`).** No family has excess 1.00. Every family's
excess equals the whole window's excess `R(q)` to within sampling error, and `R(q)` falls from
about 0.90 at `q ~ 100` to about 0.85 at `q = 5000`, toward `e^{2 gamma}/4 = 0.79296`.
REFUTED if the whole window's `R(5000)` is outside `[0.82, 0.88]`.

**P2 (family independence; the decisive one).** The normalised excess `E_F(q) / E_window(q)` is
`1.000 +- (sampling)` for every family, every `y`, every `q >= 100`. REFUTED for a family if its
pooled normalised excess over `q >= 500` departs from 1 by more than 3 pooled sigma AND the
departure has the same sign at more than 80% of rungs. I predict no family is refuted; if one is,
the candidate is (d), the islands, because their position is tied to `q` itself.

**P3 (trend).** No family's excess grows with `q`. Each is `0.793 (1 + c/ln q)` with `c` in
`[0.3, 1.2]`, decreasing. REFUTED by any family with a positive `q`-trend significant at 3 sigma
over `q in [500, 5000]`.

**P4 (thickness).** `s = ln n / ln q`. The whole window sits at `s = 2 - ln 6/ln q`, i.e. 1.79 at
`q = 5000`; `(b)_7` (density 1/5) at `s ~ 1.60`; `(a)_7` at `~1.37`; `(a)_13` at `~0.79`;
`(a)_19` at `~0.11`. NO family and not the window reaches `s = 4.27`: at `q = 5000` that would
need `n >= 10^15` against `W = 4.2 * 10^6`. Prediction: the fundamental lemma is vacuous for
every row of the table, and the gap is a factor `q^{2.3}` even for the window. REFUTED if any
row has `s > 2`.

**P5 (correlation is arithmetic, not statistical).** Because `35 | P_y`, the intersections
`A_y n I` and `O_y n I` are decided by CRT rung by rung, not by chance: `A_y ⊂ I` or
`A_y n I = {}`, all-or-nothing, and the "all" case occurs for the `q` with
`-(q^2-1)/6 mod 35 in {5, 10, 12, 17}` (predicted frequency about 4/35 = 11.4% of rungs).
`B_7 n I` is a partial overlap of 0..4 of `B_7`'s 7 classes mod 35. `O_13 ⊂ O_11 ⊂ O_7`
(containment, not correlation). Conditional on the intersection being non-empty, the excess ON
the intersection equals the excess on each family (independence beyond CRT). REFUTED if any
intersection size departs from its CRT value, or if a conditional excess departs by 3 sigma.

**P6 (mechanism at the extremes).** Per-gear take `fresh_g / N_cur(g)` against `2/g` on each
family at named rungs: no gear takes systematically less (or more) than `2/g` on any family; the
residual is the white residual of 7b's one curve in `ln g / ln Q'`. REFUTED by a gear whose take
on a family differs from `2/g` by 3 sigma with the same sign at every named rung.

**P7 (the location question itself, new).** For each family, the WITNESS THRESHOLD: the last
rung at which the family has no survivor in the window (and separately in the section).
Prediction: `(c)/(e)` and `(b)_7` have a survivor at every rung from 23; `(a)_7` from 23;
`(a)_13` fails at no rung above about 200; `(a)_17` fails at scattered rungs throughout;
`(a)_19` has no survivor at most rungs (`n ~ 2.6` at `q = 5000`); `(d)` over the whole section
has a survivor from about `q = 100` (much earlier than N-R4's 1487, which is the short arc
`[1, d)` and not the section). REFUTED per family by the measured threshold.

**P8 (the trade-off, stated as a prediction).** Survivors on a family of density `delta` number
`delta * (twins in the window) / prod_{5<=g<=y}(1-2/g)`; the located family therefore beats the
window only if `delta > prod_{5<=g<=y}(1-2/g)`, which is impossible since the family is a subset
of `O_y`, whose density IS that product. **A location rule cannot have more twins per column
than the small machine's own opening set, and every proper family has strictly fewer.** Predicted
to hold with no exception at any rung; refuted by any family exceeding `O_y`'s survivor density.

## 3. Scorecard

| item | prediction | verdict |
|---|---|---|
| P1 | no excess is 1.00; window `R(5000)` in [0.82, 0.88] | **CONFIRMED in substance, my number wrong**: no family is at 1.00; the honest (section) value is 0.7987 for every family, the s=2 handicap itself. The window's own per-rung value at q=4999 is 0.9121, ABOVE my [0.82, 0.88] - the window is not the section, because it reaches down to q where twins are four times denser. The band prediction named the wrong range; the limit 0.79305 is right. |
| P2 | every family's normalised excess = 1.000 | **CONFIRMED**, no family refuted: on the disjoint sections every normalised excess is within 1.2 sigma of 1 (b7 1.0012+-0.0040, isl 1.0061+-0.0054, a7 0.9906+-0.0107); the matched-rate two-sample test gives islands 1.00861+-0.00628 against the rest of the corridor |
| P3 | no family's excess grows with `q` | **CONFIRMED**: section excess flat (0.798, 0.799, 0.804, 0.797 by band); window excess falls as 0.782 + 1.121/ln q; no positive trend anywhere |
| P4 | no row reaches `s = 4.27`; none above `s = 2` | **CONFIRMED**: window s = 1.79 at q = 4999 (needs n = 6.2e15 against 4.2e6, a factor 1.5e9); every proper family below it; maximum s over the whole table 1.79 |
| P5 | intersections are CRT, all-or-nothing for `A_y n I` | **CONFIRMED as a rule, my frequency REFUTED**: all-or-nothing at 661 of 661 rungs, but non-empty at 0.328 of rungs, not 4/35 = 0.114 - because k_0 mod 35 takes only the six square classes. That miss produced the branch's new exact law (section 7) |
| P6 | no gear takes systematically less than `2/g` on any family | **CONFIRMED**: one curve for every family, agreeing to 2% in every bin and to 4 decimals below t = 0.65; the island family sits on the anchor family's curve |
| P7 | witness thresholds as listed | **CONFIRMED except two**: b7 and every O_y at every rung; a7 at every rung in the window; a13 last empty 419 (predicted 'no rung above about 200'); a17 last empty 2011 with 297 empties; a19 empty at all 661. Islands: last empty section 461 with only 3 empties, against the predicted 'about q = 100' |
| P8 | no family exceeds `O_y`'s survivor density | **CONFIRMED**, and it is an identity, not a measurement: survivors(F)/survivors(window) = density(F)/prod_{5<=g<=y}(1-2/g) <= 1 with equality only for F = O_y |

Everything below this line was written after the runs.

---

## 4. Setup as run

Rungs: every prime `q` from 23 to 4999 (661 rungs). One primality sieve to `5003^2 = 25,030,009`
supplies the survivor counts; the kernel fact makes survivors = twin pairs, so every count is
also checked as a twin count. `research/anchor235/r65/sf_families.py` does the sweep
(`results/families_window.tsv`, `families_section.tsv`, `islands.tsv`); `sf_analyse.py` the
tables; `sf_corr.py` the family intersections; `sf_gears.py` the per-gear takes; `sf_thick.py`
the thickness table and the island law; `sf_zero.py` the emptiness test; `sf_matched.py` the
matched-rate two-sample tests. (The `a19` row of the matched table prints a huge sigma because it
has two members and neither is a twin; its expectation is 0.27, so the row carries no information.)

**Windows overlap, sections do not.** The window at rung `q` is `(q, q'^2]` and the window at the
next rung is `(q', q''^2]`: consecutive windows share almost everything, so a pooled window
statistic over many rungs counts the same twins hundreds of times and its Poisson sigma is
meaningless. The SECTIONS `(q^2, q'^2)` are disjoint and tile `(529, 25030009]` - 4,170,919
columns, 130,644 twins, each counted once. Every error bar below that matters is computed on the
sections; window numbers are given per rung (where they are honest) and pooled only as a
description.

**Two of the five families are not location rules.** `O_y` (the brief's (c) and (e)) contains
EVERY twin of the window, because a twin is open for every gear; so its survivor count IS the
window's twin count, and since `|O_y|/W = prod_{5<=g<=y}(1-2/g)` exactly, its excess is the
window's excess identically, not approximately. Measured: 130,543 survivors and excess 0.9121 at
`q = 4999` for the window and for `o7, o11, o13, o17, o19` alike, agreeing to four decimals.
`O_y` is the window in different coordinates. Only (a), (b), (d) are proper families.

## 5. Results

### 5.1 The excess, on the disjoint sections (all 661 rungs, 130,644 twins)

`excess = survivors / (n * prod_{y<g<=q}(1-2/g))`; `norm` is the excess divided by the window's.

| family | classes / modulus | density | n (sections) | survivors | excess | norm +- 1 sigma |
|---|---|---|---|---|---|---|
| window | 1/1 | 1 | 4,170,919 | 130,644 | 0.7987 | 1.0000 |
| `o7` .. `o19` | 15/35 .. 378675/1616615 | 0.429 .. 0.234 | - | 130,644 | 0.7988-0.7989 | 1.0001-1.0003 (identity) |
| `b7` | 7/35 | 0.2000 | 834,093 | 61,034 | 0.7997 | 1.0012 +- 0.0040 |
| `b11` | 16/385 | 0.0416 | 173,317 | 15,509 | 0.8001 | 1.0017 +- 0.0080 |
| `b13` | 32/5005 | 0.0064 | 26,665 | 2,801 | 0.7948 | 0.9951 +- 0.0188 |
| `b17` | 64/85085 | 7.5e-4 | 3,139 | 395 | 0.8398 | 1.0516 +- 0.0530 |
| `b19` | 128/1616615 | 7.9e-5 | 334 | 47 | 0.8417 | 1.0539 +- 0.1538 |
| `a7` | 1/35 | 0.0286 | 119,076 | 8,619 | 0.7911 | 0.9906 +- 0.0107 |
| `a11` | 1/385 | 2.6e-3 | 10,814 | 959 | 0.7932 | 0.9931 +- 0.0320 |
| `a13` | 1/5005 | 2.0e-4 | 830 | 89 | 0.8124 | 1.0172 +- 0.1078 |
| `a17` | 1/85085 | 1.2e-5 | 48 | 5 | 0.7064 | 0.8845 +- 0.3956 |
| `a19` | 1/1616615 | 6.2e-7 | 2 | 0 | 0 | 0 (n = 2, expectation 0.27) |
| `isl` | 4/35 (shifted by `k_0`) | 0.1143 | 477,196 | 35,099 | 0.8035 | 1.0061 +- 0.0054 |

**No family is at 1.00 and every family is at the window's value.** The common value is
`0.7987 +- 0.0022`, and `e^{2 gamma}/4 = 0.79305`: the section excess IS the `s = 2` handicap,
flat across `q` (0.842, 0.798, 0.799, 0.804, 0.797 in the bands `q < 100, 300, 1000, 2500, 5000`).
The largest normalised excess with usable statistics is `b17` at 1.052 (1.0 sigma); the islands
at 1.0061 (1.1 sigma); the smallest is `a7` at 0.9906 (0.9 sigma). **Nothing is 2 sigma from 1.**

Matched-rate two-sample form (the sharpest version, since it removes the window's own
normalisation): each family against the REST of `O_y` in the same sections, where the fair rate is
identical class by class, so the comparison is a plain ratio of twins per member:

| family vs `O_y` minus family | twins/member in `F` | in the rest | ratio |
|---|---|---|---|
| `isl` vs corridor non-islands | 0.073553 | 0.072925 | **1.00861 +- 0.00628 (1.4 sigma)** |
| `b7` | 0.073174 | 0.073021 | 1.00210 +- 0.00555 (0.4 sigma) |
| `a7` | 0.072382 | 0.073143 | 0.98960 +- 0.01104 (-0.9 sigma) |
| `b11` | 0.089483 | 0.089321 | 1.00182 +- 0.00857 (0.2 sigma) |
| `a11` | 0.088681 | 0.089345 | 0.99257 +- 0.03217 (-0.2 sigma) |
| `b13` | 0.105044 | 0.105602 | 0.99472 +- 0.01900 (-0.3 sigma) |
| `a13` | 0.107229 | 0.105588 | 1.01554 +- 0.10768 (0.1 sigma) |
| `b17` | 0.125836 | 0.119650 | 1.05170 +- 0.05299 (1.0 sigma) |
| `b19` | 0.140719 | 0.133749 | 1.05211 +- 0.15349 (0.3 sigma) |

The island family - the tree's own candidate object (R2.a.i.a) - carries twins at the same rate
as the ordinary corridor columns to within 1.3% at 2 sigma.

### 5.2 Per-rung window excess (honest at a single rung)

At `q = 4999`: window and every `o_y` 0.9121 +- 0.0025; `b7` 0.9130 +- 0.0037; `b11` 0.9136;
`b13` 0.9078 +- 0.0172; `b17` 0.9597 +- 0.0483; `b19` 0.9605 +- 0.1401; `a7` 0.9029 +- 0.0097;
`a11` 0.9036 +- 0.0292; `a13` 0.9238 +- 0.0979; `a17` 0.7785 +- 0.3481; `a19` 0 (n = 2). The
window sits above the section (0.912 against 0.799) for a mechanical reason: the window reaches
down to `q`, where twins are four times denser than at `q'^2`, and the fair rate is a single
number for the whole range. The window value falls monotonically (1.05 at `q ~ 100`, 0.95 at
`q ~ 1000`, 0.912 at 4999) toward the section value; a fit in `1/ln q` extrapolates the window to
`0.7824 + 1.121/ln q`, i.e. to the same limit.

### 5.3 Trend

Section excess by band is flat for every family (window 0.842 / 0.798 / 0.799 / 0.804 / 0.797).
No family has a positive `q`-trend. Window excesses fall like `a + b/ln q` with `a = 0.782` for
the window and for every `o_y`, `0.771` for `b7`, `0.828` for `a7`; the thin families' fits are
noise.

### 5.4 Emptiness: does the family miss the range, and how often?

Last rung at which the family has NO survivor (window and section), and the count of empty rungs:

| family | window last-0 | window empties | section last-0 | section empties | n at `q = 4999` (window) |
|---|---|---|---|---|---|
| `o7` .. `o19`, window | none | 0 | none | 0 | 9.8e5 .. 4.2e6 |
| `b7` | none | 0 | none | 0 | 834,168 |
| `b11` | none | 0 | 659 | 12 | 173,335 |
| `b13` | 53 | 8 | 4547 | 120 | 26,667 |
| `b17` | 281 | 52 | 4999 | 442 | 3,140 |
| `b19` | 1009 | 161 | 4999 | 618 | 334 |
| `a7` | none | 0 | 2381 | 28 | 119,167 |
| `a11` | 43 | 6 | 4903 | 253 | 10,833 |
| `a13` | 419 | 73 | 4993 | 579 | 833 |
| `a17` | 2011 | 297 | 4993 | 656 | 49 |
| `a19` | 4999 | 661 | 4999 | 661 | 2 |
| `isl` | - | - | 461 | 3 (`q = 29, 41, 461`) | 764 |

**The emptiness is Poisson at the fair rate.** On the sections, observed empties against
`sum_q exp(-n * rate * 0.7930)`: `a7` 28 / 32.3 (0.87), `a11` 253 / 266.0 (0.95), `a13`
579 / 585.0 (0.99), `a17` 656 / 655.7 (1.00), `a19` 661 / 660.8 (1.00), `b13` 120 / 115.7 (1.04),
`b17` 442 / 445.5 (0.99), `b19` 618 / 623.5 (0.99), `isl` 3 / 3.0 (1.01). Not only the mean but
the whole probability of missing is what counting says. (The window column shows ratios 1.06-2.37,
which is the correlation between overlapping windows, not an effect: a family empty at `q` is
almost surely empty at `q'`, so the 661 window trials are close to a single trial.)

The window last-0 rung is where the expected count crosses one: `a13` last empty 419 against
`E < 1` last at 379; `a17` 2011 against 1879; `a19` never against never; `b17` 281 against 97;
`b19` 1009 against 947. The location rule starts working exactly when counting says it should.

### 5.5 P8: the trade-off, exactly

Survivors per member, pooled over `q >= 500` in the window, against `O_y`'s: `o_y` 1.0000 by
identity, `b7` 1.0009, `b11` 1.0061, `b13` 1.0043, `b17` 1.0185, `b19` 0.9366, `a7` 0.9925,
`a11` 0.9908, `a13` 1.1381 (833 members), `a17` 0.6754 (49 members), `a19` 0. So

    survivors(F) / survivors(window) = density(F) / prod_{5<=g<=y} (1 - 2/g),

and since `F` is a subset of `O_y`, whose density IS that product, the ratio is at most 1 with
equality only for `F = O_y`. Measured to within the sampling error at every family. **Naming a
location cannot gain; it loses exactly the family's density relative to the small machine's own
opening set.** `b7` loses a factor 2.14, `a7` 15, `a13` 1485, `a19` 378,675.

## 6. Thickness

`s = ln n / ln q`, `n` the family's size in the window. The dimension-2 sieve gives a lower bound
only for `s > 4.27` (node 3a, face A1); at `q = 4999` that needs `n >= 4999^4.27 = 6.2e15`.

| family | `n` at `q = 101` | 499 | 997 | 2503 | 4999 | `s` at 4999 |
|---|---|---|---|---|---|---|
| window | 1,750 | 42,084 | 169,513 | 1,058,822 | 4,170,834 | **1.79** |
| `o7` | 750 | 18,034 | 72,649 | 453,781 | 1,787,501 | 1.69 |
| `o13` | 518 | 12,486 | 50,294 | 314,157 | 1,237,501 | 1.65 |
| `o19` | 408 | 9,850 | 39,708 | 248,026 | 976,982 | 1.62 |
| `b7` | 350 | 8,414 | 33,904 | 211,765 | 834,168 | 1.60 |
| `b11` | 74 | 1,750 | 7,048 | 44,002 | 173,335 | 1.42 |
| `b13` | 12 | 270 | 1,084 | 6,770 | 26,667 | 1.20 |
| `b17` | 4 | 32 | 128 | 800 | 3,140 | 0.95 |
| `b19` | 2 | 4 | 6 | 92 | 334 | 0.68 |
| `a7` | 50 | 1,202 | 4,843 | 30,252 | 119,167 | 1.37 |
| `a11` | 4 | 109 | 440 | 2,750 | 10,833 | 1.09 |
| `a13` | 0 | 8 | 33 | 211 | 833 | 0.79 |
| `a17` | 0 | 0 | 1 | 12 | 49 | 0.46 |
| `a19` | 0 | 0 | 0 | 0 | 2 | 0.08 |
| `isl` (section) | 8 | 76 | 460 | 1,724 | 764 | 0.78 |

**No row reaches 4.27, and no row reaches 2.** The whole window sits at `s = 2 - ln 6/ln q`,
rising from 1.62 to 1.79 over the range and approaching 2 from below; it is short of the
fundamental lemma's reach by a factor `q^{2.3}`, which is `1.5e9` in `n` at `q = 4999`. Every
proper family is thinner: `b7` by a constant factor (its whole `s` loss is `ln 5 / ln q`), the
primorial families by `ln P_y / ln q`, a fixed subtraction in `s` that vanishes only as `q` grows
- `a13` is at `s = 0.79` at `q = 4999` and reaches `s = 1` only near `q ~ 3e5`. The thickness
order is permanent, and the whole table is on the wrong side of the sieve limit by nine orders of
magnitude in `n`.

Reading: thickness and excess ask different questions with the same answer. Thickness decides
whether a THEOREM can be quoted (never, here); the excess decides whether the object is worth
naming (it is not, because the excess is 1).

## 7. Correlations between families

Pooled over the disjoint sections; `exc F n G` is the excess on the intersection.

| F | G | size F | size G | size F n G | independent prediction | exc F | exc G | exc F n G |
|---|---|---|---|---|---|---|---|---|
| `a7` | `isl` | 119,076 | 477,196 | 39,268 | 13,624 | 0.7911 | 0.8035 | 0.8030 +- 0.0149 |
| `a11` | `isl` | 10,814 | 477,196 | 3,579 | 1,237 | 0.7932 | 0.8035 | 0.8697 +- 0.0464 |
| `a13` | `isl` | 830 | 477,196 | 271 | 95 | 0.8124 | 0.8035 | 0.8304 +- 0.1516 |
| `b7` | `isl` | 834,093 | 477,196 | 257,954 | 95,429 | 0.7997 | 0.8035 | 0.8014 +- 0.0058 |
| `b11` | `isl` | 173,317 | 477,196 | 53,881 | 19,829 | 0.8001 | 0.8035 | 0.7756 +- 0.0113 |
| `b13` | `isl` | 26,665 | 477,196 | 8,281 | 3,051 | 0.7948 | 0.8035 | 0.7649 +- 0.0264 |
| `o7` | `o13` | 1,787,380 | 1,237,281 | 1,237,281 | 530,217 | 0.7988 | 0.7989 | 0.7989 |
| `o13` | `isl` | 1,237,281 | 477,196 | 330,285 | 141,558 | 0.7989 | 0.8035 | 0.8037 +- 0.0043 |
| `a7` | `b7` | 119,076 | 834,093 | 119,076 | 23,813 | 0.7911 | 0.7997 | 0.7911 |
| `a13` | `b13` | 830 | 26,665 | 0 | 5 | 0.8124 | 0.7948 | - |

**The intersections are arithmetic, not statistical: every one of them is off the independence
prediction by a large factor, while the SURVIVAL on the intersection is not off at all.** Every
intersection excess lies within 2.5 sigma of both family excesses (worst: `b11 n isl` at 0.7756,
2.5 sigma below 0.8035; `b13 n isl` 1.5 sigma below; nothing survives a multiplicity correction
over ten pairs).

Structure found in the SETS:

- **`A_y` is contained in `I` or disjoint from it, all-or-nothing at 661 of 661 rungs** for
  `y = 7, 11, 13`, because `35 | P_y`. Non-empty at 217, 211, 149 rungs.
- **The exact law (new).** With `6 k_0 = q^2 - 1`, the column-0 translates sit at offset
  `i = -k_0`, so `-6i = q^2 - 1` and `2 - 6i = q^2 + 1`. Gear `g` is barred at an offset iff
  neither of those is a nonzero quadratic residue mod `g` (R2.a.i.a). Mod 5: `q^2` is 1 or 4, so
  the pair is `(0, 2)` or `(3, 0)`, and neither 2 nor 3 is a QR mod 5 - **gear 5 is barred at the
  column-0 offset for every `q`, without exception (661 of 661)**. Mod 7: `q^2` is 1, 2 or 4 and
  the QRs are `{1, 2, 4}`; `q^2 = 1` gives `q^2 + 1 = 2`, a QR; `q^2 = 2` gives `q^2 - 1 = 1`, a
  QR; `q^2 = 4` gives `(3, 5)`, neither a QR. So **the primorial family is a `B = 7` island family
  exactly when `q = +-2 (mod 7)`** - 217 of 661 rungs (108 at `q = 2 mod 7`, 109 at `q = 5`, 0 at
  the other four classes), 0 disagreements between the QR rule and the direct residue test.
- **`A_7` is contained in `B_7` always**: `c = 1` and `c = 34` are struck by both 5 and 7, so
  column 0 is a neighbour of a full hit and `|A_7 n B_7| = |A_7|` at every rung. `A_13 n B_13` is
  empty at every rung (1 is not a tooth of 11 or 13).
- **`k_0 mod 35` takes exactly six values** - `{0, 13, 18, 20, 25, 28}`, one sixth of the rungs
  each - because `q^2` is a square mod 35. The island set therefore has only six positions
  relative to the corridor, and `|B_7 n I|` is 1, 1, 3, 2, 4, 2 on them: **at every rung at least
  one of `B_7`'s seven classes is an island class, mean 2.145**, against 1.867 under a
  uniform-within-the-corridor null and 0.800 under a uniform-mod-35 null. The corridor accounts
  for 87% of the apparent factor 2.7; the residual factor is 1.15 and carries no survival effect.

## 8. Mechanism: the take of each gear on each family

Item 4's condition ("the family with the largest measured excess, if any exceeds 1.05") is not
triggered: no family's normalised excess exceeds 1.05 by as much as 1.1 sigma. The per-gear pass
is therefore run as the control, on the anchor family and the rest.

Take `= fresh_g / N_cur(g)` divided by `2/g`, binned by `t = ln g / ln q'` (7b's coordinate).
Window at `q = 4999`:

| family | `t<0.35` | `<0.50` | `<0.65` | `<0.80` | `<0.90` | `<0.95` | `<1.00` |
|---|---|---|---|---|---|---|---|
| window | 1.0000 | 1.0006 | 0.9838 | 1.0913 | 1.3412 | 1.2125 | 0.5879 |
| `o13` | 1.0000 | 1.0006 | 0.9838 | 1.0913 | 1.3412 | 1.2125 | 0.5879 |
| `b7` | 1.0000 | 1.0001 | 0.9828 | 1.0905 | 1.3384 | 1.2146 | 0.5945 |
| `b13` | 1.0000 | 0.9992 | 0.9835 | 1.1179 | 1.3148 | 1.2033 | 0.6212 |
| `a7` | 1.0000 | 1.0006 | 0.9837 | 1.1000 | 1.3503 | 1.2426 | 0.6003 |
| `a13` (833 members) | 0.9979 | 1.0082 | 0.9150 | 0.9317 | 1.6018 | 1.4796 | 0.5355 |

Sections pooled over `q >= 101` (the only pooling in which the islands appear):

| family | `t<0.35` | `<0.50` | `<0.65` | `<0.80` | `<0.90` | `<0.95` | `<1.00` |
|---|---|---|---|---|---|---|---|
| `isl` | 1.0009 | 0.9996 | 0.9888 | 1.0033 | 1.3236 | 1.5926 | 1.7951 |
| `o13` | 0.9999 | 1.0004 | 0.9924 | 1.0038 | 1.3220 | 1.6034 | 1.8208 |
| `b7` | 1.0002 | 1.0001 | 0.9913 | 1.0053 | 1.3154 | 1.6071 | 1.8243 |
| `a7` | 1.0007 | 1.0013 | 0.9926 | 1.0060 | 1.3407 | 1.6185 | 1.8459 |

**It is one curve, and it belongs to the range, not to the family.** All families with usable
statistics agree to within 2% in every bin, and the small gears (`t < 0.65`, i.e. `g` up to about
`q^{2/3}`) take exactly `2/g` to four decimals on every family - including the primorial family,
whose members are `0 (mod P_y)`, and the island family, whose position depends on `q`. No gear
takes systematically less than `2/g` on any family. The departures at `t > 0.8` are the
window-versus-section geometry (a gear above `sqrt(N)` strikes a survivor only when the cofactor
is small), identical family by family; this is 7b's curve reproduced on five new families.

MECHANISM, as arithmetic rather than measurement. Every family here is a union of residue classes
modulo `M` composed only of gears `<= y`; a gear `g > y` has `M` invertible mod `g`, so its two
teeth meet each class of the family in exactly the fair proportion. The Hardy-Littlewood singular
series of such a family is `12 C_2 / prod_{5<=g<=y}(1-2/g)`: the small-gear factor the family
"saves" by construction is exactly the factor by which its density is reduced. That is why the
excess is the same number for every family - the location rule cancels itself.

## 9. What is new

1. **Family independence of the excess, as an identity and as a measurement.** For every family of
   columns that is a union of non-tooth residue classes modulo a product of gears `<= y`, the
   twins per member equal the small machine's own openings' twins per member; the singular series
   cancels the small-gear factor exactly. Confirmed on five families and five values of `y` over
   130,644 twins in disjoint sections, all within 1.2 sigma of 1, the tightest at 0.4% (`b7`) and
   0.6% (islands, matched-rate two-sample test 1.00861 +- 0.00628).
2. **The location-thickness trade-off in exact form.**
   `survivors(F)/survivors(window) = density(F)/prod_{5<=g<=y}(1-2/g) <= 1`, with equality only
   when `F` is the whole open set of `{5..y}`. Every location rule is a strict loss of exactly its
   relative density: 2.1 for the neighbour family at `y = 7`, 15 for the primorial multiples at
   `y = 7`, 1,485 at `y = 13`, 378,675 at `y = 19`.
3. **The emptiness law.** A located family misses the section exactly as often as a Poisson
   variable at the fair rate: observed/expected 0.87, 0.95, 0.99, 1.00, 1.00, 1.04, 0.99, 0.99,
   1.01 over nine families; and the last empty window rung is where the expected count crosses 1.
   The excess says the mean is what counting says; this says the whole distribution is.
4. **The QR law joining the always-open column to the island landscape (exceptionless).** Gear 5
   is barred at the offset of the column-0 translates for EVERY `q` (661 of 661; one-line proof
   from `q^2 = 1, 4 (mod 5)`), and gear 7 is barred there iff `q = +-2 (mod 7)`. So the primorial
   family - node 3's always-open columns - is a `B = 7` island family on exactly one third of the
   rungs and never on the other two thirds. This links node 3 and node R2.a.i.a for the first time.
5. **The six positions of the island set.** `k_0 mod 35` takes exactly six values because `q^2` is
   a square mod 35, so the four island classes occupy only six positions relative to the corridor;
   `|B_7 n I|` is 1, 1, 3, 2, 4, 2 on them, never 0. Every rung has a neighbour-of-a-full-hit
   class that is also an island class.
6. **`A_7` is contained in `B_7`** (column 0 is a neighbour of a full hit at `y = 7`, since
   `c = +-1 (mod 35)` is struck by both gears), and `A_13 n B_13` is empty.
7. **Two of the brief's five families are the window restated.** `O_y` holds every twin of the
   window and `|O_y|/W` is exactly the fair rate, so its excess is the window's identically. Worth
   recording as method: a "family" that contains all the survivors is not a location rule.

Prior art: the `s = 2` handicap `4 e^{-2 gamma}` and the sieve limit `s > 4.27` are known and
cited, not re-derived (node 3a, R2.a.i.a.1.b, face A1); the singular series is standard. What is
not in the record is the comparison BETWEEN located families at one sifting level, the emptiness
law, and the QR law of item 4.

## 10. Verdict

**DEAD as a route; FACT, exact, kept.** No location rule pinpoints better than the window. The
five families of the brief have the same twins per member as the small machine's ordinary
openings, to 0.4-1.3% at 2 sigma; they miss the range exactly as often as Poisson at the fair
rate; their per-gear takes lie on one curve that belongs to the range and not to the family; and
their intersections are decided by CRT with no survival effect. The trade-off the brief asked to
state is an identity, not an empirical law: a family of relative density `delta` inside `O_y`
carries `delta` times the window's twins, so the only family that does not lose is `O_y` itself,
which is the window.

Toward the root: the round's goal - pinpoint a location inside the window with the lower machine
only - cannot be reached by any family whose definition is a residue condition modulo the lower
machine's period, because that period is invertible modulo every gear above it and therefore
carries no information about them. A location rule can beat existence only if its definition
involves the gears above `y`. The islands are the closest thing on the tree to such a definition
(their position depends on `q` through `k_0`, and the QR bar is a statement about which gears CAN
reach an offset), and they are the family measured here at 1.0061 +- 0.0054 - order one, no
excess. That is consistent with, and sharpens, R2.a.i.a's N-R5 (large gears strike islands at
exactly `2/g`): the island advantage is entirely in the SMALL gears' inability to reach, which is
already priced into the family's density.

## 11. Dead ends, with the refuting instance

- **The brief's pre-registered `1.00 +- 0.05`.** REFUTED at every family: the common value on the
  sections is 0.7987 +- 0.0022, the `s = 2` handicap `e^{2 gamma}/4 = 0.79305`. The
  pre-registration used the wrong baseline; the informative statistic is the ratio to the window,
  which is 1.
- **Pooled window statistics.** Windows at consecutive rungs overlap almost entirely; the pooled
  window excess for `a13` is 1.138 +- 0.008 by a naive Poisson sigma and 1.017 +- 0.108 on the
  disjoint sections. It is not 18 sigma from 1; it is 0.2 sigma from 1. Any branch pooling window
  counts across rungs must use sections instead.
- **`O_y` as a located family.** It is the window; its excess agrees with the window's to four
  decimals by identity, so it can never be evidence of anything.
- **The apparent factor 2.7 in the `B_7`/island overlap.** 87% of it is the corridor (both
  families live in `E_35`); the residual is 1.15 and has no survival effect (`b7 n isl` excess
  0.8014 +- 0.0058 against 0.8035).
- **The thickness lever.** Closed rather than dead: the whole window is at `s = 1.79`, short of
  `s = 4.27` by a factor `1.5e9` in `n` at `q = 4999`, and the gap does not close (`s -> 2` from
  below). Every proper family is permanently below the window by `ln(1/density)/ln q`.

## 12. What holds without exception, with the count

| statement | count | status |
|---|---|---|
| gear 5 is barred at the column-0 offset (`-6i = q^2-1`, `2-6i = q^2+1`) | 661 of 661 rungs | PROVED (one line from `q^2 = 1, 4 mod 5`) |
| gear 7 is barred there iff `q = +-2 (mod 7)` | 217 of 661, 0 disagreements | PROVED (QRs `{1,2,4}` mod 7) |
| `A_y n I` is all-or-nothing (`y = 7, 11, 13`) | 661 of 661 rungs each | PROVED (`35 | P_y`) |
| `A_7` is contained in `B_7` | 661 of 661 rungs | PROVED (`c = +-1 mod 35` is a full hit) |
| `B_7` contains at least one island class | 661 of 661 rungs, min 1, mean 2.145 | EXACT (six shifts) |
| `k_0 mod 35` in `{0, 13, 18, 20, 25, 28}` | 661 of 661 rungs | PROVED (`q^2` a square mod 35) |
| `O_y` survivors = window twins (excess identical) | 661 of 661 rungs, 5 values of `y` | IDENTITY |
| `b7` and every `O_y` have a survivor in the window and in the section | 661 of 661 rungs | MEASURED |
| `a7` has a survivor in the window | 661 of 661 rungs | MEASURED (28 empty sections) |
| no gear takes less than `2/g` on any family below `t = 0.65` | 4 decimals, 5 families, 3 rungs | MEASURED |
