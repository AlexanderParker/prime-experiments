# Branch R2.a.i.a.1.b - SQUARES ARE EVEN

Parent: node R2.a.i.a.1.a (the cover number `K(d)`, `research/proof/cover_number.md`). The
observation that spawned this branch: that document's obstruction, as the manager restated it -
a cover of the islands of `[1, d)` is realised by exactly `2^K` classes of `q` modulo a product
that exceeds `q^2`, but there are about `2.7^m` covers, so the union bound is vacuous by `10^24`;
the honest open question is **why the real phase vector `(q^2 mod g)` never realises one of those
covers**, and nothing on the tree distinguishes the real vector from a random one (P7: the phase
vector is a square in every coordinate, a set of density `2^{-pi(q)}`, and the walk length does
not notice it - percentile 0.5270).

The owner's suggestion, verbatim: *"squares are even might be the culprit?"* This branch takes it
literally and separates its three readings with one experiment.

Scripts: `research/anchor235/r42/sv_*.py`. Result outputs (untracked):
`research/anchor235/r42/results/sv_*.txt`. Every number this document relies on is written into
the document.

---

## 0. Pre-registered (written before any computation of this branch)

### 0.1 The three readings of "squares are even", stated exactly

**(a) Index parity - what evenness forbids, and what it does not.** At a gear `g` the group
`(Z/g)^*` is cyclic of even order `g - 1`, so "`q^2` is a square" is exactly the statement that
the phase `r_g = q^2 mod g` has **even discrete-log index** - one bit per gear, `2^{-pi(q)}` of
the phase space jointly. Gear `g` strikes offset `i` iff `r_g = -6i` or `r_g = 2 - 6i (mod g)`,
so evenness forbids one thing and one thing only: **a gear can never touch an offset at which
both `-6i` and `2 - 6i` have odd index** (are non-residues). That is the bar (P6), and the islands
are precisely the offsets barred at gears 5 and 7 at once, so the whole island frame is a
consequence of index parity - the target set exists *because* squares are even.

What evenness does **not** forbid, and this is the point on which the branch turns. It does not
reduce the number of strikes. Gear `g` strikes offset `i` for exactly `2 chi_g(i)` of the `g - 1`
nonzero phases, where `chi_g(i) in {0, 1, 2}` counts how many of `-6i`, `2 - 6i` are nonzero
squares; under an unrestricted phase the count is 2 out of `g - 1` at every offset. Since
`chi_g` averages to exactly 1 over offsets - and, by N-R5, **exactly 1 over the islands as well
for every gear above 7** - the mean strike rate is `2/(g-1)` under both laws. Evenness therefore
**redistributes** strikes (rate 0 on a quarter of offsets, `4/(g-1)` on a quarter, `2/(g-1)` on a
half) without removing a single one; the first moment of the number of open islands is identical
for square phases and for free phases. It also leaves the *shape* of a gear's strike set alone:
two classes at the fixed separation `d_g = 2 x 6^{-1}`, the phase sliding freely over the
`(g-1)/2` squares. So index parity can act on the witness in only two ways: through the
**dispersion** it creates across islands (some islands are barred from many gears at once and are
therefore systematically hard to strike), and through the fact that the phases at different gears
are **not independent** - they are one integer's square. The first is reading (a)/(b); the second
is reading (c); the experiment separates them.

**(b) mod 24, and the reciprocity form.** `gcd(q, 6) = 1` gives `q^2 = 1 (mod 24)`, hence
`k_0 = (q^2 - 1)/6 = 0 (mod 4)`. Offset 0 needs `q^2 = 2 (mod g)`, so only gears with
`chi_g(2) = 1`, i.e. `g = +-1 (mod 8)`, can reach it (P6, Gauss's second supplement - named once
as the classical input). For a general offset `i` the two conditions are the Legendre symbols
`(-6i / g)` and `((2 - 6i) / g)`, and by quadratic reciprocity each is a character in `g` to a
modulus dividing `4 |6i|` and `4 |2 - 6i|`: **which gears can reach island `i` is a condition on
`g` modulo `lcm(24 i, 8 |3i - 1|) = 24 i |3i - 1| / gcd(...)`, and after cancelling squares it is
a condition modulo `8 rad(3 i (3i - 1))`.** Section 2.1 writes out the exact modulus and the exact
class list for `i = 12, 47, 82` (the first three islands of the class `i = 12 mod 35`). This is
quadratic reciprocity applied to a fixed integer; it is named as classical and stopped after the
three islands are written out. Its role here is only to make reading (a) concrete: the bar at an
island is a fixed union of arithmetic progressions of gears, *the same for every `q`*.

**(c) The global integer.** The vector `(q^2 mod g)_{g}` is not merely a square at each gear: it
is one integer's square at all gears at once. By N-C6/N-C7 a cover `C` with phases pins `q^2`
modulo `P = prod_{g in C} g`, and `P > q^2` from `d = 70` on, so the residue vector determines
`q^2` **as an integer**. A locally-square vector carries no such constraint: its coordinates are
`pi(q)` independent square choices, and the CRT lift of any sub-vector is an arbitrary integer
below the modulus, almost never a perfect square.

### 0.2 The experiment (pre-registered construction)

Fix a prime `q_0`. That fixes the arc `d = 2 x 6^{-1} (mod q_0)`, the gear set
`G = {primes g : 7 < g <= q_0}` and the island set `I = {i : 1 <= i < d, i mod 35 in {5,10,12,17}}`,
`m = |I|`. **Gears 5 and 7 keep their real square phases throughout and therefore strike no island
by definition of the island set** - randomising them would change the target set rather than the
phase vector, so the anchor slot is held fixed and this is stated rather than sampled. A phase
vector is `r = (r_g)_{g in G}`; gear `g` strikes offset `i` iff `i = (2 - r_g) u_g` or `-r_g u_g
(mod g)`. Four kinds, all on the same `d`, the same `G`, the same `I`:

| kind | construction |
|---|---|
| **REAL** | `r_g = q^2 mod g` for an integer `q` coprime to 30 with `gcd(q, g) = 1` for all `g in G` (primes `q > q_0`, so one integer's square at every gear; `q^2` is astronomically below `prod G`) |
| **LS** (locally square) | `r_g` independent uniform on the `(g-1)/2` nonzero squares mod `g` |
| **RANDOM** | `r_g` independent uniform on the `g - 1` nonzero residues mod `g` |
| **LS-INERT** | LS at every gear below `q_0`, and `r_{q_0}` set to the real machine's value `0` (so the top gear strikes `i = 0` and `i = d` only and is inert on `[1, d)`, as in the real walk) |

`>= 20,000` vectors of each kind at each of six `q_0` per band (three short arcs `q_0 = 5 mod 6`,
three long arcs `q_0 = 1 mod 6`), bands `q_0 ~ 200, 500, 1000, 2000, 5000`. Measured per vector:
the number of **open islands** (islands struck by no gear of `G`), the **failure** indicator
(0 open islands), the **walk length** `L` (first offset in `[1, d)` struck by no gear), and on a
subsample the **minimum blocking set** of the struck islands (exact ILP, HiGHS).

### 0.3 Predictions, with numbers, and what refutes each

**The deciding outcome (pre-registered three ways, as the brief requires).**

* **Outcome A - `REAL = LS < RANDOM`: evenness is the mechanism.** The bar's redistribution of
  strikes creates islands that many gears cannot touch; a failure needs *every* island struck, so
  the over-dispersed island-freeness of a square vector makes failure rarer at equal mean.
* **Outcome B - `REAL < LS = RANDOM`: the global integer is the mechanism.** Being one integer's
  square, not merely a square at each gear, is what forbids covers.
* **Outcome C - all three equal: the square structure is irrelevant**, and the object's truth is a
  property of the numbers `q^2 + 6i` alone - covers are rare among *all* phase vectors. Then the
  quantity to report is the random-vector failure rate at each `d`, and whether it is already zero
  at the `d` where real failures stop (`q = 2849`, `d = 950`).

**My prediction, before computing: a fourth, mixed outcome, `REAL = LS < RANDOM` in failure rate
but with all three identical in the first moment.** Written as testable statements:

* **S1 (the first moment is blind).** The **mean** number of open islands agrees across REAL, LS
  and RANDOM to within **3%** at every `q_0`, because `chi_g` averages to exactly 1 over the
  islands for every `g > 7` (N-R5) so `E[#open]` is the same functional of the gear set under all
  three laws. REFUTED by a 5% discrepancy at any `q_0`.
* **S2 (the tail is not blind: LS fails less than RANDOM).** The failure rate satisfies
  `fail(LS) < fail(RANDOM)` at every `q_0` where RANDOM fails at least 20 times, with a ratio
  `fail(RANDOM)/fail(LS)` between **1.5 and 10**. Mechanism, stated before computing: with
  `sum_i p(i)` fixed, `P(no island open) ≈ prod_i (1 - p(i))` and `log(1 - p)` is concave, so
  spreading the per-island freeness `p(i)` - which is exactly what the bar does, the depth
  function ranging 2.05 to 5.82 against a constant 3.18 - strictly lowers the failure probability.
  REFUTED by `fail(LS) >= fail(RANDOM)` at any band, or by a ratio outside `[1.5, 10]`.
* **S3 (the global integer does NOT matter for the witness).** `fail(REAL) = fail(LS)` to within a
  factor **2** at every band, and the two open-island distributions agree in mean, minimum and
  shape. Mechanism: the failure event is decided by `K(d) ~ 6-14` gears whose product is `10^8` to
  `10^{20}`, and over a range of `q` far shorter than that product the vector `(q^2 mod g_j)` is
  equidistributed over square vectors for any *fixed* small set of gears; the global constraint
  binds only through the CRT lift, which no *statistic of the vector* can see. REFUTED by a factor
  above 2 in either direction - and if REFUTED downward (`REAL < LS`) that is Outcome B and is the
  branch's result.
* **S4 (which gears carry the difference: the small ones, overwhelmingly).** The dispersion the
  bar creates at gear `g` has variance `Var_i(2 chi_g(i)/(g-1)) = 2/(g-1)^2`, so gears
  11, 13, 17, 19, 23, 29, 31 carry `0.0568` of a total `0.0582` - **97.6%** of it - and all gears
  above 100 carry `0.0014`, i.e. **2.4%**. Prediction: re-drawing gear 11's coordinate alone from
  the square law, starting from RANDOM, moves the failure rate further than re-drawing **all**
  gears above 100 together; and the hybrid "square below `G*`, random above" reaches within 10% of
  the full LS failure rate already at `G* = 31`. REFUTED if gears above 100 move the rate more
  than gear 11, or if `G* = 31` leaves more than a quarter of the gap.
* **S5 (the top gear is irrelevant).** LS-INERT and LS differ in failure rate by less than **15%**
  relative, because a random top-gear phase strikes an expected `2 (d/q_0)(4/35) ≈ 0.08` islands.
  REFUTED by a difference above 25%.
* **S6 (`L` is blind to all of it - P7's null re-tested).** The mean and median of `L` agree across
  the four kinds to within **5%** and **1 offset** at every `q_0`. REFUTED by a 10% difference in
  the mean. (P7 measured this for real vectors against random blocked columns and got percentile
  0.5270; the prediction is that the controlled version agrees.)
* **S7 (the blocking set of the struck islands is blind too).** The mean minimum blocking set of
  the struck islands agrees across REAL, LS and RANDOM to within 1 gear at every `q_0` tested.
  REFUTED by a gap above 2 gears.
* **S8 (the proxy is faithful).** The REAL-vector failure rate at `(d, G)` taken from `q_0` equals
  the real object's own failure rate in that band (integers coprime to 30 with their own machine
  and their own arc) to within a factor 2, at the bands where the latter is nonzero. REFUTED
  otherwise, and then the fixed-`(d, G)` design is reported as a proxy only.
* **S9 (where random failure stops).** The RANDOM failure rate is a strictly decreasing function
  of the band and is **below `10^-4` by `q_0 ~ 2000` and unmeasurable (0 in 20,000) at
  `q_0 ~ 5000`**, so the arc at which the real object stops failing (`q = 2849`, `d = 950`) is
  already an arc at which a *random* vector almost never fails. Prediction: at `d = 950` the
  RANDOM failure rate is between `10^-5` and `10^-3`. REFUTED outside that interval.

**The sharp form of the owner's idea (item: the global-square test).**

* **S10 (a cover's CRT lift is not a square, and each outside gear costs a factor 2).** Take the
  failing LS vectors at a `q_0` where the real object never fails. From each, extract a minimum
  cover `C` (exact ILP) and its phases; let `R` be the CRT lift in `[0, P)`, `P = prod_{C} g`.
  Predicted: (i) `P > q_0^2` at every such cover (N-C7 reproduced); (ii) **`R` is a perfect square
  in 0 of them**; (iii) the fraction of covers whose `R` is additionally a quadratic residue
  modulo the next `t` gears outside `C` is `2^{-t}` to within a factor 2 for `t <= 12`, so
  essentially none survive `t = 20`. REFUTED by any `R` a perfect square, or by a survival
  fraction differing from `2^{-t}` by more than a factor 4 at `t = 8`.
* **S11 (and why that is not yet a bound - pre-registered as the honest limit).** The QR screen is
  *implied* by "`R` is a perfect square", so it cannot multiply the `10^{-1/2 log P}` the square
  condition already gives; the two coincide when `2^t ≈ sqrt(P)`, i.e. at `t ≈ (log_2 P)/2 ≈ 61`
  gears at `d = 1120`. Predicted: the screen's independence breaks down (measured survival exceeds
  `2^{-t}`) once `t` passes about half of `log_2 P`. Recorded as a limit, not a finding.

### 0.4 Scorecard

| # | prediction | verdict and evidence |
|---|---|---|
| S1 | mean open islands equal across kinds to 3% | **CONFIRMED at the stated tolerance**: max relative spread over 30 machines x 4 kinds is **3.12%**, under 1% at most; minima identical (3.1) |
| S2 | `fail(LS) < fail(RANDOM)`, ratio 1.5-10 | **REFUTED on the magnitude, confirmed only in the pooled sign**: `LS/RND = 0.9886` pooled (`z = -3.51`), i.e. a 1.1% effect against a predicted 50-900%; and the sign is not universal (`-2.9%` at band 200, `+7.0%` at band 500) (3.1, 5) |
| S3 | `fail(REAL) = fail(LS)` to a factor 2 | **CONFIRMED and far exceeded**: ratio `0.9984 +- 0.0033` over 6,300,000 vectors of each kind (`z = -0.49`); also equal in the open-island distribution, the per-offset profile (1%), `L` (4 figures) and the minimum blocking set (0.4 gears) (3.1, 6.3) |
| S4 | gear 11 alone beats all gears above 100; `G* = 31` closes 90% | **SPLIT**: first clause holds at `q_0 = 491` (7.8% vs 1.6%), fails at `q_0 = 463, 1571` (6.0%, 10.5% vs 18.7%, 13.6%); second clause **REFUTED** - the hybrid ladder overshoots the gap fourfold at `G* = 13` and returns (5) |
| S5 | LS-INERT vs LS within 15% | **CONFIRMED**: `LSI/LS = 0.9778` pooled, a 2.2% difference (3.1) |
| S6 | `L` equal across kinds to 5% | **CONFIRMED for REAL vs LS** (mean 14.562/14.561, 17.108/17.133, 18.708/18.781; medians equal); the RND arm is not comparable - gears 5 and 7 were freed in the walk measurement only, and gear 5 owns 2/5 of the path (6.3) |
| S7 | blocking set equal across kinds to 1 gear | **CONFIRMED**: means 10.33/10.19/10.35, 24.50/24.79/24.88, 22.54/22.30/22.36 - max gap 0.4 gears (6.3) |
| S8 | the fixed-`(d, G)` proxy matches the real object's band rate to a factor 2 | **SPLIT, and the failure is the finding**: object rates 0.1622, 0.0283, 0.0076 against proxy 0.0825, 0.0189, 0.0023 - factors 1.96, 1.50, **3.3**. The proxy is low by exactly the `4 e^{-2 gamma}` opening handicap of section 4, which does not exist at the proxy's sifting level (4.2, 4.3) |
| S9 | RANDOM failure rate at `d = 950` in `[10^-5, 10^-3]` | **CONFIRMED**: `1.467e-5` at `d = 954` (44 in 3,000,000); LS `1.267e-5`, REAL `1.056e-5` (3.2) |
| S10 | no cover lift is a square; QR screen decays as `2^{-t}` | **CONFIRMED**: 0 of 82 failing locally-square vectors at `d = 954` has a perfect-square lift; survivors 82, 40, 17, 9, 7, 3, 2, 2, 0 at `t = 0..8` against `2^{-t}`; control - 21 of 21 real failures with `P > q^2` have `R = q^2` exactly (6.1, 6.2) |
| S11 | the screen saturates at the square condition near `t = (log_2 P)/2` | **AS PREDICTED, untestable at this sample size**: `sqrt(P) = 10^{45.5}` gives saturation at `t ~ 151`; 82 vectors resolve only `t <= 8`. Recorded as an analytic limit, and the square condition is in any case weaker than the range condition already used (6.2) |

---

## 1. Setup (exact ranges)

Scripts in `research/anchor235/r42/`; outputs (untracked) in `research/anchor235/r42/results/`.
No sampling except where a row says so; every simulated rate carries its draw count.

| object | range | script |
|---|---|---|
| the reachability class list of an island, exact, and the census over classes mod `M(i)` | `i = 12, 47, 82`; checked against every prime gear `<= 200,000` | `sv_bar.py` |
| the four-kind comparison: failure rate, open islands, `L` | 5 bands x 6 primes `q_0` (3 short arcs, 3 long), 300,000 vectors per kind per `q_0` at `q_0 <= 1013`, 100,000 at 2,000, 50,000 at 5,000 - **6,300,000 vectors of each kind in all** | `sv_main.py` |
| the mechanism: hybrids `H(G*)` and one-gear swaps, paired (common random numbers) | `q_0 = 191` (1,000,000 vectors), `491, 463, 1571` (600,000 each), 37 configurations per `q_0` | `sv_mech.py` |
| the real object against the model, per prime, at its own machine and arc | every prime `q` in `[11, 6000]` (779 machines), 4,000 LS and 4,000 RND vectors each | `sv_object.py` |
| the exact first moment and the failure count after the classical correction | every prime `q` in `[11, 6000]`, exact rates, no sampling | `sv_pred.py` |
| real open islands against the exact model, at the object's own configuration | 20 primes per band, 7 bands, `q = 200 .. 80,000` | `sv_hl.py` |
| the same as a function of the sifting ratio `s = log(q^2)/log(z)`, gear set and window fixed | `z = 5009`, window `[1, 1670)`, 192 islands, 667 gears; `q` prime in four decades to `1.5 x 10^7` | `sv_s.py` |
| the failure rate of each kind against the arc, adaptive to 3,000,000 draws | 9 arcs `d = 60 .. 1100` | `sv_rate.py` |
| the global-square test: minimum covers of failing vectors, their CRT lift | control: every real failure `q` coprime to 30 with `q <= 3000`; test: locally-square vectors at `q_0 = 2861` (`d = 954`, past the last real failure) | `sv_cover.py` |
| the minimum blocking set of the struck islands by kind | `q_0 = 491, 991, 1571`, 200 vectors per kind (exact ILP, HiGHS) | `sv_mbs.py` |

**The vector constructions, exactly as run.** Gears 5 and 7 hold square phases in every kind; they
can strike no island at all (that is what an island is), so the island statistics do not see them
and only `L` does. Gears above 7 carry the phase vector. REAL takes `r_g = q^2 mod g` for a prime
`q` (`q > 6000` in `sv_main`, `q > 12000` in `sv_rate`, so `gcd(q, g) = 1` at every gear and the
phase is a nonzero square everywhere); REALNEAR the same for primes `q_0 < q <= 20 q_0`. LS draws
`s` uniform in `[1, g)` and sets `r = s^2 mod g` - uniform on the `(g-1)/2` nonzero squares, since
each has exactly two preimages. RND draws `r` uniform in `[1, g)`. LSI is LS with the top gear
`q_0` inert (`r = 0`, so it strikes `i = 0` and `i = d` only), which is what the real machine does
at its own `q`. Writing `b = -r u_g` and `a = b + d_g`, the struck offsets are `i = a` and `i = b
(mod g)`; `r` nonzero square is the same condition as `b` in the reachable half, so LS is exactly
"the pair of classes sits where a real square can put it" and RND is "anywhere".

## 2. Results - reading (b): the bar as a condition on the gear

### 2.1 Which gears can reach an island, exactly

For a fixed offset `i` the two conditions are `chi_g(-6i) = 1` and `chi_g(2 - 6i) = 1`. Replacing
each argument by its squarefree kernel `s` makes each symbol a character in `g` of conductor
`|s|` (when `s = 1 mod 4`) or `4|s|`, so reachability of island `i` is decided by `g` modulo the
lcm of the two conductors. Computed and then checked against every prime gear up to 200,000:

| island `i` | `-6i` | `2 - 6i` | conductors | **`M(i)`** | barred classes | fraction | inconsistencies |
|---|---|---|---|---|---|---|---|
| 12 | `-2 x 6^2` | `-70` | 8, 280 | **280** | 24 of 96 | **1/4** exactly | 0 |
| 47 | `-282` | `-70 x 2^2` | 1128, 280 | **39,480** | 2,208 of 8,832 | **1/4** exactly | 0 |
| 82 | `-123 x 2^2` | `-10 x 7^2` | 123, 40 | **4,920** | 320 of 1,280 | **1/4** exactly | 0 |

For `i = 12` the condition collapses to two symbols: `(-72/g) = (-2/g)` and `(-70/g) =
(-2/g)(35/g)`, so **gear `g` is barred from island 12 iff `g = 5` or `7 (mod 8)` and `(35/g) = 1`**
- twenty-four classes mod 280, the first being 13, 23, 29, 31, 109, 111, 117, 127. This is
quadratic reciprocity applied to a fixed integer (named once as the classical input) and it is
recorded only because it makes reading (a) concrete: **the bar at an island is a fixed union of
arithmetic progressions of gears, the same at every `q`, of density exactly 1/4.** Nothing about
`q` enters. The offset-0 case is the classical `g = +-1 (mod 8)` and is already on record (P6).

That exhausts reading (b): mod 24 fixes `k_0 = 0 (mod 4)` and the reciprocity conditions fix the
island set. Both are `q`-free, so neither can distinguish one `q` from another - which is the
whole point of what follows.

## 3. Results - the deciding experiment

### 3.1 The four kinds, by band (6,300,000 vectors of each kind)

Failure = no open island in `[1, d)`. Each band pools six machines (three short arcs, three long).

| band | draws per kind | **REAL** | **LS** | **RND** | **LSI** | REAL/LS (z) | LS/RND (z) |
|---|---|---|---|---|---|---|---|
| 200 | 1,800,000 | 0.08252 | 0.08268 | 0.08518 | 0.08068 | 0.9980 (-0.54) | 0.9707 (**-8.16**) |
| 500 | 1,800,000 | 0.01893 | 0.01890 | 0.01766 | 0.01863 | 1.0017 (+0.22) | 1.0703 (**+8.71**) |
| 1,000 | 1,800,000 | 0.00231 | 0.00235 | 0.00229 | 0.00231 | 0.9806 (-0.90) | 1.0269 (+1.21) |
| 2,000 | 600,000 | 0.000087 | 0.000058 | 0.000073 | 0.000073 | 1.49 (+1.82) | 0.80 (-1.01) |
| 5,000 | 300,000 | 0 | 0 | 0 | 0 | - | - |
| **pooled** | **6,300,000** | **0.029653** | **0.029700** | **0.030042** | **0.029040** | **0.9984 (-0.49)** | **0.9886 (-3.51)** |

> **N-S1 (the real phase vector is a locally-square vector, to 0.3%).** Over 6.3 million vectors of
> each kind on 30 machines, the failure rate of REAL and of LS agree at
> `0.029653` against `0.029700`, a ratio of `0.9984 +- 0.0033` (`z = -0.49`). At no band does the
> ratio depart from 1 by more than one standard error where the counts allow a test. **Being one
> integer's square rather than an independent square at each gear changes the failure rate of the
> island witness by less than 0.7% (95% bound).** REALNEAR (primes `q_0 < q <= 20 q_0`) agrees
> too, on its smaller samples.

> **N-S2 (index parity is worth about 1%, with a sign that is a property of the arc).** LS against
> RND is `0.029700` against `0.030042` pooled - LS fails **1.14% less often** (`z = -3.51`) - but
> the sign is not universal: at band 200 LS fails 2.9% *less* (`z = -8.16`) and at band 500 it
> fails 7.0% *more* (`z = +8.71`). The square constraint moves the failure rate by a few percent
> in a direction decided by how the bar's fixed class list falls on that particular arc's islands,
> and the effects very nearly cancel in aggregate.

**S1 (the first moment is blind) - confirmed.** The mean number of open islands over the four
kinds differs by at most **3.12%** at any of the 30 machines and by under 1% at most of them
(e.g. at `q_0 = 5011`: REAL 30.730, LS 30.754, RND 30.567, LSI 30.726 on `m = 383` islands). The
minimum over vectors is identical: 0 wherever failures occur, 3 at `d = 1333..1345`, 11-13 at
`d = 3329..3341`, in every kind alike.

**S5 (the top gear is irrelevant) - confirmed.** LSI/LS = 0.9778 pooled, a 2.2% difference against
the pre-registered tolerance of 15%. A random phase at the top gear places two classes mod `q_0`
of which about `d/q_0` land in the arc and `4/35` of those on an island: an expected 0.08 extra
island strikes, which is what 2.2% of the failure rate is.

### 3.2 The rate against the arc, to `10^-5` (S9)

Nine short-arc machines, adaptive sampling to 3,000,000 draws (REAL capped by the prime pool).
`INDEP` is the parent's C10 model, `prod_i (1 - p(i))` with the exact rates.

| `d` | `q_0` | `m` | gears | **LS** | **RND** | **REAL** | INDEP |
|---|---|---|---|---|---|---|---|
| 60 | 179 | 8 | 36 | 1.323e-1 | 1.359e-1 | 1.314e-1 | 1.475e-1 |
| 130 | 389 | 16 | 72 | 5.398e-2 | 5.244e-2 | 5.371e-2 | 6.158e-2 |
| 200 | 599 | 24 | 104 | 2.418e-2 | 2.054e-2 | 2.436e-2 | 3.061e-2 |
| 338 | 1,013 | 40 | 165 | 3.965e-3 | 3.695e-3 | 3.945e-3 | 6.003e-3 |
| 500 | 1,499 | 57 | 234 | 7.500e-4 | 7.900e-4 | 7.800e-4 | 1.289e-3 |
| 676 | 2,027 | 78 | 302 | 1.154e-4 | 1.300e-4 | 1.116e-4 | 2.458e-4 |
| 800 | 2,399 | 92 | 352 | 3.733e-5 | 4.300e-5 | 4.072e-5 | 8.453e-5 |
| **954** | 2,861 | 109 | 411 | **1.267e-5** | **1.467e-5** | **1.056e-5** | 2.631e-5 |
| 1,100 | 3,299 | 127 | 458 | 3.33e-7 (1/3,000,000) | 2.67e-6 (8/3,000,000) | 1.51e-6 (1/663,141) | 6.561e-6 |

> **S9 confirmed, at the pre-registered arc.** The last real failure of the object is `q = 2849`,
> arc `d = 950`. At `d = 954` a **free** phase vector fails `1.467e-5` of the time (44 in
> 3,000,000), inside the pre-registered interval `[1e-5, 1e-3]`; a locally-square vector
> `1.267e-5`; a real one `1.056e-5` (7 in 663,141). At the next rung, `d = 1100`, a locally-square
> vector fails once in 3,000,000. **The arc at which the real object stops failing is already an
> arc at which a random vector almost never fails** - within a factor 1.4 of the real rate, with no
> square structure anywhere in the calculation.

REAL sits inside the LS 95% interval at every arc down to `d = 954`; the last rung carries only
1, 8 and 1 failure events and settles nothing between the kinds. The three kinds fall together by a
factor of about 3,500 from `d = 60` to `d = 800`, i.e. `log(rate)` is very nearly linear in the
island count `m` - the failure rate is `exp(-c m)` with `c = 0.097` from `d = 60` to `d = 800` and `0.108` out to `d = 1100`, which is
just `-log(1 - p)` at the mean opening chance. Two further facts the table carries:

* the **independent-island** model `INDEP` **over**-predicts the simulated rate, and increasingly:
  by 12% at `d = 60`, 26% at `d = 200`, 72% at `d = 500` and 126% at `d = 800`. Islands sharing a
  gear are positively correlated in openness (one phase strikes two classes), which makes "every
  island struck" rarer than independence says. This is a correction to the parent's C10 in the
  direction that makes its miss worse, and it is measured here for the first time;
* by `d = 800` a random phase vector already fails only 4 times in 100,000. The real object's last
  failure is at `q = 2849`, `d = 950`.

## 4. Results - reading (c): where the global integer does show itself

Sections 3.1 and 3.2 hold the arc and the gear set fixed at a `q_0` and take the phases from some
*other* integer `q`. That is the honest test of "one integer's square versus independent squares",
and it says the two are the same. But the object's own configuration is not that: at the real
machine the gear set is `{5..q}` and the columns being sifted are `q^2 + 6i`, so **the machine
sifts exactly to the square root of the numbers it is sifting**. That is the one place the global
integer is visible, and it is visible by a large factor.

### 4.1 The real object has a fifth fewer openings than any model of its phase vector

For a prime `q` at its own machine, an island `i` is open iff `q^2 + 6i - 2` and `q^2 + 6i` have no
prime factor at all in `(7, q]` - i.e. iff the pair is a twin prime pair. The model of the phase
vector predicts `E[#open] = sum_i prod_{7 < g < q} (1 - 2 chi_g(i)/(g-1))`, and that expectation is
**exact** if `q` were uniform modulo `prod g` - which it cannot be, since that product is `e^{q}`.
Measured (20 primes per band, real count and exact model, no sampling in the model):

| band of `q` | real open islands | model | Hardy-Littlewood | **model / real** | HL / real |
|---|---|---|---|---|---|
| 200 - 400 | 2.50 | 3.15 | 2.61 | 1.2595 | 1.0454 |
| 400 - 900 | 4.10 | 4.82 | 3.94 | 1.1768 | 0.9619 |
| 900 - 2,000 | 7.35 | 9.26 | 7.38 | 1.2604 | 1.0038 |
| 2,000 - 5,000 | 14.25 | 17.52 | 13.91 | 1.2292 | 0.9763 |
| 5,000 - 12,000 | 25.85 | 33.63 | 26.67 | 1.3008 | 1.0318 |
| 12,000 - 30,000 | 59.70 | 72.82 | 57.80 | 1.2198 | 0.9682 |
| 30,000 - 80,000 | 131.20 | 165.69 | 131.48 | **1.2628** | **1.0021** |

> **N-S3 (the object's opening count is Hardy-Littlewood, and the phase-vector model overstates it
> by exactly `4 e^{-2 gamma}`).** The real number of open islands is the Hardy-Littlewood twin
> count for the island columns, `m x 12 C_2 / (ln q^2)^2 / ((3/5)(5/7))`, to within 3% at every
> band and to 0.2% at the top one. The model of the phase vector exceeds it by a ratio that sits at
> **1.2628 at `q ~ 50,000`** against the constant

```
    prod_{11 <= p <= q} (1 - 2/p)       28 C_2 e^{-2 gamma} / ln^2 q
    -----------------------------  =   ----------------------------  =  4 e^{-2 gamma} = 1.26190
    P(island column is a twin)              7 C_2 / ln^2 q
```

> Prior art, named once and stopped: this is the classical fact that the Mertens product
> over-counts the sifted set when the sifting level reaches the square root - `2 e^{-gamma} =
> 1.1229` per sieve dimension, squared here because the twin problem has dimension 2. The tree
> already carries the same `s = 2` obstruction from a different direction (branch 3a, the
> dimension-2 sifting limit).

### 4.2 It is the sifting level, not the squares - the deciding control

Hold the gear set and the window fixed (`z = 5009`, offsets `[1, 1670)`, 192 islands, 667 gears,
exact model `E[#open] = 15.4331`) and vary only how big the integer being sifted is, i.e. the
sifting ratio `s = log(q^2)/log(z)`:

| `q` range | `s` | primes | real open islands | **model / real** |
|---|---|---|---|---|
| 5,009 - 15,027 | 2.13 | 1,086 | 13.7127 | **1.1255** |
| 50,090 - 150,270 | 2.67 | 8,731 | 15.6910 | 0.9836 |
| 500,900 - 1,502,700 | 3.21 | 20,000 | 15.4281 | **1.0003** |
| 5,009,000 - 15,027,000 | 3.75 | 20,000 | 15.4154 | **1.0011** |

> **N-S4 (the global integer is invisible except at `s = 2`).** With the gear set and the island
> window fixed, the exact independent-gear model of the phase vector reproduces the real
> open-island count of an integer's square to **0.03% at `s = 3.21` and 0.11% at `s = 3.75`** -
> 20,000 primes each. It is 12.6% high already at `s = 2.13` and, from section 4.1 where `s = 2`
> exactly, 26% high. So "one integer's square" is worth nothing at all as a constraint on the
> phase vector; what is worth something is that the machine's top gear is the square root of the
> column. The deviation is not monotone (`0.9836` at `s = 2.67`), which is the Buchstab-type
> oscillation of the sifted density - classical, named, stopped.

### 4.3 What that repairs, and in which direction

The parent's first-moment model (C10) predicted 0.286 failures in `[1000, 3000)` against 4
observed and called the factor-14 miss an honest miss, attributing it to correlation between
neighbouring islands. Section 3.2 shows that attribution is **wrong in sign**: island correlation
makes `INDEP` *over*-predict failures by up to a factor 2.3, not under-predict. The whole miss is
`4 e^{-2 gamma}`. Over every prime `q <= 6000` at its own machine (778 machines, exact rates, no
sampling):

| band | primes | observed failures | INDEP | **INDEP after `p -> p / 4 e^{-2 gamma}`** | real open | model open |
|---|---|---|---|---|---|---|
| 11 - 100 | 20 | 6 | 4.042 | 5.778 | 1.300 | 1.451 |
| 100 - 300 | 37 | 6 | 3.774 | 6.084 | 1.649 | 2.303 |
| 300 - 1,000 | 106 | 3 | 1.937 | 4.094 | 3.802 | 4.998 |
| 1,000 - 3,000 | 262 | **2** | **0.143** | **0.545** | 8.859 | 11.253 |
| 3,000 - 6,000 | 353 | 0 | 0.000 | 0.005 | 16.861 | 21.075 |
| **total** | **778** | **17** | **9.90** | **16.51** | | |

> **N-S5 (the correction closes the parent's miss).** The exact first moment predicts **9.90**
> prime failures below 6,000 against **17** observed - the parent's miss, `P(X >= 2) = 0.009` for
> the `[1000, 3000)` band alone. Feeding the same model the true opening density instead of the
> Mertens product predicts **16.51 against 17**. In the band where the miss was worst the
> prediction goes from 0.143 to 0.545 against 2 observed, and `P(X >= 2)` rises from 0.9% to 10%,
> i.e. from significant to unremarkable. The residual pull the other way (islands are positively
> correlated, section 3.2, worth about a factor 0.9 on the total) leaves the corrected prediction
> near 14.8 against 17.

**And the direction matters.** The correction makes the real machine **worse off** than a random
phase vector, not better: it has a fifth fewer open islands than the model, so it fails *more*
often, not less. Whatever keeps the island witness alive above `q = 2849`, it is not that the
phase vector is a square.

## 5. Results - which gears carry the (small) difference

The comparisons below are **paired**: one uint64 bitmask per gear per vector is built twice, once
under each law, on the same random stream, so a one-gear swap changes exactly one column and
nothing else. `SWAP-IN g` is unrestricted everywhere except gear `g`, which is square; `SWAP-OUT
g` is square everywhere except gear `g`. The quoted swing is
`(SWAP-IN - SWAP-OUT)/RND`, i.e. what gear `g`'s squareness alone is worth.

| machine | `d` | `m` | draws | LS | RND | (LS-RND)/RND |
|---|---|---|---|---|---|---|
| `q_0 = 191` | 64 | 8 | 1,000,000 | 0.144036 | 0.147152 | **-2.1%** |
| `q_0 = 491` | 164 | 20 | 600,000 | 0.035438 | 0.033387 | **+6.1%** |
| `q_0 = 463` | 309 | 36 | 600,000 | 0.001365 | 0.001370 | -0.4% |
| `q_0 = 1571` | 524 | 60 | 600,000 | 0.000595 | 0.000652 | -8.7% |

Per-gear swings in the failure rate (the `q_0 = 191` and `q_0 = 491` columns are far outside
Monte-Carlo error: 147,000 and 20,000 failure events; the last two machines carry only ~370 events
each and their swings are at the 2-sigma level):

| gear | `q_0 = 191` | `q_0 = 491` | `q_0 = 463` | `q_0 = 1571` |
|---|---|---|---|---|
| 11 | **-14.3%** | +7.8% | +6.0% | +10.5% |
| 13 | **-22.8%** | **-18.6%** | -17.5% | -5.4% |
| 17 | **+23.0%** | -6.2% | +12.7% | -5.9% |
| 19 | +2.7% | **-15.3%** | +3.8% | +9.7% |
| 23 | +1.8% | +2.2% | -7.2% | +11.8% |
| 29 | +3.0% | +1.1% | +3.0% | +13.0% |
| 31 | +2.2% | -8.6% | +1.2% | +7.4% |
| 37 | +11.4% | -2.4% | +2.2% | -2.6% |
| **all `g > 100` together** | +13.3% | +1.6% | +18.7% | +13.6% |

> **N-S6 (squareness is a per-gear, per-arc accident, not a resource, and it does not add up).**
> Making one small gear square, with every other gear left free, moves the failure rate by 5-25%,
> and the **sign depends on the gear and on the arc**: at `q_0 = 191` gear 13 lowers it by 22.8%
> while gear 17 raises it by 23.0%; at `q_0 = 491` gear 11 raises it by 7.8% while gear 13 lowers
> it by 18.6%. Making *every* gear square - the real situation - lands within 2-9% of the
> free-phase rate, because the individual effects cancel. The hybrid ladder shows the same
> non-additivity directly: at `q_0 = 191` the failure rate runs 0.14715 (all free), 0.13886
> (square at 11 only), **0.12020** (square at 11 and 13), 0.13472, 0.13492, 0.13439, 0.13397,
> 0.13340, 0.13272, 0.14171, 0.13868, **0.14404** (square everywhere) - it dips 18% below the
> free-phase value at `G* = 13` and comes back to within 2% of it once all the gears are square.

The one systematic part is in the **first moment**, and there the small gears do carry it, as
pre-registered: at `q_0 = 1571` the mean open-island count runs 6.3672 (all free) up to 6.4868 at
`G* = 251` and 6.4670 with every gear square, and the whole of the rise is present by `G* = 101`.
Per gear, the open-count swing is `+-0.01` to `+-0.19` for gears 11-37 and `+-0.10` to `+-0.16`
for the whole tail `g > 100` collectively - consistent with the pre-registered dispersion budget
`Var_i(2 chi_g(i)/(g-1)) = 2/(g-1)^2`, which puts 97.6% of the total in gears 11..31 as a *sum*
but leaves the tail able to match a single small gear when the arc is short enough for the tail
gears to reach the islands at all.

**S4 is SPLIT.** Its first clause (gear 11 alone moves the rate more than all gears above 100
together) holds at `q_0 = 491` (7.8% against 1.6%) and fails at `q_0 = 463` and `q_0 = 1571`
(6.0% and 10.5% against 18.7% and 13.6%). Its second clause (`G* = 31` closes 90% of the LS-RND
gap) is **refuted outright**: the hybrids overshoot the gap by a factor of four and come back.

## 6. Results - the sharp form of the owner's idea: is a cover's lift a square?

### 6.1 The control: at a real failure the lift IS the square

For every integer `q` coprime to 30 up to 3,000 whose islands are all struck by gears above 7
(24 of them; the 7 further failures below 3,000 are multiples of 7, where gear 7 itself does the
covering and is excluded here), take the **exact** minimum cover of the islands of `[1, d)` at the
real phases (ILP, HiGHS), form `P = prod g_j` and the CRT lift `R` of the phases in `[0, P)`:

| `q` | 17 | 23 | 29 | 41 | 53 | 73 | 77 | 113 | 119 | 121 | 137 | 161 | 173 | 197 | 233 | 247 | 263 | 341 | 353 | 461 | 683 | 1151 | 1487 | 1649 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `K` | 1 | 1 | 1 | 3 | 4 | 5 | 3 | 4 | 4 | 8 | 5 | 5 | 8 | 5 | 7 | 11 | 8 | 12 | 12 | 15 | 18 | 22 | 24 | 25 |
| `log10 P/q^2` | -1.4 | -1.6 | -1.8 | 1.0 | 2.7 | 3.5 | 0.8 | 3.1 | 2.2 | 7.8 | 4.5 | 3.1 | 9.3 | 2.6 | 7.1 | 12.2 | 8.1 | 18.8 | 16.8 | 22.9 | 29.5 | 38.2 | 44.0 | 45.4 |
| `R = q^2`? | no | no | no | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |

**21 of 24, and the three exceptions are exactly the three covers with `P < q^2`** (`q = 17, 23,
29`, single-gear covers of a single island). N-C7 reproduced independently, and the `K` values
reproduce the parent's `R(q)` ladder exactly.

### 6.2 The test: failing locally-square vectors past the last real failure

`q_0 = 2861`, `d = 954`, `m = 109` islands, 411 gears - one arc past the last real failure ever
recorded (`q = 2849`, `d = 950`). **82 failing locally-square vectors in 8,000,000 draws**
(rate `1.03e-5`, agreeing with the ladder's `1.267e-5` at the same arc). For each, the exact
minimum cover and its lift:

| quantity | value |
|---|---|
| cover size `K` | mean **39.41**, min 33, max 49 |
| `log10 P / q_0^2` | mean **84.00**, min 66.39 |
| covers whose lift `R` is a **perfect square** | **0 of 82** |
| `R` a nonzero QR mod the first `t` gears outside the cover | `t = 0..8`: 82, 40, 17, 9, 7, 3, 2, 2, **0** |
| the same as a fraction, against `2^{-t}` | 1.000/1, 0.488/0.500, 0.207/0.250, 0.110/0.125, 0.085/0.063, 0.037/0.031, 0.024/0.016, 0.024/0.008, 0/0.004 |

> **N-S7 (a locally-square vector's cover is not an integer's square, and each outside gear halves
> the chance).** Of 82 explicit failing locally-square vectors at an arc where the real object has
> never failed, **not one** has a minimum cover whose CRT lift is a perfect square, and the
> fraction still consistent with being a square after screening `t` further gears follows `2^{-t}`
> to within a factor 2 out to `t = 5` and dies at `t = 8`. The failing vectors' covers use
> **33 to 49 gears** where the adversarial optimum at that arc is `K(954) ~ 18` - a ratio of
> 1.8 to 2.7, the same range the parent measured for real failures - and their moduli exceed
> `q_0^2` by `10^{66}` to `10^{100}`.

> **N-S8 (why that is still not a bound - the limit, stated exactly).** The QR screen is *implied*
> by "`R` is a perfect square", so it can never be stronger than the square condition, and the two
> must coincide once `2^t` reaches `sqrt(P) = 10^{45.5}`, i.e. at `t ~ 151` gears - far beyond
> what 82 samples can resolve. And the square condition is itself **weaker than the range
> condition already in the parent's hands**: `R` must not merely be a square, it must be `q_0^2`
> with `q_0 ~ 3d`, which costs `10^{-84}` against the square condition's `10^{-45.5}`. So the
> owner's sharp test is exactly N-C7 in another costume and adds no factor to it. The
> arithmetic is unchanged: about `2.7^m = 10^{47.0}` covers at `d = 954` against a cheapest-cover
> density `2^{18}/10^{28.19} = 10^{-22.8}`, i.e. **vacuous by `10^{24}`**, the parent's number.

### 6.3 The blocking set and the walk are blind too

Exact minimum blocking sets of the **struck** islands (ILP, 200 vectors of each kind):

| machine | `d` | `m` | REAL | LS | RND |
|---|---|---|---|---|---|
| `q_0 = 491` | 164 | 20 | 10.33 (7-15) | 10.19 (7-14) | 10.35 (7-14) |
| `q_0 = 991` | 661 | 76 | 24.50 (18-31) | 24.79 (19-32) | 24.88 (18-31) |
| `q_0 = 1571` | 524 | 60 | 22.54 (15-29) | 22.30 (17-30) | 22.36 (16-29) |

All three kinds agree to within 0.4 gears in the mean and share the same range - S7 confirmed.

The per-offset opening probability is the sharpest form of N-S1. At `q_0 = 491`, over 147,495 real
vectors and 200,000 locally-square ones:

| offset `i` | 1 | 2 | 3 | 4 | **5** | 6 | 7 | 8 | 9 | **10** | 11 | **12** | 13 | 14 | 15 | 16 | **17** | 18 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| REAL | 0 | .0412 | .0714 | .0470 | **.0997** | 0 | .0644 | .0449 | .0220 | **.2296** | 0 | **.1366** | .0579 | .0465 | .0674 | 0 | **.1981** | .0448 |
| LS | 0 | .0417 | .0703 | .0477 | **.0988** | 0 | .0644 | .0448 | .0219 | **.2311** | 0 | **.1360** | .0582 | .0468 | .0669 | 0 | **.1980** | .0444 |

Identical to within 1% at every offset, islands (bold) included; and the walk length agrees to
four figures - mean 14.562 against 14.561, median 12 against 12, `P(L <= 10)` 0.4977 against
0.4981. The same holds at `q_0 = 991` and `q_0 = 1571`. **S6 is confirmed for the comparison it
was meant for.** (The RND row of the same table is not comparable: in the walk measurement the
unrestricted law was applied to gears 5 and 7 as well, and gear 5 alone owns two of every five
path columns, so RND's `L` is shorter by 3% for a reason that has nothing to do with the islands.
The island statistics never see gears 5 and 7.)

The RND profile is nevertheless worth one line as the picture of what the bar does: it is flat
(0.038-0.087 across the same eighteen offsets) where the square profile has hard zeros at
`i = 1, 6, 11, 16` and peaks of 0.14 to 0.23 at the four islands. **The bar does not open more
columns; it decides which columns are the open ones** - the same conservation the first moment
shows.

## 7. Mechanism

**What index parity buys, and where it is spent.** At a gear the phase carries one bit - the
parity of its discrete log - and that bit does exactly one thing: it forbids the gear from ever
touching an offset at which both `-6i` and `2 - 6i` are non-residues. That is the bar, and it is a
condition on `g` alone, a union of arithmetic progressions of density exactly `1/4` modulo an
explicit `M(i)` (280 at `i = 12`, 4,920 at `i = 82`, 39,480 at `i = 47`; 0 disagreements against
every prime gear to 200,000). The islands are the offsets at which gears 5 and 7 are both barred.
**So the entire power of "squares are even" has already been spent by the time the island set is
written down.** Above gear 7 the bit buys nothing further in the first moment, because the number
of `q`-classes at which gear `g` strikes offset `i` is `2 chi_g(i)` with `chi_g` averaging exactly
1 over the islands: the bar moves strikes between offsets, it does not remove them. Measured: the
mean open-island count is the same for real, locally-square and free phases to within 3.1% at
every one of 30 machines, and the *shape* is the same too - the per-offset opening probability of
a real vector agrees with a locally-square vector's to 1% at every offset (zeros at `i = 1, 6, 11,
16`, peaks of 0.14-0.23 at the islands), while a free vector's is flat.

**What is left for parity to act on is the tail, and it acts on it by accident.** With the first
moment fixed, the only room is dispersion, and the dispersion budget of gear `g` is
`Var_i(2 chi_g(i)/(g-1)) = 2/(g-1)^2` - `0.0582` in total, `0.0568` of it in gears 11..31. That is
a 24% relative spread in the per-island opening chance, worth about 1% in the failure rate by
concavity. And that is what the paired swaps find: making one small gear square, all others free,
moves the failure rate by 5-25% **with a sign that depends on the gear and on the arc** (at
`q_0 = 191`, gear 13 down 22.8% and gear 17 up 23.0%; at `q_0 = 491`, gear 11 up 7.8% and gear 13
down 18.6%), and the effects cancel: LS against RND is `-2.1%`, `+6.1%`, `-0.4%`, `-8.7%` at the
four machines and `-1.14%` pooled over 6.3 million vectors each. Squareness is not a resource that
accumulates; the hybrid ladder overshoots the total gap by a factor of four at `G* = 13` and comes
back to it.

**Where the global integer does act, and why it is the sifting level rather than the squares.**
Fix the gear set and the offset window and feed the phases from a genuine integer's square. If the
integer is large the exact independent-gear model is right to 0.03% (`s = 3.21`, 20,000 primes)
and 0.11% (`s = 3.75`); it is 12.6% high at `s = 2.13` and 26% high at `s = 2`. The object's own
configuration is `s = 2` exactly: the machine `{5..q}` sifts the columns `q^2 + 6i`, and its top
gear is the square root of what it is sifting. There the model is not a model of a random vector
at all - the surviving islands are exactly the twin prime pairs just above `q^2`, their count is
the Hardy-Littlewood one (measured `HL/real = 1.0021` at `q ~ 50,000`), and the phase-vector model
exceeds it by

```
    prod_{11 <= p <= q} (1 - 2/p) / P(island column is twin) = 4 e^{-2 gamma} = 1.26190 ,
```

measured 1.2628. The mechanism is the classical one and is named once: the Mertens product
over-counts a sifted set when the sifting level reaches the square root, by `2 e^{-gamma}` per
sieve dimension, and the twin problem has dimension 2. **The real machine therefore has a fifth
fewer open islands than any model of its phase vector, and so fails more often, not less.** That
repairs the parent's C10: 9.90 predicted failures below `q = 6000` against 17 observed becomes
16.51 against 17 once the density is the true one.

**Why the witness still holds is therefore not a fact about the phase vector at all.** At the arc
of the last real failure (`d = 950`) a free vector already fails only 1.5 times in 100,000, a
locally-square vector 1.3 times, a real one 1.1 times; one arc further (`d = 1100`) a
locally-square vector fails once in 3,000,000. The witness holds above `q = 2849` because covers
are rare among *all* vectors at those arcs, and the object's own 26% handicap is not enough to
overcome the exponential decay in the island count. That is Outcome C of the pre-registration, with
the handicap as the one genuine global-integer effect - and it points the wrong way.

## 8. What is new

Screened line by line against `docs/novel/README.md` - in particular `reachability-landscape`,
`island-witness-integers`, `walk-path-transforms` (P6, P7), `walk-path-parts`, `j2-upper-bound`
(the dimension-2 sifting ceiling), `potential-arity-ladder` (the Mertens no-go),
`twin-percentile`, `corridor-law` - and against `research/proof/reachability.md`,
`island_witness.md`, `cover_number.md`, `walk_transforms.md`. The register carries the bar and the
square phase vector as *facts* (P6, P7) and carries `L`'s indifference to them as a *null*
(percentile 0.5270 against random blocked columns). It carries no controlled comparison of phase
vector laws, no measurement of what index parity is worth, and no correction of the first moment.

* **N-S1 (the real phase vector is a locally-square vector).** Over 6.3 million vectors of each
  kind on 30 machines the island-witness failure rate of real phase vectors and of independent
  locally-square vectors agree at 0.029653 against 0.029700 - ratio `0.9984 +- 0.0033` - and the
  per-offset opening profile, the walk length (mean 14.562 against 14.561), the open-island
  distribution and the minimum blocking set of the struck islands agree as well. The joint
  constraint of density `2^{-pi(q)}` that P7 records is worth **less than 0.7%** on the witness.
  New: P7 recorded that `L` does not notice; this measures how much *nothing* it is, on the
  witness rather than on the length, with a matched control.
* **N-S2 (index parity is worth about 1%, with a sign belonging to the arc).** Square against free
  phases: `-2.9%` at band 200, `+7.0%` at band 500, `-1.14%` pooled. Per gear, one small gear's
  squareness moves the rate by 5-25% with an arc-dependent sign, and the hybrid ladder is
  non-monotone (18% below the free-phase rate at `G* = 13`, back within 2% at `G* = q_0`). The
  dispersion budget `sum_g 2/(g-1)^2 = 0.0582` predicts the ~1% aggregate. New.
* **N-S3 / N-S4 (the object's opening count is Hardy-Littlewood, and the phase-vector model
  overstates it by `4 e^{-2 gamma}` - only at `s = 2`).** With the gear set fixed and the sifted
  integer varied, the model is exact to 0.03% at `s = 3.21` and 26% high at `s = 2`; at the
  object's own configuration `model/real = 1.2628` at `q ~ 50,000` against `4 e^{-2 gamma} =
  1.26190`, and `HL/real = 1.0021`. Prior art named once: Mertens over-counting at sifting level
  `sqrt(x)`, `2 e^{-gamma}` per dimension; the register's nearest relative is the dimension-2
  sifting ceiling in `j2-upper-bound`, which is a statement about what a sieve can *prove*, not a
  measured density correction to this object. New as the measurement and as the identification of
  *where in the object* the global integer acts.
* **N-S5 (the correction closes the parent's factor-14 miss).** Exact first moment over 778 prime
  machines: 9.90 predicted failures below 6,000 against 17 observed; after `p -> p/4e^{-2 gamma}`,
  16.51 against 17, and the `[1000, 3000)` band's `P(X >= 2)` goes from 0.9% to 10%. The parent's
  attribution of the miss to island correlation is refuted in sign: island correlation makes the
  independent model *over*-predict failures, by 12% at `d = 60` rising to 126% at `d = 800`. New.
* **N-S6 (the failure-rate ladder against the arc, all three laws).** `1.3e-1, 5.4e-2, 2.4e-2,
  4.0e-3, 7.5e-4, 1.2e-4, 3.7e-5, 1.3e-5, 3.3e-7` at `d = 60 .. 1100` for locally-square vectors,
  with free and real vectors alongside at every rung. At the arc of the last real failure a random
  vector fails 1.5 times in 100,000. New as exact data, and it is the "how rare" the brief asked
  for under Outcome C.
* **N-S7 (the global-square test, run).** 82 explicit failing locally-square vectors at `d = 954`,
  one arc past the last real failure: exact minimum covers of 33-49 gears (against the adversarial
  `K(954) ~ 18`), moduli `10^{66}` to `10^{100}` times `q_0^2`, **0 of 82** with a perfect-square
  CRT lift, and the QR screen over the outside gears decaying as `2^{-t}` (82, 40, 17, 9, 7, 3, 2,
  2, 0 at `t = 0..8`). Control: at 21 of 24 real failures - every one with `P > q^2` - the lift is
  exactly `q^2`. New as an explicit family of counterexample vectors; the square pin itself is
  N-C7.
* **N-S8 (and why it is not a bound).** The QR screen is implied by the square condition and must
  coincide with it at `t ~ (log_2 P)/2 = 151` gears; the square condition (`10^{-45.5}`) is itself
  weaker than the range condition already used (`10^{-84}`). The owner's sharp test is N-C7 in
  another costume. Recorded as the limit, not a finding.
* Filed, not claimed: the exact reciprocity class lists for islands 12, 47, 82 (modulus 280,
  39,480, 4,920; exactly 1/4 of classes barred, 0 disagreements against every gear to 200,000) -
  quadratic reciprocity applied to a fixed integer, stopped; the top gear's inertness is worth
  2.2% of the failure rate; the minimum blocking set of the struck islands is the same to 0.4
  gears in all three laws.

## 9. What holds without exception, with counts

| statement | range | exceptions |
|---|---|---|
| reachability of island `i` is decided by `g mod M(i)`, with exactly 1/4 of the classes barred | `i = 12, 47, 82`; every prime gear `<= 200,000` | **0** |
| the failure rate of real phase vectors equals that of locally-square vectors | 6,300,000 vectors of each kind, 30 machines, 5 bands | ratio `0.9984 +- 0.0033`, no band outside 1 s.e. where testable |
| the mean open-island count is the same under all four laws | 30 machines x 4 kinds | max relative spread **3.12%** |
| the minimum open-island count is the same under all four laws | 30 machines | **0** |
| the per-offset opening probability of a real vector equals a locally-square vector's | 18 offsets x 3 machines, 147,495 real and 200,000 LS vectors | max deviation **1%** |
| the walk length of a real vector equals a locally-square vector's | 3 machines | mean 14.562/14.561, 17.108/17.133, 18.708/18.781 |
| the minimum blocking set of the struck islands is the same under all three laws | 3 machines x 200 vectors, exact ILP | max gap **0.4 gears** |
| at a real failure with `P > q^2` the cover's CRT lift is exactly `q^2` | 21 covers, `q = 41 .. 1649` | **0** (the 3 with `P < q^2` are `q = 17, 23, 29`) |
| a failing locally-square vector's cover lift is not a perfect square | 82 vectors at `d = 954` | **0 squares** |
| `K(d)` non-decreasing and the failure rate decreasing along the arc ladder | 9 arcs, `d = 60 .. 1100` | **0** |

## 10. Verdict

**FACT, and the owner's suggestion answered in a way that changes one number on the tree.**

**In one sentence: reading (b) holds and is already spent - index parity is exactly what defines
the island set and nothing more; reading (a) is worth about 1% of the failure rate with a sign
that belongs to the arc; and reading (c), the global integer, matters only where the machine sifts
to the square root of its own columns, where it costs the real object a fifth of its openings by
the classical constant `4 e^{-2 gamma}` and therefore works against the witness, not for it.**

*"Squares are even"* is true and it is the reason the object exists: index parity is exactly what
bars a gear from an offset, the bar is a condition on the gear alone (a union of progressions of
density `1/4` modulo an explicit `M(i)`, 0 disagreements against every gear to 200,000), and the
islands are the offsets barred at gears 5 and 7 at once. **But that is the whole of it.** Above
gear 7 evenness moves strikes between offsets without removing any - the first moment is
conserved exactly - and what is left, the dispersion, is worth about 1% of the failure rate with a
sign that belongs to the arc, not to the arithmetic: `-2.9%` at band 200, `+7.0%` at band 500,
`-1.14%` over 6.3 million vectors of each kind. And the joint constraint, the one P7 records as a
set of density `2^{-pi(q)}`, is worth **less than 0.7%**: real phase vectors and independent
locally-square vectors give the same failure rate (0.029653 against 0.029700), the same open-island
distribution, the same per-offset profile to 1%, the same walk length to four figures and the same
minimum blocking set to 0.4 gears.

So the pre-registered outcome is **C**: at a fixed arc and gear set the square structure is
irrelevant, and the witness holds because covers are rare among *all* phase vectors. The rate is
now on record - `1.3e-1, 5.4e-2, 2.4e-2, 4.0e-3, 7.5e-4, 1.2e-4, 3.7e-5, 1.3e-5, 3.3e-7` for
`d = 60 .. 1100` - and at `d = 954`, one arc past the last real failure, a **free** vector already
fails only 1.5 times in 100,000.

The branch's own finding is the thing that survives when the three readings are separated. There
*is* one place where being one integer's square matters, and it is not the phase vector: it is
that the machine's top gear is the **square root** of the column it is sifting. With the gear set
held fixed and the sifted integer varied, the independent model reproduces a real integer's
opening count to 0.03% at `s = log(q^2)/log(z) = 3.21` and is 26% high at `s = 2`. At `s = 2` -
the object's own configuration - the open islands are exactly the twin prime pairs above `q^2`,
their count is Hardy-Littlewood (`HL/real = 1.0021` at `q ~ 50,000`), and the phase-vector model
overstates it by `4 e^{-2 gamma} = 1.26190` (measured 1.2628). That correction closes the parent's
honest miss: the exact first moment predicts 9.90 failures below `q = 6000` against 17 observed,
and after the correction 16.51 against 17.

**And it points the wrong way.** The real machine has a fifth fewer open islands than any model of
its phase vector, so it should fail *more* often than a random vector, not less. Nothing found
here helps the witness; what was found is that the first-moment model on the tree was optimistic
by a factor of `4 e^{-2 gamma}` in the openings and by an order of magnitude in the failure count,
and it now matches.

The sharp form of the owner's idea was run to the end: 82 explicit failing locally-square vectors
one arc past the last real failure, each with an exact minimum cover of 33-49 gears and a modulus
`10^{66}`-`10^{100}` times `q_0^2`, **none** with a perfect-square CRT lift, and the
quadratic-residue screen over the gears outside the cover decaying as `2^{-t}` exactly. That is
N-C7 seen from the other side and it adds no factor to it: the screen is implied by the square
condition, and the square condition (`10^{-45.5}` per cover) is weaker than the range condition
already in hand (`10^{-84}`). The union bound over `2.7^m = 10^{47}` covers is vacuous by `10^{24}`,
unchanged.

Toward the root: no bound, and one route closed. The next interaction is unchanged from the
parent's, minus one candidate explanation - **bound the number of covers a real machine can
produce** - and this branch says the bound cannot come from the phase vector being a square,
because a real phase vector and an independent locally-square vector are indistinguishable on
every statistic of the witness that was measured.

## 11. Dead ends (do not re-enter)

* **"The phase vector being a square is why the witness holds."** Refuted at 6.3 million vectors
  per kind on 30 machines: real and locally-square failure rates agree to `0.9984 +- 0.0033`, and
  locally-square against free agrees to 1.1% pooled with an arc-dependent sign. Every derived
  statistic (open-island mean and minimum, per-offset profile, `L`, minimum blocking set) agrees
  too. The constraint of density `2^{-pi(q)}` is worth under 0.7%.
* **"Index parity acts through the small gears, additively."** Refuted: the hybrid ladder is
  non-monotone (`0.14715` free, `0.12020` at `G* = 13`, `0.14404` fully square at `q_0 = 191`),
  single-gear swaps carry signs that differ between gears and between arcs (gear 13 `-22.8%`,
  gear 17 `+23.0%` at the same machine), and at two of four machines the whole tail `g > 100`
  moves the rate more than gear 11 does.
* **"The parent's factor-14 miss is island correlation."** Refuted in sign: the independent-island
  model *over*-predicts the true failure rate of the same law by 12% at `d = 60` rising to 126% at
  `d = 800`. The miss is `4 e^{-2 gamma}` in the opening density.
* **The quadratic-residue screen over the gears outside a cover as a new counting factor.** It is
  implied by "the lift is a perfect square" and must saturate at `t ~ (log_2 P)/2 = 151` gears; the
  square condition is in turn weaker than the range condition the parent already uses. Measured
  decay `2^{-t}` on 82 failing vectors, recorded, no factor gained.
* **Extending the `s`-dependence into a Buchstab-function study.** The deviation of the sifted
  density from the Mertens product as a function of `s` is the classical Buchstab/2-dimensional
  sieve function (the non-monotonicity at `s = 2.67`, `model/real = 0.9836`, is its oscillation).
  Named and stopped at the stop line; the register already carries the dimension-2 sifting ceiling
  from the bound side (`j2-upper-bound`).
