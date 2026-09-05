# Branch R2.a.i.a.1.a.2 - THE SECOND MOMENT OVER q

Parent: node R2.a.i.a.1.a (the cover number, the transfer obstruction;
`research/proof/cover_number.md`, and the reopened entry R2.a.i.a.1.a Idea 2 in
`research/proof/dead_branches_reopened.md`). The observation that spawned this branch: the parent's
first moment gets the island witness right to within a factor (16.5 expected failures against 17
observed, with the `s = 2` correction `4 e^{-2 gamma}`), but a first moment cannot separate "rare"
from "never", and nobody has written down the SECOND moment of the open-island count over `q`. Thin
place 5 of `research/proof/the_wall.md` names it: prove that a VANISHING FRACTION of `q` fail.

Scripts: `research/anchor235/r50/mom_*.py`. Result outputs (untracked):
`research/anchor235/r50/results/`. Every number this document relies on is written into the
document.

---

## 0. Pre-registered (written before any computation of this branch)

### 0.1 The object, chosen and stated

Fix an integer `q` with `gcd(q, 30) = 1`. Write `u_g = 6^{-1} (mod g)`, `d = d(q) = 2 u_q (mod q)`
(the top gear's forward tooth arc: `(q+1)/3` if `q = 5 (mod 6)` - the SHORT arc; `(2q+1)/3` if
`q = 1 (mod 6)` - the LONG arc).

**The island set.** I take the ONE-CLASS witness set of N-I2 and the FULL arc:

```
    I(q) = { i : i = 12 (mod 35),  1 <= i < d(q) },      m(q) = |I(q)|
```

Not the arc fraction `0.152 d` and not the fixed cap 2,392: the full arc maximises the island
count `m ~ d/35`, hence maximises the mean `E[N]`, and it is the largest set on which the witness
is known to hold with 0 exceptions (N-I2: class 12 alone works from `q = 5477`). Both alternatives
are strictly smaller sets and would only weaken every moment bound; they are recorded as
alternatives, not used.

**Open.** Island `i` is OPEN at `q` iff no gear `g` prime with `7 < g <= q` divides `q^2 + 6i - 2`
or `q^2 + 6i`, i.e. iff

```
    q^2 != -6i  (mod g)   and   q^2 != 2 - 6i  (mod g)      for every gear  7 < g <= q.
```

Equivalently `i != -q^2 u_g (mod g)` and `i != (2 - q^2) u_g (mod g)`.

**The count.** `N(q) = #{ i in I(q) : i open }`. The witness at `q` is `N(q) >= 1`. The band is

```
    A(X) = { q in [X, 2X] : gcd(q, 30) = 1 },       |A(X)| = (8/30) X + O(1).
```

**Note on what an opening is.** For `q = 5 (mod 6)` (short arc) `6i < 2q`, so an open island is a
pair of integers in `(q^2, q^2 + 2q) subset (q^2, (q+1)^2)` free of every prime factor `<= q`:
both members are PRIME, so an open island IS a twin prime pair. For `q = 1 (mod 6)` (long arc)
`6i` can reach `4q` and the identification has a negligible exception class (a member could be a
product of two primes both in `(q, q+4)`); it is reported, not used.

### 0.2 The exact pairwise joint density (stated before computing)

By the doubling law (N-R6), gear `g` strikes offset `i` for exactly `2 chi_g(i)` classes of
`q mod g`, where `chi_g(i)` in `{0, 1, 2}` counts how many of the two targets `-6i`, `2 - 6i` are
NONZERO quadratic residues mod `g`. Write `a_g(i) = 2 chi_g(i) / (g - 1)`. Let

```
    o_g(i, j) = # classes of q mod g that strike BOTH i and j.
```

Over a full period of `prod_{7 < g <= q} g` (CRT, one factor per gear, exact) the joint density is

```
    rho_2(i, j) = prod_{7 < g <= q} ( 1 - a_g(i) - a_g(j) + o_g(i, j)/(g - 1) )
    rho_1(i)    = prod_{7 < g <= q} ( 1 - a_g(i) )
```

and I pre-register the exact evaluation of `o_g`. Gear `g` strikes `i` at `q^2 in {-6i, 2-6i}` and
`j` at `q^2 in {-6j, 2-6j}`. With `delta = j - i`:

* if `g | delta` the two target pairs coincide, so `o_g = 2 chi_g(i)` (0, 2 or 4);
* else the pairs meet iff `-6i = 2 - 6j` or `-6j = 2 - 6i (mod g)`, i.e. iff `3 delta = -+1 (mod g)`,
  i.e. iff `g | 3 delta - 1` or `g | 3 delta + 1`; then the shared target is a single value `x` and
  `o_g = 2` if `x` is a nonzero QR mod `g`, else `o_g = 0`;
* else `o_g = 0`.

So `o_g` is 0, 2 or 4 - never odd - and the gears that correlate two islands are exactly the prime
factors above 7 of `delta`, `3 delta - 1` and `3 delta + 1`. (Prior art in one line: the letter-gear
rule "the gears of a middle gap `v` are the prime factors of `3v -+ 1`" is on record in
`research/proof/separability.md`; its appearance here as the pair-correlation support is the new
use, not the rule.)

### 0.3 The theory

**T.** The open-island count `N(q)` is a Poisson-like sum: within a `q` the CRT variance equals the
mean to within a few percent, because the DISJOINTNESS deficit of a generic gear (a gear cannot
strike two islands at once, which correlates the two islands NEGATIVELY by `-a_g(i) a_g(j) ~ -4/g^2`)
is cancelled, to leading order, by the two coincidence families of 0.2 (a gear with `g | delta`
contributes `+2/g^2` on average and the two separation families `g | 3 delta -+ 1` contribute
`+2/g^2` between them). If T holds, Chebyshev on the normalised count bounds the failing fraction
by `~1/mu(q)`, which vanishes like `(ln X)^2 / X`.

### 0.4 Predictions, with numbers, and what refutes each

* **M1 (the overlap is never odd, and its support is the letter gears).** `o_g(i, j)` in `{0, 2, 4}`
  and `o_g > 0` only if `g | delta`, `g | 3 delta - 1` or `g | 3 delta + 1`. Predicted 0 exceptions
  over every gear `11 <= g <= 500` and every island pair with `delta <= 3500`, exhaustive.
  REFUTED by one odd value or one out-of-support gear.
* **M2 (the joint-density formula is exact).** The formula of 0.2 matches brute force over a full
  period of the gear product for the gear sets `{11, 13}`, `{11, 13, 17}`, `{11, 13, 17, 19}` and
  every island pair with `delta <= 700`, to within `1e-12`. REFUTED by one mismatch.
* **M3 (the cancellation).** The mean over island pairs of the per-gear log correction
  `L_g(i, j) = log[(1 - a_i - a_j + o/(g-1)) / ((1 - a_i)(1 - a_j))]` is `O(1/g^3)`, not `O(1/g^2)`:
  predicted `|mean_pairs L_g| < 10 / g^3` for every gear `11 <= g <= 1000`, and the total
  `sum_{i != j} C(i, j) / mu^2` below 0.05 in absolute value at every sampled `q`. REFUTED by a mean
  of definite sign at order `1/g^2` (which would be `|mean L_g| > 1/g^2` for most gears).
* **M4 (Poisson within `q`).** `Var_model(q) / mu(q)` in `[0.90, 1.10]` at every sampled `q` from
  `q = 1000` up. REFUTED by a sampled `q` outside `[0.8, 1.2]`.
* **M5 (the raw band variance is dominated by the arc dichotomy).** The measured `Var[N]` over
  `A(X)` is NOT Poisson-like: predicted `Var/E > 3` at every `X >= 4000` and growing linearly in
  `X`, because `mu(q)` itself runs over a factor of about 2 inside the band (short arc against long
  arc) and over a further factor 2 from `q = X` to `q = 2X`. Consequently the raw Chebyshev ratio
  `Var/E^2` tends to a CONSTANT (predicted between 0.05 and 0.20) and does not vanish.
  REFUTED if `Var/E^2` falls below 0.02 at any `X`.
* **M6 (the normalised bound vanishes).** With `mu-hat(q) = m(q) prod_{7 < g <= q}(1 - 2/g)` and
  `B(X) = (1/|A|) sum_q (N(q) - mu-hat(q))^2 / mu-hat(q)^2`, predicted `B(X) < 1` from `X = 4000` on
  and `B(64000)/B(8000) < 0.25` (the `(ln X)^2/X` rate). REFUTED if `B` fails to fall by a factor 2
  per doubling of `X` above 8,000.
* **M7 (failures of the one-class object).** Class 12 alone fails below `q = 5477` (N-I2). Predicted
  failure counts in the bands: `X = 1000` at least 3, `X = 2000` at least 1, `X = 4000` at least 1,
  and exactly 0 at `X >= 8000`. REFUTED by a failure above 11,000.
* **M8 (the equidistributed core is 4 gears).** Over `[X, 2X]` the count of `q` in a class modulo a
  product `P` is `X/P + O(1)`, so only gear sets with `prod <= X` are equidistributed. Predicted the
  largest such set of gears above 7 is `{11, 13}` at `X = 1000, 2000`, `{11, 13, 17}` (2,431) at
  `X = 4000 .. 32000`, `{11, 13, 17, 19}` (46,189) at `X = 64000` - i.e. at most FOUR gears of the
  roughly 6,000 in the machine. Predicted the variance contribution computable from the core alone
  is below 1% of the measured `Var`. REFUTED if a core of more than 5 gears is available at any `X`.
* **M9 (the theorem attempt fails at the FIRST moment, not the second).** `sum_{q in A} N(q)` counts
  twin prime pairs in `(q^2, q^2 + 2q)` in a fixed class mod 35, summed over `q` in `[X, 2X]`; a
  lower bound for it is a lower-bound sieve of dimension 2 at sifting level `s = 2`. Predicted: no
  rigorous `B(X) -> 0` follows, and the blocking term is the first moment's lower bound, not any
  second-moment error. REFUTED if a rigorous chain from the pairwise densities to `B(X) -> 0` can be
  written with only the doubling law and per-class counting.
* **M10.** Everything that holds without exception over the sweep, with counts.

### 0.5 Scorecard

| # | prediction | verdict and evidence |
|---|---|---|
| M1 | `o_g` in `{0,2,4}`, support = prime factors of `delta`, `3 delta -+ 1` | **CONFIRMED**, 0 exceptions in 359,712,683 (gear, ordered residue pair) cells over 299 gears - a far larger range than pre-registered (2.1) |
| M2 | joint-density formula exact against brute force | **CONFIRMED**: 630 checks over three gear sets, worst deviation 1.11e-16; the derived variance also matches the full-period variance to 5.4e-12 (2.2) |
| M3 | mean per-gear log correction is `O(1/g^3)`; total below 0.05 | **REFUTED on the main clause**: the mean is `O(1/g^2)` with a fixed negative sign, `-0.28` to `-2.69` per `g^2`; `\|mean L_g\| g^3` reaches 2,644. The mechanism is right but the repayment is 89-94% below `g = m`, exactly 50.0% at `g = m`, 32-39% above (2.3). Second clause refuted only at the smallest band (0.0601) |
| M4 | `Var_model/mu` in `[0.90, 1.10]` | **REFUTED**: 0.7625 to 0.8147 over 42 sampled `q`, never above 0.82. The count is SUB-Poisson, split by arc (0.802-0.815 short, 0.762-0.793 long) (2.5) |
| M5 | raw `Var/E > 3` from 4,000; `Var/E^2` tends to a constant 0.05-0.20 | **CONFIRMED on the main clause** (`Var/E^2` = 0.684 -> 0.159, ratios rising to 0.924, limit near 0.145); subsidiary clause refuted at `X = 4000, 8000` (1.53, 2.13) (3.1, 3.2) |
| M6 | `B(X) < 1` from 4,000; falls by more than 2 per doubling | **CONFIRMED and exceeded**: 0.507, 0.329, 0.188, 0.120, 0.0653, 0.0393, 0.0228; ratio 0.58 per doubling; `B(X) X/(ln X)^2` flat at 11.4 +- 0.7 (4.2) |
| M7 | failures 3+, 1+, 1+, then 0 from `X = 8000` | **REFUTED**: 52, 37, 12, **3**, 0, 0, 0; the band `[8000, 16000]` holds `q = 10403, 11663, 11921`, two of them above the pre-registered refutation line of 11,000 (3.1) |
| M8 | core of at most 4 gears; the core is a negligible part of the sifting | **CONFIRMED**: cores of 2, 2, 3, 3, 3, 3, 4 gears against 299 to 11,983 in the machine; equidistributed fraction of the sifting `9.2e-5` at `X = 64000`, gear pairs with `g h <= 2X` `1.9e-4` of all pairs (5.1) |
| M9 | the wall is the first moment, not the second | **CONFIRMED, and sharpened past the prediction**: the first moment is itself a twin count (6.1), AND the conclusion is of the target's own strength (N-M6, 6.2) |
| M10 | exceptionless statements | nine, listed in section 8 |

---

## 1. Setup (exact ranges)

No sampling except where a row says so.

| object | range | script |
|---|---|---|
| `N(q)`, `m(q)`, `d(q)` exactly | **every** `q` coprime to 30 in `[X, 2X]` for `X = 1000, 2000, 4000, 8000, 16000, 32000, 64000` - 33,868 machines covering `[1000, 128000]` contiguously, gears to 128,000 | `mom_scan.py` |
| `o_g(i, j)` exhaustive | every gear `11 <= g <= 2000` (299 gears) and every ordered residue pair mod `g`: 359,712,683 cells - wider than the pre-registered `g <= 500`, `delta <= 3500` | `mom_pair.py` A |
| the joint-density formula against a full gear-product period | gear sets `{11,13}`, `{11,13,17}`, `{11,13,17,19}` (periods 143, 2,431, 46,189), every pair among the first 20 islands: 630 checks | `mom_pair.py` B |
| mean per-gear log correction over all island pairs | gears 11..997 at `q = 65003` (`m = 619`, 382,542 ordered pairs) | `mom_pair.py` C |
| exact `mu(q)` and `Var_model(q)` (full pair sum, exact `chi_g` per island per gear, all gears to `q`) | 42 sampled primes, 6 per band, 3 short arc and 3 long | `mom_pair.py` D |
| the whole variance machinery against the exact distribution of `N` over a full period | 4 complete periods, up to 46,189 classes and 80 islands | `mom_struct.py` E |
| exact `mu(q)` against the mean-rate proxy `mu-hat(q)` | the same 42 `q` | `mom_struct.py` F |
| the covariance profile by island separation | `q = 15259, 30307, 60727` (all separations, `m` up to 1,157) | `mom_struct.py` G |
| `N/mu*` by the least prime factor of `q`, by `omega_{>7}(q)`, by prime/composite | all 33,868 machines | `mom_div.py` |
| the band moments, the arc split, the variance decomposition, `B(X)`, the equidistributed core | all bands | `mom_bound.py`, `mom_final.py` |
| twin, cousin and balanced semiprime families against matched controls | every `q = p(p+2)`, `p(p+4)` and `p x nextprime(p+50)` up to `10^7` (81, 88, 441 machines) and 972 controls | `mom_twin.py` |

### 1.1 A correction to the object, forced by gear 7

The brief's definition sifts by the gears in `(7, q]` only. For `q` coprime to 210 that is the
real machine on the island classes, because gears 5 and 7 are barred there by definition. For
`7 | q` it is not: `q^2 = 0 (mod 7)` and `i = 12 (mod 35)` gives `i = 5 (mod 7)`, so gear 7
divides `q^2 + 6i - 2` at EVERY island of the class. An "open" island at such a `q` is a pair
`(7M, 7M+2)` with `M` and `7M+2` prime, not a twin pair. It is also measurably easier: the
cofactor `M ~ q^2/7` is sifted at a smaller size, so the open count runs above the model by
`ln(q^2)/ln(q^2/7)`. Measured over all 33,868 machines, by the least prime factor of `q`:

| least prime factor of `q` | 7 | 11 | 13 | 17 | 19 | 23 | 29 | 31 | 37 | 41 | 43 | 47 | 53+ | `q` prime |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| count | 4,838 | 2,639 | 2,028 | 1,433 | 1,208 | 943 | 714 | 641 | 515 | 454 | 427 | 385 | 5,824 | 11,819 |
| `sum N / sum mu*` | **1.1189** | 0.9988 | 1.0023 | 1.0128 | 1.0094 | 1.0073 | 0.9985 | 1.0011 | 1.0060 | 0.9969 | 1.0085 | 0.9898 | 1.0021 | 0.9977 |

Every class except `7 | q` sits on 1.000 to within 1%; the multiples of 7 sit 11.9% high, which is
the cofactor effect and nothing else. **So the object is reported twice**: as the brief defines it
(`q` coprime to 30) and restricted to `q` coprime to 210, which is the version in which an open
island is an opening of the real machine. Everything in sections 4 to 7 that speaks about twin
primes uses a third, strictly certified version defined in 4.3.

---

## 2. Results - the exact pairwise joint density (item 2)

### 2.1 The overlap law, exhaustive

> **N-M1 (the overlap law).** For two offsets `i != j` with `delta = j - i` and a gear `g > 7`,
> the number `o_g(i, j)` of classes of `q mod g` that strike both is
>
> * `2 chi_g(i)` if `g | delta` (the two target pairs coincide) - 0, 2 or 4;
> * `2` if `g | 3 delta - 1` or `g | 3 delta + 1` AND the single shared target
>   (`-6i` in the first case, `2 - 6i` in the second) is a nonzero quadratic residue mod `g`;
> * `0` otherwise.
>
> In particular `o_g` is NEVER odd, and **the gears that correlate two islands at separation
> `delta` are exactly the prime factors above 7 of `delta`, `3 delta - 1` and `3 delta + 1`.**

Exhaustive verification over every gear `11 <= g <= 2000` (299 gears) and every ordered residue
pair `(i, j) mod g` - **359,712,683 cells**, of which 276,734 carry `o_g > 0`:

| check | exceptions |
|---|---|
| `o_g` odd | **0** |
| `o_g` outside `{0, 2, 4}` | **0** |
| `o_g > 0` with `3 delta != +-1 (mod g)` and `g` not dividing `delta` | **0** |
| a gear whose diagonal count `2 chi_g(i)` is odd or exceeds 4 | **0** |

M1 CONFIRMED. Prior art in one line: the same divisor condition `g | 3v -+ 1` is the letter-gear
rule of `research/proof/separability.md` ("the letter gears of a middle gap `v` are the prime
factors of `3v - 1` and `3v + 1`"); what is new here is that this rule is exactly the support of
the pair correlation of the island process, and that the value on that support is forced to be
even by the doubling law.

### 2.2 The joint density, against brute force

With `a_g(i) = 2 chi_g(i)/(g-1)`, over a full period of the gear product (CRT, `q` coprime to it):

```
    rho_2(i, j) = prod_{7 < g <= q} ( 1 - a_g(i) - a_g(j) + o_g(i, j)/(g - 1) )
```

Checked against brute force over the complete period, for every island pair among the first 20
islands of the class:

| gear set | period | checks | worst \|measured - formula\| |
|---|---|---|---|
| `{11, 13}` | 143 | 210 | 1.11e-16 |
| `{11, 13, 17}` | 2,431 | 210 | 1.11e-16 |
| `{11, 13, 17, 19}` | 46,189 | 210 | 1.11e-16 |

M2 CONFIRMED (630 checks, worst deviation 1.11e-16 - floating point).

The whole variance machinery (not only the pair formula) was then validated end to end against the
exact distribution of `N` over a full period:

| gear set | period | islands | brute `E[N]` | brute `Var[N]` | model `mu` | model `Var` | `dE` | `dVar` |
|---|---|---|---|---|---|---|---|---|
| `{11,13}` | 143 | 30 | 20.633333 | 0.765556 | 20.633333 | 0.765556 | 0 | 1.05e-12 |
| `{11,13,17}` | 2,431 | 30 | 18.108333 | 1.304931 | 18.108333 | 1.304931 | 0 | 1.97e-13 |
| `{11,13,17,19}` | 46,189 | 40 | 21.557407 | 2.117075 | 21.557407 | 2.117075 | 3.6e-15 | 1.36e-12 |
| `{11,13,17,19}` | 46,189 | 80 | 43.369444 | 2.614437 | 43.369444 | 2.614437 | 7.1e-15 | 5.44e-12 |

### 2.3 The mechanism: a generic gear costs `4/g^2`, the coincidences repay most of it

For a gear `g` and an island pair, write
`L_g(i,j) = log[(1 - a_i - a_j + o/(g-1)) / ((1 - a_i)(1 - a_j))]`. Expanding in `1/g`:

```
    L_g  =  o/(g-1)  -  a_i a_j  +  O(1/g^3)
```

so a gear with no coincidence costs `-a_i a_j`, whose mean over pairs is `-4/g^2` (a gear cannot
strike two islands at once: the two islands compete for the gear's classes). The repayment is
`+2/g^2` from the divisor family `g | delta` (frequency `1/g`, value `+2/g`) and `+2/g^2` from the
two separation families `g | 3 delta -+ 1` (frequency `2/g`, value `+1/g` each). The pre-registered
theory T said these cancel exactly. Measured at `q = 65003` (short arc, `d = 21668`, `m = 619`
islands, 382,542 ordered pairs), the mean of `L_g` over all pairs, gear by gear:

| gear | 11 | 13 | 17 | 19 | 23 | 29 | 31 | 43 | 53 | 101 | 199 | 283 | 409 | **619** | 647 | 907 | 983 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| generic-only mean `x g^2` | -9.08 | -7.73 | -6.05 | -5.83 | -5.41 | -5.02 | -4.94 | -4.68 | -4.59 | -4.27 | -4.06 | -4.05 | -4.00 | -4.04 | -4.05 | -4.05 | -3.95 |
| total mean `x g^2` | -1.01 | -0.82 | -0.55 | -0.47 | -0.39 | -0.33 | -0.31 | -0.28 | -0.28 | -0.38 | -0.65 | -0.89 | -1.04 | **-2.02** | -2.03 | -2.45 | -2.69 |
| repaid | 88.8% | 89.4% | 90.9% | 91.9% | 92.8% | 93.4% | 93.7% | 94.0% | 93.9% | 91.1% | 83.9% | 78.1% | 73.9% | **50.0%** | 49.9% | 39.4% | 31.9% |

The generic deficit is `-4.05/g^2` to within 1% for every gear from 31 to 983 - the predicted
`-4/g^2`, measured. The repayment is 89% to 94% while `g` is well below the island count, then
**falls to exactly 50.0% at `g = 619 = m`** and to 32-39% above it: for `g > m` no pair separation
`delta = 35k` with `k < m` is divisible by `g`, so the divisor family is empty and precisely half
the repayment - the `+2/g^2` half - disappears. That is the mechanism confirmed at a number, and it
is why the cancellation is not exact.

**M3 is REFUTED as stated**: the mean is `O(1/g^2)` with a definite negative sign
(`-0.28` to `-2.7` per `g^2`), not `O(1/g^3)`; `|mean L_g| g^3` runs from 8.97 to 2644 over
gears 11..997, far outside the pre-registered bound 10. The sum over gears 11..997 of the mean
`L_g` is `-2.038e-2`. The second clause of M3 (`|sum C / mu^2| < 0.05`) is refuted only at the
smallest band, where it reaches 0.0601; from `q = 2207` on it is inside.

### 2.4 Where the correlation lives: the separation profile

`C(i,j) = rho_2(i,j) - rho_1(i) rho_1(j)`, summed over all pairs at a fixed separation
`delta = 35 k`, at `q = 30307` (long arc, `m = 577`, `mu = 31.50`):

| `k` | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `sum C` at that separation | -0.124 | **+0.112** | -0.420 | **-0.471** | -0.422 | -0.141 | -0.296 | -0.350 | **+0.029** | -0.474 | **+0.539** | -0.148 |
| smallest correlating gear | 13 | **11** | 79 | none < 400 | 131 | 17 | 23 | 29 | **11** | none < 1000 | **11** | 13 |

The profile is read off the gear list of `delta`: among the separations inspected, those whose
smallest correlating gear is 11 (`k = 2, 9, 11, 33`) are the only POSITIVE ones; the most negative
(`k = 4, 10, 15, 32, 56, 60`) are exactly those for which `delta`, `3 delta - 1` and `3 delta + 1`
have no prime factor above 7 below a few hundred, so the disjointness deficit is unpaid.
`k <= 12` carries 0.39 to 0.53 of the total and `k <= 100` carries 0.52 to 0.88; the rest is a
thin tail over all separations.

The totals are stable in the ratio that matters:

| `q` | arc | `m` | `mu` | `sum_{i != j} C` | `sum C / mu` |
|---|---|---|---|---|---|
| 15,259 | long | 291 | 18.12 | -2.7017 | -0.1491 |
| 30,307 | long | 577 | 31.50 | -4.8549 | -0.1541 |
| 60,727 | long | 1,157 | 55.50 | -8.7189 | -0.1571 |

`sum C` is LINEAR in `mu`, not quadratic: the `m^2` generic pairs very nearly cancel and what
survives is a per-island constant of about `-0.15`. That is why the variance stays positive.

### 2.5 The CRT variance is sub-Poisson, at two constants

`Var_model(q) = mu + sum_{i != j} rho_2 - mu^2 = mu - sum_i rho_1(i)^2 + sum_{i != j} C(i,j)`,
computed exactly (every gear, every island, exact `chi_g`) at 42 sampled primes, 6 per band:

| band | `q` | arc | `m` | `mu(q)` | `Var_model` | `Var/mu` | `sum C / mu^2` | `N(q)` |
|---|---|---|---|---|---|---|---|---|
| 1,000 | 1,097 | short | 11 | 1.2547 | 1.0115 | 0.8062 | -0.060144 | 2 |
| 1,000 | 1,867 | long | 36 | 3.6194 | 2.7599 | 0.7625 | -0.035268 | 0 |
| 4,000 | 4,373 | short | 42 | 3.4618 | 2.7759 | 0.8019 | -0.031078 | 2 |
| 4,000 | 7,561 | long | 144 | 10.4481 | 8.1453 | 0.7796 | -0.013611 | 10 |
| 16,000 | 30,203 | short | 288 | 15.6608 | 12.7100 | 0.8116 | -0.008289 | 15 |
| 16,000 | 30,307 | long | 577 | 31.5021 | 24.7785 | 0.7866 | -0.004892 | 30 |
| 64,000 | 121,493 | short | 1,157 | 49.1365 | 40.0316 | **0.8147** | -0.002832 | 34 |
| 64,000 | 121,351 | long | 2,312 | 98.2509 | 77.9443 | **0.7933** | -0.001633 | 75 |

Over all 42: `Var_model/mu` in **[0.7625, 0.8147]**, never once above 0.82, and split cleanly by
arc - **0.802 to 0.815 on the short arc, 0.762 to 0.793 on the long arc**, rising slowly with `q`
in both. **M4 is REFUTED**: the count is not Poisson, it is sub-Poisson by 19-24%, and the deficit
is exact:

```
    Var_model/mu  =  1  -  (mean opening density)  -  |sum C|/mu
                  =  1  -  0.04..0.06             -  0.14 (short) / 0.155 (long).
```

The long arc is more sub-Poisson than the short arc for a stated reason: it has twice as many
islands, so each island has more partners inside the correlation range, and `|sum C|/mu` is larger.

---

## 3. Results - the measured moments over `q` (item 1)

Exact `N(q)` for **every** `q` coprime to 30 in `[X, 2X]`, gears `(7, q]`, islands `i = 12 (mod 35)`
in `[1, d(q))`. 33,868 machines in all, covering `[1000, 128000]` contiguously.

### 3.1 The bands

| `X` | `\|A\|` | `E[N]` | `Var[N]` | `Var/E` | `Var/E^2` | failures | failing fraction | `E[mu*]` | `E[N]/E[mu*]` |
|---|---|---|---|---|---|---|---|---|---|
| 1,000 | 268 | 1.8619 | 2.373 | 1.274 | 0.68441 | 52 | 1.940e-01 | 1.8476 | 1.0077 |
| 2,000 | 532 | 3.1767 | 4.439 | 1.397 | 0.43985 | 37 | 6.955e-02 | 3.0797 | 1.0315 |
| 4,000 | 1,068 | 5.3127 | 8.101 | 1.525 | 0.28700 | 12 | 1.124e-02 | 5.2145 | 1.0188 |
| 8,000 | 2,132 | 9.1173 | 19.394 | 2.127 | 0.23332 | 3 | 1.407e-03 | 8.9479 | 1.0189 |
| 16,000 | 4,268 | 15.8503 | 49.896 | 3.148 | 0.19860 | **0** | 0 | 15.5240 | 1.0210 |
| 32,000 | 8,532 | 27.6643 | 131.580 | 4.756 | 0.17193 | **0** | 0 | 27.1889 | 1.0175 |
| 64,000 | 17,068 | 48.8606 | 379.243 | 7.762 | 0.15885 | **0** | 0 | 48.0194 | 1.0175 |

`mu* = mu-hat / (4 e^{-2 gamma})` is the mean-rate model with the `s = 2` correction. It lands on
the truth to 2% at every band (and to 0.3% once the multiples of 7 of 1.1 are removed) - the
`4 e^{-2 gamma} = 1.26190` handicap of `research/proof/square_vector.md` confirmed on 33,868
machines rather than at one `q`. The exact-`chi` first moment `mu(q) = sum_i rho_1(i)` differs
from the mean-rate proxy `mu-hat` by less than 1% from `q = 4000` on (mean ratio 0.988 over the
42 sampled `q`, running 0.937 at `q = 1097` to 1.0007 at `q = 30307`), so nothing below turns on
which of the two is used.

**M7 is REFUTED as stated.** The failure counts are 52, 37, 12, 3, 0, 0, 0 - the pre-registered
"exactly 0 at `X >= 8000`" is wrong: the band `[8000, 16000]` contains three failures,
`q = 10403 = 101 x 103`, `11663 = 107 x 109` and `11921 = 7 x 13 x 131`. The pre-registered
refutation condition ("a failure above 11,000") is met twice.

### 3.2 `Var/E^2` does not vanish - the arc dichotomy

`Var/E` climbs 1.27, 1.40, 1.53, 2.13, 3.15, 4.76, 7.76 (roughly `x 1.5` per doubling) and
`Var/E^2` falls 0.684, 0.440, 0.287, 0.233, 0.199, 0.172, 0.159 with successive ratios
0.643, 0.652, 0.813, 0.851, 0.866, **0.924** - converging to 1, i.e. `Var/E^2` tends to a positive
constant near 0.145. **M5 CONFIRMED on its main clause** (the constant is inside the pre-registered
[0.05, 0.20]); its subsidiary clause "`Var/E > 3` at every `X >= 4000`" is refuted at `X = 4000`
and 8,000 (1.53, 2.13) and holds from 16,000.

The cause is that `mu(q)` is not constant across a band. Splitting by arc:

| `X` | `E` short | `Var/E` short | `E` long | `Var/E` long |
|---|---|---|---|---|
| 1,000 | 1.0597 | 0.884 | 2.6642 | 0.946 |
| 8,000 | 6.1238 | 1.014 | 12.1107 | 1.210 |
| 64,000 | 32.5600 | 1.745 | 65.1612 | 2.613 |

and by the decomposition `Var[N] = Var[mu*] + Var[N - mu*] + 2 Cov`:

| `X` | `Var[N]` | `Var[mu*]` | `Var[N-mu*]` | `2 Cov` | `Var[N-mu*]/E[N]` |
|---|---|---|---|---|---|
| 1,000 | 2.373 | 0.451 | 1.617 | +0.304 | 0.8687 |
| 8,000 | 19.394 | 10.947 | 8.450 | -0.003 | 0.9268 |
| 16,000 | 49.896 | 33.216 | 13.859 | +2.820 | 0.8744 |
| 32,000 | 131.580 | 102.490 | 25.177 | +3.913 | 0.9101 |
| 64,000 | 379.243 | 321.517 | 45.736 | +11.990 | 0.9360 |

By `X = 64000`, **85% of the measured variance is the systematic run of `mu(q)` across the band**
(the short/long arc factor 2 and the factor 2 from `q = X` to `q = 2X`) and only 12% is
fluctuation. Raw Chebyshev on `N` therefore cannot vanish, and any bound that is to vanish must
normalise by `mu(q)`.

### 3.3 The measured fluctuation against the CRT prediction

The CRT model predicts `Var_model(q) = 0.813 mu` (short) / `0.789 mu` (long) with
`mu = mu-hat` to within 1%, i.e. `Var_model ~ 1.009 mu*` - essentially Poisson **against the
true mean**. Measured:

| `X` | `Var[N - mu*]` | CRT prediction `E[Var_model]` | ratio |
|---|---|---|---|
| 1,000 | 1.617 | 1.848 | 0.875 |
| 2,000 | 2.895 | 3.080 | 0.940 |
| 4,000 | 4.352 | 5.214 | 0.835 |
| 8,000 | 8.450 | 8.947 | 0.944 |
| 16,000 | 13.859 | 15.523 | 0.893 |
| 32,000 | 25.177 | 27.188 | 0.926 |
| 64,000 | 45.736 | 48.017 | **0.953** |

The real fluctuation is 5-17% BELOW the CRT prediction, and the ratio rises monotonically over the
last three bands (0.893, 0.926, 0.953) toward 1. So the CRT second moment is not merely
qualitatively right: it is quantitatively right to within 5% at the largest band and getting
better. That is the strongest positive result of the branch, and it is a measurement, not a bound.

---

## 4. The bound `B(X)`, and what it is not

### 4.1 The identity behind it

For every `q` with `N(q) = 0` the quantity `(N(q) - mu*(q))^2 / mu*(q)^2` equals exactly 1, so

```
    #{ q in A(X) : N(q) = 0 } / |A(X)|   <=   B(X) := (1/|A|) sum_{q in A} (N(q) - mu*(q))^2 / mu*(q)^2
```

with no hypothesis at all. This is Chebyshev applied per `q` rather than per band; the per-band
form `Var/E^2` is the same inequality after replacing `mu*(q)` by the band mean, and 3.2 shows
that replacement costs everything.

### 4.2 `B(X)` measured

| `X` | `B(X)` | ratio to previous | raw `Var/E^2` | actual failing fraction | `B(X) X / (ln X)^2` |
|---|---|---|---|---|---|
| 1,000 | 0.50701 | - | 0.68441 | 1.940e-01 | 10.63 |
| 2,000 | 0.32938 | 0.6496 | 0.43985 | 6.955e-02 | 11.40 |
| 4,000 | 0.18839 | 0.5720 | 0.28700 | 1.124e-02 | 10.95 |
| 8,000 | 0.12009 | 0.6374 | 0.23332 | 1.407e-03 | 11.89 |
| 16,000 | 0.06527 | 0.5435 | 0.19860 | 0 | 11.14 |
| 32,000 | 0.03928 | 0.6018 | 0.17193 | 0 | 11.68 |
| 64,000 | 0.02283 | 0.5813 | 0.15885 | 0 | 11.93 |

**M6 CONFIRMED**: `B(X) < 1` from the first band computed, it falls by a factor 0.58 +- 0.04 per
doubling, and `B(64000)/B(8000) = 0.190 < 0.25`. The last column is flat: `B(X) = 11.4 (ln X)^2 / X`
to within 6% over a 64-fold range in `X`, which is exactly `1/mu(q)` averaged over the band, as
theory T predicts. Chebyshev is loose by a factor 2.6 at `X = 1000` rising to 85 at `X = 8000`,
which is the usual gap between a second-moment bound (`1/mu`) and a Poisson tail (`e^{-mu}`).

### 4.3 The twin-certified object

To speak about twin primes the object has to be restricted so that an open island is provably a
twin pair. Take `q = 5 (mod 6)` and `gcd(q, 7) = 1`; then `d = (q+1)/3`, `i < d` gives
`6i <= 2q - 4`, so both members lie in `(q^2, (q+1)^2)`; and `i = 12 (mod 35)` with
`gcd(q, 210) = 1` forces both members coprime to 2, 3, 5 and 7. An open island is then a pair of
integers below `(q+1)^2` with no prime factor at most `q`: **both are prime, and they are twins.**

| `X` | `\|A''\|` | `E[N]` | `Var` | `Var/E` | failures | failing fraction | `B(X)` | `B X/(ln X)^2` |
|---|---|---|---|---|---|---|---|---|
| 1,000 | 115 | 1.0696 | 0.969 | 0.906 | 37 | 3.217e-01 | 0.64405 | 13.50 |
| 2,000 | 227 | 2.0220 | 2.057 | 1.017 | 29 | 1.278e-01 | 0.42623 | 14.76 |
| 4,000 | 458 | 3.4803 | 3.306 | 0.950 | 11 | 2.402e-02 | 0.25108 | 14.60 |
| 8,000 | 914 | 6.0066 | 5.779 | 0.962 | 2 | 2.188e-03 | 0.14276 | 14.14 |
| 16,000 | 1,829 | 10.2471 | 10.938 | 1.068 | **0** | 0 | 0.07988 | 13.64 |
| 32,000 | 3,657 | 18.0254 | 23.840 | 1.323 | **0** | 0 | 0.04925 | 14.64 |
| 64,000 | 7,315 | 32.0472 | 53.889 | 1.682 | **0** | 0 | 0.02749 | 14.36 |

Same law, `B(X) = 14.2 (ln X)^2 / X`.

---

## 5. The error terms, stated honestly (item 3)

### 5.1 What is equidistributed over a band of length `X`

The count of `q` in `[X, 2X]` lying in a prescribed class modulo `P` is `X/P + O(1)`, so a
condition modulo `P` is equidistributed only while `P <= X`. The event "island `i` is open at `q`"
is a condition modulo `P_q = prod_{7 < g <= q} g = e^{theta(q) - theta(7)}`. At `X = 64000` that
modulus is about `10^{55,000}` against a band of 17,068 integers: **every class contains 0 or 1
machines, and density says nothing about which.** The pre-registered core is the exact measure of
the equidistributed part:

| `X` | gears above 7 up to `2X` | the core (product `<= X`) | product | all gear pairs | pairs with `g h <= 2X` | fraction | `ln(2X)/theta(2X)` |
|---|---|---|---|---|---|---|---|
| 1,000 | 299 | `{11,13}` | 143 | 44,551 | 145 | 3.26e-03 | 3.93e-03 |
| 2,000 | 546 | `{11,13}` | 143 | 148,785 | 333 | 2.24e-03 | 2.12e-03 |
| 4,000 | 1,003 | `{11,13,17}` | 2,431 | 502,503 | 739 | 1.47e-03 | 1.14e-03 |
| 8,000 | 1,858 | `{11,13,17}` | 2,431 | 1,725,153 | 1,572 | 9.11e-04 | 6.10e-04 |
| 16,000 | 3,428 | `{11,13,17}` | 2,431 | 5,873,878 | 3,302 | 5.62e-04 | 3.28e-04 |
| 32,000 | 6,409 | `{11,13,17}` | 2,431 | 20,534,436 | 6,755 | 3.29e-04 | 1.74e-04 |
| 64,000 | 11,983 | `{11,13,17,19}` | 46,189 | 71,790,153 | 13,749 | 1.92e-04 | **9.22e-05** |

**M8 CONFIRMED and sharpened.** The core is 2, 2, 3, 3, 3, 3, 4 gears - four out of 11,983 at the
top band. The brief's split by gear pairs `g h <= X` is the more generous version and is still
`1.9e-4` of all pairs at `X = 64000`, falling like `1/X`. The equidistributed fraction of the
sifting itself, `ln(2X)/theta(2X)`, is `9.2e-5` and falls like `ln X / X`.

### 5.2 The size of the non-equidistributed part, measured

There is no way to isolate "the contribution of the pairs with `g h > X`" as a separate positive
quantity, because the open-island event is a single product over all gears, not a sum over gear
pairs. What CAN be measured, and is, is the total discrepancy between the CRT prediction and the
truth: it is the ratio column of 3.3. **The whole of the fluctuation variance is
non-equidistributed in the rigorous sense, and the whole of it is nevertheless predicted by the
CRT model to within 5% at `X = 64000`, 17% at worst over the seven bands, with the error
shrinking.** The honest summary: the model is right and unprovable, which is exactly face D.

The trivial counting bound for the non-equidistributed part is vacuous by a margin that can be
written down. The inclusion-exclusion (Legendre) expansion of the second moment over subsets of
the gear set has `2^{pi(2X) - 4}` terms, each carrying an `O(1)` counting error; at `X = 64000`
that is `2^{11983} ~ 10^{3607}` against a main term of order `X = 6.4 x 10^4`. This is the same
"+1 per class" obstruction as the parent's, at `10^{3600}` instead of `10^{24}`.

---

## 6. The theorem attempt, and why it stops (items 4, 5)

### 6.1 What a theorem would need

Chebyshev needs `sum_q (N - mu*)^2` bounded ABOVE and `mu*` bounded BELOW, both a priori. The
second is where it stops, and it stops before the second moment is reached:

```
    sum_{q in A} N(q)  =  # { (q, i) : q in [X,2X], i = 12 mod 35, i < d(q),
                                       q^2 + 6i - 2 and q^2 + 6i both prime }
```

by 4.3 - a count of twin prime pairs in the intervals `(q^2, (q+1)^2)`. A lower bound for it is a
lower-bound sieve of dimension 2 at sifting level `s = 2`, which is the parity obstruction (face A
of `the_wall.md`, and Iwaniec's exponent 4.27 in branch 3a). **M9 CONFIRMED: the blocking term is
the first moment, not any second-moment error.** The upper bound on `sum N^2` is available in
principle (an upper-bound sieve of dimension 4) but is never reached, because the chain is already
broken.

### 6.2 The statement is not weaker than the target - a proof

> **N-M6.** Let `F(X)` be the number of `q = 5 (mod 6)`, `gcd(q, 7) = 1`, in `[X, 2X]` with no open
> island of class `12 (mod 35)` in `[1, (q+1)/3)`. If `F(X) < |A''(X)|` for infinitely many `X` -
> that is, if ANY bound strictly below 1 on the failing FRACTION holds for infinitely many bands -
> then there are infinitely many twin primes.
>
> Proof. `F(X) < |A''(X)|` gives a `q` in `[X, 2X]` with an open island `i`. By 4.3 the pair
> `(q^2 + 6i - 2, q^2 + 6i)` lies in `(q^2, (q+1)^2)`, is coprime to 2, 3, 5, 7 by the class of `i`
> and to every gear in `(7, q]` by openness, and both members are below `(q+1)^2`, hence both are
> prime; they differ by 2. Letting `X` run over the infinitely many bands gives infinitely many
> twin prime pairs, all distinct because `q^2 -> infinity`. []

So `B(X) -> 0` is not a weakening of the target: **it implies the twin prime conjecture.** So does
`B(X) < 1` infinitely often; so does `B(X) <= 1 - epsilon`. The chain "vanishing fraction, then
never" has no first link that is easier than the last. Thin place 5 asked for a theorem-shaped
statement weaker than "all `q`"; the honest answer is that on this object there is no such
statement, because the object's failure set is what a twin prime is.

The measured `B(X) = 11.4 (ln X)^2 / X` is therefore a heuristic evaluation of a quantity that,
if proved at any nontrivial size, would settle twin primes. What it does say is how much slack a
proof would have: at `X = 64000` the mean of the certified object is 32.0 open twin pairs per
machine with standard deviation 7.3, so a failing machine would sit 4.4 standard deviations below
the mean - and none of the 7,315 machines in that band is even close.

### 6.3 The smallest improvement, if the first moment were granted

If the first moment were given (say by assuming Hardy-Littlewood for twins on average over
`q` in `[X, 2X]`, which is exactly what 3.1 measures to 2%), then the second moment as computed
here is enough: `B(X) = 11.4 (ln X)^2 / X -> 0`. The improvements that would sharpen the rate,
in order of cost:
1. the third moment, or a Janson/Suen inequality on the island indicators, which would replace the
   Chebyshev `1/mu` by an exponential `e^{-c mu}`; the pairwise data of section 2 is already the
   input Suen needs, and the correlation is NEGATIVE (`sum C = -0.15 mu`), which is the good
   direction;
2. a large sieve over `q` for the big-gear pairs, which would not help: the pairs with `g h > X`
   are 99.98% of all pairs at `X = 64000` and the large sieve is silent when the modulus exceeds
   the range.
Neither is worth opening while 6.2 stands.

---

## 7. Toward the root (item 6)

In the machine's vocabulary the branch's measured statement is:

> for all but a fraction `B(X) = 11.4 (ln X)^2 / X` of the machines whose top gear lies in
> `[X, 2X]`, an island past the square is open - and on the certified object the number of open
> islands per machine is 32.0 on average at `X = 64000` with standard deviation 7.3, never once
> zero in the 13,295 machines above `q = 11663`.

What would be needed to convert it to "never" is not an improvement of the moment method. It is
face D in its exact form: the open-island event is a condition on `q^2` modulo
`prod_{7 < g <= q} g ~ 10^{55,000}` while `q` runs over `10^4` integers, so every residue class
holds 0 or 1 machines and no equidistribution statement - not Bombieri-Vinogradov, not
Elliott-Halberstam, not the large sieve - reaches a modulus above `q^2`. The second moment does
not soften that: it needs the same first moment, which is itself a twin count (6.1), and its
conclusion is itself of the target's strength (6.2).

The one thing the branch adds toward the root is negative and useful: **thin place 5 is closed.**
Moments over `q` are an excellent instrument for measuring the object - they reproduce it to 5% -
and they are not a route, because the weakened statement they produce is not weaker.

---

## 8. What holds without exception (item 7)

| statement | range | exceptions |
|---|---|---|
| `o_g(i,j)` in `{0, 2, 4}`, never odd, and `o_g > 0` only if `g` divides `delta`, `3 delta - 1` or `3 delta + 1` | every gear 11..2000 and every ordered residue pair mod `g`: 359,712,683 cells | **0** |
| the joint-density formula equals the full-period density | 630 island pairs over gear sets `{11,13}`, `{11,13,17}`, `{11,13,17,19}` | **0** (worst 1.11e-16) |
| `Var = mu + sum_{i!=j} rho_2 - mu^2` equals the full-period variance | 4 complete periods, up to 46,189 classes and 80 islands | **0** (worst 5.4e-12) |
| `Var_model(q) < mu(q)` - the CRT count is sub-Poisson | 42 sampled `q`, `q = 1097 .. 121493` | **0** |
| `Var_model(q)/mu(q)` larger on the short arc than on the long arc, band by band | 7 bands, 21 short/long comparisons | **0** |
| a free island of class `12 (mod 35)` exists in `[1, d)` | every `q` coprime to 30 in `(11921, 128000]`: 30,955 machines | **0** |
| the same, for `q` coprime to 210 | every such `q` in `(11663, 128000]`: 26,591 machines | **0** |
| the same, on the twin-certified object (`q = 5 mod 6`, coprime to 7, arc `(q+1)/3`) | 13,295 machines above 11,663 | **0** |
| `B(X)` exceeds the actual failing fraction | 7 bands (an identity, recorded for the margin: 2.6x to 85x) | **0** |

The complete failure list of the one-class witness over `[1000, 128000]` is 104 machines, the last
seven being 5129, 5429, 5477, 6341, 10403, 11663, 11921. This extends N-I2 (class 12 alone works
from `q = 5477`, tested on primes) to all integers: the threshold over the integers coprime to 30
is `q = 11921` and over the integers coprime to 210 is `q = 11663`.

---

## 9. What is new

Screened against `docs/novel/README.md` (in particular the lines `reachability-landscape`,
`island-witness-integers`, `covering-hierarchy-exactness`, `moment-degree-ceiling`), against
`research/proof/cover_number.md`, `square_vector.md`, `separability.md` and `the_wall.md`. The
register has the island witness, the doubling law, the first moment and the `s = 2` correction; it
has no pair statistic of the island process, no second moment over `q`, and no statement about the
strength of the vanishing-fraction target.

* **N-M1 (the overlap law).** The gears that correlate two islands at separation `delta` are
  exactly the prime factors above 7 of `delta`, `3 delta - 1` and `3 delta + 1`, and the overlap
  count is 0, 2 or 4 - never odd (0 exceptions in 359,712,683 cells over 299 gears). Prior art in
  one line: the divisor condition is the letter-gear rule of `separability.md`; new is that it is
  the exact support of the island pair correlation and that the doubling law forces the value even.
* **N-M2 (the exact pairwise joint density, verified).** `rho_2(i,j) = prod_g (1 - a_i - a_j +
  o_g/(g-1))`, matching the full-period density to 1.1e-16 over three gear sets, and the derived
  variance matching the full-period variance to 5.4e-12.
* **N-M3 (the island count is sub-Poisson, and by how much).** `Var_model/mu` lies in
  `[0.7625, 0.8147]` at all 42 sampled `q`, splitting by arc into 0.802-0.815 (short) and
  0.762-0.793 (long). The deficit is exactly `1 - (opening density) - |sum C|/mu` with
  `|sum C|/mu = 0.14` (short), `0.155` (long), stable over a factor 4 in `m`. Mechanism, measured
  at a number: a generic gear costs `-4.05/g^2` (predicted `-4/g^2`), the divisor family `g | delta`
  repays `+2/g^2` and the two separation families `g | 3 delta -+ 1` repay `+2/g^2`; the repayment
  is 89-94% while `g < m`, **exactly 50.0% at `g = m`** (where the divisor family runs out of
  separations) and 32-39% above. The residual negative correlation sits on the separations whose
  smallest correlating gear is large, and the only POSITIVE separations are those correlated by
  gear 11.
* **N-M4 (the band moments, and why raw Chebyshev cannot work).** Exact `E[N]`, `Var[N]` and the
  failure count for every `q` coprime to 30 in `[1000, 128000]`. `Var/E^2` tends to a constant near
  0.145 because 85% of the variance is the systematic run of `mu(q)` across a band; the normalised
  bound `B(X) = 11.4 (ln X)^2/X` (14.2 on the twin-certified object), flat to 6% over a 64-fold
  range. The `s = 2` handicap `4 e^{-2 gamma}` confirmed to 2% (0.3% off the multiples of 7) on
  33,868 machines.
* **N-M5 (the one-class threshold over the integers).** The last failure of the `i = 12 (mod 35)`
  witness is `q = 11921` over the integers coprime to 30 and `q = 11663` over those coprime to 210;
  0 exceptions in the 30,955 and 26,591 machines above. Also: for `7 | q` the one-class object is
  degenerate - gear 7 divides the lower member at every island of the class, and the open count
  runs 11.9% above the model by the cofactor effect `ln(q^2)/ln(q^2/7)`.
* **N-M6 (the vanishing-fraction target is not weaker - a proof).** Any bound strictly below 1 on
  the failing fraction of the twin-certified object, holding for infinitely many bands, implies the
  twin prime conjecture. Closes thin place 5.
* Filed, not claimed: the balanced-semiprime measurement of section 10.

---

## 10. Verdict

**FACT for the measurements, DEAD for the route.**

The second moment over `q` was asked for as a way from "rare" to "a vanishing fraction of `q`
fail". It gives, exactly and cheaply, everything a second moment can give: the pairwise joint
density in closed form with its overlap law verified on 3.6e8 cells, the variance of the open-island
count computed exactly at 42 machines and measured exactly at 33,868, the discovery that the count
is sub-Poisson at a stable ratio 0.79 to 0.81 with the mechanism resolved gear by gear, and the
normalised Chebyshev bound `B(X) = 11.4 (ln X)^2/X` falling by half per doubling. The CRT model
reproduces the real fluctuation to 5% at the largest band and is improving.

And the route is dead, for two reasons that are independent and each fatal. The first moment is
itself a count of twin primes in `(q^2, (q+1)^2)`, so nothing in the chain can be started (M9).
And the conclusion is not weaker than the target: any bound strictly below 1 on the failing
fraction implies the twin prime conjecture (N-M6). Thin place 5 said "a theorem-shaped statement no
branch has yet written down"; the statement can be written down, and writing it down shows it is
the target wearing a fraction.

What survives and where it goes: the exact pair law (N-M1, N-M2) is a new handle on the island
process that is not a rate and not a count - it says WHICH gears couple two islands - and the one
thing on the tree it feeds is face A's opening ("proofs that use which residues"). The sub-Poisson
mechanism (N-M3) says the machine's own arithmetic makes openings MORE evenly spread than chance,
which is the good direction for the root and is the first measured statement of that kind. Neither
is a route by itself; both are inputs to the adversarial covering line, where the same coincidence
families (`g | delta`, `g | 3 delta -+ 1`) are what makes covering harder than counting.

---

## 11. Dead ends and closed clues (do not re-enter)

* **Thin place 5 (moments over `q` as a weaker target).** Closed by N-M6. The failing set of the
  object IS the set of squares with no twin pair above them; any nontrivial statement about its
  density is the twin prime conjecture.
* **The first moment as the easy half.** Refuted: `sum_q N(q)` is itself a twin count in short
  intervals above squares. Every moment method on this object needs it.
* **The pairwise-correlation cancellation as exact.** Refuted (2.3): the coincidence families repay
  89-94% of the disjointness deficit below `g = m` and only 32-50% above it. The count is genuinely
  sub-Poisson, by 19-24%.
* **`N` as Poisson.** Refuted (2.5): `Var_model/mu` never exceeded 0.815 at any of 42 sampled `q`.
* **Raw Chebyshev over a band.** Dead: `Var/E^2` tends to 0.145, not 0, because the band mixes
  short and long arcs and a factor 2 in `q`. Only the per-`q` normalised form falls.
* **The large sieve for the big-gear pairs.** Not opened, and stated why: the pairs with
  `g h > 2X` are 99.98% of all gear pairs at `X = 64000` and the moduli exceed the range by
  `10^{55,000}`; the large sieve is silent above the range.
* **Twin-prime products `q = p(p+2)` as bad machines.** Opened because the two largest failures of
  the one-class witness are `10403 = 101 x 103` and `11663 = 107 x 109`, and the 16 such `q` in the
  sweep have `sum N / sum mu* = 0.862` against 1.018 for every other `q`. Closed at scale: over
  `q <= 10^7` the 81 twin products give `sum N/sum mu-hat = 0.79241`, the 88 cousin products
  `p(p+4)` give 0.79362, and 441 balanced semiprimes `p x nextprime(p+50)` give 0.79295 - all three
  families identical to 0.15%. Against 972 nearest-neighbour controls matched on `q mod 6` the three
  looked 1.3% low (`z = -2.62, -2.49, -7.36`); rerunning with the controls also required coprime to
  7 - which the three families are automatically, having both factors above 7 - the control ratio
  falls to **0.79283** and every family lands on it (`z = -0.10, +0.21, +0.09`). The whole apparent
  deficit was the control group's own multiples of 7 (section 1.1). Small-sample artefact at the
  16 values, a control artefact at 10^7; there is no twin-product effect.
