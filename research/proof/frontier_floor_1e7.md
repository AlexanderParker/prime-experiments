# The frontier floor at every prime cut to 10^7 (lead U3)

Prover, 2026-09-10. Parent: research/proof/tree_review.md section 2.1(d) and its lead U3, and
research/proof/lengthen_never_precede.md (laws E6-E8, sections 3(d)-(f) and 5). The brief: E8
carries the floor constant c = 4.625 from one cut to the next provided the run that straddles the
cut satisfies `x_s >= c L_s` whenever `L_s >= d_0`; the review says the floor extends to every cut
below 10^7 if two computable quantities behave at every cut - the straddling run's ratio
`x_s / L_s` and the top-run share `tau <= 0.0833`. Both are computed here, exactly, at every prime
cut to 10^7. Scripts: `research/stack/r6/straddle_scan.py` (the scan), `straddle_report.py`,
`tau_prefix.py`; outputs in `research/stack/r6/results/` (gitignored; every number used is in this
text). One core, 91 s, peak under 300 MB. Nothing is committed. Laws are numbered E10 onward
(E9 is issued in ends_or_middles.md).

## The finding, first

**The floor carries.** At every prime cut `p` in `[7, 10^7]` - 664,576 cuts, `p` from 7 to
9,999,991, the full range with no memory limit reached - the straddling run's ratio `x_s / L_s` was
computed exactly. The frontier's hypothesis `L_s >= d_0` bites at **exactly 24 cuts, the last at
p = 487**, and it is the same 24 cuts that the measured range to 20,011 already had: **no new
biting cut appears anywhere between 20,011 and 10^7.** At 23 of the 24 the ratio is at least 4.75,
minimum **4.7500 at p = 11**; at the 20 cuts with `p >= 23` (the range in which the floor 4.625 is
claimed) the minimum is **6.7273 at p = 31**. The one cut below 4.625 is `p = 7`, ratio 4.0000,
whose own constant on the staircase is 3.250, not 4.625 (lengthen_never_precede section 3(d)); it
is not a new brick.

**Pre-registered expectation** (from the brief): the floor carries, minimum ratio near 4.75 as in
the measured range. **CONFIRMED, on the number**: 4.7500, at `p = 11`, and it is the same minimiser.

**The second quantity, tau, is confirmed with four orders of margin.** Over every cut whose lower prime `q` is
at or above 23, `max tau = 0.083333`, attained at the single cut `p = 31` and nowhere else - the
identical maximum the measured range had. Above the old reach the share collapses: over the
586,080 cuts with `p` in `[10^6, 10^7]` the maximum is `4.64 x 10^-9`. The review's sufficient
condition `(1 - tau) W >= 4.625 (tau W + L_1 + 1)` fails at exactly two cuts, `p = 7` and `p = 11`,
and holds at all 664,574 others; among `p >= 41` its tightest instance is `p = 53` at
`LHS/RHS = 3.48`, i.e. with a factor 3.5 of slack.

**What this buys.** By E8 (`c <= 6.25`, `p >= 118`) plus the base `H_4.625(q)` on `q = 23..19,997`,
`H_4.625` now holds at every cut to 10^7: 500 times the range of the previous measurement, from a
sieve rather than a frontier scan. What it does not buy: nothing is proved above 10^7. The
straddling condition remains ROOT in face E's sense - it is a two-sided gap bound at a prime square
and asks strictly more than the existence statement at that cut.

## 1. The exact objects

Column `k` is the pair `(6k - 1, 6k + 1)`. At the cut `p` (a prime; `q = prevprime(p)` is the prime
below it, and the composite machine acting there is the engine `{5..p}`) the **square column** is
`W = (p^2 - 1)/6`, the column whose upper member is `p^2`. Under `{5..p}` that column is struck by
`p` itself, so a blocked run runs through it: the **straddling run**. By reduction (R) the openings
of `{5..p}` below the top of its prefix are the twin columns with lower member above `p`, so with

    t_0 = the last twin lower member below p^2,     k_0 = (t_0 + 1)/6
    t_1 = the first twin lower member above p^2,    k_1 = (t_1 + 1)/6

the straddling run is `(x_s, L_s) = (k_0 + 1, k_1 - k_0 - 1)`. Two parts of it are named
separately: `L_top = W - k_0`, the part at or below the square column, and `L_1 = k_1 - W`, the
first-twin offset above `p^2` in columns. `d_0(p)`, the first opening of `{5..p}`, is the column of
the first twin whose lower member exceeds `p` - the first-twin distance; the frontier's hypothesis
reaches the straddling run only when `L_s >= d_0(p)`.

**E10 (the straddling run's exact decomposition).** `L_s = L_top + L_1 - 1` at every one of the
664,576 cuts, with no exception. (The review's 2.1(d) writes `L_s = L_0 + L_1 + 1`; the correct
bookkeeping is `L_top + L_1 - 1`, a shift of two columns. It changes no conclusion there - the
review's algebra is an inequality with a factor of 3.5 of slack - but the identity is what the
scan uses and it is exact, not approximate.)

The **top-run share** is `tau = L_top' / (W - q//6)`, where `L_top'` is the run of the SMALLER
machine `{5..q}` ending at the top of its prefix and the denominator is that prefix's column count
(the columns `k` with `6k - 1 > q`, up to `W`). This is valve_existence.md table 2's own
definition and it differs from `L_top` in one case that matters: under `{5..q}` the column `W` is
open when `p^2 - 2` is prime, because the only divisor of `p^2` is `p` and `p` is not a gear of
`{5..q}` - reduction (R)'s clause "plus `W(q)` when `q'^2 - 2` is prime". So

    L_top' = 0                if p^2 - 2 is prime,        L_top' = L_top = W - k_0 otherwise.

That happens at **74,911 of the 664,576 cuts (11.27 %)**, which is the density `3/ln(p^2)` for a
number that is odd and, since `p^2 = 1 (mod 3)`, never divisible by 3. The straddling run itself is
untouched by this: under `{5..p}` the column `W` is always struck.

## 2. Method, and the check that it is the same object

For each cut a small sieve of the columns around `W` by the primes `5 <= g < 1000` (column `k` is
struck by `g` iff `k = +-6^{-1} mod g`), then `gmpy2.is_prime` on the survivors outward from `W` in
both directions until a twin is met on each side; the half-width starts at 512 columns and doubles
if either side is empty. The sieve only prefilters - every reported twin is tested exactly - and a
gear is prevented from striking the member that IS the gear. 664,576 cuts in 91 s on one core.

The scan reproduces, independently and exactly, every published number of the two documents it
extends:

| quantity | on record | this scan |
|---|---|---|
| biting cuts to 20,011, and the last one | 24, last `q' = 487` (lengthen 3(e)) | 24, last `p = 487` |
| min `x_s/L_s` at the biting cuts | 4.75 at `q' = 11`, `>= 5.36` from 19 (lengthen 5.2) | 4.7500 at `p = 11`; 5.3636 at 19 |
| min `x_s/L_s` over cuts `>= 23` | 6.727 at `q' = 31`, run `(148, 22)` (lengthen 3(e)) | 6.7273 at `p = 31`, `(148, 22)` |
| min ratio by band: 7-100 / 100-300 / 1000-3000 / 10000-20011 | 4.00 (7) / 49.65 (157) / 1,434 (1231) / 65,802 (10589) | 4.0000 (7) / 49.646 (157) / 1,434.0 (1231) / 65,802 (10589) |
| max `L_s` over `q' <= 20,011` | 484 (lengthen 5.2) | 484 |
| max `tau`, `q` in 23-100 | 0.0833 at `q = 29` (valve_existence table 2) | 0.083333 at `q = 29` (`p = 31`) |
| `tau` bands 100-300 / 300-1000 / 1000-3000 / 3000-10^4 / 10^4-19997 | 0.0123 (109) / 0.00275 (337) / 0.000273 (1123) / 0.000077 (3041) / 0.000009 (10691) | 0.012322 (109) / 0.0027484 (337) / 0.00027326 (1123) / 0.000077475 (3041) / 0.0000088949 (10691) |
| max `L_top'` in those bands | 13 / 51 / 62 / 156 / 195 / 386 | 13 / 51 / 62 / 156 / 195 / 386 |
| cuts with `L_top' = 0` in those bands | 7/17, 13/37, 26/106, 58/262, 149/799, 195/1033 | 7/17, 13/37, 26/106, 58/262, 149/799, 195/1033 |
| first-twin offset against the long arc, `q <= 10^7` | 0 exceptions, max fraction 0.7714 at `q = 53` (research/stack/r3) | 0 exceptions in 664,576, max 0.7714 at `p = 53` |

Table 2's "`L_top = 0`" column is the one place where reading `L_top` as the twin gap would have
been wrong; it is the `p^2 - 2` prime case of section 1, and getting those six counts right is the
evidence that the two scans are measuring the same thing.

## 3. The straddling run to 10^7

**The 24 cuts where the condition bites** (`L_s >= d_0`), complete:

| `p` | `x_s` | `L_s` | `d_0` | `x_s/L_s` | | `p` | `x_s` | `L_s` | `d_0` | `x_s/L_s` |
|---|---|---|---|---|---|---|---|---|---|---|
| 7 | 8 | 2 | 2 | **4.000** | | 73 | 881 | 22 | 17 | 40.045 |
| 11 | 19 | 4 | 3 | **4.750** | | 101 | 1,691 | 21 | 18 | 80.524 |
| 13 | 26 | 4 | 3 | 6.500 | | 103 | 1,756 | 29 | 18 | 60.552 |
| 19 | 59 | 11 | 5 | 5.364 | | 113 | 2,103 | 34 | 23 | 61.853 |
| 23 | 88 | 7 | 5 | 12.571 | | 131 | 2,839 | 26 | 23 | 109.192 |
| 31 | 148 | 22 | 7 | **6.727** | | 137 | 3,091 | 61 | 25 | 50.672 |
| 37 | 221 | 17 | 7 | 13.000 | | 139 | 3,203 | 27 | 25 | 118.630 |
| 43 | 299 | 13 | 10 | 23.000 | | 157 | 4,071 | 82 | 30 | 49.646 |
| 47 | 358 | 15 | 10 | 23.867 | | 163 | 4,378 | 69 | 30 | 63.449 |
| 53 | 468 | 27 | 10 | 17.333 | | 199 | 6,586 | 52 | 38 | 126.654 |
| 61 | 613 | 15 | 12 | 40.867 | | 233 | 9,003 | 64 | 40 | 140.672 |
| | | | | | | 347 | 20,014 | 101 | 70 | 198.158 |
| | | | | | | 487 | 39,528 | 87 | 87 | 454.345 |

Exactly one is below 4.625 (`p = 7`, at 4.000, where the staircase's own constant is 3.250);
`p = 11` and `p = 19` are the only others below 6.727.

**The whole range, by band of the cut:**

| band of `p` | cuts | min `x_s/L_s` (at `p`) | max `L_s` | max `L_1` | max `tau` (at `p`) |
|---|---|---|---|---|---|
| 7 - 100 | 22 | 4.000 (7) | 27 | 27 | 0.083333 (31) |
| 100 - 1,000 | 143 | 49.65 (157) | 102 | 87 | 0.012322 (113) |
| 1,000 - 10^4 | 1,061 | 1,434 (1231) | 444 | 402 | 0.00027326 (1129) |
| 10^4 - 10^5 | 8,363 | 6.58 x 10^4 (10589) | 636 | 522 | 8.8949 x 10^-6 (10709) |
| 10^5 - 10^6 | 68,906 | 3.04 x 10^6 (106907) | 1,162 | 1,097 | 3.1814 x 10^-7 (106907) |
| 10^6 - 10^7 | 586,081 | 1.52 x 10^8 (1253089) | 1,822 | 1,683 | 4.6414 x 10^-9 (1010567) |

Largest straddling run anywhere in the range: `L_s = 1,822` at `p = 8,675,573`. Largest first-twin
offset: `L_1 = 1,683` columns at `p = 1,253,089`. Median `L_s` over the 664,576 cuts: 188 columns.

**E11 (the floor at every prime cut to 10^7; measured, 0 exceptions).** For every prime cut `p` in
`[11, 10^7]` the straddling run of `{5..p}` satisfies `x_s >= 4.625 L_s` whenever `L_s >= d_0(p)`.
The hypothesis holds at 23 cuts, all in `[11, 487]`, and at each the ratio is at least 4.75. At
`p = 7` the hypothesis holds and the ratio is 4.000, consistent with that cut's own staircase value
3.250. Refuted by one cut in the range with `L_s >= d_0` and `x_s < 4.625 L_s`.

## 4. Why the condition stops biting, and how far it is from biting again

This is the mechanism, not a trend. `d_0(p)` is the column of the first twin above `p`, so
`d_0 = p/6` up to the first-twin distance: over the 664,576 cuts the median of `d_0/(p/6)` is
1.0000 exactly, and `d_0` runs from 2 to 1,666,690. `L_s` is a twin gap at height `p^2`, so it is
of order `ln^2(p^2)` in numbers, i.e. of order `(2 ln p)^2 / (6 x 2C_2)` columns: median `L_s` is
1.66 times that quantity, and its largest observed value anywhere below 10^7 is 1,822. So the
condition `L_s >= d_0` needs a twin gap across `p^2` of at least `p/6` columns, which is a demand
growing linearly in `p` set against a supply growing like `ln^2 p`. The two cross once, near
`p = 500`, and never again in the range:

- the last cut at which they meet is `p = 487`, where `L_s = d_0 = 87` exactly;
- above 487 the largest value of `L_s / d_0` anywhere below 10^7 is **0.8263, at `p = 1,231`**;
- in the whole range only 79 cuts reach `L_s >= d_0/2` and only 189 reach `L_s >= d_0/4` (the 24
  biting cuts are among them), out of 664,576.

So the frontier's hypothesis is not "nearly" satisfied at large cuts and then rescued by the
ratio; it is comfortably false, and the ratio is astronomically large where it is false
(`1.5 x 10^8` at the worst cut above 10^6). The floor's whole content at large cuts lives in the
inherited runs and the interior section runs, which E7 and E8-a handle without measurement.

## 5. The review's 2.1(d) condition, checked exactly

The review derives, from `L_s = L_top + L_1 + 1` and `x_s = W - L_top`, the sufficient condition

    (1 - tau) W >= 4.625 (tau W + L_1 + 1),

and argues it holds for every `p >= 41` from `tau <= 0.0833` and `L_1 < (2p + 1)/3`. Checked cut by
cut with the exact `tau` and `L_1`:

- **failures: 2, at `p = 7` and `p = 11`.** It holds at all 664,574 other cuts, including every
  `p >= 13`.
- among `p >= 41` the tightest instance is `p = 53`, where `LHS/RHS = 3.4799`; the condition is
  never within a factor 3.4 of failing anywhere above 41.
- in its reduced form (`tau <= 0.0833` substituted, leaving `L_1 + 1 <= 0.13292 W`) there are 3
  failures, all with `p <= 13`; from `p = 17` on the reduced form alone suffices.
- the long-arc input is confirmed independently: `L_1 < (2p + 1)/3` at **all 664,576 cuts, 0
  exceptions**, the largest fraction being 0.7714 at `p = 53` - the same maximiser and the same
  number the r3 first-twin scan found with a different method (sympy per candidate rather than a
  column sieve), which is a genuine cross-check of that scan, not a re-use of it.

The review's algebra is therefore sound and its margin is real, with one correction: its
`L_s = L_0 + L_1 + 1` should be `L_top + L_1 - 1` (E10), which makes the condition slightly easier,
not harder.

## 6. tau beyond the old reach

Table 2 of valve_existence.md stopped at `q = 19,997`. Its continuation, computed here in the same
sense:

| band of `q` | cuts | max `tau` (at the cut `p`) | max `L_top'` | cuts with `L_top' = 0` | median `tau` |
|---|---|---|---|---|---|
| 20,011 - 10^5 | 7,330 | 3.1321 x 10^-6 (20,389) | 491 | 1,141 | 5.7 x 10^-8 |
| 10^5 - 10^6 | 68,906 | 3.1814 x 10^-7 (106,907) | 1,122 | 8,953 | 1.0 x 10^-9 |
| 10^6 - 10^7 | 586,080 | 4.6414 x 10^-9 (1,010,567) | 1,593 | 64,366 | 1.5 x 10^-11 |

Two cuts in the whole range have `tau > 0.0833`, both below 13 (`p = 11` at 0.1053 and `p = 7`);
over cuts whose `q` is at or above 23 the maximum is 0.083333 and it is attained once, at `q = 29`. Against the
frontier's own ceiling `1/(c + 1) = 0.1778` at `c = 4.625`, the slack at the worst cut in the whole
range above `q = 23` is a factor 2.1 - unchanged from the measured range, because the worst cut is
the same one.

## 7. What this settles and what it does not

Settled, at the reach `p <= 10^7`:

1. E8's proviso is satisfied at every cut where it is not vacuous, and it is vacuous at 664,552 of
   the 664,576 cuts. With the base `H_4.625(q)` on `q = 23..19,997` and E8-a (interior section runs
   never below 6.25 for `p >= 118`, and checked directly below), `H_4.625(p)` holds at every prime
   cut to 10^7.
2. The review's two computable quantities behave as it guessed, with the margins now measured: the
   ratio's minimum where it bites is 4.75, and `tau`'s maximum above `q = 23` is 0.0833, both
   attained at cuts below 32 and never approached again.
3. The r3 first-twin scan's arc bound is independently confirmed by a second method, and its
   extreme case (`p = 53`, fraction 0.7714) is the same.

Not settled, and the honest statement of the gap:

- Nothing here is a proof above 10^7. Both inputs are measurements of twin gaps at prime squares.
  The straddling condition is exactly "a twin gap across a prime square that is longer than the
  prime is at most 21.6 % of its start", which by lengthen_never_precede 5.2 implies a twin in
  `(p^2, 1.216 p^2 + p]` at every cut, hence infinitely many twins: it is the target statement in a
  stronger, two-sided local form, and no part of this scan weakens that.
- The scan does show where a counterexample would have to live: a cut with a twin gap across `p^2`
  of at least `p/6` columns. Below 10^7 the largest such gap reaches 83 % of `p/6` once (at
  `p = 1,231`) and 0.03 % of it at `p = 10^7`. Any future search for a failure of E8's proviso
  should be a search for an unusually long twin gap at a prime square with `p` small, not a scan of
  large cuts.

For the tree: node R4.c.iii.a's lead U3 is answered, verdict FACT with the reach stated (the floor
carries to 10^7, 0 exceptions above `p = 7`, 24 biting cuts unchanged from the measured range);
node R4.d.i.a keeps its ROOT mark, since the straddling condition itself is untouched. The review's
sentence "the first-twin scan already extends the frontier floor 4.625 to every [cut] to 10^7
PROVIDED tau stays below 0.0833 there" is confirmed with the proviso now measured rather than
assumed.
