# Branch R2.a.i.a.1 - THE ISLAND WITNESS UNDER PRESSURE

Parent: node R2.a.i.a (the reachability landscape, `research/proof/reachability.md`). The
observation that spawned this branch: the parent's central positive result **N-R4** - for every
prime `q` in `(1487, 20000]` some offset `i = 5, 10, 12` or `17 (mod 35)` with `1 <= i < d` is
struck by no gear of `{5..q}` - together with its 17 failures below 1,487, which the parent listed
but did not take apart.

The theory this branch tests: the witness is an object of the two quadratics, not of the prime
`q`; it survives being pushed (further in `q`, off the primes, up in `B`); and what it costs to
defeat it - the minimum number of gears that must cooperate - is the quantity that decides whether
the witness can be turned into a proof.

Scripts: `research/anchor235/r40/iw_sweep.py`, `iw_failures.py`, `iw_cover.py`, `iw_adv.py`,
`iw_slack.py`, `iw_class.py`.
Result outputs (untracked): `research/anchor235/r40/results/iw_*.txt`. Every number this document
relies on is written into the document.

---

## 0. Pre-registered (written before any computation of this branch)

### 0.1 Definitions, exact

**The object, for a prime (the parent's N-R4).** Fix a prime `q >= 5`; the machine is
`M = {5..q}` (all primes up to `q`). `k_0 = (q^2 - 1)/6` is the column of `q^2`; the column at
**offset** `i` is `k_0 + i` and carries the two members `q^2 + 6i - 2` and `q^2 + 6i`. Gear `g`
**strikes** offset `i` iff

```
    q^2 = 2 - 6i  (mod g)      or      q^2 = -6i  (mod g),
```

equivalently `i = (2 - q^2) u_g` or `i = -q^2 u_g (mod g)` with `u_g = 6^{-1} (mod g)`. An offset
struck by no gear of the machine is an **opening**. `d = 2 u_q (mod q)` is the distance from the
`q^2` column to the top gear's next tooth (`(q+1)/3` if `q = 5 mod 6` - the **short arc**;
`(2q+1)/3` if `q = 1 mod 6` - the **long arc**). An **island for bound `B`** is an offset that no
gear `5 <= g <= B` can strike at any `q` at all (the quadratic-residue bar): for `B = 7` exactly
`i = 5, 10, 12, 17 (mod 35)`; for `B = 11` twelve classes mod 385; for `B = 13` forty-eight
classes mod 5,005. **The witness holds at `q`** iff some island for bound `B` in `[1, d)` is an
opening; it **fails** iff every island in `[1, d)` is struck.

**The all-integers version (item 1), exact.** Let `q >= 5` be any integer with `gcd(q, 6) = 1`
(prime or not). Then `q^2 = 1 (mod 24)`, so `k_0 = (q^2 - 1)/6` is an integer, and `6^{-1} (mod q)`
exists, so `d = 2 u_q (mod q)` is defined by the same formula. The machine is still
`M = {5..q}` = the primes up to `q`, and gear `g` strikes offset `i` by the same congruence. Every
definition above carries over verbatim; the only thing lost is that `q` itself need not be a gear
and that `q^2 (mod g)` may be `0` for a gear `g | q`. Nothing else in the object refers to the
primality of `q`.

For `q` even or `3 | q` the object does not exist: `6 | q^2 - 1` iff `gcd(q, 6) = 1`, so `q^2` is
not adjacent to a column at all and `6^{-1} (mod q)` does not exist. The natural surrogate is
stated and tested in section 3.5.

### 0.2 What would count as a rule

As the parent: a statement about positions or residues with an exact exception count over a stated
range, uniform in `q`. A density, a fitted curve, an average, or a restatement of the tooth rule,
the bar, the island CRT system, the doubling law or N-R5 (large gears strike islands at `2/g`) is
**not** a finding. If a sub-question reduces to quadratic reciprocity or to a Mertens/Merten-type
sum it is named in one line as classical and stopped.

### 0.3 Predictions, with numbers, and what refutes each

**Item 1 - all integers.**

* **W1 (the divisor-gear mechanism; the sharp answer to "is primality irrelevant").** If a gear
  `g` divides `q` then `q^2 = 0 (mod g)`, so `g`'s two targets are `x = 0` and `x = -2`; both make
  a member `= 0 (mod g)`, and `0` is not a nonzero quadratic residue, so **a gear dividing `q`
  strikes exactly two of its own barred classes** - offsets that for every prime `q` it can never
  reach. Consequence, computed by hand before any run: gear 5's barred classes are `i = 0, 2
  (mod 5)` and the four `B = 7` islands are `5, 10, 12, 17`, i.e. `0, 0, 2, 2 (mod 5)`. So **if
  `5 | q` gear 5 strikes every island class and the witness fails at every such `q`, at every
  size.** Predicted: 0 integers `q = 0 (mod 5)` with a free `B = 7` island, over the whole sweep.
  REFUTED by one. Likewise `7 | q` kills exactly the two island classes `5, 12 (mod 35)` (both
  `= 5 mod 7`) and leaves `10, 17`, so multiples of 7 fail more often than average but not always.
* **W2 (off the multiples of 5, primality is irrelevant).** For integers `q` coprime to 30, the
  witness holds from some point on and composites behave like primes: predicted the largest
  failing `q` coprime to 30 in the sweep is below 20,000, and the failure rate of composites
  coprime to 30 in a `q` band matches the primes' in that band to within a factor of 2. REFUTED by
  a failure coprime to 30 above 20,000, or by a composite/prime failure-rate ratio outside
  `[0.5, 2]` in a band above 5,000.
* **W3 (prime powers).** `q = p^k` is the `g = p` case of W1: the witness fails at every power of
  5 and behaves like a generic integer at powers of 7, 11, 13, ... . Predicted 0 free islands at
  `q = 25, 125, 625, 3125, 15625`.

**Item 2 - the 17 failures.**

* **W4 (the short arc).** The brief pre-registers "all 17 have `d` in the short arc
  (`2 u_q < q/2`)". **My number, computed by hand from the residues: 16 of 17.** `q = 73` is
  `1 (mod 6)` and therefore sits in the long arc `d = (2q+1)/3 = 49`; every other failure is
  `5 (mod 6)`. REFUTED (in the brief's direction) if all 17 are short; refuted (in mine) if the
  count is not exactly 16.
* **W5 (the failures are cheap covers).** The exact minimum number of gears needed to cover every
  island in `[1, d)` at each of the 17 failures is small and grows slowly with `d`: predicted
  `<= 6` at every one of the 17, and the optimum uses only gears below 100 at 15 of the 17.
  REFUTED by an optimum above 6.
* **W6 (the mechanism is the short arc, not a residue coincidence).** Predicted: the 17 failures
  are explained by `d` being small - the number of islands in `[1, d)` at a failure is below 60 at
  all 17 - and not by any shared residue of `q` modulo a small gear; specifically, the failures'
  residues `q mod 11`, `q mod 13` are predicted to be spread (no class holding more than 5 of the
  17). REFUTED by a class holding 7 or more.

**Item 3 - the minimum blocking set.**

* **W7 (inside the machine it does not exist).** For every prime `q` in `(1487, 20000]` a free
  island exists (N-R4), so **no** subset of `(7, q]` covers the islands of `[1, d)`: the minimum
  blocking set is `+infinity` at every such `q`. That is N-R4 restated and is not a finding; the
  non-vacuous questions are the two below.
* **W8 (the adversarial blocking set is bounded - the scale-free wall again).** Give each gear
  `g > 7` its full freedom: it may choose any nonzero quadratic residue `r` for `q^2 (mod g)`,
  which places its two strike classes at `-r u_g` and `(2 - r) u_g`, i.e. **two classes mod `g` at
  the fixed separation `d_g = 2 u_g`**. The minimum number of gears that can then cover every
  island of `[1, d)` is predicted **bounded in `q`**, between 8 and 16 for every `d` tested,
  because each gear's coverage scales with the island count exactly as N-R5 says (a gear covers
  about `2m/g` of the `m` islands, and `sum_{7 < g <= G} 2/g` first reaches 1 at `G ~ 50`).
  REFUTED if the exact adversarial minimum passes 20 and keeps climbing with `d`.
* **W9 (the real cost of defeating the witness grows).** With the real phases, the number of extra
  gears beyond `q` that would be needed - each large gear covering at most 2 offsets of `[1, d)` -
  is at least `ceil(free/2)`, and `free` grows with `q` (0, 0, 0, 4, 12 by band in the parent).
  Predicted min free island count over `q` in `(50000, 100000]` exceeds 25. REFUTED by a prime in
  that band with fewer than 13 free islands.

**Item 4 - the slack law.**

* **W10.** Open islands per `q` grow like `(4d/35) prod_{7 < g <= q} (1 - 2/g)`, i.e. linearly in
  `q` divided by `(ln q)^2`; the minimum by band is the quantity of interest and is predicted
  strictly increasing over the bands `(5k, 10k], (10k, 20k], (20k, 50k], (50k, 100k]`. REFUTED by
  a band whose minimum falls.
* **W11 (which gears do the work).** Predicted: the small gears 11, 13, 17 remove the most islands
  (rate `2/g`), and they are also the commonest **sole** strikers (the gear whose removal would
  free an island), because sole-striker counts scale as `2/g` too. So there is no "large gears by
  position" effect. REFUTED if a gear above 100 is a sole striker more often than gear 11.
* **W12 (free islands are enriched at higher `B`).** Among free `B = 7` islands, the fraction that
  are also `B = 11` islands exceeds the base rate `12/44 = 0.2727`, because a `B = 11` island has
  two fewer gears able to strike it. Predicted measured above 0.30. REFUTED below 0.28.

**Item 5 - `B = 11` and `B = 13`.**

* **W13 (the nesting direction in the brief is backwards, and provably).** `S_13 ⊆ S_11 ⊆ S_7`
  (an island for a larger bound is barred by strictly more gears), so `[1, d)` contains **fewer**
  islands at larger `B` and failure is **easier**: `Fail_7 ⊆ Fail_11 ⊆ Fail_13`. The brief's
  direction ("a `q` failing at `B = 13` fails at `B = 11` and 7") is predicted **false**, with
  many counterexamples; the true nesting is the reverse and is forced by set inclusion in one
  line, so it is a gate, not a finding. Predicted 0 exceptions to `Fail_7 ⊆ Fail_11 ⊆ Fail_13`.
* **W14 (the higher bounds keep failing).** The parent's last `B = 11` failure is `q = 9281` and
  its last `B = 13` failure `q = 18839`, both near the top of its sweep. Predicted: extending to
  100,000 produces **no** new `B = 11` failure above 9,281 but **at least three** new `B = 13`
  failures above 18,839. REFUTED either way.
* **W15 (N-R4 extends).** Predicted 0 exceptions to the `B = 7` witness for primes in
  `(1487, 100000]` - a five-fold extension of the parent's certified range. REFUTED by one prime.

**Item 6 - toward the root.**

* **W16.** The smallest statement is predicted to be unchanged in form by the all-integers result
  but changed in hypothesis: it must acquire the condition `gcd(q, 35) = 1` (from W1), which for a
  prime `q > 7` is free. So the object is about the two quadratics **plus** the condition that the
  square `q^2` is a nonzero residue at gears 5 and 7 - the primality of `q` enters only there.

**Item 7.**

* **W17.** Report everything that holds for every `q` in the sweep without exception, with counts.

### 0.4 Scorecard

| # | prediction | verdict and evidence |
|---|---|---|
| W1 | a gear dividing `q` strikes only its own barred classes; every `q = 0 (mod 5)` fails | **CONFIRMED**. Divisor rule exact by `g mod 8`, 0 exceptions in 301 gears; **13,333 of 13,333** multiples of 5 fail (2.1, 2.2) |
| W2 | off multiples of 5, composites behave like primes; last failure below 20,000 | **CONFIRMED**. Largest failure coprime to 35 is `q = 1649` (composite); 0 failures coprime to 210 above 1,649 in 45,338 integers; composite/prime failure rates 0.0722 vs 0.0629 and 0.0035 vs 0.0040 (2.2) |
| W3 | powers of 5 all fail | **CONFIRMED**. All six powers of 5 below 200,000 fail; of 110 prime powers only `25, 49, 121, 125, 625, 3125, 15625, 78125` fail (2.2) |
| W4 | 16 of 17 failures in the short arc, `q = 73` the exception | **CONFIRMED exactly, brief REFUTED**: 16 of 17 in the short arc, the exception `q = 73` (3) |
| W5 | exact min cover `<= 6` at all 17 | **REFUTED**. Exact minima run to **24** (`q = 1487`) and 25 (`q = 1649`); above 6 at 10 of the 21 (3) |
| W6 | small `d`, no shared residue class holding 7 or more | **CONFIRMED**. Islands at a failure never exceed 57 (64 with composites); largest residue class 3 of 17 at `q mod 11, 13, 17, 19` (3) |
| W7 | no blocking set inside the machine above 1,487 | **CONFIRMED** (gate, not a finding): no blocking set exists inside `(7, q]` at any prime above 1,487 (4.1) |
| W8 | adversarial blocking set bounded, 8..16 | **REFUTED**. `K(d)` is exact 3, 4, 6, 9, 14, 20 at `d = 35..1120` and at least 21 at 2,240 - unbounded - while the counting bound stalls at 2, 4, 5, 7, 9, 10 (4.3) |
| W9 | min free above 25 in `(50000, 100000]` | **CONFIRMED**. Minimum free islands in `(50000, 100000]` is **57** (`q = 52553`) (5.1) |
| W10 | min free strictly increasing by band | **CONFIRMED**. 2, 4, 12, 21, 57, 107 over the six bands, strictly increasing (5.1) |
| W11 | gear 11 the commonest sole striker; no large-gear position effect | **CONFIRMED as stated** (gear 11 the commonest sole striker, 0.0710; no gear above 100 above 0.008) **but qualified**: 53.7% of all sole strikes come from gears above 100 collectively (5.2) |
| W12 | free islands enriched on `B = 11` islands above 0.30 | **CONFIRMED**. 0.3231 against base 0.2727 (ratio 1.185, matching `(1-2/11)^-1 (1-2/13)^-1`) (5.3) |
| W13 | nesting is `Fail_7 subset Fail_11 subset Fail_13`, brief's direction false | **CONFIRMED**. 0 exceptions to `Fail_7` in `Fail_11` in `Fail_13` over 17,982 primes; the brief's direction fails at 224 primes (6) |
| W14 | no new `B = 11` failure; at least 3 new `B = 13` failures | **CONFIRMED on both halves**. No `B = 11` failure above 9,281; five new `B = 13` failures above 18,839, largest 33,623 (6) |
| W15 | 0 exceptions for the `B = 7` witness to 100,000 | **CONFIRMED and exceeded**: 0 exceptions for primes in `(1487, 200000]` - 17,748 primes, a tenfold extension (2.2, 8) |
| W16 | the statement gains `gcd(q, 35) = 1` and nothing else | **CONFIRMED with a correction**: the hypothesis is `gcd(q, 5) = 1`, not `gcd(q, 35) = 1` - multiples of 7 fail only 9 times, the last at 2,849 (2.1, 7.3) |
| W17 | exception-free statements | ten, listed in section 8 |

---

## 1. Setup (exact ranges)

No sampling anywhere except where a row says so. The sweep was run to `q = 200000` rather than
the pre-registered 100,000, since it cost 32 seconds. Scripts in `research/anchor235/r40/`; the island
residue systems are reused from `research/anchor235/r39/results/rl_isl_*.npy` (the parent's exact
CRT sets).

| object | range | script |
|---|---|---|
| free islands in `[1, d)` at `B = 7, 11, 13` | **every** integer `q` coprime to 6 in `[5, 200000]` - 66,666 values, of which 17,982 prime | `iw_sweep.py` |
| the failures taken apart: islands, strikers, exact minimum cover (ILP, HiGHS, proved optimal) | all 21 failing `q` coprime to 35 and all 17 prime failures | `iw_failures.py` |
| minimum blocking set of the **struck** islands, exact ILP | every prime `q <= 200`, then every 37th prime to 20,000 (146 machines) | `iw_cover.py` |
| `K(d)`, the adversarial cover number, exact ILP over the complete candidate list | `d = 35, 70, 140, 280, 560, 1120` exact; `d = 2240` bounded | `iw_adv.py` |
| island strikes and sole strikes per gear; the position of the first free island | every prime `q <= 12000` (1,436 machines, 1.16 million island strikes) | `iw_slack.py` |
| the divisor rule against `g mod 8` | every gear `5 <= g <= 2000` (301 gears), exhaustive over the offset classes | `iw_slack.py` |
| the witness restricted to one island class; the shortest arc that still works | every prime `q <= 200000` (17,982) | `iw_class.py` |

## 2. Results - the witness over all integers (item 1)

### 2.1 The divisor rule: what a gear that divides `q` does

If a gear `g` divides `q` then `q^2 = 0 (mod g)`, so `g`'s two targets are `x = 0` and `x = -2`
and it strikes exactly the two offset classes

```
    i = 0  (mod g)          [the upper member q^2 + 6i is 0 mod g]
    i = 2 u_g  (mod g)      [the lower member q^2 + 6i - 2 is 0 mod g]
```

Neither target is a *nonzero* square, so these are classes the gear could never reach at a `q`
coprime to it - **exactly when the corresponding character says so**. Class `0` is a barred class
iff `chi_g(2) = -1`; class `2 u_g` is barred iff `chi_g(-2) = -1`. By `g mod 8` (Gauss's second
supplement, named once as the classical input):

| `g mod 8` | 1 | 3 | 5 | 7 |
|---|---|---|---|---|
| how many of the two divisor classes are barred classes of `g` | 0 | 1 | **2** | 1 |
| gears `5..2000` measured | 68 | 76 | 79 | 78 |
| exceptions | 0 | 0 | 0 | 0 |

**301 gears, 0 exceptions.** The case that matters is `g = 5`, which is `5 (mod 8)`: **both** of
gear 5's divisor classes are barred, so when `5 | q` gear 5 strikes precisely
`Bar(5) = {0, 2} (mod 5)`. The four `B = 7` islands `5, 10, 12, 17` are `0, 0, 2, 2 (mod 5)`.
Hence:

> **N-I1 (the multiple-of-five law).** If `5 | q` then gear 5 strikes **every** island class, so
> no `B = 7` island of `[1, d)` is ever free and the witness fails - at every size, with no
> exception. Measured: **13,333 of 13,333** multiples of 5 coprime to 6 up to 200,000 fail.

Gear 7 is `7 (mod 8)`, so only one of its divisor classes is barred: when `7 | q` gear 7 strikes
`i = 0, 5 (mod 7)`, the islands are `5, 3, 5, 3 (mod 7)`, and it kills the two island classes
`5, 12 (mod 35)` while leaving `10, 17`. Multiples of 7 therefore fail more often at small `q` but
not systematically: **9 failures, the largest `q = 2849`, in 7,619 values.**

### 2.2 The witness for every integer coprime to 6

| class of `q` | integers in `[5, 200000]` | failures | largest failure |
|---|---|---|---|
| all `q` coprime to 6 | 66,666 | 13,364 | 199,985 |
| `5 \| q` | 13,333 | **13,333 (all)** | 199,985 |
| `7 \| q`, `5` not | 7,619 | 9 | 2,849 |
| coprime to 35 | 45,714 | 22 | 1,649 |

The 22 failures coprime to 35 are

```
   11, 17, 23, 29, 41, 53, 73, 113, 121, 137, 173, 197, 233, 247, 263,
   341, 353, 461, 683, 1151, 1487, 1649
```

- the parent's 17 primes (`q = 11` has no island in `[1, d)` at all), plus **four composites**:
`121 = 11^2`, `247 = 13 x 19`, `341 = 11 x 31`, `1649 = 17 x 97`. Above `q = 1649` there is not a
single failure coprime to 35 in 45,338 integers; above `q = 2849` not one coprime to 30 in 52,574.

Primes and composites behave alike where the small gears allow it (W2):

| `q` band | primes coprime to 30 | fail | composites coprime to 30 | fail |
|---|---|---|---|---|
| 5 - 100 | 22 | 8 | 3 | 3 |
| 100 - 1,000 | 143 | 9 (0.0629) | 97 | 7 (0.0722) |
| 1,000 - 5,000 | 501 | 2 (0.0040) | 567 | 2 (0.0035) |
| 5,000 - 20,000 | 1,593 | 0 | 2,407 | 0 |
| 20,000 - 200,000 | 15,722 | 0 | 30,880 | 0 |

Prime powers (W3): of 110 prime powers coprime to 6 below 200,000, exactly **8 fail** -
`25, 125, 625, 3125, 15625, 78125` (every power of 5, by N-I1) and the two small generic cases
`49` and `121`. Every power of 7, 11, 13, ... above 121 has a free island.

**The answer to the brief's question, sharply.** The witness is *not* a statement about the two
quadratics alone: it is **false for every `q` divisible by 5**, at every size, for an exact
reason. It *is* a statement about `q^2 + 6i - 2` and `q^2 + 6i` **plus the single hypothesis
`gcd(q, 5) = 1`** - equivalently, that `q^2` is a *nonzero* square mod 5. Primality of `q` is used
nowhere else: composites coprime to 30 satisfy it at exactly the primes' rate and stop failing at
the same place.

### 2.3 `q` even, or divisible by 3

The object does not exist there, and the two requirements coincide: `k_0 = (q^2 - 1)/6` is an
integer iff `gcd(q, 6) = 1`, and `6^{-1} (mod q)` (hence `d`) exists iff `gcd(q, 6) = 1`. So the
column frame and the tooth arc appear and disappear together; there is no version of the object
for even `q` in which one half survives. The natural surrogate - keep the machine `{5..q}` and
start at the column of `q'^2` for the largest `q' <= q` coprime to 6 - is just the object at `q'`,
and every such `q'` is inside the sweep above.

## 3. Results - the 17 failures taken apart (item 2)

Exact minimum covers by ILP (HiGHS, proved optimal). `d` is the arc, `m` the islands in `[1, d)`.

| `q` | prime | `q mod 6` | arc | `d` | `m` | **min cover** | one optimal cover |
|---|---|---|---|---|---|---|---|
| 17 | yes | 5 | short | 6 | 1 | 1 | 11 |
| 23 | yes | 5 | short | 8 | 1 | 1 | 13 |
| 29 | yes | 5 | short | 10 | 1 | 1 | 13 |
| 41 | yes | 5 | short | 14 | 3 | 3 | 17, 29, 37 |
| 53 | yes | 5 | short | 18 | 4 | 4 | 17, 41, 43, 47 |
| 73 | yes | **1** | **long** | 49 | 7 | 5 | 11, 17, 19, 61, 71 |
| 113 | yes | 5 | short | 38 | 4 | 4 | 37, 61, 67, 101 |
| 121 | no | 1 | long | 81 | 10 | 8 | 13, 17, 19, 23, 43, 47, 61, 79 |
| 137 | yes | 5 | short | 46 | 6 | 5 | 11, 67, 79, 83, 113 |
| 173 | yes | 5 | short | 58 | 8 | 8 | 11, 17, 29, 59, 97, 101, 131, 157 |
| 197 | yes | 5 | short | 66 | 8 | 5 | 13, 17, 23, 47, 71 |
| 233 | yes | 5 | short | 78 | 9 | 7 | 11, 19, 29, 71, 89, 137, 139 |
| 247 | no | 1 | long | 165 | 20 | 11 | 11, 13, 17, 23, 29, 31, 41, 43, 79, 83, 167 |
| 263 | yes | 5 | short | 88 | 12 | 8 | 13, 17, 23, 37, 41, 43, 113, 227 |
| 341 | no | 5 | short | 114 | 13 | 12 | 13, 23, 29, 31, 59, 113, 181, 229, ... |
| 353 | yes | 5 | short | 118 | 15 | 12 | 13, 19, 41, 53, 59, 67, 79, 107, ... |
| 461 | yes | 5 | short | 154 | 19 | 15 | 11, 17, 19, 29, 31, 47, 79, 101, ... |
| 683 | yes | 5 | short | 228 | 28 | 18 | 11, 13, 19, 23, 31, 53, 59, 61, ... |
| 1151 | yes | 5 | short | 384 | 44 | 22 | 11, 13, 17, 23, 29, 31, 37, 43, ... |
| **1487** | yes | 5 | short | 496 | 57 | **24** | 11, 13, 17, 19, 23, 37, 41, 47, ..., 1409, 1453 |
| 1649 | no | 5 | short | 550 | 64 | 25 | 11, 13, 17, 19, 23, 29, 31, 53, ..., 1259, 1601 |

**The short arc: 16 of the 17 prime failures are `q = 5 (mod 6)`** and the one exception is
`q = 73`, exactly as pre-registered (W4). The brief's "all 17" is refuted. The reason is
elementary and worth one line: `q = 5 (mod 6)` gives `d = (q+1)/3`, half the arc of
`q = 1 (mod 6)`, so a short-arc machine carries half as many islands to defend.

**The mechanism is not a few small gears.** The minimum cover is a large fraction of the islands
and the fraction is not falling towards zero: `24/57 = 0.42` at `q = 1487`, then `22/44 = 0.50`,
`18/28 = 0.64`, `15/19 = 0.79` going down. Most gears in an optimal cover strike exactly **one**
island. W5 (cover `<= 6`) is refuted at 10 of the 21: the covers run to 24 and 25 gears.

**The failures are fragile.** At 20 of the 21 failures some island has exactly one striker, so
deleting a single gear from the machine would free an island: 19 such islands at `q = 1487`, 15 at
`q = 1151`, 10 at `q = 683`. The one exception is `q = 29`, whose single island `i = 5` is struck
by both 11 and 13.

**No residue coincidence** (W6 confirmed). Over the 17 prime failures the largest class is 3 of 17
at `q mod 11`, `q mod 13`, `q mod 17`, `q mod 19`, and 2 of 17 from `q mod 23` on. The failures
share nothing but a small `d`: the island count `m` at a failure never exceeds 57 (64 including
composites), while a prime of the same size that does not fail carries the same `m` and simply has
one island left over.

The smallest striker of an island, pooled over the 17 failures (227 islands): gear 11 takes
0.1674, gear 17 0.1145, gear 13 0.1101, gear 19 0.0881 - i.e. `2/g` thinned by the gears below,
the parent's N-R5 rate seen on a tiny sample.

## 4. Results - the blocking sets (item 3)

### 4.1 Inside the machine there is none

For a prime `q > 1487` a free island exists (section 5), and a free island is by definition struck
by no gear of `(7, q]`. So **the minimum blocking set of all the islands of `[1, d)` inside the
machine does not exist**, at every prime above 1,487 in the sweep. That is N-R4 restated; it is
recorded as the gate it is (W7) and the two non-vacuous quantities are below.

### 4.2 The blocking set of the *struck* islands, exact

How concentrated is the covering work that the machine does do? Minimum number of gears of
`(7, q]` accounting for every island of `[1, d)` that is struck (exact ILP, proved optimal):

| `q` | `d` | islands | free | struck | **MBS** | MBS/struck |
|---|---|---|---|---|---|---|
| 127 | 85 | 11 | 2 | 9 | 5 | 0.556 |
| 2,017 | 1,345 | 155 | 9 | 146 | 51 | 0.349 |
| 5,051 | 1,684 | 192 | 15 | 177 | 49 | 0.277 |
| 10,009 | 6,673 | 764 | 39 | 725 | 126 | 0.174 |
| 15,289 | 10,193 | 1,165 | 52 | 1,113 | 196 | 0.176 |
| 19,699 | 13,133 | 1,501 | 65 | 1,436 | 220 | 0.153 |

The MBS grows without bound - roughly `0.17 x` the island count, i.e. about `0.02 d` - so the
covering work is spread over a number of gears growing linearly in the arc. It is **not** a
handful of small gears doing the job: at `q = 19699` it takes 220 gears to account for the
strikes, of which only 21 are below 100.

### 4.3 `K(d)`: how many gears would have to cooperate to defeat the witness

This is the `q`-free version and the one that bears on the root. Give every gear `g > 7` its full
freedom - any nonzero quadratic residue `r` for `q^2 (mod g)`, which is exactly the freedom a real
`q` has - putting its two strike classes at `(2 - r) u_g` and `-r u_g`, two classes mod `g` at the
fixed separation `d_g = 2 u_g`; each gear may be used at one phase. `K(d)` is the fewest gears
that can then strike **every** island of `[1, d)`.

*Method and certificate.* Writing `a = (2-r) u_g`, `b = -r u_g` gives `6(a - b) = 2 (mod g)`. A
covered set of size `>= 3` needs two islands in one class mod `g`, so `g < d`; a covered set of
size 2 taken from the two different classes needs `g | 3(i - j) - 1` for two islands of `[1, d)`,
so `g <= 3d`; a covered set of size 1 is available for every island at infinitely many gears above
`3d`. Enumerating every gear `11 <= g <= 3d + 2` at every nonzero-QR phase, plus one generic
singleton per island, therefore enumerates **every set the adversary can play**, and the ILP over
that list (with the once-per-gear constraint) is exact, certified optimal by HiGHS.

| `d` | islands `m` | candidate sets | **`K(d)`** | status | counting lower bound | an optimal cover |
|---|---|---|---|---|---|---|
| 35 | 4 | 3 | **3** | exact | 2 | 11, 37, + one singleton |
| 70 | 8 | 20 | **4** | exact | 4 | 11, 23, 37, 71 |
| 140 | 16 | 58 | **6** | exact | 5 | 11, 13, 17, 23, 37, 127 |
| 280 | 32 | 228 | **9** | exact | 7 | 11, 13, 17, 19, 29, 53, 59, 61, 263 |
| 560 | 64 | 719 | **14** | exact | 9 | 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 67, 101, 109, 751 |
| 1,120 | 128 | 2,086 | **20** | exact | 10 | 11 .. 61, then 73, 89, 101, 107, 157, 421 |
| 2,240 | 256 | 6,221 | **21 .. 31** | ILP bounds | 11 | 11 .. 211, then 617, 683 |

**W8 is refuted, and in the direction that matters.** The counting lower bound - the one the
parent's N-R5 controls, "how many gears does it take for the strike budget to reach the island
count" - is 2, 4, 5, 7, 9, 10, 11: it grows like `log d` and is on its way to the constant that
`sum_{7 < g <= G} 2/g = 1` fixes (about a dozen gears, whatever `d`). The **actual** cover number
is 3, 4, 6, 9, 14, 20 and at least 21: it is double the counting bound by `d = 1120` and the gap
widens at every step. Counting says the adversary needs about a dozen gears; the geometry says it
needs `K(d)`, and `K(d)` grows with the arc.

### 4.4 What that forces on `q` (item 6's second half)

A failure at `q` needs at least `K(d)` distinct gears each in one of at most four prescribed
classes, so it **pins `q` modulo the product of those gears**, and the smallest product available
is the product of the `K(d)` smallest gears above 7:

| `d` | `K(d)` | smallest possible modulus | `q` at that arc (long / short) | modulus / `q` |
|---|---|---|---|---|
| 35 | 3 | 2,431 | 52 / 104 | 47 |
| 70 | 4 | 46,189 | 104 / 209 | 444 |
| 140 | 6 | 3.08e7 | 209 / 419 | 1.5e5 |
| 280 | 9 | 1.45e12 | 419 / 839 | 3.5e9 |
| 560 | 14 | 5.59e20 | 839 / 1,679 | 6.7e17 |
| 1,120 | 20 | 1.13e32 | 1,679 / 3,359 | 6.7e28 |

The modulus of the failure condition outgrows `q` itself super-exponentially. Below `d = 70` the
modulus is comparable with `q` and failures are common; from `d = 280` on the modulus exceeds `q`
by nine orders of magnitude and more, so a failure is not a residue class that `q` can fall into -
it is a single coincidence per astronomically long period. That is the sharpest structural reason
this branch found for why the witness holds, and it is a statement about the gears, not a density.

## 5. Results - the slack law (item 4)

### 5.1 The slack grows, and its minimum grows

Free `B = 7` islands in `[1, d)`, primes only:

| `q` band | primes | **min free** | at `q` | median | max | min islands |
|---|---|---|---|---|---|---|
| 1,487 - 5,000 | 433 | **2** | 1,787 | 12 | 35 | 57 |
| 5,000 - 10,000 | 560 | **4** | 5,309 | 23 | 51 | 192 |
| 10,000 - 20,000 | 1,033 | **12** | 15,161 | 38 | 87 | 382 |
| 20,000 - 50,000 | 2,871 | **21** | 20,369 | 75 | 180 | 764 |
| 50,000 - 100,000 | 4,459 | **57** | 52,553 | 143 | 302 | 1,907 |
| 100,000 - 200,000 | 8,392 | **107** | 102,317 | 253 | 528 | 3,812 |

The minimum is strictly increasing over every band (W9, W10 confirmed): the witness is never close
to failing again. The free fraction `free/islands` falls slowly (0.0503, 0.0368, 0.0327 over the
last three bands) - that is `prod_{7 < g <= q}(1 - 2/g)` with the singular-series correction, a
rate, stopped.

### 5.2 Which gears remove the islands

Over every prime `q <= 12000` (1,436 machines, 1,163,342 island strikes, 102,494 islands with
exactly one striker):

| gear | island strikes | share | `(2/g)/sum(2/g)` | **sole strikes** | share |
|---|---|---|---|---|---|
| 5, 7 | **0** | 0 | - | **0** | 0 |
| 11 | 83,881 | 0.0721 | 0.0545 | 7,272 | **0.0710** |
| 13 | 70,424 | 0.0605 | 0.0461 | 5,959 | 0.0581 |
| 17 | 54,197 | 0.0466 | 0.0352 | 4,482 | 0.0437 |
| 19 | 48,397 | 0.0416 | 0.0315 | 3,987 | 0.0389 |
| 23 | 40,004 | 0.0344 | 0.0261 | 3,227 | 0.0315 |
| 29 | 31,582 | 0.0271 | 0.0207 | 2,484 | 0.0242 |

Gear 11 is the commonest striker and the commonest **sole** striker, and no gear above 100 comes
near it (the largest single share above 100 is under 0.008). W11 is confirmed as stated - there is
no "large gears by position" effect at the level of a single gear. But the aggregate says the
opposite of what the brief's phrasing suggests: **53.7% of all sole strikes are made by gears
above 100 and 22.8% by gears above 1,000**, spread over a thousand gears each contributing a
fraction of a percent. The last island standing is normally the private business of one large
gear; it is just never the *same* large gear. (The largest gear ever a sole striker in the sweep
is 11,909.)

### 5.3 The free islands sit on higher-`B` islands more often than chance

Over 9,605,439 free `B = 7` islands (all `q` coprime to 6 to 200,000):

| | measured | base rate inside `S_7` | ratio |
|---|---|---|---|
| the free island is also a `B = 11` island | **0.3231** | `12/44 = 0.2727` | 1.185 |
| the free island is also a `B = 13` island | **0.1132** | `48/572 = 0.0839` | 1.349 |

W12 confirmed. The mechanism is the parent's bar read one level up. A `B = 11` island is a
`B = 7` island that gear 11 is additionally barred from; by the parent's N-R6 the mean strike rate
of gear 11 over the islands is exactly `2/11`, so removing it multiplies the survival chance by
`(1 - 2/11)^{-1} = 11/9 = 1.222`. A `B = 13` island removes gear 13 as well:
`(1 - 2/11)^{-1}(1 - 2/13)^{-1} = 13/9 = 1.444`. Measured 1.185 and 1.349 - **0.97 and 0.93 of the
independent-gear prediction**, the same slight shortfall the parent found at the landing. Order
one, no residue left over.

## 6. Results - `B = 11` and `B = 13`, and the nesting (item 5)

| `B` | primes with no island in `[1,d)` | failures (primes `<= 200000`) | largest failure | failures above 20,000 |
|---|---|---|---|---|
| 7 | 3 (`q = 5, 7, 11`) | **17** | **1,487** | **0** |
| 11 | 7 (`q <= 29`) | 73 | **9,281** | 0 |
| 13 | 7 (`q <= 29`) | 237 | **33,623** | 5 (22511, 27437, 28433, 31247, 33623) |

W14 is confirmed on both halves: extending from 20,000 to 200,000 adds **no** `B = 11` failure
above 9,281 (the last two are 5,261 and 9,281) and adds **five** `B = 13` failures above 18,839,
the largest 33,623. The thresholds are 1,487 / 9,281 / 33,623 - each about `4-6x` the last, while
the island density falls by `3-4x` each step. Raising `B` buys sparsity and pays for it in
threshold, exactly as the parent's dead end said.

**The nesting is the reverse of the brief's.** `S_13` inside `S_11` inside `S_7` (an island for a
larger bound is barred by strictly more gears), so `[1, d)` holds **fewer** islands at larger `B`
and failure is **easier**:

```
    Fail_7  inside  Fail_11  inside  Fail_13         0 exceptions in 17,982 primes
```

and the brief's direction ("a `q` failing at `B = 13` fails at `B = 11` and 7") fails at **224**
primes for `B = 7` and **164** for `B = 11`. The true nesting is forced by set inclusion in one
line and is recorded as a gate, not a finding (W13).

## 7. Results - squeezing the witness, and the smallest statement (items 6, 7)

### 7.1 The witness set can be one class instead of four

The witness set `{5, 10, 12, 17} (mod 35)` has density `4/35`. Each class **on its own** carries a
free island for every prime above its own threshold (primes to 200,000):

| class mod 35 | primes with no free island of that class | largest such `q` | above 20,000 |
|---|---|---|---|
| 5 | 116 | **7,109** | 0 |
| 10 | 129 | **11,717** | 0 |
| 12 | 146 | **5,477** | 0 |
| 17 | 126 | **13,001** | 0 |

> **N-I2 (the one-class witness).** For every prime `q` in `(13001, 200000]` **each** of the four
> island classes separately carries a free island in `[1, d)`: 0 exceptions in 16,436 primes. For
> the single class `i = 12 (mod 35)` the threshold is `q = 5477`, with 0 exceptions in 17,261
> primes. The witness set can therefore be taken to be **one arithmetic progression of density
> `1/35`**, and above 13,001 all four work at once.

Two classes together (density `2/35`) are exception-free from `q = 3467` (`{5, 10}`; only 1,787
and 3,467 fail above 1,487).

### 7.2 The arc can be a seventh of `d`

The witness asks for a free island in `[1, d)`. Where does the first one actually sit?

| `q` band | primes | max (first free island)/`d` | at `q` | median | max absolute offset |
|---|---|---|---|---|---|
| 1,487 - 5,000 | 433 | 0.6704 | 1,619 | 0.0555 | 635 |
| 5,000 - 20,000 | 1,593 | 0.4598 | 5,591 | 0.0182 | 957 |
| 20,000 - 50,000 | 2,871 | **0.1516** | 26,513 | 0.0085 | 1,370 |
| 50,000 - 100,000 | 4,459 | **0.1125** | 50,723 | 0.0046 | 2,145 |
| 100,000 - 200,000 | 8,392 | **0.0446** | 103,919 | 0.0025 | 2,392 |

> **N-I3 (the short-arc witness).** For every prime `q` in `(20000, 200000]` a free island sits
> inside `[1, 0.152 d)`: 0 exceptions in 15,722 primes; and the first free island's absolute
> offset never exceeds **2,392** anywhere in the sweep, against an arc `d` reaching 133,331. The
> witness does not merely bound the walk by `d`, it bounds it by a seventh of `d` and in absolute
> terms by a few thousand columns.

(Since a free island is an opening and `L` is the first opening, `L <=` the first free island
always; that is the gate the parent recorded, and the table gives it teeth.)

### 7.3 The smallest thing that would have to be proved

The parent's statement was

> for every prime `q` there is an `i = 5, 10, 12, 17 (mod 35)` with `1 <= i < d` such that for
> every gear `g` in `(7, q]` and every root `s` of `-6i` or `2 - 6i` mod `g`, `q != +- s (mod g)`.

This branch changes it in three ways and leaves its shape alone:

1. **The hypothesis is `gcd(q, 5) = 1`, not primality.** Section 2 proves the "only if": the
   statement is false for every multiple of 5, because a gear dividing `q` strikes the classes
   where a member vanishes and gear 5's two such classes are exactly `Bar(5)`, which contains all
   four island classes. It is true for composites coprime to 30 at exactly the primes' rate. So
   the object is about the two quadratics `q^2 + 6i - 2` and `q^2 + 6i` **plus** the condition
   that `q^2` is a *nonzero* square mod 5 - and nothing else about `q`.
2. **The witness set shrinks to `i = 12 (mod 35)`** (density `1/35`), and the range shrinks to
   `1 <= i < d/6`. Both are measured with 0 exceptions above their thresholds.
3. **What defeating it costs is now quantified.** A failure needs `K(d)` gears to cooperate,
   `K(d) = 3, 4, 6, 9, 14, 20` at `d = 35 .. 1120` exactly, against a counting requirement of
   `2, 4, 5, 7, 9, 10`; and those gears pin `q` modulo their product, which is `1.1e32` at
   `d = 1120` where `q` is about `3 x 10^3`.

So the smallest statement, in the machine's terms, is now:

> Fix the class `i = 12 (mod 35)`. For every integer `q` with `gcd(q, 30) = 1` and `q > 5477`
> there is an `i = 12 (mod 35)` with `1 <= i < d` such that no gear `g` in `(7, q]` has
> `q = +- s (mod g)` for a root `s` of `-6i` or of `2 - 6i`.

and what would have to happen for it to fail is that at least `K(d)` gears above 7 hit prescribed
classes simultaneously - that is, that `q` lands in one of finitely many classes modulo a number
that exceeds `q` by many orders of magnitude.

## 8. Mechanism, and what holds without exception

**Where the primality of `q` actually enters.** Nowhere except at gear 5, and there absolutely. A
gear that divides `q` does not lose its strikes - it *relocates* them onto the two classes at
which one member vanishes, and those are classes the gear can never reach otherwise. Whether they
are barred classes is decided by `chi_g(2)` and `chi_g(-2)`, i.e. by `g mod 8`; gear 5 is the one
gear that is `5 (mod 8)` and small enough for its barred pair to be the whole island system. That
is why the witness is a theorem about two quadratics with one coprimality condition rather than
about primes - and why the condition cannot be dropped.

**Why the failures are all small.** The 17 prime failures are the machines whose arc is too short
to hold enough islands: 16 of them are `q = 5 (mod 6)` (half the arc), the island count at a
failure never passes 57, and the minimum cover is a large fraction of the islands - the failures
are not efficient covers by small gears, they are a row of large gears each taking exactly one
island. That is why they die out: as `d` grows the number of islands grows linearly while the
number of gears able to take two or more of them does not.

**Where the counting wall does and does not bite.** The parent's N-R5 is exact and this branch
does not touch it: large gears strike islands at exactly `2/g`, so strikes/islands is `sum 2/g` at
every `B` and no counting argument through islands can force a free island. But the exact cover
numbers say counting is not what the adversary faces. To strike **every** island the adversary
must solve a covering problem, and its cost `K(d)` grows - 3, 4, 6, 9, 14, 20 against a counting
requirement of 2, 4, 5, 7, 9, 10 - because the two classes a gear may place are at a *fixed*
separation `d_g = 2 x 6^{-1}` and cannot be tuned to the island pattern. That is the one place in
this branch where the machine's own arithmetic beats the density heuristic, and it is where a
proof would have to live.

**What holds without exception, with counts** (item 7):

| statement | range | exceptions |
|---|---|---|
| a gear `g` dividing `q` strikes exactly `i = 0` and `i = 2 u_g (mod g)`, barred per `g mod 8` | every gear `5..2000`, exhaustive | **0** |
| `5 \| q` implies no free `B = 7` island in `[1, d)` | 13,333 integers to 200,000 | **0** |
| a free `B = 7` island exists (the parent's N-R4, extended) | every prime in `(1487, 200000]`, 17,748 primes | **0** |
| a free `B = 7` island exists for every **integer** coprime to 210 above 1,649 | 45,338 integers | **0** |
| a free `B = 7` island exists for every integer coprime to 30 above 2,849 | 52,574 integers | **0** |
| each of the four island classes separately carries a free island | every prime in `(13001, 200000]`, 16,436 | **0** |
| a free island inside `[1, 0.152 d)` | every prime in `(20000, 200000]`, 15,722 | **0** |
| the first free island's absolute offset is at most 2,392 | every prime in `(1487, 200000]` | **0** |
| `Fail_7` inside `Fail_11` inside `Fail_13` (forced by the island inclusion; a gate) | 17,982 primes | **0** |
| the minimum free-island count strictly increases by band (2, 4, 12, 21, 57, 107) | six bands to 200,000 | **0** |

## 9. What is new

Screened against `docs/novel/README.md` (every index line, in particular `reachability-landscape`,
`walk-path-parts`, `walk-path-transforms`, `walk-tooth-frame`, `anchor-235-layer-laws`,
`cover-half-counter-ladder`, `corridor-law`), `docs/proofs/`, and the parent's document. The
register's `reachability-landscape` line carries the island witness for primes 1489..19997;
nothing in the register mentions integer `q`, a divisor rule, a one-class witness, a shortened
arc, or a cover number.

* **N-I1 (the multiple-of-five law, and the exact role of primality).** A gear `g` dividing `q`
  strikes exactly the two offset classes `i = 0` and `i = 2 u_g (mod g)`, and each is a barred
  class of `g` according to `chi_g(2)`, `chi_g(-2)` - so 0, 1, 2, 1 of them by `g mod 8`
  (0 exceptions, 301 gears). Gear 5 is `5 (mod 8)`, its two divisor classes are exactly `Bar(5)`,
  and all four `B = 7` islands lie in them, so **the island witness fails at every `q` divisible
  by 5 and at no other integer above 2,849**. Consequence: the witness is a statement about
  `q^2 + 6i - 2` and `q^2 + 6i` under the single hypothesis `gcd(q, 5) = 1`; primality is not
  used. Prior art in one line: Gauss's second supplement supplies `chi_g(+-2)`; what is new is the
  object - the relocation of a divisor gear's strikes onto its own unreachable classes, and the
  exact consequence for the witness.
* **N-I2 (the one-class witness).** Each island class alone suffices: `i = 12 (mod 35)` from
  `q = 5477`, all four classes at once from `q = 13001`, 0 exceptions to 200,000. The witness set
  is an arithmetic progression of density `1/35`, not `4/35`. New.
* **N-I3 (the short-arc witness).** A free island sits inside `[1, 0.152 d)` for every prime in
  `(20000, 200000]` and at absolute offset at most 2,392 anywhere in the sweep. New; it turns
  N-R4 from "the walk ends before the top gear's next tooth" into "the walk ends in the first
  seventh of that arc".
* **N-I4 (the cover number `K(d)`, and the gap it opens).** With every gear free to choose any
  nonzero-QR phase (two classes at the fixed separation `d_g`) and used once, the exact minimum
  number of gears that can strike every island of `[1, d)` is `K(d) = 3, 4, 6, 9, 14, 20` at
  `d = 35, 70, 140, 280, 560, 1120`, certified optimal, against a counting requirement of
  `2, 4, 5, 7, 9, 10`. The counting requirement is bounded (it is the parent's `sum 2/g = 1`
  threshold); the cover number is not. A failure therefore pins `q` modulo the product of at least
  `K(d)` gears, which is `1.1e32` already at `d = 1120` where `q ~ 3 x 10^3`. New, and it is the
  branch's contribution toward the root: the first quantity in this line that grows while the
  counting margin does not.
* **N-I5 (the witness extended, and the failure thresholds).** The `B = 7` witness holds for every
  prime in `(1487, 200000]` (17,748 primes, 0 exceptions - a tenfold extension of the parent's
  certified range) and for every integer coprime to 210 above 1,649; the minimum free-island count
  by band is 2, 4, 12, 21, 57, 107. The `B = 11` threshold is `q = 9281` and the `B = 13`
  threshold `q = 33623` (five failures above the parent's range). New as exact data.
* Filed, not claimed: the four composite failures `121, 247, 341, 1649`; the fragility of the
  failures (a single gear deletion frees an island at 20 of 21); the enrichment of free islands on
  higher-`B` islands (0.3231 against 0.2727, exactly the parent's order-one bar); the nesting
  `Fail_7` inside `Fail_11` inside `Fail_13` (forced by set inclusion, a gate); the free fraction
  tracking `prod (1 - 2/g)` (a rate, stopped).

## 10. Verdict

**FACT, and one thing of a different kind.**

The witness survives every pressure applied to it and comes out sharper. Its range is extended
tenfold (0 exceptions in 17,748 primes to 200,000), its witness set is cut from four classes to
one, its arc from `d` to `d/6`, and its hypothesis from "`q` prime" to "`gcd(q, 5) = 1`" - the
last with an exact mechanism and an exact converse, since **every** multiple of 5 fails and does
so because gear 5's divisor classes are its barred classes. That converse is the branch's cleanest
new object and it settles the brief's question: the statement is about the two quadratics, but not
about them alone.

The 17 failures have no conspiracy in them. They are the machines with too short an arc: 16 of 17
in the short arc, never more than 57 islands, no shared residue, and covers that need up to 24
gears with most gears taking a single island. They are also fragile - delete one gear and the
failure is gone at 20 of 21.

The thing of a different kind is `K(d)`. The parent proved the island frame scale-free for
counting: strikes per island is `sum_{B < g <= q} 2/g` whatever `B`, so no count forces a free
island. This branch computes what the adversary actually has to do - cover a fixed sparse set with
two classes per gear at a **fixed separation** - and finds the exact cover number growing (3, 4,
6, 9, 14, 20) while the counting requirement stalls (2, 4, 5, 7, 9, 10). The gap is the machine's
own arithmetic asserting itself over the density heuristic. It does not yet bound anything:
`K(d)` is measured at six arcs and its growth law is not proved. But it is the first quantity in
this line that moves in the right direction, and the modulus consequence - a failure pins `q`
modulo a number `10^28` times `q` at `d = 1120` - is the shape a proof would have to take.

Toward the root: no length bound. What the branch adds is a much smaller and much better
conditioned target for the interaction that R2.a.i and R2.a.i.a both stop at.

## 11. Dead ends (do not re-enter)

* **The witness as a statement about the two quadratics with no hypothesis on `q`.** False, with
  an infinite family of counterexamples: every `q = 0 (mod 5)` coprime to 6, 13,333 of 13,333 in
  the sweep. The hypothesis `gcd(q, 5) = 1` is necessary and (measured) sufficient above 2,849.
* **"The failures are a few small gears covering many islands."** Refuted: the exact minimum cover
  at the failures is 24 and 25 gears at `q = 1487, 1649` - 0.42 and 0.39 of the island count - and
  most gears in an optimal cover take exactly one island. The failures are short arcs, not
  efficient covers.
* **"The failures share a residue."** Refuted: over the 17 primes the largest class is 3 of 17 at
  `q mod 11, 13, 17, 19`, and 2 of 17 from 23 on.
* **The brief's nesting direction (`Fail_13` inside `Fail_7`).** False by set inclusion and
  refuted at 224 primes; the true direction is a one-line gate, not a finding.
* **`B = 11` or `B = 13` as the witness frame.** Confirmed dead at a larger scale than the parent
  had: the thresholds are 9,281 and 33,623 against 1,487, and `B = 13` still fails five times
  above the parent's sweep. Sparsity is bought at a threshold that rises faster than the density
  falls.
* **The minimum blocking set inside the machine as a growing quantity.** It does not exist at all
  above `q = 1487` (a free island is uncoverable by the machine). Only the blocking set of the
  *struck* islands (4.2) and the `q`-free cover number `K(d)` (4.3) are meaningful, and only the
  second bears on the root.
