# The exhaust, first pass (branch R4.b.viii.a)

Parent: node R4.b.viii (the owner's stack of machines). The observation that spawned this branch
is the ledger's own line: **the exhaust is the only object of the three with no measurement and no
branch document**; every number ever quoted about a tier above the wheels is a top-machine number
read at a raised split. This branch makes the exhaust an investigated object in its own right.

Deliverable, in the owner's words: make the exhaust an investigated object - the tiers measured as
objects, `CutMono` given its exact elementary form and a written proof, the zones and the
redundancy lemma stated in the stack's coordinate, the exhaust's own record stated, and a census
of what the exhaust actually does below and above the window.

What this branch can find that is not already known: the exhaust has **no** measurement on the
record at all, so every number here is new to the object; and `CutMono` (O-X1) is carried as a
hypothesis in the kernel with the note "the attack is to find the right elementary form", which is
a definite unsolved item, not a translation of a known result. The known results met on the way -
Bertrand's postulate, Chebyshev's `theta`, Stormer's theorem, the sieve to the square root - are
cited in one line each and not rewritten.

---

## 1. Pre-registered (written before any computation of this branch)

### The theory

**T1 (self-similarity is real, not just formal).** X1 says every top-machine law is stated in
`(smallest gear, gear count, range)` only, so tier 3 is the top machine at a raised split. If that
is true then tier 3 at `q = 5` - gears above 30 - obeys L1-L4, L6, L10, L15 of `top_machine_1.md`
and L46, L57-L59 of `top_machine_4/6.md` **with `q' = 31` and "smooth" meaning 30-smooth**, with
zero exceptions, because the proofs never mention where the tier sits.

**T2 (`CutMono` is elementary and its failure has an exact cause).** The cut sequence climbs
because a cut is a product of **many** primes above the previous cut, and Bertrand puts a prime in
every dyadic interval. The degeneracy at `q = 2, 3` is not a small-number accident: it is the
statement that the first ratio `cut_1 / cut_0` is too small for two dyadic intervals to fit.

**T3 (the redundancy lemma is the exhaust's whole action on a range).** On `[1, N]` a gear above
`sqrt(N)` cannot strike twice with a cofactor above 1, so its only non-echo action is its own
number. The three gear zones of a range - repeating, non-repeating, silent - are therefore an
exact partition of the incidence count, computable in closed form from `pi` and `floor(N/g)`.

**T4 (the exhaust's record is the conjecture again).** By L59/L65 the bottom stratum of any tier's
quiet zone is that tier's family `(1, 1)`, the twin primes above its top gear. So a tier's own
record is bounded below by prime gaps and above by nothing available; O-X4 is ROOT, not open.

**T5 (the exhaust starts to matter exactly at `p_1^2`).** Below `Q^2` every exhaust strike is a
home strike or an echo (X5). The first strike that is neither needs two gears above `Q` on one
number, so it is at `nextprime(Q)^2` and not before - the same height at which L58 says the zone
rule fails, seen from the other side.

### Predictions, with numbers and refutation conditions

**The owner's predictions (from the brief), recorded so a refutation in the owner's favour is
visible:**

| # | the owner's prediction | refuted by |
|---|---|---|
| O1 | all the wheels' laws hold for tier 3 with its own parameters, since they are theorems in intrinsic parameters | one exception in any law |
| O2 | the record on the range is the smooth-zone record, linear in the largest gear used (L46/L47) | the record lying above the smooth zone at the measured machines |
| O3 | every exhaust strike below the window and inside it is a home strike or an echo, 0 exceptions | one strike that is neither |
| O4 | the exhaust's own record is the conjecture in disguise, exactly as for the wheels | a bound found that is not a twin-prime bound |

**The prover's predictions:**

| # | prediction | number | refuted by |
|---|---|---|---|
| P1 | tier-3 wheels at `q = 5, 7, 11` obey L1-L10, L15 exactly | 0 exceptions; open count `prod(g - 2)`, run ceiling `q' - 3`, chain ceiling `q' - 2`, dominoes `prod(g - 4)`, `prod(g - 3)`, clump `2(q' - 3) + 1` | any mismatch |
| P2 | the loaded tier 3 at `q = 5` has, for `n <= Q - 2`, pair `n` open iff `n`, `n + 2` are both **30-smooth** | 0 exceptions to `N = 10^7` | one exception |
| P3 | `CutMono` holds for every `q >= 5` and fails for `q = 2, 3`, the cause being the first ratio | `prod primes below q` = 1, 2, 6, 30 at `q` = 2, 3, 5, 7 | a `q >= 5` with a falling cut |
| P4 | the dyadic bound proves the step: `a >= 5`, `b >= 4a` implies `prod primes in (a, b] > b` | crude bound `> 2a^2 >= b` | a counterexample pair |
| P5 | on `[1, N]` every strike by a gear `g > sqrt(N)` is a home strike or an echo | 0 exceptions | one exception |
| P6 | the strike census splits exactly as `sum floor(N/g)` over the three zones | exact identity | a mismatch |
| P7 | the first exhaust strike in `(Q^2, Q^3]` that is neither home nor echo is at `nextprime(Q)^2` | `q = 5`: `961`; `q = 7`: `44521` | an earlier one |
| P8 | the density of neither-home-nor-echo numbers above `Q^2` is **not** negligible | measured counts | a different shape |
| P9 | the exhaust's own record is bounded below by a prime gap (L63), upper bound a twin-prime statement (L65) | ratio truth / bound 3 to 24 | a smaller obstruction |

**What would refute the branch as a whole.** A tier-3 law that fails with its own parameters would
refute X1 (self-similarity) and would be the most valuable outcome here. A `q >= 5` with a falling
cut would refute `CutMono` outright and would change the kernel's hypothesis from removable to
real.

### Scorecard

| item | prediction | outcome |
|---|---|---|
| 1 tiers as objects | O1, P1, P2 | **O1 CONFIRMED, P1 CONFIRMED with one sharpening (X10), P2 CONFIRMED** |
| 2 CutMono | O-X1, P3, P4 | **P3 CONFIRMED and sharpened (q = 4 also fails), P4 CONFIRMED; O-X1 CLOSED - `CutMono` is now a theorem for `q >= 5`** |
| 3 zones and redundancy | P5, P6 | **both CONFIRMED, 0 exceptions in 7,357,725 strikes; lemma proved in a stronger form (no primality, one inequality)** |
| 4 the exhaust's record | O2, O4, P9 | **O2 REFUTED as stated and replaced by a regime law (X20): the record is the smooth-zone record only above a crossover height that the split pushes upward. O4 CONFIRMED, P9 CONFIRMED (ratio 7.2 to 13.0)** |
| 5 what the exhaust adds | O3, P7, P8 | **O3 CONFIRMED, 0 exceptions in 57,344 incidences; P7 CONFIRMED exactly; P8 CONFIRMED with the numbers (0% on the window, 36.9% one decade above, 86.6% five decades above at `q = 5`)** |

---

## 2. Setup as computed

Scripts in `research/exhaust/r1/`, results (untracked) in `research/exhaust/r1/results/`. Every
number in this document is in the document.

| script | what it computes | exactness |
|---|---|---|
| `common.py` | primes, smooth lists, tier wheels and range machines, run statistics | - |
| `e1_tiers.py` | tier-3 wheels at `q = 5, 7, 11` over **full periods**; the loaded tier 3 on `[1, N]`, `N = 10^6, 10^7`, at `q = 5, 7` | exact |
| `e2_cutmono.py` | the cuts for `q = 2..13`; `theta(cut_2)` at `q = 5` by segmented sieve; the sharp threshold of the elementary lemma for `a = 2..3000` | exact / `theta` to double precision |
| `e3_zones.py` | the three gear zones of `[1, N]`, `N = 10^4..10^7`, closed counts; the redundancy lemma exhaustively; silent gears in pair coordinates | exact |
| `e4_record.py` | the record of the loaded tier 3, sweep `N = 10^4..10^8` at `q = 5, 7`; quiet-zone strata; L59 and the L63 bound | exact |
| `e5_below.py` | the exhaust's incidences on `[1, Q]`, on `(Q, Q^2]` classified, and above the window to `Q^3`; the share curve to `10^8` | exact |

The tier-3 wheels used (a tier's first gears; the full tier is not enumerable and is not needed,
because every wheel law is stated in `(q', m)`):

| base `q` | `cut_1 = q#` | tier-3 wheel | `q'` | `m` | period `W` |
|---|---|---|---|---|---|
| 5 | 30 | 31, 37, 41, 43 | 31 | 4 | 2,022,161 |
| 7 | 210 | 211, 223, 227 | 211 | 3 | 10,681,031 |
| 11 | 2310 | 2311, 2333 | 2311 | 2 | 5,391,563 |
| (control) 5 | 30 | tier **2**: 7, 11, 13 | 7 | 3 | 1,001 |

Total residues examined over full periods: **18,095,756**; total open positions **17,398,284**.

The loaded exhaust on a range: tier 3 at base `q` restricted to the gears that act on `[1, N]`,
i.e. the primes in `(q#, Q]` with `Q` the largest gear used (`Q = ` the largest prime below
`sqrt(N)`). This is the honest object at `q = 7` and above, where tier 3 - the primes in
`(210, cut_2]` with `cut_2` about `3.7 x 10^79` - **cannot be enumerated**; its initial segment
can, and the laws are stated in the segment's own parameters, so nothing is lost.

---

## 3. Results

### 3.1 The tiers as objects: the wheel laws at a raised split (item 1, part A)

Every law checked exactly over the full period. `EXC` is the exception count.

| law | statement in the part's own parameters | `q = 5` wheel | `q = 7` wheel | `q = 11` wheel | control tier 2 | EXC |
|---|---|---|---|---|---|---|
| L1/L2 | two teeth at `0`, `-2`; `g - 2` slots; arcs `g - 3` and `1`; singleton at `-1` (the shield) | ok | ok | ok | ok | **0** |
| L3 | the two teeth of one gear are at cyclic distance exactly 2 | ok | ok | ok | ok | **0** |
| L4 | no gap of exactly 4 between consecutive open pairs | 0 occurrences | 0 | 0 | 0 | **0** |
| L5 | open count `= prod(g - 2)` | 1,622,985 | 10,392,525 | 5,382,279 | 495 | **0** |
| L6 | shield `n = -1` open; origin clump `2(q' - 3) + 1` | 57 = 2(28)+1 | 417 = 2(208)+1 | 4617 = 2(2308)+1 | 9 | **0** |
| L6' | the antipode pairs `n = 2`, `n = -4` open | ok | ok | ok | ok | **0** |
| L7 | mirror `n -> -n - 2` preserves the open set | 0 mismatches | 0 | 0 | 0 | **0** |
| L9 | every gap length has an even count except length 1 | ok | ok | ok | ok | **0** |
| L10 | longest run `= q' - 3`; longest chain `= q' - 2` | 28 / 29 | 208 / 209 | 2308 / 2309 | 4 / 5 | **0** |
| L10 counts | run starts of `L`: `prod(g - 2 - L)` for `L >= 2`; chain starts: `prod(g - 1 - L)` for `L >= 1` | all `L` | all `L` | all `L` | all `L` | **0** |
| L15 | adjacent `prod(g - 4)`; sharing a member `prod(g - 3)` | 1,285,713 / 1,447,040 | 10,109,259 / 10,250,240 | 5,373,003 / 5,377,640 | ok | **0** |

**Total: 15 law checks x 4 wheels, 0 exceptions.** O1 and P1 confirmed. The self-similarity X1 is
no longer "FACT (reasoning)"; it is measured.

**One sharpening, found only at the raised split (X10 below).** L10's run-start count
`prod(g - 2 - L)` is stated for all `L`; it is **false at `L = 1`** and true for every `L >= 2`.
At the `q = 5` tier-3 wheel the number of run starts of length 1 is 1,622,985 (the open count
`prod(g - 2)`), while `prod(g - 3)` is 1,447,040. The chain count `prod(g - 1 - L)` is correct at
every `L >= 1`, including `L = 1`. The mechanism is the shield: one gear's open residues form two
arcs, `g - 3` and the singleton `{-1}`; a window of length `L >= 2` fits only in the long arc, so
each gear contributes `g - 2 - L`, but a window of length 1 fits in the singleton too, so each
gear contributes `g - 3 + 1 = g - 2`. In the step-2 order the two teeth `0` and `g - 2` are
adjacent (`0 = (g - 2) + 2`), so the open residues form a **single** arc of length `g - 2` and
there is no exception: the chain law is the cleaner of the two.

*A measurement bug worth recording, because it would have looked like a refutation.* The step-2
chain must be counted on the **single** cycle `0, 2, 4, ..., W - 1, 1, 3, ..., W - 2` - the period
`W` is odd, so doubling is a full cycle. Counting the two parity classes as separate sequences
produced `q' - 3` spurious mismatches at every wheel including the control tier-2 wheel, which is
what identified it as an instrument error rather than a tier-3 property.

### 3.2 The loaded exhaust on a range: the zone laws at a raised split (item 1, part B)

`Q` = the largest gear used; "smooth" means `q#`-smooth (30-smooth at `q = 5`, 210-smooth at
`q = 7`).

| `q` | `N` | gears | `m` | `q'` | L46 range | L46 EXC | L57 range | L57 EXC | `p_1` | first admissible non-smooth |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | `10^6` | (30, 997] | 158 | 31 | `[1, 995]` | **0** | `[1, 994009]` | **0** | 1009 | 1009 = `p_1` |
| 5 | `10^7` | (30, 3137] | 436 | 31 | `[1, 3135]` | **0** | `[1, 9840769]` | **0** | 3163 | 3163 = `p_1` |
| 7 | `10^6` | (210, 997] | 122 | 211 | `[1, 995]` | **0** | `[1, 994009]` | **0** | 1009 | 1009 = `p_1` |
| 7 | `10^7` | (210, 3137] | 400 | 211 | `[1, 3135]` | **0** | `[1, 9840769]` | **0** | 3163 | 3163 = `p_1` |

- **L46** (the gear-zone identity): for `n <= Q - 2` the pair `n` is open iff `n` and `n + 2` are
  both `q#`-smooth. **0 exceptions**; P2 confirmed.
- **L57** (the zone rule): for `n <= Q^2`, `n` is admissible iff `n = s P` with `s` `q#`-smooth and
  `P` either 1 or a prime above `Q`. Verified cell by cell over `[1, 9,840,769]` at each machine
  by comparing the sieve against the generative rule: **0 exceptions**, 21,669,556 cells in all.
- **L58** (the edges): the first admissible non-smooth number is exactly `p_1 = nextprime(Q)`, at
  every machine. **0 exceptions.**
- **L4 on the range**: gap 4 never occurs (0 occurrences at all four machines).
- **L10 ceilings on the range**: the longest run is exactly `q' - 3` and the longest chain exactly
  `q' - 2` at every machine (28/29 at `q = 5`, 208/209 at `q = 7`) - attained, not merely bounded.

### 3.3 `CutMono`: the cuts, and the exact elementary form (item 2)

**The cuts.**

| `q` | `prod primes below q` | `cut_1 = q#` | tier 2 gears | `cut_2` | `cut_1 <= cut_2`? |
|---|---|---|---|---|---|
| 2 | 1 | 2 | 0 | 1 (empty product) | **no** |
| 3 | 2 | 6 | 1 | 5 | **no** |
| 4 (not prime) | - | 6 | 1 | 5 | **no** |
| 5 | 6 | 30 | 7 | **215,656,441** (9 digits) | yes |
| 7 | 30 | 210 | 42 | 80 digits | yes |
| 11 | 210 | 2310 | 338 | 973 digits | yes |
| 13 | 2310 | 30030 | 3242 | 12,930 digits | yes |

**`cut_3` at `q = 5`, computed.** `cut_3 = ` the product of the primes in `(30, 215,656,441]`.
Segmented sieve: `pi(cut_2) = 11,896,487`, `theta(cut_2) = 215,639,987.078`, so
`log cut_3 = theta(cut_2) - theta(30) = 215,639,964.488` and

> **`cut_3` at `q = 5` has 93,651,247 digits.**

(`theta(cut_2) / cut_2 = 0.999924` - Chebyshev's `theta(x) ~ x`, cited in a line, is the only
analytic input and it is used only to report a digit count, never in a proof here.)

For `q >= 7`, `cut_2` is already beyond enumeration, so `cut_3` is reported from `theta` alone:
`log cut_3 = theta(cut_2) - theta(cut_1) ~ cut_2`, so `cut_3` has about `cut_2 / ln 10` digits -
at `q = 7` a **digit count that itself has 80 digits**; at `q = 11`, 973 digits; at `q = 13`,
12,930 digits. Tier 3 at `q = 7` is not an enumerable object and this branch does not pretend
otherwise; it is studied through its initial segment, the loaded exhaust on a range.

**The sharp threshold of the elementary lemma.** For each `a`, the largest `b` with
`prod of the primes in (a, b] <= b`:

| `a` | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|
| largest failing `b` | 4 | 6 | 6 | 10 | 10 | 12 | 12 | 12 |
| `b / a` | 2.000 | 2.000 | 1.500 | **2.000** | 1.667 | 1.714 | 1.500 | 1.333 |

Over `a = 2..3000`: for `a >= 5` the largest failing `b` **never reaches `4a`**; the maximum ratio
is exactly 2.0000, attained only at `a = 5` (`b = 10`). So the hypothesis `b >= 4a` is safe with a
factor-2 margin, and `b > 2a` appears to be the true threshold. P4 confirmed.

**The proof.** (Written for transcription; mathlib has `Nat.exists_prime_lt_and_le_two_mul`.)

> **Lemma A (the dyadic product bound).** Let `a >= 1` and `t >= 1` be integers with
> `2^t a <= b`. Then
>
>         prod { p prime : a < p <= b }  >  a^t * 2^{t(t-1)/2} .
>
> *Proof.* For each `i = 0, ..., t - 1` Bertrand's postulate gives a prime `p_i` with
> `2^i a < p_i <= 2^{i+1} a`. Since `2^{i+1} a <= 2^t a <= b`, each `p_i` lies in `(a, b]`; the
> intervals are disjoint so the `p_i` are distinct. Every other prime of `(a, b]` is at least 2,
> so the full product is at least `prod_i p_i > prod_i 2^i a = a^t 2^{0 + 1 + ... + (t-1)}`. QED

> **Lemma B (the step).** Let `a >= 5` and `b >= 4a`. Then
> `prod { p : a < p <= b } > b`. If moreover `a >= 16` or `b >= 8a`, then
> `prod { p : a < p <= b } > 4b`.
>
> *Proof.* Let `t = floor(log_2(b/a))`, so `t >= 2` and `2^t a <= b < 2^{t+1} a`. By Lemma A the
> product exceeds `Z = a^t 2^{t(t-1)/2}`.
> (i) `Z >= 2^{t+1} a >= b` follows from `a^{t-1} 2^{t(t-1)/2} >= 2^{t+1}`: at `t = 2` this is
> `2a >= 8`, true since `a >= 4`; for `t >= 3` the left side is at least `5^{t-1} 2^{t(t-1)/2}`
> and the exponent `(t-1) log_2 a + t(t-1)/2` is at least `2.32(t-1) + t(t-1)/2`, which exceeds
> `t + 1` at `t = 3` (7.64 against 4) and grows quadratically while the right side grows linearly.
> (ii) `Z >= 2^{t+3} a >= 4b` follows from `a^{t-1} 2^{t(t-1)/2} >= 2^{t+3}`: at `t = 2` this is
> `2a >= 32`, i.e. `a >= 16`; at `t = 3` it is `8a^2 >= 64`, i.e. `a >= 3`, true; for `t >= 4` the
> same monotone comparison applies (`2.32 x 3 + 6 = 12.96` against `7` at `t = 4`). QED

> **Theorem X12 (`CutMono` from `q = 5`).** For every prime `q >= 5` and every `k >= 1`,
>
>         cut_{k+1}  >  4 cut_k .
>
> Hence the cut sequence is strictly increasing from `cut_1` on; together with
> `cut_0 = q <= q# = cut_1` (already in the kernel as `cutMono_one`) the hypothesis
> `CutMono q k` holds for **every** `k`, unconditionally, for every prime `q >= 5`.
>
> *Proof.* **Base `k = 1`.** `cut_2` is the product of the primes in `(q, q#]`.
> For `q = 5` the primes in `(5, 30]` are 7, 11, 13, 17, 19, 23, 29 and already
> `7 * 11 * 13 = 1001 > 120 = 4 cut_1` (the full product is 215,656,441). For `q >= 7`,
> `cut_1 / cut_0 = q# / q` is the product of the primes below `q`, which is at least
> `2 * 3 * 5 = 30 >= 8`; so `b = q# >= 8a` with `a = q >= 7 >= 5`, and Lemma B(ii) gives
> `cut_2 > 4 cut_1`.
> **Step.** Let `k >= 2` and put `a = cut_{k-1}`, `b = cut_k`. By the induction hypothesis
> `b >= 4a`, and `a >= cut_1 = q# >= 30 >= 16`, so Lemma B(ii) gives
> `cut_{k+1} = prod { p : a < p <= b } > 4b = 4 cut_k`. QED

> **Theorem X13 (the exact cause of the degeneracy).** The base case of X12 is exactly the
> inequality `cut_1 >= 4 cut_0`, i.e. `prod_{p <= q} p >= 4q`; for prime `q` this is
> `prod_{p < q} p >= 4`. It **fails exactly at `q = 2, 3` (and at the non-prime `q = 4`)**, where
> the products are 1, 2 and 6 against the thresholds 8, 12 and 16, and it holds from `q = 5` on
> (6 against 4). The failure is not a small-number accident: with `cut_1 < 4 cut_0` there is at
> most one dyadic interval inside `(cut_0, cut_1]`, so Bertrand guarantees only one prime, and a
> single prime `p <= b` never exceeds `b`. That is literally the observed `q = 3` failure:
> `(3, 6]` contains only 5, and `cut_2 = 5 < 6 = cut_1`.

This **closes O-X1**. The kernel's `CutMono` hypothesis is discharged for every base the project
uses; the theorems `stack_eq_primesLE`, `exhaust_gear_gt_cut`, `exhaust_silent` and
`stack_open_iff_twin` become unconditional at `q >= 5`.

### 3.4 The zones and the redundancy lemma in the stack's coordinate (item 3, O-X2)

The machine is `{all primes <= Q}` with the exhaust above `Q`; the zones are defined by the
**range** `[1, N]`, not by the tier.

| `N` | `pi(N)` | repeating `g^2 <= N` | non-repeating `sqrt N < g <= N/2` | silent `g > N/2` | strikes repeating | strikes non-rep. | strikes silent | total |
|---|---|---|---|---|---|---|---|---|
| `10^4` | 1,229 | 25 | 644 | 560 | 18,016 | 5,724 | 560 | 24,300 |
| `10^5` | 9,592 | 65 | 5,068 | 4,459 | 202,219 | 59,722 | 4,459 | 266,400 |
| `10^6` | 78,498 | 168 | 41,370 | 36,960 | 2,198,007 | 618,741 | 36,960 | 2,853,708 |
| `10^7` | 664,579 | 446 | 348,067 | 316,066 | 23,492,474 | 6,321,777 | 316,066 | 30,130,317 |

Above `sqrt(N)`, home strikes against echoes:

| `N` | strikes above `sqrt N` | home | echo | echo share |
|---|---|---|---|---|
| `10^4` | 6,284 | 1,204 | 5,080 | 80.84% |
| `10^5` | 64,181 | 9,527 | 54,654 | 85.16% |
| `10^6` | 655,701 | 78,330 | 577,371 | 88.05% |
| `10^7` | 6,637,843 | 664,133 | 5,973,710 | 89.99% |

The home count above `sqrt(N)` is `pi(N) - pi(sqrt N)` by construction; it was recomputed
independently by a least-prime-factor sieve and agrees exactly (664,133 at `N = 10^7`).
**0 exceptions to the redundancy lemma in 7,357,725 strikes checked** (`N = 10^5, 10^6, 10^7`).
Silent gears in pair coordinates: 5,019 gears checked at `N = 10^4, 10^5`, **0 exceptions**.

> **Theorem X15 (the redundancy lemma, range form).** Let `N >= 1` and let `g` be **any** integer
> with `g^2 > N`. Then every multiple of `g` in `[1, N]` is either `g` itself or has a prime
> factor strictly smaller than `g`. Consequently, on `[1, N]`:
>
> (a) a gear `g` with `g^2 > N` makes exactly one non-echo strike, its home strike at `n = g`
>     (present iff `g <= N`); every other strike of `g` is an echo of a strictly smaller gear;
> (b) a gear `g > N/2` makes only its home strike;
> (c) in pair coordinates a gear `g > N/2` touches exactly the two positions `g` and `g - 2`,
>     both of which contain the number `g` - the home column and its partner tooth;
> (d) the incidence counts are exact: `sum_{g^2 <= N} floor(N/g)` in the repeating zone,
>     `sum_{sqrt N < g <= N/2} floor(N/g)` in the non-repeating zone of which one per gear is
>     home, and one home strike per silent gear.
>
> *Proof.* Write the multiple as `n = g m <= N`. If `m = 1` then `n = g`. If `m >= 2` then
> `g m <= N < g^2` gives `m < g`; the least prime factor `r` of `m` satisfies `r <= m < g` and
> `r | m | n`, so `r` is a strictly smaller gear striking `n`. (b) is `m >= 2 => n >= 2g > N`.
> (c) `g | n` in `[1, N]` forces `n = g`, and `g | n + 2` forces `n + 2 = g`. (d) is counting
> multiples. **No primality of `g` is used anywhere.** QED

**How this differs from the kernel's `exhaust_home_or_echo`.** The kernel form needs a two-sided
window `C < n <= C^2` and a divisor `p > C`. The range form needs a **single** inequality,
`g^2 > N`, with `N` the top of the range, and no lower bound on `n` at all: it says the same thing
about the whole range at once rather than about one cell. The window form is the case
`N = C^2`, `g > C`. This is the shape O-X2 asked for - the lemma in the stack's coordinate rather
than against bottom-open columns - and it is ready for the Formalist.

**Reading.** The three zones say which gears of a tier do work on a given range. At `N = 10^7`
there are 664,579 gears and only **446** of them repeat; the other 664,133 - 99.93% of the machine
- contribute exactly one home strike each plus 5,973,710 echoes. The echo share of the work above
`sqrt(N)` climbs 80.8%, 85.2%, 88.1%, 90.0% across the four decades: **the higher the range, the
more of the exhaust's action is duplication.**

### 3.5 The exhaust's own record (item 4, O-X4)

The record of the loaded tier 3 on `[1, N]`, with `Q` the largest gear used, `F_smooth` the record
of the smooth zone `[1, Q]` (the largest gap of the finite `q#`-smooth pair list, L47's form) and
`F_quiet` the record of the quiet zone `(Q, Q^2]`:

| `q` | `N` | `Q` | `m` | `F` | at | zone | `F_smooth` | `F_quiet` | `F / Q` |
|---|---|---|---|---|---|---|---|---|---|
| 5 | `10^4` | 97 | 15 | 10 | 363 | quiet | 6 | 10 | 0.1031 |
| 5 | `10^5` | 313 | 55 | 37 | 2,041 | quiet | 19 | 37 | 0.1182 |
| 5 | `10^6` | 997 | 158 | 66 | 1,793 | quiet | 34 | 66 | 0.0662 |
| 5 | `10^7` | 3,137 | 436 | 210 | 2,661 | **smooth** | 210 | 122 | 0.0669 |
| 5 | `10^8` | 9,973 | 1,219 | 735 | 7,039 | **smooth** | 735 | 203 | 0.0737 |
| 7 | `10^5` | 313 | 19 | 9 | 26,556 | quiet | 1 | 9 | 0.0288 |
| 7 | `10^6` | 997 | 122 | 20 | 297,003 | quiet | 9 | 20 | 0.0201 |
| 7 | `10^7` | 3,137 | 400 | 35 | 183,624 | quiet | 16 | 35 | 0.0112 |
| 7 | `10^8` | 9,973 | 1,183 | 55 | 62,066,475 | quiet | 40 | 55 | 0.0055 |

**O2 is refuted as stated and replaced by a regime law.** At `q = 5` tier 3 the record leaves the
quiet zone and becomes the smooth-zone record between `N = 10^6` and `N = 10^7`; at `q = 7` tier 3
it has not done so by `N = 10^8`, where the two are 40 against 55 - the crossover is just above.
The mechanism is exactly the raised split: "smooth" means `q#`-smooth, so raising the split makes
the smooth condition much weaker and the smooth-pair list much denser, and its gaps much smaller.
The smooth-pair list below `Q`:

| `Q` | `q = 5` (30-smooth): pairs / `F_smooth` | `q = 7` (210-smooth) | `q = 11` (2310-smooth) |
|---|---|---|---|
| 1,000 | 169 / 34 | 642 / 9 | 998 / 0 |
| 10,000 | 282 / 735 | 2,697 / 40 | 7,483 / 9 |
| 100,000 | 360 / 12,167 | 8,566 / 204 | 41,877 / 29 |
| 1,000,000 | 396 / 149,057 | 21,384 / 1,021 | 210,838 / 74 |
| 10,000,000 | 413 / 2,612,609 | 44,542 / 6,321 | 907,148 / 208 |

The `q = 5` list **saturates** (Stormer's theorem, 1897, cited in a line: the list is finite): the
largest 30-smooth pair found below `10^13` is `354,365,440, 354,365,442`, and by `Q = 10^7` there
are only 413 pairs. Once `Q` exceeds that last pair, `F_smooth = Q - 2 - 354,365,440` exactly -
slope 1, linear in the largest gear used, which is L48's bound at its saturated value. The `q = 7`
list is nowhere near saturation at `10^11` (233,659 pairs, the largest at `99,994,530,568`).

> **X20 (the record of a tier migrates into its smooth zone, and the split says when).** A tier's
> record on `[1, N]` is the larger of a smooth-zone record - the largest gap of the finite list of
> pairs smooth to the tier's lower cut - and a quiet-zone record, which is a prime-gap object of
> size `O(log^2 Q)`. The smooth-zone record is eventually `Q - 2 - s`, linear in `Q` with slope 1,
> where `s` is the largest smooth pair; so the smooth zone wins at every base, but only above a
> crossover height, and **raising the split pushes the crossover up**: `Q` about `2 x 10^3` at
> `q = 5` tier 3, above `10^4` at `q = 7` tier 3, far higher at `q = 11`. Measured: 9 machines,
> the winner correctly predicted at all 9 by comparing the two lists.

**The quiet zone, and the exhaust's own record.**

| `q` | `N` | `Q` | L59 exceptions on `(Q, 2Q]` | open pairs there: twin + smooth-member | L63 prime-gap bound | truth | ratio |
|---|---|---|---|---|---|---|---|
| 5 | `10^6` | 997 | **0** | 25 + 86 | 7 | 66 | 9.43 |
| 5 | `10^7` | 3,137 | **0** | 64 + 123 | 17 | 122 | 7.18 |
| 7 | `10^6` | 997 | **0** | 25 + 528 | 1 | 13 | 13.00 |
| 7 | `10^7` | 3,137 | **0** | 64 + 1,052 | 3 | 23 | 7.67 |

The twin counts are **identical** at `q = 5` and `q = 7` for the same `Q` (25 and 64) - this is
L61's `q`-independence of a family, seen in the exhaust's coordinate: the family `(1, 1)` of a
tier is the twin primes above its top gear and does not depend on the split at all. What the
split changes is the number of open pairs carrying a smooth member (86 against 528 at `Q = 997`).

Dyadic strata of the quiet zone (records, bottom stratum first):

- `q = 5`, `N = 10^7`: 122, 74, 75, 64, 55, 52, 49, 47, 51, 49, 54, 66 - the **U** of L64, with the
  bottom stratum making the record.
- `q = 7`, `N = 10^7`: 23, 20, 21, 27, 30, 35, 33, 32, 30, 31, 29, 31 - **not a U**: the record is
  made in the middle-upper strata, because at the raised split the bottom stratum is no longer
  family-starved (1,052 open pairs with a smooth member against 123).

> **X21 (the exhaust's own record is the conjecture in disguise - ROOT).** By L59 the bottom
> stratum `(Q, 2Q]` of a tier's quiet zone contains only primes and numbers smooth to the tier's
> lower cut, so its open pairs are the tier's family `(1, 1)` - **twin primes above `Q`** -
> together with pairs having a smooth member. The tier's own record on its quiet zone is therefore
> bounded below by the prime gaps of `(Q, 2Q]` that contain no smooth number (L63, verified with
> ratios 7.18 to 13.00 at four machines, 0 exceptions), and an **upper** bound on it is a lower
> bound on the density of twin primes in a short interval above `Q`. O4 confirmed: this is the
> conjecture, at the exhaust's rung, with the same missing instrument.

### 3.6 What the exhaust adds below, inside and above the window (item 5)

**(a) Below the cut, `[1, Q]` with `Q = q#`.** The exhaust's smallest gear is `nextprime(q#)`,
which exceeds the whole range, so **the exhaust makes no strike at all on `[1, q#]`** - not a
weakened action, none. (The motor and wheels together leave 0 open pairs on `[1, 30]` and on
`[1, 210]`, since gears 2 and 3 are present.)

**(b) The window `(Q, Q^2]`.** Every incidence of every exhaust gear classified:

| `q` | window | motor+wheels-open pairs | of which twin primes | exhaust home strikes | echoes | **neither** |
|---|---|---|---|---|---|---|
| 5 | (30, 900] | 30 | **30** | 287 | 755 | **0** |
| 7 | (210, 44100] | 621 | **621** | 9,085 | 47,217 | **0** |

**57,344 incidences, 0 exceptions.** O3 confirmed. The first open pairs at `q = 5` are
41, 59, 71, 101, 107, 137, 149, 179, 191, 197, 227, 239 - the twin-prime lower members.

> **X17 (the exhaust's action on an open pair of the window is exactly two home strikes).** In
> `(Q, Q^2]` a pair left open by the primes `<= Q` has both members prime and above `Q`, so the
> only exhaust gears dividing either member are the members themselves. Measured: **exactly 2.000
> exhaust strikes per open pair** at both bases (60 on 30 pairs, 1,242 on 621 pairs), 0 exceptions.
> The exhaust is not merely harmless on the window: its entire visible action there on the objects
> that matter is the two gears announcing that the two members are prime.

**(c) Above the window, `(Q^2, Q^3]`.**

| `q` | `Q` | `p_1` | first neither-home-nor-echo strike | equals `p_1^2`? | height above `Q^2` | first open pair that is **not** twin |
|---|---|---|---|---|---|---|
| 5 | 30 | 31 | **961** | yes | `Q^2 + 61` | 1367 (`Q^2 + 467`), with `1369 = 37^2` |
| 7 | 210 | 211 | **44,521** | yes | `Q^2 + 421` | 44,519 (`Q^2 + 419`), with `44,521 = 211^2` |

P7 confirmed exactly. At `q = 5` the first non-twin open pair is not `p_1^2 - 2 = 959`, because
`959 = 7 x 137` is struck by gear 7; it is 1367, whose partner is `37^2`.

The share of the work that is genuinely the exhaust's, by height (open pairs of the machine
`{primes <= Q}` above `Q`, split into twin primes and the rest):

| up to | `q = 5`: open / twin / exhaust's | share | `q = 7`: open / twin / exhaust's | share |
|---|---|---|---|---|
| `Q^2` | 30 / 30 / 0 | **0.000%** | 621 / 621 / 0 | **0.000%** |
| `10 Q^2` | 293 / 185 / 108 | 36.860% | 5,967 / 4,115 / 1,852 | 31.037% |
| `10^2 Q^2` | 2,988 / 1,111 / 1,877 | 62.818% | 63,473 / 29,179 / 34,294 | 54.029% |
| `10^3 Q^2` | 29,872 / 7,467 / 22,405 | 75.003% | 631,289 / 214,141 / 417,148 | 66.079% |
| `10^4 Q^2` | 298,703 / 53,862 / 244,841 | 81.968% | - | - |
| `10^5 Q^2` | 2,986,817 / 401,085 / 2,585,732 | 86.571% | - | - |

> **X18 (the exhaust's first non-redundant height, and its depth).** The first number on which an
> exhaust gear does work that no lower gear does is `p_1^2` with `p_1 = nextprime(Q)`, and not one
> cell earlier. More generally a number carrying `j` gears above `Q` needs `n >= p_1^j`, so at
> height `x` the exhaust's **depth** - the number of its gears that can sit on one number - is at
> most `floor(log_{p_1} x)`, i.e. 1 on the whole window (home strikes only) and 2 from `p_1^2` to
> `p_1^3`. The exhaust cap X5 is the depth-1 statement; every layer above is a depth-`j`
> statement at height `p_1^j`.

> **X19 (the exhaust's share of the work).** The fraction of the machine's open pairs above `Q`
> that are not twin primes - the fraction the exhaust still has to kill - is exactly 0 on
> `(Q, Q^2]` and then 36.9%, 62.8%, 75.0%, 82.0%, 86.6% over the next five decades at `q = 5`
> (31.0%, 54.0%, 66.1% at `q = 7`). The cap is not a statement that the exhaust is small; it is a
> statement that the exhaust is **exactly zero** on one stretch and the majority partner one
> decade later.

---

## 4. Laws

Numbered from **X9**, continuing the exhaust's own register in the objects ledger.

**X9 (SELF-SIMILARITY, MEASURED).** Every wheel law of `top_machine_1.md` holds verbatim for a
tier-3 wheel with its own `(q', m)`: L1, L2, L3, L4, L5, L6, L7, L9, L10 (ceilings and counts),
L15. Evidence: 4 wheels (`q = 5, 7, 11` tier 3 and a `q = 5` tier-2 control), full periods,
18,095,756 residues, 15 law checks each, **0 exceptions**. *This converts X1 from
"FACT (reasoning)" to measured. Proof: each law's proof in `top_machine_1.md` mentions only `q'`
and `m`.*

**X10 (THE RUN-START COUNT HAS EXACTLY ONE EXCEPTION, AND THE CHAIN COUNT HAS NONE).** For a
wheel of gears `G` with `q' = min G`:

        #{x : x, x+1, ..., x+L-1 all open}   =  prod (g - 2 - L)   for  2 <= L <= q' - 3 ,
        #{x : x, x+2, ..., x+2(L-1) all open} =  prod (g - 1 - L)   for  1 <= L <= q' - 2 ,

and the first fails at `L = 1`, where the count is `prod(g - 2)`, not `prod(g - 3)`.
*Proof.* Per gear, the open residues form the arcs `g - 3` and `{-1}` (L2), so a window of length
`L >= 2` fits only in the long arc (`g - 2 - L` positions) while a window of length 1 fits in both
(`g - 3 + 1`); CRT multiplies. In the step-2 order the teeth `0` and `g - 2` are adjacent, so the
open residues form a single arc of length `g - 2` and a window of length `L` has `g - 1 - L`
positions for every `L >= 1`. QED
*Evidence.* 4 wheels, every `L`, **0 exceptions** to the corrected statement.
*Reading: the shield is the only defect in the run law, and the chain coordinate removes it. This
sharpening was invisible at the wheels' own split and appeared only because the raised split gave
a fresh instance to check.*

**X11 (THE ZONE LAWS AT A RAISED SPLIT).** For a tier with lower cut `C` and largest gear used
`Q`: L46 (`n <= Q - 2`: open iff `n`, `n + 2` both `C`-smooth), L57 (`n <= Q^2`: admissible iff
`s P`, `s` `C`-smooth, `P = 1` or prime `> Q`), L58 (the first admissible non-smooth number is
`nextprime(Q)`), L4 (no gap 4) and L10's two ceilings all hold with `C = q#`.
*Evidence.* 4 machines, `q = 5, 7`, `N = 10^6, 10^7`; 21,669,556 cells for L57; **0 exceptions.**

**X12 (`CutMono` IS A THEOREM FROM `q = 5`).** For every prime `q >= 5` and every `k >= 1`,
`cut_{k+1} > 4 cut_k`; with `cut_0 <= cut_1` (Bertrand) the hypothesis `CutMono q k` holds for
every `k`. *Proof in 3.3 from Lemma A (dyadic Bertrand) and Lemma B; one finite evaluation at
`q = 5`.* **Closes O-X1.**

**X13 (THE EXACT CAUSE OF THE DEGENERACY).** The stack is monotone from its first step iff
`prod_{p <= q} p >= 4 q`, i.e. (for prime `q`) iff the product of the primes below `q` is at least
4; this fails exactly at `q = 2, 3` (and at `q = 4`) and holds from `q = 5`. Mechanism: below the
threshold `(cut_0, cut_1]` admits at most one dyadic interval, so Bertrand supplies only one prime
and a single prime `<= b` never exceeds `b`. *Verified: `q = 2, 3, 4` fail; `q = 5..13` hold.*

**X14 (THE THREE GEAR ZONES OF A RANGE, EXACT CLOSED COUNTS).** On `[1, N]` with all primes as
gears, the incidence count is `sum_{g <= sqrt N} floor(N/g)` (repeating) `+
sum_{sqrt N < g <= N/2} floor(N/g)` (non-repeating) `+ (pi(N) - pi(N/2))` (silent), and the three
sum to `sum_{p <= N} floor(N/p)`. *Evidence: 4 ranges, identity exact.* Only `pi(sqrt N)` gears
repeat: 446 of 664,579 at `N = 10^7`.

**X15 (THE REDUNDANCY LEMMA, RANGE FORM).** Stated and proved in 3.4: any integer `g` with
`g^2 > N` strikes on `[1, N]` only its own number and echoes of strictly smaller gears; no
primality of `g` is used; the hypothesis is the single inequality `g^2 > N`.
*Evidence.* 7,357,725 strikes at `N = 10^5, 10^6, 10^7`, **0 exceptions**. **Closes O-X2's first
half.** Ready for the Formalist as a range companion to `exhaust_home_or_echo`.

**X16 (SILENT GEARS IN PAIR COORDINATES).** A gear `g > N/2` touches exactly the two pair
positions `g` and `g - 2` of `[1, N]`, both containing the number `g`: the home column and its
partner tooth. *Proof: `g | n` in the range forces `n = g`; `g | n + 2` forces `n + 2 = g`.*
*Evidence.* 5,019 silent gears at `N = 10^4, 10^5`, **0 exceptions.** **Closes O-X2's second
half**, in the stack's coordinate rather than against bottom-open columns.

**X17 (THE EXHAUST'S ACTION ON THE WINDOW IS TWO HOME STRIKES PER OPEN PAIR).** On `(Q, Q^2]`,
every pair left open by the primes `<= Q` receives exactly two exhaust strikes, both home, from
the gears `n` and `n + 2` themselves; and every other exhaust incidence there is a home strike or
an echo. *Proof: X5 plus the fact that both members are primes above `Q`.*
*Evidence.* 57,344 incidences at `q = 5, 7`; 2.000 strikes per open pair; **0 neither**.

**X18 (THE FIRST NON-REDUNDANT HEIGHT, AND THE EXHAUST'S DEPTH).** The first number on which an
exhaust gear does work no lower gear does is `p_1^2`, `p_1 = nextprime(Q)`; a number carrying `j`
exhaust gears is at least `p_1^j`, so the exhaust's depth at height `x` is at most
`floor(log_{p_1} x)`. *Evidence.* `q = 5`: 961 = `31^2`, `Q^2 + 61`; `q = 7`: 44,521 = `211^2`,
`Q^2 + 421`; **0 earlier**. *The exhaust cap X5 is the depth-1 case of a ladder of statements.*

**X19 (THE EXHAUST'S SHARE ABOVE THE WINDOW).** The fraction of the machine's open pairs above
`Q` that are not twin primes is 0 on `(Q, Q^2]`, then 36.9%, 62.8%, 75.0%, 82.0%, 86.6% over the
next five decades at `q = 5` and 31.0%, 54.0%, 66.1% over three at `q = 7`.
*Evidence.* Exact counts to `10^8` at both bases.

**X20 (THE TIER'S RECORD MIGRATES INTO ITS SMOOTH ZONE, AND THE SPLIT SETS THE CROSSOVER).** As in
3.5. *Evidence.* 9 machines: `F_range = max(F_smooth, F_quiet)` exactly at all 9, **0 exceptions** -
the range record is the larger of the two zone records and nothing else. The `q = 5` smooth pair
list saturates at 423 pairs below `10^13` with largest member 354,365,440 (finiteness is
Stormer's theorem; no further pair occurs in four and a half further orders of magnitude).

**X21 (THE EXHAUST'S OWN RECORD IS ROOT).** As in 3.5. *Evidence.* L59 with 0 exceptions at 4
machines; L63's bound with ratios 7.18, 9.43, 7.67, 13.00.

**X22 (THE U-PROFILE IS NOT UNIVERSAL - it belongs to the split, not to the zone).** L64's
observation that the record of the quiet zone sits in the bottom stratum holds at `q = 5` tier 3
(122 against 74, 75, 64, ...) and **fails** at `q = 7` tier 3 (23 in the bottom stratum against a
maximum of 35 in the sixth). Mechanism: the bottom stratum is family-starved only while the smooth
list is thin; at the raised split it carries 1,052 open pairs with a smooth member against 123, so
it is no longer the sparse stratum. *Evidence: 4 machines, full strata tables.*

**X23 (W1 TRANSFERS EXACTLY, AND ITS EXCEPTION IS EXACTLY GEAR 7).** `top_machine_1.md`'s
measured fact W1 - the counts of gap 3 and gap 5 are equal in a wheel whose gears all exceed 7,
and unequal when 7 is a gear - holds exactly at every tier-3 wheel: 12,986 against 12,986 at
`q = 5`, 1,292 against 1,292 at `q = 7`, 2 against 2 at `q = 11`; and fails on the tier-2 control
wheel `{7, 11, 13}`, 32 against 34. *A wheel-level fact with no mechanism, now with three more
confirming instances and the same single cause.*

**X24 (A TIER'S FAMILY `(1, 1)` DOES NOT DEPEND ON THE SPLIT).** The twin-prime count of the
bottom stratum `(Q, 2Q]` is identical at `q = 5` and `q = 7` for the same `Q` (25 at `Q = 997`,
64 at `Q = 3,137`), while the count of open pairs with a smooth member is not (86 against 528,
123 against 1,052). *This is L61's `q`-independence read in the exhaust's coordinate: raising the
split adds families and changes none, and the family that carries the record's obstruction is the
one that never changes.*

---

## 5. The exhaust's open items, for the ledger

**Closed here.**
- **O-X1 (`CutMono`)**: proved for every prime `q >= 5` in the strong form `cut_{k+1} > 4 cut_k`
  (X12), with the exact cause of the `q = 2, 3, 4` degeneracy (X13). Ready for transcription;
  mathlib's `Nat.exists_prime_lt_and_le_two_mul` is the only import needed beyond arithmetic.
- **O-X2 (the zones and the redundancy lemma)**: stated in the stack's own coordinate and proved
  (X14, X15, X16), with an exhaustive check. The range form is strictly stronger than the window
  form already in the kernel (single inequality, no primality).
- **O-X4 (the tower's own record)**: stated (X21) and classified ROOT, with numbers, so it cannot
  be rediscovered as an open item.
- **X1 (self-similarity)** upgraded from reasoning to measurement (X9), with one sharpening of the
  wheels' own law register (X10).

**Measurement with no structural content.** The digit count of `cut_3` (93,651,247 at `q = 5`) and
the `theta`-only estimates above; the strike census tables. They are the object's scale, not its
shape.

**The root question in disguise.**
- **O-X3 (no in-use theory)** stands, and X21 says exactly why: an upper bound on any tier's quiet
  zone record is a lower bound for the family `(1, 1)`, the twin primes above that tier's top
  gear. Same instrument at every rung, as X7 said; now with numbers.

**Genuinely open on the exhaust alone.**
- **O-X5 (prior art)** stands. Nothing here was found in the literature under the stack framing,
  but no systematic search was run; the pieces met are Bertrand (Chebyshev 1852), Chebyshev's
  `theta`, Stormer 1897 (finiteness of smooth pairs), and Eratosthenes/Legendre (the cap). X10 and
  X13 are the two items most likely to be genuinely new and the two cheapest to check.
- **New: O-X6, the crossover height.** X20 says a tier's record becomes the smooth-zone record
  above a crossover in `Q` set by the tier's smooth-pair list; the exact crossover is not in closed
  form. Statement: find `Q*(C)` such that for `Q > Q*(C)` the largest gap of the `C`-smooth pair
  list below `Q` exceeds the record of `(Q, Q^2]`. The left side is computable from the finite
  list; the right side is a prime-gap quantity, so an exact `Q*` is again a prime-gap statement -
  but an upper bound for `Q*` needs only an **upper** bound on a prime gap, which is available in
  the literature, unlike the twin case. *This is the one item in the exhaust where the missing
  instrument is a known theorem rather than the conjecture.*

---

## 6. What is new

1. **The exhaust has numbers.** Before this branch the object had none. Sections 3.1-3.6 are its
   first measurement: 18,095,756 residues over full periods, 21,669,556 cells for the zone rule,
   7,357,725 strikes for the redundancy lemma, 57,344 window incidences, 9 record machines.
2. **`CutMono` is a theorem** (X12), in a strong form, from Bertrand alone, with the degeneracy
   explained by an exact inequality on `prod_{p <= q} p` (X13). The kernel hypothesis can be
   discharged.
3. **The redundancy lemma in the range coordinate** (X15) is stronger than the window form in the
   kernel: one inequality `g^2 > N`, no primality, and it covers the whole range at once.
4. **X10**: the run-start count of L10 is false at `L = 1` and the chain-start count is not - a
   correction to the wheels' own law register, found by exercising it at a raised split.
5. **X17**: the exhaust's whole action on the window's open pairs is two home strikes per pair.
   The cap says the exhaust does no harm; this says what it does instead, exactly.
6a. **X23**: W1, the wheels' one unexplained gap-census fact, transfers verbatim to tier 3 and its
   exception is exactly the gear 7 - three new instances, no new mechanism.
6. **X18/X19**: the exhaust's action is graded by depth `floor(log_{p_1} x)`, starts at `p_1^2` and
   becomes the majority of the work one decade above the window - so "nothing above the wheels
   matters" is true precisely on `(Q, Q^2]` and false immediately after, with the numbers.
7. **X20/X22/X23**: the split, not the zone, decides where a tier's record lives and whether the
   U-profile holds; but the family that carries the obstruction - twin primes above `Q` - is
   exactly the one the split cannot touch.

Not new, cited in a line each: Bertrand's postulate; Chebyshev's `theta(x) ~ x`; Stormer's theorem
on the finiteness of smooth pairs; the sieve to the square root (which X15 is, in range form);
Mertens (not used here).

---

## 7. Verdict

**The exhaust is now an investigated object, and it is more completely understood than the
wheels.** Its structure laws (X9, X11) are measured with 0 exceptions, its arithmetic
(`CutMono`, X12) is proved rather than assumed, its action on a range (X14-X16) is exact and
closed-form, and its action relative to the window (X17-X19) is a census with 0 exceptions inside
the window and a measured share outside it.

**Node status: FACT for the structure and the arithmetic; ROOT for the record.** Nothing in the
exhaust bounds a twin-free run, and X21 says why in the object's own terms: a tier's record on its
own quiet zone is bounded below by prime gaps and above by nothing short of a twin-prime density
statement. The exhaust does not move the wall; it maps its own face of it precisely, and it hands
back one item (O-X6) whose missing instrument is a **known** theorem rather than the conjecture.

**Gate consequence.** Of the exhaust's five open items, three are closed here (O-X1, O-X2, O-X4),
one is the root question named (O-X3, and the ledger already rules it out as a gate item), and one
is a prior-art check (O-X5). On the ledger's own criterion the exhaust no longer has an open
structural item.

---

## 8. Dead ends

- **The record as a smooth-zone object at every split.** Refuted at `q = 7` tier 3 by `N = 10^8`:
  the record is 55 in the quiet zone against a smooth-zone record of 40. What survived is X20, the
  regime statement with the crossover, which is the honest form.
- **The U-profile of the quiet zone as a law.** Refuted at `q = 7` tier 3: strata records 23, 20,
  21, 27, 30, **35**, 33, 32, 30, 31, 29, 31 - the maximum is in the sixth stratum, not the first.
  What survived is X22, which names the mechanism (the bottom stratum's family starvation is a
  property of a thin smooth list, not of the zone).
- **Enumerating tier 3 at `q = 7`.** Not a dead end but a dead end for enumeration: `cut_2` has 80
  digits, so tier 3 there has no listing. The loaded exhaust on a range is the object, and because
  every law is in intrinsic parameters nothing is lost by working with it.
- **An instrument error recorded so it is not repeated.** Counting step-2 chains on a full period
  by splitting into parity classes is wrong (the period is odd, so doubling is one full cycle);
  it manufactured `q' - 3` false exceptions at every wheel, including the control.
