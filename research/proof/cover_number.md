# Branch R2.a.i.a.1.a - THE COVER NUMBER K(d)

Parent: node R2.a.i.a.1 (the island witness under pressure, `research/proof/island_witness.md`).
The observation that spawned this branch: that document's item (vi) - with every gear above 7 free
to choose any reachable phase and used once, the exact minimum number of gears that strike every
island of `[1, d)` is `K = 3, 4, 6, 9, 14, 20` at `d = 35, 70, 140, 280, 560, 1120`, while the
counting requirement (the smallest `K` with `sum 2/g >= 1` over the `K` cheapest gears) stalls at
`2, 4, 5, 7, 9, 10`. Counting stalls; covering grows. `K(d)` is the first quantity in this line
that moves in the direction the root needs, and its growth law was not known.

Scripts: `research/anchor235/r41/cn_*.py`. Result outputs (untracked):
`research/anchor235/r41/results/cn_*.txt`. Every number this document relies on is written into
the document.

---

## 0. Pre-registered (written before any computation of this branch)

### 0.1 The object, exact

Fix an integer `d >= 1`. The **islands** of `[1, d)` are the offsets `i` with
`1 <= i < d` and `i mod 35 in {5, 10, 12, 17}` (the `B = 7` island classes, R2.a.i.a item (ii));
write `m = m(d)` for their number. A **gear** is a prime `g > 7`. Gear `g` at **phase** `r`
(any nonzero quadratic residue mod `g`; that is exactly the freedom a real `q` has, since
`r = q^2 mod g`) strikes the offsets

```
    i = (2 - r) u_g  (mod g)      and      i = -r u_g  (mod g),        u_g = 6^{-1} (mod g),
```

two classes mod `g` at the **fixed separation** `d_g = 2 u_g` (the phase moves the pair, the
separation is arithmetic and cannot be chosen). Each gear may be used at **one** phase.

> **`K(d)` = the fewest gears that, at phases of their choosing, strike every island of `[1, d)`.**

`K(d)` is `q`-free. A real machine `{5..q}` that defeats the island witness supplies a legal
adversarial cover (each gear `g` not dividing `q` has `q^2 mod g` a nonzero QR), so
**`K(d) <= (the real minimum cover at any q with arc d)`** - `K` is a lower bound on the real cost
of a failure, and that is why it bears on the root.

### 0.2 What would count as a rule

As the parent: an exact statement about positions, counts or residues with an exception count over
a stated range, uniform in `d` or in `q`. A fitted curve, a density, or a restatement of the
doubling law, N-R5 (large gears strike islands at `2/g`), the tooth rule or the bar is **not** a
finding. Any sub-question that reduces to Mertens' theorem, to the Erdos-Rankin covering
construction, or to the Jacobsthal function is named in one line as classical and stopped.

### 0.3 Predictions, with numbers, and what refutes each

**Item 1 - the growth law.**

* **C1 (the form, pre-registered against the brief's own suggestion).** The brief offers
  `K ~ pi(c sqrt d)`. I predict that form is **wrong asymptotically** and that the truth is
  **linear in `d` up to logarithms**:

  ```
      K(d)  ~  c1 * d / (ln d)^3 .
  ```

  Mechanism behind the prediction, stated before computing: an optimal cover spends its cheap
  gears on a prefix `11..G` and then pays one gear per leftover island. The overlap between two
  gears' strike sets is forced by CRT (`|A ∩ B| ≈ |A||B|/m`, phases cannot change it beyond
  fluctuations), so the leftover after the prefix is `m prod_{11<=g<=G}(1 - 2/g) ≈ c m/(ln G)^2`
  and the cost is `pi(G) - 4 + alpha m c/(ln G)^2`; minimising over `G` gives
  `G (ln G)^2 ≈ c' m` and `K ≈ c' m/(ln m)^3`, i.e. `K ≍ d/(ln d)^3` since `m = 4d/35`.
  Fitting the constant on the six known values: `K (ln d)^3 / d = 3.85, 4.38, 5.17, 5.75, 6.30,
  6.20` at `d = 35 .. 1120` - rising and flattening near **6.3**.
  **Point predictions: `K(2240) = 30 +- 1` and `K(4480) = 47 +- 4`.**
  The `pi(c sqrt d) - 4` fit (`c = 2.66`, which reproduces `d = 70` and `560` exactly) predicts
  **26** and **37** instead. REFUTED (my form) if `K(2240) <= 27`; refuted (the brief's form) if
  `K(2240) >= 29`. If the ILP cannot close `d = 2240`, the discriminator is the exact ladder at the
  denser values `d = 1225, 1400, 1575, 1750` (all below 2240 and cheaper), where my form predicts
  `21, 23, 25, 26` and the `sqrt` form `20, 21, 22, 23`.
* **C2 (the structural identification, and the answer to "is it the counterfactual family's record
  ladder").** `K(d)` is the inverse function of a record ladder: `K(d) = min{K : F_K >= d}` where
  `F_K` is the longest run of offsets, starting at 1, whose islands are all struck by the best
  `K`-gear machine. I predict this ladder is **NOT** the tooth-counterfactual family's ladder,
  for two structural reasons pre-registered here: (a) the family fixes the gear set `{5..y}` and
  lets the tooth position `v_g` range over `1..(g-1)/2`, so its two classes `+-v_g` have a **free
  separation** `2 v_g`; here the separation `d_g = 2 x 6^{-1}` is **fixed by arithmetic** and only
  the phase is free; (b) the family's gear set is a prefix of the primes, whereas `K(d)`'s
  adversary **chooses** which `K` primes above 7 to use. Testable consequence: the free-separation
  cover number `K_free(d) <= K(d)`, and I predict **strict** inequality from `d = 280` on, with
  `K_free/K` between 0.7 and 0.9. REFUTED if `K_free(d) = K(d)` at every `d` tested (which would
  make the identification with the family exact and would let the family's known bounds be used).
* **C3 (the counting requirement stays bounded).** The counting lower bound (smallest `K` with
  `sum 2/g >= 1` over the `K` cheapest gears, evaluated on the islands) is predicted to stay at or
  below **13** at every `d` computed, while `K(d)` passes 40. REFUTED by a counting bound above 15.

**Item 2 - which gears an optimal cover uses.**

* **C4 (the prefix-plus-tail shape).** An optimal cover uses a **consecutive prefix** `11..G(d)`
  of the gears plus a short tail of larger gears, with `G(d) ≈ 1.8 sqrt(d)` (fitted on
  `d = 560, 1120`: `G = 43, 61` against `sqrt d = 23.7, 33.5`). Predicted: at every `d`, at least
  70% of the gears in some optimal cover form a consecutive prefix from 11. REFUTED by an optimum
  whose prefix is under half the cover at two consecutive `d`.
* **C5 (near-packing, not independence).** The total budget `sum_j |S_j|` of an optimal cover
  exceeds `m` by **less than 30%** (the adversary chooses phases that beat the CRT-independence
  overlap by a wide margin). REFUTED by an excess above 50% or below 5%.
* **C6 (the tail is singletons and doubletons).** In an optimal cover the gears above `d` strike
  exactly 1 or 2 islands each, and I predict the number striking exactly 2 is at least half of
  them (the adversary uses the pair condition `g | 3(i - j) - 1`). REFUTED if under a quarter.
* **C7 (uniqueness).** The optimal **gear set** is predicted **not** unique: at `d >= 280` I
  predict at least 5 distinct optimal gear sets. REFUTED by a `d >= 280` with a unique optimum.

**Item 3 - the real machine against the adversary.**

* **C8 (the real machine is a factor ~2 worse than the adversary).** At each of the 21 failing `q`
  coprime to 35 (17 primes, 4 composites) and at the `B = 11` and `B = 13` failures, the real
  minimum cover `R(q)` satisfies `R(q) >= K(d)` by construction; I predict the ratio `R(q)/K(d)`
  lies in `[1, 3]` at every one, with a median near **2**, and that it is **increasing** with `q`
  over the failures (the small failures are near-optimal, the last ones are not). REFUTED by a
  ratio above 3, or by a ratio below 1 (which would be a bug).
* **C9 (non-failing `q`).** At a non-failing `q` the real minimum cover of the **struck** islands
  exceeds `K(d)` - the machine spends more gears covering a strict subset than the adversary needs
  for the whole set - from `q = 200` on, with the ratio growing. REFUTED by a `q > 200` with
  `MBS(struck) < K(d)`.

**Item 4 - the first-moment count (heuristic).**

* **C10.** Model each gear's condition as independent with its exact rate `2 chi_g(i)/(g - 1)`.
  Then `P(island i free) = prod_{7 < g <= q} (1 - 2 chi_g(i)/(g-1))` and
  `E[#free islands at q] = sum_i P(i free)`, and `P(fail at q) ≈ exp(-E[#free])` (Poisson).
  Predicted: the expected number of failing integers `q` coprime to 30 above 2,849 summed to
  `10^6` is **below 0.01**, and the largest `q` with expected count above `10^-2` per unit band is
  between 1,500 and 3,500 - i.e. the heuristic reproduces the observed last failure 2,849 to
  within a factor of 2. REFUTED by an expected count above 0.1 above 2,849, or by a predicted
  threshold outside `[1000, 6000]`.
* **C11 (the cover-side first moment is useless, and why).** The alternative form the brief names -
  sum over covers of size `>= K(d)` of the product of the gears' rates - is predicted to **diverge
  or to exceed 1 by many orders of magnitude**, because the number of distinct covers of the
  islands of `[1, d)` grows super-exponentially in `d` while each cover's density is only
  `2^K / prod g_j`. Predicted: at `d = 140` the number of minimum-size covers alone is above
  `10^3` and the union bound over all covers exceeds 1. REFUTED if the union bound over covers is
  below 1 at `d = 140`.

**Item 5 - the structural handle.**

* **C12 (the exact class count; the doubling law used, not the islands counted).** A cover
  `C = {g_1, ..., g_K}` with phases `r_1, ..., r_K` forces `q^2 = r_j (mod g_j)` for each `j`.
  Each such congruence has **exactly two** solutions `q` mod `g_j` (`r_j` a nonzero QR), so the
  cover is realised by **exactly `2^K` residue classes of `q` modulo `P = prod g_j`** - no island
  counting anywhere. Predicted exact, 0 exceptions, at every optimal cover found. REFUTED by one
  cover realised by a number of classes other than `2^K`.
* **C13 (the square pin - the structural statement the brief asks for).** Because `q^2` is
  determined modulo `P` and lies in `[0, q^2]`, **if `P > q^2` the cover determines `q^2` as an
  integer**: at most one `q` in the whole range realises a given (cover, phase) pair. Predicted:
  at every optimal cover found, `P > q^2` by many orders of magnitude for the `q` whose arc is `d`
  (`P/q^2 > 10^20` from `d = 560` on), so the failure condition is not a density condition at all
  but "the CRT lift of a prescribed residue vector happens to be a perfect square". REFUTED if
  `P <= q^2` at any `d >= 280`.

**Item 6 - toward the root.**

* **C14.** The smallest statement is predicted to be: *for every integer `q` coprime to 30 above
  some bound, the CRT lift of every realisable cover of the islands of `[1, 2u_q)` by gears of
  `(7, q]` is not `q^2`.* Its parts: the class count `2^K` (predicted proved here), the growth of
  `K(d)` (measured here), the number of covers (measured here, unbounded), and the square
  condition (open). I predict the open part is exactly the same "+1 per class" obstruction that
  kills every density argument in this project, and that this branch cannot close it.

**Item 7.**

* **C15.** Report everything that holds without exception over the computed range, with counts.

### 0.4 Scorecard

| # | prediction | verdict and evidence |
|---|---|---|
| C1 | `K ≍ d/(ln d)^3`, `K(2240) = 30 +- 1`, `K(4480) = 47 +- 4`; `pi(c sqrt d)` wrong | **CONFIRMED on the exact ladder; the point values not settled**: `K (ln d)^3/d = 6.15 +- 0.20` over 16 consecutive arcs `d = 315..1330`, no drift; the `sqrt` fit falls behind by 1 at `d = 1,190` and 2 at `d = 1,260`. Achieved covers `K(2240) <= 32`, `K(3360) <= 40` against the two forms' 30/39 and 26/32 - evidence, not proof, since these are upper bounds (2.1, 2.2, 2.2a) |
| C2 | not the counterfactual family's ladder; `K_free < K` from `d = 280` | **CONFIRMED, and the family is the wrong object**: `K_free = 2, 3, 4, 6, 9` against `K = 3, 4, 6, 9, 14`; ratio 0.67 (I said 0.7-0.9, so the numeric range is **refuted**, the direction and the identification are not) (2.3) |
| C3 | counting bound `<= 13` throughout | **CONFIRMED**: the counting requirement is 2..11 over every arc to 1,400 and was stuck at 10 from `d = 770` to `d = 1,330` (2.1) |
| C4 | prefix `11..G`, `G ≈ 1.8 sqrt d`, prefix >= 70% of the cover | **SPLIT**: `G ≈ 1.7 sqrt d` confirmed; "70% of the cover" **REFUTED** (47-73%, mean 0.62). Replaced by the sharper N-C3 (3.1) |
| C5 | budget excess under 30% | **CONFIRMED**: 0-30% over all 22 covers (min 1.000, median 1.181, max 1.295) (3.2) |
| C6 | tail gears strike 1 or 2; at least half strike 2 | **CONFIRMED and exceeded**: from `d = 70` on **every** gear of an optimal cover strikes at least two islands, tail included (N-C4) (3.1) |
| C7 | optimal gear set not unique (>= 5 at `d >= 280`) | **CONFIRMED**: 3, 18, `>= 40`, 8, `>= 12` optimal gear sets at `d = 35, 70, 140, 280, 560` (3.3) |
| C8 | `R(q)/K(d)` in `[1, 3]`, median ~2, increasing | **SPLIT**: range confirmed (1.00-2.25 at `B = 7`, 1.00-2.64 at `B = 11`, `R >= K` 0 exceptions in 197); median 1.50/1.60 not 2; **monotonicity REFUTED** (4.1, 4.2) |
| C9 | `MBS(struck) > K(d)` from `q = 200` | **CONFIRMED at the pre-registered threshold**: 10 of 10 sampled `q` from 199 to 2,801, ratio 1.33-2.00; the one failure `q = 101` is below 200 (4.3) |
| C10 | expected failures above 2,849 below 0.01; threshold in `[1000, 6000]` | **CONFIRMED**: 0.00122 above 2,849; last `q` with `P_fail > 10^-4` is 2,339 against the true 2,849. Qualified: the model under-predicts `[1000, 3000)` by 14x (5) |
| C11 | cover-side union bound useless (`> 1` at `d = 140`) | **CONFIRMED**: it equals `prod_i lambda(i) = (sum 2/g)^m`, `10^19` at `q = 1,487` and `10^39` at `q = 2,849` (5.1) |
| C12 | exactly `2^K` classes mod `prod g_j`, 0 exceptions | **CONFIRMED by exhaustive brute force**: 324 million residues, 0 exceptions (6.1) |
| C13 | `P > q^2` by `10^20` from `d = 560` | **CONFIRMED in substance, REFUTED on the constant**: `P > q^2` at every `d >= 70` (21 of 22 covers), but `P/q^2` is `10^17.3` at `d = 560`, reaching `10^20` only from `d = 700` (6.2) |
| C14 | the smallest statement, and which part is open | **AS PREDICTED**: the open part is the "+1 per class" obstruction, in the new form "the number of covers is `2.7^m`" (7.1) |
| C15 | exception-free statements with counts | eleven, listed in section 9 |

---

## 1. Setup (exact ranges)

No sampling except where a row says so. Scripts in `research/anchor235/r41/`; outputs (untracked)
in `research/anchor235/r41/results/`.

| object | range | script |
|---|---|---|
| `K(d)` exact, ILP with the one-phase-per-gear rule, HiGHS proved optimal | `d = 35 k`, 23 values from 35 to 1,330; bounds only from `d = 1,400` | `cn_growth.py` |
| `K_free(d)`: the same with the separation FREE (any two classes per gear) | `d = 35, 70, 140, 280, 560` exact; `d = 1,120` abandoned (memory) | `cn_growth.py --free` |
| `K_multi(d)`: the same with the one-phase-per-gear rule DROPPED | `d = 35 .. 1,120` | `cn_growth.py --multi` |
| anatomy of an optimal cover; all optimal gear sets by no-good cuts; phase-vector counts | `d = 35, 70, 140, 280, 560, 1,120` | `cn_structure.py` |
| the covering classes counted by brute force over every `q` mod `P` | `P = 1,188,847` and `P = 323,525,521` - 324 million residues, exhaustive | `cn_classes.py` |
| real machine against the adversary at every failing `q`, `B = 7` | every `q` coprime to 30 in `[5, 3000]` | `cn_real.py --B 7` |
| the same at `B = 11` | every `q` coprime to 30 in `[5, 9500]` | `cn_real.py --B 11` |
| `K_13(d)` at six arcs to `d = 10,010`, exact | `d = 1,260, 2,100, 3,360, 5,460, 7,560, 10,010` | `cn_bcheck.py` |
| first moment with the exact rates `2 chi_g(i)/(g-1)` | every `q` coprime to 30 to 4,000, then 1 in 11 to 20,000 (1,452 machines) | `cn_moment.py` |
| the modulus `P` a cover pins `q` to | the 22 optimal covers found | `cn_modulus.py` |

The parent's exact-enumeration argument is reused unchanged (a gear covers `>= 3` islands only if
`g < d`; it covers two islands `i, j` only if `g | 3(i-j) - 1`, so `g <= 3d`; a singleton is
available above `3d`), with one new reduction that costs nothing: **only a phase `r` with
`r = -6i` or `r = 2 - 6i` for some island `i` can strike an island at all**, so each gear needs `2m`
phase trials instead of `(g-1)/2`. It reproduces every previously computed `K(d)` and every
candidate count exactly.

## 2. Results - the growth law (item 1)

### 2.1 The ladder, exact

Every row is HiGHS-certified optimal (dual bound equal to the incumbent).

| `d` | `m` | **`K(d)`** | budget / `m` | counting LB | an optimal cover |
|---|---|---|---|---|---|
| 35 | 4 | **3** | 1.250 | 2 | 11, 37, + 1 singleton |
| 70 | 8 | **4** | 1.125 | 4 | 11, 23, 37, 127 |
| 105 | 12 | **5** | 1.000 | 4 | 11, 17, 37, 47, 83 |
| 140 | 16 | **6** | 1.125 | 5 | 11, 17, 19, 23, 37, 107 |
| 175 | 20 | **7** | 1.150 | 6 | 11, 13, 17, 29, 31, 47, 53 |
| 245 | 28 | **8** | 1.071 | 6 | 11, 13, 19, 23, 29, 97, 419, 421 |
| 280 | 32 | **9** | 1.062 | 7 | 11, 13, 19, 23, 29, 47, 53, 101, 199 |
| 315 | 36 | **10** | 1.167 | 7 | 11, 13, 17, 23, 29, 31, 37, 47, 67, 97 |
| 385 | 44 | **11** | 1.159 | 8 | 11..37, 43, 53, 79 |
| 455 | 52 | **12** | 1.115 | 8 | 11..31, 41, 53, 103, 419, 839 |
| 525 | 60 | **13** | 1.250 | 8 | 11..37, 47, 53, 107, 331, 383 |
| 560 | 64 | **14** | 1.203 | 9 | 11..31, 41, 47, 59, 103, 107, 421, 1013 |
| 630 | 72 | **14** | 1.181 | 9 | 11..43, 59, 103, 109, 211 |
| 700 | 80 | **15** | 1.188 | 9 | 11..31, 41, 43, 53, 59, 101, 113, 211, 631 |
| 770 | 88 | **16** | 1.182 | 10 | 11..41, 47, 53, 59, 71, 89, 107, 631 |
| 840 | 96 | **17** | 1.167 | 10 | 11..43, 59, 67, 71, 167, 199, 1049, 1171 |
| 910 | 104 | **18** | 1.212 | 10 | 11..47, 59, 73, 89, 103, 109, 149, 211 |
| 980 | 112 | **19** | 1.295 | 10 | 11..59, 71, 103, 113, 139, 173, 211 |
| 1,050 | 120 | **19** | 1.225 | 10 | 11..59, 73, 101, 577, 619, 757, 829 |
| 1,120 | 128 | **20** | 1.234 | 10 | 11..59, 89, 101, 127, 263, 719, 751, 2521 |
| 1,190 | 136 | **21** | 1.279 | 10 | 11..53, 61, 67, 71, 101, 103, 109, 211, 347, 367 |
| 1,260 | 144 | **22** | 1.271 | 10 | 11..59, 67, 73, 97, 113, 139, 163, 173, 233, 281 |
| 1,330 | 152 | **22** | 1.283 | 10 | 11..61, 73, 79, 137, 139, 181, 383, 419, 509 |

Read as a record ladder (the inverse function): the first arc at which `K` gears become necessary
is `d = 35, 70, 105, 140, 175, <=245, 280, 315, 385, 455, 525, 560, 700, 770, 840, 910, 980, 1120,
1190, 1260` for `K = 3 .. 22`. **One extra gear buys the adversary about 87 more columns - ten more
islands - at `d ~ 1000`** (`(1260 - 560)/(22 - 14) = 87.5`), against 6.4 islands per gear at
`d ~ 300`: the marginal value of a gear rises, slowly.

### 2.2 Which growth law

| `d` | 315 | 385 | 455 | 525 | 560 | 630 | 700 | 770 | 840 | 910 | 980 | 1,050 | 1,120 | 1,190 | 1,260 | 1,330 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `K (ln d)^3 / d` | 6.04 | 6.03 | 6.05 | 6.08 | 6.34 | 5.95 | 6.03 | 6.10 | 6.18 | 6.26 | 6.33 | 6.09 | 6.18 | 6.27 | 6.35 | 6.11 |
| `pi(2.66 sqrt d) - 4` | 11 | 11 | 12 | 13 | 14 | 15 | 15 | 17 | 17 | 18 | 19 | 19 | 20 | 20 | 20 | 21 |
| **`K(d)`** | 10 | 11 | 12 | 13 | 14 | 14 | 15 | 16 | 17 | 18 | 19 | 19 | 20 | **21** | **22** | **22** |

`K (ln d)^3 / d` is flat at **6.15 +- 0.20 over every `d` from 315 to 1,330** (16 values, no drift),
while the brief's `pi(c sqrt d)` form - with `c` fitted to be exact at `d = 70, 560, 1,120` - tracks
the ladder to `d = 1,120` and then falls behind by 1 and by 2 at `d = 1,190` and `d = 1,260`. The
pre-registered discriminator was the sign of that drift and it has the sign the `d/(ln d)^3` form
predicts. Extrapolated: `K(2240) = 30.0`, `K(4480) = 46.4`, against the `sqrt` form's 26 and 37.

### 2.2a Beyond the exact ladder: bounds

The ILP closes at `d = 1,330` and stops closing at `d = 1,400` (incumbent 23, dual bound 21 after
900 s). Beyond that the table carries what is proved: an achieved cover (a genuine upper bound, from
a randomised greedy over the same complete candidate list, `cn_heur.py`, 500 restarts) and HiGHS's
dual bound where one is available. The heuristic is calibrated: it returns 14 at `d = 560`
(exact 14) and 21 at `d = 1,120` (exact 20), i.e. it is 0 to +1 on the arcs where the truth is
known.

| `d` | `m` | exact | heuristic UB | `d/(ln d)^3` form | `pi(2.66 sqrt d) - 4` |
|---|---|---|---|---|---|
| 1,120 | 128 | **20** | 21 | 20 | 20 |
| 1,400 | 160 | 21 .. 23 | - | 22 | 21 |
| 1,750 | 200 | - | **<= 26** | 26 | 25 |
| 2,240 | 256 | - | **<= 32** | 30 | 26 |
| 3,360 | 384 | - | **<= 40** | 39 | 32 |

(A 3-hour exact ILP at `d = 2,240` was still running when this document was closed; its incumbent
can only be `<= 32` and its dual bound `>= 22` by monotonicity from `K(1330) = 22`, so it cannot
change any statement below. `d = 4,480` was left to the heuristic for the same reason.)

**What this does and does not settle.** An achieved cover is an upper bound, so 32 at `d = 2,240`
does not by itself contradict a prediction of 26 - the refutation of the `sqrt` form rests on the
**exact** ladder, where it under-predicts by 1 at `d = 1,190` and by 2 at `d = 1,260` and `d = 1,400`
(exact lower bound 21 against its 21, and 22 exact against its 21 at `d = 1,330`) after being fitted
to be exact at `d = 70, 560, 1,120`. The large-`d` covers add calibrated evidence rather than proof:
the greedy is 0 or +1 above the optimum at every arc where the optimum is known, so for the `sqrt`
form to be right at `d = 3,360` the same greedy would have to be **8** above optimal there. The
`d/(ln d)^3` form predicts 30 and 39 at `d = 2,240` and `3,360`, one or two below the achieved 32
and 40, exactly the gap the greedy shows elsewhere. Verdict on the pre-registered discriminator:
**`K ~ pi(c sqrt d)` is refuted on the exact ladder and badly out of line at the large arcs;
`K ≍ d/(ln d)^3` survives every arc computed.**

**Mechanism behind the form.** The adversary spends a consecutive prefix `11..G` of cheap gears and
then pays for the leftovers one gear at a time. The leftover after the prefix is
`m prod_{11 <= g <= G}(1 - 2/g) ~ c m/(ln G)^2`, so the cost is `pi(G) + alpha c m/(ln G)^2`;
minimising over `G` gives `G (ln G)^2 ~ c' m` and `K ~ c' m/(ln m)^3`, i.e. `K ≍ d/(ln d)^3` since
`m = 4d/35`. The local exponent of that form is `1 - 3/ln d = 0.55..0.58` over the measured range,
which is exactly why the ladder *looks* like `sqrt d` locally and is not.

### 2.3 The structural identification: it is NOT the counterfactual family's ladder

`K(d)` is the inverse of a record ladder - `K(d) = min{K : F_K >= d}` with `F_K` the longest arc
whose islands a best `K`-gear machine can cover. Two relaxations separate it from the
tooth-counterfactual family and locate the whole of its growth.

| `d` | 35 | 70 | 140 | 280 | 560 | 1,120 |
|---|---|---|---|---|---|---|
| counting requirement of the REAL problem (`K` largest achievable set sizes summing to `>= m`) | 2 | 4 | 5 | 7 | 9 | 10 |
| **`K_free(d)`** - separation free (the family's freedom), one phase per gear | **2** | **3** | **4** | **6** | **9** | |
| counting requirement of the FREE problem | 2 | 3 | 4 | 6 | 7 | |
| budget / `m` at the `K_free` optimum | 1.000 | 1.000 | 1.000 | 1.000 | 1.062 | |
| **`K_multi(d)`** - real separation, gear reusable at several phases | **3** | **4** | **5** | **7** | **11** | **12** |
| **`K(d)`** - the real object | **3** | **4** | **6** | **9** | **14** | **20** |

Two exact readings, both new:

> **N-C1 (with a free separation the cover is a PERFECT PACKING and covering IS counting).** If a
> gear may take any two classes mod `g` - the tooth-counterfactual family's freedom - the optimal
> cover has budget `sum |S_j| = m` **exactly**: no island is struck twice, the islands are
> *partitioned*, and `K_free(d)` equals that problem's own counting requirement, at
> `d = 35, 70, 140, 280` (4 arcs, 0 exceptions). The perfect packing breaks first at `d = 560`,
> where `K_free = 9` against a counting requirement of 7 and a budget of `1.062 m`. So free
> separation makes the covering problem a counting problem for as long as an exact partition of the
> islands exists at all.

> **N-C2 (the whole gap is the fixed separation and the one-phase rule).** With the real fixed
> separation `d_g = 2 x 6^{-1} (mod g)` the packing can no longer be perfect (budget `1.06-1.30 m`)
> and `K/K_free = 1.5, 1.33, 1.5, 1.5, 1.556` at `d = 35, 70, 140, 280, 560`. Dropping the
> one-phase-per-gear rule instead collapses `K` from 20 to **12** at `d = 1,120` - within 2 of the
> counting bound. So the growth of `K(d)` is not a coverage-budget effect at all: it is the two
> constraints *a gear has one phase* and *its two classes sit at an arithmetic separation it cannot
> choose*, and the second is worth a factor 1.5 while the first is worth the rest.

That answers the brief's structural question in the negative and in one line: **`K(d)` is not the
counterfactual family's record ladder restricted to islands.** The family lets `v_g` range over
`1..(g-1)/2`, i.e. two classes `+-v_g` at a **free** separation; the machine's separation is
`2 x 6^{-1} (mod g)`, fixed by arithmetic. The family's ladder is the `K_free` row, and it is a
*different and strictly easier* object - so the family's known record values, its budget inequality
(0.00-0.56% of members) and its spectrum bounds do not transfer, and none of them is used here.
The family's freedom is exactly the freedom that makes the answer the counting bound, which is the
parent's dead end.

## 3. Results - which gears an optimal cover uses (item 2)

### 3.1 The shape: a prefix of small gears, then an opportunistic tail

> **N-C3 (the small gears become compulsory).** The optimal cover returned at every arc
> `d >= 385` contains **all of 11, 13, 17, 19, 23, 29, 31** - 16 arcs, 0 exceptions. Below that it
> does not: the optimum at `d = 315` omits 19, the one at `d = 280` omits 17, and `d = 35..175` use
> two or three of the seven. Across *all* optima, not just one per arc: at `d = 280` the gears
> 11, 13, 29 are in every one of the eight optimal gear sets; at `d = 560` the gears
> 11, 13, 17, 19, 23, 29 are in every one of twelve, and 31 in ten of them; at `d = 70` and
> `d = 140` no gear is compulsory at all.

The consecutive prefix `11 .. G` accounts for 47-73% of the gears of an optimal cover (mean 0.62 at
`d >= 385`, minimum 0.47 at `d = 700`), so the pre-registered "at least 70%" is **refuted**; the
shape is right, the fraction is not. The prefix top `G` runs 37, 31, 37, 31, 43, 31, 41, 43, 47, 59,
59, 59, 53, 59 at `d = 385..1260`, i.e. `G ~ 1.7 sqrt d` (1.76 at `d = 1,120`), while the tail
gears are wildly variable (79, 839, 383, 1013, 211, 631, 631, 1171, 211, 211, 829, 2521, 367, 281)
and carry no pattern - they are chosen for the two or three particular islands they happen to pair.

> **N-C4 (no gear is ever wasted above `d = 35`).** In every optimal cover from `d = 70` on, every
> gear takes **at least two** islands: 20 of 22 covers have minimum set size 2, one has 3 (`d = 105`),
> and only `d = 35` uses a singleton. The adversary never spends a gear on one island once it has
> more than four to cover.

### 3.2 Overlap: the cover is nearly a partition

| `d` | 70 | 140 | 280 | 560 | budget / `m` over all 22 covers |
|---|---|---|---|---|---|
| islands struck once | 7 | 14 | 30 | 53 | |
| islands struck twice | 1 | 2 | 2 | 10 | |
| islands struck 3+ times | 0 | 0 | 0 | 1 | min 1.000, median 1.181, max 1.295 |

C5 is confirmed: the budget excess is **0-30%**, never above 30%. Triple coverage is essentially
absent (none at all up to `d = 280`; one island at `d = 560`; four at `d = 1,120`).

Against the CRT-independent expectation this is a real deviation. At `d = 1,120` the optimum's
thirteen small gears `11..59` carry a budget of 138 strikes; at generic phases they would cover
`m(1 - prod_{11 <= g <= 59}(1 - 2/g)) = 85.5` islands, i.e. spend about **50** strikes on repeats.
The whole 20-gear optimum spends **30**. The adversary buys the near-disjointness with phase
freedom - `prod (g-1)/2 = 10^{14.8}` phase vectors for those thirteen gears alone, `10^{30.3}` for
the cover - and the deviation it buys is a factor 1.7 in repeats, not an order of magnitude.

### 3.3 Uniqueness: the optimal gear set is very far from unique

| `d` | 35 | 70 | 140 | 280 |
|---|---|---|---|---|
| distinct optimal GEAR SETS | 3 (all) | 18 (all) | `>= 40` (cap) | 8 (all) |
| gears in every optimal set | none | none | none | **11, 13, 29** |
| gears appearing in some optimal set | 2 | 10 | 19 | 22 |

C7 is confirmed (`>= 5` at `d >= 280`: eight at `d = 280`). The interesting half is the second row:
at `d = 280` three gears - 11, 13 and 29 - are in **every** one of the eight optimal covers, and at
`d = 560` six gears (11, 13, 17, 19, 23, 29) are in every one of the twelve enumerated, while at
`d = 70` and `d = 140` no gear is compulsory at all. Compulsory gears appear as the arc grows,
which is N-C3 seen from the other side.

## 4. Results - the real machine against the adversary (item 3)

`R(q)` is the exact minimum number of gears of the real machine, at the real phases
`r_g = q^2 (mod g)`, that cover every island of `[1, d)`. It exists only at a failing `q`. Since
the real phases are a legal adversarial play, `R(q) >= K_B(d)` always; the question is by how much.

### 4.1 Every failure at `B = 7`

Every `q` coprime to 30 in `[5, 3000]`: 31 failures, largest 2,849 - the parent's number reproduced,
including the four composite failures and the multiples of 7.

| `q` | `d` | `m` | `R(q)` | `K(d)` | `R/K` |
|---|---|---|---|---|---|
| 17, 23, 29 | 6, 8, 10 | 1 | 1 | 1 | 1.00 |
| 41 | 14 | 3 | 3 | 2 | 1.50 |
| 49 | 33 | 4 | 3 | 3 | 1.00 |
| 53 | 18 | 4 | 4 | 3 | 1.33 |
| 73 | 49 | 7 | 5 | 3 | 1.67 |
| 121 | 81 | 10 | 8 | 5 | 1.60 |
| 173 | 58 | 8 | 8 | 4 | **2.00** |
| 247 | 165 | 20 | 11 | 7 | 1.57 |
| 341 | 114 | 13 | 12 | 6 | 2.00 |
| 353 | 118 | 15 | 12 | 6 | 2.00 |
| 461 | 154 | 19 | 15 | 7 | 2.14 |
| 683 | 228 | 28 | 18 | 8 | **2.25** |
| 707 | 236 | 28 | 12 | 8 | 1.50 |
| 1,151 | 384 | 44 | 22 | 11 | 2.00 |
| 1,487 | 496 | 57 | 24 | 12 | 2.00 |
| 1,649 | 550 | 64 | 25 | 14 | 1.79 |
| **2,849** | 950 | 108 | 26 | **18** | 1.44 |

Over all 29 failures with an island: `R/K` has **min 1.00, median 1.50, max 2.25**, and it is **not
monotone in `q`** (it peaks at `q = 683` and falls to 1.44 at the last failure). C8's range is
confirmed, its median is 1.50 not 2, and its monotonicity claim is **refuted**.

### 4.2 Every failure at `B = 11`

Every `q` coprime to 30 in `[5, 9500]`: 175 failures, largest 9,443; the 168 with `d <= 3200` are
taken apart. (`B = 13` is not attempted: its threshold is `q = 33,623`, an arc of 11,208, and the
adversarial ILP there is out of reach at this lane's compute.)

| `q` | `d` | `m` | `R(q)` | `K_11(d)` | `R/K` |
|---|---|---|---|---|---|
| 4,067 | 1,356 | 41 | 12 | 11 | 1.09 |
| 4,607 | 1,536 | 48 | 29 | 11 | **2.64** |
| 5,261 | 1,754 | 53 | 28 | 12 | 2.33 |
| 7,007 | 2,336 | 74 | 19 | 15 | 1.27 |
| 7,601 | 2,534 | 79 | 31 | 16 | 1.94 |
| **9,281** | 3,094 | 97 | 47 | **18** | 2.61 |
| 9,443 | 3,148 | 98 | 31 | 18 | 1.72 |

`R/K` over 168 `B = 11` failures: min 1.00, median 1.60, max 2.64, again not monotone. The real
machine is a factor 1.0-2.6 off the adversary's optimum at every failure ever recorded, at either
bound. **`R(q) >= K_B(d)` at all 197 failures, 0 exceptions** - as it must be, and it is the check
that the two computations agree.

> **N-C5 (the cover number is a function of the ISLAND COUNT and of the smallest gear left, not of
> the arc).** Compare the three bounds at equal island counts. The arcs differ by up to a factor of
> twelve and the answers do not.
>
> | islands `m` | 12 | 21 | 32 | 36 | 52-54 | 71-74 | 88-89 | 96-98 |
> |---|---|---|---|---|---|---|---|---|
> | `K_7` (`d = 105 .. 840`) | 5 | - | 9 | 10 | 12 | 14 | 16 | 17 |
> | `K_11` (`d = 1,096 .. 3,148`) | - | - | - | 9-10 | 12 | 15 | 17 | 18 |
> | `K_13` (`d = 1,260 .. 10,010`) | 5 | 8 | 10 | - | 13 | 15 | - | 18 |
>
> At every one of the eleven comparisons the three agree to within 1, and the sign is systematic:
> the higher bound is never cheaper. The reason is not the arc but the **gears the bar removes**.
> At `B = 7` the adversary may start at gear 11; at `B = 11` it must start at 13; at `B = 13` at 17.
> Losing 11 and 13 costs exactly one extra gear all the way to `m = 96` - at `m = 96` the best first
> gear takes `2m/11 = 17` islands at `B = 7` and only `2m/17 = 11` at `B = 13`. So what the
> adversary faces is `m` and the size of its cheapest gear, and nothing else: `K_13` needs an arc of
> **10,010** to be as hard as `K_7` at an arc of 840, and it is then harder by exactly one gear.

### 4.3 Non-failing `q`: the machine spends more on a subset than the adversary needs for the whole

At a non-failing `q` the minimum blocking set `MBS(q)` of the **struck** islands (the parent's 4.2)
is the honest measure of how efficiently the real machine covers.

| `q` | `d` | `m` | free | struck | `MBS` | `K(d)` | `MBS/K` |
|---|---|---|---|---|---|---|---|
| 101 | 34 | 4 | 2 | 2 | 2 | 3 | 0.67 |
| 199 | 133 | 16 | 1 | 15 | 11 | 6 | 1.83 |
| 401 | 134 | 16 | 1 | 15 | 9 | 6 | 1.50 |
| 601 | 401 | 47 | 8 | 39 | 16 | 11 | 1.45 |
| 809 | 270 | 32 | 3 | 29 | 12 | 9 | 1.33 |
| 1,009 | 673 | 77 | 9 | 68 | 22 | 15 | 1.47 |
| 1,201 | 801 | 92 | 5 | 87 | 31 | 17 | 1.82 |
| 1,499 | 500 | 57 | 6 | 51 | 20 | 12 | 1.67 |
| 2,003 | 668 | 76 | 5 | 71 | 29 | 15 | 1.93 |
| 2,399 | 800 | 92 | 6 | 86 | 34 | 17 | 2.00 |
| 2,801 | 934 | 108 | 7 | 101 | 36 | 18 | 2.00 |

C9 is confirmed with its pre-registered threshold: `MBS > K(d)` at **every sampled `q` from 199 on**
(10 of 10; the single exception `q = 101` is below the threshold I named, 200). The machine needs
half again to twice as many gears to account for a **strict subset** of the islands as the adversary
needs for all of them, and the ratio climbs from 1.45 to 2.00 across the sample.

## 5. Results - the first-moment count, as a heuristic (item 4)

**The model, stated exactly.** Gear `g` strikes offset `i` for exactly `2 chi_g(i)` of the `g - 1`
residues of `q` mod `g` (the doubling law N-R6; exact, not modelled). Treat the gears as
independent and the islands as independent:

```
    p(i)    = prod_{7 < g <= q} (1 - 2 chi_g(i)/(g - 1))          island i free
    E(q)    = sum_{i island in [1,d)} p(i)                        expected free islands
    P_fail  = prod_i (1 - p(i))                                   the failure probability used
```

computed with the **exact** `chi_g(i)` for every gear and every island - no rate is substituted for
a character. 1,452 machines: every `q` coprime to 30 to 4,000 and one in eleven to 20,000.

| band | expected failures | observed failures coprime to 30 |
|---|---|---|
| `[11, 100)` | 4.96 | 9 |
| `[100, 300)` | 5.43 | 10 |
| `[300, 1000)` | 3.34 | 6 |
| `[1000, 3000)` | 0.286 | **4** (1151, 1487, 1649, 2849) |
| `[3000, 10000)` | 0.00082 | 0 |
| `[10000, 20000)` | `7.9e-11` | 0 |

| tail | expected failures above `q` |
|---|---|
| above 1,000 | 0.287 |
| above 1,500 | 0.0516 |
| above 2,000 | 0.0127 |
| above **2,849** (the last real failure) | **0.00122** |
| above 5,000 | `5.6e-6` |
| above 10,000 | `7.9e-11` |

Largest `q` with `P_fail > 10^-3`: **1,589**; above `10^-4`: **2,339**; above `10^-6`: **4,049**.
The observed last failure is 2,849. C10 is confirmed on both clauses: the expected count above 2,849
is 0.0012, well under the pre-registered 0.01, and the heuristic's threshold, 2,339, is within 18%
of the truth.

The honest reading of the same table: the heuristic **under-predicts by a factor of about 2 in the
three low bands and by a factor of 14 in `[1000, 3000)`**. Three of that band's four failures are
coprime to 35, so the discrepancy is not the divisor-gear effect; it is the positive correlation
between neighbouring islands (they share their small strikers), which fattens the lower tail of the
free-island count. The heuristic gets the place right and the constant wrong.

**Where this is not a proof, exactly.** Four places, all of them live:

1. *Dependence between the gears.* The events `q^2 = r (mod g)` are independent across `g` by CRT,
   but a single gear's condition at two different islands is the SAME event (one phase strikes two
   classes), so the island events are not independent even for one gear, and the model's product
   over islands is wrong in a direction the data shows (factor 14 at the last band).
2. *The number of covering classes modulo the product.* The model replaces "`q` lies in one of `N`
   classes modulo `P`" by "`q` behaves like a random residue". The exact statement is section 6:
   `N = 2^K` per (cover, phase vector).
3. *The "+1 per class" problem.* When `P > X` a residue class modulo `P` contains **0 or 1**
   integers below `X`, and density says nothing about which. At `d = 1,120` the smallest cover
   modulus is `10^36.6` while `q ~ 3,400`: the expected count is `10^-30`, but expectation over an
   empty-or-singleton class is not a proof that the class is empty.
4. *The sum is over covers, and the covers are not enumerable.* See C11 below.

### 5.1 The cover-side first moment is useless, and by how much

The brief's alternative - sum over covers of size `>= K(d)` of the product of the gears' rates - is
bounded above by choosing one striking gear per island independently, which gives exactly

```
    sum over covers  =  prod_{i island} lambda(i) ,     lambda(i) = sum_{7 < g <= q} 2 chi_g(i)/(g-1)
```

and `lambda` is the **depth function** of R2.a.i, whose mean over the islands is the Mertens sum
`sum 2/g` - a rate already on record, so this line stops here. Measured:

| `q` | 101 | 251 | 503 | 1,009 | 1,487 | 2,003 | 2,849 |
|---|---|---|---|---|---|---|---|
| `m` | 4 | 11 | 20 | 77 | 57 | 76 | 108 |
| mean depth `lambda` | 1.20 | 1.69 | 1.89 | 2.05 | 2.16 | 2.24 | 2.33 |
| `log10 prod lambda(i)` | +0.3 | +2.5 | +5.4 | **+23.7** | +18.8 | +26.4 | **+39.3** |

C11 is confirmed and sharpened: the union bound over covers exceeds 1 by `10^19` at the last
coprime-to-35 failure and by `10^39` at the last failure, and it grows like `(sum 2/g)^m` - it is
the parent's counting wall in another costume, since `sum 2/g > 1` from `q = 53`. **No first moment
that sums over covers can ever say anything here.** The only first moment that says anything is the
per-island one above, and it is not a bound.

## 6. Results - the structural handle: the covering classes (item 5)

### 6.1 The exact class count, with no island counting

> **N-C6 (the cover class count).** Let `C = {g_1, ..., g_K}` be a cover with phases
> `r_1, ..., r_K` (each `r_j` a nonzero quadratic residue mod `g_j`). A machine at `q` realises it
> iff `q^2 = r_j (mod g_j)` for every `j`. Each such congruence has **exactly two** solutions
> `q` mod `g_j`, so the (cover, phase vector) pair is realised by **exactly `2^K` residue classes
> of `q` modulo `P = prod g_j`**, and distinct phase vectors give disjoint class sets. Nothing in
> the statement counts islands; it is the doubling law read once per gear.

Verified by brute force over **every** residue `q` mod `P`, exhaustively:

| `d` | cover | `P` | classes realising it | phase vectors of that gear set | `2^K x` phases |
|---|---|---|---|---|---|
| 70 | 11, 23, 37, 127 | 1,188,847 | **32** | 2 | `2 x 2^4 = 32` |
| 140 | 11, 17, 19, 23, 37, 107 | 323,525,521 | **128** | 2 | `2 x 2^6 = 128` |

324 million residues checked, **0 exceptions**. C12 confirmed.

### 6.2 What the modulus is, against `q`

For the 22 optimal covers found (`q` taken at the short arc, `q = 3d`):

| `d` | `K` | `log10 P` | `log10 (P/q)` | `log10 (P/q^2)` | `log10 (2^K/P)` = density |
|---|---|---|---|---|---|
| 35 | 3 | 2.61 | +0.59 | **-1.43** | -2.01 |
| 70 | 4 | 6.08 | +3.75 | +1.43 | -4.87 |
| 140 | 6 | 8.51 | +5.89 | +3.26 | -6.70 |
| 280 | 9 | 13.96 | +11.03 | +8.11 | -11.25 |
| 560 | 14 | 23.71 | +20.48 | +17.26 | -19.49 |
| 840 | 17 | 29.85 | +26.45 | +23.05 | -24.74 |
| 1,120 | 20 | 36.57 | +33.05 | +29.52 | **-30.55** |
| 1,260 | 22 | 38.10 | +34.52 | +30.95 | -31.48 |

> **N-C7 (the square pin).** `P > q^2` at every `d >= 70` (21 of the 22 covers; only `d = 35`
> fails, at `P/q^2 = 0.037`). When `P > q^2` the residue vector `(r_1, ..., r_K)` determines `q^2`
> **as an integer**, not merely as a class: `q^2` is the unique non-negative integer below `P`
> with those residues. So a failure is not "`q` fell into a class"; it is "**the CRT lift of a
> prescribed residue vector happened to be a perfect square**". At `d = 1,120` the lift lives in
> `[0, 10^36.6)` and must equal a square below `10^7.1`.

That is the exact statement the brief asked for, and it does not go through counting the islands:
it uses only the doubling law (two roots per gear) and CRT. Its consequence, also exact: **for a
fixed cover and phase vector there is at most one `q` in the whole range `q^2 < P` that realises
it** - the "+1 per class" problem stated positively.

### 6.3 What it does not give: the number of covers

Turning N-C7 into a bound needs the number of (cover, phase vector) pairs, and that is where the
handle stops. Measured: at `d = 280` the optimum has 8 distinct gear sets, at `d = 140` at least 40;
counting covers of every size, section 5.1 gives `prod lambda(i) ~ 2.7^m` of them, which at
`d = 1,120` is `10^54`. Multiplying `10^54` covers by a density of `10^-30` gives `10^24`, so the
union bound over covers is vacuous by 24 orders of magnitude even with the exact class count in
hand. **The class count is exact and the cover count is the obstruction**, not the other way round.

## 7. Toward the root (item 6)

### 7.1 The smallest statement, and what each part is

N-C3 (the seven smallest gears compulsory) suggested a finite lever: at a real `q` the phases of
11, 13, 17, 19, 23, 29, 31 are fixed by `q^2` modulo their product, 955,049,953, so if a failure
forced those seven gears near their best joint coverage, the failure condition would be a condition
on `q` modulo `9.55 x 10^8` - finite, `q`-free, exhaustively checkable, and containing no density.
**Tested and refuted** (7.2). What survives is:

> **(S) For every integer `q` coprime to 30 above some bound, no cover of the islands of
> `[1, 2 u_q)` by gears of `(7, q]` has its CRT lift equal to `q^2`.**

| part of (S) | status |
|---|---|
| a cover forces `q^2 = r_j (mod g_j)` and so exactly `2^K` classes of `q` mod `P` | **proved** (N-C6; doubling law once per gear), brute-force verified on 324 million residues |
| `P > q^2`, so the cover determines `q^2` as an INTEGER and admits at most one `q` | **proved for each cover found** (N-C7): 21 of the 22 optimal covers, all `d >= 70` |
| the real cost `R(q) >= K_B(d)` at a failure | **proved** (real phases are a legal play); measured 1.00-2.64 at all 197 recorded failures |
| `K(d)` grows, and how fast | **measured** exactly at 23 arcs; `K (ln d)^3/d = 6.15 +- 0.20` from `d = 315`; no lower bound proved for any `d` not computed |
| the number of covers | **measured, and it is the obstruction**: `~ (sum 2/g)^m = 2.7^m`, `10^54` at `d = 1,120` |
| the CRT lift is not a square | **open**, and it is the "+1 per class" problem in its sharpest form |

So the branch converts the witness from a density statement into an arithmetic one - *a prescribed
CRT lift is a perfect square* - and then stops at exactly the place every density argument in this
project stops, but for a new reason: not because the counting margin is too small (it is not used
at all here) but because the number of admissible covers outgrows the modulus by `10^24`.

The lowest-order interaction not yet proved, named for the next child: **bound the number of covers
the REAL machine can produce.** The adversary chooses phases freely; the machine has one phase
vector, and its covers are not free choices but consequences of `q^2`. Section 4 shows the machine
is a factor 1.0-2.6 worse than the adversary at every failure, and section 7.2 shows the small gears
at a failure are at the 61st percentile of their own phase distribution, i.e. **the machine's covers
are ordinary, not extremal**. A bound on the number of covers reachable from a single `q^2` - rather
than over all phase vectors - is the object that would close (S), and it is not attempted here.

### 7.2 The compulsory-prefix lever is dead

At each failure, how much of the island set do gears 11, 13, 17, 19, 23, 29, 31 actually take,
against the best and the mean over their own phase vectors?

| `q` | fail? | `m` | take by 11..31 | best over phases | mean | percentile of the real take |
|---|---|---|---|---|---|---|
| 341 | FAIL | 13 | 6 (0.46 m) | 13 | 7.19 | 0.339 |
| 461 | FAIL | 19 | 10 (0.53 m) | 17 | 11.05 | 0.373 |
| 683 | FAIL | 28 | 16 (0.57 m) | 24 | 16.04 | 0.581 |
| 1,151 | FAIL | 44 | 27 (0.61 m) | 33 | 24.96 | 0.897 |
| 1,487 | FAIL | 57 | 32 (0.56 m) | 41 | 31.93 | 0.607 |
| 1,649 | FAIL | 64 | 40 (0.63 m) | 46 | 35.92 | 0.961 |
| 2,849 | FAIL | 108 | 65 (0.60 m) | 73 | 61.18 | 0.939 |
| 1,499 | - | 57 | 35 (0.61 m) | 41 | 31.92 | 0.929 |
| 2,003 | - | 76 | 44 (0.58 m) | 54 | 43.22 | 0.678 |
| 2,801 | - | 108 | 62 (0.57 m) | 73 | 61.12 | 0.684 |
| 3,251 | - | 124 | 70 (0.57 m) | 83 | 70.11 | 0.558 |
| 5,003 | - | 192 | 103 (0.54 m) | 123 | 108.65 | 0.070 |

The nine failures sit at percentiles 0.34-0.96 (median 0.61) of their own small-gear coverage
distribution, and the seven non-failures at 0.07-0.93 (median 0.68). **A failure does not require
the small gears to do anything unusual** - it takes 0.46-0.63 of the islands from them, exactly what
a non-failing `q` of the same size takes. The failure is decided above 31, so no condition modulo
`9.55 x 10^8` can capture it. Dead end, recorded.

## 8. Mechanism

**Why `K(d)` grows while the counting requirement does not.** The adversary's difficulty is not
coverage budget. Three exact ladders separate the causes at `d = 1,120`: the counting requirement
is **10**, the same problem with the one-phase-per-gear rule dropped is **12**, and the real cover
number is **20**. Drop the rule and the covering problem is within 2 of pure counting; keep it and
the answer doubles. Independently, give the gears a free separation instead of `d_g = 2 x 6^{-1}`
and `K` falls by a factor 1.5 (and at `d <= 280` the cover becomes a perfect partition of the
islands, budget exactly `m`). So:

> the growth of `K(d)` is caused by (i) **each gear having exactly one phase** and (ii) **the two
> classes of a gear sitting at a separation the gear cannot choose** - and by nothing about how many
> strikes the machine has.

**What the optimum looks like.** A near-partition: budget `1.00-1.30 m`, at most a handful of
islands struck twice and essentially none three times, every gear taking at least two islands from
`d = 70` on. The adversary buys that near-disjointness with phase freedom - `10^{30.3}` phase
vectors at `d = 1,120` - against a CRT-independent expectation of 50 repeat strikes among the
thirteen small gears alone, where the whole optimum spends 30. It spends a compulsory prefix of small gears (all of 11..31 from
`d = 385`, up to about `1.7 sqrt d`) and then one gear per two or three leftover islands, chosen
opportunistically: the tail gears (79, 839, 383, 1013, 211, 631, 1171, 2521, 367, 281 ...) follow no
pattern at all, they are whichever primes happen to divide `3(i - j) - 1` for a surviving pair.

**Why the growth law is `d/(ln d)^3` and not `sqrt d`.** Leftovers after a prefix `11..G` are
`m prod (1 - 2/g) ~ c m/(ln G)^2` and each costs about one gear; balancing `pi(G)` against that
gives `G (ln G)^2 ~ c' m` and `K ~ c' m/(ln m)^3`. Its local exponent `1 - 3/ln d` is 0.55-0.58 over
the computed range, which is why a `sqrt d` fit tracks the ladder for a decade and then falls
behind - as it does, by 1 at `d = 1,190` and by 2 at `d = 1,260`.

**What it means for the witness.** A failure at `q` needs `R(q) >= K(d)` gears to cooperate; the
`K(d)` cheapest gears have a product `P` that passes `q^2` at `d = 70` and reaches `10^30 q^2` at
`d = 1,120`. Because `P > q^2`, the failure condition stops being "a residue class `q` might fall
into" and becomes "**a prescribed CRT lift is exactly `q^2`**" - one integer, not a density. That is
the branch's contribution to the shape of a proof. It does not close, because the covers are
`2.7^m` in number.

## 9. What holds without exception, with counts (item 7)

| statement | range | exceptions |
|---|---|---|
| a cover with phases is realised by exactly `2^K` classes of `q` mod `prod g_j` | brute force over every residue mod 1,188,847 and mod 323,525,521 (324 million) | **0** |
| `P = prod g_j > q^2` for an optimal cover (`q = 3d`, the short arc) | 21 optimal covers, `d = 70 .. 1,260` | **0** (only `d = 35`, at `P/q^2 = 0.037`, is below) |
| `R(q) >= K_B(d)` at a failure | 197 failures: 29 at `B = 7` (`q <= 3000`), 168 at `B = 11` (`q <= 9500`) | **0** |
| the optimal cover returned contains all of 11, 13, 17, 19, 23, 29, 31 | 16 arcs `d = 385 .. 1,330` | **0** |
| 11, 13, 17, 19, 23, 29 in EVERY optimal gear set enumerated | 12 optimal sets at `d = 560` | **0** (31 in 10 of 12) |
| every gear of an optimal cover takes at least two islands | 22 covers, `d = 70 .. 1,330` | **0** |
| no island struck three times in an optimal cover | `d = 35 .. 280` | **0** (one island at `d = 560`, four at `d = 1,120`) |
| the budget `sum |S_j|` of an optimal cover is at most `1.31 m` | all 23 covers | **0** |
| `K_free(d)` (free separation) equals the counting requirement | `d = 35, 70, 140, 280` | **0** (fails at `d = 560`: 9 against 7) |
| `K_B` at equal island counts agrees across `B = 7, 11, 13` to within 1, higher `B` never cheaper | 11 comparisons, `m = 12 .. 98`, arcs 105 .. 10,010 | **0** |
| a free-separation optimum is a perfect partition (budget `= m`) | `d = 35, 70, 140, 280` | **0** (fails at `d = 560`: 1.062 m) |
| `MBS(struck) > K(d)` at a non-failing `q` | 10 sampled `q` from 199 to 2,801 | **0** (`q = 101` is below the threshold) |
| `K(d)` non-decreasing in `d` | 23 arcs | **0** |

## 10. What is new

Screened against `docs/novel/README.md` line by line, in particular `reachability-landscape`,
`island-witness-integers`, `tooth-counterfactual-percentile`, `cover-half-counter-ladder`,
`covering-lp-certificates`, `restricted-covering-certificates`, `j2-lower-ladder`,
`layered-erdos-rankin` and `jk-family`; and against `docs/proof-search/alignment-rules.md` section 5
(the tooth-counterfactual family) and section 6 (the ceilings). The register's
`island-witness-integers` line carries `K(d) = 3, 4, 6, 9, 14, 20` at `d = 35..1120` and nothing
else about the cover number.

**Prior art, named once and stopped.** The **free-separation** version of this problem - each prime
choosing any two residue classes, covering an interval - is exactly the project's own
`jk-family` covering restatement of the two-class Jacobsthal function `j_2 = h_2`
(Ziller-Morack), restricted to the island sublattice and with the gear set free. That is why
`K_free` behaves like a counting problem, and it is why nothing below is claimed about it.
**The real object is not in that family**: `j_k` lets every prime choose its classes, while a gear's
two classes sit at the fixed separation `d_g = 2 x 6^{-1} (mod g)` and only the phase is free.

* **N-C1 / N-C2 (the cause of the growth, isolated).** Three exact ladders at `d = 1,120`: counting
  requirement 10, cover number with the one-phase-per-gear rule dropped 12, real cover number 20;
  and at `d <= 280` the free-separation cover is a **perfect partition of the islands** whose size
  equals the counting requirement exactly. So the gap between counting and covering - the parent's
  headline - is caused entirely by *one phase per gear* and *a separation the gear cannot choose*,
  in that order of size, and not at all by the strike budget. New.
* **N-C3 / N-C4 (the shape of an optimal cover).** From `d = 385` the optimal cover contains all
  of 11, 13, 17, 19, 23, 29, 31 (16 arcs, 0 exceptions), and six of the seven are in every one of
  the twelve optimal gear sets enumerated at `d = 560`; from `d = 70` every gear in an optimal
  cover takes at least two islands (21 covers, 0 exceptions); the cover is a near-partition (budget
  `1.00-1.30 m`, triple coverage essentially absent) which beats the CRT-independent overlap by a
  factor 1.7; the optimal gear set is very far from unique (3, 18, `>= 40`, 8, `>= 12` distinct
  optimal sets at `d = 35, 70, 140, 280, 560`) yet acquires compulsory members as `d` grows. New.
* **N-C5 (the cover number is a function of the island count and of the cheapest gear left).**
  `K_7`, `K_11` and `K_13` at equal island counts agree to within 1 at all eleven comparisons
  available (`m = 12 .. 98`), although the arcs differ by up to a factor of twelve - `K_13` needs
  `d = 10,010` to reach the `m = 96` that `K_7` reaches at `d = 840`. The higher bound is never
  cheaper, by exactly one gear, and the reason is that the bar removes gears 11 and 13 from the
  adversary's hand: the best first gear takes `2m/11` islands at `B = 7` and `2m/17` at `B = 13`.
  So the arc does not enter at all. New.
* **N-C6 (the exact class count, without counting islands).** A cover with phases is realised by
  **exactly `2^K`** classes of `q` modulo `prod g_j`, by the doubling law applied once per gear;
  verified exhaustively over 324 million residues, 0 exceptions. New as a statement about covers
  (the doubling law itself is the parent's N-R6).
* **N-C7 (the square pin).** `prod g_j > q^2` for every optimal cover from `d = 70` on, so a cover
  with phases determines `q^2` **as an integer** and admits at most one `q` in the whole range. The
  failure condition is therefore not a density condition but "a prescribed CRT lift is a perfect
  square". New, and it is the branch's contribution toward the root.
* **N-C8 (the growth law, measured).** `K(d)` exact at 23 arcs to `d = 1,330`; `K (ln d)^3/d` flat
  at `6.15 +- 0.20` from `d = 315` on, with the mechanism (prefix plus one gear per leftover) that
  produces the form; a `pi(c sqrt d)` fit tracks the ladder to `d = 1,120` and then falls behind by
  1 and 2. One extra gear buys the adversary ten more islands at `d ~ 1,000` and 6.4 at `d ~ 300`.
  New as exact data and as a form.
* Filed, not claimed: the real machine is a factor 1.00-2.64 off the adversary at all 197 recorded
  failures, median 1.50 (`B = 7`) and 1.60 (`B = 11`), not monotone in `q`; the per-island first
  moment puts the last failure at `q ~ 2,339` against the true 2,849 and expects 0.0012 failures
  above it, while under-predicting the `[1000, 3000)` band by a factor 14; the cover-side first
  moment is `prod_i lambda(i) ~ 2.7^m`, i.e. `10^39` at `q = 2,849` (the parent's counting wall in
  another costume, stopped).

## 11. Verdict

**FACT, with one statement of a different kind, and a named next interaction.**

The cover number was the parent's one growing quantity and this branch takes it apart. It grows:
`K(d) = 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 15, 16, 17, 18, 19, 19, 20, 21, 22, 22` over
23 exactly-solved arcs from 35 to 1,330, every one HiGHS-certified optimal, against a counting
requirement that goes 2 .. 10 and stops. The growth is `d/(ln d)^3` to the accuracy the data can
see (`K (ln d)^3/d = 6.15 +- 0.20` over sixteen consecutive arcs), not `pi(c sqrt d)`.

The cause is now isolated and is not what the parent's phrasing suggested. It is not the strike
budget: dropping the rule that a gear has one phase brings `K(1120)` from 20 down to 12, within two
of pure counting. It is not island geometry either: give the gears a free separation and the cover
becomes a *perfect partition* of the islands whose size is the counting requirement exactly. The
growth is bought by two facts about the machine - **a gear has one phase, and its two teeth sit at
a separation it cannot choose** - and the second is worth a factor 1.5 while the first is worth the
rest. That also settles the brief's structural question: `K(d)` is **not** the counterfactual
family's record ladder restricted to islands, because the family's teeth `+-v_g` have a free
separation; the family's ladder is the `K_free` row, which is the island restriction of the
published two-class Jacobsthal covering problem, so none of the family's record values or bounds is
used or usable here.

The statement of a different kind is N-C7. A cover pins `q^2` modulo the product of its gears, and
that product passes `q^2` at `d = 70` and reaches `10^30 q^2` by `d = 1,120`; combined with the
exact class count `2^K` (proved, and checked over 324 million residues), a failure ceases to be
"`q` fell into a class of some density" and becomes "the CRT lift of a prescribed residue vector is
exactly the integer `q^2`". At most one `q` per cover, in the whole range. That is the right shape
for a proof and it is the first time in this line the witness has had one.

It does not close, and the reason is now exact rather than vague: the number of covers is about
`(sum_{7<g<=q} 2/g)^m = 2.7^m`, which is `10^54` at `d = 1,120` against a class density of
`10^-30`. The union bound over covers is vacuous by 24 orders of magnitude - it is the parent's
counting wall wearing a different hat, since `sum 2/g > 1` from `q = 53`. The finite lever that
would have avoided it - the compulsory prefix 11..31 forcing a condition on `q` modulo `9.55 x 10^8`
- is dead: at a failure the seven smallest gears take 0.46-0.63 of the islands, at the 61st
percentile of their own distribution, indistinguishable from a non-failing `q` of the same size.

**The next interaction, named.** Bound the number of covers a REAL machine can produce. The
adversary picks phases; a machine has one phase vector, and section 4 shows it is a factor 1.0-2.6
worse than the optimum at every failure ever recorded while section 7.2 shows its small gears are
entirely ordinary. The gap between "covers that exist" and "covers a single `q^2` can realise" is
where the remaining `10^24` has to come from, and nothing here bounds it.

## 12. Dead ends (do not re-enter)

* **`K ~ pi(c sqrt d)`** (the brief's suggested form). The fit that is exact at `d = 70, 560, 1,120`
  falls behind by 1 at `d = 1,190` and by 2 at `d = 1,260`, while `K (ln d)^3/d` stays flat over
  sixteen arcs; and at `d = 3,360` it would need the greedy that is 0-1 above optimal everywhere
  else to be 8 above optimal. The ladder's local exponent is 0.54-0.58 because `1 - 3/ln d` is, not because the
  growth is a square root.
* **`K(d)` as the tooth-counterfactual family's record ladder restricted to islands.** False by
  construction and by measurement: the family's separation is free, and with a free separation the
  cover number drops by a factor 1.5 and becomes the counting requirement (a perfect partition of
  the islands at `d <= 280`). The family's record values, its budget inequality and its spectrum
  bounds do not apply and were not used.
* **The counting/coverage budget as the cause of `K`'s growth.** Refuted at `d = 1,120`: with the
  one-phase-per-gear rule dropped the cover number is 12 against a counting requirement of 10 and a
  true `K` of 20. Counting explains 12 of 20; the rule explains the other 8.
* **The cover-side first moment** (sum over covers of size `>= K(d)` of the gears' rate products).
  It is bounded below by `prod_i lambda(i)` with `lambda` the depth function, i.e. `(sum 2/g)^m`,
  which exceeds 1 by `10^19` at `q = 1,487` and `10^39` at `q = 2,849`. It can never say anything;
  it is the parent's `sum 2/g > 1` wall re-derived. Stopped in one line.
* **The compulsory-prefix lever** (a failure forcing a condition on `q` modulo the product of gears
  11..31, `9.55 x 10^8`). Refuted: the nine failures sit at percentiles 0.34-0.96 (median 0.61) of
  the small gears' own coverage distribution, and seven non-failing `q` of the same sizes sit at
  0.07-0.93 (median 0.68). A failure is decided above gear 31.
* **The plain set-cover relaxation as a lower bound on `K(d)`.** It allows one gear at several
  phases and is far too weak: 3, 4, 5, 7, 11, 12 against 3, 4, 6, 9, 14, 20, with gear 11 used four
  times at `d = 560`. Kept only as the instrument that isolates the mechanism (N-C2).
