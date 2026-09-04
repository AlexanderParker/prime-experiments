# Branch R2.a.i.a - THE REACHABILITY LANDSCAPE

Parent: node R2.a.i (the path taken apart, `research/proof/walk_path.md` and
`research/proof/walk_transforms.md`). The observation that spawned this branch, from both parent
provers: the path from `q^2` is the least-prime-factor structure of the two quadratics
`q^2 + 6i - 2` and `q^2 + 6i` along the offset `i`, so gear `g` can strike offset `i` **for some
`q` at all** only if `2 - 6i` or `-6i` is a square mod `g` - the QUADRATIC-RESIDUE BAR. Which
gears can ever reach an offset is therefore a property of the offset alone, free of `q`; the
per-offset mean depth is a fixed arithmetic function of `i`; and the landing avoids the offsets
where that function is large (0 landings on the eight highest, 500 of 2,260 on the eight lowest).

The theory this branch tests: the offsets form a **`q`-free landscape**; its low points are
**islands** - offsets that no gear up to a bound `B` can ever reach, at any `q`; the walk lands
on an island; and the interaction that decides the walk length `L` is between the islands inside
`[1, d)` and the large gears that can reach them.

Scripts: `research/anchor235/r39/rl_landscape.py`, `rl_landing.py`, `rl_null.py`,
`rl_interact.py`, `rl_double.py`, `rl_margin.py`, `rl_rate.py`. Result outputs (untracked):
`research/anchor235/r39/results/rl_*.txt`. Every number this document relies on is written into
the document.

---

## 0. Pre-registered (written before any computation of this branch)

### 0.1 Definitions, exact

Fix a prime `q >= 5`; the machine is `M = {5..q}`. `k_0 = (q^2-1)/6` is the column of `q^2`; the
column at **offset** `i` is `k_0 + i` and carries the members `q^2 + 6i - 2` (lower) and
`q^2 + 6i` (upper). Gear `g` strikes offset `i` iff

```
    q^2 = 2 - 6i  (mod g)      [lower member]      or      q^2 = -6i  (mod g)   [upper member]
```

(the tooth rule in the offset coordinate; docs/proofs/02, kernel; parent P5, 493,101,490 checks).

Write `x = -6i (mod g)`, so the two targets are `x` and `x + 2`.

* **`g` is ADMISSIBLE at offset `i`** iff `x` or `x + 2` is a **nonzero quadratic residue mod
  `g`**. (For a gear `g != q` the value `q^2 mod g` is a nonzero square, so a target `= 0`
  cannot be met.) `G(i)` = the set of admissible gears; `Bar(g)` = the set of offset classes
  `i mod g` at which `g` is **barred**, i.e. not admissible.
* **`lambda(i) = sum_g 2 chi_g(i) / (g - 1)`**, `chi_g(i)` = how many of the two targets are
  nonzero residues mod `g` (`0`, `1` or `2`) - the fixed depth function of the parent branches.
* **island for bound `B`**: an offset `i` barred for **every** gear `5 <= g <= B`. `S_B` = the
  island set; `P_B = prod_{5<=g<=B} g`; `rho_B` = the island density.
* `d` = the top gear's forward tooth arc, `d = 2 * 6^{-1} mod q` (`(q+1)/3` if `q = 5 mod 6`,
  `(2q+1)/3` if `q = 1 mod 6`); `L` = the walk length (the first offset `>= 1` no gear strikes).

The **anchor** 2, 3, 5 is one object: gears 2 and 3 are already inside the column frame, gear 5
is the first gear of the machine and is treated like every other gear here.

### 0.2 What would count as a rule

As the parents: (i) a statement about positions or residues, not a rate; (ii) an exact exception
count over a stated range; (iii) uniform in `q`. A density, a fitted curve or an average is not a
rule. Restating the tooth rule, the two-teeth kill-spacing law, the quadratic-residue bar itself,
the gear-5 lock, W1-W4 or the Hardy-Littlewood count is **not** a finding: it is noted in one
line and the sub-question stops. If a sub-question reduces to a quadratic-reciprocity class
statement (Gauss's supplements: `chi_g(2) = 1` iff `g = +-1 mod 8`, `chi_g(-2) = 1` iff
`g = 1, 3 mod 8`) it is named in one line as classical and only the machine consequence is
pursued.

### 0.3 Predictions, with numbers, and what refutes each

**The landscape (item 1).**

* **I1 (the bar-size closed form).** `|Bar(g)| = (g + 1 - chi_g(2) - chi_g(-2))/4` exactly, i.e.
  `(g-1)/4, (g+1)/4, (g+3)/4, (g+1)/4` for `g = 1, 3, 5, 7 (mod 8)`. Predicted 0 exceptions over
  every gear `5 <= g <= 20000`. REFUTED by one gear.
* **I2 (no gear reaches every offset).** `|Bar(g)| >= (g-1)/4 >= 1` for every gear, so every gear
  is barred somewhere; gear 5 reaches exactly the offsets `1, 3, 4 (mod 5)` and gear 7 exactly
  `0, 1, 2, 4, 6 (mod 7)`. Predicted 0 exceptions.
* **I3 (the offsets every gear reaches).** An offset is admissible for *every* gear iff `-6i` is
  a perfect square, i.e. `i = -6t^2`; and even there gear `g` is barred iff `g | 6t` and
  `g != +-1 (mod 8)`. `2 - 6i` is never a perfect square (`m^2 = 2 mod 3` is unsolvable). So the
  fully reachable offsets are exactly the parent's forced-composite columns `k_0 - 6t^2`, all
  **behind** the walk. Predicted 0 exceptions; REFUTED by one positive offset admissible for
  every gear `<= 20000`.
* **I4 (the mirror).** The map `i -> d_g - i (mod g)` sends the target pair `(x, x+2)` to
  `(-(x+2), -x)`, so it preserves `Bar(g)` when `g = 1 (mod 4)` and maps `Bar(g)` into the
  admissible set when `g = 3 (mod 4)`. Predicted exact for every gear; the global island set is
  therefore **not** mirror-symmetric whenever some gear `<= B` is `3 (mod 4)`. REFUTED by one
  gear either way.

**Islands (item 2).**

* **I5 (island density in closed form).** `rho_B = prod_{5<=g<=B} |Bar(g)|/g` exactly, with
  island counts per period `P_B` of **4** (`B=7`, `P=35`), **12** (`11`, `385`), **48**
  (`13`, `5005`), **192** (`17`, `85085`), **960** (`19`, `1616615`), **5760**
  (`23`, `37182145`). Predicted exact by CRT; REFUTED by one count.
* **I6 (the first islands).** The `B = 7` islands are `i = 5, 10, 12, 17 (mod 35)`; the first
  island past offset 0 is **5** at every `B <= 7`. Predicted exact.

**The landing on the landscape (item 3).**

* **I7 (the landing is an island - my number, against the brief's).** The brief pre-registers
  "the landing is a `B = 7` island in more than 90% of walks". I pre-register the opposite, from
  the mechanism: gear 5 bars 2 of its 5 classes and *strikes* 2 more, so a landing avoiding the 2
  struck classes is an island for `B = 5` with probability about `2/3`; gear 7 bars 2 of 7 and
  strikes 2, leaving `2/5`. My numbers, before computing:
  `P(B = 5 island) ~ 0.67`, `P(B = 7 island) ~ 0.27`, `P(B = 11) ~ 0.09`,
  `P(B = 13) ~ 0.024`. REFUTED (in the brief's direction) if the `B = 7` figure exceeds 0.5;
  refuted (in mine) if it falls below 0.15.
* **I8 (enrichment is exactly the per-gear conditional).** The landing is enriched on islands
  relative to a random offset by exactly the factor the independent per-gear conditional gives
  (`rho_B` versus `prod (bar_g/(g-2))`), to within 10%. If the measured enrichment exceeds the
  conditional prediction by more than 25%, the landscape carries structure beyond order one and
  the branch has a lever; predicted it does not.
* **I9 (landing concentration on the first islands).** The four smallest `B = 7` islands
  `5, 10, 12, 17` take more than 15% of the 2,260 landings; the offsets `= 1 (mod 5)` take none.

**The doubling (item 4).**

* **I10.** For every gear `g` and every offset `i`, the number of residue classes
  `r = q mod g` in `(Z/g)^*` at which `g` strikes offset `i` is **exactly `2 chi_g(i)`** - two
  square roots per admissible target, never one, never three. Exact statement, predicted 0
  exceptions over all gears `<= 500` and offsets `<= 500` exhaustively, and 0 exceptions over the
  real sweep. The empirical rate over primes `q <= 20000` then equals `2 chi_g(i)/(g-1)` up to
  prime equidistribution only; deviations there are a rate and are reported as such, not as
  exceptions.

**The interaction (item 5, 6).**

* **I11 (the island frame is vacuous at small `q`).** `[1, d)` contains no `B = 13` island at all
  until `d` is a few hundred, i.e. until `q ~ 300`; predicted more than 40 primes with zero
  `B = 13` islands in range and 0 primes above `q = 1000` with none.
* **I12 (a free island always exists once the frame is non-vacuous).** For every `q` with at
  least one `B = 13` island in `[1, d)`, at least one of those islands is struck by no gear at
  all - hence is an opening, hence `L < d`. Predicted 0 exceptions; REFUTED by one `q` whose
  `[1, d)` islands are all struck (which would be a walk running to the top gear's next tooth,
  i.e. `q = 53`'s failure re-appearing).
* **I13 (the counting margin is unchanged by the frame).** The exact ratio (strikes by gears
  `> B` on the islands of `[1, d)`) / (number of those islands) equals
  `2(ln ln q - ln ln B) + O(1)` and **exceeds 1 at every `q` above a small threshold**, for every
  `B` - because passing to islands divides numerator and denominator by the same `rho_B`.
  Predicted ratio 2.4-3.0 at `q ~ 20000` for `B = 13`, and predicted to be **the same** for
  `B = 7` and `B = 19` to within the `ln ln B` term. If the ratio were below 1 for some `B` the
  frame would give a counting proof; predicted it never is.
* **I14 (what decides which large gear strikes an island).** Gear `g` strikes island `i` iff
  `q = +- s (mod g)` with `s^2 = -6i` or `s^2 = 2 - 6i (mod g)` - a condition on `q`, not on the
  island. So "some island of `[1, d)` is free" is a statement that `q` avoids a fixed system of
  `2 chi_g(i)` classes mod every gear `g > B`, for at least one `i` in the fixed set
  `S_B ∩ [1, d)`. Predicted: this is the smallest interaction, and it is a covering statement in
  `q`-space rather than in offset space.

**Item 7.**

* **I15.** Report anything the landscape shows that holds for every `q` in the sweep without
  exception, with its count.

### 0.4 Scorecard

| # | prediction | verdict and evidence |
|---|---|---|
| I1 | `\|Bar(g)\| = (g+1-chi(2)-chi(-2))/4`, 0 exceptions | **CONFIRMED**, 0 of 2,260 gears; becomes N-R1 |
| I2 | no gear reaches every offset | **CONFIRMED**, min `\|Bar\| = 2` at `g = 5`, 2,260 of 2,260 |
| I3 | fully reachable offsets are exactly `i = -6t^2` | **CONFIRMED**, 0 forward offsets in `1..20000` admissible for all 2,260 gears; the `t = 5` exception `g = 5 \| 30` occurs as predicted |
| I4 | mirror `i -> d_g - i` preserves / inverts `Bar(g)` by `g mod 4` | **CONFIRMED**, 1,125 of 1,125 (`g=1 mod 4`) and 1,135 of 1,135 (`g=3 mod 4`); becomes N-R2 |
| I5 | island counts 4, 12, 48, 192, 960, 5760 | **CONFIRMED** exactly, all six bounds |
| I6 | `B=7` islands `5, 10, 12, 17 (mod 35)`; first island 5 | **CONFIRMED** exactly |
| I7 | landing an island: 0.67 / 0.27 / 0.09 / 0.024 | **CONFIRMED, brief REFUTED**: measured 0.6681 / 0.3226 / 0.1473 / 0.0960 against my 0.67 / 0.27 / 0.09 / 0.024 and the brief's ">90%" at `B=7` |
| I8 | enrichment = the per-gear conditional, within 10% | **REFUTED as posed, CONFIRMED in substance**: the naive conditional is the wrong null (out by 3x at `B=13`); against the correct order-one FIRST-PASSAGE null the measured rates are 0.99, 0.93, 0.88, 0.92 of prediction - the landing's island preference is order one and slightly *below* it |
| I9 | first four islands take > 15% of landings | **CONFIRMED**: 472 of 2,260 = **20.9%**; landings at `= 1 (mod 5)`: 1 (the degenerate `q = 5`) |
| I10 | exactly `2 chi_g(i)` striking classes of `q mod g` | **CONFIRMED**, 0 of 21,531 (gear, offset) cells; 0 cells with an odd count |
| I11 | frame vacuous below `q ~ 300` at `B = 13` | **REFUTED as stated**: only 7 primes have no `B=13` island in `[1,d)` (`q <= 29`), not 40; the frame is non-vacuous from `q = 31` |
| I12 | a free island exists whenever the frame is non-vacuous | **REFUTED at `B = 13`** (232 primes, largest `q = 18839`); **CONFIRMED at `B = 7` above `q = 1487`**: 0 exceptions in the 2,026 primes of `(1487, 20000]`, 17 below. Becomes N-R4 |
| I13 | strikes/islands ratio `2(lnln q - lnln B)`, always > 1 | **CONFIRMED and sharpened to an identity**: the ratio is `sum_{B<g<=q} 2/g` to four digits (2.703 vs 2.699 at `B=7`; 2.5212 vs 2.517; 2.3616 vs 2.363; 2.2426 vs 2.246) and exceeds 1 at every `q >= 101`. Becomes N-R5 |
| I14 | the smallest interaction is a covering statement in `q`-space | **CONFIRMED** as the formulation; stated in section 5 |
| I15 | exception-free statements over the sweep | seven, listed in section 6 |

---

## 1. Setup (exact ranges)

No sampling anywhere. Scripts in `research/anchor235/r39/`.

| object | range | script |
|---|---|---|
| `\|Bar(g)\|` against the closed form; `Bar(g)` written out | every gear `5 <= g <= 20000` (2,260) | `rl_landscape.py` |
| `\|G(i)\|`, `lambda(i)`, the fully-reachable offsets, the mirror | offsets `0..20000` x 2,260 gears (45.2 million admissibility cells) | `rl_landscape.py` |
| island residue sets by iterated CRT | `B = 5, 7, 11, 13, 17, 19, 23`, periods 5 to 37,182,145 | `rl_landscape.py` |
| the walk `L`, `\|G(L)\|`, `lambda(L)`, island status, landing histogram | every prime `q = 5..20000` (2,260 walks) | `rl_landing.py` |
| the order-one first-passage null (per-gear independence, offsets independent) | the same 2,260 machines, offsets `0..1599` | `rl_null.py` |
| islands in `[1, d)`, their strikes, free islands, the smallest striker | every prime `q = 5..20000`; smallest striker for `q <= 3000` | `rl_interact.py` |
| the doubling, exhaustive | every gear `5..500` and every offset class: 21,531 cells | `rl_double.py` |
| the free-island margin, the full failure lists | every prime `q = 5..20000` | `rl_margin.py` |
| the large gears' strike rate on islands | 40 gears `17..199` x 103,899 island sightings | `rl_rate.py` |

## 2. Results - the landscape (item 1)

### 2.1 The bar set of one gear, in closed form

Gear `g` is barred at offset `i` iff neither `x = -6i` nor `x + 2` is a nonzero quadratic
residue mod `g`. Counting those `x` by the standard character sum
`sum_x chi(x(x+2)) = -1` and adding the two zero cases gives

```
    |Bar(g)|  =  ( g + 1 - chi_g(2) - chi_g(-2) ) / 4
```

and by Gauss's second supplement (`chi_g(2) = 1` iff `g = +-1 mod 8`; `chi_g(-2) = 1` iff
`g = 1, 3 mod 8`) the bar-set size is decided by `g mod 8` alone:

| `g mod 8` | 1 | 3 | 5 | 7 |
|---|---|---|---|---|
| `\|Bar(g)\|` | `(g-1)/4` | `(g+1)/4` | `(g+3)/4` | `(g+1)/4` |
| gears in `5..20000` | 556 | 570 | 569 | 565 |
| mean `\|Bar\|/g` | 0.249871 | 0.250176 | 0.250755 | 0.250203 |

**0 mismatches over all 2,260 gears.** The classical input is named in one line (Gauss's second
supplement); what is new is that this is the exact size of the machine's own unreachable set at
an offset.

The bottom of the landscape, written out:

| gear | barred at `i =` (mod `g`) | `\|Bar\|` | reaches |
|---|---|---|---|
| 5 | 0, 2 | 2 | 1, 3, 4 |
| 7 | 3, 5 | 2 | 0, 1, 2, 4, 6 |
| 11 | 0, 6, 10 | 3 | 1, 2, 3, 4, 5, 7, 8, 9 |
| 13 | 0, 9, 10, 12 | 4 | 1..8, 11 |
| 17 | 2, 4, 8, 15 | 4 | the other 13 |
| 19 | 0, 1, 5, 11, 17 | 5 | the other 14 |
| 23 | 1, 3, 8, 9, 12, 16 | 6 | the other 17 |

**No gear reaches every offset**: `|Bar(g)| >= 2` for every gear, minimum 2 at `g = 5` and
`g = 7`. Gear 5's reachable set `{1, 3, 4} mod 5` is the parent's gear-5 pattern seen from the
`q`-free side: whichever of `{1,4}` or `{1,3}` a particular `q` gives, the union over `q` is
`{1,3,4}` and the pair `{0,2}` is unreachable at every prime.

### 2.2 The landscape as a whole

Over offsets `1..20000` against the 2,260 gears:

| quantity | value |
|---|---|
| `\|G(i)\|` (admissible gears) | min **1,623** (`i = 18983`), max **1,768** (`i = 14967`), mean 1,694.40 |
| the generic value `3/4 x 2260` | 1,695.0 |
| `lambda(i)` | min **1.9250** (`i = 13870`), max **5.0945** (`i = 13106`), mean 3.4433 |
| `lambda(0)`, `\|G(0)\|` | 1.5595, **1,121** = exactly the gears `= +-1 (mod 8)` (agrees exactly; classical, filed) |
| variance of `lambda` from gears `<= 13` alone | **0.21801 of 0.25099 = 86.9%**; correlation 0.932 |

The last row is the mechanism that connects this branch to its parent: **the landscape's relief
is almost entirely the small gears.** A low-`lambda` offset is essentially an offset that the
first four gears are barred from - that is, an island. The minimum-`lambda` offset of the whole
range, `i = 13870`, is a `B = 19` island; the eight lowest-`lambda` offsets of `1..80` are
`10, 17, 32, 47, 52, 55, 75, 77`, of which **five** - `10, 17, 47, 52, 75` - are `B = 7` islands
(`47 = 12`, `52 = 17`, `75 = 5` mod 35) and `10` is a `B = 13` island.

### 2.3 Which offsets every gear reaches

`-6i` is a perfect square iff `i = -6t^2` (`m^2 = 0 mod 6` forces `6 | m`), and `2 - 6i` is
never a perfect square (`m^2 = 2 (mod 3)` is unsolvable). Hence:

> **Every gear is admissible at `i = -6t^2`, except a gear `g | 6t` with `g != +-1 (mod 8)`.**

Measured: at `t = 1, 2, 3, 4, 6, 7, 8` no gear below 1,231 is barred; at `t = 5` exactly gear 5
is barred (`5 | 30`, and `chi_5(2) = -1`), as predicted. And **0 of the 20,000 forward offsets is
admissible for all 2,260 gears** - the fully reachable offsets are exactly the parent's
forced-composite columns `k_0 - 6t^2` (member `(q-6t)(q+6t)`), all behind the walk. So the
landscape is one-sided: the guaranteed-blocked offsets lie behind `q^2`, the walk goes forward
into the part of the landscape where no offset is reachable by everything.

### 2.4 The mirror

`i -> d_g - i (mod g)` (with `d_g = 2 * 6^{-1}`, the gear's own tooth separation) sends the
target pair `(x, x+2)` to `(-(x+2), -x)`, so it multiplies both Legendre symbols by `chi_g(-1)`:

* `g = 1 (mod 4)`: `Bar(g)` is **preserved** - **1,125 of 1,125 gears, 0 failures**;
* `g = 3 (mod 4)`: `Bar(g)` is mapped **entirely into the admissible set** - **1,135 of 1,135**.

Consequence: the island set `S_B` has a mirror symmetry only when every gear `<= B` is
`1 (mod 4)`. Already at `B = 7` it does not (`7 = 3 mod 4`), and the measured island set
`{5, 10, 12, 17} mod 35` is indeed not symmetric about `3^{-1} = 12 (mod 35)`. The period
mirror of the machine and the landscape's mirror are different objects.

## 3. Results - the islands (item 2)

By iterated CRT on the residue conditions (each `Bar(g)` is a set of classes mod `g`; the gears
are coprime), exact:

| `B` | `P_B` | islands per period | density `rho_B` | one island per | first island `> 0` | max gap | min gap |
|---|---|---|---|---|---|---|---|
| 5 | 5 | 2 | 0.400000 | 2.5 | 2 | 3 | 2 |
| 7 | 35 | **4** | 0.1142857 | 8.75 | **5** | 23 | 2 |
| 11 | 385 | **12** | 0.0311688 | 32.1 | 10 | 77 | 5 |
| 13 | 5,005 | **48** | 0.0095904 | 104.3 | 10 | 313 | 12 |
| 17 | 85,085 | **192** | 0.0022566 | 443.2 | 87 | 1,687 | 23 |
| 19 | 1,616,615 | **960** | 0.0005938 | 1,684.0 | 87 | 12,882 | 23 |
| 23 | 37,182,145 | **5,760** | 0.0001549 | 6,455.2 | 4,520 | 49,357 | 23 |

The island count is exactly `prod_{5<=g<=B} |Bar(g)| = 2, 4, 12, 48, 192, 960, 5760` - the
closed form of 2.1 multiplied out. The explicit small sets:

```
   B = 5 :  0, 2                                   (mod 5)
   B = 7 :  5, 10, 12, 17                          (mod 35)
   B = 11:  10, 17, 87, 110, 187, 215, 220, 285, 292, 297, 325, 362      (mod 385)
   B = 13:  10, 87, 220, 285, 325, 402, 572, 780, 857, 880, 985, ...     (mod 5005; 48 in all)
```

Gap spectrum (cyclic, one period):

| `B` | min gap | median | mean | max | distinct gap values |
|---|---|---|---|---|---|
| 7 | 2 | 5 | 8.8 | 23 | 3 |
| 11 | 5 | 28 | 32.1 | 77 | 9 |
| 13 | 12 | 77 | 104.3 | 313 | 21 |
| 17 | 23 | 380 | 443.2 | 1,687 | 52 |
| 19 | 23 | 1,122 | 1,684.0 | 12,882 | 213 |

The maximal gap grows faster than the mean (313 against 104 at `B = 13`; 12,882 against 1,684 at
`B = 19`), which is what decides whether the frame is usable at a given `q`: the frame is
non-vacuous on `[1, d)` once `d` exceeds the largest gap, and useful well before that.

## 4. Results - the landing on the landscape (item 3)

### 4.1 Is the landing an island?

| `B` | measured | naive conditional null `prod \|Bar\|/(g-2)` | **order-one first-passage null** | island density `rho_B` | measured / order-one null |
|---|---|---|---|---|---|
| 5 | 1,510 / 2,260 = **0.6681** | 0.6667 | 0.6768 | 0.400000 | **0.987** |
| 7 | 729 / 2,260 = **0.3226** | 0.2667 | 0.3469 | 0.114286 | **0.930** |
| 11 | 333 / 2,260 = **0.1473** | 0.0889 | 0.1671 | 0.031169 | **0.882** |
| 13 | 217 / 2,260 = **0.0960** | 0.0323 | 0.1044 | 0.009590 | **0.919** |
| 17 | 31 / 2,260 = 0.0137 | 0.0086 | 0.0092 | 0.002257 | 1.491 |
| 19 | 31 / 2,260 = 0.0137 | 0.0025 | 0.0092 | 0.000594 | 1.492 |

The brief's pre-registration (">90% at `B = 7`") is **refuted**: the figure is 0.3226. My
pre-registered numbers 0.67 / 0.27 / 0.09 / 0.024 are right at `B = 5, 7, 11` and low by a factor
3 at `B = 13`, for the reason the table shows: the naive conditional
`P(barred | not struck) = |Bar(g)|/(g-2)` is the **wrong null**. The landing is not a random
missed offset, it is the **first** missed offset, and an offset that more gears are barred from
is likelier to be missed by all of them. Redone as a first-passage computation with every gear
independent (`p_g(i) = 2 chi_g(i)/(g-1)`, `pi(i) = prod (1 - p_g)`,
`P(L = i) = pi(i) prod_{j<i}(1 - pi(j))`), the order-one null reproduces the measured island
rates to 7-12%, and the measured value is **below** it in every case at `B <= 13`.

Two further order-one agreements: mean `lambda(L)` measured **3.0852** against null **3.0745**
(0.3%); the null's mean `L` is 32.3 against the measured 39.2 (the null has no gear-to-gear
correlation and so lands slightly early).

So the landing's preference for islands is real and large - a `B = 13` island is **10 times**
likelier to be a landing than a random offset - but it is **entirely the per-gear bar**, with no
residue left for an interaction. `I8` is refuted as posed (my null was wrong) and confirmed in
substance (the landscape carries no order-two structure at the landing).

### 4.2 The landing histogram

| offset | landings | `B=7` island | `B=13` island | `lambda(i)` | order-one null | measured/null |
|---|---|---|---|---|---|---|
| 10 | **183** | yes | yes | 2.3674 | 214.8 | 0.85 |
| 17 | **107** | yes | no | 2.5108 | 133.9 | 0.80 |
| 12 | **93** | yes | no | 2.8055 | 111.8 | 0.83 |
| 5 | **89** | yes | no | 3.0330 | 111.1 | 0.80 |
| 22 | 71 | no | no | 2.8797 | 70.2 | 1.01 |
| 3 | 68 | no | no | 3.2672 | 85.6 | 0.79 |
| 32 | 63 | no | no | 2.6097 | 69.8 | 0.90 |
| 25 | 63 | no | no | 2.8961 | 60.5 | 1.04 |

**The four smallest `B = 7` islands, `5, 10, 12, 17`, are the four commonest landings and take
472 of 2,260 landings = 20.9%** (I9 confirmed) - out of 153 distinct landing offsets and a range
running to 402. Landings at offsets `= 1 (mod 5)`: **1**, the degenerate `q = 5` (gear 5 strikes
every other such offset at every `q`; parent N-W1).

### 4.3 The landing's position in the landscape

| quantity | value |
|---|---|
| percentile of `lambda(L)` among `lambda(1..d-1)` | mean **0.3013**, median **0.2405**; below 0.25 at 1,152 of 2,259, above 0.75 at 94 |
| `\|G(L)\|` as a fraction of the machine `{5..q}` | mean 0.7467, min 0.3333, max 0.8684 (generic 0.75) |
| `lambda(L)` | mean **3.0852** against 3.4433 over all offsets; min 2.1792, max 4.1234 |

The landing sits in the lower quartile of the landscape but is not extreme, and the admissible
count `|G(L)|` is barely below generic - because `|G|` counts all 2,260 gears equally while
`lambda` weights them by `2/(g-1)`. **The landing is chosen by the small gears' bar, not by the
size of the admissible set.**

### 4.4 By machine size

| `q` band | walks | `B=5` | `B=7` | `B=11` | `B=13` |
|---|---|---|---|---|---|
| 5-100 | 23 | 0.6522 | 0.3913 | 0.2174 | 0.2174 |
| 100-1,000 | 143 | 0.6573 | 0.3497 | 0.1888 | 0.1469 |
| 1,000-5,000 | 501 | 0.6527 | 0.3293 | 0.1737 | 0.1098 |
| 5,000-20,000 | 1,593 | 0.6742 | 0.3170 | 0.1343 | 0.0854 |

The `B = 5` rate is flat at `2/3` (it is fixed by gear 5's three-class law and nothing else); the
higher-`B` rates decay slowly, as the first-passage null says they must, because longer walks
reach deeper into offsets where the small gears' bar is diluted.

## 5. Results - the interaction (items 4, 5, 6)

### 5.1 The doubling (item 4)

Because the phase enters only as the square `q^2`, an admissible target contributes exactly its
**two** square roots as classes of `q mod g`. Exhaustively over every gear `5..500` and every
offset class - **21,531 (gear, offset) cells**:

* cells where the number of striking classes `r = q mod g` differs from `2 chi_g(i)`: **0**;
* cells with an **odd** number of striking classes: **0**.

So gear `g` strikes offset `i` at the exact rate `2 chi_g(i)/(g-1)` over the residues of `q`,
never `1/(g-1)` and never `3/(g-1)`: the square phase forbids an odd count. Two exact corollaries:

* summed over the offsets, `sum_i chi_g(i) = g - 1` for every gear (checked, 0 exceptions in the
  first 60 gears), so **the mean strike rate of gear `g` over all offsets is exactly `2/g`** -
  the machine's own rate - while the rate is **exactly 0** on the `~ g/4` barred classes and
  correspondingly higher elsewhere. The bar does not cost the gear any strikes; it concentrates
  them.
* the empirical rate over the real primes `q <= 20000` matches `2 chi_g(i)/(g-1)` to within
  0.0014-0.0046 for gears 5..37 (max deviation 0.00457 at gear 13). That is prime
  equidistribution mod `g` - a rate; noted and stopped.

### 5.2 Are the islands of `[1, d)` all struck? (item 5)

Every striker of an island is a gear `> B`, by the definition of an island. So the island set
turns the covering question into: *do the large gears cover a fixed sparse set?*

| `B` | walks with `>= 1` island in `[1,d)` | walks with none | walks with `>= 1` **free** island | primes whose islands are **all struck** |
|---|---|---|---|---|
| 7 | 2,257 | 3 (`q = 5, 7, 11`) | **2,240** | 17, largest `q = 1487` |
| 11 | 2,253 | 7 | 2,180 | 73, largest `q = 9281` |
| 13 | 2,253 | 7 | 2,021 | 232, largest `q = 18839` |
| 17 | 2,218 | 42 | 1,338 | 880, largest `q = 19937` |

The 17 failures at `B = 7` are exactly

```
   17, 23, 29, 41, 53, 73, 113, 137, 173, 197, 233, 263, 353, 461, 683, 1151, 1487
```

and **above `q = 1487` there are none: 2,026 primes, 0 exceptions.** This is the branch's
central positive result:

> **N-R4.** For every prime `q` in `(1487, 20000]` at least one offset `= 5, 10, 12 or 17
> (mod 35)` inside `[1, d)` is struck by no gear of `{5..q}` at all. Since such an offset is an
> opening, this **witnesses `L < d` on a fixed set of density `4/35`**, and the witness set does
> not depend on `q`.

I11 is refuted as stated (the `B = 13` frame is non-vacuous from `q = 31`, not `q ~ 300`) and
I12 is refuted at `B = 13` (232 primes have every `B=13` island struck) but confirmed at `B = 7`
above 1,487.

**A correction to the brief.** An all-struck island set does **not** mean the walk runs past `d`:
232 primes have every `B=13` island of `[1,d)` struck and still land well before `d`, because the
landing is then a non-island opening. Only `q = 53` has `L >= d` in the whole sweep (parent P3),
and it is in the `B = 7` failure list for a different reason - `d = 18`, and all four islands
`5, 10, 12, 17` below it happen to be struck.

Where the first free island sits relative to the landing (it can never be below `L`, since every
offset below `L` is struck - a gate, not a finding):

| `B` | first free island `= L` | `> L` | `< L` | islands strictly below `L` (all struck): median / max |
|---|---|---|---|---|
| 7 | 729 | 1,511 | **0** | 4 / 47 |
| 11 | 333 | 1,847 | **0** | 2 / 13 |
| 13 | 217 | 1,804 | **0** | 1 / 5 |
| 17 | 31 | 1,307 | **0** | 0 / 2 |

### 5.3 The margin of the frame

| `B` | `q` band | walks | min free | median | mean | max | islands (median) | free/islands |
|---|---|---|---|---|---|---|---|---|
| 7 | 5-100 | 20 | 0 | 1 | 1.30 | 3 | 4 | 0.3333 |
| 7 | 100-1,000 | 143 | 0 | 3 | 3.24 | 16 | 27 | 0.1068 |
| 7 | 1,000-5,000 | 501 | 0 | 11 | 12.21 | 35 | 149 | 0.0724 |
| 7 | 5,000-10,000 | 560 | **4** | 23 | 24.91 | 51 | 383 | 0.0580 |
| 7 | 10,000-20,000 | 1,033 | **12** | 38 | 42.56 | 87 | 761 | 0.0498 |
| 13 | 10,000-20,000 | 1,033 | 0 | 5 | 5.27 | 16 | 66 | 0.0726 |

The minimum free-island count at `B = 7` grows with `q` (0, 0, 0, 4, 12 by band): the frame is
not merely satisfied, its slack grows. The free fraction sits 20-25% **below** the independent-gear product
`prod_{B<g<=q}(1 - 2/g)` (0.04984 measured against 0.06299 at `q = 10000..20000`, `B = 7`;
0.07262 against 0.09099 at `B = 13`) - the twin-prime singular-series correction, a rate;
stopped.

### 5.4 Counting strikes against islands, exactly (item 6)

| `B` | `q` band | walks | islands (median) | **strikes/island** | `2(lnln q - lnln B)` | exact `sum_{B<g<=q} 2/g` |
|---|---|---|---|---|---|---|
| 7 | 100-1,000 | 143 | 27 | 1.901 | 2.328 | 2.044 (at `q=1000`) |
| 7 | 1,000-5,000 | 501 | 149 | 2.353 | 2.820 | 2.458 (at 5,000) |
| 7 | 5,000-10,000 | 560 | 383 | 2.557 | 3.046 | 2.614 (at 10,000) |
| 7 | 10,000-20,000 | 1,033 | 761 | **2.703** | 3.194 | **2.699** (at 15,000) |
| 11 | 10,000-20,000 | 1,033 | 208 | **2.521** | 2.776 | **2.517** |
| 13 | 10,000-20,000 | 1,033 | 66 | **2.362** | 2.642 | **2.363** |
| 17 | 10,000-20,000 | 1,033 | 15 | **2.243** | 2.443 | **2.246** |

The measured ratio is the exact Mertens sum `sum_{B<g<=q} 2/g` to **four significant figures at
every `B`**. The mechanism is a CRT identity, and it is the answer to item 6:

> **N-R5.** The island set `S_B` is a union of residue classes mod `P_B`, and every gear `g > B`
> is coprime to `P_B`, so `g`'s two strike classes mod `g` meet the island classes in exactly the
> proportion in which they meet all offsets. A large gear therefore strikes **islands at exactly
> the machine's own rate `2/g`** - the landscape gives it no discount. Passing to islands divides
> the number of targets and the number of strikes by the same `rho_B`, and **the counting margin
> is unchanged**.

Checked directly gear by gear over 103,899 island sightings (`B = 13`, `q <= 20000`): the ratio
of the measured per-gear rate to `2/g` over the 40 gears `17..199` has **mean 0.9956**, min
0.8517, max 1.1613 (the scatter is the 2,260 available residues of `q`, not structure).

Consequence, stated plainly: the exact sum `sum_{B<g<=q} 2/g` first exceeds 1 at `q = 53`
(`B=7`), `79` (`B=11`), `113` (`B=13`) and `163` (`B=17`) and grows from there, so **no counting
argument over the island set can force a free island** at any `B`. The frame does not buy a
counting proof, and it cannot: it is scale-free in exactly the wrong way.

### 5.5 Which large gear strikes an island

Gear `g` strikes island `i` iff `q = +- s (mod g)` where `s^2 = -6i` or `s^2 = 2 - 6i (mod g)` -
**a condition on `q`, not on the island**. Measured (smallest striker of a struck `B = 13`
island, `q <= 3000`, 2,732 struck islands):

| smallest striker | 17 | 19 | 23 | 31 | 29 | 37 | 43 | 47 |
|---|---|---|---|---|---|---|---|---|
| share | 0.1255 | 0.0930 | 0.0706 | 0.0593 | 0.0465 | 0.0414 | 0.0293 | 0.0253 |
| `(2/g) prod_{13<h<g}(1-2/h)` | 0.1176 | 0.0929 | 0.0687 | 0.0466 | 0.0497 | 0.0400 | 0.0304 | 0.0252 |

242 distinct smallest strikers, largest 2,837. The distribution is the order-one prediction (the
29/31 inversion is a 3-sigma sampling wobble in this subsample; the direct rate test of 5.4 shows
both gears at `2/g`). **Nothing decides which gear strikes an island except that gear's own
residue condition against `q`.**

## 6. Mechanism, and what holds without exception

**The landscape is a `q`-free relief carved by the small gears.** Each gear `g` is barred from
`(g + 1 - chi(2) - chi(-2))/4` of the offset classes - a quarter of them, exactly, with a
correction fixed by `g mod 8`. Because `q^2` is a square modulo every gear below `q`, the bar is
absolute: no prime `q` whatever can make gear `g` strike a barred offset. The bar's *relief*,
though, is almost all in the first four gears: 86.9% of the variance of `lambda` comes from
`{5, 7, 11, 13}`, because a gear contributes `O(1/g)` to the depth. So "a low point of the
landscape" and "an island for a small `B`" are the same object, and the parent's observation
(the landing avoids high-`lambda` offsets) is the statement that **the landing prefers islands**.

**Why it prefers them, and by exactly how much.** The landing is the first offset every
progression misses. An offset from which four gears are permanently barred starts with those
four misses free. The gain is quantified by the first-passage computation and is entirely order
one: measured 0.99, 0.93, 0.88, 0.92 of the independent-gear prediction at `B = 5, 7, 11, 13`.
There is no interaction term at the landing. The four smallest islands `5, 10, 12, 17` take 21%
of all landings for this reason and no other.

**Why the frame does not become a proof.** Restricting the covering question to the islands is a
genuine reduction of the *target*: from `d` offsets to `rho_B d` of them, on a fixed set
independent of `q`. But CRT makes the large gears' rate on that set exactly `2/g`, the same rate
they have everywhere, so the *capacity* falls by the same factor `rho_B`. The ratio
capacity/target is `sum_{B<g<=q} 2/g` - measured to four digits - and it crosses 1 at `q = 53, 79, 113, 163` for
`B = 7, 11, 13, 17`.
The island frame is therefore **scale-free**: it cannot change a counting margin, only the size
of the objects being counted. This is the same wall the tree records as
`docs/novel/cover-half-counter-ladder.md` ("no exposure-only argument bounds `L` uniformly"), met
here in the sparsest available coordinates, and it says why sparsity alone will not do it.

**The smallest interaction that would have to be proved** (item 6, in the machine's terms):

> Fix `B = 7`. For every prime `q` there is an offset `i` with `i = 5, 10, 12` or `17 (mod 35)`
> and `1 <= i < d` such that, for every gear `g` in `(7, q]` and every square root `s` of `-6i`
> or of `2 - 6i` modulo `g`, `q != +- s (mod g)`.

Everything below this is proved or provable in a line: the bar (2.1), the island set and its
spacing (section 3), the doubling (5.1), the large gears' rate on islands (5.4). What the
statement asks for is that `q` **avoid a fixed system of at most four residue classes modulo each
gear, for at least one member of a fixed set of density `4/35`**. Two things about that shape are
worth recording, because they are what the landscape adds that the plain covering problem lacks:

1. **The unknown has moved.** The plain statement `L < d` quantifies over offsets and asks about
   the joint behaviour of `2 pi(q)` progressions. This one quantifies over a **fixed, `q`-free**
   set of targets `S_7 ∩ [1, d)` and asks about the residues of the **single number `q`**. It is
   a covering statement in `q`-space, not in offset space - formally the same kind of object as
   the Jacobsthal problem the project already knows, but with `q` as the sifted variable.
2. **The target count grows and the per-gear reach does not.** `|S_7 ∩ [1,d)| = 4d/35 + O(1)`
   grows linearly in `q`, while each gear `g > 7` reaches at most two offsets per period `g` and
   so at most `2d/g + 2` islands. That is the only asymmetry the frame supplies, and 5.4 shows it
   is not enough: summed, it gives capacity `2 sum 1/g` per target, above 1.

**What holds for every `q`, or every gear, without exception** (item 7):

| statement | range | exceptions |
|---|---|---|
| `\|Bar(g)\| = (g + 1 - chi_g(2) - chi_g(-2))/4` | every gear `5..20000` (2,260) | **0** |
| every gear is barred somewhere (`\|Bar\| >= 2`) | 2,260 gears | **0** |
| no forward offset is admissible for every gear | offsets `1..20000` | **0** |
| the mirror `i -> d_g - i` preserves `Bar(g)` iff `g = 1 (mod 4)` | 1,125 + 1,135 gears | **0** |
| island count per period `= prod \|Bar(g)\|` | `B = 5..23`, six bounds | **0** |
| exactly `2 chi_g(i)` striking classes of `q mod g`; never odd | 21,531 (gear, offset) cells | **0** |
| **a free `B = 7` island exists in `[1, d)`** | every prime `q` in `(1487, 20000]`, 2,026 primes | **0** |
| the landing is never below the first free island | 2,240 walks (a gate: offsets below `L` are struck) | **0** |

## 7. What is new

Screened against `docs/novel/README.md` (walk-path-parts, walk-path-transforms, walk-tooth-frame,
anchor-235-layer-laws, cover-half-counter-ladder, j2-lower-ladder, layered-erdos-rankin,
corridor-law, golden-spectral-gap), `docs/proofs/`, and the parents' two documents. No entry in
the register defines an unreachable-offset set, its density, or its use as a covering target;
"island" appears nowhere.

* **N-R1 (the bar-size closed form).** `|Bar(g)| = (g + 1 - chi_g(2) - chi_g(-2))/4`, i.e. the
  size of the machine's unreachable set at an offset is fixed by `g mod 8`. Prior art in one
  line: the character sum `sum_x chi(x(x+2)) = -1` and Gauss's second supplement are classical;
  what is new is the object being counted - the offsets of the walk that a gear can never reach.
* **N-R2 (the landscape mirror).** `i -> d_g - i (mod g)` preserves `Bar(g)` exactly when
  `g = 1 (mod 4)` and carries it into the admissible set when `g = 3 (mod 4)`; hence the island
  set is not mirror-symmetric from `B = 7` on. New; the machine's period mirror `k -> -k` is a
  different map.
* **N-R3 (the island system).** The islands for bound `B` are a union of exactly
  `prod_{5<=g<=B} |Bar(g)|` classes mod `P_B` - `4, 12, 48, 192, 960, 5760` at
  `B = 7, 11, 13, 17, 19, 23` - with the explicit small sets `{0,2} mod 5` and
  `{5, 10, 12, 17} mod 35`, densities and maximal gaps as tabulated. The four smallest islands
  take 20.9% of all 2,260 landings. New.
* **N-R4 (the `B = 7` island witness).** For every prime `q` in `(1487, 20000]` some offset
  `= 5, 10, 12, 17 (mod 35)` in `[1, d)` is struck by no gear: `L < d` is witnessed on a fixed
  set of density `4/35`, with 0 exceptions in 2,026 primes and a minimum slack growing (0, 0, 0,
  4, 12 free islands by `q` band). The 17 failures below 1,487 are listed. New.
* **N-R5 (the frame is scale-free).** Large gears strike islands at exactly the machine's own
  rate `2/g` (CRT; mean measured/predicted 0.9956 over 40 gears and 103,899 sightings), so
  strikes/islands `= sum_{B<g<=q} 2/g` to four digits and the counting margin is identical to the
  unrestricted problem. New as a statement about this frame; it is the counter-ladder verdict
  (`cover-half-counter-ladder`) reached in the sparsest coordinates.
* **N-R6 (the doubling).** The number of classes of `q mod g` at which gear `g` strikes a given
  offset is exactly `2 chi_g(i)` and is never odd (0 of 21,531 cells); summed over offsets
  `sum_i chi_g(i) = g - 1`, so the gear's mean rate is exactly `2/g` while it is exactly 0 on a
  quarter of the offsets. The bar concentrates a gear's strikes, it does not reduce them. New.
* Filed, not claimed: `|G(0)| =` the gears `+-1 (mod 8)` (parent P6, classical); the
  fully-reachable offsets `= -6t^2` are the parent's forced-composite columns `k_0 - 6t^2`
  (parent P10) - what is added is the exact exception `g | 6t` with `g != +-1 mod 8`; the
  landing's preference for low `lambda` (parent N-W4) - what is added is that it is exactly the
  island preference and exactly order one; `L` tracking the twin-gap null and the free fraction
  tracking `prod (1-2/g)` are rates, stopped.

## 8. Verdict

**FACT, with one candidate reduction; not a route by itself.**

The landscape exists, is exact, and is now described in closed form: every gear is permanently
barred from a quarter of the offsets, the size decided by `g mod 8`; the islands of any bound
form an explicit residue system with computable spacing; the walk's landing prefers them by a
factor of up to 10; and that preference is exactly what per-gear independence predicts, with no
interaction left over. Two of the parent branches' loose ends are closed by this: the parent's
per-offset depth function is the landscape's relief and is 87% the first four gears, and the
parent's "landing avoids high-depth offsets" is the landing preferring islands.

The one thing of a different kind is **N-R4**: the covering statement `L < d`, which both parents
named as the first unproven interaction, is witnessed on a **fixed set of density `4/35`** for
every prime above 1,487 in the sweep, with growing slack. That is a genuine reduction of what has
to be proved - the target no longer depends on `q` - and it is the branch's contribution toward
the root.

But the branch also proves, exactly, why the reduction does not finish the job. By CRT the large
gears strike the fixed set at exactly the rate they strike everything, `2/g`; the counting margin
is `sum_{B<g<=q} 2/g` regardless of `B`, measured to four digits, crossing 1 at `q = 53, 79,
113, 163` for `B = 7, 11, 13, 17`. Any
route through the islands must therefore use something other than a count - and what is left is
the statement that `q`, as a single number, avoids a fixed residue system for at least one member
of a fixed sparse set. That is the smallest interaction, it is written out in section 6, and it
is a covering problem with `q` as the sifted variable rather than the offset.

Toward the root: no length lever. The branch produces position facts, an exact closed form and
one exact reduction of the covering target, and it stops at the same statement every branch under
R2 and R3 stops at.

## 9. Dead ends (do not re-enter)

* **Counting on the island set.** Strikes/islands is `sum_{B<g<=q} 2/g` for every `B` (four-digit
  agreement at `B = 7, 11, 13, 17`), because CRT makes the large gears' rate on a union of
  classes mod `P_B` equal to their rate everywhere. No choice of `B`, and no sparser `q`-free
  target set of the same kind, can move a counting argument below 1. Sparsity is free of charge
  on both sides.
* **The islands as an enrichment effect at the landing.** The landing's island rate is 0.88-0.99
  of the order-one first-passage null at `B = 5..13`: there is nothing above order one to
  explain. (The naive conditional null `|Bar|/(g-2)` is wrong by 3x at `B = 13`; do not use it.)
* **`B = 13` or higher as the witness frame.** 232 primes up to 18,839 have every `B = 13` island
  of `[1, d)` struck, and 880 have every `B = 17` island struck; only `B = 7` is exception-free
  above a threshold. Raising `B` sparsifies the target faster than it helps.
* **"All islands struck implies the walk passes `d`".** False: 232 primes have every `B = 13`
  island struck and land far below `d`, because the landing is then a non-island opening. The
  only walk with `L >= d` in the sweep is `q = 53` (parent P3).
* **A global mirror on the landscape.** `Bar(g)` is mirror-symmetric only for `g = 1 (mod 4)`;
  from `B = 7` on the island set has no reflection symmetry. The period mirror `k -> -k` does not
  descend to the offset landscape.
