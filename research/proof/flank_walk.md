# Branch R3.h.i - THE FLANK BRICK: the two-sided walk from a tooth

Parent: R3.h (`research/proof/ends_or_middles.md`), whose answer was *the record is made at the
ends*: a record of `M + q'` is a row of ordinary gaps of `M` glued at junctions where the new
gear `q'` strikes an old opening, and `F = flank + letters + flank` with only the two flanks
free.  The observation that spawned this branch is the owner's reading of that answer, recorded
as W5 in `research/proof/the_wall.md`: bricks and mortar are proved objects (merge law, chain
law, kill-spacing, the bare-word cap); **the one unfitted brick is the flank**, and a flank is a
walk - from a junction column `x`, walk in the old machine `M = {5..q}` to the next old opening,
backward (length `L^-`) and forward (length `L^+`).  Branch W.a/W.t decomposed the *one-sided*
walk from `q^2`; nothing on the record decomposes the walk **two-sided and at every junction**.

Scripts in `research/anchor235/r44/`; result outputs (untracked) in
`research/anchor235/r44/results/`.  Every number this document relies on is written here.

---

## 0. Pre-registered (written before any computation of this branch)

### 0.1 Objects, stated exactly

Machine `M = {5..q}`, gears the primes `5 <= g <= q`, `u_g = 6^{-1} mod g`, teeth `+-u_g`.
Period `P = prod_{g <= q} g`.  `q'` = the next prime after `q`.  `d_g = 2 u_g mod g`;
`a_g = min(d_g, g - d_g)` is the gear's **short arc** and `g - a_g` its **long arc**
(`3 a_g = g -+ 1`; kill-spacing law).

* An **opening** of `M` is a column no gear of `M` strikes.
* A **junction** is an opening `x` of `M` that is also a **tooth of `q'`**:
  `x = +u_{q'}` or `x = -u_{q'} (mod q')`.  Both teeth are used throughout.
* The **flanks** at `x` are `L^+(x)` = (next opening of `M` above `x`) `- x` and
  `L^-(x)` = `x -` (previous opening of `M` below `x`).  They are the two consecutive gaps of
  `M` at `x`; `S = L^- + L^+` is the **span** and the two-sided stretch is the blocked run
  `(x - L^-, x + L^+)` minus the single opening at `x`.
* The **forward bucket** `b_g^+(x)` = distance from `x` to gear `g`'s next tooth above `x`;
  the **backward bucket** `b_g^-(x)` = distance to its previous tooth below.  Both are `>= 1`
  because `x` is open.
* The **strike table** of a walk lists, per offset, every gear striking it and which member
  (lower `6k-1` or upper `6k+1`); the **smallest-striker word** is the sequence of smallest
  strikers; the **depth profile** the sequence of strike counts.

The pair statement (prover A, `research/proof/pair_statement.md`) in this language is
`S <= F(M) + q'` at every junction, and the chain statement bounds the deeper fusions.

### 0.2 The theory

The flank is not a free length: it is a walk in a machine whose parts are all proved, and the two
flanks are not two independent walks but **one object seen from both sides**, because L6 makes the
backward tiling the negated forward tiling gear by gear.  The theory is that the negation pins the
two walks to each other **per gear** (an exact identity on the bucket vector) and therefore pins
which gears may act on both sides, but pins nothing about the two lengths jointly; that the
junction condition (`x` a tooth of `q'`) adds nothing at all, because it is a condition modulo
`q'` and the old machine lives modulo `P`, coprime to `q'`; and that the only inverse-shape rule
of the two-sided walk that can hold without exception is the one that runs through the gears the
stretch **misses**, not the gears it uses.

### 0.3 Predictions, with numbers, and what refutes each

**The parts (item 1).**

* **F1 (the arc identity, L6 made exact on the bucket vector).**  At every opening `x` of `M`
  and every gear `g`, `b_g^+(x) + b_g^-(x) in {a_g, g - a_g}` - the sum of the two buckets is one
  of the gear's two arcs, never anything else.  Predicted 0 exceptions over the full periods
  m11..m23.  REFUTED by one pair whose sum is not an arc.
* **F2 (the both-sides bar).**  A gear that strikes at least one column on each side of `x`
  inside the two-sided stretch satisfies `a_g <= S`, i.e. `g <= 3S + 1`.  Predicted 0
  exceptions.  REFUTED by one gear above the bar striking both flanks.
* **F3 (two-sided kill-spacing).**  If `g` strikes forward offset `j` and backward offset `i`
  then `i + j = 0` or `+-d_g (mod g)`; when `i + j < g` this forces `i + j in {a_g, g - a_g}`
  and the two strikes are on **opposite** teeth (`+-d_g`, distance an arc) or on the **same**
  tooth (`0 mod g`, distance a full period).  Predicted 0 exceptions.  (This is the kill-spacing
  law read two-sidedly - a tool, named once; the census of which junctions use which case is the
  finding.)
* **F4 (small gears cannot miss).**  Every gear whose long arc is shorter than the stretch -
  `g - a_g < S - 1` - strikes at least one column of the two-sided stretch.  Predicted 0
  exceptions.
* **F5 (L6 pins nothing jointly).**  Given `L^+`, the conditional distribution of `L^-` at
  junctions has full support on the machine's gap spectrum, for every value of `L^+` with at
  least 30 junctions, at m19 and m23.  REFUTED if some `L^+` value pins `L^-` to a proper subset
  of the realised gaps.

**The pair as an object (item 2).**

* **F6 (the budget, measured).**  `max_junctions S <= F(M) + q'` at m11..m23 with slack at least
  9 (the recorded `F_2` slacks).  Additional pre-registration: over ONE period of `M` the maximum
  over junctions is **strictly below** `F_2(M)` at at least 3 of the 5 machines (junctions are
  only `2/q'` of the openings).  REFUTED by a junction with `S > F + q'`.
* **F7 (exchangeability).**  The junction set is closed under `x -> -x` and `(L^-, L^+)` at `-x`
  is `(L^+, L^-)` at `x`; the joint distribution over junctions is exactly symmetric.  Predicted
  0 exceptions.
* **F8 (the suppression law at junctions).**  Pearson correlation of `(L^-, L^+)` over junctions
  is NEGATIVE at all five machines, of the same size as over all openings (the suppression law is
  a property of the machine, not of the junction condition), and `E[L^- | L^+ >= 0.7 F] < ` mean
  gap.  REFUTED by a positive correlation at any machine, or by junctions differing in sign from
  all openings.

**The junction's own structure (item 3).**

* **F9 (the CRT independence theorem; direction pre-registered as ZERO).**  Over the period of
  `M + q'` the multiset of `(L^-, L^+)` at junctions is exactly the multiset over all openings of
  `M`, each opening class appearing exactly twice.  So the junction condition neither lengthens
  nor shortens the walks - the pre-registered direction is *no effect*.  In ONE period of `M` the
  junction means differ from the all-opening means with no consistent sign across the five
  machines and by less than 3 standard errors at each.  REFUTED by a consistent signed deviation.
* **F10 (column 0 is never a junction, and this does not help).**  `q'` never strikes column 0
  (the shield: `q' | 6*0`), so the twin-Bertrand instance of the pair statement does not sit at a
  junction.  But the CRT translate `x = 0 (mod P)`, `x = +-u_{q'} (mod q')` IS a junction and
  carries the same flanks `(d_0, d_0)`, so the obstruction recurs verbatim.  Predicted: the
  wrap pair appears at junctions at every machine, and the only thing special about column 0 is
  that `q'` is at zero phase there too.  REFUTED if the wrap flanks do not occur at a junction.

**The longest walks (item 4).**

* **F11 (the long flanks are themselves fusions, and share gears).**  At the ten longest `S` per
  machine: each flank's layer nest reaches piece count 1 only at the top two or three gears (no
  flank is a single low-machine gap), and at least one gear strikes both flanks at every one of
  the ten (forced by F4 for the small gears).  REFUTED by a top-ten junction with no shared gear.

**The inverse-shape test (item 5).**

* **F12 (the naive bucket bound fails).**  `L^+ <= b_(1) + b_(2)` (the two smallest forward
  buckets) is REFUTED, with a count; likewise `L^+ <= b_(1) * k` for any fixed `k`.  Pre-registered
  as a failure: the buckets say where the nearest teeth are, not how long the cover lasts.
* **F13 (the missing-gear bound holds).**  For every gear `g` that strikes NO column of the
  forward walk, `L^+ - 1 <= g - a_g` (the long arc); two-sided, for every gear missing the whole
  stretch, `S - 1 <= g - a_g`.  Hence `S <= 1 + (g_miss - a_{g_miss})` for the SMALLEST gear
  missing the stretch.  Predicted 0 exceptions, and predicted to be the ONLY exceptionless
  inverse-shape rule found.

**Toward the root (item 6).**

* **F14 (the certificate at junctions).**  L4's single-gear re-phasing certificate (re-phase one
  gear onto `x`, walk out in the translated machine) has loss `<= q'` at every junction of
  m11..m23.  Predicted 0 exceptions; predicted that the certifying gear is usually NOT the top
  gear.

### 0.4 Scorecard

| # | prediction | verdict | evidence |
|---|---|---|---|
| F1 | bucket sums are arcs | **CONFIRMED** and proved in one line | 0 exceptions in 10,341,945 (opening, gear) pairs over m11..m23 plus every window junction of 152 rungs |
| F2 | both-sides bar `g <= 3S+1` | **CONFIRMED**, and it follows from F1 | 0 exceptions in 4,048,942 (junction, gear) cells |
| F3 | two-sided kill-spacing | **CONFIRMED**; sharpened | the two nearest strikes are on OPPOSITE teeth at every shared gear, `b^+ + b^- =` an arc; short arc 52.0%, long arc 48.0% at m23 |
| F4 | small gears cannot miss | **CONFIRMED and sharpened** to `long arc >= S + 2` | 0 exceptions; band I strikes at 100.00% of 2.28 million cells at m23 and at 100.00% at all 18 deep window rungs |
| F5 | L6 pins nothing jointly | **REFUTED**, and the pinning is the ANCHOR not L6 | `L^+ = 1 (mod 5)` forces `L^- = 0, 2, 4 (mod 5)`; 24 of the m23 `L^+` values have a proper support; 0 exceptions in 7.95 million openings |
| F6 | budget, and max < F_2 in one period | budget **CONFIRMED** (slack 9, 12, 12, 17, 26); second clause **REFUTED** | max span over junctions **equals** `F_2(M)` at m11, m13, m17, m19 and is `F_2 - 2` at m23 |
| F7 | exchangeability | **CONFIRMED** | 0 exceptions in 235,470 junctions; the two mirror classes `x = 2, 3 (mod 5)` carry identical correlations to 4 decimals at all five machines |
| F8 | negative correlation at junctions | **CONFIRMED** in the period, **REVERSED** in the window before detrending | period `r` = -0.319, -0.075, -0.052, -0.040, -0.044 (junctions) against -0.118, -0.062, -0.040, -0.042, -0.045 (all openings); window raw **+0.048**, detrended -0.023 |
| F9 | CRT independence, no effect | **CONFIRMED** (and proved) | mean span difference +0.49, -0.40, +0.07, +0.02, -0.07 s.e. - no sign, no size |
| F10 | column 0 never a junction; obstruction recurs | **CONFIRMED** both halves | wrap flanks `(d_0, d_0)` at 0, 1, 62, 959, 15,107 junctions; and in the WINDOW the bottom junction has `L^- = d_0` exactly at all 28 occurrences |
| F11 | long flanks are fusions and share gears | **CONFIRMED** | 20 of 20 top-ten flanks at m23 are 2- or 3-piece fusions closed by 17, 19 or 23; 10 of 10 top junctions have 6 or 7 shared gears |
| F12 | naive bucket bound fails | **CONFIRMED, and no fixed `k` works** | `L^+ <= b_(1) + b_(2)` fails at 62% of m23 junctions; the smallest sufficient `k` reaches all 7 gears, and at 705 junctions the sum of EVERY bucket is still below `L^+` |
| F13 | missing-gear bound holds | **CONFIRMED** and sharpened to `S <= (g - a_g) - 2`; but it is the umbrella fact, filed as a tool | 0 exceptions everywhere; the bound is TIGHT (slack exactly 2 attained at every machine and every window rung) |
| F14 | L4 certificate loss `<= q'` at junctions | **CONFIRMED** | max loss 4, 6, 12, 13, 18 against `q'` = 13, 17, 19, 23, 29; the certifying gear is not the top gear at 4 of the 5 worst cells |

### 0.5 What this branch could find that is not already known

Known, to be named once and not re-derived: the tooth rule and kill-spacing (`TwoTeeth`), the
chain law and neighbour-of-hit (`AnchorChain`), the merge law, the gear-5 lock, L2/L4/L6 of
`pair_statement.md`, the one-sided decomposition of the walk from `q^2` (W.a, W.t), the
suppression law's phenomenon, R3.h's `F = flank + letters + flank`.  Not on the record: the
two-sided walk at a general junction in any form - no bucket identity, no both-sides bar, no
junction-versus-opening comparison, no census of gears shared by the two flanks, no
inverse-shape rule for a two-sided walk.

---

## 1. Setup (exact ranges)

No sampling anywhere.  Scripts in `research/anchor235/r44/`.

| object | range | script |
|---|---|---|
| every opening and every junction of the full period, all buckets, all strikes, the pair, the correlation, the bands, the layer nests | m11 (P = 385), m13 (5,005), m17 (85,085), m19 (1,616,615), m23 (37,182,145) - 8,354,745 openings, 583,881 junctions | `fw_period.py`, `fw_deep.py` |
| the anchor coupling and the correlation decomposition by `x mod 5` and `x mod 35` | every opening of m11..m23 (8,354,745) | `fw_deep.py` |
| the L4 re-phasing certificate | every junction with `S >= F - 4` (21, 86, 133, 416, 378 cells) | `fw_deep.py` |
| the window: every opening and every junction, the budget, the correlation | every prime rung `q` = 59..997 (152 rungs), columns `q//6 + 1 .. (q'^2-1)/6`; 462,727 window openings, 70 window junctions | `fw_window.py` |
| the window with every gear resolved (bands, umbrella, shared gears) | 18 deep rungs 59..997, all 40,000 openings there, all gears | `fw_window.py` |
| the window's junction identity and the walk-from-a-square identity | all 152 rungs | `fw_extra.py` |
| the inverse-shape family (`k` smallest buckets) | every junction of m11..m23 and every opening of m17 | `fw_shape.py` |

`F(M)` = 7, 11, 18, 25, 34 and `F_2(M)` = 11, 16, 25, 31, 39 at m11..m23, recomputed here from
the full periods and agreeing with the record.  `d_0` = 3, 3, 5, 5, 5.

## 2. Results

### 2.1 The parts: what L6 forces, exactly

**The arc identity (the exact form of L6 on the bucket vector).**  Let `x` be an opening.  The
tooth of `g` immediately above `x` and the tooth immediately below are *adjacent* teeth of `g`,
so their separation is one of `g`'s two arcs, and that separation is `b_g^+ + b_g^-`.  Hence

        b_g^+(x) + b_g^-(x)  in  {a_g,  g - a_g}      for every gear g and every opening x,

with `a_g = min(d_g, g - d_g)` and `3 a_g = g -+ 1`.  **0 exceptions in 10,341,945
(opening, gear) pairs.**  This is L6 ("the left tiling is the negated right tiling") plus the
kill-spacing law, written as an identity on the state vector; it is a tool, named once.

What it forces about `L^-` given `L^+` is exactly two things, and no more:

* **The both-sides bar.**  If `g` strikes a column of the forward flank and a column of the
  backward flank then `b_g^+ <= L^+ - 1` and `b_g^- <= L^- - 1`, so `a_g <= b_g^+ + b_g^- <=
  S - 2`: **a gear can act on both flanks only if `a_g <= S - 2`, i.e. `g <= 3(S-2) + 1`.**
  0 exceptions in 4,048,942 junction-gear cells.
* **The umbrella bar.**  If `g` strikes nothing in the two-sided stretch then `b_g^+ >= L^+ + 1`
  and `b_g^- >= L^- + 1`, so the arc containing `x` is `>= S + 2`, and since the short arc is the
  smaller one, **`g - a_g >= S + 2` for every gear that misses the stretch.**  0 exceptions, and
  the bound is attained (slack exactly 2) at every machine and every window rung.

Nothing else is forced.  In particular the two lengths are *not* pinned to each other: the
identity is per gear, and the walk length is the first offset every progression misses.

**Where the two flanks ARE coupled: the anchor, not L6.**  Openings sit in three classes mod 5
(`{0, 2, 3}`; the teeth are `+-1`).  Since `x`, `x + L^+` and `x - L^-` are all openings, the pair
`(L^-, L^+)` is coupled mod 5:

| `L^+ mod 5` | forced `x mod 5` | allowed `L^- mod 5` | forbidden |
|---|---|---|---|
| 0 | 0, 2, 3 | 0, 1, 2, 3, 4 | none |
| 1 | **2 only** | 0, 2, 4 | 1, 3 |
| 2 | 0, 3 | 0, 1, 2, 3 | 4 |
| 3 | 0, 2 | 0, 2, 3, 4 | 1 |
| 4 | **3 only** | 0, 1, 3 | 2, 4 |

**0 exceptions in 8,354,745 period openings and 462,727 window openings.**  Adding gear 7 leaves
**931 of the 1,225** classes of `(L^- mod 35, L^+ mod 35)` admissible - 24% of the pair classes
are forbidden by `{5, 7}` alone - again 0 exceptions.  This is what produced the conditional
supports the data shows (at m23, 24 of the `L^+` values have a proper `L^-` support; e.g.
`L^+ = 1` allows only `L^- = 2, 4, 5, 7, 9, 10, 12, ... `, all `= 0, 2, 4 (mod 5)`).  It is a
corollary of the anchor's three open classes, order 0, and is filed as such.

**Strikes, words, depths.**  Per junction at m23: gears striking both flanks mean 2.431 (max 7),
one flank 3.601, missing the stretch 0.969; 96,607 of 548,411 junctions (17.6%) have **no** gear
acting on both sides, and 282,098 (51.4%) have every gear striking.  Mean depth of a blocked
column inside a junction stretch: 1.0583, 1.1971, 1.3110, 1.4094, 1.4906 at m11..m23, against the
machine's own `sum 2/g` = 0.8675, 1.0214, 1.1390, 1.2443, 1.3312 - the two-sided stretch is
**12-22% deeper than the machine average**, which is neighbour-of-hit again (the stretch is
conditioned on being blocked).  For a shared gear the two nearest strikes are always on
**opposite** teeth (that is what `b^+ + b^- =` an arc says); the arc used is the short one at
691,957 of 1,332,987 shared cells at m23 (51.9%).

**The three bands.**  Sorting the gears at a junction by size against the span `S`:

| band | condition | may it act? | m11 | m13 | m17 | m19 | m23 |
|---|---|---|---|---|---|---|---|
| I | `g - a_g < S + 2` | **must strike** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| II | `a_g <= S <= g - a_g - 2` | may strike, at most once per side | 0.7273 | 0.7312 | 0.7562 | 0.7646 | 0.7791 |
| III | `a_g > S` | at most one strike in the whole stretch | 1.0000 | 0.3125 | 0.4327 | 0.4162 | 0.4289 |

(fraction of cells in which the gear actually strikes; band I's 1.0000 is proved, not measured).
In the window the same three numbers are 1.0000 / 0.7912-0.7985 / 0.3588 falling to 0.1843 over
the 18 deep rungs `q` = 59..997 - **band II's rate is constant at 0.796 +- 0.004 across a factor
of 17 in `q`**, while band III's falls like the density of the large gears.

### 2.2 The pair as an object

| machine | `q'` | junctions | max span `S` | `F + q'` | slack | `F_2(M)` | argmax junction | flanks |
|---|---|---|---|---|---|---|---|---|
| m11 | 13 | 21 | 11 | 20 | 9 | 11 | `x = 128` | (6, 5) |
| m13 | 17 | 173 | 16 | 28 | 12 | 16 | `x = 122` | (5, 11) |
| m17 | 19 | 2,346 | 25 | 37 | 12 | 25 | `x = 117` | (7, 18) |
| m19 | 23 | 32,930 | 31 | 48 | 17 | 31 | `x = 1,479,287` | (10, 21) |
| m23 | 29 | 548,411 | 37 | 63 | 26 | 39 | `x = 28,300,080` | (22, 15) |

The pair statement holds at every junction with slack 9, 12, 12, 17, 26 - and **the junction
restriction buys nothing**: the maximum over junctions already equals `F_2(M)` at four of the five
machines even inside a single period of `M`.

**Mechanism at the maximum.**  m23, `x = 28,300,080`, tooth `-u'` of 29, flanks (22, 15):

    backward word (outward)  5 23 7 5 7 5 17 11 5 7 5 7 17 5 11 5 7 13 5 19 5   ends on gear 5
    forward  word (outward)  5 7 11 5 19 5 11 13 5 17 5 13 23 5                 ends on 5 and 11
    shared gears  5(1,1 short) 7(2,3 long) 11(3,4 long) 13(8,1 long)
                  17(4,7 long) 19(5,1 short) 23(13,2 long)

Every gear of the machine acts on both sides.  The last blocked column on each side is held by
gear 5 (backward) and by 5 and 11 (forward), so what *ends* both walks is the bottom of the
machine, not the top - the same signature R3.h found for the ends of a record ("the ends of a
record are made at the bottom of the machine").  At m19 the maximum `(10, 21)` ends on gear 11
backward and gear 13 forward; at m17 the maximum `(7, 18)` ends on `{5, 7}` backward and
`{5, 7, 11}` forward.  **At all five maxima the outermost blocked column of each flank is held by
a gear at or below 13.**

**The correlation.**

| machine | junctions `r` | all openings `r` | within `x mod 5` | within `x mod 35` | `E[L^- given L^+ >= 0.7F]` (junctions) | mean gap |
|---|---|---|---|---|---|---|
| m11 | -0.3190 | -0.1177 | -0.0938 | -0.0714 | 2.33 | 2.85 |
| m13 | -0.0754 | -0.0622 | -0.0537 | -0.1180 | 3.50 | 3.37 |
| m17 | -0.0516 | -0.0398 | -0.0401 | -0.1105 | 3.57 | 3.82 |
| m19 | -0.0399 | -0.0421 | -0.0437 | -0.1025 | 3.48 | 4.27 |
| m23 | -0.0438 | -0.0453 | -0.0464 | -0.0892 | 3.15 | 4.68 |

Three things worth stating.  (i) The junction correlation and the all-opening correlation agree
to within the junction sample error at every machine (F9's theorem in action).  (ii) The anchor
coupling does **not** explain the anti-correlation: conditioning on `x mod 5` leaves it
unchanged, and conditioning on `x mod 35` **doubles** it (from -0.045 to -0.089 at m23).  The
suppression is not a residue artefact - the residue structure was partly *masking* it.  (iii) The
mirror check: the correlations in the classes `x = 2` and `x = 3 (mod 5)` are identical to four
decimals at all five machines, which is F7 read off the data.

**Exchangeability.**  `x -> -x` maps openings to openings and junctions to junctions (`+-u_{q'}`
swap), and swaps the flanks: `(L^-, L^+)(-x) = (L^+, L^-)(x)`.  0 exceptions in 235,470 junctions.
So the pair distribution is exactly symmetric and any asymmetry in a measurement is sampling.

### 2.3 The junction's own structure: it forces nothing (proved)

`x` is an opening of `M` **and** a tooth of `q'`.  The first condition lives modulo
`P = prod_{g <= q} g`; the second modulo `q'`, which is coprime to `P`.  Over one period of
`M + q'` (length `P q'`) every opening class mod `P` occurs `q'` times, of which exactly two are
teeth of `q'`.  Therefore:

> **The junction theorem.**  Over the period of `M + q'` the multiset of flank pairs
> `(L^-, L^+)` at junctions is exactly the multiset over all openings of `M`, each with
> multiplicity 2.  In particular `max_{junctions} (L^- + L^+) = F_2(M)` exactly, and the pair
> statement at junctions IS the pair statement.

Measured in one period of `M` alone (where the junctions are a genuine `2/q'` subset), the mean
span at junctions differs from the mean over all openings by +0.49, -0.40, +0.07, +0.02, -0.07
standard errors at m11..m23 - no sign, no size.  The pre-registered direction (no effect) is
confirmed, and the neighbour-of-hit law, which governs `x +- 1` relative to a `q'`-hit, acts on
the machine `M + q'`, not on `M`: it changes which columns are open *after* `q'` is added and
cannot change the old walks at all.

**Column 0.**  `q'` never strikes column 0 (the shield: `q'` divides the midpoint `6*0`), so the
twin-Bertrand instance of the pair statement, which sits at `x = 0` with flanks `(d_0, d_0)` and
correlation `+1`, is not itself a junction.  It recurs verbatim: the CRT translate
`x = 0 (mod P)`, `x = +-u_{q'} (mod q')` is a junction with the same flanks, and the wrap pair
`(d_0, d_0)` is realised at 0, 1, 62, 959 and 15,107 junctions of m11..m23.  The **only**
difference at column 0 is that `q'` is at zero phase there too, which is a codimension-one extra
condition on a point that already has every gear of `M` at zero phase - one point per period of
`M`.  The junction restriction therefore does not weaken the obstruction by one bit.

### 2.4 The mechanism of the longest two-sided walks

Layer nest of each flank at the ten longest junctions of m23 (`k_g` = pieces of the flank at
layer `g`):

| junction | flank | `k_5, k_7, k_11, k_13, k_17, k_19, k_23` | closed by | pieces joined |
|---|---|---|---|---|---|
| `x=28300080` (22, 15) | back 22 | 13, 8, 6, 5, 3, 2, 1 | 23 | 2 |
| | fwd 15 | 9, 8, 6, 4, 3, 2, 1 | 23 | 2 |
| `x=22669005` (28, 8) | back 28 | 17, 11, 8, 7, 5, 3, 1 | 23 | 3 |
| | fwd 8 | 5, 5, 4, 2, 1, 1, 1 | 17 | 2 |
| `x=2278090` (15, 20) | back 15 | 9, 8, 6, 5, 4, 3, 1 | 23 | 3 |
| | fwd 20 | 12, 7, 6, 4, 4, 2, 1 | 23 | 2 |
| `x=25175978` (26, 9) | back 26 | 16, 11, 8, 6, 5, 3, 1 | 23 | 3 |
| | fwd 9 | 5, 4, 4, 3, 2, 1, 1 | 19 | 2 |

**20 of 20 flanks in the top ten are themselves fusions**, closed by 17, 19 or 23, joining 2 or 3
pieces - never a single gap of a lower machine.  So the brick is not an atom: *a flank of a
record of `M + q'` is itself a record-shaped object of `M`, made at ITS top layers*, and
`F = flank + letters + flank` recurses one level down.  Each flank at these junctions is cut into
9-17 pieces by gear 5 and closed by one of the top three gears, exactly the shape R3.h found for
a record.

**Shared gears.**  At all ten top junctions of m23, six or seven of the seven gears strike both
flanks; the largest shared gear is 23 at nine of the ten.  Over all junctions the mean is 2.431
and 17.6% share nothing.  The sharing is forced from below by the umbrella bar (band I must
strike, and a band-I gear whose one arc is shorter than each flank strikes both) and permitted
from above by the both-sides bar (`a_g <= S - 2`); between the bars, band II shares at a rate that
is constant in `q`.

### 2.5 The inverse-shape test: no bucket statistic bounds the walk

The smallest `k` for which `L^+ <=` the sum of the `k` smallest forward buckets:

| machine | gears | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | no k works |
|---|---|---|---|---|---|---|---|---|---|
| m13 | 4 | 22 | 74 | 42 | 31 | - | - | - | 4 |
| m17 (all openings) | 5 | 2,457 | 8,240 | 5,658 | 4,670 | 1,079 | - | - | 171 |
| m19 | 6 | 3,221 | 10,609 | 7,670 | 7,942 | 2,559 | 808 | - | 121 |
| m23 | 7 | 48,293 | 158,121 | 116,783 | 139,145 | 57,080 | 22,607 | 5,677 | 705 |

`L^+ <= b_(1) + b_(2)` fails at 62.4% of m23 junctions; the required `k` grows with the machine;
and at 705 junctions of m23 the sum of **every** bucket in the machine is still below `L^+`.
`L^+ >= b_(1)` also fails (48,293 times at m23 - whenever the nearest tooth is not adjacent, the
walk stops at once).  The same test two-sided (`S` against the `k` smallest `b^+ + b^-`) needs
`k = 7` at 22 junctions of m23.

**The one exceptionless rule is the umbrella bound, and it uses the gears the walk MISSES:**
`S <= (g - a_g) - 2` for every gear `g` striking no column of the stretch, hence
`S <= min_{g misses} (g - a_g) - 2`.  0 exceptions everywhere, and tight (slack exactly 2 is
attained at every machine and every window rung).  At m23 the smallest missing gear is 7 to 23
(median slack 5, max 12); in the windows it is 7 at every rung with median rising 29 -> 79 over
`q` = 59..997 and median slack rising 9 -> 17.  This is the umbrella fact of the record
(`umbrellas-and-shields`), read on the two-sided walk; it is a tool, and the census is the
finding.  It does not bound `S` uniformly, for the reason node 5g already gives: which gear is
the smallest that misses is not controlled, and it grows with the machine.

### 2.6 The window band, `q` = 59..997

**The window has at most two junctions, and we know which.**  Inside the window both members of an
opening are prime, so a member divisible by `q'` must **be** `q'` or `q'^2` (a larger multiple of
`q'` has a cofactor `>= q'`, putting it above the window).  So:

> The junctions of the window of `M = {5..q}` are exactly the column of `q'` (present iff `q'` is
> a member of a twin pair) and the column of `q'^2` (present iff `q'^2 - 2` is prime, which is
> W.a's SQUARE GATE).  **0 mismatches over 152 rungs**; 28 bottom junctions, 42 top junctions,
> 70 in all, and **0 of any other kind**.

Two identifications follow, both checked exactly:

* **The top junction is the walk from a square.**  At the `q'^2` column the flanks under `{5..q}`
  equal the two-sided walk from `q'^2` under `{5..q'}` - branch W.a's `L(q')` forward and W.t's
  `L^-(q')` backward.  **0 mismatches of 42.**  (The reason is N-W5: `q'` strikes no other column
  within its arc, so adding it changes no opening of the stretch.)  So the flank brick at the top
  of the window IS the object W.a and W.t already decomposed one-sidedly, and this branch supplies
  its second side.
* **The bottom junction carries `d_0`.**  At the `q'` column, `L^- = u_{q'} = round(q'/6)` at all
  28 occurrences: the previous opening of `M` is column 0 itself, because every column strictly
  between 0 and the first prime above `q` has a member with a factor `<= q`.  So the bottom
  junction's backward flank is exactly `d_0(M)`, the twin-Bertrand quantity of node 1e.  Spans at
  the bottom junction: min 17, median 72, max 170; at the top junction: min 5, median 28, max 93.

**The budget in the window** (over all window openings, not only the two junctions): the largest
two-sided span is below `F_W + q'` at **152 of 152 rungs**, with slack min 54, median 446, max 928;
`max S / F_W` runs 1.120 to 1.923 with median 1.292.

**The correlation flips sign in the window.**  Raw over 462,727 window openings: **+0.0479**.  The
cause is the twin-density trend across the window (gaps near `q'^2` are longer than gaps near
`q`), and it is not a property of the pattern: normalising each gap by the local mean over blocks
of 200 openings gives **-0.0233**, the same sign and about half the size of the period value.  The
suppression law survives in the window only after detrending - worth recording, because a raw
window correlation would read as *positive* dependence.

### 2.7 The L4 certificate at junctions

Re-phasing one gear `q0` so that a tooth lands on `x` turns `M` into a translate of itself, so the
blocked run through `x` in the re-phased machine is a genuine gap of `M`.  Over every junction with
`S >= F - 4`:

| machine | cells | min loss | median | max loss | `q'` | exceptions |
|---|---|---|---|---|---|---|
| m11 | 21 | -3 | 0 | 4 | 13 | 0 |
| m13 | 86 | -4 | -1 | 6 | 17 | 0 |
| m17 | 133 | -4 | 0 | 12 | 19 | 0 |
| m19 | 416 | -4 | 0 | 13 | 23 | 0 |
| m23 | 378 | -4 | 2 | 18 | 29 | 0 |

(loss `= S - cert`; negative loss means the re-phased machine has a run longer than the span.)
The certifying gear at the worst cell is 11, 5, 13, 17, 17 - the top gear at none of the five.
The ratio `max loss / q'` runs 0.31, 0.35, 0.63, 0.57, 0.62: the certificate covers the pair
statement at every junction with content, but its margin is not shrinking.

## 3. Mechanism

A junction is an ordinary opening.  That is the first and most consequential thing this branch
found, and it is a theorem, not a measurement: the junction condition is a congruence modulo `q'`,
the old machine is periodic modulo `P`, and `gcd(P, q') = 1`, so over the period of `M + q'` the
flank pairs at junctions are the flank pairs at all openings, twice each.  Everything the record
hoped the junction condition might supply - a shorter walk, a pinned residue, a neighbour law - it
does not supply, and cannot.

What DOES structure the two-sided walk is two bars, one from below and one from above, both exact
corollaries of the arc identity `b_g^+ + b_g^- in {a_g, g - a_g}`:

* every gear whose long arc is shorter than `S + 2` **must** strike the stretch (band I, 100.00%
  of 2.28 million cells at m23 and of every deep window rung);
* every gear whose short arc exceeds `S` can strike **at most once** in the whole two-sided
  stretch, and can never strike both sides (band III).

Between them sits band II, whose gears strike at a rate of 0.796 that does not move over a factor
of 17 in `q`.  So the two-sided stretch is covered by a bottom that is forced, a top that is nearly
inert, and a middle whose participation rate is constant - and the length is decided in the middle.
That is why no bucket statistic bounds the walk: the bucket vector is the state of the bottom
(gear 5 or 7 is the nearest tooth most of the time), while the length is decided by which of the
band-II and band-III gears happen to align, and the number of them needed grows with the machine
(the sum of ALL buckets fails to reach `L^+` at 705 junctions of m23).

The coupling that does exist between the two flanks is the anchor's, not L6's.  L6 is per gear:
it says the two buckets of one gear sum to an arc, which pins which gears can act on both sides
and which cannot miss, but says nothing about the two lengths.  The anchor pins the two lengths
mod 5 (and, with gear 7, forbids 24% of the pair classes mod 35), and this is exactly what the
conditional supports show.  But the anti-correlation of the two flanks is not the anchor: it
survives conditioning on `x mod 5` unchanged and **doubles** under conditioning on `x mod 35`.

Finally, the flank is not an atom.  At every one of the twenty longest flanks at m23 the flank is
itself a two- or three-piece fusion closed by one of the top three gears - the shape R3.h found
for a record.  The brick recurses: `F = flank + letters + flank` where each flank is again
`flank + letters + flank` one machine down, and the recursion bottoms out only at the gears 5 and
7 that end every long walk (at all five period maxima the outermost blocked column of each flank
is held by a gear `<= 13`).

## 4. Toward the root: what would have to be proved

The flank brick's bound is the pair statement, and in the two-sided-walk language it reads:

> **(PS-walk)** For every machine `M` and every opening `x`, the two-sided walk from `x` has
> `L^-(x) + L^+(x) <= F(M) + q'`.

By the junction theorem, restricting `x` to junctions changes nothing, so no gain is available
there.  What is proved, and what is not, on the way from the parts to that bound:

*Proved and used here:* the tooth rule and kill-spacing; L6 in its exact bucket form (the arc
identity); the two bars that follow from it (band I must strike, band III at most once); the
umbrella bound `S <= (g - a_g) - 2` for any missing gear; the mirror (exchangeability); the merge
law (each flank is itself a fusion); L4 (re-phasing gives a certificate); the anchor's mod-5 and
mod-35 coupling of the pair.  The chain law and neighbour-of-hit act on `M + q'` and are therefore
**not** available to the old walks - a correction worth recording, since the brief expected
neighbour-of-hit to bear on `x +- 1`.

*The lowest-order interaction not proved.*  Write `G_miss(x)` for the smallest gear that strikes
no column of the two-sided stretch at `x`.  The umbrella bound is exact and tight:
`S(x) <= (G_miss - a_{G_miss}) - 2 ~ (2/3) G_miss`.  The record's own `F(M)` obeys the same bound
with its own missing gear.  So PS-walk would follow from

> **(H)** at every opening `x`, `(2/3) G_miss(x) <= F(M) + q'`,

i.e. *the smallest gear that misses a two-sided stretch is at most `(3/2)(F + q')`*.  Everything
below (H) is proved; (H) itself is a joint statement about `pi(q)` gears - it says the gears below
a threshold cannot all strike the stretch - and it is the same unbounded-order covering statement
W.a named for the one-sided walk (`L < d`).  It is not reachable by composing the pairwise laws:
band II's participation is a rate, and node 5g's hinge result already shows the relevant gear
falls as the machine grows at fixed length.  So the first unproven interaction on this brick is
**the joint miss statement (H)**, and it is the two-sided form of the wall's face A.

*The column-0 obstruction, exactly.*  At `x = 0` every gear of `M` is at zero phase: `b_g^+ =
b_g^- = u_g` for all `g`, the flanks are `(d_0, d_0)`, and the mirror makes the correlation `+1`.
That configuration is not a junction (`q'` cannot strike 0), but its CRT translate is, with the
same flanks; and in the WINDOW the bottom junction has `L^- = d_0` exactly at all 28 occurrences,
so the twin-Bertrand quantity is literally one of the two flanks of the window's own bottom
junction.  The other junctions do not carry the same difficulty in the same form - at a generic
junction the two flanks are independent-looking and anti-correlated, and L4's certificate closes
the gap with loss `<= q'` at every junction with content - but they carry a weaker version of it:
the certificate's loss is a statement about where one gear's sole columns sit, and its margin
(`max loss / q'` = 0.31, 0.35, 0.63, 0.57, 0.62) is not shrinking with the machine.

## 5. What holds without exception (with counts)

1. **The arc identity.**  `b_g^+ + b_g^- in {a_g, g - a_g}` at every opening and every gear.
   0 exceptions in 10,341,945 pairs.  (Tool: L6 + kill-spacing.)
2. **The both-sides bar.**  A gear acting on both flanks has `a_g <= S - 2`.  0 exceptions in
   4,048,942 junction-gear cells.  (Proved from 1.)
3. **The umbrella bar.**  A gear missing the two-sided stretch has `g - a_g >= S + 2`.  0
   exceptions; attained with slack exactly 2 at every machine and every window rung.  (Proved
   from 1; the umbrella fact.)
4. **Band I strikes.**  Every gear with `g - a_g < S + 2` strikes the stretch: 100.00% of
   2,281,412 cells at m23 and of every one of the 18 deep window rungs.
5. **The anchor coupling of the pair.**  `L^+ = 1 (mod 5)` forces `L^- = 0, 2, 4 (mod 5)`;
   `L^+ = 4` forces `L^- = 0, 1, 3`; `L^+ = 2` forbids `L^- = 4`; `L^+ = 3` forbids `L^- = 1`;
   `L^+ = 0` free.  0 exceptions in 8,354,745 period openings and 462,727 window openings.  With
   gear 7: 931 of 1,225 pair classes mod 35 admissible, 0 exceptions.
6. **Exchangeability.**  `(L^-, L^+)(-x) = (L^+, L^-)(x)` and junctions map to junctions.
   0 exceptions in 235,470 junctions; the mirror classes `x = 2, 3 (mod 5)` carry identical
   correlations at all five machines.
7. **The junction theorem.**  Over the period of `M + q'` the junction flank multiset is the
   all-opening multiset doubled; measured deviation in one `M`-period: `|mean difference|` at most
   0.49 s.e. at m11..m23, no consistent sign.
8. **The window's two junctions.**  Exactly the column of `q'` (iff `q'` is a twin member) and the
   column of `q'^2` (iff `q'^2 - 2` is prime).  0 mismatches over 152 rungs, 70 junctions.
9. **The top junction is the square walk.**  Flanks at the `q'^2` column under `{5..q}` equal the
   two-sided walk from `q'^2` under `{5..q'}`.  0 mismatches of 42.
10. **The bottom junction carries `d_0`.**  `L^- = round(q'/6) = d_0(M)` at all 28 occurrences.
11. **The pair statement at junctions.**  0 violations at 583,881 period junctions (slack 9, 12,
    12, 17, 26) and at 70 window junctions; and over all window openings `max S <= F_W + q'` at
    152 of 152 rungs.
12. **The L4 certificate.**  Loss `<= q'` at every junction with `S >= F - 4`: 0 exceptions in
    1,034 cells.

## 6. What is new

Screened against `docs/novel/README.md` (walk-path-parts, walk-path-transforms, suppression-law,
renewal-ladder, corridor-law, anchor-235-layer-laws, reachability-landscape), `docs/proofs/`,
`docs/proof-search/alignment-rules.md`, `research/proof/pair_statement.md`,
`research/proof/ends_or_middles.md`.

* **The junction theorem** (2.3) - no located prior art.  It is the sharp answer to "what else is
  forced at a junction": nothing, by CRT, and therefore the pair statement at junctions is the
  pair statement.  It also closes, before it is opened, any programme that hopes to prove the
  budget inequality by exploiting the junction condition.
* **The two bars and the three bands** (2.1) - the exact bucket form of L6 is a restatement, but
  the two bars it yields for a TWO-SIDED stretch, the band decomposition by gear size, and the
  measured constancy of band II's participation (0.796 +- 0.004 over `q` = 59..997) are not on the
  record.
* **The anchor coupling of the flank pair** (2.1) - order 0, elementary, but not on the record in
  pair form; it is what produces the conditional supports, and it is measured to be NOT the source
  of the anti-correlation.
* **The correlation decomposition** (2.2) - that conditioning on `x mod 35` *doubles* the flank
  anti-correlation, and that the raw window correlation is POSITIVE (+0.048) and only turns
  negative (-0.023) after detrending against the twin density.  The suppression law's phenomenon
  is on record; this refinement is not, and the sign flip is a trap worth recording.
* **The window's junction classification** (2.6) - that a window has at most two junctions, that
  they are the columns of `q'` and `q'^2`, that the top one exists exactly under W.a's square gate
  and its flanks ARE the two-sided walk from a square, and that the bottom one's backward flank is
  exactly `d_0`.  None of this is on the record and it links R3.h, W.a/W.t and node 1e into one
  object.
* **The flank recursion** (2.4) - that every one of the twenty longest flanks at m23 is itself a
  two- or three-piece fusion closed by one of the top three gears, so `F = flank + letters +
  flank` recurses and the brick is not an atom.
* **The inverse-shape negative** (2.5) - no fixed number of buckets bounds the walk, and at 705 of
  548,411 m23 junctions the sum of every bucket in the machine is below `L^+`.  The only
  exceptionless bucket rule is the umbrella bound, which uses the gears the walk misses.

Not new, named once and not re-derived: the tooth rule, kill-spacing and umbrellas; L6 and L4 of
`pair_statement.md`; the merge law; the gear-5 lock; the suppression law's phenomenon; W.a's
`L < d` and N-W5; R3.h's `F = flank + letters + flank`; node 5g's hinge.

## 7. Verdict

**FACT (exact, kept), with one theorem that closes a route.**  The flank brick has been taken
apart two-sidedly at every junction of m11..m23 and at every window junction of `q` = 59..997.
Its parts are all proved and now stated exactly for the two-sided walk: the arc identity, the
both-sides bar, the umbrella bar, the anchor coupling, exchangeability, and the recursion of the
flank into a fusion of its own.  The one structural hope the branch was opened to test - that a
junction is a special opening - is **false, and provably so**: by CRT the junction condition is
independent of the old machine, the flank distribution at junctions is the flank distribution at
all openings, and `max S` at junctions is exactly `F_2(M)`.  So the flank brick cannot be fitted by
finding structure in the junction; the pair statement is the whole of it.

Toward the root: the branch supplies the exact two-sided statement of the parts and names the
lowest-order unproven interaction, (H) - *the smallest gear that misses a two-sided stretch is at
most `(3/2)(F + q')`* - which is the two-sided form of the same joint covering statement W.a met
one-sidedly.  It is not a length lever, and the branch says why: the bottom of the machine is
forced to strike, the top is nearly inert, and the length is decided by a band whose participation
rate is constant in `q` and whose members needed grow without bound.

## 8. Dead ends (do not re-enter)

* **The junction as a special opening.**  Proved impossible (2.3).  No measurement conditioned on
  "`x` is a tooth of `q'`" can differ from the same measurement over all openings, over the true
  period.  This also kills the idea of using neighbour-of-hit on the old walks: that law acts on
  `M + q'`, not on `M`.
* **Bucket statistics as a length bound.**  The required number of buckets grows with the machine
  and at 705 m23 junctions no number suffices (2.5).  This is the order-one ceiling of W.a met
  two-sidedly.
* **L6 as a joint pin on the two lengths.**  L6 is per gear.  It gives the two bars and nothing
  more; the joint coupling that exists is the anchor's, and the anti-correlation is neither.
* **The raw window correlation as evidence of dependence.**  It is +0.048 and it is the twin
  density trend; only the detrended value (-0.023) is about the pattern.
* **Restricting the pair statement to junctions to make it easier.**  Same statement, by the
  junction theorem, and the maximum is already attained at a junction inside a single period at
  four of the five machines.
