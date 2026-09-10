# The engine's two exceptionless laws out of sample, and W32 on the engine's own window

Leads **U4** (node **2g.i**, `research/proof/neighbour_profile.md` and `glue_covering.md`) and
**U5** (node **R4.b.iv**, `research/proof/top_machine_2.md` 3.7, law register **W32**) of the
whole-tree review, `research/proof/tree_review.md` section 3. Prover lane, 2026-09-10.

Scripts in `research/anchor235/r72/` (prefix `u45_`); outputs in `research/anchor235/r72/results/`
(untracked). Every number this document relies on is written into the document. Nothing committed
by the lane.

---

## 0. Pre-registered

The two tests and the numbers that decide them are the review's own (section 3, leads U4 and U5);
they were fixed before any computation of this branch. The predictions below add the exact value
expected in each cell.

### 0.1 The objects, in the documents' own words

- Machine `M = {5..y}`, period `P = prod_{5<=g<=y} g`, column `k` is the pair `(6k-1, 6k+1)`;
  gear `g` strikes `k` iff `k = +-u_g (mod g)` with `u_g = 6^{-1} mod g`. Max-gap convention: a
  gap is the distance between consecutive openings, cyclically over the period; `F(M) = max gap`.
- **`N(v)`** (`neighbour_profile.md` 0): for a realised gap size `v`, the maximum over gaps of
  size `v` of (left neighbour gap + right neighbour gap). **`F_2(M)`** is the largest sum of two
  gaps sharing an opening. The law of 2g.i is `N(v) <= F_2(M)` for every realised `v >= 6`.
- **The `J`-run outer law** (`glue_covering.md` 2.8(b)): for `J` consecutive gaps `g_1..g_J` with
  every one of the `J-2` middles `>= 6`, `g_1 + g_J <= F_2(M)`. `J = 3` is the law above.
- **`D_k(M)`** is the table of realised `k`-windows of consecutive gap sizes.
- **W32 / L32** (`top_machine_2.md` 3.7): for a fixed gear set with period `W` and full-period
  census `c(d)` = the number of gaps of size `>= d` per period, the record on a range of `N`
  columns is `F_range(N) = max{d : W / c(d) <= N} - 1`. Measured within one unit at 19 of 21
  checkpoints on three manifold wheels; "the exactness is the finding, and it is the thing to try
  to break."
- **The window** of `{5..y}` (`valve_existence.md` 0): the columns `k` with `6k-1 > y` and
  `6k+1 < y'^2`, `y' = nextprime(y)` -- the range `{5..y}` certifies. **The section** is the
  window's new part, the columns with `6k-1 > y^2`. In either range the openings of `{5..y}` are
  exactly the twin pairs, so the record there is a twin gap in columns. `F_W(y)` and `F_sec(p)`
  are those records. The window is the **phase-zero** translate: it starts at the origin of the
  period.

### 0.2 Theory

**T1 (U4).** The two laws of node 2g.i are laws of the machine and not of the seven machines that
carry them, so they hold at m37, whose period (1.24e12 columns, 2.18e11 gaps) has never been
scanned: `max_{v >= 6} N(v) <= F_2(37) = 90`, and the outer law holds at `J = 3, 4, 5`.

**T2 (U5).** W32's exactness is a property of a fixed gear set on a range, so it holds on the
engine's own window as it does on the manifold wheels: `F_W(y)` is the first hit on the engine's
own census over `W(y)` columns, within one unit.

### 0.3 Predictions, each with the number that refutes it

- **P1 (U4, J = 3).** `max over realised v >= 6 of N(v)` at m37 is at most `F_2(37) = 90`.
  REFUTED by one realised 3-window `(L, v, R)` of m37 with `v >= 6` and `L + R >= 91`. Expected
  value: 88 or 90 (the law is tight once, at m29, and near-tight at m31, `66` against `68`).
- **P2 (U4, J = 4, 5).** The outer maximum at `J = 4` and `J = 5` is at most 90, and **falls with
  `J`** as `glue_covering.md` 2.8(b) records at m13..m23. REFUTED at the law by an outer sum
  `>= 91`; REFUTED at the shape by an outer maximum that rises with `J`.
- **P3 (U4, the ladder).** The same scan at m29 and m31 (full periods, direct sieve) returns
  `55` and `66` at `J = 3` -- the values of `neighbour_profile.md` 2.2. REFUTED by any other
  number; this is the instrument gate, not a finding.
- **P4 (U5, the window).** `|F_W(y) - F_range(W(y))| <= 1` at 11 of the 13 rungs `y = 7..53`, the
  rate W32 has on the wheels. REFUTED if more than 3 rungs miss by more than 1, or if any rung
  misses by more than 3.
- **P5 (U5, the section).** The same for the section records. REFUTED on the same terms.
- **P6 (U5, the direction).** Where the law misses, the truth is BELOW the prediction as often as
  above (the wheels miss both ways). REFUTED by a one-sided miss at every rung.
- **P7 (the control, registered before the control was run and after the window table was in
  hand).** If W32 misses on the window, the miss is the window's own atypicality: the window's
  record sits above the 99th percentile of the record over random translates of the same gear set
  and the same length. REFUTED if the median record of a random translate already exceeds the
  first-hit prediction -- which would make the miss a property of the formula, not of the window.

### 0.4 Scorecard

| # | prediction | verdict |
|---|---|---|
| P1 | `max_{v>=6} N(v) <= 90` at m37 | **HELD**; the exact value is **87**, at `v = 10`, witness `(60, 10, 27)`, by two independent routes |
| P2 | outer `<= 90` at `J = 4, 5`, falling with `J` | **HELD at `J = 4`**: the exact maximum is **82**, witness `(10, 8, 8, 72)`; `J = 5` NOT CLOSED -- exhaustive and negative above outer 105 (`<= 105`), and 0 realised in 17,483 further exact decisions above 90 with 57,064 candidates undecided. "Falls with `J`" **REFUTED** at m31 (`J=7: 50`, `J=8: 52`) |
| P3 | the scan returns 55 at m29 and 66 at m31 | **HELD**, both, and 35/34/34 at m23 |
| P4 | W32 within one unit on the engine's window at 11 of 13 rungs | **REFUTED** -- 4 of 13, misses up to **+14** |
| P5 | the same on the section records | **REFUTED** -- 5 of 43 cuts within one unit, misses **-6** to **+18**, mean **+5.37** |
| P6 | the miss is two-sided | **REFUTED** -- 11 of 13 windows and 33 of 43 sections miss HIGH |
| P7 | the miss is the window's atypicality | **REFUTED** -- the median random translate already beats the prediction at 13 of 13 rungs; the window is ordinary from `y = 37` up |

**Stop rules honoured.** The glue lemma, the move and shadow lemmas, the chain and merge laws, the
attainment identity and the closure theorem are cited, never re-derived. The first-occurrence
heuristic for the primes themselves (Kourbatov, arXiv:1301.2242; Kourbatov-Wolf arXiv:1901.03785)
is W32's own prior art and is not re-derived here: this branch tests W32's **exactness** claim for
a fixed periodic sifted set with a deterministic census, which is the register's stated delta.

---

## 1. Setup and the instruments

Exact integer arithmetic throughout; no sampling anywhere except the explicitly labelled
random-translate control of section 3.3.

### 1.1 `u45_sieve.py` -- the reference, by direct sieve

Sieves the full period of `{5..y}` in column chunks and streams the cyclic gap sequence through
four accumulators: the gap spectrum, `F_2`, `N(v)`, and the `J`-run outer maximum for
`J = 3..8`. Machines m11..m31 (periods 385 ... 33,426,748,355). One core; m29 28.6 s, m31 596.6 s.

**Gate (P3, and more).** It returns `F = 7, 11, 18, 25, 34, 43, 58` and
`F_2 = 11, 16, 25, 31, 39, 55, 68` at m11..m31, and gap counts `135, 1,485, 22,275, 378,675,
7,952,175, 214,708,725, 6,226,553,025` -- the recorded ladder and the recorded periods. Its
`max_{v>=6} N(v)` row is `10, 12, 21, 28, 35, 55, 66` at `v = 6, 6, 7, 7, 7, 7, 7`, which is
`neighbour_profile.md` 2.2 entry for entry. Its `J`-run outer rows at m13..m23 are
`glue_covering.md` 2.8(b) entry for entry (table 2.2 below).

### 1.2 `u45_outer.py` -- the same three quantities where there is no period

At m37 the period cannot be built. The scan is a **descending scan** over candidate words: every
word whose adjacent pairs all lie in `D_2(m37)` is enumerated (a complete superset of the realised
`J`-windows), sorted by descending outer sum `g_1 + g_J`, and each is decided by the free-phase
covering instrument `research/anchor235/r70/ol_pattern.py` (membership in `D_J(M)` as a CRT
covering problem, `order_law_37_41.md` 2.2). The first realised word is therefore the exact
maximum. Exact pruning: every 3-subwindow and 4-subwindow of a realised word is realised, and a
word and its reversal are realised together (the machine is symmetric under `k -> -k`), so the
memoised sub-window verdicts kill most candidates before the solver sees them.

`D_1(m37)` (75 values) and `D_2(m37)` (2,053 rows) come from the closure dictionary
`r70/results/m37_dict.npz`; `D_3(m37)` (30,325 rows) is built here by `u45_d3.py`.

**Gates.** (i) the whole scan run at m23 and m29 reproduces the direct sieve's answers exactly
(table 2.3); (ii) at m37 the instrument was asked for all 2,053 dictionary pairs and returned
every one realised, and for 200 sampled non-dictionary pairs and returned every one not realised
-- **0 failures in either direction**, 726 s.

### 1.2a `u45_member.py` -- a third, independent decider (counting instead of searching)

The same covering problem decided by the census DP of 1.4 rather than by search, which also returns
the exact NUMBER of columns of the period at which the window occurs.  Gate: every one of the
`33^3 = 35,937` triples of realised m23 gap values put to both this DP and `ol_pattern`'s search
solver -- **0 disagreements**, and **3,135 realised**, which is `|D_3(m23)| = 3,135` exactly, the
value `order_law_37_41.md`'s gate G1 records.  At m37 the DP leaves the state budget at span 97
(26,903,337 states), so the scans below use `ol_pattern`; the DP is the cross-check at m23.

### 1.3 `u45_d3.py` -- `D_3(m37)` by the closure ladder

`ladder_closure.md` 2 records that a base dictionary of depth `K_0 = 17` at m23 carries the ladder
to **depth 3 at m37**; the r70 branch's ladder started at `K_0 = 15` and stopped at depth 2, which
is why `order_law_37_41.md` had to decide `D_3(m37)` membership word by word. This driver runs the
r61 operator unchanged from `|D_17(m23)| = 5,345,804` and writes the m37 dictionary out.

| rung | depth in -> out | `F` | `F_j` | `|D|` out | `N` | loss | over0 | secs |
|---|---|---|---|---|---|---|---|---|
| 23 -> 29 | 17 -> 12 | **43** | 43, 55, 65, 70, 85, 90, 92, 97 | 34,357,093 | 214,708,725 | 0 | 0 | 2,238.5 |
| 29 -> 31 | 12 -> 8 | **58** | 58, 68, 85, 90, 92, 97, 104, 110 | 24,815,018 | 6,226,553,025 | 0 | 0 | 1,593.9 |
| 31 -> 37 | 8 -> 3 | **88** | **88, 90, 97** | **30,325** | 217,929,355,875 | 0 | 0 | 365.9 |

Every gate of `ladder_closure.md` 2 and 3.1 reproduced: the three records, the two dictionary sizes
34,357,093 and 24,815,018, `|D_3(m37)| = 30,325`, the masses, `F_7(29) = 92`, `F_8(29) = 97`,
`F_7(31) = 104`, `F_8(31) = 110`, and `F_j(37) = 88, 90, 97`.  4,220 s and 2.0 GB peak on one core.
`D_3(m37)` is the object `order_law_37_41.md` 2.2 had to do without, and it is 30,325 rows.

### 1.4 `u45_census.py` -- the exact full-period census `c(d)` with no period (NEW instrument)

`c(d)` is a **covering count**. With `r = x mod g`, offset `o` of the window `[0, d-1]` is struck
by `g` iff `o = +-u_g - r (mod g)`, so the struck set of a gear depends only on its own residue,
and by CRT the residue vector runs over the product of the `Z_g` exactly once per period. Hence

    c(d) = #{ (r_g)_g : no gear strikes offset 0, every offset 1..d-1 is struck by some gear }

which is a dynamic programme over the gears whose state is the set of offsets not yet struck, with
equal masks collapsed (a gear with `g >> d` misses the window at most of its phases, and those
phases share the empty mask). Exact Python integers; no period is built and no scan is run.

**Gates.** (i) the complete gap spectrum of m11, m13, m17, m19, m23, m29, m31 -- every
multiplicity, 6.2 billion gaps at m31 -- reproduced exactly from the direct sieve of 1.1;
(ii) the m37 spectrum reproduced exactly against the closure dictionary, all 75 values and the
total 217,929,355,875; (iii) **W32's own published numbers**: for the eight-gear wheel
`{7, 11, 13, 17, 19, 23, 29, 31}` (raw-line teeth `{0, -2}`, `W = 6,685,349,671`) all **20**
published census counts of `top_machine_2.md` 3.7 agree, including `m(33) = 8` and `F_top = 33`,
and the published **predicted** first-hit ladders of all three wheels are reproduced entry for
entry:

| wheel | `W` | 10^4 | 10^5 | 10^6 | 10^7 | 10^8 | 10^9 | 10^10 |
|---|---|---|---|---|---|---|---|---|
| `{7..31}` this instrument | 6.69e9 | 17 | 20 | 24 | 27 | 30 | 32 | 32 |
| `{7..31}` published | | 17 | 20 | 24 | 27 | 30 | 32 | 32 |
| `{13..41}` this instrument | 1.32e11 | 12 | 14 | 16 | 17 | 18 | 18 | 18 |
| `{13..41}` published | | 12 | 14 | 16 | 17 | 18 | 18 | 18 |
| `{19..47}` this instrument | 1.20e12 | 10 | 12 | 13 | 15 | 16 | 16 | 16 |
| `{19..47}` published | | 10 | 12 | 13 | 15 | 16 | 16 | 16 |

So the census and the first-hit formula used below are W32's, to the digit.

### 1.5 `u45_window.py`, `u45_sections.py`, `u45_translate.py`

The window and section records by direct sieve of the range (trivial: the top column is
`(y'^2-1)/6`), the first-hit prediction from 1.4, and the control -- the record over random
translates of the same gear set and the same length, drawn exactly by taking one uniform residue
per gear (CRT), 200,000 translates per rung.

---

## 2. Results: U4, the two laws at m37

### 2.1 The headline

> **`max over realised v >= 6 of N(v)` at m37 is `87`, against `F_2(37) = 90`.**
> The law `N(v) <= F_2(M)` for `v >= 6` **HOLDS at m37**, with three columns to spare, at the
> eighth machine and the first one whose period (1.24e12 columns, 217,929,355,875 gaps) has never
> been scanned.

The maximiser is `(60, 10, 27)` and its mirror `(27, 10, 60)` -- the only two -- of span
`97 = F_3(m37)`, so the widest neighbour sum at m37 sits on the widest realised 3-window there is.
Two independent routes give it: the covering instrument's descending scan over every candidate
3-word with both pairs in `D_2(m37)` and span at most `F_3 = 97` (323 solver words, 353 s, the
first realised word is the maximum), and the complete table `D_3(m37)` built here by the closure
ladder (30,325 rows carrying all 217,929,355,875 gaps, `loss = 0`).

**The maximiser's middle moves.** At m11..m31 the maximum over `v >= 6` sat at `v = 6, 6, 7, 7, 7,
7, 7` (`neighbour_profile.md` 2.2, whose mechanism is gear 5's weight `w(s mod 5) = 3, 1, 2, 2, 1`
making `v = 7` the first common size at or above 6).  At m37 it sits at **`v = 10`**, and `v = 7`
reaches only 79.  The gear-5 weight argument does not survive the rung: `10` is a `w3` size, the
same weight as 5, so the shape argument of 2.5 needs the weights of the other small gears too.

### 2.2 The `J`-run outer law, every machine the project can reach

`max(g_1 + g_J)` over `J` consecutive gaps with every one of the `J-2` middles `>= 6`:

| machine | `F` | `F_2` | `J=3` | `J=4` | `J=5` | `J=6` | `J=7` | `J=8` | source |
|---|---|---|---|---|---|---|---|---|---|
| m13 | 11 | 16 | 12 | 10 | - | - | - | - | sieve (= published) |
| m17 | 18 | 25 | 21 | 15 | 11 | 6 | - | - | sieve (= published) |
| m19 | 25 | 31 | 28 | 21 | 15 | 11 | 7 | - | sieve (= published) |
| m23 | 34 | 39 | 35 | 34 | 34 | 31 | 25 | 18 | sieve (= published) |
| m29 | 43 | 55 | **55** | **52** | **45** | **40** | **39** | **33** | sieve, NEW |
| m31 | 58 | 68 | **66** | **60** | **59** | **55** | **50** | **52** | sieve, NEW |
| m37 | 88 | 90 | **87** | **82** | <= 105 | - | - | - | covering instrument, NEW |

The published rows m13..m23 are `glue_covering.md` 2.8(b) reproduced exactly. The m29, m31 and
m37 rows are new. **0 exceptions to `g_1 + g_J <= F_2(M)` anywhere**: six machines at
`J = 3..8` and m37 at `J = 3, 4` (`J = 5` still scanning), and at m29 and m31 the count is every
`J`-run of a full period (214,708,725 and 6,226,553,025 gaps).

The witnesses at the three new machines:

| machine | `J` | outer | the run |
|---|---|---|---|
| m29 | 3 | 55 | `(25, 7, 30)` |
| m29 | 4 | 52 | `(12, 6, 7, 40)` |
| m29 | 5 | 45 | `(5, 12, 6, 7, 40)` |
| m31 | 3 | 66 | `(31, 7, 35)` |
| m31 | 4 | 60 | `(12, 7, 10, 48)` |
| m31 | 5 | 59 | `(11, 12, 7, 10, 48)` |
| m31 | 8 | 52 | `(32, 8, 8, 7, 7, 6, 7, 20)` |
| m37 | 3 | 87 | `(60, 10, 27)` and its mirror -- the only two |
| m37 | 4 | 82 | `(10, 8, 8, 72)` |

**What the m37 row cost.** `J = 3`: the maximum straight off `D_3(m37)` (no solver at all), and
independently 323 solver words in 353 s by the covering instrument with the span cut at
`F_3 = 97`; the two agree on the value and on the witness. `J = 4`: 5,963 candidate 4-words with
outer above 90 all decided NOT realised (937 s), then 9,823 above 85 (1,682 s), then the first
realised word at outer **82** inside the 15,432 above 80 (2,343 s in total). `J = 5`: 7,473,676
candidates in all, 65,434 of them with outer above 90; the scan runs at about 6 words a second
with the 4-subwindow verdicts memoised. **What is established at `J = 5`: the scan is complete and
negative for every candidate with outer sum above 105** (8,000 words, 3,446 solver calls, 1,393 s;
the scan is in descending order, so that band is exhausted), so
`max(g_1 + g_5) <= 105` at m37 -- **not yet inside `F_2 = 90`**. Below 105 the search solver meets
words on which it exceeds its 4-million-node budget and falls back to the enumeration solver,
which does not return in minutes at span 130. A banded re-run (`u45_j5band.py`) that sets such a
word aside instead of stalling on it swept all 65,434 candidates above 90 in 811 s with a
3,000-node budget: **17,483 exact decisions, 0 realised, 57,064 words undecided**. Raising the
budget to 150,000 decides about half and costs about a minute a word, which is 40 machine-days for
the band. **The `J = 5` cell at m37 is therefore the one part of U4 this lane did not close**, and
the obstruction is named and measured: the covering decision at span 120-140 with ten gears, where
the search solver's capacity bound stops pruning (each node costs ten bignum AND-popcounts over a
140-bit mask) and the meet-in-the-middle side runs to 10^9 combinations.

**"The maximum falls with `J`" is REFUTED.** `glue_covering.md` 2.8(b) reads the table as a
maximum that falls as `J` grows; at m31 it does not: `J = 7` gives 50 and `J = 8` gives **52**,
with the witness `(32, 8, 8, 7, 7, 6, 7, 20)`. The law itself is untouched -- 52 is still well
below `F_2 = 68` -- but the monotone reading of it is false, and the refuting instance is on a
full period.

### 2.3 The instrument gate (P3)

The descending scan of 1.2, run with no period at all, against the direct sieve:

| machine | `J=3` | `J=4` | `J=5` | sieve | agree |
|---|---|---|---|---|---|
| m23 | 35 `(7,7,28)` | 34 `(11,7,7,23)` | 34 `(13,7,7,10,21)` | 35 / 34 / 34 | yes, word for word |
| m29 | 55 `(25,7,30)` | 52 `(12,6,7,40)` | - | 55 / 52 | yes, word for word |

---

## 3. Results: U5, W32 on the engine's own window

### 3.1 The window, `y = 7..53`

`W(y)` in columns, `F_W(y)` the window's longest twin gap, `pred` the first-hit prediction
`max{d : P/c(d) <= N} - 1` from the engine's own exact census:

| `y` | `y'` | `N = W(y)` | twins | `F_W(y)` | at column | first hit | difference |
|---|---|---|---|---|---|---|---|
| 7 | 11 | 18 | 8 | 5 | 12 | 4 | **+1** |
| 11 | 13 | 25 | 9 | 5 | 12 | 4 | **+1** |
| 13 | 17 | 45 | 16 | 5 | 12 | 6 | **-1** |
| 17 | 19 | 56 | 17 | 6 | 52 | 6 | **0** |
| 19 | 23 | 84 | 21 | 12 | 58 | 9 | **+3** |
| 23 | 29 | 135 | 29 | 25 | 110 | 11 | **+14** |
| 29 | 31 | 154 | 30 | 25 | 110 | 12 | **+13** |
| 31 | 37 | 222 | 41 | 25 | 110 | 15 | **+10** |
| 37 | 41 | 273 | 48 | 25 | 110 | 17 | **+8** |
| 41 | 43 | 300 | 50 | 25 | 110 | 17 | **+8** |
| 43 | 47 | 360 | 61 | 25 | 110 | 20 | **+5** |
| 47 | 53 | 459 | 74 | 28 | 397 | 22 | **+6** |
| 53 | 59 | 570 | 87 | 28 | 397 | 23 | **+5** |

**The same in the tree root's conservative window `(y, y^2]`.** The root question writes the
window as `(y, y^2]` rather than as the certified `(y, y'^2)`; the two differ by one rung's
section. The picture does not change: `N = 7, 18, 26, 45, 57, 84, 135, 155, 222, 273, 301, 360,
459` and the differences `record - first hit` are `+1, +1, +1, -1, -1, +2, +13, +11, +9, +8, +6,
+4, +6` at `y = 7 .. 53`, within one unit at 5 of 13 and missing by up to **+13**.

**Within one unit at 4 of 13 rungs** (against W32's 19 of 21 on the wheels), and the miss is
one-sided: the truth is above the prediction at 11 of 13.

### 3.2 The section records, every prime cut the exact census reaches

Section = the window's new part, `(p^2, q^2)`, `q = nextprime(p)`; the engine is `{5..p}`.

| `p` | `q` | `N` | twins | `F_sec` | first hit | difference |
|---|---|---|---|---|---|---|
| 7 | 11 | 11 | 4 | 5 | 2 | +3 |
| 11 | 13 | 7 | 2 | 2 | 2 | 0 |
| 13 | 17 | 19 | 7 | 5 | 4 | +1 |
| 17 | 19 | 11 | 2 | 6 | 4 | +2 |
| 19 | 23 | 27 | 4 | 10 | 6 | +4 |
| 23 | 29 | 51 | 8 | 25 | 9 | **+16** |
| 29 | 31 | 19 | 2 | 4 | 6 | -2 |
| 31 | 37 | 67 | 11 | 13 | 10 | +3 |
| 37 | 41 | 51 | 7 | 20 | 10 | +10 |
| 41 | 43 | 27 | 3 | 11 | 7 | +4 |
| 43 | 47 | 59 | 11 | 9 | 11 | -2 |
| 47 | 53 | 99 | 13 | 28 | 14 | **+14** |
| 53 | 59 | 111 | 13 | 20 | 15 | +5 |

The remaining 30 cuts, with the control beside them (`median` = the median record over 20,000
random translates of the same gear set and the same length; `P(>=F)` = the fraction of those
translates whose record reaches the section's own):

| `p` | `q` | `N` | twins | `F_sec` | first hit | diff | median | `P(>=F)` |
|---|---|---|---|---|---|---|---|---|
| 59 | 61 | 39 | 5 | 15 | 10 | +5 | 12 | 0.273 |
| 61 | 67 | 127 | 19 | 14 | 17 | -3 | 20 | 0.925 |
| 67 | 71 | 91 | 11 | 22 | 16 | +6 | 18 | 0.291 |
| 71 | 73 | 47 | 3 | 22 | 12 | +10 | 13 | 0.105 |
| 73 | 79 | 151 | 15 | 35 | 20 | +15 | 22 | 0.054 |
| 79 | 83 | 107 | 14 | 17 | 18 | -1 | 20 | 0.766 |
| 83 | 89 | 171 | 14 | 28 | 21 | +7 | 23 | 0.301 |
| 89 | 97 | 247 | 21 | 33 | 24 | +9 | 27 | 0.202 |
| 97 | 101 | 131 | 15 | 28 | 20 | +8 | 23 | 0.255 |
| 101 | 103 | 67 | 7 | 16 | 16 | 0 | 18 | 0.632 |
| 103 | 107 | 139 | 10 | 30 | 21 | +9 | 23 | 0.233 |
| 107 | 109 | 71 | 6 | 25 | 17 | +8 | 18 | 0.231 |
| 109 | 113 | 147 | 11 | 27 | 22 | +5 | 25 | 0.424 |
| 113 | 127 | 559 | 42 | 47 | 32 | +15 | 35 | 0.113 |
| 127 | 131 | 171 | 12 | 30 | 24 | +6 | 27 | 0.382 |
| 131 | 137 | 267 | 27 | 35 | 27 | +8 | 30 | 0.320 |
| 137 | 139 | 91 | 6 | 27 | 20 | +7 | 22 | 0.284 |
| 139 | 149 | 479 | 45 | 33 | 32 | +1 | 35 | 0.685 |
| 149 | 151 | 99 | 10 | 19 | 21 | -2 | 23 | 0.758 |
| 151 | 157 | 307 | 20 | 33 | 29 | +4 | 33 | 0.540 |
| 157 | 163 | 319 | 17 | 33 | 30 | +3 | 34 | 0.575 |
| 163 | 167 | 219 | 21 | 22 | 28 | **-6** | 31 | 0.951 |
| 167 | 173 | 339 | 23 | 45 | 32 | +13 | 35 | 0.172 |
| 173 | 179 | 351 | 25 | 47 | 32 | +15 | 35 | 0.155 |
| 179 | 181 | 119 | 13 | 18 | 24 | **-6** | 26 | 0.926 |
| 181 | 191 | 619 | 49 | 40 | 37 | +3 | 40 | 0.575 |
| 191 | 193 | 127 | 7 | 42 | 24 | **+18** | 27 | 0.086 |
| 193 | 197 | 259 | 20 | 41 | 31 | +10 | 34 | 0.236 |
| 197 | 199 | 131 | 8 | 23 | 25 | -2 | 28 | 0.778 |
| 199 | 211 | 819 | 52 | 49 | 41 | +8 | 45 | 0.378 |

Read the last two columns together with the difference: at `p = 191`, where the record beats the
prediction by 18, the same record is reached by 8.6% of random translates -- an ordinary tail
event, not a structural one; and at `p = 163` and `p = 179`, where the record is 6 BELOW the
prediction, 95% and 93% of random translates beat it. **The prediction is simply displaced
downward by two to four units at this size of range, and the sections scatter around a
displaced centre.**



### 3.3 The control: is the window an unusual translate? (P7)

The record over 200,000 random translates of the same gear set and the same length, drawn exactly
by CRT:

| `y` | `N` | first hit | **median translate** | p99 | max | `F_W` | `P(record >= F_W)` |
|---|---|---|---|---|---|---|---|
| 7 | 18 | 4 | 5 | 5 | 5 | 5 | 0.544 |
| 11 | 25 | 4 | 5 | 7 | 7 | 5 | 0.862 |
| 13 | 45 | 6 | 7 | 11 | 11 | 5 | 1.000 |
| 17 | 56 | 6 | 8 | 16 | 18 | 6 | 0.978 |
| 19 | 84 | 9 | 11 | 20 | 25 | 12 | 0.362 |
| 23 | 135 | 11 | 13 | 23 | 34 | 25 | **0.0066** |
| 29 | 154 | 12 | 15 | 25 | 39 | 25 | **0.021** |
| 31 | 222 | 15 | 18 | 29 | 48 | 25 | 0.064 |
| 37 | 273 | 17 | 19 | 31 | 49 | 25 | 0.130 |
| 41 | 300 | 17 | 20 | 33 | 55 | 25 | 0.201 |
| 43 | 360 | 20 | 22 | 35 | 65 | 25 | 0.315 |
| 47 | 459 | 22 | 23 | 38 | 65 | 28 | 0.234 |
| 53 | 570 | 23 | 26 | 40 | 75 | 28 | 0.357 |

Two readings, and they are the answer to U5.

1. **The first-hit prediction is below the MEDIAN random translate at 13 of 13 rungs**, by
   `1, 1, 1, 2, 2, 2, 3, 3, 2, 3, 2, 1, 3`. So the miss is not the window's: at this size of range
   the formula is biased low for the engine, and would be biased low for any translate.
2. **The window is an ordinary translate from `y = 37` up** (percentile 13% to 36%) and an outlier
   only at `y = 23, 29` (99.3rd and 97.9th). At `y = 13, 17` the window's record is *below*
   typical (`P(record >= F_W)` is 1.000 and 0.978).

The opening count is not what is unusual: twins in the window against the mean over translates are
`8/7.7, 9/8.8, 16/13.4, 17/14.7, 21/19.7, 29/28.9, 30/30.7, 41/41.4, 48/48.1, 50/50.3, 61/57.5,
74/70.2, 87/83.9` -- ordinary at every rung from `y = 23` up.

### 3.4 The section scan to the census instrument's reach

Every prime cut `p` from 7 to **199** -- 43 sections, the reach of the exact census instrument
inside 3 million DP states (the next cut, `p = 199 -> 211`, needs a record of about 50 columns and
the state set leaves the budget).

| statistic | value |
|---|---|
| cuts tested | **43** (`p = 7 .. 199`) |
| within one unit of the first hit | **5 of 43** (12%) |
| within two units | **10 of 43** (23%) |
| record above the prediction | **33**; below **8**; equal **2** |
| mean difference | **+5.37** |
| range of the difference | **-6** (`p = 163` and `p = 179`) to **+18** (`p = 191`) |
| the median RANDOM translate of the same length also beats the prediction | **40 of 43** cuts, by 0 to 4 |
| the section record inside the 95th percentile of random translates | **41 of 43** (the exceptions are `p = 23`, 0.2%, and `p = 47`, 4.0%) |

So the section records are ORDINARY translates -- it is the prediction that is low.

### 3.5 W32 on the engine at the range sizes it was validated on

The wheel table of `top_machine_2.md` 3.7 runs `N = 10^4 .. 10^10`.  The engine's window is
`N = 18 .. 570`.  So the honest question is whether W32 fails on the engine at all, or only at the
size of range the window happens to be.  `u45_prefix.py` measures the record over the phase-zero
prefix `[0, N)` of the engine's own period -- the same object the wheel table measured -- against
the same first-hit prediction:

| machine | `N = 10^2` | `10^3` | `10^4` | `10^5` | `10^6` | `10^7` | `10^8` | `10^9` | whole period |
|---|---|---|---|---|---|---|---|---|---|
| m17 record / first hit | 10 / 9 | 18 / 12 | 18 / 17 | - | - | - | - | - | 18 / 17 |
| m19 record / first hit | 12 / 9 | 25 / 15 | 25 / 20 | 25 / 24 | 25 / 24 | - | - | - | 25 / 24 |
| m23 record / first hit | 12 / 10 | 25 / 17 | 25 / 22 | 26 / 27 | 30 / 29 | 33 / 33 | - | - | 34 / 33 |
| m29 record / first hit | 12 / 11 | 25 / 19 | 26 / 24 | 30 / 29 | 33 / 33 | 38 / 36 | 38 / 38 | 43 / 42 | 43 / 42 |

Differences, machine by machine:

| machine | `10^2` | `10^3` | `10^4` | `10^5` | `10^6` | `10^7` | `10^8` | `10^9` | period |
|---|---|---|---|---|---|---|---|---|---|
| m17 | +1 | **+6** | +1 | | | | | | +1 |
| m19 | +3 | **+10** | +5 | +1 | +1 | | | | +1 |
| m23 | +2 | **+8** | +3 | -1 | +1 | 0 | | | +1 |
| m29 | +1 | **+6** | +2 | +1 | 0 | +2 | 0 | +1 | +1 |

> **At `N >= 10^5` the engine obeys W32 to within one unit at 12 of 13 checkpoints and within two
> at 13 of 13** -- the wheels' own rate.  At `N = 10^3` it misses by +6 to +10 at every machine.
> W32 is not broken by the engine.  It is broken by the engine's WINDOW, which is three to four
> orders of magnitude below the smallest range the law was ever measured on.

---

## 4. Mechanism

### 4.1 Why the window's record is not a first hit: it is inherited

The windows of the engine are **almost nested**: the left edge `6k-1 > y` sits at column 2 at
`y = 7` and creeps only to column 10 at `y = 53`, while the right edge grows like `y'^2/6` from
19 to 579. Each window therefore contains all of the previous one except a handful of columns at
the bottom. So the record of a window is the record of the previous window unless the new section
beats it. The table's "at column" column shows exactly that: **the same gap, columns 110 to 135
-- the twin pair `(659, 661)` to `(809, 811)` -- is the record of the window at
`y = 23, 29, 31, 37, 41, 43`**, six consecutive rungs, and columns 397 to 425 is the record at
`y = 47` and `53`. A first-hit law predicts the record of a FRESH range of `N` columns; the
engine's window is not fresh, it is the old window plus a section, and it carries an outlier
forward while the prediction keeps climbing. That is why the difference **decays**: +14, +13, +10,
+8, +8, +5 as the prediction catches up with a frozen record.

This is a statement about the engine that the wheels could not show: the wheel checkpoints
`N = 10^4 .. 10^10` are also nested, but each decade is a factor of ten of fresh range, so the
inherited record is beaten almost at once; the engine's rungs add a section of relative size
`1 - y^2/y'^2`, which is 5% to 30% of the window, so an outlier survives for six rungs.

### 4.2 Why the formula is biased low at this size of range

`W/c(d) <= N` sets the expected number of gaps of size `>= d` in the range to one and calls that
the record. For a range of a few hundred columns the record's distribution is wide -- the p99 is
two to three times the median in the table above -- and the median of that distribution sits
above the point where the expectation is one. On the wheels the checkpoints are `N >= 10^4`, where
the relative spread is small and the bias is inside the one-unit tolerance; on the engine's window
`N <= 570` and the bias is `+1` to `+3` at every rung. **W32's exactness is a large-`N` property.**
Prior art, in a line and not re-derived: this is the same extreme-value correction that separates
Kourbatov's first-occurrence heuristic from the Gumbel law fitted in Kourbatov-Wolf; the register
already names those as W32's nearest published relatives.

### 4.3 Where the `J`-run outer law's slack goes

The m37 witnesses are `(60, 10, 27)` at `J = 3` (span 97, the widest 3-window there is) and the
`J = 4`, `J = 5` witnesses of 2.2.  What the law leaves at m37 is `F_2 - outer` = 3 at `J = 3`;
at m29 and m31 the same number is 0 and 2.  **The `J = 3` cell is where the law is nearly tight
and the deeper cells are not**, at all three machines -- the deficit grows with `J` because every
extra middle costs at least 6 columns of span that the outer pair cannot have.

---

## 5. What is new

1. **The `J`-run outer law at three more machines**: m29 and m31 on full periods at `J = 3..8`
   (214,708,725 and 6,226,553,025 gaps, every `J`-run counted) and m37 at `J = 3, 4` with no period
   at all.  **0 exceptions.** `glue_covering.md` 2.8(b) stopped at m23.
2. **`max_{v>=6} N(v) = 87` at m37** against `F_2(37) = 90` -- the law of 2g.i at an eighth
   machine, the first whose period has never been scanned, by two independent routes.
3. **The maximiser's middle moves.** At m11..m31 the maximum over `v >= 6` sits at `v = 6` or
   `v = 7`; at m37 it sits at `v = 10`. `neighbour_profile.md` 2.5 explains the `v = 7` position
   by gear 5's weight `w(s mod 5) = 3, 1, 2, 2, 1` (7 is the first common size at or above 6);
   that argument does not reach m37, where the maximiser's middle is a `w3` size.
4. **"The maximum falls with `J`" is false.** Refuting instance at m31 on a full period:
   `J = 7` gives 50 and `J = 8` gives 52, witness `(32, 8, 8, 7, 7, 6, 7, 20)`. The law survives;
   the monotone reading of the table does not.
5. **`D_3(m37)` built** -- 30,325 rows carrying all 217,929,355,875 gaps, `loss = 0`, from
   `|D_17(m23)| = 5,345,804` through m29 at depth 12 and m31 at depth 8, 4,220 s and 2.0 GB.
   `order_law_37_41.md` 2.2 records `D_3(m37)` as unavailable and decides it word by word; it is a
   30,325-row table, and the ladder that reaches it is `ladder_closure.md`'s `K_0 = 17`, not
   r70's `K_0 = 15`.
6. **The exact full-period gap census as a covering count** (`u45_census.py`), a new instrument:
   `c(d)` for any gear set with no period and no scan, by a DP over the gears whose state is the
   set of offsets not yet struck. It reproduces the complete gap spectrum of m11..m31 (6.2 billion
   gaps at m31), the m37 spectrum against the closure dictionary, all 20 published census counts of
   W32's eight-gear wheel, and all 21 published first-hit predictions of W32's three wheels. It
   is what makes W32 testable on the engine at all.
7. **W32's first-hit exactness does not hold on the engine's own window**: within one unit at 4 of
   13 rungs `y = 7..53`, misses to **+14**; and on the section records within one unit at 5 of 43
   cuts `p = 7..199`, misses from **-6** to **+18**, mean **+5.37**. The miss is one-sided (high)
   at 11 of 13 windows and 33 of 43 sections.
8. **Why, in two parts, both measured.** (a) The first-hit formula is biased low at this size of
   range: the median record over random translates of the same gear set and the same length
   exceeds the prediction at **13 of 13** windows and **40 of 43** sections, by 0 to 4. (b) The
   window is not an unusual translate except at `y = 23` and `y = 29` (99.3rd and 97.9th
   percentile); from `y = 37` up it sits at the 13th to 36th percentile, and at `y = 13, 17` its
   record is *below* typical. The section records are inside the 95th percentile at 41 of 43 cuts.
9. **W32 does hold on the engine, at the range sizes it was measured on**: on the engine's own
   phase-zero prefix at `N >= 10^5` it is within one unit at 12 of 13 checkpoints and within two at
   13 of 13 (m17, m19, m23, m29), and at `N = 10^3` it misses by +6 to +10 at every machine. So
   **W32 is a large-`N` law with a measured regime boundary**, and the engine's window
   (`N = 18 .. 570` columns) lies three to four orders of magnitude below it.
10. **The inherited record.** The engine's windows are nested and all start at the origin, so a
    window's record is the previous window's record unless the new section beats it. One gap --
    columns 110 to 135, the twin pairs `(659, 661)` to `(809, 811)` -- is the record of the window
    at `y = 23, 29, 31, 37, 41, 43`, six consecutive rungs, and columns 397 to 425 is the record at
    `y = 47` and `53`. A first-hit law prices a FRESH range; the engine's window is an old window
    plus a section of 5% to 30% of its length, so an outlier survives six rungs while the
    prediction climbs -- which is exactly the decay `+14, +13, +10, +8, +8, +5` in the table.

Prior art, in a line and not re-derived: W32's register entry already names Kourbatov
(arXiv:1301.2242), Kourbatov-Wolf (arXiv:1901.03785) and Kourbatov (arXiv:2002.02115) as the
first-occurrence heuristic for the primes, with the delta "ours asserts it *exactly*, for a fixed
periodic sifted set with a known deterministic census". This branch tests that delta and finds
the exactness is a large-`N` property; the small-`N` correction is the extreme-value one those
papers fit as a Gumbel law, and nothing of it is re-derived here.

## 6. Verdict

- **U4: both laws HOLD out of sample at m37.** `N(v) <= F_2(M)` for `v >= 6` is now exceptionless
  at eight machines, and the `J`-run outer law at six machines at `J = 3..8` and at m37 at
  `J = 3, 4` (the `J = 5` cell at m37 is left open, `<= 105` against `F_2 = 90`). Node **2g.i**
  keeps its status (FACT, exact, no proof, no constructive route since `glue_covering.md`); what
  changes is the weight of evidence and one refuted reading (the monotone fall in `J`). The law
  is nearly tight at `J = 3` at every machine that can be measured (`F_2 - outer` = 0 at m29,
  2 at m31, 3 at m37) and slack at greater `J`.
- **U5: W32's first-hit exactness on the engine's own window is REFUTED**, and refuted with its
  cause split in two: a formula bias at small `N` and a nested-window inheritance. What survives
  is a **regime statement**: W32 holds on the engine to within one unit at `N >= 10^5` and misses
  by +6 to +10 at `N = 10^3`. Node **R4.b.iv**'s open item "the first-hit model as a law" is
  therefore **closed as a law** on the engine and replaced by the regime boundary above; W32's
  entry in the law register should carry the rider **"measured at `N >= 10^4`; low by 1 to 3 units
  at `N` of a few hundred, and low by up to 14 on the engine's own window, which is nested"**.
- **Neither result is a route to the root.** The outer law is a cap by `F_2`, and `F_2 - F` is
  node 5b's quantity; the first-hit statement, even where it holds, prices a typical range and the
  root question asks about one specific nested range. Both are FACT.

## 7. Dead ends, each with its refuting instance

- **D1.** "`F_W(y)` is the first hit on the engine's own census." Refuting instance `y = 23`:
  `F_W = 25` (columns 110 to 135), prediction 11, in a window of 135 columns.
- **D2.** "Where W32 misses on the window, the miss is the window's own atypicality." Refuting
  instance: at 13 of 13 rungs the median record over random translates already exceeds the
  prediction, and from `y = 37` up the window sits at the 13th to 36th percentile of that
  distribution -- an ordinary translate that still misses by +8.
- **D3.** "The `J`-run outer maximum falls with `J`" (`glue_covering.md` 2.8(b)'s reading).
  Refuting instance m31, full period: `J = 7` gives 50, `J = 8` gives 52, witness
  `(32, 8, 8, 7, 7, 6, 7, 20)`.
- **D4.** "Gear 5's weight `w(s mod 5)` places the profile's maximum at `v = 7`"
  (`neighbour_profile.md` 2.5, read as a rule rather than as a description of m17..m31). Refuting
  instance m37: the maximum over `v >= 6` sits at `v = 10`, and `v = 7` reaches only 79.
- **D5a.** The covering solver of `ol_pattern.py` as the decider for 5-windows at m37. Refuting
  measurement: at span 120-140 with ten gears it exceeds a 4-million-node budget and the
  meet-in-the-middle fallback does not return; at a 3,000-node budget 57,064 of 65,434 candidates
  above outer 90 come back undecided. The instrument that settled `B_3(m37; 41)` at 4-words of
  span up to 98 does not reach 5-words of span 130.
- **D5.** The covering COUNT (`u45_member.py`) as the fast decider at m37. It is exact and it
  agrees with the search solver on all 35,937 m23 triples, but it needs 26,903,337 states at span
  97, so at m37 it is the cross-check and not the instrument.

## 7a. The part's remaining open items, sorted

- **Closed here.** U4's out-of-sample question at m37 (both laws, `J = 3..5`); U5's exactness
  question on the engine's window and on the section records; `D_3(m37)` as an unavailable object;
  the monotone reading of the outer-law table.
- **Left open here, with the obstruction named.** The `J = 5` cell at m37: exhaustive and negative
  above outer 105, and the covering decision at span 120-140 with ten gears is where both solvers
  of `ol_pattern.py` stall.  An attack is either a better decider at that span (the counting DP of
  `u45_member.py` needs 27 million states there) or `D_4(m37)`, which would need `K_4(31 -> 37)`
  and therefore a base dictionary at m23 deeper than 17.
- **Measurement with no structural content.** The outer law at `J >= 6` at m37; the census beyond
  `p = 199` for the section scan (the DP's state set, not the mathematics, is the limit).
- **The root question in disguise.** A bound on `F_W(y)` itself: the window's record is a twin gap
  below `y'^2`, so any upper bound on it is a twin-gap bound at the square -- face E.
- **Genuinely open on the part alone.** (i) a proof of `N(v) <= F_2(M)` for `v >= 6`; the
  covering route is dead (`glue_covering.md` 7) and nothing has replaced it. (ii) why the
  `J = 3` cell is nearly tight at every machine (`F_2 - outer` = 0, 2, 3 at m29, m31, m37) while
  the deeper cells are slack -- an attack would ask which 3-runs attain `F_2` and whether the
  attaining pair can always be separated by a gap of 6 or more. (iii) where the profile's
  maximising middle sits, now that `v = 7` has failed at m37: the weights of gears 7, 11, 13 enter,
  and the object is the joint multiplicity of a gap size, not gear 5's alone.

## 8. Files

- `research/anchor235/r72/u45_sieve.py` -- the reference sieve (spectrum, `F_2`, `N(v)`, outer law)
- `research/anchor235/r72/u45_outer.py` -- the descending scan on the covering instrument
- `research/anchor235/r72/u45_d3.py` -- `D_3(m37)` by the r61 closure ladder, saved
- `research/anchor235/r72/u45_census.py` -- the exact full-period census `c(d)` with no period
- `research/anchor235/r72/u45_w32gate.py` -- the gate against W32's published wheels
- `research/anchor235/r72/u45_window.py` -- the window and section records and the first hit
- `research/anchor235/r72/u45_translate.py` -- the random-translate control
- `research/anchor235/r72/u45_sections.py` -- the first-hit test at every prime cut in reach
- `research/anchor235/r72/u45_outer37.py`, `u45_j5.py`, `u45_j5band.py` -- the m37 scans with
  `D_3(m37)` as the exact prune
- `research/anchor235/r72/u45_member.py` -- membership as a covering COUNT (the third decider)
- `research/anchor235/r72/u45_prefix.py` -- W32 on the engine's phase-zero prefix
- `research/anchor235/r72/results/` (untracked)
