# Node 4.i.b.ii.a - CAP THE PADDED WORD

Parent: node **4.i.b.ii, the monotone functional** (`research/proof/monotone_functional.md`),
whose order law `k* = L(M) + 1` gave `L` a job: a cap `L <= c` turns the budget inequality into a
bounded-order statement about the table `D_{c+1}`. The engine's ledger (`objects_ledger.md`
O-M1) has exactly one structural item left, "is `L(M)` bounded?", and `L = max(L_bare, L_pad)`
with `L_bare <= PSORD(q' mod 210) <= 5` proved (docs/proofs/12). This branch is on the open half,
`L_pad`.

Scripts in `research/anchor235/r67/` (prefix `pc_`); outputs in `research/anchor235/r67/results/`
(untracked). Every number the document relies on is written into the document. Laws found here
are numbered `E1, E2, ...` (engine laws; the register's W-numbers from W103 belong to the
manifold).

---

## 0. Pre-registered (written before any computation of this branch)

### 0.1 The object, exactly

`M = {5..y}` the anchored machine on columns, period `P`, `N` openings, cyclic gap sequence;
`q'` the next gear, `u = 6^{-1} mod q'`, `d = 2u mod q'`, `a = min(d, q' - d)` (the letter floor
`a_L`), `b = q' - a`, so `{a, b} = {d, q' - d}` as residues and `a + b = q'`.

A gap value `v` is a **legal letter** iff `v mod q' in {0, d, -d}`; class `PAD` (`0`), `UP`
(`+d`), `DOWN` (`-d`). A word of letters is **legal** iff no two consecutive nonzero letters are
equal (pads transparent; docs/proofs/05 (F)). A legal word is **realised** in `M` if it occurs as
consecutive gaps of `M`. `L(M)` is the longest realised legal word.

- The **bare letters** are `a` and `b`: the only legal values below `q'`.
- A **padded letter** is a legal letter that is not bare, i.e. a legal value `>= q'`:
  `k q'` (PAD), `a + k q'` and `b + k q'` (`k >= 1`). This is the brief's definition and
  alignment-rules.md 3.8's ("at least one non-bare letter"). The instrument of r66
  (`lc_core.word_stats`) counts instead words with at least one PAD-class letter; both are
  measured and written `L_pad` (brief) and `L_pad0` (class 0 only). On the record to m53 the
  two agree, because every padded letter inside a recorded word of length `>= 2` is `q'` itself.
- `L_bare(M)`: longest realised legal word over `{a, b}`; `L_pad(M)`: longest realised legal word
  containing a padded letter; `L = max(L_bare, L_pad)`.
- The **PAD alphabet** `A_pad(M)` is the set of padded letters realised as gap sizes of `M`
  (with multiplicities). The **legal alphabet** `A_leg(M) = {a, b} intersect spectrum, union
  A_pad(M)`.
- The **small alphabet** is `{a, b, q'}`: the bare letters and the single pad `q'`. A **skip
  letter** is any other padded letter (`2q'`, `a + q'`, `b + q'`, ...). Why "skip": read the
  word's openings from `x_0` in units of `q'`. Every opening of the word is on one of the two
  teeth of one phase, so its offset from `x_0` is `m q'` (start tooth, class 0) or `s + m q'`
  (other tooth, class 1) with `s in {a, b}` fixed by the start tooth. Over the small alphabet
  the multiplier `m` of consecutive openings rises by 0 or 1 (letter `s`: class 0 -> 1, same
  `m`; letter `q' - s`: class 1 -> 0, `m + 1`; pad `q'`: same class, `m + 1`); a skip letter
  jumps `m` by `>= 2` in one step, or by 1 with a class change in the wrong direction, i.e. it
  passes over a tooth column that `M` has blocked.
- The **skeleton** of a realised legal word is the pair `(S_0, S_1)` of multiplier sets,
  `|S_0| + |S_1| = L + 1`, `S_0 ni 0`.
- **Level-`k` admissible** legal word: every `k` consecutive letters form a realised legal
  `k`-word (for length `<= k`: realised). `L^{(k)}(M)` = the longest level-`k` admissible legal
  word (infinite if the `k`-word de Bruijn graph on legal words has a cycle). `L^{(k)} = L` for
  `k >= L + 1`. `k_L(M) := min {k : L^{(k)} = L}`, the order at which `L` is decided.
- **`CORRCAP_3(c)`** for `c` coprime to 210: the longest legal word over the small alphabet with
  values `a_c = aOfClass(c)`, `b_c = c - a_c`, `q' = c` (mod 35 is all that matters) whose
  prefix-sum walk from some `r in E_35` stays in `E_35` (docs/proofs/14 (a)), i.e. fits gears
  5 and 7; `infinity` if the walk can cycle. Same object as PSORD (docs/proofs/12) with the
  letter `q'` added, and the same object as the record's `CORRCAP(q', F)` (uniform-order-bound,
  alignment-rules 6.2) with the alphabet cut to `{a, b, q'}` instead of every legal value
  `<= F`.

**Facts cited, not re-derived.** `J_max = L + 2` and the same-tooth lemma (docs/proofs/10);
`L_bare <= PSORD <= 5` (docs/proofs/12); `L <= 2 floor((F(M+q') - 2)/q') + 1`, letter-aware
`m <= 2T + 1 - p` (docs/proofs/11); the corridor `E_35`, the AP lemma (no four openings in AP
with difference coprime to 5), the padding laws (docs/proofs/14); the record's `CORRCAP` row
`4, 2, 3, 5, 25, 25, 11, 5` at 19->23 .. 47->53 and infinite from 53->59; the collision laws
(docs/proofs/21: twin gears collide at `(g+4)/3`); the closure step and dictionaries of r61/r66;
the corpus rows `L = 1,1,1,2,1,3,3,2,2,2,4,3` and `L_pad = 0,0,0,1,1,1,2,2,2,2,3,3` at m11..m53,
with the recorded padded words `(12,37)` at m31, `(41,14)` at m37, `(18,35,53)`, `(18,53,35)`,
`(35,18,53)` at m47 and `(35,71,35)` undecided at m47.

### 0.2 The theory

**T. `L_pad` has two halves of different nature, and the record has only ever seen the tame one.**
Padded words over the small alphabet `{a, b, q'}` are a residue object: their offsets in units of
`q'` never skip, so gears 5 and 7 see the whole word through `q' mod 210` exactly as they see a
bare word, and `CORRCAP_3(q' mod 210)` caps them uniformly. Every padded word on record (to m53)
is small-alphabet, so the measured growth `0 -> 3` of `L_pad` is growth *inside* a uniformly
capped family, driven by the class `q' mod 210` and not by the machine's size. The half that is
genuinely open is the **skip half**: words with a letter `2q'`, `a + q'`, `b + q'`, ... Skip
letters exist as gaps from m31 (`49 = 12 + 37`), but a skip letter in a word of length `>= 2`
needs `F_2(M) >= 2q'`, first true at m37 (`90 >= 82`). No residue argument caps the skip half
(the record's `CORRCAP` is infinite from 53->59 for exactly this reason), and no counting
argument does either; its growth is the shadow of the record in gear units, `F(M)/q'`.

### 0.3 Predictions, each with the number that would refute it

- **P1 (instrument).** Full periods m5..m23 and the r66 ladder m29 (depth 10), m31 (depth 6),
  m37 (depth 2) reproduce `L = 1, 0, 1, 1, 1, 2, 1, 3, 3, 2` and `L_pad0 = 0,0,0,0,0,1,1,1,2,2`
  at m5..m37, `F` at every rung, and the r66 densities `W_1/N`, `Z_1/N`. REFUTED by any mismatch.
- **P2 (the PAD alphabet, exact).** `A_pad` is empty to m17; `{23}` at m19; `{29}` at m23;
  `{31}` at m29 (`41 = 10 + 31` is a spectral hole); `{37, 49}` at m31; `{41, 55, 68}` at m37
  (`82 = 2 * 41` is a hole). The legal alphabet is `{8, 15, 23}`, `{10, 19, 29}`, `{10, 21, 31}`,
  `{12, 25, 37, 49}`, `{14, 27, 41, 55, 68}` at m19..m37. `|A_pad|` grows like
  `3 F/q' - 2` (the record's alphabet-size reading, `~3F/q'`). REFUTED by a padded value
  realised that is not on the list, or one on the list absent.
- **P3 (small alphabet, measured).** Every realised legal word of length `>= 2` at m5..m37 uses
  only `{a, b, q'}`: 0 exceptions. The single skip letters `49` (m31), `55`, `68` (m37) occur as
  gaps but never adjacent to a legal letter in a legal 2-word. At m37 the four size-feasible
  skip 2-words `(27, 55)`, `(55, 27)`, `(14, 68)`, `(68, 14)` (span 82 `<= F_2(37) = 90`) are
  all unrealised in `D_2(m37)`. REFUTED by one realised skip 2-word; that word is then the
  first of the skip half and the finding.
- **P4 (`CORRCAP_3`, the small-alphabet cap; candidate law E1).** `CORRCAP_3(c)` is finite for
  all 48 classes `c mod 210`, `PSORD(c) <= CORRCAP_3(c) <= 7` everywhere, and its maximum over
  classes is at most 7. Hence **E1: every realised legal word over `{a, b, q'}` in any machine
  containing 5 and 7 has length `<= CORRCAP_3(q' mod 210) <= max_c CORRCAP_3`**, by the proof of
  docs/proofs/12 with the letter `q' = c (mod 210)` added to the alternation (the pad keeps the
  tooth, so legality is still "nonzero classes alternate", and the offsets mod 35 depend on `c`
  alone). REFUTED by a class with a cycle (then gears 5 and 7 do not cap the small-alphabet
  family; the test moves to gears 5, 7, 11 mod 2310). Test against the record: `L(M) <=
  CORRCAP_3(q' mod 210)` at every rung m5..m53 where all realised words are small-alphabet
  (predicted: all of them), and the cap is tight (`L = CORRCAP_3`) at at least two rungs.
- **P5 (the counting cap, pre-registered as VACUOUS).** For the longest realised legal word at
  each rung, with span `S`: the interior `S - L` columns must be struck, and the machine's
  capacity `sum_{g in M} max_g(S + 1)` (docs/proofs/20, Lemma 2) exceeds `S - L` by a factor
  `>= 2` at every rung from m11 on; the junction version (the `2(L+1)` neighbour columns
  `x_i +- 1` against the capacity) is vacuous by a factor `>= 5`. Mechanism: gear 5 alone
  strikes two of every five columns. No count of strikes caps a word; only *which* gear strikes
  which junction neighbour can. REFUTED by a rung where capacity is below need (impossible, the
  word is realised) or by a ratio below 2.
- **P6 (the skeleton and the skip law; candidate law E2).** For every realised legal word,
  `|S_0| + |S_1| = L + 1`, no class contains four consecutive multipliers (the AP lemma with
  difference `q'`), and consecutive openings differ by 0 or 1 in `m` iff the word is
  small-alphabet. So a word of length `L` over the small alphabet has `m_max <= L` and each
  class's multiplier set is a union of runs of length `<= 3`. Prediction: every word on record
  to m53 has both classes' runs `<= 3` (0 exceptions), and the longest words at m29
  (`(10,21,10)`), m47 (`(18,35,18,35)`, `(18,35,53)`) have a class with a full run of 3 exactly
  where `L >= 3`. REFUTED by a run of 4 (impossible by the lemma) or a small-alphabet word with
  a multiplier jump `>= 2`.
- **P7 (the order at which `L` is decided).** `k_L(M) = L(M) + 1` at every rung with `L >= 2`
  (m19, m29, m31, m37 within depth): a legal `(L+1)`-word exists all of whose `L`-windows are
  realised, so no table of order `<= L` decides `L` - the same table `D_{L+1}` that the order law
  needs is the one that decides the depth. At `L = 1` rungs `k_L = 2` trivially (both bare
  letters realised). REFUTED by a rung with `k_L <= L`.
- **P8 (letter count against `L_pad`).** `L_pad(M) <= |A_pad(M)|` at every rung (trivially true
  while padded words carry one pad), and NOT a cap: `|A_pad|` is 1, 1, 1, 2, 3 at m19..m37
  against `L_pad = 1, 1, 1, 2, 2`, tight at four of five, and `|A_pad| ~ 3F/q' - 2` grows. The
  sharpest true size statement remains docs/proofs/11's letter-aware `L_pad <= 2T` with
  `T = floor((F(M+q') - 2)/q')`: `2, 2, 2, 4, 4, 4, 4, 4, 4` at m19..m53 against
  `1, 1, 1, 2, 2, 2, 2, 3, 3`, never tight.
- **P9 (the shadow).** If E1 holds, the growth of `L_pad` on record is not the shadow of the
  machine at all: it is `CORRCAP_3(q' mod 210)` moving with the class (predicted
  `CORRCAP_3(31) = 3`, `CORRCAP_3(37) = 5`, `CORRCAP_3(53) >= 3`). The uncapped object is the
  skip half, whose alphabet is `{v <= F(M) : v = 0, +-d mod q', v >= q' + a}` of size
  `~ 3(F - q')/q'`, so it is the shadow of **`F(M)/q'`, the record in gear units** - not of
  the merge forest's depth (`J_max` is small and non-monotone), not of the corridor (which is
  what caps the tame half), not of the gear-5 lock (proved for every gap already). One concrete
  test per candidate is run in section 6.

**Stop rules.** Any sub-question reducing to the budget, the pair statement or the chain
statement is stopped in one line. `PSORD`, the AP lemma, the corridor and the spectrum bound are
cited, never re-derived. The manifold, the valves and the exhaust are not touched.

### 0.4 Scorecard

| # | prediction | verdict | evidence |
|---|---|---|---|
| P1 | instrument reproduces `L`, `L_pad0`, `F`, densities | **CONFIRMED** - ten of ten `L`, ten of ten `L_pad0`, `F` at every rung, `W_k` rows digit for digit, scans and dictionaries agree to the unit | 1, 2.2 |
| P2 | the PAD alphabet at every rung | **CONFIRMED exactly** - `{23}, {29}, {31}, {37, 49}, {41, 55, 68}` at m19..m37; holes 41, 82 as predicted | 2.1 |
| P3 | every word of length `>= 2` is small-alphabet to m37; the four m37 skip 2-words unrealised | **CONFIRMED** - 0 exceptions in every period and dictionary; `(27,55)`, `(55,27)`, `(14,68)`, `(68,14)` absent from the complete `D_2(m37)`; the skip half enters the record at m53 | 5.1 |
| P4 | `CORRCAP_3` finite, `<= 7`, caps every recorded `L`, tight twice | **CONFIRMED with one correction** - finite at 48 of 48, maximum **8** (classes 103, 107), not 7; `L_small <= CORRCAP_3` at 14 of 14 rungs; tight at m29 and m53; gears 11, 13 lower no class | 3.1, 3.2 |
| P5 | the counting cap is vacuous, factor `>= 2` | vacuous **CONFIRMED**; the factor **REFUTED** (1.40 at m13, 1.54 at m23, 1.66-1.70 at the pad words) | 2.5 |
| P6 | skeleton law: runs `<= 3`, jumps `<= 1` iff small-alphabet | **CONFIRMED** on every word to m37 and every corpus word; wording corrected (a small-alphabet word can skip inside one class, `(18, 53, 35)`) | 2.3 |
| P7 | `k_L = L + 1` at every rung with `L >= 2` | **REFUTED** at m29 (`k_L = 3 = L`, the palindrome `(10, 21, 10)`); holds at m19, m31, m37; replaced by `k_L in {L, L+1}` with the chaining criterion | 2.2 |
| P8 | `L_pad <= |A_pad|`, not a cap | **CONFIRMED** - 5 of 5, tight at 4; `|A_pad| ~ 3F/q' - 2` | 5.2 |
| P9 | the shadow is `F(M)/q'` through the skip alphabet | **CONFIRMED**, made exact by E2: `L + 1 <= 2 Omega(T + 1)`, `T = floor((F(M+q') - 2)/q')`; the three other candidates refuted by their tests | 4, 6 |

Not pre-registered, found in the running: **E2** (the skeleton law and the pullback machine,
4.1) and **E3** (the exact corridor-plus-span cap, 4.2), and the regime statement `L <= 5`
whenever `F(M + q') <= 5q' + 1`.

(The owner's own prediction on record, `objects_ledger.md` O-M1: "a uniform cap on `L_pad` of the
shape of the bare cap - a residue obstruction on the padded alternation". P4 is that prediction
made exact, for the small-alphabet half; P3 says whether the half the record has seen is that
half.)

---

## 1. Setup (exact ranges)

Everything is exact integer arithmetic; no sampling anywhere. Scripts in
`research/anchor235/r67/`, results in `research/anchor235/r67/results/` (untracked).

| object | range | cost | script |
|---|---|---|---|
| the machines m5..m23 by direct sieve; spectrum, every maximal realised legal word with its position and count, skeletons, `k_L`, junction analysis, the record of `M + q'` with witnesses, the (order, value) mass of the rung | full periods (`P_23 = 37,182,145`, `N_23 = 7,952,175`) | 8 s | `pc_words.py` |
| m29 (depth 10, 15,240,585 rows), m31 (depth 6, 2,678,901 rows), m37 (depth 2, 2,053 rows) off the r61/r66 closure from `D_15^#(m23)` (4,407,350 rows): spectrum, every realised legal `k`-word with multiplicity, `k_L`, the rung's (order, value) mass, record witnesses | complete dictionaries, `loss = 0`, `over0 = 0` at all three rungs | 611 s | `pc_ladder.py` |
| positions on m29 (29 copies of `P_23`), m31 (899 copies) and m37 (33,263 copies) by the copy law, with the exact counts `W_1, W_2, W_3` as gates against the dictionaries | whole periods | 33 s (m29); m31, m37 see 2.4 | `pc_phase.py` |
| `CORRCAP_3(c)` for the 48 classes mod 210 (gears 5, 7), mod 2310 (5, 7, 11: 480 classes) and mod 30030 (5, 7, 11, 13: 5,760 classes); the PSORD gate | all classes | 25 s | `pc_corrcap.py` |
| the pullback opening count `Omega(n)` for gears {5}, {5,7}, {5,7,11}, {5,7,11,13}, `n <= 40`; the exact corridor + span cap `CC`; corridor carriers; skeletons of the corpus words m41..m53 | all invertible steps and starts | 60 s | `pc_skip.py` |

**Instrument gates, all passed.** `F = 2, 5, 7, 11, 18, 25, 34` at m5..m23 and `43, 58, 88`
at m29, m31, m37 (corpus); `L = 1, 0, 1, 1, 1, 2, 1, 3, 3, 2` and
`L_pad0 = 0, 0, 0, 0, 0, 1, 1, 1, 2, 2` at m5..m37 (r66 3.5; ten of ten); `W_2, W_3` at m29,
m31, m37 = `13,000 / 4`, `70,964 / 216`, `3,052 / -` (r66 3.8 and the corpus
`n_4(37 -> 41) = 3,052`); the PSORD table of docs/proofs/12 class for class (48 of 48); the
copy-law scan at m29 gives `W_1, W_2, W_3 = 8,022,924 / 13,000 / 4` and its maximal-word
counts agree with the dictionary (`(10, 21)`: 6,496 maximal words + 4 inside `(10, 21, 10)` =
6,500 windows; `mult(10) = 7,815,766`, `short_letter_row.md` 2.1); the pad letter's (order,
value) mass at 23 -> 29 from the period and from the ladder agree
(`31: {1: 500, 2: 1280, 3: 310}`); the record witnesses at 29 -> 31 and 31 -> 37 are r66's and
r61's, `(18, 10, 30)`, `(23, 10, 25)` and `(11, 12, 37, 28)`.

## 2. Results

### 2.1 The alphabet at every rung (item 1; P2)

`a, b` the bare letters, multiplicity per period in brackets, then the padded letters realised.
"size-feasible" means a legal value `<= F(M)`.

| `M` | `q'` | `a`, `b` | `F(M)` | realised legal values [mult] | size-feasible holes | `A_pad` | `|A_leg|` | `|A_pad|` |
|---|---|---|---|---|---|---|---|---|
| m5 | 7 | 2, 5 | 2 | 2 [2] | - | - | 1 | 0 |
| m7 | 11 | 4, 7 | 5 | none | 4 | - | 0 | 0 |
| m11 | 13 | 4, 9 | 7 | 4 [6] | - | - | 1 | 0 |
| m13 | 17 | 6, 11 | 11 | 6 [60], 11 [12] | - | - | 2 | 0 |
| m17 | 19 | 6, 13 | 18 | 6 [1,022], 13 [66] | - | - | 2 | 0 |
| m19 | 23 | 8, 15 | 25 | 8 [10,462], 15 [1,236], **23 [86]** | - | {23} | 3 | 1 |
| m23 | 29 | 10, 19 | 34 | 10 [243,370], 19 [440], **29 [6]** | - | {29} | 3 | 1 |
| m29 | 31 | 10, 21 | 43 | 10 [7,815,766], 21 [205,068], **31 [2,090]** | **41** (`a + q'`) | {31} | 3 | 1 |
| m31 | 37 | 12, 25 | 58 | 12, 25, **37 [26,366]**, **49 [46]** (`a + q'`) | - | {37, 49} | 4 | 2 |
| m37 | 41 | 14, 27 | 88 | 14, 27, **41 [61,460]**, **55 [9,910]** (`a + q'`), **68 [60]** (`b + q'`) | **82** (`2q'`) | {41, 55, 68} | 5 | 3 |

P2 CONFIRMED at all ten rungs, value for value. The PAD alphabet is empty to m17 and is
`{23}, {29}, {31}, {37, 49}, {41, 55, 68}` at m19..m37. The skip letters first appear at m31
(`49 = a + q'`, 46 occurrences per period) and m37 (`55 = a + q'`, 9,910; `68 = b + q'`, 60);
the two size-feasible legal values that are spectral holes, `41 = 10 + 31` at m29 and
`82 = 2 * 41` at m37, have non-empty corridor carriers (`{12, 17, 32}` and 8 residues), so
their absence is the cover half - a 40-column blocked stretch two below the record 43, an
81-column one seven below 88 - not a residue exclusion. The pad `q'` is a near-record gap at
m19 (23 against `F = 25`, 86 per period) and m23 (29 against 34, 6 per period) and ordinary
from m29 on (31 against 43: 2,090; 37 against 58: 26,366; 41 against 88: 61,460).

### 2.2 `L` and its halves at every rung (item 1; P1, P7)

| `M` | `q'` | `L` | `L_bare` | `L_small` | `L_pad` (brief) | `L_pad0` (class 0) | `L_skip` | corpus `L` / `L_pad` | `k_L` | `L^{(1)}, L^{(2)}, ...` |
|---|---|---|---|---|---|---|---|---|---|---|
| m5 | 7 | 1 | 1 | 0 | 0 | 0 | 0 | - | 1 | 1, 1 |
| m7 | 11 | 0 | 0 | 0 | 0 | 0 | 0 | - | 1 | 0 |
| m11 | 13 | 1 | 1 | 0 | 0 | 0 | 0 | 1 / 0 | 1 | 1, 1 |
| m13 | 17 | 1 | 1 | 0 | 0 | 0 | 0 | 1 / 0 | 2 | inf, 1 |
| m17 | 19 | 1 | 1 | 0 | 0 | 0 | 0 | 1 / 0 | 2 | inf, 1 |
| m19 | 23 | 2 | 2 | 1 | 1 | 1 | 0 | 2 / 1 | 3 | inf, inf, 2 |
| m23 | 29 | 1 | 1 | 1 | 1 | 1 | 0 | 1 / 1 | 2 | inf, 1 |
| m29 | 31 | 3 | 3 | 1 | 1 | 1 | 0 | 3 / 1 | **3** | inf, inf, 3, 3 |
| m31 | 37 | 3 | 3 | 2 | 2 | 2 | 1 | 3 / 2 | 4 | inf, inf, inf, 3 |
| m37 | 41 | 2 (depth 2; corpus 2) | 1 | 2 | 2 | 2 | 1 | 2 / 2 | 3 (from corpus `L = 2`) | inf, inf, (2) |

P1 CONFIRMED: ten of ten `L`, ten of ten `L_pad0`. The two definitions of `L_pad` agree at
every rung, because every padded letter inside a realised word of length `>= 2` is `q'` itself
(the skip letters 49, 55, 68 occur only as 1-words) - see section 5. `L_skip = 1` at m31 and
m37 is the single skip letter standing alone.

**P7 REFUTED, with a mechanism, and replaced.** `k_L`, the order at which the table decides `L`,
is `L + 1` at m13, m17, m19, m23, m31 (and m37 by the corpus) and `L` at m29 - and at m5, m11,
where the legal alphabet has one letter. At m29 the only realised 3-word is the palindrome
`(10, 21, 10)` (4 per period); its two overlaps `(21, 10)` and `(10, 21)` are realised, but the
4-words they could form, `(10, 21, 10, 21)` and `(21, 10, 21, 10)`, each need `(21, 10, 21)`,
which is not realised, so the 3-window table already has no legal 4-walk. At m31 both
`(12, 25, 12)` (188) and `(25, 12, 25)` (28) are realised, they chain, the 3-window table has a
cycle (`L^{(3)} = inf`), and only the 4-window table decides. The same palindrome obstruction is
the record's "overlap lemma" at m53 (the sole 3-word `(20, 98, 20)`), so `k_L(53) = 3 = L`.
Sharpest true statement: **`k_L in {L, L + 1}`, and `k_L = L` iff the realised `L`-words do not
chain into a legal `(L+1)`-word** (an identity, once said). With the order law `k* = L + 1`
this means the table that decides the budget, `D_{L+1}`, decides `L` too, except at the rungs
where a palindrome closes the alphabet one order earlier.

### 2.3 Every realised legal word of length `>= 2`, and every padded word, with its skeleton (item 2; P6)

`x N` is the number of occurrences per period; `S_0`, `S_1` the multiplier sets of the two tooth
classes (0.1); runs = the lengths of the maximal runs of consecutive multipliers per class;
jumps = the multiplier increments between consecutive openings. (Occurrence counts at m29, m31,
m37 are window multiplicities off the dictionaries.)

| `M` | word | kind | occurrences | `S_0` | `S_1` | runs | skips | jumps |
|---|---|---|---|---|---|---|---|---|
| m19 | (8, 15), (15, 8) | bare | 31, 31 | {0, 1} | {0} | 2 + 1 | 0 | 0, 1 |
| m19 | (23) | small | 86 | {0, 1} | - | 2 | 0 | 1 |
| m23 | (29) | small | 6 | {0, 1} | - | 2 | 0 | 1 |
| m29 | (10, 21, 10) | bare | 4 | {0, 1} | {0, 1} | 2 + 2 | 0 | 0, 1, 0 |
| m29 | (10, 21), (21, 10) | bare | 6,500, 6,500 | {0, 1} | {0} | 2 + 1 | 0 | 0, 1 |
| m29 | (31) | small | 2,090 | {0, 1} | - | 2 | 0 | 1 |
| m31 | (12, 25, 12) | bare | 188 | {0, 1} | {0, 1} | 2 + 2 | 0 | 0, 1, 0 |
| m31 | (25, 12, 25) | bare | 28 | {0, 1} | {0, 1} | 2 + 2 | 0 | 0, 1, 0 |
| m31 | (12, 25), (25, 12) | bare | 35,314 each | {0, 1} | {0} | 2 + 1 | 0 | 0, 1 |
| m31 | (12, 37), (37, 12) | small | 150 each | {0}, {0, 1} | {0, 1}, {1} | 1 + 2 | 0 | 0, 1 / 1, 0 |
| m31 | (25, 37), (37, 25) | small | 18 each | {0}, {0, 1} | {0, 1}, {1} | 1 + 2 | 0 | 0, 1 / 1, 0 |
| m31 | (37) | small | 26,366 | {0, 1} | - | 2 | 0 | 1 |
| m31 | (49) | skip | 46 | {0} | {1} | 1 + 1 | 0 | **1 with a class change** |
| m37 | (14, 41), (41, 14) | small | 1,525 each | {0}, {0, 1} | {0, 1}, {1} | 1 + 2 | 0 | 0, 1 / 1, 0 |
| m37 | (27, 41), (41, 27) | small | **1 each** | {0}, {0, 1} | {0, 1}, {1} | 1 + 2 | 0 | 0, 1 / 1, 0 |
| m37 | (41) | small | 61,460 | {0, 1} | - | 2 | 0 | 1 |
| m37 | (55), (68) | skip | 9,910, 60 | {0} | {1} | 1 + 1 | 0 | 1 with a class change |

The corpus words above the ladder (skeletons computed, `pc_skip.py`):

| `M` | `q'` | word | kind | `S_0` | `S_1` | skips | jumps |
|---|---|---|---|---|---|---|---|
| m41 | 43 | (43, 43) | small | {0, 1, 2} | - | 0 | 1, 1 |
| m43 | 47 | (47, 47) | small | {0, 1, 2} | - | 0 | 1, 1 |
| m47 | 53 | (18, 35, 18, 35) | bare | {0, 1, 2} | {0, 1} | 0 | 0, 1, 0, 1 |
| m47 | 53 | (18, 35, 53), (35, 18, 53) | small | {0, 1, 2} | {0} | 0 | 0, 1, 1 |
| m47 | 53 | (18, 53, 35) | small | {0, 2} | {0, 1} | 1 | 0, 1, 1 |
| m47 | 53 | (35, 71, 35), undecided on record | **skip** | {0, 2} | {0, 2} | 2 | 0, **2**, 0 |
| m53 | 59 | (20, 39), (20, 59) | bare, small | {0, 1}, {0} | {0}, {0, 1} | 0 | 0, 1 |
| m53 | 59 | (20, 98), (20, 118) | **skip** | {0, 2}, {0} | {0}, {0, 2} | 1 | 0, **2** |
| m53 | 59 | (20, 98, 20) | **skip** | {0, 2} | {0, 2} | 2 | 0, **2**, 0 |

P6 CONFIRMED on every word computed and every corpus word: no class has a run of four
multipliers (the AP lemma), and a consecutive-opening jump of `>= 2` - or a jump of 1 with the
class changing from 0 to 1, which is the letter `a + q'` or `b + q'` - occurs exactly on the
skip words. One correction to the pre-registered wording: a small-alphabet word CAN skip a
multiplier in one class - `(18, 53, 35)` at m47 has `S_0 = {0, 2}` - because a pad in class 1
carries the multiplier past a class-0 slot; what the small alphabet forbids is a jump `>= 2`
between consecutive openings, so the union `S_0 u S_1` is the whole interval `[0, m_max]`.
That is the fact E1's mechanism uses (3.2).

### 2.4 Where the longest padded word sits (item 1)

At m19, m23, m29 the longest padded word is the single pad `q'` (86, 6 and 2,090 occurrences);
at m31 it is `(12, 37)` / `(37, 12)` (150 each) and `(25, 37)` / `(37, 25)` (18 each); at m37
`(14, 41)` / `(41, 14)` (1,525 each) and `(27, 41)` / `(41, 27)` (one each). Positions and
gears (`pc_words.py` and `pc_phase.py` junction analysis; six instances at m19 and m23, three
at m29; m31 and m37 positions: see the addendum 2.4a):

- **Every gear of `M` is a sole striker inside the pad letter**, at 6 of 6 instances at m19
  (gears 5..19), 6 of 6 at m23 (gears 5..23), and 2 of 3 at m29 (gears 5..29; the instance at
  `X_0 = 1,479,277` does without gear 29). At m19 and m23 the pad is two and five columns below
  the record and behaves as an above-record stretch in L4's sense (docs/proofs/19: every gear a
  sole striker); at m29 it is twelve below and the law no longer holds.
- **The corridor fixes the start residue.** The gap 29 at m23 starts at residue 3 mod 35 at
  all 6 occurrences: the corridor carrier of the shape `(0, 29)` is exactly `{3}`. The gap 23
  at m19 starts at residue 5 or 7 (carrier `{2, 5, 7}`; residue 2 unused in the six analysed);
  the gap 31 at m29 starts at 2, 7 or 32 at all 400 positions sampled (carrier `{2, 7, 32}`,
  all used). The start residue mod `q'` is unconstrained (all 31 residues occur at m29): the
  pad is not tied to a phase of the incoming gear, as the copy law requires.
- **Junction neighbours.** The columns `x_0 - 1` and `x_L + 1` next to the word's ends are
  struck by gear 5 or 7 in every instance but one; at m23 the left neighbour of the start is
  struck by 23 alone at 5 of 6 instances (13 and 23 at the sixth) and the right neighbour of
  the end by 23 alone at 5 of 6 - gear 23's teeth frame the gap 29 at distance 31, not a tooth
  separation of 23 (`2u_23 = 8`): a coincidence of six positions, not a law.
- The bare longest words share needed gears across consecutive letters: `(8, 15)` at m19
  needs `{11, 17, 19}` in the 8 and `{5, 7, 11, 13, 17}` in the 15, shared `{11, 17}`;
  `(15, 8)` shares `{5, 19}`. Two consecutive letters can and do share a gear; the twin pair
  `(17, 19)` (collision onset `(17 + 4)/3 = 7`, docs/proofs/21) serves the same letter at
  m19's `(8, 15)`. The collision laws bound what two gears cover jointly; they do not forbid a
  gear serving two letters, and nothing here turns them into a cap.

**2.4a Positions on m31 and m37 (copy-law scans).** m31: all 899 copies, 661 s, every position
of the 336 longest padded words recorded (150 + 150 + 18 + 18). m37: copies 0..599 of 33,263
(1.8 % of the period, 138 s on 3 processes), 50 positions of `(14, 41)` / `(41, 14)` (26 + 24
against 55 expected from the multiplicity 3,050), 1,075 of `(41)`, 178 of `(55)`, 3 of `(68)`,
and no 3-word in the sample (the corpus `L(37) = 2`). The full m37 scan (about 2.3 h) was not
run: the dictionary already carries m37's words exactly, and the positions serve only the
picture below.

- The pad `q'` inside a 2-word needs nearly every gear: at m31 the 37 needs 8 or 9 of the 9
  gears 5..31 at all six instances shown, the bare letter next to it 3 to 5 gears, and the two
  letters share 3 to 5 needed gears (always including 5); at m37 the 41 needs 9 or 10 of the
  ten gears 5..37 (all ten at 3 of 6 instances), the 14 needs 5 or 6, shared 5 or 6. The
  counting ratio at these words is 1.66 (m31) and 1.70 (m37); the junction ratio 13 and 15.
- The start residue mod 35 of `(12, 37)` at m31 is 23, 18, 30 in the instances shown and of
  `(37, 12)` 3, 33, 3 - inside the carriers; the residues mod `q'` are unconstrained, as
  before. At m37 `(14, 41)` starts at 18, 33, 18 and `(41, 14)` at 32, 17, 32 mod 35.
- The junction between the two letters (the opening struck by `q'` in the fusion) has gear 5
  on at least one side at 11 of the 12 instances shown, and gear 7 on the other side at 8.

### 2.5 The counting cap (item 2; P5): vacuous, and not by the factor predicted

For the realised word of span `S` and length `L`, the interior `S - L` columns are struck and the
capacity `sum_{g in M} max_g(S + 1)` (docs/proofs/20 Lemma 2) bounds what the gears can strike:

| `M` | word | `S` | need `S - L` | capacity | ratio | junction need | junction ratio | gears above the span |
|---|---|---|---|---|---|---|---|---|
| m13 | (11) | 11 | 10 | 14 | 1.40 | 4 | 3.5 | 13 |
| m17 | (6) | 6 | 5 | 11 | 2.20 | 4 | 2.75 | 11, 13, 17 |
| m19 | (23) | 23 | 22 | 34 | 1.55 | 4 | 8.5 | none |
| m19 | (8, 15) | 23 | 21 | 34 | 1.62 | 6 | 5.67 | none |
| m23 | (29) | 29 | 28 | 43 | 1.54 | 4 | 10.75 | none |
| m23 | (10) | 10 | 9 | 19 | 2.11 | 4 | 4.75 | 13, 17, 19, 23 |
| m29 | (31) | 31 | 30 | 50 | 1.67 | 4 | 12.5 | none |

The capacity exceeds the need at every word (it must: the word is realised), by a factor between
1.4 and 2.2 - NOT `>= 2` as pre-registered: at the pad letters the ratio is 1.54-1.67, because a
pad is a near-record gap and the record is where the machine's capacity is nearly all spent
(docs/proofs/21: "the record stretch is almost exactly maximal gear by gear"). The junction
version is vacuous by 2.75-12.5. **A count of strikes caps nothing about a word, at any rung**:
the machine has spare capacity at every span, and the pre-registered cap ("`L + 1` junction
gears in a span bounded by `L` times the largest letter, gears above the span striking at most
twice") reads `L + 1 <= capacity / 2`, which is `>= S/5` by gear 5 alone and so never below
`L`. P5: vacuous CONFIRMED, the factor `>= 2` REFUTED (1.40 at m13, 1.54 at m23). Brick: the
only thing that can cap a word is *which* gear strikes *which* column - a residue statement -
and that is what E1 and E2 are.

### 2.6 The record's composition and the pad letter's fusion depth (shadow tests (b), (a))

**(b) The record of `M + q'` and its middles**, from the record-attaining fusions of each rung
(`lc_core.witnesses`, exact):

| rung | `F(M+q')` | record fusion (flank, middles, flank) | `J` | middles' kind |
|---|---|---|---|---|
| 5 -> 7 | 5 | (2, 2, 1) | 3 | bare |
| 7 -> 11 | 7 | (2, 5) | 2 | none |
| 11 -> 13 | 11 | (6, 5) | 2 | none |
| 13 -> 17 | 18 | (5, 11, 2) | 3 | bare (`11 = b`) |
| 17 -> 19 | 25 | (18, 7) and (7, 13, 5) | 2, 3 | none; bare |
| 19 -> 23 | 34 | (7, 15, 8, 4) | 4 | bare |
| 23 -> 29 | 43 | (23, 10, 10) | 3 | bare (`10 = a`) |
| 29 -> 31 | 58 | (18, 10, 30), (23, 10, 25) | 3 | bare |
| 31 -> 37 | 88 | (11, 12, 37, 28) | 4 | **small** (pad 37) |
| 37 -> 41 | 91 | (., 41, 14, .) (r61) | 4 | **small** (pad 41) |

The record's middles are padded exactly at the two rungs where `L_pad(M) = 2` and nowhere
below; at m31 the record takes the padded 2-word `(12, 37)` (span 49) over the bare 3-words
`(12, 25, 12)`, `(25, 12, 25)` (spans 49, 62), whose `J = 5` fusions reach only `Q*_5 = 68`.
So the padded word carries the record from 31 -> 37 on, and at both rungs the longest padded
word is the record's middle. Two rungs, not a law; it says that the pad `q'` is a
record-sized letter, which 2.1 already says.

**(a) The pad letter as a fusion of the rung below**, from the (order, value) mass of the
closure step (exact):

| machine | padded letter | order 1 (inherited) | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| m19 | 23 | - (`23 > F(m17) = 18`) | 48 | 38 | | |
| m23 | 29 | - (`29 > 25`) | | 6 | | |
| m29 | 31 | 500 | 1,280 | 310 | | |
| m31 | 37 | 2,268 | 10,132 | 9,278 | 4,688 | |
| m31 | 49 | - (`49 > 43`) | 8 | 26 | 12 | |
| m37 | 41 | 4,422 | 28,500 | 25,244 | 3,294 | |
| m37 | 55 | 1,122 | 3,936 | 3,528 | 1,214 | 110 |
| m37 | 68 | - (`68 > 58`) | 4 | 10 | 26 | 20 |

The pad is an ordinary product of the forest: shallow, and once `q' <= F(M)` mostly inherited or
two-fold. Its depth is not what makes it a padded letter; its size is. **The merge forest's
depth is not what the PAD alphabet is the shadow of** (test (a): refuted as a candidate).

## 3. The cap on the small-alphabet half: E1 (proved, uniform)

### 3.1 `CORRCAP_3` at the 48 classes (P4)

`CORRCAP_3(c)` = the longest legal word over `{a_c, b_c, c}` whose offset walk mod 35 stays in
`E_35` (`pc_corrcap.py`; the PSORD gate of docs/proofs/12 passes at 48 of 48 classes).
**Finite at every class.** Distribution over the 48 classes: value 2 at 4 classes, 3 at 16, 4 at
12, 5 at 10, 6 at 4, **8 at 2** (classes 103 and 107); never 7. The pre-registered `<= 7` is
refuted by one unit; the maximum is **8**.

| `c` mod 210 | PSORD | `CORRCAP_3` | longest small-alphabet word fitting 5 and 7 |
|---|---|---|---|
| 1, 61, 89, 121, 149, 179, 209 | 3 | 3 | e.g. `(20, 41, 20)` at 61 |
| 11, 19, 71, 73, 137, 139, 191, 199 | 1 | 3 | `(4, 11, 7)`, `(19, 6, 19)`, `(73, 73, 49)`, ... |
| 13, 17, 193, 197 | 1 | 6 | `(13, 13, 9, 13, 13, 4)`, `(17, 11, 17, 17, 6, 17)` |
| 23, 67, 97, 113, 143, 187 | 3 | 4 | `(15, 8, 15, 23)`, `(67, 45, 22, 45)` |
| 29, 59, 151, 181 | 2 | 2 | `(29, 10)`, `(20, 59)` |
| **31** | 3 | **3** | `(10, 21, 10)` - the realised word of m29 |
| **37**, 53, 83, 127, 157, 173 | 5 | 5 | `(25, 12, 25, 12, 25)`, `(35, 18, 35, 18, 35)` |
| **41**, 43, 79, 131, 167, 169 | 1 | 4 | `(27, 41, 14, 41)`, `(43, 43, 14, 43)` |
| **47**, 101, 109, 163 | 1 | 5 | `(31, 47, 47, 16, 47)`, `(67, 101, 34, 101, 67)` |
| **103, 107** | 1 | **8** | `(103, 103, 34, 103, 103, 69, 103, 103)` |

Adding gear 11 (classes mod 2310, 480 of them) and gear 13 (mod 30030, 5,760) lowers the
maximum of **no** class mod 210: the value at every lift equals the value at the class, so gears
5 and 7 are the whole obstruction on the small alphabet at every scale.

### 3.2 The law and its proof

> **E1 (the small-alphabet cap).** Let `M` contain gears 5 and 7 and let `q' >= 11` be a prime not
> in `M`, `c = q' mod 210`. Every legal word realised in `M` over the small alphabet
> `{a, b, q'}` has length at most `CORRCAP_3(c)`, and `CORRCAP_3(c) <= 8` for every class.
> In particular `max(L_bare, L_small) <= 8` for every machine and every scale.

*Proof.* (1) A word realised at the opening `x` has offsets `o_0 = 0 < o_1 < ... < o_m` (its
partial sums) that are all openings, so `x + o_i mod 35 in E_35` for every `i` (docs/proofs/14
(a)), and `x mod 35 in E_35`. (2) `a = aOfClass(c) (mod 70)` and `q' = c (mod 210)`
(docs/proofs/12, step 3), so the three letter values are the class values mod 35, and the
offset walk `r -> r + letter (mod 35)` from `r = x mod 35` is a walk on `E_35` with the class
steps. (3) Legality is the combinatorial condition "no two consecutive nonzero letters equal"
on the letter sequence (docs/proofs/05 (F)), the same for the class word. (4) Hence the word is
a walk in the finite graph `G_c` on states `E_35 x {last nonzero class}` with the three class
steps; `CORRCAP_3(c)` is the length of the longest walk, which is finite because `G_c` is
acyclic - checked for the 48 classes, and forced by the density mechanism of 3.3 below, which
gives `<= 25` by hand. Nothing depends on `M` beyond gears 5 and 7. QED

**Against the record.** `L_small <= CORRCAP_3(q' mod 210)` at every rung from m5 to m37 (ten
of ten, computed) and at m41, m43, m47, m53 (corpus words):

| `M` | `q'` | `CORRCAP_3` | `L_bare` | `L_small` | `L` | tight? |
|---|---|---|---|---|---|---|
| m19 | 23 | 4 | 2 | 1 | 2 | |
| m23 | 29 | 2 | 1 | 1 | 1 | |
| m29 | 31 | **3** | 3 | 1 | **3** | yes (`(10, 21, 10)`) |
| m31 | 37 | 5 | 3 | 2 | 3 | |
| m37 | 41 | 4 | 1 | 2 | 2 | |
| m41 | 43 | 4 | 1 | 2 | 2 | |
| m43 | 47 | 5 | 1 | 2 | 2 | |
| m47 | 53 | 5 | 4 | 3 | 4 | |
| m53 | 59 | **2** | 1 | 2 | **3** | yes for the small half (`(20, 59)`); `L = 3` is a SKIP word |

**And it explains a datum the record left as a computation.** At 53 -> 59 the record found
`(20, 39, 20)`, `(20, 59, 39)`, `(39, 59, 20)` and every other 3-word over `{20, 39, 59}` ZERO,
by the band table, the screen and one SAT call, and the realised 3-word to be the palindrome
`(20, 98, 20)` with `98 = b + q'` (mechanic round 27, agents-shared 3114-3131). E1 says the
first fact in one line - `CORRCAP_3(59) = 2`, the corridor carriers of those words are empty
(`pc_skip.py`) - and it says what the second fact IS: at m53 the machine's depth `L = 3` is
carried by a word **outside the small alphabet**, the first such word on record. The record's
`L_pad = 3` at m47 (`(18, 35, 53)`) is small-alphabet; at m53 it is not.

### 3.3 The mechanism: the two tooth classes carry 15/35 each, and the small alphabet needs 35/35

Read the word in the skeleton coordinate (0.1, 2.3). Over the small alphabet every consecutive
pair of openings differs by 0 or 1 in the multiplier `m`, so `S_0 u S_1 = {0, 1, ..., m_max}`
and `m + 1 <= |S_0| + |S_1| <= 2 (m_max + 1)`. Each class is an arithmetic progression of
difference `q'` in the column, so in any window of `n` consecutive multipliers it holds at most
`Omega_{5,7}(n)` openings, the most that `n` consecutive terms of an AP with step coprime to 35
can put into `E_35`:

    n            1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 ... 24 25 26 27 ... 35
    Omega_{5,7}  1  2  3  3  3  4  5  6  6  6  7  7  8  8  9 ... 12 13 13 13 ... 15

(`pc_skip.py`; gear 5 alone gives `1, 2, 3, 3, 3, 4, 5, 6, 6, 6, ...`, three in every five.)
So a small-alphabet word with `m_max + 1 = n` needs `n <= |S_0| + |S_1| <= 2 Omega_{5,7}(n)`,
and `2 Omega_{5,7}(n) < n` from `n = 27` on (`26 < 27`): `m_max <= 25`, hence `L + 1 <= 52`
crudely, and finiteness is proved by hand; the exact graph computation replaces 25 by the
class value, at most 8. **Gear 5 alone does not cap the small alphabet**: the pattern
`(s, q', q', q' - s, q', q', ...)` puts each class on the residues `{0, 3, 4}` mod 5 for ever
(three of five, the AP lemma satisfied), and it is gear 7 that breaks it. That is why the bare
cap of docs/proofs/12 and E1 are corridor statements and not gear-5 statements, and why
neither gear 11 nor 13 lowers a single class: at the small alphabet the union of the two
classes must be everything, and 15/35 + 15/35 < 1 is already a contradiction; the larger gears
only thin a set that is already too thin.

The same mechanism says exactly what a longer word must do: **skip**. A letter `2q'`,
`a + q'` or `b + q'` moves the multiplier by 2 (or by 1 with the class going the wrong way),
leaving a multiplier that neither class occupies - a tooth column of the phase that `M` has
blocked. With skips the union need not be everything, the density contradiction disappears, and
no bounded set of gears caps the word: that is the record's `CORRCAP = infinity` from 53 -> 59
(alignment-rules 6.2), now with its reason. So `L_pad` splits:

    L_pad(M) = max( L_small-padded(M),  L_skip(M) ),    L_small-padded <= CORRCAP_3(q' mod 210) <= 8,

and the open half of gate item 1 is `L_skip`, the words with a skip letter.

## 4. The skip half: the skeleton law E2 and the exact corridor-plus-span cap E3

### 4.1 The pullback machine

For a realised legal word at `x_0`, the class-`j` openings are `x_0 + s_j + m q'` (`s_0 = 0`,
`s_1 = s in {a, b}`). The column `x_0 + s_j + m q'` is open in `M` iff for every gear `g`

    m  !=  (+-u_g - x_0 - s_j) q'^{-1}   (mod g),

two forbidden residues mod `g` at separation `2 u_g q'^{-1} mod g`. So `S_j` is a set of
openings of a two-tooth machine on the multiplier line - the **pullback** `M^{(q')}` of `M`
along the tooth AP, same gears, separations `2u_g q'^{-1}` - restricted to `[0, T]` with
`T = floor((F(M+q') - 2)/q')`, because the word's span is at most `F(M+q') - 2`
(docs/proofs/11, step 3). Writing `Omega_M(n)` for the most openings the pullback can have in
`n` consecutive multipliers over all its phases:

> **E2 (the skeleton law).** For every realised legal word of `M` with respect to `q'`,
>
>     L + 1  =  |S_0| + |S_1|  <=  2 Omega_M(T + 1),     T = floor((F(M+q') - 2)/q'),
>
> and `Omega_M(n) <= Omega_{5,7}(n; q' mod 35) <= Omega_{5,7}(n)`, where `Omega_{5,7}(n; r)` is
> the most terms of an AP of step `r` and length `n` that lie in `E_35`. A gear `g >= 2n + 1`
> can be phased off any window of `n` multipliers (its two teeth fit in the complement), so
> only the gears `<= 2n` bite - the completeness lemma of docs/proofs/14 (d) in the pullback.

*Proof.* `|S_0| + |S_1| = L + 1` is the definition of the skeleton (every opening of the word is
on one of the two teeth of the phase, docs/proofs/10 Theorem 1). Each `S_j` lies in `[0, T]`
and is a set of openings of the pullback (the display above, which is the tooth rule
`x = +-u_g (mod g)` read along the AP, `q'` invertible mod `g`). The last clause is the
corridor: an AP of step `q'` mod 35 in the column is an AP of step `q' mod 35` in `Z_35`. QED

Consequences, all proved:

- **`L <= 5` whenever `F(M + q') <= 5 q' + 1`** (i.e. `T <= 4`), by gear 5 alone: five
  consecutive multipliers meet every residue mod 5 once, two are teeth, so `Omega_5(5) = 3`
  and `L + 1 <= 6`. The whole corpus has `T = 1, 1, 1, 2, 2, 2, 2, 2, 2` at m19..m53
  (`F(M+q')/q'` at most `161/59 = 2.73`), so `L <= 5` there, and the bound stays `5` up to
  `F(M+q') = 5q' + 1`, where docs/proofs/11's `2T + 1` already reads 9.
- The uniform cap table `L <= 2 Omega_{5,7}(T + 1) - 1`:

      T           1  2  3  4  5  6   7   8   9  10  11  12  13  14  15  16  17  18
      E2          3  5  5  5  7  9  11  11  11  13  13  15  15  17  17  17  19  19
      file 11     3  5  7  9 11 13  15  17  19  21  23  25  27  29  31  33  35  37

  equal at `T <= 2`, and `(6/7) T + O(1)` against `2T + 1` from `T = 3` on; gears 11 and 13
  change the row by at most one unit (`T = 19, 20`: 19 instead of 21).
- The per-class value `2 Omega_{5,7}(T + 1; q' mod 35) - 1` at the corpus rungs is `3, 3, 3,
  5, 3, 5, 5, 5, 3` at m19..m53 against `L = 2, 1, 3, 3, 2, 2, 2, 4, 3`: it holds at 9 of 9,
  and it is **tight at m29** (`(10, 21, 10)`) **and at m53** (`(20, 98, 20)`), and it says
  `L(37) <= 3` and `L(53) <= 3` where the uniform row says 5 - because an AP of step
  `41 = 6` or `59 = 24 (mod 35)` and length 3 puts at most two terms in `E_35`.

### 4.2 The exact corridor-plus-span cap: E3

E2 is a count; the exact object is the longest legal word with letter values `<= F(M)`, total
span `<= F(M+q') - 2` and offset walk inside `E_35` - the record's `CORRCAP(q', F)` (uniform-
order-bound; a letter cap only) with docs/proofs/11's span cap added. A dynamic programme on
`(span used, residue mod 35, last nonzero class)` computes it exactly (`pc_skip.py`,
`corrcap_span`):

> **E3.** `L(M) <= CC(q', F(M), F(M+q'))`, the longest legal word over the legal values
> `<= F(M)` with span `<= F(M+q') - 2` whose offsets fit gears 5 and 7.

| `M` | `q'` | `F(M)` | `F(M+q')` | `CC` | its witness | `L` | slack |
|---|---|---|---|---|---|---|---|
| m19 | 23 | 25 | 34 | 3 | (8, 15, 8) | 2 | 1 |
| m23 | 29 | 34 | 43 | 2 | (29, 10) | 1 | 1 |
| m29 | 31 | 43 | 58 | **3** | (10, 21, 10) | **3** | **0** |
| m31 | 37 | 58 | 88 | 4 | (25, 12, 25, 12) | 3 | 1 |
| m37 | 41 | 88 | 91 | 3 | (14, 41, 27) | 2 | 1 |
| m41 | 43 | 91 | 103 | 3 | (43, 43, 14) | 2 | 1 |
| m43 | 47 | 103 | 118 | 3 | (47, 16, 47) | 2 | 1 |
| m47 | 53 | 118 | 145 | 5 | (35, 18, 35, 18, 35) | 4 | 1 |
| m53 | 59 | 145 | 161 | **3** | **(20, 98, 20)** | **3** | **0** |

**Nine of nine, slack 0 or 1 at every rung, tight at m29 and m53 - and at m53 the cap's own
witness is the realised skip word.** This is the sharpest true statement about `L` this branch
can make with the corridor: the record's `CORRCAP` row is `4, 2, 3, 5, 25, 25, 11, 5` at the
same rungs (19 -> 23 .. 47 -> 53), so the span cap is what removes the 25s (at 37 -> 41 the
letters `<= 88` allow a 25-walk in `E_35`, the span `<= 89` allows three letters). E3 is still
not uniform: `CC` grows with `T` like E2's count, since a walk of `T + 1` multipliers per class
is what the span permits.

### 4.3 What E1-E3 say about gate item 1, exactly

- The tame half is closed: words over `{a, b, q'}` have length `<= CORRCAP_3(q' mod 210) <= 8`,
  uniformly, by gears 5 and 7 (E1). This includes the padded words the record has measured at
  every rung up to m47.
- The open half is the skip half, and E2 locates its growth: a skip word of length `L` needs
  `2 Omega(T + 1) >= L + 1`, i.e. a span of at least `q' * (Omega^{-1}((L+1)/2) - 1)` columns
  inside a fusion of `M + q'`; with gears 5 and 7, `L >= 6` needs `T >= 5`, i.e.
  `F(M + q') >= 5 q' + 2`; `L >= 8` needs `T >= 6`; `L >= 10` needs `T >= 7`. On the corpus
  `T = 2` at every rung from m31 to m53, and the skip half has reached `L = 3` at m53 - the
  per-class maximum (4.1). So **on the whole corpus `L` is at the corridor's cap or one
  below it**, and every unit of growth above 5 costs the record `F(M+q')` at least one more
  `q'`.
- Whether `L_skip` is bounded is therefore exactly the question whether `T = F(M+q')/q'` stays
  below a constant - the record in units of the gear - which grows along the corpus
  (`0.54 .. 2.73`) and, if the budget inequality holds with any slack, keeps growing. **The cap
  on `L` is not a residue question and not a counting question; it is the record's growth
  rate in gear units, and that is the root question restated (ROOT).** E2 makes the exchange
  rate exact: `(6/7)` of a letter per unit of `F/q'`, with gears 5 and 7; the larger gears
  buy at most one unit more by `T = 20`.

## 5. The skip half on the record: where it starts (item 3; P3)

### 5.1 The first skip words

A skip letter needs `F(M) >= q' + a` (the letter `a + q'`) or `>= 2q'`; a skip word of length
2 needs `F_2(M) >= 2q'` (the cheapest are `(b, a + q')`, `(a, b + q')`, `(q', q')` with span
`2q'`; `(q', q')` is small-alphabet). The thresholds along the corpus, with what the machine
does at each:

| `M` | `q'` | `F(M)` | `F_2(M)` | skip letters size-feasible | realised as gaps | `2q' <= F_2`? | skip 2-words realised | skip 3-words |
|---|---|---|---|---|---|---|---|---|
| m19 | 23 | 25 | 31 | none (`31 > 25`) | - | no | - | - |
| m23 | 29 | 34 | 39 | none (`39 > 34`) | - | no | - | - |
| m29 | 31 | 43 | 55 | 41 | **hole** | no (`62 > 55`) | - | - |
| m31 | 37 | 58 | 68 | 49 | 49 [46] | no (`74 > 68`) | none possible by size | - |
| m37 | 41 | 88 | 90 | 55, 68, 82 | 55 [9,910], 68 [60]; 82 a hole | **yes** (`82 <= 90`) | **none**: `(27,55)`, `(55,27)`, `(14,68)`, `(68,14)` all absent from the complete `D_2(m37)` | - |
| m41 | 43 | 91 | 103 | 57, 72, 86 | (not on record) | yes (`86 <= 103`) | not on record | not on record |
| m43 | 47 | 103 | 118 (`<= F(53)`) | 63, 78, 94 | (not on record) | yes | not on record | not on record |
| m47 | 53 | 118 | `<= 145` | 71, 88, 106 | (not on record) | yes | not on record | `(35, 71, 35)` undecided (record) |
| m53 | 59 | 145 | 159 | 79, 98, 118, 138 | 98, 118 at least | yes (`118 <= 159`) | **`(20, 98)`, `(20, 118)`** and reverses (record) | **`(20, 98, 20)`** (record) |

P3 CONFIRMED to m37: every realised legal word of length `>= 2` at m5..m37 is over the small
alphabet - 0 exceptions in every word of every period and dictionary - and at m37, the first
rung where a skip 2-word is size-feasible, none of the four is realised although each has a
corridor carrier of four residues (`{5, 18, 25, 33}` for `(27, 55)` and `(55, 27)`,
`{18, 23, 28, 33}` for `(14, 68)`, `{0, 5, 25, 30}` for `(68, 14)`): what excludes them is the
cover half (two adjacent near-record gaps, 27 + 55 or 14 + 68 = 82 against `F_2 = 90`, with
the shape's start pinned to four residues mod 35). The skip half enters the record at m53
(and possibly at m41..m47, which no census reaches): `(20, 98)` and `(20, 118)` as 2-words,
`(20, 98, 20)` as the 3-word that carries `L(53) = 3`, with `F_2(53) = 159 >= 118 = 2q'`.

### 5.2 The letter count against `L_pad` (item 3; P8)

| `M` | `|A_pad|` | `|A_leg|` | `L_pad` | `2T` (file 11, letter-aware with one pad) | `CC` (E3) |
|---|---|---|---|---|---|
| m19 | 1 | 3 | 1 | 2 | 3 |
| m23 | 1 | 3 | 1 | 2 | 2 |
| m29 | 1 | 3 | 1 | 2 | 3 |
| m31 | 2 | 4 | 2 | 4 | 4 |
| m37 | 3 | 5 | 2 | 4 | 3 |

`L_pad <= |A_pad|` at 5 of 5 and tight at 4 of 5 (P8 confirmed as a fact), and it is not a cap:
`|A_pad|` is the number of legal values in `[q', F(M)]`, about `3 F(M)/q' - 2`, and grows with
the record in gear units exactly as E2's `T` does. Every cap in this document, and every cap on
record, is a function of `F/q'`: the letter count (`|A_pad| ~ 3F(M)/q'`), the letter-aware
span cap (`2T`, docs/proofs/11), the skeleton count (`2 Omega(T+1) - 1`, E2), the exact
corridor+span cap (`CC`, E3). They differ in the constant, `3`, `2`, `6/7`, and E3 is the
sharpest (slack 0 or 1 at nine of nine rungs); none is uniform, and section 4.3 says why none
can be while `F/q'` grows.

## 6. The shadow (item 4; P9)

The brief's four candidates, one concrete test each:

| candidate | test | result |
|---|---|---|
| the merge forest's depth | the fusion order of every padded letter at every rung (2.6 (a)) | REFUTED as the shadow: pads are order 1-3 products, mostly inherited or two-fold once `q' <= F(M)`; `J_max` is 3..5 while `|A_pad|` climbs 1, 1, 1, 2, 3 |
| the record's composition | is the longest padded word the record's middle? (2.6 (b)) | yes at m31, m37 (the two rungs with `L_pad = 2`), no below; a correlation of two rungs |
| the corridor mod 35 | do the corridor carriers exclude the missing padded letters and skip words? (5.1, 2.1) | NO: the holes 41 (m29), 82 (m37) and the four m37 skip 2-words all have non-empty carriers; the corridor caps the SMALL alphabet (E1) and is transparent to skips (3.3) |
| the gear-5 lock | the per-class run law: does each tooth class saturate gear 5 (a run of three multipliers) on the longest words? (2.3) | a run of 3 appears exactly at `(43, 43)`, `(47, 47)`, `(18, 35, 18, 35)`, `(18, 35, 53)`, `(35, 18, 53)` - the words of length `>= 2` at m41..m47 - and nowhere on the ladder to m37; gear 5 is saturated by the long words but does not cap them (3.3) |

**What the growth of the PAD alphabet is the shadow of: the record in gear units,
`F(M)/q'` (letters) and `T = floor((F(M+q') - 2)/q')` (words).** Every padded letter is a legal
value in `[q', F(M)]`; the alphabet has `~3F(M)/q' - 2` entries and the machine realises them
all but the one or two nearest the record (2.1). Every padded WORD lives on two tooth APs of
step `q'` inside a span `<= F(M+q') - 2` (E2), and the count of tooth-AP openings the corridor
allows in that span is `Omega_{5,7}(T + 1)` per class. The small alphabet cannot use the span
(its union must be everything, and `2 * 15/35 < 1`); the skip letters can, and they are what
`F(M)/q' >= 1 + a/q'` makes available. So `L_pad` is capped uniformly on the half the record has
measured to m47 (E1, `<= 8`), and on the other half it is `<= 2 Omega(T + 1) - 1` - which is
bounded iff `T` is, i.e. iff `F(M + q') <= C q'` along the ladder. That is the budget
inequality's own growth rate: `F(M+q') <= F(M) + q'` summed gives `F <= sum of the gears`,
i.e. `F/q' <= pi(q') - 2`, no constant. **The shadow is the record's growth in gear units, and
a constant cap on `L` would be a statement that the record grows no faster than a constant
times the gear - strictly stronger than the budget inequality, and ROOT.**

One exact and cheap consequence that is NOT root-shaped: the corpus-wide statement
**`L(M) <= 5` for every machine with `F(M + q') <= 5 q' + 1`** (E2 with gear 5), which covers
every rung to m53 and beyond it until `F/q'` passes 5. On the corpus `F(M+q')/q'` is
`1.48, 1.48, 1.87, 2.38, 2.22, 2.40, 2.51, 2.74, 2.73` at m19..m53; if it grows like
`log q'` (the Jacobsthal scale), `T = 5` is reached only far above m53, and up to there the
budget at every rung is a statement about `D_6` at most (the order law, r66 4.3).

## 7. What is new

1. **E1, the small-alphabet cap** (3.2): every realised legal word over `{a, b, q'}` has length
   `<= CORRCAP_3(q' mod 210) <= 8`, in every machine containing 5 and 7 and at every scale;
   the exact 48-class table (values 2, 3, 4, 5, 6, 8; never 7), with gears 11 and 13 lowering
   no class. It extends docs/proofs/12 from `{a, b}` to the alphabet that carries every padded
   word on record to m47, and it closes the record's 53 -> 59 word census in one line
   (`CORRCAP_3(59) = 2`: the eight zero 3-words over `{20, 39, 59}` have empty carriers).
   Its mechanism is new: each tooth class holds at most 15/35 of the multipliers and the small
   alphabet needs the two classes to cover all of them; gear 5 alone does not cap (the
   `{0, 3, 4}` pattern), gear 7 does, and nothing above 7 is needed.
2. **The skeleton and the pullback machine** (0.1, 4.1): a realised legal word is a pair of
   multiplier sets `(S_0, S_1)` on two tooth APs of step `q'`, `|S_0| + |S_1| = L + 1`, and
   each is a set of openings of the pullback `M^{(q')}` (same gears, separations
   `2u_g q'^{-1}`); the skip letters are exactly the multiplier jumps `>= 2` (or a jump of 1
   with a class change), i.e. tooth columns `M` has blocked; the corpus words' skeletons (2.3).
3. **E2, the skeleton law** (4.1): `L + 1 <= 2 Omega_M(T + 1)`, `T = floor((F(M+q') - 2)/q')`,
   with `Omega` the pullback's opening count; the uniform table `3, 5, 5, 5, 7, 9, 11, ...` at
   `T = 1, 2, ...` against docs/proofs/11's `2T + 1`, i.e. `(6/7)T` against `2T`; the per-class
   form tight at m29 and m53; the regime statement **`L <= 5` whenever `F(M+q') <= 5q' + 1`**
   (gear 5 alone), which holds on the whole corpus with `F(M+q')/q' <= 2.73`.
4. **E3, the exact corridor-plus-span cap** (4.2): `L <= CC(q', F(M), F(M+q'))`, nine of nine
   corpus rungs, slack 0 or 1 everywhere, tight at m29 and m53 with the realised words as its
   witnesses (`(10, 21, 10)`, `(20, 98, 20)`). It combines the record's `CORRCAP` (letter cap)
   with the span cap, and it removes the record's 25s at 37 -> 41 and 41 -> 43.
5. **The split of gate item 1** (3.3, 4.3): `L_pad = max(L_small-padded, L_skip)`; the first
   half is uniformly capped (E1); the second is the record's growth in gear units (E2, ROOT).
   The first skip word on record is `(20, 98, 20)` at m53, identified as such; at m37, the
   first rung where a skip 2-word is size-feasible, none of the four candidates is realised
   although the corridor allows each (5.1).
6. **The exact tables** (2.1-2.3): the PAD alphabet with multiplicities at every rung to m37
   (`{23}, {29}, {31}, {37, 49}, {41, 55, 68}`; holes 41 and 82 with non-empty carriers);
   every realised legal word with count and skeleton to m37 (new at m31: `(25, 12, 25)` x28,
   `(25, 37)` x18; at m37: `(27, 41)` realised exactly once per period); the pad letters'
   fusion orders; `k_L in {L, L+1}` with the palindrome mechanism (2.2).
7. **Two bricks** (2.5, 6): no count of strikes caps a word (capacity/need 1.4-2.2 at every
   longest word); the merge forest's depth, the record's composition and the gear-5 lock are
   not what the PAD alphabet is the shadow of.

**Prior art, in one line each.** The AP lemma and the corridor are docs/proofs/14; the bare cap
is docs/proofs/12 (E1 is its extension by one letter, with a new mechanism); the span cap and
the pairing of alternating letters are docs/proofs/11 (E2 replaces the pairing by the
pullback count, E3 by the exact walk); `CORRCAP(q', F)` is uniform-order-bound (E3 adds the
span cap); the padded alternation `(s, q' + (q' - s), s)` is mechanic round 27's named
construct, here identified as the first skip word and placed on the skeleton; the pullback
along an AP is CRT bookkeeping and claims nothing beyond the tooth rule. No published result is
used; prior art outside the repository is not checked.

## 8. Verdict

**Node 4.i.b.ii.a: the padded word is capped on the half the record has ever measured, and
the other half is the root question in gear units.**

- E1 (proved, uniform): words over `{a, b, q'}` have `L <= CORRCAP_3(q' mod 210) <= 8`, by
  gears 5 and 7, exceptionless at 14 of 14 corpus rungs and tight at m29 and m53. Every padded
  word on record to m47 is in this class. Gate item 1's padded half, as the record measured
  it (`L_pad = 0..3`), is closed by this.
- What remains open is the skip half (letters `2q'`, `a + q'`, `b + q'`, ...), which enters
  the record at m53 with `(20, 98, 20)`. E2 and E3 cap it by `T = floor((F(M+q') - 2)/q')`
  through the corridor - `2 Omega_{5,7}(T + 1) - 1`, tight at two rungs, and the exact `CC`,
  slack 0 or 1 at nine of nine - and by nothing uniform: a constant cap on `L_skip` is
  equivalent to `F(M + q') <= C q'` along the ladder, the record growing no faster than the
  gear, which is stronger than the budget inequality. **ROOT.**
- The regime is explicit: `L <= 5` for every machine with `F(M + q') <= 5 q' + 1` (gear 5
  alone), which is the whole corpus (`F/q' <= 2.73`) and every rung until `F/q'` passes 5.
  Within it the order law makes the budget a statement about `D_6` at most.
- The shadow: the PAD alphabet is `~3F(M)/q' - 2` legal values, all realised but the one or
  two nearest the record; its growth, and the skip half's, is the record in units of the gear.
  Not the merge forest, not the record's composition, not the corridor (which caps the other
  half), not the gear-5 lock.

Scorecard: P1, P2, P3, P6, P8 CONFIRMED exactly; P4 CONFIRMED with the maximum 8 instead of 7;
P5 vacuous CONFIRMED, its factor REFUTED (1.40); P7 REFUTED at m29 and replaced by
`k_L in {L, L + 1}` with the palindrome mechanism; P9 CONFIRMED, with E2 making the exchange
rate exact.

## 9. Dead ends (bricks), each with its refuting instance

| idea | dies at | the instance | why it cannot be revived |
|---|---|---|---|
| the counting cap (strikes against span, or junction neighbours against capacity) | every rung | m13 `(11)`: need 10, capacity 14; m23 `(29)`: 28 against 43; junction ratio 3.5-15 | the machine has spare capacity at every span (gear 5 alone gives `S/5`); only a residue statement can cap a word |
| gear 5 alone as the cap on the small alphabet | the class pattern | `(s, q', q', q' - s, q', q', ...)` puts each class on `{0, 3, 4}` mod 5 for ever | the AP lemma caps runs, not density; gear 7 is needed and suffices |
| gears 11, 13 as a sharper cap on the small alphabet | all 48 classes | the maximum over the 10 (resp. 120) lifts equals the class value at every class | at the small alphabet 5 and 7 already leave `2 * 15/35 < 1`; larger gears thin a set that is already too thin |
| the corridor as a cap on skip words | 53 -> 59 (the record's `CORRCAP = inf`) and m37 | the four m37 skip 2-words have carriers of four residues; `(20, 98, 20)` has carrier `{5, 12, 25, 32}` | a skip leaves a multiplier to neither class, and the density contradiction disappears |
| `k_L = L + 1` as a law | m29 | the sole 3-word `(10, 21, 10)` is a palindrome and `(21, 10, 21)` is unrealised | `k_L = L` whenever the realised `L`-words do not chain; both values occur |
| the merge forest's depth as the shadow | m29..m37 | the pad 31 is inherited 500 times; 41 is order 1-4, 55 order 1-5 | the pad's depth is ordinary; its size is what makes it a letter |
| the letter count `|A_pad|` as a cap | its growth | `1, 1, 1, 2, 3` at m19..m37, `~3F/q' - 2` | every cap here is a function of `F/q'`, and `F/q'` grows |
| the counting relaxation E2 as uniform | `T >= 5` | `2 Omega_{5,7}(6) - 1 = 7` | `Omega(n) ~ 3n/7`; only a cap on `T = F(M+q')/q'` closes it, and that is ROOT |

## 10. The part's remaining open items, sorted

- **Closed here.** The padded half of gate item 1 for words over `{a, b, q'}`: `L <= 8`
  uniformly (E1). The 53 -> 59 word census's zero 3-words over `{20, 39, 59}` (one line). The
  four m37 skip 2-words (exact, unrealised). The PAD alphabet and every realised legal word to
  m37, with counts and skeletons.
- **Measurement with no structural content.** The positions of the pad letters (start residues
  inside the carriers; every gear a sole striker at m19, m23; 8-10 of the gears at m31, m37);
  the fusion orders of the pad letters; the record's middles being padded at m31 and m37.
- **Root question in disguise.** A constant cap on `L_skip`, equivalently on
  `T = F(M+q')/q'`: the record growing no faster than a constant times the gear. E2 gives the
  exchange rate (`6/7` letter per unit of `T` by gears 5, 7; at most one unit better with 11,
  13 to `T = 20`), E3 the exact cap at each rung.
- **Genuinely open on the part alone, with the attack.** (i) The cover half of the skip words
  at m41, m43, m47: are `(29, 57)`, `(14, 72)` at m41, `(31, 63)`, `(16, 78)` at m43 and
  `(35, 71, 35)` at m47 realised? Size permits all of them; the corridor permits all of them
  (carriers non-empty); only a copy-law scan or a CRT search decides, and the answer says at
  which rung the skip half begins (m53 is only an upper bound). (ii) The exact `Omega_M` with
  all gears (not only 5, 7, 11, 13) for the pullback separations `2u_g q'^{-1}`: a covering
  computation of file 20/21's kind on a two-tooth machine with non-real separations, which
  would sharpen E2 for `T >= 6` and say whether the arc floor (docs/proofs/21 Theorem 3, a
  real-teeth fact) has a pullback analogue. (iii) The class-1 range refinement of E2
  (`m <= floor((F(M+q') - 2 - s)/q')` for class 1, docs/proofs/11's PARITY step), a unit at
  most. (iv) Formalisation: E1 is `BareAlt` of docs/proofs/12 with one more letter and the same
  48-class `decide`; E2's inequality is the tooth rule along an AP plus a counting lemma about
  APs in `E_35`, both within the kernel's reach; E3 is a finite DP per rung and should be left
  as a certificate.
