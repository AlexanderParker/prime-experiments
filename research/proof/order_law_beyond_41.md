# The order law out of sample beyond 41: 41 -> 43, 43 -> 47, 47 -> 53, 53 -> 59

Child of node **4.i.b.ii** (`research/proof/monotone_functional.md` 4.3, the order law) and of
`research/proof/order_law_37_41.md`, which decided the rung `37 -> 41` with the covering-problem
instrument and closed with "the functional `Phi = B_{L+1}` is budget-monotone at every rung of the
ladder the project has". *The ladder the project has* stopped at 41. This document takes the
instrument to the four rungs above it.

What spawned it: 2.2 of `order_law_37_41.md` -- membership of a local pattern in `D_k(M)` is one
covering problem over the gears, decided with no scan and no dictionary. If that is true it is
true at every engine, and the closure ladder (which stalls at depth 2 already at m37) is not the
limit of the order law. The rungs `41 -> 43 .. 53 -> 59` are unreachable by any ladder the project
has; they are reachable by the instrument.

Scripts in `research/anchor235/r71/` (prefix `ol2_`); outputs in `research/anchor235/r71/results/`
(untracked). Every number this document relies on is written into the document.

**Result, up front: the order law is FALSE.** It holds at `41 -> 43` (`B_3 = 118 <= 134`) and at
`47 -> 53` (`B_5 = 145 <= 171`), and fails at `43 -> 47`, where `B_{L+1} = B_3 = 153` against a
budget of `F(m43) + 47 = 150`. The value is exact, its binding word is `(45, 16, 47, 45)` at phase
2 of 47, and the two triples that carry it occur at columns 1,669,802,076,752,677 and
1,103,716,997,185,117 of m43. The budget inequality itself is untouched at that rung
(`F(m47) = 118 <= 150`); what fails is the relaxation used to prove it.

---

## 1. Pre-registered

Written before any computation at m41, m43, m47 or m53. The only thing computed before this
section was fixed is the validation of the instrument at m37 (gate G0 below), whose target numbers
were already published in `order_law_37_41.md`.

### 1.1 The law under test

    k*(M; q')  =  L(M) + 1  =  J_max - 1,

where `B_k(M; q')` is the widest span of a level-`k` admissible word that fuses at some phase of
`q'` (level-`k` admissible: every `k` consecutive entries lie in `D_k(M)`), `L(M)` is the longest
realised legal letter word with respect to `q'`, `J_max = L + 2`, and `k*` is the least `k` with
`B_k <= F(M) + q'`.

### 1.2 The predictions, each with the number that would refute it

- **P1 (the law's first half).** `B_{L+1}(M; q') <= F(M) + q'` at every step attempted:
  `41 -> 43` (budget `91 + 43 = 134`), `43 -> 47` (`103 + 47 = 150`), `47 -> 53`
  (`118 + 53 = 171`), `53 -> 59` (`145 + 59 = 204`). **REFUTED by one step with
  `B_{L+1} > F(M) + q'`** -- and a refutation here is a refutation of the order law itself, not of
  this document.
- **P2 (the law's second half).** `B_L(M; q') > F(M) + q'` at every step, so the order `L` is
  never enough. REFUTED by one step with `B_L <= F(M) + q'` (which would put `k* < L + 1`).
- **P3.** Hence `k* = L + 1` at all four steps.
- **P4 (the mechanism of 4.4).** The binding term -- the `J` attaining `B_{L+1}`, and the `J`
  attaining `B_L` -- is the deepest fusion `J = J_max` at every step.
- **P5 (the instrument's gate at each new engine).** The exact row `Q*_J(M; q')`, `J = 1..J_max`,
  computed from the instrument alone, has `max_J Q*_J = F(M + q')` equal to the certified record:
  103, 118, 145, 161 at the four steps. REFUTED by one disagreement, which would condemn the
  instrument rather than the law.
- **P6 (strictness).** `B_{L+1} >= F(M + q')` holds by construction; predicted STRICT at every
  step (the relaxation is larger than the machine), as at 5 of the 7 earlier decisive rungs and at
  `37 -> 41` (98 against 91).
- **P7 (`L`).** `L(m41) = 2` is on the corpus record (`monotone_functional.md` M5); the
  instrument must reproduce it. `L(m43)`, `L(m47)`, `L(m53)` are not on record anywhere; predicted
  small (`<= 3`) and NOT monotone, as `L` has been at every earlier rung
  (1, 1, 1, 2, 1, 3, 3, 2, 2 at m11..m41).

### 1.3 The certified inputs, quoted and never guessed

The record table, from `docs/proof-search/agents-shared.md` (line 40) and
`docs/proof-search/mechanic.md` 3060 -- the corpus ladder is complete to `y = 53`, and
`F(59) = 161` was computed on machine 23's period:

| `y` | 5 | 7 | 11 | 13 | 17 | 19 | 23 | 29 | 31 | 37 | 41 | 43 | 47 | 53 | 59 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `F(y)` | 2 | 5 | 7 | 11 | 18 | 25 | 34 | 43 | 58 | 88 | **91** | **103** | **118** | **145** | **161** |

Every value used below is exact; none is a bound. The two caps the search ranges need are
`F(M)` itself (a realised gap value is at most the record) and `F_2(M) <= F(M + q')` (the
deletion-ladder cap: at some phase of `q'` the single interior opening of a 2-window is deleted,
so the 2-window's span is an opening-free stretch of `M + q'`).

### 1.4 Scorecard

| # | prediction | verdict | evidence |
|---|---|---|---|
| P1 | `B_{L+1} <= F + q'` at all four steps | **REFUTED at `43 -> 47`** | `B_3(m43; 47) = 153 > 150`, exact, 4.4 |
| P2 | `B_L > F + q'` at all four steps | HOLDS at 3, OPEN at `53 -> 59` | 176 > 134, 213 > 150, 198 > 171; `B_3(m53) >= 203` against 204, 4.6 |
| P3 | `k* = L + 1` | **REFUTED at `43 -> 47`** (`k* = L + 2 = J_max = 4`); holds at `41 -> 43`, `47 -> 53` | 4.1, 4.4, 4.5 |
| P4 | binding term is `J = J_max` | **REFUTED at `47 -> 53`**: `B_5`'s maximum is the RECORD term at `J = 4 < J_max = 6` | 4.5 |
| P5 | `max_J Q*_J = F(M + q')`, certified | PASSED at `41 -> 43` (full row `91, 103, 103, 100`); at `43 -> 47` and `47 -> 53` passed in the exhibiting direction only (a realised fusion of span exactly the record, with its column) | 4.1, 4.4, 4.5 |
| P6 | `B_{L+1} > F(M + q')` strictly | **REFUTED at `47 -> 53`**: `B_5 = 145 = F(m53)` exactly | 4.5 |
| P7 | `L(m41) = 2` reproduced; `L` small and non-monotone above | HOLDS | `L = 2, 2, 4, 3` at m41, m43, m47, m53; 4.2 |

---

## 2. Setup: the instrument, and what it replaces

### 2.1 The one idea

`order_law_37_41.md` 2.2 decides membership of a local pattern in `D_k(M)` by a covering problem
over the gears' free phases: gear `g` strikes exactly two residue classes of a window, `{t_g,
t_g + d_g}` with `d_g = 2u_g mod g` fixed and `t_g` free and (by CRT) independent across gears, so

> a pattern -- offsets `OPEN` required open, the complementary offsets of `[0, S]` required struck
> -- is realised in `M` iff there is a choice of `t_g in Z_g`, one per gear, with (i) no gear
> striking any offset of `OPEN` and (ii) every offset outside `OPEN` struck by some gear.

That statement has no reference to the machine's size. It is therefore an instrument for **every**
engine, not only for m37, and this branch is the test of that: the four engines below have periods
`5.3 x 10^13`, `2.3 x 10^15`, `1.1 x 10^17`, `5.7 x 10^18` columns and no ladder reaches any of
them. The solver is `r70/ol_pattern.py`, imported unchanged (`realised_search`, the exact-cover
search, with `realised`, the meet-in-the-middle reference, as the fallback).

### 2.2 What is built, and what is not

For each engine `M` the branch builds only the depth-2 dictionary, and builds it from the
instrument rather than from a closure:

    D_1(M) = { v <= F(M)         : the 1-window of span v is realisable }        (F(M) covering problems)
    D_2(M) = { (a, b), a+b <= C  : the 2-window is realisable },  C = F(M + q')  (~|D_1|^2/2 problems)

using the mirror `k -> -k` of the strike set (a window is realised iff its reversal is) to halve
the second list. Everything above depth 2 -- the `k`-windows the relaxation quantifies over, the
legal letter words that fix `L(M)`, the exact `J`-fusions that reproduce the next record -- is
decided window by window, on demand, and never tabulated.

### 2.3 Gate G0: the instrument reproduces the m37 dictionary with no ladder

Before any new engine, the method was run at the one engine where the answer is published.
`order_law_37_41.md` 3 has `D_1(m37)` = 75 values (1..72 complete, then 77, 85, 88) and
`|D_2(m37)| = 2,053` rows carrying all 217,929,355,875 gaps, both produced by the closure ladder
`T_29, T_31, T_37` from `D_15(m23)` (623 s, under a gigabyte).

`ol2_dict.py` at `y = 37`, from the covering instrument alone, with no ladder and no scan:

| | ladder (`order_law_37_41.md` 3) | instrument (`ol2_dict.py 37`) | |
|---|---|---|---|
| `|D_1(m37)|` | 75 | **75** | identical |
| absent values `<= 88` | 73-76, 78-84, 86, 87 | **73, 74, 75, 76, 78, 79, 80, 81, 82, 83, 84, 86, 87** | identical |
| `max D_1` | 88 = `F(37)` | **88** | the certified record |
| `|D_2(m37)|` | 2,053 | **2,053** | identical |
| widest 2-window | 90 = `F_2(m37)` | **90** | the corpus `F_2` |
| mirror-asymmetric rows of `D_2` | -- | **0 of 2,053** | the `k -> -k` symmetry, verified not assumed |

88 covering problems for `D_1` (124 s) and 3,799 for `D_2` (479 s), 8 processes, no fallback to
the reference solver. **The depth-2 dictionary of m37 is a property of the gears, not of the
ladder** -- and the ladder is not needed at any engine.

### 2.4 Gate G0b: the whole rung `37 -> 41` recomputed, twice

The step script (`ol2_step.py`) was run at `y = 37` against the published rung, once before and
once after the cost changes of 2.6, and both runs reproduce `order_law_37_41.md` entry for entry:

| quantity | `order_law_37_41.md` | `ol2_step.py 37` | |
|---|---|---|---|
| `L(m37)` | 2 | **2**, witness `(14, 41)` | `J_max = 4` |
| `B_2 = B_L` | 161, word `(35, 41, 27, 58)` at phase 20 | **161**, `(35, 41, 27, 58)` at phase **20** | identical |
| `B_3 = B_{L+1}` | 98, word `(21, 14, 41, 22)` at phase 20 | **98**, the mirror `(22, 41, 14, 21)` at phase **19** | identical |
| level-2 `J = 3` term | 143, `(58, 27, 58)` at phase 38 | **143**, `(58, 27, 58)` at phase **38** | identical |
| `Q*_J`, `J = 1..4` | 88, 90, 90, 91 | **88, 90, 90, 91** | identical |
| `Q*_3` witness | `(28, 14, 48)` at phase 13 | `(28, 14, 48)` at phase **13** | identical |
| `Q*_4` witness | `(21, 14, 41, 15)` at phase 20 | `(21, 14, 41, 15)` at phase **20** | identical |
| binding word realised? | NO | **NO** | identical |
| `k*` | 3 | **3** | identical |

523 s for the first run, **217 s** for the second. The pipeline is therefore gated end to end
before it is pointed at an engine with no published answer.

### 2.5 The four rungs, in the gear's own arithmetic

`u = 6^{-1} mod q'`, `d = 2u mod q'`, letters `PAD` (`= 0`), `UP` (`= +d`), `DOWN` (`= -d`),
letter floor `a_L = min(d, q' - d)`; the letters listed are the values at most `F(M)`, i.e. the
only ones that can be an interior gap of a fusion.

| rung | `q'` | `u` | `d` | `a_L` | `F(M)` | budget `F + q'` | certified `F(M + q')` | letters `<= F(M)` |
|---|---|---|---|---|---|---|---|---|
| `37 -> 41` | 41 | 7 | 14 | 14 | 88 | 129 | 91 | UP 14, 55; DOWN 27, 68; PAD 41, 82 |
| `41 -> 43` | 43 | 36 | 29 | 14 | 91 | **134** | 103 | DOWN 14, 57; UP 29, 72; PAD 43, 86 |
| `43 -> 47` | 47 | 8 | 16 | 16 | 103 | **150** | 118 | UP 16, 63; DOWN 31, 78; PAD 47, 94 |
| `47 -> 53` | 53 | 9 | 18 | 18 | 118 | **171** | 145 | UP 18, 71; DOWN 35, 88; PAD 53, 106 |
| `53 -> 59` | 59 | 10 | 20 | 20 | 145 | **204** | 161 | UP 20, 79, 138; DOWN 39, 98; PAD 59, 118 |

`3 a_L = q' -+ 1` at every one (`3 x 14 = 43 - 1`, `3 x 16 = 47 + 1`, `3 x 18 = 53 + 1`,
`3 x 20 = 59 + 1`), the real-teeth identity of `alignment-rules.md`.

### 2.6 Where the instrument's cost lives, measured

This is a finding about the instrument, not a note about implementation, and it decided the shape
of every run below. A YES verdict is a witness and is found in milliseconds. A NO verdict is an
exhaustive refutation, and **its cost is governed by how many offsets the pattern holds OPEN**,
because each open offset deletes phases from every gear and so constrains the search:

| pattern | open offsets | measured cost of a NO at m41 (one core) |
|---|---|---|
| 1-window, span near `F(M)` | 2 | the three absent values 84, 87, 89 dominated a 215 s run over all 91 |
| 2-window, span near `F_2(M)` | 3 | **30 - 69 s each** -- `(91,12) 55s`, `(51,52) 69s`, `(45,58) 46s`, `(60,43) 40s`, `(88,15) 31s`, `(70,33) 53s`, all NOT realised |
| 3-window, span near `F_3(M)` | 4 | **0 - 13 s each** (median about 8) |
| 4-window | 5 | faster again |

So the expensive object is the WIDE, THIN pattern -- exactly the Jacobsthal question itself -- and
the cheap object is the deep fusion the order law is about. Three consequences, all used below:

1. **`D_2(M)` is not built above m37.** Tabulating it costs thousands of the 30-70 s refutations
   (the m41 attempt was abandoned after it became clear it was hours). It is not needed: level-`k`
   admissibility for `k >= 2` is tested on the `k`-windows themselves, so replacing `D_2` by the
   superset `{(a,b) : a + b <= F_2 cap}` only adds candidate words, which the exact `k`-window test
   then rejects. **Every `B_k` with `k >= 3` below is exact.**
2. **The certified caps do the rejecting for free.** `F_j(M) <= F(M + the next j-1 primes)` is the
   deletion ladder (docs/proofs/07, proved), so a `j`-window wider than that cap is not realised
   and needs no solver call at all: `F_2(m41) <= 103`, `F_3(m41) <= 118`, `F_4(m41) <= 145`,
   `F_5(m41) <= 161`, and the corresponding rows at m43, m47, m53.
3. **`Q*_1` is never asked.** `F(M)` is a realised gap (the certified record) and no gap is wider,
   so `Q*_1 = F(M)` as soon as `F(M)` fuses at some phase -- an arithmetic check, not a covering
   problem. The one place a wide thin pattern is still unavoidable is `B_L` at `L = 2`, and 4.4
   says exactly what was and was not obtained there.

### 2.7 How each quantity is obtained (so the reader can check the logic, not only the numbers)

- **`L(M)`.** Enumerate the legal letter words (letters of `q'` at most `F(M)`; legality =
  the struck class starts at `0` or `d`, a PAD keeps it, an UP takes `0 -> d`, a DOWN takes
  `d -> 0`, so no two equal nonzero letters are adjacent and pads are transparent), by length,
  with adjacent pairs filtered through the span cap. At each length only EXISTENCE matters, so a
  cheap witness settles it; the full solver is called only at the length where the answer is
  "none" -- which is the length that fixes `L`, and there an exhaustive refutation of every
  candidate is genuinely required and is done. `J_max = L + 2` (docs/proofs/10).
- **`B_k`.** For each `J <= J_max`, enumerate every `J`-word over the value alphabet whose
  adjacent pairs satisfy the certified `F_2` cap and which FUSES at some phase `z` of `q'`
  (offsets `o_1 .. o_{J-1}` struck, `o_0` and `o_J` unstruck -- brute force over all `q'` phases,
  no word theory). Sort by descending span and test each word's `k`-subwindows on the instrument;
  the first word all of whose `k`-subwindows are realised is the `J`-term, because the scan is
  strictly descending. `B_k = max_J` of the terms.
- **`F(M + q')`.** The same scan with `k = J_max`, where level-`J_max` admissible means the word
  itself is realised: `max_J Q*_J = F(M + q')` is the record law (docs/proofs/09). This is the
  gate: the answer must be the certified record, and the instrument has no way of knowing it.

---

## 3. Gates at the new engines

Each new engine gets an independent exact check that shares nothing with the instrument: a bounded
column range of the machine `{5..y}` is sieved directly in the anchored column coordinate, and
every window that OCCURS in that range is put to the instrument. A single NO is a false negative
and condemns the instrument. Every question the gate asks has the answer YES, which is the cheap
direction, so the gate is affordable at every engine.

`ol2_gate.py`, 150,000,000 columns from column 0 at each engine, every distinct 1-, 2- and
3-window that occurs put to the instrument, plus 6,000 of the 4-windows drawn at random:

| engine | openings in the scan | density | widest gap seen | `D_1` | `D_2` | `D_3` | `D_4` (of) | windows checked | **disagreements** | secs |
|---|---|---|---|---|---|---|---|---|---|---|
| m41 | 25,141,514 | 0.167610 | 63 | 57 | 1,168 | 12,915 | 6,000 (88,147) | **20,140** | **0** | 203 |
| m43 | 23,972,117 | 0.159814 | 65 | 62 | 1,334 | 15,204 | 6,000 (106,982) | **22,600** | **0** | 211 |
| m47 | 22,951,996 | 0.153013 | 65 | 64 | 1,467 | 17,443 | 6,000 (126,108) | **24,974** | **0** | 203 |
| m53 | 22,085,919 | 0.147239 | 72 | 67 | 1,605 | 19,818 | 6,000 (147,484) | **27,490** | **0** | 199 |

**95,204 windows checked across the four new engines, 0 disagreements.** Every column of the
`D_1`, `D_2`, `D_3` columns is COMPLETE for the range scanned -- not a sample -- so a false NO on
any window that occurs in the first 1.5 x 10^8 columns of any of the four machines would have been
caught. Together with G0 (the instrument reproducing the m37 ladder dictionary row for row, 2.3)
and the record gates of section 4 (`F(M + q')` recomputed and equal to the certified record at
every completed rung), the instrument is checked from three directions at each engine.

A cross-lane check as well: node 4.i.b.ii.c (`research/proof/fusion_lemma.md`, r73) independently
put the legal letter words of m41 to the same characterisation and found `L(m41) = 2` with exactly
**five** maximal words, `(14,43), (43,14), (29,43), (43,29), (43,43)`. This branch's `L` search at
m41 found 5 of the 15 legal length-2 words realised and 0 of the 41 length-3 words -- the same
count, the same `L`, from a separately driven search.

---

## 4. Results

### 4.1 `41 -> 43`: the law holds, `k* = 3`, margin 16

`F(m41) = 91`, `q' = 43`, budget **134**. `D_1(m41)` was tabulated by `ol2_dict.py` (91 covering
problems, 215 s): **88 realised gap values, max 91 = `F(41)`** -- the certified record reproduced
by the instrument -- with exactly three absent values below it, **84, 87 and 89**.

**`L(m41) = 2`**, the corpus value (`monotone_functional.md` M5) reproduced, so `J_max = 4` and the
law predicts `k* = 3`. The witness is `(14, 43)` = DOWN, PAD. Of the 15 legal letter words of
length 2, **5 are realised**; of the 41 of length 3, **0** (every one refuted, 3 of them needing
the full solver).

| level `k` | `J = 1` | `J = 2` | `J = 3` | `J = 4` | `B_k` | budget 134 |
|---|---|---|---|---|---|---|
| 2 `= L` | 91 | 103 | 156 | **176** | **>= 176** | **above by 42** |
| 3 `= L+1` | 91 | 103 | 103 | **118** | **118** | **inside by 16** |

- `B_2 >= 176`, attained by `(45, 43, 43, 45)` at phase 27 -- a certified level-2 admissible fusing
  4-word, so `B_2 > 134` and `k* >= 3` is PROVED. The exact value of `B_2` was not determined:
  1,973 wider words have a pair whose refutation costs more than the soft budget, and the level-2
  superset bound is 206, so `176 <= B_2 <= 206`. This is the wide-thin cost of 2.6 and it does not
  touch the law, which needs only `B_2 > budget`.
- **`B_3 = 118` exactly**, and `118 <= 134`: **the law's first half holds with margin 16.**
  Everything in the `k = 3` row is exact -- 38,758 candidate 4-words scanned in descending span,
  1,055 covering problems, every deferred window resolved with the full solver, 0 undecided.

**The binding word.**

    (28, 14, 43, 33) at phase z = 1,  span 118,  letters  BAD, DOWN, PAD, BAD
    and its mirror (33, 43, 14, 28)

`J = 4 = J_max`, the deepest fusion, as at 8 of 8 earlier rungs. Its two triples `(28, 14, 43)` of
span 85 and `(14, 43, 33)` of span 90 are both realised; **the 4-word itself is NOT realised** --
again two windows the machine contains, overlapping in a way it never realises.

**The exact row and the record gate.**

    Q*_J(m41; 43) = 91, 103, 103, 100   at J = 1, 2, 3, 4
    max_J Q*_J = 103 = F(43), the certified record.  GATE PASSED.

with witnesses `(91)` at phase 1, `(28, 75)` at phase 1, `(26, 14, 63)` at phase 3, and
`(28, 43, 14, 15)` at phase 1 (ties `(23, 14, 43, 20)`, `(20, 43, 14, 23)`, `(15, 14, 43, 28)`).

Two things in that row are worth naming.

1. **`Q*_2 = 103 = F_2(m41)`.** The corpus has `F_2(41) = 103` exactly
   (`alignment-rules.md` 3.7, the m41 row). The instrument attains it, at `(28, 75)`, and the
   attaining 2-window also fuses -- so a second independent corpus number is reproduced on the way.
2. **The record is NOT attained at `J_max` here.** `Q*_4 = 100` while `Q*_2 = Q*_3 = 103`: the new
   record of m43 is a two- and three-piece fusion, not the fourfold one. This is the first rung on
   the ladder where the record's argmax and the relaxation's argmax come apart -- `B_3`'s maximum
   is at `J = 4`, the record's is at `J = 2, 3`. The relaxation binds where the machine does not.

**Strictness.** `B_3 - F(43) = 118 - 103 = 15 > 0`, so the relaxation is strictly larger than the
machine, as at `37 -> 41` (7) and at 5 of the 7 earlier decisive rungs.

Cost: 11,204 s on 4 processes.

**Agreement with the other lane.** Node 4.i.b.ii.c (`research/proof/fusion_lemma.md`, addendum,
r73) settled the upper half of this same rung independently and reports
`R(m41; 43) = 118 = B_3 <= 134`, `L(m41) = 2` with **five** maximal words, and `P(43, 43) = 5`.
This branch has `B_3(m41; 43) = 118` exactly against the same budget 134 with the same margin 16,
`L(m41) = 2`, and 5 of the 15 legal length-2 words realised. **Every number agrees**, from two
separately driven searches -- the other lane through the fusion lemma's characterisation, this one
through the covering instrument's descending scan. The one thing this branch adds at the rung is
the binding word `(28, 14, 43, 33)` with its phase and its non-realisation.

### 4.2 The merge depths, and where they come from

`L(M)` was not on record above m41. All four are computed here, each with its witness and each
with the complete refutation at length `L + 1`:

| engine | `q'` | bare letters `{a, b}` | `L(M)` | witness | length-`(L+1)` candidates, all refuted | `J_max` | law's `k*` | measured `k*` |
|---|---|---|---|---|---|---|---|---|
| m37 | 41 | 14, 27 | 2 | `(14, 41)` UP·PAD | 41 | 4 | 3 | 3 |
| m41 | 43 | 14, 29 | **2** | `(14, 43)` DOWN·PAD | 41 | 4 | 3 | 3 |
| m43 | 47 | 16, 31 | **2** | `(16, 47)` UP·PAD | 41 | 4 | 3 | **4** |
| m47 | 53 | 18, 35 | **4** | `(18, 35, 18, 35)` UP·DOWN·UP·DOWN | 247 | **6** | **5** | **5** |
| m53 | 59 | 20, 39 | **3** | `(20, 98, 20)` UP·DOWN·UP | 169 | **5** | **4** | open (4.6) |

`L(m41) = 2` is the corpus value reproduced. `L(m43) = 2`, **`L(m47) = 4`** and `L(m53) = 3` are new.
The counts behind each row, `candidates / realised` by length: m43 `6/1, 15/4, 41/0`; m47
`6/1, 15/9, 41/8, 97/2, 247/0`; m53 `7/1, 19/7, 59/1, 169/0`. Every `0` is a complete refutation
with the full solver, not a failed search.

**`L(m47) = 4` has a mechanism, and it is gears 5 and 7.** The witness carries no pad at all: it is
the bare alternation `a, b, a, b` with `a = 2u' = 18`, `b = q' - a = 35`. The bare-word cap
(docs/proofs/12, kernel-checked) says `L_bare(M) <= PSORD(q' mod 210)`, and `PSORD` takes only the
values 1, 2, 3, 5 over the 48 classes -- **`53` is one of the six classes with `PSORD = 5`**
(the others are 37, 83, 127, 157, 173), while `41, 43, 47` all sit in the `PSORD = 1` row. So at
`47 -> 53` gears 5 and 7 permit a bare run four long and the machine takes it, and at the three
rungs either side they permit one, so the depth-2 words there must spend a PAD -- which is exactly
what `(14, 41)`, `(14, 43)`, `(16, 47)` do. The jump in `L` is not a property of m47; it is a
property of `53 mod 210`, and the cap predicted where it could happen before it was measured.

This also makes `47 -> 53` the sharpest test on the ladder: `J_max = 6`, so the law predicts
`k* = 5`, an order of interaction never yet needed (the previous maximum is `k* = 4` at
`29 -> 31` and `31 -> 37`).

**`L(m53) = 3`**, witness `(20, 98, 20)`, and the length-4 refutation is complete: 169 legal
candidates, 0 realised. `PSORD(59) = 2`, so `L_bare(m53) <= 2`; the witness is `UP·DOWN·UP` with
`98 = 59 + 39` -- a DOWN carrying one extra lap of `q'`, which is how a word longer than the bare
cap is bought without a PAD. Exactly one of the 59 legal length-3 words is realised.

### 4.3 An identity that removes most of the cost, and is worth stating on its own

While running `41 -> 43` it became clear that the expensive half of every `B_k` was being computed
for nothing. The terms of `B_k = max_{J <= J_max}` split at `J = k`:

> **`B_k(M; q') = max( F(M + q'), the relaxed terms J = k+1 .. J_max )`, exactly.**
>
> *Proof.* For `J <= k` a level-`k` admissible `J`-word IS a realised window, so that term is the
> exact `Q*_J`, and `Q*_J <= max_J Q*_J = F(M + q')` by the record law (docs/proofs/09). And
> `B_k >= F(M + q')` for every `k`, because the true extremal fusion is realised and hence
> level-`k` admissible. So the whole block of exact terms is bounded by, and dominated by, the
> single number `F(M + q')`. []

Nothing is lost and nothing is assumed beyond the certified record: the exact terms need never be
computed. They are also exactly the terms that ask the instrument the expensive wide-thin
questions (`J = 1` is a 1-window, `J = 2` a 2-window), so this identity is where the cost of a
rung goes from hours to minutes. It was confirmed against the published rung: with the exact terms
dropped, `ol2_step.py 37` still returns `B_2 = 161`, `B_3 = 98`, the same binding words and the
same `Q*_J` row, in **170 s** instead of 523.

A second free tool, used from `47 -> 53` on, where the record table runs out (`F(61)` is not on
record, so the deletion ladder caps m47 only to `j = 3` and m53 only to `j = 2`):

> **Subadditivity of the window records.** If `(g_1, ..., g_j)` is realised then so are
> `(g_1..g_a)` and `(g_{a+1}..g_j)`, hence **`F_j(M) <= F_a(M) + F_b(M)` for every `a + b = j`.**

Closing the certified caps under it gives, per engine, a cap at every depth:

| engine | `F_1` | `F_2` | `F_3` | `F_4` | `F_5` | `F_6` |
|---|---|---|---|---|---|---|
| m41 | 91 | 103 | 118 | 145 | 161 | 236 |
| m43 | 103 | 118 | 145 | 161 | 263 | 279 |
| m47 | 118 | 134 | 161 | **268** | **295** | **322** |
| m53 | 145 | 159 | **304** | **318** | **463** | **477** |

(bold = from subadditivity, the rest from the deletion ladder). A candidate window wider than its
cap is refuted with no solver call at all.

### 4.3b The certificates: every YES in 4.4 and 4.5 is an explicit column of the machine

Before the two new rungs are read, the direction that carries them is made independent of the
instrument. A lower bound `B_k >= S` rests on YES verdicts (the binding word's `k`-subwindows are
realised); the refutation at `43 -> 47` rests on exactly two of them. `ol2_verify.py` re-runs the
covering search so that it returns the PHASE `t_g` of every gear instead of a boolean, then uses
`t_g = (-u_g - x) mod g` and CRT to turn the phase vector into an explicit COLUMN `x` of the
machine, and verifies the window there directly against the strike rule (column `c` is struck by
gear `g` iff `c = +-u_g mod g`). No covering argument, no dictionary, no instrument: an integer.

`ol2_certgate.py` gates the certifier against a direct sieve of the whole period at the engines
small enough to hold one, in BOTH directions -- every window that occurs must certify at a column
that really carries it, and every window that does not occur must be refuted:

| engine | period | open columns | 1-, 2-, 3-windows certified | absent 1- and 2-windows refuted | disagreements |
|---|---|---|---|---|---|
| m11 | 385 | 135 | 7 + 20 + 24 | 2 + 16 | **0** |
| m13 | 5,005 | 1,485 | 10 + 45 + 89 | 3 + 33 | **0** |
| m17 | 85,085 | 22,275 | 17 + 113 + 323 | 3 + 77 | **0** |

### 4.4 `43 -> 47`: the law's first half FAILS, `B_3 = 153` against a budget of 150

`F(m43) = 103`, `q' = 47`, budget **150**, certified `F(m47) = 118`. `D_1(m43)` was NOT tabulated
(2.6.1): the superset `1 .. 103` is used, and the level-2 filter is the superset of **6,693** pairs
of span at most `F_2(m43) <= F(47) = 118` (the deletion-ladder cap; `F_2(43)` is not on the corpus
record exactly). Every `k`-window with `k >= 3` below is decided exactly by the instrument, so a
superset here only adds candidate words for the exact test to reject.

**`L(m43) = 2`**, witness `(16, 47)` = UP·PAD; of the 41 legal length-3 words **0** are realised
(14 of them needing the full solver; 483 s). So `J_max = 4` and the law predicts `k* = 3`. The
fusing-word counts: 103 at `J = 1`, 6,498 at `J = 2` (spans to 118), 26,460 at `J = 3` (to 220),
74,873 at `J = 4` (to 236).

| level `k` | `J = 1` | `J = 2` | `J = 3` | `J = 4` | `B_k` | budget 150 |
|---|---|---|---|---|---|---|
| 2 `= L` | *subsumed* | *subsumed* | 175 | **213** | **>= 213** | **above by 63** |
| 3 `= L+1` | *subsumed* | *subsumed* | *subsumed* | **153** | **153** | **ABOVE by 3** |

- `B_2 >= 213`, attained by `(56, 47, 31, 79)` at phase 7 -- letters BAD·PAD·DOWN·BAD, `J = 4`.
  Its three 2-subwindows are realised at the columns `(56,47)` **1,393,938,514,172,582**,
  `(47,31)` **1,164,971,402,129,285**, `(31,79)` **1,263,370,711,272,502** of m43, each verified
  directly; the 4-word itself is NOT realised. So `B_2 > 150` and `k* >= 3` is PROVED. (The `J = 3`
  term is `>= 175` at `(65, 31, 79)`, phase 45.) The exact value of `B_2` was not determined and is
  not needed (8).
- **`B_3 = 153` exactly, and `153 > 150`: the law's first half is REFUTED, margin -3.**

**The binding word.**

    (45, 16, 47, 45) at phase z = 2,   span 153,   letters  BAD, UP, PAD, BAD
    tie: (45, 47, 16, 45)

`J = 4 = J_max`. Its two triples are realised at explicit columns of m43 --

    (45, 16, 47)  span 108  at column  1,669,802,076,752,677
    (16, 47, 45)  span 108  at column  1,103,716,997,185,117

-- and **the 4-word itself is NOT realised**: the same figure as at every rung below, two windows
the machine contains overlapping in a way it never realises. It fuses at phase 2 of 47: the three
interior offsets 47, 63, 110 lie in the struck classes (`0, 16, 16` mod 47) and the two flanks 2
and 155 lie in neither (`2` and `14` mod 47).

**`B_3 = 153` is exact, not a lower bound.** The descending scan resolved every deferral with the
full solver (42,179 words scanned, 1,680 solver calls, 8.5 x 10^8 search nodes, 219 windows sent to
the full solver, 12,873 s on 3 processes), and the audit of `ol2_verify.py` re-enumerated the
candidate space independently: of the 74,873 fusing 4-words, **41,405 have span above 153, and
0 of them are level-3 admissible, with 0 undecided**. Nothing wider than 153 qualifies.

**`k* = 4`, not 3.** `J_max = 4`, so at `k = 4` every term has `J <= k` and level-4 admissible
means realised: `B_4 = max_J Q*_J = F(m47) = 118 <= 150`. The first level at which the relaxation
comes inside the budget is therefore `L + 2 = J_max`, not `L + 1`.

**The record gate.** The exhibiting direction passed: a realised fusion of span exactly
`F(m47) = 118`,

    (2, 31, 85) at phase 14,  J = 3,  realised at column 2,161,962,392,309,550 of m43

(2 of the 351 fusing words of span 118 are realised). With `max_J Q*_J <= F(M + q')` from the
record law (docs/proofs/09) this pins `max_J Q*_J = 118` = the certified record. The FULL `Q*_J`
row was not computed: the exact `J = 2` term is a wide-thin scan at span 118 = the `F_2` cap, and
the run was still in it after six hours (8).

**The overshoot, which is what actually failed.** The budget inequality itself is untouched here:
`F(m47) = 118 <= 150` with 32 to spare. What failed is the relaxation:

    overshoot  B_3 - F(m47)          = 153 - 118 = 35
    slack      (F(m43) + 47) - F(m47) = 150 - 118 = 32          35 > 32.

Cost of the rung: the `B_2` and `B_3` scans, 13,829 s on 3 processes.

### 4.5 `47 -> 53`: the law holds, `k* = 5`, margin 26, and the relaxation is EXACTLY the machine

`F(m47) = 118`, `q' = 53`, budget **171**, certified `F(m53) = 145`. `D_1` superset `1 .. 118`;
level-2 filter the superset of **8,671** pairs of span at most `F_2(m47) = 134` (exact, corpus).

**`L(m47) = 4`**, witness `(18, 35, 18, 35)` -- the bare alternation, no pad -- with the complete
refutation at length 5 (247 candidates, 0 realised). `J_max = 6`, and the law predicts `k* = 5`,
an order of interaction never needed before. Fusing words: 118, 8,448, 34,684, 98,273, 249,595,
**623,871** at `J = 1 .. 6` (spans to 392).

| level `k` | `J = 1..4` | `J = 5` | `J = 6` | `B_k` | budget 171 |
|---|---|---|---|---|---|
| 4 `= L` | *subsumed* | 175 | **198** | **>= 198** | **above by 27** |
| 5 `= L+1` | *subsumed* | *subsumed* | 138 | **145** = `F(m53)` | **inside by 26** |

- `B_4 >= 198`, attained by `(40, 35, 18, 35, 18, 52)` at phase 31, `J = 6 = J_max`, letters
  BAD·DOWN·UP·DOWN·UP·BAD. Its three 4-subwindows are realised at the columns `(40,35,18,35)`
  **26,914,858,374,121,270**, `(35,18,35,18)` **39,311,849,256,836,467**, `(18,35,18,52)`
  **31,164,400,356,159,577** of m47, each verified directly; the 6-word itself is NOT realised.
  So `B_4 > 171` and `k* >= 5` is PROVED. (The `J = 5` term is `>= 175` at
  `(52, 18, 35, 18, 52)`, phase 1.)
- **`B_5 = 145` exactly, and `145 <= 171`: the law's first half holds with margin 26**, at the
  deepest order the ladder has ever needed. 622,655 words scanned, 15,546 solver calls,
  6.3 x 10^8 nodes, 390 windows sent to the full solver, 16,047 s; and the independent audit finds
  **621,575 of the 623,871 fusing 6-words above 145, 0 of them level-5 admissible, 0 undecided**.

**The binding term is the RECORD, and it is realised.** This is the new thing at this rung. The
only relaxed term, `J = 6`, gives 138 -- and that word is itself realised:

    (15, 35, 18, 35, 18, 17) at phase 3,  span 138,  BAD·DOWN·UP·DOWN·UP·BAD
    realised at column 49,446,827,490,946,762 of m47

so it is not a relaxation at all, it is an exact fusion, and it falls 7 short of the record. The
maximum of `B_5` is therefore the certified record term itself, attained at `J = 4`:

    (70, 35, 18, 22) at phase 1,  span 145 = F(m53),  realised at column 82,799,441,296,736,535

(2 of the 674 fusing words of span 145 are realised.) Two consequences:

1. **`B_5 = F(m53) = 145`: the relaxation is EXACTLY the machine.** P6 predicted strictness and is
   refuted here. At every earlier rung the relaxation overshot the record -- 7 at `37 -> 41`, 15 at
   `41 -> 43`, **35** at `43 -> 47` -- and here the overshoot is **0**.
2. **The binding word is realised** -- for the first time on the ladder. At `41 -> 43` and
   `43 -> 47` the binding word is a pair of overlapping realised windows that the machine never
   realises together; at `47 -> 53` there is no such word above the record at all.

`k* = 5 = L + 1`. The record gate passed in the exhibiting direction (the column above); the full
`Q*_J` row was not computed (8).

### 4.6 `53 -> 59`: `L = 3`, and `B_L` comes within 1 of the budget

`F(m53) = 145`, `q' = 59`, budget **204**, certified `F(m59) = 161`. Letters at most `F(M)`:
UP 20, 79, 138; DOWN 39, 98; PAD 59, 118. `D_1` superset `1 .. 145`; level-2 filter the superset of
pairs of span at most `F_2(m53) = 159` (exact, corpus). Certified caps
`F_j(m53) <= 145, 159, 304, 318, 463` at `j = 1 .. 5`.

**`L(m53) = 3`**, witness `(20, 98, 20)` = UP·DOWN·UP, with the complete refutation at length 4
(169 legal candidates, 0 realised); 9,643 s. `J_max = 5`, and the law predicts `k* = 4`. Fusing
words: 145, 12,089, 53,159, 170,491, **509,972** at `J = 1 .. 5` (spans to 436).

What is measured so far:

| quantity | value | status |
|---|---|---|
| `B_3 = B_L`, `J = 4` term | `>= 203`, word `(68, 39, 20, 76)` at phase 11 | LOWER BOUND, 20,737 wider words deferred |
| `B_3`, `J = 5` term | `>= 177`, word `(22, 20, 98, 20, 17)` at phase 37 | LOWER BOUND, 60,062 deferred |
| `B_3` against the budget 204 | `203 <= 204` **so far** | **P2 NOT settled: this is the first rung where `B_L` has not cleared the budget** |
| every span from 205 to 215 | **14,487 fusing 4-words** (1,302 at span 205 rising to 1,332 at 215), **no level-3 admissible word found in any of them** | soft verdicts: a word with a deferred 3-subwindow is skipped, so this is strong evidence, not a refutation |
| `B_4 = B_{L+1}` | the descending `k = 4`, `J = 5` scan is running: from span 436 down to span 409 so far, no level-4 admissible word yet | RUNNING |

The resume point is exact: `uv run python research/anchor235/r71/ol2_step.py 53 5 norec`, which
reloads `results/memo_m53.json` and restarts the `k = 4`, `J = 5` descending scan; and
`uv run python research/anchor235/r71/ol2_lb.py 53 3 215 <procs>` to continue the ascending hunt
for a level-3 admissible fusing word above the budget (spans to 215 are already done), which is
all `B_L` needs.

`B_3(m53; 59) >= 203` against a budget of 204 is worth naming on its own. At every rung below,
`B_L` cleared the budget by tens (42 at `41 -> 43`, 63 at `43 -> 47`, 27 at `47 -> 53`). Here the
best word found so far is INSIDE the budget by 1, and the eleven span levels immediately above the
budget -- 14,477 fusing 4-words -- have produced nothing. If the deferred words above 203 contain
no level-3 admissible word, then `B_3 <= 204`, `k* <= 3 = L`, and the law's SECOND half fails here
-- the mirror image of what happened at `43 -> 47`, and on the same mechanism read the other way:
`L(m53) = 3` is one deeper than `L(m43) = 2`, so the relaxation is taken one level deeper and
tightens, exactly as it did at `47 -> 53` where `L = 4` drove the overshoot to 0.

---

## 5. Mechanism

### 5.1 What the binding word is, every time

At `37 -> 41` the binding word was `(22, 41, 14, 21)` at phase 19 -- two BAD flanks around the
two-letter middle `PAD, UP` of total 55. At `41 -> 43` it is

    (28, 14, 43, 33) at phase 1,   BAD, DOWN, PAD, BAD,   middle 14 + 43 = 57

-- the same shape, in the same slot, with the same two ingredients: **one tooth-step and one PAD**.
The middle is not the letter floor and it is not the cheapest thing the alphabet allows; it is the
cheapest thing the ENGINE offers. At `37 -> 41` the census was explicit (of the 36 letter pairs
only seven are realised 2-windows of m37, and only four of those are legal), and the same holds
here: `L(m41) = 2` means the machine realises legal letter words of length 2 and no more, and the
five it realises are `(14,43), (43,14), (29,43), (43,29), (43,43)` -- every one of them contains a
PAD. The alphabet's cheapest legal pair `(14, 29)` (sum 43, the minimum possible) is **not** a
realised 2-window of m41 at all. So the deepest fusion has to buy its middle at 57 rather than 43,
and the relaxation's extra span is what the two flanks can then carry.

That is the pattern across the ladder: **the middle of the deepest fusion is fixed by `L(M)` and by
which letter words the engine actually realises, and the relaxation's overshoot is what the flanks
add on top.** The engine does not offer a cheap middle at any price -- and where it does (m47,
where the bare alternation `18, 35, 18, 35` is realised because `53 mod 210` is a `PSORD = 5`
class), `L` jumps and `J_max` jumps with it.

`43 -> 47` is the same word in the same slot once more:

    (45, 16, 47, 45) at phase 2,   BAD, UP, PAD, BAD,   middle 16 + 47 = 63

-- one tooth-step and one PAD, two BAD flanks. Three rungs, three identical shapes. What changes
from rung to rung is not the shape but the SIZE OF THE FLANKS, and 5.2 is now able to say exactly
what fixes them.

### 5.2 The flank identity: `B_{L+1} = A + C - m`

The binding term at `J = J_max = L + 2` has exactly two `(L+1)`-subwindows, `w[0..L]` and
`w[1..L+1]`, and they overlap in the `L` letters of the middle. So its span is the span of the
first plus the span of the second minus the span of the middle, and maximising it means
maximising each end independently:

> **The deepest term of `B_{L+1}` is `A + C - m = m + a + c`**, where `m` is the span of the
> realised legal `L`-word in the middle, `A = m + a` is the widest realised `(L+1)`-window ENDING
> in that middle and `C = m + c` the widest realised `(L+1)`-window BEGINNING with it. So it is the
> middle plus the widest flank the engine allows on each side.

`ol2_flank.py` measures both profiles exhaustively -- every flank value from `F(M)` downwards put
to the instrument until one is realised, so the values above the maximum are REFUTED, not unsearched:

| rung | middle | `m` | widest left flank `a` (refuted above) | widest right flank `c` (refuted above) | `m + a + c` | the deepest term |
|---|---|---|---|---|---|---|
| `41 -> 43` | `(14, 43)` DOWN·PAD | 57 | **28** (34 refuted) | **33** (29 refuted) | 57 + 28 + 33 = **118** | 118 `= B_3` |
| `43 -> 47` | `(16, 47)` UP·PAD | 63 | **45** (38 refuted) | **45** (38 refuted) | 63 + 45 + 45 = **153** | 153 `= B_3` |
| `47 -> 53` | `(35, 18, 35, 18)` DOWN·UP·DOWN·UP | 106 | **15** (104 refuted) | **17** (102 refuted) | 106 + 15 + 17 = **138** | 138, `J = 6` |

The prediction is exact at all three rungs: the fusion condition costs nothing, the widest flanks
the engine allows are attainable together, and the deepest term is their sum. That makes the
refutation legible. From `41 -> 43` to `43 -> 47` the middle got 6 wider (57 to 63), which HELPS
the law, but the two flanks went from 28 and 33 to 45 and 45, and `35 = 17 + 12 + 6` is the whole
increase in `B_{L+1}`. The budget only moved by `F(43) - F(41) + (47 - 43) = 12 + 4 = 16`.

And it says what `47 -> 53` is doing. There the middle is 106 wide -- an `L = 4` word, four letters
-- and the engine then allows flanks of only **15 and 17**, with 104 and 102 wider values refuted.
The `(L+1)`-windows themselves are no narrower than at m43 (121 and 123 against 108 and 108); it is
the middle that has eaten them. **A long middle is self-limiting: the same `(L+1)`-window budget
has to contain it, so the flanks it can still carry shrink.** That is why the deepest term falls to
138, below the record 145, and `B_5` collapses onto the machine.

**The relaxation's binding span is the middle plus what the engine will hang on either side of it,
and that quantity is not paid for by `F(M) + q'`.**

### 5.3 Where the relaxation and the machine come apart, and why the law broke

`B_{L+1} - F(M + q')` is the whole content of the lemma of `monotone_functional.md` 6.3(ii). Put
next to the budget's own slack, it is the law:

| rung | `B_{L+1}` | `F(M + q')` | **overshoot** | budget | **slack** | law `overshoot <= slack` |
|---|---|---|---|---|---|---|
| `37 -> 41` | 98 | 91 | 7 | 129 | 38 | holds, 31 to spare |
| `41 -> 43` | 118 | 103 | 15 | 134 | 31 | holds, 16 to spare |
| `43 -> 47` | **153** | 118 | **35** | 150 | 32 | **FAILS by 3** |
| `47 -> 53` | 145 | 145 | **0** | 171 | 26 | holds, 26 to spare |

The budget inequality `F(M + q') <= F(M) + q'` is true at all four (slack 38, 31, 32, 26). **What
the refutation kills is the relaxation, not the budget.** The order law was a route TO the budget:
`F(M + q') <= B_{L+1} <= F(M) + q'`. The upper half of that sandwich is now false at `43 -> 47`,
so `B_{L+1}` cannot be the monotone functional. The lower half, `B_{L+1} >= F(M + q')`, is true by
construction at every `k` and survives.

The two failures are opposite in kind and that is the most informative thing in the branch:

- at `43 -> 47` the overshoot is **35**, the largest of the four, and `L = 2` the smallest;
- at `47 -> 53` the overshoot is **0**, the smallest possible, and `L = 4` the largest.

`L` is fixed by `q' mod 210` through `PSORD` (4.2), not by the engine. So the depth at which the
relaxation is taken, `k = L + 1`, is set by an arithmetic property of the INCOMING gear, while the
thing it has to bound is a property of the engine. When `L` is small the relaxation is taken too
shallow: level-3 admissibility constrains only overlapping triples, and the engine is free to hang
a 45 on either side of `(16, 47)`. When `L` is large the relaxation at level `L + 1 = 5` is so
constrained that the only words that survive it are realised windows, and `B_5` collapses onto
`F(M + q')` exactly. **The order law asks a fixed shallow question of a system whose depth is set
elsewhere.**

A second feature, first seen at `41 -> 43` and now on the record twice: the record's argmax and the
relaxation's argmax come apart. At `41 -> 43` `F(43) = 103` is attained at `J = 2, 3`
(`Q*_J = 91, 103, 103, 100`) while `B_3`'s maximum is at `J = 4 = J_max`. At `47 -> 53` it goes the
other way: `B_5`'s maximum is the record term at `J = 4`, while `J_max = 6` gives only 138. So P4
("the binding term is the deepest fusion") is refuted: it held at 10 of 11 rungs and fails at
`47 -> 53`, where the deepest fusion is no longer the widest thing level-`L+1` admissibility
allows.

---

## 6. What is new

1. **The order law is tested out of sample at engines no ladder reaches.** m41, m43, m47, m53 have
   periods `5.3 x 10^13`, `2.3 x 10^15`, `1.1 x 10^17`, `5.7 x 10^18` columns. Before this branch
   the law's evidence stopped at `37 -> 41`.
2. **The depth-2 dictionary is a property of the gears, not of a closure.** `D_1(m37)` (75 values)
   and `D_2(m37)` (2,053 rows) come out of the covering instrument alone, identical to the ladder's,
   and `D_1(m41)` (88 values, max 91 = `F(41)`, absent 84, 87, 89) is produced at an engine the
   ladder cannot reach.
3. **THE ORDER LAW IS FALSE.** `B_{L+1}(m43; 47) = 153 > 150 = F(m43) + 47`, exact, with the
   binding word `(45, 16, 47, 45)` at phase 2 and an exhaustive audit of the 41,405 wider fusing
   4-words. The functional `Phi = B_{L+1}` is not budget-monotone, and the rung that breaks it is
   the first one past the published ladder but one. The law's OTHER half survives:
   `B_{L+1} >= F(M + q')` holds by construction.
4. **`k* = L + 2 = J_max` at `43 -> 47`**, the first rung where the least level that comes inside
   the budget is not `L + 1`. `B_4 = F(m47) = 118 <= 150` because at `k = J_max` level-`k`
   admissible means realised.
5. **`L(m43) = 2`, `L(m47) = 4` and `L(m53) = 3`**, with witnesses and with the complete refutation
   one length higher. `L(m47) = 4` is the largest merge depth on the ladder and the first that needs
   no pad; `L(m53) = 3` buys its third letter with a lap (`98 = 59 + 39`) instead of a pad.
6. **Its mechanism is `PSORD`.** The jump in `L` at `47 -> 53` is explained, before it is measured,
   by the kernel-checked bare-word cap (docs/proofs/12): `53` is one of only six classes mod 210
   with `PSORD = 5`, while `41, 43, 47` sit in the `PSORD = 1` row. **`L` is not monotone along the
   ladder and its size is a property of `q' mod 210`, not of the engine.**
7. **The flank identity** (5.2): the deepest term of `B_{L+1}` is `m + a + c` -- the engine's
   cheapest realised legal `L`-word plus the widest flank the engine allows on each side --
   measured exactly at three rungs (`57+28+33 = 118`, `63+45+45 = 153`, `106+15+17 = 138`), with
   every wider flank value refuted. This is the closed form of the binding term, and it is what
   the law failed to bound.
8. **A long middle is self-limiting** (5.2): at m47 the `L = 4` middle is 106 wide and the flanks
   collapse to 15 and 17 (104 and 102 wider values refuted), so the deepest term falls BELOW the
   record and `B_{L+1} = F(M + q')` exactly -- the relaxation is the machine, overshoot 0, the
   first such rung.
9. **`B_k = max(F(M + q'), relaxed terms J > k)` exactly** (4.3) -- an identity that removes the
   whole exact-term block from every `B_k` computation, and with it the expensive wide-thin
   covering problems.
10. **Subadditivity of the window records**, `F_j <= F_a + F_b`, which caps every depth at engines
    where the record table has run out.
11. **The instrument's cost law**, measured (2.6): a NO costs what the pattern's OPEN count buys,
    so the Jacobsthal-shaped question (wide, thin) is the expensive one and the fusion-shaped
    question (deep, many open offsets) is cheap. That is why this branch can go where the ladder
    cannot.
12. **A membership verdict can be turned into an explicit column** (4.3b): the covering search's
    phase vector plus CRT names a column of the machine, verified there against the strike rule.
    Every YES this branch relies on is published as an integer -- e.g. `(45, 16, 47)` occurs at
    column 1,669,802,076,752,677 of m43 -- and the certifier is gated in both directions against a
    full-period sieve at m11, m13, m17 with 0 disagreements.

---

## 7. Verdict

**The order law `k*(M; q') = L(M) + 1` is REFUTED.** It holds at `41 -> 43` and `47 -> 53` and
fails at `43 -> 47`, where `B_{L+1} = 153` against a budget of 150 and `k* = L + 2 = J_max = 4`.
Node 4.i.b.ii's functional `Phi = B_{L+1}` is therefore not budget-monotone, and the route
`F(M + q') <= B_{L+1} <= F(M) + q'` is closed at its upper half.

The final table, every number exact or a certified bound with its witness:

| step | `F(M)` | budget | `L` | `J_max` | `B_L` | `B_{L+1}` | margin | `k*` | binding word | phase | `J` | realised? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `37 -> 41` | 88 | 129 | 2 | 4 | 161 | 98 | +31 | 3 | `(22, 41, 14, 21)` | 19 | 4 | NO |
| `41 -> 43` | 91 | 134 | 2 | 4 | `>= 176` | **118** | +16 | 3 | `(28, 14, 43, 33)` | 1 | 4 | NO |
| `43 -> 47` | 103 | 150 | 2 | 4 | `>= 213` | **153** | **-3** | **4** | `(45, 16, 47, 45)` | 2 | 4 | NO |
| `47 -> 53` | 118 | 171 | 4 | 6 | `>= 198` | **145** | +26 | 5 | `(70, 35, 18, 22)` = the record | 1 | 4 | **YES** |
| `53 -> 59` | 145 | 204 | 3 | 5 | `>= 203` (running) | running | -- | -- | -- | -- | -- | -- |

`B_{L+1}` is EXACT at every completed rung (the audits of 4.4 and 4.5: 41,405 and 621,575 wider
fusing words, 0 level-`k` admissible, 0 undecided). `B_L` is a certified lower bound with a witness
at each, which is all the law's second half asks for -- except at `53 -> 59`, where the best
witness so far is 203 against a budget of 204 and the second half is therefore OPEN, possibly about
to fail the other way.

What survives the refutation, and is worth carrying to the next node:

- the **flank identity** `m + a + c` (5.2), a closed form for the binding term with every input
  measured exactly;
- the **overshoot/slack reading** (5.3): the law is `overshoot <= slack`, measured
  `7 <= 38`, `15 <= 31`, `35 > 32`, `0 <= 26`. The budget's own slack sits near 30 and does not
  grow; the overshoot is free to;
- **the depth mismatch**: `k = L + 1` is set by `q' mod 210` (PSORD) and the thing it must bound is
  set by the engine. Any repaired law has to choose its depth from the engine, not from `q'`. The
  obvious candidate the data suggests -- `k* <= J_max`, which is true at all five rungs -- is
  vacuous as a route, since `B_{J_max} = F(M + q')` makes it the budget inequality itself;
- the **certifier** (4.3b), which turns any future YES into a column of the machine.

---

## 8. Dead ends and what they cost

- **Tabulating `D_2(M)` above m37 is dead.** 2,537 candidate pairs at m41, each NO costing 30-70 s
  (six measured: `(91,12) 55s`, `(51,52) 69s`, `(45,58) 46s`, `(60,43) 40s`, `(88,15) 31s`,
  `(70,33) 53s`, all NOT realised). The run was abandoned. What survived: the observation that the
  table is not needed at all (2.6, 4.3), which is what made the rungs affordable.
- **Computing the exact `Q*_J` row for `J <= k` is dead as a means to `B_k`** -- the identity of
  4.3 shows those terms can never bind. It is kept only as a GATE, where it is affordable
  (`41 -> 43`, where the whole row `91, 103, 103, 100` was recomputed and matched `F(43) = 103`).
- **The exact `Q*_J` row is dead as a gate above m41 too.** Both step runs reached the record row,
  found the record WITNESS in minutes (span 118 at `(2, 31, 85)`, span 145 at `(70, 35, 18, 22)`,
  with their columns) and then stalled in the exact `J = 2` scan -- the wide-thin pattern of 2.6 --
  for six and four hours with no further output, their worker pools dead. Both were stopped. What
  replaces the row: the exhibiting half alone, which is cheap, plus `max_J Q*_J <= F(M + q')` from
  the record law. That pins the gate without a single wide-thin refutation.
- **The meet-in-the-middle reference solver is unusable above ten gears.** It was tried as the
  independent second opinion on the binding words of m43 and m47 and did not return in 10 minutes
  on a single word: with 12 or 13 gears the enumerated side runs to millions of rows and the
  residual sweep is quadratic in them. What replaced it is strictly better (4.3b): a YES is
  certified by an explicit COLUMN of the machine, checked against the strike rule, which depends on
  no solver at all. The reference solver stays where it is sound and affordable, at the small
  engines of the certifier's gate.
- **The ascending witness hunt is the wrong tool at `k = 2`.** `ol2_lb.py` scans upward from the
  budget and stops at the first admissible word, which is right for a lower bound on `B_L` when
  `L >= 3` (deep, cheap patterns). At `k = 2` its subwindows are wide-thin pairs and a single span
  level cost over 600 s at m43; the attempt was abandoned in favour of the descending scan's own
  witness. At m53 (`k = 3`) it works, at about 200 s a span level once the memo is warm.
- **The exact value of `B_L` above m37 is open, and does not matter -- except once.** At each rung
  `B_L` is reported as a certified lower bound with a witness; the law needs only
  `B_L > F(M) + q'`, which the witness proves. The exception is `53 -> 59`, where the best witness
  is 203 against a budget of 204: there the exact value (or at least one witness above 204) is
  needed, and 4.6 says where the search stands.

---

## 9. The remaining open items, sorted

**Closed here.**

- `k* = L + 1` as a law: REFUTED at `43 -> 47` (4.4). `Phi = B_{L+1}` is not budget-monotone.
- `L(m43)`, `L(m47)`, `L(m53)`: 2, 4, 3, each with a witness and a complete refutation one length
  higher (4.2).
- `B_{L+1}` at `41 -> 43`, `43 -> 47`, `47 -> 53`: exact, with audits (4.1, 4.4, 4.5).
- The closed form of the binding term: `m + a + c`, measured exactly at three rungs (5.2).
- Whether a membership YES can be exhibited rather than trusted: yes, as a column (4.3b).

**Measurement with no structural content.**

- The exact `Q*_J` rows at `43 -> 47` and `47 -> 53`. The record law already gives the upper half
  and the exhibited column gives the lower half; the row would only fill in which `J` attains the
  record. Expensive (8) and not load-bearing.
- The exact value of `B_L` at `41 -> 43`, `43 -> 47`, `47 -> 53` (`>= 176`, `>= 213`, `>= 198`).

**Root question in disguise.**

- `k* <= J_max`. True at all five rungs, but `B_{J_max} = F(M + q')` identically, so the statement
  IS the budget inequality `F(M + q') <= F(M) + q'`. Not a route; do not open it as one.

**Genuinely open on this part alone.**

1. **`B_3(m53; 59)` against 204.** Exact statement: is there a fusing 4- or 5-word of m53 with span
   above 204, all of whose 3-subwindows are realised 3-windows of m53? A single witness settles it
   yes; the descending scan with `resolve_hard` settles it no. Attack: `ol2_lb.py 53 3 215` upward
   (spans 205 to 215 are done, 14,487 words, nothing; about 200 s a span level with a warm memo),
   or the running `ol2_step.py 53 5 norec`. A NO here makes `k* <= L` at this rung and the law
   fails on BOTH sides of `47 -> 53`.
2. **`B_4(m53; 59)` and `k*(m53; 59)`.** The `k = 4`, `J = 5` descending scan runs from span 436
   downwards and had reached span 409 when this document was written (5 processes). It stops at the
   first level-4 admissible fusing word; only if that word's span is at most 204 does the law's
   first half survive this rung. Resume as in 4.6.
3. **Is the flank identity a theorem?** `m + a + c` was measured at three rungs; it is not proved
   that the widest left flank and the widest right flank can always be taken together (the fusion
   condition might obstruct at some rung). Attack: the fusion condition at `J = L + 2` is a
   statement about `z` modulo `q'` only, so it is a finite check per middle; prove it, or find the
   rung where the flanks cannot be combined.
4. **Which depth repairs the law?** The data says the right depth is not `L + 1`. The engine-side
   candidate suggested by 5.2 is a depth chosen so that the middle is long enough to strangle the
   flanks -- at `47 -> 53` that is `L = 4` and it works. Exact question: is there a function
   `k(M)` computable from the engine, with `B_{k(M)} <= F(M) + q'` at every rung and
   `k(M) < J_max` at infinitely many? Attack: measure `B_k` for all `k` from `L` to `J_max` at the
   five rungs and look at where each row crosses the budget -- cheap, since 4.3 subsumes every
   `J <= k` term.
5. **Why is `a = c = 45` at m43 and `a != c` at m41?** The left and right flank families are not
   mirrors of one another (the mirror of `(f, mid)` is `(rev mid, f)`, not `(mid, f)`), so the
   equality is not forced. Attack: `ol2_flank.py <y> <mid> <procs> full` records the WHOLE profile
   (which flank values work, not only the largest); compare the two profiles across rungs. Only
   the maxima and the refuted counts are on disk so far (`results/flank_m*.json`).
