# The order law out of sample at 37 -> 41

Node **4.i.b.ii** (`research/proof/monotone_functional.md`), item U2 of the whole-tree review
(`research/proof/tree_review.md` section 3). The review left one rung of the order law undecided
and named the number that would decide it. This document decides it.

Scripts in `research/anchor235/r70/` (prefix `ol_`); outputs in `research/anchor235/r70/results/`
(untracked). Every number this document relies on is written into the document.

---

## 1. What was to be decided

The order law of `monotone_functional.md` 4.3 says that the order of interaction the budget
inequality needs is exactly one more than the merge depth,

    k*(M; q')  =  L(M) + 1  =  J_max - 1,

where `B_k(M; q')` is the widest span of a level-`k` admissible word that fuses at some phase of
`q'` (a word is level-`k` admissible when every `k` consecutive entries of it lie in `D_k(M)`, the
table of realised `k`-windows), and `k*` is the least `k` with `B_k <= F(M) + q'`. Both halves of
the law were measured with no exception: `B_{L+1} <= F + q'` at 9 of 9 computable rungs and
`B_L > F + q'` at 7 of 7 rungs from `13 -> 17` up, so `k* = L + 1` at 6 of 6 decisive rungs.

The one rung left open was the top of the ladder. At `37 -> 41`:

- the engine is `{5..37}` in the column coordinate (column `k` is the pair `(6k - 1, 6k + 1)`;
  gear `g` strikes `k` iff `k = +-u_g (mod g)` with `u_g = 6^{-1} mod g`), the new gear is
  `q' = 41`, and `u = 7`, `d = 2u = 14`, so the letters of 41 are `PAD` (a gap `= 0 mod 41`),
  `UP` (`= 14`), `DOWN` (`= 27`), and the letter floor is `a_L(41) = min(14, 27) = 14`;
- `F(m37) = 88`, so the budget is `F + q' = 88 + 41 = 129`;
- `L(m37) = 2` and `J_max = L + 2 = 4`, so the law predicts `k* = 3`;
- `B_2(m37; 41) = 161 > 129` was known, which gives `k* >= 3`;
- `B_3(m37; 41)` was **not computable**, because both closure ladders leave m37 at depth 2 and
  `D_3(m37)` was unavailable. m37 has period `1.24 x 10^12` columns and `2.18 x 10^11` gaps and
  has never been scanned.

So the prediction under test was sharp and one-sided: `B_3` had to fall from 161 to **at most 129**,
a drop of at least 32 columns, or the law had its first exception.

## 2. The instrument

Two instruments, one old and one new.

### 2.1 The closure ladder (unchanged operator)

`ol_ladder.py` runs the merge closure `T_29, T_31, T_37` from `D_15(m23)`, importing the closure
step from `research/anchor235/r61/lc_core.py` through `r66/mo_core.py` -- so the operator here is
literally the one that computed `F(37) = 88` and `F(41) = 91` in `ladder_closure.md`. It is
`r66/mo_ladder.py` with the readouts this branch does not need dropped and one thing added: the m37
window dictionary itself is written to disk, which the r66 run did not do.

| rung | input depth -> output | `F` | corpus | `Q*_J` | `|D|` out | `N` | `loss` | `over0` | `L` (bare/pad) | corpus `L` | secs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base m23 | `|D_15(m23)| = 4,407,350` | -- | -- | -- | 4,407,350 | 7,952,175 | -- | -- | -- | -- | 14.1 |
| `23 -> 29` | 15 -> 10 | **43** | 43 | 34, 39, 43 | 15,240,585 | 214,708,725 | 0 | 0 | 3 (3/1) | 3 | 412.4 |
| `29 -> 31` | 10 -> 6 | **58** | 58 | 43, 55, 58, 55, 55 | 2,678,901 | 6,226,553,025 | 0 | 0 | 3 (3/2) | 3 | 173.0 |
| `31 -> 37` | 6 -> 2 | **88** | 88 | 58, 68, 85, 88, 68 | 2,053 | 217,929,355,875 | 0 | 0 | **2 (1/2)** | 2 | 23.0 |

Ten minutes and well under a gigabyte. Every gate passes: the three corpus records, `loss = 0` and
`over0 = 0` at all three rungs, the dictionary sizes 15,240,585 at m29 and 2,678,901 at m31 equal to
`monotone_functional.md` 1 and 2.2 row for row, `L(m37) = 2` equal to the corpus, and the m37 mass
217,929,355,875 equal to the recorded `N(m37)`. So `J_max(37 -> 41) = 4`, and the ladder stops at
depth 2 exactly as reported: it gives `D_1(m37)` and `D_2(m37)` and no more.

### 2.2 The pattern instrument (new): membership in `D_k(m37)` with no scan and no ladder

`D_3(m37)` cannot be reached by deepening the ladder -- the review priced that and both existing
ladders stall at depth 2. It does not have to be. Membership of one window in `D_k(M)` is decidable
exactly, window by window, from the machine's own definition:

> Gear `g` strikes column `k` iff `k = +-u_g (mod g)`. Read that in the OFFSET coordinate of a
> window starting at column `x`: offset `o` is struck by `g` iff `o = +-u_g - x (mod g)`, i.e. `g`
> strikes exactly TWO residue classes of the window, `{t_g, t_g + d_g}` with `d_g = 2u_g mod g`
> fixed and `t_g = -u_g - x (mod g)` free. By CRT the map `x -> (x mod g)_g` is a bijection from
> one period onto the product of the `Z_g`, so as `x` runs over the period the vector `(t_g)_g`
> runs over every combination, independently.
>
> **Hence a local pattern -- a set `OPEN` of offsets required open and the complementary set
> `CLOSED` of offsets of `[0, S]` required struck -- is realised in `M` iff there is a choice of
> `t_g in Z_g`, one per gear, such that (i) no gear strikes any offset of `OPEN` and (ii) every
> offset of `CLOSED` is struck by some gear.**

Condition (i) is independent gear by gear (it deletes at most `2|OPEN|` phases of each); condition
(ii) is a covering problem on at most `S + 1` offsets and ten gears. `ol_pattern.py` decides it two
ways, both exact and both used:

- `realised` -- meet in the middle. The gears are split into two groups, every surviving phase
  combination of each group is enumerated as a bitmask of struck offsets, and every pair is tested.
  No pruning, no heuristics; it is the brute-force reference.
- `realised_search` -- exact cover by search. Take the closed offset with the fewest surviving ways
  of being struck, branch over them, recurse, with a capacity bound (the offsets still uncovered
  cannot exceed what the free gears can strike at their best single phases). Same verdicts, one to
  three orders of magnitude faster at m37.

A YES verdict carries a certificate. `ol_witness.py` turns the solver's phase vector back into a
column, `x = -(u_g + t_g) (mod g)` for every gear, CRTs it to one integer `x mod P`, and then
re-derives the whole window from scratch -- for every offset of the span it tests `x + o` against
every gear's two teeth, with no reference to the solver. So each realised window quoted below comes
with a column of m37 that anyone can check by hand.

This is the object the review said was missing, obtained without the ladder: `D_3(m37)` is never
built as a table, and it does not need to be. `B_3` quantifies over words, and each word's
admissibility is one covering problem.

### 2.3 Gates on the new instrument

| gate | what | result |
|---|---|---|
| **G1** m23, complete | the period of m23 (37,182,145 columns, 7,952,175 gaps) is sieved directly, giving `D_3(m23)` exactly (3,135 rows); every one of the `33^3 = 35,937` triples of realised gap values is put to the instrument and compared | **0 disagreements** |
| **G2** m29, complete on the realised side | `D_3(m29)` exactly off the closure from `D_9(m23)` (three new gaps fuse at most `3 J_max = 9` old ones, `loss = 0`): 7,184 rows carrying all 214,708,725 gaps, the count of `monotone_functional.md` 4.5 | all 7,184 realised triples certified, **0 missed**; 4,000 sampled unrealised triples, **0 false positives** |
| **G3** m29, the search solver | the same complete test for `realised_search` | 7,184 realised, **0 missed**; 6,000 sampled unrealised, **0 false positives** |
| **G4** m37, the record itself | `Q*_4(m37; 41) = 91 = F(41)` is the span of a realised 4-window of m37 that fuses; the instrument is asked for it with no m37 table deeper than 2 in hand | **91**, the record reproduced; the witness is `(21, 14, 41, 15)` at phase 20, found after 1,902 words, 209 s |
| **G5** m37, `F_2` | the widest realised 2-window of m37, straight off `D_2` | **90**, the corpus `F_2(m37) = 90` |
| **spot** | `(34, 29, 34)`, the level-1 extremal word of `23 -> 29`, which `monotone_functional.md` 4.4 states is realised by no window of m23 | **NOT realised**, as recorded |
| **spot** | `(18, 10, 30)` and `(23, 10, 25)`, the band witnesses of 7.2 at `29 -> 31` | both **realised**, with certified columns 278,620,515 and 390,658,037 of m29 |

## 3. The tables

At m37 the tables the relaxation quantifies over are small, and the third is not a table at all.

| table | size | what it stands for |
|---|---|---|
| `D_1(m37)` | **75** realised gap values | 1 .. 72 complete, then 77, 85, 88 (73-76, 78-84, 86, 87 are absent) |
| `D_2(m37)` | **2,053** rows | all 217,929,355,875 gaps of m37, a compression of `1.06 x 10^8` |
| `D_3(m37)` | never built | decided window by window; **200** distinct triples occur in the 4-words that could beat 98, of which **148** were put to the instrument before the answer was settled, and 654 fusing 3-words besides |

Among the 75 realised values the letters of 41 are `UP = 14, 55`, `DOWN = 27, 68`, `PAD = 41`. The
second pad, 82, is **not** a realised gap of m37 -- a fact that does real work below.

## 4. `B_1` and `B_2`, recomputed

`ol_b2.py` runs the relaxation's dynamic programme of `r66/mo_order.py`, imported unchanged, on the
`D_1` and `D_2` the ladder produced.

| level `k` | `J = 1` | `J = 2` | `J = 3` | `J = 4` | `B_k` | budget 129 |
|---|---|---|---|---|---|---|
| 1 | 88 | 176 | 244 | **299** | **299** | above by 170 |
| 2 | 88 | 90 | 143 | **161** | **161** | above by 32 |

Both reproduce `monotone_functional.md` 4.2 exactly: `B_1 = 299`, `B_2 = 161`. The maximum is at
`J = J_max = 4` at both levels, so **`B_L = B_2 = 161 > 129` stands, recomputed from a fresh ladder
run: `k* >= 3`.**

The level-2 binding word is worth naming, because level 3 refuses it:

    B_2 = 161:  (35, 41, 27, 58) at phase z = 20,  letters  BAD, PAD, DOWN, BAD
                (and its mirror (58, 27, 41, 35) at phase 38)

Its two 3-subwindows are `(35, 41, 27)` of span 103 and `(41, 27, 58)` of span 126. Both are put to
the instrument and both come back **NOT realised** -- as they had to be, since the corpus row
`F_3(m37) = 97` (`monotone_functional.md` 3.2) makes 103 and 126 impossible spans for a realised
3-window. Level 2 lets a word be assembled from pairs that never sit in one window; level 3 does
not, and that is the whole difference between 161 and 98.

## 5. `B_3`

`B_3 = max over J <= J_max = 4` of the widest level-3 admissible `J`-word that fuses at a phase of
41. At level 3 the terms `J <= 3` are not relaxed at all -- a level-3 admissible word of length at
most 3 IS a realised window -- so only the `J = 4` term is relaxed, and its relaxation is one de
Bruijn step: the word `(g_1, g_2, g_3, g_4)` needs `(g_1, g_2, g_3)` and `(g_2, g_3, g_4)` in
`D_3(m37)` and nothing more.

### 5.1 The three exact terms

`J = 1` and `J = 2` come straight off `D_1` and `D_2`: `Q*_1 = 88` (the record gap itself, at phase
1) and `Q*_2 = 90` (the word `(88, 2)` at phase 8). Both equal the recorded `Q*_J` row
`88, 90, 90, 91` of `monotone_functional.md` 2.2.

`J = 3` is the exact `Q*_3`. `ol_j3.py` settles it completely, by the same descending scan: of the
3,122 level-2 admissible fusing 3-words of m37, **654 have span above 89**, and every one of them
is put to the instrument.

    654 words of span above 89, every one decided; the widest four realised are
        (28, 14, 48) at phase 13,  (48, 14, 28) at phase 34,
        (35, 27, 28) at phase 20,  (28, 27, 35) at phase 27,      all of span 90,
    and NOTHING above 90 is realised
    (1.7 x 10^7 search nodes, 0 fallbacks to the reference solver, 304 s).

> **`Q*_3(m37; 41) = 90`**, with the witness `(28, 14, 48)` -- exactly the value the closure with
> the span-threshold prune recorded (`monotone_functional.md` 2.2: `Q*_J = 88, 90, 90, 91`), here
> recomputed from an instrument that shares nothing with it but the depth-2 table.

So the `J = 3` term is 90 and cannot lift `B_3`. The widest candidate, `(58, 27, 58)` of span 143,
is not realised; nor is any other above 90. The corpus row `F_3(m37) = 97` is consistent
(`Q*_3 <= F_3` always, and a fusing 3-window is a constrained one).

### 5.2 The relaxed term, `J = J_max = 4`

`ol_b4.py` enumerates every level-2 admissible fusing 4-word -- all 41 phases, all 75 realised gap
values in each slot, adjacent pairs filtered exactly through `D_2(m37)` -- **2,520 words**, spans
161 down to 58, sorts them by descending span, and puts each word's two triples to the instrument.
Descending order means the first word whose triples are both realised is the maximum.

    1,624 words tested, 148 distinct triples decided (22 of them realised), 69 s, 3.9 x 10^6
    search nodes, 0 fallbacks to the reference solver.

> **The `J = 4` term at level 3 is 98**, attained by
>
>     (22, 41, 14, 21) at phase z = 19,  span 98,  letters  BAD, PAD, UP, BAD
>
> and by its mirror `(21, 14, 41, 22)`. Nothing wider is level-3 admissible.

### 5.3 `B_3`

    B_3(m37; 41)  =  max(88, 90, 90, 98)  =  98.

**`B_3 = 98 <= 129`.** The margin `budget - B_3` is **31**.

Two consistency checks the number has to pass, and does. `B_3 >= B_4 = F(41) = 91` by construction,
and 98 exceeds 91 by 7 -- so the relaxation is strictly larger than the machine at this rung, as at
4 of the 6 earlier decisive rungs. And `B_1 >= B_2 >= B_3 >= B_4`: 299, 161, 98, 91.

`B_4 = F(41) = 91` is not taken on trust here: `ol_g4.py` asks the instrument for it directly (gate
G4), enumerating the same 2,520 words in descending span and testing each 4-word for membership in
`D_4(m37)`. The first hit is at span **91**, the word `(21, 14, 41, 15)` at phase 20 -- the record
of m41 recomputed from `D_1(m37)`, `D_2(m37)` and ten covering problems, with the closure used only
to produce the depth-2 table.

## 6. The binding word

    (22, 41, 14, 21) at phase z = 19,  and its mirror  (21, 14, 41, 22) at phase z = 20
    span 98,  J = 4 = J_max

Take the mirror form and check it by hand. Offsets 0, 21, 35, 76, 98 against the phase 20:
`0 + 20 = 20` is unstruck (20 is neither `0` nor `d = 14` mod 41), `21 + 20 = 41 = 0`,
`35 + 20 = 55 = 14` and `76 + 20 = 96 = 14` are all struck, and `98 + 20 = 118 = 36` is unstruck. So
the two flanks are BAD and the two middles are letters -- `14` is an UP and `41` is a PAD, and UP
followed by PAD is legal (pads are transparent, and no two equal non-zero letters are adjacent).

Its two triples are realised, each with a certified column of m37 (`P = 1,236,789,689,135`), and
each column re-verified from scratch against all ten gears:

| triple | span | witness column `x` |
|---|---|---|
| `(22, 41, 14)` | 77 | 957,039,832,745 |
| `(41, 14, 21)` | 76 | 1,060,887,849,347 |
| `(21, 14, 41)` (mirror) | 76 | 606,089,557,672 |
| `(14, 41, 22)` (mirror) | 77 | 1,086,351,827,488 |

**The 4-word itself is NOT realised in m37.** The instrument says so directly, and the record law
says so too: a realised fusing 4-window of span 98 would make `F(41) >= 98`, and `F(41) = 91`. So
the binding word is exactly what the relaxation is for -- two windows the machine does contain,
overlapping in a way it never realises.

**And the overshoot is one substitution, in one slot.** Gate G4 found the true record fusion to be

    Q*_4 = 91:   (21, 14, 41, 15)  at phase z = 20,   realised

-- the same three gaps, at the same phase, with the closing flank 15 instead of 22. The machine
realises `(21, 14, 41)` and it realises `(14, 41, 22)`; it never realises them overlapping. That
single unrealised overlap is the whole of `B_3 - F(41) = 98 - 91 = 7`, and it is the smallest
instance of the lemma of `monotone_functional.md` 6.3(ii) that this rung has to offer.

**Is the binding term the deepest fusion?** Yes: `J = 4 = J_max`, as at 7 of 7 earlier rungs. It is
also the binding term one level down (`B_2`'s maximum is at `J = 4` too), so the pattern of 4.4
holds here in both places -- the term that violates the budget at level `L` and the term that
attains the bound at level `L + 1` are the same term, the deepest fusion, and one de Bruijn step is
what separates 161 from 98.

**Where the span comes from, and it is not where it came from at `29 -> 31`.** At `29 -> 31` the
extremal chains paid exactly the letter floor in the middle and bought their span with the flanks
(`(18, 10, 30)`, `(23, 10, 25)`, middle `= a_L = 10`). Here the opposite: the flanks are 21 and 22,
small, and 55 of the 98 columns sit in the middle, `14 + 41` -- one tooth-step plus one PAD, which
is the new gear itself. The reason is exact, and it is not a tendency but a census. The middles of
a 4-fusion are two adjacent letters, so they must be an adjacent pair of `D_2(m37)` drawn from the
letters `{14, 55}` (UP), `{27, 68}` (DOWN), `{41, 82}` (PAD). Of those 36 pairs, `D_2(m37)` contains
exactly **seven**, with these multiplicities out of 217,929,355,875 gaps:

| pair | letters | sum | openings carrying it |
|---|---|---|---|
| `(14, 41)`, `(41, 14)` | UP-PAD, PAD-UP | 55 | **1,525** each |
| `(27, 27)` | DOWN-DOWN | 54 | 662 -- **illegal**, two equal non-zero letters |
| `(14, 55)`, `(55, 14)` | UP-UP | 69 | 3 each -- **illegal**, two equal non-zero letters |
| `(27, 41)`, `(41, 27)` | DOWN-PAD, PAD-DOWN | 68 | **1** each |

So the deepest fusion at this rung has exactly **four** legal middles available to it -- `(14, 41)`,
`(41, 14)`, `(27, 41)`, `(41, 27)` -- and two of the four occur at a single opening of m37 apiece.
The cheapest legal letter pair, `(14, 27)` of sum 41 = the minimum the alphabet allows, is **not a
realised 2-window of m37 at all**, and neither is any pair involving 68; the second pad, 82, is not
even a realised gap value. The engine does not offer this rung a cheap middle at any price, and
`B_3 = 98` is the widest flank pair the machine will hang on the one middle it does offer:
`(21, 14, 41)` of span 76 on the left and `(14, 41, 22)` of span 77 on the right, meeting at 98.

## 7. Verdict

| | value | budget 129 | |
|---|---|---|---|
| `B_2 = B_L` | **161** | above by 32 | so `k* >= 3` |
| `B_3 = B_{L+1}` | **98** | inside by 31 | so `k* <= 3` |

> **`k*(m37; 41) = 3 = L(m37) + 1 = J_max - 1`. The order law's out-of-sample prediction is
> CONFIRMED, exactly.**

And the whole exact row is recomputed on the way: `Q*_J(m37; 41) = 88, 90, 90, 91` at `J = 1..4`,
the row of `monotone_functional.md` 2.2 entry for entry, with `Q*_4 = 91 = F(41)`.

The law's two halves now stand at:

- `B_{L+1} <= F(M) + q'` at **10 of 10** computable rungs (`5 -> 7 .. 37 -> 41`), margins
  3, 6, 9, 7, 7, 13, 3, 16, 7, **31**;
- `B_L > F(M) + q'` at **7 of 7** rungs from `13 -> 17` up, by 5, 12, 8, 34, 11, 23, **32**;
- hence `k* = L + 1` at **7 of 7 decisive rungs, 0 exceptions** (the two excluded rungs are
  `5 -> 7` and `11 -> 13`, where even `B_1` is inside the budget).

The binding term is the deepest fusion `J = J_max` at **8 of 8** rungs.

This was a real test, not a confirmation of something already known. `B_3` had to drop by at least
32 columns from `B_2 = 161` to satisfy the law; it dropped by 63, to 98, with 31 to spare -- the
largest margin on the ladder, at the rung whose record slack (38) is also the largest. The
functional `Phi = B_{L+1}` is therefore budget-monotone and a bound on the next record at every
rung of the ladder the project has, with no exception, and the theorem of `monotone_functional.md`
6.2 -- that `B_{L+1}(M; q') <= F(M) + q'` for every anchored machine gives the budget at every rung
-- keeps its clean measured record at the one rung that could have broken it.

What the rung does NOT do is close the item. `L` is still uncapped (gate item 1), and the lemma of
6.3(ii) is still open. What it adds is one more decisive instance, and the smallest instance of the
lemma at this rung, written out:

> `(21, 14, 41)` and `(14, 41, 22)` are realised 3-windows of m37; `14` and `41` are letters of 41
> and the flanks 21 and 22 are BAD at phase 20. Then `21 + 14 + 41 + 22 = 98 <= 88 + 41 = 129`,
> with 31 columns to spare -- and the machine itself stops 7 short of that, at 91.

## 8. What the run cost

| step | script | cost |
|---|---|---|
| the closure ladder m23 -> m29 -> m31 -> m37, `D_2(m37)` saved | `ol_ladder.py` | 623 s, peak under 1 GB |
| `B_1`, `B_2` off `D_1`, `D_2` | `ol_b2.py` | seconds |
| G1 (m23 complete) + G2 (m29 complete) | `ol_gate.py` | 504 s |
| G3 (m29, the search solver) | `ol_gate37.py` | 52 s |
| the `J = 4` term at level 3, and with it `B_3` | `ol_b4.py` | 69 s |
| the `J = 3` term, complete above 98 and then above 89 | `ol_j3.py` | 160 s + 304 s |
| G4 (`Q*_4 = 91` from the instrument alone) | `ol_g4.py` | 209 s |

Well inside the review's estimate of an hour and 3 GB, and the reason is 2.2: once membership in
`D_3(m37)` is a covering problem rather than a table to be built, the rung costs a few hundred
covering problems -- 148 to settle `B_3`, 654 more to settle the `J = 3` term, 1,902 for the gate
on the record -- instead of a ladder that does not exist. (`ol_b3.py`, which computes the whole
`B_3` table in one pass, was left behind by its own descending scan on the reference solver and was
stopped; `ol_j3.py` and `ol_b4.py` do the same two terms with the search solver in four minutes.)

## 9. Files

- `research/anchor235/r70/ol_ladder.py` -- the closure ladder, m37 dictionary saved
- `research/anchor235/r70/ol_pattern.py` -- the free-phase membership instrument (both solvers)
- `research/anchor235/r70/ol_witness.py` -- phase vector -> column, and the from-scratch re-check
- `research/anchor235/r70/ol_b2.py` -- `B_1`, `B_2`
- `research/anchor235/r70/ol_b3.py` -- the word enumeration (imported by the three below) and the
  exact terms `J = 1, 2`
- `research/anchor235/r70/ol_b4.py` -- the `J = 4` term at level 3 alone
- `research/anchor235/r70/ol_j3.py` -- the `J = 3` term above the deciding floor
- `research/anchor235/r70/ol_gate.py`, `ol_gate37.py`, `ol_g4.py` -- the gates
- `research/anchor235/r70/results/` (untracked) -- `ladder.json`, `m37_dict.npz`, `b2.json`,
  `b4.json`, `memo4.json` (every triple verdict), `j3.json`, `gate.json`, `g4.json`, and the logs
  `ladder.log`, `gate.log`, `gate37.log` (G3), `b4.log`, `j3.log`, `j3_89.log`, `g4.log`
