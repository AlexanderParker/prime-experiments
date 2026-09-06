# Node 4.i.a - THE FRONTIER'S COLLAPSE AT THE TOP: why `Rest(a)` dies as `a` reaches `F(M)`

Parent: node **4.i, the merge forest** (`research/proof/merge_forest.md`, FACT, 2026-09-06), whose
closing section named this child. What spawned it is one line of that branch's section 3.3: the
forest rewrites the record exactly as

    F(M + q') = max over old sizes a of ( a + Rest(a) ),

and the measured frontier `a -> Rest(a)` **rises to the record at an interior `a` of about
0.6-0.7 `F_old` and then collapses**, so that a gap which swallows the old record whole gains at
most 2 columns at m31 (`Rest(F_old) = 3, 2, 3, 7, 7, 5, 5, 2` at the eight rungs). The budget
inequality `F(M + q') <= F(M) + q'` is exactly `Rest(a) <= F_old + q' - a` for every `a`: a line of
slope `-1` from `q' + F_old` at `a = 0` down to `q'` at `a = F_old`. The measured `Rest` lies under
that line everywhere, with its **smallest slack at the interior maximiser and a huge slack at the
top**. So "why does `Rest` collapse at the top" is the mechanism of the budget inequality asked at
the one place where the inequality is loose rather than tight.

Scripts in `research/anchor235/r57/`; result outputs in `research/anchor235/r57/results/`
(untracked). Every number this document relies on is written into the document.

---

## 0. Pre-registered (written before any computation of this branch)

### 0.1 The objects, defined exactly

Machines `M_0 = {5}`, ..., `M_8 = {5..31}`; `q'` the incoming gear; `F_old = F(M)`,
`F = F(M + q')`; `N_old` the number of openings per old period; `m_old(v)` the number of old gaps
of size `v` per old period. `u = 6^{-1} mod q'`, `d = 2u`, letters
`a_L = min(2u mod q', q' - 2u mod q')` and `b_L = q' - a_L` (file 05 T1), so `{0, +-d}` mod `q'`
has least positive representatives `{q', a_L, b_L}`.

By the merge law every gap `G` of `M + q'` is a run of `J >= 1` consecutive old gaps
`(p_1, ..., p_J)` -- its **fusion word** -- whose `J - 1` interior openings are all struck by `q'`
in one copy and whose two outer openings are not. Write

- `mx(G) = max_i p_i` (the **largest piece**), `rest(G) = |G| - mx(G)` (the **rest**);
- `Rest(a) = max { rest(G) : mx(G) = a }`, over every gap of one full period of `M + q'`;
  `Rest(a) = 0` at least, for every realised old size `a` (a survival has `J = 1`);
- the **budget line** `B(a) = F_old + q' - a`, and the **slack profile**
  `s(a) = B(a) - Rest(a) = F_old + q' - a - Rest(a)`.

`F = max_a (a + Rest(a))` exactly, and the budget inequality is `s(a) >= 0` for every `a`. Two
values of `s` are named: the **budget slack** `min_a s(a) = F_old + q' - F` (its minimiser `a*` is
where the record is made) and the **top slack** `s(F_old)`.

Neighbour quantities of the OLD machine, per old gap size `v`:
`n1(v) = max over gaps of size v of max(left neighbour, right neighbour)`;
`N(v) = max over gaps of size v of (left + right)` (the profile of branch 2g.i);
`m_old(v)` as above.

Classification of an attaining fusion word `(p_1, ..., p_J)` with the maximum at index `i`:
**left-heavy** if `sum_{k<i} p_k > sum_{k>i} p_k`, **right-heavy** if `<`, **balanced** if `=`;
and **one-sided** if the maximum is an END piece (`i = 1` or `i = J`), **interior** otherwise.

### 0.2 The theory

**T. The collapse at the top is two different facts wearing one name, and neither is the chain
law's alternation.**

1. As `a` grows the number of old gaps of size `a` collapses (at m29, `m(43) = 2` against
   `m(25)` in the millions). `Rest(a)` is a MAXIMUM over those occurrences, so it collapses with
   the sample, not with any change in what a neighbour looks like. Call this **M3 (rarity)**; it is
   pre-registered here as a third mechanism beside the brief's two.
2. What the chain law contributes is not a difficulty in striking a junction -- by CRT every single
   old opening is struck in exactly two of the `q'` copies, so a ONE-junction (two-piece) fusion is
   essentially always available -- but a difficulty in being an INTERIOR piece: an interior gap of a
   fusion must have size `= 0, +d` or `-d (mod q')` and hence `>= a_L` (file 05 T2). A gap that
   cannot be interior can never claim BOTH of its neighbours, so its `Rest` is a one-sided sum and
   never the neighbour sum `N`. Call this **M2 (chain law)**; its sharp form is the arithmetic
   accident `F_old mod q' in {0, a_L, b_L}`.
3. **M1 (suppression)** -- the old record's neighbours are short -- then acts only on the one-sided
   sum that M2 leaves.

So the prediction is that the collapse is M3 x M2, with M1 present but not separating the regimes,
and that the whole phenomenon exists only from rung 23 on (below it the record IS made at
`a = F_old`, so there is no collapse to explain).

### 0.3 Predictions, each with the number that would refute it

- **P1 (the top slack is an identity, and where the minimum sits).** `s(F_old) = q' - Rest(F_old)`
  identically, so the "value at `a = F_old`" is `4, 9, 10, 10, 12, 18, 24, 29` at rungs 7..31 and
  is not a measurement. NEW: the minimiser of `s` is at `a = F_old` at exactly the rungs where the
  record's largest piece IS the old record (7, 11, 17, 19 by merge_forest 2.3) and in the interior
  at 13, 23, 29, 31. REFUTED by one rung where the two classifications disagree.
- **P2 (shape).** On `a >= a_L`, `s(a)` is unimodal with one interior minimum at rungs 23, 29, 31
  and is NOT convex (some discrete second difference `< 0`) at m31. REFUTED if `s` is convex at any
  of the three top rungs, or if it has two strict local minima on `a >= a_L`.
- **P3 (`a*/F_old` stable).** `a*/F_old in [0.55, 0.75]` at rungs 23, 29, 31 (0.600, 0.676,
  0.581/0.698 on record) and at 15 or more of 20 family members at m13, m17, m19. REFUTED by fewer
  than 15 of 20.
- **P4 (2-piece at the top, 3-piece in the interior).** For `a >= 0.9 F_old` the attaining fusion
  is `J = 2` at all 8 rungs; for `a in [0.5, 0.8] F_old` it is `J >= 3` at rungs 23, 29, 31.
  REFUTED by one counterexample at either end.
- **P5 (one-sided at the top rungs).** At the interior maximiser the largest piece is an END piece
  of the fusion word at rungs 29 and 31, and interior at 23 (word `4 8 15 7`). REFUTED if it is
  interior at two or more of the three.
- **M1 (suppression) test.** `Rest(F_old) = n1(F_old)` -- the attaining rest at the top IS the
  largest single neighbour of an old record -- at 6 or more of 8 rungs. REFUTED if `n1(F_old)`
  exceeds `Rest(F_old)` by `>= 2` at three or more rungs (that would mean legality, not shortness,
  is doing the work).
- **M2 (chain law) test.** `F_old mod q' in {0, a_L, b_L}` at exactly 2 of the 8 rungs (5->7, where
  `F_old = 2 = a_L(7)`, and 13->17, where `F_old = 11 = b_L(17)`); at the other 6 the old record can
  never be an interior piece, hence `Rest(F_old) < N(F_old)` strictly at those 6. REFUTED by a
  fusion whose interior piece is an old record at a rung where the residue condition fails, or by
  `Rest(F_old) = N(F_old)` at such a rung.
  *What CRT gives, exactly:* file 05 (A) -- each old opening is struck in exactly two of the `q'`
  copies, and the map copy -> deletion phase is a bijection. *What it does not give:* that the two
  OUTER openings of the intended fusion are unstruck in one of those two copies (file 05 (C): the
  outer opening at distance `w` from the junction is struck in the same copy iff
  `w = 0, +-d (mod q')`), nor anything at all about a second junction.
- **M3 (rarity) test.** Define the **rarity null** `n1_0(a)` = the largest `r` such that
  `m_old(a) * P(a neighbour of a uniformly random old gap is >= r) >= 1`, i.e. the largest
  neighbour one expects from `m_old(a)` unconditional draws. Predict `|n1(a) - n1_0(a)| <= 3` for
  every `a >= 0.5 F_old` at rungs 23, 29, 31 (rarity explains the top), against M1's prediction that
  `n1(a)` sits well BELOW `n1_0(a)` (conditional shortening). REFUTED for M3 if the gap exceeds 3 at
  more than a third of those cells; REFUTED for M1 if `n1(a) >= n1_0(a) - 1` at more than half.
- **P6 (the failure at 29->31 and the weakened strengthenings).** The max rest 34 at rung 31 sits at
  `a = 21` with sum 55; predict its fusion is `J = 3` with a letter (10 or 21) as the interior piece
  and that it is NOT the run `(18, 10, 30)` that killed the `F + 1` law. Predict `Rest(a) <= q' + 3`
  at all 8 rungs (`c = 3`, attained once, at 29->31) and `Rest(a) <= q'` for `a >= 0.61 F_old` at
  all 8 rungs. REFUTED by `c > 3`, or by a violation of the second form above `0.61 F_old`.
- **P7 (the residual).** The three bounds available -- `Rest(a) <= (J_max - 1) a` (deep-chain cap),
  `Rest(a) <= N(a) <= F_2(M)` for `J <= 3` (the 2g.i law), `Rest(a) <= q'` for `a >= a_0` -- do NOT
  cover the whole range of `a` at m31: predict a non-empty uncovered band, and that the interior
  maximiser lies inside it. REFUTED if the three cover every `a` at m31.
- **P8 (the top buys nothing for the interior).** Predict the record is attained at some `a < a_0`
  at m31 (i.e. the interior maximiser is below the threshold at which the top bound starts), so a
  proof of the collapse at the top leaves `F` untouched. REFUTED if every maximiser is `>= a_0`.

**Stop rules.** Any sub-question that reduces to the merge law or chain law (docs/proofs/05), the
attainment identity (08), the record law (09), the neighbour law `N(v) <= F_2` for `v >= 6`
(2g.i, `neighbour_profile.md`), the glue/shadow/move lemmas (2g.i.a) or the layer decomposition of
the records (R3.h, `ends_or_middles.md`) is stopped in one line and cited.

### 0.4 Scorecard

| # | prediction | verdict | evidence |
|---|---|---|---|
| P1 | `s(F_old) = q' - Rest(F_old)`; minimiser at `a = F_old` iff the record's largest piece is `F_old` | CONFIRMED, 8 of 8 rungs (identity, plus the classification) | 2.1 |
| P2 | `s` unimodal on `a >= a_L` at the top rungs, not convex | REFUTED in the unimodality half (5, 7, 7 strict local minima at 23, 29, 31); CONFIRMED in the convexity half (7, 13, 12 negative 2nd differences) | 2.2 |
| P3 | `a*/F_old` in [0.55, 0.75] at 23, 29, 31 and at 15+/20 family members | CONFIRMED at the three real top rungs (0.600, 0.676, 0.581/0.698); REFUTED on the family (8, 11, 5 of 20) | 2.1, 2.7 |
| P4 | `J = 2` attaining above `0.9 F_old`, `J >= 3` in the interior at the top rungs | REFUTED as stated and replaced by a law: `J = 2` at 12 of 15 top cells, and the 3 exceptions are exactly the `a` that are themselves letters, where `J = 3` puts `a` in the middle -- 15 of 15 under the corrected statement. Interior half CONFIRMED (26 of 39 cells `J >= 3`, all 26 with letter interiors) | 2.3 |
| P5 | largest piece is an END piece at the interior maximiser at 29 and 31 | CONFIRMED exactly as pre-registered (end at 29 and 31, interior at 23) | 2.3 |
| M1 | `Rest(F_old) = n1(F_old)` at 6+ of 8 rungs | CONFIRMED at the 6 rungs where `F_old` is not interior-legal, 6 of 6; at the other 2 it is `N(F_old)` -- the top law | 2.4 |
| M2 | `F_old mod q'` legal at exactly 2 of 8; `Rest(F_old) < N(F_old)` at the other 6 | CONFIRMED, 2 of 8 (rungs 7 and 17) and strict at 6 of 6; sharpened by the fusion-rate identity (137 cells, 0 exceptions) | 2.4 |
| M3 | `\|n1(a) - n1_0(a)\| <= 3` for `a >= 0.5 F_old` at the top rungs | REFUTED (mean deficit 7-11 columns in that range; `has(a)` a tenth of its rarity expectation at `a ~ 0.65 F_old`). Rarity survives as one factor of about 2.3 out of a total 3.2-5.3 | 2.5 |
| P6 | rung-31 max rest is a `J = 3` letter-middle fusion, not `(18, 10, 30)`; `c = 3`; `a_0 = 0.61 F_old` | HALF REFUTED on the fusion (letter-middle yes, but `J = 5`, word `(7, 10, 21, 10, 7)`); CONFIRMED that it is not `(18, 10, 30)`; CONFIRMED `c = 3` and `a_0 = 0.605 F_old` | 2.6 |
| P7 | the three bounds leave an uncovered band at m31 containing the maximiser | CONFIRMED, with one of the three bounds withdrawn as invalid (3.3): the deep-chain cap and the top bound leave `a in [15, 25]` uncovered at m31, and the maximiser `a = 25` is inside it | 3.3, 3.4 |
| P8 | the record is attained at some `a < a_0` at m31 | CONFIRMED (`a = 25 < 26 = a_0`); the collapse at the top buys nothing for the interior, and no monotonicity of `Rest` connects them | 3.4 |

---

## 1. Setup (exact ranges)

Everything exact: full periods or the streamed exact recursion of r56, integer arithmetic, no
sampling.

| object | range | script |
|---|---|---|
| the full frontier `Rest(a)`, its attaining fusion word, and the old machine's `m(v)`, `n1(v)`, `N(v)` | rungs 5->7 .. 19->23 on full periods (7,952,175 gaps at m23) | `fc_frontier.py` |
| the same at rung 23->29 | the whole m29 period (1,078,282,205 columns) streamed as 29 copies of m23 | `fc_frontier.py` |
| the same at rung 29->31 | the whole m31 period (33,426,748,355 columns) by the merge law over the streamed m29 period, every merge of every order, not only the big ones | `fc_top31.py` |
| the rarity null and the neighbour profiles of m23 and m29 | full period / streamed | `fc_frontier.py` |
| the family: frontier, slack shape, minimiser, budget violators | 20 members plus the real machine at 13->17, 17->19, 19->23, full periods each; 400 further members at 13->17 and 17->19 with the incoming gear at its real tooth | `fc_family.py` |
| the three recorded budget violators (node 2f.i), incoming tooth swept | full periods | `fc_violators.py` |
| the summary tables and the exceptionless checks | no new sieving | `fc_analyse.py`, `fc_checks.py`, `fc_checks2.py` |

**Instrument gates, all passed.** The rung-31 pass returns `F = 58`, `N = 6,226,553,025`,
`sum v m = 33,426,748,355` and the order histogram `5,805,160,589 / 413,380,422 / 7,999,018 /
12,992 / 4`, every one matching `merge_forest.md` 2.1-2.2; the whole m31 frontier
`10->19, 11->20, 12->22, ..., 40->7, 43->2` reproduces merge_forest 3.3 cell for cell; and
`max_a (a + Rest_{J>=3}(a)) = 8, 18, 25, 34, 43, 58` at rungs 13..31, which is
`neighbour_profile.md` 2.3's `max_{J>=3} Q*_J` at all six rungs it records.

## 2. Results

### 2.1 The frontier profile and the slack line, every rung (item 1)

`s(F_old) = q' - Rest(F_old)` is an identity, so the top slack is `4, 9, 10, 10, 12, 18, 24, 29`
at rungs 7..31 and carries no new information beyond `Rest(F_old)`. The measured summary:

| rung `q'` | `F_old` | `F` | budget line at `a = 0` = `F_old + q'` | budget slack `min s` | at `a*` | `a*/F_old` | record attained at | `s(F_old)` | `Rest(F_old)` | letters `(a_L, b_L)` | `F_old mod q'` | `F_old` interior-legal? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 7 | 2 | 5 | 9 | 4 | 2 | 1.000 | 2 | 4 | 3 | (2, 5) | 2 | **YES** |
| 11 | 5 | 7 | 16 | 9 | 5 | 1.000 | 5 | 9 | 2 | (4, 7) | 5 | no |
| 13 | 7 | 11 | 20 | 9 | 6 | 0.857 | 6 | 10 | 3 | (4, 9) | 7 | no |
| 17 | 11 | 18 | 28 | 10 | 7 and 11 | 0.636, 1.000 | 7, 11 | 10 | 7 | (6, 11) | 11 | **YES** |
| 19 | 18 | 25 | 37 | 12 | 13 and 18 | 0.722, 1.000 | 13, 18 | 12 | 7 | (6, 13) | 18 | no |
| 23 | 25 | 34 | 48 | 14 | 15 | 0.600 | 15 | 18 | 5 | (8, 15) | 2 | no |
| 29 | 34 | 43 | 63 | 20 | 23 | 0.676 | 23 | 24 | 5 | (10, 19) | 5 | no |
| 31 | 43 | 58 | 74 | 16 | 25 and 30 | 0.581, 0.698 | 25, 30 | 29 | 2 | (10, 21) | 12 | no |

The minimiser of `s` is exactly the `a` at which the record is made, at 8 of 8 rungs (it must be:
`s(a) = F_old + q' - (a + Rest(a))`). It sits at `a = F_old` at rungs 7, 11, 17, 19 and in the
interior at 13, 23, 29, 31 -- and merge_forest 2.3's classification "the record's largest piece IS
the old record" holds at exactly 7, 11, 17, 19. **P1 confirmed, 8 of 8**, with the one wrinkle that
rung 13 has an interior minimiser (`a = 6 = 0.857 F_old`) while its largest piece is 6, not
`F_old = 7`, so the two agree there too. So *the collapse at the top is a phenomenon of the last
three rungs*, 23, 29 and 31, plus rung 13; below that the record IS the old record extended and
there is nothing to explain.

The three top slack profiles in full (`a : s(a)`), the budget line being `s = 0`:

    rung 23 (F_old = 25, budget 48)
      1:47 2:44 3:42 4:41 5:38 6:37 7:34 8:25 9:29 10:20 11:27 12:20 13:22 14:21 15:14
      16:22 17:29 18:17 20:15 21:17 22:21 23:17 25:18
    rung 29 (F_old = 34, budget 63)
      1:62 2:59 3:57 4:56 5:53 6:52 7:49 8:47 9:46 10:34 11:32 12:33 13:28 14:26 15:31
      16:27 17:30 18:28 19:26 20:23 21:25 22:24 23:20 25:26 26:28 27:29 28:27 29:23
      30:28 31:28 32:26 33:28 34:24
    rung 31 (F_old = 43, budget 74)
      1:73 2:70 3:68 4:67 5:64 6:63 7:60 8:58 9:57 10:45 11:43 12:40 13:38 14:37 15:35
      16:33 17:32 18:29 19:27 20:26 21:19 22:19 23:19 24:22 25:16 26:23 27:19 28:23
      29:25 30:16 31:25 32:19 33:21 34:24 35:19 36:29 37:26 38:29 39:32 40:27 43:29

### 2.2 The shape of the slack line (item 5)

| rung | cells (`a >= a_L`) | strict local minima of `s` | convex? | negative 2nd differences | `s(a_L)` | `min s` | `s(F_old)` |
|---|---|---|---|---|---|---|---|
| 7 | 1 | 0 | yes | 0 | 4 | 4 | 4 |
| 11 | 1 | 0 | yes | 0 | 9 | 9 | 9 |
| 13 | 4 | 1 (at 6) | yes | 0 | 12 | 9 | 10 |
| 17 | 5 | 1 (at 7) | no | 2 | 12 | 10 | 10 |
| 19 | 12 | 3 (7, 10, 13) | no | 5 | 21 | 12 | 12 |
| 23 | 16 | 5 (10, 12, 15, 20, 23) | no | 7 | 25 | 14 | 18 |
| 29 | 24 | 7 (11, 14, 16, 20, 23, 29, 32) | no | 13 | 34 | 20 | 24 |
| 31 | 32 | 7 (25, 27, 30, 32, 35, 37, 40) | no | 12 | 45 | 16 | 29 |

**P2 is refuted in its unimodality half and confirmed in its convexity half.** `s` is a saw, not a
valley: 5, 7 and 7 strict local minima at the three top rungs, and 7, 13, 12 negative second
differences. The saw's teeth are the residue classes -- the deep local minima sit at `a` where a
`J >= 3` fusion is available (`a` itself legal, or `a` with a letter-sized neighbour), the local
maxima at `a` where only `J = 2` is available. At m31 the sequence 21:19, 22:19, 23:19, 24:22,
25:16, 26:23 is exactly "deep, deep, deep, shallow, deepest, shallow", and `a = 26` is the first
`a` above the letter band with no `J >= 3` fusion at all.

The location of the minimum is stable in the band [0.55, 0.75] at the three top rungs
(0.600, 0.676, 0.581 and 0.698) but at rungs 7-19 it is at 1.000, 1.000, 0.857, 0.636/1.000,
0.722/1.000. **P3 is confirmed at the real top rungs and refuted on the family** (section 2.7).

### 2.3 The attaining fusions (item 2)

Every entry of the frontier comes with its attaining fusion word. Two exceptionless statements
fall out.

**(A) Large `a`: two pieces, one junction -- unless `a` is a letter.** For `a >= 0.9 F_old`, over
all 8 rungs there are 15 cells; the attaining fusion is `J = 2` at 12 of them, and at the other 3
(`a = 2` at rung 7, `a = 11` at rung 17, `a = 23` at rung 23) `a` is itself interior-legal and the
attaining fusion is `J = 3` with `a` in the MIDDLE. 15 of 15, 0 exceptions:

| rung | `a` | `a/F_old` | `J` | `a` legal? | word |
|---|---|---|---|---|---|
| 7 | 2 | 1.000 | 3 | YES | 1 **2** 2 |
| 11 | 5 | 1.000 | 2 | no | 2 **5** |
| 13 | 7 | 1.000 | 2 | no | 3 **7** |
| 17 | 10 | 0.909 | 2 | no | 4 **10** |
| 17 | 11 | 1.000 | 3 | YES | 5 **11** 2 |
| 19 | 18 | 1.000 | 2 | no | 7 **18** |
| 23 | 23 | 0.920 | 3 | YES | 3 **23** 5 |
| 23 | 25 | 1.000 | 2 | no | **25** 5 |
| 29 | 31, 32, 33, 34 | 0.912-1.000 | 2, 2, 2, 2 | no | **31** 4 / 5 **32** / 2 **33** / **34** 5 |
| 31 | 39, 40, 43 | 0.907-1.000 | 2, 2, 2 | no | **39** 3 / 7 **40** / 2 **43** |

**(B) Interior `a`: three pieces with a letter between.** In the band `a in [0.5, 0.8] F_old`
there are 39 cells at rungs 17..31; 26 are attained by `J >= 3` and 13 by `J = 2`, and in **all 26
of the 26 the interior pieces are letters** (`= 0, +-d mod q'` and `>= a_L`) -- which is forced
(file 05 T2) and so is not the finding; the finding is that the deeper word is available at all
and pays. The record's own word carries the same signature: `4 8 15 7` at m23 (interiors 8 and 15,
the two letters of 23), `10 10 23` at m29 (interior 10 = `a_L(29)`), `23 10 25` and `18 10 30` at
m31 (interior 10 = `a_L(31)`).

**Sidedness.** Classifying the attaining word by where the maximum sits: at the interior maximiser
the largest piece is an END piece at rungs 29 (`10 10 23`, right end, left-heavy) and 31
(`23 10 25` and `18 10 30`, both right end, left-heavy), and an INTERIOR piece at rung 23
(`4 8 15 7`, position 3 of 4, left 12 right 7, left-heavy). **P5 confirmed as pre-registered.**
Over the whole frontier at the three top rungs the left-heavy / right-heavy / balanced counts of
the attaining words are 6 / 16 / 1 (m23, 23 cells), 15 / 17 / 1 (m29, 33 cells) and 18 / 21 / 2
(m31, 41 cells): no side preference at the two deep rungs (the m23 tilt is on 23 cells and is not
read as a signal). The structural asymmetry is in where the MAXIMUM sits, not in which flank is
longer.

### 2.4 The mechanism at the top: the top law (item 3)

> **THE TOP LAW (new, exceptionless, 8 of 8 rungs).**
> `Rest(F_old) = N(F_old)` if `F_old = 0, +d` or `-d (mod q')` (and `>= a_L`), and
> `Rest(F_old) = n1(F_old)` otherwise.
> In words: *a gap of the old record's size can gain only its single largest neighbour -- unless
> its own size is one of the incoming gear's letters, in which case it can gain both.*

| rung | `F_old` | interior-legal? | `Rest(F_old)` | `n1(F_old)` | `N(F_old)` | predicted | match |
|---|---|---|---|---|---|---|---|
| 7 | 2 | YES | 3 | 2 | 3 | 3 | yes |
| 11 | 5 | no | 2 | 2 | 3 | 2 | yes |
| 13 | 7 | no | 3 | 3 | 4 | 3 | yes |
| 17 | 11 | YES | 7 | 5 | 7 | 7 | yes |
| 19 | 18 | no | 7 | 7 | 10 | 7 | yes |
| 23 | 25 | no | 5 | 5 | 7 | 5 | yes |
| 29 | 34 | no | 5 | 5 | 6 | 5 | yes |
| 31 | 43 | no | 2 | 2 | 4 | 2 | yes |

So merge_forest's measured `Rest(F_old) = 3, 2, 3, 7, 7, 5, 5, 2` is decoded: it is the old
record's **largest single neighbour** at six rungs and its **neighbour sum** at the two rungs where
`F_old` happens to be a letter of the incoming gear (`2 = a_L(7)`, `11 = b_L(17)`). Both branches
of the law are small because the old record's neighbours are short.

**Why no third piece.** A `J >= 3` fusion with the old record at an END needs the record's own
neighbour to be an INTERIOR gap of the fusion, hence `= 0, +-d (mod q')` and `>= a_L` (file 05 T2).
Measured: **no occurrence of the old record has a letter-sized neighbour at 7 of the 8 rungs**
(`has(F_old) = 0`; the exception is rung 7, where `a_L = 2` and the neighbour 2 is itself the
letter). At 6 of those 7 the obstruction is SIZE -- `n1(F_old) < a_L`:

| rung | 7 | 11 | 13 | 17 | 19 | 23 | 29 | 31 |
|---|---|---|---|---|---|---|---|---|
| `n1(F_old)` | 2 | 2 | 3 | 5 | 7 | 5 | 5 | 2 |
| `a_L(q')` | 2 | 4 | 4 | 6 | 6 | 8 | 10 | 10 |
| `n1 < a_L`? | no | yes | yes | yes | **no** | yes | yes | yes |

and at rung 19 the obstruction is RESIDUE instead: `n1(18) = 7 >= 6 = a_L`, but 7 is not
`0, +-d (mod 19)` (the letters are 6 and 13), so the 7-neighbour is still illegal as an interior
gap. Both obstructions are of the same shape and together they are exceptionless.

**The two regimes side by side** (the brief's three quantities at `a = F_old` and at
`a` about `0.65 F_old`, plus the availability gate and the attaining order):

| rung | `a` | `a/F_old` | `m(a)` | `n1(a)` | `N(a)` | `n1_0(a)` | occurrences with a letter-sized neighbour | `Rest(a)` | `J` |
|---|---|---|---|---|---|---|---|---|---|
| 23 | 15 (the record's `a`) | 0.600 | 1,236 | 13 | 17 | 21 | **62** | 19 | 4 |
| 23 | 16 | 0.640 | 876 | 10 | 17 | 20 | **0** | 10 | 2 |
| 23 | 25 = `F_old` | 1.000 | 20 | 5 | 7 | 12 | **0** | 5 | 2 |
| 29 | 22 | 0.647 | 2,314 | 15 | 18 | 25 | **44** | 17 | 3 |
| 29 | 23 (the record's `a`) | 0.676 | 5,598 | 14 | 16 | 26 | **32** | 20 | 3 |
| 29 | 34 = `F_old` | 1.000 | 4 | 5 | 6 | 8 | **0** | 5 | 2 |
| 31 | 25 (the record's `a`) | 0.581 | 88,548 | 30 | 37 | 33 | **1,858** | 33 | 3 |
| 31 | 28 | 0.651 | 24,418 | 23 | 27 | 32 | **230** | 23 | 3 |
| 31 | 30 (the record's `a`) | 0.698 | 10,862 | 25 | 28 | 30 | **92** | 28 | 3 |
| 31 | 43 = `F_old` | 1.000 | 2 | 2 | 4 | 7 | **0** | 2 | 2 |

Every column falls from the interior to the top, so no single one of them "separates the regimes"
by falling; what separates them is that **one column reaches zero and the others do not**. The
letter-neighbour count is 62, 32, 1,858, 230, 92 in the interior and 0 at `a = F_old` at every
rung, and it is exactly that zero which removes the second junction and pins `Rest` to
`Rest_2`. `m` and `n1` fall by factors of 62-44,274 and 2.6-15 respectively, but a factor is not
a mechanism; a zero is.

**What CRT gives and what it does not, exactly.** By file 05 (A) each old opening is struck by `q'`
in exactly two of the `q'` copies, and the copies realise every deletion phase once. That is enough
to make a junction available at EVERY occurrence of every gap, and it yields a closed identity we
verified over **137 (rung, size) cells with 0 exceptions**:

> **THE FUSION-RATE IDENTITY (new, exceptionless).** Over the `q'` copies, an old gap of size `v`
> lies inside a merged (`J >= 2`) gap in exactly `4` copies if `v != 0, +-d (mod q')`, in `3` if
> `v = +-d`, and in `2` if `v = 0 (mod q')`; and it is an INTERIOR piece in exactly `0`, `1`, `2`
> copies respectively.

(Proof in a line: the gap's two openings are struck in 2 copies each; the two pairs coincide in
0, 1 or 2 copies according as `v` is generic, `+-d`, or `0` mod `q'` -- the chain law, file 05 (C).
Fused = union, interior = intersection.) So CRT gives fusibility unconditionally, at a fixed rate
`4/q'`; what it does NOT give is anything about the SIZE of the piece on the far side of the
junction, nor about a second junction. Both of those are the machine's, not CRT's.

### 2.5 Rarity against suppression (item 3, the separating quantity)

Two candidate reasons for the collapse were pre-registered as M1 (the neighbours are short) and M3
(there are too few occurrences to draw a long neighbour from). Both are measurable against the
rarity null `n1_0(a)`, the largest neighbour expected from `2 m(a)` unconditional draws.

| rung | `a/F_old` band | cells | mean `n1` | mean `n1_0` | mean deficit | max deficit |
|---|---|---|---|---|---|---|
| 23 | [0, 0.25) | 6 | 22.5 | 25.0 | 2.5 | 10 |
| 23 | [0.25, 0.5) | 6 | 17.2 | 23.7 | 6.5 | 10 |
| 23 | [0.5, 0.75) | 6 | 10.5 | 19.2 | 8.7 | 10 |
| 23 | [0.75, 1] | 5 | 7.0 | 13.8 | 6.8 | 10 |
| 29 | [0, 0.25) | 8 | 31.2 | 32.9 | 1.6 | 5 |
| 29 | [0.25, 0.5) | 8 | 22.1 | 30.0 | 7.9 | 12 |
| 29 | [0.5, 0.75) | 8 | 14.9 | 25.6 | 10.8 | 13 |
| 29 | [0.75, 1] | 9 | 5.9 | 14.0 | 8.1 | 13 |
| 31 | [0, 0.25) | 10 | 38.0 | 39.5 | 1.5 | 5 |
| 31 | [0.25, 0.5) | 11 | 32.1 | 36.4 | 4.3 | 8 |
| 31 | [0.5, 0.75) | 11 | 22.9 | 30.3 | 7.4 | 13 |
| 31 | [0.75, 1] | 9 | 9.9 | 16.8 | 6.9 | 13 |

**M3 as pre-registered is REFUTED and M1 CONFIRMED, with a correction to both.** The deficit is
0-2 columns at `a < 0.25 F_old` and 7-11 columns from `a >= 0.5 F_old`, far outside the
pre-registered `<= 3`; so rarity alone does not explain the observed neighbour sizes. But rarity is
not idle either: the null itself falls by a factor about 2.3 across the range (25.0 -> 13.8,
32.9 -> 14.0, 39.5 -> 16.8) while the observation falls by 3.2, 5.3, 3.8. **The collapse is the
product of two factors of comparable size: a rarity factor of about 2.3 and a suppression factor of
about 1.6-2.3.** Neither is the whole of it, and the pre-registered "whichever quantity separates
the two regimes" has the answer: at the very top it is the ABSOLUTE size of `n1` against the letter
`a_L` that switches the regime, because it decides whether a third piece exists at all.

The same test on the letter-neighbour channel (the quantity that actually gates `J >= 3`),
`has(a)` against `m(a) * pbar` with `pbar` the machine-wide rate:

| rung | `pbar` | `a` | `m(a)` | `has(a)` | expected | ratio |
|---|---|---|---|---|---|---|
| 23 | 0.0616 | 21 / 22 / 23 / 25 | 48 / 26 / 86 / 20 | 0 / 0 / 0 / 0 | 3.0 / 1.6 / 5.3 / 1.2 | 0 |
| 29 | 0.0595 | 23 / 25 / 28 / 34 | 5,598 / 1,404 / 322 / 4 | 32 / 8 / 0 / 0 | 333 / 84 / 19 / 0.24 | 0.10 / 0.10 / 0 / 0 |
| 31 | 0.0726 | 30 / 34 / 40 / 43 | 10,862 / 548 / 8 / 2 | 92 / 2 / 0 / 0 | 788 / 40 / 0.6 / 0.15 | 0.12 / 0.05 / 0 / 0 |

At `a = F_old` the expected count is below 1 at every rung (0.15 at m31), so `has(F_old) = 0` is
what rarity alone predicts; but at `a` around `0.65 F_old` the expectation is in the hundreds and
the observation is a tenth of it -- suppression again, and by a factor 10, not 2.

**The regime switch, stated exactly, and exceptionless (40 cells, 0 exceptions):**

> if `a` is not interior-legal and no occurrence of size `a` has a letter-sized neighbour, then
> **no `J >= 3` fusion exists with `a` as the largest piece**, so `Rest(a) = Rest_2(a) =` the
> largest neighbour of an `a`-gap that is at most `a`.

The `a` for which this happens are exactly the top of the spectrum: at m23 `a in {16, 17, 21, 22,
25}`, at m29 `a in {17, 26, 27, 28, 30, 31, 32, 33, 34}`, at m31 `a in {36, 37, 38, 39, 40, 43}`.

### 2.6 The failure at 29 -> 31 (item 4)

`max_a Rest(a) = 34 > 31 = q'` at rung 31; it is the only rung where `Rest <= q'` fails.

| rung | 7 | 11 | 13 | 17 | 19 | 23 | 29 | 31 |
|---|---|---|---|---|---|---|---|---|
| max `Rest` | 3 | 2 | 5 | 11 | 13 | 19 | 23 | **34** |
| at `a` | 2 | 2, 3, 5 | 5, 6 | 7 | 7, 10 | 15 | 14 | **21** |
| `q'` | 7 | 11 | 13 | 17 | 19 | 23 | 29 | 31 |
| max `Rest - q'` | -4 | -9 | -8 | -6 | -6 | -4 | -6 | **+3** |

**The attaining gap.** `a = 21`, `rest = 34`, size 55, order **`J = 5`**, fusion word
**`(7, 10, 21, 10, 7)`** -- a palindrome whose three interior pieces `10, 21, 10` are the two
letters of 31 -- at m29 position 220,171,095, multiplicity `w = 1` (one of the 31 phases). It is
NOT the run `(18, 10, 30)`: that run is the m31 record class, `a = 30`, `rest = 28`, at m29
position 278,620,515. And `(7, 10, 21, 10, 7)` is not new to the project: it is exactly
`neighbour_profile.md` 1's recorded `Q*_5` maximiser at m29, so the object attaining the failure is
already on the register under another name -- cited, not re-derived. **P6's fusion prediction is
half right**: it is a letter-middle fusion but of order 5, not 3.

`Rest(a) <= q'` fails at exactly four `a` at rung 31: `21 -> 34`, `22 -> 33`, `23 -> 32`,
`25 -> 33`; all four are `<= 0.58 F_old`. What the strengthening would have needed there: 3 fewer
columns at `a = 21`. The weakened forms:

- **`Rest(a) <= q' + c` holds at all 8 rungs with `c = 3`**, attained once (rung 31, `a = 21`).
  On its own it gives the budget only for `a <= F_old - 3`; together with the top law it leaves the
  three cells `a in {F_old - 2, F_old - 1, F_old}` to be covered separately, where the measured
  `Rest` is 2-5 at every rung (m29: `32 -> 5`, `33 -> 2`, `34 -> 5`; m31: 41 and 42 unrealised,
  `43 -> 2`).
- **`Rest(a) <= q'` for `a >= a_0` holds at all 8 rungs with `a_0 = 0.605 F_old`** (`a_0 = 26` at
  rung 31, `a_0 = 1` at every other rung: there is nothing to exclude). **P6's `0.61` confirmed.**

### 2.7 The family (item 5)

20 members plus the real machine at each of 13->17, 17->19, 19->23, full periods.

- **Not one member of 63 has a convex slack line.** Negative second differences 1-7 per member.
- **The minimiser's location is NOT stable across the family.** `a*/F_old` in [0.55, 0.75] at
  8 of 20 (13->17), 11 of 20 (17->19), 5 of 20 (19->23) -- **P3 refuted on the family** (predicted
  15 of 20). The commonest single family value is `a*/F_old = 1.000`, 11 of the 60 members
  (9, 1, 1 at the three rungs):
  a typical member's frontier does NOT collapse at the top, it peaks there.
- **Budget violators break at the top, or at a letter, and always at an interior-legal `a`.** One
  violator appeared among the 63 shape members (17->19, teeth `(1,2,1,4,4,5)`: `F_old = 14`,
  `F = 34` against a budget of 33, `s(10) = -1`), one more in 400 members with the incoming gear at
  its real tooth (0.50% at 17->19, 0 of 400 at 13->17), and the three violators recorded on the tree
  (node 2f.i) were reproduced exactly by sweeping the incoming tooth:

| member | `F_old` | `q'` | `F` | budget | `s < 0` at `a` | `a/F_old` | `a` interior-legal? |
|---|---|---|---|---|---|---|---|
| `{5..17}` teeth (1,3,4,4,4), `v_19 = 4` | 19 | 19 | 40 | 38 | 19 | 1.000 | YES (`19 = 0 mod 19`) |
| `{5..17}` teeth (2,3,3,3,3), `v_19 = 3` | 18 | 19 | 38 | 37 | 13 | 0.722 | YES (`13 = b_L`) |
| `{5..11}` teeth (1,1,5), `v_13 = 1` | 11 | 13 | 25 | 24 | 11 | 1.000 | YES (`11 = b_L`) |
| `{5..17}` teeth (1,3,2,1,6), `v_19 = 3` | 15 | 19 | 35 | 34 | 13 | 0.867 | YES (`13 = b_L`) |
| `{5..17}` teeth (1,2,1,4,4), `v_19 = 5` | 14 | 19 | 34 | 33 | 10 | 0.714 | YES (`10 = b_L`) |

> **5 of 5 budget violators break at an `a` that the chain law allows to be an interior piece**,
> and 2 of 5 break at `a = F_old` exactly -- the place where the real machine has its largest
> slack. The counterfactual machines that break the budget are precisely the ones whose frontier
> does NOT collapse at the top: at the first violator `Rest(F_old) = 21` against the real
> machine's `<= 7`.

## 3. Mechanism

### 3.1 The collapse, in the machine's terms

A gap of the new machine is a run of old gaps whose interior openings the new gear strikes. To make
a long one out of a big old gap `a` you must add pieces on one or both sides, and the machine
imposes three separate tolls, in this order:

1. **The junction is free.** By CRT the joint opening is struck in exactly 2 of the `q'` copies, and
   the whole gap is fused in exactly 4 (fusion-rate identity, 137 cells, 0 exceptions). Nothing at
   the top of the spectrum is rare because a junction is hard to find.
2. **The second junction is not free.** A second junction turns the first added piece into an
   INTERIOR gap, and an interior gap must be `= 0, +-d (mod q')` and hence `>= a_L ~ q'/3`
   (file 05 T2). So a third piece exists only if the big gap has a neighbour of letter size.
3. **The old record has no such neighbour** -- at 7 of 8 rungs no occurrence of it does, because its
   neighbours are shorter than `a_L` (6 rungs) or of the wrong residue (rung 19). Hence only the
   one-junction fusion survives, and its yield is the single largest neighbour: the top law.

The collapse is therefore not a weakening of the fusion mechanism at the top; it is the **letter
floor `a_L ~ q'/3` colliding with the shortness of a record's neighbours**. The two quantities
that meet are `n1(F_old)` (measured 2, 2, 3, 5, 7, 5, 5, 2) and `a_L(q')` (2, 4, 4, 6, 6, 8, 10,
10): one is bounded and the other grows like `q'/3`, so the collision gets more decisive with every
rung, and `s(F_old) = q' - n1(F_old)` grows like `q'`. That is the shape of the whole phenomenon.

At the interior maximiser everything is the other way round: `a ~ 0.65 F_old` is common
(`m(a)` in the thousands at m29 and the tens of thousands at m31), so some occurrence does have a
letter-sized neighbour, the second junction opens, and the word becomes `(big, letter, bigger)` --
the recorded record words `10 10 23`, `23 10 25`, `18 10 30`, all with the letter `a_L` as the
interior piece. The record is made where the two requirements -- a big piece and a letter-sized
neighbour -- first stop being mutually exclusive.

### 3.2 Why the slack line is a saw, not a valley

`s(a)` has 5, 7 and 7 strict local minima at the three top rungs. The teeth are not noise: `s` dips
wherever a `J >= 3` fusion is available at `a` and rises wherever only `J = 2` is. Availability is a
residue-and-size condition (`a` itself legal, or an occurrence of `a` with a letter-sized
neighbour), so the saw's period is the letters' arithmetic, not anything smooth. Two consequences:

- No convexity or unimodality argument can be used on `s`. **A proof must be case-split on
  availability, not on the size of `a`.**
- The correct coordinate is not `a` but the pair (`a`, is a `J >= 3` fusion available at `a`).

### 3.3 What the neighbour law does and does not bound (a correction)

`neighbour_profile.md`'s law `N(v) <= F_2(M)` for `v >= 6` bounds the sum of the two IMMEDIATE
neighbours. It bounds `Rest(a)` only for `J = 2` (one neighbour) and for `J = 3` with `a` in the
MIDDLE. For `J = 3` with `a` at an END the rest reaches two steps away, and it does exceed `N(a)`:
7 cells over the 8 rungs have `Rest(a) > N(a)`, 5 of them at `J = 3` (m23 `a = 20`, word `5 8 20`,
13 > 11; m29 `a = 20` and `a = 23`, words `10 10 20` and `10 10 23`, 20 > 18 and 20 > 16; m31
`a = 32` and `a = 34`) and 2 at `J >= 4` (m23 `a = 15`, word `4 8 15 7`; m31 `a = 21`, word
`7 10 21 10 7`, 34 > 30). **So the 2g.i law is not a bound on the frontier**, and the branch's
pre-registration was wrong to list it as one.

What IS an identity, at 7 of 8 rungs:

> **`max_a (a + Rest_2(a)) = F_2(M)`** -- the two-piece half of the frontier is exactly the largest
> adjacent pair of the old machine. (Values: 3, 7, 11, 16, 25, 31, 39, 55 against
> `F_2 = 3, 8, 11, 16, 25, 31, 39, 55`; the single failure is rung 11, where the `F_2` pair `3 + 5`
> is not legally fusible.)

and, as a gate rather than a finding, `max_a (a + Rest_{J >= 3}(a)) = 8, 18, 25, 34, 43, 58` at
rungs 13..31, which is `max_{J>=3} Q*_J` on record (2g.i 2.3), 6 of 6. So the frontier splits along
the attainment identity (docs/proofs/08): **the `J = 2` half of the frontier is the PAIR statement
(node 1) and the `J >= 3` half is the CHAIN statement (node 2)**, in the frontier's coordinates.
That placement is a known result restated and is stopped here in one line; what is new is that the
frontier says at WHICH `a` each obligation binds -- the pair statement binds at `a = 35` at m31
(not at the record), the chain statement at `a = 25` and `a = 30`.

### 3.4 The smallest statement that gives the budget, and the exact residual (item 6)

The budget is `a + Rest(a) <= F_old + q'` for every `a`. Split by availability:

- **`J = 2` part.** `a + Rest_2(a) <= F_2(M)` (in fact with equality at the maximiser, 7 of 8), so
  the whole two-piece half of the budget is the pair statement `F_2(M) <= F(M) + q'` -- node 1,
  OPEN, free through m31 (`F_2 - F = 4, 5, 7, 6, 5, 12, 10` against `q' = 7, ..., 31`). **The
  collapse at the top lives entirely inside this part**: `a = F_old` is a `J = 2` cell at 6 of 8
  rungs.
- **`J >= 3` part.** This is the chain statement, node 2, and nothing here bounds it.

So, stated minimally: **the budget follows from (i) the pair statement and (ii)
`a + Rest_{J>=3}(a) <= F_old + q'`.** Proved parts: the fusion-rate identity (proved here from
file 05 (A) and (C)); the interior-gap floor `>= a_L` and the alternation grammar (file 05 T2-T3,
kernel); `max order = 1 + D_{q'}` (merge_forest 3.1, proved); the deep-chain cap
`Rest(a) <= (J_max - 1) a`, which covers `a <= (F_old + q')/J_max` = 14 at m31. Measured only: the
top law, the neighbour shortness `n1(F_old) <= 7` at 8 of 8 rungs, and the suppression deficit.

**The exact residual, at m31.** Deep-chain cap covers `a <= 14`; the top bound `Rest <= q'` covers
`a >= 26`; the uncovered band is **`a in [15, 25]`**, i.e. `0.35 F_old` to `0.58 F_old`, and it
contains one of the two record maximisers (`a = 25`, the word `23 10 25`, `a + Rest = 58 = F`).
At every other rung the uncovered band is empty because `Rest <= q'` holds everywhere.

**Does the collapse at the top buy anything for the interior? No, directly.** Proving
`Rest(F_old) <= n1(F_old)` and `n1(F_old)` small settles `a` near `F_old`, where the slack is
already 16-29 columns; the record is made 12-18 columns lower, in the uncovered band, where a
different fusion order attains. What WOULD connect them is a monotonicity: **if `Rest(a)` were
non-increasing in `a` above the last `a` at which a `J >= 3` fusion is available, the top law would
propagate downward.** Measured, `Rest` is not monotone anywhere (m31: `26 -> 25`, `27 -> 28`,
`28 -> 23`, `29 -> 20`, `30 -> 28`), and the reason is exactly the saw of 3.2. The honest
connection is the other one, and it is the branch's contribution toward the root:

> the availability gate is monotone even though `Rest` is not. `has(a) = 0` at every `a` above
> 35 at m31, above 25 at m29, above 20 at m23 (with the legal `a` excepted), and once `has(a) = 0`
> the frontier is `Rest_2` and hence the pair statement. **The obligation therefore reduces to: for
> which `a` is `has(a) > 0`?** -- a statement about the old machine alone, of the same shape as
> `N(v) <= F_2` but about the RESIDUE of a neighbour rather than its size.

### 3.5 The family reading

The counterfactual family says the collapse is not decorative. Its budget violators (5 of 5) break
at interior-legal `a`, and 2 of 5 break at `a = F_old` with `Rest(F_old) = 21` and `14` -- values
the real machine never approaches (its maximum over 8 rungs is 7). So **the real machine's top-law smallness is
what those violating machines lack**, and the common feature of all five is weaker and sharper:
they break at an `a` the chain law lets be an interior piece. The family also kills the shape as a
universal: `a*/F_old = 1.000` is the modal family value (11 of 60), and no member's slack line is
convex.

## 4. What is new

1. **The top law**, exceptionless at 8 of 8 rungs: `Rest(F_old) = N(F_old)` when
   `F_old = 0, +-d (mod q')` and `= n1(F_old)` otherwise. It decodes merge_forest's
   `Rest(F_old) = 3, 2, 3, 7, 7, 5, 5, 2` into one measured quantity of the OLD machine (the
   record's largest neighbour) and one arithmetic accident (`F_old` being a letter).
2. **The fusion-rate identity**, proved from file 05 and verified at 137 (rung, size) cells with 0
   exceptions: an old gap is fused in exactly `4/q'`, `3/q'` or `2/q'` of the copies, and is an
   interior piece in `0`, `1/q'` or `2/q'`, by its residue class mod `q'`. This is the exact
   content of "CRT makes a junction available", and it shows that junction availability is NOT what
   collapses at the top.
3. **The availability gate**: `a` not interior-legal and no occurrence of `a` with a letter-sized
   neighbour implies no `J >= 3` fusion at `a` (40 cells, 0 exceptions), so `Rest(a) = Rest_2(a)`.
   The set of such `a` is exactly the top of the spectrum at each of the three top rungs.
4. **The letter floor against neighbour shortness** as the mechanism of the collapse:
   `n1(F_old) = 2, 2, 3, 5, 7, 5, 5, 2` against `a_L = 2, 4, 4, 6, 6, 8, 10, 10`; one bounded, one
   growing like `q'/3`, so `s(F_old) = q' - n1(F_old)` grows like `q'`.
5. **The frontier is a saw, not a valley**: 5, 7, 7 strict local minima and 7, 13, 12 negative
   second differences at the top rungs, with the teeth at the residue-legal `a`. No member of the
   family (63 of 63) has a convex slack line. This rules out convexity and unimodality arguments.
6. **`max_a (a + Rest_2(a)) = F_2(M)`** at 7 of 8 rungs, which locates the pair statement inside the
   frontier and shows that the collapse at the top is an event in the pair statement's half.
7. **The rung-31 failure decomposed**: max rest 34 at `a = 21`, word `(7, 10, 21, 10, 7)`, order 5,
   at m29 position 220,171,095 -- the recorded `Q*_5` maximiser, not the `(18, 10, 30)` run; the
   weakened forms `Rest <= q' + 3` (8 of 8) and `Rest <= q'` for `a >= 0.605 F_old` (8 of 8).
8. **The family's budget violators all break at an interior-legal `a`** (5 of 5), and two of them
   break at `a = F_old` with `Rest(F_old)` three times the real machine's worst.

Prior art inside the project: the frontier and `Rest(F_old) <= 7` are merge_forest 3.3 (the parent);
`N(v) <= F_2` for `v >= 6` and the `(18, 10, 30)` run are 2g.i; `(7, 10, 21, 10, 7)` is the recorded
`Q*_5` maximiser (2g.i 1); the attainment split `J = 2` / `J >= 3` is docs/proofs/08; the
suppression of neighbours of large gaps is docs/novel/suppression-law.md (a conditional-mean
statement; the extremal form `n1(F_old) <= 7` is not on it). Outside: not checked (no web access).

## 5. Verdict

**FACT, exact, with one new law and one new identity; a partial route, and it points at a different
statement from the one the branch was opened to prove.**

- The question "why does `Rest` collapse at the top" has a complete answer at 8 of 8 rungs: the top
  law, with the letter floor as its mechanism. The collapse is real, it is a property of the old
  machine's neighbour profile meeting `a_L ~ q'/3`, and it is what 2 of the 5 budget-violating
  counterfactuals lack (they break at `a = F_old` with `Rest(F_old) = 21` and `14`); the other
  three break in the interior, at an `a` that is itself a letter.
- It does **not** close the budget, because the record is made 12-18 columns below the top, in the
  uncovered band `a in [15, 25]` at m31, where the fusion order is 3 or 5 and the availability gate
  is open. Proving the top law would buy nothing there, and no monotonicity connects them: `Rest` is
  not monotone in `a` (the saw).
- **What the branch hands forward** is a reduced obligation with a cleaner shape than `Rest`: the
  availability gate `has(a) > 0` -- for which old sizes `a` does some occurrence have a neighbour of
  size `= 0, +-d (mod q')` and `>= a_L`? It is a statement about `M` alone; it is monotone at the
  top where `Rest` is not; and once it is 0 the frontier is `Rest_2`, i.e. the pair statement. That
  is the child this branch names.

## 6. Dead ends, each with its refuting instance

- **`s(a)` is unimodal, so the budget is a one-minimum problem.** Refuted: 5, 7 and 7 strict local
  minima at rungs 23, 29, 31 (m31 minima at `a = 25, 27, 30, 32, 35, 37, 40`).
- **`s(a)` is convex.** Refuted at every rung from 17 up, and at 63 of 63 family members.
- **Rarity alone (M3) explains the collapse.** Refuted: the deficit `n1_0 - n1` is 7-11 columns on
  average for `a >= 0.5 F_old`, against a pre-registered tolerance of 3; and `has(a)` is a tenth of
  its rarity expectation at `a ~ 0.65 F_old` (32 against 333 at m29).
- **The neighbour law `N(v) <= F_2` bounds `Rest(a)`.** Refuted: 7 cells have `Rest(a) > N(a)`,
  5 of them at `J = 3` with `a` at an end (m29 `a = 23`: `Rest = 20 > 16 = N`).
- **The attaining fusion at large `a` is always two-piece.** Refuted at 3 of 15 cells, and refuted
  in a way that is a law: the exceptions are exactly the `a` that are themselves letters
  (`a = 2` at rung 7, `11` at 17, `23` at 23), where a three-piece fusion puts `a` in the middle.
- **The rung-31 failure is a three-piece letter-middle fusion.** Refuted: it is `J = 5`,
  `(7, 10, 21, 10, 7)`.
- **`a*/F_old` is a stable machine invariant.** Refuted on the family (8, 11, 5 of 20 in the band at
  the three rungs; the plurality value is 1.000) and at the real rungs below 23 (1.000, 1.000,
  0.857, 0.636, 0.722).
- **Proving the collapse at the top helps the interior.** Refuted by measurement: at m31 the record
  is attained at `a = 25`, inside the band that no available bound covers, while the top bound only
  starts at `a = 26`.

## 7. What holds without exception (item 7)

| statement | count | status |
|---|---|---|
| the fusion-rate identity `4/q'`, `3/q'`, `2/q'` and interior rate `0`, `1/q'`, `2/q'` | 137 (rung, size) cells, 8 rungs | proved here from file 05 (A), (C) + verified |
| the top law: `Rest(F_old) = N(F_old)` if `F_old` interior-legal, else `n1(F_old)` | 8 of 8 rungs | measured |
| `has(F_old) = 0` (no occurrence of the old record has a letter-sized neighbour) | 7 of 8 rungs (rung 7 excepted) | measured |
| `n1(F_old) < a_L` or (`n1(F_old) >= a_L` and no neighbour of legal residue) | 8 of 8 rungs | measured |
| the availability gate: not legal and `has(a) = 0` implies no `J >= 3` fusion | 40 cells, 0 exceptions | proved (file 05 T2) + verified |
| `a >= 0.9 F_old` implies the attaining fusion is `J = 2`, unless `a` is interior-legal | 15 of 15 cells | measured |
| in the band `a in [0.5, 0.8] F_old`, every interior piece of every attaining word is a letter | 26 of 26 `J >= 3` cells (of 39) | forced (file 05 T2) + verified |
| `s(F_old) = q' - Rest(F_old)` | 8 of 8 | identity |
| `Rest(a) <= q' + 3` | 8 of 8 rungs | measured (tight once) |
| `Rest(a) <= q'` for `a >= 0.605 F_old` | 8 of 8 rungs | measured |
| `max_a (a + Rest_2(a)) = F_2(M)` | 7 of 8 (rung 11 fails, 7 against 8) | measured |
| `max_a (a + Rest_{J>=3}(a)) = max_{J>=3} Q*_J` | 6 of 6 rungs on record | gate against 2g.i |
| family budget violators break at an interior-legal `a` | 5 of 5 | measured |
| no convex slack line | 63 of 63 family members; 5 of 5 real rungs from 17 up (rungs 7 and 11 have one cell, rung 13 four) | measured |
