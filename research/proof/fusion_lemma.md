# Node 4.i.b.ii.c - THE FUSION LEMMA: PROVE IT, OR FIND THE EXACT OBSTRUCTION

Parent: node **4.i.b.ii, the monotone functional** (`research/proof/monotone_functional.md`),
whose theorem 6.2 says `B_{L+1}(M; q') <= F(M) + q'` for every engine gives the budget at every
step, and whose section 6.3 leaves two costs: a cap on `L` (worked in 4.i.b.ii.a, `pad_cap.md`,
ROOT on the skip half) and one lemma, "the level-`(J_max - 1)` relaxed deepest fusion is within
budget", with its smallest instance `(25, 10, 25)` at `23 -> 29` (`60 <= 63`) and, at
`37 -> 41` (`order_law_37_41.md`), `(21, 14, 41, 22)` at phase 20 (`98 <= 129`). Spawned by the
observation of 4.i.b.ii.b that the overshoot `B_3 - F(41) = 98 - 91 = 7` is "one substitution,
in one slot": the machine realises `(21, 14, 41)` and `(14, 41, 22)` and never realises them
overlapping. This branch asks whether that is what the relaxation always is, and whether the
lemma can be proved from the covering-problem characterisation of `D_k` membership.

Scripts in `research/anchor235/r73/` (prefix `fl_`); outputs in `research/anchor235/r73/results/`
(untracked). Every number this document relies on is written into the document.

---

## 0. Pre-registered (written before any computation of this branch)

### 0.1 The objects, by construction

- **Engine** `M = {5..q}`: the primes `5..q` acting on columns over the anchor `2, 3, 5`. Column
  `k` is the slot `(6k - 1, 6k + 1)`; gear `g` strikes column `k` iff `k = +-u_g (mod g)`,
  `u_g = 6^{-1} mod g`. An **opening** is a column no gear strikes; the openings are periodic
  with period `P = prod M` and there are `N` per period; a **gap** is the distance between
  consecutive openings; `F(M)` is the largest gap.
- **The next gear** `q'` (the prime after `q`), `u = 6^{-1} mod q'`, `d = 2u mod q'`. At a
  **phase** `z in Z_{q'}` the gear strikes offset `o` iff `o + z in {0, d} (mod q')`: two
  residue classes, the two **teeth**, at distance `d`.
- **Letters.** A gap value `v` is `PAD` if `v = 0`, `UP` if `v = d`, `DOWN` if `v = -d`
  (mod `q'`), else `BAD`. A word of gaps is **legal** iff every letter is `PAD`/`UP`/`DOWN`
  and no two consecutive nonzero letters are equal (docs/proofs/05 (F)); equivalently
  (docs/proofs/10 Theorem 1) its `|w| + 1` openings all sit on the two teeth of ONE phase.
  A legal word with a nonzero letter has exactly one **consistent start tooth**, hence one
  phase; an all-pad word has two.
- `D_k(M)`: the set of realised `k`-windows, i.e. `k` consecutive gaps that occur in `M`.
  Membership is decidable exactly by the covering problem of `order_law_37_41.md` 2.2.
- `L = L(M)`: the length of the longest realised legal word; `J_max = L + 2`
  (docs/proofs/10). A **maximal word** is a realised legal word of length exactly `L`.
- A `J`-**fusion at phase `z`**: `J` consecutive gaps with openings at offsets
  `o_0 = 0 < o_1 < ... < o_J`, with `o_1 .. o_{J-1}` struck at phase `z` and `o_0`, `o_J` not.
  `Q*_J(M; q')` = the largest span of a REALISED `J`-fusion; the record law (docs/proofs/09):
  `F(M + q') = max_{J <= J_max} Q*_J`.
- **Level-`k` admissible** `J`-word: every `k` consecutive gaps of it lie in `D_k(M)`.
  `B_k(M; q')` = the largest span of a level-`k` admissible `J`-word, `J <= J_max`, that fuses
  at some phase (`monotone_functional.md` 4.1). `Phi := B_{L+1}`.
- **The fusion lemma** (the brief's statement): for every engine `M` and next gear `q'`, every
  level-`(L+1)` admissible word that fuses at some phase of `q'` has span `<= F(M) + q'`;
  i.e. `Phi(M; q') <= F(M) + q'`.
- **Neighbourhood of a maximal word `m`** (this branch's objects):
  `P(m) := max { a : (a, m) in D_{L+1}(M) }`, the widest gap that ever precedes a realisation of
  `m`; `S(m) := max { c : (m, c) in D_{L+1}(M) }`, the widest that ever follows one;
  `N(m) := max { a + c : (a, m, c) in D_{L+2}(M) }`, the widest pair of flanks that ever
  flank ONE realisation of `m`. `|m|` = the sum of `m`'s letters (its span).
  `R(M; q') := max over maximal words m of [ P(m) + |m| + S(m) ]`.

### 0.2 The theory

**T. The relaxation at order `L + 1` is a flank substitution and nothing else, and the fusion
lemma is the budget inequality plus a statement about the neighbourhoods of the maximal words.**
Precisely: (a) once the middle of a `J_max`-word is a maximal legal word, both flanks are
unstruck automatically (a struck flank would extend the word to a realised legal word of length
`L + 1`), so the fusion condition is empty at `J = J_max`; (b) hence the relaxed `J_max`-term of
`B_{L+1}` is exactly `R(M; q')`, and `Q*_{J_max}` is the same maximum with the two flanks taken
at one realisation (`N(m)` in place of `P(m) + S(m)`); (c) hence
`Phi = max(F(M + q'), R(M; q'))`; (d) hence the fusion lemma is `F(M + q') <= F(M) + q'` (the
budget, ROOT: it contains the pair statement at `J = 2` at every step with `L >= 1`, node 1e)
together with `R(M; q') <= F(M) + q'`, a bound on the neighbourhoods of the maximal words that
the budget does not imply. The brief's reading, "one gear adds at most `q'` columns in one
interval", is the budget itself, not a mechanism for it: a fused word has TWO flank gaps of `M`
and letters summing to more than `q'` (55 at `37 -> 41`), so the span exceeds `F(M)` by a second
flank plus the letters, and bounding that by `q'` is the conjecture at the step.

### 0.3 Predictions, each with the number that refutes it

Flag: the values of `B_{L+1}`'s per-`J` rows (`r66/results/order.json`) and the record rows
were in front of the branch when these were written; predictions T1-T3 are therefore checks of
a proof against recorded numbers, not blind predictions. T6 is blind.

- **T1 (closed form).** `Phi = B_{L+1} = max(F(M + q'), R)` at all ten steps, with `R` computed
  directly from the maximal words' neighbourhoods and no dynamic programme: `R = 6, 10, 10, 21,
  30, 35, 60, 55, 75, 98` at `5 -> 7 .. 37 -> 41` (the recorded `J = J_max` entries of the
  level-`(L+1)` rows), so `Phi = 6, 10, 11, 21, 30, 35, 60, 58, 88, 98`. REFUTED by one step
  where the direct `R` differs from the recorded relaxed term, or `max(F(M+q'), R)` from the
  recorded `B_{L+1}`.
- **T2 (automatic flanks).** For every maximal word `m` and every `(a, m)`, `(m, c)` in
  `D_{L+1}`, the word `(a, m, c)` fuses at the phase of `m`: 0 exceptions over every maximal
  word at every step. And `Q*_{J_max} = max { a + |m| + c : (a, m, c) in D_{L+2}, m maximal }`
  with no phase quantifier reproduces the recorded row `5, 7, 8, 18, 25, 34, 43, 55, 68, 91`.
  REFUTED by one struck flank or one wrong `Q*`.
- **T3 (the overshoot is a substitution).** `Phi - F(M + q') = max(0, R - F(M + q'))` equals the
  recorded overshoot `1, 3, 0, 3, 5, 1, 17, 0, 0, 7`; at every step with positive overshoot the
  binding word `(P(m), m, S(m))` is NOT in `D_{L+2}` and `R - Q*_{J_max} = P(m) + S(m) - N(m)`
  for the binding `m`. REFUTED by a realised binding word with positive overshoot.
- **T4 (mirror, a fact).** `P(m) = S(reverse m)` for every maximal word at every step, by the
  column-`0` mirror `k -> -k`. Listed so the tables' symmetry is not read as a finding.
- **T5 (the obstruction; proof, not computation).** The fusion lemma implies the budget
  inequality at the step, and at every step with `L >= 1` it contains the pair statement
  `Q*_2 <= F(M) + q'` as its `J = 2` term. No proof of it exists that does not prove the budget.
  Its non-root remainder is `R <= F(M) + q'`, and under the budget as hypothesis this reads
  `P(m) + S(m) - N(m) <= slack := F(M) + q' - F(M + q')` for every maximal `m`.
- **T6 (blind: is the remainder a law of two-tooth machines?).** On the counterfactual tooth
  families at `13 -> 17` (gears `5, 7, 11, 13` with teeth `+-v_g`, `v_g` free, and `q' = 17`
  with free tooth: `2 x 3 x 5 x 6 x 8 = 1,440` machines) and `17 -> 19` (`x 8 x 9 = 12,960`),
  the neighbour bound `R <= F(M) + q'` FAILS on between 1% and 30% of the machines on which the
  budget holds; i.e. the remainder is not a consequence of the budget and not generic. The
  refutation is 0 failures at both steps, which would say the remainder is generic for two-tooth
  machines and worth a mechanism hunt of its own. (Round 30, `tooth-counterfactual-percentile`
  5C, found `L` bounded is not structural on the same family; the record law is family-wide.)

**Stop rules.** The record law, `J_max = L + 2`, the legality criterion, the copy law and the
covering characterisation are cited, never re-derived. Any sub-question that reduces to the pair
or chain statement is stopped in one line and named.

### 0.4 Scorecard

Filled in section 5.

---

## 1. The construction: what a fused word of level `L + 1` is

Build it from the engine up, with nothing assumed.

1. **The engine's openings and gaps.** `M = {5..q}`. Column `x` is open iff no gear strikes it.
   Enumerate the openings `op(0) < op(1) < ...`; `gap(n) = op(n+1) - op(n)`; the gap sequence
   is periodic with period `N` (the openings per period `P`). A `k`-window is
   `(gap(n), ..., gap(n+k-1))`; `D_k(M)` is the set of `k`-windows that occur.
2. **The next gear's two tooth progressions.** `q'` strikes `x` iff `x = +-u (mod q')`. Read
   at offsets from a start column `x_0`: offset `o` is struck iff `o + z in {0, d}` with
   `z = x_0 + u (mod q')` (so `x_0 + o = -u` or `+u`), `d = 2u`. The set of struck offsets is
   the union of two arithmetic progressions of step `q'`, `{-z + q' t}` and `{-z + d + q' t}`.
   As `x_0` runs over one period of `M + q'` every phase `z` occurs (the copy law, docs/proofs/05
   (A)), so "some phase" and "somewhere in `M + q'`" are the same quantifier.
3. **A struck chain and its word.** `k + 1` consecutive openings of `M` all struck at one phase
   are a chain; their `k` gaps, read mod `q'`, are letters of `{PAD, UP, DOWN}` forming a legal
   word, and conversely (docs/proofs/10 Theorem 1: the reading walks the tooth, PAD keeps it,
   UP takes `-u -> +u`, DOWN the reverse). So a legal word with a nonzero letter has ONE start
   tooth and so, at a given position, ONE phase at which all its openings are struck; an
   all-pad word (or the empty word) has two.
4. **The deepest fusion.** `L = L(M)` is the longest realised legal word, so the deepest chain
   has `L + 1` struck openings and the deepest fusion `J_max = L + 2` gaps: a flank gap `g_1`,
   a maximal word `m = (g_2, ..., g_{L+1})`, a flank gap `g_{L+2}`, with the flank openings
   `o_0` and `o_{L+2}` unstruck.
5. **The relaxation at order `L + 1`.** A level-`(L+1)` admissible `J_max`-word need not be a
   window of `M`; it needs its two `(L+1)`-subwindows `(g_1, m)` and `(m, g_{L+2})` in
   `D_{L+1}(M)`. `B_{L+1}` takes the widest such word that fuses, over `J <= J_max`; for
   `J <= L + 1` admissible means realised, so those terms are the exact `Q*_J`.
6. **What the covering problem says about each `(L+1)`-window.** `(g_1, m) in D_{L+1}` iff
   there is one phase `t_g in Z_g` per gear of `M` such that no gear strikes any of the `L + 2`
   opening offsets and every other offset of `[0, g_1 + |m|]` is struck by some gear
   (`order_law_37_41.md` 2.2; the gears' phases are independent by CRT). So the left window is
   one covering problem, the right window another, and the whole word a third; the relaxation
   drops the third.

## 2. The proof: what the relaxation is, exactly

Every step is a named statement; each uses only the constructions above and the two cited
theorems (legality criterion docs/proofs/05 (F) / docs/proofs/10 Theorem 1; record law
docs/proofs/09). Fix `M`, `q'`, `L = L(M)`, `J_max = L + 2`.

**Definition D1.** For a maximal word `m` (realised, legal, length `L`; the empty word when
`L = 0`): `pred(m) = {a : (a, m) in D_{L+1}}`, `succ(m) = {c : (m, c) in D_{L+1}}`,
`P(m) = max pred(m)`, `S(m) = max succ(m)`, `N(m) = max {a + c : (a, m, c) in D_{L+2}}`,
`R(M; q') = max_m [P(m) + |m| + S(m)]`.

**Lemma 1 (phases of a word).** Let `m` be legal, at a position where its first opening has
residue `t_1 = o_1 + z in {0, d}`. If `m` has a nonzero letter, exactly one `t_1` makes every
opening of `m` struck; if `m` is all-pad or empty, both do. *Proof.* Reading (step 3): PAD keeps
the tooth, UP is only consistent from `t = 0` to `t = d`, DOWN only from `d` to `0`; the first
nonzero letter fixes `t_1`, the rest follow; with no nonzero letter both readings are
consistent. QED

**Lemma 2 (automatic flanks).** Let `m` be maximal, `a in pred(m)`, and `z` any phase at which
every opening of `m` is struck (Lemma 1). Then the opening before `m` in the window `(a, m)` is
NOT struck at `z`. Symmetrically for `c in succ(m)` and the opening after `m`.
*Proof.* If it were struck, the `L + 2` openings of the realised window `(a, m)` would all lie
on the two teeth of the phase `z`, so `(a, m)` would be a legal word (docs/proofs/10 Theorem 1)
realised in `M` of length `L + 1`, contradicting the definition of `L`. For `L = 0` the same:
`(a)` would be a realised legal 1-word. QED

**Theorem A (the relaxed deepest term is the neighbourhood maximum).**
(i) For every maximal `m`, every `a in pred(m)`, every `c in succ(m)`: the word `(a, m, c)` is
level-`(L+1)` admissible and fuses at every phase of `m`.
(ii) Every level-`(L+1)` admissible `J_max`-word that fuses at some phase is of the form
`(a, m, c)` with `m` maximal, `a in pred(m)`, `c in succ(m)`.
(iii) The relaxed `J = J_max` term of `B_{L+1}(M; q')` equals `R(M; q')`.
(iv) `Q*_{J_max}(M; q') = max_m [N(m) + |m|]`: the widest realised `(L+2)`-window whose middle
is a maximal legal word, with no phase quantifier.
(v) `B_{L+1}(M; q') = max( F(M + q'), R(M; q') )`.
*Proof.* (i) Admissible: `(a, m)`, `(m, c)` are in `D_{L+1}` by D1. Fuses: the interior
openings are those of `m`, struck at any phase of `m`; the two flank openings are unstruck at
that phase by Lemma 2. (ii) A fusing `J_max`-word has its `L + 1` interior openings struck at
one phase, so its `L` middles form a legal word (docs/proofs/10 Theorem 1), realised because
the left `(L+1)`-subwindow is realised and contains it: `m` is maximal; `a in pred(m)` and
`c in succ(m)` are the two subwindows. (iii) By (i)-(ii) the relaxed term is
`max { a + |m| + c : m maximal, a in pred(m), c in succ(m) }`, and `a`, `c` range
independently, so the maximum is `max_m [P(m) + |m| + S(m)]`. (iv) A realised `J_max`-fusion
is a realised `(L+2)`-window `(a, m, c)` with `m` legal of length `L` and its flanks unstruck;
by Lemma 2 the flank condition is automatic, so `Q*_{J_max}` is the maximum of `a + |m| + c`
over all realised `(a, m, c)` with `m` maximal, i.e. `max_m [N(m) + |m|]`. (v) The terms
`J <= L + 1` of `B_{L+1}` are the exact `Q*_J`; the term `J_max` is `R >= Q*_{J_max}`
(`N(m) <= P(m) + S(m)`); the record law gives `F(M + q') = max_{J <= J_max} Q*_J`. Hence
`B_{L+1} = max(Q*_1, ..., Q*_{L+1}, R) = max(Q*_1, ..., Q*_{J_max}, R) = max(F(M+q'), R)`. QED

**Corollary B (the overshoot is one flank substitution).**
`B_{L+1} - F(M + q') = max(0, R - F(M + q'))`, and for the maximal word `m*` attaining `R`:
`R - Q*_{J_max} <= P(m*) + S(m*) - N(m*)`. If `(P(m*), m*, S(m*)) in D_{L+2}` then
`B_{L+1} = F(M + q')`. In words: the relaxation lets the widest predecessor of `m*` and the
widest successor of `m*` be taken from two different realisations of `m*`; the machine's own
record takes them from one.

**Theorem C (the fusion lemma decomposed).** For every `M`, `q'`:

    Phi(M; q') <= F(M) + q'    <==>    F(M + q') <= F(M) + q'   AND   R(M; q') <= F(M) + q'.

*Proof.* Theorem A (v). QED
Consequences. (a) The fusion lemma implies the budget inequality at the step; it is NOT weaker
than the conjecture's step, it is the step plus a second statement. (b) At every step with
`L >= 1` the term `J = 2` of `B_{L+1}` is the exact `Q*_2`, so the fusion lemma contains the
pair statement `Q*_2 <= F(M) + q'`, node 1e's obstruction (at column 0 it reads
`2 d_0 <= F + q'`; every route to it is twin-Bertrand-shaped). (c) The second statement,
`R <= F(M) + q'`, is a bound on the neighbourhoods of the maximal words that the budget does
not imply (section 3.4: it fails on 8.6% and 18.3% of the tooth-family machines on which the
budget holds). It is the whole of what `Phi` asserts beyond the budget, and it is the whole of
the order law's upper half.

**What the brief's reading is, exactly.** "A fused word is a run of `M` with the `q'`-letters
inserted, a run of `M` spans at most `F(M)`, so one gear can add at most `q'`." A fused
`J_max`-word is `(a, m, c)`: TWO gaps of `M` (each `<= F(M)`) and a word of letters between
them whose sum is not bounded by `q'` (`|m| = 55` at `37 -> 41`, `86` at `41 -> 43`). Its
span exceeds `F(M)` by `a + |m| + c - F(M)`, which is at least the second flank plus the
letters; bounding that by `q'` is `Q*_{J_max} <= F(M) + q'`, the budget's own deepest term. No
accounting of "columns one gear adds" proves it: the gear adds the letters (Lemma 2 says it
adds nothing else, the flanks are the engine's), and the letters alone can exceed `q'`.

**What the covering characterisation gives, and does not.** It makes `R` a finite certificate
at every step: the maximal words are finitely many (letters `<= F(M)`, length `L <= CC` of
`pad_cap.md` E3), `P(m)` is the first realised `(a, m)` in a descending sequence of covering
problems, `S(m)` likewise, `N(m)` a descending scan of joint problems. At `41 -> 43`, a step
no scan reaches, this is a few hundred covering problems (section 3.3). It gives no mechanism
for `P(m) <= F(M) + q' - |m| - S(m)`: the covering problem for `(a, m)` is satisfiable exactly
up to `a = P(m)`, and nothing in the gears' phases relates `P(m)` to `q'` (the family in 3.4 is
the counter-construction: same gears, other teeth, `P(m) + S(m)` past the budget while the
record stays inside it).

**Formalisation notes.** Lemma 1 is `WordLegal` reading with a fixed start tooth (exists).
Lemma 2 is one application of `WordLegal.chain_iff_word` (the `=>` half: a struck chain gives
a legal word) plus the definition of `L` as a maximum (`WordLegal.realisedWord_mono` gives the
prefix closure needed to say "length `L + 1` is not realised"). Theorem A (iii)-(v) is
bookkeeping over finite maxima with the record law (`docs/proofs/09` Theorem 1 (ii)) as the
only imported theorem; (v) needs `Q*_{J_max} <= R`, i.e. `N <= P + S`, which is
`max (a + c) <= max a + max c`. Nothing here needs primality, periodicity beyond the copy law,
or the sizes of any gap.

## 3. Results (exact; scripts `fl_flank.py`, `fl_family.py`, `fl_next.py`, `fl_ladder.py`)

### 3.1 Setup and gates

| object | range | cost |
|---|---|---|
| m5..m23 by direct sieve of the full period (m23: `P = 37,182,145`, `N = 7,952,175`); every realised `(L+2)`-window | full periods | 13 s |
| m29 (depth 10, 15,240,585 rows) and m31 (depth 6, 2,678,901 rows) rebuilt by the r61/r66 closure from `D_15(m23)`; gates `F = 43, 58`, dictionary sizes to the row, `loss = 0`, `over0 = 0` at both rungs (`fl_ladder.py`) | complete dictionaries | 847 s, under 1 GB |
| m37: `D_1`, `D_2` from r70; every `D_3`/`D_4` membership needed, by the covering instrument `r70/ol_pattern.py` (memoised; 0 new problems beyond r70's) | 4 maximal words | 0.1 s |
| the tooth families at `13 -> 17` (1,440 machines) and `17 -> 19` (12,960), each machine sieved with its own teeth and the bigger machine sieved directly for `F(M + q')` | complete families | 60 s, 225 s |
| m41: the 28 legal 2-words over the letters of 43, then the neighbourhoods of the realised ones, by covering problems on 11 gears | see 3.3 | see 3.3 |

Gates at the ten steps: `L = 1, 0, 1, 1, 1, 2, 1, 3, 3, 2` reproduced (ten of ten); the relaxed
term `R` equals the recorded level-`(L+1)`, `J = J_max` entry at ten of ten; `Q*_{J_max}`
(computed with NO phase quantifier, Theorem A (iv)) equals the recorded row
`5, 7, 8, 18, 25, 34, 43, 55, 68, 91` at ten of ten; `max(F(M+q'), R)` equals the recorded
`B_{L+1}` at ten of ten; on the families the real-teeth member reproduces
`F, F(M+q'), L, R = 11, 18, 1, 21` and `18, 25, 1, 30`.

### 3.2 The ten steps: the maximal words and their neighbourhoods

`m*` is the maximal word attaining `R`; `c | P` is the widest successor of `m*` in a
realisation whose predecessor is `P(m*)` (the flank the machine actually hangs on the widest
left flank).

| step | `F` | `F(M+q')` | budget | `L` | maximal words | `m*` | `P` | `abs(m*)` | `S` | `N` | `c` given `P` | `R = P + abs(m) + S` | `Q*_{J_max} = max N + abs(m)` | `Phi = max(F(M+q'), R)` | rec `B_{L+1}` | overshoot | margin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `5 -> 7` | 2 | 5 | 9 | 1 | (2) | (2) | 2 | 2 | 2 | 3 | 1 | **6** | 5 | 6 | 6 | 1 | 3 |
| `7 -> 11` | 5 | 7 | 16 | 0 | () | () | 5 | 0 | 5 | 7 | 2 | **10** | 7 | 10 | 10 | 3 | 6 |
| `11 -> 13` | 7 | 11 | 20 | 1 | (4) | (4) | 3 | 4 | 3 | 4 | 1 | 10 | 8 | **11** | 11 | 0 | 9 |
| `13 -> 17` | 11 | 18 | 28 | 1 | (6), (11) | (11) | 5 | 11 | 5 | 7 | 2 | **21** | 18 | 21 | 21 | 3 | 7 |
| `17 -> 19` | 18 | 25 | 37 | 1 | (6), (13) | (6) | 12 | 6 | 12 | 17 | 5 | **30** | 25 | 30 | 30 | 5 | 7 |
| `19 -> 23` | 25 | 34 | 48 | 2 | (8,15), (15,8) | (8,15) | 5 | 23 | 7 | 11 | 3 | **35** | 34 | 35 | 35 | 1 | 13 |
| `23 -> 29` | 34 | 43 | 63 | 1 | (10), (19), (29) | (10) | 25 | 10 | 25 | 33 | 2 | **60** | 43 | 60 | 60 | 17 | 3 |
| `29 -> 31` | 43 | 58 | 74 | 3 | (10,21,10) | (10,21,10) | 7 | 41 | 7 | 14 | 7 | 55 | 55 | **58** | 58 | 0 | 16 |
| `31 -> 37` | 58 | 88 | 95 | 3 | (12,25,12), (25,12,25) | (12,25,12) | 13 | 49 | 13 | 16 | 3 | 75 | 68 | **88** | 88 | 0 | 7 |
| `37 -> 41` | 88 | 91 | 129 | 2 | (14,41), (41,14), (27,41), (41,27) | (14,41) | 21 | 55 | 22 | 36 | 15 | **98** | 91 | 98 | 98 | 7 | 31 |

The other maximal words, for the record: `(6)` at m13: `7 + 6 + 7 = 20`, `N = 12`; `(13)` at
m17: `7 + 13 + 7 = 27`, `N = 12`; `(15, 8)` at m19: `7 + 23 + 5 = 35`, `N = 11` (the mirror);
`(19)` at m23: `15 + 19 + 15 = 49`, `N = 18`; `(29)` at m23: `8 + 29 + 8 = 45`, `N = 11`;
`(25, 12, 25)` at m31: `3 + 62 + 3 = 68`, realised jointly (it IS `Q*_5(m31; 37) = 68`);
`(27, 41)`, `(41, 27)` at m37: `5 + 68 + 2 = 75`, realised jointly.

**T2, automatic flanks: 0 struck flanks** among every predecessor and successor of every
maximal word at every consistent phase, at all ten steps (exhaustive at m5..m31; at m37 the
only struck-flank candidates are the four letter extensions `(41,14,41)`, `(14,41,27)`,
`(27,41,14)`, `(41,27,41)`, each put to the instrument and unrealised: exactly the words
`L(m37) = 2` forbids). And 0 struck flanks on all 14,400 machines of the two tooth families.

**T3, the overshoot is a substitution.** At the seven steps with positive overshoot the binding
word `(P, m*, S)` is not realised (seven of seven), and `R - Q*_{J_max} <= P + S - N` at the
binding word: `1 <= 1`, `3 <= 3`, `3 <= 3`, `5 <= 7`, `1 <= 1`, `17 <= 17`, `7 <= 7` (equality
except at `17 -> 19`, where `F(19) = 25` is carried by the other word `(13)` and by `Q*_2`).
At `29 -> 31` and `31 -> 37` the widest relaxed word is realised jointly (`(7, 10,21,10, 7)`
and `(3, 25,12,25, 3)`), so `R = Q*_5` and the overshoot is 0. **T4**: `P(m) = S(reverse m)`
at every word, 0 exceptions.

**The mechanism at the tight step, in the machine's own terms.** At `23 -> 29`, `L = 1`, the
letters of 29 are `10` (UP, `d = 10`), `19` (DOWN) and `29` (PAD). The widest gap of m23 that
ever precedes a 10 is 25, and by the mirror the widest that ever follows one is 25; but the
widest that follows a 10 PRECEDED by 25 is 2 (`(25, 10, 2)`, span 37), and the widest pair of
flanks around one 10 is `10 + 23` (the record fusion `(23, 10, 10)` mirrored, span 43). So the
relaxation buys `60 - 43 = 17` columns by taking the two 25s from two different occurrences of
the gap 10, and the remainder `60 <= 63` asserts, of the real teeth of `{5..23}`, that no gap
of size 27..34 ever sits next to a gap of 10. That is a fact about which gaps neighbour the
letter 10 in m23; it is decided by covering problems, and it has no visible reason.

### 3.3 The remainder past the ladder: `R(m41; 43)` by covering problems alone

Engine `{5..41}` (11 gears, period `5.3 x 10^13`, never scanned), `q' = 43`, `u = 36`,
`d = 29`; letters up to `F(m41) = 91`: DOWN `14, 57`; UP `29, 72`; PAD `43, 86`. Corpus:
`L(41) = 2` with the word `(43, 43)`; `F(43) = 103`; budget `91 + 43 = 134`. Everything below
is by covering problems on the 11 gears (`fl_next.py`, `r70/ol_pattern.py`), with no table of
m41 in hand; the run was stopped at the one-hour rule with three of the five words done
(3,837 s, 728 covering problems; every verdict is in `results/m41_memo.json`, and the script
resumes from it).

- **The maximal words of m41 w.r.t. 43 are five, not one.** Of the 28 legal 2-words over the
  six letters, exactly `(14, 43)`, `(43, 14)`, `(29, 43)`, `(43, 29)`, `(43, 43)` are
  realised; the 23 others (every bare pair `(14, 29)`, `(29, 14)`, every skip pair, `(86, 86)`)
  are not. The corpus recorded only `(43, 43)`. Then all 35 legal 3-word extensions of the five
  were tested and **none is realised**: `L(m41) = 2` from the instrument, agreeing with the
  corpus.
- **Neighbourhoods** (each `P`, `S` a descending scan from `a = 91`; `N` the descending joint
  scan):

| `m` | `abs(m)` | `P` | `S` | `N` (at) | `P + abs(m) + S` | `N + abs(m)` | joint `(P, m, S)` realised |
|---|---|---|---|---|---|---|---|
| `(14, 43)` | 57 | 28 | 33 | 43 `(15, 28)` | **118** | **100** | no |
| `(43, 14)` | 57 | 33 | 28 | 43 `(20, 23)` | 118 | 100 | no |
| `(29, 43)` | 72 | 13 | 12 | 21 `(13, 8)` | 97 | 93 | no |
| `(43, 29)` | 72 | 12 | 13 | 21 | 97 | 93 | (mirror of the row above, T4) |
| `(43, 43)` | 86 | not reached | | | | | |

So `R(m41; 43) >= 118` and the deepest realised term `Q*_4(m41; 43) >= 100`, against
`F(43) = 103` (consistent: `Q*_4 <= F(43)`, and the record is within 3 of the deepest term's
floor) and the budget 134. **The remainder at `41 -> 43` holds on the four words decided
(`118 <= 134`, margin `>= 16` for them) and is undecided on `(43, 43)`**: it needs
`P(43, 43) + S(43, 43) <= 48`, i.e. by the mirror `P(43, 43) <= 24`, one descending scan of
at most 67 covering problems from `a = 91` (the next lane's first job; the memo resumes it).
Note the shape: at this step the widest relaxed word hangs flanks 28 and 33 on the padded
word `(14, 43)`, and the machine's own widest flanks on one realisation of it sum to 43 — the
same picture as `37 -> 41` (`(21, 14, 41, 22)` against `(21, 14, 41, 15)`), one letter up.

### 3.4 T6, the tooth families: the remainder is not a consequence of the budget

Every gear `g in {5..y}` with teeth `+-v_g`, `1 <= v_g <= (g-1)/2`, and the new gear `q'` with
teeth `+-v'`; the real machine is one member (`v_g = 6^{-1} mod g` up to sign).

| step | machines | budget holds | budget fails | remainder `R <= F + q'` fails while the budget holds | share | `L` on the family | worst excess `R - budget` |
|---|---|---|---|---|---|---|---|
| `13 -> 17` | 1,440 | 1,439 | 1 (its remainder fails too) | **124** | **8.6 %** | 0: 2, 1: 1,386, 2: 48, 3: 4 | 10, teeth `(1,1,5,1)`, `v' = 1`: `F = 25`, `R = 52`, `F(M+q') = 28`, budget 42 |
| `17 -> 19` | 12,960 | 12,924 | 36 (35 with the remainder failing, 1 with it holding) | **2,371** | **18.3 %** | 1: 7,302, 2: 5,053, 3: 605 | 16, teeth `(1,3,2,4,4)`, `v' = 4`: `F = 27`, `R = 62`, `F(M+q') = 37`, budget 46 |

So on two-tooth engines with the same gears and other teeth the budget is nearly universal
(fails at 0.07 % and 0.28 %) and the remainder is not (fails at 8.6 % and 18.3 % of the
budget-holding members, with excesses up to 16 columns). The remainder is a fact of the real
teeth, not a law of the construction, and the family is the counter-construction for any
attempt to derive it from the gears' phases alone. **Where the real teeth sit**: the real
member's remainder margin (7) is at the 57th percentile at `13 -> 17` and its relaxation excess
`R - F(M+q')` is exceeded by 63 % of the budget-holding members; at `17 -> 19` the margin (7)
is at the 55th percentile and the excess (5) is exceeded by 45 %. The real
teeth are ordinary on this statistic, as they were on the budget slack (round 28: 59 % / 37 %),
unlike on `F` and `F_2` (17-26th percentile).

The record law held family-wide again (`F(M + q') >= Q*_{J_max}` at 14,400 of 14,400; the
equality is round 30's result and was not re-derived).

## 4. The obstruction, exactly, and what is still missing

### 4.1 What is proved (general, no hypothesis)

Theorem A and Corollary B: for every engine `M = {5..q}` and next gear `q'`,

    Phi(M; q') = B_{L+1}(M; q') = max( F(M + q'),  max over maximal words m of [P(m) + |m| + S(m)] ),

with `P`, `S` the widest gap ever preceding / following a realisation of `m`; the deepest
realised fusion `Q*_{J_max}` is the same maximum with both flanks at one realisation, no phase
quantifier; and the relaxation's whole overshoot is the difference between "widest predecessor
and widest successor at two realisations" and "at one".

### 4.2 What cannot be proved here, and why

Theorem C: the fusion lemma is `[budget at the step] AND [R <= F(M) + q']`.

- The first conjunct is the root question at the step (node R1: any per-step bound implies a
  twin-Bertrand-type statement; node 1e: the `J = 2` term at column 0 is `2 d_0 <= F + q'`).
  The fusion lemma contains it, so no proof of the fusion lemma from the covering
  characterisation, or from anything else, exists that does not prove the budget. Stop line:
  the exact terms `J <= L + 1` of `Phi` ARE `Q*_1, ..., Q*_{L+1}`; `J = 2` is the pair
  statement, `J >= 3` the chain statement. **ROOT.**
- The second conjunct is not root-shaped: it says nothing about `F(M + q')`. It is a bound on
  the neighbourhoods of the maximal words, true at ten of ten real steps (margins
  `budget - R` = 3, 6, 10, 7, 7, 13, 3, 19, 20, 31) and FALSE on 8.6 % and 18.3 % of the
  tooth-family machines where the budget holds (3.4). So it is not implied by the budget, not
  implied by the two-tooth construction, and carried at the real steps by facts of the real
  teeth (at `23 -> 29`: no gap of 27..34 of m23 is adjacent to a 10) with no mechanism found.
  It is exactly the order law's upper half, `B_{L+1} <= F + q'`, minus the budget.

### 4.3 The hypotheses under which the fusion lemma follows

- **Hypothesis H1 (the budget at the step) plus H2 (`R <= F(M) + q'`)**: then the lemma holds,
  by Theorem C; both hypotheses hold at ten of ten steps. Nothing weaker suffices: H1 is
  necessary (Theorem C), H2 is necessary (Theorem C).
- **H1 plus "the widest relaxed word is realised jointly"** (`(P(m*), m*, S(m*)) in D_{L+2}`):
  then `Phi = F(M + q')` and the lemma is the budget. Holds at `29 -> 31` and `31 -> 37` (and
  vacuously at `11 -> 13`, where `R < F(13)`), fails at the other seven steps.
- No hypothesis on `L` helps: `L` enters only through which words are maximal; the remainder
  fails on the family at `L = 1` (every worst member of 3.4 has `L = 1`).
- No cap from E2 helps: E2 caps `L`, not `P(m)`.

### 4.4 The smallest lemma still missing, with its smallest instances

**The missing lemma (the non-root remainder), in its `L = 1` form:** for every realised letter
`b` of `M` with respect to `q'` (a gap `b = 0, +-d (mod q')` when `L(M) = 1`),

    2 P(b) + b  <=  F(M) + q',      P(b) = the widest gap of M ever adjacent to a gap of size b

(by the mirror `P = S`). In general: `P(m) + |m| + S(m) <= F(M) + q'` for every maximal `m`.

- Smallest instance with content (the relaxed word unrealised): engine `{5}`, `q' = 7`, letter
  `b = 2` (`d = 5`, so 2 is DOWN), word `(2, 2, 2)` at phase `z = 3` (offsets `0, 2, 4, 6`
  read `3, 5, 0, 2` mod 7: unstruck, struck, struck, unstruck): `6 <= 9`. The 3-window
  `(2, 2, 2)` is not a window of m5 (its cycle is `2, 1, 2`).
- Tightest instance on the ladder: engine `{5..23}`, `q' = 29`, letter `b = 10` (UP), word
  `(25, 10, 25)` at phase `z = 4`: `60 <= 63`, margin 3; `(25, 10)` and `(10, 25)` realised,
  `(25, 10, 25)` not; the widest successor of a 10 preceded by 25 is 2.
- Largest instance computed: engine `{5..37}`, `q' = 41`, word `(14, 41)`,
  `(21, 14, 41, 22)` at phase 19: `98 <= 129`; and `41 -> 43`, section 3.3.
- The counter-instances (same gears, other teeth) that show it is not a law of the
  construction: `13 -> 17`, teeth `v = (1, 1, 5, 1)` for `5, 7, 11, 13` and `v' = 1` for 17:
  `F = 25`, `R = 52`, `F(M+17) = 28 <= 42` but `R = 52 > 42`.

**The root lemma** underneath it, in the phase-free form Theorem A (iv) gives it: the widest
realised `(L+2)`-window of `M` whose middle is a maximal legal word (and every shallower
`Q*_J`) is at most `F(M) + q'`. Tightest at `31 -> 37`: `(11, 12, 37, 28)`, `88 <= 95`, at
`J = 4 < J_max = 5`.

## 5. Scorecard

| # | prediction | verdict | evidence |
|---|---|---|---|
| T1 | `Phi = max(F(M+q'), R)`, `R = 6, 10, 10, 21, 30, 35, 60, 55, 75, 98` | **CONFIRMED** ten of ten, `R` from the neighbourhoods alone | 3.1, 3.2 |
| T2 | automatic flanks, 0 exceptions; `Q*_{J_max}` phase-free reproduces the row | **CONFIRMED** 0 struck flanks at ten steps and 14,400 family machines; row ten of ten | 3.2 |
| T3 | overshoot = flank substitution; binding word unrealised when positive | **CONFIRMED** seven of seven; `R - Q* <= P + S - N` with equality at six | 3.2 |
| T4 | `P(m) = S(reverse m)` | **CONFIRMED** 0 exceptions (a fact) | 3.2 |
| T5 | the lemma contains the budget and the pair statement; the remainder is `R <= F + q'` | **PROVED** (Theorem C) | 2, 4 |
| T6 | the remainder fails on 1-30 % of budget-holding family members | **CONFIRMED** 8.6 % and 18.3 %; the real teeth ordinary (55-57th percentile) | 3.4 |
| brief | "one gear adds at most `q'` in one interval" is a mechanism for the lemma | **REFUTED as a mechanism**: the gear adds the letters (Lemma 2), whose sum is 55 at `37 -> 41` and 86 at `41 -> 43`; the flanks are the engine's | 2 |

## 6. What is new

1. **The closed form of the monotone functional** (Theorem A (v)): `B_{L+1} = max(F(M+q'), R)`
   with `R = max_m [P(m) + |m| + S(m)]` over the maximal legal words. `Phi` is not a table walk;
   it is the record plus the neighbourhood table of at most a handful of words (1 to 5 words at
   every step to m41).
2. **Automatic flanks** (Lemma 2): every realisation of a maximal legal word is a `J_max`-fusion;
   the deepest fusion needs no phase condition, so `Q*_{J_max}` is the widest realised
   `(L+2)`-window with a maximal-word middle. The deepest term of the record law is a
   phase-free statement about `D_{L+2}(M)`.
3. **The relaxation is a flank substitution** (Corollary B), at every step, with the exact
   excess `P + S - N`; the r70 observation at one step is the general rule.
4. **The exact decomposition of the fusion lemma** (Theorem C): budget AND remainder; hence
   ROOT, and the non-root part named and measured: `R <= F(M) + q'`, ten of ten, margins
   3..31, and its `L = 1` form `2 P(b) + b <= F + q'` for every realised letter `b`.
5. **The remainder is a real-teeth fact, not a law of two-tooth engines** (3.4): fails at
   8.6 % / 18.3 % of budget-holding tooth-family members; the real teeth sit at the 55-57th
   percentile of its margin.
6. **New exact facts of m41 with respect to 43** (3.3): five maximal words, not one; every
   legal 3-word over them unrealised (`L(41) = 2` from the instrument); their neighbourhoods.
7. **The dictionaries `D_10(m29)`, `D_6(m31)` on disk** (untracked, `fl_ladder.py`), which r66
   computed and did not save.

Prior art, one line: Lemma 2 is docs/proofs/10 Theorem 1 read at the maximum; the record law
and the copy law are docs/proofs/09 and 05; the covering instrument is r70. Nothing outside the
repository is used or claimed.

## 7. Verdict

**Node 4.i.b.ii.c: the fusion lemma cannot be proved in general, because it is the budget
inequality plus a second statement, and the budget is the root; the second statement is
proved to be the whole of the relaxation (a flank substitution), named exactly, measured true
at ten of ten real steps, and shown to be a fact of the real teeth rather than of the
construction. ROOT, with the non-root remainder split off as a PARTIAL object.**

- PROVED (in writing, formalisable): Theorem A, Corollary B, Theorem C. Kernel: none yet.
- MEASURED: `R <= F(M) + q'` at ten of ten (margins 3, 6, 10, 7, 7, 13, 3, 19, 20, 31);
  `R(m41; 43)` in 3.3.
- ROOT: the fusion lemma itself; equivalently `Phi <= F + q'`; equivalently the order law's
  upper half; its `J = 2` term is node 1e's pair statement.
- What survived and where it went: the order law is now `budget AND remainder`, so the upper
  half of `k* = L + 1` is a statement about neighbourhoods of `<= 5` words, computable past the
  scan wall by covering problems (`fl_next.py`); the lower half (`B_L > F + q'`) is untouched.
- Where the difficulty moved: nowhere new. The relaxation added a true, unexplained,
  non-generic inequality about letter neighbourhoods on top of the budget; removing it (taking
  `Phi = F(M+q')`) loses nothing but the bounded-order form, and keeping it costs an extra
  measured fact per step. `Phi`'s value as a sufficient condition stands exactly as
  `monotone_functional.md` 6.2 stated it, with its cost now priced: it is the budget plus `R`.

## 8. Dead ends (bricks), each with its refuting instance

| idea | dies at | instance | why it cannot be revived |
|---|---|---|---|
| "one gear adds at most `q'` columns" as a mechanism for `Phi <= F + q'` | every step with `abs(m) > q'` | `37 -> 41`: `abs(14, 41) = 55 > 41`; `41 -> 43`: `86 > 43` | the gear adds the letters, whose sum is unbounded by `q'`; the flanks are the engine's (Lemma 2) |
| bounding `P(m)` from the covering problem's capacity | every step | pad_cap.md 2.5: capacity/need 1.4-2.2 | the covering problem is satisfiable up to `a = P(m)` exactly; capacity never binds |
| deriving the remainder from the budget | the tooth family | `13 -> 17`, teeth `(1,1,5,1)`, `v' = 1`: budget `28 <= 42`, `R = 52` | the remainder fails at 8.6 % / 18.3 % of budget-holding members |
| `2 F_{L+1}(M) - L a_L <= F + q'` as a sufficient condition for the remainder | `17 -> 19`, `23 -> 29` | `2 x 25 - 6 = 44 > 37`; `2 x 39 - 10 = 68 > 63` | the separate maxima `F_{L+1} - abs(m)` are far above `P(m)`; the joint structure is needed |

## 9. Open items on the part alone, sorted

- **Closed here.** The form of the relaxation at order `L + 1` (Theorem A); the phase-free
  form of the deepest record term; the decomposition of the fusion lemma; the maximal words
  and neighbourhoods at every step to m41.
- **Measurement with no structural content.** The real teeth's percentile on the remainder's
  margin (55-57th); the mirror identity.
- **Root question in disguise.** `Phi <= F + q'`; the exact terms `Q*_J <= F + q'`.
- **Genuinely open on the part alone, with the attack.** (i) `R <= F(M) + q'` at the real
  steps beyond m41 (43, 47, 53, 59): `fl_next.py` does a step in a few hundred covering
  problems; the maximal words of m43 (`(47, 47)` and its bare and padded neighbours), m47
  (`(18,35,18,35)`, `(18,35,53)`, `(35,18,53)`, `(18,53,35)`), m53 (`(20,98,20)`). A failure
  at any step ends `Phi`'s role (then `k* = J_max` there and the order law's upper half is
  false); ten more successes make the remainder a measured law with margins, still without a
  mechanism. (ii) A mechanism for `P(b) <= (F + q' - b)/2` at `L = 1`: which gears forbid a
  gap of 27..34 beside a 10 in m23? A covering-problem census of `(a, 10)` for `a = 26..34`
  with the gear that kills each (the `killed_by_gear` / capacity / cover verdicts of
  `ol_pattern.realised`) would name the obstruction per `a`; it is a fact of the real teeth,
  so the answer is a table, not a law. (iii) Formalisation of Lemma 2 and Theorem A (v) in
  `WordLegal`: the statements are finite-maximum bookkeeping over `D_{L+2}` and need only
  `chain_iff_word` and the record law.

## 10. Files

- `research/anchor235/r73/fl_core.py` - sieve, letters, legality, maximal words,
  neighbourhoods, the fuse check, the mirror check
- `research/anchor235/r73/fl_flank.py` - the ten steps (m5..m23 periods, m29/m31 dictionaries,
  m37 covering problems); `results/flank.json`
- `research/anchor235/r73/fl_ladder.py` - the closure ladder with `D_10(m29)`, `D_6(m31)` saved;
  `results/m29_dict.npz`, `m31_dict.npz`, `ladder.json`, `ladder.log`
- `research/anchor235/r73/fl_family.py` - the tooth families; `results/family_13_17.json`,
  `family_17_19.json` (every member's row), `family_17.log`
- `research/anchor235/r73/fl_next.py` - `R(m41; 43)` by covering problems; `results/next_41.log`,
  `next_41_43.json`, `m41_memo.json` (every covering verdict)
- `results/` is untracked.

## Addendum (manager, 2026-09-11): the deciding number at 41 -> 43

The scan resumed from the 725 memoised verdicts (`fl_next.py 41`, 351 further covering problems,
1658 s) and finished all five maximal words of m41 with respect to 43:

| maximal word | P | S | N (flank pair) | relaxed P + |m| + S | exact N + |m| |
|---|---|---|---|---|---|
| (14, 43) | 28 | 33 | 43 at (15, 28) | 118 | 100 |
| (43, 14) | 33 | 28 | 43 at (20, 23) | 118 | 100 |
| (29, 43) | 13 | 12 | 21 at (13, 8) | 97 | 93 |
| (43, 29) | 12 | 13 | 21 at (8, 13) | 97 | 93 |
| (43, 43) | 5 | 5 | 7 at (2, 5) | 96 | 93 |

Struck flanks realised: none, at all five. So P(43, 43) = 5 <= 24, R(m41; 43) = 118, and by
Theorem A, Phi = B_3(m41; 43) = max(F(43), 118) = max(103, 118) = 118 <= 134 = F(41) + 43: the
remainder R <= F + q' holds at 41 -> 43 with margin 16, the order law's upper half
B_{L+1} <= F + q' stands at 11 of 11 steps (the first past the scan wall, by covering problems
alone), and the phase-free deepest term Q*_{J_max}(m41; 43) = 100 <= F(43) = 103 is consistent
with the record. The lower half at this step (B_2(m41; 43) > 134, k* = 3) is the order-law
prover's item (order_law_beyond_41.md), not decided here.
