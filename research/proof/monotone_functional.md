# Node 4.i.b.ii - A MONOTONE FUNCTIONAL OF THE MERGE CLOSURE

Parent: node **4.i.b, the branching identity and the closure theorem**
(`research/proof/branching_identity.md`), PROVED/FACT, whose verdict the owner corrected on
2026-09-06: the closure is ROUTE-SHAPED, and "nothing here bounds it" is a brick, not a verdict.
The parent named three candidate functionals - legal-word density per opening by depth, all-pad
density, the order variance - and never ran them. This is the motor's gate item 3
(`research/proof/objects_ledger.md` O-M3, "the single named unrun item on the motor"), opened
under the owner's rule that the motor's gate items are worked before the clutch. It carries gate
item 1 (`L(M)` bounded) and gate item 2 (the chain statement at `J = 3, 4` on the band `[15, 36]`
at `29 -> 31`) as exact finite checks off the same instrument.

Scripts in `research/anchor235/r66/` (prefix `mo_`); result outputs in
`research/anchor235/r66/results/` (untracked). Every number this document relies on is written
into the document.

---

## 0. Pre-registered

Written before any computation of this branch. The corpus rows this branch already had in front
of it when the predictions were written are flagged where they are used, so no prediction is
credited as blind that was not.

### 0.1 The object, defined exactly

`M` a machine with gears `5..y`, period `P`, `N` openings, cyclic gap sequence; `q'` the next
gear, `u = 6^{-1} mod q'`, `d = 2u mod q'`, teeth `{u, -u}`, letters `PAD` (`v = 0 mod q'`),
`UP` (`+d`), `DOWN` (`-d`), `BAD` otherwise, legality = no two consecutive equal nonzero letters
with pads transparent (docs/proofs/05 (F)); `L(M)` the longest realised legal word,
`J_max = L + 2` (docs/proofs/10); `a_L(q') = min(d, q' - d)`, the letter floor.

**The operator.** `T_{q'}` acts on the multiset of realised `K`-windows of consecutive gap sizes,
`D_K^#(M)`, and returns `D_m^#(M + q')` exactly (`branching_identity.md` Theorem 5, implemented
as a machine in `ladder_closure.md` 1). One application is: for every row `w = (g_1..g_K)` with
multiplicity `mult(w)` and every phase `z in Z_{q'}`, mark the opening at offset
`o_i = g_1 + ... + g_i` struck iff `o_i + z = 0` or `d (mod q')`; if `o_0` survives the pair
`(w, z)` is one opening of `M + q'` and the new gaps at it are the successive differences of the
surviving offsets. On the count side the same operator is the branching identity
`n_J = C_{J-1} - 2 C_J + C_{J+1}` with `C_r = W_{r-1} + Z_{r-1}` (`branching_identity.md` 2.2,
2.3). The gap multiset alone is NOT a state for `T`: the operator needs the windows, which is the
first thing this branch has to say about "a functional of the spectrum".

`Q*_J(M; q')` is the largest span of a realised `J`-fusion; the record law is
`F(M + q') = max_{J <= J_max} Q*_J(M; q')`. Written out, a `J`-fusion is a run of `J` consecutive
gaps of `M` whose two flanks are BAD and whose `J - 2` middles form a legal word - so `Q*_J` is a
constrained `F_J`, and `F_J(M)` (the widest stretch of `M` carrying `J - 1` openings) is its free
relaxation.

**Useful.** A functional `Phi` on the closure's state is USEFUL if (i) `Phi(S) >= F(S)`, and
(ii) `Phi(T_{q'} S) <= Phi(S) + c(q')` with `c` explicit and `c(q') <= q'` - because with both,
the budget inequality follows.

### 0.2 The theory

**T. There is no useful functional of the closure that is not one of the three statements the
tree already has, and the reason is dimensional.** A functional that bounds `F` must be measured
in columns and must therefore grow along the ladder at least as fast as `F` does; the closure's
own invariants (densities, variances, order distributions) are scale-free and bounded, so they
cannot bound anything that grows. The only column-valued functionals the closure offers are the
`Q*_J` / `F_J` family, and their monotonicity statements are, `J` by `J`: `J = 1` the budget
inequality, `J = 2` the pair statement (twin-Bertrand), `J >= 3` the chain statement. Hence the
search for a monotone functional is not a separate problem; it collapses onto the three, and its
value is to say exactly WHERE the collapse happens and what survives as a measured law.

### 0.3 Predictions, each with the number that would refute it

- **M1 (the operator).** `T` implemented as the closure step reproduces `F(23) = 34`,
  `F(29) = 43`, `F(31) = 58`, `F(37) = 88`, `F(41) = 91` from the rung below, with
  `over0 = 0`, `sum m = (q'-2) N` and `sum v m = q' P` at every rung. REFUTED by one wrong record
  or one nonzero `over0` on an unpruned rung.
- **M2 (candidate (a): the record itself).** `Phi = F`. Prediction: `F(M + q') <= F(M) + q'` at
  every rung `5 -> 7 .. 37 -> 41`, slack never negative and never monotone. Verdict fixed in
  advance: NOT USEFUL as a tool - `Phi = F` is monotone with `c(q') = q'` exactly when the budget
  holds, so it assumes what it must prove. REFUTED (as a measurement) by a negative slack.
- **M3 (candidate (b): the `F_J` ladder, the second-largest gap, the top-`J` sum).**
  `F_J(M + q') <= F_J(M) + q'` for every `J`. Prediction: HOLDS for `J <= 5` at every rung and
  FAILS at `J = 6`; the refuting rung is predicted to be `19 -> 23`, where the corpus rows give
  `F_6(m19) = 50` and `F_6(m23) = 77`, an increment of 27 against `q' = 23`. (Flag: this
  refutation is read off the corpus `F_j` rows before computing; the branch's job is to verify it
  on the instrument and to find the merge that does it.) Second-largest realised gap value
  `F^{(2)}`: predicted increments `<= q'` at every rung. Top-`J` sum of realised values: predicted
  to fail earlier than `F_J`, because it adds `J` independent record-sized values.
- **M4 (candidate (c): excess over threshold).** `E_x(M) = sum over gaps of (g - x)_+`, and its
  densities `E_x / P` and `E_x / N`. Prediction: fusion is superadditive, `(a+b-x)_+ >=
  (a-x)_+ + (b-x)_+`, so the closure moves tail mass strictly UP at every rung and every fixed
  `x`: the density `E_x / N` rises at every rung, so there is no decreasing functional in this
  family. And `E_x >= F - x`, so it bounds `F` only through `x`, which is circular. REFUTED
  (and a finding) by one rung at which `E_x / N` falls.
- **M5 (candidate (d): merge depth; gate item 1).** `J_max = L + 2`. Prediction: `L` measured
  along the ladder is `1, 1, 1, 2, 1, 3, 3, 2, 2` at `m11..m41` (corpus rows, to be reproduced on
  the instrument), `L_bare <= 3` and `L_pad` climbing `1, 2, 2` at m29, m31, m37; `J_max` is
  `3, 2, 3, 3, 3, 4, 3, 5, 5, 4` - NOT monotone, and a small integer while `F` grows, so it
  cannot bound `F`. Gate item 1 stays OPEN and the ladder cannot close it: predicted, no
  mechanism on this instrument caps `L_pad`. REFUTED by a monotone `J_max`, or by an `L` on the
  ladder disagreeing with the corpus.
- **M6 (candidate (e): mine).** (i) The letter-floor discount
  `Phi_letter(M; q') = max_{J >= 2} [Q*_J - (J - 2) a_L(q')]` - the record's span with the
  mandatory interior cost removed, since every middle of a `J`-fusion is a letter and so is
  `>= a_L`. Prediction: the max is attained at `J = 2` at a majority of rungs, so `Phi_letter` is
  the pair statement wearing a discount, and its increments are not below `q'`. (ii) The chain
  half `Phi_3 = max_{J >= 3} [Q*_J - (J-2) a_L]`. Prediction: `Phi_3 - F(M)` is not of one sign
  (predicted values around `0, -1, +5, +15, -12` at rungs 23, 29, 31, 37, 41 from the recorded
  `Q*_J` rows). (iii) The scale-free triple the parent named: `W_1/N`, `Z_1/N`,
  `S = sum_{r>=2} C_r / N`, `Var(order)`. Prediction: all bounded in `[0, 1]`, hence USELESS as
  bounds whatever their monotonicity; and `W_1/N` is already known non-monotone (it rises
  `0.00775 -> 0.00905` at `41 -> 43`, `ladder_closure.md` 7.1). REFUTED as useless by any of the
  four exceeding 1 or growing with the machine.
- **M7 (the negative, pre-registered so the branch cannot claim a route).** No functional tested
  here will be both useful and provable. Predicted shape of the failure: every column-valued
  candidate's increment is bounded by `q'` only through the budget / pair / chain statements
  themselves. REFUTED by one column-valued functional bounding `F` whose increment is provably
  strictly below `q'` by an argument that does not assume any of the three.
- **M8 (the band; gate item 2).** At `29 -> 31`, enumerate every realised 3- and 4-piece fusion
  of m29 whose largest piece `a` lies in the band `[15, 36]`. Prediction: the maximum span on the
  band is exactly **58** (the record `(18, 10, 30)`, largest piece 30, is in the band), so the
  chain statement `Q*_J <= F(m29) + 31 = 74` holds on the band with margin **16**; and every
  `a` in the band has its own finite maximum, a curve that peaks in the interior. REFUTED by a
  span above 74 (a budget violation, to be double-checked before reporting) or by a maximum on
  the band other than 58.

**Stop rules.** The closure theorem and the size formula (`branching_identity.md` 2.6, 5.1), the
record law (docs/proofs/09), `J_max = L + 2` (docs/proofs/10), the `L_bare` cap
(docs/proofs/12), the branching identity, the span-threshold prune (`ladder_closure.md` 4.4) and
the corpus record ladder are cited and reused, never re-derived. Any sub-question that reduces to
the pair statement or the chain statement is stopped in one line and named.

### 0.4 Scorecard

| # | prediction | verdict | evidence |
|---|---|---|---|
| M1 | the operator reproduces the five records | **CONFIRMED** - 34, 43, 58, 88, 91 each from the rung below, `over0 = 0` and `loss = 0` at every unpruned rung, `sum m = (q'-2)N` and `sum v m = q' P` at all ten | 1, 2.2, `results/theta_K21_t89.json` |
| M2 | `Phi = F`: budget holds, not a tool | **CONFIRMED** - 10 rungs, no violation, slack 4, 9, 9, 10, 12, 14, 20, 16, 7, 38, not monotone; empty as a tool by construction | 3.1, 5.1 |
| M3 | `F_J` ladder holds to `J = 5`, fails at `J = 6`, rung `19 -> 23` | **CONFIRMED exactly** - holds `J <= 5` at all 8 rungs, fails at `J = 6, 7, 8` at `19 -> 23` (50 -> 77, 58 -> 83, 63 -> 88 against `q' = 23`); refuting merge exhibited; and impossible uniformly in `J` | 3.2, 5.2 |
| M4 | tail mass moves up monotonically, no decreasing functional | **CONFIRMED** - 54 of 54 density entries rise, by the identity `(a+b-x)_+ >= (a-x)_+ + (b-x)_+`; no rung is an exception | 3.4, 5.4 |
| M5 | `J_max` non-monotone, cannot bound; `L` open | **CONFIRMED** - `J_max` = 3, 2, 3, 3, 3, 4, 3, 5, 5, 4; 8 of 8 corpus `L` reproduced; `L_pad` uncapped on this instrument | 3.5, 5.5, 7.1 |
| M6 | letter discount collapses onto `J = 2`; scale-free triple useless as bounds | **CONFIRMED** - argmax at `J = 2` at 8 of 10 rungs (tie at a ninth); `Phi_3 - F(M)` = 0, -1, +5, +15, -12 at rungs 23, 29, 31, 37, 41, the predicted values exactly; the triple bounded in `[0,1]`. CORRECTED: `W_1/N` rises at five rungs, not three | 3.7, 3.8, 5.7, 5.8 |
| M7 | no useful functional; the collapse onto the three statements | **CONFIRMED, with the collapse located** - no candidate is both useful and proved; but the search leaves one measured survivor, `B_{L+1}`, useful and budget-monotone at every rung, whose remaining cost is a cap on `L` plus one finite-order lemma | 4, 5.9, 6 |
| M8 | the band max is 58, margin 16 | **CONFIRMED exactly** - band max 58 at `J = 3`, witnesses `(18,10,30)` and `(23,10,25)`, margin 16; complete enumeration, `n_J` reproduced digit for digit | 7.2 |

Not pre-registered, found in the running and reported as such: **the order law** of section 4.3
(`k* = L + 1`, 6 of 6 decisive rungs, 0 exceptions), and with it the bounded-order functional
`B_k` that makes candidate (e-iv).

---

## 1. Setup (exact ranges)

Everything is exact integer arithmetic; no sampling anywhere. Scripts in
`research/anchor235/r66/`, results in `research/anchor235/r66/results/` (untracked).

| object | range | cost | script |
|---|---|---|---|
| the machines m5..m23 by direct sieve in the anchored column coordinate; spectrum, `F`, `F_J` to `J = 12`, mean gap, tail excesses | full periods (`P_23 = 37,182,145` columns, `N_23 = 7,952,175` gaps) | 5 s | `mo_rungs.py` |
| one application of `T_{q'}` at each of the rungs `5->7 .. 23->29`, from `D_8^#(M)` | the whole new machine, never built | 3 s | `mo_rungs.py` |
| the iterated ladder `T_29, T_31, T_37` from `D_15^#(m23)` | m29 (214,708,725 gaps), m31 (6,226,553,025), m37 (217,929,355,875) | 587 s | `mo_ladder.py` |
| the maximum excursion `X(M)` from the centred opening walk, exactly (the integer walk `N op(j) - j P`) | m5..m23, full periods | 1 s | `mo_excursion.py` |
| the de Bruijn relaxation `B_k` and the order of interaction | every rung `5->7 .. 31->37`, `k = 1..5` | seconds per rung on top of the ladder | `mo_order.py` |
| the same order table off a second, deeper ladder (`K_0 = 16`: m29 at depth 11, 23,827,139 rows; m31 at depth 7, 8,977,010 rows) as an independence check | identical dictionaries and identical `B_k` | 1,532 s | `mo_order.py 4 16` |
| the band check at `29 -> 31`: every realised 2-, 3- and 4-piece fusion of m29 | `D_4^#(m29)` = 45,854 rows carrying all 214,708,725 openings, `loss = 0`, `over0 = 0` | 20 s | `mo_band.py` |
| the pruned ladder to m41 (span threshold `theta = 89`, `ladder_closure.md` 4.4) from `D_21^#(m23)` | m29, m31, m37, m41; `over0 = 0` and the corpus record at all four rungs | 2,578 s | `mo_theta.py` |
| the refuting merges, each pulled back to the machine below (which openings the rung deletes inside the run) | exact, full periods, m13..m23 | 20 s | `mo_witness.py` |

The closure step itself is **not** reimplemented: `mo_core.py` imports
`research/anchor235/r61/lc_core.py`, so the operator tested here is literally the one that
computed `F(37) = 88` and `F(41) = 91` in `ladder_closure.md`.

**Instrument gates, all passed.** Every rung reproduces its corpus record: `F = 5, 7, 11, 18, 25,
34, 43, 58, 88` at m7..m37 from the machine one rung below, with `over0 = 0` and `loss = 0` at
every unpruned rung; `sum m = (q'-2) N` and `sum v m = q' P` exactly at all ten rungs; the `F_j`
rows match `alignment-rules.md` 3.7 entry for entry at m13, m17, m19, m23, m29, m31; `L` matches
the corpus at m11, m13, m17, m19, m23, m29, m31, m37; `n_J` and `Q*_J` reproduce
`branching_identity.md` 4.4 and `ladder_closure.md` 3.2 digit for digit, including
`n_J(29->31) = 5,805,160,589 / 413,380,422 / 7,999,018 / 12,992 / 4`.

## 2. Item 1: the operator, and what its state has to be

### 2.1 `T_{q'}` written out

Adding `q'` makes `q'` copies of `M`'s period, and copy `j` realises deletion phase
`r_j = -u - jP (mod q')`, a bijection of `Z_{q'}` (docs/proofs/05 (A)). So one application of the
operator is a sum over (window, phase) pairs:

> **`T_{q'}`.** Input the multiset `D_K^#(M)` of realised `K`-windows of consecutive gap sizes.
> For every row `w = (g_1, ..., g_K)` with multiplicity `mult(w)` and every `z in Z_{q'}`, put
> `o_0 = 0`, `o_i = g_1 + ... + g_i`, and mark `o_i` STRUCK iff `o_i + z = 0` or `d (mod q')`,
> with `d = 2 * 6^{-1} mod q'`. If `o_0` is unstruck, the pair `(w, z)` is one opening of
> `M + q'`, and the gaps of `M + q'` at it are the successive differences of the unstruck offsets.
> Adding `mult(w)` to that tuple's count over all pairs gives `D_m^#(M + q')` exactly, provided
> every pair reaches `m` new gaps inside the `K` old ones (`loss = 0` certifies it).

On the count side the same operator is the branching identity `n_J = C_{J-1} - 2C_J + C_{J+1}`
with `C_r = W_{r-1} + Z_{r-1}` (`branching_identity.md` 2.2, 2.3); on the size side it is
`m_{M+q'}(v) = sum_J sum over J-windows of span v of eps_J`, `eps_J in {0, 1, 2}`
(`branching_identity.md` 2.6).

**The first thing to record is a negative about the state.** The brief allows the operator to act
"on gap multisets (or on the spectrum `C_r`)". It cannot. The spectrum of `M` does not determine
the spectrum of `M + q'`, because the new sizes are sums over *consecutive* runs: the operator's
state is the window multiset `D_K^#(M)` and nothing smaller. Section 4 makes that quantitative
rather than rhetorical - the best bound on `F(M + q')` derivable from the set of realised sizes
alone (the spectrum's support) is `B_1`, the order-1 relaxation, and it exceeds the budget from the
rung `13 -> 17` on, by a factor rising to 2.4 at `29 -> 31` (179 against 74).

### 2.2 The five records, each from the rung below

| rung | input | depth in -> out | `F(M + q')` | corpus | `n_J` | `Q*_J` | `over0` |
|---|---|---|---|---|---|---|---|
| `19 -> 23` | `D_8^#(m19)`, 83,681 rows | 8 -> 4 | **34** | 34 | 7,206,695 / 733,672 / 11,746 / 62 | 25, 31, 33, 34 | 0 |
| `23 -> 29` | `D_8^#(m23)`, 661,333 rows | 8 -> 3 | **43** | 43 | 199,048,197 / 15,416,706 / 243,822 | 34, 39, 43 | 0 |
| `29 -> 31` | `D_10^#(m29)`, 15,240,585 rows | 10 -> 6 | **58** | 58 | 5,805,160,589 / 413,380,422 / 7,999,018 / 12,992 / 4 | 43, 55, 58, 55, 55 | 0 |
| `31 -> 37` | `D_6^#(m31)`, 2,678,901 rows | 6 -> 2 | **88** | 88 | 205,591,124,261 / 12,223,428,142 / 114,732,724 / 70,532 / 216 | 58, 68, 85, 88, 68 | 0 |
| `37 -> 41` | `D_5^#(m37)` pruned at `theta = 89` | 5 -> 2 | **91** | 91 | -- | 88, 90, 90, 91 | 0 |

Two `F_j` entries fall out that are not on record: **`F_9(m29) = 99` and `F_10(m29) = 110`**, from
the depth-10 dictionary the `K_0 = 15` ladder leaves at m29 (the recorded rows stop at
`F_8(m29) = 97`).

## 3. Items 2 and 3: the candidates, exactly, at every rung

Write `Phi` for a functional of the closure's state. USEFUL means `Phi >= F` and
`Phi(T_{q'} S) <= Phi(S) + c(q')` with `c(q') <= q'`; the two together give the budget.

### 3.1 Candidate (a): the record itself

| rung | `F(M)` | `q'` | `F(M+q')` | budget | slack |
|---|---|---|---|---|---|
| 5->7 | 2 | 7 | 5 | 9 | 4 |
| 7->11 | 5 | 11 | 7 | 16 | 9 |
| 11->13 | 7 | 13 | 11 | 20 | 9 |
| 13->17 | 11 | 17 | 18 | 28 | 10 |
| 17->19 | 18 | 19 | 25 | 37 | 12 |
| 19->23 | 25 | 23 | 34 | 48 | 14 |
| 23->29 | 34 | 29 | 43 | 63 | 20 |
| 29->31 | 43 | 31 | 58 | 74 | 16 |
| 31->37 | 58 | 37 | 88 | 95 | 7 |
| 37->41 | 88 | 41 | 91 | 129 | 38 |

Ten rungs, no violation; the slack is 4, 9, 9, 10, 12, 14, 20, 16, **7**, 38 and is not monotone.
`Phi = F` is trivially useful in form and empty in content: it is monotone with `c(q') = q'`
exactly when the budget holds. It is on the list only so the scorecard can record that the branch
did not smuggle the target in as a tool. **Verdict: NOT A TOOL** (fixed in advance).

### 3.2 Candidate (b): the `F_J` ladder

`F_J(M)` is the widest stretch of `M` carrying `J - 1` openings - the free relaxation of `Q*_J`
(drop the flank and legality conditions). Exact rows, the ladder's own plus the corpus rows where
the instrument's depth ran out:

| machine | `F_1` | `F_2` | `F_3` | `F_4` | `F_5` | `F_6` | `F_7` | `F_8` |
|---|---|---|---|---|---|---|---|---|
| m5 | 2 | 4 | 5 | - | - | - | - | - |
| m7 | 5 | 7 | 11 | 13 | 16 | 18 | 21 | 23 |
| m11 | 7 | 11 | 16 | 18 | 23 | 26 | 28 | 30 |
| m13 | 11 | 16 | 23 | 26 | 28 | 31 | 34 | 38 |
| m17 | 18 | 25 | 28 | 33 | 35 | 40 | 43 | 48 |
| m19 | 25 | 31 | 35 | 38 | 47 | 50 | 58 | 63 |
| m23 | 34 | 39 | 50 | 58 | 65 | 77 | 83 | 88 |
| m29 | 43 | 55 | 65 | 70 | 85 | 90 | 92 | 97 |
| m31 | 58 | 68 | 85 | 90 | 92 | 97 | 104 | 110 |
| m37 | 88 | 90 | 97 | - | - | - | - | - |

The increments `F_J(M + q') - F_J(M)`, with `*` marking one above `q'`:

| rung | `q'` | `J=1` | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| 7->11 | 11 | 2 | 4 | 5 | 5 | 7 | 8 | 7 | 7 |
| 11->13 | 13 | 4 | 5 | 7 | 8 | 5 | 5 | 6 | 8 |
| 13->17 | 17 | 7 | 9 | 5 | 7 | 7 | 9 | 9 | 10 |
| 17->19 | 19 | 7 | 6 | 7 | 5 | 12 | 10 | 15 | 15 |
| 19->23 | 23 | 9 | 8 | 15 | 20 | 18 | **27\*** | **25\*** | **25\*** |
| 23->29 | 29 | 9 | 16 | 15 | 12 | 20 | 13 | 9 | 9 |
| 29->31 | 31 | 15 | 13 | 20 | 20 | 7 | 7 | 12 | 13 |
| 31->37 | 37 | 30 | 22 | 12 | - | - | - | - | - |

**The `F_J` budget holds at `J <= 5` at every rung tested and FAILS at `J = 6, 7, 8` at the rung
`19 -> 23`** (`F_6`: 50 -> 77, an increment of 27 against `q' = 23`; `F_7`: 58 -> 83; `F_8`:
63 -> 88). That is the pre-registered refutation, verified on the instrument. It is not an
accident of that rung:

> **Law (proved here).** Let `mu(M) = P/N` be the mean gap and `X(M) = max_J (F_J(M) - J mu(M))`
> the maximum excursion (finite; 3.5). Then `F_J(M) <= J mu(M) + X(M)` by the definition of `X`,
> and `F_J(M + q') >= J mu(M + q')` because the widest stretch of `J` gaps is at least the
> average. Since `mu(M + q') = mu(M) q'/(q'-2)`,
>
>     F_J(M + q') - F_J(M)  >=  2 J mu(M) / (q' - 2)  -  X(M),
>
> which exceeds `q'` for every `J > (q' + X(M))(q' - 2) / (2 mu(M))`. **So the `F_J` budget must
> fail for all large `J`, at every rung, for a reason that has nothing to do with records: the
> mean gap itself grows by the factor `q'/(q'-2)` per rung.** The bound is weak (it predicts
> failure beyond `J = 404` at `19 -> 23`, where the true first failure is `J = 6`), but it settles
> the shape: candidate (b) in its uniform-in-`J` form is impossible, not merely refuted.

### 3.3 Candidate (b'): the second-largest gap and the top-`J` sum

| rung | `q'` | 2nd largest value `M -> M+q'` | incr | top-3 value sum `M -> M+q'` | incr |
|---|---|---|---|---|---|
| 5->7 | 7 | 1 -> 3 | 2 | 3 -> 10 | 7 |
| 7->11 | 11 | 3 -> 6 | 3 | 10 -> 18 | 8 |
| 11->13 | 13 | 6 -> 10 | 4 | 18 -> 29 | 11 |
| 13->17 | 17 | 10 -> 16 | 6 | 29 -> 49 | **20\*** |
| 17->19 | 19 | 16 -> 23 | 7 | 49 -> 70 | **21\*** |
| 19->23 | 23 | 23 -> 33 | 10 | 70 -> 99 | **29\*** |
| 23->29 | 29 | 33 -> 40 | 7 | 99 -> 122 | 23 |
| 29->31 | 31 | 40 -> 55 | 15 | 122 -> 166 | **44\*** |
| 31->37 | 37 | 55 -> 85 | 30 | 166 -> 250 | **84\*** |

**The second-largest realised value is budget-monotone at 9 of 9 rungs** - increments 2, 3, 4, 6,
7, 10, 7, 15, 30, every one at or below `q'`. That is a new measured law and it is not implied by
the budget, since it is a statement about the second value and not the first; it is also not
useful, because the second value does not bound the first. **The top-3 sum fails from `13 -> 17`
on**, as pre-registered: it adds three independent record-sized values and a rung moves all three.

### 3.4 Candidate (c): excess over threshold

`E_x(M) = sum over gaps of (g - x)_+` per period; densities `E_x/N` (per gap):

| machine | `E_2/N` | `E_5/N` | `E_10/N` | `E_20/N` | `E_30/N` | `E_40/N` |
|---|---|---|---|---|---|---|
| m7 | 0.5333 | 0 | 0 | 0 | 0 | 0 |
| m11 | 1.0074 | 0.0889 | 0 | 0 | 0 | 0 |
| m13 | 1.4977 | 0.2828 | 0.0081 | 0 | 0 | 0 |
| m17 | 1.9300 | 0.5153 | 0.0481 | 0 | 0 | 0 |
| m19 | 2.3665 | 0.7898 | 0.1194 | 0.0012 | 0 | 0 |
| m23 | 2.7638 | 1.0669 | 0.2083 | 0.0051 | 7.29e-06 | 0 |
| m29 | 3.1045 | 1.3197 | 0.3040 | 0.0115 | 7.20e-05 | 2.79e-08 |
| m31 | 3.4443 | 1.5842 | 0.4175 | 0.0222 | 3.13e-04 | 2.38e-06 |
| m37 | 3.7470 | 1.8271 | 0.5329 | 0.0359 | 8.52e-04 | 1.64e-05 |

**Every column rises at every rung, 0 exceptions** - and the mechanism is one line, not a trend:
`(a + b - x)_+ >= (a - x)_+ + (b - x)_+` for `a, b >= 0`, so fusing gaps can only move excess mass
up, and fusing is what a rung does. So `E_x` and every density built from it is monotone in the
WRONG direction, at every threshold, forever; there is no decreasing functional in this family.
As a bound it is circular: `E_x(M) >= F(M) - x`, so `F <= x + E_x` says nothing that `x` did not
already say. **Verdict: REFUTED as a route, by an identity rather than by a rung.**

### 3.5 Candidate (d): merge depth, and `L(M)` (gate item 1)

| machine | `L(M)` w.r.t. the next gear | `L_bare` | `L_pad` | `J_max = L + 2` | corpus `L` |
|---|---|---|---|---|---|
| m5 | 1 | 1 | 0 | 3 | - |
| m7 | 0 | 0 | 0 | 2 | - |
| m11 | 1 | 1 | 0 | 3 | 1 |
| m13 | 1 | 1 | 0 | 3 | 1 |
| m17 | 1 | 1 | 0 | 3 | 1 |
| m19 | 2 | 2 | 1 | 4 | 2 |
| m23 | 1 | 1 | 1 | 3 | 1 |
| m29 | 3 | 3 | 1 | 5 | 3 |
| m31 | 3 | 3 | 2 | 5 | 3 |
| m37 | 2 | 1 | 2 | 4 | 2 |

Eight of eight corpus entries reproduced. `J_max` is `3, 2, 3, 3, 3, 4, 3, 5, 5, 4` - **not
monotone in either direction** - and it is a small integer while `F` runs 2 to 91, so it cannot
bound `F`. **Verdict: not useful, on both counts.**

**Gate item 1, status, stated exactly.** `L = max(L_bare, L_pad)`. `L_bare <= 5` is proved
(docs/proofs/12, KERNEL) and the measured `L_bare` is `1, 0, 1, 1, 1, 2, 1, 3, 3, 1` - at or below
3 everywhere on this ladder, with its maximum at m29 and m31, not at the top. `L_pad` is the open
half: `0, 0, 0, 0, 0, 1, 1, 1, 2, 2` here, and the corpus continues `2, 2, 3, 3` at m41, m43, m47,
m53 (`objects_ledger.md` O-M1). **Nothing on this instrument caps it**, and the reason is
structural, not computational: a PAD letter is any gap divisible by `q'`, so the padded alphabet
is nonempty as soon as the machine realises the size `q'`, which it does from m19 on - the all-pad
density `Z_1/N` is 2.271e-04, 7.545e-07, 9.734e-06, 4.234e-06, 2.820e-07 at m19..m37 and never
returns to 0. `L_pad` grows because the padded alphabet grows with the machine, and no functional
measured here changes that. **Gate item 1 stays OPEN; this branch adds the ladder values and the
mechanism, not a cap.**

### 3.6 Candidate (e-i), mine: the maximum excursion

`X(M) = max_J (F_J(M) - J mu(M))`, the largest amount by which a run of consecutive gaps outruns
the machine's own mean rate. It is the natural column-valued functional that is not one of the
three statements, it is exactly computable from the centred opening walk in one pass
(`X = max_j S_j - min_j S_j` with `S_j = op(j) - jP/N`, kept as the integer `N S_j`), and it
bounds the record: `F(M) <= X(M) + mu(M)`.

| machine | `mu` | `F` | `X` | `X + mu` | increment of `X` | `q'` |
|---|---|---|---|---|---|---|
| m5 | 1.66667 | 2 | 0.667 | 2.333 | - | - |
| m7 | 2.33333 | 5 | 4.667 | 7.000 | 4.00 | 7 |
| m11 | 2.85185 | 7 | 14.148 | 17.000 | 9.48 | 11 |
| m13 | 3.37037 | 11 | 23.037 | 26.407 | 8.89 | 13 |
| m17 | 3.81975 | 18 | 56.094 | 59.914 | **33.06\*** | 17 |
| m19 | 4.26914 | 25 | 141.546 | 145.815 | **85.45\*** | 19 |
| m23 | 4.67572 | 34 | 251.256 | 255.932 | **109.71\*** | 23 |

**REFUTED at `13 -> 17`, and then twice more.** The mechanism is exact and worth keeping: the run
that realises `X` is 2,081 gaps long at m17 and **245,506 gaps at m19, 232,994 at m23** (spanning
1,048,240 and 1,089,666 columns). `X` is not a record statistic at all - it is the machine's
long-range imbalance, a discrepancy of the opening set over a fifth of its period, and it exceeds
`F` by a factor of 7.4 at m23. Any functional that lets `J` run free measures equidistribution,
not the record; that is candidate (b)'s large-`J` failure seen from the other side, and it is why
every surviving candidate below has `J` capped at `J_max`.

### 3.7 Candidate (e-ii), mine: the letter-floor discount

Every middle of a `J`-fusion is a letter, hence `= 0` or `+-d (mod q')` and so at least
`a_L(q') = min(d, q' - d)`, which is `(q' -+ 1)/3` for the real teeth (`3a = q' -+ 1`). Discount
it: `Phi_letter(M) = max_{J >= 2} [Q*_J - (J-2) a_L]`, with chain half
`Phi_3 = max_{J >= 3} [Q*_J - (J-2) a_L]`.

| rung | `a_L` | `Q*_J` | `Phi_letter` | argmax `J` | `Phi_3` | `F(M)` | `Phi_letter - F(M)` |
|---|---|---|---|---|---|---|---|
| 5->7 | 2 | 2, 3, 5 | 3 | 3 | 3 | 2 | 1 |
| 7->11 | 4 | 5, 7 | 7 | 2 | - | 5 | 2 |
| 11->13 | 4 | 7, 11, 8 | 11 | 2 | 4 | 7 | 4 |
| 13->17 | 6 | 11, 16, 18 | 16 | 2 | 12 | 11 | 5 |
| 17->19 | 6 | 18, 25, 25 | 25 | 2 | 19 | 18 | 7 |
| 19->23 | 8 | 25, 31, 33, 34 | 31 | 2 | 25 | 25 | 6 |
| 23->29 | 10 | 34, 39, 43 | 39 | 2 | 33 | 34 | 5 |
| 29->31 | 10 | 43, 55, 58, 55, 55 | 55 | 2 | 48 | 43 | 12 |
| 31->37 | 12 | 58, 68, 85, 88, 68 | 73 | 3 | 73 | 58 | 15 |
| 37->41 | 14 | 88, 90, 90, 91 | 90 | 2 | 76 | 88 | 2 |

`Phi_letter` is monotone with `c(q') = q'` at 9 of 9 rungs (increments 4, 4, 5, 9, 6, 8, 16, 18,
17) and exceeds `F(M)` everywhere - **and it is empty, because its maximum sits at `J = 2` at 8 of
10 rungs, where the discount is zero and `Phi_letter = Q*_2`; and `Q*_2 <= F(M) + q'` IS the pair
statement**, which `objects_ledger.md` lists as the conjecture in disguise (at column 0 it reads
`2 d_0 <= F + q'`, and every route to it is twin-Bertrand). Worse, it does not bound the NEXT
record: `Phi_letter(M) < F(M + q')` at 7 of 10 rungs (3 against 5, 16 against 18, 31 against 34,
39 against 43, 55 against 58, 73 against 88, 90 against 91), so its monotonicity carries nothing
forward. **Verdict: the pair statement wearing a discount** - pre-registered and confirmed. The
one rung where the maximum moves off `J = 2` is `31 -> 37`, the narrow-slack rung, and there
`Phi_3 = 73` exceeds `F(m31)` by 15: the letter floor buys back less than the fourfold fusion
spends.

### 3.8 Candidate (e-iii): the three functionals the parent named

`W_r/N` the legal-word density by depth, `Z_r/N` the all-pad density, `S = sum_{r>=2} C_r / N`,
and `Var(order) = 2[(q'-4) + S(q'-2)]/(q'-2)^2` (`branching_identity.md` 2.5).

| machine | `W_1/N` | `W_2/N` | `Z_1/N` | `S` | `Var(order)` |
|---|---|---|---|---|---|
| m5 | 0.6666667 | 0 | 0 | 0.6666667 | 0.506667 |
| m7 | 0 | 0 | 0 | 0 | 0.172840 |
| m11 | 0.0444444 | 0 | 0 | 0.0444444 | 0.156841 |
| m13 | 0.0484848 | 0 | 0 | 0.0484848 | 0.122020 |
| m17 | 0.0488440 | 0 | 0 | 0.0488440 | 0.109553 |
| m19 | 0.0311190 | 1.637e-04 | 2.271e-04 | 0.0315099 | 0.089169 |
| m23 | 0.0306603 | 0 | 7.545e-07 | 0.0306610 | 0.070858 |
| m29 | 0.0373665 | 6.055e-05 | 9.734e-06 | 0.0374369 | 0.066791 |
| m31 | 0.0184449 | 1.140e-05 | 4.234e-06 | 0.0184606 | 0.054932 |
| m37 | 0.0077489 | 1.401e-08 | 2.820e-07 | 0.0077493 | 0.049050 |

- **`W_1/N` is NOT monotone**, at five of the ten rungs: `7 -> 11` (0 -> 0.044444),
  `11 -> 13` (0.044444 -> 0.048485), `13 -> 17` (0.048485 -> 0.048844), `23 -> 29` (0.030660 ->
  0.037367) and `37 -> 41` (0.007749 -> 0.00905, `ladder_closure.md` 7.1). The same for `S`, which
  equals `W_1/N` to four figures.
- **`Z_1/N` is NOT monotone**: it is 0 up to m17, becomes nonzero at `17 -> 19`, and rises again at
  `23 -> 29` (7.545e-07 -> 9.734e-06, a factor of 13) and at `37 -> 41` (2.820e-07 -> 1.977e-06).
- **`Var(order)` IS monotone**, strictly decreasing at 9 of 9 rungs: 0.5067, 0.1728, 0.1568,
  0.1220, 0.1096, 0.0892, 0.0709, 0.0668, 0.0549, 0.0490. Its monotonicity is very nearly an
  identity rather than a fact about the machine: `Var = 2[(q'-4) + S(q'-2)]/(q'-2)^2` is
  `2/q' + O(1/q'^2)` plus `2S/(q'-2)`, so it falls because `q'` grows, and the machine enters only
  through `S`, whose contribution `Var - floor` is 0.00227, 0.00258, 0.00105, 0.00040 at the top
  four rungs.

**All three are dimensionless and bounded** - `W_r/N`, `Z_r/N`, `S` and `Var` lie in `[0, 1]` and
tend to 0 - so whatever their monotonicity they cannot bound a record that grows without bound.
The one that is monotone is monotone for the gear's reason, not the machine's.
**Verdict: the parent's three named candidates are settled - two refuted as non-monotone with the
exact rung, one monotone and empty.**

---

## 4. The order table and its sharp law

This is the branch's own candidate and the one thing here that is neither a re-derivation nor a
refutation. It is candidate **(e-iv)**, and it was **not pre-registered**: it was constructed after
2.1 showed that the operator's state cannot be smaller than the window multiset, which raised the
question this section answers - relax the closure to bounded order and ask how much order the
budget actually needs. It is reported as unregistered in the scorecard (0.4).

### 4.1 The relaxation, defined

The record law says `F(M + q') = max_{J <= J_max} Q*_J(M; q')`, and a `J`-fusion is a run of `J`
consecutive gaps of `M` that is REALISED in `M` and fuses at some phase of `q'`. "Realised in `M`"
is a condition on the whole window. Relax it to a condition of bounded order:

> A `J`-word `(g_1, ..., g_J)` is **level-`k` admissible** iff every `k` consecutive entries of it
> lie in `D_k(M)`, the set of realised `k`-windows (for `J <= k`, iff the word itself lies in
> `D_J(M)`). Put
>
>     B_k(M; q') := max { span of a level-k admissible J-word that fuses at
>                         some phase, over J <= J_max }.

`D_k(M)` is a de Bruijn-style dictionary and level-`k` admissibility is a walk in its transition
graph, so `B_k` is computed exactly by a dynamic programme on the state
`(last k-1 gaps, (offset + phase) mod q')` with the `q'` phases tested by brute force - no word
theory is assumed (`mo_order.py`, `relaxed_J`). Three facts hold by construction:

- `B_1 >= B_2 >= ... >= B_{J_max}`, because level-`(k+1)` admissible implies level-`k` admissible;
- `B_k(M; q') >= F(M + q')` for every `k`, because the true extremal fusion is realised and hence
  level-`k` admissible for all `k`; so **every `B_k` bounds the next record**;
- `B_k(M; q') = F(M + q')` for `k >= J_max`, because then no relaxation is left.

So `B_k` is exactly the object the brief asks for: a functional of the closure's state that bounds
the record, computed from `D_k(M)` **alone** - a finite table - and the question "how much of the
machine does the budget need?" becomes the sharp question "at which `k` does `B_k` fall to or below
`F(M) + q'`?" Call that `k*`, the **order of interaction** the budget needs.

### 4.2 The table (exact, `mo_order.py`, `results/order.json`)

| rung | `q'` | `L(M)` | `J_max` | budget `F+q'` | `B_1` | `B_2` | `B_3` | `B_4` | `B_5` | `F(M+q')` | `k*` | rows in `D_{k*}` | gaps `N(M)` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5->7 | 7 | 1 | 3 | 9 | 6 | 6 | 5 | - | - | 5 | **1** | 2 | 3 |
| 7->11 | 11 | 0 | 2 | 16 | 10 | 7 | 7 | 7 | 7 | 7 | **1** | 4 | 15 |
| 11->13 | 13 | 1 | 3 | 20 | 18 | 11 | 11 | 11 | 11 | 11 | **1** | 7 | 135 |
| 13->17 | 17 | 1 | 3 | 28 | 33 | 21 | 18 | 18 | 18 | 18 | **2** | 52 | 1,485 |
| 17->19 | 19 | 1 | 3 | 37 | 49 | 30 | 25 | 25 | 25 | 25 | **2** | 133 | 22,275 |
| 19->23 | 23 | 2 | 4 | 48 | 96 | 56 | 35 | 34 | 34 | 34 | **3** | 1,216 | 378,675 |
| 23->29 | 29 | 1 | 3 | 63 | 97 | 60 | 43 | 43 | 43 | 43 | **2** | 429 | 7,952,175 |
| 29->31 | 31 | 3 | 5 | 74 | 179 | 111 | 85 | 58 | 58 | 58 | **4** | 45,854 | 214,708,725 |
| 31->37 | 37 | 3 | 5 | 95 | 239 | 160 | 118 | 88 | 88 | 88 | **4** | 115,193 | 6,226,553,025 |
| 37->41 | 41 | 2 | 4 | 129 | 299 | 161 | ? | - | - | 91 | ? | ? | 217,929,355,875 |

`L(M)` is the longest realised legal word with respect to the incoming gear and `J_max = L + 2`
(section 3.5); `k*` is the least `k` with `B_k <= F(M) + q'`.

### 4.3 The sharp law

> **The order law (measured, no exception).** The order of interaction the budget needs is exactly
> one more than the merge depth:
>
>     k*(M; q')  =  L(M) + 1  =  J_max - 1.
>
> Both halves, separately:
>
> - **`B_{L+1} <= F(M) + q'` at 9 of 9 rungs where it is computable** (m5..m31), margins
>   `budget - B_{L+1}` = 3, 6, 9, 7, 7, 13, **3**, 16, 7. The tight one is `23 -> 29`
>   (`B_2 = 60` against a budget of 63).
> - **`B_L > F(M) + q'` at 7 of 7 rungs from `13 -> 17` upward** (every rung with `F(M) >= 11`),
>   by 5, 12, 8, 34, 11, 23 and 32 columns: `B_1 = 33, 49, 97` against budgets 28, 37, 63 at
>   `13->17, 17->19, 23->29` (`L = 1`); `B_2 = 56, 161` against 48, 129 at `19->23, 37->41`
>   (`L = 2`); `B_3 = 85, 118` against 74, 95 at `29->31, 31->37` (`L = 3`).
>
> The only two rungs where `k* < L + 1` are `5 -> 7` and `11 -> 13`, the two machines small enough
> that even the order-1 relaxation is inside the budget (`B_1 = 6 <= 9`, `18 <= 20`). Excluding
> those, **`k* = L + 1` at 6 of 6 rungs, 0 exceptions.**

The law is sharp in both directions at those six rungs: order `L + 1` always suffices, order `L`
never does.

### 4.4 The mechanism: it is always the deepest fusion that fails

Not a trend - the per-`J` table (`results/order.json`, `per_J`) says which term of
`max_{J <= J_max} Q*_J` is doing it, and the answer is the same at every rung.

**At level `k = L`, the maximum of `B_L` is attained at `J = J_max` at 7 of 7 rungs** - the deepest
fusion, the one with `L` interior letters, relaxed by two de Bruijn steps. At `29 -> 31`, for
instance, the per-`J` row at `k = 3` is `43, 55, 58, 67, 85`: the exact terms `J <= 3` are inside
the budget, and the relaxed `J = 5` term at 85 is the whole of the violation. At level `k = L + 1`
that same term drops to 21, 30, 35, 60, 55, 75 (from 33, 49, 56, 97, 85, 118) and the maximum
lands inside the budget.

Why one step and not two. At `k = J_max - 1` exactly one term of the record law is relaxed, and it
is relaxed minimally: the word must be realised on every window of length `J_max - 1`, so only the
joint realisability of the two flanks with everything between them is dropped. At `k = J_max - 2`
two consecutive de Bruijn steps are free, and two free steps are enough to concatenate the widest
realised piece with a full letter chain that never occurs together in `M`. The witnesses are
explicit: at `23 -> 29` the level-1 extremal `J = 3` word is `(34, 29, 34)` at phase `z = 5`,
span **97** - the record gap of m23 taken twice with a PAD (`29 = 0 mod 29`) between them, a
configuration no window of m23 contains, where the true `Q*_3` is 43. Two free de Bruijn steps buy
the record twice; one free step does not.

**What the law buys.** At `k = L + 1` the state is a finite table and a small one: at `29 -> 31`
the budget is decided by `D_4(m29)`, **45,854 rows** standing for 214,708,725 gaps (a compression
of 4,682); at `31 -> 37` by `D_4(m31)`, **115,193 rows** for 6,226,553,025 gaps (54,053). So the
budget inequality at these rungs is not a statement about a machine with `10^9` gaps: it is a
statement about a table with `10^5` rows and a walk on it, and the order of that table is set by
`L`.

**Consequence for gate item 1.** `L(M)` is not one invariant among many: it is the order of
interaction. A cap `L <= c` turns the budget into a bounded-order statement about `D_{c+1}` -
uniformly in the rung - and that is the first thing on this instrument that makes gate item 1
*worth* the fight rather than merely open.

### 4.5 The table is not an artefact of the ladder, and the one rung not yet decided

**Independence check (`results/order_K15.json` vs `results/order_K16.json`).** The whole ladder was
rebuilt from a deeper base, `K_0 = 16` instead of 15 (`|D_16(m23)| = 4,897,851` rows; m29 comes out
at depth **11** with 23,827,139 rows instead of depth 10 with 15,240,585; m31 at depth **7** with
8,977,010 rows instead of depth 6 with 2,678,901). The order table is **identical**: the dictionary
sizes `|D_1..D_4|` are 41, 730, 7,184, 45,854 at m29 and 55, 1,253, 15,019, 115,193 at m31 from
both ladders, and `B_1..B_4` are 179, 111, 85, 58 and 239, 160, 118, 88 from both, with the same
`k* = 4`. So the `k`-window dictionaries the law is computed on are the complete ones and the law
is not an artefact of how deep the ladder happened to reach.

**The rung still undecided.** At `37 -> 41` both ladders leave m37 at depth 2 (the `m31 -> m37`
step's `mmin` stays at 2 when the input width goes 6 -> 7, so widening the base does not help), so
`D_3(m37)` is still unavailable. What is known is `B_2 = 161 > 129`, hence `k* >= 3`; and
`L(m37) = 2`, so the law predicts exactly `k* = 3`. That is the law's next test and a real one:
`B_3` must fall by at least 32 columns to satisfy it. Reaching it needs a ladder that gives m37
depth 3 - a base wider than `K_0 = 16`, or the span-threshold prune carried through with a
threshold low enough to keep every ancestor of a medium gap (`theta <= 33`), which the `theta = 89`
run of `mo_theta.py` does not.

## 5. The verdict on each candidate

USEFUL means both halves: `Phi >= F` (it bounds the record) and `Phi(T_{q'} S) <= Phi(S) + c(q')`
with `c(q') <= q'` explicit. The increments below are exact, one column per rung.

### 5.0 Every candidate's increment at every rung

`incr` is the change of the functional across the rung; `*` marks an increment above `q'`.

| candidate | 5->7 | 7->11 | 11->13 | 13->17 | 17->19 | 19->23 | 23->29 | 29->31 | 31->37 | 37->41 | `c(q')` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) `F` | 3 | 2 | 4 | 7 | 7 | 9 | 9 | 15 | 30 | 3 | `q'` |
| (b) `F_5` | - | 7 | 5 | 7 | 12 | 18 | 20 | 7 | - | - | `q'` (to `J=5`) |
| (b) `F_6` | - | 8 | 5 | 9 | 10 | **27\*** | 13 | 7 | - | - | none |
| (b') 2nd value | 2 | 3 | 4 | 6 | 7 | 10 | 7 | 15 | 30 | - | `q'` |
| (b') top-3 sum | 7 | 8 | 11 | **20\*** | **21\*** | **29\*** | 23 | **44\*** | **84\*** | - | none |
| (c) `E_5/N` | +0 | +0.089 | +0.194 | +0.232 | +0.275 | +0.277 | +0.253 | +0.264 | +0.243 | - | rises always |
| (d) `J_max` | - | -1 | +1 | 0 | 0 | +1 | -1 | +2 | 0 | -1 | not monotone |
| (e-i) `X` | 4.00 | 9.48 | 8.89 | **33.06\*** | **85.45\*** | **109.71\*** | - | - | - | - | none |
| (e-ii) `Phi_letter` | - | 4 | 4 | 5 | 9 | 6 | 8 | 16 | 18 | 17 | `q'` |
| (e-iii) `W_1/N` | -0.667 | **+0.044** | **+0.004** | **+0.000** | -0.018 | -0.000 | **+0.007** | -0.019 | -0.011 | **+0.001** | not monotone |
| (e-iii) `Var(order)` | -0.334 | -0.016 | -0.035 | -0.012 | -0.020 | -0.018 | -0.004 | -0.012 | -0.006 | - | decreasing |
| (e-iv) `B_{L+1}` | - | 4 | 1 | 10 | 9 | 5 | 25 | -2 | 30 | - | `q'` |

### 5.1 (a) The record itself - NOT A TOOL (fixed in advance)

Values `F` = 2, 5, 7, 11, 18, 25, 34, 43, 58, 88, 91 at m5..m41; increments 3, 2, 4, 7, 7, 9, 9,
15, 30, 3; slack `F(M)+q'-F(M+q')` = 4, 9, 9, 10, 12, 14, 20, 16, **7**, 38. Bounds `F`: by
definition. Monotone with `c(q') = q'`: at 10 of 10 rungs. Mechanism: every merge moves it - the
record of `M + q'` is a fusion of a run of `M`, so `F` is moved by the deepest fusion available.
Useful in form, empty in content: `F(M+q') <= F(M)+q'` **is** the budget. Verdict: **NOT A TOOL.**

### 5.2 (b) The `F_J` ladder - REFUTED at `19 -> 23`, and impossible for large `J`

Exact rows in 3.2. Holds for `J <= 5` at all 8 rungs measured; fails at `J = 6, 7, 8` at
`19 -> 23` (`F_6`: 50 -> 77 against `q' = 23`; `F_7`: 58 -> 83; `F_8`: 63 -> 88).

**The merge that breaks it** (`mo_witness.py`): the widest 6-run of m23 spans 77 and pulls back to
**ten** consecutive gaps of m19,

    m19:  12   2   5   10   8   5   3   4   3   25          (10 gaps, span 77)
    m23:  12       7        23           3   4   28         (6 gaps, span 77)

with fusion orders `1, 2, 3, 1, 1, 2` - the rung deletes **four** openings inside the run. That is
the mechanism, and it is general: if a `J`-run of `M + q'` is made by deleting `t` openings inside
a run of `M`, then `F_J(M + q') >= F_{J+t}(M)`, so the `F_J` budget needs
`F_{J+t}(M) - F_J(M) <= q'` for the largest `t` the machine realises inside a `J`-run - and
`F_{J+t} - F_J` grows like `t mu(M)` while `q'` does not care about `t`.

At this rung the inequality is an **equality**, and that is the sharpest way to see the failure.
The two `F_J` rows to `J = 12` are

    m19:  25  31  35  38  47  50  58  63  65  77  83  88
    m23:  34  39  50  58  65  77  83  88  90  95  97 102

and `F_J(m23) = F_{J+t}(m19)` exactly for `J = 3..8`, with `t = 3, 3, 4, 4, 4, 4`:
`50 = F_6(m19)`, `58 = F_7`, `65 = F_9`, **`77 = F_10`**, `83 = F_11`, `88 = F_12`. The rung's own
widest runs are simply the old machine's widest runs read `t` deletions later. So the `F_6` budget
is asking `F_10(m19) - F_6(m19) <= 23`, i.e. `77 - 50 = 27 <= 23`, which is false - and no rung
can help, because `t` is set by the arithmetic of the deletions and not by `q'`. (At `J = 1, 2`
the identity fails in the safe direction: `F_1(m23) = 34` sits strictly between `F_2(m19) = 31`
and `F_3(m19) = 35`, because the widest 3-run of m19 does not fuse.) The section 3.2 mean-gap law
says the same thing asymptotically (failure forced for all `J` above an explicit threshold, at
every rung). Bounds `F`:
yes for `J = 1` only, which is candidate (a). Verdict: **REFUTED, with the refuting merge; and
impossible uniformly in `J`, not merely refuted.**

### 5.3 (b') The second-largest realised value - MONOTONE, NEW, AND NOT A BOUND

Values 1, 3, 6, 10, 16, 23, 33, 40, 55, 85 at m5..m37; increments 2, 3, 4, 6, 7, 10, 7, 15, 30.
**Budget-monotone at 9 of 9 rungs**, `c(q') = q'`, with slack `q' - incr` = 5, 8, 9, 11, 12, 13,
22, 16, 7 - never below 5, and tightest at the top rung (`31 -> 37`: 30 against 37). This is a
measured law that the budget does not imply - it is a statement about the second
value, and no argument here derives it from the first. Mechanism: the second value is carried by
the second-widest fusion, which competes with the widest for the same deep merges, so a rung that
moves `F` a long way tends to move `F^{(2)}` with it (both jump 15 and 30 at the top two rungs).
Bounds `F`: **no** - `F^{(2)} < F` by definition, so a bound on it bounds nothing. Verdict:
**monotone and true at every rung, useless as a bound; kept as a new measured law.**

The **top-3 value sum** (3, 10, 18, 29, 49, 70, 99, 122, 166, 250) fails from `13 -> 17` on -
increments 7, 8, 11, **20**, **21**, **29**, 23, **44**, **84** against `q'` = 7, 11, 13, 17, 19,
23, 29, 31, 37 - as pre-registered: it adds three independent record-sized values (m17's top three
are 15, 16, 18; m13's are 8, 10, 11) and one rung moves all three. Verdict: **REFUTED at
`13 -> 17`.**

### 5.4 (c) Excess over threshold - REFUTED BY AN IDENTITY, at every rung and every threshold

`E_x/N` rows in 3.4. Every column rises at every rung, 0 exceptions in 54 entries. Mechanism, one
line and not a trend: `(a + b - x)_+ >= (a - x)_+ + (b - x)_+`, so **every** merge moves tail mass
up, and a rung is nothing but merges; the number of gaps falls by the factor `(q'-2)/q'` while the
mass above `x` cannot fall at all, so the density must rise. Monotone: yes - in the wrong
direction, with no rung and no threshold as an exception. Bounds `F`: only circularly
(`E_x >= F - x`). Verdict: **REFUTED as a route by an identity rather than by a rung** - there is
no decreasing functional in this family, now or at any rung above.

### 5.5 (d) Merge depth `J_max = L + 2` - NOT MONOTONE, CANNOT BOUND

Values 3, 2, 3, 3, 3, 4, 3, 5, 5, 4 at the rungs `5->7 .. 37->41`; increments -1, +1, 0, 0, +1,
-1, +2, 0, -1. Non-monotone in both directions: it falls from the rung `19->23` to `23->29`
(4 -> 3) and from `31->37` to `37->41` (5 -> 4), while `F` rises by 9 and by 3 across those same
rungs, and it jumps by +2 from `23->29` to `29->31`. Mechanism: `J_max` is set by the longest legal
word over the new gear's alphabet, so it is a property of the incoming gear's letter structure and
of the machine's realised gaps, not of the machine's size; the padded half `L_pad` is what climbs.
Bounds `F`: no - a small integer against a record that runs 2 to 91. Verdict: **not useful, on
both counts.** But see 4.4: `L` is the *order* the budget needs, which is a different and much
better job for it.

### 5.6 (e-i) The maximum excursion `X = max_J (F_J - J mu)` - REFUTED at `13 -> 17`

Values 0.667, 4.667, 14.148, 23.037, 56.094, 141.546, 251.256 at m5..m23; increments 4.00, 9.48,
8.89, **33.06**, **85.45**, **109.71** against `q'` = 7, 11, 13, 17, 19, 23. Bounds `F`: yes,
`F <= X + mu` - it is a genuine column-valued bound. Monotone: **no**, and it fails by a factor of
4.8 at `19 -> 23`. Mechanism, exact: the run that realises `X` is 2,081 gaps long at m17 and
**245,506 gaps at m19, 232,994 at m23**, spanning 1,048,240 and 1,089,666 columns - a fifth of the
period. `X` is not a record statistic at all; it is the discrepancy of the opening set, and it
exceeds `F` by a factor of 7.4 at m23. The merge that breaks it is not one merge: it is the
accumulation of the `2/q'` density loss over a quarter of a period, which is exactly what a rung
does to a long run. Verdict: **REFUTED - and the refutation is the same fact as (b)'s large-`J`
failure seen from the other side. Any functional that lets `J` run free measures equidistribution,
not the record.** That is why the surviving candidate (4.1) caps `J` at `J_max`.

### 5.7 (e-ii) The letter-floor discount - THE PAIR STATEMENT WEARING A DISCOUNT

Values 3, 7, 11, 16, 25, 31, 39, 55, 73, 90; increments 4, 4, 5, 9, 6, 8, 16, 18, 17, all at or
below `q'`; exceeds `F(M)` at 10 of 10 rungs. So it passes both formal tests and is still empty,
for a reason the numbers make exact: **its maximum sits at `J = 2` at 8 of 10 rungs, and ties
there at a ninth** (`5 -> 7`, argmax `J = 2, 3`), where the
discount `(J-2) a_L` is zero and `Phi_letter = Q*_2`, and `Q*_2 <= F(M) + q'` IS the pair
statement (at column 0, `2 d_0 <= F + q'`; `objects_ledger.md` lists it as the conjecture in
disguise). Worse, it does not bound the next record: `Phi_letter(M) < F(M + q')` at 7 of 10 rungs
(3<5, 16<18, 31<34, 39<43, 55<58, 73<88, 90<91), so its monotonicity carries nothing forward.
Mechanism: the letter floor buys back `a_L = (q' -+ 1)/3` per interior piece, and a deep fusion
spends more than that per piece, so the discounted maximum retreats to the shallowest fusion. The
one rung where the argmax moves off `J = 2` is `31 -> 37`, the narrow-slack rung, where
`Phi_3 = 73` exceeds `F(m31) = 58` by 15. Verdict: **stopped in one line as the pair statement**
(stop rule of 0.3).

### 5.8 (e-iii) The parent's three - TWO NON-MONOTONE, ONE MONOTONE AND EMPTY

Rows in 3.8. `W_1/N` (and `S`, equal to it to four figures) rises at **five** of the ten rungs -
`7 -> 11` (0 -> 0.04444), `11 -> 13` (0.04444 -> 0.04848), `13 -> 17` (0.04848 -> 0.04884),
`23 -> 29` (0.03066 -> 0.03737) and `37 -> 41` (0.00775 -> 0.00905); `Z_1/N` first becomes nonzero
at `17 -> 19` and then rises at `23 -> 29` (factor 13) and `37 -> 41` (factor 7). `Var(order)` falls
at 9 of 9 rungs, and falls for the gear's reason: `Var = 2[(q'-4) + S(q'-2)]/(q'-2)^2` is
`2/q' + O(1/q'^2)` plus `2S/(q'-2)`, so it is `q'` that makes it fall - the machine's contribution
`Var - 2(q'-4)/(q'-2)^2` is 0.00227, 0.00258, 0.00105, 0.00040 at the top four rungs. All four are
dimensionless, lie in `[0,1]` and tend to 0. Bounds `F`: **none of them can**, whatever their
monotonicity, because `F` grows without bound and they do not. Verdict: **settled - two refuted
with the exact rung, one monotone and empty for a dimensional reason.**

### 5.9 (e-iv) `B_{L+1}`, the bounded-order relaxation - THE ONE THAT SURVIVES

Values 6, 10, 11, 21, 30, 35, 60, 58, 88 at the rungs `5->7 .. 31->37`; increments 4, 1, 10, 9, 5,
25, -2, 30, **every one at or below `q'`** (8 of 8). Bounds `F`: **yes, and the next one** -
`B_{L+1}(M; q') >= F(M + q')` by construction, at 9 of 9 rungs with overshoot 1, 3, 0, 3, 5, 1,
**17**, 0, 0. Monotone with `c(q') = q'`: 8 of 8. Mechanism: the merges that move it are the
deepest fusions, `J = J_max`, exactly as for `F` - it is the record law with one flank condition
relaxed. Verdict: **the only candidate on the list that is both a bound and budget-monotone at
every rung, and the only one whose state is a finite table rather than the machine.** Section 6
says what it would give and what it still costs.

## 6. The monotone one

### 6.1 What it is

    Phi(M; q')  =  B_{L(M)+1}(M; q')  =  the widest span of a level-(L+1) admissible word,
                   of order J <= J_max, that fuses at some phase of q'.

Its state is `D_{L+1}(M)` - 52, 133, 1,216, 429, 45,854, 115,193 rows at the rungs `13->17`
upward. It bounds the next record at 9 of 9 rungs and it is budget-monotone at 8 of 8.

### 6.2 The theorem it would give

> **If `B_{L(M)+1}(M; q') <= F(M) + q'` for every anchored machine `M` and its next gear `q'`, the
> budget inequality holds at every rung**, because `F(M + q') <= B_k(M; q')` for every `k`.

Two things make this more than a restatement.

1. **It is strictly stronger than the budget, and still true.** At 4 of the 6 decisive rungs
   `B_{L+1} > F(M+q')` - 21 against 18, 30 against 25, 35 against 34, 60 against 43 - so the
   hypothesis asserts the budget for a strictly larger class of words than the machine realises,
   and the measurement says the machine has room for that: margins 7, 7, 13, 3, 16, 7. It is not
   the budget with extra notation; it is a sufficient condition that discards information about
   `M` and survives.
2. **It is a bounded-order statement.** It quantifies over words in a table of `10^2`-`10^5` rows,
   with a walk of length at most `J_max`, and mentions the machine only through `D_{L+1}(M)`. That
   is a different kind of object from "the widest gap of a machine with `6 x 10^9` gaps".

### 6.3 What remains to prove it, exactly

Two items, and 4.4 localises both.

- **(i) A cap on `L`.** `Phi`'s index is `L(M) + 1`. Without a cap the statement is a family
  indexed by an unbounded parameter and proves nothing uniform. This is gate item 1, and section
  7.1 says what is exact and what is open. A cap `L <= c` makes `Phi = B_{c+1}` one functional.
- **(ii) One lemma, at the deepest order.** By 4.4 the only term of `B_{L+1}` that is not already
  exact is `J = J_max`, and its relaxation is a single de Bruijn step. So the whole content is:
  > **Lemma (open).** Let `w = (g_1, ..., g_{J_max})` be a word every `(J_max - 1)`-subwindow of
  > which is realised in `M`, whose `J_max - 2` interior offsets are struck by `q'` at a common
  > phase and whose two flanks are not. Then `span(w) <= F(M) + q'`.
  >
  > Measured value of the left side at the six decisive rungs: 21, 30, 35, 60, 55, 75, against
  > budgets 28, 37, 48, 63, 74, 95. In its smallest instance (`J_max = 3`, `L = 1`) it reads: if
  > `(a,b)` and `(b,c)` are realised 2-windows of `M`, `b` is a letter (`b = 0` or `+-d mod q'`)
  > and the two flanks are BAD, then `a + b + c <= F(M) + q'`. That instance is the whole of the
  > budget at `13->17`, `17->19` and `23->29`, and its tightest measured case is `23 -> 29`, where
  > the extremal word is
  >
  >     (25, 10, 25) at phase z = 4:   25 + 10 + 25 = 60  <=  34 + 29 = 63,   margin 3,
  >
  > with `(25,10)` and `(10,25)` both realised 2-windows of m23 and the middle equal to the letter
  > floor `a_L(29) = d = 10`. The word `(25,10,25)` is **not** a realised 3-window of m23 - the true
  > `Q*_3` there is 43 - so the lemma at this rung is a genuine assertion about a word the machine
  > does not contain, and it holds with three columns to spare.

The lemma is not the pair statement (it is `J = 3`, not `J = 2`) and not the chain statement (it
asks about level-`(J_max-1)` admissible words, a strictly larger class than the realised chains
`Q*_J`). It is what the branch has to hand the tree in place of a monotone functional: a single
finite-order statement, with the exact instance to attack first.

### 6.4 Why the other candidates all failed, and what the failures are the shadow of

The pre-registered theory (0.2) said the search would collapse onto the three statements for a
dimensional reason. It did, and the collapse has a shape that the failures name precisely:

- Functionals that let `J` run free - `F_J` for large `J`, the excursion `X` - measure
  **equidistribution**, not the record. They fail because `mu(M+q') = mu(M) q'/(q'-2)`: the mean
  gap grows by a fixed factor per rung, so anything that sums `J` gaps grows like `J` times that
  factor and outruns `q'` once `J` is large. (b) and (e-i) are one failure seen twice.
- Functionals that are scale-free - `W_r/N`, `Z_r/N`, `S`, `Var(order)`, `E_x/N` - are bounded, so
  they cannot bound something unbounded, whatever their monotonicity. (c) and (e-iii) are one
  failure seen twice.
- Functionals that are column-valued and capped at `J_max` - `Q*_2`, `Phi_letter`, `B_k` - are the
  only ones left, and they *are* the three statements: `J = 1` the budget, `J = 2` the pair
  statement, `J >= 3` the chain statement.

So **the failures are the shadow of the mean gap.** Every rung multiplies the mean gap by
`q'/(q'-2)` and adds at most `q'` to the record; a functional survives only if its dependence on
the machine is confined to a bounded number of consecutive gaps. That is precisely why the one
survivor is a bounded-order functional, and why its order is `L + 1`: a rung welds at most
`J_max = L + 2` consecutive gaps into one, and the order at which that weld can be read off is one
less than the weld's own length.

## 7. The gate items

### 7.1 Gate item 1: `L(M)` bounded - what is exact, what the ladder shows

`L = max(L_bare, L_pad)`, `J_max = L + 2`.

- **Exact and proved:** `L_bare <= 5` (docs/proofs/12, KERNEL). Measured on this ladder
  `L_bare = 1, 0, 1, 1, 1, 2, 1, 3, 3, 1` at m5..m37 - at or below 3 everywhere, with its maximum
  at m29 and m31, **not** at the top.
- **Open:** `L_pad`. Measured `0, 0, 0, 0, 0, 1, 1, 1, 2, 2` at m5..m37, and the corpus continues
  `2, 2, 3, 3` at m41, m43, m47, m53 (`objects_ledger.md` O-M1). Eight of eight corpus `L` values
  reproduced on the instrument (3.5).
- **What the ladder shows, and it is a negative with a mechanism:** nothing on this instrument
  caps `L_pad`, and the reason is structural. A PAD letter is any gap divisible by `q'`, so the
  padded alphabet is nonempty as soon as the machine realises the size `q'`, which it does from
  m19 on; the all-pad density `Z_1/N` is 2.271e-04, 7.545e-07, 9.734e-06, 4.234e-06, 2.820e-07 at
  m19..m37 and never returns to 0, and it is not even monotone (it rises at `23 -> 29` and
  `37 -> 41`). `L_pad` climbs because the padded alphabet grows with the machine.
- **What is new here:** the order law of 4.3 gives `L` a job. `L` is not one invariant among many;
  it is the order of interaction the budget needs, `k* = L + 1`, at 6 of 6 decisive rungs. So a cap
  on `L` is exactly what turns the budget into a uniform bounded-order statement (6.2), and gate
  item 1 is now a gate on a named theorem rather than an unattached question.

**Status: OPEN.** This branch adds the ladder values, the mechanism of the obstruction, and the
consequence of a cap - not a cap.

### 7.2 Gate item 2: the chain statement at `J = 3, 4` on the band `[15, 36]`, at `29 -> 31`

An exact finite check, not a search (`mo_band.py`, `results/band_29_31.json`). `D_4^#(m29)` has
**45,854 distinct rows carrying all 214,708,725 gaps of m29**, with `loss = 0` and `over0 = 0`, so
enumerating every realised 2-, 3- and 4-window of m29 and testing all 31 phases enumerates every
fusion of the rung. The gate: the fusion masses reproduce the branching identity exactly -
45,532 windows / mass **413,380,422** at `J = 2`, 3,269 / **7,999,018** at `J = 3`, 62 /
**12,992** at `J = 4`, which are `n_2, n_3, n_4` of `branching_identity.md` 4.4 digit for digit.

| `J` | `Q*_J` (all `a`) | band `[15,36]` max span | argmax `a` | witness | margin to `F(m29)+31 = 74` | band windows | band mass |
|---|---|---|---|---|---|---|---|
| 2 | 55 | 55 | 30, 35 | `(25, 30)`, `(20, 35)` | 19 | 22,191 | 20,762,744 |
| 3 | 58 | **58** | 25, 30 | `(23, 10, 25)`, `(18, 10, 30)` | **16** | 1,840 | 449,604 |
| 4 | 55 | 55 | 22 | `(2, 21, 10, 22)` | 19 | 62 | 12,992 |

> **The chain statement holds on the band at `29 -> 31`, exactly and completely: the maximum span
> of a 3- or 4-piece fusion whose largest piece lies in `[15, 36]` is 58, against the budget 74 -
> margin 16.** Pre-registered value 58 with margin 16, with the record witness `(18, 10, 30)`:
> confirmed.

Three mechanisms fall out of the enumeration, and they are the useful part:

- **The extremal chain pays the letter floor and nothing more.** `d = 2 * 6^{-1} = 21 mod 31`, so
  the letters are `0, 21, 10` and `a_L = min(21, 10) = 10`. The span-maximising `J = 3` witness has
  middle **exactly 10** at 19 of the 21 values of `a` in the band - `(14,10,15)`, `(15,10,16)`,
  `(17,10,18)`, ..., `(23,10,25)`, `(18,10,30)`, `(8,10,35)` - the two exceptions being `a = 21`,
  where the witness is `(9, 21, 19)` with middle `+d`, and `a = 31`, where it is `(4, 31, 14)` with
  a PAD middle (`31 = 0 mod 31`). The `J = 4` extremal is `(2, 21, 10, 22)`: middles `21 = +d` then
  `10 = -d`, alternating, because two equal consecutive nonzero letters are illegal. The chain buys
  its span with the flanks and pays the minimum the alphabet allows in the middle.
- **The largest old piece falls with the depth of the chain.** Over all `a`, the largest piece that
  supports a fusion at all is 43 at `J = 2` (the record of m29), **35** at `J = 3`, **22** at
  `J = 4`; and the smallest is 2, 10, 21. So a 4-chain cannot contain a piece above 22 - deep
  chains are made of medium pieces, which is the band statement's own content, measured.
- **The `a`-curve peaks in the interior**, as pre-registered. At `J = 3` the maximum span by
  largest piece runs 29, 31, 34, 36, 37, 39, 41, 42, 45, 47, 48, 49, 50, 55, 52, **58**, 45, 55,
  51, 49, **58**, 49, 55, 51, 50, 53 for `a = 10..35`: it climbs to 58 at `a = 25` and `a = 30`,
  both inside the band, and falls away on both sides. At `J = 4` only `a = 21` and `a = 22` occur
  at all (spans 52 and 55).

**Status: the finite check the gate asked for is DONE and PASSES at `29 -> 31`, `J = 3, 4`, band
`[15, 36]`, margin 16.** It is one rung and one band; the gate item as a general statement is
untouched by it.

## 8. What is new

1. **The order law** (4.3): the order of interaction the budget needs is exactly `L(M) + 1`.
   `B_{L+1} <= F + q'` at 9 of 9 computable rungs (margins 3..16); `B_L > F + q'` at 7 of 7 rungs
   from `13 -> 17` up (by 5, 12, 8, 34, 11, 23, 32). Zero exceptions on both halves. The binding
   term is always the deepest fusion `J = J_max`, at 7 of 7 rungs.
2. **A bounded-order functional that bounds the record** (4.1, 5.9, 6): `B_k(M;q') >= F(M+q')` for
   every `k`, computed from the `k`-window table alone - 45,854 rows for m29's 214,708,725 gaps,
   115,193 for m31's 6,226,553,025. The budget at those rungs is a statement about a table of
   `10^5` rows. Two independent ladders (`K_0 = 15` and `K_0 = 16`, m29 at depth 10 and 11, m31 at
   depth 6 and 7) give the same dictionaries and the same table digit for digit (4.5).
3. **`Phi = B_{L+1}` is budget-monotone at 8 of 8 rungs and bounds the next record at 9 of 9**, the
   only candidate on the list with both properties, and strictly stronger than the budget at 4 of
   the 6 decisive rungs.
4. **The lemma the branch hands the tree** (6.3(ii)), with its smallest instance written out and
   its tightest measured case at `23 -> 29`: the level-2 word `(25, 10, 25)` at phase 4, span 60
   against the budget 63, a word m23 does not realise (its true `Q*_3` is 43).
5. **The second-largest realised value is budget-monotone at 9 of 9 rungs** (5.3) - a measured law
   the budget does not imply.
6. **The `F_J` failure has an exact merge, and at `19 -> 23` an exact identity**: the widest 6-run
   of m23 pulls back to ten gaps of m19 with four deletions inside,
   `12 2 5 10 8 5 3 4 3 25 -> 12 7 23 3 4 28`, span 77; in general
   `F_J(M+q') >= F_{J+t}(M)` for a merge with `t` deletions, and at this rung it is an equality,
   `F_J(m23) = F_{J+t}(m19)` for `J = 3..8` with `t = 3, 3, 4, 4, 4, 4` (5.2).
7. **The tail-excess identity** (5.4): `(a+b-x)_+ >= (a-x)_+ + (b-x)_+` closes the whole family at
   every rung and every threshold, by an identity rather than a rung.
8. **The maximum excursion is not a record statistic** (5.6): the run realising `X` is 245,506 gaps
   long at m19 and `X/F = 7.4` at m23.
9. **The band mechanisms** (7.2): extremal chains pay exactly the letter floor in the middle and
   alternate `+-d`; the largest old piece falls 43, 35, 22 with the chain depth `J = 2, 3, 4`.
10. **New corpus rows**: `F_9(m29) = 99`, `F_10(m29) = 110` (2.2); and `F(41) = 91` recomputed from
    the closure alone with the span-threshold prune, `over0 = 0` at all four rungs
    (`results/theta_K21_t89.json`).

## 9. Verdict

**Node 4.i.b.ii: the search for a monotone functional of the merge closure collapses onto the
three statements, as pre-registered - and it collapses at a definite place, which is the finding.**

There is one useful functional, `Phi = B_{L+1}`: it bounds the next record at every rung, it is
budget-monotone at every rung, and its state is a finite table of `k`-windows rather than the
machine. It is not a route on its own, because its two remaining costs are exactly the tree's own
open items: a cap on `L` (gate item 1) and one finite-order lemma about the deepest fusion, whose
smallest instance is a statement about two overlapping realised 2-windows. Prediction M7 stands:
no functional tested here is both useful and proved, and every column-valued candidate's increment
is bounded by `q'` only through the budget, the pair statement or the chain statement.

The value delivered is the order law: `k* = L + 1`, 6 of 6, no exception - which converts gate
item 1 from an unattached question into the hypothesis of a named theorem, and reduces the budget
at a rung from a statement about `10^9` gaps to a statement about `10^5` rows.

Both gate items were carried as exact finite checks. Gate item 2 at `29 -> 31`, `J = 3, 4`, band
`[15, 36]`: **passes with margin 16**, complete enumeration, `n_J` reproduced digit for digit.
Gate item 1: **stays open**, with the ladder values, the mechanism of the obstruction, and its new
consequence recorded.

## 10. Dead ends (bricks of the wall), each with its refuting rung and merge

| candidate | dies at | the merge that does it | why it cannot be revived |
|---|---|---|---|
| `F_J` uniform in `J` | `19 -> 23`, `J = 6` | 10 gaps of m19 `12 2 5 10 8 5 3 4 3 25` (span 77) lose 4 openings and become 6 gaps of m23 | `F_J(M+q') >= F_{J+t}(M)` for a `t`-deletion merge, and `mu` grows by `q'/(q'-2)` per rung: failure is forced for all large `J` at every rung |
| top-`J` value sum | `13 -> 17` (`J = 3`) | the rung that moves m13's top three `8, 10, 11` to m17's `15, 16, 18` at once | it adds `J` independent record-sized values; one rung moves all of them |
| excess over threshold `E_x`, any `x`, any density | every rung | every merge | `(a+b-x)_+ >= (a-x)_+ + (b-x)_+`: fusion moves tail mass up by an identity, and `N` falls; the density must rise |
| merge depth `J_max` | `19->23` to `23->29` (4 -> 3), and `31->37` to `37->41` (5 -> 4) | the rungs at which the new gear's legal alphabet shortens while `F` rises by 9 and by 3 | it is a small integer against an unbounded record |
| maximum excursion `X` | `13 -> 17` (+33.06 against 17) | not one merge: the `2/q'` density loss accumulated over a 245,506-gap run | free `J` measures equidistribution, not the record |
| letter-floor discount `Phi_letter` | not by a rung - by its argmax | the maximum sits at `J = 2` at 8 of 10 rungs and ties there at a ninth | it is the pair statement, and it fails to bound `F(M+q')` at 7 of 10 rungs anyway |
| `W_1/N`, `S` | five rungs: `7->11`, `11->13`, `13->17`, `23->29`, `37->41` | the rungs at which new legal words appear (the alphabet gains a letter the machine can realise twice) | bounded in `[0,1]`; cannot bound an unbounded record even when monotone |
| `Z_1/N` | `23 -> 29` (x13), `37 -> 41` (x7); nonzero from `17 -> 19` | the rungs at which the machine first realises multiples of the new gear in adjacent gaps | same |
| `Var(order)` | not refuted - monotone at 9 of 9 | falls because `q'` grows: `Var = 2/q' + O(q'^-2) + 2S/(q'-2)` | dimensionless and tending to 0; monotone for the gear's reason, not the machine's |
| the spectrum as the operator's state | `13 -> 17` | `B_1`, the best bound derivable from the realised sizes alone, is 33 against a budget of 28, and 179 against 74 by `29 -> 31` (factor 2.4) | the new sizes are sums over *consecutive* runs; the window multiset is the smallest state `T` has |
| `B_k` at fixed `k`, as a monotone functional | `B_1` at `19 -> 23` (+47), `B_2` at `19 -> 23` (+26), `B_3` at `29 -> 31` (+42) | the rung whose deepest fusion is deeper than `k` | at `k >= J_max` it equals `F(M+q')` and its monotonicity IS the budget; below `L+1` it is not monotone |
