# Node 4.i.a.i - THE AVAILABILITY GATE: which old sizes can be the largest piece of a deep fusion

Parent: node **4.i.a, the frontier's collapse at the top** (`research/proof/frontier_collapse.md`,
FACT, 2026-09-06), whose section 3.4 named this child in one sentence:

> the availability gate is monotone even though `Rest` is not. [...] once `has(a) = 0` the frontier
> is `Rest_2` and hence the pair statement. **The obligation therefore reduces to: for which `a` is
> `has(a) > 0`?** -- a statement about the old machine alone, of the same shape as `N(v) <= F_2` but
> about the RESIDUE of a neighbour rather than its size.

What spawned it exactly: the gate is *closed at the top at every rung to 31* (`has(F_old) = 0` at 7
of 8), it is a property of `M` alone, and it is monotone in the region where `Rest` is a saw. If the
gate can be shown closed above some `a_0(M)`, then above `a_0` the frontier is the **pair statement**
(node 1), whose slack at the top is the identity `s(F_old) = q' - Rest(F_old)`, and the budget
inequality reduces to the band below `a_0` plus the deep-chain caps that reach up from `a = 1`.

Scripts in `research/anchor235/r58/`; result outputs in `research/anchor235/r58/results/`
(untracked). Every number this document relies on is written into the document.

---

## 0. Pre-registered (written before any computation of this branch)

### 0.1 The objects, defined exactly

Machines `M_0 = {5}`, ..., `M_8 = {5..31}`; a **rung** is a pair `(M, q')` with `M = M_{n-1}` and
`q' = ` the incoming gear; `F_old = F(M)`, `F_2 = F_2(M)` the largest sum of two adjacent gaps of
`M`, `N_old` the number of gaps per period of `M`. `u = 6^{-1} mod q'`, `d = 2u`, letters
`a_L = min(2u mod q', q' - 2u mod q')` and `b_L = q' - a_L` (file 05 T1).

The **legal set** of the incoming gear is

    L_{q'} := { v >= a_L : v = 0, +d or -d (mod q') } = { v >= a_L : v mod q' in {0, a_L, b_L} },

the sizes an old gap may have if it is to be an INTERIOR piece of a fusion (file 05 T2). Call a size
`v` **legal** if `v in L_{q'}`.

For an old size `a`, over one full period of `M`, counting **occurrences** (gaps, not (gap, side)
pairs) and taking neighbours cyclically:

- `hasL(a)` = number of occurrences of `a` whose LEFT neighbour is legal;
- `hasR(a)` = number whose RIGHT neighbour is legal;
- `has(a)` = number with at least one legal neighbour (this is `frontier_collapse.md`'s `has`);
- `has2(a)` = number with BOTH neighbours legal -- what a 4-piece fusion `(p, v, a, w)` ... needs on
  both sides of `a`, and what a `J >= 4` word with `a` interior needs;
- `hasM(a)` = number with at least one legal neighbour `v` **of size `v <= a`** (so that `a` is still
  the largest piece: this is the gate that `Rest(a)` actually consumes).

The **gate** at `a` is OPEN if `a` is itself legal (then a `J = 3` word may carry `a` in the middle)
or `hasM(a) > 0`; CLOSED otherwise. Define

    a_gate(M, q')  := max { a realised : gate open at a },
    a_has(M, q')   := max { a realised : has(a) > 0 },
    a_gate2(M, q') := max { a realised : has2(a) > 0 }.

The **level-2 dictionary** of `M` is the set `Dict_2(M) := { (a, v) : some gap of size a has its
right neighbour of size v }`, with multiplicities `D[a][v]`; `F_2(M) = max { a + v : (a,v) in
Dict_2 }`. The gate is the question "for which `a` does `Dict_2` contain `(a, v)` with `v in L_{q'}`
and `v <= a`?".

`Rest(a)`, the budget line `B(a) = F_old + q' - a` and the slack `s(a) = B(a) - Rest(a)` are as in
`frontier_collapse.md` 0.1 and are taken from that branch's exact outputs (`fc_frontier.json`,
`fc_top31.json`) after re-gating.

**The caps that reach up from `a = 1`** (both proved):

- **the letter-floor cap.** If `a < a_L` then every interior piece of a fusion with largest piece `a`
  would have to be both `>= a_L` (file 05 T2) and `<= a` -- impossible -- so `J <= 2` and
  `Rest(a) <= a`, giving `a + Rest(a) <= 2a`.
- **the deep-chain cap.** Every piece is `<= a`, so `a + Rest(a) <= J_max a` with `J_max = 1 + D_{q'}`
  (merge_forest 3.1, proved) and `J_max <= 6` forever (the bare-word cap).

Write `a_low` for the largest `a` these two cover, i.e. the largest `a` with
`min(2a [if a < a_L], J_max a) <= F_old + q'`. The **residual band** is `[a_low + 1, a_gate]`.

### 0.2 The theory

**T. The gate is a statement about the level-2 dictionary of `M` and about it alone, it closes at
`a` well below `F_old`, and its closed form is the collision of two proved facts: a legal neighbour
must be at least `a_L`, and two adjacent gaps sum to at most `F_2(M)`. Hence `a_gate <= F_2 - a_L`,
and the budget inequality splits into the pair statement above `a_gate` and a band below it.**

The mechanism, stated before measuring: an old gap of size `a` can be the largest piece of a
`J >= 3` fusion only if the fusion has an interior piece, and an interior piece adjacent to `a` is a
neighbour of `a` of legal size. `a` and that neighbour `v` are two ADJACENT gaps of `M`, so
`a + v <= F_2(M)` by the definition of `F_2`; and `v >= a_L` by T2. So the gate cannot be open above
`F_2 - a_L`. Whether it closes EARLIER is a which-residues question about `Dict_2`.

### 0.3 Predictions, each with the number that would refute it

- **G1 (`a_gate` and where it sits).** From `frontier_collapse.md` 3.4 (`has(a) = 0` above 35 at
  m31, above 25 at m29, above 20 at m23) I pre-register `a_has = 20, 25, 35` at rungs 19->23,
  23->29, 29->31, i.e. `a_has/F_old = 0.800, 0.735, 0.814`, and `a_gate/F_old in [0.60, 1.00]` at
  all 8 rungs. *(The brief's guess "about 25/58 = 0.43" conflates the gate with the uncovered band
  `[15, 25]` of frontier_collapse 3.4; the band's TOP is the point above which the measured bound
  `Rest <= q'` holds, not the point above which the gate is closed. I predict the gate closes much
  higher, at 0.73-0.81 of `F_old`.)* REFUTED by any of the three values differing, or by a ratio
  outside `[0.60, 1.00]` at any rung.
- **G2 (holes: the gate-open set is NOT an initial segment).** Predict realised sizes below `a_gate`
  at which the gate is closed, exactly `{16, 17}` at rung 19->23 and `{17}` at rung 23->29 and none
  at 29->31 (read off frontier_collapse 2.5's list of "not legal and `has = 0`" sizes). REFUTED by a
  different hole set at any of the three, and REFUTED-as-monotone if there are no holes at all.
- **G3 (the closed form).** `a_gate <= F_2(M) - a_L` at 8 of 8 rungs (this is a proof, so a
  violation is an instrument failure); predicted values `F_2 - a_L = 23, 29, 45` at the three top
  rungs against `a_gate = 20, 25, 35`, so deficits `3, 4, 10`; and the cap is VACUOUS at rung
  29->31 (`45 > 43 = F_old`). REFUTED as a useful closed form if the deficit is 0 at 3 or more
  rungs (then it would be the gate exactly); REFUTED as an instrument if it is ever violated.
- **G4 (`has2` and the deep words).** `a_gate2 <= a_gate`, strictly at 5 or more of 8 rungs; and the
  two deepest attaining words on record -- `4 8 15 7` at rung 19->23 (`a = 15`, `J = 4`) and
  `7 10 21 10 7` at rung 29->31 (`a = 21`, `J = 5`) -- have `has2(a) > 0` at their `a`. REFUTED by
  `has2(15) = 0` at rung 23 or `has2(21) = 0` at rung 31.
- **G5 (the band).** With `a_low` from the two proved caps at the MEASURED `J_max` (3, 3, 5 at the
  three top rungs, merge_forest 2.2) the residual band is predicted `[13, 20]`, `[22, 25]`,
  `[15, 35]` at rungs 19->23, 23->29, 29->31, widths `8, 4, 21`; with the universal bare-word cap
  `J_max = 6` it is `[9, 20]`, `[11, 25]`, `[13, 35]`, widths `12, 15, 23`, which GROW. The band is
  empty at rungs 5->7 .. 13->17. The budget's minimiser `a*` lies INSIDE the band at all three top
  rungs, so the minimum slack on the band equals the global budget slack `14, 20, 16`. REFUTED by an
  `a*` outside its band, or by a band that is empty at a top rung, or by widths that shrink under
  the universal cap.
- **G6 (the family).** On 20 counterfactual members plus the real machine at each of 13->17, 17->19,
  19->23: predict at least 15 of 20 members at some rung have `a_gate = F_old` (their gate never
  closes), against the real machine's `a_gate < F_old` at 19->23. And the five recorded budget
  violators (frontier_collapse 2.7) break at an `a` with the gate OPEN: 5 of 5 (they all break at an
  interior-legal `a`, so this is close to forced -- it is a gate check, not a discovery). REFUTED by
  one violator whose `a` has the gate closed.
- **G7 (the real machine's band is narrower).** The real machine's `a_gate/F_old` is at or below the
  family median at 2 or more of the three rungs. REFUTED if it is above the median at 2 or more.
- **G8 (which legal `v` does the work at the top).** At `a = a_gate` the only legal neighbour that
  occurs is the short letter `a_L` itself, at all three top rungs. REFUTED if a legal `v > a_L`
  occurs adjacent to `a_gate` at 2 or more of the three.

**Stop rules.** Any sub-question that reduces to the merge law or chain law (docs/proofs/05), the
attainment identity (08), the peel bound / triple inequality / middle-sum lemma (16), the neighbour
law `N(v) <= F_2` (2g.i), the glue/shadow/move lemmas (2g.i.a), the fusion-rate identity or the top
law (4.i.a) is stopped in one line and cited. The level-2 dictionary as an object is the LP lane's
(`research/window_dict.py`, docs/novel/README.md round 25-27: "prescribing open positions decides
adjacent-gap-pair realisability [...] the level-2 gap dictionary the chain and the merge law
consume"); it is rebuilt here by exact full-period scan and cited, not re-derived.

### 0.4 Scorecard

| # | prediction | verdict | evidence |
|---|---|---|---|
| G1 | `a_has = 20, 25, 35` at the three top rungs; ratio in [0.60, 1.00] at all 8 | CONFIRMED on the three values, exactly (20, 25, 35); REFUTED on the range -- rung 7->11 has `a_gate = 0`, the gate closed at every `a` | 2.1 |
| G2 | holes `{16,17}`, `{17}`, none | CONFIRMED under the parent's convention (holes among the not-legal sizes): `{16,17}`, `{17}`, none. The raw `has = 0` sets add the legal sizes 19 (rung 29) and 31 (rung 31) | 2.1, 2.2 |
| G3 | `a_gate <= F_2 - a_L`, 8 of 8; deficits 3, 4, 10; vacuous at 31 | CONFIRMED with a correction: the cap bounds `a_hasM`, not `a_gate` (the "`a` is itself legal" branch is not capped; rung 13->17 has `a_gate = 11 > 10 = F_2 - a_L`). `a_hasM <= F_2 - a_L` at 8 of 8 real rungs and 60 of 60 family members; deficits 3, 4, 10 at the top three exactly as pre-registered; vacuous at rung 29->31 | 2.3, 3.2 |
| G4 | `a_gate2 <= a_gate`, strict at 5+; `has2 > 0` at the two deep words' `a` | HALF REFUTED: `has2(15) = 0` at rung 19->23 -- the `J = 4` word `4 8 15 7` needs `a` legal plus ONE legal neighbour, not two; CONFIRMED at rung 29->31, `has2(21) = 4`, and 4 is exactly the number of order-5 gaps of m31. `a_gate2 <= a_gate` 8 of 8, strict at 7 of 8 | 2.4 |
| G5 | band `[13,20]`, `[22,25]`, `[15,35]`; `a*` inside; widths grow under the universal cap | CONFIRMED in every clause: bands exactly as pre-registered, `[9,20]`/`[11,25]`/`[13,35]` under the bare-word cap (widths 12, 15, 23, growing), `a*` inside at 3 of 3, min slack on the band = the global budget slack 14, 20, 16 at 3 of 3 | 2.5 |
| G6 | 15+/20 family members with `a_gate = F_old`; 5 of 5 violators inside the band | REFUTED on the family clause (11, 10, 6 of 20, never 15); CONFIRMED on the violators, 5 of 5, every one at an `a` the chain law lets be a middle | 2.6 |
| G7 | the real machine at or below the family median at 2+ of 3 rungs | CONFIRMED, 3 of 3 (1.000 vs 1.000, 0.722 vs 1.000, 0.920 vs 0.933) | 2.6 |
| G8 | only `a_L` occurs as a legal neighbour at `a_gate`, 3 of 3 | CONFIRMED at 6 of 6 rungs with `a_hasM > 0`, and SHARPENED to an identity: `a_hasM = max { a : (a, a_L) in Dict_2(M) }`, 7 of 8 | 2.2, 3.1 |

---

## 1. Setup (exact ranges)

Everything exact: full periods, integer arithmetic, no sampling.

| object | range | script |
|---|---|---|
| `m(a)`, `hasL`, `hasR`, `has`, `has2`, `hasM`, the largest/smallest legal neighbour, and the whole level-2 dictionary `D[a][v]` | old machines `{5}`, `{5,7}`, ..., `{5..23}` on full periods (7,952,175 gaps at m23) | `ag_gate.py` |
| the same for the old machine `{5..29}` | the whole m29 period, 214,708,725 gaps, built exactly as 29 copies of the m23 period with gear 29's two teeth removed | `ag_gate.py` (`build_m29_gaps`) |
| the per-`J` gate ladder verified against the exact frontier, the three regimes, the residual band, the slack profile on it | all 8 rungs, all 137 realised `(rung, a)` cells | `ag_band.py`, consuming r57's `fc_frontier.json` and `fc_top31.json` |
| the family: gate quantities on 20 members plus the real machine | 13->17, 17->19, 19->23, full periods each | `ag_family.py` |
| the five recorded budget violators | full periods | `ag_family.py` |

**Instrument gates, all passed.** The m29 gap array returns 214,708,725 gaps summing to
1,078,282,205 with `F = 43` (merge_forest 2.1). Ten `has(a)` cells reproduce
`frontier_collapse.md` 2.4-2.5 exactly, with their multiplicities: rung 19->23
`m(15), has(15) = 1236, 62`, `has(16) = 0`, `has(25) = 0`; rung 23->29 `m(22), has(22) = 2314, 44`,
`m(23), has(23) = 5598, 32`, `has(34) = 0`; rung 29->31 `m(25), has(25) = 88548, 1858`,
`m(28), has(28) = 24418, 230`, `m(30), has(30) = 10862, 92`, `has(43) = 0`. `F_2(M)` read off the
dictionary returns `4, 7, 11, 16, 25, 31, 39, 55`, the recorded ladder (`neighbour_profile.md` 1).

## 2. Results

### 2.1 The gate at every rung (item 1)

`hasL(a) = hasR(a)` at **137 of 137 cells** -- the machine's opening set is mirror-symmetric
(`k -> P - k`, teeth at `+-u`), so the dictionary is symmetric, `D[a][v] = D[v][a]`, and left and
right carry the same count. Counting sides separately therefore adds nothing, and
`has(a) = 2 hasL(a) - has2(a)` identically (137 of 137). Everything below is stated for `has`,
`has2` and `hasM`.

| rung `q'` | `F_old` | `F_2(M)` | `a_L` | `b_L` | `a_gate` | `a_gate/F_old` | `a_hasM` | `a_hasM/F_old` | `a_has` | `a_gate2` | `F_2 - a_L` | cap deficit `F_2 - a_L - a_hasM` | max order (merge_forest) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 7 | 2 | 4 | 2 | 5 | 2 | 1.000 | 2 | 1.000 | 2 | 1 | 2 | 0 | 3 |
| 11 | 5 | 7 | 4 | 7 | **0** | 0.000 | 0 | 0.000 | 0 | 0 | 3 | 3 | **2** |
| 13 | 7 | 11 | 4 | 9 | 4 | 0.571 | 0 | 0.000 | 3 | 0 | 7 | 7 | 3 |
| 17 | 11 | 16 | 6 | 11 | 11 | 1.000 | 7 | 0.636 | 7 | 0 | 10 | 3 | 3 |
| 19 | 18 | 25 | 6 | 13 | 13 | 0.722 | 12 | 0.667 | 12 | 0 | 19 | 7 | 3 |
| 23 | 25 | 31 | 8 | 15 | 23 | 0.920 | 20 | 0.800 | 20 | 7 | 23 | 3 | 4 |
| 29 | 34 | 39 | 10 | 19 | 29 | 0.853 | 25 | 0.735 | 25 | 13 | 29 | 4 | 3 |
| 31 | 43 | 55 | 10 | 21 | 35 | 0.814 | 35 | 0.814 | 35 | 23 | 45 | 10 | 5 |

`a_has = 20, 25, 35` at the three top rungs, exactly the values `frontier_collapse.md` 3.4
recorded in passing; the ratios are `0.800, 0.735, 0.814`, not the brief's guessed 0.43 (which was
the top of the *uncovered band*, a different object). **The gate does not close low; it closes at
about four fifths of `F_old`, and the ratio `a_hasM/F_old` DRIFTS UP over the top rungs
(0.636, 0.667, 0.800, 0.735, 0.814).** That is the deciding number of the branch: the gate is not
tightening as the machine grows.

Two structural facts fall out of the table.

- **The gate is closed at every `a < a_L`**, at every rung, by definition and not by measurement:
  an interior piece must be `>= a_L` and `<= a`. So the gate is closed at BOTH ends of the
  spectrum, and the letter-floor cap of 0.1 is the gate's own low end rather than a separate tool.
- **`a_gate = 0` iff the rung's merge forest has maximum order 2.** Rung 7->11 is the only rung
  with `a_gate = 0` and the only rung whose maximum order is 2 (merge_forest 2.2:
  `3, 2, 3, 3, 3, 4, 3, 5`). 8 of 8. The gate, computed from the OLD machine alone, reproduces the
  new machine's order ceiling without touching the new machine.

### 2.2 The gate has two branches, and only one of them is capped

A `J >= 3` fusion whose largest piece is `a` needs an interior piece, and that interior piece is
either `a` itself or a neighbour of `a` inside the run. So the gate is a disjunction:

    gate open at a   iff   a is legal (a can be a MIDDLE)   or   hasM(a) > 0 (a can be an END).

The two branches behave completely differently. The END branch is capped by the dictionary
(`a_hasM <= F_2 - a_L`, section 3.2); the MIDDLE branch is not capped at all -- at rungs 5->7 and
13->17 the old record is itself a letter and the gate is open at `a = F_old`. That is the same
arithmetic accident the parent branch's TOP LAW turns on, seen from the other side.

Holes: the gate-open set is **not** an initial segment. Realised sizes in `[a_L, a_hasM]` with
`hasM = 0`:

| rung | 7 | 11 | 13 | 17 | 19 | 23 | 29 | 31 |
|---|---|---|---|---|---|---|---|---|
| holes | none | none | none | `{6}` | `{6, 8, 11}` | `{11, 16, 17}` | `{17, 19}` | `{31}` |

so 5 of 8 rungs have interior holes. Restricted to the not-legal sizes (the parent branch's
convention) the top-rung hole sets are `{16, 17}`, `{17}` and none -- G2 exactly. The holes matter:
`a = 16, 17` at rung 19->23 and `a = 17` at 23->29 sit *inside* the residual band and are
discharged by the pair statement even though the band brackets them.

**Which legal `v` does the work.** At `a = a_hasM` the largest AND the smallest legal neighbour
that occurs is the short letter `a_L` itself, at 6 of 6 rungs with `a_hasM > 0` (values
`2, 6, 6, 8, 10, 10`). Lower down the long letter also appears -- at rung 29->31 the sizes
`a = 22, 25, 27` have `b_L = 21` as a legal neighbour -- but the top of the gate is decided by
`a_L` alone.

### 2.3 The gate reduced to one row of the dictionary (item 2)

    a_hasM(M, q')  =  max { a : (a, a_L) is an adjacent pair of M },   7 of 8 rungs.

| rung `q'` | 7 | 11 | 13 | 17 | 19 | 23 | 29 | 31 |
|---|---|---|---|---|---|---|---|---|
| `a_L` | 2 | 4 | 4 | 6 | 6 | 8 | 10 | 10 |
| max `a` adjacent to `a_L` | 2 | 0 | 3 | 7 | 12 | 20 | 25 | 35 |
| `a_hasM` | 2 | 0 | 0 | 7 | 12 | 20 | 25 | 35 |

The one mismatch is rung 11->13, where the largest neighbour of a 4-gap is 3, below `a_L = 4`, so
no legal neighbour survives the `v <= a` test at all and `a_hasM = 0`. Everywhere else the two
agree exactly. So the whole availability question is **one row of the level-2 dictionary**: does
`M` ever put a gap of the short letter's size next to a gap of size `a`?

That is a much smaller object than `Rest`. `Dict_2(M)` is the LP lane's level-2 gap dictionary
(docs/novel/README.md rounds 25-27, `research/window_dict.py`: "prescribing open positions decides
adjacent-gap-pair realisability"); the object is cited, and what is new here is that the chain
obligation's availability is a single ROW of it, indexed by the incoming gear's short letter.

### 2.4 The per-`J` gate ladder (new, proved, 0 exceptions in 137 cells)

Sharper than the parent's single gate, and it separates the orders:

> **THE GATE LADDER.** For a fusion of order `J` whose largest piece is `a`:
> `J = 2` -- no condition;
> `J = 3` -- `a` is legal (`a` in the middle) OR `hasM(a) > 0` (`a` at an end);
> `J >= 4` -- `hasM(a) > 0`, always.
>
> Proof of the last line: in a `J >= 4` run the interior positions are `2 .. J-1`, at least two
> consecutive ones, so whichever position `a` occupies, one of its two run-neighbours is an
> interior piece and hence legal, and it is `<= a` because `a` is the largest. Hence above
> `a_hasM` only `J = 3` with `a` legal survives, and above `a_gate` only `J = 2`.

Verified against branch 4.i.a's exact frontier at all 137 realised `(rung, a)` cells: **0
violations of "gate closed implies no `J >= 3`" and 0 violations of "`hasM(a) = 0` implies no
`J >= 4`"**. (The parent branch checked the first statement at 40 cells; this is the full range,
and the second statement is new.)

`has2` -- both neighbours legal -- is NOT what a 4-piece fusion needs, which refutes half of G4:
`has2(15) = 0` at rung 19->23 while the record's own word there is the `J = 4` run `4 8 15 7`, in
which `a = 15` is legal and has the legal neighbour 8 on one side and the FLANK 7 on the other.
`has2` is what a run needs when `a` sits strictly inside the interior with interior pieces on both
sides, i.e. from `J = 5` with `a` at the centre. There it is exact:

> `has2(21) = 4` at rung 29->31, and the number of order-5 gaps of m31 is **4**
> (merge_forest 2.2). The four occurrences are the four palindromes `(7, 10, 21, 10, 7)`.

### 2.5 The three regimes and the residual band (item 3)

The gate ladder cuts the frontier into three regimes, and each has a different obligation:

| regime | range of `a` | orders available | the obligation there |
|---|---|---|---|
| top | `a > a_gate` | `J <= 2` | `a + Rest_2(a) <= F_2(M)`, i.e. the **PAIR statement** `F_2 <= F_old + q'` (node 1) |
| middle | `a_hasM < a <= a_gate`, `a` legal | `J <= 3` with `a` in the middle | `a + N(a) <= F_old + q'` -- the depth-3 chain term at a legal middle |
| band | `a <= a_hasM`, gate open | up to `J_max` | the full **CHAIN statement** (node 2) |

**Top regime.** `max_a (a + Rest_2(a)) = F_2(M)` at 7 of 8 rungs (the parent's identity,
`frontier_collapse.md` 3.3), so the top regime is discharged by the pair statement with slack
`F_old + q' - F_2 = 5, 9, 9, 12, 12, 17, 24, 19` at the 8 rungs. Free through m31.

**Middle regime.** The legal sizes in `(a_hasM, a_gate]` are `4` (rung 13), `11` (17), `13` (19),
`23` (23), `29` (29), none at rung 31. There `Rest(a) = Rest_3(a)` is a flank sum and
`a + Rest_3(a) <= a + N(a)`; measured `a + N(a) = 8, 18, 25, 31, 40` against budgets
`20, 28, 37, 48, 63`: **5 of 5 under budget**, slack `12, 10, 12, 17, 23`. This is not new content
-- `a + N(a)` at legal `a` is the depth-3 chain term of `neighbour_profile.md` 2.3 -- and it is
stopped here in one line and cited. What is new is that the gate says this regime is *all* that is
left between the band and the pair statement.

**The band.** The deep-chain cap reaches `a <= (F_old + q')/J_max`; the gate closes at `a_hasM`.
What is left:

| rung | budget | `J_max` | deep cap reaches | `a_hasM` | residual band (measured `J_max`) | realised sizes in it | band under the bare-word cap 6 | min slack on the band | global budget slack | `a*` in the band |
|---|---|---|---|---|---|---|---|---|---|---|
| 7 | 9 | 3 | 3 | 2 | empty | 0 | `[2, 2]` | - | 4 | no |
| 11 | 16 | 2 | 8 | 0 | empty | 0 | empty | - | 9 | no |
| 13 | 20 | 3 | 6 | 0 | empty | 0 | empty | - | 9 | no |
| 17 | 28 | 3 | 9 | 7 | empty | 0 | `[6, 7]` | - | 10 | no |
| 19 | 37 | 3 | 12 | 12 | empty | 0 | `[7, 12]` | - | 12 | no |
| 23 | 48 | 4 | 12 | 20 | **`[13, 20]`** | 5 | `[9, 20]` | 14 | 14 | YES (`a* = 15`) |
| 29 | 63 | 3 | 21 | 25 | **`[22, 25]`** | 3 | `[11, 25]` | 20 | 20 | YES (`a* = 23`) |
| 31 | 74 | 5 | 14 | 35 | **`[15, 35]`** | 21 | `[13, 35]` | 16 | 16 | YES (`a* = 30`) |

The slack profile on the band (`a : s(a)`):

    rung 29->31: 15:35 16:33 17:32 18:29 19:27 20:26 21:19 22:19 23:19 24:22 25:16 26:23
                 27:19 28:23 29:25 30:16 31:25 32:19 33:21 34:24 35:19
    rung 23->29: 22:24 23:20 25:26
    rung 19->23: 13:22 14:21 15:14 18:17 20:15

The minimum of the slack on the band is the global budget slack at 3 of 3 top rungs -- the whole of
the budget's tightness lives in the band, none of it in the two discharged regimes.

**Does the band shrink? No.** In realised sizes it is 5, 3, 21 at the three top rungs; as an
integer interval 8, 4, 21; under the universal bare-word cap 12, 15, 23. It GROWS at the rung that
matters. And it is WIDER than the parent branch's residual `[15, 25]` at m31, because the parent
closed `a >= 26` with the MEASURED bound `Rest(a) <= q'` while the gate only closes `a >= 36`. The
honest comparison: `[15, 25]` is the smaller band but rests on a measurement; `[15, 35]` is the
larger band but everything outside it is discharged by a proof plus the pair statement.

### 2.6 The family and the violators (item 4)

20 members plus the real machine at each of 13->17, 17->19, 19->23 (the alignment-rules section 5
family, the SAME 20 members as `fc_family.py`, same seed), full periods.

| rung | real `a_gate/F_old` | family min | median | max | members below the real value | members whose gate never closes (`a_gate = F_old`) | members obeying `a_hasM <= F_2 - a_L` |
|---|---|---|---|---|---|---|---|
| 13->17 | 1.000 | 0.538 | 1.000 | 1.000 | 9 of 20 | 11 of 20 (real: YES) | 20 of 20 |
| 17->19 | 0.722 | 0.652 | 1.000 | 1.000 | 2 of 20 | 10 of 20 (real: no) | 20 of 20 |
| 19->23 | 0.920 | 0.767 | 0.933 | 1.000 | 8 of 20 | 6 of 20 (real: no) | 20 of 20 |

**The real machine's gate is at or below the family median at 3 of 3 rungs** -- a which-residues
fact, but a weak one: the median is 1.000, 1.000, 0.933 and the real value 1.000, 0.722, 0.920, so
the real machine is typical-to-slightly-narrow, not extreme. The pre-registered "15 or more of 20
members never close their gate" is REFUTED (11, 10, 6 of 20); the modal family member does close
its gate, just later than the real machine at 17->19.

**The five recorded budget violators all break inside the band**, 5 of 5:

| violator | `F_old` | `q'` | `a_L` | violating `a` | `a/F_old` | `a` legal? | `hasM(a)` | gate | `a_gate` |
|---|---|---|---|---|---|---|---|---|---|
| `{5..17}` (1,3,4,4,4), `v_19 = 4` | 19 | 19 | 8 | 19 | 1.000 | YES | 4 | OPEN | 19 |
| `{5..17}` (2,3,3,3,3), `v_19 = 3` | 18 | 19 | 6 | 13 | 0.722 | YES | 28 | OPEN | 18 |
| `{5..11}` (1,1,5), `v_13 = 1` | 11 | 13 | 2 | 11 | 1.000 | YES | 1 | OPEN | 11 |
| `{5..17}` (1,3,2,1,6), `v_19 = 3` | 15 | 19 | 6 | 13 | 0.867 | YES | 4 | OPEN | 14 |
| `{5..17}` (1,2,1,4,4), `v_19 = 5` | 14 | 19 | 9 | 10 | 0.714 | YES | 34 | OPEN | 14 |

Every one is legal AND has a legal neighbour, so both branches of the gate are open at the point of
failure. That is a check, not a discovery -- the parent branch already recorded that all five break
at an interior-legal `a` -- but it confirms that the gate never has to be trusted where it is not
open: **no counterfactual machine on file breaks the budget at an `a` the gate closes.**

## 3. Mechanism

### 3.1 The gate in the machine's terms

To make a `J >= 3` gap out of a big old gap `a`, the new gear must strike two openings that both
bound one and the same old gap -- an INTERIOR piece -- and by the chain law (file 05 (C), T2) an
interior piece has size `0` or `+-d (mod q')` and hence at least `a_L ~ q'/3`. Either that interior
piece is `a` itself, or it is a neighbour of `a`; and if it is a neighbour it must also be no larger
than `a`, or `a` is no longer the largest piece and the frontier cell is a different one. So
availability is exactly the question

> does the old machine ever put a gap of a legal size, at most `a`, next to a gap of size `a`?

and by section 2.2 the legal size that decides it at the top is always the SHORT LETTER `a_L`. So
the gate is one row of the level-2 dictionary and nothing else. The gate is closed below `a_L`
because no legal size fits under `a`, and closed above `a_hasM` because the dictionary has no
`(a, a_L)` entry there. `Rest` is a saw; the gate is an interval with holes, and an interval with
holes is what a proof can attack.

### 3.2 The closed form, and exactly when it is vacuous (item 5)

> **THE PAIR CAP ON THE GATE (proved).** `a_hasM(M, q') <= F_2(M) - a_L(q')`.
>
> Proof, one line: if some occurrence of `a` has a legal neighbour `v <= a`, then `a` and `v` are
> two adjacent gaps of `M`, so `a + v <= F_2(M)` by the definition of `F_2`; and `v >= a_L` by
> file 05 T2. Hence `a <= F_2 - a_L`.

Verified at 8 of 8 real rungs and 60 of 60 family members (68 of 68), deficits
`0, 3, 7, 3, 7, 3, 4, 10`. Two things follow, one good and one bad.

**Good.** It bounds the gate by two quantities of `M` and `q'` alone, with no reference to the new
machine, no counting, and no measurement -- exactly the shape the root asks for. It is tight at
rung 5->7 (2 = 2). The cap says nothing at all about the MIDDLE branch of the gate, which is why
rung 13->17 has `a_gate = 11 > 10 = F_2 - a_L`: that is the legal old record, not a neighbour.

**Bad, and it is the branch's deciding negative.** The cap is useful only when `F_2 - a_L < F_old`,
i.e. when

    F_2(M) - F(M)  <  a_L(q').

Measured `F_2 - F = 2, 2, 4, 5, 7, 6, 5, 12` at the 8 old machines against
`a_L = 2, 4, 4, 6, 6, 8, 10, 10`: it **fails at rung 17->19 (7 > 6) and at rung 29->31 (12 > 10)**,
and those are exactly the two rungs at which `neighbour_profile.md` 4 recorded the same inequality
failing when it tried to close the depth-3 chain from the `F_2` cap ("`F_2 - F = 4, 5, 7, 6, 5, 12,
10` against `a = 4, 6, 6, 8, 10, 10, 12`: it FAILS at m17 and at m29"). **The same inequality, at
the same two rungs, blocks two different routes.** That is a new connection between branch 2g.i and
this one, and it says the obstruction is one object -- the excess `F_2 - F` of the old machine
against the incoming gear's short letter -- and not two.

So at the one rung where the band matters most (29->31) the proved cap gives `a <= 45 > 43 = F_old`
and closes nothing; the measured `a_hasM = 35` is 10 below it.

### 3.3 What would have to be proved, and the exact residual (item 5)

The smallest theorem that reduces the budget inequality to the pair statement plus a bounded band:

> **THE DICTIONARY-ROW THEOREM (target).** For every machine `M` and incoming gear `q'` there is
> `a_0(M, q')` with `a_0 <= c F(M)`, `c < 1`, such that `M` has no adjacent pair `(a, a_L)` with
> `a > a_0`; equivalently the row `v = a_L` of `Dict_2(M)` is empty above `a_0`.

With it (and the gate ladder, proved) the budget inequality reduces to exactly three pieces:

1. `a > max(a_0, largest legal size <= F_old)`: the **PAIR statement** `F_2(M) <= F(M) + q'`
   (node 1, OPEN, free through m31 with slack 5..24).
2. `a` legal, `a > a_0`: `a + N(a) <= F(M) + q'`, the depth-3 chain term at a legal middle
   (node 2 at `J = 3`; `neighbour_profile.md` 2.3's table, under budget at 5 of 5 cells on file).
3. `a <= a_0`: the **CHAIN statement** on the band `[(F_old + q')/J_max, a_0]`.

**The residual, named exactly.** What is *proved* today is `a_0 <= F_2 - a_L`, which is vacuous
whenever `F_2 - F >= a_L` -- at 2 of 8 rungs including the top one. What is *measured* is
`a_0 = a_hasM = 20, 25, 35` at the three top rungs, `a_hasM/F_old = 0.800, 0.735, 0.814` with no
sign of falling. So the residual is:

- the dictionary-row theorem itself with any `c < 1` bounded away from 1 -- unproved, and the only
  proved cap for it is vacuous at rung 29->31;
- piece 3 above: the chain statement on `[15, 35]` at m31 (21 realised sizes, minimum slack 16,
  containing both record maximisers `a = 25` and `a = 30`).

The gate therefore **shrinks the obligation but does not shrink the hard part**: it removes the top
of the spectrum from the chain statement's territory by a proof rather than a measurement (6 sizes
at m31: `36, 37, 38, 39, 40, 43`, exactly the sizes at which the parent branch's TOP LAW lives),
and it leaves the record itself inside the band, as the parent branch predicted it would.

## 4. What is new

1. **The gate ladder**, proved and verified with 0 exceptions in 137 cells: `J = 3` needs `a` legal
   or a legal neighbour `<= a`; `J >= 4` needs a legal neighbour `<= a`, always. It separates the
   orders, which the parent branch's single gate did not, and it is what makes the three regimes
   well defined.
2. **The gate is one row of the level-2 dictionary.**
   `a_hasM = max { a : (a, a_L) is an adjacent pair of M }` at 7 of 8 rungs; the short letter is
   the only legal neighbour that occurs at the top of the gate, at 6 of 6 rungs.
3. **The pair cap on the gate**, `a_hasM <= F_2(M) - a_L(q')`, proved in one line from the
   definition of `F_2` and file 05 T2, verified at 68 of 68 (8 rungs + 60 family members), and
   **vacuous exactly when `F_2 - F >= a_L`** -- the same inequality, failing at the same two rungs
   (17->19 and 29->31), that `neighbour_profile.md` found blocking the `F_2` cap from closing the
   chain. One obstruction, two routes.
4. **The gate has two branches and only the END branch is capped.** The MIDDLE branch (`a` itself
   legal) is an arithmetic accident of `F_old mod q'` and is what lets `a_gate = F_old` at rungs
   5->7 and 13->17; it is the same accident the parent's TOP LAW turns on.
5. **`a_gate = 0` iff the rung's maximum merge order is 2**, 8 of 8 -- the gate computed from the
   OLD machine alone reproduces the new machine's order ceiling.
6. **`hasL = hasR` at 137 of 137 cells** (the mirror makes the dictionary symmetric), so
   `has = 2 hasL - has2` identically; counting sides separately adds nothing.
7. **`has2(21) = 4` at rung 29->31 equals the number of order-5 gaps of m31**, and the four
   occurrences are the four palindromes `(7, 10, 21, 10, 7)`; `has2` is the `J = 5`-centre gate,
   not the `J = 4` gate (`has2(15) = 0` at rung 19->23, where the record's own word is `4 8 15 7`).
8. **The three regimes with their separate obligations, and the band measured at every rung**:
   empty at rungs 5->7 .. 17->19, `[13, 20]`, `[22, 25]`, `[15, 35]` at the top three, containing
   the budget minimiser at 3 of 3, carrying the whole of the budget's tightness (min slack on the
   band = global budget slack, 3 of 3), and GROWING.

Prior art inside the project: `has(a)` as a quantity and the values 20, 25, 35 are the parent
branch's (`frontier_collapse.md` 2.5, 3.4); the level-2 dictionary as an object is the LP lane's
(`research/window_dict.py`, README rounds 25-27); `N(v) <= F_2` and the `a + N(a)` table are 2g.i;
the merge and chain laws and T2 are docs/proofs/05; the max-order identity is merge_forest 3.1;
the five budget violators are node 2f.i via frontier_collapse 2.7. Outside the project: not checked
(no web access).

## 5. Verdict

**FACT, exact, with one proved ladder, one proved closed form and one new cross-branch
identification; a partial route whose deciding number is negative.**

- The gate is now a completely explicit object: closed below `a_L` by the letter floor, closed above
  `a_hasM` by the dictionary, with holes in between, and `a_hasM` is one row of the level-2
  dictionary of `M`. The gate ladder is proved and exceptionless in 137 cells, and it splits the
  budget inequality into the pair statement, a five-cell depth-3 obligation at legal middles, and
  the band.
- The closed form `a_hasM <= F_2 - a_L` is proved and holds 68 of 68 -- and it is **vacuous at rung
  29->31**, the rung that matters, because `F_2(m29) - F(m29) = 12 > 10 = a_L(31)`. That is the
  same inequality that killed the `F_2`-cap route to the chain statement in branch 2g.i, at the
  same two rungs.
- The band the gate leaves does not shrink: `[13, 20]`, `[22, 25]`, `[15, 35]` at the three top
  rungs, and `a_hasM/F_old = 0.800, 0.735, 0.814` with no downward trend. So the gate as
  pre-registered -- "if the gate can be shown closed above some `a_0`, the budget reduces to the
  pair statement plus a bounded band" -- is TRUE as a reduction and EMPTY as a bound, because
  nothing on file bounds `a_0` away from `F_old` at the top rung.
- **What the branch hands forward** is a strictly smaller and more concrete statement than the one
  it was opened with: *the row `v = a_L` of the level-2 dictionary of `M` is empty above `c F(M)`*.
  It is finite, it is about `M` alone, it is about ONE gap size rather than a whole profile, and
  the LP lane already certifies level-2 dictionary cells by duality at m19 and m23 -- which is
  where a child branch should start.

## 6. Dead ends, each with its refuting instance

- **`a_gate <= F_2 - a_L`** (the cap as pre-registered, on the full gate). Refuted at rung 13->17:
  `a_gate = 11 > 10`, because 11 is itself a letter of 17 and the MIDDLE branch of the gate is not
  capped. The cap survives on `a_hasM`.
- **The gate-open set is an initial segment.** Refuted at 5 of 8 rungs; holes `{11, 16, 17}` at
  rung 19->23, `{17, 19}` at 23->29, `{31}` at 29->31.
- **`has2` is the 4-piece gate.** Refuted: `has2(15) = 0` at rung 19->23 while the record's word
  there is the `J = 4` run `4 8 15 7`. `has2` is the `J = 5`-centre gate.
- **The gate closes low (the brief's `0.43 F_old`).** Refuted: `a_hasM/F_old = 0.800, 0.735, 0.814`
  at the three top rungs. 0.43 was the top of the parent's uncovered band, a different object.
- **The gate tightens as the machine grows.** Refuted: `a_hasM/F_old` over rungs 17..31 is
  `0.636, 0.667, 0.800, 0.735, 0.814`; the band grows from 5 to 21 realised sizes.
- **The gate's residual band is smaller than the parent's.** Refuted at m31: the gate leaves
  `[15, 35]`, the parent's measured bound left `[15, 25]`. The gate's band is bigger and its
  discharge is proof-shaped; that is the trade, and it is not a reduction in size.
- **"15 or more of 20 family members never close their gate."** Refuted: 11, 10, 6 of 20.

## 7. What holds without exception (item 6)

| statement | count | status |
|---|---|---|
| the gate ladder: gate closed at `a` implies no `J >= 3` fusion with largest piece `a` | 137 (rung, `a`) cells, 8 rungs | proved (file 05 T2) + verified |
| the gate ladder: `hasM(a) = 0` implies no `J >= 4` fusion with largest piece `a` | 137 cells, 8 rungs | proved here + verified |
| `a_hasM <= F_2(M) - a_L(q')` (the pair cap on the gate) | 8 real rungs + 60 family members = 68 | proved here + verified |
| `hasL(a) = hasR(a)` | 137 of 137 cells | mirror symmetry + verified |
| `has(a) = 2 hasL(a) - has2(a)` | 137 of 137 cells | identity |
| the gate is closed at every realised `a < a_L` | 8 of 8 rungs | proved (file 05 T2 + "`a` is the largest piece") |
| `a_hasM = max { a : (a, a_L) in Dict_2(M) }` | 7 of 8 rungs (rung 11->13 has no such `a >= a_L`) | measured |
| the only legal neighbour occurring at `a = a_hasM` is `a_L` | 6 of 6 rungs with `a_hasM > 0` | measured |
| `a_gate = 0` iff the rung's maximum merge order is 2 | 8 of 8 rungs | measured (against merge_forest 2.2) |
| `a + N(a) <= F_old + q'` at every legal `a` in the middle regime | 5 of 5 cells | measured (2g.i's depth-3 term, cited) |
| the minimum slack on the residual band equals the global budget slack | 3 of 3 top rungs | identity (the minimiser is in the band) |
| the recorded budget violators break at an `a` with the gate OPEN | 5 of 5 | measured |
