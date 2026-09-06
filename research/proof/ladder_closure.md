# Node 4.i.a.i - THE LADDER PAST THE WALL: the closure step run as a recursion

Parent: node **4.i.a, the branching identity and the closure theorem**
(`research/proof/branching_identity.md`), PROVED/FACT since 2026-09-06. What spawns this branch
is one sentence of its section 6: "the ladder of spectra can in principle be run without ever
constructing a period - which is what the m31 column of 4.1 actually is". The parent ran the
closure step **once** at each rung, always from a period it had built. This branch asks whether
the step can be **iterated**: dictionary in, dictionary out, so that the certified record ladder
- which stops at `F(59) = 161` because `P_59` is astronomical - can be continued past the wall
by a machine that never touches a period after the first.

Scripts in `research/anchor235/r61/` (prefix `lc_`); result outputs in
`research/anchor235/r61/results/` (untracked). Every number this document relies on is written
into the document.

---

## 0. Pre-registered

Written before any computation of this branch.

### 0.1 The objects

`M` a machine with gears `5..y`, period `P`, `N` openings, cyclic gap sequence. `q'` the next
gear, `c = 6^{-1} mod q'`, `d = 2c`. Letters `PAD / UP / DOWN / BAD` and legality as in
`branching_identity.md` 0.1; `L(M)` the longest realised legal word, `J_max = L + 2`.

`D_K^#(M)` is the multiset of realised `K`-windows of consecutive gap **sizes** (with
multiplicity, one entry per opening of `M`). `K_m(M -> q')` is the largest number of consecutive
old gaps spanned by `m` consecutive new gaps.

**The closure step** (the parent's Theorem 5, made into a machine): given `D_K^#(M)` and `q'`,
for every distinct window `w = (g_1..g_K)` with multiplicity `mult(w)` and every phase
`z in Z_{q'}`, mark the opening at offset `o_i = g_1 + ... + g_i` struck iff
`o_i + z = 0 or d (mod q')`; if `o_0` is unstruck the pair `(w, z)` is one opening of `M + q'`,
and the first `m` new gaps at it are the successive differences of the unstruck offsets. Adding
`mult(w)` to that `m`-tuple's count over all `(w, z)` gives `D_m^#(M + q')` **exactly**, provided
every `(w, z)` reaches `m` new gaps inside the `K` old ones. The number of pairs that do not is
the **loss**; `loss = 0` certifies the step.

Two identities gate every step: `sum mult = N`, and `sum mult(new) + loss = (q' - 2) N`.

### 0.2 The theory

**T. The record ladder is a recursion on dictionaries, and its cost is depth, not period.** The
period of `M + q'` is `q'` times that of `M`; the dictionary of `M + q'` at the depth the record
law needs is a function of `M`'s dictionary at a depth bigger by a bounded amount. If the depth
bound is small the ladder runs; if it grows the instrument stops, and where it stops is a
measurable property of the machine, not of the hardware.

**T2 (the span-bounded variant, pre-registered as the fix for the depth question).** The window
set `{ windows of total span <= S }` is **exactly closed** under the step with no depth
truncation at all, because the span of `m` new gaps equals the span of the old gaps they fuse.
So a dictionary indexed by span rather than by depth has `loss = 0` by construction for every
`m` whose new span is `<= S`.

### 0.3 Predictions, each with the refuting number

- **P1 (validation).** The iterated machine reproduces, from `D^#(m23)`: `F(29) = 43`,
  `|Spec(m29)| = 41`, `m(4) = 14,178,528`, `m(6) = 10,497,320`, `m(24) = 1,180`, `m(36) = 38`,
  `sum m = 214,708,725`, `sum v m = 1,078,282,205`, and the order distribution
  `n_1..n_3 = 199,048,197 / 15,416,706 / 243,822`. From the result of that step (not from a
  period): `F(31) = 58`, `|Spec(m31)| = 55`, absent `{54, 56, 57}`, `m(4) = 398,923,200`,
  `m(6) = 299,202,120`, `m(24) = 174,704`, `m(36) = 3,152`, `m(41) = 134`,
  `sum m = 6,226,553,025`, `sum v m = 33,426,748,355`. Then `F(37) = 88`. REFUTED by one
  multiplicity.
- **P2 (the depth).** `K_m - m` is bounded by `2 (J_max - 1)` over the `m` reached, and `K_m`
  tracks `J_max` and not the rung (the parent measured `K_2 - K_1 = 1, 1, 1, 1, 1, 2, 2`).
  REFUTED by one measured `K_m - m > 2 (J_max - 1)`.
- **P3 (the push).** The exact instrument reaches at least `F(41) = 91` and `F(43) = 103` inside
  3 GB and one hour per rung, matching the corpus ladder `5, 7, 11, 18, 25, 34, 43, 58, 88, 91,
  103, 118, 145, 161` at `y = 7 .. 59`. REFUTED by a mismatch with a corpus value, which would be
  an instrument failure, not a discovery.
- **P4 (the budget slack).** At every rung the instrument reaches, `F(M + q') <= F(M) + q'`, and
  the slack `F(M) + q' - F(M + q')` is reported. A violation would be the first ever recorded and
  is pre-committed to be double-checked by an independent route (the `Q*_J` extremes read off the
  same dictionary, and a direct CRT witness) before being reported at all.
- **P5 (span-bounded exactness).** With windows cut at span `S`, `loss = 0` for every step whose
  new windows have span `<= S`. REFUTED by a positive loss.
- **P6 (dictionary growth).** `|D_K(M)|` grows geometrically in `K` with a falling ratio at fixed
  `M` (the parent measured `41, 730, 7,184, 45,854, 208,668, 720,527` at m29) and by a factor
  1.7-2.6 per rung at fixed `K`. Predicted: the product of the two is what stops the ladder, and
  the stopping rung is between m41 and m53. REFUTED by reaching m59 exactly, or by stopping
  before m37.
- **P7 (truncation).** At a depth below `K_m` the machine still gives a **rigorous lower bound**
  on `F` (every window it produces is realised) and `max_{J <= J_max} F_J(M)` is a **rigorous
  upper bound** (`Q*_J <= F_J`). Predicted: the interval at the first unreachable rung is wide
  enough to be useless for the budget question. REFUTED by an interval of width 0.
- **P8 (the negative, pre-registered so the branch cannot claim a route).** A recursion computes;
  it does not bound. Nothing here will bound an extreme of the new dictionary by the extremes of
  the old one. Predicted: the residual is node R1.2's chain statement, unchanged.

### 0.4 Scorecard

| # | prediction | verdict | evidence |
|---|---|---|---|
| P1 | validation at m29, m31, m37 | **CONFIRMED**, every gate, plus five families not asked for (`n_J`, `Q*_J`, `C_r`, m37's thirteen holes, the record witnesses) | 2 |
| P2 | `K_m - m <= 2(J_max - 1)` | **REFUTED** at `23 -> 29`: `K_9 - 9 = K_10 - 10 = K_12 - 12 = 5 > 4 = 2(J_max - 1)`. What survives: `K_m - m` grows with `m`, slowly, and still not with the rung | 4.1 |
| P3 | reaches `F(41) = 91` and `F(43) = 103` | **HALF CONFIRMED**: `F(41) = 91` exact in 33 minutes and 2.4 GB from m23's period alone, with `Q*_J(37->41) = 88, 90, 90, 91`; `F(43)` is out of budget and the reason is measured | 5 |
| P4 | budget slack `>= 0` at every new rung | **CONFIRMED** (14, 20, 16, 7, 38); no violation, and the slack is NOT monotone | 3.3 |
| P5 | span-bounded closure has zero loss | **CONFIRMED**: `loss = 0` by construction, three rungs verified against direct builds - and the form is unusable, `\|V_100(m23)\| = 6,819,348` against `N = 7,952,175` | 4.3 |
| P6 | growth stops the ladder between m41 and m53 | **CONFIRMED at the near edge**: exact through m41, stops at m43 | 5.1 |
| P7 | truncation gives a lower bound and a wide interval | **CONFIRMED**: the depth-`K` floor is `max_{J <= K} Q*_J` = 88, 90, 90, 91, so depth 3 returns `F(41) >= 90`, off by one; the interval is `[91, 180]` with the budget 129 inside it | 6 |
| P8 | no bound is claimed | **CONFIRMED** | 7.3 |

**Stop rules honoured.** The record law (docs/proofs/09), the closure theorem and the size
formula (`branching_identity.md` 2.6, 5.1), the word reduction `J_max = L + 2` (docs/proofs/10)
and the corpus ladder are cited, never re-derived. The one place where this branch's instrument
meets the lap-phase transfer (`alignment-rules.md` 3.5, the vehicle behind `F(59) = 161`) is
stated in a paragraph (7.2) and the sub-question stopped there.

---

## 1. Setup: the closure step as a machine

Exact integer arithmetic throughout; no sampling anywhere. `research/anchor235/r61/lc_core.py`.

A dictionary is a pair `(win, mult)`: `win` a `uint8` array of `n` distinct windows of `K`
consecutive gap sizes, `mult` an `int64` array of multiplicities summing to `N`. One rung is:

1. offsets `o_0 = 0, o_i = g_1 + ... + g_i` and their residues mod `q'`, computed once per
   window;
2. for each phase `z in Z_{q'}`, a length-`(q'+1)` lookup marks `o_i` struck iff
   `o_i + z in {0, d} (mod q')` (one gather, not two comparisons);
3. the surviving offsets are **compacted** in place (a running count scatters `o_i` into slot
   `c_i - 1`), and the new gaps are the successive differences of the compacted offsets - so the
   walk costs `~8K` passes per phase rather than the `~20` per emitted gap a next-free table
   costs;
4. rows are grouped by an exact sort (`ceil(K/8)` packed `uint64` keys, `lexsort`, `reduceat` on
   `int64` multiplicities - no floating accumulator anywhere in the multiplicities).

The step reports, per rung: `loss` (multiplicity-weighted pairs that ran out of window),
`over0` (pairs whose FIRST new gap could not be determined - the fatal one), `mmin` (the depth
every pair can complete: the largest `m` with `loss = 0`), `kmax` (`K_m`, the largest number of
old gaps spanned), the exact `(order, value)` mass table, and `L`, `L_bare`, `L_pad`, `W_m`,
`Z_m` for the next gear.

The ladder (`lc_fixed.py`) makes two passes per rung: a measuring pass with no grouping, which
returns `mmin` and the whole `(order, value)` table, and a building pass at `m = mmin`, which is
the dictionary the next rung stands on. Because `m = mmin`, the completeness test
`cnt - 1 >= m` holds for every surviving pair, so **`loss = 0` on the unpruned ladder and, on the
pruned ladder of 4.4, `loss` is exactly the deliberate prune and nothing else. `over0 = 0` at
every rung of both.** Every number reported below is therefore exact, not a bound.

Ranges: base machine m23 by direct sieve (`P = 223,092,870`, `N = 7,952,175` gaps), depth `K_0`;
rungs 29, 31, 37, 41. Four cores; resident peak 2.4 GB at the heaviest setting
(`K_0 = 17` unpruned and `K_0 = 21` pruned), well inside the 3 GB lane budget, and the branch
stopped rather than exceed it.

---

## 2. Validation: the known rungs, from m23's period and nothing else

`lc_fixed.py 14 47 23` - base `|D_14(m23)| = 3,877,610`, three rungs, **195 s in total**.

| rung | `K_in` | depth out | `F` | `F_1..F_8` | `\|D_out\|` | `N` | loss | over0 | secs |
|---|---|---|---|---|---|---|---|---|---|
| m29 | 14 | 9 | **43** | 43, 55, 65, 70, 85, 90, **92**, **97** | 8,818,629 | 214,708,725 | 0 | 0 | 86.5 |
| m31 | 9 | 5 | **58** | 58, 68, 85, 90, 92 | 636,575 | 6,226,553,025 | 0 | 0 | 104.5 |
| m37 | 5 | 1 | **88** | 88 | 75 | 217,929,355,875 | 0 | 0 | 4.5 |

Deeper bases push the same three rungs further down the `F_j` ladder:

| base depth | m29 | m31 | m37 |
|---|---|---|---|
| `K_0 = 15` | depth 10, `F_j = 43,55,65,70,85,90,92,97` | depth 6, `F_j = 58,68,85,90,92,97` | depth 2, `F_j = 88, 90` |
| `K_0 = 17` | depth 12, same | depth 8, `F_j = 58,68,85,90,92,97,104,110` | depth 3, `F_j = 88, 90, 97` |

**Gates passed, every one, and several were not asked for.**

- `F(29) = 43`, `F(31) = 58`, `F(37) = 88` - the corpus ladder at `y = 29, 31, 37`.
- `F_j(29) = 43, 55, 65, 70, 85, 90` (all six on record) and `F_j(31) = 58, 68, 85, 90, 92, 97`
  (all six on record, from the `K_0 = 15` run up), and `F_2(37) = 90`, `F_3(37) = 97` (both on
  record) - `alignment-rules.md` 3.7 and 3.5, entry for entry.
- `sum m` at every rung equals `(q' - 2) N(M)` exactly - `27 x 7,952,175 = 214,708,725`,
  `29 x 214,708,725 = 6,226,553,025`, `35 x 6,226,553,025 = 217,929,355,875` - and `sum v m`
  equals the period: `1,078,282,205`, `33,426,748,355`, `1,236,789,689,135`.
- the individual multiplicities the parent records as its instrument gates: at m29
  `m(4) = 14,178,528`, `m(6) = 10,497,320`, `m(24) = 1,180`, `m(36) = 38`, `|Spec| = 41`, absent
  `{41, 42}`; at m31 `m(4) = 398,923,200`, `m(6) = 299,202,120`, `m(24) = 174,704`,
  `m(36) = 3,152`, `m(41) = 134`, `|Spec| = 55`, absent `{54, 56, 57}`. Every one exact.
- at m37, `|Spec| = 75` and the absent set
  `{73, 74, 75, 76, 78, 79, 80, 81, 82, 83, 84, 86, 87}` - **exactly the thirteen holes
  `cov_spectrum.md` 0a records for m37** ("m37 misses `v = 73,74,75,76,78..84,86,87` and then
  realises 88"), here with their multiplicities and from a different vehicle.
- the order distributions `n_J`: `199,048,197 / 15,416,706 / 243,822` at `23 -> 29` and
  `5,805,160,589 / 413,380,422 / 7,999,018 / 12,992 / 4` at `29 -> 31` - the parent's table 3.1,
  digit for digit.
- the per-depth extremes `Q*_J`: `34, 39, 43` at `23 -> 29` and `43, 55, 58, 55, 55` at
  `29 -> 31` - the parent's table 4.4, entry for entry.
- the chain counts, rebuilt inside this branch from the legal-word and all-pad counts
  (`C_r = W_{r-1} + Z_{r-1}`): `230,613,075 / 15,904,350 / 243,822 / 0` at `23 -> 29` and
  `6,655,970,475 / 429,417,450 / 8,025,014 / 13,000 / 4 / 0` at `29 -> 31`, with
  `S = 0.030661, 0.037437` and `Var = 0.070858, 0.066791` - the parent's rows 3.1 and 3.2.
- `L = 3, 3, 2` at m29, m31, m37 with `L_bare = 3, 3, 1` and `L_pad = 1, 2, 2` -
  `alignment-rules.md`'s measured rows, reproduced on this instrument. (`L_pad` is the longest
  realised legal word using at least one PAD letter, so that `L = max(L_bare, L_pad)`.)

The point of the table is not the numbers, which were known; it is the **provenance**. m31 has
6,226,553,025 gaps and m37 has 217,929,355,875, and neither was built: the whole of both came out
of a 3.9-million-row dictionary of a 7.95-million-gap machine, in three minutes, with the loss
counter at zero at every step.

### 2.1 What is new at the top rung

`31 -> 37` was the first rung the parent could not check against a direct build. This branch
computes it exactly:

- **`n_J(31 -> 37) = 205,591,124,261 / 12,223,428,142 / 114,732,724 / 70,532 / 216`** (`J = 1..5`),
  and `sum_J J n_J = 230,382,461,925 = 37 N(m31)` as conservation requires.
- **`Q*_J(31 -> 37) = 58, 68, 85, 88, 68`** (`J = 1..5`). So **`F(37) = 88` is carried by
  `J = 4`** - a fourfold fusion, three interior openings of m31 killed in one phase - while
  `J = 3` reaches only 85 and `J = 5` falls back to 68.
- **`C_r(31 -> 37) = 230,382,461,925 / 12,453,106,050 / 114,874,436 / 70,964 / 216 / 0`**, from
  which `n_J = C_{J-1} - 2C_J + C_{J+1}` reproduces the five `n_J` above exactly - the parent's
  identity, verified at a rung it could not reach.
- `S(31 -> 37) = 0.01846055`, `Var(order) = 0.054932` against the teeth-free floor `0.053878`.
- **`F_7(29) = 92`, `F_8(29) = 97`, `F_7(31) = 104`, `F_8(31) = 110`** - four entries past the
  recorded `F_j` rows.
- the whole m37 spectrum with multiplicities (75 values; the top is `m(70) = 44`, `m(71) = 8`,
  `m(72) = 2`, `m(77) = 2`, `m(85) = 4`, `m(88) = 2`).

### 2.2 The record's composition at each rung, read off the dictionary

`lc_detail.py` extracts, for each `Q*_J`, a realised window of `M` that attains it. The record
row (the `J` that attains `F(M + q')`) is in bold:

| rung | `J` | span | the window of `M` | letters |
|---|---|---|---|---|
| 23->29 | 1 | 34 | (34) | BAD |
| 23->29 | 2 | 39 | (5, 34) | BAD BAD |
| 23->29 | **3** | **43** | **(23, 10, 10)** | **BAD UP UP** |
| 29->31 | 2 | 55 | (20, 35) | BAD BAD |
| 29->31 | **3** | **58** | **(18, 10, 30)** | **BAD DOWN BAD** |
| 29->31 | 4 | 55 | (2, 21, 10, 22) | BAD UP DOWN BAD |
| 29->31 | 5 | 55 | (7, 10, 21, 10, 7) | BAD DOWN UP DOWN BAD |
| 31->37 | 3 | 85 | (18, 37, 30) | BAD PAD BAD |
| 31->37 | **4** | **88** | **(11, 12, 37, 28)** | **BAD DOWN PAD BAD** |
| 31->37 | 5 | 68 | (3, 25, 12, 25, 3) | BAD UP DOWN UP BAD |

Three independent confirmations fall out of this table. The `23 -> 29` record is the m23 3-run
with middle `10`, and `10 = 2u_{29}` is exactly the binding word `(10)` that
`alignment-rules.md` 3.3 records for that step. The `29 -> 31` record is `(18, 10, 30)` - the run
`neighbour_profile.md` names as the m31 record's ancestor, with middle `10 = q' - 2u_{31}`. And
the `31 -> 37` record is `(11, 12, 37, 28)`, whose two middles are `12` and `37` - **exactly the
padded even-`J` maximiser `(12, 37)` at m31** that `alignment-rules.md` 3.6 records with the note
"middle sums `12 mod 37`". The instrument reproduces all three witnesses without being told about
them.

The shape is the same every time: **flank + alternating letters (pads transparent) + flank**,
with both flanks BAD - they must be, or they would strike an endpoint and the fusion would not
stop there. The deepest cells show the alternation plainly: `(7, 10, 21, 10, 7)` at `29 -> 31` is
DOWN UP DOWN between two flanks of 7, which is `L(m29) = 3` realised.

---

## 3. The extended ladder

### 3.1 The table

| rung `q'` | `F` | `F_2` | budget `F(M)+q'` | slack | `L(M)` | `L_bare` | `L_pad` | `K_1 = J_max` | deepest `K_m` measured | `\|D\|` at that depth | time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 23 | 34 | 39 | 48 | 14 | **2** (m19) | 2 | **1** | 4 | `K_4 = 8` | 15,696 | 0.5 s |
| 29 | **43** | 55 | 63 | **20** | **1** (m23) | <= 1 | **1** | 3 | `K_12 = 17` | 34,357,093 | 1,536 s |
| 31 | **58** | 68 | 74 | **16** | **3** (m29) | **3** | **1** | 5 | `K_8 = 12` | 24,815,018 | 1,157 s |
| 37 | **88** | 90 | 95 | **7** | **3** (m31) | **3** | **2** | 5 | `K_3 = 8` | 30,325 | 306 s |
| 41 | **91** | - | 129 | **38** | **2** (m37) | **1** | **2** | 4 | `K_1 = 4` | section 5 | section 5 |

`L` is `L(M)` for the machine below the rung, i.e. the word length that sets `J_max = L + 2` for
that step; `L_bare` and `L_pad` are the corpus decomposition `L = max(L_bare, L_pad)`, with
`L_pad` the longest realised legal word using at least one PAD letter. Bold entries are measured
on this instrument - the m19 and m23 rows on the span-bounded ladder of 4.3, the rest on the
fixed-depth ladder. Measured: `L = 2, 1, 3, 3, 2` and `L_pad = 1, 1, 1, 2, 2` at
m19, m23, m29, m31, m37, reproducing `alignment-rules.md`'s two rows entry for entry, with
`L_bare = 3, 3, 1` at the top three: **the bare part stays at or below 3 and the padded part climbs 1, 2, 2** -
docs/proofs/12's `L_bare` cap and docs/proofs/10's open rider, seen on this instrument, and it is
why the depth the step must be given keeps rising.

### 3.2 The order distribution and the extremes, all rungs

| rung | `n_1` | `n_2` | `n_3` | `n_4` | `n_5` | `Q*_1` | `Q*_2` | `Q*_3` | `Q*_4` | `Q*_5` | record by |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 19->23 | 7,206,695 | 733,672 | 11,746 | 62 | - | 25 | 31 | 33 | 34 | - | `J = 4` |
| 23->29 | 199,048,197 | 15,416,706 | 243,822 | - | - | 34 | 39 | 43 | - | - | `J = 3` |
| 29->31 | 5,805,160,589 | 413,380,422 | 7,999,018 | 12,992 | 4 | 43 | 55 | 58 | 55 | 55 | `J = 3` |
| 31->37 | 205,591,124,261 | 12,223,428,142 | 114,732,724 | 70,532 | 216 | 58 | 68 | 85 | **88** | 68 | `J = 4` |
| 37->41 | 8,065,074,943,615 | 432,481,162,322 | 1,688,770,136 | 3,052 | 0 | 88 | 90 | 90 | **91** | - | `J = 4` |

The `31 -> 37` and `37 -> 41` rows are new. Read across the `Q*` columns: the per-depth frontier
rises to a maximum at `J = 4, 3, 3, 4, 4` and, wherever `J_max` leaves room above the peak, then
**collapses** - `43,55,58,55,55` and `58,68,85,88,68` are the two rungs with `J_max = 5`, and
both fall away by 3 and 20 after the peak, the collapse the parent measured at `29 -> 31` now
seen one rung higher. At `37 -> 41` there is nothing above the peak to collapse: `J_max = 4` and
the peak is at `J = 4`. **The peak is at `J <= 4` at all five rungs, including both rungs where
`J_max = 5`.**

`n_4(37 -> 41) = 3,052` deserves its own line: of the `8,499,244,879,125` gaps of m41, exactly
**3,052** are fourfold fusions, and the record `F(41) = 91` is one of them. The record of a
machine with eight and a half trillion gaps lives on an event of relative frequency `3.6e-10`.

### 3.3 The budget slack

| rung | `F(M)` | `q'` | budget | `F(M + q')` | slack |
|---|---|---|---|---|---|
| 19->23 | 25 | 23 | 48 | 34 | 14 |
| 23->29 | 34 | 29 | 63 | 43 | 20 |
| 29->31 | 43 | 31 | 74 | 58 | 16 |
| 31->37 | 58 | 37 | 95 | 88 | **7** |
| 37->41 | 88 | 41 | 129 | 91 | **38** |

No violation at any rung. The slack does not move monotonically: it widens to 20, narrows to 16
and then to **7** at `31 -> 37`, and springs back to 38 at `37 -> 41`. The narrow rung is the one
whose record climbs a depth (`J = 3` at `29 -> 31` to `J = 4` at `31 -> 37`, buying 30 units of
span for a budget increase of 6); the wide rung is the one where the record does not climb
(`J = 4` again, buying 3 units for a budget increase of 4). **The slack is a race between the
record's depth and the gear spacing, and the one narrow rung on this ladder is the rung where the
depth won.**

## 4. The depth question

### 4.1 `K_m` measured, and P2 refuted

`K_m(M -> q')` is the largest number of old gaps that `m` consecutive new gaps span; it is the
exact depth the closure step must be given for a depth-`m` output. Every entry below comes from
an unpruned run with `over0 = 0` in which the depth given was exactly consumed, so each is the
true value and not a bound.

| rung | `J_max` | measured `K_m` |
|---|---|---|
| 19->23 | 4 | `K_4 = 8` |
| 23->29 | 3 | `K_1 = 3`, `K_2 = 5`, `K_3 = 6`, **`K_9 = 14`**, **`K_10 = 15`**, **`K_12 = 17`** |
| 29->31 | 5 | `K_1 = 5`, **`K_5 = 9`**, **`K_6 = 10`**, **`K_8 = 12`** |
| 31->37 | 5 | **`K_1 = 5`**, **`K_2 = 6`**, **`K_3 = 8`** |
| 37->41 | 4 | **`K_1 = 4`** |

(`K_1, K_2, K_3` at 19->23 and 23->29 are the parent's, reproduced; the bold entries are new.
`K_1(37 -> 41) = 4` is exact without a deeper run: `K_1 = J_max` always, and `n_4 = 3,052 > 0`
shows the fourth order is realised.)

- `K_1 = J_max` at all five rungs, as the record law requires.
- **P2 is REFUTED.** It predicted `K_m - m <= 2 (J_max - 1)`. At `23 -> 29`, `J_max = 3` so the
  bound is 4, and `K_9 - 9 = K_10 - 10 = K_12 - 12 = 5`. The failure is not marginal in kind:
  `K_m - m` counts the MERGES inside a run of `m` new gaps, and a longer run has more chances to
  merge, so nothing depending on `J_max` alone can bound it. What survives is that it grows
  **slowly**: at `23 -> 29` the excess is `2, 3, 3` at `m = 1, 2, 3` and still only `5` at
  `m = 12`; at `29 -> 31` it is `4, 4, 4, 4` at `m = 1, 5, 6, 8`; at `31 -> 37` it is `4, 4, 5`
  at `m = 1, 2, 3`.
- **`K_m` still does not grow with the RUNG at fixed `m`.** `K_1 = 3, 5, 5, 4` at
  `23->29, 29->31, 31->37, 37->41`, on machines with `2.1e8`, `6.2e9`, `2.2e11` and `8.5e12`
  gaps: ordered by `J_max`, not by the machine, exactly as the parent found two rungs lower.
  `K_2 = 5, -, 6, -` and `K_3 = 6, -, 8, -` tell the same story.

### 4.2 The depth budget, and what it costs

A rung **spends** depth: given `K`, it returns `mmin(K)`, and since `K = K_{mmin}` the spend is
exactly `K_m - m`. Measured:

| rung | spend `K -> mmin` |
|---|---|
| 19->23 | `8 -> 4` (4) |
| 23->29 | `14 -> 9`, `15 -> 10`, `17 -> 12`, `21 -> 16` (5 every time) |
| 29->31 | `9 -> 5`, `10 -> 6`, `12 -> 8` (4), `16 -> 11` (5) |
| 31->37 | `5 -> 1`, `6 -> 2` (4), `8 -> 3`, `11 -> 5` (5, 6) |
| 37->41 | `3 -> 1` (2), `5 -> 2` (3) |

Against that, the record law needs `J_max = 3, 5, 5, 4` left over at the target. **A rung costs
two to six units of depth - rising slowly with the depth carried, because the spend IS
`K_m - m` - and returns three to five, so the ladder is close to break-even and its reach is set
by how deep a dictionary of the base machine is affordable.** The dictionary sizes that schedule
it:

| `K` | 1 | 2 | 3 | 4 | 5 | 6 | 9 | 10 | 12 |
|---|---|---|---|---|---|---|---|---|---|
| `\|D_K(m23)\|` | 33 | 429 | 3,135 | 15,696 | - | 158,066 | - | 1,587,434 | 2,732,706 |
| `\|D_K(m29)\|` | 41 | 730 | 7,184 | 45,854 | 208,668 | 720,527 | **8,818,629** | **15,240,585** | **34,357,093** |
| `\|D_K(m31)\|` | 55 | - | - | - | **636,575** | **2,678,901** | - | - | **24,815,018** (`K = 8`) |
| `\|D_K(m37)\|` | **75** | **2,053** | **30,325** | - | - | - | - | - | - |

(Bold entries are new; `|D_K(m23)|` at `K = 14, 15, 16, 17, 18, 20` is
`3,877,610 / 4,407,350 / 4,897,851 / 5,345,804 / 5,747,876 / 6,401,905`, against
`N(m23) = 7,952,175`.) The ratio per unit of depth at m29 falls from 2.30 (`K = 6..9`) to 1.73
(`9 -> 10`) to 1.50 (`10 -> 12`) - a falling ratio on a quantity that must saturate at
`N(m29) = 214,708,725`.

### 4.3 The span-bounded form: exactly closed, and exactly too big

P5 predicted that cutting windows at a span `S` rather than a depth `K` removes the truncation
entirely. It does, and the proof is one line: the span of `m` new gaps equals the span of the old
gaps they fuse, so "the input window ran out" and "the next new gap would leave the span cap" are
the same event. `V_S(M)`, the multiset of opening patterns of `M` inside `[x, x + S]`, is
therefore **exactly closed** under the rung step, with `loss = 0` by construction and no depth
bookkeeping at all.

Measured (`lc_smoke.py` gate 4, `S = 40`, from m13's period): m13 1,412 rows -> m17 15,311 ->
m19 132,374 -> m23 581,820, `loss = 0` at every step, and the spectrum EXACT against the directly
built machine at all three rungs (`F = 18, 25, 34`, `F_2 = 25, 31, 39`).

And it is unusable above the small machines. `|V_S(M)|` counts distinct patterns in a window of
`S` columns, which saturates at `N(M)`: **`|V_100(m23)| = 6,819,348` against
`N(m23) = 7,952,175`** - 86% of the machine's openings already carry a distinct span-100 pattern,
at the very machine the ladder starts from, and `S = 100` is barely above `F(37) = 88`. One rung
up, `|V_100(m29)|` is of order `2 x 10^8` rows of width 24. The fixed-depth form is the practical
one precisely because it forgets everything past `K` gaps, and forgetting is what costs the loss.
**The two forms are the two ends of one trade and this branch measured both ends.**

### 4.4 The span-threshold prune, which is what actually buys a rung

The way past 4.3 is not to keep more of the dictionary but to keep less of it, on a criterion
that provably loses nothing at the top.

> **Prune lemma.** Fix `theta`. If the whole stored window of a row spans less than `theta`, then
> no sub-run of it spans `theta` or more, so that row cannot be the ancestor of a gap of size
> `>= theta` at ANY later rung. Dropping it therefore leaves the multiset of windows of span
> `>= theta` unchanged at every rung above.

The threshold is free: `F(M + q') >= F_2(M) > F(M)` with no computation (`alignment-rules.md`
3.7, "the lower side is forced"), so `theta = F(M) + 1` is a valid lower bound for the next rung
and, since `F` is increasing, for every rung above it. Everything the pruned ladder reports about
values `>= theta` is exact; everything below `theta` is deliberately absent.

Measured (`lc_theta.py 21 89 41 23`, `theta = 89 = F(37) + 1`, base `D_21(m23)` pruned to span
`>= 89`, **1,992 s in total, 2.4 GB peak**):

| rung | depth in -> out | `\|D\|` after the prune | mass kept | `F(>= 89)` | `Q*_J` | over0 | secs |
|---|---|---|---|---|---|---|---|
| m23 (base) | - / 21 | 5,738,852 | 6,701,325 of 7,952,175 | - | - | - | 32 |
| m29 | 21 -> 16 | 28,360,251 | 43,560,966 of 214,708,725 | 43 | 34, 39, 43 | 0 | 1,005 |
| m31 | 16 -> 11 | 10,111,200 | 20,389,019 of 6,226,553,025 | 58 | 43, 55, 58, 55, 55 | 0 | 809 |
| m37 | 11 -> 5 | 18,185 | 56,566 of 217,929,355,875 | 88 | 58, 68, 85, 88, 68 | 0 | 146 |
| m41 | 5 -> 2 | **186** | 2,656 of 8,499,244,879,125 | **91** | 88, 90, 90, **91** | 0 | 0.1 |

The mass column is the point. At m41 the pruned dictionary is **186 distinct windows carried by
2,656 openings** out of eight and a half trillion - and it contains the record. The prune costs
nothing in exactness above `theta` and it removes, at m37, all but 56,566 of 218 billion
openings.

Note that `m = mmin` at every rung, so the completeness condition `cnt - 1 >= m` holds for every
surviving pair: the whole of the reported `loss` is the deliberate prune, none of it is
truncation, and `over0 = 0` says the first new gap was determined for every pair.

---

## 5. F(41) = 91, exactly, with no period above m23

This is the branch's push. `F(41) = 91` and

    Q*_J(37 -> 41)  =  88,  90,  90,  91   for J = 1, 2, 3, 4,

so **the record of m41 is a fourfold fusion** of m37 gaps, as the record of m37 was of m31 gaps.
The value agrees with the corpus (`F(41) = 91`, on record from COV-SAT at machine 41 complete,
`cov_spectrum.md` 0a) - a gate, not a discovery - but the route is new: a dictionary ladder from a
7,952,175-gap period, four rungs, 33 minutes, 2.4 GB, `over0 = 0` throughout.

What is new at this rung, and is not on record:

- **`n_J(37 -> 41) = 8,065,074,943,615 / 432,481,162,322 / 1,688,770,136 / 3,052 / 0`.** The
  first three are exact from the unpruned `K_0 = 17` ladder (a gap of order `<= 3` is fully
  determined by a depth-3 window); the fourth is exact because `L(m37) = 2` forces
  `J_max = 4`, so the entire mass that a depth-3 dictionary cannot place - measured as
  `over0 = loss = 3,052` - is `n_4`, and `n_5 = 0`.
- **`C_r(37 -> 41) = 8,935,103,590,875 / 435,858,711,750 / 1,688,776,240 / 3,052 / 0`**, with
  `C_2 = W_1 + Z_1 = 1,688,714,780 + 61,460` and `C_3 = W_2 + Z_2 = 3,052 + 0`. The branching
  identity `n_J = C_{J-1} - 2 C_J + C_{J+1}` then returns `n_4 = C_3 = 3,052` - the same number
  the truncated run measured as its `over0`, by a completely different route, which is the
  cross-check that makes `n_4` exact rather than inferred.
- **Only 3,052 gaps of m41 are fourfold fusions**, out of `8,499,244,879,125`; the record is one
  of them. Relative frequency `3.6 x 10^-10`.
- the budget slack at this rung is **38**, the widest on the ladder.

### 5.1 Where the exact instrument stops, and why that is the interesting number

The ladder is exact through `F(41) = 91` and stops there; `F(43) = 103` is out of reach on this
budget. P3 is therefore half confirmed and half refuted, and P6 - which predicted the stop
between m41 and m53 - is confirmed at its near edge.

The stopping condition is an inequality, not a hardware limit. Write `c(q')` for the depth a rung
spends (measured 2 to 6, rising slowly with the depth carried, 4.2) and `J_max(r)` for the depth
the record law needs at the target. Reaching rung `r` from base `M_0` needs

    K_0  >=  J_max(r)  +  sum over the intervening rungs of c(q'),

and it costs `|D_{K_i}(M_i)|` at every intervening machine. For `F(41)` that schedule was
`21 -> 16 -> 11 -> 5 -> 2` and the binding cost was `28,360,251` pruned rows at m29. For `F(43)`
the same spends give `23 -> 18 -> 13 -> 7 -> 4`, and the binding cost is the m29 dictionary at
depth 18, where two things go wrong at once:

1. `|D_K(m29)|` is `41, 730, 7,184, 45,854, 208,668, 720,527` at `K = 1..6` and
   `8,818,629 / 15,240,585 / 34,357,093` at `K = 9, 10, 12` (measured), with the ratio per unit of
   depth falling 2.30 -> 1.73 -> 1.50 as it saturates at `N(m29) = 214,708,725`; extrapolating
   the measured ratios puts `|D_18(m29)|` at or above `10^8`.
2. **the threshold prune stops working exactly there.** It bites when the depth the schedule needs
   is well below `theta / mean gap`; at m29 the mean gap is `5.021` and `theta = 92` gives
   `theta / mean = 18.3`, so at depth 18 the average window already spans about the threshold and
   little is dropped. At depth 16 with `theta = 89` the prune was worth a factor of about four
   (28,360,251 rows against an unpruned `|D_16(m29)|` extrapolated at `10^8`); at depth 18 it is
   worth close to nothing.

Both facts say the same thing in the end: **to certify a record of size `v` the instrument must
resolve the base-side machine's opening pattern over stretches of length `v`, and the number of
distinct such patterns is within a small factor of the machine's own opening count**
(`|V_100(m23)| = 6,819,348` against `N = 7,952,175` makes it concrete). The ladder buys rungs by
forgetting; the record is the one thing that cannot be forgotten.

---

## 6. The truncated form, and what it is worth

At a depth below `K_m` the instrument is sound in one direction only, and this branch has the
unusual luxury of knowing the true answer for the rung it truncated.

- **Lower bound, rigorous.** Every window a truncated run emits is a genuine `(window, phase)`
  pair of the real machine, so its largest value is a realised gap. Truncating at depth `K` sees
  fusions of order `J <= K` only, so the bound it returns is `max_{J <= K} Q*_J`. At `37 -> 41`,
  where `Q*_J = 88, 90, 90, 91`:

  | depth kept at m37 | 1 | 2 | 3 | 4 |
  |---|---|---|---|---|
  | lower bound on `F(41)` | 88 | 90 | 90 | **91 (exact)** |

  The `K_0 = 17` ladder ended with depth 3 at m37 and duly returned `F(41) >= 90`, **off by
  exactly one**, with `over0 = 3,052` flagging that a fourfold fusion existed and had not been
  seen. Truncation is not noisy here; it is a clean floor that rises one step at a time and hits
  the answer exactly when the depth reaches `J_max`.
- **Upper bound, rigorous but useless.** `Q*_J <= F_J` and emptiness is upward closed
  (`alignment-rules.md` 3.7), so `F(M + q') <= max_{2 <= J <= J_max} F_J(M)`. At `37 -> 41` with
  `J_max = 4` the surviving depth gives `F_2(37) = 90` and `F_3(37) = 97`; `F_4(37)` is out of
  reach and the free bound `F_4 <= 2 F_2 = 180` is all there is, so the interval is
  `91 <= F(41) <= 180` with the budget 129 strictly inside it.

P7 is CONFIRMED as pre-registered: **a truncated closure is a lower-bound instrument**, and that
is the same side SAT fails on past the wall (`cov_spectrum.md`: `F(61) >= 171`, `F(67) >= 175`,
`F(71) >= 185`, no upper bound past m41). Two vehicles with nothing in common fail on the same
side, and the reason is the same in both: exhibiting a long run is finding one object, ruling one
out is quantifying over all of them.

---

## 7. Toward the root

### 7.1 What the new rungs say

- **The slack is not monotone and its jumps have a cause.** 14, 20, 16, **7**, **38** at
  `19->23, 23->29, 29->31, 31->37, 37->41`. The one narrow rung, `31 -> 37`, is the rung whose
  record climbs from `J = 3` to `J = 4`; the wide rung above it is the one where the record's
  depth stays at 4. A unit of record depth is worth roughly one middle gap of `M` - 30 units of
  span at `31 -> 37` - while a rung of budget is worth `q' - q'_prev`, which is 4 to 6. So the
  budget question is, at least on these five rungs, entirely a question about **when the record's
  depth increases**, and the depths measured are `4, 3, 3, 4, 4`.
- **The `Q*_J` profile peaks at `J <= 4` at every rung**, including both rungs where
  `J_max = 5`: `25,31,33,34`; `34,39,43`; `43,55,58,55,55`; `58,68,85,88,68`; `88,90,90,91`.
  Wherever `J_max` leaves room above the peak the profile collapses (by 3 at `29 -> 31`, by 20 at
  `31 -> 37`). The collapse is the flank envelope: a legal word of length
  `J - 2` forces its middles into the two letter classes, of sizes about `q'/3` and `2q'/3`, so
  past the peak each extra letter costs more span than the two flanks can return.
- **The record's composition is the same shape at every rung**: BAD flank, then an alternating
  legal word with pads transparent, then a BAD flank (2.2). Both flanks must be BAD, or they
  would strike an endpoint and the run would not stop there. The middles at the record are
  `(10)`, `(10)`, `(12, 37)` at `23->29`, `29->31`, `31->37` - two bare letters and then a
  bare-plus-pad pair, which is `L_pad` climbing from 1 to 2 seen at the record itself.
- **The functionals**, all exactly computable at each rung from the dictionary alone, along the
  extended ladder. Nothing here is claimed monotone; they are listed so the next branch can test
  them.

| rung | `W_1/N` (legal 1-word) | `W_2/N` | `W_3/N` | `Z_1/N` (all-pad) | `S` | `Var(order)` | floor `2(q'-4)/(q'-2)^2` | `Var - floor` |
|---|---|---|---|---|---|---|---|---|
| 23->29 | 0.0306603 | 0 | 0 | 7.54511e-07 | 0.0306610 | 0.070858 | 0.068587 | 0.002271 |
| 29->31 | 0.0373665 | 6.05471e-05 | 1.86299e-08 | 9.73412e-06 | 0.0374369 | 0.066791 | 0.064209 | 0.002582 |
| 31->37 | 0.0184449 | 1.13970e-05 | 3.46901e-08 | 4.23445e-06 | 0.0184606 | 0.054932 | 0.053878 | 0.001054 |
| 37->41 | 0.0077489 | 1.40050e-08 | 0 | 2.82018e-07 | 0.0077492 | 0.049050 | 0.048652 | 0.000397 |
| 41->43 | 0.0090484 | - | - | 1.97683e-06 | - | - | 0.046401 | - |

  `W_1/N`, the density of openings whose next gap is a letter of the incoming gear, falls by a
  factor of 4.8 from `29 -> 31` to `37 -> 41` while `q'` grows by only 32% - **much faster than
  `1/q'`** - and then RISES again at `41 -> 43` (0.00905 against 0.00775). It is not monotone,
  and the `41 -> 43` row is computed on a dictionary missing `3,052` of `8.5 x 10^12` openings,
  which cannot account for it. `Z_1/N`, the all-pad density, does the same: `9.73e-06`,
  `4.23e-06`, `2.82e-07`, then up to `1.98e-06`. `Var - floor = 2S/(q'-2)` falls at every rung
  measured (`0.00227, 0.00258, 0.00105, 0.00040`) - the parent's "the teeth are seen only through
  `S`" continuing to hold as the machine grows. And `W_3/N` RISES, from `1.86e-08` at `29 -> 31`
  to `3.47e-08` at `31 -> 37`, while `W_1/N` and `W_2/N` both fall over the same step - the
  deepest legal words get commoner as the machine grows even as the shallow ones get rarer,
  before vanishing at `37 -> 41` where `L(m37) = 2` forbids them. Those two crossings are the
  cleanest things here for the next branch to test for monotonicity.
- **The dictionary's growth, as a functional**: `|D_K(m29)|` per unit of depth has ratios
  `2.30 (K=6..9), 1.73 (9->10), 1.50 (10->12)`, and `|D_K(m31)|` gives `636,575 -> 2,678,901`
  at `K = 5 -> 6` (ratio 4.2) and `24,815,018` at `K = 8` (ratio 3.0 per step). The ratio at
  fixed `K` grows with the rung while the ratio in `K` at fixed machine falls - which is the
  arithmetic of the wall in 5.1.

### 7.2 Where this meets the lap-phase transfer (stopped in a paragraph)

`alignment-rules.md` 3.5 already computes `F_J(M')` for a distant `M'` on a small machine's
period by enumerating phase tuples of several new gears at once; it is the vehicle behind
`F(59) = 161` from m23's period. The instrument here is the same identity applied **one gear at a
time, carrying the whole dictionary forward**, which is why it produces the entire spectrum,
`n_J` and `Q*_J` and not only the extremes - and why, for the record question alone, it is the
more expensive of the two. The two are not competitors; this branch does not attempt the
multi-gear form, and the sub-question is stopped here.

### 7.3 The negative, as pre-registered

Nothing above bounds anything. `F(M + q') = max_J Q*_J(M)` is a maximum over the old dictionary's
realised windows; this branch gives a faster and more informative way to compute that maximum, not
a reason it cannot be large. The residual is node R1.2's chain statement, unchanged.

---

## 8. What holds without exception

| statement | count | status |
|---|---|---|
| the iterated closure step reproduces every corpus gate it can reach (`F`, `F_j`, `sum m`, `sum v m`, `\|Spec\|`, absent sets, individual multiplicities, `n_J`, `Q*_J`, `C_r`, `L`, `L_bare`, `L_pad`) | 4 rungs, 12 gate families, 0 exceptions | exact |
| `over0 = 0` at every reported rung, so the reported spectrum is exact | 4 rungs, unpruned and pruned ladders | exact |
| `K_1 = J_max` at every rung | 5 of 5 | exact (it is the record law) |
| `K_m` tracks `J_max` and not the rung | 5 rungs, machines from `7.9e6` to `8.5e12` gaps | measured, 0 exceptions |
| the depth a rung spends is exactly `K_m - m`: 2 to 6, rising slowly with the depth carried, and identical at every input depth tried on a given rung | 5 rungs, four input depths at `23 -> 29` all spending 5 | measured |
| the span-bounded dictionary has `loss = 0` by construction | proved; 3 rungs verified | proved (4.3) + verified |
| the prune lemma: a window spanning less than `theta` has no sub-run spanning `theta` | proved; 4 rungs, `F` unchanged under the prune at every one | proved (4.4) + verified |
| `F(M + q') <= F(M) + q'` | 5 of 5, slacks 14, 20, 16, 7, 38 | measured, no violation |
| `Q*_J` peaks at `J <= 4`, and collapses after the peak wherever `J_max` leaves room | 5 of 5 (peaks at `J = 4, 3, 3, 4, 4`) | measured |
| the record's window is BAD flank + alternating legal word + BAD flank | 3 rungs with witnesses, all `J` | measured, and forced (a letter flank would strike the endpoint) |
| `L_bare = 3, 3, 1` while `L_pad = 1, 2, 2` at m29, m31, m37 | 3 of 3 measured, matching the corpus rows | measured |

---

## 9. What is new

1. **The closure step as an iterated machine.** Dictionary in, dictionary out, with `loss` and
   `over0` as its own certificate. The whole spectrum of m37 - `217,929,355,875` gaps, period
   `1,236,789,689,135` - came out of a 3.9-million-row dictionary of m23 in 195 seconds, and
   `F(41) = 91` came out of the same base in 33 minutes.
2. **The span-threshold prune (4.4), with its lemma**: a row whose stored window spans less than
   `theta` can never be the ancestor of a gap of size `>= theta`, so it may be dropped at every
   level without disturbing anything above `theta`; and `theta = F(M) + 1` is free because
   `F(M + q') >= F_2(M) > F(M)`. This is what buys the last rung: at m41 the pruned dictionary is
   **186 windows carried by 2,656 openings out of 8,499,244,879,125**, and it contains the record.
3. **`31 -> 37` computed exactly**: `n_J = 205,591,124,261 / 12,223,428,142 / 114,732,724 /
   70,532 / 216`, `Q*_J = 58, 68, 85, 88, 68`, `C_r = 230,382,461,925 / 12,453,106,050 /
   114,874,436 / 70,964 / 216 / 0`, the m37 spectrum with all 75 multiplicities,
   `S = 0.01846055`, `Var = 0.054932`.
4. **`37 -> 41` computed exactly**: `F(41) = 91` with `Q*_J = 88, 90, 90, 91`, so the record is a
   fourfold fusion; `n_J = 8,065,074,943,615 / 432,481,162,322 / 1,688,770,136 / 3,052 / 0`, with
   `n_4 = 3,052` confirmed twice over (as the mass a depth-3 dictionary cannot place, and as
   `C_3 = W_2 + Z_2 = 3,052` through the branching identity). **Of eight and a half trillion gaps
   of m41, 3,052 are fourfold fusions and the record is one of them.**
5. **The record's composition at each rung, from the dictionary** (2.2): BAD flank + alternating
   legal word + BAD flank, with the witnesses `(23, 10, 10)`, `(18, 10, 30)`, `(11, 12, 37, 28)`.
   The instrument recovers, without being told, the binding word `(10)` of `alignment-rules.md`
   3.3, the run `(18, 10, 30)` of `neighbour_profile.md`, and the padded even-`J` maximiser
   `(12, 37)` of `alignment-rules.md` 3.6.
6. **`F_7(29) = 92`, `F_8(29) = 97`, `F_7(31) = 104`, `F_8(31) = 110`** - four entries past the
   recorded `F_j` rows; `K_4(19->23) = 8`, `K_9(23->29) = 14`, `K_10 = 15`, `K_12 = 17`,
   `K_5(29->31) = 9`, `K_6 = 10`, `K_8 = 12`, `K_1(31->37) = 5`, `K_2 = 6`, `K_3 = 8`,
   `K_1(37->41) = 4`; `|D_9(m29)| = 8,818,629`, `|D_10| = 15,240,585`, `|D_12| = 34,357,093`,
   `|D_5(m31)| = 636,575`, `|D_6| = 2,678,901`, `|D_8| = 24,815,018`, `|D_1(m37)| = 75`,
   `|D_2| = 2,053`, `|D_3| = 30,325`, `|V_100(m23)| = 6,819,348`.
7. **The span-bounded dictionary is exactly closed under the rung step** - no depth truncation at
   all - and its size is within 14% of the machine's own opening count already at m23, which is
   why the depth-truncated form is the practical one. Both ends of that trade measured.
8. **`K_m - m` grows with `m`** (2, 3, 3, ..., 5 at `23 -> 29`), refuting the pre-registered
   `2(J_max - 1)` bound; `K_m` still does not grow with the rung, over machines spanning six
   orders of magnitude in size.
9. **The wall, as an inequality rather than a hardware limit** (5.1): a rung spends `K_m - m`
   units of depth (2 to 6, measured) and returns `J_max = 3..5`, and the threshold prune stops
   paying at the depth where the
   average window already spans `theta` - which is `theta / (mean gap)`, i.e. exactly the depth
   the record needs.
10. **The budget slack is not monotone**: 14, 20, 16, **7**, **38**, and the narrow rung is the
    one where the record's depth climbs (`J = 3 -> 4` at `31 -> 37`) while the wide rung is the
    one where it does not.
11. **The functionals along the ladder** (7.1), with two non-monotone crossings recorded for the
    next branch: `W_1/N` falls 0.0374, 0.0184, 0.00775 and then RISES to 0.00905 at `41 -> 43`;
    `W_3/N` rises `1.86e-08 -> 3.47e-08` while `W_1/N` and `W_2/N` fall over the same step.

**Prior art inside the project.** `branching_identity.md` (the closure theorem and the size
formula - this branch is its iteration, and reproduces its tables as gates); docs/proofs/09 (the
record law); docs/proofs/10 (`J_max = L + 2`); docs/proofs/12 (`L_bare` capped);
`alignment-rules.md` 3.3 (the binding words, recovered here as witnesses), 3.5 (the lap-phase
transfer, compared in 7.2 and not attempted), 3.6 (the padded even-`J` maximiser, recovered), 3.7
(the `F_j` rows used as gates, `F(M+q') >= F_2(M)` used to license the threshold, and the
spectrum-plus-depth certificate used for the upper half of the m41 interval);
`neighbour_profile.md` (the run `(18, 10, 30)`, recovered); `cov_spectrum.md` (m37's hole list and
`F(41) = 91`, both reproduced here as gates; and the same lower-bounds-only asymmetry past the
wall).

**Prior art outside.** Not checked (no web access this round).

---

## 10. Verdict

**INSTRUMENT, exact, with a measured wall; not a route.** The closure step iterates. The ladder
runs on dictionaries alone - from a 7,952,175-gap base to a machine with `8.5 x 10^12` gaps -
with `over0 = 0` at every rung, every corpus gate passed (twelve families of them, including
thirteen spectral holes at m37 that were on record only from SAT and three record witnesses that
were on record in three different documents), and it produces the first exact `n_J`, `Q*_J`,
`C_r` and spectrum for `31 -> 37` and `37 -> 41`. It says `F(37) = 88` and `F(41) = 91` are both
fourfold fusions, and that only 3,052 of m41's eight and a half trillion gaps are fourfold at all.

The push stops at m41. The reason is measured and is a property of the object: **a rung spends
two to six units of dictionary depth and returns three to five, and the one prune that makes
the dictionary affordable - drop every window spanning less than the target - stops paying at
exactly the depth the record needs**, because that depth is `theta / (mean gap)` and there the
average window already clears the threshold. Truncating below that depth restores the reach and
destroys the upper half of the answer: the depth-`K` floor is `max_{J <= K} Q*_J`, which at
`37 -> 41` reads 88, 90, 90, 91 - one short until the depth reaches `J_max`. The pre-registered
negative stands: this is a faster and far more informative way to compute `max_J Q*_J(M)`, not a
reason it stays small.

**Child named** (the lowest-order interaction not yet proved on the way from these parts to the
shape): **the `Q*_J` peak sits at `J <= 4` at every rung, including the two rungs where
`J_max = 5`.** Measured peaks: `J = 4, 3, 3, 4, 4` at the five rungs, with `Q*_5` falling back to
55 and 68 where it exists at all. If "the peak is at `J <= 4`" could be proved for every machine,
the record law's maximum would be over four depths instead of `L + 2`, and `L` bounded - the open
rider of docs/proofs/10 that this branch watched grow (`L_pad = 1, 2, 2` measured at m29, m31,
m37) - would stop
mattering to the record even while it keeps mattering to the recursion. That is the first
statement on this ladder that would change what the record depends on rather than how fast it can
be computed.

---

## 11. Dead ends

- **"The span-bounded dictionary is the right object because it has no truncation."** Half true
  and pre-registered as P5: it has no truncation and it does not fit. `|V_100(m23)| = 6,819,348`
  against `N(m23) = 7,952,175`; the object the record needs is, at that resolution, the machine.
- **"`K_m - m` is bounded by a function of `J_max` alone."** REFUTED at `23 -> 29`:
  `K_9 - 9 = K_10 - 10 = K_12 - 12 = 5 > 2(J_max - 1) = 4`. The excess counts merges inside a run
  of `m` new gaps and a longer run has more chances to merge; what is true is that it grows
  slowly and does not grow with the rung.
- **"A deeper base dictionary buys more rungs."** It buys one, and only with the prune: unpruned,
  `K_0 = 17` reaches m37 exactly and returns `F(41) >= 90`; pruned at `theta = 89`, `K_0 = 21`
  reaches `F(41) = 91`. The next rung needs the m29 dictionary at depth 19-21, where the prune no
  longer pays.
- **"The truncated closure gives a usable interval past the wall."** REFUTED as pre-registered:
  `[91, 180]` at m41, budget 129 strictly inside. The truncated form is a lower-bound instrument,
  like SAT and for the same reason - both can exhibit a witness and neither can exhaust the
  alternatives.
- **"Computing the whole dictionary is the way to the record."** Half refuted by 4.4: the pruned
  ladder, which throws away 99.99999997% of the mass at m41, gets the record and the unpruned one
  does not, inside the same budget. The record is a property of a vanishingly small part of the
  dictionary, and the instrument that finds it is the one that knows which part to keep.
