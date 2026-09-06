# Node 4.i.a.i.a - THE SHORT-LETTER ROW: how high the row `v = a_L` of the adjacent-pair dictionary reaches

Parent: node **4.i.a.i, the availability gate** (`research/proof/availability_gate.md`, FACT,
2026-09-06), whose section 5 handed forward exactly one statement:

> **What the branch hands forward** is a strictly smaller and more concrete statement than the one
> it was opened with: *the row `v = a_L` of the level-2 dictionary of `M` is empty above `c F(M)`*.
> It is finite, it is about `M` alone, it is about ONE gap size rather than a whole profile, and
> the LP lane already certifies level-2 dictionary cells by duality at m19 and m23 -- which is
> where a child branch should start.

What spawned this branch exactly: the parent's identity `a_hasM = max { a : (a, a_L) in Dict_2(M) }`
(7 of 8 rungs) reduced the whole availability question to one row of one matrix, and the only proved
bound on that row, `r(a_L) <= F_2 - a_L`, is **vacuous at the rung that matters** (rung 29->31:
`F_2(m29) - a_L(31) = 45 > 43 = F(m29)`).

Scripts in `research/anchor235/r60/`; result outputs in `research/anchor235/r60/results/`
(untracked). Every number this document relies on is written into the document.

---

## 0. Pre-registered (written before any computation of this branch)

### 0.1 The objects, defined exactly

Machines `M_0 = {5}`, ..., `M_8 = {5..31}`; a **rung** is `(M, q')` with `q'` the incoming gear.
Gaps are cyclic over the full period, `F(M)` the largest, `F_2(M)` the largest sum of two adjacent
gaps. `u_g = 6^{-1} mod g`, teeth `T_g = {u_g, -u_g} mod g`, `d_g = 2u_g`; the letters of `q'` are
`a_L = min(2u_{q'} mod q', q' - 2u_{q'} mod q')` and `b_L = q' - a_L` (file 05 T1), and
`3 a_L = q' -+ 1`.

**The dictionary.** `D[a][v]` = the number of positions of the period at which a gap of size `a` is
immediately followed by a gap of size `v`; `Dict_2(M) = { (a,v) : D[a][v] > 0 }`;
`F_2(M) = max { a + v : (a,v) in Dict_2 }`.

**The row of a size `v`:**

    R(v) := { a : (a, v) in Dict_2(M) or (v, a) in Dict_2(M) }      (the sizes that ever sit next to a v-gap)
    r(v) := max R(v),  0 if empty                                    (the ROW MAXIMUM)

`r(v)` is the largest SINGLE neighbour a `v`-gap can have; `N(v)` (branch 2g.i) is the largest SUM
of its two neighbours. Both are read off `D`. The **short-letter row** is `R(a_L)`; the **long-letter
row** is `R(b_L)`; the **padded row** is `R(q')`.

**The deficit.** `r(v) <= F(M)` trivially and `r(v) <= F_2(M) - v` by the definition of `F_2`, so

    d(v) := min( F(M), F_2(M) - v ) - r(v)  >= 0

is the amount by which the row falls short of the two free caps. `d(a_L)` is the whole content of
the residual statement.

**The endpoint cost of a size.** For a gear `p` and a gap of size `v` with left end `x`, both ends
are openings, so `x` must avoid `T_p` and `x + v` must avoid `T_p`:

    c_p(v) := | T_p  union  (T_p - v) |   in {2, 3, 4}

is the number of residues mod `p` forbidden to the left end of a `v`-gap. Exactly:
`c_p(v) = 2` iff `p | v`; `c_p(v) = 3` iff `p | 3v - 1` or `p | 3v + 1` (and `p` does not divide
`v`); `c_p(v) = 4` otherwise. Write

    Leg(v) := { p prime >= 5 : c_p(v) <= 3 } = { p : p | v (3v-1) (3v+1) }

-- the gears for which the chain law's "both ends struck in one copy" condition
`v = 0, +d_p, -d_p (mod p)` (file 05 (C)) can hold. These are the gears that can **close an entire
`v`-gap**, i.e. carry it as an interior piece of a fusion.

### 0.2 The theory

**T. The height of the short-letter row is not set by the letter's size, and not by `F_2`; it is set
by the SMALLEST gear's endpoint cost at that size. `a_L = (q' -+ 1)/3` is a size whose endpoint cost
at gear 5 is decided by `q' mod 5` alone, and the row reaches high exactly at the rungs where that
cost is 2 or 3 and low where it is 4. So `r(a_L) <= c F(M)` with `c < 1` is FALSE as a uniform
statement over rungs unless `c` is close to 1, and the honest object is the profile `r(v)` with its
gear-5 stratification.**

The mechanism, stated before measuring: a pair `(a, v)` needs three openings `x_0 < x_1 < x_2` with
`x_1 - x_0 = a`, `x_2 - x_1 = v`, and every column between them blocked. Gear 5 blocks two of every
five columns and is the machine's dominant blocker; the three openings cost it
`|T_5 union (T_5 - a) union (T_5 - a - v)|` classes out of five, and when `v` is such that
`T_5 - a - v` collapses onto `T_5 - a` or `T_5` the pair is far cheaper to realise. `c_5(v) = 2`
exactly when `5 | v`, `= 3` when `5 | 3v -+ 1`. For `v = a_L` the second condition reads
`5 | q'` (impossible) or `5 | q' -+ 2` -- so the letter's gear-5 cost is a statement about
`q' mod 5` and `q' mod 3` and nothing else.

### 0.3 Predictions, each with the number that would refute it

- **S1 (instrument).** `r(a_L) = 2, 0, 3, 7, 12, 20, 25, 35` at rungs 5->7 .. 29->31
  (`availability_gate.md` 2.3, where it is `max a adjacent to a_L`); `F = 2, 5, 7, 11, 18, 25, 34,
  43` and `F_2 = 4, 7, 11, 16, 25, 31, 39, 55` at `M_0..M_7`. Any mismatch is an instrument failure.
- **S2 (the brief's row bound).** `r(a_L) <= 0.85 F(M)` at every rung. I expect it REFUTED at rung
  5->7 only (there `r(a_L) = F = 2`) and held at 7 of 8, with the top three at
  `0.800, 0.735, 0.814`. REFUTED as a uniform statement by any ratio above 0.85; refuted as a
  near-uniform statement if 3 or more rungs exceed 0.85.
- **S3 (the profile).** `d(v) >= 0` by definition. Predict `d(v) = 0` -- the row reaching its free
  cap exactly -- at a MAJORITY of realised `v` at every machine m17..m29; and predict `r(v)` is NOT
  monotone in `v` above any threshold. REFUTED on the first clause if `d = 0` at fewer than half the
  realised sizes at 2 or more machines; on the second if `r` is non-increasing from `v = 7` up at
  every machine m17..m29.
- **S4 (the deciding mechanism, new).** Stratify the realised sizes of one machine by `c_5(v)`.
  Predict the median deficit is strictly ordered `d(c_5 = 2) < d(c_5 = 3) < d(c_5 = 4)` at every
  machine m13..m29 (5 of 5). And across rungs, `c_5(a_L) = 3, 4, 4, 4, 4, 3, 2, 2` at
  5->7 .. 29->31; predict every rung with `c_5(a_L) = 4` has `r(a_L)/F <= 0.70` and every rung with
  `c_5(a_L) <= 3` has `r(a_L)/F >= 0.73`. REFUTED by one crossing of that gap, or by a median
  ordering that fails at 2 or more machines.
- **S5 (the closers, exact).** `Leg(v) = { p : p | v } union { p : p | 3v-1 } union { p : p | 3v+1 }`
  agrees with the chain-law test `v = 0, +-d_p (mod p)` at every `(v, p)` with `v <= 120`,
  `p <= 120`. For `v = a_L`: `{3a_L - 1, 3a_L + 1} = {q', q' -+ 2}` exactly, hence
  **the only gear outside `M` that can close an `a_L`-gap of `M` is `q'` itself**, unless
  `q' = 2 (mod 3)` and `q' + 2` is prime, in which case `q' + 2` also can. Predict 8 of 8 rungs,
  with the twin-above exception at rungs 7->11 (`13`), 13->17 (`19`) and 23->29 (`31`).
- **S6 (initial segments).** Every realised size `a <= r(a_L)` occurs in `R(a_L)`: predict at most 2
  holes per rung. REFUTED by 3 or more holes at any rung.
- **S7 (twin rungs).** At the four rungs where `q' - 2` is prime and in `M` (5->7, 11->13, 17->19,
  29->31) the lower twin lies in `Leg(a_L) cap M`. Predict this by itself does NOT raise the row:
  the two rungs whose only member of `Leg(a_L) cap M` is a big gear (11->13, 17->19) have
  `r(a_L)/F = 0.636, 0.667`, strictly below every rung with `5 in Leg(a_L)`. REFUTED by a crossing.
- **S8 (the LP certificates).** The LP lane's windowed vehicle (`research/star_case.py`
  `two_gap_geometry` + `decide_star`, the construct of `docs/novel/restricted-covering-certificates`
  RESULT 4) certifies the cell `(a, a_L)` as unrealisable for every `a > r(a_L)` up to `F(M)`, at
  m19 (`a_L = 8`, `a = 21..25`, spans 29..33) and at m23 (`a_L = 10`, `a = 26..34`, spans 36..44),
  with gear 5 held and gear 7 held where needed -- 14 cells, and a scan-free proof of the row
  statement at two machines. Predict 14 of 14 certified and the vehicle REFUSING at `a = r(a_L)`
  (tightness). REFUTED if 3 or more cells stall, or if any certified cell is realised (soundness
  failure).
- **S9 (out of sample: rung 31->37).** `a_L(37) = 12`, `3 a_L - 1 = 35 = 5 * 7`, so
  `Leg(12) cap M = {5, 7}` and `c_5(12) = 3`. Predict `r(12) >= 0.73 F(m31) = 42.3`, i.e.
  `r(12) >= 43`, and `r(12) <= min(58, 68 - 12) = 56`. REFUTED by `r(12) <= 42`.

- **S10 (added after the m5..m29 table was computed and before the m31 result was read; a post-hoc
  law on the eight known rungs with a genuine out-of-sample test on the ninth).** Reading the eight-rung table, `r(a_L) + a_L - F(M) =
  2, -, 0, 2, 0, 3, 1, 2` at the seven rungs where `a_L` is realised -- i.e.
  **`r(a_L) <= F(M) - a_L + 3`**, which is a bound BELOW `F(M)` as soon as `a_L > 3`, and is 9
  below the proved pair cap `F_2 - a_L` at the top rung. It is not a property of a general size
  `v` (at m29, `v + r(v)` reaches `F_2 = 55` at `v = 20, 25, 30, 35`), so it is a statement about
  the letter. OUT OF SAMPLE at rung 31->37: it predicts `r(12) <= 58 - 12 + 3 = 49`; with S9 that
  brackets `43 <= r(12) <= 49`. REFUTED by `r(12) > 49` (or by `r(12) < 43`, which refutes S9).

**Stop rules.** Anything that reduces to the merge/chain law and the letters (file 05), the
attainment identity (08), the peel bound / triple inequality / middle-sum lemma (16), the
neighbour law `N(v) <= F_2` and the glue lemma (2g.i, `neighbour_profile.md`), the gear-5 weight
`w(s mod 5) = 3,1,2,2,1` (`neighbour_profile.md` 2.5), the divisor law `q | 3q_1 - 1`
(`alignment-rules-index.md` L20), the gate ladder and the pair cap `r(a_L) <= F_2 - a_L`
(`availability_gate.md` 2.4, 3.2), or the LP lane's windowed dictionary vehicle
(`research/window_dict.py`, `docs/novel/restricted-covering-certificates.md` RESULT 4) is stopped
in one line and cited.

### 0.4 Scorecard

| # | prediction | verdict | evidence |
|---|---|---|---|
| S1 | instrument: `r(a_L) = 2,0,3,7,12,20,25,35`; `F`, `F_2` ladders | CONFIRMED exactly, all three ladders | 1 |
| S2 | `r(a_L) <= 0.85 F` at every rung | CONFIRMED at 8 of 9; REFUTED at rung 5->7 only (`r = F = 2` on the machine `{5}`). The ratio is FLAT at about four fifths over the top five rungs: 0.667, 0.800, 0.735, 0.814, 0.793 | 2.1 |
| S3 | `d(v) = 0` at a majority of realised `v`; `r` not monotone | first clause REFUTED (the row reaches its free cap at 11-26% of sizes from m19 up); second clause CONFIRMED (7 of 9 machines) | 2.2 |
| S4 | median `d` ordered by `c_5`; the `0.70 / 0.73` separation | cross-rung clause CONFIRMED **9 of 9** with a clear gap (`c_5 = 4` at or below 0.667, `c_5 <= 3` at or above 0.735); the within-machine MEDIAN-DEFICIT clause REFUTED (3 of 7), and replaced by the trend-free LOCAL CONTRAST, ordered at 6 of 7 | 2.3 |
| S5 | `Leg(v)` divisor form; only `q'` (and `q'+2` when prime) close an `a_L`-gap | CONFIRMED: 0 of 1800 `(gear, size)` mismatches, 9 of 9 rungs, and the second outside closer appears at exactly the three predicted rungs (7->11, 13->17, 23->29) | 2.4 |
| S6 | at most 2 holes in the short-letter row | REFUTED (3 or more holes at 4 of 8 rungs, 10 at rung 31->37) -- but the holes are exactly the pair filter's forbidden class mod 5, plus one extra per rung | 2.6 |
| S7 | the twin's big gear does not raise the row | CONFIRMED in content: the two rungs whose only `Leg cap M` member is a big gear sit at 0.429 and 0.667, below every rung with `5 in Leg(a_L)`; and twin status by itself gives NO separation of the row height | 2.5 |
| S8 | 14 of 14 LP certificates at m19, m23; tight at `r(a_L)` | CONFIRMED and EXCEEDED: 14 of 14 cells at m19 and m23 (14,134 and 79,483 exact operations), 0 unsound, the vehicle refusing at `a = r(a_L)` at both; and the same at **m29, the rung that matters** -- 8 of 8 cells `a = 36..43` certified in 270,070 exact operations, refusing at `a = 35` | 4.3, 4.4 |
| S9 | out of sample `r(12) >= 43` at m31 | CONFIRMED: `r(12) = 46 = 0.793 F(m31)`, computed after the prediction was written | 2.1, 2.3 |
| S10 | out of sample `r(12) <= 49`; `a_L + r(a_L) <= F + 3` | CONFIRMED: `r(12) = 46`, `a_L + r(a_L) = 58 = F(m31)` exactly; the law holds 8 of 8 with `G(a_L) - F = 2, 0, 2, 0, 3, 1, 2, 0` | 2.7 |

---

## 1. Setup (exact ranges)

Everything exact: full periods, integer arithmetic, no sampling.

| object | range | script |
|---|---|---|
| the dictionary `D[a][v]`, the multiplicities, the row maxima `r(v)`, the deficits `d(v)`, the endpoint costs `c_p(v)`, `Leg(v)`, and the three named rows with counts and holes | old machines `{5}`, `{5,7}`, ..., `{5..23}` on full periods (7,952,175 gaps at m23) and `{5..29}` (214,708,725 gaps, built as 29 copies of the m23 period with gear 29's teeth removed) | `slr_row.py` |
| the same at `M = {5..31}` (period 33,426,748,355; 6,226,553,025 gaps), streamed in 3 processes, 114.8 s -- the NINTH rung, 31->37, out of the parent branch's sample | full period | `slr_m31.py` |
| the gear-5 pair filter (all 25 class pairs), the same for gears 7, 11, 13; the local contrast; the closers `Leg(a_L)` inside and outside `M`; the striker census of every `a_L`-gap; the striker map of the extremal cell | full periods to `{5..23}` (the census), all 9 rungs (the closers) | `slr_mech.py` |
| the residue-explained share of the deficit, the gear-5 cost of the extremal cell, `G(v) = v + r(v)` and its rank, the residual band | all 9 rungs | `slr_res.py` |
| scan-free LP-duality certificates for the row cells above `r(a_L)` | m19 (`v = 8`), m23 (`v = 10`), m29 (`v = 10`) | `slr_lp.py` |

**Instrument gates, all passed.** `F = 2, 5, 7, 11, 18, 25, 34, 43, 58` and
`F_2 = 4, 7, 11, 16, 25, 31, 39, 55, 68` at `M_0 .. M_8`, the recorded ladders
(`neighbour_profile.md` 1). `D = D^T` at all nine machines (the mirror `k -> P - k`), so the row
is side-blind and `r(v)` is one function, not two. `r(a_L) = 2, 0, 3, 7, 12, 20, 25, 35` at rungs
5->7 .. 29->31 -- exactly `availability_gate.md` 2.3's "max `a` adjacent to `a_L`". The m29 gap
array returns 214,708,725 gaps summing to 1,078,282,205; the streamed m31 pass returns
6,226,553,025 openings with `F = 58`, `F_2 = 68` (`neighbour_profile.md` 1, computed there by a
different chunking). The closed form `c_p(v) = 2 / 3 / 4` and the identification
`Leg(v) = {p : c_p(v) <= 3}` with the chain law's `v = 0, +-d_p (mod p)` agree at 1800 of 1800
`(gear, size)` cells (15 gears x 120 sizes), 0 mismatches. **S1 CONFIRMED.**

## 2. Results

### 2.1 The row, exact, at every rung (item 1)

| rung `q'` | `M` | `F` | `F_2` | `a_L` | `c_5(a_L)` | `Leg(a_L)` in `M` | `Leg(a_L)` outside `M` | `r(a_L)` | `r/F` | `r/F_2` | `F_2 - a_L` | `d(a_L)` | `a_L + r(a_L) - F` | mult(`a_L`) | holes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 7 | `{5}` | 2 | 4 | 2 | 3 | 5 | 7 | 2 | 1.000 | 0.500 | 2 | 0 | +2 | 2 | none |
| 11 | `{5,7}` | 5 | 7 | 4 | (4) | - | 11, 13 | **0** | 0.000 | - | 3 | - | - | 0 | (`a_L` unrealised) |
| 13 | `{5..11}` | 7 | 11 | 4 | 4 | 11 | 13 | 3 | 0.429 | 0.273 | 7 | 4 | +0 | 6 | 2 |
| 17 | `{5..13}` | 11 | 16 | 6 | 4 | - | 17, 19 | 7 | 0.636 | 0.438 | 10 | 3 | +2 | 60 | 1, 3, 4, 6 |
| 19 | `{5..17}` | 18 | 25 | 6 | 4 | 17 | 19 | 12 | 0.667 | 0.480 | 19 | 6 | +0 | 1,022 | 1, 3, 6, 8, 11 |
| 23 | `{5..19}` | 25 | 31 | 8 | 3 | 5 | 23 | 20 | 0.800 | 0.645 | 23 | 3 | +3 | 10,462 | 1, 6, 11, 16, 17 |
| 29 | `{5..23}` | 34 | 39 | 10 | 2 | 5 | 29, 31 | 25 | 0.735 | 0.641 | 29 | 4 | +1 | 243,370 | 17, 19 |
| 31 | `{5..29}` | 43 | 55 | 10 | 2 | 5, 29 | 31 | 35 | 0.814 | 0.636 | 45 | 8 | +2 | 7,815,766 | 31 |
| **37** | `{5..31}` | 58 | 68 | 12 | 3 | 5, 7 | 37 | **46** | **0.793** | 0.676 | 56 | 10 | **+0** | 109,884,182 | 4, 9, 14, 19, 24, 29, 34, 39, 44, 45 |

The last row is new: the parent branch stopped at rung 29->31; this branch computed `M = {5..31}`
on its full period.

**S2.** `r(a_L)/F = 1.000, 0.000, 0.429, 0.636, 0.667, 0.800, 0.735, 0.814, 0.793`. The brief's
`r(a_L) <= 0.85 F` holds at 8 of 9 rungs and is REFUTED at rung 5->7, where the machine `{5}` has
`F = 2` and its only two gap sizes are adjacent to each other. **Above the trivial rung the bound
holds, and it does not tighten**: the ratio over the top five rungs is
`0.667, 0.800, 0.735, 0.814, 0.793` -- flat, at four fifths.

**The occurrence counts of the short-letter row** (`R(a_L)` with `D[a][a_L] + D[a_L][a]`), which
is the object the residual statement is about, at the three rungs that matter:

    rung 19->23, R(8):  2:5684 3:4248 4:1312 5:2924 7:4486 8:344 9:4 10:1168 12:338 13:66
                        14:230 15:62 18:52 20:6
    rung 23->29, R(10): 1:85536 2:142166 3:19624 4:61648 5:65792 6:26018 7:25206 8:29044
                        9:1284 10:576 11:19692 12:250 13:5026 14:40 15:966 16:1660 18:1378
                        20:474 21:276 22:44 23:32 25:8
    rung 29->31, R(10): 1:2341104 2:4173988 3:716672 4:1923564 5:2192588 6:862988 7:1083668
                        8:884418 9:107106 10:62956 11:692752 12:42422 13:271798 14:8832
                        15:63408 16:84644 17:2694 18:65662 19:1220 20:23516 21:13000 22:3884
                        23:5996 24:68 25:1850 26:118 27:232 28:230 29:14 30:92 32:28 33:14
                        34:2 35:4

The top of the row is thin -- 6, 8 and 4 occurrences per period at the three rungs -- and the
counts are not monotone in `a`: at rung 29->31, `a = 11` occurs 692,752 times and `a = 10` only
62,956. That is the gear-5 rarity of the sizes `= +-1 (mod 5)` that `neighbour_profile.md` 2.5
records for the multiplicities, now seen inside one row (cited, not re-derived).

**The long letter and the padded letter.** `r(b_L) = 0, 0, 0, 5, 7, 13, 15, 27, 40` and
`r(q') = -, -, -, -, -, 5, 8, 14, 30` (both are often unrealised as gap sizes at the small
machines). Against `F`: `r(b_L)/F = 0.455, 0.389, 0.520, 0.441, 0.628, 0.690` at rungs 17..37 and
`r(q')/F = 0.200, 0.235, 0.326, 0.517` at rungs 23..37. The long letter's row is below the short
letter's at every rung, and the padded row below that -- as the free cap `F_2 - v` requires.

### 2.2 The profile `r(v)`, and what does not decide it

**`d(v) = 0` is rare and gets rarer.** The row reaches its free cap `min(F, F_2 - v)` at 2/2,
3/4, 4/7, 3/10, 6/17, 5/23, 4/33, 5/41, 6/55 of the realised sizes at `M_0 .. M_8`: a majority at
the smallest machines, 11% at m31. **S3's first clause is REFUTED** -- neither `F` nor `F_2 - v`
is what decides the row anywhere above m17.

**`r(v)` is not monotone** at 7 of 9 machines (S3's second clause CONFIRMED): at m29 it rises at
`v = 7, 11, 13, 16, 27, 31, 33`, at m31 at 13 sizes. The full profile at the deepest machine
reachable before the streamed one (`v : r(v) [free cap] {deficit}`):

    M = {5..29}, F = 43, F_2 = 55:
      1:39[43]{4} 2:43[43]{0} 3:40[43]{3} 4:38[43]{5} 5:38[43]{5} 6:34[43]{9} 7:40[43]{3}
      8:37[43]{6} 9:36[43]{7} 10:35[43]{8} 11:37[43]{6} 12:33[43]{10} 13:35[42]{7} 14:31[41]{10}
      15:35[40]{5} 16:32[39]{7} 17:30[38]{8} 18:30[37]{7} 19:28[36]{8} 20:35[35]{0} 21:27[34]{7}
      22:28[33]{5} 23:30[32]{2} 24:23[31]{8} 25:30[30]{0} 26:25[29]{4} 27:23[28]{5} 28:23[27]{4}
      29:15[26]{11} 30:25[25]{0} 31:14[24]{10} 32:16[23]{7} 33:20[22]{2} 34:10[21]{11}
      35:20[20]{0} 36:9[19]{10} 37:11[18]{7} 38:7[17]{10} 39:3[16]{13} 40:7[15]{8} 43:2[12]{10}

The pattern is visible without statistics: every multiple of 5 (`20:35, 25:30, 30:25, 35:20`)
stands above both its neighbours by 7 to 10, and the sizes `= +-1 (mod 5)` (`19:28, 21:27, 24:23,
29:15, 31:14, 34:10, 36:9, 39:3`) sit at the bottom.

### 2.3 The mechanism: gear 5 is the only gear that can forbid a neighbour (item 2; S4)

> **THE PAIR FILTER (proved here; exact and machine-independent).** Let `x_0 < x_1 < x_2` be three
> consecutive openings of any machine containing gear 5, `x_1 - x_0 = a`, `x_2 - x_1 = v`. For
> each gear `p`, `x_0` must avoid `T_p union (T_p - a) union (T_p - a - v)`, a set of at most
> 6 residues; so for every `p >= 7` some `x_0 (mod p)` survives for EVERY `(a, v)`, and only
> `p = 5` can exclude a class outright. Enumerating gear 5's 25 class pairs: exactly **6 of the
> 25 pairs `(a mod 5, v mod 5)` are impossible** -- `(1,1), (1,3), (2,4), (3,1), (4,2), (4,4)`.
> Equivalently, the number of classes `a mod 5` forbidden to a neighbour of a `v`-gap is
> **`c_5(v) - 2`**: none when `5 | v`, one when `v = +-2 (mod 5)`, two when `v = +-1 (mod 5)`.

Checked against every realised adjacent pair of every machine to `{5..23}`: 0 of 872 pairs
(3 + 9 + 25 + 52 + 133 + 221 + 429) lies in a forbidden class. It is also exactly the LP vehicle's
zeroth-order kill: `RelaxStar` marks a cell `dead` when some gear has no phase leaving all three
prescribed positions open, and at m19 the cell `(21, 8)` -- class `(1, 3)` -- is killed by that
clause alone, with no LP run at all.

**The letter's own arithmetic fixes its gear-5 class.** `3 a_L = q' -+ 1`, so `c_5(a_L) = 2` iff
`5 | a_L`, and `= 3` iff `5 | 3a_L -+ 1`, i.e. iff `5 | q' -+ 2` (never `5 | q'`). Measured
`c_5(a_L) = 3, 4, 4, 4, 4, 3, 2, 2, 3` at the nine rungs, and the row height separates on it
**9 of 9 with a gap**:

| `c_5(a_L)` | rungs | `r(a_L)/F` |
|---|---|---|
| 2 (`5` divides `a_L`) | 23->29, 29->31 | 0.735, 0.814 |
| 3 (`5` divides `q' -+ 2`) | 5->7, 19->23, 31->37 | 1.000, 0.800, **0.793** |
| 4 | 7->11, 11->13, 13->17, 17->19 | 0.000, 0.429, 0.636, 0.667 |

Every `c_5 = 4` rung is at or below 0.667; every `c_5 <= 3` rung is at or above 0.735.
**S4's cross-rung clause CONFIRMED 9 of 9**, and the ninth rung (31->37, `r = 46`, ratio 0.793)
was predicted before it was computed (S9, `r(12) >= 43`: CONFIRMED).

**The confound, and the trend-free test.** The `c_5 = 4` rungs are the small machines, so the
cross-rung table alone cannot separate "gear 5" from "machine size". The trend-free statistic is
the LOCAL CONTRAST `delta(v) = r(v) - (r(v-1) + r(v+1))/2`, which cancels any smooth decline:

| machine | `n`, mean `delta` at `c_5 = 2` | at `c_5 = 3` | at `c_5 = 4` | ordered? |
|---|---|---|---|---|
| `{5..11}` | 1, +2.00 | 2, +0.75 | 2, -1.50 | YES |
| `{5..13}` | 1, +2.50 | 3, +0.17 | 2, -1.00 | YES |
| `{5..17}` | 3, +0.67 | 6, +0.83 | 5, -1.10 | no (2 and 3 inverted) |
| `{5..19}` | 3, +4.83 | 8, -0.19 | 7, -2.21 | YES |
| `{5..23}` | 5, +1.70 | 13, +0.35 | 11, -1.36 | YES |
| `{5..29}` | 7, +5.50 | 16, +1.41 | 15, -4.07 | YES |
| `{5..31}` | 10, +3.45 | 21, +1.29 | 20, -3.02 | YES |

**6 of 7 machines strictly ordered, and the spread widens with the machine** (`+5.50` against
`-4.07` at m29). The stratification is gear 5's, not the machine's size. S4's within-machine
clause AS PRE-REGISTERED (median DEFICIT strictly ordered) is REFUTED -- it holds at only 3 of 7
(`{5..19}`, `{5..29}`, `{5..31}`), because the deficit mixes in the `v`-dependent cap; the local
contrast is the right statistic and it holds at 6 of 7.

**But the filter does not explain the HEIGHT.** Write `r5(v)` for the largest realised
`a <= min(F, F_2 - v)` in a gear-5-allowed class; `r(v) <= r5(v)` is proved. Splitting the
deficit:

| rung `q'` | 7 | 13 | 17 | 19 | 23 | 29 | 31 | 37 |
|---|---|---|---|---|---|---|---|---|
| free cap `min(F, F_2 - a_L)` | 2 | 7 | 10 | 18 | 23 | 29 | 43 | 56 |
| `r5(a_L)` (after the gear-5 filter) | 2 | 6 | 10 | 15 | 23 | 29 | 43 | 55 |
| `r(a_L)` (true) | 2 | 3 | 7 | 12 | 20 | 25 | 35 | 46 |
| residue-explained | 0 | 1 | 0 | 3 | 0 | 0 | 0 | 1 |
| **unexplained** | 0 | 3 | 3 | 3 | 3 | **4** | **8** | **9** |

The residue obstruction removes 0 to 3, and the unexplained residue GROWS (3, 3, 3, 3, 4, 8, 9).
So the answer to item 2's question -- glue lemma or residue obstruction? -- is **neither**. The
`F_2` cap (the glue lemma's shape, `neighbour_profile.md` 2.5, cited) is the free cap, which the
row misses by 3 to 10; the residue obstruction accounts for at most 3 of that; what is left is a
covering-capacity fact with no closed form on file.

What the filter DOES decide is **where the top of the row sits**. Write
`C_5(a, v) = |T_5 u (T_5 - a) u (T_5 - a - v)|` for the number of gear-5 classes the whole
three-opening configuration costs. The maximising `a` attains the MINIMUM of `C_5(., a_L)` over
the admissible `a` at **7 of 8 rungs** (all but 31->37), with values `4, 4, 4, 4, 3, 2, 2`, and
the extremal `a = 20, 25, 35` at the three rungs that matter are all `= 0 (mod 5)`. At rung
23->29 the extremal occurrence `(25, 10)` at `x_0 = 4,731,515` has
`x_0 = x_1 = x_2 = 0 (mod 5)`: **all three openings in one gear-5 class**, `C_5 = 2`, the whole
35-column window costing gear 5 a single residue. Its striker map (offset:strikers, `+` = several)

    left gap (25):  1:5+7+11+19  2:13  3:7+23  4:5  5:11  6:5+13  7:17  8:7  9:5  10:7  11:5
                    12:11  13:17  14:5+19  15:7+13  16:5+11  17:7  18:23  19:5+13  20:19  21:5
                    22:7  23:11  24:5+7+17
    right gap (10): 1:5+23  2:11  3:13  4:5+7  5:17  6:5+7  7:13  8:19  9:5+11

shows every gear of `M` used and gear 5 carrying 10 of the 33 blocked columns.

### 2.4 The closers of an `a_L`-gap, exactly (item 2; S5)

> **THE CLOSER LAW (proved here).** A gear `p` can strike both ends of a gap of size `v` in one
> copy iff `v = 0, +d_p, -d_p (mod p)` (file 05 (C)), i.e. iff `p | v (3v - 1)(3v + 1)`. For
> `v = a_L`, `3 a_L = q' -+ 1`, so `{3a_L - 1, 3a_L + 1} = {q', q' -+ 2}` exactly. Every prime
> factor of `q' -+ 2` other than `q' + 2` itself is smaller than `q'`, hence already a gear of
> `M`; and both ends of a gap of `M` are OPENINGS of `M`, which no gear of `M` strikes. Hence
> **the only gears that can close an `a_L`-gap of `M` are `q'` itself, and `q' + 2` when
> `q' = 2 (mod 3)` and `q' + 2` is prime.**

Measured at all nine rungs (`Leg(a_L)` split by membership of `M`):

| rung `q'` | `a_L` | `3a_L` | `{3a_L-1, 3a_L+1}` | `Leg(a_L)` | in `M` | OUTSIDE `M` |
|---|---|---|---|---|---|---|
| 7 | 2 | `q'-1` | 5, 7 | 5, 7 | 5 | **7** |
| 11 | 4 | `q'+1` | 11, 13 | 11, 13 | - | **11, 13** |
| 13 | 4 | `q'-1` | 11, 13 | 11, 13 | 11 | **13** |
| 17 | 6 | `q'+1` | 17, 19 | 17, 19 | - | **17, 19** |
| 19 | 6 | `q'-1` | 17, 19 | 17, 19 | 17 | **19** |
| 23 | 8 | `q'+1` | 23, 25 | 5, 23 | 5 | **23** |
| 29 | 10 | `q'+1` | 29, 31 | 5, 29, 31 | 5 | **29, 31** |
| 31 | 10 | `q'-1` | 29, 31 | 5, 29, 31 | 5, 29 | **31** |
| 37 | 12 | `q'-1` | 35, 37 | 5, 7, 37 | 5, 7 | **37** |

9 of 9, and the second outside closer `q' + 2` appears at exactly the three predicted rungs
(7->11 with 13, 13->17 with 19, 23->29 with 31 -- the rungs where `q' = 2 (mod 3)` and `q' + 2`
is prime). **S5 CONFIRMED.** The reading: an `a_L`-gap is rigid in the precise sense that the set
of gears able to carry it as an interior piece of a fusion is `{q'}` or `{q', q'+2}` and never
anything else. That is why the availability gate is a statement about the letter and not about a
general legal size.

**The striker census of the `a_L`-gaps** (full periods; which gears of `M` block the `a_L - 1`
interior columns of each `a_L`-gap):

| rung | `a_L` | `a_L`-gaps | blocked columns by gear (with multiplicity) | sole-striker columns by gear | 1 / 2 / 3+ strikers |
|---|---|---|---|---|---|
| 13->17 | 6 | 60 | 5:120 7:120 11:80 13:68 | 5:96 7:62 11:34 13:26 | 218 / 76 / 6 |
| 17->19 | 6 | 1,022 | 5:2044 7:2044 11:1316 13:1100 17:782 | 5:1344 7:900 11:448 13:344 17:182 | 3,218 / 1,620 / 272 |
| 19->23 | 8 | 10,462 | 5:31386 7:20924 11:16480 13:15432 17:10824 19:9812 | 5:15388 7:8728 11:7658 13:7216 17:3780 19:3352 | 46,122 / 22,822 / 4,290 |
| 23->29 | 10 | 243,370 | 5:973480 7:682298 11:486740 13:398788 17:309486 19:282202 23:216486 | 5:430296 7:303906 11:155326 13:138586 17:94498 19:84262 23:63782 | 1,270,656 / 707,054 / 212,620 |

At rung 23->29, gear 5 blocks exactly `4 = 2 x 2` of every 10-gap's 9 interior columns at every
single occurrence (973,480 = 4 x 243,370), which is forced: `10 = 0 (mod 5)` puts the two ends in
one gear-5 class, and each of gear 5's two teeth then strikes the interior exactly twice. That is
`c_5(a_L) = 2` seen from the inside, and it is why the letter 10 is a cheap gap to make.

### 2.5 The twin-rung structure decides the CLOSERS, not the ROW (item 4; S7)

The four rungs with `q' - 2` prime and in `M` (`q'` the upper member of a twin gear pair) are
5->7, 11->13, 17->19, 29->31; there `Leg(a_L) cap M` contains the lower twin and the only outside
closer is `q'`. The three rungs with `q' + 2` prime (7->11, 13->17, 23->29) have a second outside
closer, which is not a gear of `M`. Row heights:

| | twin below (`q'-2` prime, in `M`) | twin above (`q'+2` prime) | neither |
|---|---|---|---|
| rungs | 5->7, 11->13, 17->19, 29->31 | 7->11, 13->17, 23->29 | 19->23, 31->37 |
| `r(a_L)/F` | 1.000, 0.429, 0.667, 0.814 | 0.000, 0.636, 0.735 | 0.800, 0.793 |

**No separation whatsoever** -- both groups span nearly the whole range, and the highest row on
record (29->31, 0.814) is a twin rung. The mechanism is clean and negative: the lower twin
`q' - 2` enters `Leg(a_L)` as a BIG gear, and a big gear in `Leg` changes its endpoint cost from
4 to 3 out of `p >= 11` classes -- one part in eleven or more, against gear 5's one part in five.
**S7 CONFIRMED in its content**: the two rungs whose only `Leg cap M` member is a big gear
(11->13 with gear 11, 17->19 with gear 17) sit at 0.429 and 0.667, below every rung with
`5 in Leg(a_L)`; and the twin structure itself is irrelevant to the row.

### 2.6 The rows are not initial segments, and the holes are gear 5's (S6, REFUTED)

Realised sizes below `r(a_L)` that are absent from the row: none, `{2}`, `{1,3,4,6}`,
`{1,3,6,8,11}`, `{1,6,11,16,17}`, `{17,19}`, `{31}`, `{4,9,14,19,24,29,34,39,44,45}` at the eight
rungs where `a_L` is realised -- 3 or more holes at 4 of 8, so **S6 is REFUTED**. The structure of
the holes is the pair filter: at rung 31->37 (`a_L = 12 = 2 (mod 5)`, forbidden neighbour class
`a = 4 (mod 5)`) the holes are exactly `4, 9, 14, 19, 24, 29, 34, 39, 44` -- every realised size
of that class below the row maximum -- plus one extra, 45. At rung 19->23 (`a_L = 8 = 3 (mod 5)`,
forbidden class `a = 1 (mod 5)`) the holes are `1, 6, 11, 16` plus one extra, 17. At rungs 23->29
and 29->31 (`5 | a_L`, no forbidden class at all) there are 2 and 1 holes. Rung by rung, holes
predicted by the filter (every realised size below `r(a_L)` in a forbidden class) against holes
observed:

| rung `q'` | 7 | 13 | 17 | 19 | 23 | 29 | 31 | 37 |
|---|---|---|---|---|---|---|---|---|
| `a_L mod 5` | 2 | 4 | 1 | 1 | 3 | 0 | 0 | 2 |
| forbidden classes | `{4}` | `{2,4}` | `{1,3}` | `{1,3}` | `{1}` | none | none | `{4}` |
| holes the filter predicts | 0 | 1 (`2`) | 3 (`1,3,6`) | 5 (`1,3,6,8,11`) | 4 (`1,6,11,16`) | 0 | 0 | 9 (`4..44`) |
| holes observed | 0 | 1 | 4 | 5 | 5 | 2 | 1 | 10 |
| extras | 0 | 0 | 1 (`4`) | **0** | 1 (`17`) | 2 (`17,19`) | 1 (`31`) | 1 (`45`) |

**Every hole the filter predicts is observed, 8 of 8 rungs, and there are 0 to 2 extras per rung.**
So the row's holes ARE the gear-5 filter up to at most two sizes, and their count is governed by
`c_5(a_L)` exactly as the height is.

### 2.7 What is special about the letter: its largest 2-run is pinned at `F` (S10)

Write `G(v) := v + r(v)`, the longest span of two adjacent gaps one of which has size `v`;
`G(v) <= F_2(M)` by definition. Measured:

| rung `q'` | 7 | 13 | 17 | 19 | 23 | 29 | 31 | 37 |
|---|---|---|---|---|---|---|---|---|
| `F` | 2 | 7 | 11 | 18 | 25 | 34 | 43 | 58 |
| `F_2` | 4 | 11 | 16 | 25 | 31 | 39 | 55 | 68 |
| `G(a_L) - F` | +2 | 0 | +2 | 0 | +3 | +1 | +2 | **0** |
| `G(b_L) - F` | - | - | +5 | +2 | +3 | 0 | +5 | +7 |
| `G(q') - F` | - | - | - | - | +3 | +3 | +2 | +9 |
| rank of `G(a_L)` among realised `v` | 2/2 | 1/7 | 3/10 | 2/17 | 12/23 | 6/33 | 9/41 | 10/55 |
| percentile | 100% | 14% | 30% | 12% | 52% | 18% | 22% | 18% |
| `#v` with `G(v)` in `[F, F+3]` | 2/2 | 5/7 | 6/10 | 6/17 | 14/23 | 28/33 | 14/41 | 18/55 |

> **THE PINNED LETTER (measured; 8 of 8 rungs, one of them out of sample).**
> `F(M) <= a_L + r(a_L) <= F(M) + 3`, i.e. `r(a_L) <= F(M) - a_L + 3`.

This is the branch's strongest exact statement about the row, and it is much better than the
proved pair cap: at rung 29->31 it gives `r <= 36` against the pair cap's vacuous `45 > 43 = F`,
and at rung 31->37 it gives `r <= 49` against `56`. It is not a property of a general size --
`G(v) > F + 3` at 23 of 41 sizes at m29 and 28 of 55 at m31, and `G` reaches `F_2` at
`v = 20, 25, 30, 35` at m29 -- and `G(a_L)` sits at or below the 22nd percentile of the `G`
distribution at
5 of 8 rungs. The prediction `r(12) <= 49` at rung 31->37 was written down before the m31 run
returned; the answer is `r(12) = 46`, `G = 58 = F` exactly. **S10 CONFIRMED, including out of
sample.**

The importance of the form: `r(a_L) <= F - a_L + 3` is a bound BELOW `F(M)` as soon as
`a_L > 3`, i.e. from `q' >= 11` on, with

    c  =  1 - (a_L - 3)/F(M)  =  0.857, 0.727, 0.833, 0.800, 0.794, 0.837, 0.845
                                 at rungs 11->13, 13->17, 17->19, 19->23, 23->29, 29->31, 31->37,

which is exactly the shape the parent branch asked for and could not get from `F_2 - a_L`. It is
measured, not proved.

## 3. Mechanism, assembled

The row asks a covering question with three prescribed openings. For `(a, v)` to be an adjacent
pair of `M` there must be a column `x_0` with `x_0`, `x_0 + a`, `x_0 + a + v` all open and every
column between them blocked. Two independent things can stop it.

1. **The three openings themselves.** Gear `p` forbids `x_0` in `T_p u (T_p - a) u (T_p - a - v)`,
   at most 6 classes, so only `p = 5` can forbid a class outright: the PAIR FILTER, six impossible
   `(a, v)` classes mod 5, and `c_5(v) - 2` forbidden neighbour classes for a `v`-gap. This is
   exact, machine-independent, and it is the whole of the row's residue content at the pair level.
2. **The blocked columns.** `a + v - 2` columns must be covered by the gears of `M` avoiding three
   fixed positions. That is the same covering problem that defines `F` and `F_2`; it has no closed
   form on file, and it is what the missing 3 to 9 of the deficit is made of.

The letter enters through (1) only, and it enters completely: `3 a_L = q' -+ 1` fixes
`c_5(a_L)` from `q' mod 5` and `q' mod 3` alone, `c_5(a_L) = 2` iff `5 | a_L`, `= 3` iff
`5 | q' -+ 2`. The measured row height, the number of holes in the row, and the local contrast all
stratify on that one number, and the twin structure of `q'` -- which decides the CLOSERS of an
`a_L`-gap by the closer law -- does not enter the row at all.

The closer law says what an `a_L`-gap is FOR: it is the unique interior piece the incoming gear
can carry, and no other gear outside `M` can carry it except `q' + 2` at the three rungs where
that is prime. So the availability gate really is a question about one size and one gear, as the
parent branch found; but the reason the row STOPS where it does is not about that size's
arithmetic, it is about how much blocking capacity `M` has left after making a gap of size `a`.

## 4. The proof attempt (item 3)

### 4.1 The statement ladder, from weakest to strongest

| statement | status | value at rung 29->31 (`F = 43`, `a_L = 10`) |
|---|---|---|
| `r(a_L) <= F(M)` | trivial | 43 -- vacuous |
| `r(a_L) <= F_2(M) - a_L` | **proved** (`availability_gate.md` 3.2, cited) | 45 -- VACUOUS |
| `r(a_L) <= r5(a_L)` (the pair filter) | **proved here** | 43 -- vacuous at this rung (`5 | a_L`) |
| `r(a_L) <= F(M) - a_L + 3` (the pinned letter) | measured, 8 of 8, one out of sample | **36** |
| `r(a_L) = 35` | **certified scan-free by LP duality at this very rung** (4.4) | **35** |

### 4.2 Why a machine-independent per-letter computation cannot work

The brief asked whether the `a_L`-gap's residue configurations can be enumerated by CRT on the
small gears, so that the row's emptiness above `c F` becomes a computation per LETTER rather than
per machine. The answer is no, and the reason is one line of counting.

> Fix any finite set `S` of gears and any `(a, v)`. The number of admissible `x_0 (mod prod S)` is
> `prod_{p in S} (p - C_p(a, v))` with `C_p(a, v) = |T_p u (T_p - a) u (T_p - a - v)| <= 6`. This
> vanishes only if some `p in S` has `C_p = p`, which needs `p <= 6`, i.e. `p = 5`. So for every
> `S` and every `(a, v)` not killed by gear 5 alone, admissible configurations exist mod `prod S`,
> for every `S`, however large.

A per-letter enumeration therefore decides exactly the pair filter and nothing more, and the pair
filter accounts for at most 3 of a deficit that runs 3, 3, 3, 3, 4, 8, 9 and grows. The row's
emptiness above `c F` is NOT a computation per letter size; it needs the machine's blocking
capacity, and that is what makes it a per-machine statement.

### 4.3 What the LP vehicle does deliver: the row certified scan-free

The row statement is a family of level-2 dictionary cells, and the LP lane's windowed vehicle
decides such a cell by duality (`docs/novel/restricted-covering-certificates.md` RESULT 4,
`research/window_dict.py`; `two_gap_geometry(W, a)` prescribes the three openings `0, a, W`,
`decide_star` returns an exact rational Farkas certificate or an exact in-polytope witness). This
branch ran it on the row cells above the measured row maximum, escalating the number of held
gears until each cell certified.

| machine | row `v = a_L` | cells `a` | verdict | held gears used | exact operations | tightness check at `a = r(a_L)` |
|---|---|---|---|---|---|---|
| `{5..19}` | 8 | 21..25 (5 cells, spans 29..33) | **5 of 5 CERTIFIED** | 0 for all five; `a = 21` killed outright by the gear-5 pair filter | 14,134 | `a = 20` REFUTED (exact in-polytope witness) at `k = 0, 1, 2` -- correct, it is realised |
| `{5..23}` | 10 | 26..34 (9 cells, spans 36..44) | **9 of 9 CERTIFIED** | 0 at six cells, 1 at `a = 28, 30, 32`, 2 at `a = 27` | 79,483 | `a = 25` STUCK at `k = 0`, REFUTED at `k = 1` and `k = 2` -- correct, it is realised |

**14 of 14 cells certified, 0 unsound (every certified cell is genuinely unrealised, against the
full-period dictionary), and the vehicle refuses at exactly the realised top of each row.
S8 CONFIRMED.** In words: *the row `v = 8` of `Dict_2({5..19})` is empty above 20* and *the row
`v = 10` of `Dict_2({5..23})` is empty above 25* now have scan-free proofs, in 14,134 and 79,483
exact rational operations against period scans of 1,616,615 and 7,952,175 columns.

### 4.4 The rung that matters, certified: the row of `{5..29}` is empty above 35

The parent branch's deciding negative was that the proved cap `F_2 - a_L = 45` exceeds
`F(m29) = 43` and therefore closes nothing at rung 29->31. The same vehicle, pushed one machine
further than the LP lane had taken it, closes it outright:

| cell `a` | span | verdict | gears held | cases vacuous | exact operations | time |
|---|---|---|---|---|---|---|
| 35 | 45 | **REFUTED** at `k = 2` (STUCK at `k = 0, 1`) -- correct, it is realised | - | - | - | 114 s |
| 36 | 46 | CERTIFIED | 0 | 0 | 5,542 | 0.1 s |
| 37 | 47 | CERTIFIED | 3 (`5, 7, 11`) | 349 of 385 | 150,434 | 2.5 s |
| 38 | 48 | CERTIFIED | 1 | 3 of 5 | 12,345 | 0.8 s |
| 39 | 49 | CERTIFIED | 2 | 32 of 35 | 21,273 | 1.8 s |
| 40 | 50 | CERTIFIED | 1 | 2 of 5 | 27,589 | 3.1 s |
| 41 | 51 | CERTIFIED | 0 | 0 | 6,366 | 0.2 s |
| 42 | 52 | CERTIFIED | 2 | 29 of 35 | 32,490 | 0.7 s |
| 43 | 53 | CERTIFIED | 0 | 0 | 14,031 | 3.8 s |

**8 of 8 cells certified, 270,070 exact rational operations in total**, against a period of
1,078,282,205 columns; and the vehicle refuses at `a = 35`, the realised top of the row, so it is
exact there too. Only `a = 37` needed a third held gear; the pattern of the LP lane's own ladder
(more held gears, primorial cost) reappears inside one row.

Since the same vehicle certifies `F(19) <= 25`, `F(23) <= 34` and `F(29) <= 43` outright at
`k = 2` (`restricted-covering-certificates.md` RESULT 3's case split, cited), there are no cells
above `F(M)` left to check, and the three statements are complete:

> **CERTIFIED SCAN-FREE.** The row `v = 8` of `Dict_2({5..19})` is empty above `a = 20`; the row
> `v = 10` of `Dict_2({5..23})` is empty above `a = 25`; the row `v = 10` of `Dict_2({5..29})` is
> empty above `a = 35`. Equivalently `a_hasM = 20, 25, 35` at rungs 19->23, 23->29, 29->31 by
> exact rational certificate, with no period of any machine scanned.

That is the parent branch's residual statement, PROVED at the three rungs that carry the budget's
tightness -- with `c = 0.800, 0.735, 0.814` -- by a vehicle whose cost is a primorial in the
number of held gears, not a period. It is not a proof for all `M`: the certificates are
per-machine objects, and section 4.2 says no per-letter enumeration can replace them.

## 5. Toward the root, and the residual band (item 5)

If `r(a_L) <= c F(M)` with `c < 1` were PROVED for all rungs, the parent's gate ladder
(`availability_gate.md` 2.4, proved) would give: above `c F` the largest piece of a fusion admits
no `J >= 3` word, so the frontier there is `Rest_2` and the obligation is the PAIR statement
`F_2(M) <= F(M) + q'`, whose slack at the top rungs is `12, 17, 24, 19`; between `c F` and the
deep-chain cap `(F + q')/J_max` the CHAIN statement remains.

**The residual band at rung 29->31, exactly** (`M = {5..29}`, `F = 43`, `q' = 31`, budget 74):

| top of the gate | value | band (`J_max = 5`, measured) | realised sizes | band (`J_max = 6`, universal) | realised sizes |
|---|---|---|---|---|---|
| proved pair cap `F_2 - a_L` | 45 (vacuous, `> F`) | `[15, 43]` | 27 | `[13, 43]` | 29 |
| the pinned letter `F - a_L + 3` | 36 | `[15, 36]` | 22 | `[13, 36]` | 24 |
| **the row, CERTIFIED scan-free (4.4)** | **35** | **`[15, 35]`** | **21** | **`[13, 35]`** | **23** |

The band's minimum slack equals the global budget slack, 16, and it contains both record
maximisers `a = 25` and `a = 30` (`availability_gate.md` 2.5, cited). Two readings:

- **What is now proved at this rung.** The certificates of 4.4 put the top of the gate at 35 by
  exact rational arithmetic, so the band `[15, 35]` -- the parent's own band, previously resting
  on a full-period scan of 1,078,282,205 columns -- is a certified object, and everything above
  it is discharged by the pair statement with slack 19. The proved-for-all-`M` cap
  `F_2 - a_L = 45` closes nothing here; the certificate closes 8 sizes (`36..43`, of which
  `36, 37, 38, 39, 40, 43` are realised).
- **What a proof of the pinned letter would add.** `F - a_L + 3 = 36` is one above the truth and
  five below the vacuous cap; it would close the same top of the spectrum from a formula in `F`
  and `a_L` alone, at every rung and not one at a time. That is where the parent branch's TOP LAW
  lives.

**Does the row's mechanism give anything on the band? No, and the reason is sharp.** The only
machine-independent part of the mechanism is the gear-5 pair filter, and at the rung that matters
`a_L = 10 = 0 (mod 5)`, so `c_5(a_L) = 2` and the filter forbids NO neighbour class at all. The
rung whose band matters most is exactly the rung where the residue mechanism is empty. That is the
same shape of negative the parent branch recorded for `F_2 - a_L` (vacuous at 29->31) and
`neighbour_profile.md` recorded for `F_2 - F <= a_L` (fails at m17 and m29): three different
routes, all silent at the same rung.

## 6. What is new

1. **THE PAIR FILTER**, proved and machine-independent: for any machine containing gear 5, exactly
   6 of the 25 classes `(a mod 5, v mod 5)` of an adjacent gap pair are impossible, and the number
   of neighbour classes forbidden to a `v`-gap is `c_5(v) - 2`; **no gear `p >= 7` can forbid a
   class at all**, because three translates of a 2-element tooth set cover at most 6 residues.
   0 of 872 realised pairs violate it. It is also exactly the LP vehicle's zeroth-order `dead`
   clause, which gives the vehicle's cheapest kills a closed form.
2. **The letter's gear-5 class is `q' mod 5` and `q' mod 3`**: `3 a_L = q' -+ 1` gives
   `c_5(a_L) = 2` iff `5 | a_L` and `= 3` iff `5 | q' -+ 2`, measured `3, 4, 4, 4, 4, 3, 2, 2, 3`
   at nine rungs -- and **the row height separates on it 9 of 9** (`c_5 = 4` at or below 0.667 of
   `F`, `c_5 <= 3` at or above 0.735), with the ninth value predicted before it was computed.
3. **The local contrast `delta(v)`**, a trend-free confirmation that the stratification is gear
   5's and not the machine's size: ordered `c_5 = 2 > 3 > 4` at 6 of 7 machines, spread widening
   to `+5.50` against `-4.07` at m29.
4. **THE CLOSER LAW**, proved: `{3a_L - 1, 3a_L + 1} = {q', q' -+ 2}`, so the only gears that can
   close an `a_L`-gap of `M` are `q'` and, when `q' = 2 (mod 3)` and `q' + 2` is prime, `q' + 2`.
   9 of 9 rungs. The `a_L`-gap is rigid in exactly this sense.
5. **THE PINNED LETTER**, measured 8 of 8 including one out-of-sample rung:
   `F(M) <= a_L + r(a_L) <= F(M) + 3`. It is the first bound on the row that is strictly below
   `F(M)` (`c = 1 - (a_L - 3)/F`), it is 9 better than the proved pair cap at the rung that
   matters, and it is not a property of a general gap size (`v + r(v)` exceeds `F + 3` at 23 of 41
   sizes at m29 and reaches `F_2` at four of them).
6. **The row's holes are the pair filter**: at rung 31->37 the holes below the row maximum are
   exactly the realised sizes `= 4 (mod 5)` (`4, 9, 14, ..., 44`) plus one, and at rungs where
   `5 | a_L` there are 1 or 2 holes only.
7. **The ninth rung, 31->37**, computed on the full 33,426,748,355-column period of `{5..31}` in
   114.8 s: `r(12) = 46`, `mult(12) = 109,884,182`, `r(b_L = 25) = 40`, `r(37) = 30`, and the whole
   `r(v)` profile. The parent branch's sample ended at rung 29->31.
8. **Scan-free certificates for the row statement at the three rungs that matter**: 14 of 14 cells
   at m19 and m23, and **8 of 8 at m29** -- the rung where the parent's proved cap is vacuous --
   for 14,134 + 79,483 + 270,070 exact rational operations, with the vehicle refusing at exactly
   the realised top of each row. The parent's residual "the row is empty above `c F`" is therefore
   PROVED, per machine, with `c = 0.800, 0.735, 0.814`. Only one cell of the 22 needed a third
   held gear.
9. **The negative that decides the branch**: the residue obstruction explains 0 to 3 of a deficit
   that runs 3, 3, 3, 3, 4, 8, 9; no finite per-letter CRT enumeration can do better, because
   `prod (p - C_p) > 0` for every gear set once gear 5 is passed. The row's height is a covering
   capacity, not an arithmetic of the letter.

Prior art inside the project: the row `max { a : (a, a_L) in Dict_2 }` and its values to rung
29->31 are the parent's (`availability_gate.md` 2.3); the level-2 dictionary and its windowed LP
decision are the LP lane's (`restricted-covering-certificates.md` RESULT 4,
`research/window_dict.py`, `research/star_case.py`); `N(v) <= F_2` and the glue lemma are 2g.i;
the gear-5 weight `w(s mod 5) = 3,1,2,2,1` of the MULTIPLICITIES is `neighbour_profile.md` 2.5 --
the pair filter here is the same arithmetic applied to the two-gap configuration and to the row
rather than to the counts, and the "only gear 5 can forbid" half is not on record; the divisor law
`q | 3 q_1 - 1` is `alignment-rules-index.md` L20; the letters and the chain law are file 05.
Outside the project: not checked (no web access).

## 7. Verdict

**FACT, exact, with two proved machine-independent laws and one measured bound of the right shape;
the residual statement is now CERTIFIED at three machines and still unproved in general.**

- The row is completely mapped at nine rungs, one of them new. `r(a_L)/F` runs
  `1.000, 0.000, 0.429, 0.636, 0.667, 0.800, 0.735, 0.814, 0.793`: **flat at about four fifths
  over the top five rungs, with no downward trend**. So the parent's residual "the row is empty
  above `c F` for some `c < 1`" is TRUE at every rung on record with `c = 0.85`, and there is no
  sign in the data of a `c` that improves as the machine grows.
- The strongest exact statement the record supports is **`F <= a_L + r(a_L) <= F + 3`** (8 of 8,
  one out of sample), i.e. `r(a_L) <= F - a_L + 3`. It is the first bound of the shape the root
  needs -- strictly below `F(M)`, from `M` and `q'` alone, with no measurement of the new machine
  -- and it is 9 below the proved pair cap at rung 29->31, where that cap says nothing.
  **It is measured, not proved, and proving it is the single thing this branch hands forward.**
- The mechanism is decided and is half arithmetic, half capacity. The arithmetic half is complete
  and proved: only gear 5 can forbid a neighbour class, it forbids `c_5(v) - 2` of them, the
  letter's class is `q' mod 5` via `3a_L = q' -+ 1`, and the closers of an `a_L`-gap are `{q'}` or
  `{q', q'+2}`. The capacity half is untouched and it is the larger half, growing 3, 3, 3, 3, 4,
  8, 9 across the rungs.
- **A machine-independent per-letter enumeration is impossible** (4.2), so the row is decided
  per machine -- and per machine it is decided cheaply. The LP vehicle certifies every cell above
  the row maximum at m19, m23 **and m29** in 14,134, 79,483 and 270,070 exact rational operations,
  refusing exactly at the realised top of each row. So at the three rungs that carry the budget's
  tightness the parent's residual statement is no longer a measurement: `a_hasM = 20, 25, 35` is
  a certificate, and at rung 29->31 that is a bound the proved pair cap could not give.

**CANDIDATE.** `F(M) <= a_L + r(a_L) <= F(M) + 3` -- the pinned letter. What would have to break
it: a machine `M` and an incoming gear `q'` with an adjacent pair `(a, a_L)` of span more than
`F(M) + 3`. Why the system may not be able to do that: not yet shown. The pair filter forbids it
only when `c_5(a_L) >= 3` and only for one or two residue classes; the `F_2` cap allows spans up
to `F_2(M)`, which exceeds `F + 3` at 23 of 41 sizes at m29. So the mechanism is NOT in hand and
the law is on 8 rungs, one out of sample. Its natural first test is a counterfactual family
(tooth-shifted machines at m17, m19, m23 as in `availability_gate.md` 2.6): if the pinned letter
survives 60 family members it is structural; if it breaks it is an accident of the real teeth.

**Children this branch opens.**
1. *The pinned letter on the counterfactual family* -- the cheapest decisive test, and the one
   that says whether to look for a proof at all.
2. *The certificate as a family, not as instances* -- 22 cells at three machines certify with 0 to
   3 held gears; the question is whether the dual weights at the cell `(a, a_L)` have a form in
   `a` and `a_L` that survives the machine, which would turn the per-machine certificates into a
   per-rung lemma. The LP lane's own ladder (`restricted-covering-certificates.md` RESULT 3) is
   the precedent: nine rungs certified by the same construct at increasing `k`.

## 8. Dead ends, each with its refuting instance

- **`r(a_L) <= 0.85 F` as a uniform statement.** Refuted at rung 5->7: `r = F = 2`. (It holds at
  the other 8 of 9 rungs.)
- **The row reaches its free cap `min(F, F_2 - v)`.** Refuted: 5 of 41 sizes at m29 and 6 of 55 at
  m31 reach it; `d(a_L) = 3, 4, 8, 10` at the top four rungs.
- **The rows of `Dict_2` are initial segments.** Refuted at 4 of 8 rungs (10 holes at rung 31->37).
- **The deficit `d(v)` is ordered by `c_5(v)`.** Refuted as a median statement (holds at 3 of 7
  machines); what survives is the local contrast, 6 of 7.
- **The twin structure of `q'` predicts the row height.** Refuted: twin rungs give
  `1.000, 0.429, 0.667, 0.814`, non-twin rungs `0.000, 0.636, 0.735, 0.800, 0.793` -- the two
  groups interleave completely. What the twin structure does decide is the closer set, exactly.
- **A residue obstruction specific to `a_L` explains the row.** Refuted: the residue-explained
  share of the deficit is `0, 1, 0, 3, 0, 0, 0, 1` against unexplained `0, 3, 3, 3, 3, 4, 8, 9`,
  and at the two rungs that matter most (`5 | a_L`) the filter forbids nothing at all.
- **The row's emptiness is a finite computation per letter size.** Refuted by the counting
  argument of 4.2: `prod_{p in S} (p - C_p) > 0` for every gear set `S` and every pair not already
  killed by gear 5.
- **`r(a_L) <= F_2 - a_L` as a useful cap.** Already recorded dead by the parent
  (vacuous at rung 29->31); this branch adds that it is vacuous at rung 31->37 too in the sense
  that it exceeds the measured row by 10 (`56` against `46`).

## 9. What holds without exception (item 6)

| statement | count | status |
|---|---|---|
| the pair filter: exactly 6 of 25 classes `(a mod 5, v mod 5)` are impossible for an adjacent pair; no gear `p >= 7` forbids any class | proved; 0 of 872 realised pairs violate it (7 machines) | proved here + verified |
| the number of neighbour classes forbidden to a `v`-gap is `c_5(v) - 2` | 25 of 25 class pairs, exhaustive | proved here |
| `c_p(v) = 2 / 3 / 4` according as `p | v` / `p | 3v -+ 1` / otherwise, and `Leg(v) = {p : c_p(v) <= 3}` is the chain law's "both ends struck" set | 1800 of 1800 `(gear, size)` cells | proved (file 05 (C)) + verified |
| the closer law: the only gears outside `M` that can close an `a_L`-gap are `q'`, and `q' + 2` when `q' = 2 (mod 3)` and `q' + 2` is prime | 9 of 9 rungs | proved here + verified |
| `D = D^T` (the dictionary is mirror-symmetric, so the row is side-blind) | 9 of 9 machines | mirror `k -> P - k` + verified |
| `r(a_L)/F <= 0.667` when `c_5(a_L) = 4` and `>= 0.735` when `c_5(a_L) <= 3` | 9 of 9 rungs | measured |
| `F(M) <= a_L + r(a_L) <= F(M) + 3` (the pinned letter) | 8 of 8 rungs where `a_L` is realised, one out of sample | measured |
| `r(a_L) <= F_2(M) - a_L` | 9 of 9 rungs | proved (parent 3.2), cited |
| every certified LP cell of the short-letter row is genuinely unrealised, and the vehicle refuses at `a = r(a_L)` | 22 of 22 cells + 3 tightness checks at m19, m23, m29 | exact rational certificates |
| gear 5 blocks exactly 4 of the 9 interior columns of every 10-gap of `{5..23}` | 243,370 of 243,370 gaps | forced by `10 = 0 (mod 5)` + verified |
