# Node 4.i.a.i.a.1.a.i - THE RECORD GAP AS A 2-RUN: the top of the spectrum as pairs

Parent: node **4.i.a.i.a.1.a, the pinned letter's arithmetic**
(`research/proof/pinned_arithmetic.md`, FACT with one refutation, 2026-09-06), whose closing line
names this branch exactly:

> "The next child branch should be about that comparison: **the record gap itself as a 2-run** --
> which sizes sit at the two ends of the record, whether the record's decomposition ever contains
> an `a_L`-gap, and why the record's jump to 88 is a `J = 4` fusion while the letter's row is stuck
> at 2."

What spawned this branch: the parent refuted the pinned letter's lower half out of sample
(`a_L + r(a_L) = 77` against `F({5..37}) = 88`) and diagnosed the reason as a comparison of two
covering optima that grow by different mechanisms. The record gap of `M` is itself a gap with two
neighbours; read as a 2-run `(F, n1(F))` its sum is `F + n1(F)`, and the frontier's top law
(`frontier_collapse.md` 2.4) says that is exactly what the old record becomes at the next rung when
`F` is not a letter. The measured largest neighbours of the record are tiny -- `2, 2, 3, 5, 7, 5,
5, 2` at `{5,7}` .. `{5..31}` -- so the suppression at the top is exact and extreme. The object of
this branch is the **profile `n1(v)`**, the largest single neighbour of a gap of size `v`, and the
**sum profile `v + n1(v)`**, whose maximum over `v` is `F_2(M)` by definition.

Scripts in `research/anchor235/r64/`; result outputs in `research/anchor235/r64/results/`
(untracked). Every number this document relies on is written into the document.

---

## 0. Pre-registered (written before any computation of this branch)

### 0.1 The objects, defined exactly

Machines `M = {5..y}` for `y = 11, 13, 17, 19, 23, 29, 31` (and `{5,7}`, `{5}` where they exist),
written **m11 .. m31**. A **column** `k` is the pair `(6k-1, 6k+1)`; gear `g` strikes `k` iff
`k = +-u_g (mod g)`, `u_g = 6^{-1} (mod g)`. A **gap** is the distance between consecutive
openings, cyclic over the period; `F(M)` is the largest gap; `m(v)` is the number of gaps of size
`v` per period.

- **the neighbour profile** `n1(v) := max over gaps of size v of max(left neighbour, right
  neighbour)`. This is the same object as the adjacent-pair row top `r(v)` of
  `short_letter_row.md`: `n1(v) = r(v)` by definition of the row.
- **the neighbour-sum profile** `N(v) := max over gaps of size v of (left + right)` (branch 2g.i).
- **the sum profile** `Sig(v) := v + n1(v)`. By definition of `F_2(M)` (the largest sum of two gaps
  sharing an opening), `max_v Sig(v) = F_2(M)` exactly; call a maximiser `v*` and the pair
  `(v*, n1(v*))` the **`F_2` pair**.
- **the top band** `T(M) := { realised v : v >= 0.8 F(M) }`; the **top-band deficit**
  `D_top(M) := F_2(M) - max_{v in T} Sig(v)`.
- **the record's 2-run**: `(F, n1(F))`, sum `F + n1(F)`; the **record deficit**
  `D_rec(M) := F_2(M) - F - n1(F)`.
- **isolation**: the `k`-th largest realised size `s_k`; `iso_k := s_k - s_{k+1}`.
- **the lineage**: by the merge law every gap of `M + q'` is a run `(p_1, ..., p_J)` of consecutive
  gaps of `M` (its fusion word). The record's lineage at a rung is the fusion word of `F(M + q')`;
  the `F_2` lineage is the fusion word pair of `F_2(M + q')`. The **rank fraction** of a piece `p`
  is `|{realised sizes >= p}| / |{realised sizes}|` (0 = the record, 1 = the smallest size).
- **the frontier** `Rest(a)` and the budget slack `s(a) = F(M) + q' - a - Rest(a)`, from
  `frontier_collapse.md`; `a*` is the minimiser of `s`, i.e. the size at which the record of
  `M + q'` is made.

### 0.2 The theory

**T. The top of the spectrum is pinned as PAIRS, not as sizes, and the pinning is a repulsion:
a gap can be long, or it can have a long neighbour, but the two ends of a long gap are bought from
gears that are then unavailable to close a long neighbour. The profile `n1(v)` should therefore
fall roughly like `C - v` near the top with `C` between `F` and `F_2`, and the record -- the
extreme point of the spectrum -- should have the shortest neighbours of any size. The consequence
for the root is negative and worth stating in advance: since `n1(F) <= 7` at every machine on
record and `q' >= 13` at every rung, the budget inequality read at the record's own 2-run,
`F + n1(F) <= F + q'`, is vacuous by a margin of at least 6. The budget's tightness is carried by
an interior maximiser `a*` at `0.58-0.70 F`, and this branch's job is to say exactly which 2-runs
and 3-runs those are and where their pieces sit in the top-of-spectrum ranking.**

Corollary theory for the growth mechanism: **the record of `M + q'` never uses the top of `M`'s
spectrum from rung 23 on** (R3.h: records are ordinary lower gaps fused at junctions). The branch
tests this to rung 37 and asks the same of `F_2(M + q')`.

### 0.3 Disclosure of what had already been read

Read before writing these predictions: `pinned_arithmetic.md` (all), `frontier_collapse.md`
sections 0-2.1, `neighbour_profile.md` sections 0-2.3 (including the `N(v)` profiles at m29 and
m31 and the attaining `Q*_J` words), `theory_tree.md` nodes 4.i and children and R3.h's summary,
`pinned_letter.md` sections 3.1 and 4.3, `docs/novel/README.md` index lines. So the following are
**post-hoc `[read]`** and count only as instrument checks: `F = 7, 11, 18, 25, 34, 43, 58` and
`F_2 = 11, 16, 25, 31, 39, 55, 68` at m11..m31; the `N(v)` profile at m31; the attaining words
`(5,6), (5,11,2), (7,18)/(5,13,7), (7,15,8,4), (10,10,23), (18,10,30), (28,37,12,11)`; the
frontier's `a*` and `Rest(F_old) = 3, 2, 3, 7, 7, 5, 5, 2`; the top law; the `n1(F)` values
`2, 2, 3, 5, 7, 5, 5, 2` quoted in the brief. The blind content is: **the whole `n1(v)` profile at
every machine** (never computed -- only `N(v)` was), the sum profile and its maximisers, the shape
of the top band, the record's configuration enumeration and the gears that close its ends, the
isolation census, the lineage rank fractions, and the out-of-sample rows at `{5..37}`.

### 0.4 Predictions, each with the number that would refute it

- **P1 (instrument).** `max_v (v + n1(v)) = F_2(M)` at m11..m31 with values `11, 16, 25, 31, 39,
  55, 68`, and `n1(v) <= N(v) - 1` for every realised `v` (a neighbour on the other side is at
  least 1). **0 exceptions.** REFUTED by one mismatch.
- **P2 (the `F_2` pair is interior).** The maximiser `v*` of `Sig` satisfies `v*/F in [0.25, 0.80]`
  at m17..m31, and `v*` is **never** the record `F` and never in the top band `T(M)`, at 7 of 7
  machines. REFUTED if `v* in T` at 2 or more machines.
- **P3 (the top band falls short of `F_2`).** `D_top(M) >= 1` at 7 of 7 machines and `D_top >= 5`
  at m29 and m31. REFUTED by `D_top = 0` at any machine, or `D_top <= 2` at both m29 and m31.
- **P4 (the record's neighbours are the shortest at the top).** `n1(F) = 2, 2, 3, 5, 7, 5, 5, 2` at
  `{5,7}`..m31 (instrument, `[read]`); blind: `n1(F) <= n1(v)` for every realised `v >= 0.8 F` at
  6 or more of the 7 machines, i.e. the record is the extreme point of the profile and not merely a
  point on it. REFUTED if some `v` in the top band has `n1(v) < n1(F)` at 3 or more machines.
- **P5 (the top pinned to `F`, the brief's pre-registration).** For `v >= 0.8 F`,
  `n1(v) <= F + c - v` with `c = 3`, i.e. `Sig(v) <= F + 3` on the top band, at 5 or more of the 7
  machines. REFUTED if `Sig(v) > F + 3` on the top band at 3 or more machines. (Note the free
  bound `Sig(v) <= F_2 = F + 4, 5, 7, 6, 5, 12, 10`, so `c = 3` is a genuine strengthening only
  where `F_2 - F > 3`.)
- **P6 (the profile's shape near the top).** On the top band `n1` is non-increasing in `v` at 5 or
  more of the 7 machines (allowing ties). REFUTED by an increase of 3 or more at 3 or more
  machines.
- **P7 (isolation is the record's alone).** `iso_1 = F - s_2` is `>= 3` at m29 and m31
  (`[read]` at m31: sizes `52, 53, 55, 58`), and blind: `iso_2 = s_2 - s_3 <= 2` at 6 or more of
  the 7 machines, so the isolation does not repeat one step down. REFUTED if `iso_2 >= 3` at 3 or
  more machines. Also blind: `m(F) < m(s_2) < m(s_3)` (multiplicity strictly increasing down the
  top three) at 6 or more of 7.
- **P8 (the lineage avoids the top).** At rungs 19->23, 23->29, 29->31, 31->37 no piece of the
  record's fusion word is among the top 3 realised sizes of `M`, and every piece's rank fraction is
  `>= 0.10` (i.e. at least a tenth of the realised sizes are above it). REFUTED by one top-3 piece.
  Blind extension: the same holds for the fusion word of `F_2(M + q')` at those four rungs.
- **P9 (the record's 2-run carries no budget tightness).** `n1(F) <= q' - 6` at 8 of 8 rungs, so the
  budget read at the record's own 2-run has slack `>= 6`; and the budget slack
  `min_a s(a) = F(M) + q' - F(M + q')` is attained at an interior `a*` with `a*/F in [0.55, 0.75]`
  at rungs 23, 29, 31 (`[read]` from `frontier_collapse.md` 2.1). Blind: the pieces of the
  tightness-carrying run at those rungs all have rank fraction `>= 0.25`, and none is the `F_2`
  pair of `M`. REFUTED if the `F_2` pair of `M` is the tightness-carrying 2-run at 2 or more rungs.
- **P10 (the mechanism at the record's ends).** Enumerate every residue configuration realising a
  record gap (there are `m(F)` of them, `4` at m31 `[read]`). Blind predictions: (a) every gear of
  `M` is a sole striker of some interior column of the record gap at every occurrence (this is L4 /
  `pinned_letter.md` 3.1 extended, an instrument line); (b) the column just past each end of the
  record is struck by **exactly one** gear at 6 or more of the 7 machines -- the end is bought
  from a single gear, which is why the neighbour cannot be long; (c) that gear is one of the top
  three gears of `M` at 5 or more of the 7 machines. REFUTED for (b) by two or more strikers at 3
  or more machines.
- **P11 (out of sample at `{5..37}`).** The record configuration enumeration is scan-free, so it
  reaches `M = {5..37}` (`F = 88`, period 1.24e12, never scanned). Predict `n1(88) <= 7` and
  `m(88) <= 40`. REFUTED by `n1(88) >= 8`. (If the enumeration exceeds its node cap the prediction
  is recorded as untested, not confirmed.)

**Stop rules.** Anything that reduces to the merge/chain law (docs/proofs/05), the tooth rule (02),
the attainment identity (08), the record law (09), the peel bound / triple inequality (16), the
glue lemma and `N(v) <= F_2` (2g.i), the pair filter / closer law / row profile (4.i.a.i.a), the
spare-gear lemma (4.i.a.i.a.1), the forced-gear law and the CRT search (4.i.a.i.a.1.a), the
frontier and its top law (4.i.a), the gate ladder (4.i.a.i), R3.h's "records are made of the ends",
or the LP windowed dictionary vehicle, is stopped in one line and cited.

### 0.5 Scorecard

| # | prediction | verdict | evidence |
|---|---|---|---|
| P1 | `max_v Sig(v) = F_2`; `n1 <= N - 1` | CONFIRMED, 7 of 7 machines and 186 of 186 realised sizes, 0 exceptions (instrument) | 2.1 |
| P2 | the `F_2` pair is interior, never in the top band | **REFUTED at 6 of 7 machines**: the `F_2` pair lies in the top band everywhere but `{5..31}`, and at `{5..13}`, `{5..17}`, `{5..23}` it IS the record's own 2-run (`F + n1(F) = F_2` exactly) | 2.1 |
| P3 | `D_top >= 1` everywhere, `>= 5` at m29, m31 | **REFUTED**: `D_top = 0, 0, 0, 0, 0, 0, 3` -- zero at six machines, and the single non-zero value is at `{5..31}`, not `{5..29}` | 2.3 |
| P4 | the record is the extreme point of the profile on the top band | **REFUTED as stated**: it holds at 6 of 8 machines (threshold was 6 of 7); the exceptions are `{5..13}` (`n1(10) = 4`) and `{5..23}` (`n1(33) = 2`, `n1(31) = 4`) | 2.2 |
| P5 | `Sig(v) <= F + 3` on the top band | **REFUTED at 7 of 7**: the least valid `c` is `4, 5, 7, 6, 5, 12, 7`. The top is pinned to `F_2`, not to `F` | 2.2 |
| P6 | `n1` non-increasing on the top band | **REFUTED**: rises at four machines, of `+1`, `+3`, `+4`, `+4` -- the profile is a saw | 2.2 |
| P7 | isolation is the record's alone; multiplicities increase downward | CONFIRMED on isolation (`iso_1 = 3` at `{5..29}`, `{5..31}` and out of sample `{5..37}`; `iso_2 <= 2` at 7 of 7 scanned); **REFUTED** on multiplicity (2 of 7) | 4.1 |
| P8 | the lineage avoids the top 3 sizes of `M` | CONFIRMED on both clauses: 4 of 4 rungs from 19->23 for `F(M+q')` (incl. 31->37) and 3 of 3 for `F_2(M+q')`; every piece's rank fraction `>= 0.268` | 3, 3.1 |
| P9 | the record's 2-run carries no tightness; the tightness is interior | CONFIRMED on the first clause, 8 of 8 rungs, slack `10, 10, 12, 18, 24, 29, 32, 39`; the blind clause is **REFUTED below rung 23** (`M`'s `F_2` pair IS the tightness run at rungs 13 and 19) and CONFIRMED at 3 of 3 rungs from 23 on | 6.1, 6.2 |
| P10 | one gear buys each end of the record | (a) CONFIRMED and made exhaustive: **68 of 68 record occurrences, every gear a sole striker inside**; (b) **REFUTED** (exactly one striker at 68 of 130 ends); (c) **REFUTED** -- the end is bought by **gear 5 at 124 of 130**, a top-three gear at only 60 | 5.1, 5.2 |
| P11 | out of sample at `{5..37}`: `n1(88) <= 7`, `m(88) <= 40` | CONFIRMED: `m(88) = 2`, `n1(88) = 2`, `N(88) = 4`, enumerated scan-free in 34 s; the record's only two fusion words over `{5..31}` are `(28,37,12,11)` and its mirror | 5.3 |
| -- | **not pre-registered, found on the way** | **THE RECORD SATURATION LAW** (every gear a sole striker inside the record gap itself, at every occurrence of eight machines) and **the first failure of the top-band attainment law** at `{5..31}`, where the `F_2` pair `(35, 33)` has its larger member at `0.603 F` | 5.1, 2.3 |

## 1. Setup (exact ranges)

Everything is exact: full periods where a period is sieved, complete enumeration over residue
classes where it is not, integer arithmetic throughout, no sampling except the named family.

| object | range | script |
|---|---|---|
| the profile `n1(v)`, the sum profile `Sig(v)`, `N(v)`, `m(v)`, a witness column for every `n1(v)`, and every occurrence of the record with its two neighbours | full periods of `{5..11}` .. `{5..31}` (385 .. 33,426,748,355 columns; 6.23 billion gaps at m31), sieved in 3e7-column chunks by 3 processes with a 4096-column margin, every gap attributed to its left endpoint | `rr_profile.py` |
| **the configuration enumerator**: every residue configuration realising a gap of a given size, with its two neighbours, its sole-striker map and the strikers of the columns just outside its ends | `{5..11}` .. `{5..31}` at the record and at the whole top band of `{5..29}`; **out of sample at `{5..37}`**, whose period is 1.24e12 columns and has never been sieved | `rr_record.py`, `rr_top37.py` |
| the lineage: every fusion word of `F(M + q')` and of `F_2(M + q')`, read off the same configurations by dropping the top gear, with the pieces' rank fractions in `M`'s spectrum | rungs 11->13 .. 31->37 | `rr_lineage.py` |
| the capacity margin and the shortest uncoverable window at the spectrum's holes near the record | `{5..29}`, sizes 38..43 | `rr_holes.py` |
| the tooth-counterfactual family (alignment-rules section 5, seed 20260906, the members of r57/r58/r62/r63) | 20 members plus the real machine at `{5..13}`, `{5..17}`, `{5..19}`, full periods each | `rr_family.py` |
| the tables | -- | `rr_analyse.py`, `rr_mech.py` |

**Instrument gates, all passed.** The sieve returns `F = 7, 11, 18, 25, 34, 43, 58` and
`F_2 = 11, 16, 25, 31, 39, 55, 68` at m11..m31, the recorded ladders, and its `N(v)` profile at
m31 agrees cell for cell with `research/anchor235/r45/results/deep_profile_31.txt`. The
configuration enumerator reproduces, exactly and scan-free, **every cell the sieve produced that
it was asked for**: `m(v), n1(v), N(v)` at all seven top-band sizes of `{5..29}`
(`35: 442/20/23`, `36: 38/9/11`, `37: 84/11/13`, `38: 22/7/9`, `39: 12/3/4`, `40: 8/7/10`,
`43: 2/2/4`), at all nine top-band sizes of `{5..31}`
(`47: 226/13/15`, `48: 228/17/20`, `49: 46/8/14`, `50: 54/10/13`, `51: 36/9/11`, `52: 10/6/11`,
`53: 34/7/9`, `55: 34/10/13`, `58: 4/5/9`), at the record cell of the five smaller machines, and at
three spectrum holes (`m(41) = m(42) = 0` at `{5..29}`, `m(24) = 0` at `{5..23}`) --
**24 of 24 cells, 0 mismatches**. Its
record words reproduce the recorded attaining words at every rung
(`(5,6)`, `(5,11,2)`, `(7,18)`/`(5,13,7)`, `(7,15,8,4)`, `(10,10,23)`, `(18,10,30)`, and at the
top rung `(28,37,12,11)`; `neighbour_profile.md` 2.3, cited).

## 2. Results: the neighbour profile `n1(v)` and the sum profile (item 1)

`n1(v)` had never been computed. Branch 2g.i computed the neighbour **sum** `N(v)`
(`neighbour_profile.md`), and `short_letter_row.md` computed the row top `r(v)` -- which is the
same object as `n1(v)` -- at a few sizes. Here it is, for **every realised size of every machine
from `{5..11}` to `{5..31}`**, on full periods, 6.23 billion gaps at the top.

### 2.1 The headline: the record as a 2-run, at every machine

| `M` | `F` | `F_2` | `m(F)` | `n1(F)` | `F + n1(F)` | `D_rec = F_2 - F - n1(F)` | the `F_2` pair(s) `(v, n1(v))` | larger member `/ F` | `D_top` |
|---|---|---|---|---|---|---|---|---|---|
| `{5..11}` | 7 | 11 | 4 | 3 | 10 | 1 | `(5,6), (6,5)` | 0.857 | 0 |
| `{5..13}` | 11 | 16 | 12 | 5 | 16 | **0** | `(5,11), (11,5)` | 1.000 | 0 |
| `{5..17}` | 18 | 25 | 20 | 7 | 25 | **0** | `(7,18), (18,7)` | 1.000 | 0 |
| `{5..19}` | 25 | 31 | 20 | 5 | 30 | 1 | `(10,21), (21,10)` | 0.840 | 0 |
| `{5..23}` | 34 | 39 | 4 | 5 | 39 | **0** | `(5,34), (34,5)` | 1.000 | 0 |
| `{5..29}` | 43 | 55 | 2 | 2 | 45 | 10 | `(20,35), (25,30), (30,25), (35,20)` | 0.814 | 0 |
| `{5..31}` | 58 | 68 | 4 | 5 | 63 | 5 | `(33,35), (35,33)` | **0.603** | **3** |

`max_v (v + n1(v)) = F_2(M)` at 7 of 7 machines and `n1(v) <= N(v) - 1` at 186 of 186 realised
sizes, 0 exceptions. **P1 CONFIRMED** (an instrument check: both are definitional, and their
agreement gates the sieve). The brief's list `n1(F) = 2, 2, 3, 5, 7, 5, 5, 2` is indexed by the
OLD machine of each rung, `{5}` through `{5..29}`; the measurement agrees at the six of those in
range (`3, 5, 7, 5, 5, 2` at `{5..11}` .. `{5..29}`) and adds `{5..31}: 5` and, out of sample,
`{5..37}: 2`.

Two things in that table were not expected and neither is definitional.

* **The record's own 2-run IS the largest adjacent pair at three machines** (`{5..13}`,
  `{5..17}`, `{5..23}`: `F + n1(F) = F_2` exactly) and misses by 1 at two more. So the
  pre-registered picture -- the record repelling its neighbours down to an unremarkable pair -- is
  wrong at the small machines: there the record plus its largest neighbour *is* the extreme
  adjacent pair of the whole machine. **P2 REFUTED**, at 6 of 7 machines.
* **`D_rec` then breaks away at the top**: `1, 0, 0, 1, 0, 10, 5`. The record's 2-run stops
  carrying `F_2` exactly at `{5..29}`.

### 2.2 The top band, with the rarity null

For each realised `v >= 0.8 F`: its multiplicity, `n1`, the sum `Sig(v) = v + n1(v)`, and the
**rarity null** `n1_0(v)` = the largest `r` with `2 m(v) P(a random gap is >= r) >= 1`, i.e. the
largest neighbour one would expect from `2 m(v)` blind draws (`frontier_collapse.md` 0.3's M3,
cited).

| `M` | band | `m(v)` | `n1(v)` | `Sig(v)` | `Sig - F` | `n1 - n1_0` |
|---|---|---|---|---|---|---|
| `{5..11}` | 6, 7 | 4, 4 | 5, 3 | 11, 10 | +4, +3 | 0, -2 |
| `{5..13}` | 10, 11 | 12, 12 | 4, 5 | 14, 16 | +3, +5 | -3, -2 |
| `{5..17}` | 15, 16, 18 | 24, 22, 20 | 7, 7, 7 | 22, 23, 25 | +4, +5, +7 | -4, -3, -3 |
| `{5..19}` | 20, 21, 22, 23, 25 | 142, 48, 26, 86, 20 | 10, 10, 5, 5, 5 | 30, 31, 27, 28, 30 | +5, +6, +2, +3, +5 | -6, -4, -7, -10, -7 |
| `{5..23}` | 28, 29, 30, 31, 32, 33, 34 | 322, 6, 112, 20, 8, 2, 4 | 8, 8, 5, 4, 5, 2, 5 | 36, 37, 35, 35, 37, 35, 39 | +2, +3, +1, +1, +3, +1, +5 | -13, -2, -13, -9, -6, -4, -3 |
| `{5..29}` | 35, 36, 37, 38, 39, 40, 43 | 442, 38, 84, 22, 12, 8, 2 | 20, 9, 11, 7, 3, 7, 2 | 55, 45, 48, 45, 42, 47, 45 | +12, +2, +5, +2, -1, +4, +2 | -3, -7, -7, -8, -10, -4, -5 |
| `{5..31}` | 47, 48, 49, 50, 51, 52, 53, 55, 58 | 226, 228, 46, 54, 36, 10, 34, 34, 4 | 13, 17, 8, 10, 9, 6, 7, 10, 5 | 60, 65, 57, 60, 60, 58, 60, 65, 63 | +2, +7, -1, +2, +2, 0, +2, +7, +5 | -10, -6, -10, -10, -9, -7, -11, -8, -5 |

Three readings, in order of how much they say:

1. **The suppression at the top is real and it is not rarity.** `n1(v) - n1_0(v)` is negative at
   34 of the 35 top-band cells (the one exception is a tie at `{5..11}`), with a deficit of 2 to 13
   columns. A near-record gap has a *shorter* largest neighbour than blind draws from the same
   machine's gap distribution would give it. This answers the M1-versus-M3 question of
   `frontier_collapse.md` 2.5 at the top of the spectrum specifically: **rarity is not the whole
   story there, suppression is present at every cell**, and the mean deficit grows with the machine
   (`-1.0, -2.5, -3.3, -6.8, -7.1, -6.3, -8.4`).
2. **"Pinned to `F` with a small `c`" is false.** The brief's pre-registration was
   `Sig(v) <= F + c` on the top band with `c` small. The measured `c = max_{band} (Sig - F)` is
   `4, 5, 7, 6, 5, 12, 7` -- it reaches 12 at `{5..29}` and is not monotone. **P5 REFUTED at 7 of
   7 machines** (`c = 3` fails at every one). What is true is the free bound `Sig(v) <= F_2`, and
   the content of `D_top` below is how close the band comes to it.
3. **The profile is a saw, not a slope.** `n1` rises somewhere on the top band at `{5..13}`
   (+1), `{5..23}` (+1, +3), `{5..29}` (+2, +4) and `{5..31}` (+4, +2, +1, +3) -- four machines
   with a rise, three of them of 3 or more. **P6 REFUTED.** The same saw that
   `frontier_collapse.md` 2.2 found in the slack profile is in the neighbour profile: there is no
   monotonicity at the top to lean on.

Also measured, and it kills the pre-registered "the record is the extreme point of the profile"
(P4): the record has the smallest largest-neighbour in its own top band at `{5..11}`, `{5..17}`,
`{5..19}`, `{5..29}`, `{5..31}` and at `{5..37}` -- but **not** at `{5..13}` (`n1(10) = 4 < 5`) or
`{5..23}` (`n1(33) = 2 < 5`, `n1(31) = 4 < 5`). 6 of 8 including the out-of-sample machine, below
the pre-registered threshold of 6 of 7. **P4 REFUTED as stated.**

### 2.3 THE TOP-BAND DEFICIT, and where it stops being zero

`D_top(M) = F_2(M) - max_{v >= 0.8 F} Sig(v)`:

| `M` | m11 | m13 | m17 | m19 | m23 | m29 | **m31** |
|---|---|---|---|---|---|---|---|
| `D_top` | 0 | 0 | 0 | 0 | 0 | 0 | **3** |
| larger member of the `F_2` pair, `/F` | 0.857 | 1.000 | 1.000 | 0.840 | 1.000 | 0.814 | **0.603** |

> **THE TOP-BAND ATTAINMENT LAW, and its first failure.** At six of the seven machines the largest
> adjacent pair of `M` contains a gap within 19% of the record, so `F_2(M)` is decided by the top
> band alone. At `{5..31}` it is not: the band reaches only 65 against `F_2 = 68`, and the unique
> `F_2` pair is `(35, 33)`, whose larger member is `35 / 58 = 0.603` of the record.

**P3 REFUTED** (it predicted `D_top >= 1` everywhere and `>= 5` at the top two; the truth is 0 at
six machines and 3 at the seventh). On the tooth-counterfactual family the law is not a real-teeth
law either: `D_top = 0` at 16 of 20 members at `{5..13}`, 17 of 20 at `{5..17}` and 18 of 20 at
`{5..19}` -- 51 of 60, with the real machine inside the majority each time.

The failure at `{5..31}` matters for the root and is taken up in 6.3: **the pair statement is not
a statement about the top of the spectrum**, and the counterexample appears at exactly the machine
where the project's other top-of-spectrum laws have started to fail.

## 3. The lineage: which sizes of `M` the next record is made of (item 3)

Every gap of `M + q'` is a run of consecutive gaps of `M` (the merge law, docs/proofs/05, cited).
Dropping the top gear from a record configuration of `M + q'` and re-reading which offsets are
open gives that occurrence's fusion word, so the enumerator returns **every** attaining word, not
one witness. The rank fraction of a piece `p` is `|{realised sizes of M >= p}| / |{realised sizes
of M}|`, so 0 is the record of `M` and 1 the smallest size.

| rung | `F(M+q')` | every record fusion word | largest piece | its rank fraction in `M` | top 3 sizes of `M` | a top-3 piece? |
|---|---|---|---|---|---|---|
| 11->13 | 11 | `(5,6)`, `(6,5)` | 6 | **0.143** | 7, 6, 5 | **YES** (6 and 5 are `s_2`, `s_3`) |
| 13->17 | 18 | `(5,11,2)`, `(2,11,5)`, `(5,6,7)`, `(7,6,5)` | 11 | **0.000** | 11, 10, 8 | **YES** (11 = `F(M)`) |
| 17->19 | 25 | `(7,18)`, `(18,7)`, `(5,13,7)`, `(7,13,5)` | 18 | **0.000** | 18, 16, 15 | **YES** (18 = `F(M)`) |
| 19->23 | 34 | `(7,15,8,4)`, `(4,8,15,7)` | 15 | 0.348 | 25, 23, 22 | no |
| 23->29 | 43 | `(10,10,23)`, `(23,10,10)` | 23 | 0.303 | 34, 33, 32 | no |
| 29->31 | 58 | `(18,10,30)`, `(23,10,25)`, `(25,10,23)`, `(30,10,18)` | 30 | 0.268 | 43, 40, 39 | no |
| **31->37** | 88 | `(28,37,12,11)`, `(11,12,37,28)` | 37 | 0.327 | 58, 55, 53 | no |

**The transition is at rung 19->23 and it is sharp.** The largest piece's rank fraction runs
`0.143, 0.000, 0.000` at the first three rungs and `0.348, 0.303, 0.268, 0.327` from rung 23 on:
below rung 23 the record of `M + q'` is built *from the top of `M`'s spectrum* (twice literally
from `F(M)` itself); from rung 23 on **no piece of any attaining word is among the top three sizes
of `M`, at 4 of 4 rungs**, and every piece's rank fraction is at least `0.268`. **P8 CONFIRMED on
both clauses.** This is R3.h's "records are ordinary lower gaps fused at junctions"
(`ends_or_middles.md`, cited), now with the rank number attached and checked at rung 31->37 by
enumeration instead of by a scan.

Two further exact facts fall out of having *every* word rather than one witness:

* **the interior pieces are always letters.** They are `11 = b_L(17)` and `6 = a_L(17)`;
  `13 = b_L(19)`; `15 = b_L(23)` and `8 = a_L(23)`; `10 = a_L(29)`; `10 = a_L(31)`; `37 = q'` (the
  pad) and `12 = a_L(37)`. This is the chain law's `0, +-d (mod q')` condition (docs/proofs/05 T2,
  cited); what the enumeration adds is that at the last three rungs the middle is the **short**
  letter, and only at 31->37 does a padded middle appear.
* **the words come in mirror pairs at every rung**, and at rung 29->31 the four words are the two
  mirror pairs `(18,10,30)/(30,10,18)` and `(23,10,25)/(25,10,23)`, whose largest pieces `30` and
  `25` are exactly the two minimisers `a* = 30, 25` of the frontier's slack recorded in
  `frontier_collapse.md` 2.1 (cited). The frontier's interior maximiser and the record's largest
  piece are the same object; here it is exhibited as a word, with both mirror pairs.

### 3.1 The same question for `F_2(M + q')`

`F_2(M + q')` is the largest sum of two adjacent gaps of `M + q'`, i.e. `max_v Sig(v)` at that
machine. Prescribing its three openings and dropping the top gear gives its lineage the same way.

| rung | `F_2(M+q')` | the attaining pair `(v*, n1(v*))` | every fusion word over `M` | piece rank fractions in `M` |
|---|---|---|---|---|
| 11->13 | 16 | `(11, 5)` | `(5,6,5)` | 0.286, 0.143, 0.286 |
| 13->17 | 25 | `(18, 7)` | `(2,11,5,7)` | 0.800, **0.000**, 0.500, 0.300 |
| 17->19 | 31 | `(21, 10)` | `(3,13,5,10)`, `(3,18,7,3)` | 0.824, 0.235, 0.706, 0.412 / 0.824, **0.000**, 0.588, 0.824 |
| 19->23 | 39 | `(34, 5)` | `(4,8,15,7,1,4)` | 0.826, 0.652, 0.348, 0.696, 0.957, 0.826 |
| 23->29 | 55 | `(35, 20)` | `(23,10,2,20)` | 0.303, 0.697, 0.939, 0.394 |
| 29->31 | 68 | `(35, 33)` | `(35,20,10,3)` | 0.146, 0.512, 0.756, 0.927 |

Same verdict, same rung: the `F_2` lineage uses `F(M)` itself at rungs 17 and 19, and from rung 23
on no piece is among the top three sizes of `M` (3 of 3 rungs; the fourth, 31->37, is untested
because `F_2({5..37})` is not known). It is also a *deeper* fusion than the record's -- four pieces
at rungs 17, 19, 29 and 31 and six at rung 23, against the record's two to four -- so `F_2` of the
new machine reaches further down the old machine's line than `F` does.

### 3.2 The two mechanisms of growth, named

The brief asks which of two mechanisms attains `F(M + q')`: a fusion of the top of `M`'s spectrum,
or a fusion of ordinary sizes. The answer is both, in that order, with the switch at rung 23:

* **rungs 11->13, 13->17, 17->19: a fusion of the top.** The largest piece is `s_2`, `F(M)`,
  `F(M)`; at 17->19 the record is literally `F_2(M)` (`25 = 18 + 7`), a 2-run of the old machine
  that survives whole.
* **rungs 19->23, 23->29, 29->31, 31->37: a fusion of ordinary sizes.** The largest piece sits at
  a rank fraction of `0.27-0.35`, i.e. between a quarter and a third of the way down the size
  ranking, and the fusion is `J = 3` or `J = 4` with a letter (or the pad) in the middle. `F` goes
  `34, 43, 58, 88` while the top of the spectrum it does *not* use goes `25, 34, 43, 58`.

That is the growth asymmetry the parent branch named: `F` is a deep fusion of the middle of the
spectrum, while the letter's row `a_L + r(a_L)` is a 2-run at the top; the two grow by different
mechanisms and there is no reason for the ladders `43, 58, 88` and `45, 58, 77` to stay together,
and at rung 37->41 they do not.

## 4. The top of the spectrum as an object (item 4)

### 4.1 Isolation: the record's, and nobody else's

`s_1 > s_2 > ...` are the largest realised sizes and `iso_k = s_k - s_{k+1}`.

| `M` | top 8 realised sizes | `iso_1 .. iso_7` | `m` of the top 5 | `n1` of the top 5 |
|---|---|---|---|---|
| `{5..11}` | 7, 6, 5, 4, 3, 2, 1 | 1, 1, 1, 1, 1, 1 | 4, 4, 22, 6, 22 | 3, 5, 6, 3, 7 |
| `{5..13}` | 11, 10, 8, 7, 6, 5, 4, 3 | 1, 2, 1, 1, 1, 1, 1 | 12, 12, 20, 84, 60 | 5, 4, 7, 8, 7 |
| `{5..17}` | 18, 16, 15, 14, 13, 12, 11, 10 | 2, 1, 1, 1, 1, 1, 1 | 20, 22, 24, 12, 66 | 7, 7, 7, 8, 7 |
| `{5..19}` | 25, 23, 22, 21, 20, 18, 17, 16 | 2, 1, 1, 1, 2, 1, 1 | 20, 86, 26, 48, 142 | 5, 5, 5, 10, 10 |
| `{5..23}` | 34, 33, 32, 31, 30, 29, 28, 27 | 1, 1, 1, 1, 1, 1, 1 | 4, 2, 8, 20, 112 | 5, 2, 5, 4, 5 |
| `{5..29}` | 43, 40, 39, 38, 37, 36, 35, 34 | **3**, 1, 1, 1, 1, 1, 1 | 2, 8, 12, 22, 84 | 2, 7, 3, 7, 11 |
| `{5..31}` | 58, 55, 53, 52, 51, 50, 49, 48 | **3**, 2, 1, 1, 1, 1, 1 | 4, 34, 34, 10, 36 | 5, 10, 7, 6, 9 |
| **`{5..37}`** | 88, 85, 77, 72, 71, 70, 69, 68 | **3**, 8, 5, 1, 1, 1, 1 | 2, 4, 2, 2, 8 | 2, 5, 11, 8, 2 |

> **`iso_1 = 3` at `{5..29}`, `{5..31}` and `{5..37}` -- three machines in a row, the last of them
> out of sample -- and `iso_2 <= 2` at 7 of the 7 scanned machines.** The isolation does not repeat
> one step down at any scanned machine. **P7 CONFIRMED on the isolation clause**; its multiplicity
> clause (`m(F) < m(s_2) < m(s_3)`) is **REFUTED**, holding at only 2 of 7 (`{5..17}`, `{5..29}`):
> the multiplicities at the top are `4, 4, 22` at m11 and `4, 34, 34` at m31, not an increasing
> sequence.

At `{5..37}` the isolation is much larger one step down (`iso_2 = 8`, `iso_3 = 5`), so even
`iso_2 <= 2` is a small-machine fact.

### 4.2 The top of the spectrum thins out

Realised sizes as a fraction of the top band `[0.8 F, F]`:

| `M` | m11 | m13 | m17 | m19 | m23 | m29 | m31 | **m37** |
|---|---|---|---|---|---|---|---|---|
| band width (integers) | 2 | 3 | 4 | 6 | 7 | 9 | 12 | **18** |
| realised | 2 | 2 | 3 | 5 | 7 | 7 | 9 | **5** |
| fraction | 1.00 | 0.67 | 0.75 | 0.83 | 1.00 | 0.78 | 0.75 | **0.28** |

The top of `{5..37}`, computed scan-free for the first time, is
`88 (m=2), 85 (4), 77 (2), 72 (2), 71 (8)` with **thirteen certified holes** at
`73, 74, 75, 76, 78, 79, 80, 81, 82, 83, 84, 86, 87` (and `89, 90` above the record). Below the
band the spectrum fills in again immediately: every size from `61` to `70` is realised, with
multiplicities `108, 546, 328, 42, 190, 8, 44, 60, 2, 44`. So the picture is not "the top thins
gradually" but **a thin cap of five sizes sitting above a full spectrum**, and the cap grows
thinner with the machine.

### 4.3 The spectrum of `{5..37}` from 55 up, computed scan-free

Nothing below is a scan: each row is a complete enumeration of the residue configurations that
realise a gap of that size at `M = {5..37}` (period 1.24e12 columns).

| `v` | 88 | 87-86 | 85 | 84-78 | 77 | 76-73 | 72 | 71 | 70 | 69 | 68 | 67 | 66 | 65 | 64 | 63 | 62 | 61 | 60 | 59 | 58 | 57 | 56 | 55 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `m(v)` | 2 | 0 | 4 | 0 | 2 | 0 | 2 | 8 | 44 | 2 | 60 | 44 | 8 | 190 | 42 | 328 | 546 | 108 | 1020 | 28 | 2902 | 924 | 758 | 9910 |
| `n1(v)` | 2 | -- | 5 | -- | 11 | -- | 8 | 2 | 18 | 5 | 13 | 10 | 7 | 20 | 21 | 25 | 28 | 9 | 22 | 11 | 30 | 20 | 32 | 17 |
| `Sig(v)` | **90** | -- | **90** | -- | 88 | -- | 80 | 73 | 88 | 74 | 81 | 77 | 73 | 85 | 85 | 88 | **90** | 70 | 82 | 70 | 88 | 77 | 88 | 72 |

Two readings:

* **`max_{v >= 55} Sig(v) = 90 = F + 2`**, attained three times (`v = 88, 85, 62`). So
  `F_2({5..37}) >= 90`, and if the larger member of its `F_2` pair is at least 55 then
  `F_2({5..37}) = 90` exactly. That would make `F_2 / F = 1.023` against `1.57, 1.45, 1.39, 1.24,
  1.15, 1.28, 1.17` at the seven scanned machines -- so the more likely reading is that the `F_2`
  pair of `{5..37}` sits *below* 55, i.e. that `D_top` has not only left zero at `{5..31}` but has
  grown. Deciding it needs the cells `v in [45, 54]`, which the enumerator can reach but this lane
  did not (`m(55)` is already 9,910 and 539M nodes).
* the brief's `Sig(v) <= F + c` with small `c` -- refuted at every scanned machine -- **holds at
  `{5..37}` with `c = 2` on the top band**, and would hold with `c = 2` all the way down to
  `v = 55`. The band's `c` runs `4, 5, 7, 6, 5, 12, 7, 2`: it is not a law, it is a number that
  moves.

### 4.4 The holes have no local mechanism

Does a size just below the record fail because it "would need a configuration the record's
uniqueness forbids"? At `{5..29}`, where `41` and `42` are the holes:

| `v` | admissible classes per gear (5..29) | capacity `sum_g max strikes` | demand `v - 1` | margin | `m(v)` | shortest uncoverable window |
|---|---|---|---|---|---|---|
| 38 | 2, 3, 7, 9, 13, 17, 20, 25 | 55 | 37 | +18 | 22 | -- |
| 39 | 1, 3, 7, 11, 13, 15, 19, 26 | 58 | 38 | +20 | 12 | -- |
| 40 | 3, 4, 8, 9, 14, 15, 19, 25 | 59 | 39 | +20 | 8 | -- |
| **41** | 1, 3, 7, 9, 13, 15, 19, 25 | 61 | 40 | **+21** | **0** | `[1, 37)` -- **36 of 40 columns** |
| **42** | 2, 5, 7, 9, 13, 15, 19, 25 | 63 | 41 | **+22** | **0** | `[2, 41)` -- **39 of 41 columns** |
| 43 | 2, 3, 7, 10, 13, 15, 19, 25 | 63 | 42 | +21 | 2 | -- |

**The answer is no, and it is a clean negative.** The capacity margin is *larger* at the two empty
sizes than at the record itself (`+21, +22` against `+21`), so no counting argument can separate
them; and the shortest sub-interval of the run that already cannot be covered is 36 of 40 columns
at `v = 41` and 39 of 41 at `v = 42` -- essentially the whole run. There is no local certificate
for a spectrum hole, exactly as there is none for the cells above the short letter's row
(`pinned_arithmetic.md` 3.4, cited: shortest uncoverable window 15 to 44 columns of runs 36 to 53
long). Two different questions at the top of the spectrum reach the same wall: **emptiness at the
top is a global covering fact of the whole machine, not a local arithmetic one.**

## 5. Mechanism: what happens at the record's two ends (item 2)

The configuration enumerator gives the record's mechanism completely, because it gives **every**
occurrence: a class `lam_g` per gear is one column of the period by CRT, so the solutions of
"offsets 0 and `F` open, every offset between them struck" are in bijection with the record gaps of
the period. There are `m(F)` of them and the enumerator finds them all.

| `M` | `F` | `m(F)` | `n1(F)` | `N(F)` | the neighbour pairs `(L, R)`, with multiplicity | every gear a sole striker inside? | sole strikers per gear (one occurrence) | sole share of the interior |
|---|---|---|---|---|---|---|---|---|
| `{5..11}` | 7 | 4 | 3 | 4 | `(3,1) x2, (1,3) x2` | YES | 5:3 7:2 11:1 | 6/6 |
| `{5..13}` | 11 | 12 | 5 | 7 | `(2,5) x4, (5,2) x4, (2,2) x4` | YES | 5:4 7:2 11:2 13:1 | 9/10 |
| `{5..17}` | 18 | 20 | 7 | 10 | 11 distinct pairs, `(5,2)` and `(2,5)` four times each | YES | 5:4 7:4 11:2 13:1 17:2 | 13/17 |
| `{5..19}` | 25 | 20 | 5 | 7 | `(3,2) x5, (2,3) x5, (3,3) x4, (2,2) x4, (5,2), (2,5)` | YES | 5:5 7:6 11:2 13:2 17:2 19:1 | 18/24 |
| `{5..23}` | 34 | 4 | 5 | 6 | `(3,3) x2, (1,5), (5,1)` | YES | 5:5 7:3 11:3 13:3 17:2 19:2 23:3 | 21/33 |
| `{5..29}` | 43 | **2** | 2 | 4 | `(2,2) x2` | YES | 5:8 7:4 11:4 13:3 17:3 19:3 23:2 29:2 | 29/42 |
| `{5..31}` | 58 | 4 | 5 | 9 | `(3,4), (4,3), (5,4), (4,5)` | YES | 5:9 7:6 11:3 13:4 17:4 19:2 23:3 29:2 31:2 | 35/57 |
| **`{5..37}`** | **88** | **2** | **2** | **4** | `(2,2) x2` | YES | 5:14 7:9 11:5 13:6 17:6 19:4 23:2 29:3 31:2 37:3 | 54/87 |

### 5.1 THE RECORD SATURATION LAW (new, exceptionless, exhaustive)

> **At every occurrence of a record gap, every gear of `M` is the sole striker of some column
> strictly inside the record gap.** 68 of 68 occurrences over eight machines, `{5..11}` through
> `{5..37}` -- not sampled, *enumerated*.

This is strictly stronger than what the spare-gear lemma gives. That lemma
(`pinned_letter.md` 3.1, cited) applied to the 2-run `(L, F)` says only that no gear is *free*
in the whole run: each gear is obstructed at the middle opening or is a sole striker somewhere in
`(x_0 - L, x_0 + F)`. The measurement says each gear is a sole striker inside the record **half
alone**, at every occurrence, with no exception and no obstruction clause needed. The consequence
is the mechanism the brief asks for:

> **THE REPULSION AT THE TOP, stated as a mechanism.** A record configuration has no movable gear.
> Re-phasing any gear `g` opens a column inside the record and destroys it. So the two neighbour
> gaps are not chosen -- they are whatever the frozen configuration leaves, and `n1(F)` is a
> maximum over exactly `2 m(F)` determined numbers. With `m(F) = 4, 12, 20, 20, 4, 2, 4, 2` the
> record's largest neighbour is a maximum over 4 to 40 draws, where an ordinary size's is a maximum
> over millions.

### 5.2 The end-buying census: gear 5 closes the record's neighbour

For every record occurrence and each end whose neighbour has size `> 1`, the column immediately
outside the record (the first column of the neighbour gap):

| statistic | value |
|---|---|
| such columns over all eight machines | 130 |
| struck by exactly 1 / 2 / 3 / 4 gears | 68 / 38 / 14 / 10 |
| **gear 5 among the strikers** | **124 of 130** |

So the pre-registered "one gear buys each end" (P10b) is **refuted as stated** -- a single striker
at 68 of 130 columns, and *every* such column has a single striker at only 1 of the 8 machines
(`{5..11}`) -- and what replaces it is sharper and was not predicted:
**the record's neighbour is closed by gear 5 at 124 of 130 ends.** All six exceptions are at `{5..23}`, where the first
outside column is struck by `[23]`, `[11]` or `[13, 23]`. P10(c), "one of the top three gears",
is not what decides it either: a top-three gear is present at 60 of the 130 columns, against gear
5's 124.

The reason is arithmetic and one line: gear 5 has `u_5 = 1`, so it strikes the columns
`k = 1, 4 (mod 5)` and leaves `k = 0, 2, 3`; the admissible classes come in the blocks `{2, 3}` and
`{0}`, so **gear 5 strikes at least one of any three consecutive columns**. A record gap's ends sit
in the admissible classes, and the record is long, so gear 5 has used its full quota inside -- it
is the leading sole striker at 7 of the 8 machines (`3, 4, 4, 5, 5, 8, 9, 14` columns against the
runner-up's `2, 2, 4, 6, 3, 4, 6, 9`; the exception is `{5..19}`, where gear 7 leads 6 to 5) -- and
its next tooth falls immediately outside.

### 5.3 Out of sample: the record of `{5..37}`

`M = {5..37}` has a period of 1.24e12 columns and has never been sieved. The enumerator decides it
from residues in 34 seconds and 76.6 million nodes:

> **`m(88) = 2`, `n1(88) = 2`, `N(88) = 4`.** The record gap of `{5..37}` occurs exactly twice per
> period, and both occurrences have a 2-gap on each side.

Dropping gear 37 from the two configurations gives the record's decomposition into gaps of
`{5..31}`: `(28, 37, 12, 11)` and its mirror `(11, 12, 37, 28)` -- the recorded `J = 4` word
(`neighbour_profile.md` 2.3, cited), now with the statement that it and its mirror are the *only*
two. **P11 CONFIRMED** (`n1(88) = 2 <= 7`, `m(88) = 2 <= 40`).

The same enumeration certifies, scan-free, that `{5..37}` has **no gap of size 89 or 90**
(24.3M and 58.1M nodes), so the recorded `F({5..37}) = 88` is not merely a lower bound from a
witness at this machine.

## 6. Toward the root: where the budget's tightness actually is (item 5)

### 6.1 The budget at the record's own 2-run is vacuous, and by a growing margin

The budget inequality `F(M + q') <= F(M) + q'` reads, on the frontier, `Rest(a) <= F + q' - a` for
every old size `a` (`frontier_collapse.md` 0.1, cited). At `a = F` it is `Rest(F) <= q'`, and by
the top law `Rest(F)` is `n1(F)` (or `N(F)` at the two rungs where `F` is interior-legal). So:

| rung `q'` | `M` | `F(M)` | `n1(F)` | `N(F)` | `Rest(F)` (top law) | budget allowance `q'` | slack |
|---|---|---|---|---|---|---|---|
| 13 | `{5..11}` | 7 | 3 | 4 | 3 | 13 | 10 |
| 17 | `{5..13}` | 11 | 5 | 7 | 7 (legal) | 17 | 10 |
| 19 | `{5..17}` | 18 | 7 | 10 | 7 | 19 | 12 |
| 23 | `{5..19}` | 25 | 5 | 7 | 5 | 23 | 18 |
| 29 | `{5..23}` | 34 | 5 | 6 | 5 | 29 | 24 |
| 31 | `{5..29}` | 43 | 2 | 4 | 2 | 31 | 29 |
| 37 | `{5..31}` | 58 | 5 | 9 | 5 | 37 | 32 |
| **41** | **`{5..37}`** | **88** | **2** | **4** | **2** | **41** | **39** |

**P9 CONFIRMED, 8 of 8 rungs**, including the out-of-sample one: `n1(F) <= q' - 6` everywhere, and
the slack is non-decreasing along the ladder, `10, 10, 12, 18, 24, 29, 32, 39`. *The record's own 2-run carries no
part of the budget's difficulty and the share it carries is shrinking.*

### 6.2 Where the tightness is instead

The budget slack `min_a s(a) = F(M) + q' - F(M + q')` is attained at the size `a*` where the record
is made (`frontier_collapse.md` 2.1, cited), and this branch's enumeration exhibits the attaining
runs themselves:

| rung | `a*` | `a*/F` | rank fraction of `a*` in `M` | the tightness-carrying run (every word) | is it `M`'s `F_2` pair? |
|---|---|---|---|---|---|
| 11->13 | 6 | 0.857 | 0.143 | `(5,6)` -- a 2-run | **YES** |
| 13->17 | 7, 11 | 0.636, 1.000 | 0.300, 0.000 | `(5,6,7)`, `(5,11,2)` -- 3-runs | contains it (`(5,11)`) |
| 17->19 | 13, 18 | 0.722, 1.000 | 0.235, 0.000 | `(5,13,7)` -- a 3-run; `(7,18)` -- a 2-run | **YES** (`(7,18)`) |
| 19->23 | 15 | 0.600 | 0.348 | `(7,15,8,4)` -- a 4-run | no |
| 23->29 | 23 | 0.676 | 0.303 | `(10,10,23)` -- a 3-run | no |
| 29->31 | 25, 30 | 0.581, 0.698 | 0.390, 0.268 | `(23,10,25)`, `(18,10,30)` -- 3-runs | no |

**P9's blind clause is half refuted and the split is again at rung 23.** `M`'s own `F_2` pair *is*
the tightness-carrying run at rungs 13 and 19 and sits inside it at rung 17; from rung 23 on it
never is, at 3 of 3 rungs. The tightness-carrying runs from rung 23 on are 3- and 4-runs whose
pieces are ordinary sizes at rank fractions `0.35-0.83`, `0.30-0.70` and `0.27-0.76` -- exactly the
"ordinary sizes near `0.6 F`" the parent branch pointed at.

### 6.3 The smallest statement about the top of the spectrum that would help -- and what it does not reach

The frontier splits along the attainment identity: the `J = 2` half is the pair statement and the
`J >= 3` half is the chain statement (`frontier_collapse.md`, PLACEMENT, cited), and
`max_a (a + Rest_2(a)) = F_2(M)` is an identity there. Composing that identity with this branch's
top-band attainment law gives the reduction:

> **The pair statement `F_2(M) <= F(M) + q'` is almost a statement about the top band alone:**
> `F_2(M) = max_{v >= 0.8 F} (v + n1(v))` at **6 of 7 machines** -- and it **fails at the deepest
> one**, `{5..31}`, where the top band reaches only 65 against `F_2 = 68`. The top-band deficit is
> `D_top = 0, 0, 0, 0, 0, 0, 3`.

So the smallest statement about the top of the spectrum as 2-runs that would help is
`Sig(v) <= F + q'` on `v >= 0.8 F` -- a maximum over the 2 to 9 sizes of the top band instead of
over all 7 to 55 realised sizes -- but it is **not sufficient**, because `D_top` has just left zero
at m31 and there is no reason on record for it to return. **And in any case it does not reach the
band `[15, 35]` at 29 -> 31**, and the numbers say why plainly:

* the pair half at that rung is `F_2({5..29}) = 55` against the budget `43 + 31 = 74` -- slack 19,
  and the whole top band of `{5..29}` sits at `Sig = 42..55`, that is 19 to 32 under the line;
* the residual band `[15, 35]` (`availability_gate.md`, `short_letter_row.md`, cited) is the
  `J >= 3` half, where the record of `{5..31}` is actually made, at `a = 25` and `a = 30` with a
  letter middle -- sizes at rank fractions `0.39` and `0.268`, well below the top band's `0.8 F`;
* the record's own 2-run at that rung is `43 + 2 = 45` against 74, slack 29.

> **Verdict on item 5, stated plainly: the record as a 2-run is not where the difficulty lives.**
> Its slack against the budget is 29 at 29 -> 31 and 39 at 37 -> 41, and it grows with the machine.
> The difficulty is at the interior maximiser -- an ordinary size near `0.6 F` fused with a letter
> -- exactly as the parent branch's diagnosis said. What the top of the spectrum *does* buy is the
> pair half, and the pair half is not the half that is open.

## 7. What is new

1. **THE NEIGHBOUR PROFILE `n1(v)`, computed.** For every realised size of every machine from
   `{5..11}` to `{5..31}`, on full periods (6.23 billion gaps at the top), plus the whole top band
   of `{5..37}` scan-free. Only the neighbour **sum** `N(v)` existed (branch 2g.i) and only three
   cells of `n1` (the letters' rows, `short_letter_row.md`). The profile, the sum profile
   `Sig(v) = v + n1(v)` and the top-band tables are 2.1-2.3.
2. **THE TOP-BAND DEFICIT `D_top`, and its first failure.** `F_2(M) = max_{v >= 0.8 F} Sig(v)` at
   six machines (`D_top = 0, 0, 0, 0, 0, 0`) and **fails at `{5..31}`** by 3, where the unique
   `F_2` pair is `(35, 33)` with `35 / 58 = 0.603`. Not a real-teeth law (51 of 60 family members).
   The consequence for the root is in 6.3: the pair statement is *nearly* a statement about the top
   of the spectrum and stops being one at exactly the machine where the project's other
   top-of-spectrum laws stop.
3. **THE RECORD SATURATION LAW**, exceptionless and *exhaustive* rather than sampled: at every one
   of the 68 record occurrences of eight machines (`{5..11}` .. `{5..37}`), every gear of `M` is
   the sole striker of some column strictly inside the record gap. This is strictly stronger than
   the spare-gear lemma's conclusion at the 2-run `(L, F)`, and it is what freezes the record's
   configuration: `n1(F)` is a maximum over `2 m(F)` determined numbers with
   `m(F) = 4, 12, 20, 20, 4, 2, 4, 2`.
4. **THE END-BUYING CENSUS**: over the 130 first-outside columns of the eight machines' record
   occurrences, **gear 5 is a striker at 124**, against a top-three gear at 60; the striker count
   is 1 at 68 of them. The record's neighbour is closed by the smallest gear, not the largest.
5. **THE CONFIGURATION ENUMERATOR** as a certification vehicle: it returns, exactly and scan-free,
   `m(v)`, `n1(v)`, `N(v)`, every occurrence's neighbours and every fusion word, for any size at
   any machine. Validated on 20 of 20 cells the sieve could produce (the whole top band of
   `{5..29}`, the record of seven machines, three spectrum holes), 0 mismatches; it then decides
   `{5..37}`, whose period of 1.24e12 columns no scan reaches.
6. **THE TOP OF `{5..37}`, out of sample**: `m(88) = 2`, `n1(88) = 2`, `N(88) = 4`, the record's
   two configurations being mirror images whose fusion words over `{5..31}` are `(28, 37, 12, 11)`
   and `(11, 12, 37, 28)` -- the only two. The band above `0.8 F` is
   `88 (2), 85 (4), 77 (2), 72 (2), 71 (8)` with thirteen certified holes inside it and `89, 90`
   certified empty above the record; every size from 61 to 70 is realised. `iso_1 = 3` for the
   third machine running.
7. **THE LINEAGE, complete rather than by witness**, at every rung including 31->37: every
   attaining fusion word of `F(M + q')` and of `F_2(M + q')`, with the pieces' rank fractions. The
   record's largest piece has rank fraction `0.143, 0.000, 0.000` at rungs 13, 17, 19 and
   `0.348, 0.303, 0.268, 0.327` from rung 23 on -- **the switch from "a fusion of the top of the
   spectrum" to "a fusion of ordinary sizes" is at rung 19->23 and it is sharp**, and it is the
   same rung at which `M`'s own `F_2` pair stops carrying the budget's tightness (3 of 3 below it,
   0 of 3 above).
8. **The spectrum holes have no local certificate.** At `{5..29}` the empty sizes 41 and 42 have a
   *larger* capacity margin (`+21, +22`) than the record (`+21`), and their shortest uncoverable
   window is 36 of 40 and 39 of 41 columns. The same wall as the cells above the short letter's
   row.
9. **The record's 2-run is loose in the budget, and increasingly so**: `Rest(F) = n1(F)` (or
   `N(F)` at the two interior-legal rungs) against the allowance `q'` gives slack
   `10, 10, 12, 18, 24, 29, 32, 39` at rungs 13 .. 41, the last computed out of sample.

Prior art inside the project, handled in a line each: `n1(v) = r(v)` is the adjacent-pair row top
of `short_letter_row.md`; `max_v (v + n1(v)) = F_2` is the definition of `F_2` and
`max_a (a + Rest_2(a)) = F_2(M)` is `frontier_collapse.md`'s identity; the top law
`Rest(F_old) = n1(F_old)` (or `N`) is `frontier_collapse.md` 2.4; "every gear is a sole striker in
an above-record stretch" is L4 (`pair_statement.md`) and its 90-of-90 sampled form is
`pinned_letter.md` 3.1 -- what is new here is the *record gap itself*, exhaustively; the spare-gear
lemma is `pinned_letter.md` 3.1; the merge and chain laws are docs/proofs/05; "records are ordinary
lower gaps fused at junctions" is R3.h (`ends_or_middles.md`); the rarity null is
`frontier_collapse.md` 0.3's M3; the recorded attaining words and `F_2` ladder are
`neighbour_profile.md` 2.3 and docs/proofs/16; spectrum holes have been certified before by the LP
lane (`restricted-covering-certificates.md`), and this branch's enumerator certifies them about
three orders of magnitude more cheaply. Outside the project: not checked (no web access).

## 8. Verdict

**Node status: FACT** -- a new exact object (the neighbour profile `n1(v)` at every machine), one
new exceptionless law (record saturation), one new law that fails at the deepest machine (the
top-band deficit), a scan-free vehicle that reaches `{5..37}`, and a clear negative answer to the
branch's own question about the root.

- **Item 1 (the profile and its shape at the top): delivered, and it refutes the pre-registered
  shape.** `n1(v)` is computed for all 186 realised sizes of the seven scanned machines and for the
  top band of `{5..37}`. `max_v Sig(v) = F_2` and `n1 <= N - 1` hold everywhere (instrument). The
  brief's pre-registration -- "the top of the spectrum is pinned to `F` as pairs,
  `n1(v) <= F + c - v` with a small `c` for `v >= 0.8 F`" -- is **REFUTED at all seven scanned
  machines**: the measured `c` is `4, 5, 7, 6, 5, 12, 7`. The top is pinned to `F_2`, not to `F`,
  and `F_2 - F` is itself `4, 5, 7, 6, 5, 12, 10`. The profile on the band is a saw, not a slope
  (rises of up to +4), and the record is *not* always its extreme point (6 of 8 machines).
  What *is* new and sharp is the **suppression against the rarity null**: `n1(v) < n1_0(v)` at 34
  of 35 top-band cells, by 2 to 13 columns, so the collapse at the top is not explained by the
  smallness of `m(v)` alone.
- **Item 2 (the mechanism at the record): delivered, and it is the record's rigidity.** Every gear
  of `M` is the sole striker of some interior column of the record gap at **all 68 record
  occurrences of eight machines** -- so no gear can be re-phased without destroying the record, the
  configuration is frozen, and the two neighbours are a maximum over `2 m(F)` determined numbers
  with `m(F)` as small as 2. The column that closes each neighbour is bought by **gear 5** at 124
  of 130 ends; the pre-registered "one gear, one of the top three" is refuted on both halves.
- **Item 3 (the two mechanisms of growth): named, with the rung at which they change.** Below rung
  23 the record of `M + q'` is a fusion of the top of `M`'s spectrum (largest piece at rank
  fraction `0.143, 0.000, 0.000`, twice literally `F(M)`); from rung 23 on it is a fusion of
  ordinary sizes with a letter in the middle (largest piece at `0.348, 0.303, 0.268, 0.327`,
  no top-3 piece at 4 of 4 rungs), and the same holds for `F_2(M + q')` at 3 of 3 testable rungs.
  This is R3.h's law with a number, checked to rung 37 without a scan.
- **Item 4 (the top of the spectrum as an object): the isolation is real, its mechanism is not
  local.** `iso_1 = 3` at `{5..29}`, `{5..31}` and (out of sample) `{5..37}`, three machines
  running; `iso_2 <= 2` at all seven scanned machines, so the isolation does not repeat one step
  down -- but at `{5..37}` it does (`iso_2 = 8`), and the top band there holds only 5 of 18
  integers. The empty sizes just below a record have a *larger* capacity margin than the record and
  no uncoverable window shorter than 36 of 40 columns: **no local certificate** (4.4).
- **Item 5 (toward the root): the record as a 2-run is not where the difficulty lives, and this is
  now a measured statement.** The budget read at the record's own 2-run has slack
  `10, 10, 12, 18, 24, 29, 32, 39` at rungs 13 .. 41 and the slack grows with the machine; the
  tightness is carried by 3- and 4-runs whose pieces are ordinary sizes at rank fractions
  `0.27-0.83`. The one thing the top of the spectrum *would* buy is the pair half of the frontier
  (`F_2(M) <= F(M) + q'`), and even that reduction fails at `{5..31}` where `D_top = 3`.

**No CANDIDATE is claimed.** Nothing here is a route to the root: the two laws that survive
(record saturation; `iso_1 = 3` three machines running) are facts about the record's rigidity and
its isolation, and neither bounds `F(M + q')`. What the branch removes is a hypothesis: **the top
of the spectrum as pairs is not the place where the budget is tight**, and the parent's diagnosis
(ordinary sizes near `0.6 F`) is confirmed from a second direction.

**The child branch this names.** The record's largest piece sits at rank fraction `0.27-0.35` from
rung 23 on and is fused with a *letter*. Both objects are now measured -- the letter's row is
certified scan-free (`pinned_arithmetic.md`) and the profile `n1` is computed here -- and the record
is `max_l (l + N(l))` over letters at 5 of 9 rungs but not at 31->37 (85 against 88,
`pinned_letter.md` 4.3, cited). The next object is therefore **the letter's neighbour SUM profile
`N(l)` as a 3-run, at the rungs where the 3-run form fails**, i.e. what the padded middle `q'` buys
that a letter middle cannot: at `{5..31}` the `F_3` maximiser is `(18, 37, 30)` with the pad in the
middle, and `F({5..37}) = 88` is a `J = 4` word `(28, 37, 12, 11)` carrying both a pad and a letter.
The enumerator built here decides such words scan-free at `{5..37}` and one machine beyond.

## 9. Dead ends, each with its refuting instance

- **"The top of the spectrum is pinned to `F` as pairs, `Sig(v) <= F + c` with a small `c`"**
  (the brief's own pre-registration, and this branch's theory T). Refuted at every scanned machine:
  `max_{v >= 0.8F} (Sig - F) = 4, 5, 7, 6, 5, 12, 7`. At `{5..29}` the size `v = 35` has
  `n1 = 20` and `Sig = 55 = F + 12`.
- **"The record repels its neighbours, so the `F_2` pair is an interior pair"** (P2). Refuted:
  `F + n1(F) = F_2` exactly at `{5..13}`, `{5..17}` and `{5..23}`, and is one short at `{5..11}`
  and `{5..19}`. At five of seven machines the record's own 2-run is the largest adjacent pair of
  the machine or within 1 of it.
- **"The top band falls short of `F_2` and the deficit grows"** (P3). Refuted: `D_top = 0` at six
  machines. What is true is the opposite and it fails only at the seventh (`D_top = 3` at
  `{5..31}`).
- **"The record is the extreme point of the profile on its top band"** (P4). Refuted at `{5..13}`
  (`n1(10) = 4 < n1(11) = 5`) and `{5..23}` (`n1(33) = 2 < n1(34) = 5`): 6 of 8, below the
  pre-registered 6 of 7.
- **"`n1` is non-increasing near the top"** (P6). Refuted at four machines, with rises of
  `+3, +4, +4` at `{5..23}`, `{5..29}`, `{5..31}`; the profile is a saw, like the frontier's slack.
- **"One gear buys each end of the record, and it is one of the top three"** (P10 b, c). Refuted:
  a single striker at 68 of 130 first-outside columns, and a top-three gear at 60 of 130 against
  gear 5's 124.
- **"The multiplicity increases as one goes down the top of the spectrum"** (P7's second clause).
  Refuted at 5 of 7 machines: `4, 4, 22` at `{5..11}`, `4, 34, 34` at `{5..31}`.
- **"`M`'s own `F_2` pair never carries the budget's tightness"** (P9's blind clause). Refuted at
  rungs 11->13 (`(5,6)` is both) and 17->19 (`(7,18)` is both), and contained in the tightness run
  at 13->17. It is true only from rung 23 on (3 of 3).
- **A counting or local certificate for the spectrum's holes at the top.** Refuted at `{5..29}`:
  capacity margin `+21, +22` at the empty sizes 41, 42 against `+21` at the record 43, and the
  shortest uncoverable window is 36 of 40 and 39 of 41 columns.
- **The drop-one-gear test as a probe of a hole's mechanism.** Withdrawn as meaningless, not
  refuted: removing a gear makes covering strictly harder, so a hole stays a hole in every
  sub-machine and the test carries no information. It was run (16 cells, all 0) before the error
  was noticed; the capacity/window test in 4.3 is what replaced it.

## 10. What holds without exception (item 6)

| statement | count | status |
|---|---|---|
| `max_v (v + n1(v)) = F_2(M)` | 7 of 7 machines | identity, verified (instrument) |
| `n1(v) <= N(v) - 1` for every realised size | 186 of 186 | identity, verified (instrument) |
| **the record saturation law**: at every occurrence of a record gap, every gear of `M` is the sole striker of some column strictly inside it | **68 of 68 occurrences, 8 machines incl. `{5..37}`** | measured, exhaustive (not sampled); strictly stronger than the spare-gear lemma's conclusion |
| every gear is a sole striker inside every gap of the top band | **20 of 21** top-band sizes of `{5..29}`, `{5..31}`, `{5..37}` | measured; the one failure is `v = 35 = 0.814 F` at `{5..29}` -- which is exactly that machine's `F_2` maximiser, the top-band cell with the longest neighbour (`n1 = 20`) and the weakest suppression (`-3`) |
| gear 5 strikes the first column outside an end of the record | 124 of 130 ends | measured |
| `n1(v) < n1_0(v)` (the rarity null) on the top band | 34 of 35 cells | measured; the exception is a tie at `{5..11}` |
| `n1(F) <= q' - 6`, i.e. the record's 2-run is inside the budget with slack `>= 6` | 8 of 8 rungs, slack `10, 10, 12, 18, 24, 29, 32, 39` | measured, one out of sample |
| `iso_1 = F - s_2 = 3` | `{5..29}`, `{5..31}`, `{5..37}` -- 3 of the 3 machines above `{5..23}` | measured, one out of sample |
| `iso_2 <= 2` | 7 of 7 scanned machines | measured -- **and false at `{5..37}`, where `iso_2 = 8`** |
| no piece of any attaining record word is among the top 3 sizes of `M` | 4 of 4 rungs from 19->23 on (0 of 3 below) | measured, one out of sample |
| the interior pieces of every attaining record word are letters of `q'` or the pad `q'` | every word at every rung, 11->13 .. 31->37 | the chain law (docs/proofs/05 T2), cited; verified here on the complete word list |
| the attaining record words come in mirror pairs | every rung | measured |
| the enumerator reproduces the sieve's `m(v)`, `n1(v)`, `N(v)` | 20 of 20 cells | instrument, 0 mismatches |
| `D_top = 0` (the `F_2` pair contains a gap within 19% of the record) | 6 of 7 -- **REFUTED at `{5..31}`**, `D_top = 3` | measured; 51 of 60 family members |
| `Sig(v) <= F + 3` on the top band | 0 of 7 -- **REFUTED everywhere** (`c = 4..12`); true at `{5..37}` alone (`c = 2`) | refuted here |
