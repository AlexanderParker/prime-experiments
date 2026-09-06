# The wheels, second pass (branch R4.b.ii)

> Law numbers in this document map to the project-wide register
> (`research/proof/law_register.md`): **L22-L38 = W22-W38**.

Parent: R4.b, *The top machine on its own terms* (`research/proof/top_machine_1.md`, 21 laws
L1-L21, two open facts W1 and W2). The observation that spawned this branch is the owner's
direction of 2026-09-06: **understand the wheels fully before the clutch**. The first pass
established the top machine as a *domino machine* through pairwise laws (chain, merge, letters)
and one-gear laws (arcs, shield, clump). Three things it left standing: the pairwise laws were
never pushed to three and four gears at once; W1 (the gap-3 / gap-5 coincidence) had a residue
characterisation and no mechanism; W2 (the range record approaching the wheel record) was a
measurement at three values of `N` with no rule.

Construction rule (R4, owner): the top machine on the **raw line**, on its own, with the bottom
machine as inspiration only. No clutch in this branch, no interpretation against the twin
conjecture.

Numbering of laws continues from **L22**. Vocabulary is the first pass's, unchanged
(`docs/proofs/22-top-machine-laws.md`): gear, pair `n = (n, n + 2)`, tooth, strike, open, wheel
`W = prod g`, slot, run, step-2 chain, gap, record `F_top`, letters `{2, g - 2}`, domino,
shield `n = -1`, origin clump, mirror `n -> -n - 2`. `m = |G|`, `q'` = smallest gear.

---

## 1. Pre-registered predictions and scorecard

Written before any computation of this branch. Each carries the numbers it predicts and what
would refute it. Where a prediction is a formula derived by hand, the derivation is stated so
that the computation is a test and not a fit.

### Section 1. Tuples: three and four gears at once

**What "the real triple" means here.** The top machine's gears have **no free phase**. Every
gear's two teeth are pinned at `n = 0` and `n = -2` of the *same* raw line: gear `g` strikes the
pair `n` iff `g | n` or `g | n + 2`, and there is no shift to choose. A **real triple** is three
distinct primes `g < h < k` (in use: three consecutive primes above the split) with those pinned
teeth, considered modulo `g h k`. The only place the pinning is visible is the **origin**, where
all three gears' teeth coincide; everywhere else CRT decouples them.

**P1 (the joint census by CRT, exact and deviation-free).** For a real triple `(g, h, k)` the
`g h k` residues split as

        struck by all three                 8
        struck by exactly two               4 [(g-2) + (h-2) + (k-2)]
        struck by exactly one               2 [(h-2)(k-2) + (g-2)(k-2) + (g-2)(h-2)]
        struck by none (open)               (g-2)(h-2)(k-2)

and for a quadruple `16`, `8 sum (g-2)`, `4 sum_{pairs} prod`, `2 sum_{triples} prod`,
`prod (g-2)`. Predicted deviation from these CRT products: **exactly 0 for every real triple and
quadruple**, because CRT is a bijection `Z_{ghk} -> Z_g x Z_h x Z_k` and the pinned phases are
carried along by it. Refuted by any triple or quadruple whose exact counts differ from the
formula in any one of the four (five) classes.

**P2 (the origin is the unique total collision).** The residues where all `m` gears strike are
exactly `2^m` classes mod `W`, of which exactly two - `n = 0` and `n = -2` - have all `m` gears
using the *same* tooth, so that all `m` dominoes coincide. Predicted: those two classes are the
only ones at which every gear's whole domino agrees, and they are adjacent to the shield.
Refuted by a third such class in any wheel.

**P3 (the three-gear collision law, the analogue of the bottom's head collision).** In a window
of `L` consecutive positions with every gear `> L + 1`, each gear's trace is empty, a singleton
(cut by the window edge), or a domino `{x, x + 2}`. Prediction: **no three traces can pairwise
intersect unless two of them are equal.** Reason: pairwise intersecting dominoes at positions
`a <= b <= c` force `c - a <= 2` and all three positions lie in `{a, a + 2}`, so two of the three
are the same domino. Consequence predicted: in an optimal record cover for even `m` there is
**zero** multiply-covered position (a perfect tiling of `[0, 2m)` by `m` dominoes), and for odd
`m` exactly **one** unit of waste. Refuted by a three-way pairwise overlap with three distinct
dominoes, or by a record cover of even `m` with an overlap.

**P4 (records of triples and quadruples of odd primes 7..97).** The parity law L17 needs every
gear `> 2m + 1`; among the 22 odd primes `7..97` the only sub-threshold gear is `7`
(`m = 3`: threshold 7; `m = 4`: threshold 9; `11 > 9`). Predicted, exhaustively over all
`C(22,3) = 1540` triples and `C(22,4) = 7315` quadruples:

        triples     without 7:  F_top = 5      with 7:  F_top = 6
        quadruples  without 7:  F_top = 8      with 7:  F_top = 9

Refuted by one set with a different record. Mechanism predicted: the parity bound breaks exactly
when a gear is small enough to show its **long letter** `g - 2` inside the window
(`g <= L + 1`), and `g - 2` is **odd** for every odd gear, so the long letter is the only piece
that crosses parity. Gear 7's long letter is 5.

**P5 (the sub-threshold reduction: the shape of the rule).** More generally, for any gear set,

        F_top(G) depends only on m and on the sub-multiset of gears that are <= F_top + 1,

i.e. two gear sets with the same number of gears and the same small gears have the same record
whatever their large gears are. Predicted 0 exceptions over all `m = 3..8` sets of consecutive
primes with `q'` in `7..97` plus a sweep of non-consecutive sets. Refuted by two sets agreeing in
`m` and in their small gears but differing in `F_top`.

### Section 2. W1, the gap-3 / gap-5 coincidence

**The derivation, by hand, before computing.** Let the gap at `n` be `d`: `n` open, `n + d` the
next open pair. For each interior position `n + j` (`1 <= j <= d - 1`) some gear must strike it,
i.e. `n = -j` or `n = -(j + 2)` mod that gear; and `n` open forbids `n = 0, -2` mod every gear,
`n + d` open forbids `n = -d, -(d + 2)` mod every gear.

* `d = 3`. `n + 1` struck needs `n = -1` or `-3`; `-3` would strike `n + 3`, so **`n = -1`**.
  `n + 2` struck needs `n = -2` or `-4`; `-2` would strike `n`, so **`n = -4`**. Forbidden per
  gear: `{0, -2, -3, -5}`, four distinct classes for `g >= 7`. Required: some gear at `-1`, some
  gear at `-4`, both of which are among the allowed classes.
* `d = 5`. `n + 3` struck needs `n = -3` or `-5`; `-5` would strike `n + 5`, so **`n = -3`** -
  and that gear strikes `n + 1` too. `n + 2` struck needs `n = -4` (as `-2` strikes `n`) - and
  that gear strikes `n + 4` too. So the interior is covered by exactly those two requirements.
  Forbidden per gear: `{0, -2, -5, -7}`, four distinct classes for `g >= 11`. Required: some gear
  at `-3`, some gear at `-4`.

**P6 (W1 as a CRT identity).** Hence, by inclusion-exclusion over the two "some gear at class c"
requirements, with `a_g` the number of allowed classes of gear `g`,

        N_3 = prod (g - 4) - 2 prod (g - 5) + prod (g - 6)
        N_5 = prod (g - 4) - 2 prod (g - 5) + prod (g - 6)          (all gears >= 11)

**the same polynomial**, because the two forbidden sets have the same size `4` and the two marked
classes are allowed in both. Predicted: exact agreement with the measured counts in every wheel,
and equality of `N_3` and `N_5` whenever every gear is at least 11.

**P7 (why 7 is the only exception).** The forbidden set for `d = 3` is `{0, -2, -3, -5}`, whose
pairwise differences are `2, 3, 5, 1, 3, 2`; it collapses only for `g | 3` or `g | 5`, i.e.
`g = 3, 5` - never a gear. The forbidden set for `d = 5` is `{0, -2, -5, -7}`, whose pairwise
differences are `2, 5, 7, 3, 5, 2`; it collapses only for `g | 3, 5, 7`, and among gears
(`g >= 7`) **only for `g = 7`**, where `-7 = 0`. So gear 7 has `4 - 1 = 3` allowed classes for
`d = 3` and `4` for `d = 5`, and the two products separate. Predicted exactly:

        with 7 a gear:   N_3 = 3 A - 2 * 2 B + 1 * C,    N_5 = 4 A - 2 * 3 B + 2 * C

where `A = prod_{g != 7}(g - 4)`, `B = prod_{g != 7}(g - 5)`, `C = prod_{g != 7}(g - 6)`.
Predicted values at `{7, 11, 13}`: `N_3 = 3*7*9 - 2*2*6*8 + 1*5*7 = 189 - 192 + 35 = 32` and
`N_5 = 4*7*9 - 2*3*6*8 + 2*5*7 = 252 - 288 + 70 = 34`; at `{11, 13, 17}` both `= 819 - 1152 +
385 = 52`. Refuted by any wheel where the measured counts differ from these formulas.

**P8 (the whole gap census as a polynomial).** The same recipe gives every gap length. Predicted:

        N_1 = prod (g - 4)                      (= L15, the dominoes)
        N_2 = prod (g - 3) - prod (g - 5)
        N_4 = 0                                 (= L4: n + 2 struck forces n or n + 4 struck)
        N_6 = prod (g - 5) - 2 prod (g - 6) + prod (g - 7)

with `N_6`'s forbidden set `{0, -2, -6, -8}` plus the class `-4` forced (it alone strikes
`n + 2` and `n + 4`) - so `d = 6` has five forbidden classes and two further requirements.
(`N_6` is written here as the prediction to be checked; the derivation is given in Results.)
Refuted by any measured count differing from the polynomial.

**P9 (further exact coincidences).** Predicted: the only pairs of gap lengths with identically
equal counts for all large-gear sets are those whose (forbidden-set size, requirement structure)
agree. `3` and `5` is one such pair. Pre-registered guess: **no other pair below the record**,
i.e. `(3, 5)` is the unique coincidence. Refuted by a second coincident pair holding in every
wheel.

### Section 3. W2, the in-use machine and the approach to the wheel record

**P10 (the record on a prefix is set by the wheel's gap census).** For a fixed gear set with
wheel `W`, let `c(L)` be the number of gaps of length `> L` per period. Predicted: the longest
pair-free run on `[1, N]` is, to within one unit,

        F_range(N) = max { L : c(L) * N / W >= 1 } ,

i.e. the record climbs when `N` reaches `W / c(L)`. Predicted therefore **logarithmic in `N`**
with the slope set by the geometric decay of `c`, monotone non-decreasing (trivially, since a
prefix record cannot fall), and reaching the wheel record `F_top` only when `N` is of order
`W / (multiplicity of the record)`. Test: exact chunked scan of the gears `7..31` machine over
its **whole period** `W = 6,692,988,905` and of the gears `13..41` machine to `N = 10^10`,
recording the first occurrence of every record value. Refuted if the first-occurrence positions
disagree with `W / c(L)` by more than a factor of about 3, or if the record climbs in a way the
census does not predict.

**P11 (where the range record sits).** Predicted: for a **fixed** gear set the first occurrence
of each record value is at a position with no distinguished relation to the origin - the origin
clump (`n` in `[-6, 4]` for `q' = 7`) is the *most open* stretch, so the record cannot be there;
predicted position of the first record block as a fraction of the period: of order `1 / c(L)`,
i.e. scattered, not near 0. This is the opposite of the **in-use** machine (L21, gears `(q, Z]`),
whose record always sits in the gear zone `[1, Z]` immediately above the clump. Refuted if the
fixed machine's record blocks cluster near the origin or near multiples of the small gears'
wheel.

### Section 4. The smallest gears as an anchor

**P12 (there is no top-machine anchor, and the reason is a count).** The bottom machine's anchor
works because `2` and `3` leave **one** slot each (`g - 2 = 1` for `g = 3`, and for `g = 2` the
two teeth *collapse* - `0 = -2 mod 2` - leaving `1` slot), so `2, 3` together leave exactly one
residue class mod 6: the fold. Predicted for the top machine: the wheel of the two smallest gears
`q' q''` leaves exactly `(q' - 2)(q'' - 2)` slots, i.e. a **corridor of density
`(1 - 2/q')(1 - 2/q'')` >= 5/7 * 9/11 = 0.58**, and of the three smallest
`(q' - 2)(q'' - 2)(q''' - 2)`, density `>= 0.49`. Predicted: **no set of top gears can anchor**,
because an anchoring gear needs `g - 2 <= 2`, i.e. `g <= 4`. Refuted by a small top wheel whose
corridor is smaller than the CRT product.

**P13 (uniform descent).** Every open pair of every larger wheel lies in one of the small wheel's
`(q'-2)(q''-2)` slots, and each slot carries **exactly** `prod_{g > q''} (g - 2)` of them - no
slot is preferred. Predicted 0 deviation in every wheel tested. Refuted by an uneven descent.

**P14 (no preferred direction).** The gap word of the small wheel, read round the cycle from the
shield `n = -1`, is a **palindrome**: the mirror `n -> -n - 2` fixes the shield and reverses the
cycle, so the top machine's anchor candidate has no left/right asymmetry at all - unlike the
bottom's gear 5. Predicted 0 exceptions. Refuted by a non-palindromic gap word.

**P15 (the smallest gear is the anchor of the metric).** Predicted as a law: for every gear set,
the run ceiling `q' - 3`, the step-2 chain ceiling `q' - 2`, the arc structure and the width
`2(q' - 3) + 1` of the origin clump are functions of `q'` alone and of nothing else in `G`; and
the parity structure of the record (L17) is a function of `m` alone once `q' > 2m + 1`. So the
smallest gear fixes the *metric* and the gear count fixes the *record*; neither fixes the other.
Refuted by a set whose ceilings depend on a gear other than `q'`.

### Section 5. Self-similarity: the removal law

**P16 (the removal law).** Let `G' = G \ {q'}` (raising the split). Predicted, all exactly:

        W(G') = W(G) / q'                                   the wheel divides
        open(G') = open(G) / (q' - 2)                       the count divides
        dominoes(G') = dominoes(G) / (q' - 4)               the domino count divides
        run ceiling      q' - 3  ->  q'' - 3                grows
        chain ceiling    q' - 2  ->  q'' - 2                grows
        origin clump  2(q'-3)+1  ->  2(q''-3)+1             grows
        F_top            2m - (m mod 2)  ->  2(m-1) - ((m-1) mod 2)
                         i.e. -3 when m is even, -1 when m is odd, independent of which gear

and, the sharp form: **for a large-gear set, `F_top(G \ {g})` is the same for every choice of
`g`** - the record is a function of `|G|` alone. Predicted 0 exceptions. Also predicted: the
gap-3 count does **not** divide on removal (its polynomial is a sum of three products, not one),
so the census laws split into "divisible" (`N_1`) and "non-divisible" (`N_2, N_3, N_5, ...`).
Refuted by a removal that changes a ceiling, or by two removals from the same large-gear set
giving different records.

**P17 (nesting).** `Open(G) subset Open(G')` with ratio exactly `prod_{removed} (1 - 2/g)` over a
full period of `W(G)`. Predicted exact, 0 deviation.

### Scorecard

| # | Prediction | Result |
|---|---|---|
| P1 | triple/quadruple joint census = CRT product, deviation 0 | |
| P2 | the origin is the unique total collision (2 classes) | |
| P3 | no three distinct dominoes pairwise overlap; perfect tiling for even `m`, one unit of waste for odd `m` | |
| P4 | triples 5 / 6, quadruples 8 / 9 by whether 7 is a gear, exhaustive 1,540 + 7,315 | |
| P5 | `F_top` depends only on `m` and the gears `<= F + 1` | |
| P6 | `N_3 = N_5 = prod(g-4) - 2 prod(g-5) + prod(g-6)` for gears `>= 11` | |
| P7 | 7 is the unique gear that separates them (`-7 = 0`); 32 vs 34 at `{7,11,13}` | |
| P8 | the whole gap census as an explicit polynomial per length | |
| P9 | `(3, 5)` is the only coincident pair | |
| P10 | `F_range(N) = max{L : c(L) N / W >= 1}` | |
| P11 | fixed-set record blocks scattered, not at the origin | |
| P12 | corridor `(q'-2)(q''-2)`, density `>= 0.58`; no top anchor is possible | |
| P13 | uniform descent, 0 deviation | |
| P14 | the small wheel's gap word is a palindrome | |
| P15 | ceilings are functions of `q'` alone | |
| P16 | the removal law, exact; `F_top(G \ {g})` independent of `g` | |
| P17 | nesting with ratio `prod (1 - 2/g)`, exact | |

---

## 2. Setup as computed

Scripts in `research/topmachine/r2/`, results (untracked) in `.../results/`; the first pass's
`research/topmachine/r1/cover.py` is reused unchanged for every record.

| script | what it computes |
|---|---|
| `tuples.py` | the joint census of triples and quadruples, the total-collision classes, the collision law, the exhaustive records of all triples and quadruples from 7..97 |
| `records2.py` | the waste in a record cover; the corrected sub-threshold reduction |
| `gapcensus.py` | the gap census law, its universal signatures, moments, degrees; W1 |
| `inuse.py` | exact chunked prefix scans: the **whole period** of the eight-gear wheel `{7..31}` and prefixes of `{13..41}`, `{19..47}` |
| `w2.py` | the range record against the wheel census; the positions of the record blocks |
| `inuse2.py` | the in-use machine (gears `(q, Z]`) at `N = 10^7, 10^8, 10^9` |
| `anchor_removal.py` | the corridor, uniform descent, the palindrome, the metric anchor, the removal law |

Every count below is exact: either over a full wheel period (by direct enumeration or by the
census law, which is itself exact) or over the stated prefix.  The chunked scanner was checked
against a direct full-period enumeration at `{11, 13, 17}` before use (identical census).

---

## 3. Results

### 3.1 Tuples: the joint census of three and four gears (P1, P2)

"The real triple" is three distinct primes with their teeth pinned at `0` and `-2` of the same
raw line - the top machine has no free phase to choose.  The joint census is nevertheless
exactly the CRT product, because CRT is a bijection `Z_ghk -> Z_g x Z_h x Z_k` that carries
the pinned phases along with it.

| gears | struck by 0 / 1 / 2 / 3 (/ 4) gears, measured | CRT product | deviation |
|---|---|---|---|
| 7, 11, 13 | 495, 398, 100, 8 | same | 0 |
| 11, 13, 17 | 1485, 798, 140, 8 | same | 0 |
| 13, 17, 19 | 2805, 1214, 172, 8 | same | 0 |
| 17, 19, 23 | 5355, 1854, 212, 8 | same | 0 |
| 19, 23, 29 | 9639, 2766, 260, 8 | same | 0 |
| 23, 29, 31 | 16443, 3618, 308, 8 | same | 0 |
| 29, 31, 37 | 24003, 4514, 380, 8 | same | 0 |
| 7, 13, 31 | 1595, 1130, 236, 8 | same | 0 |
| 11, 19, 41 | 3159, 1922, 340, 8 | same | 0 |
| 7, 11, 13, 17 | 7425, 6960, 2296, 320, 16 | same | 0 |
| 11, 13, 17, 19 | 25245, 16536, 3976, 416, 16 | same | 0 |
| 13, 17, 19, 23 | 58905, 31104, 6040, 512, 16 | same | 0 |
| 17, 19, 23, 29 | 144585, 62286, 9436, 656, 16 | same | 0 |
| 7, 13, 19, 31 | 43095, 32266, 8348, 800, 16 | same | 0 |

**Total deviation over the fourteen tuples: 0.**  The counts are the ones pre-registered:
`8` all-struck, `4 sum (g-2)` exactly-two, `2 sum_pairs prod` exactly-one, `prod (g - 2)` open,
and the quadruple analogues with `16, 8, 4, 2`.

**The origin is the unique total collision (P2, held).**  All `m` gears strike simultaneously on
exactly `2^m` classes mod `W` - 14 of 14 wheels - and of those exactly **two**, `n = 0` and
`n = -2`, have every gear using the *same* tooth, so that all `m` dominoes coincide.  Those two
classes straddle the shield `n = -1`.  The origin clump (L6) is therefore not a separate fact: it
is the shadow of the machine's single point of total collision.  Every other all-struck class is
a *mixed* collision, where the gears share the position but not the domino.

### 3.2 The three-gear collision law (P3, held)

In a window of `L` consecutive positions with every gear `> L + 1`, a gear's trace is empty, a
singleton (cut by the window edge) or a domino `{x, x + 2}`.  Exhaustively over all windows
`L = 1..12` and all 1,354 unordered triples of distinct traces: **0 triples that pairwise
intersect.**  The one-line reason: pairwise intersecting distance-2 dominoes at positions
`a <= b <= c` force `c - a <= 2`, so all three positions lie in `{a, a + 2}` and two of the three
traces are the same trace.  This is the top machine's analogue of the bottom machine's head
collision, and it is much stronger: **three gears can share a position only if two of them are
doing exactly the same job.**

The consequence in the record.  With every gear large the record window is covered by `r`
distance-2 dominoes, `r = ceil(ceil(L/2)/2) + ceil(floor(L/2)/2)`:

| `m` | `L = F_top` | pieces | coverage | waste |
|---|---|---|---|---|
| 2 | 4 | 2 | 4 | **0** |
| 3 | 5 | 3 | 6 | **1** |
| 4 | 8 | 4 | 8 | **0** |
| 5 | 9 | 5 | 10 | **1** |
| 6 | 12 | 6 | 12 | **0** |
| 7 | 13 | 7 | 14 | **1** |
| 8 | 16 | 8 | 16 | **0** |
| 9, 10, 11, 12 | 17, 20, 21, 24 | 9, 10, 11, 12 | 18, 20, 22, 24 | 1, 0, 1, 0 |

**The record cover of an even-gear-count machine is a perfect tiling with no collision at all;
the odd case carries exactly one unit of waste.**  That single unit is the parity law's
`- (m mod 2)`, seen as geometry rather than as arithmetic: 11 of 11 cases, 0 exceptions.

### 3.3 The record of triples and quadruples, exhaustively (P4, held)

All `C(22, 3) = 1,540` triples and `C(22, 4) = 7,315` quadruples of odd primes `7..97`, each
record exact by the covering formulation:

| | `F_top = 5` | `F_top = 6` | `F_top = 8` | `F_top = 9` |
|---|---|---|---|---|
| triples | **1,330** (all without 7) | **210** (all with 7) | - | - |
| quadruples | - | - | **5,985** (all without 7) | **1,330** (all with 7) |

`1,330 = C(21,3)`, `210 = C(21,2)`, `5,985 = C(21,4)`, `1,330 = C(21,3)`.  **The record of a
three- or four-gear top machine takes exactly two values, and which one is decided by a single
bit: is 7 a gear.**  The sizes of the other gears - anywhere from 11 to 97 - make no difference
whatsoever.

**The mechanism.**  The parity bound (L17) works because a distance-2 domino never crosses
parity.  It breaks exactly when a gear is small enough to show its **long letter** `g - 2` inside
the window, i.e. when `g <= L + 1`; and `g - 2` is **odd** for every odd gear, so the long letter
is the only piece in the machine that crosses parity.  Gear 7's long letter is 5.  At `m = 3` the
threshold is `2m + 1 = 7` and at `m = 4` it is `9`, so among the primes `7..97` only 7 is
sub-threshold - hence exactly two values and exactly one bit.  Concretely at `m = 3`: `[0, 6)` is
covered by 7's long letter `{0, 5}` plus two dominoes `{1, 3}` and `{2, 4}`; at `L = 7` gear 7
repeats but still contributes only 2 positions, so `7 > 2 + 2 + 2` and the record stops at 6.

### 3.4 The sub-threshold reduction (P5, refuted as written, held when corrected)

As pre-registered the rule was tested with "small" meaning `g <= 2m + 3`; that failed, **12
exceptions in 70 cases** - because once `F` is much larger than `2m` a gear of size 19 or 23 is
still inside the window and still shows its long letter.  Re-tested in the self-consistent form -
`S = {g in G : g <= F(G) + 1}`, and only sets whose small part really is `S` are compared - the
rule holds: **90 comparable cases, 0 exceptions**, over `m = 3..8` with the large gears drawn
from four disjoint pools (`29..97`, `43..97`, `89..211`, `139..241`).

> **The record of a top machine is a function of two things only: how many gears it has, and
> which of its gears are no larger than the record itself.  The large gears enter only by their
> number.**

The rule tabulated (an extract; the full table is in `results/records2.json`), against the
parity-law value `2m - (m mod 2)`:

| `m` | small part `S` | `F_top` | parity law |
|---|---|---|---|
| 6 | - | 12 | 12 |
| 6 | 7 | 13 | 12 |
| 6 | 11 | 13 | 12 |
| 6 | 7, 13 | 16 | 12 |
| 6 | 7, 11, 13 | 19 | 12 |
| 7 | - | 13 | 13 |
| 7 | 11 | 16 | 13 |
| 7 | 11, 13 | 17 | 13 |
| 7 | 11, 13, 17 | 18 | 13 |
| 7 | 7, 11, 13, 17, 19 | 25 | 13 |
| 8 | - | 16 | 16 |
| 8 | 17 | 16 | 16 |
| 8 | 13 | 17 | 16 |
| 8 | 11, 13 | 20 | 16 |
| 8 | 11, 13, 17, 19 | 24 | 16 |
| 8 | 7, 11, 13, 19, 23 | 31 | 16 |

### 3.5 W1 solved: the gap census law and why 3 and 5 agree (P6, P7, P8, P9)

**The law.**  Fix `d >= 1`.  A gap of `d` at `n` means `n` open, `n + d` open, and every interior
position struck.  Per gear `g`, `n mod g` must avoid `{0, -2}` (from `n` open) and `{-d, -d-2}`
(from `n + d` open); and each interior position `n + j` needs *some* gear with `n = -j` or
`n = -(j+2)`.  Inclusion-exclusion over the set `S` of interior positions left uncovered gives,
exactly,

        N_d(G) = sum over S subset of [1, d-1] of  (-1)^|S|  prod_{g in G} ( g - |E_g(S)| )
        E_g(S) = ( {0, -2, -d, -d-2}  union  {-j, -(j+2) : j in S} )  mod g .

This is exact for any pairwise coprime gears (CRT).  Verified: **15 wheels, every gap length,
0 mismatches**, and against the exact full-period census of the **eight-gear** wheel `{7..31}`
(`W = 6,685,349,671`, `2,075,517,675` open pairs), `d = 1..16`, **0 mismatches** - including the
nine-figure counts `N_3 = 159,289,858` and `N_5 = 184,619,162`.

**The universal signatures.**  If every gear exceeds `d + 2` no class collapses, `|E_g(S)|` is
the same integer `e(S)` for every gear, and `N_d` is a universal polynomial in the gears:

        N_1 = prod(g-4)
        N_2 = prod(g-3) - prod(g-5)
        N_3 = prod(g-4) - 2 prod(g-5) + prod(g-6)
        N_4 = 0
        N_5 = prod(g-4) - 2 prod(g-5) + prod(g-6)          <-- identical to N_3
        N_6 = prod(g-4) - prod(g-5) - 3 prod(g-6) + 5 prod(g-7) - 2 prod(g-8)
        N_7 = prod(g-4) - 2 prod(g-5) - prod(g-6) + 4 prod(g-7) - prod(g-8) - 2 prod(g-9) + prod(g-10)
        ...   (computed to d = 16 in results/gapcensus.json)

**W1 is now a one-line identity.**  The forbidden set for `d = 3` is `{0, -2, -3, -5}` and for
`d = 5` is `{0, -2, -5, -7}`.  Both have **four** elements; in both cases the interior collapses
to **exactly two** requirements (`d = 3`: a gear on its shield `n = -1`, and a gear at `n = -4`;
`d = 5`: a gear at `n = -3`, which then strikes `n + 1` and `n + 3`, and a gear at `n = -4`,
which then strikes `n + 2` and `n + 4`); and in both cases the two marked classes are allowed.
Four forbidden and two marked in each - so the two inclusion-exclusions are the *same polynomial*
term by term.  **There is no bijection of `Z_W` behind it** (the first pass proved there is
none); the equality is an equality of class-count vectors, gear by gear.

**Why 7 and only 7 breaks it.**  A gear `g` can change `|E_g(S)|` only by dividing a difference
of two of the classes involved.  For `d = 3` those differences are `1, 2, 3, 5`, so only
`g = 2, 3, 5` could collapse - **never a gear**.  For `d = 5` they are `2, 3, 5, 7`, so among
gears (`g >= 7`) **only `g = 7`**, where `-7 = 0`.  Explicitly:

| gear | `d = 3` forbidden classes | allowed | `d = 5` forbidden classes | allowed |
|---|---|---|---|---|
| 7 | `{0, 2, 4, 5}` | **3** | `{0, 2, 5}` (collapsed: `-7 = 0`) | **4** |
| 11 | `{0, 6, 8, 9}` | 7 | `{0, 4, 6, 9}` | 7 |
| 13 | `{0, 8, 10, 11}` | 9 | `{0, 6, 8, 11}` | 9 |
| 17 | `{0, 12, 14, 15}` | 13 | `{0, 10, 12, 15}` | 13 |
| 19 | `{0, 14, 16, 17}` | 15 | `{0, 12, 14, 17}` | 15 |
| 23 | `{0, 18, 20, 21}` | 19 | `{0, 16, 18, 21}` | 19 |

With `A = prod_{g != 7}(g-4)`, `B = prod_{g != 7}(g-5)`, `C = prod_{g != 7}(g-6)` the two counts
become `N_3 = 3A - 4B + C` and `N_5 = 4A - 6B + 2C`, and they separate.  Measured, 16 of 16:

| wheel | `N_3` | `N_5` | equal | 7 a gear |
|---|---|---|---|---|
| 7, 11, 13 | 32 | 34 | no | yes |
| 11, 13, 17 | 52 | 52 | yes | no |
| 13, 17, 19 | 68 | 68 | yes | no |
| 17, 19, 23 | 88 | 88 | yes | no |
| 19, 23, 29 | 112 | 112 | yes | no |
| 23, 29, 31 | 136 | 136 | yes | no |
| 7, 11, 13, 17 | 538 | 590 | no | yes |
| 11, 13, 17, 19 | 1,162 | 1,162 | yes | no |
| 13, 17, 19, 23 | 1,978 | 1,978 | yes | no |
| 17, 19, 23, 29 | 3,386 | 3,386 | yes | no |
| 11, 17, 23, 29 | 2,522 | 2,522 | yes | no |
| 7, 13, 19, 31 | 1,562 | 1,658 | no | yes |
| 7, 11, 13, 17, 19 | 9,604 | 10,766 | no | yes |
| 11, 13, 17, 19, 23 | 28,196 | 28,196 | yes | no |
| 7, 11, 17, 23, 29 | 26,764 | 29,286 | no | yes |
| 7..31, eight gears, full period | 159,289,858 | 184,619,162 | no | yes |

In every wheel without 7 the common value is the universal polynomial; in every wheel with 7,
`N_3` is still the universal polynomial's value for the *other* gears times 7's collapsed
factors, and `N_5` is not.  **W1 is closed.**

**`(3, 5)` is the only coincidence (P9, held).**  Comparing universal signatures over
`d = 1..16`: the only pair of gap lengths with identically equal polynomials is `(3, 5)`; the
only identically zero length is `d = 4` (L4).

### 3.6 What the census law also explains: L18, L9, L4 and the parity law

**The degree, and universal record multiplicity (L18).**  Writing
`prod_g (g - e) = sum_k (-e)^k sigma_{m-k}(gears)` with `sigma` the elementary symmetric
polynomials, the census law becomes

        N_d = sum_{k=0..m} (-1)^k sigma_{m-k}(gears) M_k(d) ,    M_k(d) = sum_e c_e(d) e^k ,

where `c_e(d)` is the signature.  So `N_d` has degree `m - r(d)` in the gears, where `r(d)` is
the number of **vanishing moments** `M_0 = ... = M_{r-1} = 0`; and `N_d` is a pure constant,
independent of the gears entirely, exactly when `r(d) = m`, with value `(-1)^m M_m(d)`.
Measured `r(d)` and `(-1)^r M_r(d)`:

| `d` | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `r(d)` | 0 | 1 | 2 | - | 2 | 3 | 4 | 4 | 4 | 5 | 6 | 6 | 6 | 7 | 8 | 8 |
| `(-1)^r M_r` | 1 | 2 | 2 | - | 2 | 18 | 96 | 24 | 24 | 480 | 6480 | 1440 | 720 | 25200 | 645120 | 120960 |

Those constants are **exactly L18's numbers**: `18` at `m = 3`; `96, 24, 24` at `m = 4` (gaps
7, 8, 9); `480` at `m = 5`; `6480, 1440, 720` at `m = 6` (gaps 11, 12, 13).  Verified directly:
three disjoint gear sets of the same `m` (all gears `> d + 2`) give identical `N_d` exactly when
`r(d) >= m` - **0 mismatches** over `m = 3, 4, 5` and `d = 1..14`.  **L18 was a measured
coincidence in the first pass; it is now a corollary.**

**`r(d)` is the parity covering number.**  For every `d` from 1 to 16 except `d = 4`,

        r(d) = ceil(ceil((d-1)/2)/2) + ceil(floor((d-1)/2)/2) ,

the number of distance-2 dominoes needed to cover `d - 1` consecutive positions.  15 of 15.
Hence `F_top(m) = max{ d : r(d) <= m } - 1`, which reproduces the parity law `2m - (m mod 2)` at
`m = 2..7` from the census alone.  And `d = 4` is the unique place where the two covering
problems differ: the **record's** covering problem (L16) has *free* boundaries and `L = 3` is
coverable by a domino plus a singleton, while the **gap's** covering problem has *closed*
boundaries - the singleton's partner would have to fall on `n` or `n + 4`, which must be open.
**L4 is a boundary condition, not a counting accident.**  The two records agree everywhere else:
census `max d - 1` equals the cover's `F_top` in 15 of 15 wheels.

**L9 (odd gap census only at length 1) becomes arithmetic.**  Every gear is odd, so
`prod (g - e)` is odd iff `e` is even.  Hence `N_d` is odd iff `sum over even e of c_e(d)` is
odd.  Computed for `d = 1..16`: that sum is `1` at `d = 1` and `0, 2, 0, 2, -4, 0, -8, 8, 0, 32,
0, 32, -64, 0, -128` afterwards - **even at every `d > 1`**.  The first pass had L9 from the
mirror involution; this is a second, independent proof.

### 3.7 W2: the range record is a first hit on the wheel's own census (P10, P11)

The first pass measured the fixed-gear machine only to `N = 10^7` and concluded that the wheel
record is a ceiling approached slowly.  Scanned further - the **whole period** for `{7..31}` -
the picture is different and simple.

**The exact full-period census of the eight-gear wheel `{7, 11, 13, 17, 19, 23, 29, 31}`**
(`W = 6,685,349,671`; `2,075,517,675` open pairs `= prod (g - 2)` exactly):

| `d` | 1 | 2 | 3 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| count | 472,665,375 | 862,511,104 | 159,289,858 | 184,619,162 | 144,217,264 | 103,162,336 | 33,487,932 | 40,269,000 | 26,762,220 | 18,694,656 | 12,247,392 | 9,432,336 |

| `d` | 26 | 27 | 28 | 29 | 30 | 31 | 32 | 33 |
|---|---|---|---|---|---|---|---|---|
| count | 3,296 | 1,036 | 688 | 116 | 92 | 100 | 12 | **8** |

**The law.**  With `c(d)` the number of gaps `>= d` per period, the record on `[1, N]` is

        F_range(N) = max { d : W / c(d) <= N } - 1 ,

i.e. the record climbs when the prefix is long enough to expect one block of that length.
Measured against the predicted ladder:

| gears | `W` | 10^4 | 10^5 | 10^6 | 10^7 | 10^8 | 10^9 | 10^10 / period |
|---|---|---|---|---|---|---|---|---|
| 7..31 measured | 6.69e9 | 19 | 21 | 24 | 27 | 30 | **32** | 32 |
| 7..31 predicted | | 17 | 20 | 24 | 27 | 30 | 32 | 32 |
| 13..41 measured | 1.32e11 | 12 | 15 | 16 | 17 | **18** | 18 | 18 |
| 13..41 predicted | | 12 | 14 | 16 | 17 | 18 | 18 | 18 |
| 19..47 measured | 1.20e12 | 11 | 12 | 14 | 14 | **16** | 16 | 16 |
| 19..47 predicted | | 10 | 12 | 13 | 15 | 16 | 16 | 16 |

**Agreement to within one unit at 19 of 21 checkpoints, two units at the remaining two.**  The
first-occurrence positions sit within a factor of a few of `W / c(d)`: at `{7..31}` the ratio
`first / (W/c)` runs `0.26, 1.91, 0.52, 0.89, 1.46, 0.93, 1.52, 2.78, 0.87` over `d = 25..33` -
exponential-waiting scatter about 1, and nothing else.

**The wheel record is reached, and early.**  `{7..31}` reaches 32 at `N = 7.3e8`, 10.9% of its
period; `{13..41}` reaches 18 at `N = 4.9e7`, 0.037% of its period; `{19..47}` reaches 16 at
`N = 6.5e7`, 0.005% of its period.  W2's "hard ceiling approached slowly from below" was an
artefact of stopping at `N = 10^7`: **the record is reached at a fraction of the period equal to
about the reciprocal of the record's multiplicity, which is a tiny number.**  The approach is
monotone (a prefix record cannot fall) and logarithmic in `N`, about `+3` per decade for
`{7..31}` and `+1.5` per decade for `{13..41}`.

**Where the record blocks sit (P11, held, and sharpened).**  The eight record blocks of
`{7..31}` are at

        725,616,551    2,040,735,004    2,457,175,028    3,149,892,405
        3,535,457,233  4,228,174,610    4,644,614,634    5,959,733,087

- four mirror pairs, each pair summing to `W - 33` (the mirror `n -> -n - 2` acting on a block of
length 33).  They are nowhere near the origin: the first is at 10.9% of the period.  But they are
**not** free in the small gears:

| | mod 7 | mod 11 | mod 13 | mod 17 | mod 19 | mod 23 | mod 29 | mod 31 | mod 1001 |
|---|---|---|---|---|---|---|---|---|---|
| the 8 record blocks | 2 values | **1** | 2 | **1** | 2 | 2 | 4 | 4 | **2**: 308, 660 |
| the 12 blocks at `F - 1` | 2 | 2 | **1** | 6 | 2 | 2 | 6 | 6 | **2**: 465, 504 |

The record block's start is **pinned modulo the small gears** - a single residue mod 11 and mod
17 for all eight blocks, and only two residues mod `7 * 11 * 13 = 1001`, which are a mirror pair
(`308 + 660 = 968 = 1001 - 33`).  This is the covering formulation read backwards: the record
tiling assigns each small gear one specific piece, and that fixes its phase.

**The in-use machine is a different object (L21, extended).**  With gears `(q, Z]`,
`Z = floor(sqrt N)`, at `N = 10^7, 10^8, 10^9`:

| `q` | `N` | `Z` | gears | `F` | at | `at / N` | `at / Z` | `F / Z` |
|---|---|---|---|---|---|---|---|---|
| 5 | 10^7 | 3,162 | 443 | 3,006 | 161 | 1.6e-5 | 0.051 | 0.951 |
| 5 | 10^8 | 10,000 | 1,226 | 9,846 | 161 | 1.6e-6 | 0.016 | 0.985 |
| 5 | 10^9 | 31,622 | 3,398 | 31,560 | 161 | 1.6e-7 | 0.005 | **0.998** |
| 7 | 10^9 | 31,622 | 3,397 | 22,972 | 8,749 | 8.8e-6 | 0.277 | 0.726 |
| 11 | 10^9 | 31,622 | 3,396 | 12,120 | 19,601 | 2.0e-5 | 0.620 | 0.383 |
| 13 | 10^9 | 31,622 | 3,395 | 10,426 | 21,295 | 2.1e-5 | 0.673 | 0.330 |
| 17 | 10^9 | 31,622 | 3,394 | 6,289 | 13,311 | 1.3e-5 | 0.421 | 0.199 |
| 19 | 10^9 | 31,622 | 3,393 | 5,881 | 13,719 | 1.4e-5 | 0.434 | 0.186 |

**The in-use record sits at `10^-5` to `10^-7` of the range** - on the scale of `N` it is at the
origin - and it fills the gear zone: at `q = 5` the record run is `[161, 31,721]`, which is
99.8% of `[1, Z]`, starting immediately above the origin clump and ending just past `Z`.  At
larger `q` the record starts further into the zone and covers less of it (`F/Z` from `0.998` down
to `0.19`), because more small primes are left out of the gear set and more `q`-smooth pairs
survive.  The fixed-gear machine and the in-use machine therefore behave in opposite ways: the
first has its record scattered through a period at positions pinned only modulo its small gears;
the second has its record at a fixed spot at the bottom of the range, at `at/N -> 0`.

### 3.8 The smallest gears: a corridor, not an anchor (P12-P15)

**The corridor (P12, held).**  Every open pair of every larger wheel must sit in an open residue
of the smallest gears' wheel; that corridor is exactly the small wheel's own slot set:

| small wheel | `W_small` | corridor slots | `= prod (g - 2)` | density |
|---|---|---|---|---|
| 7, 11 | 77 | 45 | yes | 0.5844 |
| 11, 13 | 143 | 99 | yes | 0.6923 |
| 13, 17 | 221 | 165 | yes | 0.7466 |
| 17, 19 | 323 | 255 | yes | 0.7895 |
| 7, 11, 13 | 1,001 | 495 | yes | 0.4945 |
| 11, 13, 17 | 2,431 | 1,485 | yes | 0.6109 |
| 13, 17, 19 | 4,199 | 2,805 | yes | 0.6680 |

Set against the bottom machine's anchor, computed in the **same pair coordinate** (an arithmetic
remark about the formula `g - 2`, not a comparison of machines):

| gears | slots per turn | density | note |
|---|---|---|---|
| 2 | 1 | 0.5000 | `0 = -2 (mod 2)`: **the two teeth collapse**, so `g - 1`, not `g - 2` |
| 3 | 1 | 0.3333 | `g - 2 = 1` |
| 2, 3 | **1 slot mod 6** | 0.1667 | the fold |
| 5 | 3 | 0.6000 | |
| 2, 3, 5 | **3 slots mod 30** | 0.1000 | |

**An anchoring gear needs `g - 2 <= 2`, i.e. `g <= 4`.**  The primes that can anchor are exactly
2 and 3, and 2 only because it is the unique prime dividing the tooth separation, which collapses
its two teeth into one.  Every top gear has `g >= 7`, hence `g - 2 >= 5` slots and a corridor of
density at least `5/7 = 0.71` on its own.  **No set of the lowest top gears can play the role
2, 3, 5 play; the obstruction is a count, not an accident of which primes were chosen, and it
cannot be removed by choosing a different split.**

**Uniform descent (P13, held).**  In 5 of 5 wheels every corridor slot is occupied and each
carries exactly `prod_{g outside the small wheel} (g - 2)` open pairs of the larger wheel - 45
slots x 165 for `{7,11}` inside `{7,11,13,17}`, then 99 x 255, 165 x 357, 495 x 255, 1485 x 357.
**0 deviation.**  The corridor is inherited perfectly evenly: the small gears constrain *where*
the open pairs are and not at all *how many* sit in each place.

**No preferred direction (P14, held).**  The cyclic gap word of the small wheel, read from the
shield `n = -1`, is a **palindrome** in 6 of 6 wheels (`{7,11}`, `{11,13}`, `{7,11,13}`,
`{11,13,17}`, `{13,17,19}`, `{7,11,13,17}`); e.g. `{7,11,13}` begins
`2 1 1 1 2 2 2 5 1 1 1 5 2 2 2 1 ...`.  The mirror fixes the shield and reverses the cycle, so
the top machine's small wheel has no left/right asymmetry at all.  The bottom's gear 5 has one;
the top machine has nothing corresponding.

**The metric anchor (P15, held).**  In 14 of 14 wheels, spanning five values of `q'` and gears
chosen consecutively and non-consecutively, the longest run of open pairs is exactly `q' - 3`,
the longest step-2 chain exactly `q' - 2`, and the origin clump exactly `2(q' - 3) + 1` slots -
independent of every other gear (`{7,11,13}`, `{7,13,19}`, `{7,17,23}`, `{7,11,13,17}` all give
`4, 5, 9`; `{11,13,17}`, `{11,19,29}`, `{11,13,17,19}` all give `8, 9, 17`; and so on to
`{23,29,31}` at `20, 21, 41`).  **0 exceptions.**

So the two "anchors" of the top machine are different objects and neither is the bottom's: `q'`
alone fixes every local metric quantity, and `m` alone fixes the record.  Neither fixes the
other, and neither folds the line.

### 3.9 The removal law (P16, P17, held)

Raising the split from `q` to `q_2` removes the smallest gears.  One of the four chains, every
entry exact:

| gears | `W` | open | dominoes | run `<=` | chain `<=` | clump | `F_top` | parity value | large-gear regime |
|---|---|---|---|---|---|---|---|---|---|
| 11,13,17,19,23 | 1,062,347 | 530,145 | 233,415 | 8 | 9 | 17 | 10 | 9 | no (`11 = 2m+1`) |
| 13,17,19,23 | 96,577 | 58,905 | 33,345 | 10 | 11 | 21 | 8 | 8 | yes |
| 17,19,23 | 7,429 | 5,355 | 3,705 | 14 | 15 | 29 | 5 | 5 | yes |
| 19,23 | 437 | 357 | 285 | 16 | 17 | 33 | 4 | 4 | yes |
| 23 | 23 | 21 | 19 | 20 | 21 | 41 | - | - | - |

At every step of every chain (16 steps over four chains): `W` divides by exactly `q'`, the open
count by exactly `q' - 2`, the domino count by exactly `q' - 4`; the run ceiling, chain ceiling
and clump width **grow**, to `q'' - 3`, `q'' - 2`, `2(q'' - 3) + 1`; and `F_top` **falls** by 3
when `m` was even and by 1 when `m` was odd, matching `2m - (m mod 2)` at each rung whenever the
regime holds.  **Nesting** is exact: `Open(G)` is a subset of `Open(G \ {q'})` with density ratio
exactly `1 - 2/q'` (`0.818182` at `q' = 11`, `0.846154` at 13, `0.714286` at 7) - 3 of 3,
0 deviation.

**The sharp form (new).**  For a large-gear machine, `F_top(G \ {g})` is the **same for every
choice of `g`**: 9 of 9 sets tested (`{11,13,17,19}` and three siblings all `8 -> 5` whichever
gear goes; `{13,17,19,23,29}` and two siblings `9 -> 8`; `{17,19,23,29,31,37}` and
`{29,31,37,41,43,47}` both `12 -> 9`).  It fails exactly where the regime fails: `{7,11,13,17}`
gives 5 if 7 goes and 6 otherwise, `{7,11,13,17,19}` gives 8 if 7 goes and 9 otherwise.
**Removing a gear from a large-gear top machine costs the same whichever gear you remove, and the
cost alternates 3, 1, 3, 1 as `m` comes down.**

The census laws split under removal into two kinds.  `N_1 = prod (g - 4)` divides exactly, by
`q' - 4`.  `N_2, N_3, N_5, ...` are sums of several products and do **not** divide by anything -
their removal rule is term by term, not a single factor.

---

## 4. Laws

Numbered from L22, continuing `docs/proofs/22-top-machine-laws.md`.

**L22 (the gap census law).**  For pairwise coprime gears, the number of gaps of length `d` per
wheel is exactly

        N_d(G) = sum over S subset of [1, d-1] of  (-1)^|S|  prod_{g in G} ( g - |E_g(S)| ) ,
        E_g(S) = ( {0, -2, -d, -d-2} union {-j, -(j+2) : j in S} ) mod g .

*Proof:* the open/struck conditions at `n` and `n + d` are per-gear residue conditions; each
interior position needs some gear on one of two classes; inclusion-exclusion over the uncovered
interior positions, then CRT.  *Verified:* 15 wheels, every gap length, and the exact
full-period census of an eight-gear wheel to `d = 16` - **0 mismatches**.  *New.  It subsumes L4,
L9, L15 and L18 and supplies every gap count in closed form.*

**L23 (the universal signature, and the collapse threshold).**  If every gear exceeds `d + 2`
then `|E_g(S)| = e(S)` is independent of `g` and `N_d` is a universal polynomial in the gears
with signature `c_e(d)`.  A gear can depart from it only by dividing a difference of the classes
involved, and those differences are at most `d + 2`.  *Verified for `d <= 16`.  New.*

**L24 (W1, closed).**  `N_3` and `N_5` have the *same* universal signature,
`prod(g-4) - 2 prod(g-5) + prod(g-6)`, because both have four forbidden classes per gear
(`{0,-2,-3,-5}` and `{0,-2,-5,-7}`) and both reduce to exactly two "some gear here" requirements
(`-1` and `-4`; `-3` and `-4`).  The gap-3 classes can collapse only for `g | 3` or `g | 5`,
never a gear; the gap-5 classes collapse for `g = 7`, where `-7 = 0`.  **Hence the counts of gap
3 and gap 5 are equal in every wheel whose gears all exceed 7, and unequal exactly when 7 is a
gear.**  16 of 16 wheels including the eight-gear full period.  Moreover `(3, 5)` is the **only**
pair of gap lengths with identical universal polynomials, and 4 the only identically zero length,
for `d <= 16`.  *The first pass's W1 is proved.*

**L25 (the degree law, and universal record multiplicity).**  Writing the census law in
elementary symmetric polynomials, `N_d = sum_k (-1)^k sigma_{m-k} M_k(d)` with
`M_k(d) = sum_e c_e(d) e^k`.  `N_d` has degree `m - r(d)` in the gears, where `r(d)` is the
number of vanishing moments; it is **independent of the gears entirely** exactly when
`r(d) = m`, with value `(-1)^m M_m(d)`.  Those values are `18` (`m = 3`); `96, 24, 24` (`m = 4`);
`480` (`m = 5`); `6480, 1440, 720` (`m = 6`) - **L18 and its sub-record counts, now derived.**
*Verified:* 0 mismatches over `m = 3, 4, 5`, `d = 1..14`, three disjoint gear sets each.  *New;
L18 was measured in the first pass.*

**L26 (`r(d)` is the parity covering number; L4 is a boundary condition).**  For every `d <= 16`
except `d = 4`, `r(d) = ceil(ceil((d-1)/2)/2) + ceil(floor((d-1)/2)/2)`, the number of distance-2
dominoes needed to cover `d - 1` consecutive positions; hence
`F_top(m) = max{d : r(d) <= m} - 1`, reproducing the parity law.  `d = 4` is the unique exception
because the gap's covering problem has **closed** boundaries (the singleton's partner would have
to land on the open end) while the record's covering problem (L16) has free ones; the two agree
on the record itself in 15 of 15 wheels.  *New.*

**L27 (L9 by arithmetic).**  Every gear is odd, so `prod (g - e)` is odd iff `e` is even; hence
`N_d` is odd iff `sum over even e of c_e(d)` is odd, which happens only at `d = 1`.  `d <= 16`.
*Second, independent proof of L9.*

**L28 (the joint census of a tuple is the CRT product; the origin is the only total collision).**
For any gear set the residues mod `W` struck by exactly `j` gears number
`sum over the j-subsets of 2^j prod_{others}(g - 2)`; deviation 0 in 14 tuples.  All `m` gears
strike on exactly `2^m` classes, of which exactly **two** - `n = 0` and `n = -2` - have every
gear on the same tooth, so that all `m` dominoes coincide; those two straddle the shield.  **The
origin clump is the shadow of the machine's unique point of total collision.**  *The count is
standard CRT bookkeeping; the identification of the origin as the unique total collision is new.*

**L29 (the collision law, and the waste in a record).**  In a window with every gear `> L + 1`,
no three distinct traces pairwise intersect: three gears can share a position only if two of them
have the same trace.  *Proof:* pairwise intersecting distance-2 dominoes lie within a span of 2,
so only two distinct dominoes fit.  0 exceptions over all windows `L <= 12` and 1,354 triples.
*Consequence:* the record cover is a **perfect tiling with zero collision when `m` is even**, and
carries **exactly one unit of waste when `m` is odd** - 11 of 11, `m = 2..12`.  That unit is the
parity law's `-(m mod 2)` in geometric form.  *New; the analogue of the bottom's head collision,
and much stronger.*

**L30 (the record of a triple or a quadruple).**  Over all 1,540 triples and 7,315 quadruples of
odd primes 7..97:

        triples:     F_top = 5 if 7 is not a gear (1,330 sets),  6 if it is (210 sets)
        quadruples:  F_top = 8 if 7 is not a gear (5,985 sets),  9 if it is (1,330 sets)

**0 exceptions.**  The sizes of the other gears never matter.  *Mechanism:* the long letter
`g - 2` is odd, so it is the only piece that crosses parity, and a gear can show it only when
`g <= L + 1`; at `m = 3, 4` the only such gear among 7..97 is 7.  *New.*

**L31 (the sub-threshold reduction).**  `F_top(G)` is a function of `m = |G|` and of the
sub-multiset `{g in G : g <= F_top(G) + 1}` alone; the larger gears enter only by their number.
**90 comparable cases, 0 exceptions**, `m = 3..8`, large gears from four disjoint pools.  *New;
it is the exact statement of which the first pass's parity law is the large-gear special case.*

**L32 (the range record is a first hit on the wheel's census).**  For a fixed gear set,
`F_range(N) = max{d : W / c(d) <= N} - 1` with `c(d)` the number of gaps `>= d` per period.
Agreement within one unit at 19 of 21 checkpoints over three eight-gear machines and
`N = 10^4 .. 10^10`; first-occurrence positions within a factor of about 3 of `W / c(d)`.
Consequently the wheel record is **reached**, at about `W` over its multiplicity: 10.9% of the
period for `{7..31}`, 0.037% for `{13..41}`, 0.005% for `{19..47}`.  *New; W2's "the wheel record
is a hard ceiling approached slowly" was an artefact of stopping at `N = 10^7`.*

**L33 (the record blocks are pinned modulo the small gears).**  The eight record blocks of the
`{7..31}` wheel occupy exactly **two** residues mod 1001 (a mirror pair, 308 and 660), one
residue mod 11, one mod 17; the twelve blocks at `F - 1` occupy two residues mod 1001 and one mod
13.  They form mirror pairs summing to `W - 33`.  *Measured, exact over a full period; new.*

**L34 (the corridor, and why the top machine can have no anchor).**  The residues mod the
smallest gears' wheel that every open pair of every larger wheel must occupy are exactly that
wheel's own `prod (g - 2)` slots, of density `prod (1 - 2/g) >= (5/7)(9/11) = 0.58` for two
gears; and every slot carries exactly `prod_{larger} (g - 2)` open pairs of the larger wheel -
the **uniform descent**, 5 of 5 wheels, 0 deviation.  An anchoring gear (one that folds the line)
needs `g - 2 <= 2`, i.e. `g <= 4`; the only prime with collapsed teeth is `g = 2`, the unique
prime dividing the separation.  **No set of top gears can anchor, for any split.**  *New as a
statement; the counts are CRT.*

**L35 (no direction).**  The cyclic gap word of a top wheel, read from the shield `n = -1`, is a
palindrome; 6 of 6 wheels.  The mirror fixes the shield and reverses the cycle.  *New.*

**L36 (the metric anchor).**  The longest run `q' - 3`, the longest step-2 chain `q' - 2` and the
origin clump `2(q' - 3) + 1` are functions of the smallest gear alone, and the record is a
function of the gear count alone (L31 in the large-gear regime).  14 of 14 wheels, 0 exceptions.
**The smallest gear is the anchor of the metric and of nothing else; the gear count is the anchor
of the record and of nothing else.**  *The three ceilings are L6/L10 restated; that they are the
machine's only anchoring, and their separation from the record, is new.*

**L37 (the removal law).**  Removing the smallest gear `q'` from `G`: `W` divides by `q'`, the
open count by `q' - 2`, the domino count `N_1` by `q' - 4`, exactly; `Open(G)` is a subset of
`Open(G \ {q'})` of density ratio exactly `1 - 2/q'`; the run, chain and clump ceilings **grow**
to `q'' - 3`, `q'' - 2`, `2(q'' - 3) + 1`; and `F_top` falls by 3 when `m` is even and by 1 when
`m` is odd.  16 removal steps over four chains and 3 of 3 nesting tests, 0 deviation.  *New as a
law; each ingredient except the `F` step is a restatement.*

**L38 (removal independence).**  In the large-gear regime `F_top(G \ {g})` is the same for every
choice of `g`: 9 of 9 sets.  It fails exactly outside the regime (`{7,11,13,17}`: 5 if 7 goes,
6 otherwise).  **The record does not know which gear left, only that one did.**  *New; the sharp
converse of the parity law.*

---

## 5. What is new

Against the first pass's 21 laws and its two open facts:

| first pass | second pass |
|---|---|
| W1: gap-3 and gap-5 counts equal unless 7 is a gear; **mechanism open**, and no bijection of `Z_W` exists | **closed** (L22-L24): both counts are the same inclusion-exclusion over the same shape of class data - four forbidden classes and two marked - and 7 is the unique gear that can collapse the gap-5 classes, because `-7 = 0` |
| W2: the range record climbs slowly toward the wheel record and does not reach it | **replaced** (L32): the record is a first hit on the wheel's own gap census, `F_range(N) = max{d : W/c(d) <= N} - 1`; the wheel record **is** reached, at about `W` over its multiplicity - 0.005% to 11% of the period |
| L4, the forbidden gap 4, proved directly | **explained** (L26): `d = 4` is the unique place where the gap's covering problem (closed boundaries) differs from the record's (free boundaries) |
| L9, even gap counts except length 1, by the mirror involution | **second proof** (L27), arithmetic: `N_d` is odd iff a signature sum over even `e` is odd |
| L15, dominoes `prod (g - 4)` | the `d = 1` case of L22 |
| L18, universal record multiplicity 18, 24, 480, 720, **measured, no mechanism** | **derived** (L25): they are `(-1)^m M_m(d)`, the top moment of the census signature, and `N_d` is gear-independent exactly when the number of vanishing moments reaches `m` |
| L17, the parity law for gears `> 2m + 1` | **generalised** (L31): `F_top` depends on `m` and on the gears `<= F + 1`; the parity law is the case where that set is empty.  Also (L26) re-derived from the census, and (L29) given a geometric form: even `m` is a perfect tiling, odd `m` wastes exactly one unit |
| L6, the origin clump | **explained** (L28): the origin is the machine's unique point of total collision - the only place where all `m` dominoes coincide |
| "its own anchor is the wheel of its smallest gears" (the construction rule's open question) | **answered, negatively and exactly** (L34): the corridor is the small wheel's own slots, of density at least 0.58, uniformly filled; anchoring needs `g <= 4`; and the small wheel is a palindrome (L35), so there is no direction either.  What the smallest gear does instead is fix the metric (L36) |
| - | new: the collision law (L29), the exhaustive triple and quadruple records (L30), the pinning of record blocks modulo the small gears (L33), the removal law and removal independence (L37, L38) |

**Prior art met and stopped.**  The counts `prod (g - 2)`, `prod (g - 3)`, `prod (g - 4)` and the
CRT arguments behind L22 are the standard Hardy-Littlewood / Schemmel local factors and ordinary
inclusion-exclusion; the record itself is a two-class Jacobsthal function (Jacobsthal 1961,
Iwaniec 1978 for the one-class bound).  Nothing here is an asymptotic.  What L22-L26 add is
**exact structure at a fixed gear set**: a closed form for every gap length, its degree in the
gears, and the identification of the gear-independent regime - which is where the record and its
multiplicity live.  No literature search has been run for the top machine as an object; all of
the above is prior art **not checked** except the two lines just given.

---

## 6. Verdict

**The top machine's gap census is a single closed formula, and everything the first pass could
not explain falls out of it.**  L22 gives the number of gaps of any length as an
inclusion-exclusion over which interior positions go uncovered, evaluated by CRT gear by gear.
Read as a polynomial in the gears it has a degree, `m - r(d)`, and a signature; the degree
collapses to zero exactly at the top of the census, which is why the record's multiplicity is a
pure number (L25 = L18) and why the record itself is a counting-and-parity quantity (L26 = L17).
Read gear by gear it says that the gap-3 and gap-5 counts are the same polynomial, and that 7 -
the smallest possible gear, and the only gear dividing 7 - is the only thing that can break the
identity (L24 = W1, closed).  Read at the boundary it says why a gap of 4 is impossible while a
struck run of 3 is not (L26).

**Three and four gears at once add nothing to the counting and everything to the geometry.**  The
joint census is exactly the CRT product with zero deviation, because the gears' pinned phases are
carried by the CRT bijection; the pinning shows only at the origin, which is the unique class
where all `m` dominoes coincide and is therefore the machine's one point of total collision
(L28) - and the origin clump is its shadow.  Three gears can never share a position without two
of them doing the same job (L29), so a record cover is a perfect domino tiling when `m` is even
and wastes exactly one position when `m` is odd.  Over every triple and quadruple of odd primes
from 7 to 97 the record takes exactly two values, decided by one bit: is 7 a gear (L30).

**The record knows two numbers and nothing else**: how many gears there are, and which of them
are no bigger than the record (L31).  Removing a gear from a large-gear machine therefore costs
the same whichever gear leaves (L38), and the whole machine transforms under a raised split by a
short list of exact divisions and growths (L37).

**There is no top-machine anchor, and the obstruction is a count.**  Folding the line requires a
gear leaving at most two slots, i.e. `g <= 4`; the top machine's gears leave `g - 2 >= 5`, so its
smallest gears give a corridor of density above 0.58, uniformly filled, with a palindromic
pattern and no direction (L34, L35).  What the smallest gear does instead is fix every local
metric - run `q' - 3`, chain `q' - 2`, clump `2(q' - 3) + 1` - and nothing else (L36).  The two
anchors of the top machine, `q'` for the metric and `m` for the record, are independent.

**On a range the fixed machine reaches its wheel record early** - at about `W` over the record's
multiplicity, which is 0.005% to 11% of the period - and gets there by first-hit statistics on
its own census (L32); its record blocks are scattered through the period but pinned to one or two
residues modulo its smallest gears (L33).  The in-use machine is a different object: its record
sits at `10^-5` to `10^-7` of the range, immediately above the origin clump, and at `q = 5` it
fills 99.8% of the gear zone.

No interpretation against the twin conjecture is offered; no clutch.

---

## 7. Scorecard, filled

| # | Prediction | Result |
|---|---|---|
| P1 | triple/quadruple joint census = CRT product, deviation 0 | **held**, 14 tuples, total deviation 0 |
| P2 | the origin is the unique total collision (2 classes) | **held**, 14 of 14 (`2^m` all-struck classes; exactly `n = 0, -2` with identical dominoes) |
| P3 | no three distinct dominoes pairwise overlap; perfect tiling for even `m`, one unit of waste for odd `m` | **held**, 0 of 1,354 triples, `L <= 12`; waste rule 11 of 11 |
| P4 | triples 5 / 6, quadruples 8 / 9 by whether 7 is a gear | **held**, exhaustive: 1,330 / 210 and 5,985 / 1,330, 0 exceptions |
| P5 | `F_top` depends only on `m` and the gears `<= F + 1` | **refuted as written** (threshold `2m + 3`: 12 exceptions in 70); **held in the self-consistent form**, 90 cases, 0 exceptions |
| P6 | `N_3 = N_5 = prod(g-4) - 2 prod(g-5) + prod(g-6)` for gears `>= 11` | **held**, every wheel; and generalised to the whole census (L22) |
| P7 | 7 is the unique gear that separates them; 32 vs 34 at `{7,11,13}` | **held** exactly, including 159,289,858 vs 184,619,162 at the eight-gear wheel |
| P8 | the gap census as an explicit polynomial per length | **held for the law**; the hand-guessed `N_6` was **wrong** (the true signature is `+1, -1, -3, +5, -2` at `e = 4..8`, not `+1, -2, +1` at `e = 5..7`) |
| P9 | `(3, 5)` is the only coincident pair | **held** for `d <= 16` |
| P10 | `F_range(N) = max{d : W/c(d) <= N} - 1` | **held**, within 1 at 19 of 21 checkpoints, within 2 at the other 2 |
| P11 | fixed-set record blocks scattered, not at the origin | **held** (first at 10.9% of the period) and **sharpened**: pinned to 2 residues mod 1001 |
| P12 | corridor `(q'-2)(q''-2)`, density `>= 0.58`; no top anchor possible | **held**, 7 of 7 small wheels |
| P13 | uniform descent, 0 deviation | **held**, 5 of 5 |
| P14 | the small wheel's gap word is a palindrome | **held**, 6 of 6 |
| P15 | ceilings are functions of `q'` alone | **held**, 14 of 14 |
| P16 | the removal law, exact; `F_top(G \ {g})` independent of `g` | **held**, 16 steps, 4 chains; independence 9 of 9 in the regime, failing exactly outside it |
| P17 | nesting with ratio `prod (1 - 2/g)`, exact | **held**, 3 of 3 |

---

## 8. Holds without exception (the count)

Every statement below was tested exhaustively over the stated range with **zero** exceptions.

| statement | range | exceptions |
|---|---|---|
| L22 census law against exact enumeration | 15 wheels, every gap length; plus an eight-gear wheel's full period, `d = 1..16` | **0** |
| L24 `N_3 = N_5` iff every gear `> 7` | 16 wheels including a full 6.7e9 period | **0** |
| L25 `N_d` gear-independent iff `r(d) >= m`, value `(-1)^m M_m` | `m = 3, 4, 5`; `d = 1..14`; three disjoint gear sets | **0** |
| L26 `r(d)` = the parity covering number | `d = 1..16`, `d != 4` | **0** |
| L26 census record = cover record | 15 wheels | **0** |
| L27 `N_d` odd only at `d = 1` | `d = 1..16` | **0** |
| L28 joint census = CRT product | 14 tuples (triples and quadruples) | **0** |
| L28 `2^m` all-struck classes, exactly 2 with identical dominoes | 14 tuples | **0** |
| L29 no three distinct traces pairwise overlap | windows `L <= 12`, 1,354 triples | **0** |
| L29 waste 0 for even `m`, 1 for odd `m` | `m = 2..12` | **0** |
| L30 record of a triple / quadruple | 1,540 + 7,315 sets, exhaustive over primes 7..97 | **0** |
| L31 sub-threshold reduction | 90 comparable cases, `m = 3..8` | **0** |
| L33 record blocks in mirror pairs, pinned mod 1001 | full period of `{7..31}`, 8 + 12 blocks | **0** |
| L34 corridor `= prod (g - 2)` | 7 small wheels | **0** |
| L34 uniform descent | 5 wheels | **0** |
| L35 palindromic gap word from the shield | 6 wheels | **0** |
| L36 run `q'-3`, chain `q'-2`, clump `2(q'-3)+1` from `q'` alone | 14 wheels | **0** |
| L37 removal: `W`, open, dominoes divide; ceilings grow; `F` step | 16 steps, 4 chains | **0** |
| L37 nesting ratio `1 - 2/q'` | 3 wheels | **0** |
| L38 `F_top(G \ {g})` independent of `g` in the large-gear regime | 9 sets | **0** |
| L5 open pairs `= prod (g - 2)` over a full 6.7e9 period | 1 eight-gear wheel | **0** |

Nearly-exceptionless, stated with its slack: **L32**, the range record as a first hit, agrees
within one unit at 19 of 21 checkpoints and within two at the other two.

---

## 9. Dead ends

* **The hand-derived `N_6`.**  Pre-registered as `prod(g-5) - 2 prod(g-6) + prod(g-7)` on the
  guess that `d = 6` would also reduce to two clean requirements.  It does not: the interior of a
  gap of 6 has a genuine disjunction (position `n + 1` can be served from `-1` or from `-3`), and
  the true signature is `+1, -1, -3, +5, -2` at `e = 4..8`.  What survived is the general recipe
  (L22), which needs no case analysis at all.
* **"`F_top` depends on `m` and the gears below `2m + 3`."**  Refuted, 12 exceptions in 70: at
  `m = 7` with small part `{7, 13}` the record is 20 with large gears `19..37` and 19 with large
  gears `73..97`, because 19 and 23 are themselves inside a window of length 20.  The threshold
  is not `2m + 3` but the record itself; restated that way the law holds with 0 exceptions
  (L31).
* **The census law by transfer matrix or DP.**  A dynamic program over gears with state "which
  interior positions are covered" has `2^{d-1}` states and is no cheaper than the
  inclusion-exclusion; for `d > 20` neither is usable, so the long tail of the `{7..31}` census
  (`d = 17..33`) is recorded from the exact scan and not from the formula.
* **Searching for a bijection behind W1** (inherited from the first pass).  Not re-entered: the
  first pass had already refuted every shift and every reflection of `Z_W`, and L24 shows why -
  the equality is an equality of per-gear class counts, not of sets, and the two class sets
  `{0,-2,-3,-5}` and `{0,-2,-5,-7}` are genuinely different subsets of `Z_g`.

---

## 10. What the next pass could take up

Left open, stated plainly and not worked:

* **L22 in the kernel.**  The census law is a written proof (CRT plus inclusion-exclusion) and is
  the natural next Lean target: it would carry L4, L9, L15 and L18 with it, and it needs the same
  `Finset`-indexed CRT lemma that the first pass's two open items (L8's group count, L17's
  attainment) are blocked on.
* **The vanishing moments.**  `M_k(d) = 0` for `k < r(d)` is verified to `d = 16` and not proved;
  a proof would make L25 (and therefore L18) a theorem, and would probably also give `r(d)` in
  closed form and hence a second proof of the parity law.
* **The census beyond `d = 20`.**  Both known routes are exponential in `d`; the long tail of a
  wheel's census is currently only obtainable by scanning the period.
* **L31 as a formula.**  The reduction says `F_top` is a function of `m` and the small gears; the
  function itself is tabulated, not derived.  The mechanism (only the odd long letter crosses
  parity) suggests a parity-refined covering bound per small gear, which was not attempted.
