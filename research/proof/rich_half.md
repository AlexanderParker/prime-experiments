# Node R4.c.ii - THE RICH HALF OF THE RECORD

Parent: the unstick pass `research/proof/dead_branches_reopened_3.md`, section "The engine's
skip half", whose reading of E2 (`pad_cap.md` 4.1) spawned this branch: a record of `M + q'` is
a poor interval of `M` whose openings, all on the two teeth of `q'`, form a rich interval of the
PULLBACK `M^(q')` (same gears, separations `2 u_g q'^-1 mod g`) in multiplier coordinates. The
one object in that pass that is not the record restated is the rich-interval function
`Omega(n)`, the most openings of the pullback in `n` consecutive multipliers over all translates.
This branch reopens the ENGINE: the pullback with EVERY gear of `M`, the record's translate, the
collision laws of `docs/proofs/21` under the twisted separations, and the record's start class.

Scripts in `research/anchor235/r68/` (prefix `rh_`), outputs in `research/anchor235/r68/results/`
(untracked). Every number the document relies on is in the document. Laws found here continue
the engine's series from **E4**.

Vocabulary (owner's, canonical). ENGINE = the primes up to `q` in the anchored column coordinate
(column `k = (6k - 1, 6k + 1)`; gear `g` strikes `k` iff `k = +-u_g mod g`, `u_g = 6^-1 mod g`);
the record `F(M)`; the ladder `M -> M + q'`; letters `a = 2u mod q'` reduced to `min(d, q' - d)`,
`b = q' - a`; the skip half (letters `2q'`, `a + q'`, `b + q'`); the PULLBACK `M^(q')`: the
machine with the same gears and separations `2 u_g q'^-1 mod g`, whose openings in `n`
consecutive multipliers are what a word of `M + q'` along the new gear's tooth progression sees;
`Omega(n)` = the rich-interval function = the most openings of the pullback in `n` consecutive
multipliers over all translates; E2/E3 = its exact caps on the skip half (`pad_cap.md`).

---

## 0. Pre-registered (written before any computation of this branch)

### 0.1 The objects, exactly

- `M = {5..q}`, `q'` the next prime, `u = 6^-1 mod q'`, `d = 2u mod q'`, `{a, b} = {d, q' - d}`.
- The **pullback** `M^(q')`: for a column `x_0` and a class offset `s_j` (`s_0 = 0`, `s_1 = s in
  {a, b}`), the multiplier `m` is OPEN iff the column `x_0 + s_j + m q'` is open in `M`; gear `g`
  forbids the two residues `m = (+-u_g - x_0 - s_j) q'^-1 mod g`, at separation `s'_g = 2 u_g
  q'^-1 mod g`. Its **twisted arc** is `a'_g = min(s'_g, g - s'_g)`. In column coordinates the
  pullback is the machine read along the slots `x_0 + s_j + m q'`, and this branch computes in
  column coordinates throughout (no inverses are needed; the twist is the slot set).
- **`Omega^full(n)`** = the most openings of the pullback of `M` in `n` consecutive multipliers
  (slots `0, q', 2q', ..., (n-1) q'`) over all translates `x_0 mod P(M)`. By the one-orbit
  reduction (`docs/proofs/21`) the translates are all phase vectors, so `Omega^full(n)` = the
  largest subset `O` of the `n` slots such that every gear `g` has a phase avoiding `O`. A gear
  `g >= 2n + 1` can always be phased off `n` slots (the complement of `n` residues has `g - n >=
  n + 1` residues, a cyclic interval long enough to hold any arc `<= (g - 1)/2`), so **only the
  gears `g <= 2n` of `M` bite**, and `Omega^full(n)` depends on `n` and on `q' mod (gears <=
  2n)` alone. The corridor value `Omega_{5,7}(n; q' mod 35)` of `pad_cap.md` 4.1 is the same
  object with gears 5 and 7 only.
- **The joint two-class value** `Omega^(2)(T; q', s)` = the largest feasible subset of the slot
  set `{m q' : 0 <= m <= T} u {s + m q' : 0 <= m <= T_1}`, `T_1 = floor((F(M+q') - 2 - s)/q')`,
  over all translates; the sharpened E2 reads `L + 1 <= max_{s in {a, b}} Omega^(2)`.
- **The record's windows.** A record stretch of `M + q'` of length `G = F(M + q')` is
  `(c, c + G)` with `c`, `c + G` open in `M + q'`; its fusion `(f_L, w_1, ..., w_L, f_R)` gives
  the openings of `M` inside, `x_0 = c + f_L`, `x_i = x_0 + w_1 + ... + w_i`, all on the two
  teeth of `q'`. The class-`j` **window** `W_j` is the set of multipliers `m` with
  `-f_L < s_j + m q' < G - f_L` (the tooth progression's slots strictly inside the stretch),
  `n_j = |W_j|`; the openings of the pullback at the record's translate in `W_j` are exactly the
  word's class-`j` openings, `|S_j|`, with `|S_0| + |S_1| = L + 1`. **The record sits at a richest
  translate** iff `L + 1 = Omega_slots`, where `Omega_slots` is the largest feasible subset of the
  slot set `W_0 u W_1` (in column offsets). The **deficit** is `Omega_slots - (L + 1)`; the
  **rank** of the record's translate is the fraction of the `P(M)` translates whose opening count
  on those slots is `>= L + 1` (exact, by a union DP over the gears' phases).
- **Collision deficits on the pullback** (`docs/proofs/21`): `c(g, h; L) = max_g(L) + max_h(L) -
  joint_max(g, h; L)` with the twisted separations; the arc floor asks `c = 0` for `2 <= L <=
  max(a'_g, a'_h)`. The rich-direction analogue: `min_g(n)` = the fewest of `n` consecutive
  multipliers gear `g` must strike, `joint_min(g, h; n)` likewise for the pair, and the
  **coincidence** `k(g, h; n) = min_g(n) + min_h(n) - joint_min(g, h; n) >= 0`. The **stacking
  deficit** `D(n) = (n - Omega^full(n)) - max_g min_g(n) >= 0`: zero iff every gear's forced
  strikes can be stacked on the strikes of the most-constrained gear.

**Facts cited, not re-derived.** E1-E3 and the skeleton (`pad_cap.md`); the record fusions with
their phases (`pad_cap.md` 2.6, `r61/results/detail_K14_base23.json`; 37 -> 41: `(15, 41, 14, 21)`
at phase 19, `docs/proof-search/mechanic.md`); the collision laws and the one-orbit reduction
(`docs/proofs/21`); the order law and `Phi = B_{L+1}` (`monotone_functional.md` 4, 6); the
corpus `F = 25, 34, 43, 58, 88, 91, 103, 118, 145, 161` at m19..m59 and `L = 2, 1, 3, 3, 2, 2, 2,
4, 3` at m19..m53.

### 0.2 The theory

**T.** The record of `M + q'` is a two-coordinate object: poorest along the column (no opening
of `M` off the teeth of `q'` for `G - 1` columns) and rich along the new gear's stride (the few
openings inside all lie on two tooth progressions, i.e. they are openings of the pullback in
short multiplier windows). `Omega^full` caps the rich half exactly and uniformly in the rung
(only gears `<= 2n` bite); whether the record actually USES the richest translate decides
whether the record's position is a formula in a growing modulus (P3) or a trade-off between the
word's richness and the flanks' length (my own P5).

### 0.3 Predictions, each with the number that would refute it

From the unstick file (the owner's brief; adopted as the scorecard):

- **P1.** `Omega^full(n) < Omega_{5,7}(n)` (the corridor value) first at some `n_0(q') <= 12` at
  every rung m19..m53, decided by the gears 11..23 whose pullback arcs exceed `n_0/2`. REFUTED
  by a rung with `n_0 > 12`, or by a drop that gears 11..23 do not produce.
- **P2.** The sharpened cap `L + 1 <= 2 Omega^full(T + 1)` is tight at `>= 5` of 9 corpus
  rungs. REFUTED by tightness at `<= 4` rungs.
- **P3.** The record stretch sits at a richest translate (`L + 1 = Omega_slots`) at every rung
  from m23 on. REFUTED at one rung with the deficit named.

My own, written before computing:

- **P4 (the sharpening is empty on the corpus).** At every corpus rung `T + 1 <= 3` (`T = 1, 1,
  1, 2, 2, 2, 2, 2, 2` at m19..m53), and no gear above 5 bites a window of `<= 3` multipliers
  (`2n + 1 = 7`), so `2 Omega^full(T + 1) - 1` EQUALS `pad_cap.md`'s per-class row
  `3, 3, 3, 5, 3, 5, 5, 5, 3`, slack `1, 2, 0, 2, 1, 3, 3, 1, 0`: tight at m29 and m53 only, and
  P2 is refuted by arithmetic. The gears above 5 first matter at `T = 3`
  (`F(M + q') >= 3q' + 2`), which no corpus rung reaches. REFUTED by any corpus rung where the
  full-gear value differs from the gear-5 value.
- **P5 (the record is not at the richest translate).** The record's windows have `n_j <= 3`
  multipliers per class at every rung to 37 -> 41, and the deficit `Omega_slots - (L + 1)` is
  `>= 1` at a majority of the rungs 13 -> 17 .. 37 -> 41: the richest translate would put an
  opening of `M` on a tooth slot inside a flank, which shortens the stretch. Deficit 0 occurs
  only where every slot inside the stretch is a word opening. REFUTED by deficit 0 at a
  majority of rungs (then P3 stands and the start class is a formula).
- **P6 (the arc floor has no pullback form).** The twisted arcs `a'_g = min(+-(3 q')^-1 mod g)`
  are not coherent in file 20's sense (`3 a_g = g -+ 1`); some are 1 (adjacent teeth), which
  file 21 step 12 names as the configuration that breaks the floor. Prediction: `c(g, h; L) = 0`
  for `2 <= L <= max(a'_g, a'_h)` FAILS on the pullback at every rung (`>= 1` exception per rung
  among the pairs of `M`), while the shared-arc law `c(g, h; a + 1) >= 1` holds with 0
  exceptions (it is proved for any separation). The twin-gear collision at `(g + 4)/3` has no
  twisted analogue: twisted twins share an arc at no rung except by coincidence (`<= 1` pair per
  rung). REFUTED by 0 floor exceptions at some rung.
- **P7 (the coincidence law).** `k(g, h; n + gh) = k(g, h; n) + 4` with 0 exceptions (the
  one-orbit argument of file 21 Theorem 1 applies to the minimum verbatim), and the stacking
  deficit `D(n) = 0` for every `n <= 9` at every rung (gears 5 and 7 stack their forced strikes)
  with `D(n) >= 1` at some `n <= 20` at every rung. REFUTED by an exception to the +4 law, or a
  rung with `D(n) = 0` for all `n <= 20`.
- **P8 (the start class).** Where the deficit is 0 the argmax slot set pins each biting gear's
  phase to `<= 2` classes and the record's actual position lies in them (a consistency check,
  0 exceptions by construction); where the deficit is `>= 1` no start class follows. The rank of
  the record's translate is `< 10 %` at every rung from 19 -> 23 on: rich relative to a typical
  translate, not richest. REFUTED by a rank `>= 10 %`.

**Stop rules.** Anything reducing to a bound on `F(M + q')/q'` is stopped in one line and marked
ROOT; the one-orbit reduction, E1-E3 and file 21's theorems are cited, never re-derived; the
manifold, valves and exhaust are not touched.

### 0.4 Scorecard

| # | prediction | verdict | evidence |
|---|---|---|---|
| P1 | `Omega^full < Omega_{5,7}` first at `n_0 <= 12`, by gears 11..23 | **REFUTED** - `n_0 = -, 11, 19, -, 15, 17, -, 20, 10` at m19..m53 (no drop to `n = 20` at three rungs; `<= 12` at two); the deciding gear is **11 at every drop**, gears 13..37 lower no value at any rung | 2.1, 2.2 |
| P2 | sharpened cap tight at `>= 5` of 9 | **REFUTED** - tight at 2 of 9 (m29, m53) in both the per-class and the joint form | 3 |
| P3 | records at richest translates from m23 on | **REFUTED at 3 of 4** - deficit 1, 1, 2 at 23 -> 29, 29 -> 31, 31 -> 37; deficit 0 at 37 -> 41 (and at 19 -> 23, 13 -> 17 below the range) | 4 |
| P4 | the sharpening is empty on the corpus (gear 5 alone at `T + 1 <= 3`) | **CONFIRMED** - the full-gear per-class row equals `pad_cap`'s `3, 3, 3, 5, 3, 5, 5, 5, 3`; the joint form equals E3's `CC` row at 9 of 9 and gears `>= 11` add nothing at 9 of 9 | 3 |
| P5 | deficit `>= 1` at a majority of rungs; `n_j <= 3` | **HALF** - `n_j <= 3` at every fusion (14 of 14); deficit `>= 1` at 3 of the 6 non-trivial rungs (a tie, not a majority); the mechanism is as predicted (a struck tooth slot inside a flank, gear 5 among its strikers at 4 of 4 positions to 29 -> 31) | 4 |
| P6 | arc floor fails on the pullback; shared-arc law holds; twisted twins do not share arcs | **CONFIRMED, and turned into a theorem** - the failures are exactly the arc-1 gears at `L = 2` (9, 6, 7, 18, 9, 18, 20, 12, 13 exceptions = (arc-1 gears) x (other gears) at 9 of 9 rungs); for arcs `>= 2` the floor holds for ANY separations (E4, proved; 0 exceptions in 2,473,871 instances); shared-arc law 0 exceptions; twisted twins share an arc at 5 of 9 rungs ((5, 7) at five, (17, 19) at two), so the `(g+4)/3` law has no pullback form | 5 |
| P7 | coincidence +4 law; `D(n) = 0` to `n = 9`, `>= 1` by `n = 20` | **+4 law CONFIRMED** (E5, 0 violations in 47,603 instances); `D(n) = 0` to `n = 9` **REFUTED** at four rungs (`D(7) = 1` at m29, m41, m47, m53); `D >= 1` by `n = 20` at 9 of 9 | 5.4 |
| P8 | rank `< 10 %` from 19 -> 23 on | **REFUTED at one fusion** - 15.5 % for `(23, 10, 25)` at 29 -> 31; 1.6, 9.4, 8.3, 5.6, 1.1 % elsewhere | 4 |

Not pre-registered, found in the running: **E4** (the arc floor is a theorem for any two-class
gears whose arcs are `>= 2`; the real teeth qualify because `a_g = (g -+ 1)/3 >= 2`), which
proves `docs/proofs/21` Theorem 3 and explains its random-separation failures to the last
instance; **E5** (the rich-direction +4 law); and the identification of E3's corridor cap with
the two-class pullback count (section 3).

---

## 1. Setup (exact ranges)

Everything is exact integer arithmetic; no sampling anywhere. Scripts in `research/anchor235/r68/`,
results in `research/anchor235/r68/results/` (untracked).

| object | range | cost | script |
|---|---|---|---|
| `Omega^full(n)`, `n <= 20`, all gears of `M`, with the prefix chain by gear, the number of argmax patterns, `min_g(n)`, the stacking deficit `D(n)` and its smallest forcing gear subset; the corridor per class and uniform; the sharpened E2 (per class and joint) at the corpus `T` | m19..m53, `q'` the next prime; `2^n` subsets per `(n, rung)` | 12 s | `rh_omega.py` |
| the cap tables by `T = 1..9` (per class, joint with {5, 7}, joint with all gears) | 9 rungs | 5 s | `rh_caps.py` |
| pair deficits `c(g, h; L)`, coincidences `k(g, h; n)`, arcs, onsets, the floor test, the +4 law, the shared-arc pairs, `D_{5,7}(n)` | every pair of `M` at 9 rungs, `L <= 60` (the +4 law to `2gh` for `gh <= 400`) | 30 s | `rh_collide.py` |
| the arc floor over EVERY separation pair `(s_g, s_h)`, `1 <= s <= (g-1)/2`, all pairs `g < h <= 47` and `g < h <= 97` | 78 and 253 pairs; 113,617 and 2,643,359 instances | 60 s; 12 min | `rh_floor.py` |
| record positions: 13 -> 17, 17 -> 19, 19 -> 23, 23 -> 29 (full periods of `M + q'` by the copy law, every occurrence); 29 -> 31 (899 copies of `P_23`); 31 -> 37 (the 4-window `(11, 12, 37, 28)` located in m31 and lifted to m37 by its phase); 37 -> 41 (the 4-window `(15, 41, 14, 21)` in m37, 33,263 copies, 3 processes) | whole periods | 4 s; 201 s; 249 s; about 1 h | `rh_scan.py` |
| the record's windows, `Omega_slots`, deficit, exact rank over all `P(M)` translates, argmax pins, strikers of the struck slots | every record fusion of 7 rungs | 10 s | `rh_records.py` |

**Instrument gates, all passed.** The record ladder `F = 18, 25, 34, 43, 58, 88` at m17..m37 with
the fusions of `pad_cap.md` 2.6 reproduced from the positions (`(5, 11, 2)`, `(7, 13, 5)`,
`(18, 7)`, `(7, 15, 8, 4)`, `(23, 10, 10)`, `(18, 10, 30)`, `(23, 10, 25)`, `(11, 12, 37, 28)` all
found, plus `(5, 6, 7)` at 13 -> 17 which the witness list omitted); the corridor row
`Omega_{5,7}(n) = 1, 2, 3, 3, 3, 4, 5, 6, 6, 6, 7, 7, 8, 8, 9, 9, 9, 10, 10, 11` of `pad_cap.md`
4.1 recomputed digit for digit; the per-class E2 row `3, 3, 3, 5, 3, 5, 5, 5, 3` and E3's `CC`
row `3, 2, 3, 4, 3, 3, 3, 5, 3` reproduced (section 3); the arc floor with the real teeth: 0
exceptions at every rung (file 21's certificate); every record position's phase vector lies in
the phases its own openings admit (14 of 14 fusions with positions).

## 2. Results: `Omega^full(n)` at every rung

### 2.1 The tables

`full` = `Omega^full(n)` with every gear of `M`; `5,7` = the corridor value at the class
`q' mod 35`; `D` = the stacking deficit `(n - Omega^full) - max_g min_g(n)`. The uniform corridor
row (max over the 24 classes) is `1, 2, 3, 3, 3, 4, 5, 6, 6, 6, 7, 7, 8, 8, 9, 9, 9, 10, 10, 11`.

    n                 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
    m19 -> 23  full   1  2  3  3  3  4  5  6  6  6  6  6  7  7  7  8  9  9  9  9
               5,7    1  2  3  3  3  4  5  6  6  6  6  6  7  7  7  8  9  9  9  9
               D      0  0  0  0  0  0  0  0  0  0  1  2  2  2  2  2  2  3  3  3
    m23 -> 29  full   1  2  2  3  3  4  5  5  6  6  6  6  7  7  8  8  8  9  9  9
               5,7    1  2  2  3  3  4  5  5  6  6  7  7  8  8  9  9  9 10 10 11
               D      0  0  0  0  0  0  0  0  0  0  1  2  1  2  1  2  3  2  3  3
    m29 -> 31  full   1  2  2  3  3  4  4  5  5  6  6  7  7  8  8  9  9  9  9  9
               5,7    1  2  2  3  3  4  4  5  5  6  6  7  7  8  8  9  9  9 10 10
               D      0  0  0  0  0  0  1  0  1  0  1  1  1  1  1  1  2  2  3  3
    m31 -> 37  full   1  2  3  3  3  4  5  6  6  6  6  6  7  7  7  8  9  9  9  9
               5,7    1  2  3  3  3  4  5  6  6  6  6  6  7  7  7  8  9  9  9  9
               D      0  0  0  0  0  0  0  0  0  0  1  2  2  2  2  2  2  3  3  3
    m37 -> 41  full   1  2  2  3  3  4  5  5  6  6  7  7  8  8  8  8  8  9  9  9
               5,7    1  2  2  3  3  4  5  5  6  6  7  7  8  8  9  9  9 10 10 11
               D      0  0  0  0  0  0  0  0  0  0  0  1  0  1  1  2  3  2  3  3
    m41 -> 43  full   1  2  3  3  3  4  4  5  5  5  6  6  7  7  7  8  8  9  9  9
               5,7    1  2  3  3  3  4  4  5  5  5  6  6  7  7  7  8  9 10 10 10
               D      0  0  0  0  0  0  1  1  1  1  1  2  2  2  2  2  3  3  3  3
    m43 -> 47  full   1  2  3  3  3  4  5  6  6  6  6  6  7  7  7  8  9  9  9  9
               5,7    1  2  3  3  3  4  5  6  6  6  6  6  7  7  7  8  9  9  9  9
               D      0  0  0  0  0  0  0  0  0  0  1  2  2  2  2  2  2  3  3  3
    m47 -> 53  full   1  2  3  3  3  4  4  5  5  5  6  6  7  7  7  8  9  9  9  9
               5,7    1  2  3  3  3  4  4  5  5  5  6  6  7  7  7  8  9  9  9 10
               D      0  0  0  0  0  0  1  1  1  1  1  2  2  2  2  2  2  3  3  3
    m53 -> 59  full   1  2  2  3  3  4  4  5  5  5  6  7  7  8  8  8  8  9  9  9
               5,7    1  2  2  3  3  4  4  5  5  6  6  7  7  8  8  9  9  9 10 10
               D      0  0  0  0  0  0  1  0  1  1  1  1  1  1  1  2  3  2  3  3

The twisted arcs `a'_g` (the pullback's short arcs) at each rung:

| rung | `q'` | `a'_5, a'_7, a'_11, a'_13, a'_17, a'_19, a'_23, ...` up to `a'_q` |
|---|---|---|
| m19 | 23 | 1, 1, 4, 3, 1, 8 |
| m23 | 29 | 2, 2, 1, 3, 8, 7, 9 |
| m29 | 31 | 2, 3, 2, 6, 2, 9, 1, 5 |
| m31 | 37 | 1, 1, 1, 2, 2, 6, 6, 6, 12 |
| m37 | 41 | 2, 2, 5, 2, 4, 2, 3, 4, 1, 3 |
| m41 | 43 | 1, 2, 4, 1, 5, 5, 5, 9, 6, 2, 7 |
| m43 | 47 | 1, 1, 5, 6, 7, 7, 8, 7, 11, 16, 16, 18 |
| m47 | 53 | 1, 3, 2, 4, 3, 8, 11, 2, 8, 10, 8, 10, 13 |
| m53 | 59 | 2, 3, 1, 5, 5, 3, 10, 10, 7, 14, 19, 17, 17, 3 |

(`a'_g = min(+-(3 q')^{-1} mod g)`: the twisted separation is `(3q')^{-1} mod g` because
`2 u_g = 3^{-1} mod g`. Unlike the real arcs `(g -+ 1)/3` they are small and incoherent, and
three rungs have `a'_5 = a'_7 = 1`.)

### 2.2 What the tables say (exact statements, each with its count)

1. **Only the gears `<= 2n` bite**, as stated in 0.1: `Omega` over the gears `<= 2n` equals `Omega`
   over all gears of `M` at 180 of 180 `(n, rung)` cells. So `Omega^full(n)` to `n = 20` is a
   function of `q' mod (5 x 7 x ... x 37)` and the tables above are exact for every machine
   containing the primes to 37.
2. **Gears 13 and above lower nothing.** The prefix chain `Omega_{5}, Omega_{5,7}, Omega_{5,7,11},
   Omega_{5..13}, ...` is constant from gear 11 on at 180 of 180 cells. To `n = 20` the pullback's
   rich function is `Omega_{5,7,11}(n; q' mod 385)`.
3. **Gear 11 lowers the corridor value at six rungs, never by more than 2**: first at
   `n_0 = 11, 19, 15, 17, 20, 10` at m23, m29, m37, m41, m47, m53, and not at all to `n = 20` at
   m19, m31, m43. The three rungs with no drop are exactly the three with `a'_5 = a'_7 = 1`
   (`q' = +-2 mod 5` and `q' = 2, 5 mod 7`: 23, 37, 47), 3 of 3 against 0 of 6; a correlation on
   nine rungs, not a law. P1 is refuted on both counts (`n_0 <= 12` at two rungs only; the
   deciding gear is 11, never 13..23).
4. **The argmax patterns are gear 5's.** Every richest window to `n = 20` is a union of runs of
   three consecutive multipliers on the three open residues mod 5 (e.g. `{0,1,2, 5,6,7, 12, 15,16}`
   at m19, `n = 20`), thinned by 7 and 11; the number of argmax patterns is 1 to 46 per cell.
5. **The stacking deficit** `D(n) = 0` for `n <= 6` at 9 of 9 rungs and first becomes positive at
   `n = 7` (m29, m41, m47, m53), `n = 11` (m19, m23, m31, m43) or `n = 12` (m37); by `n = 20` it
   is 3 at every rung. Its smallest forcing subset is the pair `{5, 7}` at 96 of 101 positive
   cells; the exceptions are `{5, 11}` (m23, `n = 11, 13, 15`), `{11, 13}` (m37, `n = 15`) and
   `{7, 11}` (m53, `n = 10`). Section 5.4 compares `D` with the pair `(5, 7)`'s own deficit.

**Mechanism** (why the large gears are invisible). A gear `g` in `(n, 2n]` is forced to strike at
least one multiplier of the window (`min_g(n) >= 1` iff `n >= g - a'_g`), and a gear `g <= n` at
least `2 floor(n/g)`; the union of the strikes of 5 and 7 at their optimum already covers
`n - Omega_{5,7}(n) >= 4n/7 - 1` multipliers. A gear with one or two forced strikes has `g`
phases to place them and needs only that its teeth land on the struck set; with more than half
of the window struck and the teeth at an arbitrary twisted separation, some phase does it at
every cell computed. Gear 11 fails to hide only where its forced strikes are two or more
(`n >= 11`) and the twisted arc `a'_11` conflicts with the runs-of-three pattern; gears 13..37
have at most two forced strikes to `n = 20` and always hide. This is the same mechanism as
`pad_cap.md` 3.3's "gears 11 and 13 lower no class": the corridor is dense enough to absorb the
large gears' teeth.

## 3. The sharpened cap

### 3.1 At the corpus rungs

`T = floor((F(M+q') - 2)/q')`; per-class cap `2 Omega^full(T+1) - 1`; joint cap
`max_s Omega^(2)(T; q', s) - 1` (the two classes on one translate, 0.1); E3's `CC` from
`pad_cap.md` 4.2.

| `M` | `q'` | `F(M+q')` | `T` | per-class, all gears | joint, gears {5,7} | joint, all gears | E3 `CC` | `L` | slack per-class / joint |
|---|---|---|---|---|---|---|---|---|---|
| m19 | 23 | 34 | 1 | 3 | 3 | 3 | 3 | 2 | 1 / 1 |
| m23 | 29 | 43 | 1 | 3 | 2 | 2 | 2 | 1 | 2 / 1 |
| m29 | 31 | 58 | 1 | 3 | 3 | 3 | 3 | **3** | **0 / 0** |
| m31 | 37 | 88 | 2 | 5 | 4 | 4 | 4 | 3 | 2 / 1 |
| m37 | 41 | 91 | 2 | 3 | 3 | 3 | 3 | 2 | 1 / 1 |
| m41 | 43 | 103 | 2 | 5 | 3 | 3 | 3 | 2 | 3 / 1 |
| m43 | 47 | 118 | 2 | 5 | 3 | 3 | 3 | 2 | 3 / 1 |
| m47 | 53 | 145 | 2 | 5 | 5 | 5 | 5 | 4 | 1 / 1 |
| m53 | 59 | 161 | 2 | 3 | 3 | 3 | 3 | **3** | **0 / 0** |

- **P4 CONFIRMED, P2 REFUTED.** With `T + 1 <= 3` no gear above 5 bites a window (`2n + 1 = 7`),
  so the per-class cap with all gears IS `pad_cap.md`'s per-class row `3, 3, 3, 5, 3, 5, 5, 5, 3`,
  slack `1, 2, 0, 2, 1, 3, 3, 1, 0`, tight at m29 and m53 only. Made explicit: at `T = 2` the cap
  is `2 Omega_5(3) - 1`, which is **3 if `q' = +-1 (mod 5)`** (twisted arc 2: the two free
  residues of three consecutive multipliers are adjacent and cannot hold the teeth) and **5 if
  `q' = +-2 (mod 5)`** (arc 1). The corpus reads `q' mod 5 = 4, 1, 2, 1, 3, 2, 3, 4` at m23..m53,
  and the two tight rungs are the two with `q' = +-1 (mod 5)` and `L = 3`. This is E2's
  per-class form with its true modulus, 5 and not 35 (`pad_cap.md` 4.1 attributed it to
  `q' mod 35`).
- **The joint two-class form reproduces E3 exactly** at 9 of 9 rungs, `3, 2, 3, 4, 3, 3, 3, 5, 3`,
  and gears `>= 11` change none of the nine values. E3 was computed as a walk in `E_35` with
  letter values `<= F(M)` and span `<= F(M+q') - 2`; the joint pullback count drops the letter
  and span bookkeeping and keeps only "a subset of the two tooth progressions that some
  translate of `M` opens", and at these sizes the two agree. So **E3's content on the corpus is
  the corridor's two-class richness**, `Omega^(2)_{5,7}(T; q' mod 35, s)`: slack 0 or 1 at nine
  of nine, tight at m29 and m53.

### 3.2 The regime above the corpus: the cap as a function of `T`

For `T >= 3` (`F(M + q') >= 3q' + 2`, reached at no corpus rung) the three forms separate
(`rh_caps.py`; the joint forms take both classes on `[0, T]`):

    T                    1   2   3   4   5   6   7   8   9
    docs/proofs/11       3   5   7   9  11  13  15  17  19
    E2 {5,7} uniform     3   5   5   5   7   9  11  11  11
    m19  per class       3   5   5   5   7   9  11  11  11   joint {5,7}  3 4 4 5 6 7 8 8 9    joint all  3 4 4 5 6 7 8 8 9
    m23                  3   3   5   5   7   9   9  11  11                2 3 4 5 7 7 8 9 10              2 3 4 5 7 7 8 8 9
    m29                  3   3   5   5   7   7   9   9  11                3 3 4 4 6 7 7 8 9               3 3 4 4 6 7 7 8 9
    m31                  3   5   5   5   7   9  11  11  11                3 5 5 5 6 7 9 9 9               3 5 5 5 6 7 8 8 8
    m37                  3   3   5   5   7   9   9  11  11                2 3 4 4 6 6 7 9 9               2 3 4 4 6 6 7 9 9
    m41                  3   5   5   5   7   7   9   9   9                2 3 4 4 6 6 7 8 9               2 3 4 4 5 6 7 8 9
    m43                  3   5   5   5   7   9  11  11  11                2 3 4 5 6 7 8 9 10              2 3 4 5 6 7 8 9 9
    m47                  3   5   5   5   7   7   9   9   9                3 5 5 5 7 7 9 9 9               3 5 5 5 7 7 8 8 9
    m53                  3   3   5   5   7   7   9   9   9                2 3 4 5 6 7 8 9 10              2 3 4 5 6 7 7 8 9

The joint form is below the per-class form by 1 to 3 letters and grows like `T` (about one
letter per unit of `T`, against `(6/7) x 2` per class); all gears of `M` lower the `{5, 7}` joint
value by at most one unit to `T = 9`. **Every form is linear in `T = F(M+q')/q'`**; the pullback
with all gears changes the constant, not the shape. ROOT, exactly as `pad_cap.md` 4.3 said, and
the sharpening buys nothing on the corpus and at most one unit per rung above it.

## 4. Records and translates

For each record fusion `(f_L, w, f_R)` of `M + q'`: `s` the class-1 offset, `W_0`, `W_1` the tooth
slots strictly inside the stretch (relative to `x_0`), `L + 1` the word's openings,
`Omega_slots` the largest subset of `W_0 u W_1` some translate of `M` opens, the deficit
`Omega_slots - (L + 1)`, the number of argmax sets, the rank (fraction of the `P(M)` translates
with at least `L + 1` of the slots open, and with more), and the occurrences per period of
`M + q'`. Mirror fusions have the same numbers and are listed once.

| rung | fusion | `s` | `W_0` | `W_1` | `L+1` | `Omega_slots` | deficit | #argmax | rank `>= L+1` | rank `> L+1` | occ/period |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 13 -> 17 | (7, 6, 5) | 6 | {0} | {6} | 2 | 2 | **0** | 1 | 3.78 % | 0 | 6 |
| 13 -> 17 | (5, 11, 2) | 11 | {0} | {11} | 2 | 2 | **0** | 1 | 4.86 % | 0 | 4 |
| 17 -> 19 | (18, 7) | 6 | {0} | {-13, 6} | 1 | 2 | 1 | 3 | 64.5 % | 14.0 % | 4 |
| 17 -> 19 | (7, 13, 5) | 13 | {0} | {-6, 13} | 2 | 2 | **0** | 3 | 14.0 % | 0 | 6 |
| 19 -> 23 | (7, 15, 8, 4) | 15 | {0, 23} | {15} | 3 | 3 | **0** | 1 | 1.62 % | 0 | 2 |
| 23 -> 29 | (23, 10, 10) | 10 | {0} | {-19, 10} | 2 | 3 | **1** | 1 | 9.38 % | 0.63 % | 1 |
| 29 -> 31 | (23, 10, 25) | 10 | {0, 31} | {-21, 10} | 2 | 3 | **1** | 2 | 15.5 % | 1.20 % | 1 |
| 29 -> 31 | (18, 10, 30) | 10 | {0, 31} | {10} | 2 | 3 | **1** | 1 | 8.28 % | 0.60 % | 1 |
| 31 -> 37 | (11, 12, 37, 28) | 12 | {0, 37, 74} | {12, 49} | 3 | 5 | **2** | 1 | 5.60 % | 0.92 % | 1 |
| 37 -> 41 | (15, 41, 14, 21) | 14 | {0, 41} | {14, 55} | 3 | 3 | **0** | 2 | 1.07 % | 0 | see 4.3 |

(Occurrences count each orientation separately: the mirror fusion occurs equally often, so
13 -> 17 has 20 record stretches per period, 17 -> 19 has 20, 19 -> 23 has 4, 23 -> 29 has 2,
29 -> 31 has 4, 31 -> 37 has 2.) Exact counts behind the ranks (translates by open-slot count,
summing to `P(M)`): 19 -> 23 `760,292 / 602,829 / 227,286 / 26,208` (of 1,616,615); 23 -> 29
`17,049,163 / 16,645,059 / 3,252,303 / 235,620` (of 37,182,145); 29 -> 31 `(23,10,25)`:
`399,094,955 / 512,512,560 / 153,701,730 / 12,972,960` (of 1,078,282,205); 31 -> 37
`12,211,479,935 / 13,498,752,555 / 5,845,039,245 / 1,564,469,160 / 284,510,700 / 22,496,760`
(of 33,426,748,355); 37 -> 41 `527,526,872,378 / 560,057,578,146 / 135,955,870,479 /
13,249,368,132` (of 1,236,789,689,135).

### 4.1 The verdict on P3 and P5

- **P3 is refuted at three of the four rungs from m23 on** (deficits 1, 1, 2 at 23 -> 29, 29 -> 31,
  31 -> 37) and holds at 37 -> 41; below the brief's range it holds at 13 -> 17 (trivially: one
  slot per class) and 19 -> 23, and at 17 -> 19 the record is attained both by a fusion at a
  richest translate (`(7, 13, 5)`) and by one a unit below it (`(18, 7)`). Over the seven rungs:
  4 have a record fusion at a richest translate, 3 do not. Not a law in either direction.
- `n_j <= 3` at every fusion (P5's size clause holds); the deficit `>= 1` at 3 of the 6
  non-trivial rungs is a tie, not the predicted majority.
- **The deficit is always a struck tooth slot inside a flank, and gear 5 is among its strikers
  wherever the position is known to 29 -> 31.** 23 -> 29: the slot `x_0 - 19` (inside the flank
  23) is struck by 5, 7, 11 and 17. 29 -> 31: `(23, 10, 25)` has `x_0 + 31` struck by 5 and 19
  and `x_0 - 21` by 5 and 11; `(18, 10, 30)` has `x_0 + 31` struck by 5 and 11. 31 -> 37:
  `x_0 + 37` (the pad slot) is struck by **23 alone** and `x_0 + 74` by 7 and 13. 17 -> 19
  `(18, 7)`: `x_0 - 13` by 11, `x_0 + 6` by 5 and 7 (and 13 at two of four positions).
- **The mechanism of the trade-off.** A stretch of length `G` holds about `2G/q'` tooth slots.
  At a richest translate the word opens `Omega_slots` of them and the flanks are short (19 -> 23:
  flanks 4 and 7, word `(15, 8)`; 37 -> 41: flanks 15 and 21, word `(41, 14)`; 17 -> 19: flanks 7
  and 5, word `(13)`). At the other rungs the record takes a long flank that swallows a tooth
  slot (23 -> 29: flank 23 over the slot at `-19`; 29 -> 31: flanks 23/25 or 30/18 over `-21` or
  `31`; 31 -> 37: flank 28 over `74`), i.e. it trades one opening of the pullback for `f_L` or
  `f_R` columns of poor interval. Which trade wins is decided by the machine's gaps near the
  teeth (the flank lengths available with the slot struck), not by the pullback: at 23 -> 29 the
  rich alternative would be a fusion `(., 19, 10, .)` with the word `(19, 10) = (b, a)` and
  flanks summing to `43 - 29 = 14`, and the record law (`pad_cap.md` 2.6: the record is the
  fusion of largest total) says m23 offers no such fusion above 43.
- The record's word is the bare letter `a` alone at the three deficit rungs (`(10)`, `(10)`) or
  begins with `a` (`(12, 37)`), and contains `b` or the pad at the deficit-0 rungs (`(13)`,
  `(15, 8)`, `(41, 14)`, `(11)`, `(6)`): the long letter or the pad fills the class-1 window, the
  short letter leaves its other slot inside a flank. Seven rungs; a reading, not a law.

### 4.2 The rank

The record's translate is rich relative to a typical one (a translate opens `>= L + 1` of the
slots with probability 1.1 % to 15.5 %; it opens more than the record does with probability 0
to 1.2 %), but P8's `< 10 %` fails at the fusion `(23, 10, 25)` of 29 -> 31 (15.5 %; its four
slots `{-21, 0, 10, 31}` are not jointly feasible, so `Omega_slots = 3` and the rank counts every
translate opening any three or the record's two). The rank is a property of the slot set's size
and residues, and it says nothing about the record: the fraction of translates at the maximum
is 1.6 %, 0.63 %, 1.2 %, 0.07 %, 1.1 % at 19 -> 23 .. 37 -> 41, while the record occurs 1 or 2
times per period.

### 4.3 The 37 -> 41 record

The record `F(41) = 91` is the fusion `(15, 41, 14, 21)` of m37 (`docs/proof-search/mechanic.md`,
phase 19; its word `(41, 14)` is the small-alphabet 2-word of `pad_cap.md` 2.3). Its slot set
`{0, 41} u {14, 55}` has `Omega_slots = 3`, attained by exactly two subsets, `{0, 41, 55}` (the
record's own openings) and `{0, 14, 55}`; so the record sits at a richest translate, and its
tooth slot `x_0 + 14` is struck. Positions in m41's period: see the addendum (section 12), from
the 33,263-copy scan.

## 5. The collision law's pullback form

### 5.1 The arc floor is a theorem, and its true hypothesis is `a >= 2` (E4)

`docs/proofs/21` Theorem 3 (the arc floor, `c(g, h; L) = 0` for `2 <= L <= max(a_g, a_h)`) is on
record as a certificate for the real teeth with no proof, refuted by random separations (134
exceptions in 12,306 sampled instances), and with the note that "any correct proof must use
`3 a_g = g -+ 1`". On the pullback the separations are twisted, and the floor was tested first:

| rung | `q'` | pairs | floor instances | exceptions | pairs failing | arc-1 gears | (arc-1 gears) x (other gears) |
|---|---|---|---|---|---|---|---|
| m19 | 23 | 15 | 53 | 9 | 9 | 5, 7, 17 | 3 x 3 = 9 |
| m23 | 29 | 21 | 116 | 6 | 6 | 11 | 1 x 6 = 6 |
| m29 | 31 | 28 | 120 | 7 | 7 | 23 | 1 x 7 = 7 |
| m31 | 37 | 36 | 185 | 18 | 18 | 5, 7, 11 | 3 x 6 = 18 |
| m37 | 41 | 45 | 113 | 9 | 9 | 31 | 1 x 9 = 9 |
| m41 | 43 | 55 | 263 | 18 | 18 | 5, 13 | 2 x 9 = 18 |
| m43 | 47 | 66 | 714 | 20 | 20 | 5, 7 | 2 x 10 = 20 |
| m47 | 53 | 78 | 604 | 12 | 12 | 5 | 1 x 12 = 12 |
| m53 | 59 | 91 | 990 | 13 | 13 | 11 | 1 x 13 = 13 |

Every exception is at `L = 2` and every failing pair contains exactly one gear of twisted arc 1;
the count is `(number of arc-1 gears) x (number of other gears)` at 9 of 9 rungs. The real-teeth
control gives 0 exceptions in 61, 103, 166, 238, 337, 467, 610, 790, 1,011 instances. Then the
floor was tested over EVERY separation pair, exhaustively (`rh_floor.py`): all pairs `g < h`, all
`(s_g, s_h)` with `1 <= s <= (g-1)/2`, every `L` in `[2, max(a_g, a_h)]`:

| range | instances with both arcs `>= 2` | exceptions | instances with an arc equal to 1 | exceptions | where |
|---|---|---|---|---|---|
| `g < h <= 47` (78 pairs) | 99,817 | **0** | 13,800 | 1,704 | all at `L = 2` |
| `g < h <= 97` (253 pairs) | 2,473,871 | **0** | 169,488 | 10,846 | all at `L = 2` |

> **E4 (the arc floor, any separations).** Let `g, h >= 5` be two-class gears with any
> separations, short arcs `a_g`, `a_h`. Then `c(g, h; L) = 0` for every `3 <= L <= max(a_g, a_h)`,
> and `c(g, h; 2) = 0` unless exactly one of the arcs is 1. In particular the arc floor of
> `docs/proofs/21` Theorem 3 holds whenever `a_g, a_h >= 2`, which the real teeth satisfy
> because `a_g = (g -+ 1)/3 >= 2` for `g >= 5`.

*Proof.* Let `a_h = max(a_g, a_h) >= L >= 2`. (i) A run of `L <= a_h` columns has all its pairwise
distances `<= L - 1 < a_h <= h - a_h`, so gear `h`'s two teeth (at cyclic distance `a_h` and
`h - a_h`) cannot both lie in it: `h` strikes at most one column of the run, and it can strike
any chosen column (its phase is free), so `max_h(L) = 1`. (ii) Take a phase of `g` attaining
`max_g(L)`. If `max_g(L) < L` some column of the run is unstruck by `g`; put `h`'s single strike
there. The union has `max_g(L) + 1 = max_g(L) + max_h(L)` columns, so `c(g, h; L) = 0`.
(iii) `max_g(L) < L` for every `L >= 3`: if `g > L` each residue mod `g` occurs at most once
in the run, so `g` strikes at most 2 columns, and `2 < L`; if `g <= L` (so `L >= 5`), `g`
strikes at most `2 ceil(L/g) <= 2L/g + 2 <= 2L/5 + 2 < L`. (iv) For `L = 2`: `max_g(2) = 2` iff the teeth are adjacent,
`a_g = 1`, and then `c(g, h; 2) = 2 + 1 - 2 = 1`; otherwise `max_g(2) = 1` and two single
strikes on different columns give `c = 0`. If `a_g = a_h = 1` the floor's range `[2, 1]` is
empty. QED

So the "property of the real teeth" that file 21 saw is only that real arcs are never 1, and
the 134 random-separation exceptions on record are the draws with an arc-1 gear (every
exception in 2,643,359 exhaustive instances is one). Prior art, one line: the statement is
elementary (two residue classes of a large modulus in a short interval); it is recorded because
file 21 marks it unproved and ascribes it to the real separation, and because the pullback's
arc-1 gears are exactly where the hypothesis fails.

### 5.2 The shared-arc law and the twins on the pullback

Theorem 2 of file 21 (`a_g = a_h = a` implies `c(g, h; a + 1) >= 1`) is proved for any
separation; on the pullbacks it holds at every shared-arc pair (0 exceptions in 44 pairs). Which
pairs share a twisted arc is a different set from the twins: `3 q' a = +-1 (mod g)` and
`(mod h)` together, so `gh` divides `(3 q' a)^2 - 1`. Per rung (pair, arc): m19 `(5,7,1), (5,17,1),
(7,17,1)`; m23 `(5,7,2)`; m29 `(5,11,2), (5,17,2), (11,17,2)`; m31 `(5,7,1), (5,11,1), (7,11,1),
(13,17,2), (19,23,6), (19,29,6), (23,29,6)`; m37 `(5,7,2), (5,13,2), (5,19,2), (7,13,2), (7,19,2),
(13,19,2), (17,29,4), (23,37,3)`; m41 `(5,13,1), (7,37,2), (17,19,5), (17,23,5), (19,23,5)`; m43
`(5,7,1), (17,19,7), (17,29,7), (19,29,7), (37,41,16)`; m47 `(7,17,3), (11,29,2), (19,31,8),
(19,41,8), (31,41,8), (37,43,10)`; m53 `(7,19,3), (7,53,3), (13,17,5), (19,53,3), (23,29,10),
(43,47,17)`. The twin pairs of `M` share a twisted arc at 5 of 9 rungs (`(5, 7)` at m19, m23,
m31, m37, m43; `(17, 19)` at m41, m43) and at none of m29, m47, m53: **the twin collision at
`(g + 4)/3` has no pullback form**; the head collision's content ("5 and 7 are twins") does not
survive the twist, and on the pullback the pairs in early collision are whichever pairs
`(3q')^{-1}` happens to give equal arcs.

### 5.3 The rich direction: the coincidence law (E5)

`Omega` is a maximum of openings, i.e. a minimum of the union of strikes; the collision deficit's
counterpart is the **coincidence** `k(g, h; n) = min_g(n) + min_h(n) - joint_min(g, h; n) >= 0`,
the strikes the two gears can share when both are at their fewest.

> **E5 (the coincidence law).** For any two gears with any separations,
> `k(g, h; n + gh) = k(g, h; n) + 4`, and `joint_min(g, h; n + gh) = joint_min(g, h; n) + 2g + 2h - 4`.

*Proof.* File 21's Theorem 1, steps 1-4, with maximum replaced by minimum: a window of length
`n + gh` is a window of length `n` plus one full period, which carries exactly `|U| = 2g + 2h - 4`
marks of the union pattern and `2h`, `2g` marks of the single patterns, whatever the start. QED
(Checked: 0 violations in 47,603 instances over all pairs with `gh <= 400` at the nine rungs,
for `c` and `k` together.)

The rich direction is not the mirror of the poor one in its onset: my working assumption that
two gears coincide as soon as both are forced to strike (both strikes at a window's end) fails
at 5, 3, 11, 6, 11, 4, 1, 10, 7 pairs per rung, e.g. `(7, 13)` at m19: both forced from `n = 10`,
first coincidence at `n = 13`, because gear 7's two forced strikes (adjacent teeth, arc 1) must
sit in the window's middle (`{3,4}, {4,5}, {5,6}`) while gear 13's single forced strike must sit
within its arc of an end (`{0,1,2,7,8,9}`). A measured fact; no law.

### 5.4 Does the pair law decide the richest window?

The pair `(5, 7)`'s own stacking deficit `D_{5,7}(n) = joint_min(5, 7; n) - max(min_5, min_7)`
against the full `D(n)` of section 2:

| rung | `D_{5,7}(n)`, `n = 1..20` | agrees with `D(n)` at |
|---|---|---|
| m19 | 0 0 0 0 0 0 0 0 0 0 1 2 2 2 2 2 2 3 3 3 | 20 of 20 |
| m23 | 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1 2 1 2 1 | 10 of 20 |
| m29 | 0 0 0 0 0 0 1 0 1 0 1 1 1 1 1 1 2 2 2 2 | 18 of 20 |
| m31 | 0 0 0 0 0 0 0 0 0 0 1 2 2 2 2 2 2 3 3 3 | 20 of 20 |
| m37 | 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1 2 1 2 1 | 14 of 20 |
| m41 | 0 0 0 0 0 0 1 1 1 1 1 2 2 2 2 2 2 2 2 2 | 16 of 20 |
| m43 | 0 0 0 0 0 0 0 0 0 0 1 2 2 2 2 2 2 3 3 3 | 20 of 20 |
| m47 | 0 0 0 0 0 0 1 1 1 1 1 2 2 2 2 2 2 3 3 2 | 19 of 20 |
| m53 | 0 0 0 0 0 0 1 0 1 1 1 1 1 1 1 2 3 2 3 3 | 15 of 20 |

At the three rungs with `a'_5 = a'_7 = 1` the richest window to `n = 20` is decided by the pair
`(5, 7)` alone (20 of 20; the other gears' strikes hide entirely inside 5 and 7's); at the other
six, gear 11 adds one unit at 1 to 10 of the 20 lengths and no other gear adds anything
(section 2.2, item 2). So the pullback form of "which gears' collisions decide the window" is:
**the coincidence of 5 and 7, corrected by 11, to `n = 20` at every rung**, the correction
falling where the twisted arc of 11 conflicts with the runs-of-three pattern of gear 5 (a
two-unit correction at m23 and m37, `n = 20`, with `a'_11 = 1` and `5`). This is a description
of the computed tables, not a proof that 13 and above stay silent at larger `n`.

## 6. The start class

Where the record sits at a richest translate the argmax set pins each gear's phase: the number
of admissible phases per gear (of `g`), and the fraction of all translates that is left:

| rung | argmax set (relative to `x_0`) | admissible phases per gear 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 | fraction of translates | record occurrences / period |
|---|---|---|---|---|
| 19 -> 23 | {0, 15, 23} | 2, 2, 6, 7, 12, 13 | 26,208 / 1,616,615 = 1.62 % | 2 |
| 37 -> 41 | {0, 41, 55} (also {0, 14, 55}) | 1, 3, 7, 7, 11, 13, 17, 23, 26, 31 | 13,249,368,132 / 1,236,789,689,135 = 1.07 % | see section 12 |
| 17 -> 19 | {0, 13} (of three argmax sets) | 1, 3, 7, 9, 14 | 14.0 % | 6 |
| 13 -> 17 | {0, 11} | 1, 3, 9, 9 | 4.86 % | 4 |

So even where P3 holds, `Omega`'s argmax fixes the record's start only to a residue SET of
density 1 to 2 % (gear 5 to one or two classes, gear 7 to two or three, the larger gears
barely), inside which the record is one or two positions per period; the other 26,206 classes
at 19 -> 23 open the same three slots and are not records because the interval around them is
not poor. Where P3 fails (23 -> 29, 29 -> 31, 31 -> 37) the argmax pins the wrong translate: at
23 -> 29 the argmax set `{-19, 0, 10}` needs `x_0 = 3 (mod 5)` and the record has `x_0 = 0
(mod 5)` at its one position, so the slot `x_0 - 19` is struck by 5.

**What would make it a law, exactly.** Two things, neither in hand: (a) deficit 0 at every rung
(refuted at 3 of 7), and (b) a second condition selecting, among the `~1 %` of translates at the
argmax, the one whose surrounding interval is poor, which is the record's own definition. The
start class in the pullback's modulus is therefore not computable from `Omega`; thin place 3 of
the wall is unchanged in this coordinate. Exception count for "the record's start lies in the
argmax classes": 3 of 7 rungs.

## 7. What is new

1. **E4, the arc floor for any separations** (5.1): `c(g, h; L) = 0` for `3 <= L <= max(a_g, a_h)`
   and for `L = 2` unless exactly one arc is 1; a four-step proof; exhaustive over every
   separation pair to 97 (0 exceptions in 2,473,871 instances with arcs `>= 2`; 10,846
   exceptions, all arc-1 and all at `L = 2`). It proves `docs/proofs/21` Theorem 3, whose status
   was "certificate, no proof, needs the real separation", with the true hypothesis
   `a_g, a_h >= 2`; the random-separation failures on record are the arc-1 draws. On the
   pullbacks the exception count is `(arc-1 gears) x (other gears)` at 9 of 9 rungs.
2. **E5, the coincidence law** (5.3): the rich-direction `+4` law, proved by file 21's argument
   for the minimum; 0 violations in 47,603 instances.
3. **The exact `Omega^full(n)` tables to `n = 20` at nine rungs** (2.1), with the facts: only gears
   `<= 2n` bite (180 of 180); gears 13..37 lower nothing (180 of 180); gear 11 lowers the corridor
   at six rungs by 1 or 2, first at `n = 10..20`, and not at the three rungs with
   `a'_5 = a'_7 = 1`; the argmax windows are gear 5's runs of three; the stacking deficit is the
   pair `(5, 7)`'s at 96 of 101 positive cells.
4. **E3 identified** (3.1): on the corpus the exact corridor-plus-span cap equals the two-class
   pullback count under gears 5 and 7 at 9 of 9 rungs, and no larger gear changes it; the
   per-class E2 at `T = 2` is `3` or `5` by `q' mod 5` alone, tight exactly at the two rungs with
   `q' = +-1 (mod 5)` and `L = 3`.
5. **The record's translate measured** (4): deficit `0, 0/1, 0, 1, 1, 2, 0` at 13 -> 17 .. 37 -> 41,
   with the struck flank slot and its strikers at every position to 31 -> 37 (gear 5 at 4 of 4
   known positions to 29 -> 31; gear 23 alone at 31 -> 37); the exact rank distributions; the
   trade-off (rich word with short flanks against a long flank over a struck slot) named with
   its instances. The 31 -> 37 record occurs exactly 2 times per period of m37 (one per
   orientation), the 23 -> 29 and 29 -> 31 records 2 and 4 times.
6. **Instrument**: the subset-feasibility computation of `Omega` for any slot set and any gear
   set (no period, `2^n` masks), the exact translate distribution by a union DP, and the
   positions of the 29 -> 31 and 31 -> 37 records by the copy law (`rh_core.py`, `rh_scan.py`).

**Prior art, one line each.** The one-orbit reduction and the deficit's +4 law are file 21's;
`Omega` with two classes per prime is the two-class admissible-tuple function noted in the
unstick file (not searched further: the values here are per class of `q'` and exact, and no
published table is claimed); E4's content is elementary and is claimed only as the proof file 21
lacked.

## 8. Verdict

**Node R4.c.ii: the rich half is the corridor's richness (gears 5, 7, and 11 by a unit), the
record does not sit at the richest translate as a rule, and the sharpened cap is E2/E3 again.**

- The sharpened cap's slack: **identical to E3's** - `1, 1, 0, 1, 1, 1, 1, 1, 0` at m19..m53 in the
  joint form (tight at m29 and m53), `1, 2, 0, 2, 1, 3, 3, 1, 0` per class; every gear above 7
  adds nothing at the corpus and at most one unit to `T = 9`. P2 refuted, P4 confirmed.
- Records at the richest translates: **at 4 of 7 rungs, not at 3** (23 -> 29, 29 -> 31, 31 -> 37,
  deficits 1, 1, 2); the record trades a pullback opening for a long flank, and the flank slot
  is struck by gear 5 (four of four known positions to 29 -> 31) or by a single large gear
  (23 at 31 -> 37). P3 refuted; the start class is not a formula (section 6).
- The collision form: the arc floor is a **theorem for any separations with arcs `>= 2`** (E4),
  which is why it holds on the real teeth and fails on random draws and on the pullback's
  arc-1 gears exactly at `L = 2`; the twin law has no pullback form; the rich direction has its
  own +4 law (E5); the richest window is decided by the coincidence of 5 and 7 corrected by 11.
- **ROOT, honestly.** `Omega` is a maximum over translates and every cap it yields is linear in
  `T = F(M + q')/q'` (3.2); the existence of a richest translate is a free-phase (CRT) fact that
  says nothing about the poorness of the interval around it, and the record is the poorest
  interval whose openings are on the teeth, a condition `Omega` does not see. Nothing here bounds
  `F(M + q')` without a count; the rich half is the record's cap restated in the pullback's
  coordinate, sharper by a constant and not in shape.

## 9. Scorecard (filled)

See 0.4. Confirmed: P4; P6 (and strengthened to E4); P7's law half. Refuted: P1 (both clauses),
P2, P3 (3 of 4), P7's `D = 0` clause, P8 (one fusion). Half: P5.

## 10. Dead ends (bricks), each with its refuting instance

| idea | dies at | the instance | why it cannot be revived |
|---|---|---|---|
| all gears sharpen E2 on the corpus | every corpus rung | `T + 1 <= 3 < 7 = 2 x 5 - 3`: only gear 5 bites | the corpus has `F(M+q') < 3q' + 2`; gear 7 first matters at `T = 3`, gear 11 at `T >= 9` |
| the sharpened cap as a law tighter than E3 | 9 of 9 rungs | joint cap = `CC` value for value | the two are the same count of the corridor's two tooth classes |
| records at richest translates (P3) | 23 -> 29 | `(23, 10, 10)`: the slot `x_0 - 19` is struck by 5, 7, 11, 17; deficit 1 | the record prefers a long flank over a rich word when the machine's gaps allow it |
| the start class from `Omega`'s argmax | 19 -> 23 even where P3 holds | 26,208 classes at the maximum for 2 records per period | richness selects 1-2 % of translates; poorness selects the record |
| the twin collision `(g+4)/3` on the pullback | m29, m47, m53 | no twin pair shares a twisted arc | the shared arc is `3 a = g -+ 1`, destroyed by the factor `q'^{-1}` |
| the arc floor as a real-teeth property | every separation pair with arcs `>= 2` | 0 exceptions in 2,473,871 | it is a theorem (E4); only arc 1 breaks it |
| the stacking deficit as a pure `(5, 7)` object | m23, `n = 11` | `D = 1`, `D_{5,7} = 0`, forcing subset `{5, 11}` | gear 11 contributes at six rungs |
| the coincidence onset "at the window's end" | `(7, 13)` at m19 | both forced at `n = 10`, first shared strike at `n = 13` | the small gear's adjacent forced pair sits in the middle |

## 11. The part's remaining open items, sorted

- **Closed here.** `Omega^full(n)` to `n = 20` at nine rungs (exact, with the biting set and the
  deciding gear); the sharpened cap at every corpus rung (equal to E3); the record's translate,
  deficit and rank at seven rungs with positions to 31 -> 37; the arc floor (E4, proved for any
  separations with arcs `>= 2`; the open item (ii) of `pad_cap.md` 10); the coincidence law (E5).
- **Measurement with no structural content.** The rank distributions; the argmax pins; the
  first-coincidence table; the correlation "no drop below the corridor iff `a'_5 = a'_7 = 1`"
  (3 of 3 against 0 of 6, nine rungs).
- **Root question in disguise.** Any cap on `L` through `Omega` (per class, joint, all gears):
  linear in `T = F(M+q')/q'`; the record's position as the poorest interval among the richest
  translates.
- **Genuinely open on the part alone, with the attack.** (i) Whether gears `>= 13` ever lower
  `Omega^full(n)` at some `n > 20` (the mechanism of 2.2 suggests not until a gear's forced
  strikes exceed what the `{5, 7, 11}` struck set can absorb; the computation is `2^n` masks and
  is cheap to `n = 26`; a proof would be a covering statement about runs of three mod 5 and the
  teeth of a gear `g` with `a'_g` arbitrary). (ii) Formalisation: E4 is four elementary steps
  about residues in an interval and belongs in the kernel beside file 20's Lemma 2; E5 is file
  21's Theorem 1 with `max` replaced by `min`.

## 12. Addendum: the 37 -> 41 record's positions

(Not filled: the 33,263-copy scan of m37 was killed by the session limit of 2026-09-07 before completion; `results/positions_37.json` does not exist. The 37 -> 41 record's translate rank stays unmeasured; positions to 31 -> 37 stand.)
