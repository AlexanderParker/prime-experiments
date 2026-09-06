# Node 4.i.a.i.a.1.1 - THE PINNED LETTER'S ARITHMETIC: the letter in tooth units

Parent: node **4.i.a.i.a.1, the pinned letter** (`research/proof/pinned_letter.md`, CANDIDATE,
2026-09-06), whose closing line names this branch exactly:

> "The next child branch is therefore not another covering construction but the arithmetic one:
> ... the place to look is what `3 a_L = q' -+ 1` does to the covering capacity, not what
> two-tooth periodicity does to it."

What spawned this branch: the parent proved that **no tooth-invariant argument can give the
constant 3** (12 of 63 family members have `E(a_L) > 3`, up to `+7`), so the only remaining input
is the real teeth `u_g = 6^{-1} (mod g)`. This branch takes that input seriously and asks what it
says about the letter's position relative to every gear's teeth.

Scripts in `research/anchor235/r63/`; result outputs in `research/anchor235/r63/results/`
(untracked). Every number this document relies on is written into the document.

---

## 0. Pre-registered (written before any computation of this branch)

### 0.1 The objects, and the arithmetic, derived by hand before measuring

Machines `M_0 = {5}`, ..., `M_8 = {5..31}`; a **rung** is `(M, q')` with `q'` the incoming gear
(`M = {5..11}` at rung `q' = 13`, ..., `M = {5..31}` at rung `q' = 37`). `u_g = 6^{-1} (mod g)`,
teeth `T_g = {u_g, -u_g}`, tooth distance `d_g = 2 u_g` (the short arc, file 02(b)).

**The sign.** Write `eps = +1` when `q' = 5 (mod 6)` and `eps = -1` when `q' = 1 (mod 6)`, so
`6 u_{q'} = q' + eps` and

    a_L = 2 u_{q'} = d_{q'} = (q' + eps)/3,      3 a_L = q' + eps.

`a_L` is literally the incoming gear's own tooth distance. `b_L = q' - a_L` and `3 b_L = 2q' - eps`.

**The identity the brief asks to verify.** For every gear `g` of `M`, `6 u_g = 1 (mod g)`, so
`3 d_g = 6 u_g = 1`, i.e. `d_g = 3^{-1} (mod g)`. Hence for ANY integer `v`,

    (I0)   v = (3v) d_g = 2 (3v) u_g   (mod g)      -- the universal form

and at `v = a_L`, where `3 a_L = q' + eps`,

    (I1)   a_L = (q' + eps) d_g = 2 (q' + eps) u_g   (mod g).

So the letter's residue modulo every gear is the fixed multiple `2(q' + eps)` of that gear's own
tooth position. **The identity is trivial and it is universal** -- it holds for every distance, not
only for the letter, because it is nothing but the change of coordinate

    (C)   n := 6 k    (the column `k` written as the even number `n` with members `n -+ 1`)

in which EVERY gear has its teeth at `n = +1` and `n = -1` and a distance `v` becomes `6 v`. That
single coordinate is the real-teeth input, and it is what the counterfactual family destroys.
Define the **tooth-unit multiple** of a distance at a gear,

    mu_g(v) := v * d_g^{-1} (mod g)          ("how many tooth-distances long is v, at gear g")

Then `mu_g(v) = 3v` for every gear of the real machine, and the vector `(mu_g(v))_{g in M}` is the
reduction of ONE integer. That is the branch's Setup, and the letter's own integer is `q' + eps`.

**The residue formula for the far end.** Let an `a_L`-gap have near end `x` and far end `x + a_L`,
and write `lam_g := 6x (mod g)` for the near end's position in tooth units (teeth at `lam = +-1`).
Then for every `t >= 0`

    (I2)   gear g strikes the column x + a_L + t   iff   lam_g = +-1 - 2(q' + eps) - 6t  (mod g)

and for every `0 <= s <= a_L`

    (I3)   gear g strikes the column x + s        iff   lam_g = +-1 - 6s  (mod g).

Both ends open means `lam_g not in {1, -1, 1 - 2(q'+eps), -1 - 2(q'+eps)}`: **at most four classes
mod `g` are forbidden to the near end, and which four is decided by `q' mod g` alone.**

**Three corollaries derived by hand (to be checked, not assumed).**

* **(D1) The closers.** `g` strikes both ends of an `a_L`-gap in one copy iff `a_L = 0` or `+-d_g`
  `(mod g)`, i.e. (by I1) iff `g | q' + eps` or `q' + eps = +-1 (mod g)`, i.e. `g | q'` (impossible
  for a gear of `M`) or `g | q' + 2 eps`. So `Leg(a_L) n M = { g in M : g | q' + 2 eps }` = the
  prime factors of `q' + 2eps` -- the parent's CLOSER LAW (`short_letter_row.md` 2.4), re-derived
  in one line from (I1). Cited, not claimed.
* **(D2) The pad.** `g | a_L` iff `g | q' + eps`; so `Pad(a_L) n M` is the set of gears dividing
  `q' + eps = 3 a_L`, and for such a gear the two ends of every `a_L`-gap lie in the SAME class mod
  `g`, whence `g` strikes exactly `2 a_L / g` of the `a_L - 1` interior columns of EVERY `a_L`-gap,
  at every occurrence.
* **(D3) The specialised obstruction law.** The parent's spare-gear lemma calls a gear `h`
  *obstructed* at a 2-run `(a, v)` when `h | a`, or `h | v`, or (`a = +d_h` or `v = -d_h`) and
  (`a = -d_h` or `v = +d_h`) mod `h`. At `v = a_L`, (I1) gives `a_L = +d_h` iff `h | q'` (never) or
  ... precisely: `a_L = +d_h` iff `q' + eps = 1`, i.e. `h | q' + eps - 1`; `a_L = -d_h` iff
  `h | q' + eps + 1`. With `eps = +1` those read `h | q'` (never) and `h | q' + 2`; with
  `eps = -1`, `h | q' - 2` and `h | q'` (never). Hence

  > **`h` is obstructed at a 2-run `(a, a_L)` iff `h | a`, or `h | q' + eps`, or
  > (`h | q' + 2 eps` and `a = -eps * d_h (mod h)`).**

  Every clause is a divisibility in `q'` and a residue of the NEIGHBOUR `a`. In particular a gear
  that divides neither `a` nor `q' + eps` nor `q' + 2 eps` is never obstructed, so by the
  spare-gear lemma it must be BUSY (a sole striker inside the run) whenever `E(a_L) > 0`.

### 0.2 The theory

**T. The pinned letter is a statement in the single coordinate `n = 6k`. In that coordinate the
letter gap is a jump of the one integer `6 a_L = 2q' + 2eps`, two away from twice the incoming
gear, and every gear of `M` sees the same jump. The bound `a_L + r(a_L) <= F + 3` should therefore
be a consequence of (i) the four forbidden classes (I2)/(I3) that the letter's two ends impose on
every gear at once, and (ii) the covering capacity left over. Prediction of the branch: (i) is
exactly computable and reproduces the certified row tops by an exact CRT search with the letter's
ends prescribed, but (i) alone does NOT produce the constant 3 -- the constant, if it has a
mechanism, lives in the interaction of (i) with the record `F`, and the branch's job is to say
which of the two halves fails.**

### 0.3 Disclosure of what had already been read

The parent's tables 2.1, 2.7 and 4.2 and the grandparent's 2.1, 2.3, 2.4, 2.7 and 4.3-4.4 were
read before these predictions were written. So every prediction about the numbers
`F, F_2, a_L, r(a_L), E(a_L), r(b_L), r(q'), c_5(a_L)` at rungs 5->7 .. 31->37, and about the
LP-certified row tops `20, 25, 35`, is **post-hoc `[read]`** and counts only as an instrument
check. The blind content of this branch is: the identity checks, the three corollaries D1-D3, the
strike tables and sole-striker maps of the attaining runs (never computed), the exact CRT search
and what its infeasibility proofs need, the family repair experiment, and the row of the other two
letters computed by the same search.

### 0.4 Predictions, each with the number that would refute it

- **P1 (the identity; instrument).** `(I1)` holds at every (rung, gear) pair -- 8 rungs, 44 pairs
  -- and `(I0)` at every `(v, g)` with `v <= 120`, `g <= 37` (1440 cells). **0 exceptions.**
  REFUTED by one. And `mu_g(v) = 3v (mod g)` at all of them, so the tooth-unit vector is constant.
- **P2 (the family is incoherent).** On the 60 counterfactual members of `pl_family.py`
  (rungs 13->17, 17->19, 19->23, seed 20260906) the vector `(mu_g(a_L))_{g in M}` is the reduction
  of a single integer `< min(g)`... more precisely: predict **0 of 60** members satisfy
  `mu_g(a_L) = 3 a_L (mod g)` at every gear, and predict the number of gears at which it does hold
  is `0, 1 or 2` at 55 or more of the 60. REFUTED if 3 or more members are fully coherent.
- **P3 (D1, D2, D3 checked).** D1 at 8 of 8 rungs (`Leg(a_L) n M` = factors of `q' + 2eps`);
  D2 exact at every `a_L`-gap of every machine to `{5..23}` (millions of gaps, 0 exceptions);
  D3 against the generic obstruction test at every realised 2-run `(a, a_L)` of machines
  `{5..11}` .. `{5..29}`, **0 mismatches**. REFUTED by one mismatch in any of the three.
- **P4 (the strike table of the attaining run).** At the attaining 2-run of `a_L` at every rung
  `13, 17, 19, 23, 29, 31`: (a) every gear is a sole striker inside the run (the parent's 90 of 90,
  `[read]`, instrument); (b) predict the near-end tooth-unit vector `(lam_g)` of the attaining
  occurrence takes **at most 2 distinct values up to the mirror `lam -> -lam`** at each rung --
  i.e. the attaining run is rigid, not one of many residue shapes. REFUTED by 3 or more mirror
  classes at 2 or more rungs; (c) predict **every gear strikes at least one column of the NEIGHBOUR
  gap** at every rung (no gear is confined to the letter gap), refuted by one gear with zero;
  (d) predict the gears of `Pad(a_L)` carry the largest share of the letter gap's columns at every
  rung (refuted at 2 or more rungs).
- **P5 (the exact CRT search -- the branch's main computation).** An exact branch-and-bound over
  the gears' residue classes decides, scan-free, whether the pair `(a_L, a)` is realisable
  (three prescribed openings at `0, a_L, a_L + a`, every column between them struck). Predict:
  - **P5a.** The largest feasible `a` equals `r(a_L)` at every rung `13, 17, 19, 23, 29, 31`
    (values `3, 7, 12, 20, 25, 35` `[read]`) and, out of the LP lane's reach, at rung 37
    (`r(12) = 46` `[read]` from the streamed scan, but never certified scan-free). REFUTED by one
    mismatch. This also re-derives the LP lane's three certified tops `20, 25, 35` by a different
    vehicle (cited; agreement is an instrument check, not a new result).
  - **P5b.** The search gives **no** formula: predict the minimal infeasible `a` (`= r(a_L) + 1`)
    needs the WHOLE machine at 4 or more of the 7 rungs -- i.e. no proper sub-machine of `M` makes
    the cell infeasible. REFUTED if a sub-machine of at most 3 gears suffices at 5 or more rungs
    (which would be a proof-shaped mechanism, and the branch would then chase it).
  - **P5c.** The pinned bound `F + 3 - a_L` is tight (equals `r(a_L)`) at exactly 1 of the 8 rungs
    (23->29, `[read]`), so the CRT search cannot reproduce the constant 3 as an equality; predict
    the search's slack `F + 3 - a_L - r(a_L)` is `3, 1, 3, 0, 2, 1, 3` at rungs
    `13, 17, 19, 23, 29, 31, 37`.
- **P6 (the family, item 3 -- the repair experiment).** For each member with `E(a_L) > 3`, set ONE
  gear back to its real tooth `u_g` and recompute. Predict a single-gear repair succeeds at **half
  or more** of the violating members, and that the repairing gear is **5** at the majority of the
  successes. REFUTED if single-gear repair succeeds at fewer than a quarter. And predict the
  coherence count (gears with `v_g = u_g`) separates: members with 2 or more coherent gears obey
  the upper half at a strictly higher rate than members with 0.
- **P7 (the twin rungs, item 4).** At a twin rung (`p = q' - 2` a gear of `M`) file 02(e) gives
  `u_p = u_{q'}`, so `d_p = a_L` and `6 a_L = 2(p + 1) = 2 (mod p)`: **the far end of the letter
  gap sits exactly 2 tooth-units beyond the near end at the twin partner**, the same jump as the
  partner's own two teeth. Predict: (a) exactly 3 classes mod `p` are forbidden to `lam_p`
  (`{1, -1, -3}`), i.e. `c_p(a_L) = 3`, at 4 of 4 twin rungs; (b) the partner strikes **at most 1**
  interior column of any `a_L`-gap, at every occurrence at every twin rung; (c) the measured
  excesses at twin rungs (`+2, 0, 0, +2` `[read]`) are the four smallest-or-equal among the rungs,
  which is already known to be true and is an instrument line only.
- **P8 (the other letters, item 5).** By the same identity, `3 b_L = 2q' - eps` so
  `Leg(b_L) n M` = primes `>= 5` dividing `2q' - eps -+ 1`, i.e. **dividing `q' - eps`**; and
  `Leg(q') n M` = primes `>= 5` dividing `3q' -+ 1`. Predict both at 8 of 8 rungs, 0 exceptions.
  For the rows: predict `b_L + r(b_L) <= F + 7` at 8 of 8 and **NOT** `<= F + 3` (it is `+5, +7` at
  two rungs, `[read]`); predict `q' + r(q') <= F + 9` at 8 of 8 and not `<= F + 3`. The blind
  clause: predict the excess is ordered by the multiplier `3l` -- `E(a_L) <= E(b_L) <= E(q')` at 6
  or more of the 8 available comparisons. REFUTED by 4 or more inversions.
- **P9 (the `q'`-closure consistency check).** Because `a_L = d_{q'}`, the incoming gear can be
  phased with one tooth on the middle opening of a 2-run `(a, a_L)`, the other tooth landing
  `a_L` away on the far side; if `a != 0, a_L (mod q')` neither end is struck, so
  `F(M + q') >= a_L + r(a_L)`. This is an instance of the attainment identity (file 08) and is
  **cited, not claimed**; predict it holds at 8 of 8 rungs with slack `4, 12, 7, 6, 8, 13, 30`
  `[read]` and is far weaker than `F + 3`.

**Stop rules.** Anything that reduces to the merge/chain law and the letters (file 05), the tooth
rule (file 02), the attainment identity (08), the peel bound / triple inequality (16), the glue
lemma and `N(v) <= F_2` (2g.i), the pair filter / closer law / row profile (4.i.a.i.a), the
spare-gear lemma and the excess-as-price reading (4.i.a.i.a.1), the gate ladder (4.i.a.i), the
divisor law L20, the half-column fibre theorem, or the LP windowed dictionary vehicle, is stopped
in one line and cited.

### 0.5 Scorecard

| # | prediction | verdict | evidence |
|---|---|---|---|
| P1 | the identity `(I1)`, universal form `(I0)`, constant tooth-unit vector | CONFIRMED, 0 exceptions in 45 + 1,200 + 1,200 checks -- **and the identity turns out to be universal**, true of every distance, not a property of the letter | 2.1 |
| P2 | the family is incoherent: 0 of 60 fully coherent, 0-2 coherent gears at 55+ | CONFIRMED on both clauses (0 of 60; 56 of 60) | 2.5 |
| P3 | D1, D2, D3 all exact | CONFIRMED: D1 9/9 rungs, D2 0 exceptions in 243,370 `a_L`-gaps, D3 0 mismatches in 5,400 cells | 2.3 |
| P4 | strike table: rigid residue shape, no gear confined, pad gears carry most | (b) **REFUTED decisively** -- one mirror class per occurrence at every rung (4, 12, 24, 6, 8, 4); (c) CONFIRMED 6/6; (d) REFUTED -- gear 5 leads at 6/6 whether or not it divides `a_L` | 3.1, 3.2 |
| P5a | the CRT search reproduces `r(a_L)` at 7 rungs | CONFIRMED 7 of 7 (`3, 7, 12, 20, 25, 35, 46`), and 22 of 22 rows counting `b_L`, `q'` and four non-letter sizes; rung 37 reached in 36 s, contrary to the doubt in the prediction | 1, 3.3, 5.2 |
| P5b | no small sub-machine certifies infeasibility | CONFIRMED and sharpened: at the two rungs that matter no cell above the row is killed by arithmetic at all, and the shortest uncoverable window is 15-44 columns of runs 36-53 long | 3.4 |
| P5c | slack `3, 1, 3, 0, 2, 1, 3` | CONFIRMED exactly at the seven scanned rungs -- and the eighth, out of sample, has slack **14** | 3.3, 5.4 |
| P6 | single-gear repair at half or more; gear 5 the repairer; coherence count separates | **REFUTED on every clause by the control**: real tooth repairs 0.44 of moves against a wrong tooth's 0.53 (upper half 0.62 vs 0.60); no gear stands out; the coherence count does not order obedience (0.67, 0.73, 0.60, 0.50) | 2.5, 4.1 |
| P7 | twin partner: 3 forbidden classes, at most 1 interior strike | CONFIRMED both, 4 of 4 twin rungs, 50 of 50 admissible classes, 0 exceptions | 4.2 |
| P8 | `Leg(b_L)` = factors of `q' - eps`; `Leg(q')` = factors of `3q' -+ 1`; the ordering | CONFIRMED: divisor forms 27/27 cells 0 exceptions; `b + r(b) <= F + 7` at 6/6 and `q' + r(q') <= F + 9` at 4/4, neither pinned to 3; the ordering by `3l` at 8 of 10 comparisons | 5.1, 5.2 |
| P9 | the `q'`-closure holds with slack `4, 12, 7, 6, 8, 13, 30` | CONFIRMED in content (8 of 8 rungs); the predicted slack list was wrong at one entry (5, not 12, at rung 17). Cited instance of file 08 | 5.3 |
| -- | **not pre-registered, found on the way** | **the pinned letter's LOWER half `F <= a_L + r(a_L)` is REFUTED out of sample at rung 37->41: `14 + 63 = 77` against `F({5..37}) = 88`** | 5.4 |

---


## 1. Setup (exact ranges)

Everything exact: full periods where a period is scanned, complete enumeration over residue
classes where it is not, integer arithmetic throughout, no sampling except the named family.

| object | range | script |
|---|---|---|
| the identity `(I1)`, the universal form `(I0)`, the tooth-unit vector, the four forbidden classes, the divisor forms of `Pad` and `Leg` for `a_L`, `b_L`, `q'`, the twin-partner shift | all 9 rungs `5->7 .. 31->37`, gears `5..37`, sizes `v <= 120` | `pa_identity.py` |
| the strike table, sole-striker map and tooth-unit vectors of the attaining 2-run of `a_L` at every occurrence; D2 on the full period; D3 exhaustively; the twin-partner count over all classes | full periods `{5..11}` .. `{5..23}`; `{5..29}` streamed as 29 copies of the m23 period | `pa_strike.py` |
| **the exact CRT search**: the largest neighbour of an `a_L`-gap, decided from residues alone | machines `{5..11}` .. `{5..31}` (rungs 13 .. 37), every cell from `F` down | `pa_crt.py` |
| what kills each cell above the row (pair filter / shortest uncoverable window); the rows of `b_L` and `q'` by the same search | the same 7 rungs | `pa_crt2.py`, `pa_extra.py` |
| the forced-gear law and the forced-cover count | all 9 rungs, complete enumeration over classes | `pa_forced.py` |
| coherence of the counterfactual family, the repair experiment and its control | rungs 13->17, 17->19, 19->23, 20 members each (seed 20260906, the same members as r57/r58/r62), full periods | `pa_family.py`, `pa_extra.py` |
| **out of sample**: the row of the letter at rung 37->41, `M = {5..37}`, whose period (1.24e12 columns) has never been scanned | the CRT search, every cell from `F = 88` down, 400M-node cap | `pa_rung41.py` |

**Instrument gates, all passed.** The search reproduces **22 of 22** row tops that were previously
obtained by full-period or streamed scans, 0 mismatches: `r(a_L) = 3, 7, 12, 20, 25, 35, 46`,
`r(b_L) = 5, 7, 13, 15, 27, 40`, `r(q') = 5, 8, 14, 30`, and four non-letter sizes at
`M = {5..31}` (`r(14) = 48`, `r(16) = 42`, `r(20) = 43`, `r(25) = 40`). Three of them are the LP
lane's certified cells (`20, 25, 35`), and six of them had only ever come from the
33,426,748,355-column streamed pass. A soundness guard was added after the first run and every
result re-verified: a gear with **no** class avoiding the three prescribed openings kills a cell
outright, and the covering search cannot see such a gear because it contributes no options. The
witness of the deepest cell was checked by explicit CRT reconstruction (5.4).

## 2. Results: the arithmetic (item 1)

### 2.1 The identity, and what it actually says

`(I1)` `a_L = (q' + eps) d_g = 2 (q' + eps) u_g (mod g)` holds at **45 of 45** (rung, gear) pairs,
0 exceptions; the universal form `(I0)` `v = (3v) d_g (mod g)` at **1200 of 1200** `(v, g)` cells;
and `mu_g(v) = v d_g^{-1} = 3v (mod g)` at **1200 of 1200**. **P1 CONFIRMED.** (The
pre-registered counts "44 pairs" and "1440 cells" were arithmetic slips in enumerating the rungs
and the gear list: there are 9 realised rungs giving 45 (rung, gear) pairs, and the universal check
runs over 10 gears and 120 sizes, 1200 cells. Nothing about the verdict turns on it.)

The honest reading of the check is a negative that shapes the whole branch:

> **The identity is universal, not a property of the letter.** Every distance `v` satisfies
> `mu_g(v) = 3v` at every gear, because `d_g = 3^{-1} (mod g)` at every gear, because
> `6 u_g = +-1`. Written once: the machine lives in the single coordinate `n = 6k`, in which
> **every gear has its teeth at `n = +1` and `n = -1`** and a distance `v` becomes `6v`. There is
> one coordinate for the whole machine, and that is the entire real-teeth input.

What is special about the letter is not that its residue is a multiple of `u_g` -- every size's is
-- but **which** multiple: `6 a_L = 2(q' + eps)`, two away from twice the incoming gear. The
letter is the incoming gear's own tooth distance, so in tooth units the letter gap is a jump of
`+-2` at `q'` itself and of the single integer `2(q' + eps)` at every gear of `M`.

### 2.2 The residue formula, and each gear's role as one congruence in `q' mod g`

With `lam_g := 6x (mod g)` the near end of an `a_L`-gap in tooth units,

    gear g strikes x + s          iff  lam_g = +-1 - 6s                  (0 <= s <= a_L)
    gear g strikes x + a_L + t    iff  lam_g = +-1 - 2(q' + eps) - 6t    (t >= 0)

so both ends open means `lam_g` avoids `{1, -1, 1 - 2(q'+eps), -1 - 2(q'+eps)}`, at most four
classes, and **which four is decided by `q' mod g` alone.** Collapsing four to three or two is the
endpoint cost, and the identity turns it into one congruence per gear:

> **THE ROLE OF A GEAR AT THE LETTER (new in this generality).** `c_g(a_L) = 2` iff
> `q' = -eps (mod g)`; `= 3` iff `q' = -2 eps (mod g)`; `= 4` otherwise. Equivalently `g | q' + eps`
> (`g` divides `a_L`) or `g | q' + 2 eps` (`g` is a closer). Cross-checked against
> `|T_g u (T_g - a_L)|` at every (rung, gear) pair, 0 mismatches.

Worked: gear 5 has `c_5 = 2` at rungs 29 (`q' = 4 mod 5`) and 31 (`q' = 1 mod 5`), `c_5 = 3` at
rungs 7, 23, 37, `c_5 = 4` at rungs 11, 13, 17, 19; gear 29 has `c = 3` at rung 31
(`q' = 2 mod 29`); gears 5 and 7 have `c = 3` at rung 37 (`q' = 2` mod both). This is the parent's
`c_5(a_L)` stratification (`short_letter_row.md` 2.3, cited) with the general gear filled in.

### 2.3 The three corollaries, checked

* **D1, the closer law** (`Leg(a_L) n M` = the prime factors of `q' + 2 eps`): 9 of 9 rungs,
  0 exceptions, and the direct chain-law test `v = 0, +-d_g (mod g)` agrees with `Pad u Leg` at all
  27 (rung, letter) cells. This is the parent's law re-derived in one line from `(I1)`; **cited,
  not claimed.**
* **D2, the pad gears** (`g | a_L` iff `g | q' + eps`, and then `g` strikes exactly `2 a_L / g`
  interior columns of EVERY `a_L`-gap). The only rung in scan range with a pad gear is 23->29
  (`5 | 10`): gear 5 strikes exactly 4 of the 9 interior columns, **0 exceptions in 243,370
  `a_L`-gaps**. Proof: `g | a_L` puts both ends in one class mod `g`, and each tooth then meets the
  interior exactly `a_L / g` times. It shows again in 2.4 as a degenerate range (`4..4`).
* **D3, the specialised obstruction law.** `h` is obstructed at a 2-run `(a, a_L)` iff `h | a`, or
  `h | q' + eps`, or (`h | q' + 2 eps` and `a = -eps d_h (mod h)`). Checked against the generic
  spare-gear obstruction test at **5,400 of 5,400** (rung, gear, neighbour `a <= 120`) cells,
  **0 mismatches. P3 CONFIRMED on all three.**

D3 is the arithmetic form of the parent's lemma at the letter: at every rung at most **one** gear
of `M` divides `q' + eps` and at most **two** divide `q' + 2 eps`, so *at most three gears of the
machine can ever be obstructed for a reason other than dividing the neighbour `a`*. Everything
else must be busy, which is why the spare-gear lemma never fires at an attaining run.

### 2.4 THE FORCED-GEAR LAW (new)

Enumerating, for each gear, the interior strike count over **all** admissible classes `lam_g`
gives an exact range whose minimum is 0 or positive:

> **THE FORCED-GEAR LAW.** Gear `g` strikes at least one interior column of **every** `a_L`-gap
> iff its long arc is too short to hold the gap, `g - d_g <= a_L + 1`, which by
> `d_g = (g + eps_g)/3` and `3 a_L = q' + eps` is the closed condition
>
>     2 g  <=  q' + eps + eps_g + 3          (eps_g = +1 if g = 5 mod 6, -1 if g = 1 mod 6)
>
> i.e. roughly `g <= (q' + 3)/2`: **the bottom half of the machine cannot avoid the letter gap, the
> top few gears can.** Verified by complete class enumeration at all 45 (rung, gear) pairs,
> 0 exceptions.

| rung `q'` | `a_L` | gears FORCED into every `a_L`-gap | gears free to miss it | interior demand `a_L - 1` | sum of forced minima | surplus |
|---|---|---|---|---|---|---|
| 7 | 2 | 5 | - | 1 | 1 | 0 |
| 11 | 4 | 5, 7 | - | 3 | 3 | 0 |
| 13 | 4 | 5, 7 | 11 | 3 | 3 | 0 |
| 17 | 6 | 5, 7, 11 | 13 | 5 | 5 | 0 |
| 19 | 6 | 5, 7, 11 | 13, 17 | 5 | 5 | 0 |
| 23 | 8 | 5, 7, 11, 13 | 17, 19 | 7 | 7 | 0 |
| 29 | 10 | 5, 7, 11, 13, 17 | 19, 23 | 9 | 10 | +1 |
| 31 | 10 | 5, 7, 11, 13, 17 | 19, 23, 29 | 9 | 10 | +1 |
| 37 | 12 | 5, 7, 11, 13, 17, 19 | 23, 29, 31 | 11 | 14 | +3 |

**THE FORCED-COVER COUNT (new, exceptionless).** The sum over gears of the *unavoidable* number of
interior columns each strikes equals the interior demand `a_L - 1` **exactly** at 6 of the 9 rungs
and exceeds it by `1, 1, 3` at the top three. A letter gap is at the exact edge of being covered by
strikes that no phase can avoid: through rung 19->23 there is **no strike to spare inside the
letter gap**, and the slack that appears at 23->29 grows with the machine.

The per-gear ranges show D2 as the degenerate case: at rungs 29 and 31 gear 5's range is `4..4`
(`5 | a_L`), every other gear's has width 1 or 2.

### 2.5 The family is incoherent, and coherence is not additive (item 3, P2)

A member is COHERENT at gear `g` iff `v_g` is the real tooth (`6 v_g = +-1 mod g`), which is
exactly when `(I1)` holds there. Over the 60 members:

| coherent gears of `M` | members | `0 <= E(a_L) <= 3` | `E > 3` | `E < 0` |
|---|---|---|---|---|
| 0 | 6 | 4 | 2 | 0 |
| 1 | 30 | 22 | 7 | 1 |
| 2 | 20 | 12 | 2 | 6 |
| 3 | 4 | 2 | 1 | 1 |

**0 of 60 members are fully coherent** and 56 of 60 have 0, 1 or 2 coherent gears. **P2 CONFIRMED
on both clauses.** But the obedience rate does **not** rise with the coherence count
(0.67, 0.73, 0.60, 0.50): coherence is not additive over gears.

## 3. The attaining run, and the exact CRT search (items 1, 2)

### 3.1 The residue shape of the attaining run is NOT rigid (P4b REFUTED)

Distinct tooth-unit vectors `(lam_g)` of the letter gap's near end over all occurrences of the
attaining 2-run, up to the mirror `lam -> -lam`: **4, 12, 24, 6, 8, 4** at rungs
13, 17, 19, 23, 29, 31 -- exactly one class per occurrence at every rung, no coincidences at all.
**P4b REFUTED decisively**: the attaining run is not one residue shape but as many as it has
occurrences.

What *is* constant is the bottom gear. `lam_5` is `2` (i.e. `x = 2 mod 5`, up to mirror) at every
occurrence at rungs 13, 17, 19 -- forced, since with `a_L = 4` or `6` only one class mod 5 leaves
both ends open -- and `0` at every occurrence at rungs 29 and 31, i.e. **both ends of the letter
gap sit in gear 5's shield class** `x = 0 (mod 5)`, where the pair filter forbids nothing and the
choice is free. At rung 23 both admissible classes occur and in each the shield carries one end.
This is the parent's "the maximising `a` attains the minimum of `C_5(., a_L)`"
(`short_letter_row.md` 2.3, cited) seen from the inside.

### 3.2 The strike table and the sole-striker map (item 1)

At every rung, at every occurrence, **every gear of `M` is a sole striker of some interior column**
and **no gear is confined to the letter gap** -- every gear also strikes inside the neighbour.
**P4c CONFIRMED 6 of 6**; the first half is the parent's 90-of-90 (cited) extended to rung 31.
Columns struck, letter gap / neighbour, at a representative occurrence:

| rung | span | 5 | 7 | 11 | 13 | 17 | 19 | 23 | 29 |
|---|---|---|---|---|---|---|---|---|---|
| 13 | 7 | 2/1 | 1/1 | 1/1 | | | | | |
| 17 | 13 | 2/3 | 2/2 | 1/2 | 1/1 | | | | |
| 19 | 18 | 2/5 | 2/4 | 1/2 | 1/2 | 0/2 | | | |
| 23 | 28 | 3/8 | 2/6 | 2/4 | 1/3 | 1/3 | 1/2 | | |
| 29 | 35 | 4/10 | 2/8 | 2/5 | 2/4 | 1/3 | 1/3 | 1/2 | |
| 31 | 45 | 4/14 | 3/10 | 2/6 | 2/5 | 1/4 | 2/3 | 1/3 | 0/3 |

Gear 5 carries the most in both halves at 6 of 6 rungs whether or not it is a pad gear, so
**P4d as stated (a `Pad(a_L)` clause) is REFUTED and replaced by a gear-5 clause**. The gears the
forced-gear law calls free are exactly those showing `0/..` or `1/..` in the letter half (17 at
rung 19, 29 at rung 31).

The sole-striker map at the rung that matters (`M = {5..29}`, `q' = 31`, word `(10, 35)`, letter
gap first, `x_0 = 22,134,900`, `lam = {5:0, 7:3, 11:9, 13:9, 17:11, 19:8, 23:17, 29:14}`):

    5:[1,14,19,26,31,34,36]  7:[18,23,25,30,37]  11:[13,17,35]  13:[3,7,33,42]
    17:[15,38]  19:[8,27,40]  23:[5,43]  29:[12,22]

28 sole-striker columns of the 44 interior ones, every gear represented, 4 of them inside the
letter gap.

### 3.3 THE EXACT CRT SEARCH: the row of a letter decided scan-free (item 2)

A choice of one class `lam_g` per gear is a configuration; by CRT every configuration occurs in the
period; so the cell `(a_L, a)` is realisable iff some choice covers every interior column while
leaving the three prescribed openings open. That is a covering feasibility problem with at most 10
gears and at most 102 columns, solved exactly by depth-first search on the lowest uncovered column
with a reachability prune.

| rung `q'` | `M` | `F` | `a_L` | cells decided | **search top** | scanned `r(a_L)` | nodes (total / max cell) | time |
|---|---|---|---|---|---|---|---|---|
| 13 | `{5..11}` | 7 | 4 | 5 | **3** | 3 | 33 / 9 | 0.0 s |
| 17 | `{5..13}` | 11 | 6 | 5 | **7** | 7 | 60 / 21 | 0.0 s |
| 19 | `{5..17}` | 18 | 6 | 7 | **12** | 12 | 1,049 / 424 | 0.0 s |
| 23 | `{5..19}` | 25 | 8 | 6 | **20** | 20 | 8,656 / 3,447 | 0.0 s |
| 29 | `{5..23}` | 34 | 10 | 10 | **25** | 25 | 185,919 / 41,628 | 0.1 s |
| 31 | `{5..29}` | 43 | 10 | 9 | **35** | 35 | 2,579,032 / 548,562 | 2.1 s |
| 37 | `{5..31}` | 58 | 12 | 13 | **46** | 46 | 43.8M / 5,980,955 | 36 s |
| **41** | `{5..37}` | 88 | 14 | 26 | **63** | never scanned | 746M / 73,301,869 | 652 s |

**P5a CONFIRMED, 7 of 7** where a scan exists (the pre-registered doubt about reaching rung 37 was
wrong: 36 seconds). Three of these are the LP lane's certified cells and agree exactly
(`20, 25, 35`; `restricted-covering-certificates.md` RESULT 4 via `short_letter_row.md` 4.3-4.4,
cited); `r(12) = 46` at `M = {5..31}` had only ever come from a streamed scan of 33,426,748,355
columns and is **certified scan-free here for the first time**.

The excesses and the pinned bound's slack:

| rung | 13 | 17 | 19 | 23 | 29 | 31 | 37 | **41** |
|---|---|---|---|---|---|---|---|---|
| `a_L + r(a_L) - F` | +0 | +2 | +0 | +3 | +1 | +2 | +0 | **-11** |
| pinned bound `F + 3 - a_L` | 6 | 8 | 15 | 20 | 27 | 36 | 49 | 77 |
| slack `F + 3 - a_L - r(a_L)` | 3 | 1 | 3 | **0** | 2 | 1 | 3 | **14** |

**P5c CONFIRMED exactly** on the seven scanned rungs. The bound is tight at exactly one rung
(19->23), so the search returns the truth and the constant 3 is an equality nowhere else: **the
search cannot produce the constant, because the constant is not what the search computes.** The
last column is section 5.4 and it changes the law.

### 3.4 What kills each cell above the row -- and where the arithmetic runs out

For every cell `a` in `(r(a_L), F]`: is it killed by the gear-5 pair filter alone (gear 5 has no
admissible class -- a one-line, machine-independent arithmetic kill), or does it need the covering
search? And for those that need the search, what is the **shortest sub-interval of the run that
already cannot be covered**? A short one would be a local, proof-shaped obstruction.

| rung | `a_L` | cells above the row | killed by the pair filter | need the search | shortest uncoverable window per hard cell (of span) |
|---|---|---|---|---|---|
| 13 | 4 | 4 | `4, 7` | 2 | 6, 3 (of 9, 10) |
| 17 | 6 | 4 | `8, 11` | 2 | 6, 9 (of 15, 16) |
| 19 | 6 | 6 | `13, 16, 18` | 3 | 14, 13, 8 (of 20, 21, 23) |
| 23 | 8 | 5 | `21` | 4 | 26, 26, 11, 26 (of 30, 31, 32, 33) |
| **29** | 10 | 9 | **none** | 9 | 33, 33, 35, 23, 33, 15, 32, 31, 21 (of 36..44) |
| **31** | 10 | 8 | **none** | 8 | 25, 44, 34, 41, 38, 25, 38, 27 (of 46..53) |
| 37 | 12 | 12 | `49, 54` | 10 | (not computed in this lane) |

**This is the branch's sharpest result and it is negative.** The share of the row's top that the
arithmetic decides on its own runs `2/4, 2/4, 3/6, 1/5, 0/9, 0/8, 2/12` and is **zero at exactly
the two rungs that carry the budget's tightness**, 23->29 and 29->31, because there `5 | a_L` and
the pair filter forbids no neighbour class at all. For those cells the shortest window that already
fails is 15 to 44 columns out of runs 36 to 53 long: the obstruction is not local. **P5b CONFIRMED
and sharpened** -- there is no short certificate, and at the rungs that matter no arithmetic
certificate of any length short of most of the run.

This is the same silence the grandparent recorded three times (`short_letter_row.md` 5: `F_2 - a_L`
vacuous at 29->31, the residue mechanism empty at 29->31, `F_2 - F <= a_L` failing at m17 and m29).
A fourth route reaches the same wall at the same rung.

## 4. The family and the twin rungs (items 3, 4)

### 4.1 The repair experiment, and its control (item 3)

For every family member violating `0 <= E(a_L) <= 3`, put ONE gear back on its real tooth and
recompute; as a control, move the same gear to a different WRONG tooth.

| | moves tried | landing in `0 <= E <= 3` |
|---|---|---|
| gear put on its REAL tooth | 70 | 31 (0.44) |
| gear put on a different wrong tooth | 154 | 82 (0.53) |
| members with `E > 3`, landing at `E <= 3` (real tooth) | 47 | 29 (0.62) |
| the same, control | 103 | 62 (0.60) |

**P6 REFUTED, and this is the item-3 answer.** Putting a gear back on its real tooth repairs a
violating member no more often than moving it to any other wrong tooth -- 0.44 against 0.53
overall, 0.62 against 0.60 on the upper half. By gear the "repairs" are spread
(`5: 3/10, 7: 6/13, 11: 7/15, 13: 10/16, 17: 3/10, 19: 2/6`) with none standing out. (Counting
single-gear repairs per member rather than per move gives 17 of 20, which looks impressive until
the control shows the same rate for a wrong tooth.)

> **The violation is not attributable to any one gear's misplaced tooth.** The identity is a
> statement about the whole tooth vector at once -- one coordinate `n = 6k` for the entire machine
> -- and restoring one coordinate of that vector buys nothing. So the brief's item-3 question
> ("which gear's misplacement lets the neighbour grow past `F + 3 - a_L`") has the answer
> **no gear's**; the count it asks for is `60 of 60 members break the identity at every gear they
> are incoherent at`, which is true and explanatorily empty.

### 4.2 The twin rungs (item 4)

At a twin rung `p = q' - 2` is a gear of `M` and shares the incoming gear's tooth
(`u_p = u_{q'}`, file 02(e)), so `d_p = a_L` and `6 a_L = 2(p + 1) = 2 (mod p)`:

> **THE TWIN-PARTNER LAW (new).** At a twin rung the letter gap's far end sits exactly **two tooth
> units** beyond its near end at the partner -- the same jump as the partner's own two teeth. So
> `lam_p` is forbidden only the three classes `{1, -1, -3}` (`c_p(a_L) = 3`) and the partner
> strikes **at most one** interior column of any `a_L`-gap.

Complete enumeration over classes (a proof for that gear, not a scan):

| rung | `p = q'-2` | admissible `lam_p` | interior strikes: 0 / 1 / more |
|---|---|---|---|
| 7 | 5 | 2 of 5 | 0 / 2 / 0 |
| 13 | 11 | 8 of 11 | 2 / 6 / 0 |
| 19 | 17 | 14 of 17 | 4 / 10 / 0 |
| 31 | 29 | 26 of 29 | 8 / 18 / 0 |

**P7 CONFIRMED on both clauses, 4 of 4 twin rungs, 0 exceptions**, and "more than one" is empty in
all `2 + 8 + 14 + 26 = 50` admissible classes. The partner is therefore always among the *free*
gears of the forced-gear law: `2p = 2q' - 4` never satisfies `2g <= q' + eps + eps_g + 3` once
`q' > 9`.

Does that explain the small excesses at twin rungs (`+2, 0, 0, +2`)? **No.** The partner is free of
the letter gap, so it is available for the neighbour, which if anything makes the neighbour
*longer*; and the non-twin excesses (`+2, +3, +1, 0`) interleave with the twin ones. This agrees
with the grandparent's finding that the twin structure decides the CLOSERS and not the ROW
(`short_letter_row.md` 2.5, cited). What the law adds is the reason: the twin partner is the one
big gear whose position inside the letter gap is completely pinned by the arithmetic, and it is
pinned to "almost never strikes".

## 5. The other letters, and toward the root (item 5)

### 5.1 The divisor forms of all three letters (P8, first clause)

Reading `(I1)` at a general letter `l` with multiplier `m = 3l`: `Leg(l) n M` is the set of gears
dividing `m - 1` or `m + 1`, `Pad(l) n M` the set dividing `m`. Hence

| letter | value | multiplier `3l` | `Pad n M` | `Leg n M` |
|---|---|---|---|---|
| `a_L` | `(q'+eps)/3` | `q' + eps` | gears dividing `q' + eps` | gears dividing `q' + 2eps` |
| `b_L` | `q' - a_L` | `2q' - eps` | gears dividing `2q' - eps` | gears dividing **`q' - eps`** |
| `q'` | `q'` | `3q'` | none | gears dividing `3q' -+ 1` |

**Confirmed at 27 of 27 (rung, letter) cells, 0 exceptions**, with the direct chain-law test
agreeing with `Pad u Leg` everywhere. The `b_L` form is new: the long letter's closers inside `M`
are the prime factors of `q' - eps`, the *other* neighbour of the incoming gear. (So the two
letters of `q'` point at `q' + 2eps` and `q' - eps` -- the same pair of numbers the half-column
fibre theorem attaches to column `u_{q'}`; `half_column.md` 2.1(e), cited.)

### 5.2 The rows of all three letters, by the same search (P8, second clause)

| rung | `F` | `a_L` | `r(a_L)` | `E(a_L)` | `b_L` | `r(b_L)` | `E(b_L)` | `q'` | `r(q')` | `E(q')` |
|---|---|---|---|---|---|---|---|---|---|---|
| 13 | 7 | 4 | 3 | **+0** | 9 | unrealisable | - | 13 | unrealisable | - |
| 17 | 11 | 6 | 7 | **+2** | 11 | 5 | +5 | 17 | unrealisable | - |
| 19 | 18 | 6 | 12 | **+0** | 13 | 7 | +2 | 19 | unrealisable | - |
| 23 | 25 | 8 | 20 | **+3** | 15 | 13 | +3 | 23 | 5 | +3 |
| 29 | 34 | 10 | 25 | **+1** | 19 | 15 | +0 | 29 | 8 | +3 |
| 31 | 43 | 10 | 35 | **+2** | 21 | 27 | +5 | 31 | 14 | +2 |
| 37 | 58 | 12 | 46 | **+0** | 25 | 40 | +7 | 37 | 30 | +9 |

Every value matches the parent's scans (18 of 18). The excess bands are `a_L: [0, 3]`,
`b_L: [0, 7]`, `q': [+2, +9]`, so **`b_L + r(b_L) <= F + 7` at 6 of 6 realised rungs and
`q' + r(q') <= F + 9` at 4 of 4**, and neither is pinned to 3 (`b_L` breaks it at rungs 17, 31, 37;
`q'` at rung 37). The excess is ordered by the letter's tooth-unit length `3l`
(`q'+eps < 2q'-eps < 3q'`) at **8 of 10** available comparisons, the inversions being rung 29
(`E(a_L) = +1 > E(b_L) = 0`) and rung 31 (`E(b_L) = +5 > E(q') = +2`). **P8 CONFIRMED on the
divisor forms and on the ordering clause**; the "b is pinned to a small constant" reading holds
only with `c = 7`.

So the pinned letter is a property of the SHORT letter specifically, and the natural parameter is
`3l`, the number of tooth units the gap spans: the shortest letter is the tightest.

### 5.3 The `q'`-closure, and the gate

`a_L = d_{q'}` means the incoming gear can be phased with one tooth on the middle opening of a
2-run `(a, a_L)`, the other tooth landing `a_L` away on the far side; when `a != 0, a_L (mod q')`
neither end is struck, so `F(M + q') >= a_L + r(a_L)`. It holds at 8 of 8 rungs with slack
`1, 4, 5, 7, 6, 8, 13, 30` (the pre-registered slack list was wrong at one entry: 5, not 12, at
rung 17). **P9 CONFIRMED in content**; it is an instance of the attainment identity (file 08),
**cited, not claimed**, and far weaker than the pinned letter (`F + q'` against `F + 3`).

**The gate, with the row now certified scan-free.** The parent closes the availability gate at
`F + 3 - a_L`; this branch closes it at the certified `r(a_L)` itself -- one to three lower, at
every rung, by an exact residue computation instead of a period:

| rung | `F` | `a_L` | pinned bound `F+3-a_L` | **certified top `r(a_L)`** | deep-chain cap | residual band | cost |
|---|---|---|---|---|---|---|---|
| 19 | 18 | 6 | 15 | **12** | 12 | empty | 1,049 nodes |
| 23 | 25 | 8 | 20 | **20** | 12 | `[13, 20]` | 8,656 nodes |
| 29 | 34 | 10 | 27 | **25** | 21 | `[22, 25]` | 185,919 nodes |
| **31** | 43 | 10 | 36 | **35** | 14 | **`[15, 35]`** | 2,579,032 nodes |
| **37** | 58 | 12 | 49 | **46** | (not measured) | -- | 43.8M nodes |

(deep-chain caps from `availability_gate.md` 2.5, cited). At rung 29->31 this reproduces the LP
lane's `[15, 35]` (270,070 exact rational operations there, 2.6 million integer nodes here) and at
rung 31->37 it adds a certified top where none existed.

### 5.4 Out of sample: the rung nobody can scan, and the law breaks

`M = {5..37}` has a period of 1.24e12 columns, so `r(14)` has never been measured. `a_L(41) = 14`
(`3 * 14 = 42 = q' + 1`, `eps = +1`) and `F({5..37}) = 88` (the recorded ladder;
`neighbour_profile.md` 1 and docs/proofs/18, "`F(37) = 88` exactly by scan"). The pinned letter
predicts `r(14) <= 88 + 3 - 14 = 77` and its lower half predicts `r(14) >= 74`.

> **THE ANSWER: `r(14) = 63`, so `a_L + r(a_L) = 77 = F - 11`.**
>
> The upper half `a_L + r(a_L) <= F + 3` **holds, with slack 14** -- the largest slack on record,
> where every earlier rung had slack 0 to 3.
> The lower half `F <= a_L + r(a_L)` is **REFUTED, by 11.**

The computation: 26 cells decided from `a = 88` down, 746 million nodes, 652 s, no node-cap
failures; every cell `64..88` infeasible and `a = 63` feasible. The witness was reconstructed
explicitly by CRT -- the column

    x = 470,382,204,623

of the period of `{5..37}` has `x`, `x + 14` and `x + 77` open and every one of the other 75
columns between them struck by a gear of `M`, verified directly. Ten of the infeasible cells
(`64, 67, 69, 72, 74, 77, 79, 82, 84, 87`) are killed by the gear-5 pair filter alone -- `a_L = 14`
is `4 (mod 5)`, so `c_5 = 4` and the filter forbids the two neighbour classes `a = 2` and
`a = 4 (mod 5)`, which is exactly that list -- and the other fifteen by the search.

**What this does to the recorded law.** The ladder of `E(a_L) = a_L + r(a_L) - F` now reads

    rung   5->7  11->13  13->17  17->19  19->23  23->29  29->31  31->37  37->41
    E       +2      0      +2       0      +3      +1      +2       0     -11

so `0 <= E <= 3` is **8 of 9, not 9 of 9**, and the failure is not marginal. The two halves part
company:

* the **upper** half `r(a_L) <= F + 3 - a_L` survives at 9 of 9 and is what the availability gate
  uses -- but its slack, `3, 1, 3, 0, 2, 1, 3, 14`, jumps by an order of magnitude at the new rung,
  so the constant 3 is not a law of the ladder, it is a coincidence of the machines small enough to
  scan;
* the **lower** half `F <= a_L + r(a_L)` -- "the letter's 2-run reaches the record" -- is simply
  false once the record grows faster than the letter's row. `F` goes `43, 58, 88` at the last three
  machines while `a_L + r(a_L)` goes `45, 58, 77`; the record's jump to 88 (a `J = 4` fusion with
  word `(28, 37, 12, 11)`, `neighbour_profile.md` 7, cited) leaves the 2-run behind.

## 6. Mechanism

1. **There is one coordinate.** `n = 6k` puts every gear's teeth at `+-1` and turns every distance
   `v` into `6v`. That is the whole real-teeth input, it is why the counterfactual family behaves
   differently (it is the same machine with no common coordinate), and it is why no tooth-invariant
   argument could reach the constant -- the parent's negative, now with its one-line reason.
2. **The letter's number is `2(q' + eps)`.** In that coordinate the letter gap is a jump of one
   integer at every gear at once; the letter is the incoming gear's own tooth distance, so the jump
   is `+-2` at `q'` itself.
3. **Each gear's role is one congruence.** `c_g(a_L) = 2, 3, 4` according as `g | q' + eps`,
   `g | q' + 2eps`, or neither; at most one gear of `M` is in the first class and at most two in
   the second, so at most three gears have any special relation to the letter, and by D3 at most
   three can ever be obstructed for a reason other than dividing the neighbour.
4. **The bottom half of the machine is forced into the letter gap**, by the exact inequality
   `2g <= q' + eps + eps_g + 3`, and the forced minima already sum to the interior demand at 6 of 9
   rungs. The letter gap is covered by strikes no phase can avoid.
5. **And that is exactly why the neighbour is long.** Everything the arithmetic pins about the
   letter gap frees the top gears for the neighbour -- including, at a twin rung, the partner,
   which strikes at most one interior column of the letter gap ever. The arithmetic contains no
   term that limits the neighbour: `r(a_L)` is decided by covering the neighbour's columns under
   the letter's four forbidden classes, and those four classes are the pair filter, which is empty
   exactly when `5 | a_L` -- at the rungs that matter.

So the letter's residues are now completely understood (points 1-4 are exact, exceptionless,
closed-form) and they do not touch the neighbour's length. Section 5.4 says why that had to be so:
the quantity the law compares `a_L + r(a_L)` with is `F(M)`, and `F(M)` grows by a different
mechanism (deep fusions) than the letter's row does.

## 7. The proof attempt, and what it produced

The brief's attempt was: prescribe the letter gap's openings, pin the residues by `q' mod g`, and
compute the maximal blocked neighbour by CRT. The computation was built and it works -- exact,
scan-free, cheap, reproducing every measured row top including six that needed billions of columns,
and reaching one machine beyond every scan. **But it is a decision procedure, not a bound.** It
returns `r(a_L)`; it does not return `F + c - a_L`, and it cannot:

* `F` never enters it. The search knows the letter, the gears and the residues; the record `F(M)`
  is a *different* covering optimum on the same machine, and the arithmetic supplies no map between
  the two. A proof of `a_L + r(a_L) <= F + 3` must compare two optima;
* the bound is tight at 1 of 9 rungs and has slack 14 at the ninth, so no argument can produce it
  as an equality;
* the part of the row's top the arithmetic decides alone vanishes at the two rungs that carry the
  tightness, and the shortest uncoverable window there is 15 to 44 columns of runs 36 to 53 long.
  There is no local certificate to grow into a lemma.

What the attempt did produce and is worth keeping: a certification vehicle far cheaper than the LP
lane's for this family of cells, which certified `r(12) = 46`, `r(25) = 40` and `r(37) = 30` at
`M = {5..31}` scan-free and then decided a rung no scan can reach -- and, in doing so, **refuted
the lower half of the law it was built to prove.**

## 8. What is new

1. **The one-coordinate reading of the real teeth.** `mu_g(v) = 3v` at every gear for every
   distance: the identity the brief asks about is universal, and it is the coordinate `n = 6k`. The
   letter's content is its integer `6 a_L = 2(q' + eps)`. Consequence: the real-teeth input
   available to any proof is one global coordinate, and the family destroys it globally, not gear
   by gear (4.1).
2. **THE FORCED-GEAR LAW**: `g` strikes inside every `a_L`-gap iff `2g <= q' + eps + eps_g + 3`
   (equivalently `g - d_g <= a_L + 1`), 0 exceptions over 45 (rung, gear) pairs by complete class
   enumeration.
3. **THE FORCED-COVER COUNT**: the forced minima sum to exactly `a_L - 1` at 6 of 9 rungs and
   exceed it by 1, 1, 3 at the top three.
4. **THE TWIN-PARTNER LAW**: at a twin rung the far end is exactly two tooth units beyond the near
   end at `p = q' - 2`, so `c_p(a_L) = 3` and the partner strikes at most one interior column of any
   `a_L`-gap -- 50 of 50 admissible classes, 4 of 4 rungs.
5. **D3, the specialised obstruction law** at a 2-run `(a, a_L)`, 5,400 of 5,400 cells; at most
   three gears of `M` can be obstructed for any reason other than dividing the neighbour.
6. **The gear's role as one congruence on `q' mod g`** (the general-gear form of the parent's `c_5`
   stratification) and **the divisor form of the long letter's closers**, `Leg(b_L) n M` = the gears
   dividing `q' - eps`.
7. **The CRT search as a certification vehicle**: 22 of 22 row tops reproduced scan-free, including
   `r(12) = 46`, `r(25) = 40`, `r(37) = 30` at `M = {5..31}`; and the census of what kills each cell
   above the row -- pair filter `2/4, 2/4, 3/6, 1/5, 0/9, 0/8, 2/12`, **zero at the two rungs that
   matter**, shortest uncoverable windows 15 to 44 columns.
8. **The repair control**, which kills the "one gear's misplaced tooth" reading of the family:
   real tooth 0.44, wrong tooth 0.53.
9. **The three letters side by side**, rows certified by the same search: excess bands `[0,3]`,
   `[0,7]`, `[2,9]`, ordered by the tooth-unit length `3l` at 8 of 10 comparisons.
10. **THE OUT-OF-SAMPLE RUNG 37->41**: `r(14) = 63` at `M = {5..37}` with an explicit verified
    witness, a value no scan can produce -- and with it **the refutation of the pinned letter's
    lower half** (`E = -11`) and the collapse of the upper half's tightness (slack 14).

Prior art inside the project: the identity's ingredients are file 02 (the tooth rule) and file 05
(the letters); the closer law, the pair filter and the row profile are `short_letter_row.md`; the
spare-gear lemma, the excess-as-price reading and the family verdict are `pinned_letter.md`; the
windowed dictionary vehicle and the three certified tops are the LP lane
(`restricted-covering-certificates.md` RESULT 4); the attainment identity is file 08; the record
`F(37) = 88` and its `J = 4` word are `neighbour_profile.md` 7 and docs/proofs/18; the half-column
fibre theorem (`half_column.md` 2.1(e)) is the same arithmetic read at the column `u_g`; the
coherence-under-CRT statement is `separation_drives_K.md` N-S2, and this branch's "one coordinate
`n = 6k`" is that statement's `c/r = 1/3` case read on the teeth rather than on separations.
Outside the project: not checked (no web access).

## 9. Verdict

**The pinned letter's arithmetic is now completely known, exact and exceptionless -- and it does
not contain the bound; and the law itself is half false. Node status: FACT (four new exceptionless
laws, a scan-free certification vehicle, 22 of 22 row tops) with one REFUTATION: `F <= a_L + r(a_L)`
fails at rung 37->41 by 11.**

- **Item 1 (the identity and the strike tables): delivered, and it demotes the identity.** `(I1)`
  holds 45 of 45 but is the universal statement `mu_g(v) = 3v` -- the coordinate `n = 6k` -- so it
  says nothing about the letter it does not say about every size. What is special is the letter's
  integer `2(q' + eps)`, and from it: each gear's role is one congruence on `q' mod g`, at most
  three gears have any special relation to the letter, the bottom half of the machine is forced
  into the letter gap by an exact inequality, and the forced minima already cover its interior. The
  attaining run's residue shape is NOT rigid (one class per occurrence), every gear is a sole
  striker, and no gear is confined to the letter gap.
- **Item 2 (the CRT bound): the computation works, the bound does not follow.** The search
  reproduces the certified tops `20, 25, 35` and, newly, `46` at `M = {5..31}` -- 7 of 7 scanned
  rungs, scan-free. It returns `r(a_L)` exactly, never a formula; the pinned bound is tight at 1 of
  9 rungs; and the arithmetic's own share of the kills above the row is **zero** at 23->29 and
  29->31.
- **Item 3 (why the family breaks it): no gear is to blame.** Restoring one real tooth repairs a
  violating member at 0.44 against a control's 0.53; coherence is not additive (0.67, 0.73, 0.60,
  0.50 by coherent-gear count) and 0 of 60 members have it.
- **Item 4 (the twin rungs): a new law that does not explain the excesses.** The partner sees the
  letter gap as its own two-tooth jump, so `c_p = 3` and it strikes at most one interior column
  ever -- which frees it for the neighbour, the opposite of what a small excess needs.
- **Item 5 (toward the root): the gate is certified lower than the pinned bound, and the pinned
  bound is loosening.** Certified tops `12, 20, 25, 35, 46` sit 1 to 3 below `F + 3 - a_L`, the
  residual band at 29->31 is `[15, 35]` at 2.6 million integer nodes, rung 31->37 is closed at 46
  for the first time -- and at rung 37->41 the certified top 63 is **14** below `F + 3 - a_L`. The
  long letter is pinned only with `c = 7`, the padded letter with `c = 9`.

**THE CANDIDATE IS NOW ONE-SIDED.** `a_L + r(a_L) <= F(M) + 3` survives at 9 of 9 rungs;
`F(M) <= a_L + r(a_L)` is REFUTED at rung 37->41 (`77` against `88`). The surviving half is the one
the availability gate uses, so nothing downstream breaks -- but its slack `3, 1, 3, 0, 2, 1, 3, 14`
says the constant 3 is an artefact of the machines small enough to scan, and the honest form of the
gate's top from here on is the **certified row top itself**, computed by the search, not
`F + 3 - a_L`.

What would have to break the surviving half: a machine and an incoming gear with an `a_L`-gap next
to a gap longer than `F + 3 - a_L`. Why the system may not be able to do that: **still not shown,
and now known not to be shown by the letter's residues either.** Two routes are closed behind it --
tooth-invariant covering (the parent) and the letter's own arithmetic (this branch) -- and what both
closings share is that the statement compares the row of one size with `F(M)`, two different optima
of the same covering problem, growing by different mechanisms (5.4). The next child branch should
be about that comparison: **the record gap itself as a 2-run** -- which sizes sit at the two ends of
the record, whether the record's decomposition ever contains an `a_L`-gap, and why the record's jump
to 88 is a `J = 4` fusion while the letter's row is stuck at 2.

## 10. Dead ends, each with its refuting instance

- **"The identity `a_L = 2(q' -+ 1) u_g` is the real-teeth input the proof needs."** It is
  universal: `v = (3v) d_g (mod g)` for every `v` and every gear, 1200 of 1200 cells. By itself it
  distinguishes nothing.
- **The attaining 2-run is a rigid residue shape.** Refuted at every rung: 4, 12, 24, 6, 8, 4
  distinct mirror classes for 4, 12, 24, 6, 8, 4 occurrences -- one each.
- **`Pad(a_L)` gears carry the letter gap.** Refuted: gear 5 carries the most at 6 of 6 rungs,
  including the four where no gear divides `a_L`.
- **A single misplaced tooth carries a family violation.** Refuted by the control: real tooth 0.44,
  wrong tooth 0.53.
- **The coherence count predicts obedience.** Refuted: 0.67, 0.73, 0.60, 0.50 at 0, 1, 2, 3
  coherent gears.
- **The CRT search with the letter's ends prescribed yields `F + c - a_L`.** Refuted: it yields
  `r(a_L)` exactly, and `F + 3 - a_L - r(a_L) = 3, 1, 3, 0, 2, 1, 3, 14`.
- **A local (short-window) certificate for the cells above the row.** Refuted at the rungs that
  matter: shortest uncoverable window 15 to 44 columns of runs 36 to 53 long, and at rungs 29 and
  31 no cell above the row is killed by the pair filter at all.
- **The twin partner explains the small excess at twin rungs.** Refuted by its own law: the partner
  is free of the letter gap, so it is spare for the neighbour.
- **`F(M) <= a_L + r(a_L)` (the pinned letter's lower half).** Refuted at rung 37->41:
  `a_L + r(a_L) = 14 + 63 = 77` against `F({5..37}) = 88`.

## 11. What holds without exception (item 6)

| statement | count | status |
|---|---|---|
| `(I1)` `a_L = 2(q'+eps) u_g (mod g)` at every rung and gear | 45 of 45 | identity, verified |
| `(I0)` `v = (3v) d_g` and `mu_g(v) = 3v` for every distance and gear | 1200 + 1200 cells | identity, verified |
| `c_g(a_L) = 2 / 3 / 4` according as `g \| q'+eps` / `g \| q'+2eps` / neither | 45 of 45 (rung, gear) pairs | proved here + verified |
| the divisor forms of `Pad` and `Leg` for `a_L`, `b_L`, `q'` (incl. `Leg(b_L) n M` = gears dividing `q'-eps`) | 27 of 27 (rung, letter) cells | proved here + verified |
| **the forced-gear law**: `g` forced iff `2g <= q' + eps + eps_g + 3` | 45 of 45, complete class enumeration | proved here + verified |
| **the forced-cover count**: forced minima sum to `a_L - 1` at 6 of 9 rungs, surplus 1, 1, 3 at the top three | 9 of 9 rungs | measured, exact |
| **D2**: `g \| a_L` implies `g` strikes exactly `2 a_L / g` interior columns of every `a_L`-gap | 243,370 of 243,370 gaps at m23 | proved here + verified |
| **D3**: the specialised obstruction law at a 2-run `(a, a_L)` | 5,400 of 5,400 cells | proved here + verified |
| **the twin-partner law**: `c_p(a_L) = 3` and at most one interior strike | 4 of 4 rungs, 50 of 50 admissible classes | proved here + verified |
| the CRT search reproduces the scanned row top | 22 of 22 rows | exact, scan-free |
| every gear of `M` is a sole striker inside the attaining 2-run, and no gear is confined to the letter gap | 6 of 6 rungs, every occurrence | measured (first half is the parent's, extended) |
| `a_L + r(a_L) <= F + 3` (the surviving half of the pinned letter) | 9 of 9 rungs, one out of sample, slack 0 to 14 | measured, unproved |
| `F <= a_L + r(a_L)` (the other half) | 8 of 9 -- **REFUTED at rung 37->41** | refuted here |
