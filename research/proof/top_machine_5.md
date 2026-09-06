# Sliding the split down (branch R4.b.v)

Parent: R4.b, *The top machine on its own terms*. The observation that spawned this branch is
the owner's question of 2026-09-06, verbatim:

> "I wonder how much we learned about the top machine can apply to the bottom one; a way to try
> that would be to imagine a top machine that starts at gear 2, then 3, then 5, etc. and see if
> the proofs hold true there (what is the smallest top machine that retains the simplicity)."

The conjugacy L19 (kernel-checked, `TopMachine.conjugacy`) says that a top machine whose gear
set starts at 5 **is** the bottom machine `{5..q}` written in the column coordinate, because the
map `n -> 6^{-1}(n + 1)` exists exactly when `6` is invertible modulo the wheel, i.e. exactly
when neither 2 nor 3 is a gear. So this experiment measures, law by law, what the top machine's
theory gives the bottom machine - and where the two small gears 2 and 3 break it.

Construction rule (R4, owner): the machine on the raw line, on its own. No clutch beyond the
conjugacy itself; no interpretation against the twin conjecture.

Numbering of new laws continues from **L50** (document 1 reached L21, document 2 L38,
document 3 L45, document 4 L46; there is a numbering clash between documents 2 and 3, both of
which use L30-L38 for different laws, so every citation below names the document).

---

## 0. Setup: the machine with a small gear

**The machine.** Gear set `G` of pairwise coprime integers, each `>= 2`. Gear `g` strikes the
pair `n = (n, n + 2)` iff `n = 0` or `n = -2 (mod g)`. Open pairs `O`, wheel `W = prod g`,
`m = |G|`, `q' = min G`.

**The three tooth regimes.** Everything in this branch turns on what the pair `{0, -2}` looks
like modulo `g`:

| `g` | teeth mod `g` | the piece a gear shows in a window | slots per turn |
|---|---|---|---|
| `2` | `{0}` - the two teeth **coincide** (`0 = -2 mod 2`) | a single cell, every 2 | `1 = g - 1` |
| `3` | `{0, 1}` - the two teeth are **adjacent** (`-2 = 1 mod 3`) | a **solid** domino `{x, x + 1}`, every 3 | `1 = g - 2` |
| `>= 5` | `{0, g - 2}` - separated by 2, and `g - 2 >= 3` | a **gapped** domino `{x, x + 2}` | `g - 2 >= 3` |

Gear 5 is the first gear whose domino is gapped *with the gap inside the period*: its open
residues are `{1, 2, 4}`, a long arc `1, 2` and the shield `4`, and its two teeth `0, 3` have an
open cell (`4`) between them going one way round and two open cells (`1, 2`) the other way.
Gear 3 has no long arc at all; gear 2 has no second tooth at all.

**Vocabulary.** Unchanged from the first pass: gear, pair, tooth, strike, open, wheel, slot,
run, step-2 chain, gap, record `F_top`, letters `{2, g - 2}`, domino, shield `n = -1`, origin
clump, mirror `n -> -n - 2`, walk `L(x)`, mex.

**The two coordinates.** Pair coordinate: the raw line, teeth `{0, -2}`. Column coordinate:
`k` with members `6k - 1`, `6k + 1`; gear `g` strikes column `k` iff `k = +-u_g (mod g)` with
`u_g = 6^{-1} mod g`. The conjugacy `n -> 6^{-1}(n + 1)` carries one to the other and exists iff
`gcd(6, W) = 1`.

---

## 1. Pre-registered predictions and scorecard

Written before any computation of this branch. Predictions are derived from each law's stated
mechanism and from the exact hypothesis its Lean statement carries
(`research/proof/top_machine_lean.md`), not from any measurement.

### 1.1 The three transitions, predicted as mechanisms

**T5 (gear 5, the gapped domino appears).** At `q' = 5` the gear's slots first form two arcs of
lengths `g - 3 = 2` and `1`; the piece is a distance-2 domino confined to one parity class; and
`6` is invertible, so the machine is conjugate to a bottom machine. Predicted: every
*structural* law of the pair view (partner, arcs, forbidden gap, run and chain ceilings, census,
correlation, transform) holds verbatim at `q' = 5`, with the ceilings taking the values
`q' - 3 = 2` and `q' - 2 = 3`. Predicted also: the *record* laws (parity law, mex, universal
multiplicity) do **not** hold at `q' = 5` except for `m <= 2`, because they need `q' > 2m`.

**T3 (gear 3, the domino becomes solid).** `-2 = 1 (mod 3)`, so gear 3's struck set is
`{x : x = 0, 1 (mod 3)}`: a *solid* block of two adjacent cells, repeating every 3. Predicted
consequences: (i) the long arc is empty, so there are no two adjacent open pairs, `N_1 = 0`, and
the longest run of open pairs is 1 while `q' - 3 = 0` - the value formula of L10 dies but its
counting form `A(L) = prod (g - 2 - L)` still gives the right answer 0; (ii) the piece crosses
parity, so the parity law's mechanism is destroyed outright and the record grows like `3` per
gear rather than `2`; (iii) the forbidden gap 4 survives, because it follows from the partner
law which is still true (`0 -> -2 = 1 -> 0`); (iv) the letters become `{2, 1}` and the
alternation degenerates; (v) the machine folds: open pairs are exactly `n = 2 (mod 3)`, so L20
("no fold") is false; (vi) the twin-candidate (three-teeth) view is **empty**, since
`{0, -1, -2} = Z_3`.

**T2 (gear 2, the teeth coincide).** `0 = -2 (mod 2)`, one tooth, one slot. Predicted: (i) the
slot count is `g - 1 = 1`, so `prod (g - 2)` is wrong (it gives 0) and every count law needs the
factor `1` at `g = 2`; (ii) the symmetry group loses a factor, `c = +1` and `c = -1` being the
same map mod 2, so the group is `(Z/2)^{m - 1}`, not `(Z/2)^m`; (iii) the mirror has two fixed
points, not one, because `W` is even; (iv) all open pairs are odd, so every gap is even and
`N_1 = 0`; (v) `n = 2` and `n = -4` are struck, so the antipode fails; (vi) the counting bound
`L <= 2m/(1 - 2 H_S)` is vacuous for every gear set containing 2, since `1/2` alone reaches the
threshold; (vii) with 2 and 3 together the open pairs are exactly `n = 5 (mod 6)` - **the
bottom machine's anchor is recovered as a two-gear top machine**, and the record grows by about
6 per further gear because a distance-2 domino can cover at most one residue `5 (mod 6)`.

### 1.2 The law table: predicted smallest `q'` at which each law still holds

`rel` means the law's truth is a relation between `q'` and `m`, not an absolute floor.

| law | document | mechanism / Lean hypothesis | predicted smallest `q'` |
|---|---|---|---|
| L1 teeth, separation 2 | 1 | none | **2** |
| L1 slot count `g - 2` | 1 | `3 <= g` (`card_open_residues`) | **3** |
| L2 arcs `(g - 3, 1)` | 1 | `3 <= g` (`open_residues`); needs a nonempty long arc | **5** |
| L2 shield `n = -1` open | 1 | `2 <= g` (`not_strikes_neg_one`) | **2** |
| L3 partner law | 1 | none (`partner`, `strikes_iff_domino`) | **2** |
| L4 forbidden gap 4 | 1 | none (`no_gap_four`) | **2** |
| L5 wheel count `prod (g - 2)` | 1 | gears `>= 3`, coprime (`wheel_count`) | **3** |
| L6 antipode `n = 2, -4` | 1 | gears `>= 5` (`two_open`) - predicted not sharp | **3** |
| L6 origin clump `2(q'-3)+1` | 1 | gears `>= q' >= 3` (`origin_clump`) | **3** |
| L7 mirror | 1 | none (`open_mirror`) | **2** |
| L7 unique fixed point | 1 | needs `W` odd | **3** |
| L8 group `(Z/2)^m` | 1 | gears prime, `>= 5` (`affine_group`); needs `+-1` distinct | **3** |
| L9 gap counts even except `d = 1` | 1 | mirror; and `N_1` odd | **5** |
| L10 run `= q' - 3`, chain `= q' - 2` | 1 | `5 <= q'` (`no_long_run`) | run **5**, chain **3** |
| L11 run spectrum `= D^2 prod(g-2-L)` | 1 | CRT only | **2** |
| L12 chain law | 1 | none (`chain_law`) | **2** |
| L13 merge law | 1 | none (`merge_law`) | **2** |
| L14 letters `{2, g-2}`, alternation | 1 | none | **3** (degenerate at 2) |
| L15 dominoes `prod(g-4)`, chains `prod(g-3)` | 1 | CRT | **5** |
| L16 record as an exact cover | 1 | CRT only | **2** |
| L17 parity law `2m - (m mod 2)` | 1 | odd, `> 2m + 1` (`parity_law`) | rel: `q' > 2m + 1` |
| L18 universal multiplicity | 1 | large-gear regime | rel: `q' > F + 1` |
| L19 the conjugacy to the column | 1 | `gcd(6, W) = 1` (`exists_column`) | **5** |
| L20 no fold (flat mod 2, 3, 6) | 1 | needs `2, 3 not in G` | **5** |
| L21 the gear zone | 1 | smoothness, no hypothesis | **2** |
| L22 the gap census law | 2 | CRT, pairwise coprime | **2** |
| L23 universal signature | 2 | every gear `> d + 2` | rel: `q' > d + 2` |
| L24 `N_3 = N_5` iff gears `> 7` | 2 | collapse at `g | 3, 5, 7` | **11** |
| L25 degree law / gear-independence | 2 | universal signature | rel: `q' > d + 2` |
| L26 `r(d)` = parity covering number | 2 | universal signature | rel: `q' > F + 1` |
| L27 `N_d` odd only at `d = 1` | 2 | every gear odd, `N_1` odd | **5** |
| L28 joint census = CRT product | 2 | CRT | **3** (2 with a per-gear tooth count) |
| L28 `2^m` all-struck classes, exactly 2 total collisions | 2 | two distinct teeth per gear | **3** |
| L29 collision law, waste 0/1 | 2 | every gear `> L + 1` | rel: `q' > F + 1` |
| L30 record of a triple/quadruple, one bit | 2 | only 7 sub-threshold among 7..97 | **7** |
| L31 sub-threshold reduction | 2 | self-consistent | **2** |
| L32 range record = first hit on the census | 2 | statistics of the census | **2** |
| L33 record blocks pinned mod the small gears | 2 | measured | **2** |
| L34 corridor; **no anchor is possible** | 2 | anchoring needs `g - 2 <= 2`, i.e. `g <= 4` | **5** |
| L35 palindromic gap word from the shield | 2 | the mirror fixes the shield | **2** |
| L36 metric anchor (`q'` fixes run, chain, clump) | 2 | the arcs | **5** |
| L37 removal law (`W`, open, dominoes divide) | 2 | CRT | `W` **2**, open **3**, dominoes **5** |
| L38 removal independence | 2 | large-gear regime | rel: `q' > F + 1` |
| L30 the mex form | 3 | every gear `> 2m` | rel: `q' > 2m` |
| L31 the location bound | 3 | odd, `> 2m + 1` | rel: `q' > 2m + 1` |
| L32 the general mex form | 3 | none | **2** |
| L33 the counting bound `2m/(1 - 2H_S)` | 3 | `H_S < 1/2` | **5**, and only for `m <= 3` |
| L34 the twin-candidate mex | 3 | every gear `> 3m`; needs `T` nonempty | **5** |
| L35 the triple record `3m` | 3 | `>= 3m + 3` | rel, and `q' >= 5` |
| L36 the `C`-identities | 3 | none | **2** |
| L37 the closed form of `C(j)` | 3 | every gear `>= j + 2` | rel: `q' >= j + 2` |
| L38 the hop collapse (chain `<= 2`) | 3 | `g > F_G + 3` | rel |
| L39 the nested form | 3 | none | **2** |
| L40 per-gear transform | 3 | two distinct teeth | **3** |
| L41 full spectral support | 3 | `g` odd for the cosine argument | **2** by direct computation |
| L42 XOR bias `prod (g - 4)` | 3 | character sum, two teeth | **3** (zero at 2) |
| L43 XOR run `= F_top` | 3 | an exactly-covered record block exists | rel |
| L44 correlation product `B(d)` | 3 | two teeth per gear | **3** |
| L45 holes `{4}` (pairs), `{2, 3}` (triples) | 3 | partner law; triple view nonempty | pairs **2**, triples **5** |

### 1.3 The five definitions of "retains the simplicity", with predicted thresholds

**(a) The mex closed form is exact.** `L(x) = mex {(-x) mod g, (-x-2) mod g}`. Predicted
threshold `q' > 2m`, and predicted **sharpened**: the correct statement is
`mex(x) < q' ==> L(x) = mex(x)`, with the failures confined to `mex(x) >= q'`; predicted
0 exceptions to the sharpened form at every `q'` down to 2. Predicted failure count rising from
0 as `q'` crosses `2m` (the known instance `{7,11,13,17}`, 36 failures of 17,017).

**(b) The parity law holds.** Predicted threshold `q' > 2m + 1`, sharp: predicted the first
failure at `q' = 2m + 1` exactly (`{7,11,13}` at `m = 3`), to be probed with odd composite but
pairwise coprime gears (`9, 15, 21, 25, 27`) so that the boundary can be hit at every `m`.

**(c) The record is a tiling by free dominoes** (its covering number equals the parity covering
number `ceil(ceil(L/2)/2) + ceil(floor(L/2)/2)`). Predicted threshold `q' > F + 1`, i.e.
`q' > 2m` for even `m` and `q' > 2m - 1` for odd `m` - marginally weaker than (b).

**(d) The symmetry group is exactly `(Z/2)^m`.** Predicted **absolute**: `q' >= 3`. The only
obstruction is gear 2, where `+1 = -1`.

**(e) The record depends on `m` and on the gears `<= F + 1` only** (L31 of document 2).
Predicted **absolute and unconditional**: holds at every `q' >= 2`, because the statement is
self-consistent (the small part is defined by `F` itself).

Predicted answer to the owner's question: **the smallest top machine that retains the
simplicity is `q' = 5` for every structural law and for the conjugacy, and there is no absolute
answer for the record laws - they hold exactly while `q' > 2m + 1`, a relation, so "simple"
means "few gears relative to the smallest", and the bottom machine `{5..q}` is never simple in
that sense once it has three gears.**

### 1.4 The transfer to the bottom machine, predicted

Predicted: via L19, the gear set `{5..q}` in the column coordinate is the bottom machine's
anchor-235 machine, and

* every **counting and symmetry** law transfers unchanged (L5, L7, L8, L9, L11, L15, L22, L25,
  L28, L44, and the census machinery), because the conjugacy is a bijection of residues;
* every **metric** law transfers only in a modified form, with the letters `{2, g - 2}` replaced
  by `{2u_g, g - 2u_g}`: the run ceiling `q' - 3` becomes the bottom's long arc, the mex form
  needs the teeth `+-u_g`, the hop collapse needs longer chains;
* the laws that fail with no analogue are those that name the number 2 as a tooth separation:
  the partner law (bottom distance `2u_g ~ g/3`), the forbidden gap 4, the origin clump's width,
  and the parity law (whose piece must lie in one parity class).

**Predicted corrected closed form for the next open column of `{5..q}`** (the deliverable):
with `u_g = 6^{-1} mod g`, `a_g = (u_g - x) mod g`, `b_g = (-u_g - x) mod g`, and a cut-off `B`,

        M_B(x) = mex ( union_g { a_g + k g, b_g + k g : k >= 0, term <= B } )

and the certificate: **if `M_B(x) <= B` then `L_col(x) = M_B(x)`**, exactly, for every gear set.
Predicted: `B = 2m` suffices at every `x` for `{5..q}`, `q <= 31`; predicted 0 mismatches over
10^6 positions per gear set; predicted term count `2 sum_g (1 + floor(B/g))`, a few dozen.

### Scorecard

| # | Prediction | Result |
|---|---|---|
| S1 | the law table above, law by law | |
| S2 | T5: the gapped domino, structural laws verbatim at `q' = 5` | |
| S3 | T3: solid domino, no adjacent open pairs, fold mod 3, empty triple view, gap 4 still forbidden | |
| S4 | T2: one tooth, `(Z/2)^{m-1}`, all gaps even, `{2,3}` recovers the six-fold | |
| S5 | (a) mex exact iff `q' > 2m`; sharpened `mex < q' ==> exact`, 0 exceptions at every `q'` | |
| S6 | (b) parity law exactly while `q' > 2m + 1`, first failure at `q' = 2m + 1` | |
| S7 | (c) domino tiling iff `q' > F + 1` | |
| S8 | (d) symmetry group `(Z/2)^m` iff `q' >= 3`, absolute | |
| S9 | (e) L31 sub-threshold reduction holds at every `q' >= 2` | |
| S10 | the transfer list (unchanged / modified / no analogue) | |
| S11 | the corrected column mex is exact on `{5..q}`, `q <= 31`, 10^6 positions | |

---

## 2. Setup as computed

Scripts in `research/topmachine/r5/`, results (untracked) in `.../results/`. Every count below is
exact over a full wheel period or exact over the stated range; nothing is sampled except where
the word "sampled" appears.

| script | what it computes |
|---|---|
| `common.py` | the machine with small gears allowed: teeth, slots, masks, walks, gaps, the all-struck count, the covering record, the mex forms, the column coordinate |
| `s1_structure.py` | the structural laws of documents 1 and 2 on 23 wheels with `q' = 2, 3, 5, 7, 11, 13` |
| `s2_walk.py` | the walk, the mex forms, the `C`-identities, the spectrum, the XOR bit, the correlation, the triple view, the covering record, on 18 wheels |
| `s3_pairwise.py` | the chain, merge and alternation laws on 18 ladder steps; the hop collapse; the removal law on 4 chains; the corridor; the odd gap length |
| `s4_thresholds.py` | the five definitions of simplicity and their exact boundaries, with odd composite pairwise-coprime gears (9, 15, 21, 25) to hit `q' = 2m + 1` at every `m` |
| `s5_bottom.py` | the conjugacy checked directly; the column machine `{5..q}`; the corrected column mex at 10^6 positions |
| `s6_range.py` | the fixed gear set `{q'..97}` on `[1, 10^7]` for `q' = 2, 3, 5, 7, 11, 13` |
| `s7_fold.py` | the anchor rescaling law for `p = 2`, `p = 3`, `p = 6` |

The essential engineering point: every count law of the three documents is written with the
constant `2` (as in `g - 2`, `prod (g - 2)`, `prod (g - 4)`) because a gear was assumed to have
**two distinct teeth**. In the scripts the constant is replaced by `|{0, -2} mod g|`, which is 2
for `g >= 3` and 1 for `g = 2`. Where a law is reported below as "holds in the corrected form",
that is the only change made.

---

## 3. Results

### 3.1 The law table

23 wheels, smallest gear 2, 3, 5, 7, 11, 13, up to six gears; plus 18 ladder steps, 6 range
machines to `N = 10^7`, and 8 column machines `{5..q}`. "measured" is the smallest smallest-gear
at which the law **as written in its document** is still true; "rel" means the law's truth is a
relation between `q'` and `m`, with the boundary given in 3.3.

| law | doc | predicted | **measured** | evidence, and the corrected form when it fails |
|---|---|---|---|---|
| L1 teeth, separation 2 | 1 | 2 | **2** | every gear; at `g = 2` the two teeth coincide, at `g = 3` they are adjacent |
| L1 slot count `g - 2` | 1 | 3 | **3** | 23 wheels. Corrected: `g - |{0,-2} mod g|`, which is `1` at `g = 2`; that holds at 2 |
| L2 arcs `(g - 3, 1)` | 1 | 5 | **5** | `g = 3` has one arc of length 1 (no long arc), `g = 2` one arc of length 1 |
| L2 shield `n = -1` open | 1 | 2 | **2** | 23 of 23 |
| L3 partner law | 1 | 2 | **2** | 23 wheels, every gear, **0 exceptions** - `0 -> -2` is a strike at distance 2 for `g = 3` (`-2 = 1`) and for `g = 2` (`-2 = 0`) too |
| L4 forbidden gap 4 | 1 | 2 | **2** | **0 gaps of 4** in 23 wheels and in 6 range machines to `10^7`, including `q' = 2, 3` |
| L5 wheel count `prod (g - 2)` | 1 | 3 | **3** | 23 wheels; at `q' = 2` the product is 0 and the truth is `prod (g - |teeth|)` |
| L6 antipode `n = 2, -4` | 1 | 3 | **3** | true at `q' = 3`, false at `q' = 2` (gear 2 strikes `n = 2`). The Lean hypothesis `5 <= g` is **not sharp**; 3 suffices |
| L6 origin clump `2(q'-3)+1` | 1 | 3 | **3** | `q' = 3`: measured 1 = predicted 1; `q' = 2`: measured 1, formula gives `-1` |
| L7 mirror `n -> -n - 2` | 1 | 2 | **2** | **0 mismatches**, 23 wheels |
| L7 unique fixed point | 1 | 3 | **3** | one fixed point for odd `W`; **two** when 2 is a gear (`n = -1` and `n = W/2 - 1`) |
| L8 group `(Z/2)^m` | 1 | 3 | **3** | brute force over all `W^2` affine maps, 7 wheels: order `2^{#odd gears}`, i.e. `2^{m-1}` when 2 is a gear. `b = c - 1` in every case; adjacency group `Z/2` in every case |
| L9 gap counts even except `d = 1` | 1 | 5 | **5** | at `q' <= 3` the count at `d = 1` is 0. The true law is L51 below (exactly one odd length, whatever `q'`) |
| L10 run `= q' - 3` | 1 | 5 | **5** | `q' = 3` and `q' = 2` give 1. Corrected: `max(q' - 3, 1)` |
| L10 chain `= q' - 2` | 1 | 3 | **3** | corrected: **the smallest ODD gear minus 2** - 14 of 14, 0 exceptions (gear 2 constrains no step-2 chain, both members having the same parity) |
| L11 run spectrum | 1 | 2 | **2** | corrected to `A(L) = prod max(g - L - 2, 0)` for `L >= 2` (the run spectrum is its second difference): **138 of 138 run counts exact**, `L = 2..7`, 23 wheels |
| L12 chain law | 1 | 2 | **2** | 18 ladder steps, **260,284 opening pairs, 0 exceptions**, including `M` containing 2 and 3 |
| L13 merge law | 1 | 2 | **2** | 18 ladder steps, **36,754 gaps, 0 exceptions** |
| L14 letters, alternation | 1 | 3 | **2** | 18 ladder steps, **918 struck runs, 0 exceptions**; the letters are `{2 mod g, (g-2) mod g}`, i.e. `{2, 1}` at `g = 3` and `{0}` at `g = 2`, and the alternation still holds |
| L15 dominoes `prod (g - 4)` | 1 | 5 | **5** | corrected `prod max(g - 4, 0)`: 0 at `q' <= 3`, exact at every `q'`, 23 wheels. Member-sharing `prod (g - 3)` holds at 3, fails at 2 |
| L16 record as an exact cover | 1 | 2 | **2** | **18 of 18** wheels: the covering record equals the scanned record, `q'` from 2 to 13 |
| L17 parity law | 1 | rel `q' > 2m+1` | **rel, sharpened** | 3.3(b): holds iff `q' >= 2m + 1` (`m` even) or `q' >= 2m + 3` (`m` odd); 35 sets, 7 boundary pairs, 0 exceptions |
| L18 universal multiplicity | 1 | rel | **rel** | fails at `q' = 5`: `{5,7,11,13}` has 4 record blocks, `{5,11,13,17}` 2, against 24 for large-gear `m = 4` |
| L19 the conjugacy | 1 | 5 | **5** | 5 wheels, **0 mismatches**; the map does not exist for `q' <= 3` because `gcd(6, W) > 1`. **This is the exact sense in which `q' = 5` is the bottom machine** |
| L20 no fold | 1 | 5 | **5** | `q' = 2`: every open pair odd; `q' = 3`: every open pair `= 2 (mod 3)`; `q' = 5`: flat mod 2, 3, 6 |
| L21 the gear zone | 1 | 2 | **not tested** | the gear-zone statement is about the *in-use* machine (gears `(q, sqrt N]`) and was not re-tested here. What was measured on `[1, 10^7]` with the fixed set `{q'..97}` is that the density matches the corrected CRT product to four figures at every `q'` (ratios 0.9998 to 1.0016) and that no gap of 4 occurs |
| L22 the gap census law | 2 | 2 | **2** | **0 mismatches**, 23 wheels, every gap length to `d = 12` - the inclusion-exclusion is written with `|E_g(S)|` and so absorbs every collapse at `g = 2, 3` automatically |
| L23 universal signature | 2 | rel `q' > d + 2` | **rel** | at `q' = 5` the signature is universal only for `d <= 2` |
| L24 `N_3 = N_5` iff gears `> 7` | 2 | 11 | **11** | measured `(N_3, N_5)`: `(18,22)`, `(186,250)`, `(486,538)`, `(2610,3686)` at `q' = 5`; `(6,0)`, `(42,0)`, `(189,0)`, `(378,0)`, `(819,0)` at `q' = 3`; equal from `q' = 11` |
| L25 degree law | 2 | rel | **rel** | same hypothesis as L23 |
| L26 `r(d)` = parity covering number | 2 | rel | **rel** | `F = max{d : r(d) <= m} - 1` gives 5 for `{5,7,11}` against the true 9 |
| L27 `N_d` odd only at `d = 1` | 2 | 5 | **5** | replaced by L51 |
| L28 joint census = CRT product | 2 | 2 | **2** | 23 wheels: all-struck classes `= prod |teeth_g|` exactly, i.e. `2^m` for odd gears and `2^{m-1}` when 2 is a gear |
| L28 exactly two total collisions | 2 | 3 | **2** | `n = 0` and `n = -2` are the only classes where every gear uses the same tooth: **23 of 23**, gear 2 included |
| L29 collision law, waste 0/1 | 2 | rel | **rel** | needs every gear `> L + 1`; a gear `<= L` repeats inside the window |
| L30 record of a triple, one bit | 2 | 7 | **7** | with small gears the record of a triple takes many values: 5 (`{11,13,17}`), 6 (`{7,11,13}`), 9 (`{5,7,11}`), 11 (`{2,3,5}`), 14 (`{3,5,7}`) |
| **L31 sub-threshold reduction** | 2 | 2 | **2** | **27 families, 117 gear sets, 0 disagreements**, with small parts `{}`, `{2}`, `{3}`, `{5}`, `{7}`, `{2,3}`, `{3,5}`, `{5,7}`, `{7,11}`, `{2,3,5}`. **The one record law that survives all the way down** |
| L32 range record, first hit | 2 | 2 | untested here | the range records of the fixed sets are recorded (3.5) but the first-hit prediction was not re-tested |
| L33 record blocks pinned | 2 | 2 | untested here | - |
| L34 corridor; no anchor | 2 | 5 | **5** | measured corridors: `{2}` 1 slot of 2, `{3}` 1 of 3, `{2,3}` **1 of 6**, `{2,3,5}` 3 of 30, `{3,5}` 3 of 15, `{5,7}` 15 of 35 (0.43), `{5,7,11}` 135 of 385, `{7,11}` 45 of 77 (0.58). An anchoring gear needs `g <= 4`: **exactly 2 and 3, and no others** |
| L35 palindromic gap word | 2 | 2 | **2** | 23 of 23 |
| L36 metric anchor | 2 | 5 | **5** | run, chain and clump are functions of `q'` alone only from `q' = 5`; below that the chain ceiling is set by the smallest **odd** gear and the run ceiling is stuck at 1 |
| L37 removal: `W` divides by `q'` | 2 | 2 | **2** | 16 steps, 4 chains |
| L37 removal: open divides by `q' - 2` | 2 | 3 | **3** | removing gear 2 leaves the open count **unchanged** (the ratio is `g - 1 = 1`) |
| L37 removal: `N_1` divides by `q' - 4` | 2 | 5 | **5** | at `q' = 5` the ratio is 1 (gear 5 forbids no domino); at `q' = 3` both counts are 0 |
| L37 removal: `F` falls by 3 or 1 | 2 | rel | **rel** | with small gears the drop is huge: `{2,3,5,7,11} -> {3,5,7,11}` is `41 -> 17` |
| L38 removal independence | 2 | rel | **rel** | large-gear regime only |
| L30 the mex form | 3 | rel `q' > 2m` | **rel, sharpened** | 3.3(a) and L50: the sharp sufficient condition is `F_top < q'`, strictly weaker than `q' > 2m` |
| L31 the location bound | 3 | rel | **rel** | same boundary as L17 |
| L32 the general mex | 3 | 2 | **2** | exact at every `q'` by construction; the content is the term count (3.4) |
| L33 the counting bound | 3 | 5 | **5, and only for small `m`** | `H_S >= 1/2` from gear 2 alone, and from `{3,5,7}`; alive at `{5,7}` (bound 4, **tight**), `{5,7,11}` (19.1 v 9), `{5,11,13,17}` (13.3 v 10); dead at `{5,7,11,13}` (`H_S = 0.511`) |
| L34 the twin-candidate mex | 3 | 5 | **5** | the triple view is **empty** whenever 2 or 3 is a gear (`{0,-1,-2} = Z_3` and `= Z_2`): 9 of 9 wheels give 0 twin candidates |
| L35 the triple record `3m` | 3 | rel, `>= 5` | **rel, `>= 5`** | at `q' = 5`, `F_3 = 18, 34, 28` against `3m = 9, 12, 12` |
| L36 the `C`-identities | 3 | 2 | **2** | all four identities, **18 of 18 wheels, 0 mismatches**, `q'` from 2 to 13 |
| L37 the closed form of `C(j)` | 3 | rel `q' >= j + 2` | **rel** | - |
| L38 hop chain `<= 2` | 3 | rel | **2** | **the bound holds at every `q'`**: 18 ladder steps, longest chain exactly 2 everywhere, and the layered landing is exact (0 mismatches). What fails below the hypothesis is the *rule*: `{2,3,5}+7`, `{3}+5`, `{3,5}+7`, `{3,5,7}+11` have double hops that are not "tooth `-2` and lower gap 2" |
| L39 the nested form | 3 | 2 | **2** | landings exact in all 18 layers |
| L40 per-gear transform | 3 | 3 | **2 (corrected)** | with the per-gear factor written as `-(1/g) sum_{t in teeth} w^{-a t}` the product formula is exact to `6.7e-16` at **59,141 frequencies in 17 wheels**, `q'` from 2 to 13 |
| L41 full spectral support | 3 | 2 | **2** | minimum `|O^(a)|` from `2.06e-2` down to `3.3e-7`, never 0, in 17 wheels |
| L42 XOR bias `prod (g - 4)` | 3 | 3 | **2 (corrected)** | the true bias is `prod (g - 2 |teeth_g|)`, exact in 18 of 18: `-1` per gear 3 (so `-3, -21, -189, -819`) and **`0` whenever 2 is a gear** |
| L43 XOR run `= F_top` | 3 | rel | **rel** | holds at `{5,7,11}` (9 = 9), `{5,11,13,17}` (10 = 10); fails at `{5,7,11,13}` (10 v 13), `{3,5,7}` (5 v 14), `{2,3,5}` (3 v 11) |
| L44 correlation product | 3 | 3 | **2 (corrected)** | the general form `B(d) = prod_g (g - |{0,-2,-d,-d-2} mod g|)` has **0 mismatches over 720 values** (40 distances, 18 wheels); the document's three-case form `{g-2, g-3, g-4}` fails only at gear 2 |
| L45 pair hole `{4}` | 3 | 2 | **2** | the only hole below the record at `q' >= 5`; at `q' = 3` the holes are every `d` not divisible by 3, at `q' = 2` every odd `d` |
| L45 triple holes `{2, 3}` | 3 | 5 | **5** | at `q' = 5` the holes are `d = 2, 3 (mod 5)`, not `{2,3}`: gear 5's triomino makes the next twin candidate `1` or `4, 5, 6` away |

### 3.2 The three transitions, as three machines

**Gear 5: the gapped domino, and the bottom machine.** Teeth `0` and `3`; slots `{1, 2, 4}`; a
long arc `1, 2` of length `g - 3 = 2` and the shield `4`. The piece a gear shows in a short
window is the distance-2 domino `{x, x + 2}` (from tooth `3` to tooth `5`), with **one open cell
between its two ends**, and the long letter `{x, x + 3}` when the window is long enough. This is
the first gear at which the domino is genuinely gapped inside the period, and therefore the first
gear at which the domino-machine description of documents 1-3 applies verbatim: the arcs, the run
ceiling `q' - 3 = 2`, the chain ceiling `q' - 2 = 3`, the clump `2(q' - 3) + 1 = 5`, the forbidden
gap 4, `N_1 = prod (g - 4)`, the flat distribution mod 2, 3 and 6 - all measured, all exact, at
every `{5, ...}` wheel tested.

It is also the first gear set for which `6` is invertible, so **`q' = 5` is exactly where the
conjugacy exists and the top machine IS the bottom machine `{5..q}` in the column coordinate**
(0 mismatches, 5 wheels). Below 5 there is no column coordinate to be conjugate to, because 2 and
3 are the gears that *make* it.

What does *not* survive at `q' = 5` is the record theory: `F` is 9, 13, 10, 24 at `{5,7,11}`,
`{5,7,11,13}`, `{5,11,13,17}`, `{5,7,11,13,17}` against the parity values 5, 8, 8, 9, because
gear 5 (and 7, 11, 13) sits inside the record window and shows its long letter. The mex form
fails at 2.9% to 6.3% of positions.

**Gear 3: the domino becomes solid.** `-2 = 1 (mod 3)`, so gear 3's struck set is
`{n = 0, 1 (mod 3)}` - a **solid** two-cell block repeating every 3, and its single slot is the
shield `n = 2 = -1`. Measured consequences, all exact:

* **no two open pairs are ever adjacent** (`N_1 = 0` in every wheel containing 3), so the longest
  run of open pairs is 1 while the formula `q' - 3` gives 0; the corrected count
  `A(L) = prod max(g - L - 2, 0)` gives 0 for `L >= 2` and is right;
* the step-2 chain ceiling is still `q' - 2 = 1` (`n` and `n + 2` cannot both be `2 mod 3`);
* **the forbidden gap 4 survives** - the partner law is still true (`0 -> -2 = 1 -> 0`), 0 gaps of
  4 in 5 wheels and in the `10^7` range machine - but it is no longer the *only* hole: every gap
  is a multiple of 3;
* **the line is folded**: every open pair is `n = 2 (mod 3)`, so L20 is false;
* the letters are `{2, 1}`, and the alternation law still holds (918 struck runs, 0 exceptions);
* the piece **crosses parity**, which is why the parity law's mechanism is gone: the records of
  `{3,5,7}`, `{3,5,7,11}`, `{3,5,7,11,13}` are 14, 17, 41 against parity values 5, 8, 9;
* the twin-candidate (three-teeth) view is **empty**: `{0, -1, -2} = Z_3`, so no `n` starts three
  consecutive open integers. Measured 0 in every wheel containing 3.

**Gear 2: the teeth coincide.** `0 = -2 (mod 2)`: one tooth, one slot, `g - 1 = 1`. Measured
consequences, all exact:

* every count law needs the factor `1` instead of `g - 2 = 0`; with that one substitution the
  wheel count, the census law, the correlation, the transform and the XOR bias are all exact;
* **all open pairs are odd**, so every gap is even, `N_1 = 0`, and the antipode `n = 2` is struck;
* the mirror has **two** fixed points (`W` even) instead of one;
* the symmetry group is `(Z/2)^{m-1}`: `c = +1` and `c = -1` are the same map modulo 2. Brute
  force over all `W^2` affine maps at `W = 30, 70, 210, 286`: orders 4, 4, 8, 4, exactly
  `2^{#odd gears}`, every one of the form `n -> c(n + 1) - 1`;
* gear 2 constrains **no** step-2 chain at all, so the chain ceiling is set by the smallest
  **odd** gear: `q_odd - 2`, 14 of 14;
* removing gear 2 leaves the open count unchanged (the removal ratio is `g - 1 = 1`);
* the counting bound `2m/(1 - 2 H_S)` is **vacuous for every gear set containing 2**, because
  `1/2` alone reaches the threshold;
* **with 2 and 3 together the open pairs are exactly `n = 5 (mod 6)`** - one slot in six, the
  bottom machine's anchor recovered as a two-gear top machine. Measured corridor densities:
  `{2}` 1/2, `{3}` 1/3, `{2,3}` **1/6**, `{2,3,5}` 3/30.

**The anchoring dichotomy, exactly.** A gear folds the line (leaves one slot) iff
`g - |teeth| = 1`, i.e. `g = 3` (two teeth, adjacent) or `g = 2` (one tooth). Every `g >= 5`
leaves at least 3 slots. So **the two anchoring gears are 2 and 3 and there are no others** -
document 2's L34 ("an anchoring gear needs `g <= 4`") confirmed from below, by exhibiting the two
gears it excludes and measuring exactly what each destroys.

### 3.3 The five definitions of simplicity, and their exact boundaries

**(a) The mex closed form is exact.** `L(x) = mex {(-x) mod g, (-x - 2) mod g : g in G}`. The
document-3 hypothesis is `q' > 2m`. Measured failure rate over the full period:

| `q'` | wheels | mex failures |
|---|---|---|
| 2 | 6 | 30.3% to 69.6% of positions |
| 3 | 4 | 22.7% to 45.4% |
| 5 | 3 | **2.86%** (`{5,7,11}`, 11 of 385), 6.25%, 3.08% |
| 7 | 3 | 0% (`{7,11,13}`), **0.21%** (`{7,11,13,17}`, 36 of 17,017), 0.65% |
| 9 | 3 | 0%, 0%, 0.042% (`{9,11,13,17,19}`, 174 of 415,701) |
| 11, 13, 15, 17, 21, 25 | 12 | **0%** |

The failure count is 0 exactly when `F_top(G) < q'`: **26 of 26 gear sets**, including the
composite-gear probes `{9,11,13}`, `{9,11,13,17}`, `{15,17,19}`, `{21,23,29,31}`, `{25,29,31,37}`.
The criterion `F < q'` is strictly weaker than `q' > 2m`: **`{6, 11, 13}`** has `q' = 6 = 2m`, so
the document's hypothesis fails, but `F = 5 < 6` and the mex form is exact at all 858 positions.
It is also not necessary: `{6,7,11}` and `{10,11,13,17,19}` have `F = q'` and the form is still
exact. The proved statement is L50 below.

**Threshold: `q' > 2m` suffices (document 3); `F_top(G) < q'` is the sharp sufficient condition.**

**(b) The parity law holds** (`F = 2m - (m mod 2)`), and **(c) the record is a tiling by free
dominoes** (its covering number equals the parity covering number). These two turned out to be
**the same condition**: over 28 gear sets, `pieces = parity pieces` iff every gear used in the
optimal cover contributes a distance-2 domino or a singleton, iff `F = 2m - (m mod 2)`.
**28 of 28, 0 exceptions.**

The boundary, probed with odd composite pairwise-coprime gears so that `q'` can equal `2m + 1` at
every `m`:

| `m` | `2m + 1` | `q' = 2m - 1` | `q' = 2m + 1` | `q' = 2m + 3` | boundary |
|---|---|---|---|---|---|
| 2 | 5 | `{3,5}` F = 8, no | `{5,7}` F = 4, **yes** | `{7,11}` F = 4, yes | `q' >= 5` |
| 3 | 7 | `{5,7,11}` F = 9, no | `{7,11,13}` F = 6, **no** | `{9,11,13}` F = 5, yes | `q' >= 9` |
| 4 | 9 | `{7,...}` F = 9, no | `{9,11,13,17}` F = 8, **yes** | `{11,...}` F = 8, yes | `q' >= 9` |
| 5 | 11 | `{9,...}` F = 13, no | `{11,13,17,19,23}` F = 10, **no** | `{13,...}` F = 9, yes | `q' >= 13` |
| 6 | 13 | `{11,...}` F = 16, no | `{13,17,19,23,29,31}` F = 12, **yes** | `{15,...}` F = 12, yes | `q' >= 13` |
| 7 | 15 | `{13,...}` F = 16, no | `{15,17,19,23,29,31,37}` F = 14, **no** | `{17,...}` F = 13, yes | `q' >= 17` |
| 8 | 17 | `{15,...}` F = 20, no | `{17,19,23,29,31,37,41,43}` F = 16, **yes** | `{19,...}` F = 16, yes | `q' >= 17` |

**The threshold alternates with the parity of `m`: `q' >= 2m + 1` for even `m`, `q' >= 2m + 3`
for odd `m`. 7 boundary pairs, `m = 2..8`, 0 exceptions.** Document 1's hypothesis `q' > 2m + 1`
is sufficient at both parities and **not necessary at even `m`**, where the gear `2m + 1` itself
is harmless.

The mechanism is the long letter, counted exactly. To beat the parity value the machine must
cover `L = F + 1` cells, and the only parity-crossing piece is a gear's long letter `g - 2`
(odd), available iff `g <= L + 1`. For **even** `m`, `L = 2m + 1` has `m + 1` even cells and `m`
odd cells; spending one long letter (one cell of each parity) leaves `m` evens and `m - 1` odds,
needing `m/2 + m/2 = m` more pieces - `m + 1` gears in all, one too many. For **odd** `m`,
`L = 2m` has `m` of each; one long letter leaves `m - 1` of each, and `m - 1` is even, so
`(m-1)/2 + (m-1)/2 = m - 1` more pieces suffice - `m` gears exactly. **The long letter is worth a
cell only at odd `m`.**

**(d) The symmetry group is exactly `(Z/2)^m`.** Brute force over all `W^2` affine maps of `Z_W`,
7 wheels: orders 4, 4, 8, 8, 8, 4, 8 at `{2,3,5}`, `{2,5,7}`, `{2,3,5,7}`, `{3,5,7}`, `{5,7,11}`,
`{2,11,13}`, `{3,5,11}` - that is `2^{#odd gears}` every time, `b = c - 1 (mod W)` every time, and
the adjacency-preserving subgroup is `{id, mirror}` every time.
**Threshold: absolute, `q' >= 3`.** Gear 2 is the only obstruction and the loss is exactly one
factor of 2.

**(e) The record depends on `m` and on the gears `<= F + 1` only** (document 2's L31).
**27 families, 117 gear sets, 0 disagreements**, with large gears drawn from four disjoint pools
(`29..37`, `41..47`, `53..61`, `67..73`) and small parts `{}`, `{2}`, `{3}`, `{5}`, `{7}`,
`{2,3}`, `{3,5}`, `{5,7}`, `{7,11}`, `{2,3,5}`. Examples: every `m = 4` set with small part
`{2,3}` has `F = 17`; every `m = 5` set with small part `{2,3,5}` has `F = 29`; every `m = 6` set
with small part `{2,3,5}` has `F = 41`.
**Threshold: absolute and unconditional, `q' >= 2`.**

**Summary of the five.**

| definition | threshold | kind |
|---|---|---|
| (a) mex closed form exact | `F_top(G) < q'` (implied by `q' > 2m`) | relation |
| (b) parity law | `q' >= 2m + 1` (`m` even), `q' >= 2m + 3` (`m` odd) | relation |
| (c) record is a free-domino tiling | **the same condition as (b)**, 28 of 28 | relation |
| (d) symmetry group `(Z/2)^m` | `q' >= 3` | **absolute** |
| (e) record depends on `m` and the small gears | none | **absolute, always** |

### 3.4 What transfers to the bottom machine

By L19 the gear set `{5..q}` in the column coordinate is the bottom machine's anchor-235 machine.
Checked directly: `n -> 6^{-1}(n + 1)` carries the pair-coordinate open set onto the column open
set with **0 mismatches** at `{5,7}`, `{5,7,11}`, `{5,7,11,13}`, `{5,7,11,13,17}`, `{7,11,13}`,
and the two open counts agree with `prod (g - 2)` in all five.

**Holds unchanged** (counting and symmetry, coordinate-free): L5 the wheel count; L7 the mirror
and its unique fixed point; L8 the symmetry group `(Z/2)^m` and the adjacency subgroup `Z/2`;
L11 the run-start counts as counts of that coordinate's runs; L16 the record as an exact cover;
L22 the gap census law (with the column's tooth set); L25 the degree law; L28 the joint census
and the two total collisions; L31 the sub-threshold reduction; L36 (document 3) the
`C`-identities; L41 full spectral support; L44 the correlation as a product; L51 the unique odd
gap length.

**Holds in a modified form, with the modification stated exactly** (metric; the tooth separation
2 becomes `2u_g` with `u_g = 6^{-1} mod g`):

| law | the modification, measured on `{5..q}` |
|---|---|
| L2 arcs `(g - 3, 1)` | arcs `(g - 2u_g - 1, 2u_g - 1)`: measured `5:(2,1)`, `7:(4,1)`, `11:(6,3)`, `13:(8,3)`, `17:(10,5)`, `19:(12,5)`, `23:(14,7)`, `29:(18,9)`, `31:(20,9)`. Gears 5 and 7 are unchanged because `u = 1`; from 11 on the short arc is no longer a single slot |
| L14 letters `{2, g - 2}` | letters `{2u_g, g - 2u_g}`: `5:{2,3}`, `7:{2,5}`, `11:{4,7}`, `13:{4,9}`, `17:{6,11}`, `19:{6,13}`, `23:{8,15}`, `29:{10,19}`, `31:{10,21}` |
| L10 the run ceiling | the long arc of the smallest gear: **measured 2 for every `{5..q}`, `q <= 31`** - the bottom's alignment law |
| L30 (document 3) the mex | teeth `+-u_g` instead of `{0, -2}`, and the recurrences are needed: L57 |
| L38 (document 3) the hop collapse | the chain is still `<= 2` (4 column layers, 0 chains of 3, landings exact), but the double-hop rule becomes "the lower gap equals the **forward letter of the landing's tooth**" (`2u_g` from `-u_g`, `g - 2u_g` from `+u_g`): **10,860 hits, 0 exceptions** |
| L44 the correlation | `c_g(d) = g - 2` if `d = 0`, `g - 3` if `d = +-2u_g`, `g - 4` otherwise |

**Fails with no analogue** (the laws that name the number 2 as a tooth separation):

| law | why |
|---|---|
| L3 the partner law | the partner of a strike is at distance `2u_g`, which differs from gear to gear, so the struck set is not a union of dominoes of one width |
| L4 the forbidden gap 4 | measured: `{5,7}` still forbids 4 (both gears have `u = 1`), but **`{5,7,11}` has no hole at all** - every gap from 1 to 6 occurs. The forbidden gap dies with the first gear whose `u_g > 1` |
| L6 the origin clump `2(q' - 3) + 1` | a distance on the raw line |
| L17 the parity law, L26, L29 | a distance-`2u_g` piece does not lie in one parity class, so there is no parity obstruction to tiling |
| L15 `prod (g - 4)` as *adjacent* openings | the number is the same in both coordinates; the object it counts is not |
| L21 the gear zone | a smoothness statement about the raw line |

**The sharp form of the transfer: the anchor rescaling law.** Adding the anchoring gears back
does not perturb the record - it *rescales* it, exactly:

        F(G + {2})     = 2 F_2(G) + 1
        F(G + {3})     = 3 F_3(G) + 2
        F(G + {2, 3})  = 6 F_col(G) + 5

where `F_r(G)` is the record of the same gear set in the sub-lattice coordinate - the machine
with teeth `{0, -2 r^{-1}}` modulo each gear - and `F_col = F_6` is the bottom machine's record in
**columns**. **30 of 30 exact** (10 base sets, three anchors each). Examples:

| `G` | `F_col(G)` | `F(G + {2,3})` | `6 F_col + 5` |
|---|---|---|---|
| `{5}` | 1 | 11 | 11 |
| `{5,7}` | 4 | 29 | 29 |
| `{5,7,11}` | 6 | 41 | 41 |
| `{5,7,11,13}` | 10 | 65 | 65 |
| `{5,7,11,13,17}` | 17 | 107 | 107 |
| `{7,11,13,17}` | 8 | 53 | 53 |

**The corrected closed form for the next open column** (the deliverable). With
`u_g = 6^{-1} mod g`, `a_g = (u_g - x) mod g`, `b_g = (-u_g - x) mod g` and a cut-off `B`,

        M_B(x) = mex ( union_{g in G} { a_g + k g , b_g + k g : k >= 0 , term <= B } )
        if M_B(x) <= B then the next open column after x is exactly x + M_B(x).

Measured on `{5..q}` for `q = 7, 11, 13, 17, 19, 23, 29, 31`, at every position of the wheel when
the wheel is below 10^6 and at 200,000 evenly spaced positions of `[0, 10^6)` otherwise:

| gears | `m` | record on the range | `B = 2m`: certified / bad / terms | `B = F`: certified / bad / terms | plain mex (no recurrences) |
|---|---|---|---|---|---|
| `{5,7}` | 2 | 4 | 34 of 34, 0, 4 | 34 of 34, 0, 4 | 0 of 34 wrong |
| `{5,7,11}` | 3 | 6 | 383 of 383, 0, 8 | 383 of 383, 0, 8 | 3 wrong |
| `{5..13}` | 4 | 10 | 4,967 of 5,003, 0, 12 | 5,003 of 5,003, 0, 14 | 141 wrong |
| `{5..17}` | 5 | 17 | 84,503 of 85,081, 0, 16 | 85,081 of 85,081, 0, 26 | 4,561 wrong |
| `{5..19}` | 6 | 24 | 198,409 of 200,000, 0, 20 | 200,000 of 200,000, 0, 36 | 12,991 wrong |
| `{5..23}` | 7 | 29 | 198,690 of 200,000, 0, 26 | 200,000 of 200,000, 0, 46 | 17,292 wrong |
| `{5..29}` | 8 | 32 | 198,892 of 200,000, 0, 30 | 200,000 of 200,000, 0, 52 | 20,784 wrong |
| `{5..31}` | 9 | 41 | 199,246 of 200,000, 0, 34 | 200,000 of 200,000, 0, 70 | 24,102 wrong |

**890,501 certified walks, 0 mismatches.** At `B = 2m` the form certifies 99.2% to 99.7% of
positions with 4 to 34 terms and never gives a wrong answer on a certified position; at `B = F`
it certifies every position with 4 to 70 terms. The uncorrected form - two residues per gear, no
recurrences - is wrong at 12% of positions for `{5..31}`, so the recurrences are not a
technicality: **the small gears 5, 7, 11, 13 recur inside the window and each contributes
`2 floor(B/g) + 2` numbers, not 2.**

### 3.5 The machine on a range with a small gear

Fixed gear set `{q'..97}` on `[1, 10^7]`:

| `q'` | gears | density | corrected CRT product | ratio | longest pair-free run | at | gaps of 4 | gaps of 1 |
|---|---|---|---|---|---|---|---|---|
| 2 | 25 | 0.0191664 | 0.0191485 | 1.0009 | 485 | 7,443,912 | **0** | 0 |
| 3 | 24 | 0.0383602 | 0.0382970 | 1.0016 | 221 | 8,724,258 | **0** | 0 |
| 5 | 23 | 0.1150577 | 0.1148911 | 1.0014 | 102 | 9,820,402 | **0** | 52,974 |
| 7 | 22 | 0.1916104 | 0.1914852 | 1.0007 | 65 | 714,641 | **0** | 264,276 |
| 11 | 21 | 0.2680944 | 0.2680793 | 1.0001 | 41 | 6,170,818 | **0** | 615,809 |
| 13 | 20 | 0.3275885 | 0.3276525 | 0.9998 | 32 | 9,632,632 | **0** | 967,386 |

The corrected CRT product (factor `1/2` at gear 2) predicts the density to four figures at every
`q'`; the forbidden gap 4 holds on the range at every `q'`, including 2 and 3; and adjacent open
pairs disappear entirely at `q' <= 3`.

---

## 4. Laws

Numbered from L50. `q'` is the smallest gear, `q_odd` the smallest odd gear, `m = |G|`,
`t_g = |{0, -2} mod g|` (2 for `g >= 3`, 1 for `g = 2`).

**L50 (the sharp mex criterion).** For any gear set and any `x`, let
`M(x) = mex {(-x) mod g, (-x - 2) mod g : g in G}`. Then

        M(x) < q'   ==>   L(x) = M(x) ,

and consequently `F_top(G) < q'` implies the mex closed form is exact at every position.

*Proof.* If `M(x) < q'` then for each `j < M(x)` some gear has `(-x) mod g = j` or
`(-x - 2) mod g = j`; since `j < q' <= g` and both sides lie in `[0, g)`, the congruence is an
equality and `x + j` is struck. At `j = M(x) < q' <= g` no gear lists `j`, and again the
congruence is an equality, so `x + M(x)` is open. QED. Conversely `M(x) >= q'` forces
`L(x) >= q'`, so every failure of the mex form sits at a position whose walk is at least `q'`.

*Evidence.* **94,774 positions with `M(x) < q'` across 18 wheels with `q'` from 2 to 13, 0
exceptions**; and in all 18 wheels every mismatch has `M(x) >= q'` (18 of 18). The criterion
`F < q'` predicts "0 failures" correctly in **26 of 26** gear sets, composite gears included.
*Strictly weaker than document 3's `q' > 2m`*: `{6,11,13}` has `q' = 2m = 6`, `F = 5 < 6`, and 0
failures in 858 positions. *Not necessary*: `{6,7,11}` and `{10,11,13,17,19}` have `F = q'` and
still 0 failures. **`q' > 2m` implies `F <= 2m < q'`, so L50 contains document 3's L30.**

**L51 (exactly one odd gap length, and which).** In every wheel exactly one gap length has an odd
count, and it is the length of the unique **mirror-self-paired** gap: the gap `[a, a + d]` with
`2a + d + 2 = 0 (mod W)`, the one carried to itself by `n -> -n - 2`.

*Evidence.* 15 wheels, `q'` from 2 to 13: exactly one odd length and exactly one self-paired gap,
the two agreeing every time. The odd length is `1` for every `q' >= 5` (recovering document 1's
L9), `3` or `15` or `21` at `q' = 3`, and `6` at `q' = 2`. *Document 1's L9 ("every length has an
even count except `d = 1`") is false as soon as `N_1 = 0`; L51 is the true statement and it needs
no hypothesis at all.* The self-paired gap at `{7,11,13}` is `[499, 500]`, a mirror pair with
`499 + 500 = -2 (mod 1001)`: **the antipodal adjacent open pair, identified as a property of the
machine rather than of a coordinate.**

**L52 (every counting law is a tooth-count law).** Replacing the constant 2 by `t_g` everywhere,

        slots of one gear          g - t_g
        wheel count                prod (g - t_g)
        starts of a run of L >= 2  prod max(g - L - 2, 0)
        starts of a step-2 chain of L   prod (g - |{0, -2, ..., -2L} mod g|)
        correlation B(d)           prod (g - |{0, -2, -d, -d - 2} mod g|)
        gap census N_d             the L22 formula, unchanged
        striker-parity bias        prod (g - 2 t_g)
        per-gear transform         -(1/g) sum_{t in teeth} w_g^{-a t}
        all-struck classes         prod t_g

every one of these is exact for **every** gear set of pairwise coprime integers `>= 2`.

*Evidence.* 23 wheels: wheel count 23 of 23, run counts 138 of 138 (`L = 2..7`), step-2 chain
counts 92 of 92 (`L = 2..5`), `N_1 = prod max(g - 4, 0)` 23 of 23, the census law every `d <= 12`
- **0 mismatches in all of them**; **720 correlation values in 18 wheels, 0 mismatches**; **59,141 frequencies in 17 wheels, max error 6.7e-16**; the parity bias
exact in 18 of 18 (and **0** whenever 2 is a gear, since `2 - 2 = 0`). *The documents' `g - 2`,
`g - 4`, `prod (g - 3)` forms are the `g >= 3` (resp. `g >= 5`) specialisations.*

**L53 (the two ceilings, corrected).** The longest run of consecutive open pairs is
`max(q' - 3, 1)`; the longest step-2 chain of open pairs is `q_odd - 2`, the smallest **odd** gear
minus two.

*Mechanism.* A run of `L >= 2` needs every gear to miss `L + 2` consecutive residues, impossible
unless `g >= L + 3`; a run of 1 always exists. A step-2 chain lives inside one parity class, on
which gear 2 imposes nothing at all, so gear 2 is invisible to the chain ceiling.
*Evidence.* run: 23 wheels; chain: **14 of 14, 0 exceptions**, including `{2,17,19}` (chain 15),
`{2,11,13}` (9), `{2,5,7}` (3), `{2,3,5}` (1).

**L54 (the symmetry group, corrected).** The affine maps preserving the open-pair set are exactly
`n -> c(n + 1) - 1` with `c = +-1` modulo every gear; the group has order
`2^{#{g in G : g odd}}`, and the adjacency-preserving subgroup is `Z/2`.

*Mechanism.* Modulo 2 the two signs coincide, so gear 2 contributes one sign choice, not two.
*Evidence.* brute force over all `W^2` affine maps of `Z_W` at `W = 30, 70, 105, 165, 210, 286,
385`: orders 4, 4, 8, 8, 8, 4, 8 - `2^{#odd gears}` every time, `b = c - 1` every time.
*Document 1's L8, and the Lean `affine_group` whose hypothesis is `5 <= g` and prime, is the
odd-gear case.*

**L55 (the sharp parity threshold; and the tiling definition is the same definition).**
`F_top(G) = 2m - (m mod 2)` if and only if the record cover is a tiling by free distance-2
dominoes, and that happens exactly when

        q' >= 2m + 1   (m even)        q' >= 2m + 3   (m odd) .

*Mechanism.* To exceed the parity value the machine must cover `L = F + 1` cells, and the only
parity-crossing piece is a gear's long letter `g - 2` (odd), available iff `g <= L + 1`. Counting
the two parity classes of `[0, L)`: at even `m` one long letter still leaves `m` pieces to buy
with `m - 1` gears; at odd `m` it leaves `m - 1` pieces to buy with `m - 1` gears, and succeeds.
*Evidence.* 35 gear sets, `m = 2..8`, with odd composite pairwise-coprime gears 9, 15, 21, 25 to
reach `q' = 2m + 1` at every `m`: **7 boundary pairs, 0 exceptions**. The tiling/parity
equivalence: **28 of 28**. *Document 1's L17 hypothesis `q' > 2m + 1` is sufficient at both
parities and not necessary at even `m` - `{9,11,13,17}`, `{13,17,19,23,29,31}` and
`{17,19,23,29,31,37,41,43}` all satisfy the parity law with `q' = 2m + 1`.*

**L56 (THE ANCHOR RESCALING LAW).** Let `p` be a gear leaving exactly one slot, i.e. `p = 2` or
`p = 3`. Then, exactly,

        F(G + {2})    = 2 F_2(G) + 1 ,
        F(G + {3})    = 3 F_3(G) + 2 ,
        F(G + {2, 3}) = 6 F_col(G) + 5 ,

where `F_r(G)` is the record of `G` in the sub-lattice coordinate - the machine with teeth
`{0, -2 r^{-1}}` modulo each gear - and `F_col = F_6` is the record of the **bottom machine** `G`
in the column coordinate.

*Mechanism.* If a gear leaves one slot mod `r`, every open pair lies in that class, so a maximal
struck run is exactly `r` times a struck run of the sub-lattice machine plus the `r - 1`
positions of the class that fall outside it at the two ends. The sub-lattice machine has teeth
`{0, -2}` rescaled by `r^{-1}`; for `r = 6` that is precisely `+-6^{-1}`, the column coordinate,
which is L19.
*Evidence.* **30 of 30 exact**, 10 base gear sets and three anchors each, records from 5 to 107.
*New; the exact quantitative statement of what the bottom machine's anchor does to the top
machine's record - it multiplies it by 6 and adds 5, and changes nothing else.*

**L57 (the next open column of the bottom machine, in closed form, with a certificate).** For the
bottom machine `{5..q}` with `u_g = 6^{-1} mod g`, `a_g = (u_g - x) mod g`,
`b_g = (-u_g - x) mod g`, and any cut-off `B`,

        M_B(x) = mex ( union_g { a_g + k g , b_g + k g : k >= 0 , term <= B } )
        M_B(x) <= B   ==>   the next open column after x is exactly x + M_B(x) ,

at a cost of `2 sum_g (1 + floor(B/g))` numbers.

*Proof.* Every `j <= B` is correctly classified: `x + j` is struck iff `j` is congruent to `a_g`
or `b_g` modulo some gear, and every such `j <= B` appears among the listed progressions. So if
the mex of the listed set is at most `B`, it is the true walk. QED.
*Evidence.* **890,501 certified walks, 0 mismatches**, over `{5..q}` for `q = 7..31`; `B = F`
certifies every position at 4 to 70 terms, `B = 2m` certifies 99.2-99.7% at 4 to 34 terms.
Without the recurrences the form is wrong at 12% of positions for `{5..31}`. *This is the bottom
machine's next-opening formula: the top machine's mex transported by L19 with the small gears'
recurrences restored.*

**L58 (the hop collapse transfers; the double-hop rule changes).** Adding a gear `g` with
`g > F_G + 3` to a machine `G`, the hop chain has length at most 2 in either coordinate; a double
hop occurs iff the `G`-gap at the landing equals the **forward letter of the landing's tooth** -
`2` in the pair coordinate (document 3's L38), `2u_g` or `g - 2u_g` in the column coordinate,
according to which tooth was hit.

*Evidence.* pair coordinate: 18 ladder steps including `M` containing 2 and 3, longest chain
exactly 2 in every one, layered landing exact (0 mismatches). Column coordinate: 4 layers of
`{5,7,11,13,17}`, **10,860 hits, 0 exceptions** to the modified rule, longest chain 2.
*The chain bound survives the change of coordinate; the rule that names the number 2 does not.*

**L59 (the anchoring dichotomy).** A gear folds the line - leaves exactly one open residue - iff
`g - t_g = 1`, i.e. `g = 3` (two teeth, adjacent) or `g = 2` (one tooth). What each destroys,
exactly: **gear 3** destroys the long arc (`N_1 = 0`, no two open pairs adjacent), the parity
mechanism (its piece is solid and crosses parity), the flatness mod 3, and the twin-candidate view
(`{0,-1,-2} = Z_3`, so no three consecutive open integers exist at all); **gear 2** destroys the
second tooth (`prod (g - 2)`, the antipode, the second sign of the symmetry group, the odd gaps,
the parity bias, and the counting bound). Everything else in documents 1-3 that is not a *value*
survives both.

*Evidence.* the whole of 3.1 and 3.2; in particular the partner law, the forbidden gap 4, the
mirror, the chain law, the merge law, the alternation, the census law, the correlation, the
transform, the `C`-identities, the covering record and the sub-threshold reduction all hold with
`q' = 2`.

---

## 5. What is new

1. **The record laws are the only ones that need a large smallest gear, and the boundary is a
   parity alternation.** `q' >= 2m + 1` for even `m` and `q' >= 2m + 3` for odd `m` (L55), sharp
   at seven boundaries; and the two definitions "the parity law holds" and "the record is a
   free-domino tiling" are the same condition (28 of 28). Document 1's `q' > 2m + 1` is not
   necessary at even `m`.
2. **The mex closed form's real hypothesis is `F_top < q'`, not `q' > 2m`** (L50), with a
   one-line proof and a witness (`{6,11,13}`) separating the two. Every failure sits at a
   position whose walk is at least `q'`.
3. **The anchor rescaling law** (L56): `F(G + {2,3}) = 6 F_col(G) + 5`, exact, 30 of 30 - the
   quantitative answer to the owner's question about the record.
4. **The bottom machine's next-opening formula** (L57): the mex over the gears' two column teeth
   *with their recurrences*, self-certifying, 890,501 walks and 0 mismatches on `{5..q}` to
   `q = 31`, at 4 to 70 numbers per call. The uncorrected two-per-gear form is wrong 12% of the
   time.
5. **Exactly one gap length has an odd count, and it is the mirror-self-paired gap** (L51) - a
   hypothesis-free statement that replaces document 1's L9 and identifies the antipodal adjacent
   open pair as a property of the machine.
6. **Every counting law is a tooth-count law** (L52): one substitution, `2 -> |{0,-2} mod g|`,
   carries the wheel count, the run counts, the census, the correlation, the transform, the
   parity bias and the all-struck count down to gear 2 with 0 mismatches everywhere.
7. **The chain ceiling belongs to the smallest odd gear** (L53), because gear 2 is invisible to a
   step-2 chain: 14 of 14.
8. **The symmetry group is `(Z/2)^{#odd gears}`** (L54), brute-forced over all `W^2` affine maps
   at seven wheels: gear 2 costs exactly one factor of 2, because `+1 = -1` mod 2.
9. **The two anchoring gears are 2 and 3 and there are no others** (L59) - document 2's L34 seen
   from below, with an exact list of what each destroys and what survives.
10. **The hop collapse survives the change of coordinate** (L58) with a modified double-hop rule:
    10,860 column hits, 0 exceptions.

**Prior art met and stopped.** The rescaling `F(G + {2,3}) = 6 F_col(G) + 5` is the familiar
observation that a Jacobsthal-type function of a sifted set scales with the modulus of the fold;
it is used here as an exact identity between two machines, not as an asymptotic.
`prod (g - 2)`, `prod (g - 3)`, `prod (g - 4)` are the usual Schemmel / Hardy-Littlewood local
factors. Nothing above is an asymptotic and nothing re-derives one.

---

## 6. Verdict

**The top machine's theory splits in two, and the split is not where the documents put it.** One
half is *structure*: the partner law, the forbidden gap 4, the mirror, the chain, merge and
alternation laws, the census law, the correlation product, the spectral product and full support,
the `C`-identities, the covering formulation of the record, and the sub-threshold reduction.
**Every one of these holds with gears 2 and 3 present**, once the single constant `2` in the count
formulas is read as `|{0, -2} mod g|`. The other half is *value*: the arcs `(g - 3, 1)`, the run
ceiling `q' - 3`, the clump width, `prod (g - 4)`, the parity law, the mex closed form, universal
multiplicity, "no fold", the twin-candidate view. Those need a floor on `q'`, and the floors are
`3`, `5`, or a relation to `m`.

**The smallest top machine that retains the simplicity is `q' = 5` for everything structural and
for the conjugacy, `q' = 3` for the symmetry group, and nothing absolute for the record.** Gear 5
is the first gear whose domino is gapped inside its period, and it is also the first gear set for
which `6` is invertible - so `q' = 5` is simultaneously "the domino machine begins" and "this IS
the bottom machine `{5..q}`". Below it, gear 3's piece is solid and crosses parity, and gear 2's
two teeth coincide; those are the two gears that fold the line, and they are the only two.

**What the record theory gives the bottom machine is a rescaling, not a transfer.** The parity law
and the mex closed form both fail at `{5..q}` for `m >= 3`, because the small gears recur inside
the window - and they fail in a way that is completely described: the record of the whole machine
`{2,3,5..q}` is exactly `6 F_col + 5`, and the next open column is exactly the mex of the gears'
two column teeth *with their recurrences included*, certified, at a few dozen numbers per call.
That closed form is exact on 890,501 walks of `{5..q}`, `q <= 31`, and is the bottom machine's own
next-opening formula.

**The one record law that transfers whole is the sub-threshold reduction**: the record depends on
the number of gears and on the gears no larger than the record, and on nothing else - 117 gear
sets, 27 families, 0 disagreements, with small parts as small as `{2, 3, 5}`. That is the law to
carry across the split.

No interpretation against the twin conjecture is offered; no clutch beyond the conjugacy itself.

---

## 7. Scorecard, filled

| # | Prediction | Result |
|---|---|---|
| S1 | the law table | **held with four corrections**: L6's antipode holds at `q' = 3` (the Lean hypothesis `5 <= g` is not sharp); L14's alternation holds at `q' = 2` (predicted 3); L40, L42, L44 hold at `q' = 2` once the per-gear factor is written with the actual tooth set (predicted 3); L38's hop bound holds at every `q'` (predicted relative) |
| S2 | T5: the gapped domino, structural laws verbatim at `q' = 5` | **held**; and `q' = 5` is exactly where the conjugacy exists |
| S3 | T3: solid domino, no adjacent open pairs, fold mod 3, empty triple view, gap 4 still forbidden | **held**, all five, exactly |
| S4 | T2: one tooth, `(Z/2)^{m-1}`, all gaps even, `{2,3}` recovers the six-fold | **held**, all four; group order verified by brute force over all `W^2` affine maps |
| S5 | (a) mex exact iff `q' > 2m`; sharpened `mex < q' ==> exact` | sharpened form **held**, 94,774 positions, 0 exceptions; and the criterion **improved** to `F_top < q'`, 26 of 26, with `{6,11,13}` separating it from `q' > 2m` |
| S6 | (b) parity law exactly while `q' > 2m + 1`, first failure at `q' = 2m + 1` | **refuted as written**: `q' = 2m + 1` fails at odd `m` and **holds** at even `m`. Corrected threshold `q' >= 2m + 1` (even `m`), `q' >= 2m + 3` (odd `m`): 7 boundary pairs, 0 exceptions |
| S7 | (c) domino tiling iff `q' > F + 1` | **refuted as written**; (c) is exactly equivalent to (b), 28 of 28 |
| S8 | (d) symmetry group `(Z/2)^m` iff `q' >= 3`, absolute | **held**, and the corrected order `2^{#odd gears}` verified at 7 wheels |
| S9 | (e) L31 holds at every `q' >= 2` | **held**, 27 families, 117 sets, 0 disagreements |
| S10 | the transfer list | **delivered** (3.4), and sharpened by the rescaling law L56 |
| S11 | the corrected column mex exact on `{5..q}`, `q <= 31` | **held**, 890,501 certified walks, 0 mismatches; `B = 2m` certifies 99.2-99.7%, `B = F` certifies all |

---

## 8. Holds without exception (the count)

| statement | range | exceptions |
|---|---|---|
| L3 the partner law | 23 wheels, every gear, `q'` from 2 | **0** |
| L4 the forbidden gap 4 | 23 wheels + 6 range machines to `10^7`, `q'` from 2 | **0** |
| L7 the mirror | 23 wheels, `q'` from 2 | **0** |
| L12 the chain law | 18 ladder steps, 260,284 opening pairs | **0** |
| L13 the merge law | 18 ladder steps, 36,754 gaps | **0** |
| L14 letters and alternation | 18 ladder steps, 918 struck runs | **0** |
| L16 the covering record = the scanned record | 18 wheels, `q'` from 2 | **0** |
| L22 the gap census law | 23 wheels, every `d <= 12` | **0** |
| L28 exactly two total collisions | 23 wheels | **0** |
| L35 the palindromic gap word | 23 wheels | **0** |
| L36 (document 3) the four `C`-identities | 18 wheels | **0** |
| L40/L41 the spectral product and full support | 17 wheels, 59,141 frequencies (max error 6.7e-16) | **0** |
| L42 the parity bias `prod (g - 2 t_g)` | 18 wheels | **0** |
| L44 the correlation product (general form) | 18 wheels, 720 values | **0** |
| L31 (document 2) the sub-threshold reduction | 27 families, 117 gear sets | **0** |
| L50 the sharp mex criterion | 94,774 positions with `mex < q'`, 18 wheels | **0** |
| L50 `F < q'` predicts 0 failures | 26 gear sets | **0** |
| L51 exactly one odd gap length = the self-paired gap | 15 wheels | **0** |
| L52 the tooth-count counting laws | 23 wheels: 23 wheel counts, 138 run counts, 92 chain counts, 23 domino counts | **0** |
| L53 the chain ceiling `q_odd - 2` | 14 wheels | **0** |
| L54 the group order `2^{#odd gears}` | 7 wheels, all `W^2` affine maps each | **0** |
| L55 the parity threshold | 35 gear sets, 7 boundary pairs, `m = 2..8` | **0** |
| L55 (b) equivalent to (c) | 28 gear sets | **0** |
| L56 the anchor rescaling law | 30 cases (10 gear sets x 3 anchors) | **0** |
| L57 the certified column mex | 890,501 walks, 8 gear sets `{5..q}` | **0** |
| L58 the hop chain `<= 2` | 18 pair layers + 4 column layers | **0** |
| L58 the modified column double-hop rule | 4 layers, 10,860 hits | **0** |
| L19 the conjugacy | 5 wheels | **0** |
| the triple view is empty when 2 or 3 is a gear | 9 wheels | **0** |

---

## 9. Dead ends

* **"The parity law fails first at `q' = 2m + 1`."** Refuted: it fails there only at odd `m`.
  What survived is the exact threshold and its mechanism - the long letter is worth a cell only
  when the remaining parity demand is even, which is the odd-`m` case (L55).
* **"The record is a free-domino tiling iff `q' > F + 1`."** Refuted at `{5,7}`, `{9,11,13,17}`,
  `{13,17,19,23,29,31}` and `{17,19,23,29,31,37,41,43}`, where `q' = F + 1` and the tiling is
  still free. What survived is the stronger statement that (b) and (c) are the same condition.
* **"The mex form needs `q' > 2m`."** Refuted by `{6,11,13}`. What survived is L50, whose
  hypothesis is a property of the record, not of the gear count.
* **The universal record multiplicity (L18) below `q' = 7`.** Not a law there: `{5,7,11,13}` has 4
  record blocks and `{5,11,13,17}` has 2, both `m = 4`. Not pursued; it is document 2's L25 with
  the gear-independence hypothesis broken, exactly as that law says.
* **Document 2's L32 and L33 (the range record as a first hit; the pinning of record blocks), and
  document 1's L21 (the gear zone), at small `q'`.** Not tested in this branch - they need whole-period scans of wheels containing 2
  and 3, which are as large as the large-gear wheels already scanned in that branch and were out
  of budget here. Left open, and noted as the only rows of the law table with no measurement.
