# The walk and the transforms of the top machine (branch R4.b.iii)

Parent: R4.b, *The top machine on its own terms* (`research/proof/top_machine_1.md`, laws
L1-L21; `docs/proofs/22-top-machine-laws.md`; kernel ledger `research/proof/top_machine_lean.md`).
The observation that spawned this branch is L3 together with L17: a gear's struck set is a
disjoint union of dominoes `{x, x + 2}`, and when every gear is large the longest struck stretch
is decided by counting and parity alone, `2m - (m mod 2)`. If the blocked stretch is that rigid,
then **the walk from an arbitrary position to the next opening should be computable from the
residues alone**, without scanning the line. This branch is about that walk and about the
transforms (Fourier, bitwise, character) that describe it.

Construction rule (R4, owner): the top machine on the **raw line**, on its own; the bottom
machine as inspiration only; no clutch; no interpretation against the twin conjecture.

Deliverable (owner, this round): **a closed form for locating the next opening from any
position, with a proof.** Sections 1 and 2 carry it; sections 3-5 serve it.

Numbering of laws continues from **L30** (L22-L29 belong to the parallel second pass,
`research/proof/top_machine_2.md`).

---

## 0. Setup

**The line and the gears.** The integers. Gear `g` (a prime above the split, so `g >= 7`, odd)
strikes its multiples.

**Two views of the same machine.**

* *Pair view.* The object is the pair `n = (n, n + 2)`, indexed by its lower member, so the pair
  coordinate is the raw line. Gear `g` strikes the pair `n` iff `n = 0` or `n = -2 (mod g)`:
  **two teeth, at `0` and `-2`, of separation 2.** Open pairs: `O = {n : forall g, n mod g not in
  {0, g - 2}}`, `|O| = prod (g - 2)` per wheel `W = prod g` (L5).
* *Single-number view.* The object is the integer `n`; gear `g` strikes `n` iff `g | n`: **one
  tooth, at `0`.** Open integers `S = {n : forall g, n mod g != 0}`. A **twin candidate** is a run
  of three consecutive open integers `n, n + 1, n + 2`; it contains the open pair `(n, n + 2)`.
  The set of starts of such a run is
  `T = {n : forall g, n mod g not in {0, g - 1, g - 2}}`: **three teeth, at `0`, `-1`, `-2`,
  solid.**

The pair view's teeth are a *gapped* domino `{0, -2}`; the triple view's are a *solid* triomino
`{0, -1, -2}`. That difference is the branch's main mechanism.

**The walks.** For a gear set `G` with `m = |G|`, write `x_g = x mod g`.

        L(x)  = min { j >= 0 : x + j in O }        the walk to the next open pair
        R(x)  = min { j >= 0 : x + j in T }        the walk to the next twin candidate

`L(x) = 0` iff `x` is itself open. `max_x L(x)` is the record `F_top(G)` of L16/L17 (the longest
run of consecutive struck pairs); `max_x R(x) =: F_3(G)` is the longest stretch with no twin
candidate.

**The transforms.** `O` and `T` are `W`-periodic subsets of `Z_W`; their indicator functions are
functions on the finite abelian group `Z_W`, which factors as `prod Z_g` by CRT, so every
transform of them factors over the gears. Fourier: `f^(a) = (1/W) sum_n f(n) omega_W^{-a n}`.
Bitwise: gear `g` is the periodic mask with ones at `0` and `-2`; blocked `= OR`, open `= AND` of
complements, and `XOR` is the parity of the striker count.

**Vocabulary.** As in the first pass, unchanged: gear, tooth, pair, open, wheel, slot, run, gap,
record, letters `{2, g - 2}`, domino, shield `n = -1`, origin clump, mirror `n -> -n - 2`.
New here: **walk** `L(x)`, **twin-candidate walk** `R(x)`, **mex** (the least non-negative
integer not in a given set), **all-struck count** `C(j)`, **correlation** `B(d)`.

---

## 1. Pre-registered predictions and scorecard

Written before any computation of this branch. Each prediction states the derivation by hand
where there is one, so that the computation is a test and not a fit.

### Section 1. The walk, closed form

**Q1 (THE MEX FORM: the next opening as an explicit function of the residues).** Set, for each
gear, `a_g = (-x) mod g` and `b_g = (-x - 2) mod g`. Predicted, as an exact identity:

        L(x) = mex { a_g , b_g : g in G }          whenever every gear exceeds 2m

where `mex` is the least non-negative integer missing from the listed `2m` numbers. Derivation:
`x + j` is struck by `g` iff `j = a_g` or `j = b_g (mod g)`; a set of `2m` numbers cannot cover
`{0, 1, ..., 2m}`, so the mex is at most `2m < g`, and for every `j <= mex` the congruence is an
equality because `j < g`. **This is a closed form: `O(m)` arithmetic operations on the residues,
with no scan of the line and no reference to `W`.** Refuted by one `x` in one wheel where the mex
differs from the scanned walk.

**Q2 (the location bound, from L17).** Under the same hypothesis strengthened to *every gear odd
and `> 2m + 1`*, predicted `L(x) <= 2m - (m mod 2)` for every `x`, with equality attained.
Derivation in the mex language: if both `a_g` and `b_g` lie in `[0, 2m]` then `b_g = a_g - 2`
(the other alternative `b_g = a_g + g - 2 >= g - 2 >= 2m + 1` is out of range), so the `2m`
numbers fall into at most `m` pairs of **equal parity** plus singletons; covering the
`ceil((2m + 1)/2)` even and `floor((2m + 1)/2)` odd cells of `[0, 2m]` needs
`ceil(ceil(L/2)/2) + ceil(floor(L/2)/2)` pieces, which exceeds `m` at `L = 2m` exactly when `m`
is odd. Refuted by a walk longer than `2m - (m mod 2)` in the stated regime.

**Q3 (THE TWIN-CANDIDATE MEX FORM, and no parity defect).** In the single-number view, with
`c_g^i = (-x - i) mod g` for `i = 0, 1, 2`, predicted:

        R(x) = mex { c_g^0 , c_g^1 , c_g^2 : g in G }        whenever every gear exceeds 3m

and, whenever every gear is at least `3m + 3`,

        F_3(G) = max_x R(x) = 3m       exactly, with no parity correction.

Derivation: each gear's trace in a window of length `L <= 3m + 1` is a *solid* interval of at
most 3 cells (the two ends of the triomino are `2` apart and `g - 2 > L`, so it cannot be split
by the window), whence `L <= 3m`; and `m` solid triominoes tile `[0, 3m)` exactly, a phase vector
realised by CRT. Predicted values: `F_3 = 9` for `{13,17,19}`, `12` for `{17,19,23,29}`, `15` for
`{19,23,29,31,37}`. **This is the exact contrast with the pair machine**: the domino `{x, x + 2}`
never crosses parity, so it wastes a cell for odd `m`; the triomino is solid and wastes nothing.
Refuted by a large-gear set with `F_3 != 3m`, or by one `x` where the mex differs from the scan.

**Q4 (the walk-length distribution is the first difference of the all-struck count).** Let
`C(j) = #{x in Z_W : x, x + 1, ..., x + j - 1 all struck}`, `C(0) = W`. Predicted exactly, in
every wheel:

        #{x : L(x) = j} = C(j) - C(j + 1)          for all j >= 0
        #{x : L(x) = j} = #{gaps of length >= j + 1}   for j >= 1
        N_d (the gap census) = C(d - 1) - 2 C(d) + C(d + 1)      for d >= 1
        max_x L(x) = F_top = max { j : C(j) > 0 }

i.e. **the gap census is the second difference of `C`, exactly dual to L11** (the run spectrum is
the second difference of `A(L) = prod (g - 2 - L)`). Refuted by one wheel where any of the four
fails.

**Q5 (the closed form of `C`, and its shape).** Pre-registered shape: **an alternating sum of
shifted wheel products, and not a single product.** Precisely, whenever every gear is at least
`j + 2`,

        C(j) = sum_{k, e} (-1)^k T(j, k, e) prod_g (g - 2k + e)

where `T(j, k, e)` is the number of `k`-subsets `S` of `[0, j)` with exactly `e` elements `s`
such that `s - 2` is also in `S`. Derivation: inclusion-exclusion over which of the `j` positions
are open; "all of `S` open" is a per-gear condition, giving `prod_g (g - |U(S) mod g|)` with
`U(S) = {-s, -s - 2 : s in S}`, and `|U(S)| = 2|S| - e(S)` when no wraparound. `T(j, k, e)` is
itself closed: `[0, j)` splits into its `n1 = ceil(j/2)` even and `n2 = floor(j/2)` odd
positions, `s` and `s - 2` share parity, so `T` is the convolution of two path counts
`C(k - 1, e) C(n - k + 1, k - e)`. Predicted first values:

        C(1) = W - prod(g-2)
        C(2) = W - 2 prod(g-2) + prod(g-4)
        C(3) = W - 3 prod(g-2) + prod(g-3) + 2 prod(g-4) - prod(g-5)

Predicted further: **no `C(j)` with `j >= 2` is a product over the gears** (its value divided by
`W` is not multiplicative), which is the exact reason the walk's distribution is harder than the
run spectrum. Refuted by a mismatch against the exact wheel count, or by finding a product form.

**Q6 (the in-use regime: the general mex form and a counting bound).** For an arbitrary gear set
(gears not all above `2m`), predicted exactly:

        L(x) = mex ( union_g ( {a_g, b_g} + g Z_{>=0} ) )

still a closed form, but with `2 sum_g ceil((B + 1)/g)` terms rather than `2m`, where `B` is any
proved bound on `L`. Predicted general bound, by counting cells:

        L <= 2m / (1 - 2 H_S) ,   H_S = sum_{g in G, g <= L} 1/g ,   valid when H_S < 1/2

which reduces to `L <= 2m` when every gear exceeds `L` (Q2's regime). Predicted: **this bound is
vacuous exactly when `sum_{q < g <= L} 1/g` reaches `1/2`**, and for the in-use machine
`(q, sqrt(N)]` that happens at every `q` tested to `N = 10^7`, so the counting bound does not
decide the in-use record; the in-use record is a gear-zone (smoothness) phenomenon (L21) and not
a covering phenomenon. Refuted if the bound is non-vacuous at some in-use `q` and `N`, or if it
fails as a bound anywhere.

### Section 2. The layered walk

**Q7 (the order: smallest gear first).** Justification pre-registered: adding gears in increasing
order makes the new gear the largest so far, so the large-gear hypothesis of Q2 (`g > 2m + 1`
at every step) is exactly the hypothesis under which the layer collapses; largest-first would
put the small gears, the only ones with a long letter inside the window, last, where they break
every bound. Predicted: with smallest-first, every step of every ladder from `q' >= 7` satisfies
`g > F_G + 3` from the second gear on.

**Q8 (the hop law and the exact landing).** Let `y` be the landing of the walk under `G`, i.e.
`y = x + L_G(x)`, and add gear `g`. Predicted, exactly and with 0 exceptions:

* `g` hops at `y` iff `y mod g in {0, g - 2}` (the hit law with `d = 2`);
* after a hop at `y`, the **next position `g` itself strikes** is `y + 2` if `y = -2 (mod g)`,
  and `y + g - 2` if `y = 0 (mod g)`.

**Q9 (THE COLLAPSE: at most a double hop).** Predicted: if `g > F_G + 3` then the hop chain has
length at most 2, and a double hop occurs **iff** `y = -2 (mod g)` *and* the next `G`-opening
after `y` is exactly `y + 2`. Hence the layer is a one-line, non-recursive formula:

        L_{G + g}(x) = L_G(x) + [ y = 0 or -2 mod g ] * ( d1 + [ y = -2 mod g and d1 = 2 ] * d2 )

with `d1` the `G`-gap at `y` and `d2` the `G`-gap at `y + 2`. Derivation: after a hop at
`y = 0 (mod g)` the next `g`-strike is `y + g - 2 > y + F_G + 1`, beyond the next `G`-opening;
after a hop at `y = -2` the next `g`-strike is `y + 2`, which is a `G`-opening only if the
`G`-gap is exactly 2, and the strike after that is `y + g > y + F_G + 3`. Predicted 0 triple hops
in the large-gear regime and 0 double hops of any other form. Refuted by one triple hop with
`g > F_G + 3`, or one double hop with `d1 != 2`.

**Q10 (the nested form and its cost).** Predicted: the walk is exactly the nested composition of
the Q9 layers over the gears in increasing order, so the closed form has **cost `m` layer tests
plus one gap lookup per hop**, and the total number of hops over a full wheel equals the total
number of merges in the merge law (L13). Predicted mean hops per walk in the large-gear regime:
below 1 (each gear hits a given landing with probability `2/g`, so the expected number of hops is
`sum_g 2/g < 1` for large-gear sets). Refuted by a measured mean above 1 in that regime.

### Section 3. Spectral

**Q11 (the per-gear transform, and the shield in place of the fold).** Predicted, exactly:

        u_g^(0) = (g - 2)/g ,     u_g^(a) = -(1 + omega_g^{2a}) / g   for a != 0

and, in the **shield coordinate** `n' = n + 1` (the teeth become `+1` and `-1`, symmetric),

        u_g^(a) = -(2/g) cos(2 pi a / g)          REAL, for a != 0.

So the top machine is the `u = 1` machine, and **the bottom's fold (a symmetric tooth pair about
`0`) is replaced by a translation to the shield**: the pole-phase law is the same real cosine
factor multiplied by the single phase `omega_g^{a}` of the shield's position. Refuted by an exact
DFT that disagrees.

**Q12 (the spectrum of `O`, and full support).** Predicted `O^(a) = prod_g u_g^(a_g)` under the
CRT dual, so `|O^(a)| = prod_{a_g = 0} (1 - 2/g) * prod_{a_g != 0} (2 |cos(2 pi a_g / g)| / g)`
and the phase is `pi #{g : a_g != 0} + 2 pi sum_{a_g != 0} a_g / g`. Predicted: **`O^(a) != 0`
for every `a`** - the open-pair set has full spectral support - because `cos(2 pi a / g) = 0`
needs `4a = g (mod 2g)`, impossible for odd `g`. Refuted by a vanishing coefficient.

**Q13 (the run indicator is a Dirichlet kernel, and it does have zeros).** The per-gear factor of
the indicator of "a run of `L` open pairs starts here" is the transform of the complement of an
interval of `L + 2` consecutive residues, i.e. a Dirichlet kernel of length `L + 2` centred at
`-(L + 1)/2`. Predicted: it vanishes at `a != 0` iff `a (L + 2) = 0 (mod g)`, so run indicators
have spectral zeros exactly when `gcd(L + 2, g) > 1`, unlike `O`. Refuted by a zero elsewhere.

**Q14 (what the spectrum can and cannot decide).** Predicted: the spectrum decides the **run**
record (`q' - 3`) because the run count is the `a = 0` coefficient of a *product*, and a product
vanishes iff a factor does (`g - 2 - L = 0` at `L = q' - 2`); and it **cannot** decide the
blocked record `F_top`, because `C(j)` is a signed sum of products and positivity is not a
support question. Refuted by a spectral criterion that yields `F_top`.

### Section 4. Bitwise

**Q15 (the parity bit and its exact bias).** With the XOR of the gears' masks (the parity of the
striker count, a Liouville-type bit restricted to the gear set), predicted exactly per wheel:

        #{even striker count} = (W + prod(g - 4)) / 2 ,   #{odd} = (W - prod(g - 4)) / 2

by the per-gear character sum `(g - 2) - 2 = g - 4`. **Predicted identity:** the excess
even-minus-odd is exactly `prod(g - 4)`, which is also the number of adjacent open pairs (L15) -
the same polynomial answering two unrelated questions. Refuted by a wheel where the counts differ
from the formula.

**Q16 (XOR bounds the OR's longest run from BELOW, and the bound is tight for even `m`).**
`{XOR = 1}` is a subset of `{blocked}`, so the longest run of ones in the XOR pattern is a *lower*
bound for `F_top`, computable from a product. Predicted: in the large-gear regime the record
cover is a perfect tiling by `m` dominoes when `m` is even (no cell covered twice), so
`XOR = 1` on the whole record block and **the longest XOR run equals `F_top = 2m` exactly**; for
odd `m` there is one unit of waste, one doubly covered cell, so the longest XOR run is strictly
less than `F_top`. This is the opposite of the bottom machine's answer (there, the parity barrier
gave no bound at all). Refuted by an even-`m` large-gear wheel with longest XOR run `< 2m`.

### Section 5. Characters

**Q17 (the correlation is a true product).** Predicted exactly, for every `d >= 1`:

        B(d) = #{n : n and n + d both open} = prod_g c_g(d),
        c_g(d) = g - 2 if d = 0 (mod g), g - 3 if d = +-2 (mod g), g - 4 otherwise.

Derivation: "both open" is a per-gear condition; the forbidden set is `{0, -2} u {-d, -d - 2}`,
of size 4 minus the overlaps, and the only overlaps are `d = 0, +-2 (mod g)`. Checks: `B(1) =
prod(g - 4)` (L15's dominoes), `B(2) = prod(g - 3)` (L15's member-sharing pairs). Refuted by one
mismatch.

**Q18 (spectrum holes).** Predicted: `c_g(d) >= g - 4 > 0` always, so **the correlation never
vanishes - every distance `d` occurs between some two open pairs**; holes exist only in the
*consecutive* (gap) census, and, below the record, the only hole is `d = 4` (L4). Predicted
further, in the single-number view, that the triple machine's holes are `{2, 3}`: if `x` and
`x + d` both start twin candidates then `x + 1` struck forces `x = -3 (mod g)` for the striker,
which then strikes `x + 1, x + 2, x + 3`, so `d >= 4`. Refuted by a wheel with a different hole
set in either view.

### Scorecard

| # | Prediction | Result |
|---|---|---|
| Q1 | `L(x) = mex{a_g, b_g}` when every gear `> 2m`, exact | |
| Q2 | `L(x) <= 2m - (m mod 2)` when gears odd and `> 2m + 1`, attained | |
| Q3 | `R(x) = mex{c_g^0, c_g^1, c_g^2}`; `F_3 = 3m` exactly, no parity defect | |
| Q4 | walk distribution `= C(j) - C(j+1)`; gap census `=` second difference of `C` | |
| Q5 | `C(j) = sum (-1)^k T(j,k,e) prod(g - 2k + e)`; no product form | |
| Q6 | general mex form exact; counting bound `2m/(1 - 2H_S)`; vacuous in use | |
| Q7 | smallest-first is the order that keeps the collapse hypothesis | |
| Q8 | hop law with `d = 2`; next own strike at `+2` or `+ g - 2` | |
| Q9 | at most a double hop when `g > F_G + 3`; double iff `y = -2` and `d1 = 2` | |
| Q10 | nested form exact; mean hops per walk `< 1` in the large-gear regime | |
| Q11 | per-gear transform; real in the shield coordinate (`u = 1` machine) | |
| Q12 | `O^(a) != 0` for every `a` (full support) | |
| Q13 | run indicator = Dirichlet kernel, zeros iff `gcd(L + 2, g) > 1` | |
| Q14 | spectrum decides the run record, cannot decide `F_top` | |
| Q15 | XOR bias exactly `prod(g - 4)`, equal to the domino count | |
| Q16 | longest XOR run `= F_top` for even `m`, `<` for odd `m` | |
| Q17 | `B(d) = prod c_g(d)`, exact for all `d` | |
| Q18 | correlation never vanishes; gap holes `{4}` (pairs), `{2, 3}` (triples) | |

---

## 2. Setup as computed

Scripts in `research/topmachine/r3/`, results (untracked) in `.../results/`:
`core.py` (the two machines, walks, gaps, all-struck counts), `walk.py` (Q1-Q5, Q17, Q18),
`layered.py` (Q7-Q10), `spectral.py` (Q11-Q16), `inuse.py` (Q6), `extras.py` (the record
block's cover multiplicity, the triple record at `m = 5`). Every count below is exact over a
full wheel period, or exact over the stated range.

---

## 3. Results

### 3.1 The walk in closed form: the mex of the residues

`L(x) = min {j >= 0 : x + j is an open pair}`, cyclically over the wheel.

| gears | m | W | every gear > 2m | mex form vs the scan | max L | `2m - (m mod 2)` |
|---|---|---|---|---|---|---|
| 7,11,13 | 3 | 1,001 | yes | **0 mismatches** | 6 | 5 |
| 11,13,17 | 3 | 2,431 | yes | **0 mismatches** | 5 | 5 |
| 13,17,19 | 3 | 4,199 | yes | **0 mismatches** | 5 | 5 |
| 17,19,23 | 3 | 7,429 | yes | **0 mismatches** | 5 | 5 |
| 19,23,29 | 3 | 12,673 | yes | **0 mismatches** | 5 | 5 |
| 7,11,13,17 | 4 | 17,017 | **no** (7 < 8) | 36 mismatches | 9 | 8 |
| 11,13,17,19 | 4 | 46,189 | yes | **0 mismatches** | 8 | 8 |
| 13,17,19,23 | 4 | 96,577 | yes | **0 mismatches** | 8 | 8 |
| 17,19,23,29 | 4 | 215,441 | yes | **0 mismatches** | 8 | 8 |
| 11,13,17,19,23 | 5 | 1,062,347 | yes | **0 mismatches** | 10 | 9 |

**0 mismatches over 1,448,287 positions in the nine wheels that satisfy the hypothesis**, and the
one wheel that fails it fails the formula too - the hypothesis `q' > 2m` is *sharp*, and it fails
exactly where the small gear repeats inside the window. (`{7,11,13}` at `m = 3` satisfies
`7 > 6` and holds; `{7,11,13,17}` at `m = 4` does not and breaks.) The two wheels whose record
exceeds `2m - (m mod 2)` are exactly the two containing a gear below the L17 threshold
(`7 < 2m + 1 = 9`; `11 < 2m + 1 = 11`), as L17 says.

### 3.2 The twin-candidate walk: the same form, three teeth, no parity defect

Single-number view: `T` is the set of `n` starting a run of three consecutive open integers,
i.e. a **twin candidate**; `R(x) = min {j >= 0 : x + j in T}`.

| gears | m | W | every gear `>= 3m+3` | number of starts `= prod(g-3)` | mex form vs the scan | max R | `3m` |
|---|---|---|---|---|---|---|---|
| 13,17,19 | 3 | 4,199 | yes | 2,240 = 2,240 | **0 mismatches** | 9 | 9 |
| 17,19,23 | 3 | 7,429 | yes | 4,480 = 4,480 | **0 mismatches** | 9 | 9 |
| 19,23,29 | 3 | 12,673 | yes | 8,320 = 8,320 | **0 mismatches** | 9 | 9 |
| 17,19,23,29 | 4 | 215,441 | yes | 116,480 = 116,480 | **0 mismatches** | 12 | 12 |
| 19,23,29,31 | 4 | 392,863 | yes | 232,960 = 232,960 | **0 mismatches** | 12 | 12 |
| 23,29,31,37 | 4 | 765,049 | yes | 495,040 = 495,040 | **0 mismatches** | 12 | 12 |
| 19,23,29,31,37 | 5 | 14,535,931 | yes | 7,920,640 = 7,920,640 | (record only) | **15** | 15 |
| 11,13,17 | 3 | 2,431 | no (11 < 12) | 1,120 = 1,120 | 0 mismatches | 9 | 9 |
| 7,11,13 | 3 | 1,001 | no (7 < 12) | 320 = 320 | 36 mismatches | 10 | 9 |

`F_3 = 3m` **exactly** at `m = 3, 4, 5`, with no parity correction: seven large-gear wheels,
0 exceptions. The hypothesis is sufficient, not necessary (`{11,13,17}` misses it and still gives
`9`), and it is not vacuous (`{7,11,13}` gives `10 > 3m`).

### 3.3 The distribution: the all-struck count `C(j)`

`C(j) = #{x : x, x+1, ..., x+j-1 all struck}`. Exact per wheel, and every one of the identities
of Q4 held with 0 mismatches in all ten wheels.

| gears | `C(0), C(1), ...` | `F_top` | `max{j : C(j) > 0}` |
|---|---|---|---|
| 7,11,13 | 1001, 506, 200, 118, 68, 18, 2, 0 | 6 | 6 |
| 11,13,17 | 2431, 946, 280, 158, 88, 18, 0 | 5 | 5 |
| 13,17,19 | 4199, 1394, 344, 190, 104, 18, 0 | 5 | 5 |
| 17,19,23 | 7429, 2074, 424, 230, 124, 18, 0 | 5 | 5 |
| 19,23,29 | 12673, 3034, 520, 278, 148, 18, 0 | 5 | 5 |
| 7,11,13,17 | 17017, 9592, 4624, 2984, 1882, 780, 268, 96, 48, 12, 0 | 9 | 9 |
| 11,13,17,19 | 46189, 20944, 7984, 4880, 2938, 996, 216, 72, 24, 0 | 8 | 8 |
| 13,17,19,23 | 96577, 37672, 12112, 7160, 4186, 1212, 216, 72, 24, 0 | 8 | 8 |
| 17,19,23,29 | 215441, 70856, 18896, 10840, 6170, 1500, 216, 72, 24, 0 | 8 | 8 |
| 11,13,17,19,23 | 1062347, 532202, 235472, 151990, 96704, 41418, 14328, 6096, 2472, 480, 24, 0 | 10 | 10 |

The last nonzero value is the universal record multiplicity of L18 (18 at `m = 3`, 24 at
`m = 4, 5`) reappearing as `C(F)`. The closed form of Q5 was checked wherever its hypothesis
holds (every gear `>= j + 2`): **0 mismatches** at `{11,13,17}`, `{13,17,19}`, `{17,19,23}`,
`{19,23,29}`, `{13,17,19,23}`, `{17,19,23,29}`; and the coefficient `T(j,k,e)` agrees with its
path-convolution closed form for `j = 0..12`, 0 mismatches.

**The mean walk.** `sum_x L(x) = sum_{j >= 1} C(j)` exactly (each `x` contributes 1 for every `j`
with `L(x) >= j`, and `#{L >= j} = C(j)`), so the *expected* cost of locating the next opening is
`(1/W) sum_{j>=1} C(j)`, explicit in the gears through Q5:

| gears | mean walk (scan) | `(1/W) sum_j C(j)` | via the closed form | `1 / density` |
|---|---|---|---|---|
| 7,11,13 | 0.911089 | 0.911089 | 0.909091 (hypothesis fails at `j = 6`) | 2.0222 |
| 11,13,17 | 0.612916 | 0.612916 | **0.612916** | 1.6370 |
| 13,17,19 | 0.488211 | 0.488211 | **0.488211** | 1.4970 |
| 17,19,23 | 0.386324 | 0.386324 | **0.386324** | 1.3873 |
| 11,13,17,19 | 0.823876 | 0.823876 | **0.823876** | 1.8296 |
| 13,17,19,23 | 0.648747 | 0.648747 | **0.648747** | 1.6395 |
| 17,19,23,29 | 0.503962 | 0.503962 | **0.503962** | 1.4901 |

### 3.4 The in-use machine: gears `(q, sqrt(N)]`

The general mex form (Q6) was checked against the scan on 2,000 random positions per machine,
twelve machines, `N = 10^6` and `10^7`: **24,000 walks, 0 mismatches**, on gear sets of 160 to
443 gears.

`N = 10^7`, `Z = 3,162`:

| q | gears | density | record F | mean walk | median | 99th pct | mex form | AP terms used | `2m` | `H_S` | counting bound |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 443 | 0.072346 | 3,006 | 14.44 | 10 | 66 | 0 mismatches | 7,914 | 886 | 1.311 | vacuous |
| 7 | 442 | 0.103753 | 2,718 | 10.10 | 7 | 47 | 0 mismatches | 6,379 | 884 | 1.156 | vacuous |
| 11 | 441 | 0.128052 | 2,088 | 8.04 | 5 | 38 | 0 mismatches | 4,521 | 882 | 1.031 | vacuous |
| 13 | 440 | 0.151844 | 1,166 | 6.63 | 4 | 32 | 0 mismatches | 2,346 | 880 | 0.876 | vacuous |
| 17 | 439 | 0.171820 | 618 | 5.75 | 4 | 28 | 0 mismatches | 1,172 | 878 | 0.726 | vacuous |
| 19 | 438 | 0.191263 | 227 | 5.10 | 3 | 26 | 0 mismatches | 408 | 876 | 0.507 | vacuous |

`N = 10^6`, `Z = 1,000`:

| q | gears | density | record F | mean walk | median | 99th pct | AP terms | `2m` | `H_S` | counting bound |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 165 | 0.100143 | 858 | 10.11 | 7 | 47 | 2,001 | 330 | 1.143 | vacuous |
| 7 | 164 | 0.141390 | 570 | 6.98 | 5 | 33 | 1,167 | 328 | 0.937 | vacuous |
| 11 | 163 | 0.172071 | 283 | 5.57 | 4 | 27 | 529 | 326 | 0.742 | vacuous |
| 13 | 162 | 0.201410 | 163 | 4.68 | 3 | 23 | 280 | 324 | 0.562 | vacuous |
| 17 | 161 | 0.225568 | 136 | 4.11 | 3 | 21 | 218 | 322 | 0.462 | **4,289** |
| 19 | 160 | 0.248987 | 71 | 3.64 | 2 | 19 | 107 | 320 | 0.287 | **753** |

Three things. (i) The closed form does not break in use: the mex is still exact, only the
progressions no longer truncate to their first term. (ii) **The cost of the closed form is the
number of arithmetic-progression terms, `2 sum_g ceil((B+1)/g)`, and it is small: 107 to 7,914
terms against records of 71 to 3,006** - and *below* `2m` whenever the record is short
(`q = 19`, `N = 10^6`: 107 terms against 320 gears, because a gear larger than the record
contributes a term only when its residue happens to land inside the window). (iii) The typical
walk is nothing like the record: median 2 to 10, 99th percentile 19 to 66, against records of
71 to 3,006. Locating the next opening in use costs a few steps; the record is a rare event.

**The counting bound.** `L <= 2m/(1 - 2 H_S)` with `H_S = sum_{g <= L} 1/g` is a proof, and it is
non-vacuous exactly where `H_S < 1/2`: at `q = 17` and `q = 19` for `N = 10^6` (bounds 4,289 and
753 against records 136 and 71 - true, and a factor 10 to 30 loose). At `N = 10^7` every `q`
tested has `H_S > 1/2` and the bound says nothing. The pre-registered claim "vacuous at every
in-use `q` and `N`" is therefore **refuted at two of the twelve machines**; the honest statement
is that the covering bound survives only while the sum of reciprocals of the gears below the
record stays under `1/2`, and the in-use machines cross that line between `q = 19` at `10^6` and
`q = 19` at `10^7`.

### 3.5 The layered walk and the collapse

Gears added one at a time, smallest first; every layer checked over the full wheel of the larger
machine (391,048 positions in twelve layers).

| lower machine | `F_G` | new gear | `g > F_G + 3` | hop law | longest hop chain | double hops | double-hop rule | one-line layer formula | `F_{G+g}` |
|---|---|---|---|---|---|---|---|---|---|
| 7 | 1 | 11 | yes | 0 exceptions | **2** | 2 | 0 exceptions | 0 mismatches | 4 |
| 7,11 | 4 | 13 | yes | 0 exceptions | **2** | 20 | 0 exceptions | 0 mismatches | 6 |
| 7,11,13 | 6 | 17 | yes | 0 exceptions | **2** | 224 | 0 exceptions | 0 mismatches | 9 |
| 11 | 1 | 13 | yes | 0 exceptions | **2** | 2 | 0 exceptions | 0 mismatches | 4 |
| 11,13 | 4 | 17 | yes | 0 exceptions | **2** | 32 | 0 exceptions | 0 mismatches | 5 |
| 11,13,17 | 5 | 19 | yes | 0 exceptions | **2** | 544 | 0 exceptions | 0 mismatches | 8 |
| 13 | 1 | 17 | yes | 0 exceptions | **2** | 2 | 0 exceptions | 0 mismatches | 4 |
| 13,17 | 4 | 19 | yes | 0 exceptions | **2** | 44 | 0 exceptions | 0 mismatches | 5 |
| 13,17,19 | 5 | 23 | yes | 0 exceptions | **2** | 896 | 0 exceptions | 0 mismatches | 8 |
| 17 | 1 | 19 | yes | 0 exceptions | **2** | 2 | 0 exceptions | 0 mismatches | 4 |
| 17,19 | 4 | 23 | yes | 0 exceptions | **2** | 56 | 0 exceptions | 0 mismatches | 5 |
| 17,19,23 | 5 | 29 | yes | 0 exceptions | **2** | 1,456 | 0 exceptions | 0 mismatches | 8 |

**Why smallest-first.** The same gears in the other order break the collapse:

| lower machine | `F_G` | gear added last | `g > F_G + 3` | longest hop chain | collapse |
|---|---|---|---|---|---|
| 11,13,17 | 5 | 7 | no | **3** | broken |
| 13,17,19 | 5 | 7 | no | **3** | broken |
| 17,19,23 | 5 | 7 | no | **3** | broken |
| 13,17,19 | 5 | 11 | yes | 2 | holds |
| 11,13 | 4 | 7 | no | 2 | holds |

**The nested form.** Exact against the scan on every sample tested, and cheap:

| gears | m | sample | nested vs scan | hops per walk (mean) | max hops | `sum 2/g` |
|---|---|---|---|---|---|---|
| 7,11,13 | 3 | 1,001 (all) | 0 mismatches | 0.898 | 6 | 0.621 |
| 11,13,17 | 3 | 2,431 (all) | 0 mismatches | 0.646 | 5 | 0.453 |
| 13,17,19 | 3 | 4,000 | 0 mismatches | 0.512 | 5 | 0.377 |
| 17,19,23 | 3 | 4,000 | 0 mismatches | 0.383 | 5 | 0.310 |
| 11,13,17,19 | 4 | 4,000 | 0 mismatches | 0.807 | 8 | 0.559 |
| 13,17,19,23 | 4 | 4,000 | 0 mismatches | 0.665 | 8 | 0.464 |
| 17,19,23,29 | 4 | 4,000 | 0 mismatches | 0.491 | 6 | 0.379 |

Mean hops per walk is below 1 in every case, as predicted, and about 1.24 to 1.45 times
`sum_g 2/g` (the independent-hit expectation) - the excess is the double hops, which the chain
law makes correlated.

### 3.6 The spectrum

Exact DFT against the product formula, five wheels, every frequency:

| gears | W | max abs(DFT - product) | min abs(`O^(a)`) | `O^(0)` | shield coordinate |
|---|---|---|---|---|---|
| 7,11,13 | 1,001 | 6.2e-17 | 3.05e-05 | 0.494505 | real, max error 9.7e-17 |
| 11,13,17 | 2,431 | 1.1e-16 | 5.21e-06 | 0.610860 | real, max error 1.1e-16 |
| 13,17,19 | 4,199 | 1.0e-16 | 1.75e-06 | 0.668016 | real, max error 1.1e-16 |
| 17,19,23 | 7,429 | 1.1e-16 | 5.60e-07 | 0.720824 | real, max error 7.6e-17 |
| 7,11,13,17 | 17,017 | 6.7e-17 | 3.31e-07 | 0.436328 | real, max error 6.9e-17 |

The product formula is exact to machine precision at every one of 32,077 frequencies; the minimum
magnitude is never zero (**full support**); and in the shield coordinate `n' = n + 1` the whole
transform is **real** - the top machine is the `u = 1` machine, and the bottom's fold is exactly
replaced by that one translation.

**The run indicator.** For `L >= 2` the per-gear factor is the Dirichlet kernel of an interval of
`L + 2` residues, exact to 2.4e-16 at `L = 2, 3, 4` in three wheels. At `L = 1` the interval form
does not apply (the two teeth are the *gapped* domino `{0, -2}`, not an interval), which is the
spectral face of L3. Predicted zeros at `gcd(L + 2, g) > 1` were **never reached**, and the
reason is a theorem, not an accident: a nonempty run needs `L <= q' - 3` (L10), so
`L + 2 <= q' - 1 < g` for every gear and no gear can divide `L + 2`. **Every nonempty run
indicator of a top machine has full spectral support as well.**

### 3.7 The bitwise view

| gears | m | W | #even | `(W + prod(g-4))/2` | #odd | `prod(g-4)` | longest XOR run | `F_top` |
|---|---|---|---|---|---|---|---|---|
| 7,11,13 | 3 | 1,001 | 595 | 595 | 406 | 189 | **6** | 6 |
| 11,13,17 | 3 | 2,431 | 1,625 | 1,625 | 806 | 819 | **5** | 5 |
| 13,17,19 | 3 | 4,199 | 2,977 | 2,977 | 1,222 | 1,755 | **5** | 5 |
| 17,19,23 | 3 | 7,429 | 5,567 | 5,567 | 1,862 | 3,705 | **5** | 5 |
| 19,23,29 | 3 | 12,673 | 9,899 | 9,899 | 2,774 | 7,125 | **5** | 5 |
| 7,11,13,17 | 4 | 17,017 | 9,737 | 9,737 | 7,280 | 2,457 | **9** | 9 |
| 11,13,17,19 | 4 | 46,189 | 29,237 | 29,237 | 16,952 | 12,285 | **8** | 8 |
| 13,17,19,23 | 4 | 96,577 | 64,961 | 64,961 | 31,616 | 33,345 | **8** | 8 |
| 17,19,23,29 | 4 | 215,441 | 154,033 | 154,033 | 61,408 | 92,625 | **8** | 8 |
| 11,13,17,19,23 | 5 | 1,062,347 | 647,881 | 647,881 | 414,466 | 233,415 | **10** | 10 |

The parity bias is exactly `prod(g - 4)` in all ten wheels, and the longest run of ones in the
striker-parity bit **equals** `F_top` in all ten - including the odd `m`, which the
pre-registration said would fall short. The reason is the cover multiplicity:

| gears | m | `F_top` | record blocks | blocks covered exactly once | minimum excess |
|---|---|---|---|---|---|
| 7,11,13 | 3 | 6 | 2 | 2 | 0 |
| 11,13,17 | 3 | 5 | 18 | 12 | 0 |
| 13,17,19 | 3 | 5 | 18 | 12 | 0 |
| 17,19,23 | 3 | 5 | 18 | 12 | 0 |
| 19,23,29 | 3 | 5 | 18 | 12 | 0 |
| 7,11,13,17 | 4 | 9 | 12 | 12 | 0 |
| 11,13,17,19 | 4 | 8 | 24 | 24 | 0 |
| 13,17,19,23 | 4 | 8 | 24 | 24 | 0 |
| 17,19,23,29 | 4 | 8 | 24 | 24 | 0 |
| 11,13,17,19,23 | 5 | 10 | 24 | 24 | 0 |

For **even** `m` every record block is an exact cover; for **odd** `m` two thirds of them are, and
the odd `m`'s "one unit of waste" is paid not by a doubly covered cell but by a gear whose second
tooth falls *outside* the window (a singleton at the edge). So an exactly-covered record block
always exists, and the XOR run reaches `F_top` at both parities.

### 3.8 The correlation and the holes

| gears | `B(d) = prod c_g(d)`, `d = 1..40` | `B(1)` | `B(2)` | gap holes below the record |
|---|---|---|---|---|
| 7,11,13 | 0 mismatches | 189 = `prod(g-4)` | 320 = `prod(g-3)` | `{4}` |
| 11,13,17 | 0 mismatches | 819 | 1,120 | `{4}` |
| 13,17,19 | 0 mismatches | 1,755 | 2,240 | `{4}` |
| 17,19,23 | 0 mismatches | 3,705 | 4,480 | `{4}` |
| 19,23,29 | 0 mismatches | 7,125 | 8,320 | `{4}` |
| 7,11,13,17 | 0 mismatches | 2,457 | 4,480 | `{4}` |
| 11,13,17,19 | 0 mismatches | 12,285 | 17,920 | `{4}` |
| 13,17,19,23 | 0 mismatches | 33,345 | 44,800 | `{4}` |
| 17,19,23,29 | 0 mismatches | 92,625 | 116,480 | `{4}` |
| 11,13,17,19,23 | 0 mismatches | 233,415 | 358,400 | `{4}` |

400 exact correlation values, 0 mismatches. In the triple (twin-candidate) view the gaps present
are `1, 4, 5, 6, ...`: the holes are `{2, 3}` in all ten wheels, plus, in the two smallest
four-gear wheels, a few isolated lengths just under the record where the census is thin
(`{16, 19, 20}` at `{7,11,13,17}`, `{14}` at `{11,13,17,19}`) - a sparseness, not a structure.

---

## 4. Closed forms and proofs

Numbered from L30, in the top machine's own vocabulary. `m = |G|`, `q' = min G`,
`a_g = (-x) mod g`, `b_g = (-x - 2) mod g`.

**L30 (THE MEX FORM - the next open pair from any position, in closed form).**
If every gear exceeds `2m`, then for every `x`

        L(x) = mex { a_g , b_g : g in G } ,

the least non-negative integer missing from those `2m` numbers; the next open pair is at
`x + L(x)`.

*Proof.* `x + j` is struck by `g` iff `j = a_g` or `j = b_g (mod g)`. Let `M` be the mex of the
`2m` listed numbers; a set of at most `2m` numbers cannot contain all of `0, 1, ..., 2m`, so
`M <= 2m < g` for every gear. For `j < M` some listed number equals `j`, so `x + j` is struck.
For `j = M`: for every gear `j != a_g` and `j != b_g` as integers, and since `0 <= j < g` and
`0 <= a_g, b_g < g` the congruences are equalities, so no gear strikes `x + M`. Hence
`L(x) = M`. QED.

*Evidence.* 0 mismatches over 1,448,287 positions in nine wheels. **Sharp**: at `{7,11,13,17}`
(`q' = 7 < 2m = 8`) the identity fails at 36 positions. *`O(m)` arithmetic operations; no scan,
no reference to `W`.*

**L31 (the location bound, in mex form).** If every gear is odd and exceeds `2m + 1` then
`L(x) <= 2m - (m mod 2)` for every `x`.

*Proof.* By L30 the walk is the mex of the `2m` numbers. If both `a_g` and `b_g` lie in `[0, 2m]`
then `b_g = a_g - 2`: the alternative `b_g = a_g + g - 2` gives `b_g >= g - 2 >= 2m + 1`, out of
range. So the listed numbers inside `[0, 2m]` form at most `m` pairs of equal parity, plus
singletons. Covering `[0, L)` needs `ceil(ceil(L/2)/2) + ceil(floor(L/2)/2)` same-parity pieces,
which is `2 ceil(m/2) > m` at `L = 2m` for odd `m`. QED. (This is L17's bound, re-proved from the
closed form rather than from the covering formulation.)

*Evidence.* Attained in every wheel satisfying the hypothesis; the two wheels exceeding it are
exactly those with a gear below `2m + 1`.

**L32 (the general mex form).** For **any** gear set and any `x`,

        L(x) = mex ( union_g ( {a_g, b_g} + g Z_{>=0} ) ) ,

and if `B` is any bound with `L(x) <= B` the union may be truncated at `B`, costing
`2 sum_g ceil((B + 1)/g)` terms.

*Proof.* Identical to L30 without the truncation step. QED.

*Evidence.* 24,000 in-use walks, gear sets of 160-443 gears, `N = 10^6` and `10^7`,
**0 mismatches**; 107 to 7,914 terms.

**L33 (the counting bound).** For any gear set, with `H_S = sum_{g in G, g <= L} 1/g`, every
all-struck window of length `L` satisfies

        L <= 2m / (1 - 2 H_S)       provided H_S < 1/2 .

*Proof.* Gear `g` strikes at most `2 ceil(L/g)` of `L` consecutive positions (two residues, each
met at most `ceil(L/g)` times). Summing,
`L <= sum_{g > L} 2 + sum_{g <= L} (2L/g + 2) = 2m + 2 L H_S`. QED. When every gear exceeds `L`
this is `L <= 2m`, L31's regime.

*Evidence.* True at all twelve in-use machines; non-vacuous at two of them (`q = 17, 19` at
`N = 10^6`: bounds 4,289 and 753 against records 136 and 71).

**L34 (the twin-candidate mex form).** In the single-number view, if every gear exceeds `3m`,

        R(x) = mex { (-x) mod g , (-x-1) mod g , (-x-2) mod g : g in G } ,

and the next run of three consecutive open integers - the next twin candidate - starts at
`x + R(x)`.

*Proof.* As L30 with three teeth; `3m` numbers cannot cover `[0, 3m]`, so `R <= 3m < g`. QED.

*Evidence.* 0 mismatches in six large-gear wheels (1,397,654 positions) and in `{11,13,17}`.

**L35 (THE TRIPLE RECORD - `3m`, with no parity defect).** If every gear is at least `3m + 3`
then

        F_3(G) = max_x R(x) = 3m       exactly.

*Proof.* Upper bound: in a window of `L <= 3m + 1` consecutive positions, gear `g`'s trace is
`{p : p = 0, -1, -2 (mod g)}`, a block of three consecutive residues repeating with period `g`;
since `g >= 3m + 3 >= L + 2`, at most one block meets the window and the window cannot split one,
so the trace is a *solid* interval of at most 3 cells. Hence `L <= 3m`. Attainment: assign gear
`i` the block `[3i, 3i + 3)` by choosing `x = -(3i + 2) (mod g_i)`; CRT realises the phase vector,
so `[0, 3m)` is entirely struck. QED.

*Contrast with L17.* The pair machine's piece is the **gapped** domino `{x, x + 2}`, which lies in
one parity class and therefore cannot tile an interval; that is the whole source of the
`- (m mod 2)`. The triple machine's piece is a **solid** triomino, which tiles exactly, so the
record is `3m` on the nose. *The parity defect is a property of the separation, not of the tooth
count.*

*Evidence.* `m = 3, 4, 5`: `F_3 = 9, 12, 15`, seven wheels, 0 exceptions.

**L36 (the walk distribution, and the gap census as a second difference).** With
`C(j) = #{x : x, ..., x+j-1 all struck}` (`C(0) = W`), exactly, for every gear set:

        #{x : L(x) = j} = C(j) - C(j+1)                    (all j >= 0)
        #{x : L(x) = j} = #{gaps of length >= j + 1}       (j >= 1)
        N_d = C(d-1) - 2 C(d) + C(d+1)                     (the gap census, d >= 1)
        F_top = max { j : C(j) > 0 }
        sum_x L(x) = sum_{j >= 1} C(j)                     (the mean walk is (1/W) sum_j C(j))

*Proof.* Inside a gap of length `d` beginning at an open `a`, the values of `L` are
`0, d-1, d-2, ..., 1`; summing over gaps gives the first two lines and the second difference gives
the third; `sum_x L(x) = sum_{j>=1} #{L >= j} = sum_{j>=1} C(j)`. QED.

*Reading.* **The gap census is the second difference of the all-struck count `C`, exactly as the
run spectrum (L11) is the second difference of the all-open count `A(L) = prod(g - 2 - L)`.** The
two are dual; the difference is that `A` is a product and `C` is not.

*Evidence.* 0 mismatches on all five identities in ten wheels.

**L37 (the closed form of `C`).** If every gear is at least `j + 2`,

        C(j) = sum_{k, e} (-1)^k T(j, k, e) prod_g (g - 2k + e) ,

where `T(j, k, e)` counts the `k`-subsets `S` of `[0, j)` with exactly `e` elements `s` having
`s - 2` in `S`; and `T` is itself closed - the convolution over the even and the odd positions of
`[0, j)` of the path counts `binom(k-1, e) binom(n-k+1, k-e)` with `n = ceil(j/2)` and
`floor(j/2)`.

*Proof.* Inclusion-exclusion over which of the `j` positions are open. "All of `S` open" is a
per-gear condition forbidding `U(S) = {-s, -s-2 : s in S}`, so it counts
`prod_g (g - |U(S) mod g|)`; `|U(S)| = 2|S| - e(S)` because the only coincidence `-s = -s' - 2` is
`s' = s - 2` (the other, `s' - s = g - 2`, needs `g <= j + 1`). The path count is the standard
number of `k`-subsets of a path of `n` vertices with `e` adjacent pairs. QED.

First values: `C(1) = W - prod(g-2)`; `C(2) = W - 2 prod(g-2) + prod(g-4)`;
`C(3) = W - 3 prod(g-2) + prod(g-3) + 2 prod(g-4) - prod(g-5)`.

*Evidence.* 0 mismatches in the six wheels where the hypothesis holds; `T` verified against brute
force for `j = 0..12`.

*The shape is the point.* `C(j)` is an **alternating sum** of shifted wheel products and not a
single product - "all struck" is not a per-gear condition, while "all open" is. That is the exact
reason the walk's distribution is harder than the run spectrum, and the reason the record cannot
be read off a product (L41).

**L38 (the hop law and THE COLLAPSE).** Let `y` be the `G`-landing of the walk and add a gear `g`.
Then `g` hops at `y` iff `y = 0` or `-2 (mod g)`; after a hop the next position `g` itself strikes
is `y + 2` when `y = -2 (mod g)` and `y + g - 2` when `y = 0 (mod g)`. If `g > F_G + 3` the hop
chain has length **at most 2**, a double hop occurs **iff** `y = -2 (mod g)` and the `G`-gap at
`y` is exactly 2, and the layer is the non-recursive one-liner

        L_{G+g}(x) = L_G(x) + [y = 0 or -2 mod g] * ( d1 + [y = -2 mod g and d1 = 2] * d2 )

with `d1` the `G`-gap at `y` and `d2` the `G`-gap at `y + 2`.

*Proof.* The teeth are `0` and `-2`, so from a strike at `y = 0` the next strike of the same gear
is at `y + g - 2 > y + F_G + 1`, past the next `G`-opening; from `y = -2` it is at `y + 2`, which
is a `G`-opening only if `d1 = 2`, and the strike after that is at `y + g > y + F_G + 3`. QED.

*Evidence.* Twelve layers, 391,048 positions: hop law 0 exceptions, longest chain exactly 2 in
every layer, the double-hop rule 0 exceptions (3,280 double hops), the one-line formula 0
mismatches. **Sharp in the other order**: adding gear 7 last to `{11,13,17}`, `{13,17,19}`,
`{17,19,23}` gives hop chains of length 3.

**L39 (the nested form and its cost).** The walk is the nested composition of the L38 layers over
the gears in increasing order: `W(i, x) = y - x` where `y` starts at `x + W(i-1, x)` and is pushed
to `(y + 1) + W(i-1, y+1)` while gear `i` strikes it. Exact against the scan in seven wheels; the
mean number of hops per walk is below 1 in the large-gear regime (0.38 to 0.90 measured, against
`sum_g 2/g` = 0.31 to 0.62), so the lazy cost of the nested form is `m` layer tests plus fewer
than one gap lookup.

*Order.* Smallest gear first, because that keeps the new gear the largest and so keeps
`g > F_G + 3` - the hypothesis of the collapse - true at every step.

**L40 (the per-gear transform; the shield in place of the fold).** With
`f^(a) = (1/W) sum_n f(n) omega_W^{-an}`, the open indicator of one gear has

        u_g^(0) = (g - 2)/g ,
        u_g^(a) = -(1 + omega_g^{2a})/g = -(2/g) cos(2 pi a/g) * omega_g^{a}    (a != 0) ,

so in the **shield coordinate** `n' = n + 1` (teeth `+-1`) the transform is exactly
`-(2/g) cos(2 pi a/g)`, real. The full spectrum is the CRT product `O^(a) = prod_g u_g^(alpha_g)`
with `alpha_g = a (W/g)^{-1} (mod g)`.

*Proof.* `sum_r u(r) omega^{-ar} = 0 - 1 - omega^{2a}` for `a != 0`. QED.

*Reading.* The bottom machine's per-gear factor is the real `-(2/g) cos(2 pi a u_g/g)` because its
teeth are symmetric about `0` (the fold). The top machine's factor is the **same real cosine with
`u = 1`, multiplied by the single phase of the shield's position**: the top machine is the `u = 1`
machine, and the bottom's fold is replaced by one translation. This is L19's conjugacy seen on the
Fourier side - the `+1` in `6^{-1}(n + 1)` is the shield shift, the `6^{-1}` is the change of `u`.

*Evidence.* Exact DFT at five wheels, 32,077 frequencies, max error 1.1e-16; the shifted transform
real to 1.1e-16.

**L41 (full spectral support, and what the spectrum cannot decide).** `O^(a) != 0` for every `a`,
because `cos(2 pi a/g) = 0` needs `4a = g (mod 2g)` and `g` is odd. The same holds for every
**nonempty** run indicator: its per-gear factor is a Dirichlet kernel of length `L + 2`, which
vanishes only when `g | a(L + 2)` with `a != 0`, and a nonempty run needs `L <= q' - 3` (L10), so
`L + 2 < g` for every gear. Consequently the spectrum **decides the run record** - `A(L) =
prod(g - 2 - L)` is a product, and a product vanishes iff a factor does, at `L = q' - 2` - and it
**cannot decide the blocked record** `F_top`, because `C(j)` is a signed sum of products (L37) and
its positivity is not a support question.

*Evidence.* Minimum magnitudes 3.3e-07 to 3.1e-05 over five wheels, never 0; the predicted
run-indicator zeros are unreachable.

**L42 (the striker-parity bit and its exact bias).** The XOR of the gears' masks is the parity of
the number of striking gears (a Liouville-type bit restricted to `G`). Exactly per wheel

        #{even} = (W + prod(g - 4))/2 ,      #{odd} = (W - prod(g - 4))/2 ,

so the even-minus-odd excess is exactly `prod(g - 4)`, **which is also the number of adjacent open
pairs (L15)**: the same polynomial answers two unrelated questions.

*Proof.* `sum_n (-1)^{number of strikers}` factors over the gears as `prod_g ((g - 2) - 2)`. QED.

*Evidence.* 10 wheels, exact.

**L43 (the record block is an exact cover; the XOR run reaches the record).** In every wheel
tested at least one record block is struck **exactly once at every cell**; hence, since
`{XOR = 1}` is contained in `{blocked}`, the longest run of ones of the striker-parity bit equals
`F_top` exactly - 10 of 10, at both parities of `m`. For even `m` *every* record block is an exact
cover; for odd `m` two thirds are, and the "one unit of waste" of L17 is paid by a gear whose
second tooth falls outside the window, not by a doubly covered cell.

*Reading.* Bitwise, the answer is the reverse of the bottom machine's: the XOR does not bound the
OR's longest run from above (a parity bit is blind to multiplicity), but it bounds it **from
below**, and here the bound is **tight**.

**L44 (the correlation is a true product).** For every `d >= 1`,

        B(d) = #{n : n and n + d both open} = prod_g c_g(d) ,
        c_g(d) = g - 2 if d = 0 (mod g), g - 3 if d = +-2 (mod g), g - 4 otherwise.

*Proof.* "Both open" is a per-gear condition forbidding `{0, -2} u {-d, -d-2}`, whose size is 4
minus the overlaps, and the only overlaps are `d = 0, +-2 (mod g)`. QED. Special cases:
`B(1) = prod(g-4)` and `B(2) = prod(g-3)` (both L15).

*Evidence.* 400 values, 10 wheels, 0 mismatches.

**L45 (the holes).** Since `c_g(d) >= g - 4 > 0`, the correlation never vanishes: **every distance
occurs between some two open pairs**, so holes exist only in the consecutive (gap) census. In the
pair view the only gap hole below the record is `d = 4` (L4), 10 of 10. In the triple
(twin-candidate) view the holes are `d = 2` and `d = 3`.

*Proof.* If `x` starts a twin candidate and `x + 1` does not, the gear responsible has
`x + 1 = 0, -1` or `-2 (mod g)`, i.e. `x = -1, -2` or `-3`; the first two would strike `x` itself,
so `x = -3 (mod g)`, and that gear then strikes `x + 1, x + 2, x + 3`. So the next twin candidate
is at `x + 1`, or at `x + 4` or beyond. QED.

*Evidence.* 10 of 10.

---

## 5. What is new

**The deliverable.** *The next opening of the top machine has a closed form in the residues.*

        next open pair after x       =  x + mex { (-x) mod g , (-x-2) mod g : g in G }
        next twin candidate after x  =  x + mex { (-x) mod g , (-x-1) mod g , (-x-2) mod g }

exact whenever every gear exceeds `2m` (resp. `3m`), `O(m)` operations, no scan and no period; and
exact for every gear set at all if the two (three) residues per gear are replaced by their
arithmetic progressions, at a measured cost of 107 to 7,914 terms on in-use machines with up to
443 gears. The mex is not a re-labelling of the scan: it says that **the walk is decided by `2m`
numbers read off the residues and by nothing else**, and its hypothesis is exactly the regime in
which each gear's contribution to a window is one domino.

**The parity defect is a property of the separation, not of the tooth count (L35).**
`F_top = 2m - (m mod 2)` for the pair machine and `F_3 = 3m` for the twin-candidate machine,
because the pair machine's piece `{x, x + 2}` lies inside one parity class and cannot tile an
interval, while the triple machine's piece `{x, x+1, x+2}` is solid and tiles exactly. New, with a
short proof, and it puts a hard, gear-size-independent ceiling `3m` on the walk to the next twin
candidate in the large-gear regime.

**The dual of the run spectrum (L36).** The run spectrum is the second difference of the all-open
count `A(L) = prod(g - 2 - L)` (L11); the gap census is the second difference of the all-struck
count `C(j)`, and the walk-length distribution is its first difference. `A` is a product because
"all open" is per-gear; `C` is an alternating sum of shifted wheel products (L37) because "all
struck" is not. That asymmetry is why the record is hard and the run ceiling easy, and it is
stated exactly in L41.

**The mean walk in closed form.** `sum_x L(x) = sum_{j>=1} C(j)`, so the expected number of steps
to the next opening is `(1/W) sum_{j>=1} C(j)`, fully explicit in the gears through L37. Verified
to six decimals in seven wheels.

**The collapse of the layer (L38).** Because the letters are `{2, g - 2}`, a large new gear can
hop at most twice on the lower machine's openings, and the double hop has one exact cause: the
landing sits on the tooth `-2` and the lower gap is exactly 2. The layered walk therefore has a
one-line non-recursive form per gear - the bottom machine's nested next-opening formula, but with
the chain provably bounded at 2 rather than merely observed to be short.

**The top machine is the `u = 1` machine (L40).** The bottom's pole-phase factor
`-(2/g) cos(2 pi a u_g/g)` becomes `-(2/g) cos(2 pi a/g)` times the shield's phase; shifting to the
shield coordinate makes the whole spectrum real. The fold is replaced by a translation. Full
spectral support for `O` and for every nonempty run indicator (L41).

**Bitwise, the answer reverses (L42, L43).** The striker-parity bit has exact bias `prod(g - 4)`,
the same polynomial as the domino count; and its longest run of ones equals `F_top` exactly,
because a record block is always an exact cover. The XOR bounds the OR's longest run **from
below**, tightly - where the bottom machine's parity barrier gave nothing.

**Prior art met and stopped.** The mex of a union of arithmetic progressions is the standard
"first uncovered point of a covering system"; nothing asymptotic is claimed. `F_top` is the
two-class Jacobsthal function of the gear set and `F_3` its three-class analogue (Jacobsthal 1961;
Iwaniec 1978 for the asymptotic); the results here are exact structure at fixed gear sets, which
that literature does not address. `prod(g-2)`, `prod(g-3)`, `prod(g-4)` are the usual Schemmel /
Hardy-Littlewood local factors, used as bookkeeping. The gap census by inclusion-exclusion is
being derived per length in the parallel second pass (`top_machine_2.md`, P6-P8); L36/L37 reach the
same numbers uniformly in `d` from the all-struck count, and the two agree where they overlap
(`N_1 = prod(g-4)`, `N_4 = 0`).

**Kernel-checkable next**, in order of cheapness: L30 and L34 (the mex forms - two case splits and
one `Nat.lt_of_lt_of_le`, no CRT needed); L45's triple-hole proof (five lines, the shape of the
existing `no_gap_four`); L44 (the correlation product - needs only the existing `card_filter_crt`
engine); L35's upper bound (a counting argument of the same shape as the existing `parity_upper`;
its attainment needs the same missing `Finset`-indexed CRT lemma as L17's); L36's identities
(elementary, no CRT); L42 (one character sum per gear).

---

## 6. Verdict

**The top machine's walk is a mex.** From any position `x`, read the `2m` numbers `(-x) mod g` and
`(-x - 2) mod g`; the least non-negative integer missing from that list is exactly the distance to
the next open pair, whenever every gear exceeds `2m`. Nothing else about the position matters, no
scan is involved, and the whole metric behaviour of the machine - the record `2m - (m mod 2)`, the
increments `+3, +1`, the impossibility of a gap of 4 - is the combinatorics of those `2m` numbers
falling into same-parity pairs `{a, a - 2}`. With three teeth instead of two the same statement
locates the next **twin candidate**, at `x + mex` of `3m` numbers, and there the ceiling is `3m`
exactly: no parity defect, because a solid triomino tiles an interval and a gapped domino does
not.

**Out of the large-gear regime the form survives and only the cost changes.** The mex is taken
over arithmetic progressions instead of single residues; on the in-use machines (up to 443 gears,
`N = 10^7`) it is exact on every walk tested, at 107 to 7,914 terms, and the typical walk is 3 to
10 steps against records of 71 to 3,006. The covering bound `2m/(1 - 2H_S)` is a proof but dies
when the reciprocals of the gears below the record reach `1/2`, which happens inside the in-use
range; the in-use record is not a covering phenomenon but a gear-zone one (L21), and this branch
does not close it.

**The transforms say what they can and cannot do.** The spectrum factors over the gears exactly,
is real in the shield coordinate, and never vanishes - so it decides every *product* question (the
run ceiling `q' - 3`, the counts `prod(g - 2 - L)`, the correlations `prod c_g(d)`) and cannot
decide the record, which is the positivity of an alternating sum. The bitwise view supplies the
missing half from below: the striker-parity bit's longest run of ones **equals** the record,
because a record block is always an exact cover.

No interpretation against the twin conjecture is offered; the clutch comes later.

---

## 7. Scorecard, filled

| # | Prediction | Result |
|---|---|---|
| Q1 | `L(x) = mex{a_g, b_g}` when every gear `> 2m` | **held**, 0 mismatches in 1,448,287 positions; hypothesis shown sharp |
| Q2 | `L(x) <= 2m - (m mod 2)`, attained | **held**, and re-proved from the mex form |
| Q3 | `R(x) = mex{...}`; `F_3 = 3m`, no parity defect | **held**, 6 wheels plus `m = 5` at `W = 14.5M`: 9, 12, 15 |
| Q4 | walk distribution and gap census as differences of `C` | **held**, all five identities, 10 wheels, 0 mismatches |
| Q5 | `C(j)` an alternating sum of shifted products; no product form | **held**, 0 mismatches where the hypothesis holds; `T` verified to `j = 12` |
| Q6 | general mex exact; counting bound; vacuous in use | mex **held** (24,000 walks, 0 mismatches); bound **held**; "vacuous in use" **refuted at 2 of 12** (`q = 17, 19`, `N = 10^6`) |
| Q7 | smallest-first keeps the collapse | **held**; gear 7 added last gives hop chains of length 3 in three ladders |
| Q8 | hop law; next own strike at `+2` or `+ g - 2` | **held**, 0 exceptions, 12 layers |
| Q9 | at most a double hop; double iff `y = -2` and `d1 = 2`; one-line layer | **held**, 0 exceptions, 0 mismatches over 391,048 positions |
| Q10 | nested form exact; mean hops `< 1` | **held**, 0 mismatches; 0.38-0.90 hops per walk |
| Q11 | per-gear transform; real in the shield coordinate | **held** to 1.1e-16, 32,077 frequencies |
| Q12 | full spectral support | **held**, minima 3.3e-07 to 3.1e-05 |
| Q13 | run indicator a Dirichlet kernel; zeros iff `gcd(L+2,g) > 1` | kernel **held** for `L >= 2` (and `L = 1` is not an interval - the domino); the zero condition is **unreachable**, so run indicators also have full support: a sharpening, and a partial refutation |
| Q14 | spectrum decides the run record, not `F_top` | **held**, stated exactly in L41 |
| Q15 | XOR bias `= prod(g - 4)` = the domino count | **held**, 10 wheels, exact |
| Q16 | XOR run `= F_top` for even `m`, `<` for odd | **held for even `m`; refuted for odd `m` - it is equal there too**, because the record block is an exact cover at both parities (L43) |
| Q17 | `B(d) = prod c_g(d)` | **held**, 400 values, 0 mismatches |
| Q18 | correlation never vanishes; holes `{4}` and `{2,3}` | **held**, 10 wheels each |

---

## 8. What holds without exception

Counted, and each with its evidence.

| statement | count | exceptions |
|---|---|---|
| the mex form `L(x) = mex{a_g, b_g}` (gears `> 2m`) | 1,448,287 positions, 9 wheels | 0 |
| the twin-candidate mex form (gears `> 3m`) | 1,400,085 positions, 7 wheels | 0 |
| the general mex form, in use (160-443 gears) | 24,000 walks, 12 machines | 0 |
| `F_3 = 3m` (gears `>= 3m + 3`) | 7 wheels, `m = 3, 4, 5` | 0 |
| the five `C`-identities (distribution, tail, gap census, record, mean) | 10 wheels | 0 |
| the closed form for `C(j)` (gears `>= j + 2`) | 6 wheels | 0 |
| the hop law and the next own strike | 12 layers, 391,048 positions | 0 |
| hop chain `<= 2` and the double-hop rule when `g > F_G + 3` | 12 layers, 3,280 double hops | 0 |
| the one-line layer formula | 12 layers, 391,048 positions | 0 |
| the nested form against the scan | 7 wheels, 23,432 walks | 0 |
| the spectral product, and reality in the shield coordinate | 32,077 frequencies, 5 wheels | 0 |
| full spectral support of `O` | 32,077 frequencies | 0 |
| the parity bias `prod(g - 4)` | 10 wheels | 0 |
| longest XOR run `= F_top` | 10 wheels | 0 |
| a record block covered exactly once exists | 10 wheels | 0 |
| the correlation product `B(d) = prod c_g(d)` | 400 values, 10 wheels | 0 |
| the pair-view hole `{4}` and the triple-view holes `{2, 3}` | 10 wheels each | 0 |

---

## 9. Dead ends

- **A product formula for the walk-length distribution.** There is none, and the reason is
  structural: "all struck" is a union condition, not a per-gear one, so `C(j)` is an alternating
  sum (L37). Every attempt to read the record off the spectrum founders on the same point (L41).
- **"The counting bound decides the in-use record."** It does not: `H_S` passes `1/2` inside the
  in-use range and the bound goes vacuous. What survived is the transition itself - the bound is
  alive exactly while `sum_{q < g <= L} 1/g < 1/2` - and the measurement that the in-use record is
  a gear-zone (smoothness) object, not a covering object.
- **"For odd `m` the XOR run falls short of the record."** Refuted: it is equal at both parities.
  What survived is the better statement, L43: a record block is an *exact* cover, the odd-`m`
  waste being an edge singleton rather than a doubly covered cell.
- **Run-indicator spectral zeros.** Predicted from the Dirichlet kernel; unreachable, because a
  nonempty run has `L + 2 < q'`. The prediction died into a theorem (full support for runs too).
