# The wheels' last open laws (branch R8)

> **Numbering note (harvester, 2026-09-06).** Law numbers W86-W93 in the original text were renumbered W95-W102 on 2026-09-06 to leave room for document 7 (`top_machine_7.md` L67-L75 = register W86-W94). The map: W86->W95 (non-cancellation), W87->W96 (minimum-cover count), W88->W97 (cover polynomial), W89->W98 (census as a cover sum), W90->W99 (bijection), W91->W100 (second moment), W92->W101 (half-turn law), W93->W102 (parity-refined capacity bound). Document-local citations L67-L75 below are register W86-W94 in that order. Future documents number from W103.

Parent: `top_machine_7.md` (branch R7), whose ledger entry (section 6(d)) leaves four items
genuinely open on the wheels alone: (1) the cost of the core minimisation in L69, (2) the
non-cancellation of the top moment `M_{r(d)}(d)` (L75, measured 0 exceptions of 24, unproved),
(3) L22 in the kernel (a Formalist's, not this branch's), (4) a parity-refined capacity bound
(L70 is the unrefined one). The observation that spawned this branch is R7's own attack note on
item 2: "the minimal covers are enumerable in closed form ... so the sum is a product of two
one-dimensional sums; evaluate them."

**The object.** The wheels: pairwise coprime odd gears `G` acting on the pair coordinate with
teeth `0` and `-2` (dominoes `{x, x+2}`), the record `F_top(G)`, the gap census `N_d(G)`, and the
covering problem both reduce to. Free wheels = every gear above `2m`; loaded wheels = small gears
present; core gears `<= L + 1`, tail gears the rest; the domino cost `D(U)` of a set of cells is
the sum over its maximal step-2 runs per parity class of `ceil(run/2)`. Nothing about the motor,
no clutch.

**Numbering.** Laws here are numbered from **W95** in the project register
(`research/proof/law_register.md`; W1-W85 taken). Document-local laws of `top_machine_7.md` are
cited as L67-L75; the register assigned them W86-W94 on 2026-09-07 (L67 = W86, ..., L75 = W94)
and this document's own laws are W95-W102 there, as the numbering note above says. Each law
below says which document law it extends.

**Vocabulary introduced here** (names chosen not to collide with existing objects):
*pieces* of length `d` = the family `J_p = {p-2, p} n [1, d-1]`, `p in [1, d+1]`, `p != 2, d`
(L73); a *cover* = a subfamily whose union is `[1, d-1]`; `C_t(d)` = the number of covers with
exactly `t` pieces; a *minimum cover* has `t = r(d)`. *Half-index* = the coordinate `i = c/2`
on the even cells and `i = (c-1)/2` on the odd cells of a window, in which a domino `{c, c+2}`
is an adjacent pair `{i, i+1}`. The *half-turn* `H = 2^{-1} mod W_core = (W_core + 1)/2` (not to
be confused with the *antipode* `n = 2, -4` of document 1 or the column antipode `(P +- 1)/2`).

---

## 1. Pre-registered predictions and scorecard

Written before any computation of this branch. What is below was derived on paper from L73's
expansion; the closed form in P1 was checked by hand against the 24 values R7 published (that
table is prior data, not a computation of this branch) before anything was run.

### Section 1. The non-cancellation of the top moment (R7 item 2, L75)

**The mechanism, derived.** Write `f(x) = 4 + sum_p u_p(x)` as in L73 and expand `f^k` as a sum
over sequences `(q_1, ..., q_k)` with each `q_i` either the constant `4` or one of the pieces.
A sequence whose distinct pieces form the set `P` contributes `4^{#4s}` times
`sum_x (-1)^{|x|} prod_{p in P} u_p(x)`, and by inclusion-exclusion over `P`

        sum_x (-1)^{|x|} prod_{p in P} u_p(x) = sum over T subset of P with union T = [1,d-1]
                                                of (-1)^{|T|}  =:  mu(P) .

(Each `u_p = 1 - prod_{j in J_p}(1 - x_j)`; expanding the product over `P` and using
`sum_x (-1)^{|x|} prod_{j in V}(1 - x_j) = [V = [1, d-1]]`.) So `mu(P) = 0` unless `P` contains
a cover, and at `k = r(d)` the only sequences that survive are the `r!` orderings of each minimum
cover `P`, for which `mu(P) = (-1)^r` (the only covering subfamily of a minimum cover is
itself). Hence

**P1 (THE NON-CANCELLATION, with the sign and the count).**

        M_{r(d)}(d) = (-1)^{r(d)} r(d)! C_{r(d)}(d) ,

every minimum cover carrying the **same** sign `(-1)^{r(d)}`, so cancellation is impossible and
`M_{r(d)}(d) != 0` exactly when a cover exists. Predicted exact at every `d = 2..26` against
R7's `moments.py` values, **including `d = 4`** where `C = 0` and `M = 0`. Refuted by one `d`
where the identity fails.

**P2 (THE MINIMUM-COVER COUNT IN CLOSED FORM).** Minimum covers split by parity class; a class
that is a step-2 run of even length has exactly one minimum cover (its tiling), a run of odd
length `2a + 1` has `a` covers with one doubly-covered interior cell plus one cover for each end
singleton available to it (`{1}` to the class of 1, `{d-1}` to the class of `d-1`). Predicted:

        C_{r(d)}(d) = d/4 - 1         (d = 0 mod 4)     [= 0 at d = 4]
                    = 1               (d = 1 mod 4)
                    = (d + 6)/4       (d = 2 mod 4)
                    = ((d + 1)/4)^2   (d = 3 mod 4) ,

predicted equal to a brute-force enumeration of minimum covers at every `d = 2..24` (0
mismatches), and predicted to give, with P1, the eight published record multiplicities
`18, 96, 24, 24, 480, 6480, 1440, 720` at `d = 6..13` as `r! C_r`: `6*3, 24*4, 24*1, 24*1,
120*4, 720*9, 720*2, 720*1`.

**P3 (UNIFORM SIGNS, checked cover by cover).** Predicted: for every minimum cover `P` of every
`d <= 14`, the brute-force value of `mu(P) = sum_x (-1)^{|x|} prod_{p in P} u_p(x)` is exactly
`(-1)^{r(d)}` - 0 exceptions over all minimum covers.

**P4 (THE WHOLE SIGNATURE IS A COVER POLYNOMIAL).** The same expansion at every `k` gives the
generating identity

        sum_e c_e(d) z^e = sum_t C_t(d) (1 - z)^t z^{d + 3 - t} ,

(proof: `sum_S (-1)^{|S|} z^{e(S)} = z^4 sum_S (-1)^{|S|} prod_p (1 + (z-1) u_p(S))`, expand the
product over subfamilies `T`, apply `mu`). Predicted exact - every coefficient `c_e(d)` - for
`d = 2..26`, with `C_t(d)` computed independently by brute force over all `2^{d-1}` subfamilies
for `d <= 20` and by a two-state transfer matrix per parity class for all `d`; transfer matrix
against brute force: 0 mismatches for `d <= 20`. Consequences predicted to fall out as
corollaries: L73 (the factor `(1-z)^t` vanishes to order `t >= r` at `z = 1`), L74, P1, L25's
degree law, and L24's gap-3 / gap-5 coincidence as the identity
`z^2 + 2(1-z)z + (1-z)^2 = 1` between the cover polynomials of `d = 3` and `d = 5`.

**P5 (THE UNIVERSAL CENSUS AS A COVER SUM).** With `P_G(e) = prod_{g in G} (g - e)` and `Delta`
the forward difference in `e`, predicted for every gear set with all `g > d + 2`:

        N_d(G) = sum_{t = r(d)}^{m} C_t(d) (-1)^t (Delta^t P_G)(d + 3 - t) ,

exact against L22 on 5 wheels for `d = 1..10` (0 mismatches), and with the top term
`C_m(d) m!` alone when `r(d) = m`.

**P6 (THE BIJECTION BEHIND L18).** In a wheel of exactly `r(d)` gears, all `> d + 2`, the gaps
of length `d` are in bijection with pairs (minimum cover, bijection gears -> pieces), so
`N_d = r(d)! C_{r(d)}(d)` by a direct CRT count with no inclusion-exclusion. Predicted 0
mismatches against full-period scans at `(d, G) = (6, {11,13,17})`, `(7, {11,13,17,19})`,
`(8, {11,13,17,19})`, `(9, {13,17,19,23})`, `(10, {13,17,19,23,29})` and against L22 at
`d = 11, 12, 13` with six gears above 15.

**P7 (THE SECOND MOMENT).** Predicted from the same expansion, for `d >= 3`, `d != 4`:

        M_{r+1}(d) = (-1)^r (r+1)!/2 * [ C_r(d) (2d + 6 - r) - 2 C_{r+1}(d) ] ,   r = r(d) .

Exact at `d = 3..26`.

**P8 (THE UNIVERSAL SIGNATURE IS POLYNOMIAL-TIME).** Because `C_t(d)` comes from a transfer
matrix, `c_e(d)` for all `e` costs `O(d^2)` rather than `2^{d-1}`. Predicted: the transfer-matrix
signature agrees with `moments.py`'s exact one at every `d <= 26`, and `c_e(d)` is then tabulated
to `d = 60`, with `M_{r(d)}(d)` to `d = 60`. This reclassifies R7's ledger item (b) "the census
beyond `d = 20`" for the universal regime.

### Section 2. The core minimisation's structure (R7 item 1)

**P9 (PARITY DECOMPOSITION AND THE HALF-TURN, exact).** `D(U) = D(U_even) + D(U_odd)` by
definition, and in half-index coordinates each core gear `g` acts on each class as a gear with
two ADJACENT teeth `{u, u-1} mod g`; the odd class's phase is the even class's shifted by
`(g-1)/2 = -2^{-1} mod g`. Hence, with `W_core = prod core`, the two parity classes of `[0, L)`
are two windows of ONE pattern - the adjacent-teeth core wheel `Q` (position `i` struck iff some
core gear has `i = u_g` or `u_g - 1 mod g`) - at positions `[0, ceil(L/2))` and
`[H, H + floor(L/2))`, `H = (W_core + 1)/2`. Predicted: the odd-class uncovered mask equals the
even-class pattern read at offset `H`, 0 mismatches on every phase vector of 200 gear sets.
Consequence: `F_top = max{L : min over x in Z_{W_core} of [c_{ceil(L/2)}(x) + c_{floor(L/2)}(x + H)]
<= t(L)}`, a scan of the CORE period with an additive window cost, not of `W`.

**P10 (DECOUPLING FAILS).** The two classes do not minimise independently: predicted
`min_x [c_e(x) + c_o(x + H)] > min_x c_e(x) + min_x c_o(x)` at `L = F + 1` on at least one set
(`{7,11,13}` at `L = 7`, worked by hand: sum-of-minima 2, true minimum 3), and the decoupled
record formula over-estimates `F_top` on **at least 100** of the loaded sets. Refuted if the
decoupled formula is exact everywhere.

**P11 (THE GEARS' DOMINOES CANNOT KEEP A FIXED PARITY PATTERN - the mechanism of waste).**
Consecutive dominoes of one core gear are `g` apart in half-index, `g` odd, so they alternate
between the two adjacent-pair grids `{2j, 2j+1}` and `{2j+1, 2j+2}`; the run between two
consecutive dominoes of the same gear, when no other gear intervenes, has length `g - 2`, odd,
and costs `(g-1)/2` pieces for `g - 2` cells: one half-piece of waste per interior run. Predicted
consequence, tested as **E2**: the run structure matters at the optimum - there are sets where
`min_v D(U(v)) > min_v [ceil(|U_e(v)|/2) + ceil(|U_o(v)|/2)]` at `L = F + 1`. Predicted at
least 1 such set; count reported.

**P12 (ANCHORING IS NOT A LAW).** Predicted: "some optimal phasing at `L = F` has a core gear
striking cell 0" holds on most loaded sets but has exceptions (>= 1); likewise for cell `L - 1`.

**P13 (HOW SPECIAL THE OPTIMUM IS - measurement).** Over all `prod(core)` phase vectors at
`L = F` and `L = F + 1`: the number of distinct `D` values and the fraction attaining the minimum.
Predicted: at `L = F` the median number of distinct values over loaded sets is between 3 and 8,
the median fraction at the minimum between 1% and 20%; and the full enumeration reproduces
`F_rule` on every set (0 mismatches, a third independent decision of the record after the scan
and the branch-and-bound).

**P14 (THE COST, honestly).** No algorithm polynomial in `L` and `|core|` is claimed. What is
claimed: the exact decision is a scan over `W_core` (P9) with the cost updated in `O(1)` per
position, and the family's largest `W_core` is reported against its `W`.

### Section 3. The parity-refined capacity bound (R7 item 4)

**P15 (THE BOUND, proved).** For a core gear `g` and window `[0, L)` let `A_g(L)` be the set of
achievable pairs `(|trace n even|, |trace n odd|)` over its `g` phases. Then `[0, L)` coverable
implies

        min over choices (a_g, b_g) in A_g(L), g in core, of
            ceil( max(0, ceil(L/2) - sum a_g) / 2 ) + ceil( max(0, floor(L/2) - sum b_g) / 2 )
        <= t(L) ,

and `Lcap2(G) = max{L : that holds}` satisfies `F_top <= Lcap2 <= Lcap` (L70). Computable by a
DP over the core with state `(sum a, sum b)`, cost `O(|core| L^2 |A|)`. Predicted 0 violations
on the whole scanned family.

**P16 (SLACK).** Predicted: `Lcap2 = F_top` on every free set and on a majority of loaded sets;
maximum slack on the family at most 4 (against L70's 9 at `{7,11,13,17}` and 6 at `{13..41}`);
`Lcap2({7,11,13,17}) = 9` exactly and `Lcap2({13,17,19,23,29,31,37,41}) = 20`, slack 2 (both
worked by hand).

**P17 (WHAT THE SLACK IS MADE OF).** The task's hypothesis - "the loose cases are exactly the
ones where core dominoes overlap each other" - is predicted **false as an iff**: three levels are
computed at `L = F + 1`, capacity (`Lcap2`), counts-with-overlap (`min_v ceil(|U_e|/2) +
ceil(|U_o|/2)` over actual phase vectors, which sees overlaps but not runs), and exact `D`; the
loose cases split into those closed by the counts level (overlap) and those closed only by `D`
(run parity). Predicted: both kinds occur, each on at least 5 sets.

### Section 4. The ledger

**P18.** Predicted: of R7's four open items, this branch closes 2 and 4 outright (proofs),
reduces 1 to a core-period scan and reclassifies it as computational with no polynomial
algorithm claimed, and leaves 3 to the Formalist; and it adds none. The wheels are predicted to
have **no open structural item on paper** after this branch.

### Scorecard

| # | Prediction | Result |
|---|---|---|
| P1 | `M_{r(d)}(d) = (-1)^r r! C_r(d)`, `d = 2..26` incl. `d = 4` | |
| P2 | `C_r(d)` closed form by `d mod 4`; the eight multiplicities as `r! C_r` | |
| P3 | every minimum cover has `mu = (-1)^r`, `d <= 14` | |
| P4 | cover-polynomial identity for all `c_e(d)`, `d = 2..26`; transfer matrix = brute force | |
| P5 | universal `N_d` as a cover sum, 5 wheels, `d = 1..10` | |
| P6 | bijection count `N_d = r! C_r` by scan (5 cases) and L22 (3 cases) | |
| P7 | `M_{r+1}(d)` formula, `d = 3..26` | |
| P8 | polynomial-time signature; tables to `d = 60` | |
| P9 | half-turn identity, 0 mismatches, 200 sets | |
| P10 | decoupling fails on >= 100 loaded sets | |
| P11 | E2: run parity matters at the optimum on >= 1 set | |
| P12 | anchoring at cell 0 / `L-1` has exceptions | |
| P13 | distinct-`D` and min-fraction medians in the stated ranges; enumeration = `F_rule` | |
| P14 | cost stated as a `W_core` scan; largest `W_core` reported | |
| P15 | `Lcap2` proved; 0 violations; `F <= Lcap2 <= Lcap` | |
| P16 | `Lcap2` exact on all free sets, majority of loaded; max slack <= 4; the two hand values | |
| P17 | slack splits into overlap and run-parity kinds, each >= 5 sets | |
| P18 | ledger: 2 closed, 1 reduced, 1 the Formalist's; none added | |

---

## 2. Setup as computed

Scripts in `research/topmachine/r8/`, results (untracked) in `.../results/`. Every count is exact:
integer inclusion-exclusion, exhaustive subfamily enumeration, full enumeration of every core
phase vector, full-period scans; nothing sampled, nothing fitted.

| script | what it computes |
|---|---|
| `covers.py` | `C_t(d)` by brute force over all subfamilies (`d <= 22`) and by the two-state transfer matrix (`d <= 60`); `c_e(d)` exactly by R7's `moments.py` (`d <= 26`, up to 33,554,432 subsets) against the cover polynomial; `M_r`, `M_{r+1}`; `mu(P)` per minimum cover; the census by scan, by L22 and by the cover sum |
| `phases.py` | the 6,659-set family of R7 (pairwise-coprime subsets of `{5,7,9,11,13,17,19,23,25,29,31,37,41,43,47,49}` of size 2..6, period `<= 24,000,000`; regenerated here and counted again as **6,659**, 5,006 loaded, 1,653 free); `F` from R7's `rule.F_rule`; at `L = F` and `L = F + 1` **every** core phase vector enumerated in half-index coordinates (largest enumeration 5,766,215 vectors), with `D`, class counts, decoupled minima, anchoring, the two bounds; the half-turn identity on every phase vector of 200 sets |
| `bound_post.py` | the bounds by core density, the minimisers under reflection, the anchoring exceptions' mechanism |

Composite gears (9, 25, 49) are included everywhere; nothing uses primality.

---

## 3. Results

### 3.1 The non-cancellation, proved, with the sign and the count (P1, P2, P3)

`M_{r(d)}(d) = (-1)^{r(d)} r(d)! C_{r(d)}(d)` at every `d = 2..26`, **0 mismatches of 25**, `d = 4`
included (`C = 0`, `M = 0`); the counted `C_r(d)` equals the closed form at every `d`,
**0 mismatches**; and every one of the 34 minimum covers of `d = 2..14` has
`mu(P) = (-1)^{r(d)}` exactly, **0 exceptions** - the signs are uniform, cover by cover.

| `d` | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `r(d)` | 1 | 2 | inf | 2 | 3 | 4 | 4 | 4 | 5 | 6 | 6 | 6 | 7 |
| `C_r(d)` | 2 | 1 | 0 | 1 | 3 | 4 | 1 | 1 | 4 | 9 | 2 | 1 | 5 |
| `(-1)^r r! C_r` | -2 | 2 | 0 | 2 | -18 | 96 | 24 | 24 | -480 | 6480 | 1440 | 720 | -25200 |
| `M_{r(d)}(d)` exact | -2 | 2 | 0 | 2 | -18 | 96 | 24 | 24 | -480 | 6480 | 1440 | 720 | -25200 |

| `d` | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `r(d)` | 8 | 8 | 8 | 9 | 10 | 10 | 10 | 11 | 12 | 12 | 12 | 13 |
| `C_r(d)` | 16 | 3 | 1 | 6 | 25 | 4 | 1 | 7 | 36 | 5 | 1 | 8 |
| `M_{r(d)}(d)` exact `= (-1)^r r! C_r` | 645120 | 120960 | 40320 | -2177280 | 90720000 | 14515200 | 3628800 | -279417600 | 17244057600 | 2395008000 | 479001600 | -49816166400 |

The eight published multiplicities are `r! C_r`: `6*3, 24*4, 24*1, 24*1, 120*4, 720*9, 720*2,
720*1 = 18, 96, 24, 24, 480, 6480, 1440, 720`, **0 mismatches of 8**. The closed form by residue
of `d` mod 4 reads: one minimum cover when both classes are even runs (`d = 1 mod 4`); a square
when both are odd runs each with one end singleton (`d = 3 mod 4`, `((d+1)/4)^2`); a linear count
when only one class is odd (`d = 2 mod 4`, the odd class has both end singletons, `(d+6)/4`;
`d = 0 mod 4`, the odd class has neither, `d/4 - 1`, which is `0` at `d = 4` - L4 again).

### 3.2 The whole signature is a cover polynomial (P4, P7, P8)

`sum_e c_e(d) z^e = sum_t C_t(d) (1-z)^t z^{d+3-t}`: every coefficient of every `c_e(d)`,
`d = 2..26`, **0 mismatches** (`d = 4`: both sides identically zero). The cover counts by the
transfer matrix agree with brute force over every subfamily for `d = 2..22`, **0 mismatches**.
The cover counts themselves, for the record:

| `d` | `C_t(d)` (`t`: count) |
|---|---|
| 2 | 1: 2, 2: 1 |
| 3 | 2: 1 |
| 5 | 2: 1, 3: 2, 4: 1 |
| 6 | 3: 3, 4: 4, 5: 1 |
| 7 | 4: 4, 5: 4, 6: 1 |
| 8 | 4: 1, 5: 6, 6: 5, 7: 1 |
| 9 | 4: 1, 5: 6, 6: 11, 7: 6, 8: 1 |
| 10 | 5: 4, 6: 14, 7: 16, 8: 7, 9: 1 |
| 11 | 6: 9, 7: 24, 8: 22, 9: 8, 10: 1 |
| 12 | 6: 2, 7: 21, 8: 40, 9: 29, 10: 9, 11: 1 |
| 13 | 6: 1, 7: 12, 8: 46, 9: 62, 10: 37, 11: 10, 12: 1 |
| 14 | 7: 5, 8: 35, 9: 86, 10: 91, 11: 46, 12: 11, 13: 1 |

L24's coincidence in this language: `C_t(3) = (t=2: 1)` and `C_t(5) = (2: 1, 3: 2, 4: 1)`, and
`(1-z)^2 z^6 + 2(1-z)^3 z^5 + (1-z)^4 z^4 = (1-z)^2 z^4 (z + (1-z))^2 = (1-z)^2 z^4`: the two
signatures agree because `(z + (1 - z))^2 = 1`.

The second moment: `M_{r+1}(d) = (-1)^r (r+1)!/2 [C_r (2d + 6 - r) - 2 C_{r+1}]` at every
`d = 3..26`, `d != 4`, **0 mismatches** (e.g. `d = 6`, `r = 3`: `-(4!/2) [3 (12 + 6 - 3) - 2 * 4]
= -12 * 37 = -444`; `d = 11`: `378000`; `d = 26`: `2092278988800`).

Beyond the enumeration: the transfer matrix gives `c_e(d)` in `O(d^2)`; `d = 27` (which would
need `2^26` subsets) has 27 nonzero coefficients, `c_4 = 1, c_5 = -2, c_6 = -21, c_7 = 64,
c_8 = 131, c_9 = -714, ...`, and `M_{r(d)}(d)` continues `r(27) = 14, C = 49, M = 4271736268800`;
`r(30) = 15, C = 9`; `r(31) = 16, C = 64`; `r(40) = 20, C = 9`; `r(50) = 25, C = 14`;
`r(60) = 30, C = 14`.

### 3.3 The universal census as a cover sum, and the bijection (P5, P6)

`N_d(G) = sum_t C_t(d) (-1)^t (Delta^t P_G)(d + 3 - t)` agrees with L22 and with full-period
scans at every universal `(d, G)` tested - `{11,13,17}` for `d = 1..8`, `{13,17,19,23}` for
`d = 1..10`, `{11,13,17,19}` for `d = 6..8`, `{13,17,19,23,29}` for `d = 9, 10`, and
`{17,19,23,29,31,37}` (period 247,110,827, by L22 only) for `d = 11..13`: **0 mismatches** of 27.
Where `r(d) = m` the census is `r! C_r` exactly: `N_6({11,13,17}) = 18`,
`N_7({11,13,17,19}) = 96`, `N_8 = 24`, `N_9({13,17,19,23}) = 24`, `N_{10}({13,17,19,23,29}) = 480`,
`N_{11,12,13}({17..37}) = 6480, 1440, 720`, **0 mismatches of 8**. And where `r(d) > m` the
census is zero (`N_7({11,13,17}) = N_8 = 0`, `N_{10}({13,17,19,23}) = 0`): the record itself,
`F_top = max{d : r(d) <= m} - 1`, L26, now with the count attached.

### 3.4 The core minimisation: what the full enumeration shows (P9, P13, P14)

**The enumeration re-decides every record.** `min D <= t` at `L = F` and `min D > t` at
`L = F + 1` on all **6,659** sets, **0 mismatches** against `F_rule` - a third independent
decision after R7's scan and branch-and-bound.

**The half-turn identity** holds on every one of the 19,896 phase vectors of 200 loaded sets,
**0 mismatches**: the odd-class uncovered mask is the even-class pattern of the adjacent-teeth
core wheel read at offset `H = (W_core + 1)/2`.

**How special the optimum is.** Over the 5,006 loaded sets at `L = F`:

| | distinct `D` values | fraction of phase vectors at the minimum |
|---|---|---|
| median | 4 | 5.71% |
| min | 2 | 2 of 3,172,455 (`{5,9,11,13,17,29}`, `F = 29`) |
| max | 12 | 44.4% |

Distribution of the number of distinct `D` values at `L = F`: 2: 1,015 sets, 3: 1,256, 4: 1,087,
5: 430, 6: 376, 7: 482, 8: 222, 9: 83, 10: 39, 11: 15, 12: 1. At `L = F + 1` the median is 3
distinct values and the median fraction at the (now unaffordable) minimum is 25%. The record
cover is unique up to reflection at the top of the family: exactly one minimising phase vector
(necessarily self-mirror under `c -> L - 1 - c`) on **1,515** loaded sets, exactly two on
**1,450**, and 19 sets with only 2 minimisers among more than a million phase vectors:

| gears | `F` | core | phase vectors | at the minimum | distinct `D` |
|---|---|---|---|---|---|
| 5,9,11,13,17,29 | 29 | all six | 3,172,455 | 2 | 11 |
| 5,9,11,13,19,23 | 27 | all six | 2,812,095 | 2 | 11 |
| 5,11,13,17,19,23 | 23 | all six | 5,311,735 | 4 | 10 |
| 5,7,11,13,17,31 | 32 | all six | 2,637,635 | 2 | 10 |
| 5,7,13,17,23,29 | 29 | all six | 5,159,245 | 4 | 11 |

**The cost.** The largest core period in the family is `W_core = 5,766,215` at
`{5,7,13,19,23,29}` - and there `W_core = W`, the tail is empty, so the scan is the full period.
Over the family, `sum W_core = 182,554,029` against `sum W = 33,595,744,071`; the median
`W / W_core` is 29,939. When the tail is empty the reduction buys nothing, which is the in-use
regime (document 4 L52); the reduction is exactly the tail's period.

### 3.5 The classes are coupled, and the runs matter (P10, P11, P12)

**Decoupling fails, massively.** At `L = F + 1`, `min_v D_e + min_v D_o < min_v (D_e + D_o)` on
**3,611 of 5,006** loaded sets, and on every one of them the decoupled criterion wrongly says
"coverable". By core size at `F + 1`: 1,060 of 1,986 single-core sets, 981 of 1,377 with two core
gears, 581 of 654 with three, and **all** 688, 250 and 51 sets with four, five and six. The
smallest instance is `{5,7}` at `L = 5` (`F = 4`, `t = 1`): gear 5 covers `{0,2}` if asked to
serve the even class (leaving `U_e = {4}`, cost 1) and `{1,3}` if asked to serve the odd class
(cost 0), so the decoupled minimum is `1 <= t`; but no single phase does both - every phase gives
`D = 2 > t`. The two classes want the one gear's domino in different places, and the half-turn
locks them to one phase.

**The run structure matters.** On **1,875** loaded sets the counts-only criterion
`min_v [ceil(|U_e|/2) + ceil(|U_o|/2)] <= t` wrongly says "coverable" at `L = F + 1`; the run
parity alone refuses them. Instance `{5,7,9,17}` at `L = 15` (`F = 14`, `t = 1`, core `{5,7,9}`):
the capacity bound says cost 0, the counts level says cost 1, the true minimum `D` is 2: the
uncovered cells can be made few but not adjacent.

**Anchoring is not a law, and the exceptions are one mechanism.** At `L = F` some minimiser has a
core gear striking cell 0 on 4,796 of 5,006 loaded sets, and cell `L - 1` on the same 4,796. The
**210 exceptions are all** the sets `{7} u {four gears > 13}` at `F = 12`, `t = 4`, core `{7}`,
where the single core gear has a **unique** optimal phase, `a = 3`, trace `{1, 3, 8, 10}`, which
leaves `U_e = {0,2,4,6}` and `U_o = {5,7,9,11}` - two runs of four, cost `2 + 2 = 4 = t`. Every
phase that strikes cell 0 (`a = 0`: `{0,5,7}`; `a = 2`: `{0,2,7,9}`) leaves an odd run in a class
and costs 5. The record cover puts the core gear's two dominoes in the middle of their classes so
that the tail's four dominoes tile what is left; that is the "fixed parity pattern" the task asked
about, and it is a pattern of the *uncovered* set, not of the gear.

### 3.6 The parity-refined capacity bound (P15, P16, P17)

`F_top <= Lcap2` on all 6,659 sets, **0 violations**; `Lcap2 <= Lcap` (L70), **0 violations**.
Exact on all 1,653 free sets. The four hand values: `{7,11,13}`: `F = 6`, `Lcap = 8`,
`Lcap2 = 6`; `{11,13,17,19,23}`: `10, 14, 10`; `{7,11,13,17}`: `9, 18, 10`; `{13,17,19,23,29,
31,37,41}`: `18, 24, 22`.

The slack is governed by the **core density at the deciding length**,
`rho = sum_{g <= F+2} 2/g` (the fraction of cells the core could cover if its traces never
overlapped):

| `rho` | sets | loaded | `Lcap2 = F` | `Lcap2 - F` median | max | beyond the scan (`>= 3m + 39`) | `Lcap = F` (L70) | `Lcap - F` median |
|---|---|---|---|---|---|---|---|---|
| `[0, 0.3)` | 3,184 | 1,531 | **3,184** | 0 | 0 | 0 | 1,349 | 1 |
| `[0.3, 0.5)` | 1,081 | 1,081 | 695 | 0 | 15 | 0 | 55 | 3 |
| `[0.5, 0.7)` | 1,066 | 1,066 | 187 | 5 | 42 | 20 | 44 | 12 |
| `[0.7, 0.85)` | 536 | 536 | 0 | 38 | 40 | 364 | 0 | 38 |
| `[0.85, 1)` | 494 | 494 | 0 | 35 | 40 | 479 | 0 | 35 |
| `>= 1` | 298 | 298 | 0 | 31 | 39 | 298 | 0 | 31 |

`Lcap2` is exact on **every** set with `rho < 0.376` (3,349 of the 3,350 with `rho < 0.4`); the
first loose set is `{9,13,17,19,23,25}`, `F = 14`, core `{9,13}`, `rho = 2/9 + 2/13 = 0.376`,
`Lcap2 = 29`; the densest exact set is `{5,7,17}`, `rho = 0.686`. Above `rho = 0.7` the bound is
never exact and almost always runs past the scan limit `3m + 40`; at `rho >= 1` it is vacuous by
construction (the capacity criterion `L(1 - rho) <= 2t + rounding` holds at every `L`). Both
capacity bounds are density bounds, and their slack is of order `1/(1 - rho)`.

**What the slack is made of.** At `L = F + 1` the bound is loose (says coverable) on **2,577**
loaded sets. The counts level - actual union sizes, so overlaps seen - refuses **702** of them
(the overlap kind: `{5,7,9}`, `{5,7,11}`, `{5,7,9,11}`, ...); the remaining **1,875** are
refused only by the run structure `D` (the run-parity kind: `{5,7,9,17}`, `{5,7,9,19}`, ...).
The task's hypothesis "loose exactly when core dominoes overlap" is refuted: the run-parity kind
is the larger by 2.7 to 1.

---

## 4. Mechanism

**The top moment.** `M_k(d)` is an alternating sum over `S`; every `S` is a hitting set of the
pieces it meets, and the alternating sum over hitting sets of a family is `mu`, a signed count of
its covering subfamilies. At `k = r(d)` only minimum covers have covering subfamilies (themselves),
so each contributes `(-1)^r`; nothing else contributes; nothing can cancel. The count `C_r(d)` is
a product over the two parity classes because dominoes never cross parity: an even run tiles in
one way, an odd run of `2a + 1` cells has `a` interior double-covers and one cover per end
singleton, and the end singletons are exactly `{1}` and `{d-1}` (L74's `p = 1` and `p = d + 1`).

**The whole signature.** The same `mu` at every `k` says `c(z) = sum_T mu(T) (z-1)^{|T|} z^{4 + n - |T|}`
summed over subfamilies `T`, which telescopes to the cover polynomial. So the universal census is
a linear functional of the cover polynomial of the domino-plus-two-singletons family on `[1, d-1]`,
and every fact of `top_machine_2.md` sections 3.5-3.6 and `top_machine_7.md` 3.5 is a property of
that polynomial: `M_k = 0` below `r` because `(1-z)^t` vanishes to order `t` at `z = 1`; the
degree of `N_d` is `m - r` because `Delta^t` lowers degree by `t`; `d = 4` has no cover.

**The bijection.** In a wheel of `m = r(d)` gears above `d + 2`, a gap of length `d` at `n` forces
each gear's trace inside `[n+1, n+d-1]` to be one piece (L67: a domino or a boundary singleton,
the phases `p = 2` and `p = d` being forbidden because they strike `n` or `n + d`), the `m` pieces
must cover, so they are a minimum cover with a bijection to the gears, and CRT makes each such
assignment exactly one `n` mod `W`. The universal record multiplicity `r! C_r(d)` is therefore a
count of labelled tilings, not an inclusion-exclusion residue.

**The two classes as two windows.** Multiplication by `2^{-1}` mod `W_core` conjugates the teeth
`{0, -2}` to `{0, -1}`; the even cells `2i` of `[0, L)` become `i in [0, ceil(L/2))` and the odd
cells `2i + 1` become `i + 2^{-1}`, i.e. the window starting half a turn round the core wheel.
The record problem is the sum of the costs of a window and its half-turn partner, minimised over
one phase `x in Z_{W_core}`. The coupling is what defeats decoupling (3.5), and the alternation
of a gear's dominoes between the two adjacent-pair grids (consecutive dominoes are `g` apart, `g`
odd) is what makes the run parity cost a half-piece per interior run (3.5, the run-parity kind).

---

## 5. Laws

Numbered from **W95** in the project register. Setting as in `top_machine_7.md` section 4:
pairwise coprime odd gears `g >= 3`, teeth `0` and `-2`, `m = |G|`; pieces, covers, `C_t(d)`,
`r(d)` as in the vocabulary above; `P_G(e) = prod_{g in G} (g - e)`; `Delta` the forward
difference in `e`.

**W95 (THE NON-CANCELLATION, WITH THE SIGN - closes `top_machine_7.md` L75).** For every
`d >= 2`, in the universal regime,

        M_{r(d)}(d) = (-1)^{r(d)} r(d)! C_{r(d)}(d) ,

and `C_{r(d)}(d) >= 1` whenever `r(d)` is finite; hence `M_{r(d)}(d) != 0` for every `d != 4`.

*Proof.* Expand `f^k = (4 + sum_p u_p)^k` as a sum over sequences of length `k` from
`{4} u pieces`. A sequence whose set of distinct pieces is `P` contributes
`4^{#4s} sum_x (-1)^{|x|} prod_{p in P} u_p(x)`, and since
`prod_{p in P} u_p = sum_{T subset of P} (-1)^{|T|} prod_{j in J(T)} (1 - x_j)` (with
`J(T) = union of J_p, p in T`) and `sum_x (-1)^{|x|} prod_{j in V} (1 - x_j) = [V = [1, d-1]]`,
the inner sum is `mu(P) = sum over covering T subset of P of (-1)^{|T|}`. At `k = r(d)` a sequence
has at most `r` distinct pieces, so `mu(P) != 0` forces `P` to be a minimum cover with no `4` and
no repetition, and then `mu(P) = (-1)^r`; there are `r!` such sequences per minimum cover. QED

*Evidence.* `d = 2..26`, **0 mismatches of 25**; the 34 minimum covers of `d <= 14` each have
`mu = (-1)^r` by direct summation, **0 exceptions**.

**W96 (THE MINIMUM-COVER COUNT AND THE UNIVERSAL RECORD MULTIPLICITY IN CLOSED FORM - extends
`top_machine_1.md` L18 and `top_machine_2.md` L25).**

        C_{r(d)}(d) = d/4 - 1  (d = 0 mod 4),   1  (d = 1 mod 4),
                      (d+6)/4  (d = 2 mod 4),   ((d+1)/4)^2  (d = 3 mod 4) ,

so a wheel of exactly `r(d)` gears, all above `d + 2`, has exactly `r(d)! C_{r(d)}(d)` gaps of
length `d` per period.

*Proof.* Minimum covers factor over the two parity classes of `[1, d-1]`: with `n = d - 1`, the
odd class has `ceil(n/2)` cells and the end singletons `{1}` and, if `n` is odd, `{n}`; the even
class has `floor(n/2)` cells and the singleton `{n}` if `n` is even. A step-2 run of even length
`2a` is covered by `a` pieces only by its tiling (capacity `2a` with no waste); a run of odd
length `2a + 1` is covered by `a + 1` pieces either with exactly one doubly-covered cell - which
must be at an even offset from the run's start, `a` choices - or with one end singleton and the
tiling of the rest. Multiply, and read off the four residues of `d` mod 4. The multiplicity
statement is W95 with L25 (`N_d = (-1)^m M_m(d)` when `r(d) = m`), or directly by W99. QED

*Evidence.* counted against the closed form `d = 2..26`, **0 mismatches**; the eight published
multiplicities, **0 mismatches of 8**; the census by scan on 5 wheels and by L22 on a sixth,
**0 mismatches of 8**.

**W97 (THE UNIVERSAL SIGNATURE IS THE COVER POLYNOMIAL - extends `top_machine_2.md` L23 and
`top_machine_7.md` L73).**

        sum_e c_e(d) z^e = sum_t C_t(d) (1 - z)^t z^{d + 3 - t} .

*Proof.* `sum_S (-1)^{|S|} z^{e(S)} = z^4 sum_S (-1)^{|S|} prod_p (1 + (z - 1) u_p(S))
= z^4 sum_T (z-1)^{|T|} mu(T) = z^4 sum_{T' covering} (-1)^{|T'|} sum_{T superset of T'} (z-1)^{|T|}
= z^4 sum_{T'} (-1)^{|T'|} (z-1)^{|T'|} z^{(d-1) - |T'|}`, using that the pieces number `d - 1`
(for `d >= 3`; `d = 2` has two pieces and the identity holds by direct check). QED

*Corollaries, each in one line.* `M_k(d) = [(z d/dz)^k c](1)` vanishes for `k < r(d)` because
`(1-z)^t` vanishes to order `t >= r(d)` at `z = 1` (L73); at `k = r(d)` only the `t = r` term
survives and gives `(-1)^r r! C_r` (W95); `r(d) = D(d-1)` because the pieces are the dominoes of
`[1, d-1]` plus two singletons (L74); `d = 4` has `C_t = 0` for all `t` (L4); L24's gap-3 / gap-5
coincidence is `(z + (1 - z))^2 = 1`.

*Evidence.* every coefficient, `d = 2..26`, **0 mismatches**; `M_{r+1}` from the same expansion,
`d = 3..26`, **0 mismatches** (W100).

**W98 (THE UNIVERSAL CENSUS AS A COVER SUM, AND ITS COST - extends `top_machine_2.md` L22, L23,
L25).** For every gear set with all `g > d + 2`,

        N_d(G) = sum_{t = r(d)}^{m} C_t(d) (-1)^t (Delta^t P_G)(d + 3 - t) ,

a sum of `m - r(d) + 1` terms; `deg N_d = m - r(d)` (L25); and the cover counts `C_t(d)` are the
coefficients of a product of two path cover polynomials, computable by a two-state transfer
matrix in `O(d^2)`. Hence the universal signature `c_e(d)` and the universal census cost
polynomial time in `d`, not `2^{d-1}`.

*Proof.* Apply the functional `z^e -> P_G(e)` to W97; `sum_j binom(t,j) (-1)^j P(a + j) =
(-1)^t Delta^t P(a)`; `Delta^t P = 0` for `t > deg P = m`. The transfer matrix: a run of `ell`
cells is covered by a subfamily of its adjacent pairs and end singletons iff, scanning left to
right, every cell is met by the pair ending at it, the pair starting at it, or its singleton; the
state is "the pair ending here was taken". QED

*Evidence.* 27 universal `(d, G)` cases against L22 and scans, **0 mismatches**; transfer matrix
against brute force `d = 2..22`, **0 mismatches**; signatures tabulated to `d = 60`.

**W99 (THE BIJECTION BEHIND THE UNIVERSAL RECORD MULTIPLICITY - extends `top_machine_1.md`
L18).** In a wheel of `m = r(d)` gears all above `d + 2`, the map
`(minimum cover P, bijection G -> P) -> n mod W` given by CRT from "gear `g` has tooth 0 at
`n + p` for its piece `J_p`" is a bijection onto the gaps of length `d`. Hence
`N_d = r(d)! C_{r(d)}(d)` with no inclusion-exclusion.

*Proof.* A gap of length `d` at `n` needs `n, n + d` open and `n + 1, ..., n + d - 1` struck. For
`g > d + 2` the `d + 3` positions `n, ..., n + d + 2` are distinct residues, so gear `g` strikes
inside `[n+1, n+d-1]` exactly the piece `J_p` where `n + p = 0 mod g`, `p in [1, d+1]`, and the
open conditions exclude `p in {0, 2, d, d + 2}`; `m` pieces covering `d - 1` cells with
`m = r(d)` form a minimum cover, each gear taking one piece; conversely any such assignment is a
phase vector, realised by exactly one `n` mod `W`. QED

*Evidence.* the 8 scan/L22 cases of 3.3, **0 mismatches**.

**W100 (THE SECOND MOMENT - extends L25).** For `d >= 3`, `d != 4`, with `r = r(d)`,
`M_{r+1}(d) = (-1)^r (r+1)!/2 [C_r(d) (2d + 6 - r) - 2 C_{r+1}(d)]`. *Proof:* the sequences of
length `r + 1` in W95's expansion are: one `4` and a minimum cover (`(r+1)!` per cover, weight
`4`), a minimum cover with one piece repeated (`r (r+1)!/2` per cover), or `r + 1` distinct
pieces forming a cover (`(r+1)!` per cover, `mu = (-1)^{r+1} + (-1)^r #(minimum covers inside)`),
and `sum_P #(minimum covers inside P) = C_r (d - 1 - r)`. QED *Evidence.* `d = 3..26`,
**0 mismatches**.

**W101 (THE HALF-TURN LAW - the two parity classes are two windows of the adjacent-teeth core
wheel; extends `top_machine_7.md` L69).** Let `Q` be the struck set of the core gears with
teeth `{0, -1}` at phases `u_g = 2^{-1} a_g mod g` (`a_g` the pair-coordinate phase), a subset
of `Z_{W_core}`, and let `c_ell(x)` be the domino cost (pieces = adjacent pairs and singletons)
of the unstruck part of `[x, x + ell)` in `Q`. Then the even cells of `[0, L)` uncovered by the
core are `[0, ceil(L/2)) \ Q` and the odd cells are `[H, H + floor(L/2)) \ Q` with
`H = (W_core + 1)/2 = 2^{-1} mod W_core`, and

        F_top(G) = max { L : min over x in Z_{W_core} of
                         c_{ceil(L/2)}(x) + c_{floor(L/2)}(x + H) <= t(L) } .

*Proof.* `n = 0` or `-2 mod g` iff `2^{-1} n = 0` or `-1 mod g` (`g` odd); the even cell `2i`
maps to `i` and the odd cell `2i + 1` to `i + 2^{-1}`, and `2^{-1} = (W_core + 1)/2` in
`Z_{W_core}`, simultaneously `2^{-1}` mod every core gear by CRT. Dominoes `{c, c + 2}` become
adjacent pairs, so `D(U) = D(U_e) + D(U_o)` is the sum of the two window costs; then L69. QED

*Evidence.* 19,896 phase vectors of 200 loaded sets, **0 mismatches**; the record re-decided by
full enumeration on 6,659 sets, **0 mismatches**.

*Measured with it.* (i) the two windows do not minimise independently: 3,611 of 5,006 loaded
sets, all sets with four or more core gears; (ii) the run structure is not reducible to counts:
1,875 sets; (iii) anchoring at the ends fails on exactly the 210 sets `{7} u {four gears > 13}`
at `F = 12`, by the unique-phase mechanism of 3.5.

**W102 (THE PARITY-REFINED CAPACITY BOUND - extends `top_machine_7.md` L70).** With `A_g(L)` the
set of achievable `(#even cells, #odd cells)` of gear `g`'s trace in `[0, L)`,

        F_top(G) <= Lcap2(G) = max { L : min over (a_g, b_g) in A_g(L) of
            ceil((ceil(L/2) - sum a_g)^+ / 2) + ceil((floor(L/2) - sum b_g)^+ / 2) <= t(L) } ,

and `Lcap2 <= Lcap` (L70). Computable by a DP over the core with state `(sum a, sum b)`.

*Proof.* At any phase vector the even class has at least `ceil(L/2) - sum a_g` uncovered cells
and needs at least half that many tail pieces, likewise the odd class, and a tail piece serves one
class (L67); minimise over the splits the core can actually show. The comparison with L70:
`ceil(x/2) + ceil(y/2) >= (x + y)/2` and `a_g + b_g <= 2 ceil(L/g)`. QED

*Evidence.* 6,659 sets, **0 violations** of either inequality; exact on all 1,653 free sets and
on every set with core density `rho < 0.376` (3,184 with `rho < 0.3`, 3,349 of 3,350 with
`rho < 0.4`); never exact above `rho = 0.7`; vacuous at `rho >= 1`. L70 is exact on 1,349 of the
3,184 low-density sets; W102 on all of them.

---

## 6. The wheels' remaining open items - the ledger entry

R7's four genuinely open items, after this branch:

| R7 item | status now |
|---|---|
| 1. the cost of `min_U D_L(U)` | **reduced and reclassified.** Every structural ingredient is now proved: the cost is additive over two windows of one pattern (W101), the windows are locked by the half-turn, the decision is a scan of `W_core` (the tail's period drops out; the median gain on the family is 29,939x, and nothing when the tail is empty, which is the in-use regime). No polynomial algorithm is found or claimed, and the classes provably cannot be decoupled (3,611 refutations) - the coupling is the problem. What remains is a **complexity question**, not a structural one: the one-class Jacobsthal function is computed by exhaustive search in the literature too. Ledger class **(b)** |
| 2. `M_{r(d)}(d) != 0` | **closed**: W95, proved, with sign and count; W96 gives the count in closed form |
| 3. L22 in the kernel | the Formalist's; W97-W99 hand it a shorter route - `N_d` is a `Finset` sum over covers with a closed-form count, no `Finset.powerset` inclusion-exclusion needed in the universal regime |
| 4. a parity-refined capacity bound | **closed**: W102, proved, with its regime measured - exact below core density 0.376, a density bound above |

Items added by this branch: none structural. Two measurements with a mechanism attached: the
exactness threshold of W102 (why the first loose set is `{9,13}` at `rho = 0.376`) and the
anchoring exceptions (explained). One computational: the `W_core` scan.

**Plainly: on paper, the wheels now have no open structural item.** What is open is (i) kernel
coverage (the Formalist's lane; L67-L69 in progress, W95-W102 written in `Finset` shape), and
(ii) the complexity of the core minimisation, which is a question about algorithms.

*The capacity route and the in-use machine, in one line.* Both capacity bounds are density
bounds with slack of order `1/(1 - rho)`; in use, `rho = sum_{q < p <= Q} 2/p ~ 2 log(log Q / log q)`
exceeds 1 as soon as `Q > q^{1.65}`, so no capacity bound reaches the in-use record - the same
fact as document 4 L52 (the tail is empty in use) and the novel index's "moment-degree-ceiling"
entry, now with the threshold.

---

## 7. What is new

**The non-cancellation is a theorem, and its value is a count.** `M_{r(d)}(d) = (-1)^r r! C_r(d)`:
the signed sum over minimum covers has one sign because a minimum cover has exactly one covering
subfamily. The universal record multiplicity - 18, 96, 24, 24, 480, 6480, 1440, 720 - is `r!`
times the number of minimum domino tilings of two parity classes, and that number is
`d/4 - 1, 1, (d+6)/4, ((d+1)/4)^2` by `d` mod 4. L4 is the case `d/4 - 1 = 0`.

**The census signature is a cover polynomial.** `sum_e c_e(d) z^e = sum_t C_t(d) (1-z)^t z^{d+3-t}`.
Everything R7 and document 2 proved or measured about `c_e(d)` and `M_k(d)` is a property of
this polynomial, and it is computable in `O(d^2)`: the universal census is no longer a
`2^{d-1}` object. R7's ledger item "the census beyond `d = 20`" is closed for the universal
regime (a wheel with a gear `<= d + 2` still needs L22's full sum).

**The record is a two-window problem on one wheel.** The parity classes of `[0, L)` are the
windows `[0, ceil(L/2))` and `[H, H + floor(L/2))` of the adjacent-teeth core wheel, `H` half a
turn. The classes are coupled through `H` and do not minimise separately (3,611 of 5,006); the
record cover is unique up to reflection on 2,965 loaded sets and on 19 sets it is 2 phase vectors
in millions.

**The refined capacity bound and its regime.** Exact below core density 0.376, worthless above
0.7, vacuous at 1: a capacity bound is a density bound, and the in-use machine sits at density
above 1.

**Prior art, in a line.** The `mu`-duality between hitting sets and covers is the Möbius
inversion on the Boolean lattice (standard); the edge-cover polynomial of a path is a Fibonacci-type
transfer matrix (standard); `2^{-1}` conjugating `{0,-2}` to `{0,-1}` is CRT (standard; document
1 L19 does the same with `6^{-1}`). What is new is the identification of the census signature
with the cover polynomial of *this* family, the closed form of the record multiplicity, the
half-turn form of the record, and the density regime of the capacity bound. Nothing asymptotic
is used.

---

## 8. Verdict

**L75 is proved.** The top moment does not cancel because every minimum cover carries the sign
`(-1)^{r(d)}`; its value is `(-1)^r r! C_r(d)` with `C_r(d)` in closed form, verified at every
`d = 2..26` and reproducing all eight published multiplicities. With L73 and L74 this completes
L25: the degree law and the universal record multiplicity are theorems.

**The whole universal counting theory collapses to one polynomial.** The signature is the cover
polynomial of the domino family; the census is its image under `z^e -> prod(g - e)`; both are
polynomial-time.

**The minimisation has its structure.** Two windows of one adjacent-teeth wheel, half a turn
apart, coupled; a scan of the core period decides it; the classes cannot be separated, the run
parity cannot be dropped, the ends cannot be assumed struck (210 exceptions, one mechanism). No
polynomial algorithm; the item is computational.

**The refined bound is proved and its regime is measured.** Exact on every low-density set,
a density bound above.

**The ledger.** Two of R7's four items closed, one reduced to a complexity question, one the
Formalist's. The wheels have no open structural item on paper. No clutch, no motor, nothing
against the twin conjecture.

---

## 9. Scorecard, filled

| # | Prediction | Result |
|---|---|---|
| P1 | `M_{r(d)}(d) = (-1)^r r! C_r(d)`, `d = 2..26` incl. `d = 4` | **held**, 0 of 25 |
| P2 | `C_r(d)` closed form; the eight multiplicities | **held**, 0 mismatches; 8 of 8 |
| P3 | every minimum cover has `mu = (-1)^r` | **held**, 34 covers, 0 exceptions |
| P4 | cover-polynomial identity; transfer matrix = brute force | **held**, 0 mismatches on every coefficient `d <= 26`; 0 on `d <= 22` |
| P5 | universal `N_d` as a cover sum | **held**, 27 cases, 0 mismatches |
| P6 | bijection count by scan and L22 | **held**, 8 of 8 |
| P7 | `M_{r+1}(d)` formula | **held**, `d = 3..26`, 0 mismatches |
| P8 | polynomial-time signature, tables to 60 | **held** |
| P9 | half-turn identity | **held**, 19,896 vectors, 0 mismatches |
| P10 | decoupling fails on >= 100 loaded sets | **held**, 3,611 |
| P11 | run parity matters on >= 1 set | **held**, 1,875 |
| P12 | anchoring has exceptions | **held**, 210, all one mechanism |
| P13 | medians in range; enumeration = `F_rule` | **held**: median 4 distinct values, 5.71% at the minimum; 0 mismatches of 6,659 |
| P14 | cost as a `W_core` scan | **held**: largest `W_core` 5,766,215 (tail empty there) |
| P15 | `Lcap2` proved, 0 violations, `F <= Lcap2 <= Lcap` | **held** |
| P16 | exact on all free, majority of loaded; max slack <= 4; hand values 9 and 20 | **half refuted**: all free exact; loaded 2,413 of 5,006 (48%, not a majority - exact on all with `rho < 0.376`, never above 0.7); "max slack 4" badly wrong (the bound is a density bound and goes vacuous at `rho >= 1`); hand values were 10 and 22, not 9 and 20 |
| P17 | slack splits into overlap and run-parity kinds, each >= 5 | **held**, 702 and 1,875 - and the run-parity kind dominates |
| P18 | ledger: 2 closed, 1 reduced, 1 the Formalist's; none added | **held** |

---

## 10. Holds without exception (the count)

| statement | count | exceptions |
|---|---|---|
| `M_{r(d)}(d) = (-1)^r r! C_r(d)` | `d = 2..26` | **0** |
| `C_r(d)` closed form | `d = 2..26` | **0** |
| `mu(P) = (-1)^r` per minimum cover | 34 covers | **0** |
| the cover-polynomial identity, every coefficient | `d = 2..26` | **0** |
| transfer matrix = brute-force cover counts | `d = 2..22` | **0** |
| `N_d` cover sum = L22 = scan (universal) | 27 cases | **0** |
| `N_d = r! C_r` where `r(d) = m` | 8 cases | **0** |
| `M_{r+1}(d)` formula | `d = 3..26` | **0** |
| the half-turn identity | 19,896 phase vectors | **0** |
| full enumeration re-decides the record | 6,659 sets | **0** |
| `F <= Lcap2 <= Lcap` | 6,659 sets | **0** |
| `Lcap2 = F` at core density `< 0.376` | 3,349 sets | **0** |
| decoupling fails at four or more core gears | 989 sets | **0** |
| anchoring exceptions are `{7} + four gears > 13`, `F = 12` | 210 | **0** |

---

## 11. Dead ends

- **Decoupling the parity classes.** `min_x [c_e(x) + c_o(x + H)] = min c_e + min c_o` is false on
  3,611 of 5,006 loaded sets, already at `{5,7}`, `L = 5`. The half-turn couples the windows; the
  one-window function `m(ell) = min_x c_ell(x)` does not decide the record.
- **Dropping the runs.** The counts-only criterion is wrong on 1,875 sets; the run parity is a
  half-piece per interior run and cannot be priced by cell counts.
- **"Each core gear's dominoes align with a fixed parity pattern."** Impossible as stated:
  consecutive dominoes of one gear are `g` apart in half-index, `g` odd, so they alternate grids.
  What is true is a pattern of the uncovered set at the optimum (3.5), not of any gear.
- **Anchoring a core gear at cell 0 as a WLOG.** 210 exceptions; the mechanism is a unique optimal
  phase with both dominoes interior.
- **"Loose exactly when the core dominoes overlap."** Refuted 1,875 to 702 the other way.
- **A capacity bound with small slack everywhere.** Both bounds are density bounds; slack of
  order `1/(1 - rho)`, vacuous at `rho >= 1`. The pre-registered "max slack 4" and the two hand
  values (9, 20; actual 10, 22) were wrong.
- **A polynomial algorithm for the minimisation.** Not found; the structural reductions (W101) are
  exact, and the remaining question is algorithmic. Not re-entered.
