# The spectrum's depletion as a sum rule (branch, prover, 2026-09-05)

Parent: node **R3.i** (the half-column map, `research/proof/half_column.md`), which died as a route
and left one live idea, listed as item 2 of the "reading as a whole" in
`research/proof/dead_branches_reopened_2.md`: **the depletion as a sum rule**. What spawned this
branch is the pair of exact facts R3.i ended on. (i) For `v < y^2/3`, `v` is uncoupled in
`M = {5..y}` iff `v` is `y`-rough and its half-column `v/2` is a twin column above `y`
(5,505 of 5,505 cells, H6). (ii) Those uncoupled sizes are the depleted sizes of the gap spectrum:
12-128 times rarer than their coupled neighbours, 10 of 10 cells, with the whole effect in the step
from zero coupling gears to one.

Put together they read: **the machine's gap spectrum is depleted exactly at twice the twin columns
above the machine.** If twins were finite then for `y > Y` no even `y`-rough size below `y^2/3`
would be depleted. So the question of this branch is whether the machine's own structure FORCES
depleted sizes to exist, through an exact identity on the multiplicity function
`m_M(v) = ` (number of gaps of size `v` per period of `M`). If some identity can only be satisfied
with depleted sizes present, the identity forces twin columns above every `y` -- a self-referential
route with a finite computation at its base.

What this branch can find that is not already known: the exact recursion the multiplicity function
obeys when a gear is added (the merge law gives the shape, not the coefficient); whether the
depletion is a coefficient effect or a merge effect; whether an uncoupled size can be absent rather
than merely depleted, and the exact rule; and whether the identities on `m` are strong enough to be
infeasible when the depleted sizes are filled in.

Scripts: `research/anchor235/r53/sr_*.py`. Result outputs (untracked):
`research/anchor235/r53/results/`. Every number this document relies on is written into the
document.

---

## 0. Pre-registered (written before any computation of this branch)

### 0.1 Definitions fixed here

- `M = {5..y}` is the machine, `P = prod_{q in M} q` its period, `O` its openings, `N = |O| =
  prod (q - 2)`. Gaps are the differences between cyclically consecutive openings; there are `N`
  of them per period. `m_M(v)` is the number of gaps of size `v` per period ("the multiplicity
  function"); `Spec(M) = {v : m_M(v) > 0}`; `F(M) = max Spec(M)`.
- Teeth of gear `q`: `T_q = {u_q, -u_q}` with `u_q = 6^{-1} mod q`; letters `a_q = 2u_q`,
  `b_q = q - 2u_q`, `d_q = 2u_q` as a residue (docs/proofs/02, 05).
- **The autocorrelation count** `A_M(v) := #{k mod P : k and k + v both open}`, and its local
  factor `c_q(v) := #{r mod q : r and r + v both open for q} = q - |T_q u (T_q - v)|`.
- **Coupled / uncoupled** as in `half_column.md`: `v` is coupled in `M` iff some `q in M` has
  `q | v` or `v = +- d_q (mod q)`; otherwise uncoupled.
- Adding a gear `q'`: `M' = M + q'`, `P' = P q'`, `q'` copies of `M`'s period, deletion phase
  `r_j` per copy, each opening of `M` struck in exactly two copies (docs/proofs/05 (A)).
- A **`J`-run** of `M` is `J >= 1` consecutive gaps of `M`, i.e. openings
  `x_0 < x_1 < ... < x_J` consecutive in `M`; its **ends** are `x_0, x_J` and its **interiors**
  are `x_1, ..., x_{J-1}`; its **span** is `x_J - x_0`.

### 0.2 Theory

**T. The multiplicity function obeys an exact linear-plus-merge recursion whose linear coefficient
is the autocorrelation local factor `c_{q'}(v)`, and the depletion of the uncoupled sizes is
carried by the merge term, not by the coefficient.** An old gap of size `v` survives as a gap of
`M'` in exactly those copies where NEITHER of its two end openings is struck by `q'`; each end is
struck in exactly two copies, and the two pairs of copies overlap by 2, 1 or 0 according to the
chain law (`v = 0`, `v = +- d_{q'}`, or neither, mod `q'`). So the survival count is
`q' - 4 + overlap = c_{q'}(v)`, the same three-valued function that appears in the open-pair count.
Hence `m_{M'}(v) = c_{q'}(v) m_M(v) + Merge_{q'}(v)`. Being uncoupled costs exactly the smallest
coefficient, `q' - 4`; the ratio to a coupled neighbour's coefficient is at most `(q'-2)/(q'-4)`,
which is `<= 5/3` for `q' >= 7` -- far too small to produce a factor of 12-128, so the depletion
must live in `Merge`.

### 0.3 Predictions, each with the number that would refute it

- **P1 (the elementary identities).** For every machine m5..m23: (a) `sum_v m(v) = prod (q - 2)`;
  (b) `sum_v v m(v) = prod q`; (c) `m(v)` is even for every `v >= 2` and `m(1)` is odd;
  (d) `m(1) = prod_{q in M} (q - 4)` exactly; (e) `m(2) = 2 . 4 . prod_{q >= 11} (q - 4)`, i.e.
  `m(2) = (8/3) m(1)` for every machine containing 5 and 7. REFUTED by one machine.
- **P2 (the survival coefficient).** At every rung `5 -> 7, ..., 23 -> 29` and every realised size
  `v` of `M`, the number of copies in which a given old gap of size `v` survives whole is exactly
  `c_{q'}(v) = q' - 2` if `q' | v`, `q' - 3` if `v = +- d_{q'} (mod q')`, `q' - 4` otherwise.
  REFUTED by one (rung, gap) instance measured against the direct sieve.
- **P3 (the exact recursion).** `m_{M+q'}(v) = c_{q'}(v) m_M(v) + Merge_{q'}(v)`, with
  `Merge_{q'}(v) = sum over J >= 2 runs of span v of w(R)`, `w(R)` = the number of copies in which
  every interior of `R` is struck and neither end is. Reproduces the directly sieved spectrum of
  `M'` at rungs `5->7 .. 23->29`, **0 error on every size**. REFUTED by one nonzero residual.
- **P4 (the autocorrelation sum rule).** `A_M(v) = prod_q c_q(v)` for every `v`, and
  `A_M(v) = sum_{j >= 1} W_j(v)` with `W_1 = m`. Pre-registered as an INSTRUMENT CHECK on a
  recorded result (`depth-sum-identity` in docs/novel/README.md = Holt arXiv:2502.20470 Cor. 1);
  it is verified in one line and cited, not claimed.
- **P5 (monotonicity: sizes are never lost).** `m_{M'}(v) >= (q' - 4) m_M(v)`, so once a size is
  realised it is realised at every larger machine, and its multiplicity is multiplied by at least
  `q' - 4` per rung. Consequently a size absent at `M` is absent at every sub-machine of `M`.
  REFUTED by one size whose multiplicity falls, m5..m31.
- **P6 (the depletion is a merge effect, not a coefficient effect).** At the rungs `23 -> 29` and
  `29 -> 31`, for the uncoupled sizes (24, 36, 41) the merge term supplies MORE than half of
  `m_{M'}(v)`, while the coefficient deficit `(q'-3)/(q'-4)` accounts for at most 8%. REFUTED if
  the survival term supplies more than half at either rung.
- **P7 (absent versus depleted).** An uncoupled size `v <= F(M)` is absent iff no `J`-run of `M` of
  span `v` has a legal strike pattern; the first appearance of every such size is a pure merge
  event with `m_M(v) = Merge` and `m_{M^-}(v) = 0`. Predicted: 4 of 4 recorded arithmetic holes
  (4 at m7; 24 at m19 and m23; 41 at m29) appear this way, and no size ever disappears again.
  REFUTED by one size that reappears as zero.
- **P8 (the counterfactual is FEASIBLE, i.e. the sum-rule route dies).** Take the measured spectrum
  at m19 and m23; impose (a), (b), (c), (d), (e), non-negativity, and the counterfactual
  "no depleted sizes": every even `y`-rough `v < y^2/3` with `v <= F` has
  `m(v) >= median{m(w) : |w - v| <= 4}`. Predicted **FEASIBLE**, with an explicit witness, because
  the uncoupled sizes carry under `10^-4` of both the count budget and the length budget.
  REFUTED (and the branch would then be a route) if the LP is infeasible; the identity that forces
  it would then be named.
- **P9 (the life cycle of a twin column).** For a twin column `c` the size `2c` is uncoupled at
  every machine below `6c - 1`; at the rung `q' = 6c - 1` the coefficient jumps from `q' - 4` to
  `q' - 3` because `2c = a_{q'} = d_{q'}` exactly (the fibre theorem H5). Predicted exact at the
  two observable columns (`c = 2` at rung 11, `c = 3` at rung 17), and predicted NOT to be the
  cause of the flood: the step ratio at the flip is dominated by merges.
- **P10 (the residual, stated before it is measured).** "There is an uncoupled even size below
  `y^2/3`" is EQUIVALENT to "there is a column `c` with `y/6 < c <= (y^2 - 1)/6`, `c` having no
  prime factor in `(3, y]`, and `(6c - 1, 6c + 1)` both prime". This is strictly stronger than the
  root (it adds the roughness of `c`), so forcing depletion proves more than the conjecture needs.
  REFUTED if some uncoupled even size below `y^2/3` does not carry a twin column.

Stop rules: any sub-question that reduces to the merge law (docs/proofs/05 (D)-(F)), the chain law
(05 (C)), the parity theorem (03 (e)), the depth-sum identity (`depth-sum-identity`, Holt Cor. 1)
or `Leg_real` / H6 (half_column.md) is stopped in one line and cited.

The scorecard for P1-P10 is filled in section 7.

---

## 1. Setup (exact ranges)

Everything is exact: full periods, integer arithmetic, no sampling anywhere.

| object | range | method |
|---|---|---|
| the identities (a)-(f) | m5, m7, m11, m13, m17, m19, m23, sieved whole (period 37,182,145 at m23) | `sr_identities.py` |
| the recursion, verified against a direct sieve | rungs `5->7, 7->11, 11->13, 13->17, 17->19, 19->23` | `sr_recursion.py` |
| the recursion, extended past the sieve | rungs `19->23`, `23->29` (m23 openings held whole), `29->31` (the m29 period of 1,078,282,205 columns streamed in 40M-column chunks with a 40-opening carry) | `sr_recursion.py`, `sr_rung31.py` |
| holes: span census versus phase census | every size absent below `F(M')` at rungs `7->11 .. 23->29`, every run of that span enumerated over the full old period, every phase tested | `sr_holes.py` |
| the feasible interval of `m(v)` under the identities | m19, m23, m29, m31; exact LP (HiGHS) | `sr_lp.py` |
| the residual after the pair count | m19, m23, m29, m31, every size `3..F` | `sr_residual.py` |
| the first uncoupled size | every prime rung `y = 5..19997` (2,260 rungs), sizes `2..2,000,000` | `sr_firstunc.py` |

**Instrument gate.** The recursion is run forward from `{5}` and its output compared with the
directly sieved spectrum at every rung it can reach: **0 error on all 94 sizes** at the six rungs
`5->7 .. 19->23`. Past the sieve it is gated against the corpus: at m29 it returns `F = 43`,
`|Spec| = 41`, absent `{41, 42}`, `m(4) = 14,178,528`, `m(6) = 10,497,320`, `m(24) = 1,180`,
`m(36) = 38`, `sum m = 214,708,725 = prod (q-2)`, `sum v m = 1,078,282,205 = P`; at m31 it returns
`F = 58`, `|Spec| = 55`, absent `{54, 56, 57}`, `m(4) = 398,923,200`, `m(6) = 299,202,120`,
`m(24) = 174,704`, `m(36) = 3,152`, `m(41) = 134`, `sum m = 6,226,553,025`,
`sum v m = 33,426,748,355`. Every one matches `half_column.md` 2.3-2.4 and the corpus. The m31
spectrum takes 30 seconds this way, against a chunked sieve of 33 billion columns.

## 2. Results

### 2.1 The identities (item 1)

All hold with **0 exceptions** at every machine m5..m23, and (a), (b) again at m29 and m31 through
the recursion.

| identity | statement | values m7..m23 |
|---|---|---|
| (a) | `sum_v m(v) = prod (q-2)` | 15; 135; 1,485; 22,275; 378,675; 7,952,175 |
| (b) | `sum_v v m(v) = P` | 35; 385; 5,005; 85,085; 1,616,615; 37,182,145 |
| (c) | `m(v)` even for `v >= 2`, `m(1)` odd | 0 odd-count sizes `>= 2` anywhere (docs/proofs/03 (e), cited) |
| (d) | `m(1) = prod (q-4)` | 3; 21; 189; 2,457; 36,855; 700,245 |
| (e) | `m(2) = 2 . 4 . prod_{q >= 11}(q-4) = (8/3) m(1)` | 8; 56; 504; 6,552; 98,280; 1,867,320 |
| (f) | `A(v) = prod_q c_q(v)` for every `v <= F` | 102 cells, 0 exceptions |

Here `c_q(v) := #{r mod q : r and r+v both open for q} = q - |T_q u (T_q - v)|`, which is `q-2` if
`q | v`, `q-3` if `v = +-d_q (mod q)`, and `q-4` otherwise -- the chain law read as a count -- and
`A(v)` is the number of open pairs at distance `v` per period.

Three consequences worth naming, because the branch turns on them.

- **`A(v) >= prod_q (q - 4)` for every `v`, with equality if and only if `v` is uncoupled in `M`.**
  Immediate from `c_q(v) in {q-4, q-3, q-2}`. The uncoupled sizes are exactly the distances with
  the SMALLEST possible number of open pairs, and that number is `prod (q-4) = m(1)`.
- **`v = 1` is uncoupled in every machine** (`1 = +-d_q (mod q)` would force `q | 4` or `q | 8`),
  and `m(1) = A(1) = prod (q-4)`: **every open pair at distance 1 is a gap.** The smallest
  uncoupled size is the one size whose multiplicity attains its own cap exactly.
- **`m(2) = A(2)`** as well, because gear 5 alone forbids three consecutive openings, so no pair
  at distance 2 has an opening between it. Hence `rho(1) = rho(2) = 1` exactly at every machine,
  writing `rho(v) := m(v)/A(v)`.

*(Prior art, one line, then stopped: identity (f) is the recorded `depth-sum-identity`, i.e. Holt
arXiv:2502.20470 Corollary 1 at the constellation `s = (2, 6g-2, 2)`; used here as an instrument,
not claimed.)*

### 2.2 The recursion (item 1d), and where it is already on record

Derivation. Let `M' = M + q'`, with `q'` copies of `M`'s period and the phase bijection of
docs/proofs/05 (A). A run of `J >= 1` consecutive gaps of `M`, from opening `x_0` to `x_J`, is a
single gap of `M'` in copy `r` iff every interior `x_1..x_{J-1}` is struck by `q'` in that copy
and neither end is. Writing `w(R)` for the number of such copies,
`m_{M+q'}(v) = sum over runs R of span v of w(R)`.

For `J = 1` there are no interiors, so `w` counts the phases avoiding both ends: the bad set is
`{x_0, x_0 - d, x_1, x_1 - d}`, of size 2 if `v = 0 (mod q')`, 3 if `v = +-d (mod q')`, and 4
otherwise -- the chain law. Hence

> **`m_{M+q'}(v) = c_{q'}(v) . m_M(v) + Merge_{q'}(v)`**, the survival coefficient being exactly
> the autocorrelation local factor, and `Merge >= 0` the weighted census of `J >= 2` runs.

For `J >= 2` the interiors must lie in `{r, r+d}` for a common `r`, so there are at most two
candidate phases (two when all interiors share one residue, one when both residues occur, none
otherwise), each then tested against the two ends: `w(R) in {0, 1, 2}`.

**Verified: `survival_{M'}(v) = c_{q'}(v) m_M(v)` at 137 of 137 (rung, size) cells,
`5->7 .. 29->31`, 0 exceptions; and the full recursion reproduces the sieved spectrum at 94 of 94
sizes over the six sieve-checkable rungs.**

**PRIOR ART, and the sub-question is stopped here.** This is the recorded `paired-holt-recursion`
(docs/novel/paired-holt-recursion.md, SCRIPT-VERIFIED, prior-art checked 2026-08-23):
`n_g(M+q') = sum_{w : sum w = g} coef(w) n_w(M)` with a position-free coefficient, whose `j = 1`
case is written there in the same three cases and identified there with the round-19
autocorrelation `c_{q'}(g)`. Nothing in the derivation above is new. What this branch adds is
arithmetic, not law: the recursion is carried three rungs past its recorded verification (which
reached the word census at `13->17` and `17->19`), to `19->23`, `23->29` and `29->31`, and is used
as the *generator* of the m29 and m31 spectra, with the survival term separated from the merge
term size by size.

Two corollaries the split gives immediately, both one line:

- **No size is ever lost.** `m_{M'}(v) >= (q'-4) m_M(v) >= m_M(v)`, so `Spec(M) subset Spec(M')`
  and a multiplicity is multiplied by at least `q'-4` per rung. **137 of 137 cells, 0 exceptions**;
  `Spec` inclusion 0 exceptions over m5..m31.
- **Every size is born a merge.** Its first appearance has `survival = 0` by definition, so the
  spectrum is built entirely by the merge term and merely propagated by the coefficient: 55 of 55
  sizes at m31.

### 2.3 Survival against merge, and what the depletion is NOT (item 2)

The split at the two deepest rungs, at the uncoupled sizes and at controls of comparable rarity:

| rung | `v` | coupling | `m_M(v) -> m_M'(v)` | ratio | `c_{q'}(v)` | survival share |
|---|---|---|---|---|---|---|
| 23->29 | 24 | none | 0 -> 1,180 | -- | 25 | 0.0% |
| 23->29 | 36 | none | 0 -> 38 | -- | 25 | 0.0% |
| 23->29 | 23 | 5,7,17 letters, 23 pad | 5,598 -> 278,558 | 49.8 | 25 | 50.2% |
| 23->29 | 25 | 5 pad, 19 letter | 1,404 -> 88,548 | 63.1 | 25 | 39.6% |
| 29->31 | 24 | none | 1,180 -> 174,704 | 148.1 | 27 | **18.2%** |
| 29->31 | 36 | none | 38 -> 3,152 | 82.9 | 27 | **32.6%** |
| 29->31 | 23 | 5,7,17 letters, 23 pad | 278,558 -> 12,709,164 | 45.6 | 27 | 59.2% |
| 29->31 | 25 | 5 pad, 19 letter | 88,548 -> 4,937,476 | 55.8 | 27 | 48.4% |
| 29->31 | 35 | 5,7 pads, 13 letter | 442 -> 70,782 | 160.1 | 27 | **16.9%** |
| 29->31 | 37 | 5,7,11 letters | 84 -> 26,366 | 313.9 | 27 | **8.6%** |

P6 is confirmed in the letter (merge supplies 81.8% at `v = 24` and 67.4% at `v = 36`) **and
refuted in its intent**: the coupled controls 35 and 37 have merge shares of 83.1% and 91.4%.
Merge dominance is a function of how rare and how large a size is, not of coupling. The
coefficient is the same `q' - 4 = 27` for 24, 35, 36 and 37 at this rung, none of them being a
letter or a pad of 31.

Merge mass by depth at `29 -> 31`: `J = 2`: 413,380,422; `J = 3`: 7,999,018; `J = 4`: 12,992;
`J = 5`: 4. Depth-2 merges carry 98.1% of everything the rung creates.

### 2.4 Absent versus depleted: every hole is a phase hole (item 2)

For every size absent below `F(M')`, the runs of the old machine of that span were counted over
the whole period and every phase of every such run was tested.

| rung | absent `v` | runs of that span in `M` | total weight | kind |
|---|---|---|---|---|
| 11->13 | 9 | 28 | 0 | phase hole |
| 13->17 | 17 | 420 | 0 | phase hole |
| 17->19 | 19 | 3,276 | 0 | phase hole |
| 17->19 | 24 | 2,457 | 0 | phase hole |
| 19->23 | 24 | 36,855 | 0 | phase hole |
| 23->29 | 41 | 700,245 | 0 | phase hole |
| 23->29 | 42 | 2,334,150 | 0 | phase hole |

**7 of 7 holes are phase holes; there is never a span hole.** And the run census is not a new
number: 28, 420, 3,276, 2,457, 36,855, 700,245, 2,334,150 are exactly `A_M(v) = prod c_q(v)` in
each case -- the sum rule (f) again, since a run of span `v` is a pair of openings at distance `v`.
So the characterisation asked for is:

> A size `v <= F(M')` is ABSENT at `M'` iff all `A_M(v)` open pairs of `M` at distance `v` fail the
> phase test; it is merely DEPLETED when some pass. The uncoupled sizes are the ones with the
> fewest pairs to try -- `A_M(v) = prod (q-4)`, the floor -- which is why the arithmetic holes
> recorded at m7, m19, m23 and m29 are uncoupled ones.

Two of the seven (41 and 42 at m29) are the sizes just below `F`; 24 at m19 and m23 is the deep
one, and it is the uncoupled one. That is the split of the holes into arithmetic and capacity
classes recorded in `half_column.md`, now with the mechanism attached: both classes are phase
holes, and what distinguishes the arithmetic class is the SIZE of the pool, not its emptiness.

### 2.5 What the depletion really is: the pair count explains most of it

`half_column.md` measured the depletion on the raw multiplicities, `r(v) = m(v) / median{m(w)}`
over coupled neighbours within 4, and got factors of 12 to 128. But `m(v) <= A(v)`, and `A(v)` is
smallest exactly at the uncoupled sizes, so part of that factor is an identity. The fair quantity
is `rho(v) = m(v)/A(v)`, the fraction of open pairs at distance `v` that are actually adjacent.
Residuals below are against a local log-linear fit of `rho` over the coupled sizes within 6. (The
raw factors below are recomputed here with the same neighbour window as `half_column.md` but
excluding `v` itself from the median, which is why 101.3 stands where the record has 128; the
comparison in each row is between two numbers computed the same way.)

| machine | `v` | `m(v)` | `A(v)` | raw depletion factor | residual factor after `A` | percentile among coupled |
|---|---|---|---|---|---|---|
| m29 | 24 | 1,180 | 17,506,125 | **101.3** | **22.5** | 2.6 |
| m29 | 36 | 38 | 17,506,125 | **6.9** | **1.3** | 23.7 |
| m29 | 41 | 0 | 17,506,125 | inf | absent | 2.6 |
| m31 | 24 | 174,704 | 472,665,375 | **36.3** | **8.9** | 7.4 |
| m31 | 36 | 3,152 | 472,665,375 | **12.5** | **2.9** | 14.8 |

The coupled control's own leave-one-out residual has geometric mean 1.09 and 0.96 and geometric
standard deviation 1.92 and 1.97 at m29 and m31, with ranges `[0.18, 5.15]` and `[0.00, 2.31]`.
Detrending `log rho` globally instead (a quadratic in `v` fitted over all coupled sizes) gives the
same picture: at m31 the uncoupled residuals are 0.165 (`v = 24`) and 0.278 (`v = 36`) against
coupled medians of 0.86 to 1.41 by coupling count and a coupled minimum of 0.092; at m29 they are
0.044 and 0.648 against a coupled minimum of 0.181.

> **The exact pair count `A(v)` accounts for a factor of 4 to 5 of the measured depletion at every
> cell (101.3 -> 22.5; 6.9 -> 1.3; 36.3 -> 8.9; 12.5 -> 2.9). Of the four uncoupled cells with
> `m > 0`, three fall inside the ordinary scatter of the coupled sizes after that division; only
> `v = 24` at m29 remains an outlier.**

And the sharpest single check that uncoupledness by itself does not depress a multiplicity:
`v = 1` is uncoupled at every machine and has `rho(1) = 1` exactly, the largest value any size can
have. Uncoupledness shrinks the POOL of candidate pairs, by `prod (q-3)/(q-4)` or
`prod (q-2)/(q-4)` over the coupling gears; it does not by itself make a pair less likely to be
adjacent.

### 2.6 The counterfactual: what the identities pin down (item 3)

Instead of testing one filled-in spectrum, the exact feasible interval of `m(v)` was computed over
the polytope cut out by every identity available: (a), (b), (d), (e), and `0 <= m(v) <= A(v)` at
every size. (Parity (c) is a congruence, not a face; it moves an endpoint by at most 1.)

| machine | size | LP interval for `m(v)` | cap `A(v)` | measured |
|---|---|---|---|---|
| m19 | 24 (uncoupled) | `[0, 18,933]` | 36,855 | 0 |
| m19 | 25 (coupled) | `[0, 17,986]` | 117,936 | 20 |
| m23 | 24 (uncoupled) | `[0, 500,757]` | 700,245 | 0 |
| m23 | 25 (coupled) | `[0, 475,720]` | 2,240,784 | 1,404 |
| m29 | 24 (uncoupled) | `[0, 15,767,400]` | 17,506,125 | 1,180 |
| m29 | 36 (uncoupled) | `[0, 9,460,460]` | 17,506,125 | 38 |
| m29 | 41 (uncoupled) | `[0, 8,108,960]` | 17,506,125 | 0 |
| m29 | 25 (coupled) | `[0, 14,937,600]` | 56,019,600 | 88,548 |
| m31 | 24 (uncoupled) | `[0, 472,665,375]` | 472,665,375 | 174,704 |
| m31 | 36 (uncoupled) | `[0, 306,390,000]` | 472,665,375 | 3,152 |

**The identities pin down nothing at an uncoupled size.** Zero is feasible at every one of them,
and so is a value four orders of magnitude above the truth; the interval for a coupled control of
the same size has the same shape. The arithmetic reason is visible in the budgets: the uncoupled
sizes carry `5.7 x 10^-6` of the count budget and `2.8 x 10^-5` of the length budget at m29
(`2.9 x 10^-5` and `1.3 x 10^-4` at m31), so (a) and (b) cannot see them at all, while (d) and (e)
are statements about `v = 1` and `v = 2` alone.

The structural reason is stronger and does not depend on the machine: **every identity in which
coupling appears at all does so through `c_q(v)`, and `c_q(v)` is MINIMAL for uncoupled `v`.** So
the identity family is monotone in the wrong direction. It can certify that an uncoupled size is
rare and it can cap it; no combination of upper caps and two global sums can force a size to be
present. P8 is confirmed and the sum-rule route is dead.

### 2.7 The life cycle of a twin column (item 4)

`v = 2c` for a twin column `c` is uncoupled at every machine below `6c - 1`, and at the rung
`q' = 6c - 1` it becomes the short letter of the arriving gear (`2c = a_{q'} = d_{q'}`, the fibre
theorem H5), so its coefficient rises from `q' - 4` to `q' - 3`. All observable cases:

| `v` | m5 | m7 | m11 | m13 | m17 | m19 | m23 | m29 | m31 |
|---|---|---|---|---|---|---|---|---|---|
| 4 (`c = 2`, flip at 11) | >F | **0** | 6 | 96 | 1,536 | 26,208 | 539,136 | 14,178,528 | 398,923,200 |
| `r(4)` | -- | 0 | 0.273 | 0.403 | 0.604 | 0.592 | 0.579 | 0.569 | 0.559 |
| 6 (`c = 3`, flip at 17) | >F | >F | 4 | 60 | 1,022 | 18,776 | 393,464 | 10,497,320 | 299,202,120 |
| `r(6)` | -- | -- | 0.182 | 0.667 | 0.629 | 0.615 | 0.587 | 0.570 | 0.556 |
| 24 (`c = 12`, flip at 71) | >F | >F | >F | >F | >F | **0** | **0** | 1,180 | 174,704 |
| `r(24)` | -- | -- | -- | -- | -- | 0 | 0 | 0.0099 | 0.0275 |
| 36 (`c = 18`, flip at 107) | >F | >F | >F | >F | >F | >F | >F | 38 | 3,152 |
| `r(36)` | -- | -- | -- | -- | -- | -- | -- | 0.144 | 0.080 |
| 41 (`= a_31 + 31`, wrapped) | >F | >F | >F | >F | >F | >F | >F | **0** | 134 |

Bold = absent while inside the spectrum's range; ">F" = above the record, so not yet a question.

- **The flip is not the flood.** At the flip rung `13 -> 17` for `v = 6` the coefficient is 14
  instead of 13; the survival term is 840 of the new 1,022, so the extra unit of coefficient is
  worth 60 gaps, **5.9%** of the total. In general the flip is worth exactly `1 + 1/(q'-4)` on the
  survival term and nothing else -- at the twin gear of column 12 that is `68/67`, a **1.5%**
  effect. `v = 6` recovered from `r = 0.18` to `r = 0.67` at `11 -> 13`, one rung BEFORE its flip,
  while still uncoupled, and by merges alone.
- **Letter status does not lift the rarity to 1.** After the flip `r(4)` and `r(6)` settle at
  0.56-0.60 and stay there for five rungs. What the flip changes is only which of three values
  `c_{q'}` takes, and by the fibre theorem a size can be the short letter of at most the two gears
  of its own column, so the total lift available is `prod (q-3)/(q-4)` over those two gears.
- **The handicap saturates.** `A(24)/A(23) = 0.315` and `A(24)/A(25) = 0.3125` at m29 and again to
  four figures at m31: a new gear coupling neither size changes nothing. The identity part of the
  depletion is a bounded constant in `y`, not a growing one.
- **First appearance versus the flip.** `v = 4` is absent at m7 and appears at m11, its flip rung;
  `v = 41` appears at m31, also its flip rung (`41 = -d_31 (mod 31)`), with survival 0 and all 134
  from merges. `v = 6` appears at m11, six rungs before its flip; `v = 24` appears at m29, far
  below its flip rung 71. So 2 of 4 first appearances coincide with the flip and 2 do not.

### 2.8 The residual, exactly (item 5)

"An uncoupled size below `y^2/3` exists" is not a spectral statement at all: it is a statement
about the gears. `v` is uncoupled in `{5..y}` iff `v` avoids the three residues `{0, +d_q, -d_q}`
modulo every gear `q <= y`. So:

> **the openings of the machine are the survivors of TWO residue classes per gear (`+-u_q`); the
> uncoupled sizes are the survivors of THREE (`0, +-d_q`). The root asks whether a survivor of the
> 2-class system lands in `(y/6, y^2/6]`; forcing depletion asks whether a survivor of the 3-class
> system lands in `[2, y^2/3]`. It is the same covering question with one more class per gear, on
> a strictly thinner set.**

Measured at every prime rung `y = 5..19997` (2,260 rungs, sizes to 2,000,000):

- the first uncoupled size `v_1(y)` lies below `y^2/3` at **2,260 of 2,260** rungs;
- the first uncoupled EVEN size `v_e(y)` lies below `y^2/3` at **2,260 of 2,260**, and its
  half-column `v_e/2` is a twin column above `y` at **2,260 of 2,260** (H6 re-verified at 2,260
  fresh cells);
- `v_e(y)/y` lies in `[0.335, 2.244]` with median 2.020 -- the first even uncoupled size sits
  around `2y`, three orders of magnitude below the window's top at the larger rungs;
- `v_1(y)` is even at only **374 of 2,260** rungs: at most rungs the first uncoupled size is odd,
  and an odd uncoupled size encodes a quarter-column prime, not a twin column;
- `v_e(y) = 2 d_0(y)` -- twice the FIRST twin column above `y` -- at only **25 of 2,260** rungs.

The last two lines are why the object is strictly harder than the root. `v_e(y)/2` is not merely a
twin column above `y`: its INDEX `c` must itself be `y`-rough (only 2, 3 and primes above `y`
divide it). While `c < y` that forces `c` to be 3-smooth, and the measured sequence is exactly the
3-smooth twin columns `c = 2, 3, 12, 18, 72, 192, 432`; past `y` the index is prime,
`c = 5023, 10357, 20123` at `y = 4999, 9973, 19997`. So

> **an uncoupled even size below `y^2/3` exists <=> a twin column with a `y`-rough index sits in
> the window.** That implies the root and is not implied by it. An identity forcing depletion
> would prove strictly more than the conjecture needs -- and the identities available force
> nothing at all (2.6).

## 3. Mechanism, stated once

Three things, and they fit together.

**1. The multiplicity function has one exact law, and it is multiplicative in the wrong place.**
Adding a gear multiplies each old size's multiplicity by `c_{q'}(v) in {q'-4, q'-3, q'-2}` and adds
a merge census. The coefficient is the only place where the arithmetic of `v` -- whether a gear can
chain at that distance -- enters at all, and it enters as a factor between 1 and `(q'-2)/(q'-4)`,
at most `5/3` at `q' = 7` and tending to 1. So the law the branch hoped would force something is a
law of gentle multipliers: it propagates a spectrum, it never creates one. Everything a spectrum
contains was made by merges (55 of 55 sizes at m31), and merges are indifferent to coupling: they
need a run of the right span and a phase, and runs of every span below `F_2` exist in abundance.

**2. The depletion of the uncoupled sizes is mostly an identity, and the identity is a pool size.**
`A(v) = prod_q c_q(v)` counts the open pairs at distance `v`, and it is minimised exactly at the
uncoupled sizes, where it equals `prod (q-4)` -- the same number as the count of adjacent openings.
An uncoupled size therefore starts from the smallest pool of candidate pairs any distance can have,
and its multiplicity is small for that reason before anything else happens. Dividing the pool out
removes a factor of 4 to 5 of the measured 6.9 to 101.3, and what remains is inside the coupled
sizes' own scatter at three of the four cells. The decisive control is `v = 1`, uncoupled at every
machine, whose `rho` is exactly 1: uncoupledness shrinks the pool, it does not make a pair less
likely to be adjacent.

**3. Two systems, not one.** Openings are what two classes per gear miss; uncoupled sizes are what
three classes per gear miss. The branch's plan was to make the second system's non-emptiness follow
from an identity, and thereby force the first system's non-emptiness in the window. But the
identities on `m` see the second system only through the factor `c_q(v)`, which is smallest
precisely when `v` is uncoupled, so they can only push an uncoupled size DOWN. The LP makes that
quantitative: the feasible interval for `m(24)` at m29 is `[0, 15,767,400]` against a truth of
1,180. Nothing in the sum-rule family knows whether the uncoupled sizes exist.

## 4. What is new

1. **`A_M(v) >= prod_q (q - 4)` with equality iff `v` is uncoupled in `M`, and `A_M(1)` attains
   it.** The uncoupled distances are exactly the machine's minimum-support distances, and their
   open-pair count is the domino count `m(1)`. One line from `c_q(v) in {q-4, q-3, q-2}`; not on
   record, and it is the identity behind the measured depletion of `half_column.md`.
2. **`rho(v) = m(v)/A(v)`, and the deflation of the depletion.** The recorded factors of 12-128 are
   4-5 times the true residual: 101.3 -> 22.5 and 6.9 -> 1.3 at m29, 36.3 -> 8.9 and 12.5 -> 2.9 at
   m31, against a coupled control of geometric sd 1.9-2.0. Three of the four uncoupled cells with
   `m > 0` fall inside the coupled scatter after the division.
3. **`rho(1) = rho(2) = 1` exactly at every machine** (`m(1) = A(1) = prod(q-4)`;
   `m(2) = A(2) = 8 prod_{q>=11}(q-4)`, because gear 5 forbids three consecutive openings), and
   `v = 1` is uncoupled at every machine: the uncoupled size with the largest possible `rho`.
4. **`m(2) = (8/3) m(1)` at every machine containing 5 and 7** (6 of 6 exact) -- the size-2
   companion of the domino law.
5. **Every spectrum hole is a PHASE hole, never a span hole: 7 of 7** at rungs `11->13 .. 23->29`,
   with the run census equal to `A_M(v)` in every case (28, 420, 3,276, 2,457, 36,855, 700,245,
   2,334,150). The exact characterisation of absent versus depleted: an absent size is one whose
   `A_M(v)` candidate pairs all fail the phase test, and uncoupled sizes are absent more often
   because their pool is the floor, not because their pool is empty.
6. **No size is ever lost, and every size is born a merge.** `m_{M'}(v) >= (q'-4) m_M(v)` and
   `Spec(M) subset Spec(M')`, 137 of 137 cells, 0 exceptions; first appearances have survival 0 at
   55 of 55 sizes.
7. **The flip is worth `1 + 1/(q'-4)`.** When `2c` becomes the short letter of the twin gear
   `6c - 1` the coefficient rises by one unit and nothing else changes: 5.9% of the total at
   `13 -> 17` for `v = 6`, 1.5% at the twin gear of column 12. The recorded "the spectrum flips
   when the machine acquires the column" is a coincidence of first appearance at 2 of 4 cells and
   not a mechanism -- `v = 6` recovered a rung before its flip, by merges, while still uncoupled.
   And the identity handicap saturates: `A(24)/A(23) = 0.315` at m29 and at m31 alike.
8. **The feasible interval of an uncoupled size's multiplicity under every identity is
   `[0, ~A(v)]`** -- `[0, 15,767,400]` at m29 against a truth of 1,180 -- with the uncoupled sizes
   carrying `10^-5` to `10^-4` of both budgets. The identities determine nothing there.
9. **The two-class / three-class reading of the root.** Openings are the survivors of `{+-u_q}`;
   uncoupled sizes are the survivors of `{0, +-d_q}`. Forcing depletion is the root's own covering
   question with one more class per gear.
10. **The first uncoupled size measured at 2,260 prime rungs to `y = 19997`:** `v_1(y) < y^2/3` at
    2,260 of 2,260; the first uncoupled EVEN size has a twin half-column above `y` at 2,260 of
    2,260, with `v_e(y)/y in [0.335, 2.244]`, median 2.020; `v_1` is even at only 374 of 2,260;
    `v_e = 2 d_0` at only 25 of 2,260. The even branch's index is 3-smooth while `c < y`
    (`c = 2, 3, 12, 18, 72, 192, 432`) and prime beyond (`5023, 10357, 20123`).
11. **The recursion carried three rungs past its recorded verification** (`19->23`, `23->29`,
    `29->31`) and used as the generator of the m29 and m31 spectra, with the survival/merge split
    by size and by depth; depth-2 merges carry 98.1% of what the top rung creates. The law itself
    is `paired-holt-recursion`, cited, not claimed.

### 4a. Exceptionless statements, with counts

- **S1.** `sum_v m(v) = prod (q-2)` and `sum_v v m(v) = P`. **9 of 9** machines m5..m31.
- **S2.** `m(v)` even for every `v >= 2`; `m(1)` odd. **7 of 7** machines sieved (= docs/proofs/03
  (e), instrument check).
- **S3.** `m(1) = prod (q-4)`. **7 of 7**. `m(2) = 8 prod_{q>=11}(q-4) = (8/3) m(1)`. **6 of 6**
  machines containing 5 and 7.
- **S4.** `A(v) = prod_q c_q(v)` for every `v <= F`. **102 of 102** cells (= `depth-sum-identity`,
  cited).
- **S5.** `A(v) >= prod (q-4)`, with equality iff `v` is uncoupled; `A(1) = prod(q-4)`. Proved;
  measured at every cell of S4.
- **S6.** `rho(1) = rho(2) = 1`. **4 of 4** machines checked directly (m19, m23, m29, m31); proved
  for all.
- **S7.** `survival_{M+q'}(v) = c_{q'}(v) . m_M(v)`. **137 of 137** (rung, size) cells,
  `5->7 .. 29->31` (= the `j = 1` coefficient of `paired-holt-recursion`, cited).
- **S8.** The recursion reproduces the directly sieved spectrum. **94 of 94** sizes at the six
  sieve-checkable rungs, **0 error**; at m29 and m31 it reproduces every corpus value, 13 of 13.
- **S9.** `m_{M+q'}(v) >= (q'-4) m_M(v)` and `Spec(M) subset Spec(M+q')`. **137 of 137** cells.
- **S10.** Every size's first appearance has survival 0. **55 of 55** sizes at m31.
- **S11.** Every size absent below `F(M')` is a phase hole, never a span hole. **7 of 7**.
- **S12.** The first uncoupled even size of `{5..y}` is twice a twin column above `y` and lies
  below `y^2/3`. **2,260 of 2,260** prime rungs `5..19997`.

## 5. Toward the root, and the residual

**What an identity forcing depletion would prove.** By S12 and H6, an uncoupled even size below
`y^2/3` is `2c` for a twin column `c` in the window whose index is `y`-rough. So a proof that some
uncoupled even size below `y^2/3` must exist is a proof of the conjecture and more: it also
supplies the roughness of the index, which the conjecture does not ask for. It is a strictly
stronger target, and the measurement says by how much -- `v_e(y) = 2 d_0(y)` at only 25 of 2,260
rungs, so the rough twin column is almost never the first twin column.

**What the identities actually force.** Exactly three things, all of them caps or propagations:
`m(v) <= A(v)`, with `A(v) = prod (q-4)` at the uncoupled sizes; `m_{M'}(v) >= (q'-4) m_M(v)`,
which propagates what already exists; and the two global sums, of which the uncoupled sizes are
`10^-5`. The LP interval `[0, 15,767,400]` for `m(24)` at m29 is the exact statement that nothing
else is forced.

**The residual, named.** To force depletion one would have to show that the three-class system
`{0, +-d_q : q <= y}` fails to cover `[2, y^2/3]`. That is the root's own covering problem with
three classes per gear instead of two -- a Jacobsthal-type statement one class higher, on a set
thinner by `prod (q-3)/(q-2)`. The branch therefore does not weaken the root; it strengthens it.
Recorded as the reason the route is dead, not as an open lemma.

**What survives and where it goes.** S5 (the autocorrelation floor and its equality case) and the
`rho` deflation belong beside `half_column.md` 2.3-2.4, which they explain and correct: the
recorded "12 to 128 times rarer, a step at zero coupling gears" is, to a factor of 4-5, the exact
pair count `A(v)`, and after that division only one of four cells is an outlier. S11 (every hole is
a phase hole, with the pool equal to `A(v)`) belongs beside the recorded spectrum holes. S9 and S10
(no size lost; every size born a merge) belong beside `paired-holt-recursion` as corollaries of its
`j = 1` coefficient.

## 6. Verdict

- **The sum-rule route is DEAD.** The identities the multiplicity function satisfies -- the two
  global sums, the parity theorem, the domino law and its size-2 companion, and the autocorrelation
  cap -- leave the multiplicity of an uncoupled size free in `[0, ~A(v)]`. Zero is feasible at
  every uncoupled size of every machine tested. No identity forces depletion, and the structural
  reason is that coupling enters every identity only through `c_q(v)`, which is minimal for
  uncoupled `v`: the family is monotone in the wrong direction.
- **The recursion is a rediscovery and is cited.** `m_{M+q'}(v) = c_{q'}(v) m_M(v) + Merge(v)` is
  the `j = 1` coefficient of `paired-holt-recursion` (SCRIPT-VERIFIED, prior-art checked
  2026-08-23). It is carried three rungs further here and used as the generator of the m29 and m31
  spectra with the survival/merge split; the law is not new. Status of the node as a route: DEAD.
- **The depletion is largely an identity, and that identity is new.** `A(v) >= prod (q-4)` with
  equality exactly at the uncoupled sizes. Status FACT, and it deflates a recorded measurement by
  a factor of 4 to 5.
- **The characterisation asked for is exact and answered.** Absent versus depleted is decided by
  the phase test on `A_M(v)` candidate pairs; 7 of 7 holes are phase holes, never span holes.
  Status FACT.
- **The life cycle is measured and the flip is not the flood.** The coefficient flip at the twin
  gear is worth `1 + 1/(q'-4)`; `v = 6` recovered a rung before its flip; the identity handicap
  saturates in `y`. Status FACT; it corrects the recorded reading of the spectrum flip.
- **The self-referential reading is exact but points the wrong way.** An uncoupled even size below
  `y^2/3` exists iff a rough twin column sits in the window -- 2,260 of 2,260 rungs to
  `y = 19997` -- and it is a strictly stronger object than the root's, since the index must be
  `y`-rough (`v_e = 2 d_0` at only 25 of 2,260). Status FACT, not a route.

## 7. Scorecard, filled

| # | prediction | result |
|---|---|---|
| P1 | the five elementary identities | **HELD, 0 exceptions**, m5..m23 (7 machines); (a), (b) again at m29 and m31 |
| P2 | survival count `= c_{q'}(v)` | **HELD, 137 of 137** cells -- and it is the recorded `j = 1` coefficient of `paired-holt-recursion`; sub-question stopped and cited |
| P3 | the recursion reproduces the next spectrum | **HELD, 0 error on 94 of 94 sizes** at the six sieve-checkable rungs; all 13 corpus values at m29 and m31 reproduced |
| P4 | the autocorrelation sum rule (instrument) | **HELD, 102 of 102** cells; = `depth-sum-identity` (Holt Cor. 1), cited |
| P5 | no size is ever lost | **HELD, 137 of 137**; `Spec` inclusion 0 exceptions m5..m31 |
| P6 | the depletion is a merge effect, not a coefficient effect | **HELD in the letter, REFUTED in intent**: merge supplies 81.8% at `v = 24` and 67.4% at `v = 36` at `29->31`, but 83.1% and 91.4% at the COUPLED controls 35 and 37. Merge share tracks rarity, not coupling |
| P7 | absent iff no run passes the phase test; first appearance a pure merge | **HELD**: 7 of 7 holes are phase holes with `A_M(v)` runs each and total weight 0; 55 of 55 first appearances have survival 0; no size ever returns to zero |
| P8 | the counterfactual is FEASIBLE (the route dies) | **HELD**: LP interval `[0, 15,767,400]` for `m(24)` at m29 against a truth of 1,180; zero feasible at every uncoupled size at m19, m23, m29, m31 |
| P9 | the flip is exact and is not the flood | **HELD**: the coefficient rises `q'-4 -> q'-3` exactly at the twin gear (2 of 2 observable columns); worth 5.9% at `13->17` and 1.5% at `q' = 71`; `v = 6` recovered one rung earlier while uncoupled |
| P10 | the residual is strictly stronger than the root | **HELD**: uncoupled even size `<=>` rough twin column in the window, 2,260 of 2,260 rungs; `v_e = 2 d_0` at only 25 of 2,260, so the roughness is a real strengthening |

## 8. Dead ends, each with its refuting instance

- **D1. "An identity on `m` forces depleted sizes to exist."** Dead. Refuting instance: at m29 the
  polytope cut out by `sum m = prod(q-2)`, `sum v m = P`, `m(1) = prod(q-4)`,
  `m(2) = 8 prod_{q>=11}(q-4)` and `0 <= m(v) <= A(v)` contains a point with `m(24) = 0` and
  another with `m(24) = 15,767,400`. The identities cannot see a quantity carrying `5.7 x 10^-6`
  of the count budget.
- **D2. "Being uncoupled makes a size rare."** Dead as stated. Refuting instance: `v = 1` is
  uncoupled at every machine and has `rho(1) = 1`, the maximum. What uncoupledness does is minimise
  the pool `A(v)`; after dividing the pool out, 3 of the 4 uncoupled cells with `m > 0` are inside
  the coupled sizes' scatter (`v = 36` at m29 sits at the 23.7th percentile).
- **D3. "The spectrum flips when the machine acquires the column."** Dead as a mechanism. The
  coefficient flip is worth `1 + 1/(q'-4)` -- 60 gaps of 1,022 at `13 -> 17`, 1.5% at `q' = 71`.
  Refuting instance: `v = 6` went from `r = 0.18` to `r = 0.67` at `11 -> 13`, one rung before its
  flip and while still uncoupled; and `r(4)`, `r(6)` sit at 0.56 five rungs after their flips.
- **D4. "An uncoupled size is absent because no run of that span exists."** Dead: 7 of 7 holes have
  `A_M(v) > 0` runs of the right span (28 to 2,334,150 of them) with total phase weight 0.
- **D5. "The merge term is where uncoupling bites."** Dead: at `29 -> 31` the merge share is 81.8%
  at the uncoupled `v = 24` and 91.4% at the coupled `v = 37`.
- **D6. "Forcing depletion is a route to the root."** Dead by strengthening: it would prove a twin
  column with a `y`-rough index in the window, which coincides with the first twin column at only
  25 of 2,260 rungs; and it is the root's own covering problem with three residue classes per gear
  instead of two.
