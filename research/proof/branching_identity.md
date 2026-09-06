# Node 4.i.a - THE BRANCHING IDENTITY, PROVED IN GENERAL, AND THE SPECTRUM'S EVOLUTION

Parent: node **4.i, the merge forest** (`research/proof/merge_forest.md`), FACT since 2026-09-05.
What spawns this branch is one line of its mechanism section (3.1): the order distribution of a
rung is the second difference of a chain-count sequence, `n_J = C_{J-1} - 2 C_J + C_{J+1}`, with
`C_0 = q' N`, `C_1 = 2 N`, `C_2 = 2 A_0 + A_d` verified at 8 rungs and proved in a paragraph. The
owner flagged the identity as interesting. This branch does three things the parent did not: it
proves the identity in general with the boundary cases done exactly, it pushes `C_r` to a closed
form for **every** `r` (not just `r <= 2`), and it asks whether the same machinery closes the
**size** side - whether the spectrum's evolution is a finite-depth recursion.

Scripts in `research/anchor235/r57/` (prefix `bi_`); result outputs in
`research/anchor235/r57/results/` (untracked). Every number this document relies on is written
into the document.

---

## 0. Pre-registered

Written from the proof, before any computation of this branch. (The proof came first; the
predictions are what the proof asserts, each with the number that would refute it.)

### 0.1 The objects, defined exactly

Machine `M` with gears `5 <= q <= y`, period `P`, openings `op(0) < op(1) < ... < op(N-1)` in
`[0, P)` extended `P`-periodically, cyclic gap sequence `gap(n) = op(n+1) - op(n)` (`N` gaps per
period, `sum gap = P`). New gear `q'`, `c = 6^{-1} mod q'`, `d = 2c`, teeth `{c, -c}`. Adding
`q'` makes `q'` **copies** of `M`'s period; copy `j` realises **deletion phase**
`r_j = -c - jP (mod q')` and `j -> r_j` is a bijection of `Z_{q'}` (docs/proofs/05 (A)), so
"copy" and "phase" are the same index and are used interchangeably below.

**Letters.** A gap value `v` is `PAD` if `v = 0 (mod q')`, `UP` if `v = +d`, `DOWN` if `v = -d`,
`BAD` otherwise; a word of letters is **legal** if no two consecutive nonzero letters are equal,
pads transparent (docs/proofs/05 (F)). `L(M)` is the longest realised legal word of consecutive
gap letters, `J_max = L + 2`, `A_kill = L + 1` (docs/proofs/10).

**Coordinates for the proof.** An opening `x` is struck in phase `r` iff `x - r in {0, d}`.
Write `x = r + c + t c` with `t in {-1, +1}` (`t = -1` for `x = r`, `t = +1` for `x = r + d`);
then a gap between two struck openings is `(t_next - t_prev) c`, i.e. `PAD` keeps `t`, `UP` takes
`t` from `-1` to `+1`, `DOWN` from `+1` to `-1`. That is file 05 (F) written as a walk.

**The counts.** For `r >= 0`,

    C_r  :=  # { (phase p, index n) : op(n), op(n+1), ..., op(n+r-1) all struck in phase p }

(`r = 0`: the condition is empty, so `C_0 = q' N`). `n_J` is the number of gaps of `M + q'` of
**order** `J` (fusions of exactly `J` old gaps) per period of `M + q'`.

**The weight.** For a `J`-window of consecutive gaps at `n`, `eps_J(n)` is the number of copies
in which those `J` gaps fuse into exactly one gap of `M + q'`: the `J - 1` interior openings all
struck, the two endpoints not struck.

**The dictionaries.** `D_K(M)` is the set of realised `K`-tuples of consecutive gap sizes;
`D_K^#(M)` is the same **with multiplicity and in position order** (the multiset of realised
`K`-windows). `K_m(M -> q')` is the largest number of consecutive old gaps spanned by `m`
consecutive new gaps.

### 0.2 The theory

**T. Both sides of a rung are local, and the locality has a fixed depth.** The count side is a
run-length inversion of one sequence `C_r`, and `C_r` is a pure word count of the old machine's
letters with a single exceptional factor of two for the all-pad word. The size side is the same
statement carrying the sizes: the new multiplicity of a value is a sum of a local weight
`eps_J in {0, 1, 2}` (`{q'-4, q'-3, q'-2}` at `J = 1`) over the old machine's realised windows.
Hence the whole evolution of the spectrum is a recursion whose depth is `J_max = L + 2`, and the
hierarchy of dictionaries closes at depth `m J_max` - bounded exactly when `L` is bounded, which
is docs/proofs/10's open rider and not one step nearer.

### 0.3 Predictions, each with the refuting number

- **P1 (the identity, general).** `n_J = C_{J-1} - 2 C_J + C_{J+1}` for every `J >= 1` and every
  machine, with `C_0 = q' N` the only boundary convention. Verified at all 8 rungs `5->7 .. 29->31`
  by two vehicles independent of the derivation. REFUTED by one mismatch at one rung.
- **P2 (the closed form of `C_r`, all `r`).** `C_r = W_{r-1} + Z_{r-1}`, `W_m` = the number of
  positions at which the `m` gaps form a legal word, `Z_m` = the number at which they are all
  `PAD`. Corollaries predicted exactly: `C_1 = 2N`; `C_2 = 2 A_0 + A_d`; `C_3 = W_2 + Z_2`;
  `C_4 = W_3 + Z_3`; `C_r = 0` for `r > L + 1`. REFUTED by one `r` at one rung.
- **P3 (tail form).** `sum_{J' >= J} n_{J'} = C_{J-1} - C_J` exactly, so the merge count is
  `2N - 2A_0 - A_d` and the merge fraction is `2/(q'-2)` minus `C_2/((q'-2)N)`. REFUTED by one rung.
- **P4 (max order).** max order `= 1 + D = L + 2 = J_max` with `D` the largest `r` with `C_r > 0`.
  8 of 8. REFUTED by one rung.
- **P5 (moments).** mean `= q'/(q'-2)` (known, one line from conservation) and
  `sum_J J^2 n_J = 2 sum_{r >= 0} C_r - q' N = q' N + 4 N + 2 sum_{r >= 2} C_r`, hence
  `Var = 2[(q'-4) + S(q'-2)]/(q'-2)^2` with `S = (1/N) sum_{r >= 2} C_r`. Exact at 8 rungs.
  REFUTED by one rung off by one.
- **P6 (what the variance says about the teeth).** The teeth enter the variance only through `S`,
  so the real machine's variance percentile on the 20-member counterfactual family equals its
  `n_3` percentile, predicted `<= 0.30` at rungs `11->13` and `13->17` (merge_forest measured
  `n_3` at percentile 0.20). REFUTED by a variance percentile above 0.5, or by the closed form
  failing on one member.
- **P7 (the size side).** `m_{M+q'}(v) = sum_{J=1}^{J_max} sum_{J-windows of span v} eps_J`,
  exact. Reproduces the full spectrum at every rung to `23 -> 29` (0 error, direct build) and the
  m29 and m31 corpus gates. REFUTED by one multiplicity.
- **P8 (least depth).** The least `K` for which the depth-`K` window dictionary reproduces the
  next spectrum exactly is `K = J_max = L + 2` at every rung, values `3, 2, 3, 3, 3, 4, 3, 5`,
  and depth `J_max - 1` is strictly wrong (nonzero error at every rung). REFUTED by an exact
  reproduction at `J_max - 1` or a failure at `J_max`.
- **P9 (closure).** The depth-`m` dictionary of `M + q'` (with multiplicity) is determined
  exactly by `D_{K_m}^#(M)` with `K_m <= m J_max`; `K_m` does NOT grow with the rung, only with
  `J_max`. Predicted `K_1 = J_max`, `K_2 <= 2 J_max`, and measured `K_2 - K_1` small (1 or 2).
  REFUTED if `K_m` grows by one per rung at fixed `J_max` (an open hierarchy in the rung).
- **P10 (the negative, pre-registered so the branch cannot claim a route).** A closed hierarchy
  gives the next spectrum but bounds nothing: the record is `F(M+q') = max_J Q*_J(M)` and the
  dictionary's EXTREMES are not bounded by the previous dictionary's extremes by anything proved
  here. Predicted: the residual is unchanged, and is the chain statement.

### 0.4 Scorecard

| # | prediction | verdict | evidence |
|---|---|---|---|
| P1 | the identity in general | **PROVED** (2.1-2.2) + verified 8 of 8 rungs, two independent vehicles, 0 exceptions | 3.1 |
| P2 | `C_r = W_{r-1} + Z_{r-1}` for every `r` | **PROVED** (2.3) + verified 8 of 8, `r = 0..5` | 3.1 |
| P3 | tail form, merge count `2N - 2A_0 - A_d` | **PROVED** (2.4) + verified 8 of 8 | 3.1, 4.2 |
| P4 | max order `= L + 2` | CONFIRMED 8 of 8 (proved: it is `C_{L+1} > 0 = C_{L+2}`) | 3.1 |
| P5 | second moment closed form | **PROVED** (2.5) + exact at 8 of 8 rungs and 42 of 42 family members | 3.2, 3.4 |
| P6 | variance percentile = `n_3` percentile, `<= 0.30` | CONFIRMED: 0.190 at both rungs, both statistics; closed form exact 42 of 42 | 3.4 |
| P7 | the size formula | **PROVED** (2.6) + EXACT at 7 rungs against the direct build and at m31 against every corpus gate | 4.1 |
| P8 | least depth `= J_max` | CONFIRMED 8 of 8, values `3, 2, 3, 3, 3, 4, 3, 5`; depth `J_max - 1` wrong at every rung, and the error is exactly `C_{J_max-1} - C_{J_max}` | 4.1, 4.2 |
| P9 | closure at `K_m <= m J_max` | **PROVED** + measured; `K_2 - K_1 = 1, 1, 1, 1, 1, 2, 2` and `K_m` tracks `J_max`, not the rung | 5.1 |
| P10 | the residual is unchanged | CONFIRMED, and made exact: `F(M+q') = max_J Q*_J(M)` with the table of `Q*_J` | 6 |

**Stop rules honoured.** The mean order (`q'/(q'-2)`) is merge_forest's identity and is used in
one line. The merge law, chain law, `J_max = L + 2`, the depth-0 monotonicity lemma and the
inflation-onset law are cited, never re-derived; where this branch's exact transfer meets the
project's over-generating transfer (docs/novel/dictionary-monotonicity-onset.md) the difference
is stated in one paragraph (5.2) and the sub-question stopped.

---

## 1. Setup (exact ranges)

Everything is exact integer arithmetic on full periods; no sampling anywhere.

| object | range | cost | script |
|---|---|---|---|
| gap sequences | m5..m23 by direct sieve; m29 (214,708,725 gaps, period 1,078,282,205) by the merge construction from m23, cached as `uint8` | 6 s | `bi_core.py` |
| `C_r`, `n_J` by inversion; `n_J` by the local weight; `n_J` by a direct copy-by-copy build of the tiled period | all 8 rungs `5->7 .. 29->31` | 130 s total, 126 s of it the top rung | `bi_counts.py` |
| the whole spectrum of `M + q'` from `M`'s window dictionary; truncation table; `|D_J(M)|`; the per-`J` extremes | all 8 rungs, m31 included (6,226,553,025 gaps, never materialised) | 95 s | `bi_spectrum.py` |
| the depth-`m` dictionary of `M + q'` by the z-walk (phase enumeration, no word theory), `m = 1, 2, 3` | rungs `5->7 .. 23->29` | 50 s | `bi_closure.py` |
| the counterfactual family, 20 members + the real machine at two rungs | full periods | 1 s | `bi_family.py` |

**Instrument gates, all passed.** The rebuilt m29 spectrum returns `F = 43`, `|Spec| = 41`,
absent `{41, 42}`, `m(4) = 14,178,528`, `m(6) = 10,497,320`, `m(24) = 1,180`, `m(36) = 38`,
`sum m = 214,708,725`, `sum v m = 1,078,282,205`; the m31 spectrum returns `F = 58`,
`|Spec| = 55`, absent `{54, 56, 57}`, `m(4) = 398,923,200`, `m(6) = 299,202,120`,
`m(24) = 174,704`, `m(36) = 3,152`, `m(41) = 134`, `sum m = 6,226,553,025`,
`sum v m = 33,426,748,355`. Every one matches `spectrum_sum_rule.md`, `half_column.md` and
`merge_forest.md`. The realised 5-tuple dictionary of m29 comes out at **208,668** tuples, the
number recorded independently in `docs/novel/dictionary-monotonicity-onset.md`. The family's
`n_3` range at the two rungs (`0 / 20 / 56` and `19 / 238 / 576`, min / median / max) reproduces
merge_forest 2.7 member for member.

---

## 2. Statement and proof

Throughout, `M`, `q'`, `N`, the letters and `C_r` are as in 0.1. One proviso is needed and is
stated where it is used:

> **(H)** Some gap of `M` is not `= 0, +d` or `-d (mod q')`.

(H) says exactly that not every opening of `M` can be struck in one phase, since if all were,
every gap would be a letter. It is verified at all 8 rungs: `A_0 + A_d` against `N` is
`2 / 3`, `0 / 15`, `6 / 135`, `72 / 1,485`, `1,088 / 22,275`, `11,784 / 378,675`,
`243,816 / 7,952,175`, `8,022,924 / 214,708,725`.

### 2.1 Lemma A (a rung is a run-length problem, per phase)

Fix a phase `p`. Let `s_n = 1` if `op(n)` is struck in phase `p` and `0` otherwise, an
`N`-periodic cyclic binary sequence, not identically 1 by (H). By the merge law
(docs/proofs/05 (D)) the openings of `M + q'` inside copy `p` are exactly the `n` with `s_n = 0`,
and the gap of `M + q'` beginning at such an `n` fuses the old gaps
`gap(n), gap(n+1), ..., gap(n + J - 1)` where `J - 1` is the length of the run of ones
immediately after `n`. So

> a new gap of order `J` in copy `p` **is** a zero followed by exactly `J - 1` ones.

Two bookkeeping facts make this exact rather than approximate. First, the gap sequence is
`N`-periodic **in the tiled index**: the gap between tiled openings `t` and `t + 1` is
`gap(t mod N)`, because the copies are consecutive blocks of the same period. Second, a run of
ones may cross a copy boundary; that is a genuine merge and is counted once, in the phase of the
copy that contains its first opening - the tiled sequence of `q' N` openings is one cyclic
sequence and the per-phase sequences are its `q'` blocks. Both are respected below by counting
over the pair `(phase, index)`.

### 2.2 Theorem 1 (the branching identity)

> For every machine `M`, every new gear `q'` and every `J >= 1`,
>
>     n_J  =  C_{J-1}  -  2 C_J  +  C_{J+1},
>
> with the single boundary convention `C_0 = q' N`.

*Proof.* Work in one phase `p` and write `R_k` for the number of cyclic positions `n` with
`s_n = s_{n+1} = ... = s_{n+k-1} = 1` (occurrences, not maximal runs), with `R_0 := N` (the empty
window occurs at every position). Let `A_k` be the number of positions `n` with `s_n = 0` and
`s_{n+1} = ... = s_{n+k} = 1` (a zero followed by **at least** `k` ones) and `G_k` the number
with exactly `k` ones after the zero.

1. `A_k = R_k - R_{k+1}` for every `k >= 0`. For `k >= 1`: among the `R_k` positions carrying `k`
   ones, those whose predecessor is also a one are in bijection with the `R_{k+1}` positions
   carrying `k + 1` ones (shift by one), and the rest are precisely the ones preceded by a zero.
   For `k = 0`: `R_0 - R_1 = N - R_1` is the number of zeros, which is `A_0`. This is where "not
   all ones" (H) is used - it makes "preceded by a zero" available at every run - and where the
   cyclicity is used: on a cycle every maximal run of ones has exactly one predecessor zero, with
   no end effects.
2. `G_k = A_k - A_{k+1}` by definition of "exactly" against "at least".
3. Hence `G_k = R_k - 2 R_{k+1} + R_{k+2}`.
4. By Lemma A, `n_J = sum over phases of G_{J-1}`. Summing step 3 over the `q'` phases and
   writing `C_r = sum_p R_r` gives `n_J = C_{J-1} - 2 C_J + C_{J+1}`. The convention
   `C_0 = sum_p R_0 = q' N` is forced, not chosen. `qed`

Two lines, as advertised, once the two boundary cases (`k = 0` and the cyclic wrap) are named.

**Corollary (conservation, known).** `C_1 = 2N` because each opening is struck in exactly two
copies (docs/proofs/05 (A)); so `sum_J J n_J = C_0 = q' N` and `sum_J (J-1) n_J = C_1 = 2N`, and
with `sum_J n_J = C_0 - C_1 = (q'-2) N` the mean order is `q'/(q'-2)` - merge_forest's identity,
cited in one line.

### 2.3 Theorem 2 (the closed form of `C_r`, for every `r`)

> For `r >= 1`,   `C_r = W_{r-1} + Z_{r-1}`,
>
> where `W_m` is the number of cyclic gap positions at which the `m` gaps
> `gap(n), ..., gap(n+m-1)` form a **legal word**, and `Z_m <= W_m` the number at which they are
> **all `PAD`**. `W_0 = Z_0 = N`.

*Proof.* Fix `r >= 1` and a tiled index `t`; put `n = t mod N`. The `r - 1` gaps between the
openings `t, ..., t + r - 1` are `gap(n), ..., gap(n + r - 2)` (the gap sequence is `N`-periodic
in the tiled index, Lemma A). By the chain law read as words (docs/proofs/05 (F) "killable iff
legal" = docs/proofs/10 Theorem 1), these `r` openings lie on the teeth of a common phase iff
that word is legal - a condition on `n` alone. Given legality, how many of the `q'` tiled indices
`t = jN + n` realise it? The residues `op(n) + jP (mod q')` run over all of `Z_{q'}` exactly once
as `j` does, and the phases that strike the whole chain are exactly the `r` with `op(n) - r`
equal to `t_1 c + c` for a consistent reading of the word. A legal word with at least one nonzero
letter admits exactly one starting tooth (the first nonzero letter fixes it, file 05 (F) step 13);
an all-`PAD` word admits both, and the two give distinct phases because `d != 0 (mod q')`. So the
count per position is `2` (all pad), `1` (legal, some nonzero letter) or `0` (illegal), and
summing over `n` gives `W_{r-1} + Z_{r-1}`. `qed`

**The factor two, exactly.** The `2` in `C_1 = 2N` and in `C_2 = 2 A_0 + A_d` is not a coincidence
of small `r`: it is the all-pad term `Z_{r-1}`, present at every `r`, and it is the only place
where a phase is not determined by the word.

**The derived closed forms.**

- `r = 1`: `W_0 = Z_0 = N`, so `C_1 = 2N`.
- `r = 2`: `W_1 = A_0 + A_d` and `Z_1 = A_0` with `A_0 = #{gaps = 0 (mod q')}` (i.e.
  `m(q') + m(2q') + ...` read off the old spectrum) and `A_d = #{gaps = +d or -d}` (i.e. the
  spectrum's mass on the two letter classes `a = 2u_{q'}` and `b = q' - a` and their translates
  by multiples of `q'`). Hence **`C_2 = 2 A_0 + A_d`**, merge_forest's form, now a special case.
  *The sign pattern*: the `+d` and `-d` classes each contribute once (a nonzero letter fixes the
  tooth), the `0` class contributes twice (a pad leaves the tooth free); there is no cancellation
  and no dependence on which of `a`, `b` is the short letter.
- `r = 3`: `C_3 = W_2 + Z_2` where, writing `P[x][y]` for the number of positions whose two
  consecutive gaps have letters `x, y`, alternation T3 gives
  `W_2 = P[0][0] + P[0][+d] + P[0][-d] + P[+d][0] + P[-d][0] + P[+d][-d] + P[-d][+d]`
  (7 of the 9 letter pairs; `(+d, +d)` and `(-d, -d)` are the excluded ones) and `Z_2 = P[0][0]`.
  So `C_3` is the **two-point residue dictionary** of the old gap sequence.
- `r = 4`: `C_4 = W_3 + Z_3`, the **three-point residue dictionary** with the same rule applied
  to consecutive nonzero letters; `Z_3` is the all-pad triple count.
- **`C_r = 0` for `r > L + 1`**, since `W_m = 0` for `m > L` by the definition of `L`.

**Which quantities of the old machine enter.** `n_J` needs `C_{J-1}, C_J, C_{J+1}`, and the whole
order distribution therefore needs `W_m` and `Z_m` for `m = 0, ..., J_max - 1` - i.e. the
`(m+1)`-point residue dictionaries of the old gap sequence for `m + 1 <= J_max - 1 + 1`, with the
deepest one that is not identically zero at `m = L = J_max - 2`. Nothing else about `M` enters:
not its sizes, not its period, not its record.

### 2.4 Corollary (the tail form, and the merge count in closed form)

Summing Theorem 1 from `J` upwards telescopes:

> `sum_{J' >= J} n_{J'} = C_{J-1} - C_J` for every `J >= 1`.

So the number of **merges** is `C_1 - C_2 = 2N - 2A_0 - A_d`, the number of gaps of order `>= 3`
is `C_2 - C_3`, and so on. Two consequences that were measured facts before:

- merge fraction `= (2N - 2A_0 - A_d)/((q'-2)N) = 2/(q'-2) - C_2/((q'-2)N)`. merge_forest's
  prediction "merge fraction at most `2/(q'-2)`" is therefore an identity with an **exact
  deficit**, and the deficit is `C_2/((q'-2)N)`, i.e. the old spectrum's mass on the pad and
  letter classes.
- max order `= 1 + D` where `D` is the largest `r` with `C_r > 0`, and by Theorem 2
  `D = L + 1 = A_kill`, so **max order `= L + 2 = J_max`** - merge_forest's measured "`1 + D`" and
  docs/proofs/10's word reduction are the same statement, joined here.

### 2.5 Theorem 3 (the second moment)

> `sum_J J^2 n_J = 2 sum_{r >= 0} C_r - q' N = q' N + 4 N + 2 sum_{r >= 2} C_r`,
>
> hence with `S := (1/N) sum_{r >= 2} C_r`,
>
> `E[J] = q'/(q'-2)`,  `E[J^2] = (q' + 4 + 2S)/(q'-2)`,
> `Var(J) = 2[(q'-4) + S(q'-2)]/(q'-2)^2`.

*Proof.* Theorem 1 is equivalent to `sum_J max(J - r, 0) n_J = C_r` for every `r >= 0` (invert the
second difference; the `r = 0` case is `sum J n_J = q' N`). Sum over `r >= 0`: for a fixed `J`,
`sum_{r >= 0} max(J - r, 0) = J + (J-1) + ... + 1 = J(J+1)/2`, so
`sum_r C_r = (1/2)(sum J^2 n_J + sum J n_J)`. Rearranging and using `sum J n_J = q' N` gives the
first form; splitting off `C_0 = q' N` and `C_1 = 2N` gives the second. The moments follow by
dividing by `sum n_J = (q'-2) N`. `qed`

**What the variance says.** The teeth enter the variance **only** through `S`, and
`S >= C_2/N = (2A_0 + A_d)/N` with the remaining terms tiny. The teeth-free floor is
`2(q'-4)/(q'-2)^2`, attained exactly when the new gear can never strike two consecutive openings.
So: the first moment is blind to the teeth (merge_forest), the second moment sees them through
one number, and that number is the old spectrum's mass on three residue classes.

### 2.6 Theorem 4 (the size side)

> For every value `v`,
>
>     m_{M+q'}(v)  =  sum_{J = 1}^{J_max}  sum_{n : gap(n) + ... + gap(n+J-1) = v}  eps_J(n),
>
> where `eps_J(n)` depends only on the LETTERS of those `J` gaps:
>
> - `J = 1`: `eps_1 = q' - 2` if the gap is `PAD`, `q' - 3` if `UP` or `DOWN`, `q' - 4` otherwise;
> - `J >= 2`: let `T` be the set of starting teeth `t_1 in {-1, +1}` for which the **middle** word
>   (the letters of gaps `2 .. J-1`, length `J-2`) reads consistently, and for `t_1 in T` let
>   `t_{J-1}` be the tooth it ends on. Then
>
>       eps_J = # { t_1 in T :  first letter not in {PAD, (UP if t_1 = +1 else DOWN)}
>                              and last letter not in {PAD, (DOWN if t_{J-1} = +1 else UP)} }.
>
> In particular `eps_J in {0, 1, 2}` for `J >= 2`, and `eps_J = 0` whenever the first or last gap
> of the window is `PAD`.

*Proof.* A gap of `M + q'` of size `v` is, by the merge law, a run of `J` consecutive old gaps
`g_1, ..., g_J` of total `v` whose `J - 1` interior openings `x_1, ..., x_{J-1}` are all struck in
one copy and whose endpoints `x_0`, `x_J` are not. Counting over (copy, position) pairs and using
the `N`-periodicity of the gap sequence in the tiled index (Lemma A) as in Theorem 2, the number
of copies realising this for a fixed position is a function of the letters alone:

- the interiors are struck in a common phase iff the middle word `g_2, ..., g_{J-1}` is legal, and
  the admissible phases are in bijection with the admissible starting teeth `t_1` at `x_1` (2 if
  the middle word is all pad, 1 if legal with a nonzero letter, 0 otherwise) - Theorem 2's count;
- in the coordinates of 0.1, `x_0 = x_1 - g_1` is struck iff `g_1 = (t_1 - t_0)c` for some
  `t_0 in {-1, +1}`, i.e. iff `g_1 = 0` (`PAD`) or `g_1 = t_1 d`, which is `UP` when `t_1 = +1`
  and `DOWN` when `t_1 = -1`;
- symmetrically `x_J = x_{J-1} + g_J` is struck iff `g_J = 0` or `g_J = -t_{J-1} d`, i.e. `PAD`,
  or `DOWN` when `t_{J-1} = +1` and `UP` when `t_{J-1} = -1`.

Requiring both endpoints unstruck removes the offending `t_1` and leaves `eps_J`. For `J = 1`
there is no interior; `x_0` and `x_1` are each struck in exactly 2 of the `q'` copies and their
struck sets overlap in `2`, `1` or `0` copies according as `g_1` is `PAD`, a nonzero letter, or
neither (chain law), so the count of copies striking neither is `q' - 4 + |overlap|`. Summing over
positions of the given span gives the theorem; the range `J <= J_max` is Theorem 2's
`C_r = 0` for `r > L + 1`, which makes `eps_J = 0` identically for `J > J_max`. `qed`

**Consistency with the count side.** `sum_v m_{M+q'}(v)` restricted to a single `J` is `n_J`, so
Theorem 4 is a second, independent closed form for the order distribution; that it agrees with
Theorem 1 at all 8 rungs is one of the checks in 3.1. Summing `eps_1` over all gaps also
reproduces `n_1 = (q'-4) N + 2 A_0 + A_d`, the `J = 1` coefficient of
`docs/novel/paired-holt-recursion.md`, which is where the two overlap.

---

## 3. The count side: verification and the moments

### 3.1 The identity and the closed forms, 8 rungs, three routes, 0 exceptions

Route A is the closed form (`C_r` from the letter dictionary, then the second difference); route B
is the local weight (`n_J = sum_n eps_J`, Theorem 4); route C is a direct copy-by-copy build of
the tiled period that never mentions a word. Routes B and C are independent of the derivation of
route A.

| rung | `N` | `q'` | `L` | `J_max` | `C_0` | `C_1` | `C_2` | `C_3` | `C_4` | `A_0` | `A_d` | `n_1 .. n_5` | B = A | C = A |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5->7 | 3 | 7 | 1 | 3 | 21 | 6 | 2 | 0 | 0 | 0 | 2 | 11, 2, 2, 0, 0 | yes | yes |
| 7->11 | 15 | 11 | 0 | 2 | 165 | 30 | 0 | 0 | 0 | 0 | 0 | 105, 30, 0, 0, 0 | yes | yes |
| 11->13 | 135 | 13 | 1 | 3 | 1,755 | 270 | 6 | 0 | 0 | 0 | 6 | 1,221, 258, 6, 0, 0 | yes | yes |
| 13->17 | 1,485 | 17 | 1 | 3 | 25,245 | 2,970 | 72 | 0 | 0 | 0 | 72 | 19,377, 2,826, 72, 0, 0 | yes | yes |
| 17->19 | 22,275 | 19 | 1 | 3 | 423,225 | 44,550 | 1,088 | 0 | 0 | 0 | 1,088 | 335,213, 42,374, 1,088, 0, 0 | yes | yes |
| 19->23 | 378,675 | 23 | 2 | 4 | 8,709,525 | 757,350 | 11,870 | 62 | 0 | 86 | 11,698 | 7,206,695, 733,672, 11,746, 62, 0 | yes | yes |
| 23->29 | 7,952,175 | 29 | 1 | 3 | 230,613,075 | 15,904,350 | 243,822 | 0 | 0 | 6 | 243,810 | 199,048,197, 15,416,706, 243,822, 0, 0 | yes | yes |
| 29->31 | 214,708,725 | 31 | 3 | 5 | 6,655,970,475 | 429,417,450 | 8,025,014 | 13,000 | 4 | 2,090 | 8,020,834 | 5,805,160,589, 413,380,422, 7,999,018, 12,992, 4 | yes | yes |

`C_2 = 2 A_0 + A_d` at 8 of 8; `C_3 = W_2 + Z_2` at 8 of 8 (values `0, 0, 0, 0, 0, 62, 0, 13,000`);
`C_4 = W_3 + Z_3` at 8 of 8 (values all `0` except `4` at the top rung). `L` reproduces
docs/proofs/10's measured row `1, 1, 1, 2, 1, 3` at m11..m29 exactly, and the max order equals
`L + 2 = 1 + D` at all 8 rungs (`3, 2, 3, 3, 3, 4, 3, 5` against `D = 2, 1, 2, 2, 2, 3, 2, 4`).

The top rung is worth stating on its own: **the whole order distribution of a machine with
6,226,553,025 gaps is `C_0 = 6,655,970,475`, `C_1 = 429,417,450`, `C_2 = 8,025,014`,
`C_3 = 13,000`, `C_4 = 4`, `C_5 = 0`** - six numbers, of which the last four are word counts of
m29's gap letters, and the second difference returns
`5,805,160,589 / 413,380,422 / 7,999,018 / 12,992 / 4`.

### 3.2 The moments, exact at all 8 rungs

| rung | `sum J^2 n_J` | `q'N + 4N + 2 sum_{r>=2} C_r` | `E[J]` | `q'/(q'-2)` | `E[J^2]` | `Var` | floor `2(q'-4)/(q'-2)^2` | `S` |
|---|---|---|---|---|---|---|---|---|
| 5->7 | 37 | 37 | 1.400000 | 1.400000 | 2.466667 | 0.506667 | 0.240000 | 0.666667 |
| 7->11 | 225 | 225 | 1.222222 | 1.222222 | 1.666667 | 0.172840 | 0.172840 | 0.000000 |
| 11->13 | 2,307 | 2,307 | 1.181818 | 1.181818 | 1.553535 | 0.156841 | 0.148760 | 0.044444 |
| 13->17 | 31,329 | 31,329 | 1.133333 | 1.133333 | 1.406465 | 0.122020 | 0.115556 | 0.048485 |
| 17->19 | 514,501 | 514,501 | 1.117647 | 1.117647 | 1.358688 | 0.109553 | 0.103806 | 0.048844 |
| 19->23 | 10,248,089 | 10,248,089 | 1.095238 | 1.095238 | 1.288715 | 0.089169 | 0.086168 | 0.031510 |
| 23->29 | 262,909,419 | 262,909,419 | 1.074074 | 1.074074 | 1.224493 | 0.070858 | 0.068587 | 0.030661 |
| 29->31 | 7,530,881,411 | 7,530,881,411 | 1.068966 | 1.068966 | 1.209478 | 0.066791 | 0.064209 | 0.037437 |

`7 -> 11` is the clean case: no two consecutive openings can be struck at all (`C_2 = 0`,
`S = 0`), and the variance sits exactly on the teeth-free floor. At every other rung the variance
exceeds the floor by `2S/(q'-2)`, and `S` exceeds `C_2/N` only through `C_3` and `C_4`: by 0.52% at `19->23` and 0.16% at `29->31`, and by nothing at all at the other six rungs, where `C_3 = 0`.

### 3.3 The variance is small, and that is the teeth

The excess `Var - floor = 2S/(q'-2)` is `0.267, 0, 0.008, 0.006, 0.006, 0.003, 0.002, 0.003` at
the eight rungs. In merge_forest's language `A_d` is dominated by the spectrum's mass on the two
letters `a = 2u_{q'}` and `b = q' - a`; the real teeth have `3a = q' -+ 1`, so `a` is about
`q'/3` and the letters are never the common small sizes. That is the whole mechanism of the small
variance, and 3.4 measures it against the family.

### 3.4 The counterfactual family: the variance percentile

20 random members plus the real machine at rungs `11 -> 13` and `13 -> 17` (teeth at `+- v_g`,
`v_g` uniform in `1..(g-1)/2`; the alignment-rules section 5 family, the same generator and seed
as merge_forest 2.7, so the rows are comparable).

| rung | real `Var` | family min | family median | family max | percentile of the real machine | percentile of `n_3` |
|---|---|---|---|---|---|---|
| 11->13 | 0.156841 | 0.148760 | 0.175696 | 0.224181 | **0.190** | **0.190** |
| 13->17 | 0.122020 | 0.117262 | 0.136925 | 0.167273 | **0.190** | **0.190** |

- the closed form `Var = 2[(q'-4) + S(q'-2)]/(q'-2)^2` reproduces the measured variance for
  **42 of 42** member-rungs;
- the branching identity and the second-moment identity hold for **42 of 42** member-rungs, i.e.
  they are teeth-free structural facts, not properties of the real machine;
- the two percentiles are equal because the variance is an affine function of `S` and `S` is
  dominated by `C_2`, which is what drives `n_3`; so merge_forest's "triple-fusion deficit at
  percentile 0.20" is, exactly, **a variance deficit**: the real machine's order distribution is
  the tightest fifth of its family, and by the closed form the only thing it can be tight about
  is the old spectrum's mass on the pad and letter classes.

---

## 4. The size side

### 4.1 The whole next spectrum from the old machine's window dictionary

Theorem 4, evaluated over the old machine's realised windows, at every rung:

| rung | `J_max` | `F(M+q')` | `|Spec|` | `sum m` | `sum v m` | period | vs direct build |
|---|---|---|---|---|---|---|---|
| 5->7 | 3 | 5 | 4 | 15 | 35 | 35 | EXACT |
| 7->11 | 2 | 7 | 7 | 135 | 385 | 385 | EXACT |
| 11->13 | 3 | 11 | 10 | 1,485 | 5,005 | 5,005 | EXACT |
| 13->17 | 3 | 18 | 17 | 22,275 | 85,085 | 85,085 | EXACT |
| 17->19 | 3 | 25 | 23 | 378,675 | 1,616,615 | 1,616,615 | EXACT |
| 19->23 | 4 | 34 | 33 | 7,952,175 | 37,182,145 | 37,182,145 | EXACT |
| 23->29 | 3 | 43 | 41 | 214,708,725 | 1,078,282,205 | 1,078,282,205 | EXACT |
| 29->31 | 5 | 58 | 55 | 6,226,553,025 | 33,426,748,355 | 33,426,748,355 | every corpus gate |

"EXACT" means every multiplicity of the directly built machine is reproduced, value by value, not
merely the totals. At `29 -> 31`, where no direct build exists inside the budget, the formula
reproduces `F = 58`, `|Spec| = 55`, the absent set `{54, 56, 57}`, and
`m(4) = 398,923,200`, `m(6) = 299,202,120`, `m(24) = 174,704`, `m(36) = 3,152`, `m(41) = 134`,
`sum m` and `sum v m` - every recorded corpus value.

### 4.2 The least depth is `J_max`, and the truncation error is a closed form

Truncating the sum at `J <= K` and comparing to the true spectrum:

| rung | `J_max` | total error at `K = 1` | `K = 2` | `K = 3` | `K = 4` | `K = 5` |
|---|---|---|---|---|---|---|
| 5->7 | 3 | 4 | 2 | **0** | 0 | - |
| 7->11 | 2 | 30 | **0** | 0 | - | - |
| 11->13 | 3 | 264 | 6 | **0** | 0 | - |
| 13->17 | 3 | 2,898 | 72 | **0** | 0 | - |
| 17->19 | 3 | 43,462 | 1,088 | **0** | 0 | - |
| 19->23 | 4 | 745,480 | 11,808 | 62 | **0** | 0 |
| 23->29 | 3 | 15,660,528 | 243,822 | **0** | 0 | - |
| 29->31 | 5 | 421,392,436 | 8,012,014 | 12,996 | 4 | **0** |

**The least `K` is exactly `J_max = L + 2` at all 8 rungs** - values `3, 2, 3, 3, 3, 4, 3, 5` -
and `K = J_max - 1` is wrong at every rung, so the depth is sharp, not merely sufficient. The
errors are not arbitrary: by 2.4 the missing mass at depth `K` is `sum_{J > K} n_J = C_K - C_{K+1}`,
and every entry of the table is that number (e.g. at `29 -> 31`,
`C_1 - C_2 = 421,392,436`, `C_2 - C_3 = 8,012,014`, `C_3 - C_4 = 12,996`, `C_4 - C_5 = 4`). So the
count side predicts the size side's truncation error exactly.

### 4.3 How the dictionary grows

`|D_J(M)|`, the number of distinct realised `J`-tuples of consecutive gap sizes:

| machine | `J=1` | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| m5 | 2 | 3 | 3 | 3 | - | - |
| m7 | 4 | 9 | 11 | - | - | - |
| m11 | 7 | 25 | 50 | 73 | - | - |
| m13 | 10 | 52 | 161 | 323 | - | - |
| m17 | 17 | 133 | 478 | 1,281 | - | - |
| m19 | 23 | 221 | 1,216 | 4,489 | 12,490 | - |
| m23 | 33 | 429 | 3,135 | 15,696 | - | - |
| m29 | 41 | 730 | 7,184 | 45,854 | **208,668** | **720,527** |

Growth in `J` at a fixed machine is geometric with a **falling** ratio (m29: 17.8, 9.8, 6.4, 4.6,
3.5), so the dictionary is far from a free product - the legality and the sizes both bite. Growth
in the rung at fixed `J` is a factor 1.7-2.6 per rung at `J = 2` (3, 9, 25, 52, 133, 221, 429, 730)
against a period growing by a factor `q'`, which is the known "dictionary grows 3-5x per gear
while the period grows 30x" of `docs/novel/qualifying-dictionary-rung.md`, reproduced here at a
different arity. `|D_5(m29)| = 208,668` matches the independently computed value on record.

### 4.4 The extremes, per `J`

The largest span a `J`-run of positive weight attains is `Q*_J(M; q')`, and `F(M+q') = max_J Q*_J`
is the record law read on the dictionary:

| rung | `Q*_1` | `Q*_2` | `Q*_3` | `Q*_4` | `Q*_5` | `F(M+q')` | `F(M)` | `F(M) + q'` |
|---|---|---|---|---|---|---|---|---|
| 5->7 | 2 | 3 | 5 | - | - | 5 | 2 | 9 |
| 7->11 | 5 | 7 | - | - | - | 7 | 5 | 16 |
| 11->13 | 7 | 11 | 8 | - | - | 11 | 7 | 20 |
| 13->17 | 11 | 16 | 18 | - | - | 18 | 11 | 28 |
| 17->19 | 18 | 25 | 25 | - | - | 25 | 18 | 37 |
| 19->23 | 25 | 31 | 33 | 34 | - | 34 | 25 | 48 |
| 23->29 | 34 | 39 | 43 | - | - | 43 | 34 | 63 |
| 29->31 | 43 | 55 | 58 | 55 | 55 | 58 | 43 | 74 |

The record is never carried by `J = 1` (a corollary of `F` strictly increasing). It is carried by
`J = 2` at `7 -> 11` and `11 -> 13`, by `J = 2` and `J = 3` jointly at `17 -> 19` (both attain 25),
by `J = 3` at `5 -> 7`, `13 -> 17`, `23 -> 29` and `29 -> 31`, and by `J = 4` at `19 -> 23`. At the
top rung `Q*_2 = 55 = F_2(m29)` on record and `Q*_3 = 58` beats it: the record at `29 -> 31` is a
triple, and depths 4 and 5 exist but fall back to 55.

---

## 5. Closure

### 5.1 The hierarchy closes at depth `m J_max`, and that is a theorem

> **Theorem 5.** The depth-`m` dictionary of `M + q'` **with multiplicity** is determined exactly
> by the depth-`K_m` window dictionary of `M` with multiplicity, where `K_m` is the largest
> number of consecutive old gaps spanned by `m` consecutive new gaps, and `K_m <= m J_max`.

*Proof.* `m` consecutive gaps of `M + q'` beginning at a surviving opening `x_0` are, by the merge
law, a run of `K = J_1 + ... + J_m` consecutive old gaps in which the `m + 1` openings
`x_0, x_{J_1}, x_{J_1+J_2}, ...` survive and all others are struck, all in one copy. As in Theorem
4 this is a condition on the letters of those `K` gaps and the number of copies realising it is a
function of those letters; the sizes of the `m` new gaps are partial sums of the `K` old sizes.
Each `J_i <= J_max`, so `K <= m J_max`. `qed`

Measured, by the **z-walk** - a vehicle that uses no word theory at all: for each position and
each `z in Z_{q'}` it reads directly which of the following openings survive (an opening at offset
`o` is struck iff `z + o in {0, d} (mod q')`, and `z` running over `Z_{q'}` runs over the copies
exactly once). Its output is gated against the directly built machine.

| rung | `J_max` | `K_1` | `K_2` | `K_3` | `2 J_max` | `3 J_max` | `|depth-2 dict|` | gate at `m = 1, 2, 3` |
|---|---|---|---|---|---|---|---|---|
| 5->7 | 3 | 3 | 4 | 7 | 6 | 9 | 9 | EXACT |
| 7->11 | 2 | 2 | 4 | 5 | 4 | 6 | 25 | EXACT |
| 11->13 | 3 | 3 | 4 | 5 | 6 | 9 | 52 | EXACT |
| 13->17 | 3 | 3 | 4 | 5 | 6 | 9 | 133 | EXACT |
| 17->19 | 3 | 3 | 4 | 6 | 6 | 9 | 221 | EXACT |
| 19->23 | 4 | 4 | 6 | 7 | 8 | 12 | 429 | EXACT |
| 23->29 | 3 | 3 | 5 | 6 | 6 | 9 | 730 | EXACT |

Reading the table:

- `K_1 = J_max` at 7 of 7, as it must be.
- **`K_m` does not grow with the rung.** It tracks `J_max`: the rungs with `J_max = 3` have
  `K_2 = 4, 4, 4, 4, 5` and the one rung with `J_max = 4` has `K_2 = 6`. The hierarchy is **not** open in
  the rung - the depth needed at `19 -> 23` (`J_max = 4`) exceeds the depth needed at the much
  bigger `23 -> 29` (`J_max = 3`).
- `K_2 - K_1 = 1, 2, 1, 1, 1, 2, 2` and `K_3 - K_2 = 3, 1, 1, 1, 2, 1, 1`: each further new gap
  costs one or two more old gaps of depth, well short of the worst case `J_max` per new gap. (The
  `5 -> 7` row is degenerate: the machine has 3 gaps, so a 3-run of new gaps wraps its whole
  period.)
- the brief's scan range for the depth-2 question was `J = 2, 3, 4, 5`; at `19 -> 23` the least
  depth is `K_2 = 6`, OUTSIDE that range, so the honest answer at that rung is "none of 2, 3, 4, 5
  suffices; 6 does", and it is the rung with the largest `J_max` on the ladder, not the largest
  machine. At every other rung the least depth is 4 or 5, inside the range.
- so the answer to "does the depth-`J` dictionary of `M + q'` follow from the depth-`(J + k)`
  dictionary of `M` for a bounded `k`" is **yes, with `k = K_m - m <= m(J_max - 1) = m(L + 1)`**,
  and the bound is uniform in the rung once `L` is bounded. `L` bounded is docs/proofs/10's open
  rider; this branch does not touch it. What it does settle is that **nothing else** stands in the
  way: there is no second, independent source of depth growth.

### 5.2 Where this meets the project's existing transfer (stopped in a paragraph)

`docs/novel/dictionary-monotonicity-onset.md` already has a **dictionary transfer** `M -> q'`
that computes a certified SUPERSET of `D_4(M + q')` from `D_4(M)`, exact below a sharp span (the
inflation onset `13, 15, 17, 25, 31, 41, 53, 68`) and over-generating above it. Theorem 5 is not a
sharpening of that result and does not compete with it: the transfer's input is the **set**
`D_4(M)` plus an order-4 closure over walks the machine need not realise, and its over-generation
is exactly the closure's fault; Theorem 5's input is the **multiset of realised windows**
`D_K^#(M)`, a strictly richer object, and with it the transfer is exact at every span with no
onset at all. The one-line moral, which is what the comparison is worth: **the onset measures the
cost of forgetting which windows are realised together, not the cost of the free phase.** The
depth-0 monotonicity lemma (`D_m(M) subset D_m(M+q')` for `q' > 2(m+1)`) is cited and used
nowhere here. Sub-question stopped.

---

## 6. Toward the root: what a closed hierarchy would and would not give

State it exactly, since the branch was pre-registered not to claim a route.

**What it gives.** Theorems 4 and 5 make the spectrum's evolution a **recursion of fixed depth**:
`Spec(M + q')` with all multiplicities is a function of `D_{J_max}^#(M)`, and
`D_m^#(M + q')` is a function of `D_{K_m}^#(M)`. So the ladder of spectra can in principle be run
without ever constructing a period - which is what the m31 column of 4.1 actually is: a
6.2-billion-gap spectrum computed from a 214-million-window dictionary. And the record comes with
it, because

> `F(M + q') = max_{J <= J_max} Q*_J(M; q')`,

the record law (docs/proofs/09) read on the dictionary; the table of 4.4 is that maximum taken
apart by `J`.

**What it does not give.** A recursion computes; it does not bound. The budget inequality is
`max_J Q*_J(M) <= F(M) + q'`, a statement about the **extremes** of the dictionary, and nothing
above bounds an extreme of `D_{J_max}^#(M + q')` by the extremes of `D_{J_max}^#(M)`. Three
reasons, each visible in the tables:

1. the recursion's state is the whole dictionary, whose size grows (4.3) - it is a finite-depth
   recursion, not a finite-STATE one, and a bound would need a finite summary of the state that is
   closed under the step;
2. the depth `J_max = L + 2` is bounded only if `L` is, and `L` bounded is open (docs/proofs/10);
   the bare half of `L` is capped forever (docs/proofs/12) and the padded half is not;
3. the max over `J` in the record law is attained at different `J` at different rungs (4.4:
   `J = 2, 2, 3, 2/3, 4, 3, 3` from `7 -> 11` up), so a per-`J` bound would have to hold at every
   `J <= J_max` simultaneously - which is exactly the chain statement, node R1.2.

**The residual, named.** Bound `max span of a word-legal J-run of M` by `F(M) + q'`, for every
`J <= J_max`. That is node R1.2's chain statement, unchanged. The branching identity is the count
side of the same dictionary and it closes; the size side computes but does not close; the extreme
side is untouched. This branch adds no route and says so.

---

## 7. What holds without exception (item 6)

| statement | count | status |
|---|---|---|
| `n_J = C_{J-1} - 2 C_J + C_{J+1}`, `C_0 = q'N` | 8 of 8 rungs, 3 routes; 42 of 42 counterfactual member-rungs | **PROVED** (2.2) + verified |
| `C_r = W_{r-1} + Z_{r-1}` for every `r >= 1` (the pad class counts twice, each nonzero class once) | 8 of 8 rungs, `r = 0..5` | **PROVED** (2.3) + verified |
| `C_1 = 2N`, `C_2 = 2A_0 + A_d`, `C_3 = W_2 + Z_2`, `C_4 = W_3 + Z_3`, `C_r = 0` for `r > L+1` | 8 of 8 | **PROVED** (2.3) + verified |
| `sum_{J' >= J} n_{J'} = C_{J-1} - C_J`; merges `= 2N - 2A_0 - A_d`; merge fraction `= 2/(q'-2) - C_2/((q'-2)N)` | 8 of 8 | **PROVED** (2.4) + verified |
| max order `= 1 + D = L + 2 = J_max` | 8 of 8 | **PROVED** (2.4, joining merge_forest and docs/proofs/10) |
| `sum_J J^2 n_J = q'N + 4N + 2 sum_{r>=2} C_r`; `Var = 2[(q'-4) + S(q'-2)]/(q'-2)^2` | 8 of 8 rungs and 42 of 42 member-rungs | **PROVED** (2.5) + verified |
| mean order `= q'/(q'-2)`, teeth-free | 8 rungs + 42 member-rungs | identity (merge_forest, cited) |
| `m_{M+q'}(v) = sum_J sum_{span v} eps_J`, `eps_J in {0,1,2}` for `J >= 2` | every value at 7 rungs against the direct build; every corpus gate at m31 | **PROVED** (2.6) + verified |
| the least depth that reproduces the next spectrum is exactly `J_max = L + 2` | 8 of 8 (`3, 2, 3, 3, 3, 4, 3, 5`), and `J_max - 1` fails at 8 of 8 | **PROVED** + verified |
| the depth-`K` truncation error is `C_K - C_{K+1}` | 8 of 8 rungs, every `K` | **PROVED** (2.4) + verified |
| depth-`m` dictionary of `M + q'` from depth-`K_m` of `M`, `K_m <= m J_max` | 7 rungs, `m = 1, 2, 3`, EXACT against the direct build at every one | **PROVED** (5.1) + verified |
| `K_m` tracks `J_max`, not the rung | 7 rungs | measured, 0 exceptions |
| `eps_J = 0` whenever the first or last gap of the window is `PAD` | structural | **PROVED** (2.6) |

---

## 8. What is new

1. **The branching identity, proved in general**, with the two boundary cases named and done
   (`C_0 = q'N` forced by the empty window; the cyclic wrap, which needs (H) and gives every
   maximal run exactly one predecessor zero). merge_forest had the identity as a paragraph at
   `r <= 2`; this is the general statement with the proviso identified.
2. **`C_r = W_{r-1} + Z_{r-1}` for every `r`** - the chain-count sequence of a rung is the old
   machine's legal-word count plus its all-pad count, and the factor two that appears in
   `C_1 = 2N` and `C_2 = 2A_0 + A_d` is the all-pad term at every depth. This is what turns
   "`C_2` has a closed form" into "the whole sequence has one", and it is why the entire order
   distribution of the 6.2-billion-gap machine m31 is five numbers.
3. **The tail form `sum_{J' >= J} n_{J'} = C_{J-1} - C_J`**, which makes the merge count
   `2N - 2A_0 - A_d` exactly and converts merge_forest's measured bound "merge fraction
   `<= 2/(q'-2)`" into an identity with the exact deficit `C_2/((q'-2)N)`.
4. **The second moment in closed form**, `sum J^2 n_J = 2 sum_r C_r - q'N`, hence
   `Var = 2[(q'-4) + S(q'-2)]/(q'-2)^2` - exact at 8 rungs and 42 member-rungs. The first moment
   is blind to the teeth; the second sees them through the single number `S`, and the real
   machine's variance percentile on the family is 0.190 at both tested rungs, equal to its `n_3`
   percentile. merge_forest's "triple-fusion deficit at percentile 0.20" is therefore, precisely,
   a **variance** deficit of the order distribution, with a closed form for what the deficit is.
5. **The size side as an exact local formula**: `m_{M+q'}(v) = sum_J sum_{span v} eps_J` with
   `eps_J in {0,1,2}` a function of the window's letters alone, and the explicit rule (the middle
   word's admissible teeth, minus the readings in which a flank letter would strike an endpoint).
   Verified value by value at 7 rungs and against every m31 corpus gate. This is the count
   identity carrying its sizes, and it is what makes the spectrum a dictionary computation.
6. **The depth is exactly `J_max = L + 2`, and the truncation error is `C_K - C_{K+1}`** - the
   count side predicts, in closed form, what a too-shallow dictionary gets wrong.
7. **Closure at depth `K_m <= m J_max`, with `K_m` measured and tracking `J_max` rather than the
   rung** (`K_2 = 4, 4, 4, 4, 6, 5` in rung order over the six non-degenerate rungs, i.e. ordered
   by `J_max` and not by the rung). The hierarchy of dictionaries
   is closed at a depth that is bounded exactly when `L` is; there is no second source of depth
   growth. The comparison with the project's existing over-generating transfer localises the
   onset: it is the cost of forgetting which windows are realised together, not of the free phase.
8. `|D_6(m29)| = 720,527` and `|D_4(m29)| = 45,854` are new; `|D_5(m29)| = 208,668` reproduces the
   value on record by a different vehicle.

**Prior art inside the project.** merge_forest 3.1 (the identity at `r <= 2`, `C_2 = 2A_0 + A_d`,
max order `= 1 + D`, mean order); docs/proofs/05 (copies and phases, chain law, merge law,
legality); docs/proofs/10 (`J_max = L + 2`, `A_kill = L + 1`, chain iff word); docs/proofs/12
(`L_bare <= PSORD <= 5`); docs/novel/paired-holt-recursion.md (the `J = 1` coefficient, which
Theorem 4 reproduces); docs/novel/dictionary-monotonicity-onset.md (the depth-0 lemma and the
inflation onset, compared in 5.2 and not re-derived); docs/novel/renewal-ladder.md is the nearest
prior work on the count side and is a different object - it gives **CRT upper bounds** on joint
qualifying-gap counts computed from per-gear counts without a period, converging to exact as
interior points are restored; `C_r` here is an **identity** in the old machine's realised word
counts, exact by construction and requiring the dictionary as input. The two meet only in that
both count configurations of consecutive gaps with prescribed blocked interiors.

**Prior art outside.** Not checked (no web access this round). The run-length inversion
`exactly-k = (>=k) - 2(>=k+1) + (>=k+2)` is elementary and certainly classical; what is not
obviously classical is its combination with the two-class word count `W_m + Z_m`, which has no
one-class counterpart (in one class every deleted point is `= 0` mod the new prime, so there are
no letters and no alternation).

---

## 9. Verdict

**PROVED, and a FACT about the machine; not a route.** The branching identity is now a theorem in
general, its chain-count sequence has a closed form at every depth in the old machine's letter
dictionary, and the same argument carrying the sizes gives the next machine's entire spectrum
exactly - verified value by value at seven rungs and against every corpus gate at the eighth,
where the machine has 6.2 billion gaps. The evolution of the spectrum is a recursion of depth
`J_max = L + 2`, sharp at every rung, and the dictionary hierarchy closes at depth `m J_max` with
the measured depth much smaller. The moment side is complete to second order in closed form, and
it says the one thing the teeth control at this level is `S`, the old spectrum's mass on the pad
and letter classes; the real machine sits at percentile 0.19 of its family on exactly that.

What it does not do is bound anything. The record is `max_J Q*_J(M)` and the extremes of a
dictionary are not bounded by the extremes of its predecessor by anything proved here; the
residual is node R1.2's chain statement, unchanged and named. The honest position: this branch
converts "the spectrum's evolution" from a scan into a finite-depth computation and closes the
count side completely, and it leaves the size question exactly where it found it.

**Child named** (the lowest-order interaction not yet proved on the way from these parts to the
shape): the extremes as a dictionary statement - *is `max_{J <= J_max} Q*_J(M; q')` bounded by
`F(M) + q'` for every `M`, given that the depth is `L + 2` and the weights `eps_J` are `0, 1` or
`2`?* The new leverage this branch supplies for it is that the whole object is now local: the
maximum is over realised `K`-windows with a `{0,1,2}`-valued weight, so a bound on the extremes is
a statement about the old dictionary's realised windows, not about the period. The natural next
step is the pair `(J, span)` frontier - for each `J`, the largest span with `eps_J > 0` - whose
values are tabulated in 4.4 and whose collapse at `J = 4, 5` at the top rung
(`58 -> 55 -> 55`) is the same collapse merge_forest measured in `Rest(a)`.

---

## 10. Dead ends

- **"The order distribution needs the sizes."** False, and worth recording as the sharp negative:
  `n_J` is a function of the old machine's gap RESIDUES mod `q'` alone (Theorem 2), so two machines
  with the same letter dictionary have the same order distribution whatever their sizes. The sizes
  enter only at Theorem 4.
- **"The factor 2 in `C_2 = 2A_0 + A_d` is a small-`r` artefact."** False: it is `Z_{r-1}`, the
  all-pad term, and it is present at every depth (`C_3 = W_2 + Z_2` with `Z_2` the all-pad pair
  count, `C_4 = W_3 + Z_3`). Nothing special happens at `r = 2`.
- **"The depth-`J_max` dictionary suffices only in the aggregate."** False in the strong direction
  and worth stating: it suffices value by value, but `J_max - 1` fails at every rung, so there is
  no slack to be traded - the pre-registered "least `J`" is exactly `J_max`, never less.
- **"Closure at bounded depth would make the spectrum a finite-state recursion, hence the record
  computable without a scan."** Half true and pre-registered as a negative (P10): the recursion is
  finite-DEPTH but its state is the dictionary, which grows (4.3: 41, 730, 7,184, 45,854, 208,668,
  720,527 at m29). A finite-state recursion would need a finite summary closed under the step, and
  none is exhibited here.
- **"`K_m` grows by one per rung (an open hierarchy)."** Refuted: `K_2 = 4, 4, 4, 4, 6, 5` in rung
  order across the six non-degenerate rungs - ordered by `J_max`, not by the rung; the biggest rung
  tested (`23 -> 29`) needs LESS depth than `19 -> 23`.
- **"The branch strengthens the transfer of `dictionary-monotonicity-onset`."** No: it computes a
  different, richer object (the multiset of realised windows). Stopped in a paragraph rather than
  developed (5.2).
