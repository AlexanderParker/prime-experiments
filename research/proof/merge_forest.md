# Node 4.i - THE MERGE FOREST: the ladder's gaps as a rooted forest

Parent: node **4, Genealogy (records recruit runner-ups)**, WEAK on the tree since 2026-09-04
(exact at 8 steps: the record's ancestor one rung down is a runner-up by 2-14, the largest gap
merged one level down at 7 of 8 steps, 1-5 generations; its theory *bounded branching bounds
growth* never tested). What spawns this branch is not that observation on its own but the pair of
facts the exact recursion left behind (`research/proof/spectrum_sum_rule.md` 2.2): **no size is
ever lost** and **every size is born a merge**. Together they say the ladder's gaps have a birth
and a survival, i.e. a genealogy that is exact and cheap to compute. The toolbox entry is set
theory and order theory: the merge law makes the gaps of successive machines a chain of
partitions of the opening set, each coarser than the last, and that chain is a rooted forest.

Scripts in `research/anchor235/r56/`; result outputs in `research/anchor235/r56/results/`
(untracked). Every number this document relies on is written into the document.

---

## 0. Pre-registered (written before any computation of this branch)

### 0.1 The object, defined exactly

Machines `M_0 = {5}`, `M_1 = {5,7}`, ..., `M_8 = {5..31}`, periods `P_n`, openings `O_n`,
`N_n = |O_n| = prod (q - 2)`. A **gap of rung n** is an ordered pair of consecutive openings of
`M_n`; on the unrolled line (all of `Z`, not one period) the gaps of rung `n` partition the
integers into intervals with an opening at each end.

By the merge law (docs/proofs/05 (D)) every gap of `M_n` is a union of consecutive gaps of
`M_{n-1}`, so **the rung-`n` partition is coarser than the rung-`(n-1)` partition**: the family of
all gaps of all rungs, as intervals of the line, is a **laminar family**, and its Hasse diagram is
a **rooted forest** whose roots are the rung-`n` gaps and whose leaves are the gaps of the base
machine `{5}`.

For a gap `G` at rung `n`:

- its **parents** are the gaps of `M_{n-1}` it is the union of (the merge law says there is at
  least one and they are consecutive);
- its **order** `J(G)` is the number of parents: `J = 1` is a **survival**, `J >= 2` a **merge**
  of `J` pieces, born at rung `n`;
- its **birth rung** `b(G)` is `n` if `J >= 2`, else `b` of its unique parent (base gaps are born
  at rung 0);
- its **depth** `D(G) = n - b(G)`, the number of rungs it has survived unchanged;
- its **lineage** is the whole ancestry: the layer word at every rung `b <= k <= n` (the sizes of
  its ancestors at rung `k`, in position order), and the **order profile**
  `(k_5, k_7, ..., k_q)`, `k_g` = the number of ancestors at layer `g` (= the `k_g` of R3.h).

Positions are handled by the identity that makes the whole computation cheap: a gap of `M_n` whose
left end is the column `x` has as parents the gaps of `M_{n-1}` starting at `x mod P_{n-1}` and
running to `x + |G|`. So a lineage is read by reduction modulo the lower periods; no search.

### 0.2 The theory

**T. The forest is thin and shallow, and its thinness is an identity, not a property of the
teeth.** Each old opening is struck by `q'` in exactly two of the `q'` copies (docs/proofs/05 (A)),
so exactly `2 N_{n-1}` junctions are made per period of `M_n` while `(q'-2) N_{n-1}` gaps are
produced: the total branching of a rung is fixed at `2 N_{n-1}` before any tooth is chosen. What
the teeth decide is only how that fixed budget is *distributed* - how often two junctions land
inside one gap (order 3) rather than in two different gaps (two orders of 2). Node 4's theory
"bounded branching bounds growth" is therefore true in its premise and empty in its conclusion:
branching is bounded by the chain law, and the growth of `F` is carried entirely by the *sizes* of
the pieces, which the forest does not bound.

### 0.3 Predictions, each with the number that would refute it

- **F1 (count conservation, exact).** At every rung `5->7 .. 29->31`:
  `|O_n| = (q'-2)|O_{n-1}|`; `sum over gaps of M_n of J = q' |O_{n-1}|`;
  `sum (J - 1) = 2 |O_{n-1}|`; mean order `= q'/(q'-2)`; number of struck openings
  `= 2|O_{n-1}|` and each is the interior of exactly one merge. Length conservation
  `P_n = q' P_{n-1}` and `sum_G |G| = P_n`. REFUTED by one violated count at one rung.
- **F2 (order distribution).** The number of gaps of order exactly `J` at rung `n` satisfies
  `sum_{J>=2} (J-1) n_J = 2 N_{n-1}` exactly; the fraction of gaps that are merges is at most
  `2/(q'-2)` (6.9% at `q' = 31`); the maximum order is `1 + D_{q'}` with `D` the chain depth, and
  is predicted to be `<= 5` at every rung to m31 and `<= 6` (the literal cap) forever. REFUTED by
  an order `>= 7` or by a violated sum.
- **F3 (the record is born fresh, and its parents are not records).** The record of rung `n` has
  order `>= 2` at every rung (one line: `F` is strictly increasing, so no survival can be the
  record - stated as a corollary, not a finding). NEW part: the record's largest immediate parent
  is not a record of its own rung at rungs 23, 29, 31 (predicted max ratio `<= 0.7`) but IS one at
  rungs 17 and 19 (ratio 1.000, from R3.h's `mx/F_g` table, cited not re-derived). REFUTED if
  some rung `>= 23` has a parent of ratio 1.
- **F4 (the record's pieces are recently made).** Mean depth of the record's immediate pieces
  `<= 1.5` at every rung 11..31, and the largest piece has depth 0 (born at rung `n-1`) at a
  majority of rungs. REFUTED by a mean depth above 1.5 at any rung, or depth-0 largest pieces at
  half or fewer of the rungs.
- **F5 (ordinary in size, extraordinary in rarity).** The record's pieces are ordinary by SIZE
  (piece size / `F(M_{n-1})` in [0.2, 0.75] at rungs 23..31, R3.h's 0.3) but extreme by MASS RANK:
  the fraction of gaps of `M_{n-1}` strictly smaller than a piece is above 0.999 for the largest
  piece at every rung 19..31. REFUTED by a largest piece below the 0.99 mass rank.
- **F6 (the top is not closed).** The top of the spectrum at rung `n` does NOT descend entirely
  from the top third of the spectrum at rung `n-1`: the smallest piece of the record is below
  `F(M_{n-1})/3` at m29 and m31 (predicted around 10/43 = 0.23 and 10/34 = 0.29). Stated as a
  prediction of REFUTATION of closure. REFUTED (i.e. closure would hold) if every piece of every
  gap of size `>= F/2` is itself in the top third of the old spectrum.
- **F7 (many families at the top).** The number of distinct order profiles among gaps of size
  `>= F/2` is at least 50 at m23 and grows with the rung; the record's own profile is carried by
  fewer than 1 in 10^4 of those gaps. REFUTED by fewer than 50 profiles at m23.
- **F8 (bounded branching does not bound growth).** The product (max order at rung `n`) x (max
  piece fraction) is above 1 at every rung, so node 4's inequality is satisfied but vacuous; and
  the max piece fraction over ALL gaps of rung `n` (not just the record) reaches 1.0 at every
  rung (a survival of the old record). The only non-vacuous quantity is the max piece fraction
  *for the record*, which is predicted to decrease over rungs 23, 29, 31. REFUTED if it increases.
- **F9 (the family differs only in the tail).** Over 20 family members at rungs 13 and 17 (teeth
  at `+-v_g`, `v_g` uniform in `1..(g-1)/2`, the alignment-rules section 5 family): the mean order
  is EXACTLY `q'/(q'-2)` for every member (an identity, teeth-free, 20 of 20), while the maximum
  order and the count of order `>= 3` vary; the real machine's values sit inside the family range.
  REFUTED by one member whose mean order differs, or by a real value outside the family range on
  both tail statistics.
- **F10 (the residual).** The forest gives `F(M_n) = sum of J <= 1 + D_{q'} pieces`, with `J`
  bounded by the chain law; the one quantity it does not bound is the joint size of the pieces,
  i.e. `max { sum of J consecutive gaps of M_{n-1} whose J-1 interior openings are struck in one
  copy }`, which is exactly `Q*_J(M_{n-1})`, the chain statement of node R1.2. Predicted: the
  forest closes on nothing new; stated so that the branch cannot claim a route it does not have.

**Stop rules.** Any sub-question that reduces to the merge law (docs/proofs/05 (D)-(F)), the chain
law (05 (C)), the record law (09), the exact recursion (`paired-holt-recursion`, R3.i.a) or R3.h's
layer decomposition of the exact records is stopped in one line and cited.

### 0.4 Scorecard

| # | prediction | verdict | evidence |
|---|---|---|---|
| F1 | count and length conservation | CONFIRMED, 0 exceptions, 8 rungs | 2.1 |
| F2 | order distribution and the branching budget | CONFIRMED, and PROVED as an identity | 2.2, 3.1 |
| F3 | record born fresh; parents not records from m23 | CONFIRMED (0.600, 0.676, 0.698 at 23, 29, 31; 1.000 at 7, 11, 17, 19) | 2.3 |
| F4 | record's pieces are recently made | REFUTED as stated (mean depth 1.67 at rung 17); survivor: the largest piece has depth 0 at 7 of 8 rungs | 2.4 |
| F5 | ordinary in size, extreme in mass rank | REFUTED in the letter (0.991 at m23, not 0.999), CONFIRMED in substance (>= 0.99 from rung 17) | 2.4 |
| F6 | the top is not closed | CONFIRMED: record's smallest piece 0.160, 0.294, 0.233 of F_old at 23, 29, 31; all-pieces-in-top-third falls 0.89 -> 0.33 | 2.5 |
| F7 | many families at the top | CONFIRMED: 274 order profiles and 1,336 full lineages at m23; the record's lineage is 1 of 59,940 | 2.6 |
| F8 | bounded branching does not bound growth | CONFIRMED in the first half (J_max x maxfrac = 2.40, 2.03, 3.49 at 23, 29, 31, all vacuous); REFUTED in the second (the record's max piece fraction RISES 0.600 -> 0.676 -> 0.698) | 2.3, 3.3 |
| F9 | the family differs only in the tail | CONFIRMED: mean order exact for 21 of 21 members at both rungs; the real machine sits at the 0.20 percentile in n_3 at both | 2.7 |
| F10 | the residual is the chain statement | CONFIRMED, and made quantitative by the frontier (3.3) | 3.2, 3.3 |

---

## 1. Setup (exact ranges)

Everything is exact: full periods or the exact recursion, integer arithmetic, no sampling.

| object | range | cost | script |
|---|---|---|---|
| the whole forest (every gap, its order, birth rung, ancestor counts at every layer) | m5, m7, m11, m13, m17, m19, m23 on full periods (7,952,175 gaps at m23, period 37,182,145) | 1.9 s | `mf_core.py`, `mf_partA.py`, `mf_partA2.py` |
| the forest at rung 23 -> 29 | the whole m29 period (1,078,282,205 columns, 214,708,725 gaps) streamed as 29 copies of the m23 period; full lineage kept for the 607,862 gaps of size >= 22 = F/2 | 7.9 s | `mf_top.py` pass 1 |
| the forest at rung 29 -> 31 | the whole m31 period (33,426,748,355 columns, 6,226,553,025 gaps) by the merge law over the streamed m29 period; full lineage kept for the 1,774,436 gaps of size >= 29 = F/2 | 33 s | `mf_top.py` pass 2 |
| the branching identity, independently | rungs 5->7 .. 23->29 by direct chain enumeration; rung 29->31 from the m29 spectrum | exact | `mf_chain.py` |
| the rest statistic and the frontier | every rung m7..m31 | exact | `mf_rest.py` |
| the counterfactual family | 20 members plus the real machine at m13 and at m17, full periods each | exact | `mf_family.py` |

**The construction.** Adding `q'` to `M` makes `q'` copies of `M`'s period. Index the old gaps of
the tiled period by `t = j N_old + i` (copy `j`, old gap `i`). The openings of `M'` are exactly the
tiled old openings whose residue mod `q'` avoids the two teeth, so if `newpos` is the sorted list
of tiled indices of the surviving openings then

- the ORDER of the `s`-th new gap is `newpos[s+1] - newpos[s]`;
- its PARENTS are the tiled old gaps `newpos[s] .. newpos[s+1]-1`;
- its BIRTH rung is `n` if the order exceeds 1, else the birth rung of its unique parent.

Ancestor counts at any lower layer use the exact tiled prefix sum
`S_tiled[t] = (t // N) Total + S[t mod N]`, so nothing of size `q' N` is ever materialised; that is
why the top two rungs cost 41 seconds in all.

**Instrument gates, all passed.** The rebuilt m29 spectrum returns `F = 43`, `|Spec| = 41`, absent
`{41, 42}`, `m(4) = 14,178,528`, `m(6) = 10,497,320`, `m(24) = 1,180`, `m(36) = 38`,
`sum m = 214,708,725`, `sum v m = 1,078,282,205`; the m31 spectrum returns `F = 58`, `|Spec| = 55`,
absent `{54, 56, 57}`, `m(4) = 398,923,200`, `m(6) = 299,202,120`, `m(24) = 174,704`,
`m(36) = 3,152`, `m(41) = 134`, `sum m = 6,226,553,025`, `sum v m = 33,426,748,355`, merge mass by
depth `J = 2, 3, 4, 5: 413,380,422 / 7,999,018 / 12,992 / 4`. Every one matches
`spectrum_sum_rule.md`. The record decompositions reproduce `ends_or_middles.md` (R3.h) letter for
letter: the m23 record at `x = 12,694,428` with layer words `19: 4 8 15 7`, `17: 4 3 5 8 7 7`,
`13: 2 2 3 5 7 1 7 3 4`, `11: 2 2 1 2 2 3 7 1 7 3 4`, `7: 2 2 1 2 2 3 2 5 1 5 2 3 2 2`; the m29
record at `x = 200,906,185` with `23: 10 10 23`, `19: 7 3 5 5 23`, `17: 7 1 2 5 5 7 13 3`, and
`k_5 = 26`; the m31 record with layer-29 word `23 10 25` at m29 position 390,658,037, which is copy
1 of the m29 period, i.e. column `390,658,037 + 1,078,282,205 = 1,468,940,242` -- R3.h's recorded
m31 record -- and the second class `18 10 30` at `278,620,515 + 20 P_29 = 21,844,264,615`, R3.h's
second class, `k_5 = 35`. (The m29 start 278,620,515 is also the run that node 2g.i.a recorded as
the one no local certificate could reach.)

## 2. Results

### 2.1 Count and length conservation (item 5)

Each old opening is struck by `q'` in exactly two of the `q'` copies (docs/proofs/05 (A)), so the
whole bookkeeping of a rung is forced before any tooth is looked at. Exact at all 8 rungs:

| rung | `N_old` | `N_new = (q'-2) N_old` | `sum J = q' N_old` | `sum (J-1) = 2 N_old` | mean order | `q'/(q'-2)` |
|---|---|---|---|---|---|---|
| 5->7 | 3 | 15 | 21 | 6 | 1.400000 | 1.400000 |
| 7->11 | 15 | 135 | 165 | 30 | 1.222222 | 1.222222 |
| 11->13 | 135 | 1,485 | 1,755 | 270 | 1.181818 | 1.181818 |
| 13->17 | 1,485 | 22,275 | 25,245 | 2,970 | 1.133333 | 1.133333 |
| 17->19 | 22,275 | 378,675 | 423,225 | 44,550 | 1.117647 | 1.117647 |
| 19->23 | 378,675 | 7,952,175 | 8,709,525 | 757,350 | 1.095238 | 1.095238 |
| 23->29 | 7,952,175 | 214,708,725 | 230,613,075 | 15,904,350 | 1.074074 | 1.074074 |
| 29->31 | 214,708,725 | 6,226,553,025 | 6,655,970,475 | 429,417,450 | 1.068966 | 1.068966 |

Length conservation `sum_G |G| = P_n = q' P_{n-1}` holds at all 8 (37,182,145 at m23;
1,078,282,205 at m29; 33,426,748,355 at m31). The exact number of junctions made per rung is
`2 N_{n-1}` -- each old opening dies in exactly two of the `q'` phases, and each death is the
interior of exactly one merge.

**The forest as a set.** Per period of `{5..23}` the forest has 88,710,327 nodes in seven layers
(22,309,287 / 15,935,205 / 13,037,895 / 11,032,065 / 9,734,175 / 8,709,525 / 7,952,175 at layers
5 / 7 / 11 / 13 / 17 / 19 / 23): 22,309,287 leaves (gaps of `{5}`, `3P/5` of them), 7,952,175
roots, 2.8054 leaves per root. Layer `k` contributes `P_n N_k / P_k` nodes, so the forest's whole
shape is fixed by the `N/P` ratios.

### 2.2 Branching: the order distribution (item 3)

| rung | `n_1` survivals | `n_2` | `n_3` | `n_4` | `n_5` | max order | merge fraction | `2/(q'-2)` |
|---|---|---|---|---|---|---|---|---|
| 5->7 | 11 | 2 | 2 | 0 | 0 | 3 | 0.266667 | 0.400000 |
| 7->11 | 105 | 30 | 0 | 0 | 0 | 2 | 0.222222 | 0.222222 |
| 11->13 | 1,221 | 258 | 6 | 0 | 0 | 3 | 0.177778 | 0.181818 |
| 13->17 | 19,377 | 2,826 | 72 | 0 | 0 | 3 | 0.130101 | 0.133333 |
| 17->19 | 335,213 | 42,374 | 1,088 | 0 | 0 | 3 | 0.114774 | 0.117647 |
| 19->23 | 7,206,695 | 733,672 | 11,746 | 62 | 0 | 4 | 0.093745 | 0.095238 |
| 23->29 | 199,048,197 | 15,416,706 | 243,822 | 0 | 0 | 3 | 0.072938 | 0.074074 |
| 29->31 | 5,805,160,589 | 413,380,422 | 7,999,018 | 12,992 | 4 | 5 | 0.067677 | 0.068966 |

Gaps of order `>= 3`: 2, 0, 6, 72, 1,088, 11,808, 243,822, 8,012,014. The maximum order is
3, 2, 3, 3, 3, 4, 3, 5 -- not monotone in the rung, and 5 at the top rung, well inside the
bare-word cap of 6. `sum_{J>=2} (J-1) n_J = 2 N_old` at all 8 rungs, 0 exceptions.

### 2.3 The record's lineage (item 2)

Every record class of every rung with its immediate parents (`depth 0` = the piece was itself born
at the rung below; mass rank = the fraction of gaps of `M_{n-1}` strictly smaller):

| rung | x | order | pieces | birth gears | depths | size / `F_old` | mass rank |
|---|---|---|---|---|---|---|---|
| 7 | 12 | 3 | 1, 2, 2 | 5, 5, 5 | 0, 0, 0 | 0.500, 1.000, 1.000 | 0.000, 0.333, 0.333 |
| 11 | 150 | 2 | 2, 5 | 5, 7 | 1, 0 | 0.400, 1.000 | 0.200, 0.867 |
| 13 | 122 | 2 | 6, 5 | 11, 11 | 0, 0 | 0.857, 0.714 | 0.941, 0.778 |
| 17 | 117 | 3 | 5, 11, 2 | 7, 13, 5 | 2, 0, 3 | 0.455, 1.000, 0.182 | 0.692, 0.992, 0.127 |
| 17 | 502 | 3 | 5, 6, 7 | 7, 11, 13 | 2, 1, 0 | 0.455, 0.545, 0.636 | 0.692, 0.873, 0.914 |
| 19 | 110 | 2 | 7, 18 | 13, 17 | 1, 0 | 0.389, 1.000 | 0.861, 0.9991 |
| 19 | 26,045 | 3 | 7, 13, 5 | 17, 17, 11 | 0, 0, 2 | 0.389, 0.722, 0.278 | 0.861, 0.9935, 0.625 |
| 23 | 12,694,428 | 4 | 4, 8, 15, 7 | 17, 19, 19, 17 | 1, 0, 0, 1 | 0.160, 0.320, 0.600, 0.280 | 0.499, 0.901, 0.991, 0.809 |
| 29 | 200,906,185 | 3 | 10, 10, 23 | 23, 23, 19 | 0, 0, 1 | 0.294, 0.294, 0.676 | 0.906, 0.906, 0.9990 |
| 31 | 1,468,940,242 | 3 | 23, 10, 25 | 29, 29, 29 | 0, 0, 0 | 0.535, 0.233, 0.581 | 0.99787, 0.88120, 0.99917 |
| 31 | 21,844,264,615 | 3 | 18, 10, 30 | 23, 29, 29 | 2, 0, 0 | 0.419, 0.233, 0.698 | 0.98859, 0.88120, 0.99992 |

Record gap counts and distinct parent words: 2 gaps / 2 words at m7; 4 / 2 at m11; 12 / 2 at m13;
20 / 4 at m17; 20 / 4 at m19; 4 / 2 at m23; 2 / 2 at m29; 4 / 4 at m31. The gap counts reproduce
R3.h's record-stretch counts exactly.

**The record is born fresh at every rung, and that is a one-line corollary, not a finding:** `F` is
strictly increasing (2, 5, 7, 11, 18, 25, 34, 43, 58), so a record can never be a survival; its
order is `>= 2` and its depth 0, at 8 of 8 rungs. The orders are 3, 2, 2, 3, 2, 4, 3, 3.

**Summary per rung:**

| rung | F | `F/F_old` | order | max piece | max piece `/F_old` | min piece `/F_old` | mean piece depth | max-piece depth | mass rank of max piece |
|---|---|---|---|---|---|---|---|---|---|
| 7 | 5 | 2.500 | 3 | 2 | 1.000 | 0.500 | 0.00 | 0 | 0.333 |
| 11 | 7 | 1.400 | 2 | 5 | 1.000 | 0.400 | 0.50 | 0 | 0.867 |
| 13 | 11 | 1.571 | 2 | 6 | 0.857 | 0.714 | 0.00 | 0 | 0.941 |
| 17 | 18 | 1.636 | 3 | 11 | 1.000 | 0.182 | 1.67 | 0 | 0.992 |
| 19 | 25 | 1.389 | 2 | 18 | 1.000 | 0.389 | 0.50 | 0 | 0.9991 |
| 23 | 34 | 1.360 | 4 | 15 | 0.600 | 0.160 | 0.50 | 0 | 0.991 |
| 29 | 43 | 1.264 | 3 | 23 | 0.676 | 0.294 | 0.33 | 1 | 0.9990 |
| 31 | 58 | 1.349 | 3 | 30 | 0.698 | 0.233 | 0.67 | 0 | 0.99992 |

Two rows of that table answer node 4.

- **The record's largest piece is not a record of its own rung from rung 23 on** (0.600, 0.676,
  0.698) but IS one at rungs 7, 11, 17 and 19 (ratio 1.000: the record of `M_{n-1}` survives whole
  into the record of `M_n`). The changeover is exactly where the record set collapses (node 5d).
- **The max piece fraction does not decrease.** Over the three top rungs it RISES: 0.600, 0.676,
  0.698. The pre-registered "pieces stay a fixed fraction of `F`" is refuted; what is true is that
  it stays in a band, 0.60-0.70 from rung 23 and 0.86-1.00 below it.

### 2.4 Depth: how old the pieces are (item 2)

Mean depth of the record's immediate pieces: 0.00, 0.50, 0.00, 1.67, 0.50, 0.50, 0.33, 0.67 at
rungs 7..31 -- the pre-registered `<= 1.5` fails at rung 17, whose pieces `5, 11, 2` have depths
2, 0, 3 (the 5 is a `{5,7}` gap that has survived two rungs, the 2 a gear-5 gap that has survived
three). **The largest piece has depth 0 at 7 of 8 rungs**; the exception is m29, whose largest
piece (23) was born at rung 19 and survived one rung. Over ALL big gaps the same statistic is
0.5940 at m29 and 0.6086 at m31, so on this the record sits at the top of an ordinary tendency
rather than outside it.

Mass rank of the largest piece: 0.333, 0.867, 0.941, 0.992, 0.9991, 0.991, 0.9990, 0.99992. From
rung 17 on the record's largest piece is in the top 1% of gaps by rarity while being 0.60-1.00 of
`F_old` by size; the record's other pieces are ordinary on both scales (the m31 record's middle
piece is 10, the 88th percentile -- and 10 is `a_31`, the top gear's short letter, which is R3.h's
result and is cited, not re-derived). So R3.h's "ordinary lower gaps" is right about size and the
sharper reading is: **ordinary in size, extreme in rarity, and only for the largest piece.**

### 2.5 Recruitment: how far down the old spectrum the pieces come from (item 4)

Over every gap of size `>= F/2`:

| rung | #gaps | mean piece mass rank | min | pieces in the old top third by size | gaps with ALL pieces in the old top third | gaps with a piece `< F_old/4` |
|---|---|---|---|---|---|---|
| 11 | 36 | 0.5126 | 0.000 | 0.9310 | 0.8889 | 0.1111 |
| 13 | 188 | 0.6333 | 0.000 | 0.7290 | 0.5638 | 0.1809 |
| 17 | 914 | 0.7518 | 0.000 | 0.8135 | 0.6871 | 0.1685 |
| 19 | 6,656 | 0.7709 | 0.000 | 0.6930 | 0.5394 | 0.2809 |
| 23 | 59,940 | 0.8207 | 0.000 | 0.5700 | 0.4683 | 0.3656 |
| 29 | 607,862 | 0.8462 | 0.000 | -- | 0.4410 | 0.4532 |
| 31 | 1,774,436 | 0.8283 | -- | -- | 0.3326 | 0.5543 |

**The top of the spectrum is NOT closed.** The fraction of big gaps built entirely from the old top
third falls monotonically 0.89, 0.56, 0.69, 0.54, 0.47, 0.44, 0.33, and the record itself always
recruits from below the top third: its smallest piece is 0.182, 0.389, 0.160, 0.294, 0.233 of
`F_old` at rungs 17, 19, 23, 29, 31. At m31, 55.4% of the big gaps contain a piece below
`F_29/4 = 10.75`. Mean piece size is 14.215 = 0.418 `F_23` at m29 and 0.379 `F_29` at m31.

### 2.6 How many families the top of the spectrum has (item 4)

| rung | threshold `F/2` | #gaps | distinct full lineages | distinct parent words | distinct order profiles | the record's lineage multiplicity |
|---|---|---|---|---|---|---|
| 11 | 4 | 36 | 10 | 8 | 4 | 2 of 36 |
| 13 | 6 | 188 | 30 | 23 | 11 | 2 of 188 |
| 17 | 9 | 914 | 94 | 43 | 26 | 2 of 914 |
| 19 | 13 | 6,656 | 327 | 122 | 80 | 4 of 6,656 |
| 23 | 17 | 59,940 | 1,336 | 249 | 274 | 1 of 59,940 |
| 29 | 22 | 607,862 | -- | -- | 1,127 | -- |
| 31 | 29 | 1,774,436 | -- | -- | 3,672 | -- |

A *full lineage* is the tuple of layer words at every layer; an *order profile* is the vector
`(k_5, ..., k_{q-}, J)` of ancestor counts. The m29 and m31 rows count profiles over the distinct
runs (phases only translate a lineage), on `k_5..k_23` plus the order. The record's own lineage is
unique among the 59,940 big gaps at m23.

**Laminarity, as an order.** Because the merge law merges *consecutive* gaps, the ancestors of a
gap at any layer are a CONTIGUOUS run of that layer's gaps. So a lineage never has to be stored as
a tree: it is a start index and a count at each layer, and the whole ladder is the laminar family
of column intervals `[x, x + |G|)` across rungs, ordered by inclusion. That is the exact sense in
which the forest is an order, and it is what puts the m31 forest inside 33 seconds.

### 2.7 The counterfactual family (item 6)

20 random members plus the real machine at m13 and at m17 (teeth at `+-v_g`, `v_g` uniform in
`1..(g-1)/2`, the alignment-rules section 5 family), each on its full period.

- **Mean order is exactly `q'/(q'-2)` for 21 of 21 members at both rungs.** It is an identity, not a
  property of the real teeth: the two-copies law fixes the total branching before any tooth is
  chosen.
- The tail does depend on the teeth, and the real machine is on the LOW side. `n_3` at m13: real 6,
  family min 0, median 20, max 56, real percentile 0.20. At m17: real 72, family min 19, median 238,
  max 576, real percentile 0.20. Max order: real 3 at both, family 2-3 at m13 and 3 at m17.
- `F` itself: real 11 against family 10-18 (percentile 0.15) at m13; real 18 against 15-25
  (percentile 0.325) at m17. Recruitment (`all pieces in the old top third`): real 0.564 (percentile
  0.40) and 0.687 (percentile 0.70). The record's max piece fraction: real 0.857 (percentile 0.20)
  and 1.000 (percentile 0.775).

So the family answer is sharp: **the forest's size is teeth-free and its branching tail is not.**
The real machine makes about a third as many triple fusions as a typical member.

## 3. Mechanism

### 3.1 The branching identity (new, proved, 8 rungs, 0 exceptions)

A gap of `M + q'` of order `J` has exactly `J - 1` struck interior openings, and they are
consecutive openings of `M`. So it contains exactly `max(J - r, 0)` chains of `r` consecutive struck
openings; and every chain of `r` consecutive openings struck in one copy lies in the interior of
exactly one gap of that copy. Writing `C_r` for the number of (copy, `r`-chain) pairs,

> **`sum_J max(J - r, 0) n_J = C_r` for every `r >= 0`, hence `n_J = C_{J-1} - 2 C_J + C_{J+1}`: the
> order distribution of a rung is the exact second difference of its chain-count sequence.**

`C_0 = q' N` (every old gap lies in one new gap in every copy) and `C_1 = 2 N` (every opening is
struck in exactly two copies), which recovers `sum J n_J = q' N` and `sum (J-1) n_J = 2 N`. The
first non-trivial term has a closed form in the old gap spectrum by residue class:

> **`C_2 = 2 A_0 + A_d`, `A_0` = the number of gaps of `M` divisible by `q'`, `A_d` = the number
> congruent to `+- d_{q'}`** -- the chain law read as a count.

Verified independently (chains enumerated directly from the openings; orders built from the
forest): the inversion returns `n_1..n_5` exactly at all 8 rungs and `C_2 = 2 A_0 + A_d` at all 8
(values `C_2` = 2, 0, 6, 72, 1,088, 11,870, 243,822, 8,025,014). At the top rung, from
`N_29 = 214,708,725`, `A_0 = m_29(31) = 2,090` and `A_d = m_29(10) + m_29(21) = 8,020,834`, so
`C_2 = 8,025,014`, `C_3 = 13,000`, `C_4 = 4`, and the inversion returns
`n_1..n_5 = 5,805,160,589 / 413,380,422 / 7,999,018 / 12,992 / 4`: the entire order distribution of
a machine with 6.2 billion gaps, from five numbers. And

> **max order `= 1 + D_{q'}`** with `D` the chain depth (the largest `r` with `C_r > 0`): 3, 2, 3, 3,
> 3, 4, 3, 5 against `D` = 2, 1, 2, 2, 2, 3, 2, 4 -- 8 of 8.

This settles node 4's premise and refutes its conclusion. Branching IS bounded, by an identity that
has nothing to do with the sizes; and the family result has its mechanism here. `A_d` is dominated
by `m(a) + m(b)` for the two letters `a = 2u_{q'}`, `b = q' - a`. The real teeth have
`3a = q' -+ 1`, so `a` is about `q'/3` and the letters are never small sizes; a family member with
`v_g` near `(g-1)/2` has `a = 1`, so `A_d` picks up `m(1)` and the triple fusions multiply. The
seven m17 members whose top gear's short letter is 1 or 2 (`v_17` = 1 or 8) have `n_3` = 212, 378,
378, 432, 480, 506, 576, every one above the real machine's 72 and six of the seven above the
family median 238. That is the whole of the family's tail effect.

### 3.2 What the forest gives the record, and what it does not (item 6)

The forest turns the record into

    F(M + q') = max over gaps G of ( largest piece of G + rest of G ),   rest = the sum of the others,

with the order of `G` bounded by `1 + D_{q'}` and hence, by the bare-word cap, by 6 forever. The
largest piece is a gap of `M`, so it is at most `F(M)`, and therefore

> **`rest(G) <= q'` for every gap would imply the budget inequality `F(M+q') <= F(M) + q'` in one
> line.**

That is the forest's natural strengthening, and it is the exact residual: nothing on the tree bounds
`rest`. Measured, max rest is 3, 2, 5, 11, 13, 19, 23, 34 at rungs 7..31 against
`q'` = 7, 11, 13, 17, 19, 23, 29, 31 -- **true at 7 of 8 rungs and FALSE at 29 -> 31**, where rests
of 32, 33 and 34 occur (on 6, 4 and 2 gaps). So the strengthening is dead as a route, and the live
residual is not a bound on `rest` alone but on the pair (largest piece, rest).

Stated in the tree's own language: `Rest(a)` is `Q*_J(M)` restricted to the runs whose largest gap
is `a`, so the forest reduces the record to node R1.2's chain statement with one extra coordinate,
and adds nothing that bounds it. That is the honest verdict on the branch as a route.

### 3.3 The frontier (new): where the budget inequality actually lives

For each value `a` of the largest piece let `Rest(a)` be the largest rest over gaps whose largest
piece is `a`. Then `F(M + q') = max_a (a + Rest(a))`, exactly. At the top three rungs:

| rung | `a` at which `a + Rest(a)` peaks | peak | `a / F_old` at the peak | `Rest(F_old)` | `F_old + Rest(F_old)` |
|---|---|---|---|---|---|
| 19->23 | 15 | 34 = F | 0.600 | 5 | 30 < 34 |
| 23->29 | 23 | 43 = F | 0.676 | 5 | 39 < 43 |
| 29->31 | 25 and 30 | 58 = F | 0.581 and 0.698 | 2 | 45 < 58 |

Lower down, `Rest(F_old)` is 3, 2, 3, 7, 7 at rungs 7, 11, 13, 17, 19, and at rungs 7, 11, 17 and 19
the peak IS at `a = F_old` (the old record extended). So:

> **A gap that swallows the old record whole can gain at most `Rest(F_old)` = 3, 2, 3, 7, 7, 5, 5, 2
> further columns; 8 of 8 rungs, and the value falls at the top three (5, 5, 2).**

The full m31 frontier, `a -> Rest(a)` with the sum in brackets: 10->19(29), 11->20(31), 12->22(34),
13->23(36), 14->23(37), 15->24(39), 16->25(41), 17->25(42), 18->27(45), 19->28(47), 20->28(48),
21->34(55), 22->33(55), 23->32(55), 24->28(52), 25->33(58), 26->25(51), 27->28(55), 28->23(51),
29->20(49), 30->28(58), 31->18(49), 32->23(55), 33->20(53), 34->16(50), 35->20(55), 36->9(45),
37->11(48), 38->7(45), 39->3(42), 40->7(47), 43->2(45).

The curve rises to the record at `a = 25` and `a = 30` and then collapses. Both ends of the frontier
are far below `F`: large `a` with a tiny rest (43->2), small `a` with a large rest (10->19). This is
the branch's contribution toward the root, as a mechanism rather than a bound -- the record is
neither the biggest available piece plus something, nor many small pieces; it is made at an interior
point of the frontier at `a` about 0.6-0.7 of `F_old`, and the reason a bigger `a` cannot be used is
that `Rest(a)` collapses there. Why it collapses is not shown here: that is the child this branch
names.

## 4. What is new

1. **The branching identity** `n_J = C_{J-1} - 2 C_J + C_{J+1}` with `C_0 = q'N`, `C_1 = 2N`,
   `C_2 = 2 A_0 + A_d`: the order distribution of a rung is the second difference of its chain-count
   sequence, and its first three terms are closed forms in the old gap spectrum by residue class.
   Proved in a paragraph, verified independently at 8 rungs including the 6.2-billion-gap rung.
   Prior art inside the project: the recursion of `paired-holt-recursion` gives the `J = 1`
   coefficient and the two agree where they overlap (`n_1 = (q'-4) N + 2 A_0 + A_d`); the ladder over
   `r` and its inversion are not on the register. Prior art outside: not checked (no web access).
2. **Mean order is teeth-free** -- `q'/(q'-2)` exactly, 21 of 21 family members at two rungs and 8 of
   8 real rungs -- so every difference between the real machine and its family lives in the tail, and
   the tail is governed by `A_d`, i.e. by how common the two letters are as gap sizes. The real
   machine makes about a third as many triple fusions as a typical member (percentile 0.20 at both
   rungs tested).
3. **The rest statistic and its frontier.** `size = largest piece + rest` is exact; `rest <= q'`
   would give the budget inequality and holds at 7 of 8 rungs, failing first at 29 -> 31 (34 > 31);
   `Rest(F_old) <= 7` at 8 of 8 and falls to 2 at the top rung; the frontier peaks at
   `a/F_old` = 0.600, 0.676, 0.581-0.698 at the three top rungs.
4. **The record's birth and depth data.** Orders 3, 2, 2, 3, 2, 4, 3, 3; the largest piece has depth
   0 at 7 of 8 rungs (exception m29); the largest piece IS a record of its own rung at rungs 7, 11,
   17, 19 and is not from rung 23 on; the mass rank of the largest piece is above 0.99 from rung 17.
   Node 4's "the ancestor is a runner-up by 2-14" is replaced by an exact statement: the ancestor is
   the largest piece; from rung 23 it sits at 0.60-0.70 of `F_old`; and its rarity, not its size, is
   what is extreme.
5. **Family counts at the top of the spectrum**: 10, 30, 94, 327, 1,336 distinct full lineages at
   m11..m23 and 274 / 1,127 / 3,672 order profiles at m23 / m29 / m31; the top is not closed
   (all-pieces-in-the-old-top-third falls 0.89 -> 0.33 across the rungs).
6. **Laminarity as the computational fact**: ancestors at every layer are a contiguous run, so a
   lineage is a (start, count) pair per layer; that is what makes the exact m31 forest a 33-second
   computation rather than an impossible one.

## 5. Verdict

**FACT, exact, with one new identity and one new object; not a route.** The forest settles node 4
completely, and in the direction opposite to its theory. Branching IS bounded, by an identity that
does not involve the teeth, and bounding it bounds nothing: `J_max x (max piece fraction)` is 2.40,
2.03 and 3.49 at the three top rungs -- above 1, hence vacuous as a growth bound. What the forest
does contribute is the reformulation `F = max_a (a + Rest(a))` and the measured frontier, which
localises the whole open question in one function `Rest(a)` and shows that the record is made at an
interior point of it, with `Rest(F_old) <= 7` at every rung and falling. Node 4 should move from
WEAK to FACT with its theory marked refuted; the live residual is the frontier, which is node
R1.2's chain statement with one extra coordinate.

**Child named** (the interaction to prove, in the machine's terms): *why does `Rest(a)` collapse as
`a` approaches `F(M)`?* Exactly: bound the largest sum of gaps of `M` adjacent to a gap of size `a`
whose separating openings are simultaneously struck by `q'`, as a function of `a`; the measured
frontier says it falls from 34 at `a = 21` to 2 at `a = 43` at rung 31, and the budget inequality is
the statement that `a + Rest(a) <= F(M) + q'` on the whole curve. This is 5b's adjacency repulsion
(the suppression law) asked at the top of the spectrum and with the chain law attached, which is a
combination no branch on the tree has taken.

## 6. Dead ends

- **"The record's ancestors are never records of their own rung."** False: at rungs 7, 11, 17 and 19
  the record's largest piece is exactly `F(M_{n-1})` (ratio 1.000). It becomes true only from rung 23
  (0.600, 0.676, 0.698), i.e. when the record set collapses.
- **"The record's pieces are recently made."** Refuted at rung 17: mean piece depth 1.67 (pieces
  `5, 11, 2` of depths 2, 0, 3). What survives is the largest piece, depth 0 at 7 of 8 rungs.
- **"The record's max piece fraction decreases."** Refuted: it rises 0.600 -> 0.676 -> 0.698 over
  rungs 23, 29, 31.
- **"`rest <= q'`, hence the budget inequality."** Refuted at the first rung where it would have
  mattered: max rest 34 against `q' = 31` at 29 -> 31. It holds at rungs 7..29 with 0 exceptions and
  dies at 31.
- **"Bounded branching bounds growth" (node 4's theory).** Refuted as an inequality with content: it
  holds trivially and the bound it gives exceeds `F_old` at every rung.
- **The 0.999 form of "extraordinary in rarity".** Refuted at m23 (0.991011); the correct threshold
  is 0.99, and at that threshold it holds from rung 17.

## 7. What holds without exception (item 7)

| statement | count | status |
|---|---|---|
| `N_new = (q'-2) N_old`, `sum J = q' N_old`, `sum (J-1) = 2 N_old`, `sum size = P` | 8 of 8 rungs | proved (docs/proofs/05 (A)) + verified |
| `n_J = C_{J-1} - 2 C_J + C_{J+1}` (the branching identity) | 8 of 8 rungs | proved here + verified independently |
| `C_2 = 2 A_0 + A_d` | 8 of 8 rungs | proved here + verified |
| max order `= 1 + D_{q'}` | 8 of 8 rungs | proved (chain law) + verified |
| mean order `= q'/(q'-2)` exactly | 8 real rungs + 21 of 21 family members at 2 rungs | identity |
| ancestors at every layer are a contiguous run (laminarity) | every gap of every rung built | proved (merge law) |
| the record has order `>= 2` and depth 0 | 8 of 8 rungs | corollary of `F` strictly increasing |
| `Rest(F_old) <= 7`: a gap containing the old record gains at most 7 | 8 of 8 rungs; 3, 2, 3, 7, 7, 5, 5, 2 | measured |
| the record's largest piece has depth 0 | 7 of 8 (fails at m29) | measured |
| `rest <= q'` | 7 of 8 (fails at 29 -> 31, 34 > 31) | measured, refuted |
