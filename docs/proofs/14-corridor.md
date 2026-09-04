# 14. The corridor of gears 5 and 7: `E_35`, completeness, the 32-cap, exclusion laws, padding

## In plain words

Gears 5 and 7 together leave open exactly fifteen of every thirty-five columns, and every
opening of every machine sits on one of those fifteen positions. That alone forbids many
arrangements of openings outright, for every machine and at every scale: certain pairs of
neighbouring gap sizes, four openings evenly spaced by any larger gear, two equal gear-sized
gaps in a row for half of all gears, and any run of more than thirty-two consecutive columns
each containing a prime. A small completeness lemma says that for shapes of up to five openings
no other gear can add a forbidden arrangement. What these facts constrain is where things sit,
never how large a gap can be.

## Vocabulary

Column `k` is the pair `(6k-1, 6k+1)`; gear `q` strikes `k` iff `k = +-u_q (mod q)`.  Gear 5
has teeth `{1, 4}` (`u_5 = 1`) and gear 7 has teeth `{1, 6}` (`u_7 = 1`).  A column `k >= 1` is
**exposed** if neither 5 nor 7 divides either member.  A **shape** is a finite list of offsets
`X = {0 = o_0 < o_1 < ... < o_{n-1}}`; the shape **occurs** at column `k` in machine `M` if
`k + o` is an opening of `M` for every `o in X`; the **carrier** of a list of gaps
`g_1, ..., g_l` is the set of residues `r` mod 35 such that every partial sum
`r, r + g_1, r + g_1 + g_2, ...` is in `E_35` (mod 35).  A **padded link** of a new gear `q'`
is a pair of consecutive openings of `M` both struck by `q'` in one copy on the same tooth, so
their gap is `= 0 (mod q')` (file 05).

Classical translation: an exposed column is a twin candidate `(6k-1, 6k+1)` with no member
divisible by 5 or 7; everything in this file is what the primes 5 and 7 alone force on the
positions of twin candidates, at every scale.

## Statement

**(a) The exposed set.**  For `k >= 1`, `k` is exposed iff `k mod 35 in`
`E_35 := {0, 2, 3, 5, 7, 10, 12, 17, 18, 23, 25, 28, 30, 32, 33}` (15 residues).  Every
opening of every machine containing 5 and 7 is exposed.

**(b) Endpoint and adjacency laws.**  If `a` and `a + G` are openings then
`a mod 35 in {r in E_35 : (r + G) mod 35 in E_35}`; for `G = 34 (mod 35)` this is
`{3, 18, 33}`.  If `a`, `a + g_1`, `a + g_1 + g_2` are openings then `a mod 35` lies in the
carrier of `(g_1, g_2)`; exactly 294 of the 1225 pairs `(g_1, g_2) mod 35` have empty carrier
and never occur as adjacent gaps in any machine containing 5 and 7.

**(c) Tier A.**  If a chain of openings with consecutive gaps `g_1, ..., g_l` occurs at column
`x` then `x mod 35` lies in the carrier of `(g_1, ..., g_l)`; an empty carrier forbids the
configuration in every machine containing 5 and 7, with no period scan.  Instances: with the
binding word and record of each step, `(7; 4; 7)`, `(11; 6; 11)`, `(18; 13; 18)`,
`(34; 19; 34)`, `(43; 10; 43)` have empty carrier (no word occurrence at 11->13, 13->17, 17->19,
23->29, 29->31 has the old record on both flanks); `(25; 8; 25)` at 19->23 has carrier
`{0, 5, 7, 12}` -- tier A does not close that step.

**(d) Completeness lemma.**  A shape of `n` points can fail to fit gear `q` (no translate avoids
both teeth) only if `q <= 2n`.  Hence a shape of `n` points occurs at some column of a machine
`M` iff it fits every gear `q in M` with `q <= 2n`; for `n <= 5` the mod-35 test (gears 5 and 7)
is the entire obstruction, for `n <= 3` gear 5 alone; gear 11 first matters at `n = 6`, gear 13
at `n = 7`.

**(e) The 32-cap.**  For `k >= 2`, `k = 1 (mod 35)` or `k = 34 (mod 35)` forces both members
composite.  Any 33 consecutive columns from column 2 on contain such a column, so a run of
columns each carrying at least one prime member has length at most 32, at every scale; and any
`W` consecutive columns contain at least `floor(W/33)` columns with both members composite.

**(f) Adjacent-gap exclusion law (mod 5), complete.**  Three openings `k`, `k + g_1`,
`k + g_1 + g_2` in a machine containing gear 5 are impossible when
`(g_1 mod 5, g_2 mod 5) in {(1,1), (1,3), (2,4), (3,1), (4,2), (4,4)}`; for each of the other
19 classes such a triple occurs in every machine.

**(g) AP lemma.**  No four openings of a machine containing gear 5 form an arithmetic
progression with common difference coprime to 5; in particular none with common difference
`q'` for any prime `q' > 5`; more generally four openings at `k + i q'` with the four `i`
distinct mod 5 are impossible.  Openings AP theorem: an arithmetic progression of `L` openings
has common difference divisible by every gear `q <= L + 1` of the machine.

**(h) Padding.**  (Onset gate) a padded link of `q'` inside `M` has interior gap a positive
multiple of `q'` and at most `F(M)`, so padded links exist only if `q' <= F(M)`.  (Count) `p`
padded links in a run of span `S <= F + (5/6) q'` satisfy `6 p q' <= 6F + 5q'`; below onset at
most one; once `F >= (13/6) q'` the budget no longer excludes three.  (Adjacent equal padded
links) three consecutive openings `r, r + q', r + 2q'` are impossible by the corridor iff
`q' mod 35 in {1, 4, 6, 9, 11, 16, 19, 24, 26, 29, 31, 34}` (12 of the 24 invertible classes),
e.g. at `q' = 41`; and the equal shape `(1, 1)` is corridor-impossible iff both unequal shapes
`(1, 2)` and `(2, 1)` are corridor-possible.

## Proof

**(a)**  `5 | 6k - 1` iff `k = 1 (mod 5)`; `5 | 6k + 1` iff `k = 4`; `7 | 6k - 1` iff
`k = 6 (mod 7)`; `7 | 6k + 1` iff `k = 1`.  So `k` is exposed iff `k mod 5 in {0, 2, 3}` and
`k mod 7 in {0, 2, 3, 4, 5}`; by CRT these `3 * 5 = 15` pairs are the residues listed.  An
opening of a machine containing 5 and 7 is struck by neither, hence exposed.

**(b)**  Openings are exposed, and `(a + G) mod 35 = (a mod 35 + G) mod 35`; the adjacency
form is the same with two steps.  For `G = 34`: `r` and `r - 1` both in `E_35` happens for
`r in {3, 18, 33}`.  The count 294 is a finite evaluation of the `35 x 35` table (kernel).

**(c)**  Every partial sum lands on an opening, hence in `E_35`, so `x mod 35` is in the
carrier; if the carrier is empty there is no such `x`.  The instances are finite evaluations.

**(d)**
1. Gear `q` blocks the translate `t + X` iff `t + o = +-u_q (mod q)` for some `o in X`, i.e.
   iff `t in {+-u_q - o : o in X}`, a set of at most `2n` residues.  If `q > 2n` some `t` is not
   in it.
2. `X` occurs at some column of `M` iff for every gear `q in M` there is a residue `t_q` with
   `t_q + X` off the teeth of `q` (by CRT, combine the `t_q` into one `k`; conversely
   `k mod q` is such a `t_q`).  By step 1 only gears `q <= 2n` can fail.  `2n <= 10` for
   `n <= 5` leaves gears 5 and 7; `2n <= 6` leaves gear 5; `11 <= 2n` first at `n = 6`,
   `13 <= 2n` at `n = 7`.

**(e)**
3. `k = 1 (mod 35)`: `k = 1 (mod 5)` gives `5 | 6k - 1` and `k = 1 (mod 7)` gives `7 | 6k + 1`;
   for `k >= 2`, `6k - 1 >= 11 > 5` and `6k + 1 >= 13 > 7`, so both are proper multiples,
   composite.  `k = 34 (mod 35)`: `k = 4 (mod 5)` gives `5 | 6k + 1`, `k = 6 (mod 7)` gives
   `7 | 6k - 1`, and both members exceed 7.
4. The residues `1` and `34` are cyclically `33` and `2` apart, so among any 33 consecutive
   integers one is `= 1` or `= 34 (mod 35)`.  A run of `L >= 33` columns from column `>= 2`
   each with a prime member would contain a column with both members composite.  Packing
   disjoint 33-blocks gives the `floor(W/33)` floor.

**(f)**
5. Openings of a machine containing 5 have `k mod 5 in {0, 2, 3}`.  For each forbidden class
   the three residues `k`, `k + g_1`, `k + g_1 + g_2` (mod 5) cannot all lie in `{0, 2, 3}`;
   the offending residue for each admissible `k` is shown:

       class    offsets mod 5    k = 0      k = 2      k = 3
       (1,1)    0, 1, 2          k+1 = 1    k+2 = 4    k+1 = 4
       (1,3)    0, 1, 4          k+1 = 1    k+4 = 1    k+1 = 4
       (2,4)    0, 2, 1          k+1 = 1    k+2 = 4    k+1 = 4
       (3,1)    0, 3, 4          k+4 = 4    k+4 = 1    k+3 = 1
       (4,2)    0, 4, 1          k+4 = 4    k+4 = 1    k+1 = 4
       (4,4)    0, 4, 3          k+4 = 4    k+4 = 1    k+3 = 1

   (each entry is one of the three residues that lands on a tooth `1` or `4` of gear 5).  A
   direct enumeration of the `25 x 5` table confirms these six classes are exactly the ones
   with no admissible `k mod 5`.
6. Completeness: the shape has `n = 3` points, so by (d) only gear 5 can obstruct it; every
   other class passes gear 5, and CRT places the triple in every machine.  (Occurring as three
   openings, not necessarily as three CONSECUTIVE openings: the exclusion is what is proved.)

**(g)**
7. Four terms `k, k + g, k + 2g, k + 3g` with `g` invertible mod 5 occupy four distinct residues
   mod 5, but openings occupy only the three residues `{0, 2, 3}`.  A prime `q' > 5` is
   invertible mod 5; and `k + i q'` with the `i` distinct mod 5 occupy distinct residues for
   the same reason.
8. Openings AP theorem: if a gear `q` does not divide the common difference `g`, the `L` terms
   `k + ig` occupy `min(L, q)` distinct residues mod `q`; if `L >= q - 1` that is more than the
   `q - 2` open residues, contradiction.  So `L >= q - 1`, i.e. `q <= L + 1`, forces `q | g`.

**(h)**
9. A padded link's two openings are on the same tooth, so their gap `w` satisfies `q' | w`,
   `w > 0`, hence `w >= q'`; and `w` is a gap of `M`, so `w <= F(M)`.
10. `p` padded links consume `>= p q'` of the span `S`; if `6S <= 6F + 5q'` then
    `6pq' <= 6F + 5q'`.  If `F < q'` this gives `p <= 1`; if `13q' <= 6F` then
    `6 * 3 q' <= 6F + 5q'`, so three are not excluded.
11. Three consecutive openings `r, r + q', r + 2q'` reduce mod 35 to `r, r + g, r + 2g` with
    `g = q' mod 35`, all in `E_35`: the carrier of `(g, g)`.  Which of the 24 invertible `g`
    have empty carrier, the count 12, the instance `g = 6` (`q' = 41`), and the dichotomy with
    `(g, 2g)` and `(2g, g)` are finite evaluations (kernel).

## Status

Kernel: `Corridor.Exposed`, `Corridor.exposedSet`, `Corridor.exposed_iff_mem` (a);
`Corridor.endpoint_law`, `Corridor.endpoint_law_34`, `Corridor.allowed3`,
`Corridor.adjacency_law`, `Corridor.no_chain_of_forbidden`, `Corridor.forbidden_first_examples`,
`Corridor.forbidden_pairs_count` (b); `TierA.offsets`, `TierA.carrier`,
`TierA.mem_carrier_of_chain`, `TierA.no_chain_of_carrier_empty`, `TierA.flanked`,
`TierA.no_maximal_flanks`, `TierA.flanks_11_13`, `TierA.flanks_13_17`, `TierA.flanks_17_19`,
`TierA.flanks_23_29`, `TierA.flanks_29_31`, `TierA.flanks_19_23_nonempty`,
`TierA.no_adjacent_maximal_13` (c); `Corridor.exists_class_in_run`,
`Corridor.both_composite_of_class`, `Corridor.both_composite_in_run`,
`Corridor.double_slot_in_run`, `Corridor.prime_adjacent_run_le`, `Corridor.n2_packing` (e);
`TierA.onset_gate`, `TierA.padding_count_le`, `TierA.padding_three_not_excluded`,
`TierA.padding_at_most_one_below_onset`, `TierA.no_adjacent_equal_padded`,
`TierA.no_adjacent_padded_41`, `TierA.equal_padding_forbidden_classes`,
`TierA.equal_padding_forbidden_card`, `TierA.padding_shape_dichotomy` (h).  Written proofs
only: (d) the completeness lemma, (f) the exclusion law, (g) the AP lemma and the openings AP
theorem.

Verified computationally: `E_35`, `A(34) = {3, 18, 33}` and the count 294 against
`research/topgap_endpoint_law.py`; the carriers against `research/flank_tierA_fix.py`; the
exclusion law against full-period censuses m11..m31 (1,589 populated lag-1 cells, none in a
forbidden class); the AP lemma over all `(r, g)` mod 5; the openings AP theorem on full periods
m13..m29 (longest run of equal gaps 3-4, always `g = 5`); the completeness lemma
(`research/corridor_complete.py`); the 32-cap unchanged by adding gears through 23; padding
shapes for every prime to 4000.

Harvest disagreement, resolved by the proof: (d) gives completeness of the mod-35 test for
`n <= 5` points; one harvest recorded only `n <= 3`.  Both agree on the bound `q <= 2n`.

Recorded limitation: corridor arithmetic constrains WHERE configurations sit, never how BIG they
are -- every pair of gap residues mod 35 is within distance 1 of an allowed pair, and lifting
the modulus adds no exclusion tier A did not already give.

## Relationship to the conjecture

Residue facts from two gears, exact and permanent, but by the recorded escape-distance-1
property they constrain positions and never sizes.  Bookkeeping for pruning shapes; no progress
on the size statement; nothing measured enters the theorems.

## Where it is used

The literal and bare caps (files 12, 13) live inside `E_35`; the exclusion and AP laws prune
padding shapes (`j = 2, 4` literal links between two padded links are impossible); the 32-cap
is the one unconditional cap on prime-adjacent runs.

## Source

Constructor round 9 and Formalist round 10 (`proofs/Corridor.lean`); Lateral rounds 15-20
(`docs/novel/corridor-law.md`, `docs/proof-search/lateral.md` items 17-21); Formalist
`proofs/TierA.lean`; `docs/proof-search/alignment-rules.md` 1.7 and 2.5.
