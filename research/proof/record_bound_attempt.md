# A proof of the record bound from the paint mechanism: what it gives, where it stops (2026-09-13)

Owner: given the mechanism for growth, make a proof for the bound. The bound wanted: for every
prime y, the m-line record R_y of the wheel of the gears 5..y (the longest run of consecutive
k with every column (12k - 1, 12k + 1) struck by some gear up to y) is below y^2/12.

## What is proved

Lemma 1 (paint per gear). In R consecutive integers k, the number of k in a given residue class
mod h is at most ceil(R/h). Gear h paints two classes, so it paints at most 2 ceil(R/h) of the R
columns. Proof: the class members are spaced exactly h apart.

Lemma 2 (the budget). If R consecutive columns are all painted by the gears 5..y then
R <= sum over the gears of 2 ceil(R/h). Proof: every column is painted by at least one gear, so
the total paint is at least R, and Lemma 1 bounds each gear's paint.

Theorem (the budget bound). If 2 sum(1/h) over the gears 5..y is below 1, then R_y is at most
the largest R with sum 2 ceil(R/h) >= R, a finite number. Values:

| y | gears | 2 sum 1/h | budget bound | true R_y | y^2/12 |
|---|---|---|---|---|---|
| 5 | 5 | 0.400 | 2 | 2 | 2.08 |
| 7 | 5, 7 | 0.686 | 8 | 4 | 4.08 |
| 11 | 5, 7, 11 | 0.868 | 36 | 7 | 10.08 |
| 13 | 5 .. 13 | 1.021 | none | 9 | 14.08 |
| 17 | 5 .. 17 | 1.139 | none | 17 | 24.08 |
| 23 | 5 .. 23 | 1.331 | none | 34 | 44.08 |
| 29 | 5 .. 29 | 1.400 | none | 43 | 70.08 |

So the mechanism proves R_y < y^2/12 at y = 5 (2 < 2.08, tight), proves finite bounds at 7 and
11 that are weaker than y^2/12, and proves nothing from 13 on: from there 2 sum(1/h) exceeds 1
and the gears bring more paint to any stretch than its length, so counting paint cannot
exclude a full run of any length.

## Why the mechanism cannot be sharpened into the bound by counting

The field shows the true runs use the paint tightly (1.00 to 1.38 per column), far below what
the budget allows (2 sum(1/h) per column, above 1 from y = 13). So the budget is not the
binding constraint; overlaps are forced by something the budget does not see: the teeth of
different gears fall on the same column exactly as the Chinese remainder theorem places them,
4 R/(h h') columns per pair, and so on for triples. Correcting the count for overlaps gives the
sieve: the number of unpainted columns in a stretch of length R is R prod(1 - 2/h) plus a
remainder, and a full run means that count is 0. Lower bounds on that count (which is what
excludes a full run) are the province of lower-bound sieves, and for two residues per prime
(sieve dimension 2) they are positive only for stretches of length at least y^4.27 (the
sieving limit for dimension 2, already recorded on the tree at IV.1 of the earlier document:
the record route is sieve-blocked at exponent 4.27). The bound wanted is at length y^2/12.
Between exponent 2 and 4.27 no counting of teeth, with or without overlaps, decides.

The other direction, the budget with the true overlap, is exact and useless: the total paint
equals R plus the overlap, and nothing bounds the overlap from below without knowing where
the teeth fall, which is the record itself.

## What a proof of the bound would have to be

Not a count. A reason, from the teeth themselves, that the union of the two-class progressions
of the gears 5..y cannot contain y^2/12 consecutive k. Facts available: the teeth are close
pairs (-+12^-1 mod h); the wheel is symmetric about every multiple of every gear, so runs come
in mirror pairs; the record run sits where 5 and 7 interleave without slack and each larger
gear places one double tooth; the record grows like y^2 with constant between 1/20 and 1/12
in reach; the certified records F(q) of the full machine sit at 0.28 to 0.42 of q^2/6. A proof
must produce a structural reason the constant stays below 1/12, and the only mechanism read so
far (the paint budget) does not reach past y = 11.

## Standing

- PROVED: Lemmas 1 and 2; the budget bound; R_5 < 25/12.
- EXACT: R_y for y <= 29 by full-period computation.
- OPEN: R_y < y^2/12 for y >= 7 (true to 29 by computation), which is the one open statement of
  the proof document in the sub-machine's coordinates.
