# 6. The saturation theorem: a far gear always produces the record `F_2(M)`

## In plain words

If the new gear is large compared with the machine's longest gap, larger than three times it,
then it can never strike two neighbouring openings in the same repeat, so every new gap is at
most two old gaps merged, and the biggest such merge always happens somewhere. The new record
is therefore exactly the old machine's best pair of neighbouring gaps, whatever the large gear
is. The catch, stated in the file: along the actual sequence of primes the next gear is always
small compared with the record, so this theorem never applies to the steps that matter.

## Vocabulary

Column `k` is the pair `(6k-1, 6k+1)`; `M` is a finite set of gears with period `P` and
openings `O_M`; `F(M)` is the largest gap between consecutive openings (max-gap convention: the
difference of the two openings); `F_2(M)` is the largest sum of two consecutive gaps, i.e. the
largest `x_2 - x_0` over three consecutive openings `x_0 < x_1 < x_2`.  For a gear `q` not in
`M`, `u_q` is its tooth value (`6u_q = q -+ 1`) and `a = 2u_q` its smaller letter (file 05).

Classical translation: `F(M)` is the paired Jacobsthal-type quantity of `M` in column units; in
the "adjacent frame" of the project's early documents (gear 3 included, positions counted
among all `n` coprime to 6) all lengths are three times the column lengths.

## Statement

**Theorem (saturation).**  Let `q` be a gear not in `M` with `F(M) < 2u_q`.  Then

    F(M + q) = F_2(M).

Since `2u_q >= (q-1)/3`, the hypothesis is implied by `3 F(M) < q - 1`; in the adjacent frame,
where the record is `3F(M)`, that is the record's form `q - 1 > F(M)`.

## Proof

Write `M' = M + q`.

1. **No two consecutive openings are both struck.**  Let `x < y` be consecutive openings of
   `M`, so `y - x <= F(M) < 2u_q`.  If both were struck by `q` (both on its teeth), they would
   differ by at least `2u_q` (file 05, T4: any two distinct struck columns of `q` are at least
   `2u_q` apart).  So at most one of two consecutive openings of `M` is struck.
2. **Upper bound `F(M') <= F_2(M)`.**  Let `y < z` be consecutive openings of `M'`.  By the
   merge law (file 05 (D)) every opening of `M` strictly between `y` and `z` is struck by `q`,
   and `z - y` is the sum of the consecutive gaps of `M` from `y` to `z`.  By step 1 there is at
   most one opening of `M` strictly between `y` and `z`, so `z - y` is one gap of `M` or the sum
   of two consecutive gaps of `M`; in either case `z - y <= F_2(M)` (a single gap is at most
   `F(M) < F_2(M)`).
3. **Lower bound `F(M') >= F_2(M)`.**  Let `x_0 < x_1 < x_2` be consecutive openings of `M`
   with `x_2 - x_0 = F_2(M)`.  By file 05 (A) there is a copy `j` in which `x_1 + jP` is struck
   by `q` (there are exactly two such `j`).  Translation by `jP` preserves the openings of `M`,
   so `x_0 + jP < x_1 + jP < x_2 + jP` are consecutive openings of `M`.  By step 1 the
   neighbours `x_0 + jP` and `x_2 + jP` are not struck, so they are openings of `M'`, and there
   is no opening of `M'` between them (the only opening of `M` between them is struck).  So
   `x_2 - x_0 = F_2(M)` is a gap of `M'`.
4. Steps 2 and 3 give `F(M') = F_2(M)`.

Remark (the hypothesis is sufficient, not necessary).  What step 1 really needs is that no gap
of `M` is congruent to `0`, `+2u_q` or `-2u_q` mod `q` (file 05, T2); `F(M) < 2u_q` is the
simplest way to guarantee it.  For example `M = {5, 7}` has `F = 5`, `F_2 = 7`, gap values
`{1, 2, 3, 5}`, and `F(M + 11) = 7 = F_2(M)` although `2u_11 = 4 <= 5`: the class `4 mod 11`
is simply not a gap of `M`.  The exact rule is the attainment identity of file 08.

Limitation recorded with the theorem: along the consecutive chain `q'` is the next prime after
the top gear of `M` and `q' < F(M)` from machine 13 on (`47` against `354` at m47), so the
theorem never applies to the rungs the budget inequality needs.

## Status

Kernel: none for the equality (written proof only).  Its two ingredients are kernel-checked:
`TwoTeeth.kills_gap_ge` (step 1), `AnchorChain.copy_phase` and `AnchorChain.phase_bijective`
(step 3), and the merge-law bookkeeping `MergeLaw.newgap_le_step` (step 2).

Verified computationally: 48 `(M, q)` pairs with zero violations (`docs/gear-recursion.md`
4b, `research/gear_recursion.py`); `{5, 7}` plus any of `q = 11, 13, 17, 19, 23, 29, 37, 41, 53`
gives `F = 7` in column units (`21` in the adjacent frame), an increment of 2 (`6`) every time.

## Prior art, and what is new

**Leverages.**  File 05's spacing bound T4 and its CRT copy count, and nothing else.  The object
being evaluated is the paired Jacobsthal function of Ziller & Morack (`h_2`, arXiv:1706.00317
and arXiv:1706.03668) in column units; the only exact "add one prime" relations in print are
Hagedorn 2009 Proposition 2.8 (`h(n+1) = 2 w(n) + 2`) and Hajdu & Saradha 2012 Lemma 2.2
(`j(2m) = 2 j(m)` for odd `m`), and neither is of this shape.

**New.**  An exact evaluation of the next record from the old machine under one threshold:
`F(M) < 2u_q` gives `F(M+q) = F_2(M)`, not an inequality and not an asymptotic.  The prior-art
check (2026-08-23, `docs/novel/saturation-theorem.md`) found nothing in print that evaluates a
Jacobsthal-type function of `P q` exactly from `P`-level data under a threshold on `q`, in
either class count; the remark that the real hypothesis is a residue condition, not a size one,
is what file 08 then makes exact.

**Not new.**  The lower half `F(M+q) >= F_2(M)` is elementary and implicit in any gap-merging
picture -- it is file 07 at `r = 1`, and Holt-Rudd's recursion gives it in one class -- and the
same check concedes that in the one-class frame the whole statement looks derivable from their
machinery: it is the statement, not the depth, that was not found.  The regime boundary is
likewise the two-class shadow of a published one-class crossover: Ziller 2020 (arXiv:2007.01808)
records `2 p_{k-1} < h(k-1)` for `k > 18`, the same place where the next prime stops being large
against the record.

## Relationship to the conjecture

Exact and closed, but in a regime provably disjoint from the rungs the conjecture needs: from
machine 13 on the next prime is smaller than the record.  No progress on the open size
statement; no measured input.

## Where it is used

It fixes `F_2(M)` as the second object of the recursion (`F(M+q) >= F_2(M)` always, with
equality in the saturated regime) and marks the boundary of what residue-free arguments deliver:
the same two steps are vacuous once `F(M) >= 2u_q`, which is the regime of every rung.

## Source

`docs/gear-recursion.md` section 4b (statement and proof in the adjacent frame, with the
deletion-spacing lemma of section 4); `docs/novel/saturation-theorem.md`; Constructor R15;
`docs/proof-search/alignment-rules.md` 2.4.
