# 11. The spectrum bound on the alignment depth: `L(M) <= 2 F(M+q')/q' + 1`

## In plain words

The longest grammatical run in the old machine cannot be longer than about twice the next
machine's record divided by the new gear, plus one. The reason: consecutive letters of the
grammar alternate between two sizes that add up to exactly the gear, so a long run must be long
in columns, and a run's total length is bounded by the next record. This turns a counting
question, how many in a row, into a size question, how big the record is. It does not make the
count bounded by a constant, because the record grows faster than the gear.

## Vocabulary

Column `k` is the pair `(6k-1, 6k+1)`; `M` is a finite set of gears, `q'` a gear not in `M`,
`u' = u_{q'}` its tooth value, `a := 2u'`, `b := q' - 2u'` (so `a + b = q'`, `a < b`,
`3a = q' -+ 1`); `c = 6^{-1} mod q'`, so `{2c, -2c} = {a, b}` as residues.  A gap value is
**padded** if `= 0 (mod q')`, **nonzero legal** if `= +-2c`; a **legal word** is a run of
consecutive gaps of `M` all padded or nonzero legal, with the nonzero classes alternating
(file 05 (F)); `L(M)` is the length of the longest legal word realised in `M` (file 10).
`G := F(M + q')` is the record of the next machine.  `T := floor((G - 2)/q')`.

Classical translation: `L(M) + 1` is the largest number of consecutive `M`-rough twin
candidates the single prime `q'` can strike in a row (file 10); the theorem bounds that
combinatorial arity by the metric quantity `2 * (record of M + q') / q'`.

## Statement

**Theorem.**  Let a realised legal word of `M` have `m` letters, of which `p` are padded and
`n = m - p` nonzero.  Then

    (SIMPLE)   m <= 2T + 1,   and letter-aware   m <= 2T + 1 - p;
    (PARITY)   m <= max( 2T,  2 floor((G - 2 - a)/q') + 1 ).

Consequently `L(M) <= 2T + 1 <= 2 F(M+q')/q' + 1`.

## Proof

1. **Class minima.**  A positive integer `= 0 (mod q')` is at least `q'`; one `= +2u'` is at
   least `a` (as `0 < a < q'`); one `= -2u'` is at least `b = q' - a` (as `0 < b < q'`).  So
   each padded letter has value `>= q'`, and each nonzero letter has value `>= a` or `>= b`
   according to its class, with `a < b`.
2. **Alternation.**  In a legal word the nonzero letters alternate between the two classes
   (file 05, T3), so any two consecutive nonzero letters (padded letters in between ignored)
   are one of each class and sum to at least `a + b = q'`.  Pairing the `n` nonzero letters in
   order,

       (sum of nonzero letters) >= floor(n/2) * q' + [n odd] * a,

   and adding the padded letters,

       span(word) >= p q' + floor(n/2) q' + [n odd] a.                       (*)

3. **Attainment.**  A realised word of `m` letters occupies `m + 1` consecutive openings
   `x_1 < ... < x_{m+1}` of `M`.  Take the openings `x_0` before and `x_{m+2}` after; then
   `x_0 < ... < x_{m+2}` is a `(m+2)`-run whose middles are exactly the word, so it is
   word-legal and by the attainment half of file 08, `x_{m+2} - x_0 <= G`.  Both flanks are
   at least 1, so

       span(word) <= G - 2.                                                 (**)

4. **(SIMPLE).**  From (*) and (**), `(p + floor(n/2)) q' <= G - 2`, so, the left side being
   an integer multiple of `q'`, `p + floor(n/2) <= T`.  Then
   `m = n + p <= 2 floor(n/2) + 1 + p <= 2(p + floor(n/2)) + 1 <= 2T + 1`, and keeping `p`
   explicit, `m = 2 floor(n/2) + [n odd] + p <= 2(p + floor(n/2)) + 1 - p <= 2T + 1 - p`.
5. **(PARITY).**  If `n` is even, (*) gives `(p + n/2) q' <= G - 2`, so `p + n/2 <= T` and
   `m = 2(p + n/2) - p <= 2T`.  If `n` is odd, (*) gives `(p + (n-1)/2) q' + a <= G - 2`, so
   `p + (n-1)/2 <= floor((G - 2 - a)/q')` and
   `m = 2(p + (n-1)/2) + 1 - p <= 2 floor((G - 2 - a)/q') + 1`.
6. Finally `2T + 1 <= 2(G-2)/q' + 1 < 2G/q' + 1`.

No use is made of the cover half of realisability, of phase saturation, or of any property of
the gears of `M` beyond the openings being distinct integers with a next and a previous one.

Consequence and its limit, on record: `L` is `O(F/q')`, not `O(1)`; `F(M+q')/q'` is measured
`0.54 .. 2.64` and growing along the corpus, so this does not bound `L` by a constant, and `L`
bounded remains open.

## Status

Kernel: none (written proof only).  Its ingredients: file 08 (attainment; written) and file 05
T3 (`TwoTeeth.spacing_from_lo`, `TwoTeeth.spacing_from_hi`, `WordLegal.legal_iff_noRepeat`;
kernel), `TwoTeeth.teeth_letters` for `a + b = q'`.

Verified computationally: the class minima, `3a = q' -+ 1`, the bound against the measured `L`,
and the accounting (*) and (**) on every realised word on record at the twelve corpus machines
m11..m53 (173 assertion gates), and (SIMPLE), (PARITY), letter-aware at every one of 165,584
rows of the tooth-counterfactual family, zero violations, including the family's `L = 5` member
where (PARITY) equals 5 exactly (`research/lateral_r31.py corpus`, `family`).  Corpus row
(PARITY) `1, 1, 2, 3, 3, 3, 5, 4, 5, 5, 5, 5` against `L = 1, 1, 1, 2, 1, 3, 3, 2, 2, 2, 4, 3`;
tight at m11, m13, m29.

## Relationship to the conjecture

The one theorem on record that bounds the alignment depth `L` by anything, but by `2F(M+q')/q'
+ 1`, which grows along the corpus (`F/q'` measured `0.54 .. 2.64`).  It retired requirement
(B) as posed and does not bound `L` by a constant; closing the budget inequality through it
needs the open padded constant.  The theorem itself has no measured input; its substituted
consequence uses the measured `c_A = 4`.

## Where it is used

It retired the project's requirement (B) as posed ("`L` bounded by an absolute constant") by
replacing it with a linear-in-`G` bound; substituted into the increment chain it gives the
budget inequality under `8F(M) <= q'^2 - (F_2 - F + 12) q' + 16` -- conditional on the open
padded case (`c_A = 4` is a literal-letter constant).

## Source

Lateral round 31, item 84; `docs/novel/spectrum-bound-on-L.md`;
`docs/proof-search/alignment-rules.md` 3.8.
