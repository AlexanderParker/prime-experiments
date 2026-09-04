# 12. The bare-word uniform cap: `L_bare(M) <= PSORD(q' mod 210) <= 5`

## In plain words

Among the grammatical runs there are the bare ones, using only the two smallest letters. Gears
5 and 7 alone forbid long bare runs: how long one can be depends only on the remainder of the
new gear on division by 210, and it is never more than five; for more than half of the possible
remainders it is at most two. This is the first part of the alignment depth that provably does
not grow. It says nothing about runs that use the larger letters, and those are the ones that
carry the record at the bigger machines.

## Vocabulary

Column `k` is the pair `(6k-1, 6k+1)`; `M` is a finite set of gears containing 5 and 7; `q'`
is a prime `>= 11` (in practice the next prime after `M`'s top gear), with tooth value `u'`
(`6u' = q' -+ 1`) and **bare letters** `a := 2u' = (q' -+ 1)/3` and `b := q' - a`.  Gear 5 has
teeth `{1, 4}` and gear 7 has teeth `{1, 6}` (file 02: `u_5 = u_7 = 1`).

- A **bare word** of length `m` is a run of `m` consecutive gaps of `M` whose values are all in
  `{a, b}`.  `L_bare(M)` is the length of the longest bare word that is a legal word (nonzero
  classes alternating, file 05 (F)) realised in `M`.
- For a list of gap values `v_1, ..., v_m` its **offsets** are the partial sums
  `0, v_1, v_1 + v_2, ..., v_1 + ... + v_m` (`m + 1` numbers).  An offset list **fits** gear
  `g` (teeth `{u, g - u}`) if some translate `t in [0, g)` has `t + o mod g` off both teeth for
  every offset `o`.
- For `c` coprime to 210 put `aOfClass(c) := (c - 1)/3` if `c = 1 (mod 6)`, else `(c + 1)/3`,
  and `bOfClass(c) := c - aOfClass(c)`; the **bare alternation** of length `m` at class `c` is
  `(a, b, a, ...)` or `(b, a, b, ...)` with these letters.  `PSORD(c)` is the largest `m` such
  that one of the two bare alternations of length `m` fits both gear 5 and gear 7.
  `S := {c : PSORD(c) <= 2}`.

Classical translation: a bare word is a run of consecutive `M`-rough twin candidates whose
spacings are exactly the two minimal strike spacings of `q'`; the cap says gears 5 and 7 alone
forbid more than `PSORD` such spacings in a row, for every `M` and every scale.

## Statement

**Theorem.**  For every machine `M` containing 5 and 7 and every prime `q' >= 11`,

    L_bare(M) <= PSORD(q' mod 210) <= 5,

and over the 48 classes `c` mod 210 coprime to 210, `PSORD` takes only the values 1, 2, 3, 5:

| `PSORD` | classes mod 210 | count |
|---|---|---|
| 1 | 11, 13, 17, 19, 41, 43, 47, 71, 73, 79, 101, 103, 107, 109, 131, 137, 139, 163, 167, 169, 191, 193, 197, 199 | 24 |
| 2 | 29, 59, 151, 181 | 4 |
| 3 | 1, 23, 31, 61, 67, 89, 97, 113, 121, 143, 149, 179, 187, 209 | 14 |
| 4 | none | 0 |
| 5 | 37, 53, 83, 127, 157, 173 | 6 |

`S` is the union of the first two rows, `|S| = 28`; `S` is closed under `c -> 210 - c`; and
`c in S` iff the literal cap of file 13 satisfies `capC(c) <= 3` (`PSORD(c) = capC(c) - 1` at
every class).

## Proof

1. **A bare legal word is an alternation.**  `a = 2u' = +2c` and `b = q' - 2u' = -2c (mod q')`
   with `c = 6^{-1}`; these are the two nonzero legal classes, distinct residues (`4u' = 0`
   would need `q' | 4u'`, impossible as `0 < 4u' < q'`), and neither is `0 (mod q')`.  So a bare
   word has no padded letter, and legality (no two consecutive nonzero letters equal) forces it
   to be `(a, b, a, ...)` or `(b, a, b, ...)`.
2. **A realised word's offsets are openings.**  If the gaps from opening `x` on are
   `v_1, ..., v_m`, then the columns `x + o` for every offset `o` of `(v_1, ..., v_m)` are
   openings of `M`, hence not on the teeth of gear 5 nor of gear 7.  Taking `t = x mod 5`
   (resp. `x mod 7`) shows the offset list fits gear 5 (resp. 7).  (Kernel:
   `BareAlt.open_of_gapWord`, `BareAlt.fitsB_of_open`; contrapositive `BareAlt.no_gapWord`,
   `BareAlt.no_bare_run`, `BareAlt.no_bare_run_ge` -- if the alternation of length `m` fits
   neither way, no run of `>= m` consecutive gaps of `M` is a bare alternation.)
3. **Fitting depends only on `q' mod 210`.**  Whether an offset list fits gear 5 depends only
   on the offsets mod 5, i.e. on `a mod 5` and `b mod 5`; likewise mod 7.  Write
   `q' = c + 210 k` with `c = q' mod 210`.  Then `c = q' (mod 6)`, so the sign in
   `a = (q' -+ 1)/3` is the sign in `aOfClass(c) = (c -+ 1)/3`, and
   `a - aOfClass(c) = 210k/3 = 70k`; hence `a = aOfClass(c) (mod 70)` and
   `b = bOfClass(c) (mod 70)`.  So `a, b` and the class letters agree mod 5 and mod 7, and the
   alternations of `M` fit exactly when the class alternations fit.  (Kernel:
   `BareAlt.aOfClass_mod_five`, `aOfClass_mod_seven`, `bOfClass_mod_five`, `bOfClass_mod_seven`,
   `bareAdmAB_congr`, `no_bare3_of_class_mem`.)
4. **The cap.**  By steps 1-3, a realised bare legal word of length `m` forces one of the two
   class alternations of length `m` to fit both gears, so `m <= PSORD(c)`.  Fitting is
   downward closed (a prefix of a fitting list fits: `BareAlt.bareAdm_downward`), so `PSORD`
   is a maximum and the bound is `L_bare(M) <= PSORD(q' mod 210)`.
5. **The table.**  `PSORD(c)` for each of the 48 classes is a finite computation: lengths
   `1..9`, two starting letters, `5 * 7` translates, `m + 1` offsets each.  The kernel decides
   it (`BareAlt.psord_le_five`: no class fits a 6-letter alternation, so `PSORD <= 5`;
   `psord_ne_four`; `psord_eq_one_iff`, `psord_eq_two_iff`, `psord_eq_five_iff` list the
   rows; `bareAlt_inadmissible_iff` and `S_card` give `S` and `|S| = 28`; `S_iff_psord`;
   `S_mirror` the closure under `c -> 210 - c`).  Cross-checks in the kernel: the same set by an
   independent vehicle (`AlternationOrder`, offsets built from `3^{-1} mod g` instead of
   integer division: `bareFits_eq_fits`, `bareAdm_eq_survMax`, `psord_succ_eq_psMax`), and
   `inadmissible_iff_capC`: `c in S` iff `LiteralCapTable.capC c <= 3`.

Honest boundary, from the kernel file's header: this bounds `L_bare`, not `L`.  A legal word
may use a padded letter (a gap `= q'`) or a non-bare literal (a gap `= a + q'`, `b + q'`, ...),
and those are untouched.  Measured: `L` exceeds `L_bare` at m37, m41, m43, m53, exactly the
`S`-machines whose record is carried by a word containing the letter `q'`.  Nothing on record
bounds `L_pad`.

## Status

Kernel: `BareAlt.Blocks`, `BareAlt.fitsB`, `BareAlt.fitsB_of_open`,
`BareAlt.not_open_of_not_fits`, `BareAlt.offsets`, `BareAlt.GapWordAt`,
`BareAlt.open_of_gapWord`, `BareAlt.no_gapWord`, `BareAlt.altWord`, `BareAlt.bareFits`,
`BareAlt.bareAdmAB`, `BareAlt.no_bare_run`, `BareAlt.no_bare_run_ge`, `BareAlt.aOfClass`,
`BareAlt.bOfClass`, `BareAlt.bareAdm`, `BareAlt.S`, `BareAlt.bareAlt_inadmissible_iff`,
`BareAlt.S_card`, `BareAlt.S_mirror`, `BareAlt.S_half_mirror`, `BareAlt.bareAdm_downward`,
`BareAlt.psord`, `BareAlt.psord_le_five`, `BareAlt.psord_ne_four`, `BareAlt.psord_eq_one_iff`,
`BareAlt.psord_eq_two_iff`, `BareAlt.psord_eq_five_iff`, `BareAlt.S_iff_psord`,
`BareAlt.psord_succ_eq_psMax`, `BareAlt.bareFits_eq_fits`, `BareAlt.bareAdm_eq_survMax`,
`BareAlt.inadmissible_iff_psMax`, `BareAlt.inadmissible_iff_capC`,
`BareAlt.aOfClass_mod_five`, `BareAlt.aOfClass_mod_seven`, `BareAlt.bOfClass_mod_five`,
`BareAlt.bOfClass_mod_seven`, `BareAlt.no_bare3_of_class_mem` (`proofs/BareAlternation.lean`,
no `decide` on any machine's period); instantiated at m23, m37, m41, m43
(`BareAltInst.m23_no_bare3`, `m23_no_bare_ge`, `m37_no_bare2`, `m37_no_bare_ge`,
`m41_no_bare_offsets`, `m41_no_bare_offsets_B`, `m43_no_bare_offsets`,
`m43_no_bare_offsets_B`).  The assembly of steps 1-5 into the displayed inequality is written.

Verified computationally: `research/bare_alt_r31.py` reproduces `S`, the one-start-letter sets,
the `PSORD` distribution and the table from an independent implementation (6 gates); at
m11..m53, `L_bare <= PSORD` everywhere, tight at m29 and m37/m41/m43; three vehicles sharing no
code agree element for element.

## Relationship to the conjecture

The first uniform, non-growing cap on part of the alignment depth: the bare half.  It does not
touch the padded half `L_pad`, which carries the record at m37, m41, m43, m53 and is unbounded
on record; so it is not progress on the open size statement.  Nothing measured enters the
theorem.

## Where it is used

The first bound on part of `L` that does not grow with the machine; the decomposition
`L = max(L_bare, L_pad)` reduces requirement (B) to "`L_pad` bounded"; at m41 and m43 every
bare decision is free.

## Source

Lateral round 30 item 79 (the observation), Lateral round 31 (`docs/novel/bare-word-uniform-cap.md`),
Formalist round 31 (`proofs/BareAlternation.lean`); `docs/proof-search/alignment-rules.md` 3.8.
