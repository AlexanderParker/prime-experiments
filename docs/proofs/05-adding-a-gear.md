# 5. Adding a gear: copies, phases, the hit and chain laws, the merge law and its grammar

## In plain words

When a new gear is added, the old machine's cycle repeats as many times as the new gear has
positions, and in each repeat the new gear's strikes fall on the old openings in a different
place: every possible placement exactly once, and each old opening is struck in exactly two of
the repeats. A gap of the new machine is made by merging neighbouring gaps of the old machine
whenever all the openings between them are struck. Two old openings can both be struck in one
repeat only if their distance leaves one of three particular remainders on division by the new
gear, and inside a run of struck openings the distances alternate between two letters in a
strict rhythm, each letter at least a third of the gear. This file writes down that grammar.

## Vocabulary

Column `k` is the pair `(6k-1, 6k+1)`.  `M` is a finite set of gears with period
`P = prod q`; its openings form a `P`-periodic, unbounded subset `O_M` of the naturals; a
**gap** of `M` is the difference of two consecutive openings, `F(M)` the largest gap and
`F_2(M)` the largest sum of two consecutive gaps.  A new gear `g` (a prime `>= 5`, not in `M`,
so `gcd(P, g) = 1`) is added: `M' := M + g`.  Write `c := 6^{-1} mod g`, so the teeth of `g`
are `T := {c, -c} = {u_g, -u_g}` (file 02), and `d := 2c`.  The two **letters** of `g` are
`a := 2u_g` and `b := g - 2u_g`; `a + b = g`, `a < b`, and `{a, b}` are the least positive
residues of `+-d`.

Classical translation: adding gear `g` deletes from the twin-candidate pairs those with a member
divisible by `g`; the residues below are residues of `k`, i.e. of `(6k-1)/6` rounded, not of the
members.

## Statement

**(A) Copies and phases.**  The period of `M'` is `Pg`, made of `g` **copies** of the period
of `M`: copy `j` (`0 <= j < g`) is the block of columns `x + jP`, `0 <= x < P`.  For an
opening `x` of `M`, the column `x + jP` is struck by `g` iff `x mod g in {r_j, r_j + d}` where
`r_j := -c - jP (mod g)`.  The map `j -> r_j` is a bijection of `Z_g`: the `g` copies realise
every **deletion phase** `r in Z_g` exactly once.  Each opening of `M` is struck in exactly two
copies (one per tooth).

**(B) Hit law.**  For a column `x`, let `next_M(x)` be the least opening of `M` above `x` and
`next_{M'}(x)` the least opening of `M'` above `x`.  Then `next_{M'}(x) = next_M(x)` iff
`next_M(x)` is not a tooth of `g` (`next_M(x) mod g not in T`).

**(C) Chain law.**  For residues `x, y in Z_g`: there is `r` with `x, y in {r, r + d}` iff
`y - x in {0, d, -d}`.  Hence two openings `x < y` of `M` are both struck by `g` in some copy
iff `y - x = 0, +d` or `-d (mod g)`, and a finite set of openings is struck entirely in some
copy iff all its pairwise differences are `0` or `+-d (mod g)`.

**(D) Merge law.**  Let `y < z` be consecutive openings of `M'`.  Then `y, z` are openings of
`M`, every opening of `M` strictly between them is struck by `g`, and `z - y` is the sum of the
consecutive gaps of `M` from `y` to `z`.  Every gap of `M'` is therefore a gap of `M` or a
**merged J-run** (Lean: `MergedWindow`; a run of gaps, not the certified window): a sum of `J >= 2` consecutive gaps of `M` whose `J - 1` interior openings are
all struck by `g`.

**(E) Grammar of a struck run.**  Let `x_0 < x_1 < ... < x_k` be consecutive openings of `M`,
all struck by `g` (in one copy, i.e. all with residues in `T` after one translation), with
spacings `w_i = x_i - x_{i-1}`.

- **T1 (alphabet).**  The residues by which two struck columns can differ are `0`, `+d`, `-d`,
  whose least positive representatives are `g`, `a = 2u_g`, `b = g - 2u_g`; `a + b = g` and
  `3a = g -+ 1`.
- **T2 (residue necessity).**  Every `w_i` is `= 0`, `a` or `b (mod g)`, and a positive gap in
  one of these classes is at least `a`.
- **T3 (alternation).**  Read the tooth of each `x_i`.  A spacing `= 0` keeps the tooth; a
  spacing `= b` occurs only from tooth `u_g` to tooth `-u_g`; a spacing `= a` only from `-u_g`
  to `u_g`.  Hence the nonzero classes among `w_1, ..., w_k` strictly alternate `a, b, a, ...`
  (spacings `= 0` being transparent), two consecutive spacings in the same nonzero class never
  occur, and two consecutive nonzero-class spacings sum to at least `a + b = g`.
- **T4 (spacing).**  Any two distinct struck columns of `g` differ by at least `a = 2u_g`.
- **T5 (fuel cap).**  `x_k - x_0 >= k a`; equivalently `k <= (x_k - x_0)/a`.

**(F) Legal words.**  Call a gap **padded** if its value is `= 0 (mod g)`, **up** if
`= +2c`, **down** if `= -2c`, and **illegal** otherwise.  Read a word of letters
`{pad, up, down}` with a current tooth `t in {+c, -c}`: `pad` keeps `t`; `up` requires
`t = -c` and sets `t = +c`; `down` requires `t = +c` and sets `t = -c`.  A word is **legal** if
some starting tooth makes the reading consistent.  Then: a word is legal iff no two consecutive
nonzero letters are equal; equivalently, writing `up = +1`, `down = -1`, `pad = 0`, iff all its
prefix sums lie in an interval of length 1.  And a list of residues `x_1, ..., x_m` lies in
`{r + c, r - c}` for some `r` iff its consecutive differences form a legal word
("killable iff legal").

**Corollary (R39).**  Every gap of `M'` is at most `max(F_2(M), max_{J >= 3} Q_J(M))`, where
`Q_J(M)` is the largest sum of `J` consecutive gaps of `M` whose `J - 2` interior gaps are all
`>= a`.

## Proof

**(A)**

1. `x + jP = +-c (mod g)` iff `x = -c - jP` or `x = c - jP = (-c - jP) + 2c`, i.e.
   `x in {r_j, r_j + d}`.
2. `P` is invertible mod `g`, so `j -> -c - jP` is a bijection of `Z_g`.
3. For fixed `x`, `x + jP = c` has exactly one solution `j` mod `g`, and `x + jP = -c` exactly
   one; they differ because `c != -c` (`2c = 0` would give `6c = 0 != 1`).  So `x` is struck in
   exactly two copies.

**(B)**

4. `next_{M'}(x)` is an opening of `M` above `x` that is not a tooth of `g`.  If
   `next_M(x)` is not a tooth, it is such an opening and the least of them, so the two agree.
   If `next_M(x)` is a tooth, then `next_{M'}(x) != next_M(x)`, and in fact
   `next_{M'}(x) > next_M(x)`.

**(C)**

5. (`=>`)  `x, y in {r, r+d}`: the four cases give `y - x in {0, d, -d}`.  (`<=`)  If
   `y - x in {0, d}` take `r = x`; if `y - x = -d` take `r = y`.
6. For a set: if all pairwise differences are in `{0, +-d}`, fix `x_0`; every other element is
   `x_0`, `x_0 + d` or `x_0 - d`, and `x_0 + d`, `x_0 - d` cannot both occur since their
   difference `2d` is not in `{0, +-d}` (`2d = 0`, `d = 0`, `3d = 0` are all impossible:
   `3d = 6c = 1 != 0` and `2d = 4c`, `d = 2c` are units times `c`).  So the set lies in
   `{x_0, x_0 + d}` or `{x_0 - d, x_0}`.  "Struck entirely in copy `j`" means all residues
   lie in `{r_j, r_j + d}`, and by step 2 every `r` is some `r_j`.

**(D)**

7. Openings of `M'` are openings of `M`.  An opening of `M` strictly between `y` and `z` is not
   an opening of `M'` (they are consecutive), so it is struck by `g`.  The gaps of `M` from `y`
   to `z` telescope to `z - y`.

**(E)**

8. T2: the residues of `x_{i-1}` and `x_i` are in `{u_g, g - u_g}`.  The four cases give
   `w_i = 0` (same tooth), `w_i = (g - u_g) - u_g = g - 2u_g = b` (from `u_g` to `g - u_g`), or
   `w_i = u_g - (g - u_g) = 2u_g = a (mod g)` (from `g - u_g` to `u_g`).  A positive gap
   `= 0 (mod g)` is `>= g > a`; `= a` is `>= a`; `= b` is `>= b > a`.
9. T1: `+-d = +-2c`, and `{2c, -2c} = {2u_g, -2u_g}` since `c = +-u_g`; least positive
   representatives `a`, `b`.  `3a = 6u_g = g -+ 1`.
10. T3: by step 8 a spacing of class `b` moves the tooth from `u_g` to `-u_g` and a spacing of
    class `a` moves it back; a spacing of class `0` keeps it.  After a class-`b` spacing the
    tooth is `-u_g`, from which the next nonzero-class spacing must be of class `a`; and
    conversely.  So nonzero classes alternate, and two consecutive nonzero-class spacings, one
    of each class, sum to at least `a + b = g`.
11. T4: two distinct struck columns differ by a positive amount in class `0`, `a` or `b`
    (step 8), hence by at least `a`.
12. T5: `x_k - x_0 = w_1 + ... + w_k >= k a` by T4.

**(F)**

13. (Legal iff no repeat.)  If the reading is consistent, an `up` leaves the tooth at `+c`, so
    the next nonzero letter must be `down`, and vice versa: no two consecutive nonzero letters
    agree.  Conversely, if nonzero letters alternate, start with tooth `-c` when the first
    nonzero letter is `up` and `+c` when it is `down` (either if there is none); the reading
    stays consistent because each nonzero letter flips the tooth to the one the next nonzero
    letter requires.
14. (Prefix-sum form.)  With the tooth encoded as `0` for `-c` and `1` for `+c`, reading a
    letter adds its value to the encoded tooth, so the reading is consistent iff the partial
    sums from the starting value stay in `{0, 1}`.
15. (Killable iff legal.)  If `x_i - r = t_i c` with `t_i in {+1, -1}`, then
    `x_{i+1} - x_i = (t_{i+1} - t_i) c in {0, 2c, -2c}` and the letter word is exactly the
    reading of the tooth sequence `t_i`, hence legal.  Conversely a legal word read from `t_1`
    gives teeth `t_1, ..., t_m`; put `r := x_1 - t_1 c`; then inductively
    `x_{i+1} - r = x_i - r + (t_{i+1} - t_i)c = t_{i+1} c`.

**Corollary.**  By (D) a gap of `M'` is a sum of `J >= 1` consecutive gaps of `M` with all
interior openings struck.  For `J <= 2` it is at most `F_2(M)` (a single gap is at most
`F(M) < F_2(M)`).  For `J >= 3` each interior gap has both endpoints struck, so by T2 it is
`>= a`, and the sum is at most `Q_J(M)`.

## Status

Kernel (all machine-free unless noted): `AnchorChain.copy_phase`, `AnchorChain.phase_bijective`,
`AnchorChain.teeth_eq_phase` (A); `AnchorChain.hop_zero` (B, the direction "not a tooth implies
equality"); `AnchorChain.chain_law` (C, both directions); `AnchorChain.no_two_up`,
`AnchorChain.no_two_down` (T3, algebraic half); `MergeLaw.MergedWindow`,
`MergeLaw.interior_gap_mod` (T2), `MergeLaw.floor_of_mod` (the floor `>= 2u`),
`MergeLaw.sub_mod_eq`, `MergeLaw.windowSum_telescope`, `MergeLaw.newgap_le`,
`MergeLaw.newgap_le_max`, `MergeLaw.newgap_le_step` (the Corollary, including the bookkeeping of
(D) for two concrete machines), `MergeLaw.D_of_qualmax`; `Spectrum.merged_eq`,
`Spectrum.windowSum`, `Spectrum.SpectrumBound`, `Spectrum.Qualifying`, `Spectrum.QualBound`;
`TwoTeeth.teeth_letters` (T1), `TwoTeeth.spacing_from_lo`, `TwoTeeth.spacing_from_hi`,
`TwoTeeth.next_kill_of_lo`, `TwoTeeth.next_kill_of_hi` (T3), `TwoTeeth.kills_gap_ge`,
`TwoTeeth.kill_spacing_min` (T4), `TwoTeeth.fuel_span_cap`, `TwoTeeth.fuel_le` (T5);
`WordLegal.Alt`, `WordLegal.Legal`, `WordLegal.NoRepeat`, `WordLegal.legal_iff_noRepeat`,
`WordLegal.alt_iff_prefixSum`, `WordLegal.killable_iff`, `WordLegal.two_mul_ne_zero`,
`WordLegal.val_injective` (F).  The "exactly two copies" count (step 3) and step 6 are written
one-line corollaries of the kernel facts.

Verified computationally: hit law and chain law on the full period of every machine `{5..23}`;
"exactly two copies" at m11..m23; the merged gap histogram reproduced against direct
construction at four extensions and `F = 18, 25, 34, 43, 58, 88` at the six steps 13->17 ..
31->37; the two recorded failure modes (literal-only merge condition undershoots at 31->37,
non-alternating condition overshoots at 23->29) are why T2 and T3 are both needed.

## Relationship to the conjecture

Exact machinery: the complete local grammar of one step, used by every certificate and every
bound on the alignment depth.  By itself it bounds nothing; the budget inequality is not
touched.  Nothing measured enters the theorems; the chain depth `D_g` that the grammar leaves
free is measured.

## Where it is used

Files 06-12 all rest on this file: saturation (06) uses T4 and (A); the attainment identity
(08) uses (D), (F) and (A); the record law (09) is (A) read on one lower period; the word
reduction (10) is (F) over an enumeration; the spectrum bound (11) uses T3.

## Source

`docs/gear-recursion.md` sections 3-4 (the merge transform, the deletion-spacing lemma in the
adjacent frame); `docs/novel/merge-law.md`; `docs/novel/two-teeth-kill-spacing.md` (T1-T5);
Constructor R39, R81, R89 (`docs/proof-search/constructor.md`);
`docs/proof-search/anchor-235.md` 9d-9f (hit law, chain law, phase form);
`docs/proof-search/alignment-rules.md` 2.1-2.3.
