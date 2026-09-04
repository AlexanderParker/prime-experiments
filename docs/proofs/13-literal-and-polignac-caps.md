# 13. The literal cap (at most 6 members, exact per class mod 210) and the Polignac cap (12)

## In plain words

The same fact seen from the gear's side: a run of consecutive strikes of one gear at the
minimal alternating spacings, all of them landing on columns that gears 5 and 7 leave open, has
at most six members, and the exact maximum for each remainder class of the gear is tabulated:
six for six classes, four, three or two for the rest, never five. For pairs of primes at any
even distance instead of two, the same kind of run has at most twelve members. These caps
concern only the tightest runs; runs containing a full-gear-sized spacing are not covered.

## Vocabulary

Column `k` is the pair `(6k-1, 6k+1)`; gear `q` strikes `k` iff `k = +-u_q (mod q)`,
`6u_q = q -+ 1`; consecutive struck columns of `q` are spaced alternately `2u_q` and `q - 2u_q`
(file 02 (b)).  The **corridor** `E_35` is the set of residues mod 35 open for both gears 5 and
7 (file 14): `{0, 2, 3, 5, 7, 10, 12, 17, 18, 23, 25, 28, 30, 32, 33}`; a column `k >= 1` is
**exposed** if `k mod 35 in E_35`.

A **literal chain** of gear `q` based at `r` with parity `ph in {0, 1}` is the sequence

    member_i := r + floor((i + ph)/2) * q + [ (i + ph) odd ] * 2u_q,    i = 0, 1, 2, ...

i.e. `r, r + 2u_q, r + q, r + q + 2u_q, r + 2q, ...` (`ph = 0`) or
`r + 2u_q, r + q, r + q + 2u_q, ...` (`ph = 1`): consecutive struck columns of `q` at the
alternating spacings.  Its **length** is the number of leading members that are exposed.  In
letters (file 05), a literal chain of `L` members carries the bare word of `L - 1` letters
`(a, b, a, ...)` or `(b, a, b, ...)`; so this is the member-count form of file 12's object, and
`capC(c) = PSORD(c) + 1`.

Classical translation: a literal chain is the longest run of consecutive columns with a
`q`-divisible member, none of which has a member divisible by 5 or 7.

## Statement

**Theorem 1 (the literal cap).**  Let `q` be a prime with `gcd(q, 210) = 1`, `r >= 1`,
`ph in {0, 1}`.  If `member_0, ..., member_{L-1}` are all exposed then

    L <= capC(q mod 210) <= 6,

where `capC(c) = 6` for `c in {37, 53, 83, 127, 157, 173}`; `4` for
`c in {1, 23, 31, 61, 67, 89, 97, 113, 121, 143, 149, 179, 187, 209}`; `3` for
`c in {29, 59, 151, 181}`; and `2` for the remaining 24 invertible classes.  The table is exact:
every class admits, in the corridor, a start and parity with `capC(c)` consecutive exposed
members.  No class has cap 5.  In word form: a literal word of `ell` letters carried by a chain
of `ell + 1` exposed members has `ell < capC(q mod 210)`.

**Theorem 2 (the Polignac cap).**  Fix an even gap `d = 2e`.  In the halved frame a position
`n` denotes the pair `(2n + 1, 2n + 1 + 2e)`, and (after one CRT translation) an odd prime `q`
blocks `n = 0` and `n = -e (mod q)`.  A **literal chain** of a prime `q' >= 11` for gap `d` is
a maximal run of consecutive strikes of `q'` (spacings alternating `e`, `q' - e`) that survive
gear 3, all of which survive gears 5 and 7; strikes that gear 3 blocks are skipped and neither
count nor end the run.  Then the length of every such chain is at most `cap(gcd(e, 105))`:

    gcd(e, 105)    1   3   5   7   15   21   35   105
    cap            6   6   6   6   10    6    6    12

so **12 is the absolute ceiling over all even gaps**, for every gear, and each cap is attained.
The twin case `e = 1` is Theorem 1's uniform bound 6.

## Proof

**Theorem 1.**

1. **Reduction mod 35.**  Exposure of `member_i` depends only on `member_i mod 35`, which is
   `(r mod 35) + floor((i+ph)/2) (q mod 35) + [(i+ph) odd] (2u_q mod 35)` reduced mod 35.  Both
   `q mod 35` and `2u_q mod 35` are functions of `c := q mod 210`: write `q = c + 210k`; then
   `c = q (mod 6)`, so `2u_q = (q -+ 1)/3` and `sOf(c) := (c -+ 1)/3` carry the same sign and
   differ by `70k`, whence `2u_q = sOf(c) (mod 35)` (kernel: `LiteralCap.s_eq`).  So the
   residue sequence of any literal chain of `q` is the walk
   `wpos(c mod 35, sOf(c) mod 35, r mod 35, ph, i)`.
2. **The finite check.**  There are 48 invertible classes `c`, 35 starts `r mod 35` and 2
   parities.  For each, the kernel evaluates whether the first `capC(c) + 1` walk members are
   all in `E_35` and finds that they never are (`LiteralCapTable.cap_table_maximal`; the
   uniform statement "never 7" is `LiteralCap.no_run_seven`).  Hence a chain with `L` exposed
   members has `L <= capC(c)` (`LiteralCapTable.literal_chain_le_capC`,
   `LiteralCap.literal_chain_le_six`).
3. **Exactness.**  For every class the kernel exhibits a start and parity whose first `capC(c)`
   walk members are in `E_35` (`LiteralCapTable.cap_table_realized`,
   `LiteralCap.cap_six_classes_sharp` for the six classes of cap 6).  The census of the table is
   `{2: 24, 3: 4, 4: 14, 6: 6}` (`cap_two_classes`, `cap_three_classes`, `cap_four_classes`,
   `cap_six_classes`, `no_cap_five`, `cap_spectrum_counts`).
4. **Word form.**  A word of `ell` letters between `ell + 1` consecutive members, all exposed,
   gives `ell + 1 <= capC(c)` (`LiteralCapTable.word_length_lt_capC`).

Scope, from the record: literal chains only.  Padded runs (a gap `= q'`, i.e. the same tooth
one lap later) escape the cap, and "killed runs are bounded by 6" is FALSE; the cap is not a
density statement either (over all 1,225 pairs `(t, s)` mod 35 the run spectrum reaches 140; the
restriction to `t` invertible and `s = sOf(t)` does real work).

**Theorem 2.**

5. **The model.**  Odd prime `q` blocks position `n` iff `q | 2n + 1` or `q | 2n + 1 + 2e`,
   i.e. `n = -2^{-1}` or `n = -2^{-1} - e (mod q)`: a translate of `{0, -e}`.  Translating all
   positions by a common `s` with `s = -2^{-1} (mod q)` for every gear in play (CRT) puts every
   gear's blocked pair at `{0, -e}`, which is the kernel's normalisation (`PolignacCap.adm e q n`).
   Consecutive strikes of `q'` are then spaced alternately `e` (from `-e` to `0`) and `q' - e`.
6. **Reduction mod 105.**  Survival of a position at gears 3, 5, 7 depends on the position mod
   105 and on `e mod 105`; the spacings mod 105 depend on `e mod 105` and `t := q' mod 105`.
   So the whole run structure of `q'`'s strike sequence is a walk on `Z_105` determined by
   `(e mod 105, t)`, `t` coprime to 105.
7. **Reduction to `gcd(e, 105)`.**  For a unit `v` mod 105, the map `n -> vn` sends the walk
   for `(e, t)` to the walk for `(ve, vt)` and preserves every blocking condition
   (`n = 0` iff `vn = 0`; `n = -e` iff `vn = -ve`, per prime factor of 105).  As `t` ranges
   over all units so does `vt`.  Hence the maximal run length over all `t` and all starts is
   the same for `e` and `ve`.  In `Z_105 = Z_3 x Z_5 x Z_7` two residues with the same
   `gcd(e, 105)` have the same pattern of zero coordinates, and their nonzero coordinates are
   unit multiples of one another, so they are unit multiples mod 105.  Therefore the cap is a
   function of `gcd(e, 105)` alone, and the eight representatives `e = 1, 3, 5, 7, 15, 21, 35,
   105` cover every even gap.
8. **The finite check.**  For each representative `e`, the kernel evaluates
   `capOK e L 26 = true` (`PolignacCap.cap_gcd_1`, `_3`, `_5`, `_7`, `_15`, `_21`, `_35`,
   `_105`, with `L = 6, 6, 6, 6, 10, 6, 6, 12`): for every unit `t` mod 105, every start
   position `r` exposed to 3, 5 and 7, and both parities, the walk of 26 steps from `r` never
   accumulates more than `L` consecutive positions that survive gear 3 and are exposed to 5 and
   7 (a position blocked by 3 is skipped without resetting the count; a position surviving 3 but
   blocked by 5 or 7 resets it).
9. **26 steps suffice.**  A run of `L + 1` counted strikes begins at a counted (hence exposed)
   position, which is one of the scanned starts, with the run's parity.  Between counted
   positions the walk skips only gear-3-blocked ones, so it is enough that `L + 1` gear-3-
   admissible positions occur within 26 steps of any admissible start.  Read the walk mod 3,
   where the steps alternate `e` and `t - e` with `t = +-1 (mod 3)`.  If `3 | e` the steps are
   `0, +-1, 0, +-1, ...`: the positions mod 3 go `n, n, n+-1, n+-1, n+-2, n+-2, ...`, exactly
   one residue (`0`) is blocked by 3, so four of every six consecutive steps are admissible,
   and from any admissible start `13` admissible positions occur within
   `1 + 2 + 4 + 2 + 4 + 2 + 4 = 19` steps.  If `3` does not divide `e`, two residues (`0` and
   `-e`) are blocked and one is admissible; the steps mod 3 are `(e, t - e) in
   {(1, 0), (1, 1), (2, 2), (2, 0)}`, so the walk either visits the three residues cyclically
   (one admissible position in every three steps) or visits each residue twice in a row with
   period six (two admissible positions in every six steps); either way `7` admissible
   positions occur within `1 + 4 + 2 + 4 + 2 + 4 + 2 = 19` steps of an admissible start.  In
   all eight classes `L + 1 <= 13` positions are seen within 19 `<= 26` steps, so the scan of
   step 8 would have found the run.  (This step is written; the kernel statement is literally
   step 8.)
10. `capOf_le_twelve`: the largest table entry is 12.  Sharpness of each cap (the scan fails at
    `cap - 1`) was checked numerically before formalising.

Modelling trap recorded with the theorem: treating gear 3 like gears 5 and 7 (as ending a run
rather than filtering the candidate list) gives caps `2/4` instead of `6/10/12`.

## Status

Kernel: Theorem 1 -- `LiteralCap.sOf`, `LiteralCap.wpos`, `LiteralCap.run7`,
`LiteralCap.no_run_seven`, `LiteralCap.s_eq`, `LiteralCap.member`,
`LiteralCap.literal_chain_le_six`, `LiteralCap.hasRun6`, `LiteralCap.cap_six_classes_sharp`;
`LiteralCapTable.runL`, `LiteralCapTable.hasRunL`, `LiteralCapTable.hasRunL_mono`,
`LiteralCapTable.capC`, `LiteralCapTable.capC_le_six`, `LiteralCapTable.cap_table_maximal`,
`LiteralCapTable.cap_table_realized`, `LiteralCapTable.literal_chain_le_capC`,
`LiteralCapTable.word_length_lt_capC`, `LiteralCapTable.cap_two_classes`,
`LiteralCapTable.cap_three_classes`, `LiteralCapTable.cap_four_classes`,
`LiteralCapTable.cap_six_classes`, `LiteralCapTable.no_cap_five`,
`LiteralCapTable.cap_spectrum_counts`; also `LiteralCapTable.tripled_teeth_antipode`
(`{3u, q - 3u} = {(q-1)/2, (q+1)/2}`).  Theorem 2 -- `PolignacCap.adm`, `PolignacCap.inE`,
`PolignacCap.stepOf`, `PolignacCap.scan`, `PolignacCap.capOK`, `PolignacCap.cap_gcd_1`,
`cap_gcd_3`, `cap_gcd_5`, `cap_gcd_7`, `cap_gcd_15`, `cap_gcd_21`, `cap_gcd_35`, `cap_gcd_105`,
`PolignacCap.capOf`, `PolignacCap.capOf_le_twelve`, `PolignacCap.exists_mul_mod_eq`
(no axioms beyond the kernel's).  Steps 5, 7 and 9 of Theorem 2 (the model, the reduction to
`gcd`, and the adequacy of 26 steps) are written.

Verified computationally: the per-class caps against a 140-step maximal-run computation at all
48 classes and class invariance against every prime to 2000 and 5000
(`research/literal_cap_gap_d.py`), zero mismatches; every realised chain length in
`research/data/fuel_census.csv` respects its class cap, saturating it at `q' = 19` and `31`;
the eight Polignac spectra reproduced row for row.

## Prior art, and what is new

**Leverages.**  Standard congruence bookkeeping modulo 210 and modulo 105, CRT, and the corridor
of file 14.  One counting ingredient is conceded KNOWN on the record: the product
`|E_e| = prod_{q in {3,5,7}} (q - r_q)` is the Hardy-Littlewood local factor, used here only as
a cross-check (`docs/novel/polignac-cap.md` section 6).

**New.**  The caps themselves.  A run of a gear's tightest alternating strikes inside the
anchor's corridor has at most six members, with the exact maximum per class of `q' mod 210` and
no class of cap 5; and over all even gaps the same object is capped by `cap(gcd(e, 105))`, with
12 the ceiling for every gear and every even gap.  Both are recorded NOVEL AS FAR AS SEARCHED
(2026-08-23), and their value to the route is that the word list of the word-indexed identity
depends on `q' mod 210` alone -- a finite dictionary rather than a per-machine search.

**Not new.**  The admissible-residue arithmetic is standard (`nu(q) = q - 2` for `q >= 5`; the
15 residues mod 35), and the `|E_e|` product is the classical local factor.  Nothing in the
Zhang-Maynard-Polymath line is prior art here: those are infinitude statements and this is a
finite structural cap, a delta the register states explicitly.  The scope limit is the file's
own: literal chains only, padded runs escape, and "killed runs are bounded by 6" is false.

## Relationship to the conjecture

The same object as file 12 from the gear's side, plus the all-gaps ceiling 12.  It bounds only
literal (tightest-spacing) chains; padded runs escape it; no bearing on the size statement
beyond file 12's.  Nothing measured enters.

## Where it is used

The word list of the word-indexed identity (alternating words of length `1 .. capC - 1`)
depends on `q' mod 210` alone; `capC(c) <= 3 iff c in S` ties the literal cap to the bare-word
cap (file 12); the Polignac cap is the first all-gap statement in the ledger.

## Source

Constructor section 23.2 (the literal walk); Formalist rounds 14-16 (`proofs/LiteralCap.lean`,
`proofs/LiteralCapTable.lean`, `proofs/PolignacCap*.lean`); Harvester's halved frame
(`docs/novel/literal-cap.md`, `docs/novel/polignac-cap.md`);
`docs/proof-search/alignment-rules.md` 3.8.
