# 8. The attainment identity: `F(M + q') = max(F_2(M), max_J Q*_J(M; q'))`

## In plain words

This is the exact rule for the record of the bigger machine. Any stretch of the old machine
whose interior spacings obey the grammar of the new gear's strikes can be lined up so that the
new gear strikes every opening inside it, and the whole stretch becomes one gap of the bigger
machine; conversely every gap of the bigger machine arises that way. So the new record is
exactly the longest grammatical stretch of the old machine. Said plainly: because this is an
equality, checking that the longest grammatical stretch fits the budget is the same task as
checking the record itself, only done on the old machine.

## Vocabulary

Column `k` is the pair `(6k-1, 6k+1)`; `M` is a finite set of gears with period `P`; `q'` is
a gear not in `M`, `c = 6^{-1} mod q'`, teeth `T = {c, -c}`; `F(M)`, `F_2(M)` as in file 06.
A **`J`-run** of `M` is a block of `J` consecutive gaps, spanning `J + 1` consecutive openings
`x_0 < x_1 < ... < x_J`; its **flanks** are the first and last gap, its **middles** the `J - 2`
gaps `x_1 -> x_2, ..., x_{J-2} -> x_{J-1}`, its **span** is `x_J - x_0`.  A gap value `v` has
letter `pad` if `v = 0 (mod q')`, `up` if `v = +2c`, `down` if `v = -2c`, and is illegal
otherwise; a word of letters is **legal** if no two consecutive nonzero letters are equal
(file 05 (F)).  A `J`-run is **word-legal** for `q'` if all its middles have letters and the
middle word is legal.  Define

    Q*_J(M; q') = the largest span of a word-legal J-run of M   (J >= 2; -inf if none),

so `Q*_2(M; q') = F_2(M)` (a 2-run has no middles).

Classical translation: a `J`-run with all its interior openings struck by `q'` becomes a single
gap of `M + q'`; the identity says the record of the bigger machine is computed on the old
machine from the residues of its gaps mod `q'`.

## Statement

**Theorem (attainment identity, R68).**

    F(M + q') = max_{J >= 2} Q*_J(M; q') = max( F_2(M), max_{J >= 3} Q*_J(M; q') ).

The two halves:

- **(Attainment, `>=`.)**  If `x_0 < ... < x_J` are consecutive openings of `M` whose middle
  word is legal for `q'`, then `x_J - x_0 <= F(M + q')`.
- **(Merge, `<=`.)**  Every gap of `M + q'` is the span of a word-legal `J`-run of `M` for some
  `J >= 2`, or a single gap of `M`.

## Proof

**Lemma (killable iff legal; file 05 (F), step 15).**  Residues `x_1, ..., x_m in Z_{q'}` all
lie in `{r + c, r - c}` for some `r` iff their consecutive differences form a legal word.

**Attainment (`>=`).**

1. Let `x_0 < ... < x_J` be consecutive openings of `M` with legal middle word (`J >= 2`; for
   `J = 2` the word is empty and legal).  By the Lemma applied to the interior openings
   `x_1, ..., x_{J-1}` (whose consecutive differences are exactly the middles) there is a
   residue `r` with `x_i = r +- c (mod q')` for `1 <= i <= J - 1`.
2. `P` is invertible mod `q'`, so there is `j` with `jP = -r (mod q')`.  Translation by `jP`
   preserves the openings of `M`, and `x_i + jP = +-c (mod q')` for every interior `i`: in copy
   `j` every interior opening is a tooth of `q'`, i.e. struck.
3. No opening of `M + q'` lies strictly between `x_0 + jP` and `x_J + jP` (every opening of
   `M` there is struck).  `M + q'` has openings, so the gap of `M + q'` containing that
   interval has length at least `x_J - x_0` (longer if an endpoint is struck too).  Hence
   `x_J - x_0 <= F(M + q')`.
4. Taking the maximum over word-legal `J`-runs: `Q*_J(M; q') <= F(M + q')` for every `J >= 2`.
   (For `J = 2` this is `F_2(M) <= F(M + q')`, file 07 with `r = 1`.)

**Merge (`<=`).**

5. Let `y < z` be consecutive openings of `M + q'`.  By the merge law (file 05 (D)) `y` and `z`
   are openings of `M`, the openings of `M` strictly between them, say `x_1 < ... < x_{J-1}`
   (`J >= 1`), are all struck by `q'`, and `z - y` is the sum of the `J` consecutive gaps of
   `M` from `y =: x_0` to `z =: x_J`.
6. If `J = 1`, `z - y` is a gap of `M`, at most `F(M) < F_2(M)`.  If `J = 2`,
   `z - y <= F_2(M)`.
7. If `J >= 3`: the interior openings `x_1, ..., x_{J-1}` all have residues in `T = {0 + c, 0 - c}`,
   so by the Lemma (with `r = 0`) their consecutive differences -- the middles of the run --
   form a legal word.  The run `x_0 < ... < x_J` is word-legal and `z - y <= Q*_J(M; q')`.
8. So every gap of `M + q'` is at most `max_{J >= 2} Q*_J(M; q')`, and with step 4 the
   identity holds.

**Which direction is kernel-checked, and which is written.**  The Lemma is kernel-checked in
both directions (`WordLegal.killable_iff`; over a machine's opening enumeration,
`WordLegal.chain_iff_word`: `k + 1` consecutive openings all on the teeth of one phase iff the
`k` gaps between them form a legal word).  The merge law's bookkeeping (step 5) is kernel-checked
as `MergeLaw.newgap_le_step`, but only in the relaxed form where each middle is required to be
`>= 2u'` (the qualifying floor) rather than to carry a legal word -- i.e. the kernel proves
`F(M+q') <= max(F_2(M), max_j Q_j(M))` with `Q_j >= Q*_j`; the sharp `<=` with `Q*_J`
(step 7) is the written assembly of `chain_iff_word` with the merge law.  In the attainment
direction the existence of the copy `j` is `AnchorChain.phase_bijective`; the translation step
(steps 2-3) is written.  No Lean theorem states the identity itself.

**Negative rider, on record and load-bearing.**  Because `max_J Q*_J` equals `F(M+q')`, the
statement "the word-legal criterion certifies `F(M+q') <= F(M) + q'`" is the same statement as
the budget inequality in another representation.  There is no slack in the criterion to exploit;
its whole value is that it is computed on the old machine without building `M + q'`.

## Status

Kernel: `WordLegal.killable_iff`, `WordLegal.chain_iff_word`, `WordLegal.killed_of_word`,
`WordLegal.word_of_killed` (the Lemma); `MergeLaw.newgap_le_step`, `MergeLaw.newgap_le`,
`MergeLaw.interior_gap_mod` (the merge half, qualifying-floor form); `AnchorChain.phase_bijective`
(the copy).  The identity as stated: written proof only (round 22 as the Kleene-star identity
`F(M+q') = L (x) K* (x) R` in max-plus, proved both ways; round 26 the standalone CRT proof,
Constructor R68).

Verified computationally: exact at the eight steps m11..m37 (`research/qstar.py`,
`research/data/r26_qstar.log`), with `max_J Q*_J = F(M+q') = 11, 18, 25, 34, 43, 58, 88, 91`;
two out-of-scan confirmations `Q*_max(43; 47) = 118 = F(47)` and `Q*_max(47; 53) = 145 = F(53)`;
`F(59) = 161` computed on machine 23's period by the same vehicle; 27,570 tooth-counterfactual
machines with zero exceptions.

## Prior art, and what is new

**Leverages.**  Standard (CRT) and the grammar of file 05.  The one-class shadow of the merge
half is Holt & Rudd, arXiv:1408.6002, Lemma 2.1 (every gap of the next stage is a sum of
consecutive gaps of the current one) with Theorem 2.3 as its CRT converse, both read first-hand
in the round-30 check (`docs/novel/spectrum-depth-certificate.md` section 6).

**New.**  The equality.  `F(M + q')` is exactly the largest span of a word-legal `J`-run of the
old machine, so the next record is computed on the old machine without building the new one, and
the criterion that makes it exact is word-legality (a residue condition on the middles), not a
size threshold as in the saturated regime of file 06.  The register's checks found no published
inequality of this shape, let alone an identity, in either class count.

**Not new.**  The `<=` half is Holt-Rudd's merge in gear language at two classes, and the `>=`
half is the CRT translation of file 07; `docs/novel/merge-law.md` records PARTIAL OVERLAP on
precisely that split (the one-class cycle recursion known, the no-reconstruction maximal-gap
formula not found).  And the identity carries no slack, which the file's own negative rider
states: certifying `max_J Q*_J` against the budget is the budget inequality in another
representation, not a weaker statement.

## Relationship to the conjecture

Exact machinery: the record of the next machine computed on the old one.  It is the frame in
which the budget inequality splits into the pair statement (`J = 2`) and the chain statement
(`J >= 3`), both open.  Being an equality it carries no slack: it relocates the size statement
to the old machine and does not make progress on it.  No measured input.

## Where it is used

The record of every machine from the one below (files 09, 17, 18); the word reduction (file
10) and the spectrum bound on `L` (file 11) both consume the attainment half; the pair statement
and chain statement split of file 19 (L1) is this identity read at `J = 2` versus `J >= 3`.

## Source

Constructor R46 (round 22, the Kleene identity) and R68 (round 26, `docs/proof-search/
constructor.md`); Mechanic's `Q*` definition (round 25); `docs/novel/kleene-generator.md`;
`docs/proof-search/alignment-rules.md` 3.2-3.3.
