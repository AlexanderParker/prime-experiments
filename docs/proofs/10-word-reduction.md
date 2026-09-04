# 10. The word reduction `J_max = L + 2`, `A_kill = L + 1`, and the same-tooth lemma

## In plain words

The number of consecutive old openings the new gear can strike in one go is not a new quantity:
it is one more than the length of the longest run of neighbouring gaps in the old machine whose
sizes fit the new gear's grammar. So the depth of any merge is a question about the old
machine's gap sizes alone. A second lemma says that in a run of struck openings with no
same-position spacing, an even number of spacings brings the strike back to the tooth it
started on.

## Vocabulary

Column `k` is the pair `(6k-1, 6k+1)`; `M` is a finite set of gears with period `P`, `q'` a
gear not in `M`, `c = 6^{-1} mod q'`, teeth `{c, -c}`.  Let `op(0) < op(1) < ...` enumerate
the openings of `M` and `gap(n) := op(n+1) - op(n)`.  Letters: a gap value is `pad` if
`= 0 (mod q')`, `up` if `= 2c`, `down` if `= -2c`, illegal otherwise; a word over
`{pad, up, down}` is **legal** if no two consecutive nonzero letters are equal (file 05 (F)).

- A **realised legal word of length `m`** is an index `i` and a legal word `w_1 .. w_m` with
  `gap(i + j - 1)` of letter `w_j` for `j = 1..m`.  `L(M)` is the largest `m` for which one
  exists.
- A **word-legal `J`-run** at gap index `i` is the block `gap(i), ..., gap(i + J - 1)` whose
  `J - 2` middles `gap(i+1), ..., gap(i+J-2)` form a legal word (file 08).
  `Q*_J(M; q') > -inf` means some word-legal `J`-run exists; `J_max(M)` is the largest such
  `J`.
- A **`k`-chain** at `n` is `k` consecutive openings `op(n), ..., op(n + k - 1)` all on the
  teeth of ONE phase `r`: `op(n + j) - r = +-c (mod q')` for `j < k`.  `A_kill(M -> q')` is
  the largest `k` for which a `k`-chain exists (the arity of the deepest deletion `q'` makes in
  one copy; by the chain law of file 05 (C) this is the chain depth `D_{q'}` of file 09).

Classical translation: `A_kill` is the largest number of consecutive `M`-rough twin candidates
that the single prime `q'` can strike in a row; `L` is the longest run of consecutive gaps of
`M` whose values, read modulo `q'`, are all in `{0, +-2c}` with the nonzero classes
alternating.

## Statement

**Theorem 1 (chain iff word).**  A `(k+1)`-chain exists at `n` iff the `k` gaps
`gap(n), ..., gap(n + k - 1)` form a legal word.

**Theorem 2 (word reduction, R89).**  Assume the gap letters are periodic in the index with
some period `N > 0` (true for every machine: `gap(n + N) = gap(n)` with `N` the number of
openings per period).  Then for every `J >= 2`:

    Q*_J(M; q') > -inf   iff   L(M) >= J - 2.

Hence `J_max(M) = L(M) + 2` and `A_kill(M -> q') = L(M) + 1`.  (The second identity needs no
periodicity.)

**Theorem 3 (same-tooth lemma, R90).**  Let `w` be a legal word read from tooth `t`, and let
`S` be the sum of its letters as residues (`pad = 0`, `up = 2c`, `down = -2c`).  Then
`S = (end tooth) - (start tooth) in {0, +-2c}`, and `S = 0 (mod q')` iff the number of
non-padded letters is even.  For a word-legal `J`-run `x_0 < ... < x_J`: the middle span
`x_{J-1} - x_1 = 0 (mod q')` iff the number of non-padded middles is even.  In particular a
**literal** (no padded middle) `J`-run with `J` even has `x_{J-1} - x_1 = 0 (mod q')`: its
first and last struck openings sit on the same tooth, and its middle span is a positive multiple
of `q'`.

## Proof

**Theorem 1.**

1. (`=>`)  If `op(n + j) - r = t_j c` with `t_j in {+1, -1}` for `j = 0..k`, then
   `gap(n + j) = (t_{j+1} - t_j) c`, which is `0` (`pad`), `2c` (`up`, from `-` to `+`) or
   `-2c` (`down`, from `+` to `-`); reading the word from tooth `t_0` reproduces the sequence
   `t_j`, so the reading is consistent and the word is legal.
2. (`<=`)  Read the legal word from a consistent starting tooth `t_0`, obtaining teeth
   `t_0, ..., t_k`; put `r := op(n) - t_0 c`.  Inductively
   `op(n + j + 1) - r = (op(n + j) - r) + gap(n + j) = t_j c + (t_{j+1} - t_j) c = t_{j+1} c`.
   So all `k + 1` openings are on the teeth of phase `r`.

**Theorem 2.**

3. (`=>`)  The middles of a word-legal `J`-run at index `i` are `J - 2` consecutive gaps
   starting at `i + 1` forming a legal word: a realised legal word of length `J - 2`.
4. (`<=`)  Let a legal word of length `J - 2` be realised at index `i`.  If `i >= 1`, the block
   `gap(i - 1), ..., gap(i + J - 2)` is a `J`-run whose middles are that word: word-legality
   constrains only the middles, so any flanks do.  If `i = 0`, periodicity moves the word to
   index `N >= 1` (`gap(N + j) = gap(j)` letter for letter), and the previous case applies.
5. `J_max = L + 2`: prefixes of legal words are legal (the reading restricts), so if a word of
   length `L` is realised, so is one of every shorter length; and none of length `L + 1` is
   realised by definition of `L`.  By steps 3-4, `Q*_J > -inf` iff `J - 2 <= L`.
6. `A_kill = L + 1`: by Theorem 1, a `(k+1)`-chain exists iff a legal word of length `k` is
   realised, iff `k <= L`.

**Theorem 3.**

7. Reading the word from `t`, each letter changes the tooth from `t_j` to `t_{j+1}` and has
   value `(t_{j+1} - t_j) c`; the values telescope to `(t_end - t_start) c`.  This is `0` if
   the end tooth equals the start tooth and `+-2c` otherwise.
8. A padded letter keeps the tooth and a non-padded letter flips it, so the end tooth equals the
   start tooth iff the number of non-padded letters is even.  Since `2c != 0 (mod q')` (from
   `6c = 1`: `2c = 0` would give `1 = 6c = 0`), `S = 0` iff the teeth agree.
9. For a run, `x_{J-1} - x_1` is the sum of the middles, whose residues are the letter values;
   apply steps 7-8.  If all `J - 2` middles are non-padded and `J - 2` is even, the count is
   even, so `x_{J-1} - x_1 = 0 (mod q')`, and it is positive, hence `>= q'`.  (The sharper
   `>= ((J-2)/2) q'` is the middle-sum lemma of file 16.)

Two riders.  (a) Theorem 3 is for literal middles: the two padded even-`J` maximisers on record,
`(12, 37)` at m31 and `(41, 14)` at m37, have middle sums `49 = 12 (mod 37)` and
`55 = 14 (mod 41)`.  (b) The reduction moves the open question, it does not close it: `L(M)`
bounded is still open; measured `L = 1, 1, 1, 2, 1, 3, 3, 2, 2, 2, 4, 3` at m11..m53.

## Status

Kernel: `WordLegal.chain_iff_word`, `WordLegal.killable_iff`, `WordLegal.word_of_killed`,
`WordLegal.killed_of_word` (Theorem 1); `WordLegal.word_of_window`, `WordLegal.window_of_word`,
`WordLegal.qstar_iff_word` (hypothesis: periodicity of the gap residues, `hper`),
`WordLegal.jmax`, `WordLegal.akill`, `WordLegal.realisedWord_mono`, `WordLegal.legal_take`,
`WordLegal.alt_take` (Theorem 2); `WordLegal.sum_eq_tooth_sub`, `WordLegal.endTooth_eq_iff`,
`WordLegal.same_tooth`, `WordLegal.middle_span`, `WordLegal.same_tooth_window`,
`WordLegal.literal_even_span`, `WordLegal.two_mul_ne_zero`, `WordLegal.val_injective`
(Theorem 3).  Instantiated at m11, m13, m17 with `L = 1` (`WordLegal11`, `WordLegal13.L13`,
`WordLegal13.jmax13`, `WordLegal13.akill13`, `WordLegal17.L17`, `WordLegal17.akill17`); the
periodicity hypothesis is supplied by `Machine11.g11_shift` (in `proofs/Machine11Per.lean`) and `Periodic.op_shift`.

Verified computationally: `J_max` and `A_kill` rows reproduced 16/16 at m11..m41
(`research/perj_window.py`, `research/perj_scanfree.py`); `D_g = A_kill` at seven gears by two
vehicles; the same-tooth lemma on all 38 realised legal words with an exact source.

## Prior art, and what is new

**Leverages.**  File 05's legality criterion and the periodicity of the gap sequence; nothing
external.  Nearest published items, read first-hand in the round-30 check
(`docs/novel/even-j-mechanism.md` section 6): Holt-Rudd remark (vi) (a run of equal gaps forces
divisibility by the small primes), Ziller 2020 (arXiv:2007.01808) `D(k)`, the one-class
dictionary at word length one, and on the prime side Shiu 2000, Banks-Freiberg-
Turnage-Butterbaugh (arXiv:1311.7003) and Maynard 2016 -- all existence results for long runs,
the opposite shape to a cap.

**New.**  The identity `J_max = L + 2`, `A_kill = L + 1`: the merge depth is not a new quantity
but the longest realised legal word of the old machine's gap letters, so every empty cell of the
per-`J` table becomes a dictionary fact instead of a search, and the open question "is the
alignment depth bounded" becomes "is `L` bounded" without a change of content.  The converse
half is the new one; the same-tooth lemma (the middle span is `0 mod q'` iff the number of
non-padded middles is even) is likewise recorded NOVEL AS FAR AS SEARCHED.

**Not new.**  Theorem 1 (chain iff word) is file 05's legality criterion re-read over an opening
enumeration, and the forward half of the reduction is Mechanic's round-28 index observation on
the record.  Ziller 2020's `D(k)` is the one-class case of "which words occur" at length one,
and Holt-Rudd remark (vi) is a one-class run constraint in gear language.

## Relationship to the conjecture

Exact bookkeeping: it identifies the alignment depth `A_kill` with the word length `L(M) + 1`,
converting the open question "is `A_kill` bounded" into "is `L` bounded" without answering it.
`L` bounded is open.  No measured input to the theorems.

## Where it is used

Every `EMPTY` cell of the per-`J` table is a dictionary fact; the depth cap of the spectrum
certificate `J_max = A_kill + 1`; the spectrum bound on `L` (file 11) and the bare-word cap
(file 12) are bounds on the `L` of this file.

## Source

Mechanic round 28 (the `=>` half, "a word-legal J-run (the round notes say window) of `J` gaps is a kill chain of arity
`J - 1`"); Constructor round 29, R89 and R90 (`docs/novel/even-j-mechanism.md` 1.1-1.2);
Formalist round 30 (`proofs/WordLegal.lean`); `docs/proof-search/alignment-rules.md` 3.6.
