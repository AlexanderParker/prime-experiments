# 16. The peel bound, the triple inequality, the middle-sum lemma and the parity of palindromes

## In plain words

Four small facts about grammatical stretches. Cutting off either end of one leaves a shorter
grammatical stretch, so each depth is bounded by the depth below plus the smaller end gap. Any
three neighbouring gaps together are at most the best pair plus the smaller end gap, with no
grammar needed. In a stretch whose interior spacings are all the two small letters, the
interior adds up to at least half a gear per interior spacing, so deeper stretches have less
room for their end gaps. And a stretch with an even number of such interior spacings can never
read the same backwards.

## Vocabulary

Column `k` is the pair `(6k-1, 6k+1)`; `M` is a finite set of gears, `q'` a gear not in `M`
with tooth value `u'`, letters `a := 2u'`, `b := q' - 2u'` (`a + b = q'`, `a < b`),
`c = 6^{-1} mod q'` so `{2c, -2c} = {a, b}` as residues.  A **`J`-run** is a block of `J`
consecutive gaps of `M` spanning openings `x_0 < ... < x_J`, with **flanks** `g_L = x_1 - x_0`,
`g_R = x_J - x_{J-1}` and **middles** `x_1 -> x_2, ..., x_{J-2} -> x_{J-1}`; it is
**word-legal** if the middles are all `= 0` or `+-2c (mod q')` with the nonzero classes
strictly alternating, and **literal** if moreover no middle is `= 0 (mod q')` (files 05, 08).
`Q*_J(M; q')` is the largest span of a word-legal `J`-run, `F_2(M) = Q*_2`, `F(M)` the record,
and `Phi_J` the largest flank sum `g_L + g_R` over literal word-legal `J`-runs.

Classical translation: a word-legal `J`-run is a stretch of `J + 1` consecutive `M`-rough twin
candidates whose `J - 1` interior ones the prime `q'` can strike in one lap; its span is a
candidate for the next record.

## Statement

**Theorem D (the peel bound).**  Deleting either flank of a word-legal `J`-run (`J >= 3`)
leaves a word-legal `(J-1)`-run.  Hence for every word-legal `J`-run,
`span - min(g_L, g_R) <= Q*_{J-1}(M; q')`, so at a run attaining `Q*_J`,

    Q*_J <= Q*_{J-1} + min(g_L, g_R).

**The triple inequality (`J = 3`, hypothesis-free).**  For ANY three consecutive gaps
`g_L, w, g_R` of `M`,

    g_L + w + g_R <= F_2(M) + min(g_L, g_R).

In particular a depth-3 run can exceed `F_2(M) + s` only if both its flanks exceed `s`.

**Theorem A (the middle-sum lemma).**  In a literal word-legal `J`-run the `J - 2` middles
alternate between the classes `a` and `b`, so with `k := floor((J-2)/2)`,

    (sum of middles) >= k q'         (J even),
    (sum of middles) >= k q' + a     (J odd),

with equality iff every middle is its least positive representative.  Consequently
`Phi_J <= Q*_J - m_min(J) <= F(M + q') - m_min(J)` with `m_min(J) = k q'` or `k q' + a`.

**Theorem B (the `J`-parity of palindromes).**  For `J` even a literal word-legal `J`-run is
never a palindrome (its gap sequence read backwards differs from itself).  For `J` odd the
middle class word is forced palindromic (`a, b, a, ..., a`), so the run is a palindrome exactly
when its middle values are symmetric and its two flanks are equal.

## Proof

**Theorem D.**

1. Let `x_0 < ... < x_J` be word-legal with middle word `w_1 ... w_{J-2}` (the letter of
   `x_i -> x_{i+1}` is `w_i`).  Deleting the left flank leaves `x_1 < ... < x_J`, a
   `(J-1)`-run with middles `w_2 ... w_{J-2}`, a suffix of the word; deleting the right flank
   leaves `x_0 < ... < x_{J-1}` with middles `w_1 ... w_{J-3}`, a prefix.  A prefix of a legal
   word is legal (the tooth reading restricts); a suffix is legal (read it from the tooth the
   full reading has reached at that point).  For `J = 3` the remaining run has no middles and
   is legal.
2. Delete the smaller flank: the remaining word-legal `(J-1)`-run has span
   `span - min(g_L, g_R) <= Q*_{J-1}`.

**The triple inequality.**

3. `g_L + w` and `w + g_R` are each sums of two consecutive gaps of `M`, so each is
   `<= F_2(M)`.  Then `g_L + w + g_R <= F_2(M) + g_R` and `<= F_2(M) + g_L`; take the
   smaller.  (No legality is used; this is Theorem D at `J = 3` with `Q*_2 = F_2`.)  If
   `g_L + w + g_R > F_2 + s` then `min(g_L, g_R) > s`.

**Theorem A.**

4. In a literal run no middle is `= 0 (mod q')`, so every middle is a nonzero letter, and the
   nonzero letters strictly alternate (file 05, T3): the `J - 2` middles are alternately of
   class `a` and class `b`, i.e. `a, b, a, b, ...` or `b, a, b, a, ...`.
5. A middle of class `a` is a positive integer `= 2u' (mod q')`, hence `>= a`; one of class
   `b` is `>= b`; and `a + b = q'`.  With `J - 2 = 2k` (even) the middles pair off into `k`
   pairs of one class each, summing to `>= k q'`.  With `J - 2 = 2k + 1` (odd) there are `k`
   such pairs and one extra middle, of class `a` or `b`, worth at least `min(a, b) = a`.
   Equality holds iff every middle is exactly `a` or `b`.
6. `span = g_L + (sum of middles) + g_R`, so `g_L + g_R <= span - m_min(J) <= Q*_J - m_min(J)`,
   and `Q*_J <= F(M + q')` by the attainment identity (file 08).

Note on the recorded consequence.  `docs/novel/per-j-window-analogues.md` 1.1 goes on to state
`Phi_J <= F_2(M) + s_min(q') - m_min(J)`.  That inequality is the displayed one with `F(M+q')`
replaced by `F_2(M) + s_min(q')`, i.e. it ASSUMES the per-`J` increment inequality
`Delta_J = Q*_J - F_2(M) <= s_min(q')`, which is measured (file 17), not proved.  What is proved
here is `Phi_J <= Q*_J - m_min(J)`; the collapse of the flank envelope "at rate `q'` per two
levels" is a theorem relative to `Q*_J`, and relative to `F_2 + s_min` only under that
hypothesis.

**Theorem B.**

7. `J` even: the middle class word has even length `2k` and alternates, so it is
   `a b a b ... a b` or `b a b a ... b a`; its first and last classes differ.  Reversing the
   run reverses the middles, so the reversed run's first middle has the class of the original's
   last middle.  Since `a` and `b` are distinct residues mod `q'` (`4u' != 0`), a middle of class
   `a` cannot equal a middle of class `b`; so the reversed gap sequence differs from the
   original at the first middle.  No palindrome.
8. `J` odd: the middle class word has odd length and alternates, so it starts and ends with
   the same class: `a b a ... a` or `b a b ... b`, which read backwards is the same class word.
   The run is then a palindrome iff the middle values themselves are symmetric and `g_L = g_R`.

## Status

Kernel: none (written proofs only).  Ingredients used: `WordLegal.alt_take`,
`WordLegal.legal_take` (prefixes of legal words are legal, step 1), `TwoTeeth.teeth_letters`
(`a + b = q'`), T3 (file 05).

Verified computationally: the `Delta_J` table at m11..m41 (`research/perj_window.py`,
`research/perj_scanfree.py`); Theorem A's margins `+5, +10, +9` at m19, m29, m31; the
palindrome dichotomy at 13 measured cells (`J = 3, 4`: reversal pairs, never palindromes;
`J = 5`: unique and self-reverse, `(7,10,21,10,7)` at m29 and `(3,25,12,25,3)` at m31); the
triple inequality discharges 6 of 8 steps outright.

What is NOT proved and must not be read into this file: `Delta_J <= s_min(q')` (the per-`J`
increment inequality) and `Delta_J = O(1)` are measured statements; `Delta_3` measured
`-3, 2, 0, 2, 4, 3, 2, 0` at m11..m37.

## Relationship to the conjecture

Exact reductions inside the per-`J` family: they discharge the depth-3 obligation except at
triples with both flanks above `s_min`, and shrink the flank envelope relative to `Q*_J`.  They
establish neither `Delta_J <= s_min` nor `Delta_J = O(1)` (both measured), and the recorded
flank consequence assumes the former.  No progress on the size statement.

## Where it is used

The reduction of the depth-3 obligation to triples with both flanks above `s_min`; the
observation that the deep layers are the cheap ones (the flank envelope shrinks by `q'` every
two levels relative to `Q*_J`); the palindrome route applies only at odd `J`.

## Source

Constructor round 28 (`docs/novel/per-j-window-analogues.md` 1.1-1.4); the triple inequality
from the manager's round-27 reduction and Constructor R78; Prover A's L2 (file 19) is its
depth-1 analogue; `docs/proof-search/alignment-rules.md` 3.7.
