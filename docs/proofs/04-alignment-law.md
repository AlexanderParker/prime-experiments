# 4. The alignment law: the longest run of consecutive openings

## In plain words

Look for the longest stretch of columns that are all open, one after another. The answer is
decided entirely by the smallest gear: it is the long untouched stretch of that gear, and
adding more gears never shortens it, because somewhere in the cycle all the other gears line up
so that their strikes miss that stretch. With gear 5 present the longest such stretch is two
columns, so openings come only as single columns and as pairs of neighbours, and the proof
counts exactly how many pairs there are. What it says nothing about is how far apart the
openings are, which is the real question.

## Vocabulary

Column `k` is the pair `(6k-1, 6k+1)`; gear `q >= 5` strikes `k` iff `k = +-u_q (mod q)`,
where `6u_q = q -+ 1` (file 02).  For a finite set `G` of gears an **opening** is a column no
gear of `G` strikes; a **run** is a maximal set of consecutive open columns.  The **long arc**
of `q` is `A(q) := q - 2u_q - 1`, the number of consecutive open residues of `q` between its
two teeth on the long side (file 02 (b)).

Classical translation: a run of `A` consecutive openings is a block of `A` consecutive columns
`(6k-1, 6k+1)`, `(6k+5, 6k+7)`, ... none of whose `2A` members is divisible by a prime of `G`.

## Statement

Let `G` be a finite nonempty set of gears with smallest gear `q_0`.

**(i)**  The longest run of consecutive openings of `G` has length exactly `A(q_0)`, whatever
the other gears are: `2, 4, 6, 8, 10, 12` for `q_0 = 5, 7, 11, 13, 17, 19`.

**(ii)**  If `5 in G` the longest run is 2: the opening set is a disjoint union of isolated
openings and adjacent pairs ("dominoes").

**(iii)**  The number of adjacent open pairs per period is `prod_{q in G} (q - 4)`.  With
`5 in G` these are the dominoes, and the isolated openings number
`prod (q - 2) - 2 prod (q - 4)`.

**(iv)**  If `5 in G`, or `7 in G`, or `|G| >= 2`, isolated openings exist: the shortest run
has length 1.

## Proof

**Lemma (the long arc is strictly increasing in the gear).**

1. If `q = 1 (mod 3)` then `6u_q = q - 1`, `2u_q = (q-1)/3`, and `A(q) = (2q - 2)/3`.  If
   `q = 2 (mod 3)` then `2u_q = (q+1)/3` and `A(q) = (2q - 4)/3`.
2. For gears `q < q'` we have `q' >= q + 2`, so `A(q') >= (2q' - 4)/3 >= 2q/3 > (2q-2)/3 >= A(q)`.

**(i)**

3. *Upper bound.*  A run of consecutive openings contains no column struck by `q_0`, so it lies
   strictly between two consecutive struck columns of `q_0`.  Those are `2u_{q_0}` or
   `q_0 - 2u_{q_0}` apart (file 02 (b)), so the run has at most `q_0 - 2u_{q_0} - 1 = A(q_0)`
   columns.
4. *Lower bound.*  For every `q in G`, `A(q) >= A(q_0)` by the Lemma, so the residues
   `u_q + 1, ..., u_q + A(q_0)` all lie in the long arc `u_q + 1, ..., q - u_q - 1` of `q` and
   are open for `q`.  By the Chinese remainder theorem choose `k` with `k = u_q (mod q)` for
   every `q in G` simultaneously.  Then `k + 1, ..., k + A(q_0)` are open for every gear: a run
   of length `A(q_0)`.

**(ii)**  With `5 in G`, `q_0 = 5`, `u_5 = 1`, `A(5) = 2`.

**(iii)**

5. Columns `k` and `k+1` are both open iff for every gear `q`,
   `k mod q not in {u_q, -u_q, u_q - 1, -u_q - 1}`.  These four residues are distinct for
   `q >= 5`: `u_q != -u_q` (as `0 < 2u_q < q`), `u_q != u_q - 1`, `u_q != -u_q - 1` (as
   `0 < 2u_q + 1 < q`), `-u_q != u_q - 1` (as `0 < 2u_q - 1 < q`).  So each gear admits
   `q - 4` residues, and by CRT there are `prod (q - 4)` adjacent open pairs per period.
6. With `5 in G` no run has three columns, so every adjacent open pair is a whole run; the
   openings number `prod (q-2)` (CRT: `q - 2` open residues per gear), each domino uses two,
   the rest are isolated.

**(iv)**

7. If `5 in G`: take `k = 0 (mod 5)`; then `k - 1 = 4 = -u_5` and `k + 1 = 1 = u_5` are both
   struck by 5, while `k` is open for 5; choose `k` open for the other gears by CRT.  The same
   works for `7` (`u_7 = 1`).
8. If `q_1 != q_2` are in `G`: take `k = u_{q_1} + 1 (mod q_1)` and `k = u_{q_2} - 1 (mod q_2)`.
   Both residues are open (`u + 1 = -u` would need `2u + 1 = 0 (mod q)`, impossible as
   `0 < 2u + 1 < q`; similarly `u - 1`), and `k - 1 = u_{q_1}`, `k + 1 = u_{q_2}` are struck.
   Choose `k` open for the remaining gears by CRT.
9. (The hypothesis is needed: a single gear `q >= 11` has shortest run `2u_q - 1 >= 3`.)

## Status

Kernel: none (written proof only).  The ingredients used are file 02 (b) and the Chinese
remainder theorem.

Verified computationally: zero failures over 103 gear sets, including sets of five and six gears
(`research/alignment.py`, `docs/twin-prime-program.md` 26c); the domino count checked at small
sets (`alignment-rules.md` 1.5).

Limitation recorded with the law: it says "somewhere in the period"; the period is the primorial
while the window of `{5..y}` is the first `~y^2/6` columns.

## Prior art, and what is new

**Leverages.**  Standard (CRT) and the arcs of file 02.  The domino count `prod (q-4)` is the
classical local-factor count for a prime quadruplet; the record marks that family KNOWN
(`docs/proof-search/alignment-rules-index.md` H2, `c_q(g) = q - nu_q({0,2,6g,6g+2})`, the
Hardy-Littlewood local factor; `docs/novel/matrix-formulation.md` records Schemmel 1869 for the
same values).

**New.**  The exact evaluation of the longest run of consecutive openings as the long arc of the
smallest gear, uniform in every other gear, together with the strictly-increasing-arc lemma that
makes adding gears harmless; and the consequences (ii)-(iv), that with the anchor's gear 5
present the opening set is exactly isolated points and dominoes, and that isolated openings
always exist.  What it is useful for is negative and load-bearing: it is the strongest positive
alignment statement the corpus has, and it is about the primorial period, not the window.

**Not new.**  The counting corollaries are the standard local-factor counts in gear language,
and the argument is the usual "CRT realises every relative phase somewhere in the period", as
the file's own source note says.  Prior art for the run-length statement itself is not checked;
it is not carried as an entry in `docs/novel`.

## Relationship to the conjecture

A positive alignment statement, but about runs of consecutive openings somewhere in the
primorial period, not about their spacing inside the window; it has no bearing on the open size
statement (the budget inequality) and depends on nothing measured.

## Where it is used

The structural fact that the opening set of any machine containing gear 5 is a union of
isolated points and dominoes (so the pattern supports prime quadruplets at every level); the
observation that `F(M) >= 2` and that runs of openings say nothing about the spacing between
them, which is the whole problem.

## Source

`docs/twin-prime-program.md` 26c-26d (statement, the CRT reason, the 103-set check);
`docs/proof-search/alignment-rules.md` 1.5.  The proof written out here is the CRT argument
stated there.
