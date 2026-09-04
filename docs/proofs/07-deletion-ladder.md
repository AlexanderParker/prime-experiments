# 7. The deletion ladder: `F_j(M) <= F(M + j-1 new gears)`

## In plain words

Take the longest stretch of the old machine that has a few openings inside it, and add that
many new gears. By lining up each new gear so that it strikes one of those interior openings,
which is always possible with one gear per opening, the whole stretch becomes a single gap of
the bigger machine. So the bigger machine's record is at least that stretch: a lower bound on
the record after adding gears, read off the old machine.

## Vocabulary

Column `k` is the pair `(6k-1, 6k+1)`; `M` is a finite set of gears with period `P` and
`P`-periodic opening set `O_M`.  `F_j(M)` is the largest sum of `j` consecutive gaps of `M`,
i.e. the largest `x_j - x_0` over `j + 1` consecutive openings `x_0 < x_1 < ... < x_j`
(a stretch spanning `j - 1` interior openings); `F_1 = F` is the record.  A gear `q` strikes
column `k` iff `k = +-u_q (mod q)`, `6u_q = q -+ 1` (file 02).

Classical translation: `F_j(M)` is the longest stretch of columns containing exactly `j - 1`
twin candidates (pairs free of the primes of `M`) strictly inside it, with candidates at both
ends.

## Statement

**Theorem.**  Let `q_1, ..., q_r` (`r >= 1`) be distinct gears not in `M`.  Then

    F_{r+1}(M) <= F(M + q_1 + ... + q_r).

In particular `F_2(M) <= F(M + q')` for any gear `q'` not in `M`, and, taking the next `j - 1`
primes after the top gear of `M`,

    F_j(M) <= F(M + the next j-1 primes).

## Proof

1. Let `x_0 < x_1 < ... < x_r < x_{r+1}` be consecutive openings of `M` with
   `x_{r+1} - x_0 = F_{r+1}(M)`; the interior openings are `x_1, ..., x_r`, one per new gear.
2. Since `gcd(P, q_i) = 1` for each `i` and the `q_i` are distinct primes, the Chinese
   remainder theorem gives an integer `s >= 0` with

       s = 0 (mod P)    and    s = u_{q_i} - x_i (mod q_i)   for i = 1, ..., r.

3. Translation by `s` preserves the openings of `M` (`s` is a multiple of `P`), so
   `x_0 + s < ... < x_{r+1} + s` are consecutive openings of `M`.  For each `i`,
   `x_i + s = u_{q_i} (mod q_i)`: the interior opening `x_i + s` is struck by gear `q_i`.
4. Let `M' = M + q_1 + ... + q_r`.  Its openings are openings of `M` not struck by any `q_i`.
   By step 3 no opening of `M'` lies strictly between `x_0 + s` and `x_{r+1} + s`.  `M'` has
   openings (`prod (q - 2) > 0` of them per period, e.g. column `0`), so the two openings of
   `M'` nearest to that interval on either side are at distance at least
   `x_{r+1} - x_0 = F_{r+1}(M)` apart.  Hence `F(M') >= F_{r+1}(M)`.

(If some endpoint `x_0 + s` or `x_{r+1} + s` happens to be struck as well, the gap of `M'` is
longer still; the inequality is unaffected.)

Remark, on record with the law: as an induction step it is circular -- it prices `F_2(M)` by
the very `F` the next rung is meant to certify -- and its slack `F(M+q') - F_2(M)` is
`3, 1, 0` at 29->31, 37->41, 41->43.

## Status

Kernel: none (written proof only).  The `r = 1` case is also the lower half of the attainment
identity (file 08), where the same translation appears with a legal word.

Verified computationally: all 32 `(M, j)` pairs at which both sides are known exactly, with one
equality `F_2(17) = 25 = F(19)` and tightest non-equality `F_2(37) = 90` against `F(41) = 91`
(`alignment-rules.md` 3.4).  Free caps past the scan wall: `F_2(41) <= 103`, `F_2(43) <= 118`,
`F_3(43) <= 145`, `F_4(43) <= 161`, `F_2(53) <= 161`.

## Relationship to the conjecture

A lower bound on the record of the bigger machine, the direction opposite to the budget
inequality; useful for capping old spectra from records above the step, and circular as an
induction step.  No measured input.

## Where it is used

Caps on the old machine's spectrum from records above the step (the spectrum-plus-depth
certificate); the reduction of the pair statement to the budget inequality at the same rung
(file 19, L1).

## Source

Mechanic (the `F_2(M) <= F(M + one gear)` form, credited "Mechanic, proved" in Constructor
X36); the `r`-gear generalisation and the three-line CRT proof in
`docs/proof-search/alignment-rules.md` 3.4.
