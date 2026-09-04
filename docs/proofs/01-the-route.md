# 1. The route: twin primes are infinite iff every window holds an opening

## In plain words

Picture the primes from 5 upward as gears turning together over a long row of columns. Each
column holds a pair of numbers two apart, and a gear strikes the columns where it divides one
of the pair. A column no gear strikes is an opening. For the gears up to some size there is a
stretch of columns, the window, where a strike is the only way a pair can fail to be twin
primes; so an opening inside the window is a twin prime pair, with no exceptions. The theorem
says the whole twin prime question is exactly this: twin primes go on for ever if and only if,
however many gears you take, the window always has an opening in it. Nothing is lost in the
translation, and a second form says it is enough that no stretch of columns without an opening
is longer than the window.

## Vocabulary

A **column** `k >= 1` is the pair `(6k-1, 6k+1)`.  A **gear** is a prime `q >= 5`; gear `q`
**strikes** column `k` if `q` divides `6k-1` or `6k+1`.  The **machine** `{5..y}` is the set of
primes in `[5, y]`; column `k` is an **opening** of the machine if no gear of the machine strikes
it.  The **window** of `{5..y}` is the set of columns whose members lie in `(y, y^2]`, i.e.
`y < 6k-1` and `6k+1 <= y^2`.

Classical translation.  Since 2 and 3 never divide `6k +- 1`, column `k` is an opening of
`{5..y}` iff no prime `q <= y` divides `6k-1` or `6k+1`.  The kernel states this for the lower
member `m = 6k-1` as `Survivor y m`: no prime `q <= y` divides `m` or `m+2`.  Everything below is
stated for `m`; the column form is recovered by `m = 6k-1`.

## Statement

**Theorem 1 (window lemma).**  Let `y < m`, `m + 2 <= y^2` and `m > 1`.  Then
`Survivor y m` holds iff `m` and `m+2` are both prime.

**Theorem 2 (the route).**  The set `{p : p and p+2 prime}` is infinite iff for every `N` there
are `y >= N` and `m` with `y < m`, `m + 2 <= y^2` and `Survivor y m`.

In column form: twin primes are infinite iff for every bound there is a machine `{5..y}` with
an opening inside its window.

**Theorem 3 (gap form).**  Fix `y`.  If every interval `(a, a+G]` contains an `m` with
`Survivor y m`, and `y + G + 2 <= y^2`, then the window of `y` contains a survivor.

**Theorem 4 (horizon).**  If `y < m < y^2` and `m` is composite, then `m` has a prime factor
`p < y` (strictly).  Consequently, if `y < m` and `m + 2 < y^2` and no prime `p < y` divides
`m` or `m+2`, then `m` and `m+2` are both prime.

## Proof

**Theorem 1.**

1. (`<=`)  Let `m` and `m+2` be prime and let `q <= y` be prime.  A prime `q` divides the prime
   `m` only if `q = m`; but `q <= y < m`.  Likewise `q <= y < m+2` gives `q` does not divide
   `m+2`.  So `Survivor y m`.
2. (`=>`)  Suppose `m` is composite.  Its least prime factor `r` satisfies `r^2 <= m`, so
   `r <= sqrt(m) <= sqrt(m+2) <= y` (using `m+2 <= y^2`).  Then `r` is a prime `<= y` dividing
   `m`, contradicting `Survivor y m`.  The same argument with `m+2` in place of `m` (least prime
   factor `r <= sqrt(m+2) <= y`) shows `m+2` is prime.  (The kernel phrases the hypothesis as
   `Nat.sqrt (m+2) <= y`, which follows from `m + 2 <= y * y`.)

**Theorem 2.**

3. (`<=`)  Assume the window hypothesis.  Given any `a`, apply it with `N = a + 2`: there are
   `y >= a+2` and `m > y` with `m + 2 <= y^2` and `Survivor y m`.  Then `m > 1`, so by Theorem 1
   `m` and `m+2` are prime, and `m > a`.  A set of natural numbers with no upper bound is
   infinite.
4. (`=>`)  Assume infinitely many twin pairs and fix `N`.  Choose a twin start `p > N^2 + 8`.
   Put `y = floor(sqrt(p+2)) + 1`.  Then:
   - `p + 2 <= y^2`, because `y > sqrt(p+2)`;
   - `y < p`: for `p >= 9`, `(p-1)^2 = p^2 - 2p + 1 > p + 2`, so `sqrt(p+2) < p - 1` and
     `y <= p - 1`;
   - `N <= y`: `N^2 <= p + 2` gives `N <= sqrt(p+2) < y`;
   - `Survivor y p` by Theorem 1 (`<=`), since `y < p`, `p + 2 <= y^2`, `p > 1`.
   So the window hypothesis holds at `N`.

**Theorem 3.**  Take `a = y`: there is `m` with `y < m <= y + G` and `Survivor y m`, and
`m + 2 <= y + G + 2 <= y^2`.

**Theorem 4.**

5. A composite `m > 1` has a prime factor `r = minFac m` with `r^2 <= m < y^2`, hence `r < y`.
   (`m > 1` because `y < m` and `y >= 1`; if `y = 0` there is no `m` with `0 < m < 0`.)
6. If no prime below `y` divides `m` and `y < m < y^2`, step 5 forces `m` prime.  Apply this to
   `m` and to `m+2` (which satisfies `y < m + 2 < y^2`).

**Column form.**  With `m = 6k-1`, `Survivor y m` says no prime `q <= y` divides `6k-1` or
`6k+1`; primes 2 and 3 never do, so this is exactly "no gear of `{5..y}` strikes `k`", i.e. `k`
is an opening.  The window condition reads `(y+1)/6 < k <= (y^2-1)/6`.  Theorem 3 is the reason
the record `F(M)` (the largest distance between consecutive openings, see file 06) is the object
of the whole programme: the openings of `{5..y}` are periodic, every `F` consecutive columns
contain one, so as soon as the window contains `F({5..y})` consecutive columns it contains an
opening, and by Theorem 1 that opening is a twin prime pair.

## Status

Kernel: `BlockedSlots.survivor_iff_twin`, `BlockedSlots.twin_of_survivor`,
`BlockedSlots.twins_infinite_of_survivor_in_window`,
`BlockedSlots.survivor_in_window_of_twins_infinite`,
`BlockedSlots.twins_infinite_iff_survivor_in_window`,
`BlockedSlots.survivor_in_window_of_gap_bound`; `Horizon.exists_prime_factor_lt`,
`Horizon.prime_of_no_prime_factor_lt`, `Horizon.twin_of_no_prime_factor_lt`
(`proofs/BlockedSlots.lean`, `proofs/Horizon.lean`).

Verified computationally: the openings of `{5..y}` inside the window and the twin pairs there
coincide for `y = 11..1009` (`alignment-rules.md` 4.1).

## Prior art, and what is new

**Leverages.**  Standard sieve arithmetic only: the Chinese remainder theorem, and the
Eratosthenes/Legendre fact that a composite below `y^2` has a prime factor below `y`, which is
Theorem 4 here.  The window statement is in print: Ziller & Morack 2017 (arXiv:1706.00317)
Theorem 4.1 proves that their Conjecture 6, `h_2(n) < p_n^2 - p_n`, implies Goldbach and the
infinitude of prime pairs for every even difference, and their Conjecture 5 is the same
`(y, y^2]` window; the one-class analogue is Mercer 2018 Theorem 1, whose Lemma 2 is Theorem 1
of this file verbatim.  The two-class object those statements are about is Ziller & Morack's
paired Jacobsthal function `h_2`, of which the record `F` here is the realised (real-teeth)
version.

**New.**  The biconditional and its mechanisation: the literature carries the sufficiency
(Ziller-Morack Theorem 4.1, Mercer Lemma 2), not the equivalence, and not a kernel-checked one.
The gap form (Theorem 3) is what makes the record `F(M)` the object the ladder certifies, since
it converts "find a twin" into "bound a stretch"; the column translation with the anchor
factored out is the frame every later file computes in.

**Not new.**  The forward implication is Ziller-Morack 2017 Theorem 4.1 in gear language, and
the horizon lemma is the standard least-prime-factor bound; the project's target `F(y) < y^2/6`
is their Conjecture 6 up to the linear term, stated for the realised teeth rather than the
maximum over class assignments.  The record also fixes why no class-count-only route closes it:
the window sits at sieve dimension two, below the Diamond-Halberstam-Richert sifting limit
`beta_2 = 4.2664` (`research/proof/iwaniec_two_class.md`), which is where the parity barrier
shows up on this side.

## Relationship to the conjecture

This is the route itself: it converts the conjecture into "every window holds an opening" and
loses nothing.  It establishes no size statement and depends on nothing measured.  Everything
else in this directory is machinery for the record `F(M)`, which enters the conjecture only
through Theorem 3.

## Where it is used

Everything.  It is the reduction of the conjecture to a statement about openings, and Theorem 3
is why the record `F(M)` and the budget inequality `F(M+q') <= F(M) + q'` (a target, not a law)
are what the search certifies rung by rung.

## Source

`docs/twin-prime-program.md` (the reduction); `docs/proof-search/alignment-rules.md` 4.1;
`proofs/BlockedSlots.lean` header.
