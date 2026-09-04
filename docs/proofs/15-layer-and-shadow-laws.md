# 15. The layer law and the shadow law

## In plain words

When the window grows from the square of one prime to the square of the next, the new part
contains composite numbers that need explaining. Almost all of them are already struck by a
gear smaller than the prime that just joined; the only new ones are that prime's square and its
products with the next few primes, a handful of explicit numbers per layer. And a gear never
explains anything below its own square. So each layer is simple; the difficulty is only in the
number of layers.

## Vocabulary

Column `k` is the pair `(6k-1, 6k+1)`; the members of a set of columns are the numbers
`6k +- 1`.  For a prime `y` and the next prime `y'`, the **layer** is the interval of members
`(y^2, y'^2)`: the new part of the window when the horizon advances from `y^2` to `y'^2`
(file 01).  For a prime `q` and a finite set `S` of members, gear `q`'s **ledger line** is
`R_q(S) :=` the number of composites `m in S` whose least prime factor is exactly `q`; its
**partners** are the cofactors `m/q` of those `m`.  "`q` exposes `m`" means `q | m`.

Classical translation: a composite in the layer is exposed by the smallest prime dividing it;
the theorems below say which gear that is, and that a gear exposes nothing below its own square.

## Statement

**(a) Slot cap.**  A prime `q >= 3` never divides both members of one column.

**(b) Which gear exposes a layer composite.**  Let `y < y'` with no prime strictly between,
and let `1 < m < y'^2` be composite.  Then the least prime factor of `m` is `< y` or `= y`.

**(c) The semiprime shape.**  If `1 < m`, the least prime factor of `m` is exactly `y`, and
`y^2 < m < y^3`, then `m = y c` with `c` prime and `y < c`.

**(d) Layer law.**  Let `y < y'` be consecutive primes with `y'^2 <= y^3` (true for
consecutive primes from `y = 3` on, since `y' < 2y`).  Every composite `m` in the open layer
`y^2 < m < y'^2` is either exposed by a prime strictly below `y`, or is `y c` with `c` prime,
`y < c`.  So the novel workload of gear `y` in the layer is exactly `{y^2} cup {y c : c prime}`,
and since `y c < y'^2` forces `c < y'^2/y < 4y`, these are at most a handful of explicit numbers.

**(e) Shadow law.**  A composite whose least prime factor is `q` is at least `q^2`.  Hence if
every member of `S` lies in `(1, q^2)`, `R_q(S) = 0`: a gear's ledger line opens at `q^2`.

**(f) One line, exactly.**  If every member of `S` lies in `(1, q^3)` and `q` is prime, then
`R_q(S)` equals the number of primes `c >= q` with `q c in S`.

## Proof

**(a)**  If `q | m` and `q | m + 2` then `q | 2`, so `q <= 2`.

**(b)**
1. Let `r` be the least prime factor of the composite `m`.  Then `r^2 <= m < y'^2`, so
   `r < y'`.
2. If `r > y` then `r` is a prime strictly between `y` and `y'`, contradicting the hypothesis.
   So `r <= y`, i.e. `r < y` or `r = y`.

**(c)**
3. `y` is prime (it is a least prime factor) and `y | m`; write `m = y c`.  From `y^2 < m`,
   `c > y`; from `m < y^3`, `c < y^2`.
4. If `c` were composite, its least prime factor `r` would satisfy `r^2 <= c < y^2`, so
   `r < y`; but `r | m` and the least prime factor of `m` is `y`, so `y <= r`.  Contradiction.
   So `c` is prime.

**(d)**
5. `m` composite with `y^2 < m < y'^2 <= y^3`.  By (b) its least prime factor `r` is `< y` or
   `= y`.  If `r < y`, `r` is a prime below `y` dividing `m`.  If `r = y`, (c) applies with
   `y^2 < m < y^3` and gives `m = y c`, `c` prime, `c > y`.
6. `c = m/y < y'^2/y`; for consecutive primes `y' < 2y` (Bertrand), so `c < 4y`: the partner
   primes lie in `(y, 4y)`, and `y c` lies in the layer only for the one to three primes `c`
   with `y'^2/y > c > y` (the record's "one to three explicit numbers per layer").
   (The kernel keeps `y'^2 <= y^3` as a hypothesis and never uses Bertrand.)

**(e)**
7. If `m > 1` is composite with least prime factor `q`, then `q^2 <= m`.  If `1 < m < q^2` for
   every `m in S`, no `m in S` is rooted at `q`, so the filtered set is empty and
   `R_q(S) = 0`.  (The guard `1 < m` matters: `minFac 0 = 2` would otherwise put `0` on gear
   2's line.)

**(f)**
8. A composite `m in S` rooted at `q` has `q^2 <= m < q^3` by (e); if `m = q^2` its partner is
   `c = q`; if `q^2 < m < q^3`, (c) gives `m = q c` with `c` prime, `c > q`.  So every member
   of the line is `q c` with `c` a prime `>= q`, and `m -> m/q` is injective on the line
   (`q | m` for every member).  Conversely, for a prime `c >= q`, `q c` is composite (a product
   of two primes) with least prime factor `q` (a prime divisor of `q c` divides `q` or `c`,
   and `q <= c`), so `q c in S` puts `c` in the partner set.  Hence
   `R_q(S) = |{c prime : q <= c, q c in S}|`.

## Status

Kernel: `Layer.slot_cap` (a); `Layer.minFac_lt_or_eq` (b); `Layer.eq_mul_prime_of_minFac_eq`
(c); `Layer.layer_novelty` (d, with `y'^2 <= y^3` as hypothesis); `Gear.sq_le_of_minFac_eq`,
`Gear.R_eq_zero_of_below_sq` (e); `Gear.semiprime_of_fiber`, `Gear.not_prime_mul`,
`Gear.minFac_mul`, `Gear.partners`, `Gear.R_eq_card_partners`, `Gear.mem_partners`,
`Gear.window_bounds` (f); the per-gear cap `Gear.R_le_card_multiples`, `Gear.R_prefix_le`
(`R_q` over the first `t` columns is at most `6t/q + 2`).  Step 6's use of Bertrand is
written.

Verified computationally: the nine layers 13->17 .. 43->47; seven of nine owe nothing in-band
beyond `y^2`, the exceptions being `221 = 13 x 17` beside the prime 223 and `437 = 19 x 23`
beside the prime 439 (`alignment-rules.md` 2.9).

## Prior art, and what is new

**Leverages.**  Standard throughout: the least prime factor of a composite `m` satisfies
`r^2 <= m` (the Eratosthenes/Legendre bound, which is file 01's Theorem 4), and step 6 uses
Bertrand's postulate for `y' < 2y` -- the kernel avoids even that by carrying `y'^2 <= y^3` as a
hypothesis.

**New.**  Nothing here is offered as new mathematics.  What it adds to the route is exact
bookkeeping: the novel workload of a section is precisely `{y^2}` together with `{y c : c
prime}`, a gear's ledger line opens at `q^2`, and one line counts partner primes exactly below
`q^3`.  That is what licenses grading a window's certificate by section without loss, and it
locates the difficulty in the number of sections rather than inside any one of them.

**Not new.**  All four statements are the standard least-prime-factor argument in the project's
vocabulary; the layer law is "a composite below the next square is caught by a prime below the
current one", i.e. the sieve of Eratosthenes read one section at a time.  Prior art for the
ledger-line form is not checked -- the result is not carried as an entry in `docs/novel`.

## Relationship to the conjecture

Bookkeeping about which gear does the work in each layer (equivalently, that graded sieving
inside a window loses nothing).  No bearing on the record or the budget inequality; nothing
measured.

## Where it is used

"The tower's complexity lives in the number of layers, never inside one"; full-set sieving is
equivalent to graded sieving inside a window (a gear first matters in its own section); the
square gate (a gear is droppable from a window's certificate iff it owns no pseudo-twin there).

## Source

The session's layer law (`docs/band-attribution.md`, `docs/gear-at-infinity.md`); Formalist
`proofs/Layer.lean`, `proofs/Gear.lean`; `docs/proof-search/alignment-rules.md` 2.9.
