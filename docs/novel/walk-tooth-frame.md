# walk-tooth-frame - the walk from q^2 starts on the top gear's tooth, and the deepest layer it uses is decided by one primality test

Status: PROVED (mechanisms below, all elementary) + SCRIPT-VERIFIED (every prime gear
5..4999; 832,915 congruence checks at the transfer rule). Established round 37, branch
R2.a (`research/proof/self_feeding.md`). Prior-art check: NOT YET CHECKED.

## 1. WHAT IT IS

Plain language. Take the machine `{5..q}` and start at the column that holds `q^2` - the
bottom of the certified window - and walk upward to the first opening (the layered walk of
`docs/proof-search/anchor-235.md` 9c, which always lands on a twin pair). Three exact facts
about that walk, all of them about the *newest* gear rather than the old ones:

1. The walk begins standing on one of the top gear's two teeth, and the top gear's other
   tooth is a third or two thirds of `q` further on, so the top gear strikes the whole walk
   exactly once - at the very first column - and is then inert for the rest of it.
2. The largest gear that makes a hop anywhere in the walk is the top gear itself **if and
   only if `q^2 - 2` is prime**. One primality test decides the deepest layer of the whole
   recursion.
3. A gear that strikes a column `j` away from a twin pair's own column strikes the column
   `i` steps into that pair's walk one level up if and only if it divides one of four numbers
   built from `i` and `j` alone - the level does not appear.

Precise form. Write `c = 6^{-1} mod q` (so gear `q`'s teeth are the columns `+-c mod q`),
`u_q = min(c, q-c) = round(q/6)`, `k_0 = (q^2-1)/6` (the column whose upper member is `q^2`),
`L` = the walk length (landing column minus `k_0`), and `T` = the largest gear that is the
smallest striker of some traversed column.

* **(W1) Tooth start and one strike.** `6 k_0 = q^2 - 1 = -1 (mod q)`, so `k_0 = -c (mod q)`:
  the walk starts on a tooth. The next strike of `q` is `d = 2c mod q` columns higher, where
  `d = 2 u_q` when `q = 5 mod 6` and `d = q - 2 u_q` when `q = 1 mod 6`. Hence the top gear
  strikes the interval `[k_0, k_0 + L]` exactly once whenever `L < d`.
* **(W2) The square-gate top layer.** `T = q` iff `q^2 - 2` is prime, provided `L < d`.
* **(W3) The level-free transfer rule.** Let `(g, g+2) = (6k-1, 6k+1)` be a twin pair, so `k`
  is its shared first tooth column, and let `k_0' = k(g-1) = 6k^2 - 2k` be the first column of
  that pair's own walk (its upper member is `g^2`). If a gear `h` strikes column `k + j` then
  `h` strikes column `k_0' + i` iff

      h | (6j)^2 + 6i - 2   or   h | (6j)^2 + 6i          (h hit the lower member at k+j)
      h | (6j+2)^2 + 6i - 2 or   h | (6j+2)^2 + 6i        (h hit the upper member at k+j)

  Neither `k` nor `g` occurs. At `i = 0` the conditions are `h | (6j)^2 - 2` and
  `h | (6j+2)^2 - 2` - the square-gate numbers of the offsets - so at `j = +-1` the entire
  admissible set is `{7, 17, 31}`.
* **(W4) The pair's frame.** `k_0' = 6k^2 - 2k` sits exactly `2k` columns below the pair's
  twin-product column `6k^2` (`tooth-sharing-pinning` (c)); those are consecutive strikes of
  each member, so both newest gears strike the next level's walk interval once each, at
  distance `2k = (g+1)/3`.

## 2. WHY IT MIGHT BE NOVEL

The pieces are individually elementary - (W1) and (W4) are the two-teeth kill-spacing law
evaluated at a distinguished phase, (W2) is the least-prime-factor lemma applied to `q^2-2`,
(W3) is congruence substitution. What is not on record anywhere in this project, and what we
have not found stated elsewhere, is the combination: that the certified window opens exactly
on a tooth of the gear that certifies it, so the new gear is structurally absent from the
first third of the region it creates; and that the deepest layer of the layered walk is
decided by a single primality test on `q^2 - 2` with no exception in 667 walks. (W3) makes
precise a negative that is usually assumed: a sieve recursion of this shape carries no
information from a survivor's neighbourhood to that survivor's own action one level up,
because the carriers are pinned by offsets alone.

Honest shadow: `q^2 - 2` prime is the project's **square gate**
(`docs/proof-search/alignment-rules.md` 4.1) in its role as a necessity criterion for a gear;
(W2) is a different statement about the same number.

## 3. PROOF

(W1) `6 u = 1 (mod q)` defines the teeth `+-c`. The column `k_0` has upper member `q^2`, so
`6 k_0 + 1 = q^2 = 0 (mod q)`, i.e. `k_0 = -c (mod q)`. The two teeth are `+-c`, so the
distance up from `-c` to `+c` is `2c mod q`, and from `+c` back to `-c` is `q - (2c mod q)`.
For `q = 6m+5`, `c = m+1 = u_q` and `2c mod q = 2u_q`; for `q = 6m+1`, `c = 5m+1 = q - u_q`
and `2c mod q = q - 2u_q`. Verified at all 667 gears `q <= 4999`, 0 exceptions, and the
strike count inside `[k_0, landing]` matched the rule at all 667.

(W2) The first traversed column carries the members `q^2 - 2` and `q^2`. Since `q^2 - 2 < q^2`,
a composite `q^2 - 2` has a prime factor below `q`; so gear `q` is the smallest striker there
iff `q^2 - 2` is prime. For `q` to be the smallest striker of a later traversed column, one
member must be `q m` with `m` free of primes below `q`, hence `m` prime and `m >= q`; `m = q`
is the start, so `m >= q+2` and the column is at least `d` further on, which needs `L >= d`.
Script: `research/anchor235/r37/sf_walks.py` - 153 walks with the gate open, all with `T = q`;
514 with it shut, none with `T = q`; 0 exceptions either way.

(W3) If `h | 6(k+j) - 1` then `6k = 1 - 6j (mod h)`, so `g = 6k - 1 = -6j` and
`g^2 = (6j)^2 (mod h)`. Column `k_0' + i` has members `g^2 - 2 + 6i` and `g^2 + 6i`; substitute.
If instead `h | 6(k+j) + 1` then `g = -(6j+2) (mod h)` and `g^2 = (6j+2)^2`. Script:
`sf_birth.py` - 341 twin pairs `g <= 20000`, `|j| <= 8`, `i <= 40`, 832,915 checks,
0 mismatches; and the `i = 0`, `j = +-1` census on real machines produced exactly
`{7: 1663, 17: 1149, 31: 281}` carry-overs of 50,906 triples.

(W4) `6 k_0' + 1 = 6k(6k-2) + 1 = (6k-1)^2 = g^2`. Modulo `g` the teeth are `+-k` and
`k_0' = -k`, so the next strike is `2k` on; modulo `g+2` the teeth are `{k, 5k+1}` and
`k_0' = 3k+1`, so the next strike is again `2k` on, at `6k^2`. Script: `sf_walks.py` W4 -
667 landings, spacing brute-forced over the whole `2k` gap at the 30 smallest, 0 failures.

Exception counts, all ranges as stated: (W1) **one** (`q = 53`, `L = 27 >= d = 18`); (W2)
**zero**; (W3) **zero**; (W4) **zero**.

## 4. IMPLICATIONS

Inside the project: the walk from `q^2` - the object branch 1e/7d identified as what a proof
must control - is decided by the *old* gears, not the new one. Gear 5 makes 40.4% of the
18,743 hops over all 667 walks, gear 7 another 17.5%, gears above `sqrt(q)` only 15.1%, and a
median of 3.2% of a machine's gears do all the work of any one walk. (W1) says the newest gear
cannot lengthen the walk beyond its own tooth arc without striking twice, which happens once in
range. This is a position fact of the same family as the gear-5 lock and the corridor law: it
constrains where the machine can act, not how long a stretch can be.

Outside: an exact local description of the boundary condition of a sieve at its own horizon -
the sifting range `(q, q^2]` begins on a zero of the last prime added, which is a small,
checkable structural statement about Eratosthenes-type sieves in the `6k+-1` frame.

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

* The twin prime conjecture through the route: (W1)'s hypothesis `L < d ~ q/3` is a
  twin-Bertrand-strength statement at scale `q/3` (an opening within `q/3` columns of `q^2/6`),
  far stronger than the root's `L < W`. It is measured, not proved; median `L/d = 0.0231`,
  max 0.5217 above `q = 53`.
* `q^2 - 2` prime: the square gate's density (153 of 667 primes `q <= 5000`) is a
  Hardy-Littlewood-type count for the polynomial `x^2 - 2`; (W2) gives it a new meaning inside
  the machine but no new information about it.

## 6. PRIOR-ART CHECK

NOT YET CHECKED (round 37 branch had no web access). Terms to try: sieve of Eratosthenes
`6k+-1` wheel "starts on a residue of the last prime"; "first prime gap above p^2"; largest
prime factor of `p^2 - 2` sieve layer; twin prime sieve layered recursion deepest level.
