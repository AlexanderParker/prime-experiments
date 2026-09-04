# 19. Prover A's proved lemmas (L2, L3, L4) and Prover C's Lemma 1

## In plain words

Four bookkeeping lemmas around the two statements the budget inequality splits into. A pair of
neighbouring gaps is within budget whenever the smaller of the two is at most the new gear, so
any violation needs both gaps large. At the first column the two neighbouring gaps are equal,
and the budget there says exactly that the first twin-candidate column above the machine's top
prime is not too far out, which is a statement about the existence of a twin pair, not about
gaps. Moving one gear's strikes to a new position just slides the whole pattern, so if a pair
of gaps beats the record then every gear must be the only striker of some column in that
stretch. And in the family of machines with symmetric but arbitrary strike positions, the two
facts the real primes supply, that no gear strikes neighbouring columns and that the new gear's
letter is a third of the gear, pin gear 5 to its real position and fix what the letters are.

All four are written proofs, not kernel theorems.  They are collected here because the pair
statement and the chain statement (defined below) are the two halves into which the budget
inequality splits (file 08), and these lemmas are what is proved about them.

## Vocabulary

Column `k` is the pair `(6k-1, 6k+1)`; `M = {5..p}` is a machine with period `P` and openings
`O_M`; `q'` is the next prime after `p`; `F(M)` is the record and `F_2(M)` the largest sum of
two consecutive gaps.  The **pair statement** PS is `F_2(M) <= F(M) + q'`; the **chain
statement** is `Q*_J(M; q') <= F(M) + q'` for all `J >= 3` (file 08).  At an opening `x` write
`g_L`, `g_R` for the gaps immediately left and right of `x`.  `d_0(M)` is the least positive
opening of `M`.  A gear `q` is the **sole striker** of a column if it is the only gear of `M`
striking it.

For Prover C's lemma, the **tooth-counterfactual family**: a symmetric two-tooth machine is a
set of primes `q >= 5` where gear `q` strikes `k = +-v_q (mod q)` for some chosen
`1 <= v_q <= (q-1)/2`; the real machine is `v_q = u_q = round(q/6)`.  For the incoming gear
`q'` with tooth `v'`, the letters are `{2v' mod q', q' - 2v' mod q'}` and `a := min` of the
two; the **separation** of an old gear is `a_q := min(2v_q mod q, q - 2v_q mod q)`.  Two named
facts: **(L)** `a = 2 round(q'/6)`; **(T)** no old gear has adjacent teeth, `2v_q != +-1 (mod q)`
(for the real machine this is `AnchorChain.neighbour_of_hit`, file 02 (d)).

Classical translation: PS says the longest stretch with exactly one twin candidate inside beats
the longest with none by at most the next prime; `d_0` is, under a size condition, the column of
the least twin prime pair above `p`.

## Statement

**L1 (reduction, on record).**  The budget inequality `F(M+q') <= F(M) + q'` holds iff both PS
and the chain statement hold; and PS is implied by the budget inequality at the same rung.

**L2 (trivial discharge).**  For every opening `x`: `g_L + g_R <= F(M) + min(g_L, g_R)`.  Hence
PS holds at every opening whose smaller flank is `<= q'`; a violating pair needs both flanks in
`[q' + 1, F]`, so PS holds outright whenever `F_2(M) <= 2q' + 1`.

**L3 (the column-0 instance).**  The two openings adjacent to column 0 are `+-d_0`, the gaps at
0 are `(d_0, d_0)`, and PS at column 0 is exactly `2 d_0 <= F(M) + q'`, i.e.
`F(M) >= 2 d_0 - q'`.  Moreover `d_0 > (p-1)/6`, and if `6 d_0 + 1 < q'^2` then `6 d_0 +- 1`
are both prime and `d_0` is the column of the least twin prime pair both of whose members
exceed `p`.

**L4 (re-phasing and the sole-striker corollary).**  Let `S` be a set of gears of `M` and for
each `q in S` let `T_q` be a translate of `q`'s tooth pair.  The machine `M'` obtained by
moving the teeth of each `q in S` to `T_q` is a translate of `M`; in particular
`F(M') = F(M)`.  Corollary: if `g_L + g_R > F(M)` at an opening `x`, then every gear of `M` is
the sole striker of at least one column of `(x - g_L, x + g_R)` other than `x`.

**Prover C, Lemma 1 (what (T) and (L) say inside the family).**
(a) (L) holds iff `v' = round(q'/6)`.
(b) (T) holds at gear `q` iff `a_q >= 2` iff `v_q != (q-1)/2`.
(c) At gear 5, (T) forces the real tooth `v_5 = 1`; at gear 7 it allows `v_7 in {1, 2}`, of
which 1 is real.
(d) Unified reading: (T) + (L) say "no gear, old or new, has letter 1, and the incoming letter
is `(q' -+ 1)/3`"; the real machine has every gear's separation equal to `(q -+ 1)/3`.

## Proof

**L1.**  By the attainment identity (file 08), `F(M+q') = max(F_2(M), max_{J>=3} Q*_J)`, so
`F(M+q') <= F(M) + q'` iff `F_2(M) <= F(M) + q'` and every `Q*_J <= F(M) + q'`.  By the
deletion ladder (file 07), `F_2(M) <= F(M+q')`, so the budget inequality gives PS.

**L2.**
1. `g_L <= F(M)` and `g_R <= F(M)` (each is a gap).  So `g_L + g_R <= F(M) + g_R` and
   `<= F(M) + g_L`; take the smaller.
2. If `min(g_L, g_R) <= q'` then `g_L + g_R <= F(M) + q'`.  So a pair with `g_L + g_R > F + q'`
   has both flanks `> q'`, and both `<= F`; then `F_2 >= g_L + g_R >= 2q' + 2`.  Contrapositive:
   `F_2(M) <= 2q' + 1` gives PS.

**L3.**
3. Column 0 is an opening (file 03 (a)) and the opening set is closed under `k -> -k`
   (file 03 (c)).  `d_0` is the least positive opening, so there is no opening in `(0, d_0)`,
   and by the mirror none in `(-d_0, 0)`; `-d_0` is an opening.  So the openings adjacent to 0
   are `-d_0` and `d_0`, with gaps `(d_0, d_0)`, and PS at 0 reads `2d_0 <= F(M) + q'`.
4. For `1 <= k <= (p-1)/6`: `6k + 1 <= p`, so `5 <= 6k - 1 < p`; `6k - 1` is coprime to 6, so
   its least prime factor is a prime in `[5, 6k-1] subset [5, p]`: a gear of `M` strikes `k`.
   Hence `d_0 > (p-1)/6`.
5. If `6d_0 + 1 < q'^2`: neither member of column `d_0` has a prime factor in `{5..p}` (it is
   an opening), nor 2 or 3, hence none below `q'` (there is no prime strictly between `p` and
   `q'`); both members are below `q'^2`, so both are prime (file 01, Theorem 4).  For a column
   `k < d_0` with `6k - 1 > p`: `k` is struck, so a member has a prime factor `<= p`, and that
   member exceeds `p`, so it is composite -- no twin pair with both members above `p` sits
   below column `d_0`.

**L4.**
6. Let `t_q` be the shift with `T_q = t_q + {+-u_q}` (mod `q`), and choose `s` by CRT with
   `s = t_q (mod q)` for `q in S` and `s = 0 (mod q)` for `q not in S`.  A column `k` is open
   in `M'` iff for `q not in S`, `k` is not `= +-u_q (mod q)`, and for `q in S`, `k` is not
   `= t_q +- u_q (mod q)`; both say exactly that `k - s` is open in `M`.  So
   `O_{M'} = O_M + s`, the gaps of `M'` are those of `M`, and `F(M') = F(M)`.
7. Corollary.  Suppose `g_L + g_R > F(M)` at the opening `x`, and suppose some gear `q_0`
   strikes no column of `(x - g_L, x + g_R) \ {x}` alone.  Take `S = {q_0}` and `T_{q_0}` the
   translate of `q_0`'s tooth pair containing `x`'s residue.  In `M'`, `x` is struck by `q_0`;
   every other column of the open interval `(x - g_L, x + g_R)` was struck in `M` (they are
   not openings of `M`), and by the supposition each is struck by some gear other than `q_0`,
   whose teeth are unchanged, so it is still struck in `M'`.  Hence `M'` has no opening in the
   open interval `(x - g_L, x + g_R)`, a stretch of `g_L + g_R - 1` blocked columns, and a gap
   `>= g_L + g_R > F(M) = F(M')`.  Contradiction.

**Prover C, Lemma 1.**
8. (a) `2v'` lies in `[2, q' - 1]`, `q' - 2v'` is odd (`q'` odd) and `2 round(q'/6)` is
   even.  If `a = 2 round(q'/6)` then `a` is even, so `a = 2v'` (not the odd candidate), giving
   `v' = round(q'/6)`.  Conversely if `v' = round(q'/6) = u'` then `2v' = 2u' < q' - 2u'` (as
   `4u' < q'`, file 02), so `a = 2u' = 2 round(q'/6)`.
9. (b) With `1 <= v <= (q-1)/2`, `2v in [2, q-1]`.  `2v = -1 (mod q)` iff `2v = q - 1` iff
   `v = (q-1)/2`; `2v = 1 (mod q)` has no solution in that range (`2v = 1` and `2v = q + 1`
   are both out of range).  So (T) at `q` iff `v_q != (q-1)/2`.  And
   `a_q = min(2v, q - 2v) >= 2` iff `q - 2v >= 2` (as `2v >= 2` always) iff `v != (q-1)/2`.
10. (c) Gear 5: `v in {1, 2}`, and `(5-1)/2 = 2` is excluded, leaving `v_5 = 1 = u_5`.
    Gear 7: `v in {1, 2, 3}`, `3` excluded, leaving `{1, 2}`; `u_7 = 1`.
11. (d) By (b), (T) at every old gear is "`a_q >= 2`", i.e. no old gear has letter 1; by (a),
    (L) is "the incoming letter is `2u' = (q' -+ 1)/3`".  For the real machine
    `a_q = 2u_q = (q -+ 1)/3` at every gear (file 02).

## Status

Kernel: none for the four lemmas (written proofs only).  Ingredients used from the kernel:
`Mirror.mirror_gear` (L3), `AnchorChain.neighbour_of_hit` (the real machine satisfies (T)),
`Horizon.prime_of_no_prime_factor_lt` (L3, step 5).  Kernel-shaped items noted by Prover A and
not built: L2 as a one-line inequality on naturals; L3's equivalence; L4's translation lemma.

Verified computationally: L2 -- PS free through m31 (`F_2 = 11, 16, 25, 31, 39, 55, 68` against
`2q' + 1 = 27, 35, 39, 47, 59, 63, 75`), first content at m37; L3 -- `d_0 <= q'` for all 78,496
primes `5 <= p <= 10^6` (max `d_0/q' = 0.2857`), and the teeth-free form
"columns `1..d_0 - 1` blocked forces a gap `>= 2d_0 - q'`" is FALSE at the family member
`V(19) (1,1,4,3,5,2)` (`F = 26`, `d_0 = 25`, `q' = 23`); L4 -- every pair with
`g_L + g_R > F` certified at m11..m23 (20/20, 88/88, 124/124, 400/400, 130/130) and one-class
`P_5..P_9`; Lemma 1 -- checked on the 142,560-row family at m19 (`research/proof/
chain_teeth_r33.py`).

What is NOT proved, stated so it is not read into this file: PS itself (its column-0 instance
`F(M) >= 2d_0 - q'` is a twin-existence statement -- a twin pair in `(p, 6q' + 1]` would suffice
and is open); the chain statement (Prover C: no proof from (T) + (L), and a family member with
`Q*_4 = F + q'` exactly, so any proof must be exact); `d_0 <= q'` for all `p`.

## Relationship to the conjecture

Bookkeeping around the open pair and chain statements.  L3 locates where the real teeth enter
the pair statement: as a twin-existence statement at column 0.  Nothing here proves the pair
statement, the chain statement or the budget inequality; the accompanying verification tables
are measured.

## Where it is used

The split of the budget inequality into the pair and chain statements is the frame of the
round-32/33 proof attempts; L2 says where PS has content (both flanks above `q'`); L3 locates
the real teeth's entry into PS at column 0; L4 is the only mechanism that converts a two-flank
statement into a gap of `M`, and it is one-class as well.

## Source

Prover A, round 32 (`research/proof/pair_statement.md`, L1-L4); Prover C, round 33
(`research/proof/chain_from_teeth.md`, Lemma 1); the manager's one-class sole-coverer argument
(round 32) for L4.
