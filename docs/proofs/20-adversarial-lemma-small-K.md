# 20. The adversarial lemma for small K, and the exact ladder A(K)

## In plain words

Every prime above three, when it is used to strike out pairs of numbers, removes exactly two
positions out of every stretch of that many positions, and the gap between the two removed
positions is not free: it is fixed by the prime itself, at one third of the way round.  Suppose
an adversary is allowed to pick any primes he likes -- not the small ones the machine actually
carries, any of them -- and to slide each one wherever he wants, and asks how long a stretch of
consecutive positions he can wipe out completely.  With one prime he manages one position, with
two he manages four, with three six, with four fifteen, with five twenty-one, with six
twenty-seven; and no choice of primes does better.  This file proves those six numbers, and it
proves the statement the project actually wants for up to ten primes: ten primes, whichever ten
they are and however placed, cannot wipe out a stretch as long as the window that the eleventh
prime opens.  The proofs work by dividing the primes into the ones smaller than the stretch,
which can strike three or more of its positions, and the ones at least as large, which can
strike at most two -- and when such a prime does strike two, the distance between them is
forced, so the leftover positions can be finished off only at particular distances, and usually
they are at the wrong ones.

## Vocabulary

Column `k` is the pair `(6k-1, 6k+1)`.  A **gear** is a prime `g >= 5`; it **strikes** column
`k` iff `g` divides `6k-1` or `6k+1`, i.e. iff `k = +-u_g (mod g)` with `u_g = 6^{-1} (mod g)`
(file 02).  The two struck residues are the **teeth**; their **separation** is
`d_g = 2 u_g = 3^{-1} (mod g)`; the **short arc** is `a_g = min(d_g, g - d_g)` and the **long
arc** is `g - a_g`.

A **run of `L`** is `L` consecutive columns.  A set `S` of gears **covers** a run of `L` if
there are phases -- one integer per gear, the position of one of its teeth -- making every
column of the run struck by some member of `S`.  Since a gear's strike set is periodic of
period `g` with two elements per period, "any phase" is the same as "slide the run anywhere":
gear `g` at phase `c` strikes the columns `k = c` and `k = c + d_g (mod g)`.

`F(S)` is the least `L` that `S` cannot cover, and

    A(K) = max{ F(S) : S a set of K distinct primes >= 5 } ,

the least run length that **no** `K` primes cover.  `W(K) = (p_{K+1}^2 - 1)/6`, with `p_j` the
`j`-th prime above 3, is the window the conjecture's route (file 01) attaches to a machine of
`K` gears: `W = 8, 20, 28, 48, 60, 88, 140, 160, 228, 280` at `K = 1..10`.

Classical translation: `A(K)` is a Jacobsthal-type covering number for two classes per prime at
a *fixed* separation, maximised over the choice of primes.  Ziller and Morack's `h_2` is the
same shape with the two classes *arbitrary* and the primes an initial segment; see Prior art.

**Convention warning.**  `A(K)` is the least run length that cannot be covered, not the longest
that can.  `A(4) = 16` means "some four primes cover 15 consecutive columns and no four primes
cover 16".

## Statement

**Theorem A (the adversarial lemma for `K <= 10`).**  Let `1 <= K <= 10`.  No `K` primes above
3, each striking two residue classes at its own separation `3^{-1} (mod g)` with any phase,
cover `W(K) = (p_{K+1}^2 - 1)/6` consecutive columns.

**Theorem B (the exact ladder to six gears).**

    K       1    2    3    4    5    6
    A(K)    2    5    7   16   22   28

that is: the longest run `K` primes can cover is `1, 4, 6, 15, 21, 27`, and covers of exactly
those lengths exist.

Four lemmas do the work, and they are of independent use.

**Lemma 1 (arc law).**  For every gear `g`, `3 a_g = g -+ 1`; hence `a_g` is **even**, the long
arc `g - a_g = 2 a_g -+ 1` is **odd**, and `g = 3 a_g +- 1`.

**Lemma 2 (capacity).**  In a run of `L`, gear `g` strikes at most

    maxstrike(g, L) = 2 floor(L/g) + e,   e = 2 if (L mod g) > a_g, 1 if 1 <= (L mod g) <= a_g,
                                          0 if L mod g = 0,

columns, and the bound is attained.  In particular every gear with `g >= L` strikes at most 2.

**Lemma 3 (span lemma).**  Let `g >= L` and suppose a run of `L` contains two columns struck by
`g`, at distance `t` (`0 < t <= L-1`).  Then `t = a_g` or `t = g - a_g`, and therefore

* `t` **even** forces `t = a_g` and `g in {3t - 1, 3t + 1}`;
* `t` **odd** forces `t = g - a_g` and `g in {(3t-1)/2, (3t+1)/2}`.

So at most two primes in all can strike two columns of the run at any given distance, and which
two is decided by the distance alone.

**Lemma 4 (type lemma).**  If `g - a_g >= L` then the set of columns of the run struck by `g`
is one of

    {} ,   {i, i + a_g} with i + a_g <= L-1 ,   {i} with i < a_g or i >= L - a_g ,

a list depending on `a_g` alone.  Consequently at level `L` the infinite pool of primes has
only finitely many types: the primes with `g - a_g <= L-1` (equivalently `g <= (3L-2)/2`), one
by one; for each even `a < L` a *domino of arc `a`* with multiplicity the number of primes among
`{3a-1, 3a+1}` that are big at `L`; and one *single-column* type of unbounded multiplicity.

## Proof

### 1. The lemmas

**Lemma 1.**  `3 d_g = 1 (mod g)` and `1 <= d_g <= g-1` give `3 <= 3 d_g <= 3g - 3`, so
`3 d_g in {g+1, 2g+1}`.

* If `3 d_g = g + 1` then `d_g = (g+1)/3` and `g - d_g = (2g-1)/3 > d_g` for `g > 2`, so
  `a_g = d_g` and `3 a_g = g + 1`, `g = 3 a_g - 1`, `g - a_g = 2 a_g - 1`.
* If `3 d_g = 2g + 1` then `d_g = (2g+1)/3 > g/2`, so `a_g = g - d_g = (g-1)/3` and
  `3 a_g = g - 1`, `g = 3 a_g + 1`, `g - a_g = 2 a_g + 1`.

In both cases `3 a_g = g -+ 1` is even because `g` is odd, and 3 is odd, so `a_g` is even; and
`g - a_g = 2 a_g -+ 1` is odd.  QED

**Lemma 2.**  By file 02(b) the struck columns of `g` are periodic of period `g`, two per
period, at distances `a_g` and `g - a_g` alternately.  Write `L = q g + r`, `0 <= r < g`.  Each
of the two teeth-classes contributes `q` or `q+1` columns; it contributes `q+1` exactly when its
residue falls in the `r` residues that occur `q+1` times, which form `r` consecutive residues.
Both teeth do so iff the two teeth lie in `r` consecutive residues, iff `r >= a_g + 1` (the two
teeth are at cyclic distances `a_g` and `g - a_g >= a_g`).  One of them does so iff `r >= 1`.
Hence the maximum is `2q + e` with `e` as stated, and every case is realised by choosing the
phase.  If `g >= L` then `q = 0` and `e <= 2`.  QED

**Lemma 3.**  `g >= L` means each residue class mod `g` meets the run at most once, so the two
struck columns are in different classes and their difference satisfies `t = +-d_g (mod g)`,
i.e. `t = +-a_g (mod g)`.  With `0 < t < g` this leaves `t = a_g` or `t = g - a_g`.  By
Lemma 1, `a_g` is even and `g - a_g` is odd, so the parity of `t` decides which.  If `t = a_g`
then `g = 3t +- 1` by Lemma 1.  If `t = g - a_g` then, substituting `a_g = (g -+ 1)/3`,
`t = (2g +- 1)/3`, i.e. `g = (3t -+ 1)/2`.  QED

**Lemma 4.**  `g - a_g >= L` gives `g = a_g + (g - a_g) >= 2 + L > L`, so Lemma 3 applies and
at most two columns are struck.  If two, at distance `t <= L-1`, then `t in {a_g, g - a_g}` and
`g - a_g >= L > L-1 >= t`, so `t = a_g`.  If one, at column `i`, the partner tooth sits at
`i + a_g` or `i - a_g` in the integers (the other representatives, at `i +- (g - a_g)`, are at
least `L` away, hence outside the run), and it must be outside the run: `i + a_g > L-1` or
`i - a_g < 0`.  For the finiteness statement, `g - a_g <= L-1` reads `(2g +- 1)/3 <= L-1`, i.e.
`g <= (3L-2)/2`; and two primes share a short arc `a` iff they are `3a-1` and `3a+1`
(Lemma 1), so a domino type has multiplicity at most 2; and primes with `a_g >= L` are
infinitely many (they are the primes `> 3L`), each able to strike a single column anywhere.
QED

Two remarks on how Lemma 4 is used.  First, the list is used only in the direction
"every realisable strike set is in the list"; the converse (every listed set is realised) is
true for the pairs and the singletons but is not needed, and `{}` is granted to the adversary
for free.  Over-granting is the safe direction when the conclusion is "no cover exists".
Second, **coverability is monotone**: if `K` gears cover a run of `L` they cover any shorter
run inside it, so the coverable lengths form an initial segment and `A(K)` is exactly the
threshold.

### 2. The split

Fix `L` and put `T(L) = {primes 5 <= g < L}`.  Let `K` gears cover a run of `L`, let
`S` be those of them that lie in `T(L)` and `m = |S|`.  By Lemma 2 the other `K - m` gears
strike at most two columns each, so

> **(C)  the counting filter.**  `sum_{g in S} maxstrike(g, L) + 2 (K - m) >= L`.

Fix in addition the phases of `S` and let `H` be the set of columns of the run that `S` leaves
unstruck.  Each auxiliary gear covers at most two holes, and by Lemma 3 it covers two only if
their distance `t` lies in `D(g) = {a_g, g - a_g} n [1, L-1]`, with the auxiliary primes
distinct.  If `P` of them cover two holes each, the number of auxiliary gears is at least
`(|H| - 2P) + P = |H| - P`.  Hence

> **(M)  the matching filter.**  `|H| - M(H) <= K - m`, where `M(H)` is the largest number of
> pairwise disjoint pairs of holes that can be assigned injectively to distinct primes
> `g >= L` with the pair's distance in `D(g)`.

`D(g)` is short and explicit: by Lemma 3, the primes able to span a distance `t` are among
`{3t-1, 3t+1}` for even `t` and `{(3t-1)/2, (3t+1)/2}` for odd `t`, restricted to those that
are prime and `>= L`.

### 3. Theorem B

**Lower bounds (the covers).**  Each is an explicit set of primes with an explicit phase, and
each is checked by direct arithmetic (`sk_theoremA.py`, engine (1); the phases are the columns
of one tooth, the other tooth being `d_g = 3^{-1} (mod g)` further on):

    K   L        gears at their phases                        covers 0..L-1
    1   1        5@0
    2   4        5@0, 7@3
    3   6        5@0, 7@3, 11@0
    4  15        7@0, 5@1, 11@9, 17@4
    5  21        5@0, 11@8, 29@3, 7@4, 23@6
    6  27        5@3, 23@1, 11@2, 37@16, 7@0, 17@5

Worked, for `K = 4`: `d_5 = 2`, so `5@1` strikes `k = 1, 3 (mod 5)`, i.e. `{1,3,6,8,11,13}`;
`d_7 = 5`, so `7@0` strikes `k = 0, 5 (mod 7)`, i.e. `{0,5,7,12,14}`; `d_11 = 4`, so `11@9`
strikes `k = 9, 2 (mod 11)`, i.e. `{2,9,13}`; `d_17 = 6`, so `17@4` strikes `k = 4, 10`, i.e.
`{4,10}`.  The union is `{0,...,14}`.  So `A(4) >= 16`, and likewise for the other rows.

**`A(1) = 2`.**  A gear never strikes two adjacent columns (file 02(d)), so no gear covers a run
of 2; gear 5 covers a run of 1.

**`A(2) = 5`.**  In a run of 5, gear 5 strikes exactly 2 (one full period) and every gear
`g >= 7 > 5` strikes at most 2 by Lemma 2.  So two gears strike at most 4 < 5.  The row above
covers 4.

**`A(3) = 7`.**  `T(7) = {5}`, `maxstrike(5,7) = 3`, and every gear `>= 7` strikes at most 2.

* If the 3-set misses 5, it strikes at most `2+2+2 = 6 < 7`.  (Filter (C).)
* If it contains 5, (C) reads `3 + 4 >= 7`: an equality.  So gear 5 must strike 3 columns, the
  two auxiliary gears must strike 2 each, and all 7 strikes must be distinct.  The four holes
  must therefore split into **two disjoint pairs**, one per auxiliary gear.

  Gear 5's five phases give the strike sets `{0,2,5}`, `{1,3,6}`, `{2,4}`, `{0,3,5}`,
  `{1,4,6}`; only the four with three strikes qualify, leaving the hole sets

        [1,3,4,6] ,  [0,2,4,5] ,  [1,2,4,6] ,  [0,2,3,5] .

  By Lemma 3 the distances a prime `>= 7` can span inside a run of 7 are: `t = 2` (only
  `g = 7`, since `3*2 -+ 1 = 5, 7` and 5 is not an auxiliary gear -- auxiliary gears are
  `>= L = 7` -- and is in `S` in any case), `t = 4` (`g = 11` or `13`), `t = 5` (only `g = 7`,
  since `(3*5 -+ 1)/2 = 7, 8`), `t = 6` (`g = 17` or `19`).  Distances 1 and 3 are spanned by no
  prime `>= 7` at all (`t = 1` gives `g in {1,2}`; `t = 3` gives `g in {4,5}`).

  The three pairings of each hole set ask for the distance multisets

        [1,3,4,6]:  {2,2}   {3,3}   {5,1}
        [0,2,4,5]:  {2,1}   {4,3}   {5,2}
        [1,2,4,6]:  {1,2}   {3,4}   {5,2}
        [0,2,3,5]:  {2,2}   {3,3}   {5,1}

  Every one of the twelve contains a distance no prime `>= 7` spans (1 or 3), or asks twice for
  a distance that only the single prime 7 supplies (`{2,2}`, `{5,2}`).  So no 3-set covers 7.

**`A(4) = 16`.**  `T(16) = {5,7,11,13}` with `maxstrike = 7, 5, 4, 3`; every gear `>= 16`
strikes at most 2.  Filter (C) leaves exactly five subsets `S` -- every other choice has
capacity below 16:

    S                  aux gears   capacity     max columns S can strike    holes at best
    {5,7}                  2          16                 11                      5
    {5,7,11}               1          18                 13                      3
    {5,7,13}               1          17                 13                      3
    {5,11,13}              1          16                 11                      5
    {5,7,11,13}            0          19                 14                      2

The fourth column is the exact maximum, over all phase vectors of `S`, of the number of columns
of a 16-run that `S` strikes together (35, 385, 455, 715 and 5005 phase vectors, 6595 in all;
`sk_head.py`, `sk_cases.py`).  In every row the holes left over -- 5, 3, 3, 5, 2 -- exceed the
`2 (K - m)` columns the auxiliary gears could ever take -- 4, 2, 2, 2, 0.  So filter (C) plus
one number per row closes all five cases, and no 4-set covers 16.

The first row is worth naming, because the same thing recurs at `K = 5` and `K = 6`:

> **The head collision.**  `maxstrike(5, L) + maxstrike(7, L)` is 12, 16, 20 at `L = 16, 22,
> 28`, but the largest number of columns gears 5 and 7 can strike *together* is 11, 15, 18.
> The two smallest gears cannot both be maximal and disjoint.  (35 phase pairs at each `L`.)

**`A(5) = 22`.**  `T(22) = {5,7,11,13,17,19}` with `maxstrike = 9,7,4,4,3,3`.  Filter (C)
leaves 18 subsets.  Fourteen of them are closed by the hole count alone, exactly as at
`K = 4`; the head-collision row `{5,7}` is one of them (holes `>= 7` against `2*3 = 6`).  Four
need filter (M), and they are the mechanism of the ladder:

    S                  aux   the best holes it can leave        distance   who could span it
    {5,7,11}            2     [4,10,15,17] (4 holes)             --        M = 0, needs 4 aux
    {5,7,11,13}         1     [10,15]                             5        (3*5-+1)/2 = 7, 8
    {5,7,11,17}         1     [15,17]                             2        3*2-+1 = 5, 7
    {5,7,11,19}         1     [10,15]                             5        (3*5-+1)/2 = 7, 8

In each of the last three rows the two surviving holes sit at a distance that only gear 5 or
gear 7 could span -- and both are already spent inside `S`.  No prime `>= 22` has short arc 2
or long arc 5.  So each needs two auxiliary gears where one is available, and no 5-set covers
22.

**`A(6) = 28`.**  `T(28) = {5,...,23}` with `maxstrike = 12,8,6,5,4,4,3`.  Filter (C) leaves 53
subsets; 51 are closed by the hole count alone (including the head-collision row `{5,7}`,
holes `>= 10` against `2*4 = 8`), and two need filter (M):

    S                    aux   best holes        why it fails
    {5,7,11,17}           2     [3,6,11,13]      of the six distances 3, 5, 7, 8, 10, 2 only
                                                 10 is spannable by a prime >= 28 (29 or 31),
                                                 so at most one pair forms and 3 aux gears
                                                 are needed
    {5,7,11,17,23}        1     [6,13]           distance 7, spannable only by
                                                 (3*7-+1)/2 = 10, 11; 11 < 28 and is spent

So no 6-set covers 28.  With the covers above, `A(K) = 2, 5, 7, 16, 22, 28`.  QED

### 4. Theorem A

By Lemma 4, at level `L` a cover by `K` primes induces a selection of at most `K` items from the
finite type list, respecting multiplicities, whose strike sets cover the run.  Write that as an
exact 0/1 feasibility program:

    x[i,o] in {0,1}                      "one gear of type i is used at option o"
    sum_o x[i,o]  <=  mult_i             a type is offered by mult_i primes
    sum_{i,o} x[i,o]  <=  K              K gears
    sum_{(i,o) : c in mask(i,o)} x[i,o]  >=  1     for every column c of the run

If this program is infeasible, no `K` primes cover `L`.  It is infeasible at `L = W(K)` for
every `K = 1..10`:

    K       1     2     3     4     5     6     7     8     9    10
    W(K)    8    20    28    48    60    88   140   160   228   280
    binaries
           53   274   513  1358  1916  3696  9272 12162 23342 34099
    result  every one INFEASIBLE (HiGHS, total under 30 seconds)

which is Theorem A.  QED

The same program is infeasible at `L = A(K)` for `K = 1..10` -- `A(K) = 2, 5, 7, 16, 22, 28,
37, 45, 68, 88` -- and feasible at `L = A(K) - 1` with an explicit cover; with monotonicity that
is the stronger statement `A(K) < W(K)`, the ratio running `0.25, 0.25, 0.25, 0.33, 0.37, 0.32,
0.26, 0.28, 0.30, 0.31`.

## The certificates, and how to reproduce them

From the repository root, `uv run python research/anchor235/r53/<script>`:

| script | what it certifies |
|---|---|
| `sk_core.py` | the two engines: a direct phase search over an explicit prime set, and the type-reduced search over all primes |
| `sk_gate.py` | gates: `F({5..q}) = 5,7,11,18,25,34,43,58` at `q = 7..31` reproduced; `A(1..6)` reproduced by the type-reduced search; every type-reduced cover realised by explicit primes |
| `sk_theoremA.py 10 --window` | Theorem A: the 0/1 program infeasible at `L = W(K)`, `K = 1..10`; also infeasible at `L = A(K)` and the explicit covers at `L = A(K)-1` |
| `sk_cases.py` | Theorem B: the split by `T(L)`, the counting filter, the span table, and the case-by-case verdicts for `K = 2..6` |
| `sk_head.py` | the head-collision numbers and the `A(3) = 7` tables |
| `sk_theoremB.py` | an independent route at `K <= 5`: for every `K`-subset of the small pool of Lemma 4, an exhaustive search restricted to it |
| `sk_distortion.py` | the audit of the withdrawn distortion route (see Status) |

Outputs land in `research/anchor235/r53/results/` (untracked).

## Status

Kernel: **none**.  Nothing in this file is in Lean.

Proved by reasoning, with no computation anywhere in the argument: Lemmas 1-4; monotonicity;
the split and both filters; `A(1) = 2`; `A(2) = 5`.

Proved by reasoning plus a table small enough to check by hand: `A(3) = 7` -- five phases of
gear 5, four hole sets, twelve pairings, and the span table of Lemma 3.

Proved by reasoning down to a stated finite list of cases, each case then settled by exhaustive
enumeration in a script: `A(4) = 16` (5 cases, 6,595 phase vectors, 1,005 search nodes), `A(5) = 22` (18 cases,
756,870 phase vectors, 15,475 nodes, of which four cases need the matching filter),
`A(6) = 28` (53 cases, 29,418,815 phase vectors nominal, 266,643 nodes once the counting prune
is applied, of which two cases need the matching filter).  The completeness of each case list is Lemma 2 plus filter
(C) and is proved, not checked.

Proved by a finite certificate that is a computation rather than a hand-checkable object:
**Theorem A**.  Its ten infeasibility results come from HiGHS through `scipy.optimize.milp` on
the program of section 4.  Corroboration, in the absence of a checkable infeasibility
certificate: (i) at `K <= 6` the same conclusions follow from the independent exhaustive search
of `sk_cases.py`, which uses no solver; (ii) at `K <= 5` a third, differently organised
exhaustive search (`sk_theoremB.py`) agrees; (iii) the values `A(K)` agree with the recorded
ladder of `research/proof/arc_multiset.md` R1, computed in round 50 by a different
implementation, which was itself cross-checked at `K = 7, L = 37` by brute force over all
77,520 seven-subsets of the primes `5..79`; (iv) the same code reproduces the certified record
ladder `F({5..q}) = 2,5,7,11,18,25,34,43,58`.

**Withdrawn, and why.**  `research/proof/distortion_method.md` R7 and `research/proof/the_wall.md`
5f record "the localised budget proves the adversarial lemma for `K <= 10`".  That is not a
proof, and this round refutes the inequality it rests on rather than merely finding it
unproved.  The claim applies to an interval the hypothesis `eta = sum_i E[alpha_i^2] < 1` of
BBMST's Theorem 3.1, whose conclusion is about covering `Z` and whose proof lives on `Z_Q` with
the product structure `Z_{Q_i} = Z_{Q_{i-1}} x Z_{p_i}`.  Evaluated honestly on an interval with
the uniform measure, the hypothesis holds for gear sets that demonstrably cover that interval:
`{5,7,11,17}` covers 15 columns with `eta = 0.693`; `{5,7,11,23,29}` covers 21 with
`eta = 0.698`; `{5,7,11,13,17,23,31,37,47}` covers 67 with `eta = 0.923` (`sk_distortion.py`
part D, eight instances at `K = 2..9`).  So "`eta < 1` implies no cover of the interval" is
false.  Separately, the quantity `eta_max` the lane actually tabulates is not an upper bound on
the localised second moment either -- exact evaluation beats it per gear at `g = 5, L = 88`
(`0.1674 > 0.1600`), at `g = 7, L = 610` (`0.0824 > 0.0816`) and at `g = 11, L = 610`
(`0.0357 > 0.0331`) -- because the step `alpha <= 2/m` is false once a fibre holds more than `g`
columns, and the code repairs that regime by substituting `4/g^2`, which is the Cauchy-Schwarz
*lower* bound.  What `eta_max` is, exactly, is the union bound `sum_g 2 ceil(L/g) / L` on the
collapsed gears plus `4/g^2` on the head; the union-bound part is a genuine theorem and is
vacuous (above 1) at every `K >= 4`, and the entire margin that puts `eta_max` below 1 for
`K <= 10` is the replacement of gears 5 and 7's capacity `0.686` by `0.242`
(`sk_distortion.py` part E).  Theorem A above replaces the claim.

## Prior art, and what is new

**Prior-art status.**  No fresh literature search was run in this round.  The statements below
are taken from the repository's own recorded checks -- `research/proof/literature_increment.md`
(sections 2a, 2b, 2c, 3d; sources first-hand, dated 2026-08-24 to 2026-08-29),
`research/proof/distortion_method.md` section 7 (2026-09-06), `research/proof/iwaniec_two_class.md`,
and `docs/novel/README.md` -- and are marked **prior art not checked** where those records do not
settle the point.

**Leverages.**  The tooth rule and the alternating spacings `2u_g`, `g - 2u_g` (file 02, itself
a one-line CRT computation); the domino/type dichotomy of `research/proof/gear_count.md` R4 and
its exact form in `research/proof/arc_multiset.md` section 0.1, which is where the type lemma
and the finite item list come from; the recorded values `A(K)` for `K = 1..12` from
`arc_multiset.md` R1, here re-derived independently for `K <= 10`.

**The neighbouring printed objects.**

* **Ziller & Morack**, arXiv:1706.00317 (paired Jacobsthal `j_2`, `h_2(n) = j_2(p_n#)`,
  Conjectures 4-6, Theorem 4.1) and arXiv:1706.03668 (values for `n <= 21`); OEIS **A072753**
  (column units) and **A288815** (`= 6 A072753 + 6`).  This is the two-class covering number
  with the two classes **arbitrary** and the primes an **initial segment**.  In column units
  its values at the gear sets `{5}, {5,7}, ..., {5..37}` are `2, 4, 10, 24, 31, 42, 60, 74, 94,
  117`, against the longest run this file's fixed-separation adversary can cover with the same
  number of primes, free to choose them: `1, 4, 6, 15, 21, 27, 36, 44, 67, 87`.  Neither
  function dominates the other by hypothesis -- theirs frees the classes, ours frees the primes
  -- and the arithmetic says the fixed separation costs more than the free choice of primes
  buys at every `K` except `K = 2`, where both are 4.  Their Conjecture 6, `h_2(n) < p_n^2 - p_n`, is the same shape as
  the open lemma `A(K) < W(K)` but about their object.
* **Stevens**, Math. Ann. 226 (1977) 95-97, via Hajdu-Saradha (1.1): `H(r) <= 2 r^{2 + 2e log r}`
  for the longest interval `r` arbitrary primes can block with **one** class each -- the only
  printed upper bound of this shape.  At `r = 4` it gives `1.10e6` against the value 16 proved
  here; at `r = 12`, `1.09e17` against 115.
* **Covering systems**: Balister-Bollobas-Morris-Sahasrabudhe-Tiba, Invent. math. 228 (2022)
  377-414 (Theorem 3.1, the engine with no lower bound on the moduli); Hough, Ann. of Math. 181
  (2015) 361-382; Filaseta-Ford-Konyagin-Pomerance-Yu, JAMS 20 (2007) 495-517 (Theorem 2, the
  one statement in that corpus with several classes per modulus).  Every conclusion in the
  corpus is about the **density** of the uncovered subset of `Z`; none has an interval in its
  conclusion (`distortion_method.md` 2.4).
* **Ford-Konyagin-Maynard-Pomerance-Tao**, *Long gaps in sieved sets*, Remark 7, is the one
  place in print where the machine's own system `I_p = {0,2} (mod p)` -- two classes at a fixed
  separation -- is named, and it is named for a lower bound.
* **Erdos problem #689** (two of the congruences satisfied by every integer of `[1,n]`) is
  adjacent in shape and open.

**New, as far as the record goes.**  Fixed-separation two-class covering numbers -- `A(K)`, the
longest run `K` primes with two classes at separation `3^{-1} (mod g)` can cover, maximised
over the primes -- do not appear in print: the entire published two-class corpus is the two 2017
Ziller-Morack preprints, whose object is the maximum over class assignments on an initial
segment, and the covering-systems corpus has no interval statement at all.  Within that, the
pieces this file adds are: the **span lemma** (Lemma 3) in the closed form "a distance `t`
inside a run shorter than the gear is spanned only by `3t -+ 1` (`t` even) or `(3t -+ 1)/2`
(`t` odd)", which turns the hole-distance dictionary of `arc_multiset.md` R7 from a measured
table into a two-line rule; the **head collision** (gears 5 and 7 cannot both be maximal and
disjoint, deficit 1, 1, 2 at `L = 16, 22, 28`); the hand proof of `A(3) = 7` (the record's
one-paragraph version in `arc_multiset.md` R7 assumes the 3-set contains both 5 and 7 and uses
"5 and 7 are already spent" for the distance-2 pair -- the case where 7 is an auxiliary gear,
and the cases where the pairing asks for distances 1, 3 or 5, are supplied here); and the
proofs of `A(4), A(5), A(6)` reduced to 5, 18 and 53 stated cases.  **Prior art not checked**
for the span lemma and the head collision as isolated statements.

**Not new.**  The numbers `A(K)` for `K <= 12` are on the project record
(`arc_multiset.md` R1, round 50); this file proves them rather than re-measuring them, and
extends nothing.  The type lemma is round 50's, restated with its proof.

## Relationship to the conjecture

These are the **finite base** of the open lemma `A(K) < (p_{K+1}^2 - 1)/6` for all `K`, and
nothing more.  The lemma is strictly stronger than the root: the root asks only that the
initial segment `{5..p_K}` fail to cover its own window, and `A(K) >= F({5..p_K})` at every `K`
(with equality at `K <= 3` and at `K = 10`, `arc_multiset.md` R1).  So Theorem A gives ten
instances of a statement that would imply ten rungs of the route of file 01, and Theorem B gives
the exact value of the adversarial function at the first six.

What they do **not** give is any induction step.  There is no argument here that passes from
`K` to `K+1`; each `K` is a separate finite computation whose cost grows with the item list, and
the recorded ratio `A(K)/W(K)` is flat at `0.26-0.37` over `K = 4..12` with no proof that it
stays bounded.  The residual named on the record is unchanged: a lower bound on the tiler
function `h_S(L)` -- "`k` prime gears cannot leave fewer than `2(K-k)+1` holes in a run of
`W(K)` columns" -- which is a capacity statement with a gear count attached, and capacity is
loose by a factor of two (face A).  Nothing measured enters the statements; the certificates are
exhaustive, not sampled.

One thing this file does settle on the record: the wall's section 5f lists "the localised budget
proves the adversarial lemma for `K <= 10`" as the one positive of the covering round.  That
positive is withdrawn (see Status), and Theorem A is what stands in its place -- proved by a
finite certificate rather than by a general method.  No general method is known to reach the
lemma at any `K`.

## Where it is used

* As the base of any induction on `K` for the adversarial lemma; the lemma is the covering form
  of the route of file 01 and the one formulation face A does not forbid
  (`the_wall.md` 5d, 5e).
* The span lemma (Lemma 3) is the sharp form of the hole-distance mechanism used throughout
  `arc_multiset.md` R7 and `gear_count.md` R4, and it is what makes those dictionaries
  predictive rather than tabulated.
* The counting filter (C) with `maxstrike` is the per-gear capacity used in file 18's case
  split and in the `h_S(L)` residual.

## Source

`research/proof/small_K_theorem.md` (this round's working: pre-registration, what was verified,
what failed); `research/proof/arc_multiset.md` (the ladder `A(K)`, the type lemma, the matching
identity, the hole-distance dictionary); `research/proof/gear_count.md` (the domino dichotomy);
`research/proof/dead_branches_reopened_2.md` (reading-as-a-whole item 1, which set this round);
`research/proof/distortion_method.md` (the withdrawn route);
`research/proof/literature_increment.md` (the two-class prior art); scripts and outputs in
`research/anchor235/r53/`.
