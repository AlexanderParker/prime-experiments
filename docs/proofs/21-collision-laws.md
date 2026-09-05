# 21. Collision laws for gear pairs

## In plain words

Two primes striking out pairs of numbers get in each other's way, and how much they get in each
other's way is not an accident of how they are slid about: it is fixed by the two primes
themselves.  Take the most positions the first prime could ever strike in a stretch of a given
length, add the most the second could ever strike, and compare that with the most the two can
strike between them.  The second number is always the smaller, and the shortfall -- call it what
the two gears lose to each other -- obeys three exact rules.  It grows by exactly four every
time the stretch is lengthened by the product of the two primes, so it is a straight line whose
slope is known in advance.  It is exactly nothing while the stretch is shorter than the wider of
the two gaps the primes leave between their own two strikes.  And the moment two primes leave
the *same* gap, they lose to each other at once, because a stretch just long enough to hold that
gap has only one place to put it, so both primes are forced onto the same two positions.  Twin
primes always leave the same gap, so twin primes always collide at the first opportunity;
5 and 7 are the smallest instance, and their collision -- which the previous file found by hand
and used at every size -- turns out to be one case of a general law.

The second half of the file turns the losses into a bound on what an adversary can cover.  The
naive form, subtract every pairwise loss from the total capacity, is not a bound at all and is
refuted outright by two covers already on record.  The correct form splits the primes into
blocks of a chosen size and subtracts one loss per block.  Blocks of two are already enough to
prove, for four primes, the statement the previous file could only reach by enumerating phases;
blocks of four do it for five and for six.  And the same identity that gives the bound says how
far it can go: the block size needed grows with the number of primes, without bound, so no rule
about pairs, or triples, or any fixed number of primes at a time, will ever reach every case.

## Vocabulary

Column `k` is the pair `(6k-1, 6k+1)`.  A **gear** is a prime `g >= 5`; with **separation**
`s_g` and **phase** `c_g` it **strikes** the columns `k = c_g` and `k = c_g + s_g (mod g)`.  The
real machine has `s_g = 3^{-1} (mod g)`, equivalently `k = +-6^{-1} (mod g)` (file 02(a)); the
**short arc** is `a_g = min(s_g, g - s_g)` and the **long arc** is `g - a_g`.  For the real
separation `3 a_g = g -+ 1` (file 20, Lemma 1), so `a_g` is even and `g > 2 a_g`.

A **run of `L`** is `L` consecutive columns.  A set of gears **covers** a run if some choice of
phases leaves no column of the run unstruck.

* `max_g(L)` is the most columns of a run of `L` that gear `g` can strike, over its phases; by
  file 20, Lemma 2 (capacity),

      max_g(L) = 2 floor(L/g) + e,   e = 2 if (L mod g) > a_g,
                                     1 if 1 <= (L mod g) <= a_g,
                                     0 if (L mod g) = 0,

  and the proof there uses only "two classes at cyclic distances `a_g` and `g - a_g`", so it
  holds for any separation, not only the real one.
* `joint_max(B; L)` is the most columns of a run of `L` that a set `B` of gears can strike
  between them (the size of the union), over all phase vectors.
* the **block deficit** of a set `B` is

      c_B(L) = sum_{g in B} max_g(L) - joint_max(B; L) >= 0 ,

  and for a pair we write `c(g, h; L) = c_{{g,h}}(L)`, the **collision deficit**.  `c_B = 0` for
  `|B| = 1`.
* the **onset** `L0(g, h)` is the least `L >= 2` with `c(g, h; L) > 0`.  (`L = 1` is excluded
  throughout: `c(g, h; 1) = 1` for every pair, both gears wanting the one column, which is a
  triviality and not a collision.)
* a **twin pair** of gears is `(g, g+2)` with both prime.

**The one-orbit reduction.**  For a fixed set of gears the moduli are pairwise coprime, so by CRT
every phase vector is a diagonal translate: `(c_g, c_h, ...) = (t, t, ...)` for a unique
`t (mod prod g)`.  Quantifying over all phases is therefore the same as sliding a window of
length `L` over one period of the one fixed pattern
`U_B = union_{g in B} ({0, s_g} + g Z)`.  This is on the record (`the_wall.md` 5a, "the adversary
with one phase per gear over all primes up to `q` is exactly the real machine over its period");
it is cited, not re-derived, and it is what makes `joint_max(B; L)` an exact maximum over a
period rather than a search.

Classical translation: `c(g, h; L)` measures how far two arithmetic progressions-of-two-classes
fall short of independence on an interval; the four residues mod `gh` that both gears strike are
the four-point CRT shape of `separation_drives_K.md` 3.1, and they are the entire source of the
deficit.

## Statement

Throughout, `W(K) = (p_{K+1}^2 - 1)/6` is the window a machine of `K` gears must be shown not to
cover (file 01, file 20): `W = 28, 48, 60, 88, 140, 160, 228, 280` at `K = 3..10`.

**Theorem 1 (the linear deficit law).**  For every pair of gears `g < h`, with **any**
separations, and every `L >= 1`,

    c(g, h; L + gh) = c(g, h; L) + 4 .

Hence `c` is linear with slope exactly `4/(gh)`; `c(g, h; L) >= 4 floor(L/gh)`; every zero of
`c` lies in `[1, gh]`; and the **permanent onset** `L1 = 1 + (last zero in [1, gh])`, beyond
which `c > 0` for good, is an exactly computable number rather than a limit.  The general form
for a block `B` with period `P = prod_{g in B} g` is

    c_B(L + P) = c_B(L) + sum_{g in B} 2P/g - (P - prod_{g in B} (g - 2)) ,

which is `+4` for a pair and `4(g+h+k) - 8` for a triple.

**Theorem 2 (the shared-arc law).**  If two gears have the same short arc, `a_g = a_h = a`, then

    c(g, h; a + 1) >= 1 .

**Corollary 2.1 (twin gears).**  For the real separation, twin gears share the short arc
`a = (g+1)/3` (file 02(e), file 20 Lemma 1), so every twin pair `(g, g+2)` collides at

    L = a + 1 = (g + 4)/3 ,

which is the earliest length the arc floor of Theorem 3 permits any pair with those arcs to
collide, and is below `g`.  `(5, 7)` at `a = 2` is the first instance.

**Theorem 3 (the arc floor) -- CERTIFICATE, not proved.**  For the **real** separation,

    c(g, h; L) = 0   for every   2 <= L <= max(a_g, a_h) ,

so `L0(g, h) >= max(a_g, a_h) + 1`.  This is a property of the real teeth: with random
separations it fails.  No proof is on record; the evidence is stated in Status.

**Theorem 4 (the head collision, and gears 5 and 7 in permanent collision).**  `(5, 7)` is the
`a = 2` instance of Theorem 2, and `c(5, 7; 16), c(5, 7; 22), c(5, 7; 28) = 1, 1, 2`, the
numbers file 20 obtained from 35 phase pairs.  Moreover

    c(5, 7; L) > 0   for every   L >= 21 ,

with `L = 20` the last zero.  So gears 5 and 7 are in permanent collision on every window
`W(K)` from `K = 4` on.

**Theorem 5 (the block bound).**  Let a set `S` of gears cover a run of `L` and let
`S = B_1 u ... u B_r` be **any** partition of `S` into blocks.  Then

    L <= sum_{i} joint_max(B_i; L) = sum_{g in S} max_g(L) - sum_{i} c_{B_i}(L) ,

hence `L` is at most the minimum of the right-hand side over all partitions.  Blocks of size 1
are file 20's counting filter (C); blocks of size 2 are a maximum-weight matching of the gears;
blocks of size `|S|` are the exact question.

**Theorem 5b (the all-pairs form is false).**  The form

    L <= sum_{g in S} max_g(L) - sum_{g < h in S} c(g, h; L)

is **not** a valid bound.  Two recorded covers refute it: the nine-gear cover of 67 columns
(`sum max = 98`, all-pairs sum `33`, "bound" `65 < 67`) and the ten-gear cover of 87 columns
(`sum max = 135`, all-pairs sum `57`, "bound" `78 < 87`).

**Corollary 5.1 (the lemma at `K = 4, 5, 6`) -- proved by reasoning plus stated computations.**
Write `bound_b(K, L)` for the maximum, over every `K`-set of gears (those `>= L` granted 2
strikes each with no deficit, the safe over-granting of file 20), of the best block-`b` value of
Theorem 5.  If `bound_b(K, W(K)) < W(K)` then no `K` gears cover `W(K)`.  Computed exhaustively:

    K            3     4     5     6
    W(K)        28    48    60    88
    b = 1       26    51    72   113
    b = 2       24    46    65   102
    b = 3       21    42    61    95
    b = 4        -    38    55    87
    proves      K=3   K=4   K=5   K=6
                b=1   b=2   b=4   b=4

so the **matching** (block-2) bound proves the adversarial lemma at `K = 4`, where the counting
filter alone does not (`51 >= 48` against `46 < 48`), and block size 4 proves it at `K = 5`
(`55 < 60`) and `K = 6` (`87 < 88`).  At `K = 7..10` block size 4 is not enough: `147, 178, 264,
335` against `140, 160, 228, 280`.

**Recorded fact 6 (the order obstruction).**  This is a fact on the record, not a theorem of this
file.  For a gear set `S` put

    rho_b(S) = sum_{g in S} 2/g - max over partitions into blocks of size <= b of sum_B rate(B) ,
    rate(B) = sum_{g in B} 2/g - (1 - prod_{g in B} (1 - 2/g)) ,

the asymptotic form of the block bound (`rate(B)` is the per-column growth rate of `c_B` read off
Theorem 1).  The block-`b` bound can bite at large `L` only where `rho_b < 1`.  For the `K`
smallest gears, the least block size with `rho_b < 1` is

    K       3   4   5   6   7   8   9  10  11  12
    least b 1   2   2   3   4   5   6   7   8   8

so `b >= K - 3` from `K = 6` on.  Read as a fact about this family of bounds: no interaction law
of bounded order can reach the adversarial lemma at every `K` -- pairs die at `K = 6`, triples at
`K = 7`, quadruples at `K = 8`.

## Proof

### 1. Theorem 1, the linear deficit law

1. By the capacity formula, `floor((L + gh)/g) = floor(L/g) + h` and
   `(L + gh) mod g = L mod g`, so `e` is unchanged and `max_g(L + gh) = max_g(L) + 2h`.
   Symmetrically `max_h(L + gh) = max_h(L) + 2g`.
2. By the one-orbit reduction, `joint_max(g, h; L)` is the maximum, over the starting position
   `t`, of the number of marks of the fixed pattern `U` of period `gh` in the window
   `[t, t + L)`.  A window of length `L + gh` starting at `t` is the window `[t, t + L)` together
   with the full period `[t + L, t + L + gh)`, which contains exactly `|U|` marks whatever `t`
   is.  So every window count rises by exactly `|U|`, and hence so does the maximum:
   `joint_max(g, h; L + gh) = joint_max(g, h; L) + |U|`.
3. `|U| = gh - (g - 2)(h - 2) = 2g + 2h - 4`, since a residue mod `gh` is unstruck iff it is
   unstruck by both gears, and each gear leaves `g - 2`, resp. `h - 2`, residues open.
4. Subtracting, `c(g, h; L + gh) = c(g, h; L) + (2h + 2g) - (2g + 2h - 4) = c(g, h; L) + 4`.
   The `4` is exactly the number of residues mod `gh` that **both** gears strike -- the
   four-point shape `{0, S_g, S_h, S_g + S_h}` of `separation_drives_K.md` 3.1.
5. Nothing in steps 1-4 used the value of either separation, so the law holds for any
   separations.  Steps 1-3 generalise verbatim to a block `B` of any size with `P = prod g`:
   `max_g(L + P) = max_g(L) + 2P/g`, and `|U_B| = P - prod (g - 2)`, giving the stated block
   form.
6. Consequences.  `c >= 0` always (the union of two sets is at most the sum of their sizes, and
   each gear strikes at most its own maximum), so iterating step 4 downwards gives
   `c(g, h; L) >= 4 floor(L/gh)`; in particular `c > 0` for every `L > gh`, so every zero lies in
   `[1, gh]` and the permanent onset is the successor of the last zero there, a finite
   computation.  QED

### 2. Theorem 2, the shared-arc law, and Corollary 2.1

Let `a_g = a_h = a` and `L = a + 1`.

7. `a = min(s, g - s) <= (g-1)/2` for odd `g`, so `g >= 2a + 1 > a + 1 = L`, and likewise
   `h > L`.  Hence each residue class mod `g` meets the run at most once, so `g` strikes at most
   two of its columns, and if two then their distance `t` satisfies `t = +-a (mod g)` with
   `0 < t <= L - 1 = a`, forcing `t = a` (the alternative `t = g - a >= a + 1 > a` is too big).
   This is file 20's span lemma (Lemma 3) and type lemma (Lemma 4) in the case `g > L`.
8. So if `g` strikes two columns of the run they are `{i, i + a}` with `0 <= i` and
   `i + a <= L - 1 = a`, which forces `i = 0`: the two columns are the two **ends** of the run.
   A run of `a + 1` columns has room for exactly one arc of length `a`, and there is only one
   place to put it.
9. `max_g(L) = 2`: with `L = a + 1 < g` the capacity formula gives `floor(L/g) = 0` and
   `L mod g = a + 1 > a_g`, so `e = 2`.  Likewise `max_h(L) = 2`, so
   `max_g(L) + max_h(L) = 4`.
10. Now bound `joint_max(g, h; a+1)`.  If both gears strike two columns, by step 8 both strike
    the same two columns, the two ends, and the union has size 2.  If at least one strikes fewer
    than two, the union has size at most `2 + 1 = 3`.  Either way `joint_max <= 3`, so
    `c(g, h; a + 1) >= 4 - 3 = 1`.  QED
11. **Corollary 2.1.**  For twin gears `(g, g+2)` under the real separation, `3 a_g = g + 1` and
    `3 a_{g+2} = (g + 2) - 1 = g + 1` (file 20, Lemma 1), so `a_g = a_{g+2} = (g+1)/3` -- the
    "twin gears share an arc" of `arc_multiset.md`, which is file 02(e)'s shared tooth
    `u = (p+1)/6` doubled.  Theorem 2 applies with `a = (g+1)/3`, giving
    `c(g, g+2; (g+4)/3) >= 1`.  Since `(g+4)/3 < g` for `g > 2`, the onset is below the smaller
    gear.  Combined with Theorem 3 (a certificate, not proved), which gives
    `L0 >= max(a_g, a_h) + 1 = a + 1`, the onset of a shared-arc pair is **exactly** `a + 1`.

### 3. Theorem 3, the arc floor -- what is and is not established

12. No proof is on record, and the one-sentence mechanism the branch document offers ("below the
    larger short arc neither gear can be forced onto the other") cannot be the whole argument,
    for a reason that is itself informative: the sentence never mentions the separation, yet the
    statement is **false** for general separations.  Under random separations a gear may have
    arc 1 -- two adjacent teeth, which the real separation never has, since `a_g` is even by file
    20 Lemma 1 -- and the floor then fails.  Any correct proof must use `3 a_g = g -+ 1`.
13. What is established is a certificate over a stated finite range: 253 pairs `g < h <= 97`,
    5,009 instances, 0 exceptions; and the failure under random separations is 134 exceptions in
    12,306 instances, carried by 134 of 759 draws.  Both are recorded in Status.
14. Theorem 3 is used below only in Corollary 2.1's "exactly", and in the reading of the twin
    table.  Nothing in Theorems 1, 2, 4, 5 depends on it.

### 4. Theorem 4, the head collision and the permanent onset of `(5, 7)`

15. Gears 5 and 7 are a twin pair with `a_5 = a_7 = 2` (`s_5 = 2`, `s_7 = 5`, short arc 2 for
    both), so Corollary 2.1 applies with `a = 2`: `c(5, 7; 3) >= 1`, and 3 is the onset.  This
    is file 20's head collision seen as the smallest instance of Theorem 2; the head collision's
    content is not that 5 and 7 are the two smallest gears but that they are a twin pair.
16. The three recorded values are reproduced by an independent route -- a window sliding over the
    period 35, rather than 35 phase pairs -- and are `1, 1, 2` at `L = 16, 22, 28`.  Kept apart
    from `c`, the deficit under joint maximality is `c_max = 1, 1, 2` and the deficit under
    disjointness is `c_dis = 1, 2, undefined`: at `L = 28` gears 5 and 7 **cannot strike
    disjointly at all**, at any phase.  So the tight case is joint maximality.
17. **Permanent collision.**  The exact profile of `c(5, 7; .)` on `[1, 35]` has its last zero at
    `L = 20`; the values at `L = 21..35` are `1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4`, all
    positive.  For `L >= 36`, Theorem 1 gives `c(5, 7; L) = c(5, 7; L - 35) + 4 >= 4 > 0` because
    `c >= 0`.  Hence `c(5, 7; L) > 0` for every `L >= 21`.  This is a complete proof: a table of
    fifteen numbers plus Theorem 1.  QED
18. `W(K) >= 48 > 21` from `K = 4` on, so the head collision is not a fact about three special
    lengths -- the pair is in permanent collision across every window the route cares about.

### 5. Theorem 5, the block bound, and why the all-pairs form fails

19. Fix a partition `S = B_1 u ... u B_r` and a phase vector under which `S` covers the run.
    Every column of the run is struck by some gear of `S`, hence lies in the union of the strikes
    of the block containing that gear.  So the run is contained in the union of the `r` block
    unions, and `L <= sum_i |strikes(B_i) n run| <= sum_i joint_max(B_i; L)`, the last step
    because each block's count at the given phases is at most its maximum over all phases.
20. By definition `joint_max(B; L) = sum_{g in B} max_g(L) - c_B(L)`, which gives the second
    form.  The inequality holds for every partition, hence for the one minimising the right-hand
    side.  Blocks of size 1 have `c_B = 0` and return `sum_g max_g(L) >= L`, exactly file 20's
    counting filter (C).  QED
21. **Why the all-pairs form is not a bound.**  Bonferroni runs the other way:
    `|union| >= sum |A_i| - sum_{i<j} |A_i n A_j|`, an inequality of the wrong sign for an upper
    bound, and subtracting every pairwise deficit subtracts each higher-order coincidence
    repeatedly.  The sign change is visible in the increment: a new gear `q'` brings capacity at
    rate `2/q'` and, in the all-pairs accounting, is charged `sum_{g in M} 4/(g q')`, for a net
    rate `(2/q')(1 - 2 sum_{g in M} 1/g)` which turns **negative** once `sum_{g in M} 1/g > 1/2`,
    that is from `M = {5,7,11,13}` on (`1 - 2 sum 1/g` runs `+0.600, +0.314, +0.133, -0.021,
    -0.139, -0.244, -0.331` for `M = {5}, {5,7}, ..., {5..23}`).  From there the all-pairs form
    subtracts more than a gear brings and would "prove" statements that are false -- and it does:
    the two recorded covers of Theorem 5b are explicit counterexamples, so this is a refutation,
    not a gap.  Under the block form each gear sits in **one** block and is charged at most
    `b - 1` collisions, and the increment stays positive.
22. **Corollary 5.1, the reasoning part.**  Suppose `K` gears cover `W(K)`.  Those with `g >= L`
    strike at most 2 columns each (file 20, Lemma 2), and are granted 2 with no deficit -- an
    over-granting, which is the safe direction for a "no cover exists" conclusion, exactly as
    file 20 does.  The remaining gears are drawn from the finitely many primes below `L`.  So the
    cover's value is at most `bound_b(K, W(K))` for every `b`, and `bound_b(K, W(K)) < W(K)`
    contradicts Theorem 5.  The computation of `bound_b` is a certificate: the maximum over all
    `K`-sets, taken exhaustively over the subsets of the pool of primes below `L` (64, 1,093,
    4,944 and 82,160 subsets at `K = 3, 4, 5, 6`), with an exact subset DP for the best
    partition of each set.  The four values that matter are `bound_1(3, 28) = 26`,
    `bound_2(4, 48) = 46`, `bound_4(5, 60) = 55`, `bound_4(6, 88) = 87`, each attained at the `K`
    smallest gears.  At `K = 4` the whole margin is one pair deficit, `c(5, 7; 48) = 5`, applied
    to the set `{5, 7, 11, 13}` where counting alone gives 51.
23. One caveat that only helps: blocks whose period exceeds 9,000,000 are not evaluated and that
    block option is dropped, which can only **raise** the minimum over partitions, i.e. weaken
    the bound.  The reported values are therefore upper bounds on the true `bound_b`, and
    `bound_b < W(K)` remains a valid conclusion.

### 6. Recorded fact 6, the order obstruction

24. Reading Theorem 1's block form as a rate, `c_B(L)/L -> rate(B) = sum_{g in B} 2/g -
    (1 - prod_{g in B}(1 - 2/g))`, which is the inclusion-exclusion tail of order `>= 2`: a block
    of size `b` captures the Bonferroni truncation at order `b`.  Block size 1 recovers file 20's
    counting filter, block size `K` the exact question.  The block bound's asymptotic content is
    `L(1 - rho_b) <= O(1)` with `rho_b` as defined, so a block size with `rho_b >= 1` gives a
    bound that is vacuous at large `L` however the constants are improved.
25. The table of `rho_b` for the `K` smallest gears is a finite computation, and it is what the
    ladder `1, 2, 2, 3, 4, 5, 6, 7, 8, 8` reports.  It is recorded here as a fact about this
    family of bounds, not proved as a theorem about all `K`: no argument here shows that the
    ladder continues, only that it does so to `K = 12`, and the mechanism -- capacity is an
    order-1 sum `sum 2/g` while the truncated tail's leading term is order 2, and a partition
    into blocks of size `b` allows only about `K/b` of the pairs -- is a reading of the identity,
    not a proof of a growth rate.
26. The finite-`L` bound is slightly weaker than the rate condition: at `K = 5`,
    `rho_2 = 0.997 < 1` but `bound_2(5, 60) = 65 > 60`; the `O(1)` slack `sum_g 2` costs the
    difference.

## The certificates, and how to reproduce them

From the repository root, `uv run python research/anchor235/r55/<script>`:

| script | what it certifies |
|---|---|
| `cl_core.py` | the machinery: `max_g`, `joint_max` by the one-orbit reduction, the three deficits `c`, `c_max`, `c_dis`, the block deficit |
| `cl_pairs.py` | the head collision reproduced with the three deficits kept apart; every pair `g < h <= 97`: onset, permanent onset, the growth law, the twin table |
| `cl_families.py` | the real separation against 20 random draws and 8 coherent families, every pair `g < h <= 61` |
| `cl_triples.py` | the block growth law on the 20 triples of `{5..19}`; the record's overlap decomposed at `m11..m23` |
| `cl_bound.py` | the all-pairs refutation on the 7 recorded covers; `bound_b(K, W(K))` exhaustive at `K = 3..6`; the `rho_b` table to `K = 12` |
| `cl_laws.py` | the exceptionless statements with their counts: the period law under random separations, the arc floor, the shared-arc law, the period floor |

Outputs land in `research/anchor235/r55/results/` (untracked).

## Status

Kernel: **none**.  Nothing in this file is in Lean.

**Proved by reasoning, with no computation in the argument:** Theorem 1 and its block form, with
their consequences (slope `4/(gh)`, `c >= 4 floor(L/gh)`, every zero inside `[1, gh]`, the
permanent onset computable); Theorem 2 and Corollary 2.1's shared arc; Theorem 5 and the
invalidity of the all-pairs form; the implication "`bound_b(K, W(K)) < W(K)` implies no `K` gears
cover `W(K)`" of Corollary 5.1.

**Proved by reasoning plus a table small enough to check by hand:** Theorem 4's permanent
collision, `c(5, 7; L) > 0` for every `L >= 21` -- fifteen values of `c` on `[21, 35]` plus
Theorem 1.

**Certificate, exhaustive over a stated finite range, not proved:** Theorem 3, the arc floor
(253 pairs `g < h <= 97`, 5,009 instances, 0 exceptions); the numerical half of Corollary 5.1
(`bound_b(K, W(K))` as the exhaustive maximum over 64, 1,093, 4,944 and 82,160 subsets at
`K = 3, 4, 5, 6`, with the exact subset DP for the best partition of each).

**Recorded fact, neither proved nor claimed general:** item 6, the block-size ladder
`1, 2, 2, 3, 4, 5, 6, 7, 8, 8` at `K = 3..12`, computed for the `K` smallest gears.

**Checks behind the proved statements** (they add nothing to the proofs; they are the gates that
caught nothing):

| statement | range | exceptions |
|---|---|---|
| `c(g,h;L+gh) = c(g,h;L) + 4`, real separations | 228 pairs `g < h <= 97`, 248,334 instances | **0** |
| the same, random separations | 120 draws, 67,400 instances | **0** |
| both together | 315,734 instances | **0** |
| `c(g,h;gh) = 4` | 228 pairs | **0** |
| `c(g,h;L) >= 4 floor(L/gh)` | 66 pairs, 134,380 instances | **0** |
| the block growth law for triples, increment `4(g+h+k) - 8` | 20 triples, 61,980 instances | **0** |
| the shared-arc law `c(g,h;a+1) >= 1` | 14,340 shared-arc configurations over all 253 pairs and all arcs, not only the real separation (value 1 in 13,328, value 2 in 1,012) | **0** |
| twin pairs have onset exactly `a + 1` | 7 of 7 twin pairs to `(71, 73)` | **0** |
| twin pairs have onset `< g`; non-twin pairs do not | 7 of 7 and 246 of 246 | **0** |
| the arc floor, real separations | 253 pairs, 5,009 instances | **0** |
| the arc floor, random separations | 12,306 instances | **134**, carried by 134 of 759 draws |
| the block bound never violated by an explicit cover | the 7 recorded covers, blocks of size 2 | **0** |
| the record ladder `F({5..q}) = 7, 11, 18, 25, 34` reproduced at `m11..m23` | 5 machines | gate passed |

The random-separation rows are **sampled**, not exhaustive, and they enter only negatively: they
are what shows the arc floor is a fact about the real teeth rather than about two-class gears in
general.  Every other row is an exact maximum over a period or an exhaustive enumeration.

**Not established, and said so.**  The arc floor has no proof.  The ladder of item 6 is a
computation on one family of gear sets, not a theorem about all `K`.  Corollary 5.1 stops at
`K = 6`: at `K = 7..10` block size 4 leaves `147, 178, 264, 335` against `140, 160, 228, 280`,
and the exhaustive maximum at block size 5 and above was not run (4.5e6 subsets at `K = 7`
already), so the rows for `b >= 5` at `K >= 7` in the branch document are the value of the `K`
smallest gears only, which decides nothing where it is below `W(K)`.

## Prior art, and what is new

**Prior-art status.**  No fresh literature search was run for this file.  The statements below are
taken from the repository's own record -- `research/proof/arc_multiset.md`,
`research/proof/separation_drives_K.md`, `research/proof/gear_count.md`,
`research/proof/small_K_theorem.md`, and file 20's own prior-art section -- and are marked
**prior art not checked** where the record does not settle the point.

**Leverages.**  The tooth rule and the arcs (file 02(a), (b), (e)); the arc law `3 a_g = g -+ 1`
and the capacity formula (file 20, Lemmas 1, 2); the span lemma and the type lemma (file 20,
Lemmas 3, 4), used in step 7 in their `g > L` case; the one-orbit reduction (`the_wall.md` 5a);
the four-point overlap shape and the mean-overlap identity `4m/(gh)`
(`separation_drives_K.md` 3.1, N-S1); the recorded ladder `A(K)`, the windows `W(K)` and the
seven explicit covers (`arc_multiset.md` R1, `small_K_theorem.md` 1.2, file 20); "two gears share
a short arc iff they are twins" (`gear_count.md` R4, restated in `arc_multiset.md`); the
certified record ladder `F({5..q})`, used as a gate.

**Nearest on the record.**  `separation_drives_K.md` N-S1 has the same 4 as a *mean* overlap over
all `gh` phase pairs, and showed there that no separation can act through the mean.  Theorem 1 is
the same 4 as an exact *increment of a maximum*, which is a different object: it makes the
deficit an exactly linear function of `L` rather than an average.  `gear_count.md` R4 has the
domino dichotomy and the fact that a shared short arc means a twin pair; file 20 has the head
collision at three lengths, obtained by enumerating 35 phase pairs.  `arc_multiset.md` R7 has the
hole-distance dictionary that file 20's span lemma turned into a rule.

**New, as far as the record goes.**

* **The linear deficit law** `c(g, h; L + gh) = c(g, h; L) + 4`, with its one-line proof and its
  block generalisation, and the consequence that the permanent onset is exactly computable.
  Prior art not checked.
* **The shared-arc law** `a_g = a_h = a` implies `c(g, h; a+1) >= 1`, with its proof, and the
  corollary that every twin pair collides at `(g+4)/3`.  This is the general form of file 20's
  head collision, which is its `a = 2` instance; the content of the head collision is that 5 and
  7 are twins, not that they are small.  Prior art not checked.
* **The arc floor** as a statement separating the real separation from a random one (certificate).
* **The block bound** as the correct form of "capacity minus forced collision", with the
  all-pairs form refuted outright by two recorded covers rather than merely left unproved, and
  with the exact statement of what each block size buys: `b = 2` proves the lemma at `K = 4`
  where counting fails, `b = 4` at `K = 5` and `K = 6`.
* **The block-size ladder** as a recorded obstruction: the least `b` with `rho_b < 1` grows with
  `K`, so this family of bounds is not a route to all `K`.

**Not new.**  The capacity formula, the arc law, the span and type lemmas are file 20's.  The
four-point shape and the `4m/(gh)` density are CRT and are `separation_drives_K.md`'s.  "Twin
gears share an arc" is `gear_count.md` R4 by way of file 02(e).  The values `A(K)`, `W(K)` and
the seven covers are `arc_multiset.md` R1 and file 20, used here and not re-derived.

## Relationship to the conjecture

These statements **bound adversarial covers** and **cut the counting-tight cases**.  The counting
filter of file 20 is the block-1 case of Theorem 5 and it is exactly what runs out at `K = 4`:
`sum_g max_g(48) = 51` for `{5, 7, 11, 13}` against a window of 48, so counting permits a cover
that does not exist.  Theorem 2 supplies the missing subtraction -- one pair deficit,
`c(5, 7; 48) = 5` -- and closes that case by reasoning plus one number, where file 20 closed the
analogous cases by enumerating phase vectors.  At `K = 5` and `K = 6` the same happens with
blocks of four.  Where file 20's Theorem A is a solver's infeasibility certificate, Corollary 5.1
is an inequality with a stated finite maximum inside it.

They **prove nothing past `K = 6` by reasoning**.  Corollary 5.1 stops at `K = 6`; at `K = 7..10`
the block-4 bound is above the window and the exhaustive maximum at larger block sizes was not
run.  There is no induction step here.  Every `K` is a separate finite computation, and the cost
of the computation grows with the block size that would be needed.

And the **order obstruction says no bounded-order interaction law reaches all `K`**.  The block
size with any chance of biting is `1, 2, 2, 3, 4, 5, 6, 7, 8, 8` at `K = 3..12`, growing without
bound; a pairwise law is vacuous from `K = 6`, a triple law from `K = 7`, a quadruple law from
`K = 8`.  This is recorded as a fact about this family of bounds, not proved for all `K`, but its
source is the identity `rate(B) = sum 2/g - (1 - prod (1 - 2/g))` -- Bonferroni truncation -- and
not any feature of the real teeth.  So the induction step the search has been looking for is not
a pairwise law, and cannot be made one by improving constants.  What a new gear costs the
adversary is not a bounded number of pairwise collisions but a share of every higher-order
overlap.

The budget inequality `F(M+q') <= F(M) + q'` is untouched by this file; nothing here is an
increment statement about the record.  Two smaller readings of the record are worth stating and
are measurements, not results: at the real machines `m11..m23` the record stretch is almost
exactly maximal gear by gear (the shortfall from per-gear maximality is `0, 1, 0, 1, 2`), so the
record's whole price is overlap; and the sum of the pairwise deficits at the record length tracks
that price to within one unit at four of five rungs (`1, 0, 5, 12, 14` against `1, 2, 5, 10,
15`), in both directions, so it is an accounting coincidence and not a bound.  The valid matching
part is much smaller -- 27% of the price at `m23` -- and that gap is precisely the difference
between the all-pairs identity and the block bound.

Nothing measured enters Theorems 1, 2, 4, 5 or the reasoning half of Corollary 5.1.  Theorem 3
and the numerical half of Corollary 5.1 are exhaustive certificates over stated ranges; the
random-separation rows are sampled and are used only to show the arc floor is not separation-free.

## Where it is used

* Corollary 5.1 gives a second, independent proof of file 20's Theorem A at `K = 4, 5, 6`, by a
  route that uses no solver and no phase enumeration; at `K = 4` it also proves what file 20's
  counting filter (C) alone could not.
* Theorem 2 replaces file 20's head collision as the general statement: any two gears sharing an
  arc collide at once, and every twin pair of gears is an instance.  Anywhere the head collision
  is invoked to close a counting-tight case, the shared-arc law is the reason.
* Theorem 1 makes the deficit of any pair a closed-form linear function, so the permanent onset
  `L1 <= gh` is computable and "this pair is in permanent collision on this window" is decidable
  by one table lookup.  Theorem 4 is that statement for `(5, 7)`, and gears 5 and 7 sit in every
  machine the route builds.
* Item 6 is a stop sign for the search: it says in advance that a pairwise, triple, or any
  fixed-order interaction law will not carry the adversarial lemma to all `K`, so effort of that
  shape is spent.
* The one candidate the branch leaves open, and it is weak: a statement of the form "at every
  `L`, some growing number of disjoint pairs of the machine's gears are past their permanent
  onset" would give a deficit growing with `K` rather than a fixed one.  Item 6 says a fixed
  number of pairs is not enough; nothing here supplies a growing number, and nothing here forbids
  it.

## Source

`research/proof/collision_laws.md` (the branch document: pre-registration, the eight predictions
and their verdicts, every number); scripts and outputs in `research/anchor235/r55/`.
Ingredients: `docs/proofs/02-tooth-rule.md` (the tooth rule, the arcs, twin gears share a tooth),
`docs/proofs/20-adversarial-lemma-small-K.md` (the arc law, capacity, the span lemma, the type
lemma, the head collision, `A(K)`, `W(K)`, the seven covers).  Record context:
`research/proof/separation_drives_K.md` (the four-point shape, the mean-overlap identity,
coherent families), `research/proof/arc_multiset.md` (the ladder `A(K)`, the hole-distance
dictionary), `research/proof/gear_count.md` (the domino dichotomy, shared arc iff twins),
`research/proof/small_K_theorem.md` (the explicit covers), `research/proof/the_wall.md` (the
one-orbit reduction, faces A and C).
