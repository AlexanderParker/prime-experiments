# Ideas generated from what the session established

Ordered by how much of the existing structure they reuse and how far they are from something
already verified. Each states what is known, what the idea is, and what would falsify it quickly.

The remaining gap throughout is section 26c of `covering-bound-route.md`: prove
`min_L h(L) = h(1)`, where `h(L) = G(L)/N(L)` is the hazard of the gap distribution.

## 1. The gap sequence is a constrained word (verified structure, strongest lead)

**Known.** Gear 3 blocks one of any two adjacent positions; gear 5 blocks one of any three spaced 3
apart (section 18a). Extending the search, each gear forbids its own minimal configurations of
exposed positions:

    q = 5   12 minimal triples, including (0,3,6), (0,3,12), (0,6,18), (0,9,12)
    q = 7    7 minimal quadruples, including (0,3,9,12), (0,9,18,27)
    q = 11   4 minimal sextuples
    q >= 13  none within a span of 27

**Idea.** Read those as forbidden *factors of the gap word*. Writing gaps in units of 3, gear 5
forbidding `(0,3,6)` says no three consecutive exposed slots, that is **the factor `11` never
occurs**; gear 7 forbidding `(0,3,9,12)` says **the factor `121` never occurs**. Verified: over gear
sets to 7, 11, 13 the counts of `11` and `121` are zero, while `212` occurs 2, 6, 30 times - so the
restriction is real and not vacuous.

A word avoiding a finite set of factors is recognised by a finite automaton, so its letter statistics
- which *are* the gap counts `n_i` - come from a transfer matrix, as eigenvalues rather than as
inclusion-exclusion sums. That is a different computational route to `n_i` than the one that stalled
in section 25b, where the inclusion-exclusion grew like `2^j`.

**Falsifies quickly.** Enumerate the forbidden factors contributed by gears to 19, build the
automaton, and check whether its letter frequencies reproduce the measured `n_1 .. n_4`. If the
automaton is not finite - because larger gears contribute longer and longer forbidden factors without
bound - the idea fails immediately.

## 2. Exposed runs have length at most 2 (immediate corollary, exact)

**Known.** The gear-5 law forbids three consecutive exposed slots.

**Idea.** So in `k`-space the exposed set is exactly a disjoint union of **isolated points and
dominoes** - nothing longer. The counts follow: dominoes number `n_1 = prod (q-4)`, total runs are
`prod (q-2) - prod (q-4)`, so singletons number `prod (q-2) - 2 prod (q-4)`. Checked at `{3,5,7}`:
9 singletons and 3 dominoes, 15 points in 12 runs, matching `A = 15` and `n_1 = 3`.

This also explains why the `prod (q - 2k)` family of section 32c collapses at `k = 3` - it counts
`k`-consecutive exposed positions, and there are none beyond `k = 2`. So that family carries exactly
two pieces of information, `A` and `n_1`, and no more.

**Use.** The pattern is now described completely at the local scale: points and dominoes, with known
counts. The gap distribution is the distribution of spacings between these objects, which is a
one-dimensional arrangement problem with two object types.

## 3. Gap counts as spectral quantities (the frequency route)

**Known.** The spectrum factorises exactly, `Ehat(k) = prod_q ehat_q(k t_q mod q)` (section 35a of the
main document), and every beat matters - the L1 norm grows like `2.06^n` (section 36b).

**Idea.** The gap counts are correlations, and correlations are spectral. The pair correlation
`C(g) = sum_m E(m) E(m+g)` equals `P sum_k |Ehat(k)|^2 omega^{kg}`, and `|Ehat|^2` factorises exactly
as `Ehat` does. Since `N(L) = sum_{g>L} (g-L) n_g` and each `n_g` is an alternating sum of
correlations, the hazard condition becomes a weighted spectral sum.

Why it might help where the direct route did not: the obstruction in section 25b was that
inclusion-exclusion over configurations grows exponentially. A spectral sum has one term per
frequency with a *product* amplitude, so the growth is in the number of frequencies rather than in
combinatorial depth, and the factorisation is exact.

**Falsifies quickly.** Write `n_3` and `n_4` as spectral sums and check they reproduce the verified
closed forms. If the spectral expression needs as many terms as the inclusion-exclusion did, nothing
is gained.

## 4. Recursion in the gear set rather than in L

**Known.** The per-`j` recipe (section 24b) recurses on `L` and stalls because the number of
conditions grows like `F_k ~ 0.055 y^2`.

**Idea.** Recurse on the *gear set* instead. Adding gear `q` deletes the exposed positions in two
residue classes mod `q` and merges the gaps on either side of each deletion. By CRT the deleted
positions are equidistributed across the existing pattern, so the new gap distribution is an explicit
transform of the old one: each gap either survives or absorbs its neighbour. Track the distribution
through that transform and the hazard condition becomes an induction on gears - one step per gear,
`pi(y)` steps, rather than `F_k` conditions.

**Falsifies quickly.** Compute the transform from the gear set to 13 and check it predicts the
measured gap distribution at 17. If the merge rule depends on more than the gap distribution - for
instance on the joint distribution of consecutive gaps - the recursion does not close, though the
word structure of idea 1 might then supply the missing state.

## 5. What not to try again

Recorded so the next attempt does not repeat them:

* monotonicity of the margin in the gear set - **false**, the ratios peak then fall (24a);
* the universal bound `h >= 1/(F_h - L)` - **circular**, presupposes `F_h` (22b);
* log-concavity of `N` - **false**, fails at `L = 3` (27a);
* tail-fraction bounds from `N(L) <= N(1) - (L-1)G(L)` - **too crude** from `L = 6` (27b);
* the per-`j` recipe beyond small `j` - **does not scale** (25b);
* any argument resting only on "gaps are multiples of 3 and at least 3" - the multiset `{3,3,15}`
  satisfies both and violates the claim (27c).
