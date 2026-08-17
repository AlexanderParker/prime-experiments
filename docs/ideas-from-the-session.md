# Ideas generated from what the session established

Ordered by how much of the existing structure they reuse and how far they are from something
already verified. Each states what is known, what the idea is, and what would falsify it quickly.

The remaining gap throughout is section 26c of `covering-bound-route.md`: prove
`min_L h(L) = h(1)`, where `h(L) = G(L)/N(L)` is the hazard of the gap distribution.

The conceptual frame these all sit inside - the machine as gears turning together forever, and the
gear at infinity - is recorded separately in `docs/gear-at-infinity.md`. It is not a proof, but four of
its six steps are theorems, and it is where the `+/-1` walk law and both blocking laws came from.

## 1. The gap sequence is a constrained word - TESTED, REFUTED, but it paid out

Tested in full; see `docs/forbidden-configurations.md`. The idea itself fails on three counts, and two
of the figures quoted in the original version of this section were wrong.

**Corrections to what was written here.** "`q >= 13`: none within a span of 27" is false - gear 13 has a
forbidden configuration of span 24. And gear 5 forbids ten minimal gap words, not just `11`
(`11, 13, 16, 24, 31, 42, 61, 121, 151, 222`); gear 7 forbids seventeen, not just `121`. The exact
minimal size for gear `q` is `(q+1)/2` positions, and the minimal span grows like `1.9 q`.

**Why the idea fails.** The antidictionary is not finite - minimal forbidden words keep appearing at
every length, still rising at the boundary of a box of length 16. And even granting an automaton, its
letter statistics count *which words can occur*, a quantity independent of `y`, whereas the `n_j` count
how often they *do* occur and scale with `P`. Measured side by side the two frequency distributions are
nowhere near each other and move in opposite directions with `y`. Getting `n_j` out would need the
automaton weighted by the CRT measure, and those weights are the `n_j` - circular in the same way
section 22b was.

**What it paid out.** Three things, all verified and all new:

* **the minimal size law** - gear `q` can force a block only in a configuration of at least `(q+1)/2`
  positions, and that bound is attained. Equivalently, in exposure form: **any `(q-1)/2` positions can
  be simultaneously exposed to gear `q`**. Gear 3 and gear 5's laws are its first two cases;
* **large gears force nothing new** - within a box of length 16 and letters to 6, gears 29 through 47
  contribute zero minimal forbidden words beyond what gears to 23 already forbid, and this is not a box
  artefact for 29 and 31, whose own minimal configurations fit inside it;
* **the step form of the remaining gap** - `G(L) = N(L) - N(L+1)`, so `min_L h(L) = h(1)` is exactly
  `rho(L) <= rho(1)` with `rho(L) = N(L+1)/N(L)`. Verified at every `L` up to `F_h`, not just the block
  starts, for all gear sets to `y = 19`.

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

## 3a. The tight cases are a short fixed list, and the recipe reaches them

**Known.** `h` rises within each block `{1,2}, {3,4,5}, {6,7,8}, ...`, so the minima of `h/d` sit at the
block starts. Ranking them (`docs/forbidden-configurations.md` section 8), the tight starts are
`1, 6, 3, 9, 21, 24` with `15, 39, 54` behind them - the **same small absolute values for every gear
set**, not a fixed fraction of `F_h`, stable from `y = 13` through `y = 23`. The four tightest are
`1, 6, 3, 9`, exactly the four already proved.

**Why the recipe now works.** Section 25b rejected the per-`j` recipe because `c_j(L)` sums over `2^L`
subsets. But the head gears annihilate almost every term, and pruning on the first fully covered gear
visits only survivors: 2548 contributing terms at `L = 39` instead of `5.5 * 10^11`. So each tight case
is an explicit finite inequality between products over the gear set, and all of them check out from
closed forms alone for `y = 23` through `199`.

**What is left, on the right scale.** `h(L)/d` tends to 1 for every `L`, `L = 1` included, so no margin
can be read off it. The scale-free quantity is `kappa(L) = (h(L)/d - 1)/d`, for which `kappa(1) = 1/(1-d)`
exactly and the claim is `kappa(L) >= 1/(1-d)` for `L >= 2`. Measured to `y = 100003` at 80-digit
precision, `kappa(1) -> 1` while every other `kappa(L)` settles above it, the smallest being
`kappa(6) -> 1.67`, and `kappa` grows with `L` to `7.58` at `L = 63`.

So the target is a **uniform lower bound, not an exact minimisation**: `kappa(L) >= 1.1` for `L >= 3`
closes the whole thing for every `y >= 13`, and the binding value is `L = 6`, already proved.
**Falsifies quickly:** if `kappa(6)` drifts towards 1 rather than settling near `1.67`, the room
disappears; it has not through nine thousand gears.

## 4. Recursion in the gear set rather than in L - BUILT, and now the leading route

Built exactly in `docs/gear-recursion.md` and `research/gear_recursion.py`. It closed better than
expected, and it also turned up a frame confusion that was affecting everything else.

**The frames.** The adjacent frame - gear 3 included, each gear blocking `{o, o+1}` - and the real
`k`-space frame - gears `q >= 5` only, teeth at `+- 6^{-1}`, gear 3 inert - are the **same problem scaled
by 3**, verified as `F_adjacent = 3 F_k` for seven gear sets. So results transfer with `L -> 3L`. Two
things follow: the adjacent-frame `L = 1` has no real counterpart, so `h(1) = d/(1-d)` is a grid artefact
and the real minimum sits at `L = 2`; and `min_L h(L) = h(1)` is **stronger than the conjecture needs**,
which only wants `h(L) >= d`, that is `kappa >= 0`, with measured room `0.68`.

**The transform and the recursion.** Adding gear `q` is `q` copies of the old pattern, each thinned at a
different phase of `q`, laid end to end - and each exposed point is deleted in exactly 2 of the `q` laps.
Deleting `k` consecutive points merges `k+1` gaps, and the deleted points must lie in `{phi, phi+1} mod q`,
so the partial sums of the interior gaps stay in `{0, 1}` or `{0, -1} mod q`. That condition gives
`F(M + q)` **exactly from the old gap word**, verified in 15 cases. Its `k = 2` case is the
deletion-spacing lemma: consecutive deletions are at least `q - 1` apart, proved and tight.

**The skeleton.** `F_adjacent(y) ~ C sum_{3<=q<=y} q` with `C` measured in `[0.81, 1.10]`; the odd primes
sit inside the odd numbers so `sum q <= (y^2+2y-3)/4` with no prime counting; and the requirement
`F_k <= (y^2-y)/6` then follows from any proved `C <= 1.8` for `y >= 29`, the smaller `y` being checked
directly. **Falsifies quickly:** if the increment `F(M+q) - F(M)` can exceed `1.8 q` along the consecutive
chain, the constant fails - measured ratios there are all at most `1.29`, though skipping a gear reaches
`1.59`.

## 4b. The old framing of idea 4

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
  satisfies both and violates the claim (27c);
* a finite automaton over the gap word, whether via a finite antidictionary or via transfer-matrix
  letter statistics - **both fail**, section 5 of `forbidden-configurations.md`;
* a **per-gear** usefulness argument for the step form - the offsets that block position `L` are
  jointly below average at covering `[0, L)` only when `L mod q >= q/4`, so for gears with
  `L mod q < q/4` the conditioning pushes the wrong way. Exact, checked over 4525 pairs `(q, L)` with
  zero exceptions. Any proof has to treat the gears jointly;
* **per-gear conditional marginals** - measured directly, they *rise* under the conditioning rather than
  fall, by up to 63%, so gear exhaustion is not the mechanism (section 7 of
  `forbidden-configurations.md`);
* **weak negative association**, `h(L) >= prod (1 - marginal_q)` - fails narrowly at small `L`, once at
  `y = 17` and twice each at `y = 11, 13`. Nearly true is not true.
