# Where the programme stands

A single statement of what the twin conjecture reduces to in this programme's terms, what is proved, what
is not, and why each attempted route fails. Written to be read on its own.

## 1. The reduction, exactly

A twin pair is `(6k-1, 6k+1)`. A prime `q >= 5` divides one member exactly when `k = +- 6^{-1} mod q`, so
gear `q` blocks two residues mod `q`; gear 3 never acts, since `6k +- 1` is never divisible by 3. Slot `k`
is a genuine twin, certified by the gears up to `y`, exactly when it is exposed to all of them **and**
`6k + 1 <= y^2`, since a larger prime factor would have to exceed `y`.

Slot `k = 0` is exposed for every gear set - every gear divides 0, so every gear shields rather than
threatens - and the exposed pattern is periodic with period `P = prod q` and symmetric about 0. Gaps of the
exposed pattern are at most `F_k(y)`, so an exposed slot exists in `(y/6, y/6 + F_k(y)]`. For that slot to
be inside the certified window:

> **The twin prime conjecture follows from `F_k(y) <= (y^2 - y)/6` for all large `y`.**

Measured, this holds with a factor of about 2.5 and the ratio is flat:

    y                7     11     13     17     19     23     29     31     37     41     43
    F_k              5      7     11     18     25     34     43     58     88     91    103
    (y^2 - y)/6    7.0   18.3   26.0   45.3   57.0   84.3  135.3  155.0  222.0  273.3  301.0
    ratio        0.714  0.382  0.423  0.397  0.439  0.403  0.318  0.374  0.396  0.333  0.342

## 2. Two frames, one problem

Everything in `covering-bound-route.md` and `forbidden-configurations.md` uses the **adjacent frame**: every
odd prime, gear 3 included, blocks the adjacent pair `{o, o+1} mod q`. Gear 3 there confines the exposed set
to one class mod 3, so every adjacent-frame gap is 3 times a `k`-space gap. Verified as
`F_adjacent = 3 F_k` and `F2_adjacent = 3 F2_k` for seven and six gear sets respectively. Results transfer
with `L -> 3L`. Lengths quoted in those documents are adjacent-frame; divide by 3 for `k`-space.

One consequence matters: the adjacent-frame `L = 1` has no `k`-space counterpart, so `h(1) = d/(1-d)` - the
"free" case the covering route is built on - is an artefact of the finer grid. In `k`-space the minimum of
`h/d` sits at `L = 2`, and the conjecture needs only `h(L) >= d`, not `min_L h(L) = h(1)`.

## 3. What is proved

Mechanics of the machine:

* **teeth**: gear `q` threatens `k = +- 6^{-1} mod q`, with `6^{-1} = (q+1)/6` or `q - (q-1)/6`; the
  symmetry follows from `6 * 6^{-1} = 1`;
* **the ±1 walk**: every gear and every sub-machine steps the 6-cycle by exactly `+-1` per rotation, since
  every prime gear is `1` or `5 mod 6` and the units mod 6 are closed under multiplication;
* **self-blocking**: the lower tooth of gear `q` is the index of the pair containing `q`; twin gears share
  their lower tooth;
* **window identity**: `survivors(y, K) = T(6K+1) - T(y)` for `6K+1 <= y^2`, exact;
* **tooth budget**: per rotation, one shield, two killers, `q-3` misses - universal;
* **minimal size law**: gear `q` can be forced to block one of a set `S` only when `|S| >= (q+1)/2`, and
  that bound is attained. Exposure form: **any `(q-1)/2` positions are simultaneously exposable to gear
  `q`**. Gear 3's and gear 5's blocking laws are its first two cases;
* **large gears force nothing new**: within a box of length 16, gears 29 to 47 add no minimal forbidden
  configuration beyond gears to 23;
* **gcd form**: `m` is a twin slot iff `gcd(36m^2 - 1, primorial(sqrt(6m+1))) = 1`;
* **generating polynomial** `prod (q - 2 + 2x)`, whose coefficients count slots threatened by exactly `j`
  gears, and whose evaluations give every alignment law;
* **factorised spectrum** `Ehat(k) = prod_q ehat_q(k t_q mod q)`, agreeing with FFT to `1.1e-16`.

The recursion on the gear set:

* **the merge transform**: adding gear `q` is `q` copies of the old pattern, each thinned at a different
  phase, laid end to end; every exposed point is deleted in exactly 2 of the `q` laps. Verified against
  direct construction on full gap histograms;
* **the chain condition**: `F(M+q)` is determined by the old gap word and `q` alone - a chain of `k`
  deletions merges `k+1` gaps, and the partial sums of the interior gaps must stay in `{0,1}` or `{0,-1}`
  mod `q`. Verified exact in 15 cases;
* **the deletion-spacing lemma**: consecutive deletions are at least `q-1` apart, from `delta >= 3` and
  `delta = 0, +-1 mod q`. Tight;
* **the saturation theorem**: if `q - 1 > F(M)` then `F(M+q) = F2(M)` exactly. Checked over 48 pairs.

And a closed-form method for the next twin prime, which did not exist before: the bite distance
`min((u_q - m) mod q, (-u_q - m) mod q)` per gear, verified to `k = 10^16`, and the explicit
`J(m0) = sum_J prod (1 - E(m0+i))` form.

## 4. What is not proved

Only the bound of section 1. In its equivalent forms:

* `F_k(y) <= (y^2 - y)/6` directly;
* the covering bound `N(L) <= P (1-d)^L`, equivalently the hazard condition `h(L) >= d` for every `L`,
  equivalently `kappa(L) >= 0` where `kappa(L) = (h(L)/d - 1)/d`. Measured minimum `kappa` in `k`-space
  settles near `0.68`;
* `F(y) <= C sum_{3<=q<=y} q` for a constant `C <= 1.8`, which with the elementary
  `sum q <= (y^2+2y-3)/4` closes section 1 for `y >= 29`. Measured `C` peaks at `1.354`.

Proved special cases: `h(1), h(3), h(6), h(9)` in the adjacent frame, and `kappa(L) >= 1` verified over all
1.67 million block starts to `L = 5 * 10^6`.

## 4a. There is no covering-design freedom: every configuration is one translate

This corrects a framing used throughout the earlier sections and in the routes of section 5.

The covering problem was posed as: choose an offset `o_q` for each gear `q <= y`, where gear `q` blocks
`{o_q, o_q + 1} mod q`, and ask which runs can be covered. That looks like a design problem over
`prod q = P` configurations. It is not. **By CRT the offset vector `(o_q)` is exactly a single residue
`c mod P`** with `c = o_q mod q` for every `q`, and then gear `q` blocks `i = c` or `c+1 mod q`. So the
uncovered set is

    { i : q does not divide (i - c) and q does not divide (i - c - 1), for every q <= y }
  = { i : i - c and i - c - 1 are both coprime to P },

which is a **translate** of the single pattern `{ n : n and n+1 both coprime to P }`. The `P` offset vectors
are precisely the `P` translations of one pattern, and nothing else.

Hence

> **`F(2,y) = 1 + ` the maximum gap of `{ n : n, n+1 both coprime to P(y) }`**, the Jacobsthal-type function
> for the pair `{0,1}` modulo the primorial.

Verified: the configuration with every offset equal to 1 - where `i` is uncovered exactly when `i-1` and
`i-2` are both free of prime factors `<= y` - attains `F(2,y)` exactly at `y = 7, 11, 13, 17, 19, 23, 29`,
ratio `1.000` in all seven cases, which is what the theorem requires. The `y = 29` check was done by a
segmented sieve over all `3.2 * 10^9` positions of the period, since it does not fit in memory whole.

Three consequences:

* the exhaustive offset searches in `maxgap.rs`, `coverbound.rs` and `holegap.rs` are sound but are
  searching a space isomorphic to `Z_P`; they explore translations, not designs;
* "the extremal configurations are efficient" (section 37b of `twin-prime-program.md`) is automatic - they
  are all the same configuration;
* **the obstruction is not combinatorial fitting.** `F` is the maximum gap of one explicit pattern, and
  bounding it is the Jacobsthal problem. In `k`-space the same statement is the obvious one:
  `F_k(y)` is the maximum gap of `{ k : 6k-1, 6k+1 both y-rough }`, the twin pattern itself, with the two
  frames related by the factor 3 of section 2.

## 5. Why every route fails, and the one reason behind it

Gear `q` covers about `2L/q` positions of a run of length `L`, so any argument that bounds `F` by comparing
total capacity against `L` needs `sum 2/q < 1`. It is not:

    y            sum 2/q (q>=5)   sum 2/q (q>=3)   overlap
    5                  0.400            1.067        6.7%
    7                  0.686            1.352       35.2%
    11                 0.868            1.534       53.4%
    13                 1.021            1.688       68.8%
    23                 1.331            1.998       99.8%
    47                 1.657            2.323      132.3%
    101                1.959            2.625      162.5%

**Capacity arguments therefore work only up to `y = 11`, and never again.** By `y = 47` the gears carry 132%
more covering capacity than a run needs.

### The two-scale version fails too, and fails worse as `y` grows

The obvious repair is to split the gears at a threshold `z`. The gears `<= z` can cover a run of length
`F(z) - 1` outright, so in a window of length `L` they leave at least `L/F(z) - 1` positions uncovered, and
the gears in `(z, y]` must cover those at a cost of at most `2(L/q + 1)` positions each. That yields a bound
on `L` exactly when

    F(z) * 2 * sum_{z < q <= y} 1/q  <  1.

Minimising the left side over `z` (excluding the degenerate `z = y`, where the sum is empty):

    y             13     17     19     23     29     37     47
    best z         3      3      3      3      3      3      3
    product    3.064  3.417  3.733  3.994  4.201  4.556  4.970

The best threshold is always `z = 3`, where the condition reduces to
`sum_{5 <= q <= y} 1/q < 1/6`, and that sum is already `0.51` at `y = 13`. So the two-scale family fails by a
factor of 3 to 5, and since `sum 1/q` diverges **the shortfall grows without bound** - splitting at a
threshold gets worse with `y`, not better.

### What the shortfall measures

The gap between capacity and feasibility is the clustering of the exposed set: its maximum gap against its
mean gap, `F(z) * d_z`.

    z            3     5     7    11    13    17    19    23    29    31    37    43    47
    1/d_z     3.00  5.00  7.00  8.56 10.11 11.46 12.81 14.03 15.07 16.11 17.03 18.77 19.61
    F(z)         3     6    15    21    33    54    75   102   129   174   264   309   354
    F(z) d_z  1.00  1.20  2.14  2.45  3.26  4.71  5.86  7.27  8.56 10.80 15.51 16.46 18.06

The maximum gap runs from equal to the mean gap at `z = 3` to 18 times it at `z = 47`, and keeps growing -
`F ~ C y^2/log y` against `1/d ~ log^2 y` gives `F d ~ y^2/log^3 y`. **That growing clustering factor is the
entire difficulty.** Capacity is abundant at every scale; the exposed set is simply far from uniform, so
every bound that treats it as uniform fails by the clustering factor. Section 4a rules out the other reading
- that the difficulty is a design question about fitting residue classes together - since there is no design
freedom to exploit.

Every closed route below is a different attempt to see past the uniform estimate, and each fails for its own
reason:

* **capacity / usefulness counting** - `sum 2/q > 1` from `y = 13`;
* **two-scale capacity counting**, splitting the gears at any threshold - short by a factor of 3 to 5, and
  the shortfall grows without bound since `sum 1/q` diverges;
* **monotonicity of the margin** in the gear set - false, the ratios peak then fall;
* **the universal bound `h >= 1/(F_h - L)`** - circular, presupposes `F_h`;
* **log-concavity of `N`** - false at `L = 3`;
* **tail-fraction bounds** - too crude from `L = 6`;
* **the per-`j` recipe** - it does scale, 2548 contributing terms at `L = 39` rather than `5.5e11`, but it
  gives one condition per `L` and the tight `L` do not terminate;
* **finite automaton over the gap word** - the antidictionary is infinite, and the automaton's letter
  statistics count which words *can* occur, a quantity independent of `y`, while `n_j` counts how often they
  *do* and scales with `P`. Weighting it needs the `n_j` themselves;
* **per-gear conditional marginals** - they *rise* under conditioning, by up to 63%, so gear exhaustion is
  not a per-gear effect;
* **weak negative association** - fails narrowly at small `L`;
* **per-gear usefulness for the step form** - the offsets blocking `L` are jointly below average only when
  `L mod q >= q/4`;
* **bounding the chain length `k` from gap structure** - `(k-1)(q-1) <= sum h_j <= (k-1) F(M)` is vacuous
  whenever `F(M) >= q-1`, which is the regime that matters;
* **per-step increment bounds** - the gear-37 step reaches `2.432 q`, above the `1.8` needed, so summing
  per-step bounds cannot give `C`.

## 6. What would count as progress

Anything that sees the *clustering* rather than the average. Concretely, in decreasing order of how close it
sits to what is already built:

1. a bound on the **average** increment `F(M+q) - F(M)` over gears up to `y`, since the per-step version is
   closed but the aggregate `C` is well behaved - the two pieces of each increment trade off against each
   other;
2. a lower bound on `kappa(L)` uniform in `L`, which needs no exact minimisation - in `k`-space the
   requirement is only `kappa >= 0` against a measured `0.68`;
3. control of the average of the pair weight `psi(delta) = 3C prod_{q|delta} (q-2)/(q-4)
   prod_{q|delta^2-1} (q-3)/(q-4)`, since `kappa(L) = L - sum_{delta<=L} psi(delta)` to second order and
   `mean psi -> 3` exactly.

What section 5 rules out, in one line: **anything that bounds `F` by comparing how much the gears can cover
against how much needs covering.** That is true at one scale, at two scales, and with the shortfall growing
in both. A proof has to see the clustering, and the clustering factor `F d` grows like `y^2/log^3 y`, so it
cannot be treated as a constant either.

The mechanical apparatus is complete and independently cross-checked. What remains, stated as plainly as
section 4a allows, is a bound on the maximum gap of one explicit pattern - `{ n : n, n+1 both coprime to
P(y) }`, or equivalently the twin pattern itself in `k`-space. That is the Jacobsthal problem for a pair, and
the capacity barrier of section 5 is why it has resisted every elementary attempt here, including all of
them made in this programme so far.
