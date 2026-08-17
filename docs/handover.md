# Twin primes via the blocked-slot machine: handover

Written for a reader who will attempt the remaining abstract proof, and who should verify rather than trust
what follows. Everything below is either proved, computationally verified at the stated scale, or explicitly
flagged as refuted. The refutations in section 6 are mathematical dead ends with counterexamples attached, and
they are load-bearing: seventeen plausible claims were recorded and then killed here, several by their own next
data point. Implementation defects found and fixed along the way are deliberately omitted - they are not part
of the mathematics and would only add noise.

Notation throughout: `q` ranges over odd primes, `y` is the gear bound, `P(y) = prod_{3<=q<=y} q`,
`d = prod (1 - 2/q)`, `A = prod (q-2)`.

---

## 1. The core thesis

**Two coordinate systems, one problem.**

*`k`-space (the real frame).* A twin pair is `(6k-1, 6k+1)`. A prime `q >= 5` divides one member exactly when
`k = +- 6^{-1} mod q`, so gear `q` blocks two residues mod `q`. Gear 3 never acts, since `6k +- 1` is never
divisible by 3. Write `F_k(y)` for the maximum gap of the set of `k` exposed to all gears `<= y`.

*The adjacent frame.* Every odd prime, gear 3 included, blocks the adjacent pair `{o, o+1} mod q`. Write
`F(2,y)` for the maximum gap. **`F(2,y) = 3 F_k(y)` exactly** - verified for seven gear sets - because gear 3
confines the exposed set to one class mod 3.

**The reduction.** Slot `k` is certified as a genuine twin by gears up to `y` exactly when it is exposed to
all of them and `6k+1 <= y^2`. Slot `k = 0` is exposed for every gear set, and gaps are at most `F_k(y)`, so an
exposed slot exists in `(y/6, y/6 + F_k(y)]`. Therefore:

> **CORE THESIS.** The twin prime conjecture follows from
>
>     F_k(y) <= (y^2 - y)/6      for all sufficiently large y.

**The equivalent form that removes all apparent combinatorial freedom** (section 3, item 17):

>     F(2,y) = 1 + max gap of { n : n and n+1 are both coprime to P(y) }.

This is the Jacobsthal-type function for the pair `{0,1}` modulo the primorial. There is **no** covering-design
freedom to exploit - see item 17. Any proof must bound the maximum gap of this one explicit pattern.

**Measured slack.** `F_k(y) / ((y^2-y)/6)`:

    y         7     11     13     17     19     23     29     31     37     41     43
    F_k       5      7     11     18     25     34     43     58     88     91    103
    ratio 0.714  0.382  0.423  0.397  0.439  0.403  0.318  0.374  0.396  0.333  0.342

The requirement holds with a factor of roughly 2.5 and the ratio is flat, not climbing. So the truth of the
thesis is not in doubt numerically; only the proof is missing.

---

## 2. Equivalent formulations of the open bound

All four are interchangeable; different attacks prefer different ones.

**(a) Direct.** `F_k(y) <= (y^2-y)/6`.

**(b) Covering / hazard.** With `N(L) = #{m : m, ..., m+L-1 all blocked} = sum_g max(0, g-L)` and
`G(L) = #{gaps > L}`:

    G(L) = N(L) - N(L+1)              (proved, item 18)
    h(L) = G(L)/N(L) = 1 - N(L+1)/N(L)

and the bound follows from `N(L) <= P (1-d)^L`, equivalently `h(L) >= d` for every `L`. In the **`k`-frame this
is all that is needed.** In the adjacent frame the stronger `min_L h(L) = h(1)` was pursued; that is an
artefact of the finer grid (item 17a below) and is not required.

**(c) Second-order form.** `h(L)/d -> 1` for every `L` as the gear set grows, so `h/d` carries no margin.
The scale-free quantity is

    kappa(L) = (h(L)/d - 1)/d,     kappa(1) = 1/(1-d) exactly.

Adjacent frame: the bound needs `kappa(L) >= 1/(1-d)` for `L >= 2`. `k`-frame: it needs only `kappa(L) >= 0`.
Measured `k`-frame minimum of `kappa` settles near **0.68** (values `0.3889, 0.6167, 0.6749, 0.6824, 0.6840,
0.6805` at `y = 7..23`), so there is absolute room, not a knife edge.

**(d) Gear-recursion form.** `F(2,y) <= C * sum_{3<=q<=y} q` for a constant `C`. Since the odd primes sit
inside the odd numbers, `sum_{3<=q<=y} q <= (y^2+2y-3)/4` with **no prime counting** (note `pi(y) < y/2` is
false at `y = 3, 5, 7`), and the thesis then follows from any proved

    C <= 2(y^2-y)/(y^2+2y-3),   which is 1.8125 at y=29, 1.85 at y=37, 1.88 at y=47, rising to 2.

So **`C <= 1.8` suffices for all `y >= 29`**, with smaller `y` checked directly. Measured `C` peaks at
**1.354** at `y = 37` and is not monotone.

---

## 3. Established ground truths

Proved outright, or verified exhaustively at the stated scale. Numbering is referenced elsewhere in this
document.

**Machine mechanics.**

1. **Teeth.** Gear `q` threatens `k = +- 6^{-1} mod q`. With `tooth(q) = (q+1)/6` for `q = 5 mod 6` and
   `(q-1)/6` for `q = 1 mod 6`, the threatened set is `{tooth(q), q - tooth(q)}`; `6^{-1}` equals `tooth(q)` in
   the first case and `q - tooth(q)` in the second. The `+-` symmetry follows from `6 * 6^{-1} = 1`.
2. **The ±1 walk.** Every gear and every sub-machine steps the 6-cycle by exactly `+-1` per rotation:
   successive multiples of `q` step by `q mod 6`, every prime gear is `1` or `5 mod 6`, and the units mod 6 are
   closed under multiplication so composites inherit it. Verified for every sub-machine of up to four gears
   drawn from 5..59: `P mod 6` is always 1 or 5.
3. **Slot 0 is always exposed** - the complete-shield position - since `6^{-1}` is never `0`.
4. **Periodicity and symmetry.** The exposed set is a union of complete residue classes mod `P`, and the threat
   set is symmetric under `m -> -m`, so the pattern is symmetric about 0.
5. **Self-blocking law.** The lower tooth of gear `q` is the index of the pair containing `q`; twin gears share
   their lower tooth. Zero failures to `q = 2000`.
6. **Window identity.** `survivors(y, K) = T(6K+1) - T(y)` for `6K+1 <= y^2`, exact; verified `y = 11` to
   `1009`. This is what makes the certified window `(y, y^2]` precise.
7. **Tooth budget.** Per rotation each gear contributes exactly one shield, two killers and `q-3` misses.
   Universal.
8. **Minimal size law.** Gear `q` can be *forced* to block one of a set `S` only when `|S| >= (q+1)/2`, and the
   bound is attained - take `S` at residues `0, 2, 4, ..., q-1`, whose dominoes `{q-1,0}, {1,2}, ...` tile `Z_q`
   with one overlap; integer positions with those residues and all `= 0 mod 3` exist by CRT. Exhaustive to
   `q = 19`; the construction checked for all 45 odd primes below 200. **Exposure form: any `(q-1)/2` positions
   can be simultaneously exposed to gear `q`, whatever their spacing.** Gear 3's and gear 5's blocking laws are
   its first two cases. Minimal span with positions restricted to multiples of 3 grows like `1.9q`, by exact
   bitmask dynamic programme.
9. **Large gears force nothing new.** Within a search box of word length 16 and letters to 6, gears 29 through
   47 contribute zero minimal forbidden configurations beyond those from gears `<= 23`. Not a box artefact for
   29 and 31, whose own minimal configurations fit inside the box.
10. **gcd characterisation.** `m` is a twin slot iff `gcd(36m^2 - 1, primorial(floor(sqrt(6m+1)))) = 1`. The 36
    is `6^2`, the block period squared.
11. **Generating polynomial.** `p(x) = prod (q - 2 + 2x)`; the coefficient of `x^j` counts slots threatened by
    exactly `j` gears. `p(1) = P`, `p(0) = prod(q-2) = A`, `p(1-k) = prod(q-2k)`, `p(0) - p(-1) = ` run count.
12. **Factorised spectrum.** `Ehat(k) = prod_q ehat_q(k t_q mod q)` with `t_q = (P/q)^{-1} mod q`,
    `ehat_q(0) = (q-2)/q`, `ehat_q(c != 0) = -(2/q) cos(2 pi c u_q / q)`. Agrees with a direct FFT to
    `1.1e-16`. The `L1` norm grows about `2.06` per gear, so **spectral truncation is not available** - a single
    low-energy beat still blocks a slot.

**The gear-set recursion.**

13. **Merge transform (exact).** Let `M` have period `P` and exposed set `E`. Adding gear `q` gives period `Pq`
    and `E' = { x : x mod P in E, x mod q not in {0,1} }`. Walking `x` upward walks `E` around `q` times; lap
    `l` keeps the points whose `e mod q` avoids `{-lP mod q, (1-lP) mod q}`, and that pair shifts by `-P mod q`
    per lap. Since `gcd(P,q) = 1` the shift generates, so **every phase of gear `q` occurs in exactly one lap,
    and every exposed point of `M` is deleted in exactly 2 of the `q` laps.** Verified against direct
    construction on full gap *histograms*, not merely maxima.
14. **Chain condition (exact).** A new gap merges `k+1` consecutive old gaps when the `k` exposed points
    between them are deleted in one lap; those points lie in `{phi, phi+1} mod q`, so the partial sums of the
    interior gaps must all lie in `{0,1} mod q` or all in `{0,-1} mod q`. Since every point is deleted somewhere,
    **`F(M+q)` is determined by the old gap word and `q` alone.** Verified exact in 15 cases (gear sets to 19,
    added gears to 31).
15. **Deletion-spacing lemma (proved, tight).** Consecutive deleted points within a lap are at least `q-1`
    apart. *Proof:* they differ by `0` or `+-1 mod q`, and gaps are at least 3, so `delta = 0 mod q` gives
    `delta >= q`, `delta = 1 mod q` gives `delta >= q+1`, and `delta = -1 mod q` gives `delta >= q-1`. Attained
    at `q = 13` and `q = 19`.
16. **Saturation theorem (proved).** If `q - 1 > F(M)` then `F(M+q) = F2(M)`, where `F2` is the largest sum of
    two adjacent old gaps - because no interior gap can reach `q-1`, so only `k=1` chains exist. Checked over 48
    pairs, zero violations. Consequence: **above the threshold the increment is independent of `q`** (gears to 7
    plus any of 11..53 all give `F = 21`).

**The CRT collapse - the most important structural fact.**

17. **Every configuration is one translate.** The offset vector `(o_q)_{q<=y}` is, by CRT, exactly one residue
    `c mod P` with `c = o_q mod q`; gear `q` then blocks `i = c` or `c+1 mod q`, so the uncovered set is
    `{ i : i-c, i-c-1 both coprime to P }` - a translate of the single pattern `{ n : n, n+1 both coprime to P }`.
    The `P` offset vectors are precisely the `P` translations of one pattern. Hence `F(2,y) = 1 + ` the maximum
    gap of that pattern. **Verified:** the all-offsets-1 configuration (where `i` is uncovered exactly when
    `i-1` and `i-2` are both free of prime factors `<= y`) attains `F(2,y)` exactly at
    `y = 7, 11, 13, 17, 19, 23, 29` - ratio `1.000` in all seven cases, the last by segmented sieve over all
    `3.2 * 10^9` positions.
    - 17a. **Corollary.** The adjacent-frame `L = 1` has no `k`-space counterpart, so `h(1) = d/(1-d)` - the
      "free" case the covering route was built on - is a grid artefact. In `k`-space the minimum of `h/d` sits at
      `L = 2`, and `min_L h(L) = h(1)` is *stronger than the conjecture needs*.
    - 17b. **Corollary.** The exhaustive offset searches (`maxgap.rs`, `coverbound.rs`, `holegap.rs`) are sound
      but search a space isomorphic to `Z_P`: they explore translations, not designs. "Extremal configurations
      are efficient" is automatic - they are all the same configuration.

**Exact identities in the hazard formulation.**

18. `G(L) = N(L) - N(L+1)`, so `h(L) = 1 - N(L+1)/N(L)`.
19. `rho(1) = N(2)/N(1) = (1-2d)/(1-d)` and `h(1) = d/(1-d)`, **proved**: the third-order term is
    `prod (1 - 3/q)`, which vanishes because gear 3 contributes `1 - 3/3 = 0`. That is what makes `h(1)` free.
20. **Factorisation law.** With `w(S) = |{s-1, s : s in S}|` over the integers, `|W_q(S)| = w(S)` for every
    `q > span(S) + 1`. The threshold is `span+1`, not `span`: `min(S)-1` and `max(S)` differ by `span+1` and
    collide exactly at `q = span+1`. Hence
    `N(L) = sum_j c_j(L) prod_{q>L}(q-j)` with `c_j(L)` **independent of `y`**; verified by reassembly at
    `y = 13, 19, 23, 31`. Validity needs `y >= L+1`.
21. **The per-`j` recipe does scale.** `c_j(L)` is a sum over `2^L` subsets, but the head gears annihilate
    nearly all of them and depth-first pruning on the first fully covered gear visits only survivors:
    `2, 4, 10, 19, 61, 181, 289, 721, 2548` contributing terms at `L = 1, 3, 6, 9, 15, 21, 24, 30, 39` against
    `2^39 = 5.5e11`. Every visited subset contributes.
22. **Second-order expansion of `kappa`.** With `psi(delta) = v(delta)/d^2` the normalised pair weight,

        kappa(L) = L - sum_{delta <= L} psi(delta) + (small)
        psi(delta) = 3C * prod_{q | delta, q>=5} (q-2)/(q-4) * prod_{q | delta^2-1, q>=5} (q-3)/(q-4)
        C = prod_{q>=5} (1 - 4/(q-2)^2) = 0.396880415,    3C = 1.190641246

    Only `delta = 0 mod 3` contributes, because gear 3 divides one of `delta-1, delta, delta+1` and the two
    off-cases give the factor `1 - 3/3 = 0` - **the gear-3 law emerging term by term rather than imposed.**
    `psi(3) = 3C` exactly. Checked against measured `kappa` at `y = 100003` for thirteen `L`, agreeing to `0.03`
    at `L=3` rising to `0.42` at `L=63`. **`mean psi -> 3` exactly** (`2.6926, 2.9320, 2.9858, 2.9980, 2.99976`
    at `L = 63 .. 300000`), so `kappa = L - sum psi` is a bounded difference of two quantities each growing like
    `L`. Minimum of the leading-order `kappa` over all block starts to `L = 5 * 10^6` is **1.6343 at `L = 6`**,
    never below 1.
23. **Repulsion form.** `v(delta) = P(0 and delta both exposed)`, so `kappa(L) >= 1` says exactly: conditioning
    on an exposed position at 0 reduces the expected count of exposed positions in `(0, L]` by at least `d` -
    one position's worth of density. `v(1) = v(2) = 0` outright, since gear 3 blocks one of any two positions
    less than 3 apart.

**Proved special cases.** `h(1), h(3), h(6), h(9)` in the adjacent frame. Also `F_h = 0 mod 3` for all thirteen
known values, and exposed runs have length at most 2 in `k`-space (so the pattern is exactly isolated points and
dominoes, counts `prod(q-4)` and `prod(q-2) - 2 prod(q-4)`).

**New computed values produced here.** `F(2,37) = 264`, `F(2,41) = 273`, `F(2,43) = 309`; `F(2,29) = 129` and
`F(2,31) = 174` in halved coordinates; and `F2(2,y) = 21, 33, 48, 75, 93, 117, 165, 204, 270` for
`y = 7, 11, 13, 17, 19, 23, 29, 31, 37`. `F2_adjacent = 3 F2_k` verified for six gear sets.

**A closed-form next-twin method, which did not previously exist.** Per gear the distance to its next tooth is
`min((u_q - m) mod q, (-u_q - m) mod q)`; taking the minimum over gears gives the bite distance, and iterating
gives the next twin without walking. Verified to `k = 10^16` (192 s to 0.081 s against the naive walk). Also an
explicit closed form `J(m0) = sum_{J>=1} prod_{i=1..J} (1 - E(m0+i))` with
`E(m) = prod_q (1 - [q prime][q | 36m^2 - 1])`, verified at `m0 = 1, 5, 20, 50, 100, 200, 400, 1000`.

---

## 4. The computational anchor

One paragraph, as requested. Every structural claim in section 3 was computed two ways wherever two ways were
reachable, in Python (`research/*.py`) and Rust (`rust2/src/bin/*.rs`), and the independent routes agree
exactly: at `y = 23` the hazard table derived from the pattern matches the Rust enumeration of all 111,546,435
offset vectors to four decimals; at `y = 29` the hole-covering search returns `F2 = 165` against `3 * 55 = 165`
from the `k`-frame pattern, the pattern route being at its memory limit there; at `y = 29` a segmented sieve
over all `3.2 * 10^9` positions gives the single-pattern maximum gap as `129` against `F(2,29) = 129` from the
offset search, confirming the CRT collapse of item 17; the merge transform reproduces not just maxima but full
gap *histograms* against direct construction for four gear-set extensions; `chain_max` reproduces `F(M+q)`
exactly in 15 cases spanning gear sets to 19 and added gears to 31; the factorised spectrum agrees with a
direct FFT to `1.1e-16`; and `holegap.rs` reproduces `F2` exactly at all seven `y` where the pattern could be
built independently. Scales reached: `kappa` and the `psi` expansion to `y = 100003` (9592 gears) at 80-digit
precision, chosen because the integer form needs products with tens of thousands of digits while float64 loses
about eight digits to cancellation (`c_0` reaches `10^8` at `L = 24` while the answer is of order 1); the
leading-order `kappa` inequality over all 1.67 million block starts to `L = 5 * 10^6` by
smallest-prime-factor sieve; `F(2,y)` by exhaustive covering search to `y = 47`; the closed-form next-twin
method to `k = 10^16`; the minimal size law exhaustively to `q = 19` with its construction checked for all 45
odd primes below 200. The refutations in section 6 are the substantive output of the computation, not
incidental: each is a counterexample or a measured divergence at a stated scale, and several arrived as the
next data point after the claim looked settled.

---

## 5. The near-proof bottleneck

**Where it stalls, precisely.** Every route reduces to a constant, and the constant resists because *every
elementary bound compares how much the gears can cover against how much needs covering* - and capacity is
abundant, not scarce.

**5.1 The one-scale capacity barrier.** Gear `q` covers about `2L/q` positions of a run of length `L`, so a
capacity contradiction needs `sum 2/q < 1`:

    y                5      7     11     13     23     47    101
    sum 2/q (q>=5)  0.400  0.686  0.868  1.021  1.331  1.657  1.959
    sum 2/q (q>=3)  1.067  1.352  1.534  1.688  1.998  2.323  2.625
    overlap          6.7%  35.2%  53.4%  68.8%  99.8% 132.3% 162.5%

Capacity arguments work **only up to `y = 11`, and never again**. By `y = 47` the gears carry 132% more
covering capacity than a run needs.

**5.2 The two-scale repair also fails, and fails worse.** Split the gears at `z`. Those `<= z` can cover a run
of `F(z)-1` outright, so in a window of length `L` they leave at least `L/F(z) - 1` uncovered, and the gears in
`(z,y]` cover at most `2(L/q + 1)` each. A bound on `L` follows exactly when
`F(z) * 2 * sum_{z<q<=y} 1/q < 1`. Minimising over `z` (excluding the degenerate `z=y`):

    y             13     17     19     23     29     37     47
    best z         3      3      3      3      3      3      3
    product    3.064  3.417  3.733  3.994  4.201  4.556  4.970

The best threshold is always `z = 3`, where the condition reduces to `sum_{5<=q<=y} 1/q < 1/6` against an
actual `0.51` already at `y = 13`. Short by 3 to 5, and since `sum 1/q` diverges **the shortfall grows without
bound.**

**5.3 What the shortfall measures: clustering.** The gap between capacity and truth is the maximum gap against
the mean gap, `F(z) d_z`:

    z            3     5     7    11    13    17    19    23    29    31    37    43    47
    1/d_z     3.00  5.00  7.00  8.56 10.11 11.46 12.81 14.03 15.07 16.11 17.03 18.77 19.61
    F(z) d_z  1.00  1.20  2.14  2.45  3.26  4.71  5.86  7.27  8.56 10.80 15.51 16.46 18.06

Equal to the mean gap at `z = 3`, eighteen times it at `z = 47`, and growing like `y^2 / log^3 y` since
`F ~ C y^2/log y` against `1/d ~ log^2 y`. **The exposed set is far from uniform, and every bound that treats
it as uniform loses exactly this factor.** Note (item 17) that the difficulty is *not* a combinatorial fitting
or design question - there is no design freedom - so the clustering is the whole of it.

**5.4 The gear-recursion route stalls on an aggregate.** `F(M+q) - F(M)` splits exactly into
`(F2(M) - F(M)) + (F(M+q) - F2(M))`, both now computable:

    add q         11     13     17     19     23     29     31     37     41     43
    F(M)          15     21     33     54     75    102    129    174    264    273
    F2(M)         21     33     48     75     93    117    165    204    270      -
    increment      6     12     21     21     27     27     45     90      9     36
    F2 - F         6     12     15     21     18     15     36     30      6      -
    excess         0      0      6      0      9     12      9     60      3      -
    incr/q     0.545  0.923  1.235  1.105  1.174  0.931  1.452  2.432  0.220  0.837

**The gear-37 step reaches `2.432 q`, above the `1.8` the chain needs, so no per-step bound
`increment <= 1.8 q` is true and summing per-step bounds cannot give `C`.** Cumulative `C` is nevertheless
`1.354`, because the neighbours are `1.452` and `0.220`. The two pieces trade off against each other rather
than each staying small - which is why the aggregate is better behaved than either, and why the constant must
be argued as a property of the running total or the average increment.

**5.5 The chain length `k` is not boundable from gap structure.** The deleted points are consecutive exposed
points, so the span holds exactly `k` of them and the interior gaps are genuine gaps of `M`:
`(k-1)(q-1) <= sum h_j <= (k-1) F(M)`. These are compatible for **every** `k` as soon as `F(M) >= q-1`, which
is the regime that matters. Above the threshold they are incompatible for `k >= 2`, which is exactly the
saturation theorem - so the argument has that reach and no more. Bounding `k` needs the arithmetic of which
gaps fall within 1 of a multiple of `q`. Empirically `k <= 3` always: for gears to 19 with `q = 23`, runs of one
qualifying interior gap occur 11808 times, of two only 62 times, of three never.

**5.6 The remaining inequality, at its cleanest.** From item 22, the bound reduces at leading order to

    sum_{delta <= L, 3 | delta} psi(delta)  <=  L - 1     for every L >= 3,

verified over all 1.67 million block starts to `L = 5 * 10^6`, minimum margin `1.6343` at `L = 6`. What is
missing is (i) a rigorous error term for the second-order expansion, currently only measured, and (ii) control
of the average of `psi` with the error in the right direction. `mean psi -> 3` is what makes it delicate: the
inequality is a statement about the *second-order* term of that average.

---

## 6. Refuted claims - do not re-derive

Mathematical dead ends, each with the counterexample or measured divergence that killed it. Each looked right
on a small sample first; several died to their own next data point. These are worth checking but not worth
re-deriving.

1. **mex law** (first exposed slot = mex of `{u_q} union {q-u_q}`) - held to `y = 37`, failed at 41, stalled at
   20 against the truth 87.
2. **Uniform adjacency as the failure condition** - 55 of 56 failures for `{5,7,11}` were *not* uniformly
   adjacent.
3. **Gear-3 lemma** (gear 3 implies the bound for any separations) - counterexample `{3,5,7,11,13,17}` with
   separations `(1,3,3,3,3,3)`: `N(6) = 148485 > 147584.435`, ratio `1.006102`. Verified twice, the second time
   by direct enumeration of all 255255 tuples.
4. **Monotonicity of the margin in the gear set** - false; `R_1` peaks at `1.705697` near `q = 83` then falls to
   `1.679767`.
5. **Collapsing margin** - a normalisation artefact; the exact integer differences were *growing*
   (`51555900`, `350759640`).
6. **Log-concavity of `N`** - false, fails at `L = 3`.
7. **Tail-fraction bounds** from `N(L) <= N(1) - (L-1)G(L)` - too crude from `L = 6`.
8. **The universal bound `h >= 1/(F_h - L)`** - circular; it presupposes `F_h`, so it cannot appear in a chain
   deriving `F_h`. An apparent "complete proof for `{3,5,7}`" built on it was hollow.
9. **Finite automaton over the gap word**, in both forms - the antidictionary is infinite (minimal forbidden
   words still appearing at length 16 with the count rising), and even granting an automaton its letter
   statistics count which words *can* occur, a `y`-independent quantity, while `n_j` counts how often they *do*
   and scales with `P`. Measured side by side at `y = 13`: letter 1 at `0.127` in the pattern against `0.044` in
   the automaton, and the two move in opposite directions with `y`. Weighting it needs the `n_j` themselves.
10. **Per-gear conditional marginals fall under conditioning** - they *rise*, by up to 63% above `2/q`, failing
    at 36 of 53 values of `L` at `y = 17`. So gear exhaustion is **not** a per-gear effect, and the intuitive
    reading of the step form is wrong at that level.
11. **Weak negative association** `h(L) >= prod (1 - marginal_q)` - fails narrowly: once at `y = 17` (`L = 2`,
    short by 2.6%), twice each at `y = 11` and `13`.
12. **Per-gear usefulness for the step form** - the offsets that block position `L` are jointly below average at
    covering `[0,L)` only when `L mod q >= q/4`. Exact, 4525 pairs `(q,L)`, zero exceptions.
13. **The `3q` / `6q^2` reading of the tight block starts** - predicted `51, 57, 150` as tight; at `y = 71` `51`
    and `57` rank 19th and 21st of 21, the loosest, while `30` and `45` (in neither family) rank 8th and 10th,
    and at `y = 31` `L = 150` ranks 33rd of 58. Refuted from both directions by two independent routes.
14. **`F2` is attained at a maximal gap, and `F2 - F` is bounded** - both false. At `y = 29` (`k`-frame)
    `F2 = 55` comes from the adjacent pair `(30, 25)`, two large-but-not-maximal gaps, and `F2 - F` reads
    `2, 4, 5, 7, 6, 5, 12` in `k`-units, doubling at `y = 29` after appearing to settle. What *is* true and
    strengthens: maximal gaps are strongly isolated - at `y = 29` the only flanking pair is `(2,2)`.
15. **Any argument resting only on "gaps are multiples of 3 and at least 3"** - the multiset `{3,3,15}`
    satisfies both and violates the claim.
16. **Per-step increment bounds** - see 5.4.
17. **Covering-design freedom** - there is none; see item 17. Any argument that appeals to choosing offsets
    cleverly is appealing to a translation.

---

## 7. Promising pathways not yet formalised

**7.1 The aggregate increment.** The per-step route is closed (5.4) but the cumulative `C` is well behaved
because the two pieces of each increment trade off. Formalise the trade-off: when `F2(M) - F(M)` is large the
chain excess is small and vice versa. Concretely, both pieces are governed by the same object - the largest
gaps of `M` and their neighbourhoods - and the isolation of maximal gaps (item 14 above, the true half) says a
large `F` forces small neighbours, which caps `F2 - F`; while a large `F2 - F` means two medium gaps are
adjacent, which limits how much a chain can add on top. A bound on the *sum* of the two, uniform in `q`, gives
`C` and hence the thesis. Tools in place: `chain_max` for the second piece, `holegap.rs` for the first, and the
saturation theorem for the regime boundary.

**7.2 A uniform lower bound on `kappa`, in the `k`-frame.** The `k`-frame needs only `kappa(L) >= 0` against a
measured minimum near `0.68` - absolute room, no minimisation required. Via item 22 this is
`sum_{delta<=L, 3|delta} psi(delta) <= L - 1`, an inequality about an explicit divisor product. Two sub-problems:
a rigorous error term for the second-order expansion (the neglected triples-and-above, measured at `0.03` to
`0.42` across `L = 3..63` at `y = 100003`), and a second-order estimate for the average of `psi` in the right
direction. The repulsion form (item 23) is the mechanical statement of the same thing and may be the better
handle: `v(1) = v(2) = 0` outright gives a deficit of `2d^2`, and the question is whether the surviving
multiples of 3 over-compensate - they need `mean psi <= 3 - 3/L` and measure `3 - 104/L` at `L = 5e6`.

**7.3 A clustering-aware bound.** Section 5.3 identifies exactly what every failed route could not see: the
ratio `F d` of maximum gap to mean gap, growing like `y^2/log^3 y`. A bound that *uses* the clustering rather
than assuming uniformity is the only kind that can work. Two concrete openings. First, the exact
`N(L) = sum_j c_j(L) prod_{q>L}(q-j)` decomposition (item 20) separates the gear set from the run length
completely, with `c_j` computable to `L = 39` in 2548 terms (item 21) - so the `L`-dependence is available in
closed form and only its large-`L` behaviour is missing. Second, the CRT collapse (item 17) means the target is
the maximum gap of one explicit pattern, `{ n : n, n+1 coprime to P(y) }`, so any technique for pair-Jacobsthal
bounds applies directly; the programme's contribution would be the explicit constant, since the requirement
`F <= (y^2-y)/2` in adjacent units is a factor of `log y` weaker than what the measured `F ~ 0.68 y^2/log y`
delivers, and that margin *widens* with `y`.

---

## 8. Files

    docs/status.md                    consolidated status, capacity barrier, the CRT collapse
    docs/gear-recursion.md            merge transform, chain condition, saturation, frames
    docs/forbidden-configurations.md  minimal size law, factorisation, step form, kappa
    docs/covering-bound-route.md      the hazard route and its proved cases (adjacent frame)
    docs/twin-prime-program.md        the long-form record, sections 17-37
    docs/gear-at-infinity.md          the conceptual frame and what of it is proved
    docs/ideas-from-the-session.md    leads with named falsification tests, and the closed list

    research/gear_recursion.py        merge transform, chain_max, deletion spacing
    research/closed_hazard.py         c_j(L) by pruned enumeration, density and kappa at 80 digits
    research/kappa_expansion.py       the psi closed form and the second-order expansion
    research/minimal_forbidden.py     minimal size law, minimal span, factorisation law
    research/covering_decomposition.py  N(L) decomposition, step form
    research/negative_association.py  the two refuted association claims
    research/hazard.py                gap counts in closed form
    research/jump_distance.py         the closed-form next-twin method
    rust2/src/bin/maxgap.rs           F(2,y) by pruned covering search
    rust2/src/bin/holegap.rs          F2(2,y) by the same search with a mandatory hole
    rust2/src/bin/coverbound.rs       exhaustive offset enumeration, N(L) dump

One caution for the reader: lengths quoted in `covering-bound-route.md` and `forbidden-configurations.md` are
**adjacent-frame**; divide by 3 for `k`-space. The `y^2/6` in `gear-at-infinity.md` is `k`-space; the `y^2/2` in
`covering-bound-route.md` is the same requirement in adjacent units.
