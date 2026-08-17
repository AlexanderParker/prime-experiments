# Adding one gear: the merge transform, the two frames, and a proof skeleton

Companion to `covering-bound-route.md` and `forbidden-configurations.md`. Script:
`research/gear_recursion.py`. This is the recursion on the gear set rather than on the run length -
idea 4 of `ideas-from-the-session.md` - built exactly and mechanically, with statistics used only at
the end to check the construction rather than to steer it.

## 1. The two frames are the same problem, scaled by 3

Two coordinate systems have been in use and they were not being distinguished carefully enough.

**The real frame (`k`-space).** A pair is `(6k-1, 6k+1)`. A prime `q >= 5` divides one member exactly
when `k = +- 6^{-1} mod q`, so gear `q` blocks two residues separated by `2 * 6^{-1}`. Gear 3 never acts:
`6k +- 1` is never divisible by 3.

**The adjacent frame.** Each odd prime `q`, gear 3 included, blocks the adjacent pair `{o, o+1} mod q`.
This is the frame of `maxgap.rs`, `coverbound.rs --adjacent`, `hazard.py` and everything in
`forbidden-configurations.md`.

They are related exactly. Gear 3 in the adjacent frame blocks one of any two adjacent positions, which
confines the exposed set to a single class mod 3, so every adjacent-frame gap is 3 times a real-frame
gap. Measured:

    y            7    11    13    17    19    23    29
    F_adjacent  15    21    33    54    75   102   129
    3 * F_k     15    21    33    54    75   102   129

**equal in all seven cases.** So results transfer with `L -> 3L`, and the hazard tables agree
term for term:

    min h/d       y = 7   11      13      17      19      23
    adjacent    1.1667  1.2162  1.2002  1.1787  1.1602  1.1455   at L = 6
    real frame  1.1667  1.2162  1.2002  1.1787  1.1602  1.1455   at L = 2

Two consequences that matter.

* The adjacent-frame case `L = 1` has **no counterpart in the real frame** - it is a length below one
  real unit. So `h(1) = d/(1-d)`, the "free" case that section 26c of `covering-bound-route.md` builds
  on, is an artefact of the finer grid. In the real frame the minimum of `h/d` sits at `L = 2`, and
  `h(1)` is *not* the minimum there: `kappa(1) = 0.748` against `min kappa = 0.68`.
* The target `min_L h(L) = h(1)` is therefore **stronger than the twin conjecture needs**. What is
  needed is only `h(L) >= d`, that is `kappa(L) >= 0`. In the real frame the measured minimum of
  `kappa` settles at about `0.68` - `0.3889, 0.6167, 0.6749, 0.6824, 0.6840, 0.6805` at
  `y = 7 .. 23` - so there is `0.68` of absolute room rather than a knife edge.

## 2. What the twin conjecture actually requires

Slot `k` is certified by gears up to `y` when `6k + 1 <= y^2`. Slot `k = 0` is always exposed, and gaps
of the exposed pattern are at most `F_k(y)`, so there is an exposed slot in `(y/6, y/6 + F_k(y)]`. For it
to lie inside the certified window,

> **`F_k(y) <= (y^2 - y)/6`.**

Measured, this holds with a factor of about 2.5 in hand, and the ratio is flat rather than climbing:

    y                7     11     13     17     19     23     29     31     37     41     43
    F_adjacent      15     21     33     54     75    102    129    174    264    273    309
    F_k              5      7     11     18     25     34     43     58     88     91    103
    (y^2 - y)/6    7.0   18.3   26.0   45.3   57.0   84.3  135.3  155.0  222.0  273.3  301.0
    ratio        0.714  0.382  0.423  0.397  0.439  0.403  0.318  0.374  0.396  0.333  0.342

`F(2,37) = 264`, `F(2,41) = 273` and `F(2,43) = 309` are new, computed with `rust2/src/bin/maxgap.rs`
started from the previous value, which is a valid lower bound since `F` is non-decreasing in `y`. Note the
increments are lumpy - `+90` at gear 37, then `+9` at gear 41, then `+36` at gear 43 - so single increments
say little; the running ratio is the stable quantity.

## 3. The merge transform, exactly

Let `M` have period `P` and exposed set `E`. Adding a gear `q` coprime to `P` gives period `Pq` and

    E' = { x in [0, Pq) : x mod P in E,  x mod q not in {0, 1} }.

Walking `x` upward walks `E` around `q` times. Lap `l` covers `[lP, (l+1)P)` and holds the points
`e + lP`; such a point survives when `(e + lP) mod q` avoids `{0, 1}`, that is when

    e mod q  avoids  { -lP mod q,  (1 - lP) mod q }.

So **each lap is `E` with two residue classes mod `q` deleted, and the deleted pair shifts by `-P mod q`
per lap.** Since `gcd(P, q) = 1` that shift generates, so across the `q` laps every phase of gear `q`
occurs exactly once. Adding a gear is `q` copies of the old pattern, each thinned at a different phase,
laid end to end - and each exposed point of `M` is deleted in exactly 2 of the `q` laps.

Verified against direct construction for four extensions - gears to 7 plus 11, to 11 plus 13, to 13 plus
17, to 17 plus 19 - matching not only the maximum gap but the **entire gap histogram**.

Deleting a point merges the two gaps either side of it; deleting `k` consecutive points merges `k+1`
gaps. Every new gap is a sum of consecutive old gaps, so the maximum gap grows only by merging.

## 4. The deletion-spacing lemma

> **Within one lap, consecutive deleted points are at least `q - 1` apart.**

*Proof.* Deleted points lie in two residue classes `{phi, phi+1}` mod `q`, so any two differ by `0` or
`+-1` mod `q`. Old gaps are at least 3, so two distinct exposed points differ by at least 3. A difference
`delta >= 3` with `delta = 0 mod q` gives `delta >= q`; with `delta = 1 mod q` gives `delta >= q + 1`,
since `delta = 1` is excluded; with `delta = -1 mod q` gives `delta >= q - 1`. So `delta >= q - 1`. **QED**

Tight: the bound is attained, at `q = 13` and `q = 19` the measured minimum spacing is exactly `q - 1`.
Measured minima against the bound: `12` vs `12`, `18` vs `16`, `18` vs `18`, `24` vs `22`.

Consequence: a stretch of length `G` contains at most `1 + G/(q-1)` deleted points, so long merges are
rare, and the growth of the maximum gap is governed by `q` rather than by the old maximum. That is the
mechanical reason an increment law in `q` is possible at all.

## 4a. The new maximum gap, exactly, from the old gap word

The deletion-spacing lemma generalises to the full chain condition, and that condition turns out to
determine `F(M + q)` completely.

A new gap merges `k+1` consecutive old gaps `g_i .. g_{i+k}` when the `k` exposed points between them are
all deleted in one lap. Those points lie in `{phi, phi+1} mod q`, so taking the first of them as origin,
the partial sums of the interior gaps must all fall in `{0, 1} mod q`, or all in `{0, -1} mod q` - the two
cases being whether the first deleted point sits on `phi` or `phi+1`. Since every exposed point is deleted
in exactly 2 of the `q` laps, every chain meeting that condition is realised in some lap. Hence

> **`F(M + q) = max over chains of the merged length`, computable from the old gap word and `q` alone.**

`k = 1` requires no interior gaps and is always available, so `F(M + q)` is at least the largest sum of two
adjacent old gaps. A single interior gap must be `0` or `+-1 mod q` and at least 3, hence at least `q - 1` -
the spacing lemma of section 4 as the `k = 2` case.

Verified against the transform in 15 cases spanning gear sets to 19 and added gears to 31 - **exact
agreement every time**:

    gears to   q     F(M)   chain_max   F(M+q)   increment   incr/q
    7          11      15          21       21           6    0.545
    7          17      15          21       21           6    0.353
    11         13      21          33       33          12    0.923
    11         17      21          48       48          27    1.588
    13         17      33          54       54          21    1.235
    17         19      54          75       75          21    1.105
    17         29      54          78       78          24    0.828
    19         23      75         102      102          27    1.174
    19         31      75         111      111          36    1.161

Note the largest ratios, `1.588` and `1.421`, come from *skipping* a gear - adding 17 or 19 to the gears
up to 11. Along the consecutive chain, which is what `F(y)` actually is, every ratio measured is at most
`1.29`.

The recursion needs only the gap word, not the pattern. It does not yet iterate on its own, because
computing `F` two gears ahead needs the new gap *word* rather than its maximum; `add_gear` supplies the
new histogram exactly but the ordering costs `A q` to materialise.

## 4b. The saturation theorem, and the anatomy of the maximising chain

> **If `q - 1 > F(M)` then `F(M + q) = F2(M)`,** where `F2(M)` is the largest sum of two adjacent old gaps.

*Proof.* A chain with `k >= 2` needs an interior gap that is `0` or `+-1 mod q` and at least 3, hence at
least `q - 1` by section 4. No gap of `M` reaches `q - 1`, so only `k = 1` chains exist, and their maximum
is `F2(M)`. **QED**

Checked over 48 pairs with zero violations. The consequence is worth stating plainly: **above the
threshold the increment does not depend on `q` at all.** For the gears up to 7, adding 11, 13, 17, 19, 23,
29, 37, 41 or 53 all give `F = 21`, an increment of 6 every time. So "increment `~ q`" from section 5 is
not a law about `q` - it is what happens when the added gear is small relative to `F(M)`, which is the
regime the consecutive chain is always in for large `y`.

Below the threshold the maximising chain is short, and its interior gaps are exactly what the condition
demands:

    gears to   q    F(M)   F2   F(M+q)   k   interior gaps   as multiples of q
    11         17     21    33      48   2   [18]            17 + 1
    11         19     21    33      48   2   [18]            19 - 1
    13         17     33    48      54   2   [33]            2*17 - 1
    13         23     33    48      54   2   [24]            23 + 1
    17         19     54    75      75   2   [39]            2*19 + 1
    17         29     54    75      78   2   [30]            29 + 1
    19         23     75    93     102   3   [45, 24]        2*23 - 1, 23 + 1
    19         31     75    93     111   3   [30, 63]        31 - 1, 2*31 + 1

Every interior gap is within 1 of a multiple of `q`, as required, and **`k` never exceeds 3** in any
maximum observed. The excess `F(M+q) - F2(M)` is small where it is nonzero: `15, 6, 6, 0, 3, 9, 18`.

### Why `k` cannot be bounded mechanically, and where that leaves things

The natural attempt fails, and it is worth recording why so it is not retried. The deleted points
`p_1 < ... < p_k` are *consecutive* exposed points, so the span `p_k - p_1` contains exactly `k` exposed
points and its interior gaps are genuine gaps of `M`. That gives two constraints:

    p_k - p_1 = sum h_j  >=  (k-1)(q-1)     each interior gap is at least q-1
    p_k - p_1 = sum h_j  <=  (k-1) F(M)     each interior gap is at most the maximum gap

Together, `(k-1)(q-1) <= (k-1) F(M)`, which is vacuous whenever `F(M) >= q - 1` - and that is precisely the
regime the consecutive chain lives in, since `F` outgrows `y`. The two inequalities are compatible for
**every** `k`. Above the threshold they are incompatible for `k >= 2`, which is the saturation theorem
again, so the argument has exactly the reach already obtained and no more.

So no bound on `k` follows from the gap structure alone - from "gaps are multiples of 3, at least 3, at most
`F`, interiors at least `q - 1`". Bounding `k` needs the *arithmetic* of which gaps fall within 1 of a
multiple of `q`, that is the joint behaviour of consecutive gaps modulo `q`. That is a counting question,
and it is the legitimate kind here: the mechanical construction - transform, chain condition, deletion
lemma, saturation theorem - is complete and verified first, and the estimate checks it rather than standing
in for it.

What the counting has to deliver is modest: each interior gap is forced to be at least `q - 1`, roughly
twice the mean gap `1/d`, so a chain of length `k` needs `k - 1` consecutive gaps that are all both unusually
large and pinned to within 1 of a multiple of `q`. Every maximum observed has `k <= 3`.

## 4c. Maximal gaps are isolated, and that is what keeps the growth polynomial

Since `F(M + q) >= F2(M)` always, `F2(M) - F(M)` is a lower bound on **every** increment. If it were a
fixed fraction of `F` the growth would be geometric in the number of gears, and `F` would blow past the
requirement. It is not:

    gears to     A         F     F2    F2-F   (F2-F)/F   gaps flanking a maximal gap
    7            15       15     21       6      0.400   (3,6) (6,3)
    11          135       21     33      12      0.571   (3,9) (9,3)
    13         1485       33     48      15      0.455   (6,6) (6,15) (15,6)
    17        22275       54     75      21      0.389   (6,6) (6,9) (6,15)
    19       378675       75     93      18      0.240   (6,6) (6,9) (6,15)
    23      7952175      102    117      15      0.147   (3,15) (9,9) (15,3)

**The gaps immediately either side of a maximal gap are always tiny** - drawn from `{3, 6, 9, 15}` in every
gear set measured, while `F` itself runs from 15 to 102. `F2 - F` stays inside `[6, 21]` across a sevenfold
growth in `F`, so the ratio falls from `0.40` to `0.147`.

That is the mechanism keeping the growth polynomial rather than geometric: **a maximal gap is locally
isolated.** A long blocked run consumes the gears in its neighbourhood, so what follows it is dense. It is
the gear-exhaustion idea - and note that the version refuted in section 7 of
`forbidden-configurations.md` was a claim about per-gear *conditional marginals*, which do rise. This is a
different claim, about a single extremal configuration, and it is what the data supports.

It also gives a consistency check on the increments: `F2(23) - F(23) = 15`, and the measured gear-29
increment is `27 >= 15`. Likewise `F(2,41) - F(2,37) = 9` forces `F2(M_37) - F(M_37) <= 9` - the neighbours
of the `264` gap are at most 9 wide.

**The sharpened target.** Both quantities are covering problems of the same kind:

    F(y)  = 1 + max length of a coverable run
    F2(y) = 1 + max length of a window coverable except for one interior position

`maxgap.rs` computes the first by exhaustive search at the leftmost uncovered position. A variant with a
mandatory hole computes the second, so `F2 - F` is directly measurable at the sizes where `F` is known. The
claim to test is that `F2(y) - F(y)` is bounded, or at worst `O(y)`.

## 5. The increment law, measured

    gears to    q added   F(M)   F(M+q)   increment   incr/q   sum of q   F/sum
    5             7          6      15           9    1.286         15   1.000
    7            11         15      21           6    0.545         26   0.808
    11           13         21      33          12    0.923         39   0.846
    13           17         33      54          21    1.235         56   0.964
    17           19         54      75          21    1.105         75   1.000
    19           23         75     102          27    1.174         98   1.041
    23           29        102     129          27    0.931        127   1.016

Individual increments straddle `q`, and the running total tracks `sum_{q <= y} q`:

    y             7     11     13     17     19     23     29     31     37     41     43
    F_adjacent   15     21     33     54     75    102    129    174    264    273    309
    sum of q     15     26     39     56     75     98    127    158    195    236    279
    C = F/sum  1.000  0.808  0.846  0.964  1.000  1.041  1.016  1.101  1.354  1.157  1.108

So `F_adjacent(y) ~ C sum_{3<=q<=y} q` and `F_k(y) ~ (C/3) sum q`, with `C` measured between `0.81` and
`1.354`. At `y = 29`, `sum q / 3 = 42.3` against the true `F_k = 43`. `C` rises to its peak at `y = 37` and
falls back to `1.108` by `y = 43`, so the peak is a spike rather than a trend.

**Correction to an earlier version of this section**, which recorded `C` in `[0.81, 1.10]` on the strength
of data up to `y = 31` plus `y = 47`. Filling in `y = 37` gives `C = 1.354`, above that range - the gear-37
increment is `+90` where gear 41's is `+9`. `C` is not monotone and its supremum is not established. Since
`sum q ~ y^2/(2 log y)`, this says `F_adjacent ~ 0.68 y^2 / log y` at these sizes.

## 6. The proof skeleton this gives

Chain the pieces:

1. **Mechanical, to prove:** `F_adjacent(y) <= C * sum_{3 <= q <= y} q` for an explicit `C`. Measured
   `C <= 1.10` through `y = 47`. This is the one open link, and section 4 is the tool for it.
2. **Elementary:** the odd primes are a subset of the odd numbers, and the odds from 3 to `y` sum to
   `((y+1)/2)^2 - 1`, so

       sum_{3 <= q <= y} q  <=  (y^2 + 2y - 3)/4.

   No prime counting is needed - not `pi(y) < y/2`, which is false for `y = 3, 5, 7` anyway.
3. **Conclude:** `F_k = F_adjacent/3 <= C (y^2 + 2y - 3)/12`, and the requirement is
   `(y^2 - y)/6 = 2(y^2 - y)/12`. So the chain closes as soon as

       C  <=  2 (y^2 - y) / (y^2 + 2y - 3),

   which is `1.8125` at `y = 29`, `1.85` at `y = 37`, `1.88` at `y = 47`, and rises to `2`. **A bound
   `C <= 1.8` suffices for every `y >= 29`**, with the finitely many smaller `y` already checked directly in
   section 2.

So the open link needs a crude constant, not a sharp one: the largest `C` measured is `1.354` at `y = 37`
against a threshold of `1.85`, a margin of `1.37`. That is narrower than the `1.6` claimed before `y = 37`
was filled in, and since `C` is not monotone its supremum is not established - **this is the weak point of
the skeleton, not the elementary step.**

Using the sharp `sum_{q <= y} q ~ y^2/(2 log y)` instead gives more room, since the ratio to the requirement
becomes `C/log y` and improves with `y`: it predicts `1.354/log 37 = 0.375` against the measured `0.396`.
Either way the constant has to be bounded, and that is what section 4b is about.

So the whole question reduces to step 1: **how far can the maximum gap grow when one gear is added.**
That is a statement about merges of consecutive gaps under deletion of two residue classes - entirely
mechanical, with the deletion-spacing lemma already in hand.

## 7. Status

Established here:

* the exact merge transform, verified against direct construction on full gap histograms;
* the deletion-spacing lemma, proved and tight;
* the chain condition, giving `F(M + q)` exactly from the old gap word and `q`, verified in 15 cases;
* the saturation theorem `q - 1 > F(M) => F(M + q) = F2(M)`, proved, checked over 48 pairs, and the
  correction it forces to the reading of section 5 - the increment is `q`-independent above the threshold;
* the anatomy of the maximising chain: `k <= 3` in every observed maximum, interior gaps always within 1 of
  a multiple of `q`, and by census `k <= 3` *everywhere* in the pattern, not merely at the maximum - for
  gears to 19 and `q = 23`, runs of one qualifying interior gap occur 11808 times, of two only 62 times, of
  three never;
* **maximal gaps are isolated**: the flanking gaps are always in `{3, 6, 9, 15}`, and `F2 - F` stays in
  `[6, 21]` while `F` grows from 15 to 102, so the ratio falls from `0.40` to `0.147`. This is what keeps
  the growth polynomial rather than geometric, and it sharpens the target to bounding `F2 - F`;
* `F_adjacent = 3 F_k` exactly, for seven gear sets, so the two frames are one problem;
* the real requirement `F_k(y) <= (y^2 - y)/6`, holding with a factor of 2.3 to 3, ratio falling;
* the real-frame minimum of `h/d` sits at `L = 2`, not `L = 1`, so `min_L h(L) = h(1)` is stronger than
  needed - what is needed is `kappa >= 0`, with measured room `0.68`;
* the increment law `F ~ C sum_{q <= y} q` with `C` measured in `[0.81, 1.354]`, and a skeleton in which
  **any proved `C <= 1.8` finishes the bound for `y >= 29`**, by an elementary step that needs no prime
  counting;
* three new values, `F(2,37) = 264`, `F(2,41) = 273` and `F(2,43) = 309`.

Corrected here: `C` was recorded as at most `1.10` from data that skipped `y = 37`; it is `1.354` there, so
the margin against the threshold is `1.37` rather than `1.6`, and `C` is not monotone.

Open: step 1 of section 6, a proved bound on the increment. Note what has changed about the shape of the
remaining work - it is no longer a knife-edge inequality needing exact minimisation, but a crude constant
with a factor of `1.6` of slack against the measurement. Section 4b reduces it further, to bounding the
chain length `k` and the interior gap sizes, both of which are constrained mechanically before any
estimate is made.

A caution for anything built on the earlier work: quantities named `F_h`, `L`, `n_j`, `kappa` in
`forbidden-configurations.md` and `covering-bound-route.md` are **adjacent-frame**. Divide lengths by 3
to read them in `k`-space. The `y^2/6` in `gear-at-infinity.md` is `k`-space; the `y^2/2` in
`covering-bound-route.md` is the same requirement in adjacent units.
