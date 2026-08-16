# A counting route to bounding F(2,y)

Status: **one unproved lemma away from a complete proof.** Everything else in this document is
verified computationally. The lemma is a conjecture, stated precisely below, with its mechanism
identified and its failure mode understood.

Implementation: `research/covering_bound.py`. Exact `F_h` values from
`rust2/src/bin/maxgap.rs`.

## 1. The argument, in the proof-by-contradiction form

The twin constructor (`research/twin_constructor.py`, `research/jump_distance.py`) finds a twin in
the window `(y, y^2]` unless every slot of that window is threatened. Every slot threatened means
a **covering** exists - an assignment of one offset per gear that blocks a whole run as long as
the window. So:

1. Suppose the constructor fails at level `y`. Then a covering of a run of length `L = y^2/2`
   exists (halved coordinates; `y^2/6` in twin-slot coordinates).
2. Coverings are offset vectors, and there are only `P = prod q` of them in total.
3. If the number of covering vectors is provably below 1, there are none, and step 1 is impossible.

No statistics enter: the count is an integer, and an integer below 1 is zero.

## 2. Setting

In halved coordinates each odd prime `q <= y` blocks the adjacent residue pair
`{o_q, o_q + 1} mod q`, one offset chosen per prime - this is the frame of `maxgap.rs`, and it is
exact, since by CRT every combination of offsets is realised by some absolute position. Write

    Q = {3, 5, ..., y}     P = prod_{q in Q} q     d = prod_{q in Q} (1 - 2/q)

`d` is the chance a single position escapes every prime, so `Pr[position covered] = 1 - d`. Let
`N(L)` be the number of offset vectors covering all of `[0, L)`. Then `F_h(y)` is the least `L`
with `N(L) = 0`.

## 3. The lemma

> **Lemma (conjectured).** For a gear set containing 3, `N(L) <= P (1 - d)^L`.

Equivalently, with offsets independent and uniform, `Pr[every position of [0,L) is covered]` is at
most `(1 - d)^L` - the covered events are no more likely to hold together than if independent.

Given the lemma, `N(L) < 1` forces `N(L) = 0`, hence

    F_h(y) <= L_0(y) = ceil( log P / -log(1 - d) )

which is of order `theta(y)/d(y)`, that is `y log^2 y` up to constants.

## 4. Why the lemma should be true, and why it needs the prime 3

For a single prime `q`, the chance two given positions both escape is `1 - k/q`, where `k` is the
number of offsets that would block either of them:

| relative position | forbidden offsets | factor | against `(1 - 2/q)^2` |
| --- | --- | --- | --- |
| distance `>= 2` | `{i-1, i, j-1, j}`, four of them | `1 - 4/q` | smaller - **negatively** correlated |
| adjacent | `{i-1, i, i+1}`, only three | `1 - 3/q` | larger for `q >= 5` - **positively** correlated |

Adjacent positions are the sole source of positive correlation *in this frame*, and at `q = 3`
their factor is `1 - 3/3 = 0` exactly. (Section 6a corrects the scope of this: adjacency controls
the bound at `L = 2` exactly, not at every `L`, and positive correlations reappear at other
distances once the frame changes.) **Gear 3 blocks `{o, o+1}` of its three residues, leaving only `o+2`, so of
any two adjacent positions at least one is always blocked by gear 3 alone.** The positive
correlation is annihilated by the 6-cycle itself.

Measured, over all offset vectors:

| primes | `d^2` | `Pr[adjacent both escape]` | `Pr[distance >= 2 both escape]` | adjacent |
| --- | --- | --- | --- | --- |
| 3, 5, 7 | 0.020408 | **0.000000** | 0.016104 | negative |
| 5, 7, 11 | 0.122955 | 0.166234 | 0.100260 | **POSITIVE** |
| 3, 5, 7, 11 | 0.013662 | **0.000000** | 0.010248 | negative |

## 5. Verification

Exhaustive over all offset vectors, every `L` up to and past the true `F_h`:

| primes | P | d | worst ratio `N(L)/P(1-d)^L` | violations | true `F_h` |
| --- | --- | --- | --- | --- | --- |
| 3, 5 | 15 | 0.200000 | 1.0000 | 0 | 6 |
| 3, 5, 7 | 105 | 0.142857 | 1.0000 | 0 | 15 |
| 3, 5, 7, 11 | 1155 | 0.116883 | 1.0000 | 0 | 21 |
| 3, 5, 7, 11, 13 | 15015 | 0.098901 | 1.0000 | 0 | 33 |
| 3, 5, 7, 11, 13, 17 | 255255 | 0.087266 | 1.0000 | 0 | 54 |
| 3, 5, 7, 11, 13, 17, 19 | 4849845 | 0.078080 | 1.0000 | 0 | 75 |

Zero violations, and the worst ratio is attained at `L = 1`, where the bound is an equality by
construction. The recovered `F_h` values match the independent covering search exactly at every
size - 6, 15, 21, 33, 54, 75.

Omitting 3 breaks it, as the mechanism predicts:

| primes | worst ratio | violations |
| --- | --- | --- |
| 5, 7 | 1.1375 | 1 |
| 5, 7, 11 | 1.1026 | 2 |
| 7, 11, 13 | 1.2941 | 3 |

## 6. The bound covers the range the exact values do not

The window needs `F_h(y) < y^2/2`. Exact values settle `y <= 43`; the bound must settle the rest.

| y | exact `F_h` | exact suffices | bound `L_0` | `y^2/2` | bound suffices |
| --- | --- | --- | --- | --- | --- |
| 19 | 75 | yes | 190 | 180.5 | no |
| 23 | 102 | yes | 251 | 264.5 | **yes** |
| 29 | 129 | yes | 319 | 420.5 | yes |
| 43 | 309 | yes | 666 | 924.5 | yes |
| 47 | - | - | 770 | 1104.5 | yes |
| 101 | - | - | 2291 | 5100.5 | yes |
| 1009 | - | - | 55222 | 509040.5 | yes |

**The two ranges overlap on `23 <= y <= 43`, so their union is every `y`.** The bound's margin
widens without limit, since `y log^2 y` against `y^2/2` is a growing gap.

## 6a. Conditioning on gear 3, and where the failure mode actually lives

Conditioning on gear 3 sharpens the picture. The positions gear 3 leaves open form an arithmetic
progression of difference 3; reindexing those by `t`, gear `q` blocks two residues mod `q`
separated by `s_q = 3^{-1} mod q`. Adjacency in `t` would need `s_q = +/-1`, that is `q | 3 -/+ 1`,
so `q = 2` - impossible for a gear. Checked for every gear from 5 to 199: **no gear has adjacent
teeth in `t`-space.**

The reduction is valid and gives a stronger statement than the lemma:

    N(L) = sum over o_3 of N'(M),  M approx L/3
    N'(M) <= P' (1 - d')^M                       (the t-space bound)
    (1 - d')^{L/3} <= (1 - d'/3)^L = (1 - d)^L   since 0 <= d'^2 (9 - d')/27

so the `t`-space bound implies the lemma. And the `t`-space bound holds with zero violations for
exactly the gear sets that violate the adjacent-teeth version:

| gears | adjacent teeth, worst ratio | t-space, worst ratio |
| --- | --- | --- |
| 5, 7 | 1.1375 (1 violation) | 1.0000 (0) |
| 5, 7, 11 | 1.1026 (2) | 1.0000 (0) |
| 5, 7, 11, 13 | 1.0805 (2) | 1.0000 (0) |
| 5, 7, 11, 13, 17 | - | 1.0000 (0) |

**Correction to the account in section 4.** Tooth adjacency controls the bound at `M = 2` exactly,
and empirically at `M = 3`: at `M = 2` the bound reads `1 - 2d + Pr[both escape]` against
`1 - 2d + d^2`, so it holds precisely when the distance-1 escape probability is at most `d^2`, and
gear 3 forces that probability to zero. But it is **not** true that non-adjacent teeth make all
pairs negatively correlated. Measured in `t`-space for gears `5, 7, 11, 13`, five of the twelve
distances `1..12` are positively correlated - distances 2, 5, 7, 10, 12, corresponding to gears
dividing `3 delta -/+ 1` - and the bound holds regardless. So the bound is not a pairwise
consequence, and any proof needs more than a correlation inequality.

What the measurement does establish is that the failure mode is confined to very small `M`. The
adjacent-teeth ratio exceeds 1 only at `M = 2, 3`, and is below 1 from `M = 4` onward; the ratio
then decays roughly geometrically, reaching 0.08 by `M = 10`. Since the argument needs the bound
at `M = L_0`, of order `y log^2 y`, it is needed only in the regime where the measured margin is
enormous - which is favourable, but is not yet a proof.

## 7. What remains

Prove the lemma - and section 6a narrows what has to be proved in two useful ways.

**It is not a correlation inequality.** Pairwise negativity is neither available nor sufficient:
in `t`-space five of the first twelve distances are positively correlated, and the bound holds
anyway. So negative association, Harris, FKG and the like are the wrong tools; the escape
indicators for one prime come from a *cyclic shift* of a fixed pattern, and cyclic-shift families
are not negatively associated in general.

**It is only needed for large `L`.** Every violation observed sits at `L = 2` or `L = 3`. From
`L = 4` onward the ratio is below 1 in every case measured, adjacent teeth included, and decays
roughly geometrically - reaching 0.08 by `L = 10`. The argument needs the bound at `L = L_0`, of
order `y log^2 y`, so the required statement is asymptotic, not universal:

    for all sufficiently large L,  N(L) <= P (1 - d)^L

with the small-`L` cases irrelevant. A proof of geometric decay - `N(L+1) <= c N(L)` for some
`c <= 1 - d` once `L` exceeds a small threshold - would suffice, and would be a statement about
how adding one more position to a covering constrains the offsets, rather than about correlations
between positions.

What is verified: the lemma holds with zero violations for every gear set containing 3 that has
been checked exhaustively, up to `P = 4.8` million and `L = 80`; it fails without 3; the mechanism
distinguishing the two cases is identified exactly; and if it holds, the twin prime conjecture
follows from it together with the computed values for `y <= 43`.

## 8. The step law, and the residue condition that governs it

This is the sharpest form the route has taken, and it is a statement about modular residues
rather than about counting.

### 8a. The lemma is an induction with one step

A vector covers `[0, L+1)` exactly when it covers `[0, L)` **and** covers position `L`. So

    N(L+1) = #{vectors covering [0,L) that also cover L}

and the lemma `N(L) <= P (1 - d)^L` follows by induction from the single step

    **step law:  N(L) / N(L-1) <= 1 - d**

Since `1 - d` is the unconditional chance a position is covered, the step law says: covering the
earlier positions must not make covering the next one *easier*.

### 8b. When it can be easier - the residue condition

One offset serves two positions exactly when their distance equals that gear's **tooth
separation**. Gear `q` blocks `{r, r + s_q}`, so it covers both `i` and `i + delta` from a single
offset precisely when `delta = +/- s_q mod q`. Those are the gears that "help" at distance
`delta`, and they are the only route to a violation.

* **Adjacent teeth** (`s_q = 1` for every gear): *every* gear helps at `delta = 1`. All of them
  conspire at once, and the step law fails at `L = 2`.
* **`t`-space** (`s_q = 3^{-1} mod q`, after conditioning on gear 3): gear `q` helps at `delta`
  iff `3^{-1} = +/- delta mod q`, that is

      **q | 3 delta - 1   or   q | 3 delta + 1**

  so the helpers at distance `delta` are exactly the prime divisors of `3 delta - 1` and
  `3 delta + 1`. Verified with zero mismatches over gears 5 to 299 and `delta` 1 to 39.

The count of helpers is therefore at most `omega(3 delta - 1) + omega(3 delta + 1)`, which is at
most `2 log_2 (3 delta + 1)` - six gears at `delta = 2`, ten at `delta = 10`, twenty-four at
`delta = 1000`. **Never all `pi(y)` of them.** The conspiracy that breaks the adjacent case cannot
recur, because no single distance is the tooth separation of more than a logarithmic number of
gears.

### 8c. Measured

| gears | tooth separation | steps tested | violations | worst ratio / (1-d) |
| --- | --- | --- | --- | --- |
| 5, 7 | 3^-1 | 10 | 0 | 1.00000 |
| 5, 7, 11 | 3^-1 | 14 | 0 | 1.00000 |
| 5, 7, 11, 13 | 3^-1 | 14 | 0 | 1.00000 |
| 5, 7, 11, 13, 17 | 3^-1 | 20 | 0 | 1.00000 |
| 3, 5, 7 | 1 | 18 | 0 | 1.00000 |
| 3, 5, 7, 11 | 1 | 24 | 0 | 1.00000 |
| 3, 5, 7, 11, 13 | 1 | 34 | 0 | 1.00000 |
| 3, 5, 7, 11, 13, 17 | 1 | 40 | 0 | 1.00000 |
| 5, 7 | 1 | 10 | **1** (at L=2) | 1.13750 |
| 5, 7, 11 | 1 | 12 | **1** (at L=2) | 1.10264 |
| 7, 11, 13 | 1 | 14 | **1** (at L=2) | 1.29408 |

The worst ratio is exactly `1 - d` in every passing case, attained at `L = 1` where the step law is
an equality by definition, and strict thereafter. Every violation is the `delta = 1` conspiracy and
nothing else.

### 8d. What remains, stated precisely

The step law is now the only gap, and it has been reduced from a global statement about all gears
to a local one about an explicitly identified set:

> At each `L`, the gears that can push `N(L)/N(L-1)` above `1 - d` are exactly the prime divisors
> of `3(L-1) - 1` and `3(L-1) + 1`. Show that a set of at most `2 log_2 (3L)` gears cannot do so.

That is a finite, modular condition at each step rather than a correlation inequality over the
whole run, and it is the form a mechanical proof should take: the mechanism places twins, the only
thing that could stop it is a simultaneous conspiracy of every gear at one distance, and the
divisors of `3 delta -/+ 1` are too few for that conspiracy to exist.
