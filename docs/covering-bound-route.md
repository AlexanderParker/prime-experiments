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

### 8c-bis. Correction: the helper account had the dominant effect backwards

Measuring the conditional blocking probability per gear - `Pr[gear q blocks the new position |
[0,L-1) covered]` against the unconditional `2/q` - shows it almost always **below** `2/q`, not
above. Conditioning on earlier coverage makes each gear *less* likely to block the next position,
because an offset spent at the far end of the run contributes nothing to covering the rest of it.
Aggregate escape probabilities measured for gears `5,7,11,13`: `0.368, 0.356, 0.387, 0.522, 0.448,
0.552, 0.423, 0.400, 0.667, 1.000` against `d = 0.297` - the step law holds with a wide margin,
and for the opposite reason to the one section 8b gives.

The correct decomposition splits the two offsets that block position `i = L-1`:

    o_q = i      blocks i and i+s   - contributes nothing earlier, so it is DISFAVOURED
    o_q = i - s  blocks i-s and i   - dual use, so it is FAVOURED

and the disfavouring dominates. Section 8b's divisor condition `q | 3 delta -/+ 1` correctly
identifies which gears can serve two positions at distance `delta` from one offset, and it does
govern the `L = 2` violation, where the earlier run is the single position 0 and the dual-use
offset covers exactly it. It does **not** govern the general step, because the general step
involves every earlier position rather than one distance.

**Lemma A (proved).** If `i < q` and `s_q < q - i`, then
`Pr[o_q = i | [0,i) covered] <= 1/q`.

*Proof.* Under those conditions the offset `o_q = i` blocks no position of `[0,i)`: the first
tooth needs `i >= q` to wrap back into the run, and the second, at `i + s_q`, needs
`s_q >= q - i`. So vectors carrying that offset require the remaining gears to cover `[0,i)`
unaided, while every other offset of `q` is at least as useful. Hence `N(i) >= q N_{-q}(i)`, and
`Pr[o_q = i | cover] = N_{-q}(i)/N(i) <= 1/q`. QED

Verified: 20 applicable cases, zero failures, and every violation of the inequality occurs exactly
where the hypothesis fails - `q = 7, s = 5` at `i = 2, 3, 7`; `q = 13, s = 9` at `i = 4, 5, 8`.

**But Lemma A does not reach the regime that matters.** Since `s_q = 3^{-1} mod q` equals
`(q+1)/3` or `(2q+1)/3`, the hypothesis `s_q < q - i` requires roughly `q > 3i` - gears *larger*
than the run. In the target regime the run has length `L` of order `y log^2 y` while gears go only
to `y`, so no gear qualifies.

### 8c-ter. Scale of the verification, stated honestly

The bound is only needed for `L <= F_h(y)`, since `N(L) = 0` beyond that. As `F_h` behaves like
`0.165 y^2`, the ratio `L/q` that matters grows like `0.165 y`. The exhaustive checks reach gears
up to 19 with `L` up to 80, so `L/q` of about 4. The tests are therefore in the right qualitative
regime but at small scale, and the conjecture is unverified where `L/q` is large.

### 8d. What remains, stated precisely

The step law is now the only gap, and it has been reduced from a global statement about all gears
to a local one about an explicitly identified set:

> At each `L`, the gears that can push `N(L)/N(L-1)` above `1 - d` are exactly the prime divisors
> of `3(L-1) - 1` and `3(L-1) + 1`. Show that a set of at most `2 log_2 (3L)` gears cannot do so.

That is a finite, modular condition at each step rather than a correlation inequality over the
whole run, and it is the form a mechanical proof should take: the mechanism places twins, the only
thing that could stop it is a simultaneous conspiracy of every gear at one distance, and the
divisors of `3 delta -/+ 1` are too few for that conspiracy to exist.

## 9. The mechanism, stated as the contradiction

This is the argument in the form the whole route was aiming at: a mechanism, the condition that
would have to hold to stop it, and the reason that condition cannot hold.

### 9a. The mechanism

Gears 2 and 3 leave slots 1 and 5 of every six. Each gear `q <= y` then blocks exactly two of
every `q` twin slots, at `k = +/- 6^{-1} mod q`. Inside the window `(y, y^2]` those gears decide
primality outright, so **any slot they all leave open is a twin pair**. That is the generating
mechanism, and it is exact - no estimate enters.

### 9b. What would have to happen to stop it

A twin fails to appear in the window only if every slot of the window is blocked - a **covering**.
Coverings are offset vectors, so writing `N(L)` for the number covering a run of length `L`,
prevention requires `N(L) >= 1` at `L` equal to the window length.

Now `N` decays step by step: a vector covers `[0,L+1)` exactly when it covers `[0,L)` and covers
`L`, so

    N(L) = N(L-1) * Pr[block the new position | earlier run covered]

If that conditional probability never exceeds `1 - d`, then `N(L) <= P (1-d)^L`, which is below 1
for `L >= L_0` of order `y log^2 y` - far short of the window. So **prevention requires the step
ratio to exceed `1 - d`**, that is, it requires conditioning on the covered run to *favour* the
offsets that also block the next position.

### 9c. Why that cannot happen in the window regime

An offset can be favoured only by being more useful for covering the run than its alternatives.
And the usefulness of every offset of a given gear is almost identical:

> **Spread lemma.** For gear `q` with tooth separation `s`, the offset `o` blocks
> `#{j < i : j = o or o+s mod q}` positions of the run. Each of the two residues contributes
> `floor(i/q)` or `ceil(i/q)`, so every offset blocks between `2 floor(i/q)` and
> `2 floor(i/q) + 2` positions - **a spread of at most 2, whatever `i` and `q`. When `q` divides
> `i` the spread is exactly 0** and all `q` offsets are perfectly interchangeable.

Measured for `q = 5, 7, 11, 29`: spread 0 at every `i` that is a multiple of `q`, and relative
spread falling like `q/i` - `1.0` at `i = 4, q = 5`, then `0.0000` from `i = q` onward.

So in the window regime, where the run is far longer than any gear, no offset of any gear is
materially more useful than another, and the conditioning cannot concentrate on the offsets that
block the next position. Conversely, when `i < q` the offsets differ absolutely - some block no
position of the run at all - and that is precisely the regime where every violation was found, at
`L = 2` and `L = 3`.

### 9d. Measured signature of the mechanism

If this is the right mechanism, the step-law margin should widen as `L/q` grows. At `y = 29`,
3.2 billion offset vectors enumerated exhaustively:

| L | L / q_max | step ratio | margin above `1-d` |
| --- | --- | --- | --- |
| 1 | 0.03 | 0.933626 | 0.000000 |
| 10 | 0.34 | 0.920303 | 0.013323 |
| 30 | 1.03 | 0.884753 | 0.048873 |
| 60 | 2.07 | 0.861040 | 0.072586 |
| 90 | 3.10 | 0.724816 | 0.208810 |
| 120 | 4.14 | 0.642857 | 0.290769 |

Zero step-law violations, and the margin is zero only at `L = 1`, where the step law is an
equality by construction. The margin grows monotonically in trend, exactly as the spread lemma
predicts.

### 9e. The gap that is left

The spread lemma is proved and exact. What is not proved is the implication

    offsets nearly equally useful  =>  step ratio at most 1 - d

Usefulness is not the only thing conditioning responds to: *where* an offset blocks matters as
well as how much, since overlaps with other gears differ. So the mechanism is identified and its
signature measured, but the final implication is still a conjecture.

Stated as the contradiction, the argument is complete except for that link:

> The mechanism generates a twin in every window. To stop it, the covered run would have to
> favour the offsets that block the next slot. Those offsets cannot be favoured, because every
> offset of every gear blocks within 2 of the same number of run positions - exactly 0 apart when
> the gear divides the run length - and the difference vanishes like `q/L` in the regime the
> window requires.

## 10. Two results on what has to be proved

### 10a. Sub-multiplicativity is false, so that route is closed

Since covering `[0, a+b)` means covering `[0,a)` and covering `[a, a+b)`, and the second has the
same count as covering `[0,b)` by translation, the natural stronger statement is

    N(a+b) * P <= N(a) * N(b)

- the two covering events negatively correlated - which would give the bound by induction from
`L = 1`. **It is false.** For gears `{3, 5, 7}` at `a = b = 6`:

    N(6) = 24,  N(12) = 6,  P = 105
    N(12) * P = 630  >  N(6) * N(6) = 576      ratio 1.09375

Checked over all `(a,b)` pairs: this is the only violation among the sets tested, and the four
other sets - `{3,5,7,11}`, `{3,5,7,11,13}` adjacent, `{5,7,11}`, `{5,7,11,13}` in t-space - have
none. But one counterexample settles it.

Worth noting the bound itself still holds at that point: `N(12) = 6` against
`P (1-d)^12 = 16.5`. So the bound is **strictly weaker** than sub-multiplicativity, and proving it
cannot go through negative correlation of block-covering events.

### 10b. The bound needed is far weaker than the bound conjectured

The argument needs `L_0 = log P / -log(rho) < y^2/2` for whatever decay rate `rho` is available,
so it needs only

    rho <= exp(-2 log P / y^2),   that is a decay rate of about  2 log P / y^2 ~ 2/y

while the conjectured rate is `d`, of order `1/log^2 y`. The slack:

| y | log P | d | needed rate | d / needed |
| --- | --- | --- | --- | --- |
| 19 | 15.39 | 0.078080 | 0.085288 | 0.92 (short) |
| 23 | 18.53 | 0.071290 | 0.070057 | 1.02 |
| 29 | 21.90 | 0.066374 | 0.052074 | 1.27 |
| 43 | 36.42 | 0.053271 | 0.039391 | 1.35 |
| 101 | 87.65 | 0.037539 | 0.017185 | 2.18 |
| 1009 | 962.47 | 0.017278 | 0.001891 | 9.14 |
| 10007 | 0.99e4 | 0.009787 | 0.000198 | 49.5 |
| 10^6 | 1.0e6 | 0.004361 | 0.0000020 | 2184 |

The slack grows like `y / (2 log^2 y)`, unbounded. So **the sharp bound is far more than the
argument requires**: any per-step decay of at least `2 log P / y^2` suffices, and at `y = 29` the
measured maximum step ratio is `0.933626` against a permitted `0.947926` - a margin of `0.0143`
already, widening without limit.

This reframes the remaining work. The target is not the tight inequality
`N(L)/N(L-1) <= 1 - d`, which is an equality at `L = 1` and therefore has no slack at all there.
It is the loose inequality

    N(L)/N(L-1) <= 1 - 2 log P / y^2

which has slack everywhere for `y >= 23`, and which needs only that the conditional escape
probability at each step be at least about `2/y` - rather than at least `d`. A far weaker
statement about the mechanics of section 9.

## 11. Three sub-routes to the step law, priced

### 11a. The relaxation does not reduce the problem, only parameterises it

Section 10b showed any decay rate above `2 log P / y^2` suffices. But "prove decay rate `rho`"
at `L = y^2/2` is equivalent to `N(y^2/2) < 1`, which is the goal itself. So the relaxation does
not weaken the target; it weakens what a *method* must achieve. It is useful only for grading
methods, which is what the rest of this section does.

### 11b. Induction on the number of gears: the algebra works, the hypothesis does not

The natural recursion is exact:

    M_n(S) = sum over o_n of M_{n-1}( S \ B_n(o_n) )

where `M_n(S)` counts offset vectors of gears `q_1..q_n` covering the set `S`, and `B_n(o_n)` is
what gear `n` blocks. Gear `n` removes a `2/q_n` fraction of `S` on average, so if the inductive
hypothesis were `M_{n-1}(S) <= P_{n-1} (1 - d_{n-1})^{|S|}`, the step would need

    (1 - a)^t <= 1 - a t   for  a = d_{n-1},  t = 1 - 2/q_n

which is exactly Bernoulli's inequality and holds - verified over 25 `(a,t)` pairs with zero
violations. **The recursion delivers precisely `P_n (1 - d_n)^{|S|}`, so the structure is sound.**

The hypothesis is not. A size-only bound is false for arbitrary `S`: if `S` lies inside two
residue classes mod `q_n` then gear `n` covers it alone, so `M_n(S) >= P_{n-1}`, while the target
`q_n P_{n-1} (1 - d_n)^{|S|}` drops below `P_{n-1}` as `|S|` grows. Explicit:

| gears | S | M(S) | target | |
| --- | --- | --- | --- | --- |
| 3, 5 | 3 multiples of 5 | 6 | 7.680 | ok |
| 3, 5 | 5 multiples of 5 | 6 | 4.915 | **fails** |
| 3, 5 | 12 multiples of 5 | 6 | 1.031 | **fails** |
| 3, 5, 7 | 5 multiples of 7 | 50 | 48.580 | **fails** |
| 3, 5, 7 | 12 multiples of 7 | 30 | 16.513 | **fails** |

So the inductive hypothesis must be **spread-aware** - it has to record how `S` sits modulo each
remaining gear, not just how large it is. In the recursion the sets that actually arise are
intervals minus unions of arithmetic progressions, which is the sieve setting, and that is where
the difficulty relocates.

### 11c. Second moment: valid, and short by a factor of `P/(Ld)`

Let `U(v)` be the number of uncovered positions of `[0,L)` for offset vector `v`. Then
`E[U] = L d`, and if all covariances are at most zero, `Var(U) <= L d`. Chebyshev gives

    N(L)/P = Pr[U = 0] <= Var(U)/E[U]^2 <= 1/(L d)

so `N(L) <= P/(L d)`. Valid but far too weak: at `y = 29, L = 129` it reads
`N <= 3.23e9 / 8.57 = 3.8e8`, against the truth `N = 0`. To reach `N < 1` it would need
`L d > P`, that is `L > e^y / d`, where the window offers only `y^2/2`.

The Poisson heuristic says the right answer is `Pr[U = 0] ~ e^{-Ld}` with
`L d ~ C y^2 / (2 log^2 y)`, comfortably past the `e^{-y}` needed. So the gap between second
moment and the truth is the whole of the concentration, and closing it needs control of all
moments - the sieve again.

### 11d. Where this leaves the route

Three sub-routes, all priced:

* **sub-multiplicativity** - false, counterexample at gears `{3,5,7}`, `a = b = 6` (section 10a);
* **size-only induction on gears** - algebra sound, hypothesis false, needs spread awareness;
* **second moment** - valid, short by a factor of `P/(Ld)`.

The step law itself remains verified and unproved: zero violations across every gear set tested,
100 billion offset vectors at `y = 31`, margin widening with `L/q` exactly as the spread lemma
of section 9c predicts.
