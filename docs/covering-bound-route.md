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

Adjacent positions are the sole source of positive correlation, and at `q = 3` their factor is
`1 - 3/3 = 0` exactly. **Gear 3 blocks `{o, o+1}` of its three residues, leaving only `o+2`, so of
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

## 7. What remains

Prove the lemma. Two properties would suffice, and both are standard in form:

1. For a single prime, the family of escape indicators over `[0, L)` behaves at least as well as
   independent when the adjacent pairs are excluded.
2. Independent gears compose, so the per-gear property lifts to the product.

The obstacle to quoting standard negative-association machinery directly is that the escape
indicators for one prime come from a *cyclic shift* of a fixed pattern, not from sampling without
replacement, and cyclic-shift families are not negatively associated in general. That is exactly
why the lemma fails for gear sets without 3, so any proof must use the `q = 3` adjacency
annihilation rather than a generic correlation inequality.

What is verified: the lemma holds with zero violations for every gear set containing 3 that has
been checked exhaustively, up to `P = 4.8` million and `L = 80`; it fails without 3; the mechanism
distinguishing the two cases is identified exactly; and if it holds, the twin prime conjecture
follows from it together with the computed values for `y <= 43`.
