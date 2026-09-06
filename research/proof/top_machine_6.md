# The zone of tranquillity (branch R4.b.vii)

> Law numbers in this document map to the project-wide register
> (`research/proof/law_register.md`): **L57-L66 = W76-W85**.  This document's L57-L59
> CLASH with `top_machine_5.md`'s; the register resolves the clash.  Future documents
> number from **W86**.

Parent: R4.b.iv, *The in-use next-opening bound* (`research/proof/top_machine_4.md`, laws
L46-L56). The observation that spawned this branch is the one thing that branch left with no
mechanism at all: below the largest gear the machine is a proved smooth-number object (L46-L48),
but **above** the largest gear it left a "small separate object", the record `A(q, N) = 24..419`,
described only by a first-hit fit and with the standing note *"OPEN: a bound above the zone"*.

The owner's direction, verbatim:

> "I think there is something close to Q - sqrt(Q), or the first gear whose square is greater
> than Q, which after which there is a zone of tranquillity; that zone should be completely
> knowable, and ideally we should be able to know if and where upper machine gap alignments
> always occur in that zone and how big they are."

*Gap alignment* here means what the owner means: several gears' dominoes lining up into a run of
struck pairs, i.e. a gap between consecutive open pairs.

Construction rule (R4, owner): the top machine on the raw line, on its own; no clutch, no bottom
machine beyond `q`-smoothness as a definition; no interpretation against the twin conjecture.

Numbering of new laws continues from **L57** (document 1 reached L21, document 2 L38, document 3
L45, document 4 L56, document 5 opened at L50 - cite by document).

---

## 0. The object

**The machine.** `G = {primes p : q < p <= Q}`, acting in pair coordinates on `[1, N]`. Gear `g`
strikes the pair `n` iff `g | n` or `g | n + 2`.

**Admissible.** Call a single integer `n` **admissible** (for `(q, Q)`) iff no gear divides it,
i.e. `n` has no prime factor in `(q, Q]`. The pair `n` is open iff `n` and `n + 2` are both
admissible. Write `A(q, Q)` for the admissible set.

**The smooth/rough split.** Every `n` factors uniquely as `n = s * r` with `s` the `q`-smooth
part (all prime factors `<= q`) and `r` the rough part (all prime factors `> q`). `n` is
admissible iff every prime factor of `r` exceeds `Q`.

**Vocabulary.** Unchanged: gear, tooth, pair, open, struck, run, gap, record, walk. New in this
branch: **admissible**; **stratum** (the part of the zone with `n/Q` in a dyadic band, which is
what limits the smooth cofactor); **family** (a pair `(s, s')` of smooth cofactors, see Z4);
**active family** (one whose smooth cofactor fits below the height in question).

---

## 1. Pre-registered predictions and scorecard

Written before any computation of this branch. Numbers are derived by hand here, or quoted from
`top_machine_4.md`, so that every computation below is a test and not a fit.

### Section 1. The edges (item 1 of the brief)

**Z1 (THE ZONE RULE - the manager's reading, predicted exact, and predicted to hold on all of
`[1, Q^2]` rather than only on `(Q, Q^2]`).** Predicted, as an identity with no exceptions:

        for every n <= Q^2 :   n admissible  <=>  n = s * P
        with s q-smooth and P either 1 or a single prime > Q.

Derivation by hand: `n` admissible means every prime factor of its rough part exceeds `Q`; two
such primes multiply to strictly more than `Q^2`, so at most one can occur, and it occurs to the
first power for the same reason. Hence the pair `n` is open iff both `n` and `n + 2` have that
form, for every `n <= Q^2 - 2`. Refuted by one admissible `n <= Q^2` not of that form, or one
`n <= Q^2` of that form that a gear strikes.

**Z2 (NO TRANSITION BAND; the handover is at `nextprime(Q)`, not at `Q` and not at
`Q - sqrt Q`).** Predicted: the rule of Z1 *contains* the rule of `[1, Q]` (L46) as the special
case `P = 1`, because for `n <= Q` a prime factor `> Q` cannot divide `n` at all. So the two
rules never disagree and there is **no transition band**. The first `n` at which the two rules
give different *sets* - the first admissible `n` with `P > 1` - is predicted to be exactly

        p_1 = nextprime(Q) ,

so the handover point is `p_1` and not `Q`; on `(Q, p_1)` the smooth rule still describes
everything. Predicted values: `p_1 = 101` at `Q = 100`, `1009` at `Q = 1000`, `10007` at
`Q = 10^4`. Predicted further: **nothing whatever happens at `Q - sqrt(Q)`** - the smooth rule
holds unchanged from 1 up to `p_1`, so no density, rule or striker change can be located there.
Refuted by any measured discontinuity at `Q - sqrt(Q)`.

**Z3 (WHERE THE RULE DIES, exactly, and how gradually).** Predicted: the rule of Z1 first fails
at `n = p_1^2` (an admissible number with two large prime factors), i.e. at a height
`p_1^2 - Q^2 = (p_1 - Q)(p_1 + Q) ~ 2 Q * (prime gap at Q)` above the zone's top, and
thereafter the density of rule-failures among admissible numbers grows like the density of
numbers with exactly two prime factors above `Q`. Predicted values of the first failure:
`10201` at `Q = 100` (201 above `Q^2`), `1018081` at `Q = 1000` (18,081 above), `100140049` at
`Q = 10^4` (140,049 above). Predicted: the general rule on `(Q^k, Q^{k+1}]` is "smooth times at
most `k` primes above `Q`", so `k = 1` - the zone - is the last level at which the rough part is
a *single* named object. Refuted by a failure below `p_1^2`.

**Z4 (THE STRATIFICATION - the zone is not homogeneous, and this is the mechanism the brief
asks for).** Predicted: at height `x` in the zone the smooth cofactor is bounded, `s <= x / p_1`,
so the admissible set at height `x` is

        A ∩ [x, x + h]  =  { s P : s q-smooth, s <= x/p_1, P prime > Q }  ∪  (q-smooth numbers) ,

and in the bottom stratum `(Q, 2Q]` the only cofactor available is `s = 1`: **there, admissible
= prime or `q`-smooth**, so an open pair in `(Q, 2Q]` is a pair of primes two apart, or a pair
one of whose members is `q`-smooth. Predicted consequence: the open-pair set of the zone is the
disjoint union, over **families** `(s, s')` of `q`-smooth cofactors with `gcd(s, s') | 2`, of the
solution sets of

        s' P' - s P = 2 ,      P, P' prime > Q ,

together with the finitely many pairs having a `q`-smooth member. Predicted number of families
available at `q = 5`, `Q = 10^4`: about 100 smooth values `s <= 10^4` (hand estimate
`(log Q^2)^3 / (6 log 2 log 3 log 5) ~ 106`), hence a few thousand ordered pairs with
`gcd | 2`. Refuted by an open pair in the zone not carrying such a label.

### Section 2. Complete knowledge (item 2)

**Z5 (EXACT ENUMERATION WITHOUT A SIEVE).** Predicted: generating `{s P}` from the `q`-smooth
list and the primes above `Q`, together with the `q`-smooth numbers, and intersecting with its
own shift by 2, reproduces the sieve's open-pair set in the zone **exactly, 0 mismatches**, at
every `(q, Q)` tested. Refuted by one mismatch.

**Z6 (THE EXACT COUNT).** Predicted exact closed form for the admissible count, needing only
`pi` and the smooth-counting function `Psi`, for every `X <= Q^2`:

        #(A ∩ [1, X])  =  Psi(X, q)  +  sum over q-smooth s <= X/p_1 of ( pi(X/s) - pi(Q) ) ,

and for the open pairs an exact three-part decomposition: the (smooth, smooth) pairs are the
finite Stormer list; the (smooth, large) and (large, smooth) pairs are computable from the
**finite smooth list alone** (test whether `(n ± 2)/smoothpart` is a prime above `Q`); only the
(large, large) part needs prime pairs, and it is the sum over families of Z4. Predicted: the
(large, large) part is over 99% of the zone's open pairs at every `(q, Q)` with `Q >= 1000`.
Refuted by a count mismatch against the sieve.

### Section 3. The alignments (item 3)

**Z7 (ALIGNMENTS ALWAYS OCCUR, AND THE PROVED LOWER BOUND IS A PRIME GAP).** Predicted theorem,
unconditional given the primes: let `p < p'` be consecutive primes in `(Q, 2Q]` with no
`q`-smooth number in `(p, p')`. Then every pair `n` with `p < n < p' - 1` is struck, so

        record in the zone  >=  max ( p' - p - 1 )   over such consecutive prime pairs.

Derivation: in `(Q, 2Q]` a smooth cofactor `s >= 2` would force `P = n/s <= Q`, so admissible
means prime or `q`-smooth (Z4); a pair `n` strictly inside a prime gap has `n` non-prime and
non-smooth, hence struck. So **gap alignments always occur in the zone, at every `(q, Q)`, and
their size is at least the largest prime gap just above `Q`.** Refuted by an open pair strictly
inside such a prime gap.

**Z8 (THE RECORD'S POSITION - the bottom stratum, and its endpoints are twin primes).**
Predicted: because the number of active families grows with height while the length available
grows too, the competition is real; predicted outcome, from the hand estimate
(bottom-stratum open-pair density `~ 1.32 / log^2 Q` against whole-zone density `~ C / log Q`),
that the **bottom stratum wins**: at every `(q, Q)` with `Q >= 1000` the zone record block starts
at a position `x` with

        1 < x / Q <= 10 ,

and both its endpoints (the open pair below the block and the open pair above it) are of family
`(1, 1)` - two primes two apart - with a small number of exceptions where a `q`-smooth member
takes one end. Predicted at `q = 5`, `Q = 10^4`: position `26,262`, i.e. `x/Q = 2.63`. Refuted by
a record above `10 Q`, or by a majority of record endpoints not of family `(1, 1)`.

**Z9 (NO UPPER BOUND, and exactly why).** Predicted: an upper bound on the zone record requires a
lower bound on the density of the family `(1, 1)` in a short interval above `Q`, so no
unconditional upper bound can be produced by this branch; what can be produced exactly is the
identity "record = the largest gap of a finite union of solution sets of `s'P' - sP = 2`".
Refuted by producing an unconditional upper bound.

### Section 4. How big (item 4)

**Z10 (CROSS-CHECK AGAINST `A(q, N)`, exact numbers pre-registered).** The zone `(Q, Q^2]` with
`Q = 10^4` **is** the above-zone region of `top_machine_4.md` at `N = 10^8`. Predicted, to the
unit, from that document's table 3.5:

| `q` | `Q` | zone record predicted | position predicted |
|---|---|---|---|
| 5 | 10,000 | 419 | 26,262 |
| 7 | 10,000 | 371 | 18,540 |
| 11 | 10,000 | 269 | 14,868 |
| 37 | 10,000 | 183 | 13,486 |
| 5 | 3,162 | 200 | - |
| 5 | 316 | 61 | 72,369 |

Refuted by any disagreement.

**Z11 (GROWTH).** Predicted: at fixed `q` the zone record **grows** with `Q`, and grows far more
slowly than the `[1, Q]` record `Q - s(q) - 2` (which is linear in `Q`): predicted zone records
61, 200, 419 at `Q = 316, 3162, 10^4` for `q = 5` against `[1, Q]` records 186, 3006, 9846 - a
ratio falling 3.0, 15.0, 23.5. Predicted: the zone record's position as a fraction of `Q^2`
tends to 0 like `1/Q`, while as a multiple of `Q` it stays bounded (Z8). Predicted growth shape:
governed by the gaps of the family `(1, 1)` near `Q`, hence of order `log^2 Q` times a slowly
growing factor; **no fit is offered until the mechanism is measured.**

**Z12 (THE WALK IN THE ZONE - the mex in a new guise).** Predicted exact closed form for the next
admissible number above `x` in the zone, with no scan of the line:

        nextadm(x) = min ( the least q-smooth number >= x ,
                           min over q-smooth s of s * nextprime( max(Q, ceil(x/s)) ) ) ,

the inner min running over the `O(polylog Q)` smooth numbers `s <= 2x/Q`; and the next open pair
from `x` is the least `n >= x` with `nextadm(n) = n` and `nextadm(n + 2) = n + 2`. Predicted 0
mismatches against the sieve over every position of the zone at the tested `(q, Q)`. Predicted
reading: **the mex over residues of L30 becomes a min over smooth scalings of the next-prime
function** - the walk in the zone is a prime-gap object exactly, not a residue object.

### Section 5. The owner's two numbers (item 5)

**Z13 (`Q - sqrt(Q)`: nothing there; and the reconciliation).** Predicted: measured as a position
on the line with `Q` the largest gear, `Q - sqrt(Q)` shows **no** change in the record's
mechanism, in the density of open pairs, or in which gears strike - because the smooth rule holds
unchanged from 1 to `p_1` (Z2). Predicted reconciliation instead: read with `Q` standing for the
range top `N`, "`Q - sqrt(Q)`" is the **length** `N - sqrt(N)` of the region above the largest
gear, and "the first gear whose square is greater than `Q`" is its **lower edge**
`sqrt(N) = the largest gear`; under that reading both of the owner's descriptions name the same
region, and it is the manager's zone `(Q, Q^2]`. Refuted by finding a real feature at
`Q - sqrt(Q)`.

**Z14 (`g0`, the first gear with `g0^2 > Q`).** Predicted: as a position on the line, `g0 ~
sqrt(Q)` sits deep inside the all-struck stretch `(s(q), Q]` and shows nothing; as a threshold in
the gear list it separates the gears that strike their own square inside `[1, Q]` from those that
do not, which is predicted to have **no effect** on the zone's rule, density, or record.
Refuted by a measured feature at `g0` or `g0^2`.

### Scorecard

| # | Prediction | Result |
|---|---|---|
| Z1 | the zone rule `n = s P`, exact on all of `[1, Q^2]`, 0 exceptions | |
| Z2 | no transition band; handover at `p_1 = nextprime(Q)`; nothing at `Q - sqrt Q` | |
| Z3 | rule dies first at `p_1^2` (10,201 / 1,018,081 / 100,140,049) | |
| Z4 | stratification `s <= x/p_1`; bottom stratum = prime or smooth; family labels | |
| Z5 | exact enumeration from the rule, 0 mismatches against the sieve | |
| Z6 | exact count `Psi + sum (pi(X/s) - pi(Q))`; (large,large) over 99% | |
| Z7 | alignments always occur; record `>=` largest prime gap above `Q` | |
| Z8 | record in the bottom stratum, `1 < x/Q <= 10`; endpoints of family `(1,1)` | |
| Z9 | no unconditional upper bound, and the exact reason | |
| Z10 | 419 @ 26,262; 371 @ 18,540; 269 @ 14,868; 183 @ 13,486; 200; 61 | |
| Z11 | record grows with `Q`; ratio to the `[1,Q]` record falls 3.0, 15.0, 23.5 | |
| Z12 | the walk = min over smooth `s` of `s * nextprime(x/s)`, 0 mismatches | |
| Z13 | nothing at `Q - sqrt(Q)`; the two descriptions name the same region | |
| Z14 | nothing at `g0` or `g0^2` | |

---

## 2. Setup as computed

Scripts in `research/topmachine/r6/`, outputs (untracked) in `.../results/`:

| script | what it computes |
|---|---|
| `common.py` | the machine two ways: `admissible_by_sieve` (strike every multiple of every gear) and `admissible_by_rule` (generate `s * P`); gap reports; prime gaps |
| `s1_edges.py` | the rule against the sieve on `[1, Q^2]`, the handover, the death above `Q^2`, the stratification, the density by stratum |
| `s2_count.py` | the rule-generated open-pair set against the sieve's, the exact count formula, the family census |
| `s3_gaps.py` | the record and its position, the record by stratum, the endpoint families, the gap spectrum, the proved prime-gap bound |
| `s4_walk.py` | the walk closed form against the sieve; the profile at `Q - sqrt(Q)`, `g0`, `g0^2` |
| `s5_size.py` | the record's endpoints over every machine; `g0^2` against `Q`; growth |

Everything below is an exact full scan, not a sample. Machines: `q in {5, 7, 11, 13, 17, 37}`,
`Q in {50, 100, 200, 316, 500, 1000, 2000, 3162, 10000}`, gear counts 8 to 1,224. Zones scanned
to `Q^2` in every case, and to `1.5 Q^2` (capped at `1.6 x 10^8`) where the rule's death above
`Q^2` was tested. Largest single scan: `q = 5..17`, `Q = 10^4`, the whole of `[1, 1.5 x 10^8]`.

---

## 3. Results

### 3.1 The rule, and where it starts and stops (45 machines, 0 exceptions)

For every `n <= Q^2`: `n` is admissible **iff** `n = s P` with `s` `q`-smooth and `P` either 1 or
a single prime above `Q`. Checked cell by cell against the gear sieve at 45 machines
(9 values of `Q` times 5 values of `q`), over ranges from 3,750 to 1.5 x 10^8 cells:
**0 exceptions.**

The two edges are exact, and they are not where the brief's candidates put them:

| `Q` | handover: first admissible `n` that is not `q`-smooth | `nextprime(Q)` | first rule failure above `Q^2` | `nextprime(Q)^2` |
|---|---|---|---|---|
| 100 | 101 | 101 | 10,201 | 10,201 |
| 200 | 211 | 211 | 44,521 | 44,521 |
| 500 | 503 | 503 | 253,009 | 253,009 |
| 1,000 | 1,009 | 1,009 | 1,018,081 | 1,018,081 |
| 3,162 | 3,163 | 3,163 | 10,004,569 | 10,004,569 |
| 10,000 | 10,007 | 10,007 | 100,140,049 | 100,140,049 |

**45 of 45** for the handover and **45 of 45** for the death, and both are independent of `q` -
they are properties of `Q` alone. So:

* there is **no transition band**. The `[1, Q]` rule "both members `q`-smooth" (L46) is the
  special case `P = 1` of the zone rule; the two never disagree. What changes at
  `p_1 = nextprime(Q)` is not the rule but its *content*: below `p_1` the rough part cannot be
  anything but 1, above it may be a prime.
* the rule dies at `p_1^2`, not at `Q^2`. Between `Q^2` and `p_1^2` - a stretch of
  `(p_1 - Q)(p_1 + Q)`, measured 201, 18,081 and 140,049 cells at `Q = 100, 1000, 10^4` - the
  rule is still exact although it is past the nominal top of the zone.

### 3.2 The stratification: the zone is layered, and the layer is `x / p_1`

At height `x` an admissible number `n = s P` has `P > Q`, so `s < x / Q`; sharply, `s` is the
largest `q`-smooth number not exceeding `x / p_1`. Measured at `q = 5`, `Q = 10^4`, taking the
largest smooth cofactor actually observed in each dyadic stratum against `floor(x_hi / p_1)`:

| stratum | `floor(x_hi/p_1)` | largest cofactor observed | admissible | open pairs | open density |
|---|---|---|---|---|---|
| (10,000, 20,000] | 1 | **1** | 1,070 | 140 | 0.0140 |
| (20,000, 40,000] | 3 | **3** | 3,368 | 465 | 0.0232 |
| (40,000, 80,000] | 7 | **6** | 8,979 | 1,777 | 0.0444 |
| (80,000, 160,000] | 15 | **15** | 20,800 | 4,941 | 0.0618 |
| (160,000, 320,000] | 31 | **30** | 44,767 | 11,589 | 0.0724 |
| (320,000, 640,000] | 63 | **60** | 91,991 | 24,677 | 0.0771 |
| (640,000, 1,280,000] | 127 | **125** | 183,378 | 49,343 | 0.0771 |
| (1,280,000, 2,560,000] | 255 | **250** | 359,539 | 95,031 | 0.0742 |
| (2,560,000, 5,120,000] | 511 | **500** | 697,287 | 178,417 | 0.0697 |
| (5,120,000, 10,240,000] | 1,023 | **1,000** | 1,343,933 | 331,118 | 0.0647 |
| (10,240,000, 20,480,000] | 2,046 | **2,025** | 2,582,175 | 611,100 | 0.0597 |
| (20,480,000, 40,960,000] | 4,093 | **4,050** | 4,955,895 | 1,125,727 | 0.0550 |
| (40,960,000, 81,920,000] | 8,186 | **8,100** | 9,510,339 | 2,071,306 | 0.0506 |
| (81,920,000, 10^8] | 9,993 | **9,720** | 4,097,114 | 870,870 | 0.0482 |

The observed maximum is in every stratum exactly the largest 5-smooth number below the bound
(1, 3, 6, 15, 30, 60, 125, 250, 500, 1000, 2025, 4050, 8100, 9720 against 1, 3, 7, 15, 31, 63,
127, 255, 511, 1023, 2046, 4093, 8186, 9993). **The bound is attained in every stratum.**

In the bottom stratum `(Q, 2Q]` the only cofactor available is `s = 1`. **There, admissible means
prime or `q`-smooth**, and an open pair means two primes two apart, or a pair with a `q`-smooth
member. That one line is the mechanism behind everything in 3.4.

The open-pair density is **not monotone**: it rises 5.5-fold from the bottom stratum to a peak
near `x = Q^{1.5}` and then falls back by a third to the top. Two forces: the number of usable
cofactors grows with height, but each family's own density falls like
`1/(log(x/s) log(x/s'))`, and at the top of the zone `log x = 2 log Q`. The pre-registered claim
that the density is monotone in height (Z4) is **refuted**; the predicted bottom-to-top ratio of
3 to 4 is held (measured 0.0482 / 0.0140 = **3.44**).

### 3.3 Complete knowledge: the enumeration, the count, the families

Generating `{s P}` from the smooth list and the primes above `Q` and intersecting it with its own
shift by 2 reproduces the sieve's open-pair set exactly. And the admissible count has a closed
form needing only `pi` and the smooth-counting function `Psi`:

        #(A ∩ [1, X])  =  Psi(X, q)  +  sum over q-smooth s <= X/p_1 of ( pi(X/s) - pi(Q) ) .

| `q` | `Q` | open pairs in `(Q, Q^2]` (sieve) | from the rule | mismatches | admissible (sieve) | the count formula |
|---|---|---|---|---|---|---|
| 5 | 1,000 | 100,058 | 100,058 | **0** | 326,079 | 326,079 |
| 17 | 1,000 | 225,407 | 225,407 | **0** | 477,991 | 477,991 |
| 5 | 3,162 | 723,261 | 723,261 | **0** | 2,772,102 | 2,772,102 |
| 17 | 3,162 | 1,717,717 | 1,717,717 | **0** | 4,172,041 | 4,172,041 |
| 5 | 10,000 | 5,376,501 | 5,376,501 | **0** | 23,900,810 | 23,900,810 |
| 7 | 10,000 | 7,765,397 | 7,765,397 | **0** | 28,334,815 | 28,334,815 |
| 11 | 10,000 | 9,665,288 | 9,665,288 | **0** | 31,459,021 | 31,459,021 |
| 13 | 10,000 | 11,564,004 | 11,564,004 | **0** | 34,295,997 | 34,295,997 |
| 17 | 10,000 | 13,194,719 | 13,194,719 | **0** | 36,565,989 | 36,565,989 |

13 machines, **0 mismatches** on the open pairs and **0** on the count.

**The families.** Every open pair `(n, n + 2)` of the zone carries a label `(s, s')`: the smooth
parts of its two members. Its rough parts are `P = n/s` and `P' = (n + 2)/s'`, each 1 or a prime
above `Q`, so the pair solves `s' P' - s P = 2`. At `q = 5`, `Q = 10^4` (5,376,501 open pairs):

| type | count | share |
|---|---|---|
| (large, large) | 5,376,052 | 0.999916 |
| (smooth, large) | 229 | 0.000043 |
| (large, smooth) | 220 | 0.000041 |
| (smooth, smooth) = Stormer | 0 | 0 |

1,510 distinct `(s, s')` families are used, out of 175 smooth values below `Q`;
**0 families with `gcd(s, s')` outside `{1, 2}`** and **0 pairs violating `s <= n / p_1`**. The
census is led by the small cofactors:

| rank | `(s, s')` | count | share of (large, large) |
|---|---|---|---|
| 1 | (1, 1) | **440,107** | 0.0819 |
| 2 | (3, 1) | 313,608 | 0.0583 |
| 3 | (1, 3) | 312,952 | 0.0582 |
| 4 | (1, 5) | 129,590 | 0.0241 |
| 5 | (5, 1) | 128,924 | 0.0240 |
| 6 | (2, 4) | 125,011 | 0.0233 |
| 7 | (4, 2) | 124,660 | 0.0232 |
| 8 | (1, 9) | 112,132 | 0.0209 |
| 9 | (9, 1) | 111,819 | 0.0208 |

The family `(1, 1)` is the set of `P, P + 2` both prime: its count 440,107 is exactly
`pi_2(10^8) - pi_2(10^4) = 440,312 - 205`. And the family counts do **not depend on `q`**: at
`q = 11`, `Q = 10^4` the same nine families carry 440,107, 313,608, 312,952, 129,590, 128,924,
125,011, 124,660, 112,132, 111,819 - identical numbers. Raising `q` adds families; it never
changes one. That is the exact sense in which the zone is completely knowable: its open-pair
count is a sum over a finite, explicitly listable index set of counts each of which is a property
of the integers and not of the machine.

### 3.4 The alignments: they always occur, and the proved bound is a prime gap

**They always occur.** Let `p < p'` be consecutive primes in `(Q, 2Q]` with no `q`-smooth number
strictly between them. Every pair `n` with `p < n < p' - 1` has `n` neither prime nor `q`-smooth,
hence (3.2) not admissible, hence struck. So the zone contains a run of at least `p' - p - 1`
struck pairs. Measured against the truth at 48 machines:

| `q` | `Q` | zone record | position | position/`Q` | position/`Q^2` | proved bound | record in `(Q, 2Q]` | `[1, Q]` record |
|---|---|---|---|---|---|---|---|---|
| 5 | 500 | **137** | 882 | 1.76 | 3.5e-3 | 13 | 137 | 360 |
| 5 | 1,000 | **119** | 1,488 | 1.49 | 1.5e-3 | 23 | 119 | 858 |
| 5 | 2,000 | **167** | 2,382 | 1.19 | 6.0e-4 | 27 | 167 | 1,864 |
| 5 | 3,162 | **200** | 9,881,102 | 3,124.95 | 9.9e-1 | 31 | 193 | 3,006 |
| 5 | 10,000 | **419** | 26,262 | 2.63 | 2.6e-4 | 51 | 371 | 9,846 |
| 7 | 10,000 | **371** | 18,540 | 1.85 | 1.9e-4 | 51 | 371 | 4,351 |
| 11 | 10,000 | **269** | 14,868 | 1.49 | 1.5e-4 | 51 | 269 | 3,721 |
| 13 | 10,000 | **227** | 14,640 | 1.46 | 1.5e-4 | 43 | 227 | 2,141 |
| 17 | 10,000 | **209** | 18,312 | 1.83 | 1.8e-4 | 29 | 209 | 2,141 |
| 37 | 10,000 | **183** | 13,486 | 1.35 | 1.3e-4 | 23 | 183 | 256 |
| 37 | 1,000 | **46** | 95,333 | 95.33 | 9.5e-2 | 5 | 37 | 30 |
| 13 | 316 | **41** | 2,037 | 6.45 | 2.0e-2 | 5 | 28 | 31 |

**Record `>=` the proved bound at 48 of 48**, ratio 3.22 to 24.00. The bound needs nothing but
the primes below `2Q` and the smooth list; no sieve estimate enters it.

**The one alignment that is always there, and where.** The largest struck block of the whole
machine begins at `s_k + 1`, one above the last `q`-smooth pair below `Q`, and runs *through the
zone's lower edge* until the first open pair above `Q`. Its part inside the zone is short,
because the first open pair above `Q` is close to `Q`: at `q = 5` it is 10,007 for `Q = 10^4`
(the twin `10007, 10009`), 3,167 for `Q = 3,162`, 2,025 for `Q = 2,000` and 1,019 for
`Q = 1,000` - 7, 5, 25 and 19 cells into the zone. So the great alignment belongs to `[1, Q]` and
only its tail reaches the zone; the zone's own record is a different block.

**Where the zone's own record sits: a U.** The record per dyadic stratum, `q = 5`, `Q = 10^4`:

| stratum | length | open pairs | record there | at | at/`Q` |
|---|---|---|---|---|---|
| (10,000, 20,000] | 10,000 | 140 | **371** | 18,540 | 1.85 |
| (20,000, 40,000] | 20,000 | 465 | **419** | 26,262 | 2.63 |
| (40,000, 80,000] | 40,000 | 1,777 | 187 | 71,068 | 7.11 |
| (80,000, 160,000] | 80,000 | 4,941 | 143 | 142,313 | 14.23 |
| (160,000, 320,000] | 160,000 | 11,589 | **116** | 219,195 | 21.92 |
| (320,000, 640,000] | 320,000 | 24,677 | 141 | 623,464 | 62.35 |
| (640,000, 1,280,000] | 640,000 | 49,343 | 141 | 851,660 | 85.17 |
| (1,280,000, 2,560,000] | 1,280,000 | 95,031 | 155 | 1,496,204 | 149.62 |
| (2,560,000, 5,120,000] | 2,560,000 | 178,417 | 203 | 3,725,460 | 372.55 |
| (5,120,000, 10,240,000] | 5,120,000 | 331,118 | 200 | 9,881,102 | 988.11 |
| (10,240,000, 20,480,000] | 10,240,000 | 611,100 | 242 | 12,134,950 | 1,213.49 |
| (20,480,000, 40,960,000] | 20,480,000 | 1,125,727 | 261 | 32,126,087 | 3,212.61 |
| (40,960,000, 81,920,000] | 40,960,000 | 2,071,306 | 270 | 78,382,014 | 7,838.20 |
| (81,920,000, 10^8] | 18,080,000 | 870,870 | **311** | 95,959,923 | 9,595.99 |

The profile is U-shaped: 371, 419, then down to 116 in the middle, then back up to 311 at the
top. **Two ends compete.** At the bottom the zone is family-starved - only `s = 1` is available,
so an open pair is two primes two apart - but the stratum is short; at the top there are 1,510
families but `log x = 2 log Q`, so the density is diluted, and the stratum is 1,800 times longer.
The middle strata - many families and a short logarithm - are the densest and carry the smallest
records.

The bottom wins at **28 of 48** machines, and the losses are close: at `q = 5`, `Q = 3,162` the
top of the zone wins 200 against 193, a margin of 3.5%. At the largest machines measured
(`Q = 10^4`, all six values of `q`) the record is at 1.35 to 2.63 `Q` in every case.

**Endpoints.** The record block is bounded below and above by open pairs, and their families say
what made the alignment. At `Q = 10^4`: `q = 5`, record 419 at 26,262, both endpoints `(1, 1)`;
`q = 7`, 371 at 18,540, both `(1, 1)`; `q = 11`, 269 at 14,868, both `(1, 1)`; `q = 17`, 209 at
18,312, both `(1, 1)`. Over all 42 machines of the endpoint census both endpoints are `(1, 1)` at
**13**; the pre-registered claim (Z8) that this is the rule is **refuted as stated**, and the
correct statement is conditional: *when the record is made in the bottom stratum it is a gap
between two pairs of primes two apart; when it is made at the top of the zone its endpoints carry
large cofactors.* At `q = 5`, `Q = 3,162` the record's endpoints are `(1, 3)` and `(2, 24)`; at
`q = 5`, `Q = 10^4` the 4th, 5th, 7th and 8th largest gaps are all high in the zone, with
endpoints `(2, 4)`, `(1, 3)`, `(1, 5)` and `(2, 12)`.

**The spectrum.** 49,433,381 gaps over the seven machines whose full spectrum was taken. Lengths
run from 1 to the record; length 1 is by far the commonest (21% to 44% of all gaps).
**A gap of exactly 3 struck pairs - distance 4 - occurs 0 times in all 49.4 million**, so L4
(`top_machine_1.md`) holds on the range and inside the zone, not only in a wheel. And the counts
of distance 3 and distance 5 (lengths 2 and 4) are nearly equal exactly when 7 is not a gear -
297,276 against 297,230 at `q = 7`, 432,223 against 432,253 at `q = 11`, but 146,337 against
186,660 at `q = 5`, where 7 *is* a gear - which is W1 (`top_machine_1.md`) surviving from the
wheel onto the range as an approximate equality.

### 3.5 The walk in the zone: the mex becomes a next-prime

The next admissible number above `x`, with no scan of the line:

        nextadm(x) = min ( the least q-smooth number >= x ,
                           min over q-smooth s of s * nextprime( max(Q, ceil(x/s)) ) )

| `q` | `Q` | `x` range tested | positions | mismatches | terms in the min |
|---|---|---|---|---|---|
| 5 | 1,000 | (1000, 500,000] | 499,000 | **0** | 85 + 1 |
| 11 | 1,000 | (1000, 500,000] | 499,000 | **0** | 191 + 1 |
| 17 | 1,000 | (1000, 500,000] | 499,000 | **0** | 286 + 1 |
| 5 | 3,162 | (3162, 4,999,122] | 4,995,960 | **0** | 125 + 1 |
| 11 | 3,162 | (3162, 4,999,122] | 4,995,960 | **0** | 325 + 1 |

**11,489,920 positions, 0 mismatches.** And the next *open pair*, by iterating it (take
`y = nextadm(x)`; accept if `nextadm(y + 2) = y + 2`, else restart from `y + 1`):

| `q` | `Q` | positions | mismatches | mean iterations | max |
|---|---|---|---|---|---|
| 5 | 1,000 | 200,000 | **0** | 2.97 | 23 |
| 11 | 1,000 | 200,000 | **0** | 2.31 | 24 |
| 5 | 3,162 | 200,000 | **0** | 3.50 | 40 |

600,000 walks, **0 mismatches**, at a cost of 2.3 to 3.5 iterations of an `O(polylog Q)`
minimum.

### 3.6 The owner's two numbers

`q = 5`, `Q = 10^4`: `Q - sqrt(Q) = 9,900`, `g0 = 101` (the first gear with `g0^2 > Q`),
`g0^2 = 10,201`, `p_1 = 10,007`, `s(5) = 160`.

| window | open pairs | density | mean strikers per struck pair | largest striker |
|---|---|---|---|---|
| [1, 200) | 13 | 0.0653 | 1.76 | 199 |
| [150, 350) | 1 | 0.0050 | 2.10 | 349 |
| [9,700, 9,900) | **0** | 0 | 3.04 | 9,901 |
| [9,900, 10,000) | **0** | 0 | 3.00 | 9,973 |
| [10,000, 10,007) | **0** | 0 | 2.71 | 5,003 |
| [10,007, 11,007) | 17 | 0.0170 | 2.87 | 5,503 |
| [20,000, 21,000) | 16 | 0.0160 | 2.90 | 6,997 |
| [100,000, 101,000) | 67 | 0.0670 | 3.09 | 9,181 |
| [5x10^7, 5x10^7 + 1000) | 45 | 0.0450 | 3.05 | 9,907 |

**Nothing is at `Q - sqrt(Q)`.** The open pairs of `[1, Q]` at `q = 5` are the thirteen Stormer
pairs 1, 2, 3, 4, 6, 8, 10, 16, 18, 25, 30, 48, 160; the whole of `(160, 10,007)` is a single
struck block of 9,846 pairs, and `Q - sqrt(Q) = 9,900`, `g0 = 101` and `Q` itself all lie
strictly inside it. No density, no rule and no striker statistic changes there, because there is
nothing there to change.

**`g0^2` is another matter.** `g0 = nextprime(sqrt Q)`, so `g0^2` is the first square of a gear
lying above `Q`, and it lands just above the zone's lower edge:

| `Q` | `g0` | `g0^2` | `(g0^2 - Q)/Q` | `(g0^2 - Q)/sqrt(Q)` |
|---|---|---|---|---|
| 1,000 | 37 | 1,369 | 0.3690 | 11.90 |
| 10,000 | 101 | 10,201 | 0.0201 | 2.01 |
| 31,623 | 179 | 32,041 | 0.0132 | 2.36 |
| 100,000 | 317 | 100,489 | 0.0049 | 1.55 |
| 316,228 | 563 | 316,969 | 0.0023 | 1.32 |
| 10^7 | 3,163 | 10,004,569 | 0.0005 | 1.44 |

`g0^2 = Q (1 + O(g0/sqrt Q))`, measured 2.0%, 1.3%, 0.5%, 0.2% and 0.05% above `Q`. **The
owner's second number is the zone's lower edge**, to a relative error that vanishes; it names `Q`
in the machine's own vocabulary - the first gear whose square has left the region in which every
gear strikes its own square.

### 3.7 How big, and how it grows

The zone record against `Q`, and against the two objects it should be compared with - the
`[1, Q]` record of L47 (linear in `Q`) and the largest prime gap just above `Q`:

| `q` | `Q` | zone record | at/`Q` | `(log Q)^2` | record/`(log Q)^2` | largest prime gap in `(Q, 2Q]` | record / that | `[1, Q]` record |
|---|---|---|---|---|---|---|---|---|
| 5 | 200 | 45 | 1.87 | 28.1 | 1.60 | 14 | 3.21 | 111 |
| 5 | 316 | 61 | 229.02 | 33.1 | 1.84 | 18 | 3.39 | 186 |
| 5 | 500 | 137 | 1.76 | 38.6 | 3.55 | 20 | 6.85 | 360 |
| 5 | 1,000 | 119 | 1.49 | 47.7 | 2.49 | 34 | 3.50 | 858 |
| 5 | 2,000 | 167 | 1.19 | 57.8 | 2.89 | 28 | 5.96 | 1,864 |
| 5 | 3,162 | 200 | 3,124.95 | 64.9 | 3.08 | 32 | 6.25 | 3,006 |
| 5 | 10,000 | 419 | 2.63 | 84.8 | 4.94 | 52 | 8.06 | 9,846 |
| 17 | 1,000 | 71 | 1.79 | 47.7 | 1.49 | 34 | 2.09 | 136 |
| 17 | 3,162 | 137 | 1.67 | 64.9 | 2.11 | 32 | 4.28 | 618 |
| 17 | 10,000 | 209 | 1.83 | 84.8 | 2.46 | 52 | 4.02 | 2,141 |
| 37 | 1,000 | 46 | 95.33 | 47.7 | 0.96 | 34 | 1.35 | 30 |
| 37 | 3,162 | 111 | 1.68 | 64.9 | 1.71 | 32 | 3.47 | 113 |
| 37 | 10,000 | 183 | 1.35 | 84.8 | 2.16 | 52 | 3.52 | 256 |

The zone record grows with `Q` at every `q` - 45, 61, 137, 119, 167, 200, 419 at `q = 5` - but
**not monotonically** (137 at `Q = 500` against 119 at `Q = 1,000`), because the record is the
maximum of two competing objects that each have their own scatter. It grows far more slowly than
the `[1, Q]` record, which is `Q - s_k - 2` and hence linear: at `q = 5`, `Q = 10^4` the two are
419 and 9,846, a ratio of **23.5**, up from 3.0 at `Q = 316`. As a fraction of the zone `Q^2` the
record's position is `2.6 x 10^-4` at `Q = 10^4` and falls like `1/Q`; as a multiple of `Q` it is
`1.35` to `2.63` at that `Q`. Nothing is fitted here: `record/(log Q)^2` climbs from 1.60 to 4.94
over the range measured, which is what a first-hit maximum over a set of density
`~ 1/(log Q)^2` in a window of length `~ Q` does (it carries an extra factor `log(Q/(log Q)^2)`),
but the sample is seven points and no law is claimed.

### 3.8 Cross-check against `A(q, N)` of `top_machine_4.md`

The zone `(Q, Q^2]` with `Q = 10^4` **is** the above-zone region of `top_machine_4.md` at
`N = 10^8`. Pre-registered to the unit before computing:

| `q` | `Q` | predicted (tm4) | measured here | position predicted | measured |
|---|---|---|---|---|---|
| 5 | 10,000 | 419 | **419** | 26,262 | **26,262** |
| 7 | 10,000 | 371 | **371** | 18,540 | **18,540** |
| 11 | 10,000 | 269 | **269** | 14,868 | **14,868** |
| 37 | 10,000 | 183 | **183** | 13,486 | **13,486** |
| 5 | 3,162 | 200 | **200** | - | 9,881,102 |
| 5 | 316 | 61 | **61** | 72,369 | **72,369** |
| 13 | 316 | 41 | **41** | 2,037 | **2,037** |
| 37 | 1,000 | 46 | **46** | 95,333 | **95,333** |

**8 of 8, value and position.** The `A(q, N)` of `top_machine_4.md` and the zone record of this
branch are the same object, and this branch has its mechanism.

---

## 4. Laws

Numbered from **L57**. `G = {primes in (q, Q]}`; `n` is *admissible* iff no gear divides it;
the pair `n` is open iff `n` and `n + 2` are admissible; `p_1 = nextprime(Q)`.

**L57 (THE ZONE RULE - the machine below `Q^2` is a smooth-times-one-prime machine).** For every
`n <= Q^2`,

        n admissible  <=>  n = s P  with s q-smooth and P either 1 or a prime > Q ,

and hence for every `n <= Q^2 - 2` the pair `n` is open iff both `n` and `n + 2` have that form.

*Proof.* Admissible means every prime factor of the rough part of `n` exceeds `Q`. Two such
primes have product `> Q^2 >= n`, so at most one occurs, and it occurs to the first power for the
same reason. Conversely `s P` has no prime factor in `(q, Q]`. QED.

*Evidence.* 45 machines, `q = 5..17`, `Q = 50..10^4`, every cell of `[1, Q^2]`: **0 exceptions.**
*This contains L46 (`top_machine_4.md`) as the case `P = 1` and extends the machine's exact
description from `[1, Q]` to `[1, Q^2]` - from a finite Diophantine list to a complete
generative rule.*

**L58 (THE TWO EDGES, exact and independent of `q`).** The first admissible `n` that is not
`q`-smooth is `p_1 = nextprime(Q)`; the first `n` at which L57 fails is `p_1^2`. There is **no
transition band**: the `[1, Q]` rule and the zone rule are the same rule, and they never
disagree anywhere.

*Proof.* Below `p_1` no prime exceeding `Q` divides any `n`, so `P = 1` is forced; `p_1` itself
is admissible and not smooth. L57 fails first where two primes above `Q` can multiply below the
tested height, i.e. at `p_1^2`. QED.

*Evidence.* 45 of 45 for each, at every `q`; the failure height is 201, 18,081 and 140,049 cells
above `Q^2` at `Q = 100, 1000, 10^4`.

**L59 (THE STRATIFICATION - the zone is layered by `x / p_1`, and the layer is attained).** At
height `x` the smooth cofactor of an admissible `n = s P` with `P > 1` satisfies
`s <= x / p_1`, and the largest `q`-smooth number below `x / p_1` is attained. In the bottom
stratum `(Q, 2Q]` therefore **admissible = prime or `q`-smooth**, and an open pair there is two
primes two apart or a pair with a `q`-smooth member.

*Proof.* `s = n/P < x/Q`, and `P >= p_1` gives `s <= x/p_1`; attainment is by taking `P = p_1`.
QED.

*Evidence.* Every dyadic stratum of `(10^4, 10^8]` at `q = 5`: observed maxima 1, 3, 6, 15, 30,
60, 125, 250, 500, 1000, 2025, 4050, 8100, 9720, each the largest 5-smooth number below the
bound; **0 pairs violating `s <= n/p_1`** among 5.4 million.

**L60 (THE FAMILY DECOMPOSITION - the zone's open pairs are a finite union of linear
twin-prime problems).** Every open pair of the zone carries the label `(s, s')` of the two smooth
parts, and

        the open pairs of (Q, Q^2] with both rough parts large
        =  the disjoint union over q-smooth pairs (s, s') with gcd(s, s') | 2, s, s' <= Q,
           of  { (sP, s'P') : P, P' prime > Q,  s'P' - sP = 2 } ,

together with the finitely many pairs having a `q`-smooth member (computable from the smooth
list alone) and the Stormer pairs. The number of families is at most `Psi(Q, q)^2`, a polylog in
`Q`.

*Proof.* The label is the unique smooth/rough split; `gcd(s, s')` divides `s'P' - sP = 2` because
`P, P'` exceed `Q` and are therefore coprime to every smooth number. QED.

*Evidence.* `q = 5`, `Q = 10^4`: 1,510 families, **0 with `gcd` outside `{1, 2}`**, the
(large, large) type carrying 99.9916% of 5,376,501 open pairs; the same at `q = 11` and at
`Q = 10^3`.

**L61 (THE EXACT COUNT, and the `q`-independence of each family).**

        #{admissible n <= X}  =  Psi(X, q) + sum over q-smooth s <= X/p_1 of ( pi(X/s) - pi(Q) )
                                                                            for every X <= Q^2,

and each family's open-pair count is a property of the integers, not of `q`: raising `q` adds
families and changes none. In particular the family `(1, 1)` contributes
`pi_2(Q^2) - pi_2(Q)` at every `q`.

*Evidence.* 13 machines, **0 mismatches** on the admissible count and on the open-pair set; the
nine leading families carry the identical counts 440,107 / 313,608 / 312,952 / 129,590 / 128,924
/ 125,011 / 124,660 / 112,132 / 111,819 at `q = 5` and at `q = 11`, and 440,107 is exactly
`pi_2(10^8) - pi_2(10^4)`.

**L62 (THE WALK IN THE ZONE - the mex becomes a next-prime).** For every `x` in the zone,

        nextadm(x) = min ( least q-smooth >= x ,
                           min over q-smooth s <= 2x/Q of s * nextprime( max(Q, ceil(x/s)) ) ) ,

an `O(Psi(x/Q, q))`-term minimum with no scan of the line; and the next open pair from `x` is
obtained by iterating it. **The mex over residues of L30 (`top_machine_3.md`) is replaced, in the
zone, by a minimum over smooth scalings of the next-prime function**: the walk in the zone is a
prime-gap object, not a residue object.

*Evidence.* 11,489,920 positions at five machines, **0 mismatches** for `nextadm`; 600,000 walks
at three machines, **0 mismatches** for the pair walk, mean 2.31 to 3.50 iterations, max 40.

**L63 (ALIGNMENTS ALWAYS OCCUR - a proved lower bound on the zone record, from prime gaps
alone).** Let `p < p'` be consecutive primes in `(Q, 2Q]` with no `q`-smooth number strictly
between them. Then every pair `n` with `p < n < p' - 1` is struck, so

        record of the zone  >=  max ( p' - p - 1 )   over such consecutive prime pairs.

*Proof.* By L59 an admissible `n` in `(Q, 2Q]` is prime or `q`-smooth; an `n` strictly inside a
prime gap is neither. QED.

*Evidence.* 48 machines, **0 exceptions**; the ratio of the truth to the bound is 3.22 to 24.00.
The bound uses only the primes below `2Q` and the finite smooth list - no sieve estimate.

**L64 (THE RECORD PROFILE IS A U, AND THE RECORD IS A COMPETITION).** The record per dyadic
stratum of the zone falls and then rises: at `q = 5`, `Q = 10^4` it is 371, 419, 187, 143, 116,
141, 141, 155, 203, 200, 242, 261, 270, 311 from the bottom stratum to the top. The bottom
stratum is family-starved (one cofactor, so the open pairs are two primes two apart) and short;
the top has every family but `log x = 2 log Q` and is 1,800 times longer; the middle, with many
families and a short logarithm, is the densest and carries the smallest records. The bottom
stratum makes the record at **28 of 48** machines and the losses are narrow (200 against 193 at
`q = 5`, `Q = 3,162`); at the largest machines measured the record sits at `1.35 Q` to `2.63 Q`,
i.e. at `1.3 x 10^-4` to `2.6 x 10^-4` of the zone.

*Evidence.* Stratum tables at seven machines; positions at 48.

**L65 (NO UPPER BOUND ON THE ZONE RECORD IS AVAILABLE, AND THE EXACT REASON).** An upper bound
on the record would need a lower bound on the density of open pairs in a short interval above
`Q`; by L59 the open pairs there are the solutions of `P' - P = 2` in primes above `Q`, so any
such bound is a lower bound for a linear twin-prime problem. The record of the zone is therefore
**exactly as hard to bound above as the gaps of the family `(1, 1)`**, while it is bounded below
by ordinary prime gaps (L63). Measured, the truth exceeds the prime-gap bound by 3.2 to 24.0.

*Reading.* The zone is completely knowable as a **rule** (L57), as an **enumeration** (L60), as a
**count** (L61) and as a **walk** (L62); it is not knowable as a **bound**, and the obstruction is
named exactly, not merely observed.

**L66 (L4 AND W1 SURVIVE ONTO THE RANGE).** In the zone, a gap of exactly 3 struck pairs -
distance 4 - never occurs: **0 in 49,433,381 gaps** over seven machines. And the counts of
distance 3 and distance 5 agree to within 0.02% exactly when 7 is not a gear (297,276 against
297,230 at `q = 7`; 432,223 against 432,253 at `q = 11`) and disagree by 28% when it is
(146,337 against 186,660 at `q = 5`).

*Evidence.* Seven full spectra; L4 (`top_machine_1.md`) was proved for a wheel, and the range
inherits it because the argument is local.

---

## 5. What is new

**The zone's rule, and the fact that it starts at 1 and not at `Q`.** For every `n <= Q^2` the
machine's admissible numbers are exactly `s P` with `s` `q`-smooth and `P` a prime above `Q` or
1. That single rule contains the `[1, Q]` smooth rule as its `P = 1` case, so **there is no
transition band anywhere**; what happens at `p_1 = nextprime(Q)` is that a second kind of
admissible number becomes available, and what happens at `p_1^2` is that a third does and the
rule dies. Both edges are exact, both are properties of `Q` alone, and both were verified at 45
machines with 0 exceptions.

**The stratification.** The zone is not a homogeneous region: at height `x` only the smooth
numbers below `x / p_1` can be cofactors, and that bound is attained in every stratum. So the
zone opens gradually, one cofactor at a time, from a bottom stratum in which the only open pairs
are two primes two apart to a top stratum with a thousand-odd cofactors. The open-pair density is
consequently **unimodal in height** - 0.0140, rising to 0.0771 near `Q^{1.5}`, falling to 0.0482
at `Q^2` - which is the first time the in-use machine has been shown to be non-uniform *inside*
the region above its gears.

**The family decomposition.** The zone's open pairs are the disjoint union, over a polylog-sized
explicit index set of smooth cofactor pairs `(s, s')` with `gcd | 2`, of the solution sets of
`s' P' - s P = 2` in primes above `Q`. Every one of 5.4 million open pairs carries such a label,
0 have a forbidden gcd, and each family's count is **independent of `q`** - raising `q` adds
families and changes none. This turns "the open pairs above the gear zone" from a statistic into
a finite list of named problems.

**The walk, in a second guise.** The mex of the residues (L30) is, in the zone, the minimum over
smooth `s` of `s * nextprime(x/s)`: 0 mismatches in 11.5 million positions. The next opening
above `Q` is decided by the primes above `Q`, exactly and constructively.

**Alignments always occur, and their size has a proved floor made of prime gaps.** Any prime gap
in `(Q, 2Q]` free of `q`-smooth numbers is a run of struck pairs, so the zone always contains an
alignment at least that long: 0 exceptions in 48 machines, with the truth 3.2 to 24.0 times the
floor.

**The record's profile is a U, and the record is a competition between the two ends of the
zone.** The bottom is thin but short, the top is thick but long, the middle is densest and
quietest. The bottom wins at 28 of 48 and by narrow margins when it loses.

**The `A(q, N)` of `top_machine_4.md` is this object.** Its eight pre-registered values and
positions (419 at 26,262; 371 at 18,540; 269 at 14,868; 183 at 13,486; 200; 61 at 72,369; 41 at
2,037; 46 at 95,333) are reproduced exactly, and it now has a mechanism.

**The owner's second number is right.** `g0^2`, the square of the first gear whose square exceeds
`Q`, is `Q(1 + O(g0/sqrt Q))` - measured 2.0%, 1.3%, 0.5%, 0.2%, 0.05% above `Q` as `Q` runs to
`10^7` - so "the first gear whose square is greater than `Q`" names the zone's lower edge in the
machine's own vocabulary.

**Prior art, in a line.** That the survivors of sieving `[1, z^2]` by the primes up to `z` are
the primes is Eratosthenes/Legendre; numbers of the form (smooth) x (one large prime) are the
*semismooth* numbers of factorisation practice (Bach-Peralta 1996), counted by
`Psi(x, y, z)`; the finiteness of the `q`-smooth pairs is Stormer 1897 / Lehmer 1964; the
families `s'P' - sP = 2` are the general linear prime pairs of the Hardy-Littlewood
conjectures, and unconditional gap bounds for the family `(1, 1)` are not available - which is
L65. Nothing asymptotic is claimed anywhere above; every number is an exact count.

---

## 6. Verdict

**The zone is `(Q, Q^2]`, and the manager's reading is the right definition - but its rule
begins at 1.** For every `n <= Q^2` a number is admissible iff it is `q`-smooth times a prime
above `Q` or times 1, and a pair is open iff both its members are. The lower edge of the zone,
in the machine's vocabulary, is `p_1 = nextprime(Q)` - the first admissible number that is not
smooth - and the owner's `g0^2` names it to within a vanishing relative error. `Q - sqrt(Q)` is
not an edge of anything: at `q = 5`, `Q = 10^4` it sits 9,740 cells inside a single struck block
that runs from 161 to 10,006. The upper edge is `p_1^2`, not `Q^2`.

**The zone is completely knowable in four senses and unknowable in one.** Knowable: the rule
(L57, 0 exceptions in 45 machines), the enumeration without a sieve (L60, 0 mismatches), the
count in closed form (L61, 0 mismatches at 13 machines), and the walk (L62, 0 mismatches in 11.5
million positions). Unknowable: the record has no available upper bound, and L65 says exactly why
- the bottom stratum of the zone is the family `(1, 1)`, two primes two apart, so bounding the
zone's record above is bounding the gaps of that family.

**Alignments always occur, and we know where and how big.** Every `q`-smooth-free prime gap in
`(Q, 2Q]` is an alignment (L63), so the floor is a prime gap, proved, 0 exceptions in 48
machines; the truth is 3.2 to 24.0 times that floor. The largest alignment of the whole machine
straddles the zone's lower edge - it starts one above the last smooth pair, far below `Q`, and
ends at the first open pair above `Q`, which is 5 to 25 cells into the zone. The zone's own
record sits at `1.35 Q` to `2.63 Q` at the largest machines measured, i.e. at a fraction
`~ 2/Q` of the zone, and it is 183 to 419 there against a `[1, Q]` record of 256 to 9,846. It
grows with `Q` but not monotonically, because it is the maximum of two competing objects at the
two ends of a U.

No interpretation against the twin conjecture is offered; the clutch does not appear.

---

## 7. Scorecard, filled

| # | Prediction | Result |
|---|---|---|
| Z1 | the zone rule, exact on all of `[1, Q^2]` | **held**, 45 machines, 0 exceptions (L57) |
| Z2 | no transition band; handover at `p_1`; nothing at `Q - sqrt Q` | **held**, 45 of 45, and nothing at `Q - sqrt Q` (L58, 3.6) |
| Z3 | rule dies first at `p_1^2` (10,201 / 1,018,081 / 100,140,049) | **held**, 45 of 45, all three values exact (L58) |
| Z4 | stratification `s <= x/p_1`; bottom stratum = prime or smooth; families | **held and sharpened** (the bound is attained in every stratum); the sub-prediction that density is monotone in height is **refuted** - it is unimodal, peaking near `Q^{1.5}`; the bottom-to-top ratio 3-4 is held at 3.44 |
| Z5 | exact enumeration from the rule, 0 mismatches | **held**, 13 machines, 0 (L60) |
| Z6 | exact count; (large, large) over 99% | **held**, 0 mismatches; 99.99% at `Q = 10^4` (L61) |
| Z7 | alignments always occur; record `>=` prime gap above `Q` | **held**, 48 of 48, ratio 3.22-24.00 (L63) |
| Z8 | record in the bottom stratum, `1 < x/Q <= 10`; endpoints family `(1,1)` | **half refuted**: the bottom stratum wins at 28 of 48 and `x/Q <= 10` at 35 of 48 - the failures are at small `Q` and at `q = 5`, `Q = 3,162`, where the *top* of the zone wins 200 against 193; both endpoints are `(1,1)` at 13 of 42, but at 4 of 6 machines with `Q = 10^4`. The correct statement is the conditional one (L64) |
| Z9 | no unconditional upper bound, and the exact reason | **held** (L65) |
| Z10 | 419 @ 26,262; 371 @ 18,540; 269 @ 14,868; 183 @ 13,486; 200; 61 @ 72,369 | **held, 8 of 8**, value and position (3.8) |
| Z11 | the record grows with `Q`; ratio to the `[1,Q]` record falls | **held in substance, refuted in detail**: the growth is not monotone (137 at `Q = 500` against 119 at `Q = 1,000`); the ratio to the `[1, Q]` record falls 3.0 -> 23.5 as predicted |
| Z12 | the walk = min over smooth `s` of `s * nextprime(x/s)`, 0 mismatches | **held**, 11,489,920 positions and 600,000 walks, 0 (L62) |
| Z13 | nothing at `Q - sqrt(Q)`; the two descriptions name the same region | **held**: `Q - sqrt Q` lies inside one struck block; both of the owner's descriptions land on the zone above the largest gear |
| Z14 | nothing at `g0` or `g0^2` | **refuted, in the owner's favour**: `g0` is inert, but `g0^2 = Q(1 + o(1))` **is** the zone's lower edge (2.0%, 1.3%, 0.5%, 0.2%, 0.05% above `Q`) |

---

## 8. What holds without exception

| statement | count | exceptions |
|---|---|---|
| the zone rule `n = s P` on `[1, Q^2]` | 45 machines, up to 1.5 x 10^8 cells each | **0** |
| handover at `nextprime(Q)` | 45 machines | **0** |
| the rule's first failure at `nextprime(Q)^2` | 45 machines | **0** |
| the stratification bound `s <= n/p_1` | 5.4 million open pairs at `q = 5`, `Q = 10^4` | **0** |
| every open pair carries a family `(s, s')` whose gcd divides 2 | 1,510 families, 5.4 million pairs | **0** |
| enumeration from the rule = the sieve's open pairs | 13 machines | **0** |
| the exact admissible count formula | 13 machines | **0** |
| the walk closed form `nextadm` | 11,489,920 positions | **0** |
| the pair walk by iteration | 600,000 walks | **0** |
| record `>=` the proved prime-gap bound | 48 machines | **0** |
| the cross-check against `A(q, N)` of tm4, value and position | 8 machines | **0** |
| no gap of distance 4 (L4) in the zone | 49,433,381 gaps | **0** |

---

## 9. Dead ends

- **`Q - sqrt(Q)` as an edge of the zone.** There is nothing there. At `q = 5`, `Q = 10^4` the
  whole of `(160, 10,007)` is one struck block of 9,846 pairs and `Q - sqrt(Q) = 9,900` is inside
  it; the rule, the density and the striker statistics are the same on both sides. What survived
  is the reading that makes the owner's phrase true: `N - sqrt(N)` is the *length* of the region
  above the largest gear, not an edge inside it.
- **A monotone density through the zone.** Refuted: the density is unimodal, 0.0140 at the bottom,
  0.0771 near `Q^{1.5}`, 0.0482 at the top. What survived is the mechanism for both halves - the
  cofactor count rises with height, each family's own density falls with `log x`.
- **"The record is always made at the bottom of the zone."** False at 20 of 48 machines. What
  survived is better: the record profile is a U with two competing ends, and the loser loses
  narrowly (200 against 193).
- **"The record block is always bounded by two twin primes."** True at 13 of 42, and at 4 of 6 at
  the largest `Q`. What survived is the conditional form: the endpoints' families say which end
  of the zone made the record - `(1, 1)` at the bottom, large cofactors at the top.
- **An upper bound on the zone record.** Not attempted beyond identifying the obstruction, which
  is exact (L65): the bottom stratum of the zone is the family `(1, 1)`. This is where the branch
  stops, by the rule against re-deriving; the object is named, not attacked.
