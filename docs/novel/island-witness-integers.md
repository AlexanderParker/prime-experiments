# The island witness for every integer: the multiple-of-five law, the one-class witness, and the cover number

Round 40, branch R2.a.i.a.1 (`research/proof/island_witness.md`). Child of
`docs/novel/reachability-landscape.md`, whose island witness (N-R4) this document sharpens.

## 1. WHAT IT IS

Take any integer `q >= 5` that is not divisible by 2 or 3. Then `q^2 - 1` is divisible by 6, so
`q^2` sits at the top of a column `k_0 = (q^2 - 1)/6`; the column `k_0 + i` carries the pair
`q^2 + 6i - 2`, `q^2 + 6i`. Let the machine be every prime `5 <= g <= q`; gear `g` **strikes**
offset `i` when it divides one of the two members. Let `d = 2 x 6^{-1} (mod q)`, the distance from
`q^2`'s column to the next place where `q`'s own tooth would land.

Four of the offset classes modulo 35 - `i = 5, 10, 12, 17` - can be struck by neither gear 5 nor
gear 7, whatever `q` is; the parent calls them the **islands** for bound 7. The parent's finding
was: for every prime `q` between 1,489 and 19,997 at least one island below `d` is struck by no
gear at all. This document says four further things about that object.

**(a) The multiple-of-five law - where the primality of `q` enters, and that it enters only
there.** If a gear `g` divides `q` then `q^2` is `0` modulo `g`, so the two offsets it strikes are
the ones where a member vanishes: `i = 0` and `i = 2 x 6^{-1} (mod g)`. Neither `0` nor `-2` is a
*nonzero* square, so those are offsets the gear can never reach when `q` is coprime to it; whether
each is one of the gear's permanently unreachable classes is decided by `g mod 8` - 0, 1, 2, 1 of
the two, for `g = 1, 3, 5, 7 (mod 8)`. Gear 5 is `5 (mod 8)`, so **both** its classes are
unreachable ones, and they are exactly `i = 0, 2 (mod 5)` - which is where all four islands lie
(`5, 10, 12, 17` are `0, 0, 2, 2` mod 5). Hence:

> If `5 | q` then gear 5 strikes every island class and the witness **fails**, at every size,
> without exception (13,333 of 13,333 multiples of 5 coprime to 6 below 200,000).
>
> If `gcd(q, 5) = 1` the witness holds for every integer `q` coprime to 6 above 2,849 - prime or
> composite (52,574 integers, 0 exceptions), and above 1,649 if also `7 !| q` (45,338 integers,
> 0 exceptions).

So the statement is about the two quadratics `q^2 + 6i - 2` and `q^2 + 6i` together with the
single hypothesis that `q^2` is a *nonzero* square modulo 5. Primality of `q` is used nowhere
else: composites coprime to 30 satisfy the witness at exactly the primes' rate and stop failing at
the same place. The full list of failures coprime to 35 is finite and explicit:
`11, 17, 23, 29, 41, 53, 73, 113, 121, 137, 173, 197, 233, 247, 263, 341, 353, 461, 683, 1151,
1487, 1649` - the parent's 17 primes plus the four composites `121 = 11^2`, `247 = 13 x 19`,
`341 = 11 x 31`, `1649 = 17 x 97`.

**(b) The one-class witness.** Each island class *on its own* carries an unstruck offset for every
prime above its own threshold: `i = 12 (mod 35)` from `q = 5477`, `i = 5` from 7,109, `i = 10`
from 11,717, `i = 17` from 13,001 - 0 exceptions in every case up to 200,000. So the witness set
may be taken to be a single arithmetic progression of density `1/35` rather than four of density
`4/35`, and above `q = 13001` all four progressions work simultaneously.

**(c) The short-arc witness.** The unstruck island is not merely somewhere below `d`; for every
prime in `(20000, 200000]` one sits inside `[1, 0.152 d)`, and its absolute offset never exceeds
**2,392** anywhere in `1487 < q <= 200000`, against an arc `d` running to 133,331.

**(d) The cover number `K(d)`.** Let every gear `g > 7` be free to take any nonzero quadratic
residue `r` as its value of `q^2 (mod g)` - exactly the freedom a real `q` gives it - which places
its two strike classes at `(2 - r) 6^{-1}` and `-r 6^{-1}`, i.e. two classes modulo `g` at the
fixed separation `2 x 6^{-1}`. Let each gear be used at one phase. Define `K(d)` as the fewest
gears that can then strike **every** island below `d`. Exactly:

| `d` | 35 | 70 | 140 | 280 | 560 | 1,120 | 2,240 |
|---|---|---|---|---|---|---|---|
| islands | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
| **`K(d)`** | **3** | **4** | **6** | **9** | **14** | **20** | 21 .. 31 |
| counting requirement | 2 | 4 | 5 | 7 | 9 | 10 | 11 |

The counting requirement - how many gears it takes for the strike budget `sum 2/g` to reach 1 - is
bounded, at about a dozen gears whatever `d`. The actual cover number is not: it is already double
the counting requirement at `d = 1120`. Consequently a failure of the witness at arc `d` forces
`q` into a residue class modulo a product of at least `K(d)` gears, and the smallest such product
is `1.1 x 10^32` at `d = 1120`, where `q` itself is about `3 x 10^3`.

## 2. WHY IT MIGHT BE NOVEL

The underlying characters are classical: Gauss's second supplement fixes `chi_g(2)` and
`chi_g(-2)` by `g mod 8`, and the density of `q` avoiding a system of residue classes is a
standard sieve heuristic. What is not standard is the object. (a) identifies the exact and only
place where the primality hypothesis is used in a statement about consecutive values of two
quadratics near `q^2` - a gear that divides `q` does not lose its strikes, it *relocates* them
onto its own permanently unreachable classes - and turns that into a complete description of the
exceptional set. (d) separates two quantities that a density argument identifies: the number of
gears needed to *pay for* a cover (bounded) and the number needed to *build* one (growing),
because the two classes a gear may place sit at a separation fixed by the modulus and cannot be
tuned. Neither is a restatement of Brun/Selberg sieve bounds, of the Jacobsthal function, or of a
covering-system result: the moduli here are forced, not chosen, and the target set is fixed and
`q`-free.

Not a restatement of: the parent's bar-size closed form (`|Bar(g)| = (g + 1 - chi(2) - chi(-2))/4`),
the parent's N-R5 (large gears strike islands at exactly `2/g`, so counting through islands is
scale-free) - (d) is precisely the statement that N-R5 does not control what the adversary must
do; the twin-prime singular series (the free-island *count* tracks it and is reported as a rate
and stopped).

## 3. PROOF

**SCRIPT-VERIFIED (finite), exact - no sampling.** Scripts in `research/anchor235/r40/`:

* `iw_sweep.py` - every integer `q` coprime to 6 in `[5, 200000]` (66,666 values, 17,982 prime):
  islands below `d` and free islands at `B = 7, 11, 13`. Reproduces the parent's counts exactly on
  the overlap (17 prime failures at `B = 7`, largest 1,487; 232 at `B = 13` with largest 18,839
  inside `q <= 20000`).
* `iw_slack.py` - the divisor rule against `g mod 8`, exhaustively over every gear `5..2000` and
  every offset class: **0 exceptions**; and the per-gear island strike and sole-strike counts over
  every prime `q <= 12000`.
* `iw_class.py` - the per-class and short-arc witnesses over every prime `q <= 200000`.
* `iw_failures.py`, `iw_cover.py` - the failures taken apart, with exact minimum covers by integer
  programming (HiGHS via `scipy.optimize.milp`, proved optimal).
* `iw_adv.py` - `K(d)`. The candidate enumeration is complete, which is what makes the ILP exact:
  writing the two strike classes as `a`, `b` with `6(a - b) = 2 (mod g)`, a covered set of size
  `>= 3` needs two islands in one class so `g < d`; a covered set of size 2 across the two classes
  needs `g | 3(i - j) - 1` so `g <= 3d`; a covered set of size 1 is available for every island at
  infinitely many gears above `3d`. Enumerating every gear `11 <= g <= 3d + 2` at every
  nonzero-QR phase plus one generic singleton per island therefore enumerates every set the
  adversary can play, and the ILP over that list with a once-per-gear constraint is `K(d)`,
  certified optimal by HiGHS at `d <= 1120`.

The one-line proofs behind the measured statements: the divisor rule is `q^2 = 0 (mod g)`
substituted into the tooth rule, and "barred" is then `chi_g(2) = -1`, `chi_g(-2) = -1`; the
multiple-of-five law is that rule at `g = 5` (which is `5 mod 8`) together with
`{5, 10, 12, 17} = {0, 2} (mod 5)`. Those two are **proved**, not merely measured. Everything
about the *existence* of a free island above a threshold is MEASURED with the exception counts
stated.

## 4. IMPLICATIONS

Inside the project: the interaction that branches R2.a.i and R2.a.i.a both stop at - that the
`2 pi(q)` progressions do not cover the arc `[1, d)` - now has a target that is smaller in three
independent ways (one residue class instead of four, an arc of `d/6` instead of `d`, and a
hypothesis of `gcd(q, 5) = 1` instead of primality), and one quantity attached to it that grows
where every counting quantity in this line has stalled. The cover number is the first thing found
under R2 that increases with scale while the counting margin does not, so it is where a proof of
the witness would have to live.

Outside: the multiple-of-five law is a clean example of a statement about two quadratic
polynomials that is true for all but finitely many integer arguments coprime to a single small
prime and false for every multiple of it - with the exceptional set completely determined by one
Legendre symbol.

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

Twin primes via the project's window formulation (`F(y) < y^2/6`, Ziller-Morack Conjecture 6 at
the real teeth): the witness is a *local* substitute for "the walk from `q^2` ends before the top
gear's next tooth", and (b), (c) make that local statement much cheaper to aim at. Jacobsthal-type
covering: `K(d)` is a covering-cost question with forced moduli and forced class separation, a
constrained relative of the covering systems used in Erdos-Rankin constructions.

## 6. PRIOR-ART CHECK

**NOT YET CHECKED** (no web access in this lane). Searches to run: "quadratic residue" +
"consecutive values" + "q^2 + 6i" ; "covering system" + "two residue classes per prime" + "fixed
separation" ; "Jacobsthal function" + "restricted moduli" ; "least prime factor" + "interval after
a square" ; and the parent's own check for `reachability-landscape`, on which this builds.

**Index line for `docs/novel/README.md` (to be added by the manager - this lane is not permitted
to edit that file):**

`- island-witness-integers - the island witness for all integers: it fails at every q divisible by 5 (a gear dividing q relocates its strikes onto its own barred classes; gear 5 is 5 mod 8 so both are barred and both island classes) and holds for every integer coprime to 30 above 2849 and every prime in (1487, 200000]; one island class of density 1/35 suffices from q = 5477; a free island always sits inside [1, 0.152 d) above q = 20000; the exact cover number K(d) = 3, 4, 6, 9, 14, 20 grows while the counting requirement stalls at 2, 4, 5, 7, 9, 10 - SCRIPT-VERIFIED (divisor rule PROVED) - prior art NOT YET CHECKED (research/proof/island_witness.md)`
