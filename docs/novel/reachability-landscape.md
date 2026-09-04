# The reachability landscape and its islands

Branch R2.a.i.a (round 39), `research/proof/reachability.md`. Scripts
`research/anchor235/r39/rl_*.py`. Status per item below; PRIOR ART CHECK **not yet checked**
(no web access in this lane).

## 1. WHAT IT IS

The walk from `q^2` under the machine `{5..q}` runs along columns `k_0 + i`, `k_0 = (q^2-1)/6`,
whose members are `q^2 + 6i - 2` and `q^2 + 6i`. Gear `g` strikes the column at offset `i` iff
`q^2 = 2 - 6i` or `-6i (mod g)`. Since `q^2` is a nonzero square modulo every gear below `q`,
gear `g` can strike offset `i` **for no prime `q` at all** unless one of the two targets is a
nonzero quadratic residue mod `g`. Which gears can ever reach an offset is therefore a property
of the offset alone - a landscape free of `q`. Write `Bar(g)` for the offset classes mod `g` that
`g` can never reach, and call an offset barred by every gear `5 <= g <= B` an **island for bound
`B`**.

**(A) The bar-size closed form.** With `chi_g` the Legendre symbol,

```
        |Bar(g)|  =  ( g + 1 - chi_g(2) - chi_g(-2) ) / 4 ,
```

so the size of a gear's unreachable set is fixed by `g mod 8`: `(g-1)/4`, `(g+1)/4`, `(g+3)/4`,
`(g+1)/4` for `g = 1, 3, 5, 7 (mod 8)`. Exact for all 2,260 gears to 20,000, 0 exceptions. In
particular gear 5 can reach only the offsets `1, 3, 4 (mod 5)` and gear 7 only
`0, 1, 2, 4, 6 (mod 7)`, and **no gear reaches every offset**.

**(B) The landscape mirror.** The map `i -> d_g - i (mod g)`, `d_g = 2 * 6^{-1}` the gear's own
tooth separation, sends the target pair `(x, x+2)` to `(-(x+2), -x)` and so multiplies both
Legendre symbols by `chi_g(-1)`. Hence `Bar(g)` is **preserved** exactly when `g = 1 (mod 4)`
(1,125 of 1,125 gears) and is carried **entirely into the reachable set** when `g = 3 (mod 4)`
(1,135 of 1,135). The island set therefore has no reflection symmetry from `B = 7` on.

**(C) The island system.** The islands for bound `B` are a union of exactly
`prod_{5<=g<=B} |Bar(g)|` residue classes modulo `P_B = prod_{5<=g<=B} g`:

| `B` | 5 | 7 | 11 | 13 | 17 | 19 | 23 |
|---|---|---|---|---|---|---|---|
| `P_B` | 5 | 35 | 385 | 5,005 | 85,085 | 1,616,615 | 37,182,145 |
| islands | 2 | **4** | 12 | 48 | 192 | 960 | 5,760 |
| density | 0.4 | 0.1143 | 0.03117 | 0.009590 | 0.002257 | 0.000594 | 0.000155 |
| max gap | 3 | 23 | 77 | 313 | 1,687 | 12,882 | 49,357 |

explicitly `{0, 2} mod 5` and `{5, 10, 12, 17} mod 35`. The only offsets reachable by *every*
gear are `i = -6t^2` (with `g | 6t` and `g != +-1 mod 8` the sole exception) - the columns
`k_0 - 6t^2` whose member factors as `(q-6t)(q+6t)`, all **behind** the walk. No forward offset
below 20,000 is reachable by all 2,260 gears.

**(D) The landing prefers islands, by exactly the order-one amount.** Over every prime
`q = 5..20000` (2,260 walks), the landing offset `L` is an island for `B = 5, 7, 11, 13` at rates
0.6681, 0.3226, 0.1473, 0.0960 - a `B = 13` island is **10 times** likelier to be a landing than
a random offset. The four smallest islands `5, 10, 12, 17` are the four commonest landings and
take **472 of 2,260 = 20.9%** of them. Against the order-one first-passage null (each gear
independent, `p_g(i) = 2 chi_g(i)/(g-1)`) the measured rates are 0.99, 0.93, 0.88, 0.92 of
prediction: the preference is entirely the per-gear bar, with nothing above order one.

**(E) The `B = 7` island witness.** For **every prime `q` in `(1487, 20000]` - 2,026 primes, 0
exceptions** - at least one offset `= 5, 10, 12` or `17 (mod 35)` inside `[1, d)` (where
`d = 2*6^{-1} mod q` is the top gear's forward tooth arc) is struck by **no gear of `{5..q}` at
all**. Such an offset is an opening, so this witnesses the walk's `L < d` on a **fixed set of
density `4/35` that does not depend on `q`**. The slack grows: the minimum number of free islands
by `q` band is 0, 0, 0, 4, 12 over `5-100`, `100-10^3`, `10^3-5·10^3`, `5·10^3-10^4`,
`10^4-2·10^4`; median 38 in the top band. The 17 failing primes below the threshold are
`17, 23, 29, 41, 53, 73, 113, 137, 173, 197, 233, 263, 353, 461, 683, 1151, 1487`.

**(F) The frame is scale-free (the negative half).** By CRT the island set is a union of classes
mod `P_B` and every gear `g > B` is coprime to `P_B`, so a large gear strikes **islands at exactly
the machine's own rate `2/g`**. Measured over 103,899 island sightings, the ratio of the per-gear
rate on islands to `2/g` over the 40 gears `17..199` has mean **0.9956**. Hence

```
        (strikes by gears > B on the islands of [1,d)) / (islands of [1,d))  =  sum_{B<g<=q} 2/g
```

- verified to four significant figures at `B = 7, 11, 13, 17` (measured 2.703, 2.5212, 2.3616,
2.2426 against the exact sums 2.699, 2.517, 2.363, 2.246 at `q ~ 15,000`). Restricting to islands
divides targets and strikes by the same `rho_B`: the counting margin is identical to the
unrestricted covering problem, and it crosses 1 at `q = 53, 79, 113, 163` for `B = 7, 11, 13, 17`.

**(G) The doubling.** Because the phase enters only as the square `q^2`, an admissible target
contributes exactly its two square roots: the number of classes `r = q mod g` at which gear `g`
strikes offset `i` is exactly `2 chi_g(i)`, and is **never odd**. Exhaustive over every gear
`5..500` and every offset class - 21,531 cells, 0 exceptions of either kind. Summed over the
offsets, `sum_i chi_g(i) = g - 1`, so a gear's mean strike rate over all offsets is exactly `2/g`
while it is exactly 0 on a quarter of them: **the bar concentrates a gear's strikes, it does not
reduce them**.

## 2. WHY IT MIGHT BE NOVEL

The underlying facts are classical and are named as such: the character sum
`sum_x chi(x(x+2)) = -1`, Gauss's second supplement (`chi(2) = 1` iff `g = +-1 mod 8`), and the
observation that the prime divisors of a quadratic's values lie in half the residue classes. What
is not classical is the object: a **fixed, prime-independent set of positions that the sieving
machine's small gears can never occupy**, its exact residue system and spacing, and its use as a
covering target for the walk from `q^2`. Item (E) is a reduction of a covering statement to a
`q`-free set of density `4/35`; item (F) is an exact proof that the reduction cannot be finished
by counting. Neither shape - the island system, or the CRT scale-freeness that defeats it -
appears in the project register.

The statement (F) is the sharpest form the project has of a wall it already knows: the
counter-ladder verdict of `cover-half-counter-ladder.md` ("no fixed-depth truncation and no
exposure-only argument bounds `L` uniformly"), reached here in the sparsest available coordinates
and with an exact identity rather than a measured trend.

## 3. PROOF

* (A), (B), (G) and the `-6t^2` family: **PROVED** (elementary, character sums and Gauss's
  supplement) and **SCRIPT-VERIFIED** exactly - `rl_landscape.py` (2,260 gears, 45.2 million
  admissibility cells), `rl_double.py` (21,531 exhaustive cells).
* (C): **PROVED** (CRT on the residue conditions) and computed exactly by iterated CRT to
  `B = 23`, `rl_landscape.py`.
* (F): the rate identity is **PROVED** (CRT: `g` coprime to `P_B`); the four-digit agreement is
  **SCRIPT-VERIFIED**, `rl_interact.py`, `rl_rate.py`, `rl_double.py`.
* (D): **MEASURED**, exact counts over every prime to 20,000, against a computed order-one null
  (`rl_landing.py`, `rl_null.py`).
* (E): **MEASURED**, exact, 0 exceptions in the 2,026 primes of `(1487, 20000]`
  (`rl_interact.py`, `rl_margin.py`). Not proved: it is `L < d` on a restricted witness set, and
  `L < d` is itself unproved (a twin-Bertrand-strength statement at scale `q/3`).

## 4. IMPLICATIONS

Inside the project: the parent branches' per-offset depth function `lambda(i)` is this landscape's
relief, and 86.9% of its variance comes from the four gears `{5, 7, 11, 13}` alone - so "the
landing avoids high-depth offsets" (round 38) *is* "the landing prefers islands", and (D) shows
the effect is exactly order one, closing it as a lever. What survives as a route-shaped object is
(E): the first unproven interaction named by both parents, `L < d`, now has a `q`-free witness
set. The smallest statement that would have to be proved becomes

> for every prime `q` there is an `i = 5, 10, 12` or `17 (mod 35)` with `1 <= i < d` such that
> `q != +- s (mod g)` for every gear `g` in `(7, q]` and every square root `s` of `-6i` or
> `2 - 6i` modulo `g`

- a covering statement in which the **sifted variable is `q` itself**, not the offset, and the
targets are a fixed set. Outside the project: (A) and (B) are small exact facts about where the
prime divisors of the pair `(x^2 + 6i - 2, x^2 + 6i)` can lie, as a function of `i`.

## 5. UNSOLVED QUESTIONS IT TOUCHES

Ziller-Morack Conjecture 6 / the paired Jacobsthal bound only indirectly: this is a statement
about the position of the first opening above `q^2`, i.e. about the first twin pair above `q^2`,
which is a twin-Bertrand-strength localisation at scale `q/3`. Item (F) is a negative result for
sieve-counting approaches to it on any sparse `q`-free target set.

## 6. PRIOR-ART CHECK

**NOT YET CHECKED** (no web access in this lane). Screened against `docs/novel/README.md`
(walk-path-parts, walk-path-transforms, walk-tooth-frame, anchor-235-layer-laws,
cover-half-counter-ladder, j2-lower-ladder, layered-erdos-rankin, corridor-law,
golden-spectral-gap), `docs/proofs/` and the two parent branch documents: no entry defines an
unreachable-offset set, its density, or its use as a covering target; the word "island" appears
nowhere in the register. Terms a checker should try: "quadratic residue admissible arithmetic
progression covering", "prime divisors of x^2+a and x^2+a+2 residue classes", "Jacobsthal
covering restricted target set", "least prime factor of consecutive quadratic values".
