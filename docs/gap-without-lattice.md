# The gap without the lattice

Answering a direct question about the original algorithm: can the gap be obtained without iterating over the
lattice? The answer splits, and one half is yes. Script: `rust/src/bin/closedgap.rs`, which implements all three
variants and checks them against each other.

## Three versions, in the order they were written

**1. `rust/src/main.rs`, `next_prime_any` - eager lattice.** With `R = ceil(sqrt(n))`: compute each gear's first
tooth as `first(p) = (p - n mod p) mod p`, then mark the whole lattice `first(p), first(p)+p, first(p)+2p, ...`
up to `R` for every gear, then scan for the first unmarked candidate offset. The first step is closed form and is
the core insight. The marking costs `sum_{p<=R} R/p ~ R log log R` writes regardless of how near the next prime is.

**2. `rust2/src/main.rs`, `get_next_prime_gap` - lazy cursors.** This version **had already removed the lattice.**
It keeps one cursor per gear and advances each only as far as the candidate under test. Two details make it
efficient, and both are sound:

* the inner loop skips `divisors[0] = 2`. For odd `n` gear 2's teeth are all at odd offsets while the candidates
  are even, so gear 2 can never block one;
* the loop stops advancing once `divisors[i] >= test_gap`, because a gear with `p >= test_gap` can only reach
  `test_gap` on its first tooth, and the first teeth were already checked.

**3. `closedgap.rs`, `gap_by_candidates` - per-candidate test.** No array and no cursors. Offset `t` is open
exactly when no `p <= R` divides `n + t`, so walk the candidates and test each directly, exiting at the first gear
that divides. The early exit in versions 1 and 2 - return `n + 2` when the first candidate is not among the first
teeth - is this same test applied to one offset; version 3 is just that generalised to every offset.

## Measured

    n                          R      gap    lattice ops   cursor ops   cand ops   lat s    cur s   cand s
    7,213,393,222         84,932        1        228,513       16,542      8,271  0.00026  0.00009  0.00002
    100,000,000,000      316,228        3        885,532       81,879     27,298  0.00068  0.00022  0.00006
    1,000,000,000,000  1,000,000       39      2,887,174    1,648,501     82,552  0.00224  0.00074  0.00020
    10,000,000,000,000 3,162,278       37      9,383,340    4,552,981    227,911  0.01074  0.00199  0.00053
    100,000,000,000,000    10^7       31     30,414,281   11,297,876    664,704  0.04011  0.00697  0.00166
    1,000,000,000,000,000 3.16e7      37     98,360,900   39,039,200  2,925,888  0.30301  0.02006  0.00691

All three return the same prime on every benchmark case, and on **28,000 consecutive odd `n`** across three ranges
- from `10^6`, `10^10` and `999,999,000,000` - with zero disagreements.

`pi(R)` at the largest case is `1,951,958`.

## What the numbers say

**The lazy-cursor version is already 15 times faster than the eager lattice** at `n = 10^15`. Removing the lattice
was the right move and it was already done.

**A further 2.9 times is available**, and the reason is specific: `get_next_prime_gap` tests membership with
`gap_buckets.contains(&test_gap)`, a linear scan over all `pi(R)` cursors, once per candidate. That is
`pi(R) * gap/2` comparisons - about 35 million of the 39 million ops at `n = 10^15`, so it dominates everything
else the function does. Replacing it with a direct per-candidate divisibility test, or with a bitset of first
teeth, removes that term.

**There is a floor, and it is `pi(R)`.** Certifying that a slot is open means consulting every gear once, which is
exactly the window identity. The candidate version costs `2.9M` against a floor of `1.95M`, so it is within `1.5`
times optimal; the eager lattice was paying `log R * log log R` times the floor.

## A latent off-by-one, not reachable as used

Generalising version 2 to even `n` for the comparison exposed something worth recording, though **it cannot occur
in the code as written**. The guard `divisors[i] < test_gap` should be `<=`: a gear with `p == test_gap` can block
`test_gap` on its *second* tooth when `p | n`, since then its first tooth is at 0 and the `contains` check does not
see it. For odd `n` this needs an even `p`, so only `p = 2`, and `2 | n` is false - the case is unreachable.
`get_next_prime_gap` is only ever called on the last known prime, so odd `n` is its whole contract and it is
correct. But the guard is one character from being wrong if the function is ever reused on an even argument, and
the sweep over even `n` produces composites - `n = 1000008` returns `1000011 = 3 * 333337` - which is what put this
in view.

## The half that is no: the gap itself has no closed form

A formula giving the gap with no iteration over anything is not available, and the reason is exact. Offset `t` is
open iff `gcd(n + t, primorial(R)) = 1`, so the gap is

    least t >= 1 with n + t coprime to primorial(R),

the joint condition across all gears at once. By CRT the open offsets are a union of residue classes modulo the
primorial, exponential in `R`, and locating the least element above a given point is precisely the localisation
problem in `docs/handover.md` section 1. Anything producing the gap in time polynomial in `log n` would bound it
and settle that question.

So the ledger:

* **per-gear next tooth** - closed form, in all three versions;
* **per-offset openness** - closed form, used once in versions 1 and 2 and for every offset in version 3;
* **the gap** - not closed form, and equivalent to the open problem.
