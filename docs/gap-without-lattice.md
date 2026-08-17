# The gap without the lattice

Answering a direct question about the original algorithm in `rust/src/main.rs`: can the gap be obtained without
iterating over the lattice? The answer splits cleanly, and one half is yes.

Script: `rust/src/bin/closedgap.rs`.

## What the original does

`next_prime_any(n)` with `R = ceil(sqrt(n))`:

1. for each gear `p <= R`, compute its first tooth as a distance from `n`:

       first(p) = (p - n mod p) mod p

2. if any first tooth landed on a candidate offset, mark the whole lattice of teeth
   `first(p), first(p) + p, first(p) + 2p, ...` up to `R`, for every gear;
3. scan the marked array for the first unmarked candidate offset.

Step 1 is **already closed form** - two modular reductions per gear, nothing walked. That is the core insight of
the algorithm and it is right. Steps 2 and 3 are the lattice: step 2 writes `sum_{p<=R} R/p ~ R log log R` marks
and step 3 reads up to `R/2` entries, so the cost is `O(sqrt(n) log log sqrt(n))` **no matter how near the next
prime actually is**.

## The half that is yes: the lattice was never needed

The test for a single offset is closed form too, and the original already uses it once. Offset `t` is open exactly
when no gear has a tooth there, that is when

    no p <= R divides n + t.

The original's `even_slot_found` flag is precisely this test applied to the *first* candidate offset: if no gear's
first tooth lands there, return `n + 2` immediately without touching the lattice. **Generalising that early exit
from the first candidate to every candidate removes the lattice entirely.** Walk the candidate offsets and test
each directly, exiting at the first gear that divides.

Measured, both methods returning the same prime:

    n                     R        gap   lattice ops   cand ops   lattice s    cand s
    7,213,393,222         84,932     1       228,513      8,271    0.000495  0.000019
    100,000,000,000      316,228     3       885,532     27,298    0.000706  0.000104
    1,000,000,000,000  1,000,000    39     2,887,174     82,552    0.002468  0.000189
    10,000,000,000,000 3,162,278    37     9,383,340    227,911    0.011479  0.000551
    100,000,000,000,000   10^7      31    30,414,281    664,704    0.043902  0.001603
    1,000,000,000,000,000 3.16*10^7 37    98,360,900  2,925,888    0.397626  0.006537

At `n = 10^15` that is **34 times fewer operations and 61 times faster**, and it never allocates the `sqrt(n)`
window at all. Verified identical on **56,000 consecutive `n`** across three ranges - from `10^6`, from `10^10`,
from `999,999,000,000` - with zero disagreements.

**Why the ratio is what it is.** The lattice costs `R log log R`. The candidate walk costs about `pi(R) ~ R/log R`,
because a composite candidate exits after a handful of divisions - small primes are dense - so essentially the
only full pass over the gears is the one that certifies the prime itself. The ratio is therefore
`log R * log log R`, which at `R = 3.2 * 10^7` is about 34, matching the measurement.

**The floor this reveals.** `pi(R)` is irreducible within the algorithm's own terms: to certify that a slot is open
you must consult every gear once, which is exactly the window identity. The lattice was paying `log R * log log R`
times that floor. The candidate walk sits on the floor.

## The half that is no: the gap itself has no closed form

A formula that outputs the gap without iterating over *anything* is not available, and the reason is exact rather
than a limitation of effort. Offset `t` is open iff `gcd(n + t, primorial(R)) = 1`, so the gap is

    least t >= 1 with n + t coprime to primorial(R),

the joint condition across all gears at once. By CRT the open offsets form a union of residue classes modulo the
primorial, which is exponential in `R`, and locating the least element of that union above a given point is
precisely the localisation problem this whole programme is stuck on. Anything that produced the gap in time
polynomial in `log n` would immediately bound it, and so would settle the open question of
`docs/handover.md` section 1.

So the honest position is:

* **per-gear next tooth** - closed form, already in the original;
* **per-offset openness** - closed form, in the original but applied only to the first candidate;
* **the gap** - not closed form, and equivalent to the open problem.

The practical consequence is that the lattice iteration was pure overhead, and removing it costs nothing in
exactness: both methods are the same algorithm consulting the same gears, differing only in whether they
precompute answers they will not use.
