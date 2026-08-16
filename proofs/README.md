# Formal statement of the algorithm

`BlockedSlots.lean` is a Lean 4 + mathlib formalisation of the blocked-slot
algorithm from `rust2/src/main.rs`.

What is defined and proved:

| Name | Content |
|------|---------|
| `Blocked n y g` | the gap `g` is ruled out from `n` by divisors up to `y`: some prime `q ≤ y` has `q ∣ n + g` |
| `blocked_iff_cursor` | that relation equals the running-cursor form: `q` blocks `(-n mod q) + k q` |
| `prime_of_not_blocked` | an unblocked slot is prime once the divisor bound reaches `sqrt(n+g)` - soundness |
| `not_blocked_of_prime` | a prime above the divisor bound is never blocked - no false negatives |
| `nextGap n` | the operation: the least unblocked slot, divisor bound tracking the candidate |
| `exists_gapOK` | the search terminates (only outside input: Euclid's theorem) |
| `nextGap_spec` | `n + nextGap n` is prime and nothing between `n` and it is prime |
| `BlockedTwin n y g` | the twin version: two cursors per divisor, `q ∣ n+g` or `q ∣ n+g+2` |
| `twin_of_not_blockedTwin` | an unblocked twin slot gives two primes |
| `not_blockedTwin_of_twin` | no false negatives for twins |
| `twinGap n` | the twin gap operation, defined where the search terminates |
| `twinGap_spec`, `no_twin_between` | it returns the next twin pair |
| `Survivor y m` | no prime `q ≤ y` divides `m` or `m + 2` |
| `survivor_iff_twin` | inside the certified window `(y, y*y]`, survivor and twin pair are the same thing |
| `twins_infinite_of_survivor_in_window` | **the reduction**: a survivor in the certified window for arbitrarily large `y` gives infinitely many twin primes |
| `survivor_in_window_of_gap_bound` | the gap form of that hypothesis |
| `survivor_step` | lockstep: moving to the next divisor destroys at most the survivor equal to that divisor |

Note the asymmetry that carries the whole problem: `exists_gapOK` is proved, so
`nextGap` is total, while the corresponding termination for `twinGap` is taken as
a hypothesis - that hypothesis *is* the twin prime conjecture.

Also included, from the centred form of the algorithm (section 12 of the program
document): `CentreSurvivor y c` (no divisor up to `y` divides `c^2 - 1`),
`centreSurvivor_iff_survivor`, and `centreSurvivor_iff_twin`.

## Status: machine-checked

`lake build` completes with no errors and no `sorry`. Every theorem listed above
depends only on the three standard axioms `propext`, `Classical.choice`,
`Quot.sound` - checked with `#print axioms`, so nothing is assumed beyond ordinary
mathematics.

Reproducing:

```
elan toolchain install $(cat lean-toolchain)
lake exe cache get      # run from this directory; the cache tool reads ./lean-toolchain
lake build
```

Toolchain `leanprover/lean4:v4.34.0-rc1`, mathlib pinned in `lake-manifest.json`.

What is *not* proved, and cannot be without settling the conjecture: the
hypothesis of `twins_infinite_of_survivor_in_window`, and the termination
hypothesis `∃ g, TwinGapOK n g` that `twinGap` takes as an argument. Everything
around them is now verified.
