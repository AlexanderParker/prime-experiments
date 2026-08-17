# Formalist workstream log

## Round 1 — Horizon theorem (2026-08-18)

### What was done

Formalised the horizon theorem as `proofs/Horizon.lean` (namespace `Horizon`,
standalone — imports only mathlib, nothing from `BlockedSlots`). Registered it
in `proofs/lakefile.toml` as a second `lean_lib` and added it to
`defaultTargets`; lake does not glob the directory, each root file needs its
own `[[lean_lib]]` entry. Added the three theorems to `proofs/AxiomCheck.lean`
(run with `~/.elan/bin/lake.exe env lean AxiomCheck.lean` from `proofs/`).

### Final theorem statements

```lean
theorem exists_prime_factor_lt {y m : ℕ} (hym : y < m) (hmyy : m < y * y)
    (hnp : ¬ m.Prime) : ∃ p, p.Prime ∧ p < y ∧ p ∣ m

theorem prime_of_no_prime_factor_lt {y m : ℕ} (hym : y < m) (hmyy : m < y * y)
    (h : ∀ p, p.Prime → p < y → ¬ p ∣ m) : m.Prime

theorem twin_of_no_prime_factor_lt {y m : ℕ} (hym : y < m) (hwin : m + 2 < y * y)
    (h : ∀ p, p.Prime → p < y → ¬ p ∣ m ∧ ¬ p ∣ (m + 2)) :
    m.Prime ∧ (m + 2).Prime
```

Note the strict bound `p < y` throughout: this is the interior form — gears
strictly below `y` decide the open window `(y, y*y)`, sharper than
`BlockedSlots.survivor_iff_twin` which uses `q ≤ y` and the closed window
`m + 2 ≤ y * y`.

### Proof route

`Nat.minFac m` is prime (`Nat.minFac_prime`, needs `m ≠ 1`, which follows from
`y < m < y*y` forcing `m ≥ 2`), divides `m` (`Nat.minFac_dvd`), and satisfies
`minFac m ^ 2 ≤ m` for composite `m` (`Nat.minFac_sq_le_self`). Then
`minFac² ≤ m < y²` gives `minFac < y` by contradiction with
`Nat.mul_le_mul` + `linarith` — no `Nat.sqrt` needed at all, cleaner than the
sketch. The pair corollary is two applications of the contrapositive form.
Adapted from `prime_of_not_blocked` in `BlockedSlots.lean:74-83`.

One iteration detail: `omega` cannot see through the nonlinear atom `y * y`,
so the window inequalities are discharged with `linarith` (which treats
`y * y` as an opaque atom) rather than `omega`. Avoided the deprecated
`push_neg` (mathlib now wants `push Not`) by using `Nat.lt_or_ge` /
`Nat.le_of_not_lt` instead.

### Build status

`~/.elan/bin/lake.exe build` from `proofs/`: **Build completed successfully
(968 jobs)**, zero `sorry`, zero warnings in `Horizon.lean`.
`BlockedSlots.lean` still builds clean (only its pre-existing `push_neg`
deprecation warnings).

### Axiom audit

```
'Horizon.exists_prime_factor_lt'      depends on axioms: [propext, Classical.choice, Quot.sound]
'Horizon.prime_of_no_prime_factor_lt' depends on axioms: [propext, Classical.choice, Quot.sound]
'Horizon.twin_of_no_prime_factor_lt'  depends on axioms: [propext, Classical.choice, Quot.sound]
```

Only the three standard axioms — nothing beyond ordinary mathematics.

### Proposed next target

The layer law's arithmetic core: one layer's novelty in stepping the horizon
from `y` to `y'` is `{y^2} ∪ {y*c : c prime, y < c, y*c < y'^2}` — i.e. a
composite that first becomes visible when the horizon passes `y` has least
prime factor exactly `y` (when `y` is prime). This is again a `minFac`
argument (`minFac m = y ↔ y ∣ m ∧ no smaller prime divides`), composes
directly with `exists_prime_factor_lt`, and needs no new machinery.
Second choice: the `h(2) ≥ d` product inequality.

## Round 2 — Layer law arithmetic core + slot-cap lemma (2026-08-18)

### What was done

New file `proofs/Layer.lean` (namespace `Layer`, mathlib-only imports),
registered as a third `[[lean_lib]]` in `proofs/lakefile.toml` and added to
`defaultTargets`. AxiomCheck.lean extended with the four new theorems.
Compiled first try, zero sorry, zero warnings.

### Final theorem statements

```lean
theorem slot_cap {q m : ℕ} (hq : 3 ≤ q) : ¬ (q ∣ m ∧ q ∣ m + 2)

theorem minFac_lt_or_eq {y y' m : ℕ}
    (hnext : ∀ q, q.Prime → y < q → q < y' → False)
    (h1 : 1 < m) (hnp : ¬ m.Prime) (hm : m < y' * y') :
    m.minFac < y ∨ m.minFac = y

theorem eq_mul_prime_of_minFac_eq {y m : ℕ} (h1 : 1 < m)
    (hfac : m.minFac = y) (hlow : y * y < m) (hhigh : m < y * y * y) :
    ∃ c, c.Prime ∧ y < c ∧ m = y * c

theorem layer_novelty {y y' m : ℕ}
    (hnext : ∀ q, q.Prime → y < q → q < y' → False)
    (hthin : y' * y' ≤ y * y * y)
    (hnp : ¬ m.Prime) (hlow : y * y < m) (hhigh : m < y' * y') :
    (∃ p, p.Prime ∧ p < y ∧ p ∣ m) ∨ ∃ c, c.Prime ∧ y < c ∧ m = y * c
```

### Design decisions

- Took the STRONGEST form: `c` is proved prime, not merely `minFac c ≥ y`.
  Bertrand was avoided entirely by carrying the thin-layer bound
  `y'^2 ≤ y^3` as an explicit hypothesis (`hthin`). For consecutive primes
  it holds from y = 3 on; the caller discharges it, this file never needs
  Bertrand. Inside the proof the range collapses to `c < y*y`, and a
  composite `c` with all prime factors `≥ y` (via `Nat.minFac_le_of_dvd`
  applied to `minFac c ∣ m` against `minFac m = y`) would need
  `y*y ≤ minFac c ^ 2 ≤ c` — contradiction.
- `hnext` phrased exactly like `BlockedSlots.survivor_step`'s gap
  hypothesis (`∀ q, q.Prime → y < q → q < y' → False`), so the two compose
  without adapters. `y` prime is NOT assumed — it falls out of
  `minFac m = y` in the semiprime branch.
- `layer_novelty`'s open bounds `y*y < m < y'*y'` deliberately exclude the
  boundary novelty `{y^2}` itself, matching "novelty = {y^2} ∪ {y*c}".
- slot_cap stated for `3 ≤ q` (no primality needed): `q ∣ m ∧ q ∣ m+2`
  gives `q ∣ 2` by `Nat.dvd_sub`, five lines total.

### Build status

`lake build`: **Build completed successfully (970 jobs)**, zero sorry.
BlockedSlots.lean and Horizon.lean unchanged and still building clean.

### Axiom audit (extended AxiomCheck.lean)

```
'Layer.slot_cap'                 depends on axioms: [propext, Quot.sound]
'Layer.minFac_lt_or_eq'          depends on axioms: [propext, Classical.choice, Quot.sound]
'Layer.eq_mul_prime_of_minFac_eq' depends on axioms: [propext, Classical.choice, Quot.sound]
'Layer.layer_novelty'            depends on axioms: [propext, Classical.choice, Quot.sound]
```

Standard axioms only; slot_cap needs neither choice nor anything classical.

### Proposed next target

The zero-slack theorem's finite core from the Constructor ledger: with
slot_cap and lpf attribution both now kernel-checked, the overlap-free
supply identity sum_q R(q) = 2N - P over a window is a Finset.card
partition argument — the natural next Lean chunk, and it is what C2
(prefix pigeonhole) formalisation would sit on. Alternative: h(2) ≥ d's
product inequality.

## Round 3 — Supply identity as a Finset partition (2026-08-18)

### What was done

New file `proofs/Supply.lean` (namespace `Supply`), the first file that
IMPORTS the earlier ones — `import Horizon` discharges the fiber-membership
obligation, `import Layer` supplies slot_cap for the slot-level corollary.
Registered as fourth `[[lean_lib]]` in lakefile.toml, added to
defaultTargets, AxiomCheck.lean extended. Compiled first try, zero sorry,
zero warnings.

### Final theorem statements

```lean
theorem minFac_mem_gears {y m : ℕ} (hym : y < m) (hmyy : m < y * y)
    (hnp : ¬ m.Prime) : m.minFac ∈ (Finset.range y).filter Nat.Prime

theorem card_composites_eq_sum_roots {y : ℕ} (S : Finset ℕ)
    (hS : ∀ m ∈ S, y < m ∧ m < y * y) :
    (S.filter fun m => ¬ m.Prime).card =
      ∑ p ∈ (Finset.range y).filter Nat.Prime,
        (S.filter fun m => ¬ m.Prime ∧ m.minFac = p).card

theorem card_eq_primes_add_sum_roots {y : ℕ} (S : Finset ℕ)
    (hS : ∀ m ∈ S, y < m ∧ m < y * y) :
    S.card = (S.filter Nat.Prime).card
      + ∑ p ∈ (Finset.range y).filter Nat.Prime,
          (S.filter fun m => ¬ m.Prime ∧ m.minFac = p).card

theorem roots_ne {m : ℕ} (h1 : 1 < m) (hodd : ¬ 2 ∣ m) :
    m.minFac ≠ (m + 2).minFac
```

### Proof route

- The partition is `Finset.card_eq_sum_card_fiberwise` (current mathlib
  signature takes `(↑s).MapsTo f t`) with `f = Nat.minFac`,
  `t = (range y).filter Nat.Prime`; the MapsTo obligation is exactly
  `minFac_mem_gears`, which is Horizon's `exists_prime_factor_lt` plus
  `Nat.minFac_le_of_dvd` (least factor ≤ the found factor < y). `m ≠ 1`
  falls out of `p ∣ m` for a prime p (`Nat.dvd_one`). The fibers are
  reshaped from nested filters to the `∧` form by `Finset.filter_filter`.
- Ledger form via `Finset.card_filter_add_card_filter_not`.
- `roots_ne`: a shared root would be a gear ≥ 3 (odd m rules out 2)
  dividing both m and m+2, killed by `Layer.slot_cap`. `3 ≤ minFac` comes
  from `lt_of_le_of_ne hp.two_le` — definitionally `2 < x` is `3 ≤ x`.
- The window hypothesis is per-member (`∀ m ∈ S, y < m ∧ m < y * y`), so S
  need not be an interval: it composes with any window Finset the
  constructor formalisation later chooses (e.g. the ±1 mod 6 members),
  and the `2N − P` reading is call-site arithmetic once S comes in pairs.

### Build status

`lake build`: **Build completed successfully (974 jobs)**, zero sorry.
BlockedSlots, Horizon, Layer unchanged, still building.

### Axiom audit (extended AxiomCheck.lean)

```
'Supply.minFac_mem_gears'            depends on axioms: [propext, Classical.choice, Quot.sound]
'Supply.card_composites_eq_sum_roots' depends on axioms: [propext, Classical.choice, Quot.sound]
'Supply.card_eq_primes_add_sum_roots' depends on axioms: [propext, Classical.choice, Quot.sound]
'Supply.roots_ne'                    depends on axioms: [propext, Classical.choice, Quot.sound]
```

### Proposed next target

The zero-slack direction: under Condition X (no twin in the window), every
slot has ≥ 1 composite member, so pairing the ledger form over slots gives
n1 + 2*n2 = C and n1 + n2 = N together with P = n1 — the census pinning
n1(t) = P(t), n2(t) = N(t) − P(t) as Finset statements over prefixes.
That is the formal substrate C2 (prefix pigeonhole) sits on. Alternative:
h(2) ≥ d's product inequality.
