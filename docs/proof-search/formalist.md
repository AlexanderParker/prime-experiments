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
