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
