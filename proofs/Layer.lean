/-
Formalisation of the layer law's arithmetic core, and the slot-cap lemma.

Stepping the horizon from `y` to the next prime `y'` opens the layer
`(y * y, y' * y')`. The layer law from the session: the only genuinely new
composites in that layer are `y * y` itself and the semiprimes `y * c` with
`c` prime - everything else is already exposed by a gear strictly below `y`.

The arithmetic core proved here: with no prime strictly between `y` and `y'`,
a composite `m` in the open layer either has a prime factor `< y`, or is
`y * c` with `c` prime and `y < c`. The split is by least factor: `minFac m`
is below `y'` because `minFac m ^ 2 ≤ m < y'^2`, so the gap hypothesis forces
it `< y` or `= y`; in the second case the cofactor `c` has every prime factor
`≥ y` while `c < y * y`, so a proper factorisation would need `y * y ≤ c` and
`c` must be prime. The only global input is the thin-layer bound
`y'^2 ≤ y^3`, kept as an explicit hypothesis - for consecutive primes it
holds from `y = 3` on (Bertrand), but this file never needs Bertrand.

The slot-cap lemma is the constructor ledger's floor: a gear `q ≥ 3` can
never block both members of a pair, since it would divide their difference 2.
So kills per slot come from distinct gears and the supply ledger is
overlap-free.
-/

import Mathlib.Data.Nat.Prime.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.IntervalCases

namespace Layer

/-- **Slot cap.** A gear `q ≥ 3` never blocks both members of a pair: it
would have to divide their difference `2`. -/
theorem slot_cap {q m : ℕ} (hq : 3 ≤ q) : ¬ (q ∣ m ∧ q ∣ m + 2) := by
  rintro ⟨h1, h2⟩
  have hd : q ∣ 2 := by
    have h := Nat.dvd_sub h2 h1
    rwa [show m + 2 - m = 2 by omega] at h
  have := Nat.le_of_dvd (by omega) hd
  omega

/-- **Which gear exposes a layer composite.** With no prime strictly between
`y` and `y'`, a composite below `y' * y'` has least factor below `y` or equal
to `y`: the least factor is at most the square root, hence below `y'`, and
the gap hypothesis leaves no room in between. -/
theorem minFac_lt_or_eq {y y' m : ℕ}
    (hnext : ∀ q, q.Prime → y < q → q < y' → False)
    (h1 : 1 < m) (hnp : ¬ m.Prime) (hm : m < y' * y') :
    m.minFac < y ∨ m.minFac = y := by
  have hp : m.minFac.Prime := Nat.minFac_prime (by omega)
  have hsq : m.minFac * m.minFac ≤ m := by
    simpa [pow_two] using Nat.minFac_sq_le_self (by omega) hnp
  have hlt : m.minFac < y' := by
    by_contra hge
    have hge' : y' ≤ m.minFac := Nat.le_of_not_lt hge
    have : y' * y' ≤ m.minFac * m.minFac := Nat.mul_le_mul hge' hge'
    linarith
  have hle : m.minFac ≤ y := by
    by_contra hgt
    exact hnext m.minFac hp (Nat.lt_of_not_le hgt) hlt
  exact hle.lt_or_eq

/-- **The semiprime shape.** If the least factor of `m` is exactly `y` and
`y * y < m < y * y * y`, then `m` is `y` times a prime above `y`: the
cofactor has every prime factor `≥ y` yet sits below `y * y`, so it cannot
factor properly. -/
theorem eq_mul_prime_of_minFac_eq {y m : ℕ} (h1 : 1 < m)
    (hfac : m.minFac = y) (hlow : y * y < m) (hhigh : m < y * y * y) :
    ∃ c, c.Prime ∧ y < c ∧ m = y * c := by
  have hyp : y.Prime := hfac ▸ Nat.minFac_prime (by omega)
  have hy2 : 2 ≤ y := hyp.two_le
  have hdvd : y ∣ m := hfac ▸ Nat.minFac_dvd m
  obtain ⟨c, rfl⟩ := hdvd
  have hyc : y < c := by
    by_contra hge
    have : y * c ≤ y * y := Nat.mul_le_mul le_rfl (Nat.le_of_not_lt hge)
    linarith
  have hcu : c < y * y := by
    by_contra hge
    have h' : y * (y * y) ≤ y * c := Nat.mul_le_mul le_rfl (Nat.le_of_not_lt hge)
    have heq : y * (y * y) = y * y * y := (Nat.mul_assoc y y y).symm
    linarith
  have hcp : c.Prime := by
    by_contra hcnp
    have hc1 : 1 < c := by omega
    have hsq : c.minFac * c.minFac ≤ c := by
      simpa [pow_two] using Nat.minFac_sq_le_self (by omega) hcnp
    have hcd : c.minFac ∣ y * c := (Nat.minFac_dvd c).mul_left y
    have h2m : 2 ≤ c.minFac := (Nat.minFac_prime (by omega)).two_le
    have hle : (y * c).minFac ≤ c.minFac := Nat.minFac_le_of_dvd h2m hcd
    have hyle : y ≤ c.minFac := hfac ▸ hle
    have : y * y ≤ c.minFac * c.minFac := Nat.mul_le_mul hyle hyle
    linarith
  exact ⟨c, hcp, hyc, rfl⟩

/-- **Layer law, arithmetic core.** With no prime strictly between `y` and
`y'` and the thin-layer bound `y'^2 ≤ y^3`, every composite in the open layer
`(y * y, y' * y')` is either exposed by a prime strictly below `y`, or is `y`
times a prime above `y`. So the layer's novelty is exactly
`{y * y} ∪ {y * c : c prime}` - the open bounds here exclude the boundary
point `y * y` itself. -/
theorem layer_novelty {y y' m : ℕ}
    (hnext : ∀ q, q.Prime → y < q → q < y' → False)
    (hthin : y' * y' ≤ y * y * y)
    (hnp : ¬ m.Prime) (hlow : y * y < m) (hhigh : m < y' * y') :
    (∃ p, p.Prime ∧ p < y ∧ p ∣ m) ∨ ∃ c, c.Prime ∧ y < c ∧ m = y * c := by
  have hy2 : 2 ≤ y := by
    rcases Nat.lt_or_ge y 2 with hy | hy
    · exfalso; interval_cases y <;> linarith
    · exact hy
  have h4 : 2 * 2 ≤ y * y := Nat.mul_le_mul hy2 hy2
  have h1 : 1 < m := by linarith
  rcases minFac_lt_or_eq hnext h1 hnp hhigh with hlt | heq
  · exact Or.inl ⟨m.minFac, Nat.minFac_prime (by omega), hlt, Nat.minFac_dvd m⟩
  · exact Or.inr (eq_mul_prime_of_minFac_eq h1 heq hlow (lt_of_lt_of_le hhigh hthin))

end Layer
