import Mathlib.Data.Nat.Prime.Basic
import Mathlib.Data.ZMod.Basic
import Mathlib.Algebra.BigOperators.Associated
import Mathlib.Tactic

/-!
# The locator closure

A *copy* is a natural number `j`; the gear (prime) `g` *strikes* the copy `j` when
`g ∣ (30 * j - 1) * (30 * j + 1)`, i.e. when `g` divides one of the pair `30j - 1`, `30j + 1`.

This file proves, in plain words:

* (a) `class_meets_every_residue`: if the prime `g` does not divide `N`, then the `g` numbers
  `r, r + N, r + 2N, ..., r + (g - 1)N` run over every residue modulo `g` (every residue is hit,
  and no residue is hit twice: `class_residues_distinct`).
* (b) `locator_closure`: if the prime `g ≥ 7` does not divide `N`, then among those `g` numbers
  there is a copy `j = r + kN` (with `j ≥ 1`, so the subtraction `30j - 1` is honest) that `g`
  strikes; in fact `g ∣ 30j - 1`. Contrapositive (`locator_closure_contra`): if no copy of the
  congruence class `j ≡ r (mod N)` is struck by `g`, then `g ∣ N`.
* (c) `locator_modulus`: if every copy `j ≡ r (mod N)` with `j ≥ 1` is unstruck by every prime
  `g` with `7 ≤ g ≤ X`, then every such prime divides `N`; hence (`locator_modulus_prod`)
  their product divides `N`, and (`locator_modulus_le`) that product is at most `N` when `N ≥ 1`.

No hypothesis `N ≥ 1` or `r ≥ 1` is needed: `¬ g ∣ N` already forces `N ≠ 0`, and the copy found
in (b) always satisfies `30j ≡ 1 (mod g)`, so `j ≥ 1` comes out as part of the conclusion.
-/

namespace RangeLine

/-- (a) The class `r + kN`, `k = 0, 1, ..., g - 1`, meets every residue modulo the prime `g`
when `g ∤ N`: for every target `t` there is `k < g` with `r + kN ≡ t (mod g)`. -/
theorem class_meets_every_residue {g N : ℕ} (hg : g.Prime) (hgN : ¬ g ∣ N) (r t : ℕ) :
    ∃ k < g, (r + k * N) % g = t % g := by
  have := Fact.mk hg
  have hN0 : (N : ZMod g) ≠ 0 := by
    rw [Ne, ZMod.natCast_eq_zero_iff]
    exact hgN
  refine ⟨(((t : ZMod g) - r) * (N : ZMod g)⁻¹).val, ZMod.val_lt _, ?_⟩
  rw [← ZMod.natCast_eq_natCast_iff']
  push_cast
  rw [ZMod.natCast_zmod_val, mul_assoc, inv_mul_cancel₀ hN0]
  ring

/-- (a, second half) The `g` numbers `r + kN`, `k < g`, are pairwise distinct modulo the prime
`g` when `g ∤ N`; together with `class_meets_every_residue` they are a complete residue system. -/
theorem class_residues_distinct {g N : ℕ} (hg : g.Prime) (hgN : ¬ g ∣ N) (r : ℕ)
    {k k' : ℕ} (hk : k < g) (hk' : k' < g)
    (h : (r + k * N) % g = (r + k' * N) % g) : k = k' := by
  have := Fact.mk hg
  have hN0 : (N : ZMod g) ≠ 0 := by
    rw [Ne, ZMod.natCast_eq_zero_iff]
    exact hgN
  rw [← ZMod.natCast_eq_natCast_iff'] at h
  push_cast at h
  have h2 : (k : ZMod g) = (k' : ZMod g) := by
    have h3 : ((k : ZMod g) - k') * N = 0 := by linear_combination h
    rcases mul_eq_zero.mp h3 with h4 | h4
    · exact sub_eq_zero.mp h4
    · exact absurd h4 hN0
  rw [ZMod.natCast_eq_natCast_iff', Nat.mod_eq_of_lt hk, Nat.mod_eq_of_lt hk'] at h2
  exact h2

/-- For a prime `g ≥ 7`, `30` is invertible modulo `g` (`g` is none of `2, 3, 5`). -/
theorem thirty_ne_zero {g : ℕ} (hg : g.Prime) (hg7 : 7 ≤ g) : ((30 : ℕ) : ZMod g) ≠ 0 := by
  rw [Ne, ZMod.natCast_eq_zero_iff]
  intro h
  have h30 : (30 : ℕ) = 2 * 3 * 5 := by norm_num
  rw [h30] at h
  rcases (Nat.Prime.dvd_mul hg).mp h with h | h
  · rcases (Nat.Prime.dvd_mul hg).mp h with h | h
    · have := Nat.le_of_dvd (by norm_num) h; omega
    · have := Nat.le_of_dvd (by norm_num) h; omega
  · have := Nat.le_of_dvd (by norm_num) h; omega

/-- The sharp form of (b): for a prime `g ≥ 7` with `g ∤ N` and any `r`, some `k < g` gives a
copy `j = r + kN` with `j ≥ 1` and `g ∣ 30j - 1`. -/
theorem locator_closure_minus {g N : ℕ} (hg : g.Prime) (hg7 : 7 ≤ g) (hgN : ¬ g ∣ N) (r : ℕ) :
    ∃ k < g, 1 ≤ r + k * N ∧ g ∣ 30 * (r + k * N) - 1 := by
  have := Fact.mk hg
  have h30 := thirty_ne_zero hg hg7
  obtain ⟨k, hk, hmod⟩ :=
    class_meets_every_residue hg hgN r (((30 : ℕ) : ZMod g)⁻¹).val
  rw [← ZMod.natCast_eq_natCast_iff', ZMod.natCast_zmod_val] at hmod
  have hone : ((30 * (r + k * N) : ℕ) : ZMod g) = 1 := by
    rw [Nat.cast_mul, hmod, mul_inv_cancel₀ h30]
  have hpos : 1 ≤ 30 * (r + k * N) := by
    by_contra hlt
    have h0 : 30 * (r + k * N) = 0 := by omega
    rw [h0, Nat.cast_zero] at hone
    exact zero_ne_one hone
  refine ⟨k, hk, by omega, ?_⟩
  rw [← ZMod.natCast_eq_zero_iff, Nat.cast_sub hpos, hone, Nat.cast_one, sub_self]

/-- (b) The locator closure: for a prime `g ≥ 7` not dividing `N` and any `r`, some copy
`j = r + kN` with `k < g` (and `j ≥ 1`, so the subtraction is safe) is struck by `g`:
`g ∣ (30j - 1)(30j + 1)`. -/
theorem locator_closure {g N : ℕ} (hg : g.Prime) (hg7 : 7 ≤ g) (hgN : ¬ g ∣ N) (r : ℕ) :
    ∃ k < g, 1 ≤ r + k * N ∧
      g ∣ (30 * (r + k * N) - 1) * (30 * (r + k * N) + 1) := by
  obtain ⟨k, hk, hpos, hdvd⟩ := locator_closure_minus hg hg7 hgN r
  exact ⟨k, hk, hpos, Dvd.dvd.mul_right hdvd _⟩

/-- (b, contrapositive) A congruence class of copies modulo `N` no copy of which is struck by
the prime `g ≥ 7` forces `g ∣ N`. -/
theorem locator_closure_contra {g N r : ℕ} (hg : g.Prime) (hg7 : 7 ≤ g)
    (hclass : ∀ j, 1 ≤ j → j ≡ r [MOD N] → ¬ g ∣ (30 * j - 1) * (30 * j + 1)) : g ∣ N := by
  by_contra hgN
  obtain ⟨k, _, hpos, hdvd⟩ := locator_closure hg hg7 hgN r
  exact hclass (r + k * N) hpos (Nat.add_mul_mod_self_right r k N) hdvd

/-- (c) The locator modulus: if every copy `j ≡ r (mod N)`, `j ≥ 1`, is unstruck by every prime
`g` with `7 ≤ g ≤ X`, then every such prime divides `N`. -/
theorem locator_modulus {N r X : ℕ}
    (hclass : ∀ j, 1 ≤ j → j ≡ r [MOD N] →
      ∀ g, g.Prime → 7 ≤ g → g ≤ X → ¬ g ∣ (30 * j - 1) * (30 * j + 1)) :
    ∀ g, g.Prime → 7 ≤ g → g ≤ X → g ∣ N := by
  intro g hg hg7 hgX
  exact locator_closure_contra hg hg7 (fun j hj hjr => hclass j hj hjr g hg hg7 hgX)

/-- (c, product form) Under the hypothesis of `locator_modulus`, the product of all primes `g`
with `7 ≤ g ≤ X` divides `N`. -/
theorem locator_modulus_prod {N r X : ℕ}
    (hclass : ∀ j, 1 ≤ j → j ≡ r [MOD N] →
      ∀ g, g.Prime → 7 ≤ g → g ≤ X → ¬ g ∣ (30 * j - 1) * (30 * j + 1)) :
    (∏ p ∈ (Finset.Icc 7 X).filter Nat.Prime, p) ∣ N := by
  apply Finset.prod_primes_dvd
  · intro a ha
    rw [Finset.mem_filter] at ha
    exact ha.2.prime
  · intro a ha
    rw [Finset.mem_filter, Finset.mem_Icc] at ha
    exact locator_modulus hclass a ha.2 ha.1.1 ha.1.2

/-- (c, size form) Under the hypothesis of `locator_modulus` and `N ≥ 1`, the modulus `N` is at
least the product of all primes `g` with `7 ≤ g ≤ X`. -/
theorem locator_modulus_le {N r X : ℕ} (hN : 1 ≤ N)
    (hclass : ∀ j, 1 ≤ j → j ≡ r [MOD N] →
      ∀ g, g.Prime → 7 ≤ g → g ≤ X → ¬ g ∣ (30 * j - 1) * (30 * j + 1)) :
    (∏ p ∈ (Finset.Icc 7 X).filter Nat.Prime, p) ≤ N :=
  Nat.le_of_dvd hN (locator_modulus_prod hclass)

end RangeLine
