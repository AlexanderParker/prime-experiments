/-
LadderEuclid (2026-09-20, prover lane F7 extended): the twin's primality propagates along the
stretch.  At an offset `j` divisible by a gear `p` other than the twin gears, the lower member
`s^2 + 6j - 1` is never struck by `p` (it is `(s-1)(s+1)` mod `p`, and `p` divides neither prime),
and the upper member `s^2 + 6j + 1` is struck by `p` exactly when `p ∣ s^2 + 1`.  So at offsets
divisible by every gear up to `x` the lower member is `x`-rough unconditionally - the Euclid device
inside the stretch.  (Tree node R5.f.xxviii.)
-/
import TwinLadderTheorem

namespace TwinLadder

/-- The lower member at offset `j` is `(s-1)(s+1) + 6j`. -/
theorem lower_member_eq {s j : ℕ} (hs : TwinCentre s) :
    s ^ 2 + 6 * j - 1 = (s - 1) * (s + 1) + 6 * j := by
  have h6 := twinCentre_ge_six hs
  obtain ⟨t, rfl⟩ : ∃ t, s = t + 1 := ⟨s - 1, by omega⟩
  rw [Nat.add_sub_cancel]
  ring_nf; omega

/-- **The Euclid device**: a gear `p` dividing the offset `j`, other than the twin gears, does not
strike the lower member of column `j`. -/
theorem lower_member_rough {s p j : ℕ} (hs : TwinCentre s) (hp : p.Prime)
    (hne1 : p ≠ s - 1) (hne2 : p ≠ s + 1) (hj : p ∣ j) : ¬ p ∣ s ^ 2 + 6 * j - 1 := by
  intro hdiv
  rw [lower_member_eq hs] at hdiv
  have h6j : p ∣ 6 * j := Dvd.dvd.mul_left hj 6
  have hprod : p ∣ (s - 1) * (s + 1) := (Nat.dvd_add_left h6j).mp hdiv
  obtain ⟨_, hp1, hp2⟩ := hs
  rcases (Nat.Prime.dvd_mul hp).mp hprod with h | h
  · exact hne1 ((Nat.prime_dvd_prime_iff_eq hp hp1).mp h)
  · exact hne2 ((Nat.prime_dvd_prime_iff_eq hp hp2).mp h)

/-- The upper member at an offset divisible by `p` is struck by `p` iff `p ∣ s^2 + 1`. -/
theorem upper_member_iff {s p j : ℕ} (hj : p ∣ j) :
    p ∣ s ^ 2 + 6 * j + 1 ↔ p ∣ s ^ 2 + 1 := by
  have h6j : p ∣ 6 * j := Dvd.dvd.mul_left hj 6
  have : s ^ 2 + 6 * j + 1 = (s ^ 2 + 1) + 6 * j := by ring
  rw [this]
  exact Nat.dvd_add_left h6j

end TwinLadder
