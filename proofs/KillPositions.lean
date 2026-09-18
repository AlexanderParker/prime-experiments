/-
KillPositions (round 96, 2026-09-19): where a gear can kill after a square - a location law.

The stretch above a gear's square is where the machine's next twins are decided (StretchRule).
A gear `h` strikes the member `p² + a` there exactly when `p² ≡ -a` modulo `h`; so `-a` must be a
square modulo `h`.  That is a law about POSITION: the offset `a` after the square fixes, by
quadratic residuosity, which gears are able to strike it at all.

  `strike_after_square_isSquare`: a gear striking `p² + a` has `-a` a square modulo itself.
  `square_neighbour_killers`: the lower member of the square's own column, `p² - 2`, can be
    struck only by gears congruent to 1 or 7 modulo 8.
  `next_lower_killers`: the next column's lower member, `p² + 4`, only by gears congruent to
    1 modulo 4.
  `next_upper_killers`: the next column's upper member, `p² + 6`, only by gears with `-6` a
    square - and so on down the stretch, one class condition per position.

These say nothing about how many kills there are.  They say which gears are eligible at each
location, and that the eligibility is decided by the position's residue character, not by the
gear's size.
-/
import Mathlib

namespace MirrorWalk

/-- **A gear striking `p² + a` has `-a` a square modulo itself.** -/
theorem strike_after_square_isSquare {h p a : ℕ} [Fact h.Prime] (hdvd : h ∣ p ^ 2 + a) :
    IsSquare (-(a : ZMod h)) := by
  refine ⟨(p : ZMod h), ?_⟩
  have h0 : ((p ^ 2 + a : ℕ) : ZMod h) = 0 := (ZMod.natCast_eq_zero_iff _ _).mpr hdvd
  push_cast at h0
  linear_combination -h0

/-- **The square's own neighbour.**  A gear (odd prime) dividing `p² - 2` is `1` or `7` modulo 8,
because `2` must be a square modulo it. -/
theorem square_neighbour_killers {h p : ℕ} [hp : Fact h.Prime] (h2 : h ≠ 2) (hpp : 2 ≤ p ^ 2)
    (hdvd : h ∣ p ^ 2 - 2) : h % 8 = 1 ∨ h % 8 = 7 := by
  have hsq : IsSquare (2 : ZMod h) := by
    refine ⟨(p : ZMod h), ?_⟩
    have h0 : ((p ^ 2 - 2 : ℕ) : ZMod h) = 0 := (ZMod.natCast_eq_zero_iff _ _).mpr hdvd
    rw [Nat.cast_sub hpp] at h0
    push_cast at h0
    linear_combination -h0
  exact (ZMod.exists_sq_eq_two_iff h2).mp hsq

/-- **The next column's lower member.**  A gear dividing `p² + 4` is `1` modulo 4, because `-1`
must be a square modulo it (`-4 = (2)² · (-1)`). -/
theorem next_lower_killers {h p : ℕ} [hp : Fact h.Prime] (h2 : h ≠ 2)
    (hdvd : h ∣ p ^ 2 + 4) : h % 4 = 1 := by
  have hs : IsSquare (-(4 : ZMod h)) := by
    have := strike_after_square_isSquare (h := h) (p := p) (a := 4) hdvd
    simpa using this
  have h2z : (2 : ZMod h) ≠ 0 := by
    intro h0
    have : (h : ℕ) ∣ 2 := by
      have := (ZMod.natCast_eq_zero_iff 2 h).mp (by exact_mod_cast h0)
      exact this
    have hle := Nat.le_of_dvd (by norm_num) this
    have := hp.out.two_le
    omega
  have hneg : IsSquare (-1 : ZMod h) := by
    obtain ⟨x, hx⟩ := hs
    refine ⟨x * (2 : ZMod h)⁻¹, ?_⟩
    have hinv : (2 : ZMod h) * (2 : ZMod h)⁻¹ = 1 := mul_inv_cancel₀ h2z
    calc (-1 : ZMod h) = -(4 : ZMod h) * ((2 : ZMod h)⁻¹ * (2 : ZMod h)⁻¹) := by
          have : (4 : ZMod h) = 2 * 2 := by norm_num
          rw [this]
          calc (-1 : ZMod h) = -((2 : ZMod h) * (2 : ZMod h)⁻¹) * ((2 : ZMod h) * (2 : ZMod h)⁻¹) := by
                rw [hinv]; ring
            _ = -(2 * 2) * ((2 : ZMod h)⁻¹ * (2 : ZMod h)⁻¹) := by ring
      _ = x * x * ((2 : ZMod h)⁻¹ * (2 : ZMod h)⁻¹) := by rw [hx]
      _ = x * (2 : ZMod h)⁻¹ * (x * (2 : ZMod h)⁻¹) := by ring
  have := (ZMod.exists_sq_eq_neg_one_iff).mp hneg
  -- h is an odd prime, so h % 4 is 1 or 3, and 3 is excluded
  have hodd : h % 2 = 1 := by
    rcases hp.out.eq_two_or_odd with h2' | hodd
    · exact absurd h2' h2
    · exact hodd
  omega

end MirrorWalk
