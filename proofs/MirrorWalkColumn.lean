/-
MirrorWalkColumn (round 51, 2026-09-14): the final step's landing is column `h` modulo the
spiral's base.

The primorial spiral lands at `E = -1 + 2 P A`, so `E ≡ -1 (mod P)` with `P` the product of the
base gears.  The final step `{3, h}` lands at `E + 6 h` (up) or `E - 6 h` (down).  Modulo any
base gear `g` (`g ∣ P`) the up landing is `(6h - 1, 6h + 1)`, column `h` itself, and the down
landing is `(-(6h + 1), -(6h - 1))`, its reflection.  Hence a base gear strikes the landing iff
it strikes column `h`: the base gears' forbidden classes of `h` are `±6⁻¹ (mod g)`, fixed by the
gear alone, not by `E` or the direction.
-/
import MirrorWalkFinal

namespace MirrorWalk

theorem dvd_iff_of_modEq {n a b : ℤ} (h : a ≡ b [ZMOD n]) : n ∣ a ↔ n ∣ b := by
  rw [← Int.modEq_zero_iff_dvd, ← Int.modEq_zero_iff_dvd]
  exact ⟨h.symm.trans, h.trans⟩

/-- **Up landing is column `h` modulo a base gear.**  With `E ≡ -1 (mod P)` and `g ∣ P`, gear
`g` strikes `(E + 6h, E + 6h + 2)` iff it strikes `(6h - 1, 6h + 1)`. -/
theorem base_strikes_up_iff {P E h : ℤ} {g : ℕ} (hE : E ≡ -1 [ZMOD P]) (hg : (g : ℤ) ∣ P) :
    ((g : ℤ) ∣ E + 6 * h ∨ (g : ℤ) ∣ E + 6 * h + 2) ↔
      ((g : ℤ) ∣ 6 * h - 1 ∨ (g : ℤ) ∣ 6 * h + 1) := by
  have hEg : E ≡ -1 [ZMOD g] := hE.of_dvd hg
  have h1 : E + 6 * h ≡ 6 * h - 1 [ZMOD g] := by
    have := hEg.add_right (6 * h); rwa [show -1 + 6 * h = 6 * h - 1 by ring] at this
  have h2 : E + 6 * h + 2 ≡ 6 * h + 1 [ZMOD g] := by
    have := hEg.add_right (6 * h + 2)
    rwa [show -1 + (6 * h + 2) = 6 * h + 1 by ring, ← add_assoc] at this
  rw [dvd_iff_of_modEq h1, dvd_iff_of_modEq h2]

/-- **Down landing is the reflection of column `h` modulo a base gear.**  Gear `g` strikes
`(E - 6h, E - 6h + 2)` iff it strikes `(6h - 1, 6h + 1)`. -/
theorem base_strikes_down_iff {P E h : ℤ} {g : ℕ} (hE : E ≡ -1 [ZMOD P]) (hg : (g : ℤ) ∣ P) :
    ((g : ℤ) ∣ E - 6 * h ∨ (g : ℤ) ∣ E - 6 * h + 2) ↔
      ((g : ℤ) ∣ 6 * h - 1 ∨ (g : ℤ) ∣ 6 * h + 1) := by
  have hEg : E ≡ -1 [ZMOD g] := hE.of_dvd hg
  have h1 : E - 6 * h ≡ -(6 * h + 1) [ZMOD g] := by
    have := hEg.add_right (-(6 * h)); rwa [show -1 + -(6 * h) = -(6 * h + 1) by ring,
      show E + -(6 * h) = E - 6 * h by ring] at this
  have h2 : E - 6 * h + 2 ≡ -(6 * h - 1) [ZMOD g] := by
    have := hEg.add_right (-(6 * h) + 2)
    rwa [show -1 + (-(6 * h) + 2) = -(6 * h - 1) by ring,
      show E + (-(6 * h) + 2) = E - 6 * h + 2 by ring] at this
  rw [dvd_iff_of_modEq h1, dvd_iff_of_modEq h2, dvd_neg, dvd_neg, or_comm]

/-- The primorial spiral's landing `-1 + 2 P A` is `-1` modulo `P`. -/
theorem primorialEnd_modEq (P A : ℤ) : -1 + 2 * P * A ≡ -1 [ZMOD P] := by
  rw [Int.modEq_iff_dvd]; exact ⟨-(2 * A), by ring⟩

/-- **Column open to the base ⟹ landing open to the base**, both directions. -/
theorem landing_open_base_iff {P E h : ℤ} {g : ℕ} (hE : E ≡ -1 [ZMOD P]) (hg : (g : ℤ) ∣ P) :
    (OpenTo g (E + 6 * h) ↔ OpenTo g (6 * h - 1)) ∧ (OpenTo g (E - 6 * h) ↔ OpenTo g (6 * h - 1)) := by
  unfold OpenTo
  have u := base_strikes_up_iff hE hg (h := h)
  have d := base_strikes_down_iff hE hg (h := h)
  have e : 6 * h - 1 + 2 = 6 * h + 1 := by ring
  rw [e]
  constructor
  · constructor
    · rintro ⟨a, b⟩; exact ⟨fun c => (u.mpr (Or.inl c)).elim a b, fun c => (u.mpr (Or.inr c)).elim a b⟩
    · rintro ⟨a, b⟩; exact ⟨fun c => (u.mp (Or.inl c)).elim a b, fun c => (u.mp (Or.inr c)).elim a b⟩
  · constructor
    · rintro ⟨a, b⟩; exact ⟨fun c => (d.mpr (Or.inl c)).elim a b, fun c => (d.mpr (Or.inr c)).elim a b⟩
    · rintro ⟨a, b⟩; exact ⟨fun c => (d.mp (Or.inl c)).elim a b, fun c => (d.mp (Or.inr c)).elim a b⟩

end MirrorWalk
