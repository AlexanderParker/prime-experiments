/-
UniversalClearance (round 118, 2026-09-19): offsets no gear can strike at any twin centre.

A twin centre `s` (both `s - 1` and `s + 1` prime) satisfies `s ≢ ±1 (mod g)` for every prime
`g < s - 1`, so `s² mod g` is a square of a residue other than `±1`.  Gear `g` strikes the offset `j`
of the stretch of `s` iff `s² ≡ 1 - 6j` or `s² ≡ -1 - 6j (mod g)`.  If neither `1 - 6j` nor `-1 - 6j`
is such a square, the offset is free of `g` at EVERY twin centre and every level - a fixed class
set `U_g` with no dependence on `s`: `U_5 = {3}`, `U_7 = {0, 2, 4}`.  (Random lane, angle 1; tree
node R5.f.xiii.)  The general lemma is stated over ZMod; the instances are decided.
-/
import Mathlib.Tactic

namespace UniversalClearance

/-- **The clearance lemma.**  If `s ≢ ±1 (mod g)` and the residue `a` is not the square of any
residue other than `±1`, then `g ∤ s² - a`.  Applied with `a = 1 - 6j` (lower member) and
`a = -1 - 6j` (upper member) it clears offset `j` of gear `g` at every twin centre. -/
theorem not_dvd_of_not_square (g : ℕ) (s a : ℤ)
    (hs : ((s : ZMod g) ≠ 1) ∧ ((s : ZMod g) ≠ -1))
    (ha : ∀ x : ZMod g, x ≠ 1 → x ≠ -1 → x ^ 2 ≠ (a : ZMod g)) :
    ¬ ((g : ℤ) ∣ s ^ 2 - a) := by
  intro h
  have h' : ((s ^ 2 - a : ℤ) : ZMod g) = 0 := (ZMod.intCast_zmod_eq_zero_iff_dvd _ _).mpr h
  push_cast at h'
  have : (s : ZMod g) ^ 2 = (a : ZMod g) := by linear_combination h'
  exact ha _ hs.1 hs.2 this

/-- Residues other than `±1` mod 5 square to `0` or `4`; `1 - 6·3 = -17 ≡ 3` and `-1 - 6·3 = -19 ≡ 1`
are neither, so **gear 5 never strikes an offset `j ≡ 3 (mod 5)` at any twin centre**. -/
theorem clear_five (s j : ℤ) (hj : j % 5 = 3)
    (hs : ((s : ZMod 5) ≠ 1) ∧ ((s : ZMod 5) ≠ -1)) :
    ¬ ((5 : ℤ) ∣ s ^ 2 + 6 * j - 1) ∧ ¬ ((5 : ℤ) ∣ s ^ 2 + 6 * j + 1) := by
  obtain ⟨k, hk⟩ : ∃ k, j = 5 * k + 3 := ⟨j / 5, by omega⟩
  subst hk
  constructor
  · have := not_dvd_of_not_square 5 s (1 - 6 * (5 * k + 3)) hs (by
      intro x h1 h2 hx
      have hm : (1 - 6 * (5 * k + 3)) % ((5 : ℕ) : ℤ) = 3 := by push_cast; omega
      have : ((1 - 6 * (5 * k + 3) : ℤ) : ZMod 5) = 3 := by rw [← ZMod.intCast_mod, hm]; rfl
      rw [this] at hx
      revert x; decide)
    intro hd; apply this
    have e : s ^ 2 - (1 - 6 * (5 * k + 3)) = s ^ 2 + 6 * (5 * k + 3) - 1 := by ring
    rw [e]; exact hd
  · have := not_dvd_of_not_square 5 s (-1 - 6 * (5 * k + 3)) hs (by
      intro x h1 h2 hx
      have hm : (-1 - 6 * (5 * k + 3)) % ((5 : ℕ) : ℤ) = 1 := by push_cast; omega
      have : ((-1 - 6 * (5 * k + 3) : ℤ) : ZMod 5) = 1 := by rw [← ZMod.intCast_mod, hm]; rfl
      rw [this] at hx
      revert x; decide)
    intro hd; apply this
    have e : s ^ 2 - (-1 - 6 * (5 * k + 3)) = s ^ 2 + 6 * (5 * k + 3) + 1 := by ring
    rw [e]; exact hd

end UniversalClearance
