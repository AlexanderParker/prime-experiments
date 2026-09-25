import Mathlib.Tactic
import RangeMissed

/-!
# Range derived: the copy map on the gears

For a number `y` whose square is `1` or `19` modulo `30`, the copy just above `y²` is
`J1 y = (y² + 29) / 30` (case 1) or `J19 y = (y² + 11) / 30` (case 19). For a prime gear
`g ≥ 7` these are exactly the copies of `RangeMissed.missed_copy_legs`. This file studies the
two maps `y ↦ J1 y` and `y ↦ J19 y` as maps on numbers.

This file proves:

* `J1_exact`, `J19_exact`: on its residue set the division is exact:
  `30 * J1 y = y² + 29` when `y² mod 30 = 1`, and `30 * J19 y = y² + 11` when `y² mod 30 = 19`.
* `copy_map_on_gear`: for a prime `g ≥ 7`, one of the two exact forms applies to `g`.
* `J1_fixed_identity`: in case 1, over the integers, `30 * (J1 y - y) = (y - 1) * (y - 29)`;
  `J1_fixed_iff`: in case 1, `J1 y = y` exactly when `y = 1` or `y = 29`.
* `sq_add_eleven_ne`: no natural `y` has `y² + 11 = 30 y`;
  `J19_no_fixed`: in case 19, `J19 y ≠ y`.
* `J1_strictMono`, `J19_strictMono` (and the `StrictMonoOn` forms `J1_strictMonoOn`,
  `J19_strictMonoOn`): on its residue set each map is strictly increasing.
* `J19_lt_self_iff`: in case 19, `J19 y < y` exactly when `1 ≤ y` and `y² + 11 < 30 y`;
  `J19_lt_self_list`: the numbers `y` in case 19 with `J19 y < y` are exactly `7, 13, 17, 23`.
* `J1_lt_self_iff`: in case 1, `J1 y < y` exactly when `1 < y < 29`;
  `J1_lt_self_list`: the numbers `y` in case 1 with `J1 y < y` are exactly `11, 19`.
-/

namespace RangeLine

/-- The case-1 copy map: `J1 y = (y² + 29) / 30`. -/
def J1 (y : ℕ) : ℕ := (y ^ 2 + 29) / 30

/-- The case-19 copy map: `J19 y = (y² + 11) / 30`. -/
def J19 (y : ℕ) : ℕ := (y ^ 2 + 11) / 30

/-- (a) In case 1 (`y² mod 30 = 1`) the division is exact: `30 * J1 y = y² + 29`. -/
theorem J1_exact (y : ℕ) (h : y ^ 2 % 30 = 1) : 30 * J1 y = y ^ 2 + 29 := by
  unfold J1
  generalize y ^ 2 = s at h ⊢
  omega

/-- (a) In case 19 (`y² mod 30 = 19`) the division is exact: `30 * J19 y = y² + 11`. -/
theorem J19_exact (y : ℕ) (h : y ^ 2 % 30 = 19) : 30 * J19 y = y ^ 2 + 11 := by
  unfold J19
  generalize y ^ 2 = s at h ⊢
  omega

/-- For a prime gear `g ≥ 7`, either `g` is in case 1 and `30 * J1 g = g² + 29`, or `g` is in
case 19 and `30 * J19 g = g² + 11`. -/
theorem copy_map_on_gear (g : ℕ) (hg : g.Prime) (h7 : 7 ≤ g) :
    (g ^ 2 % 30 = 1 ∧ 30 * J1 g = g ^ 2 + 29) ∨
      (g ^ 2 % 30 = 19 ∧ 30 * J19 g = g ^ 2 + 11) := by
  rcases sq_mod30_cases g hg h7 with h | h
  · exact Or.inl ⟨h, J1_exact g h⟩
  · exact Or.inr ⟨h, J19_exact g h⟩

/-- (b) In case 1, over the integers, `30 * (J1 y - y) = (y - 1) * (y - 29)`. -/
theorem J1_fixed_identity (y : ℕ) (h : y ^ 2 % 30 = 1) :
    30 * ((J1 y : ℤ) - (y : ℤ)) = ((y : ℤ) - 1) * ((y : ℤ) - 29) := by
  have e : ((30 * J1 y : ℕ) : ℤ) = ((y ^ 2 + 29 : ℕ) : ℤ) := by rw [J1_exact y h]
  push_cast at e
  linear_combination e

/-- (b) In case 1, `J1 y = y` exactly when `y = 1` or `y = 29`. -/
theorem J1_fixed_iff (y : ℕ) (h : y ^ 2 % 30 = 1) : J1 y = y ↔ y = 1 ∨ y = 29 := by
  have key := J1_fixed_identity y h
  constructor
  · intro hfix
    have hz : ((y : ℤ) - 1) * ((y : ℤ) - 29) = 0 := by
      rw [← key, hfix]; ring
    rcases mul_eq_zero.mp hz with h1 | h1
    · left; omega
    · right; omega
  · intro hy
    have hz : ((y : ℤ) - 1) * ((y : ℤ) - 29) = 0 := by
      rcases hy with rfl | rfl <;> norm_num
    rw [hz] at key
    have : (J1 y : ℤ) = (y : ℤ) := by linarith
    exact_mod_cast this

/-- (c) No natural number `y` satisfies `y² + 11 = 30 y` (the discriminant `856` of
`y² - 30 y + 11` is not a square). -/
theorem sq_add_eleven_ne (y : ℕ) : y ^ 2 + 11 ≠ 30 * y := by
  intro hy
  have hle : y ≤ 30 := by nlinarith
  interval_cases y <;> omega

/-- (c) In case 19, the map has no fixed point: `J19 y ≠ y`. -/
theorem J19_no_fixed (y : ℕ) (h : y ^ 2 % 30 = 19) : J19 y ≠ y := by
  intro hfix
  have e := J19_exact y h
  rw [hfix] at e
  exact sq_add_eleven_ne y e.symm

/-- (d) On the case-1 residue set, `J1` is strictly increasing: if `y < y'` then
`J1 y < J1 y'`. -/
theorem J1_strictMono (y y' : ℕ) (h : y ^ 2 % 30 = 1) (h' : y' ^ 2 % 30 = 1) (hlt : y < y') :
    J1 y < J1 y' := by
  have e := J1_exact y h
  have e' := J1_exact y' h'
  have hsq : y ^ 2 < y' ^ 2 := Nat.pow_lt_pow_left hlt (by norm_num)
  omega

/-- (d) On the case-19 residue set, `J19` is strictly increasing: if `y < y'` then
`J19 y < J19 y'`. -/
theorem J19_strictMono (y y' : ℕ) (h : y ^ 2 % 30 = 19) (h' : y' ^ 2 % 30 = 19)
    (hlt : y < y') : J19 y < J19 y' := by
  have e := J19_exact y h
  have e' := J19_exact y' h'
  have hsq : y ^ 2 < y' ^ 2 := Nat.pow_lt_pow_left hlt (by norm_num)
  omega

/-- (d) `J1` is strictly increasing on the set `{y | y² mod 30 = 1}`. -/
theorem J1_strictMonoOn : StrictMonoOn J1 {y : ℕ | y ^ 2 % 30 = 1} :=
  fun _ hy _ hy' hlt => J1_strictMono _ _ hy hy' hlt

/-- (d) `J19` is strictly increasing on the set `{y | y² mod 30 = 19}`. -/
theorem J19_strictMonoOn : StrictMonoOn J19 {y : ℕ | y ^ 2 % 30 = 19} :=
  fun _ hy _ hy' hlt => J19_strictMono _ _ hy hy' hlt

/-- (e) In case 19, `J19 y < y` exactly when `1 ≤ y` and `y² + 11 < 30 y`. -/
theorem J19_lt_self_iff (y : ℕ) (h : y ^ 2 % 30 = 19) :
    J19 y < y ↔ 1 ≤ y ∧ y ^ 2 + 11 < 30 * y := by
  have e := J19_exact y h
  constructor
  · intro hlt
    exact ⟨by omega, by omega⟩
  · rintro ⟨_, hlt⟩
    omega

/-- (e) The numbers `y` in case 19 with `J19 y < y` are exactly `7, 13, 17, 23`. -/
theorem J19_lt_self_list (y : ℕ) :
    (y ^ 2 % 30 = 19 ∧ J19 y < y) ↔ (y = 7 ∨ y = 13 ∨ y = 17 ∨ y = 23) := by
  constructor
  · rintro ⟨h, hlt⟩
    have hb := ((J19_lt_self_iff y h).mp hlt).2
    have hle : y < 30 := by nlinarith
    interval_cases y <;> simp_all
  · rintro (rfl | rfl | rfl | rfl) <;> decide

/-- (e) In case 1, `J1 y < y` exactly when `1 < y` and `y < 29`. -/
theorem J1_lt_self_iff (y : ℕ) (h : y ^ 2 % 30 = 1) : J1 y < y ↔ 1 < y ∧ y < 29 := by
  have key := J1_fixed_identity y h
  constructor
  · intro hlt
    have hneg : ((y : ℤ) - 1) * ((y : ℤ) - 29) < 0 := by
      rw [← key]
      have : (J1 y : ℤ) < (y : ℤ) := by exact_mod_cast hlt
      linarith
    rcases lt_or_ge (y : ℤ) 1 with h1 | h1
    · nlinarith
    rcases lt_or_ge (y : ℤ) 29 with h2 | h2
    · refine ⟨?_, by omega⟩
      rcases eq_or_lt_of_le h1 with h3 | h3
      · rw [← h3] at hneg; norm_num at hneg
      · exact_mod_cast h3
    · nlinarith
  · rintro ⟨h1, h2⟩
    have h1' : (1 : ℤ) < y := by exact_mod_cast h1
    have h2' : (y : ℤ) < 29 := by exact_mod_cast h2
    have hneg : ((y : ℤ) - 1) * ((y : ℤ) - 29) < 0 :=
      mul_neg_of_pos_of_neg (by linarith) (by linarith)
    have : (J1 y : ℤ) < (y : ℤ) := by linarith
    exact_mod_cast this

/-- (e) The numbers `y` in case 1 with `J1 y < y` are exactly `11, 19`. -/
theorem J1_lt_self_list (y : ℕ) :
    (y ^ 2 % 30 = 1 ∧ J1 y < y) ↔ (y = 11 ∨ y = 19) := by
  constructor
  · rintro ⟨h, hlt⟩
    have hb := (J1_lt_self_iff y h).mp hlt
    obtain ⟨_, hle⟩ := hb
    interval_cases y <;> simp_all
  · rintro (rfl | rfl) <;> decide

end RangeLine
