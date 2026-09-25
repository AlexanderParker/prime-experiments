import Mathlib.Data.Nat.Prime.Basic
import Mathlib.Data.Nat.Prime.Int
import Mathlib.Data.ZMod.Basic
import Mathlib.Tactic

/-!
# Range copies: which gears strike a copy, and which pairs of copies a gear strikes together

The copy `j` (for `j ≥ 1`) is the pair of numbers `30 j - 1` and `30 j + 1`, its two legs.
A number `g` strikes copy `j` when `g` divides one of the two legs.

This file proves:

* `copy_product`, `copy_strike_iff`: the product of the two legs of copy `j` is `900 j² - 1`,
  so `g` divides the product exactly when it divides `900 j² - 1` (any `g`, any `j`).
  `copy_strike_iff_legs`: for a prime `g`, dividing the product is the same as striking a leg.
* `minus_leg_iff_mod`, `plus_leg_iff_mod`, `copy_strike_class`: for `g ≥ 2` and `j ≥ 1`,
  `g` divides the minus leg exactly when `30 j mod g = 1`, and the plus leg exactly when
  `30 j mod g = g - 1`; so `g` strikes copy `j` exactly when `30 j ≡ ±1 (mod g)`.
* The four leg rules, for a prime `g ≥ 7` that already divides one leg of copy `j`:
  - minus leg to minus leg of copy `j + D`, and plus leg to plus leg: iff `g ∣ D`
    (the legs differ by `30 D`);
  - minus leg of copy `j` to plus leg of copy `j + D`: iff `g ∣ 15 D + 1`
    (the legs differ by `30 D + 2 = 2 (15 D + 1)`);
  - plus leg of copy `j` to minus leg of copy `j + D` (with `D ≥ 1`): iff `g ∣ 15 D - 1`
    (the legs differ by `30 D - 2 = 2 (15 D - 1)`).
* `copy_leg_rule`: if a prime `g ≥ 7` strikes copy `j` and copy `j + D`, then
  `g ∣ D` or `g ∣ 15 D - 1` or `g ∣ 15 D + 1`.
* `copy_leg_rule_converse`: conversely, each of these three conditions gives a copy `j`
  with `1 ≤ j < g` such that `g` strikes both copy `j` and copy `j + D`.
* `copy_pair_iff`: the two directions together, as one equivalence.
-/

namespace RangeLine

/-! ## Small facts about a prime gear `g ≥ 7` -/

/-- A number at least 7 does not divide 2. -/
lemma not_dvd_two {g : ℕ} (hg7 : 7 ≤ g) : ¬ g ∣ 2 := fun h => by
  have := Nat.le_of_dvd (by norm_num) h
  omega

/-- A prime at least 7 does not divide 30 = 2 · 3 · 5. -/
lemma not_dvd_thirty {g : ℕ} (hg : g.Prime) (hg7 : 7 ≤ g) : ¬ g ∣ 30 := by
  intro h
  have h' : g ∣ 2 * (3 * 5) := by norm_num; exact h
  rcases (Nat.Prime.dvd_mul hg).mp h' with h2 | h15
  · have := Nat.le_of_dvd (by norm_num) h2
    omega
  · rcases (Nat.Prime.dvd_mul hg).mp h15 with h3 | h5
    · have := Nat.le_of_dvd (by norm_num) h3
      omega
    · have := Nat.le_of_dvd (by norm_num) h5
      omega

/-- In `ℤ`, a prime `g ≥ 7` divides `2 x` exactly when it divides `x`. -/
lemma int_two_mul_iff {g : ℕ} (hg : g.Prime) (hg7 : 7 ≤ g) {x : ℤ} :
    (g : ℤ) ∣ 2 * x ↔ (g : ℤ) ∣ x :=
  ⟨fun h => ((Nat.prime_iff_prime_int.mp hg).dvd_or_dvd h).resolve_left
      (fun h2 => not_dvd_two hg7 (by exact_mod_cast h2)),
   fun h => dvd_mul_of_dvd_right h 2⟩

/-- In `ℤ`, a prime `g ≥ 7` divides `30 x` exactly when it divides `x`. -/
lemma int_thirty_mul_iff {g : ℕ} (hg : g.Prime) (hg7 : 7 ≤ g) {x : ℤ} :
    (g : ℤ) ∣ 30 * x ↔ (g : ℤ) ∣ x :=
  ⟨fun h => ((Nat.prime_iff_prime_int.mp hg).dvd_or_dvd h).resolve_left
      (fun h30 => not_dvd_thirty hg hg7 (by exact_mod_cast h30)),
   fun h => dvd_mul_of_dvd_right h 30⟩

/-- If `g` divides `x`, then `g` divides `y` exactly when it divides the step `y - x`. -/
lemma int_step_iff {g x y z : ℤ} (hx : g ∣ x) (hyz : y - x = z) : g ∣ y ↔ g ∣ z := by
  rw [← hyz]
  exact ⟨fun hy => dvd_sub hy hx, fun h => by simpa using dvd_add h hx⟩

/-! ## (a) The product of the two legs -/

/-- The two legs of copy `j` multiply to `900 j² - 1` (natural-number subtraction; true for
every `j`, and at `j = 0` both sides are `0`). -/
theorem copy_product (j : ℕ) : (30 * j - 1) * (30 * j + 1) = 900 * j ^ 2 - 1 := by
  rcases Nat.eq_zero_or_pos j with rfl | hj
  · simp
  · obtain ⟨k, rfl⟩ : ∃ k, j = k + 1 := ⟨j - 1, by omega⟩
    have e1 : 30 * (k + 1) - 1 = 30 * k + 29 := by omega
    have e2 : 900 * (k + 1) ^ 2 = (900 * k ^ 2 + 1800 * k + 899) + 1 := by ring
    rw [e1, e2, Nat.add_sub_cancel]
    ring

/-- (a) `g` divides the product of the two legs of copy `j` exactly when `g` divides
`900 j² - 1`. This holds for every `g` and every `j`; no primality or size condition is used. -/
theorem copy_strike_iff (g j : ℕ) :
    g ∣ (30 * j - 1) * (30 * j + 1) ↔ g ∣ 900 * j ^ 2 - 1 := by
  rw [copy_product]

/-- For a prime `g`, dividing the product of the legs of copy `j` is the same as striking the
copy (dividing one of its legs). -/
theorem copy_strike_iff_legs {g j : ℕ} (hg : g.Prime) :
    g ∣ 900 * j ^ 2 - 1 ↔ g ∣ 30 * j - 1 ∨ g ∣ 30 * j + 1 := by
  rw [← copy_product]
  exact Nat.Prime.dvd_mul hg

/-! ## (b) The strike class `30 j ≡ ±1 (mod g)` -/

/-- For `g ≥ 2` and `j ≥ 1`: `g` divides the minus leg `30 j - 1` exactly when
`30 j mod g = 1`. -/
theorem minus_leg_iff_mod {g j : ℕ} (hg : 2 ≤ g) (hj : 1 ≤ j) :
    g ∣ 30 * j - 1 ↔ 30 * j % g = 1 := by
  rw [← Nat.modEq_iff_dvd' (by omega : 1 ≤ 30 * j)]
  unfold Nat.ModEq
  rw [Nat.mod_eq_of_lt (by omega : 1 < g)]
  exact eq_comm

/-- For `g ≥ 2`: `g` divides the plus leg `30 j + 1` exactly when `30 j mod g = g - 1`. -/
theorem plus_leg_iff_mod {g j : ℕ} (hg : 2 ≤ g) :
    g ∣ 30 * j + 1 ↔ 30 * j % g = g - 1 := by
  have hr := Nat.mod_lt (30 * j) (by omega : 0 < g)
  rw [Nat.dvd_iff_mod_eq_zero, Nat.add_mod, Nat.mod_eq_of_lt (by omega : 1 < g)]
  constructor
  · intro h
    rcases Nat.lt_or_ge (30 * j % g + 1) g with hlt | hge
    · rw [Nat.mod_eq_of_lt hlt] at h
      omega
    · omega
  · intro h
    rw [h, Nat.sub_add_cancel (by omega : 1 ≤ g), Nat.mod_self]

/-- (b) For `g ≥ 2` (in particular every prime `g ≥ 7`) and `j ≥ 1`: `g` strikes copy `j`
exactly when `30 j mod g` is `1` or `g - 1`, that is, `30 j ≡ ±1 (mod g)`. -/
theorem copy_strike_class {g j : ℕ} (hg : 2 ≤ g) (hj : 1 ≤ j) :
    (g ∣ 30 * j - 1 ∨ g ∣ 30 * j + 1) ↔ (30 * j % g = 1 ∨ 30 * j % g = g - 1) := by
  rw [minus_leg_iff_mod hg hj, plus_leg_iff_mod hg]

/-! ## The four leg rules -/

/-- Minus leg to minus leg: if a prime `g ≥ 7` divides `30 j - 1` (`j ≥ 1`), then it divides
`30 (j + D) - 1` exactly when `g ∣ D`. -/
theorem leg_minus_minus {g j D : ℕ} (hg : g.Prime) (hg7 : 7 ≤ g) (hj : 1 ≤ j)
    (h : g ∣ 30 * j - 1) : g ∣ 30 * (j + D) - 1 ↔ g ∣ D := by
  have hstep : ((30 * (j + D) - 1 : ℕ) : ℤ) - ((30 * j - 1 : ℕ) : ℤ) = 30 * (D : ℤ) := by
    omega
  rw [← Int.natCast_dvd_natCast] at h
  rw [← Int.natCast_dvd_natCast, int_step_iff h hstep, int_thirty_mul_iff hg hg7,
    Int.natCast_dvd_natCast]

/-- Plus leg to plus leg: if a prime `g ≥ 7` divides `30 j + 1`, then it divides
`30 (j + D) + 1` exactly when `g ∣ D`. -/
theorem leg_plus_plus {g j D : ℕ} (hg : g.Prime) (hg7 : 7 ≤ g)
    (h : g ∣ 30 * j + 1) : g ∣ 30 * (j + D) + 1 ↔ g ∣ D := by
  have hstep : ((30 * (j + D) + 1 : ℕ) : ℤ) - ((30 * j + 1 : ℕ) : ℤ) = 30 * (D : ℤ) := by
    omega
  rw [← Int.natCast_dvd_natCast] at h
  rw [← Int.natCast_dvd_natCast, int_step_iff h hstep, int_thirty_mul_iff hg hg7,
    Int.natCast_dvd_natCast]

/-- Minus leg to plus leg: if a prime `g ≥ 7` divides `30 j - 1` (`j ≥ 1`), then it divides
`30 (j + D) + 1` exactly when `g ∣ 15 D + 1` (the legs differ by `2 (15 D + 1)`). -/
theorem leg_minus_plus {g j D : ℕ} (hg : g.Prime) (hg7 : 7 ≤ g) (hj : 1 ≤ j)
    (h : g ∣ 30 * j - 1) : g ∣ 30 * (j + D) + 1 ↔ g ∣ 15 * D + 1 := by
  have hstep : ((30 * (j + D) + 1 : ℕ) : ℤ) - ((30 * j - 1 : ℕ) : ℤ)
      = 2 * ((15 * D + 1 : ℕ) : ℤ) := by
    omega
  rw [← Int.natCast_dvd_natCast] at h
  rw [← Int.natCast_dvd_natCast, int_step_iff h hstep, int_two_mul_iff hg hg7,
    Int.natCast_dvd_natCast]

/-- Plus leg to minus leg: if a prime `g ≥ 7` divides `30 j + 1` and `D ≥ 1`, then it divides
`30 (j + D) - 1` exactly when `g ∣ 15 D - 1` (the legs differ by `2 (15 D - 1)`). -/
theorem leg_plus_minus {g j D : ℕ} (hg : g.Prime) (hg7 : 7 ≤ g) (hD : 1 ≤ D)
    (h : g ∣ 30 * j + 1) : g ∣ 30 * (j + D) - 1 ↔ g ∣ 15 * D - 1 := by
  have hstep : ((30 * (j + D) - 1 : ℕ) : ℤ) - ((30 * j + 1 : ℕ) : ℤ)
      = 2 * ((15 * D - 1 : ℕ) : ℤ) := by
    omega
  rw [← Int.natCast_dvd_natCast] at h
  rw [← Int.natCast_dvd_natCast, int_step_iff h hstep, int_two_mul_iff hg hg7,
    Int.natCast_dvd_natCast]

/-! ## (c) The leg rule -/

/-- (c) If a prime `g ≥ 7` strikes copy `j` (`j ≥ 1`) and copy `j + D`, then `g ∣ D` or
`g ∣ 15 D - 1` or `g ∣ 15 D + 1`. (No condition on `D` is needed: at `D = 0`, `g ∣ D`.) -/
theorem copy_leg_rule {g j D : ℕ} (hg : g.Prime) (hg7 : 7 ≤ g) (hj : 1 ≤ j)
    (h1 : g ∣ 30 * j - 1 ∨ g ∣ 30 * j + 1)
    (h2 : g ∣ 30 * (j + D) - 1 ∨ g ∣ 30 * (j + D) + 1) :
    g ∣ D ∨ g ∣ 15 * D - 1 ∨ g ∣ 15 * D + 1 := by
  rcases h1 with h1 | h1 <;> rcases h2 with h2 | h2
  · exact Or.inl ((leg_minus_minus hg hg7 hj h1).mp h2)
  · exact Or.inr (Or.inr ((leg_minus_plus hg hg7 hj h1).mp h2))
  · rcases Nat.eq_zero_or_pos D with hD | hD
    · exact Or.inl (hD ▸ dvd_zero g)
    · exact Or.inr (Or.inl ((leg_plus_minus hg hg7 hD h1).mp h2))
  · exact Or.inl ((leg_plus_plus hg hg7 h1).mp h2)

/-! ## (d) The converse -/

/-- Every prime `g ≥ 7` divides the minus leg of some copy `j` with `1 ≤ j < g`
(`j` is the inverse of `30` modulo `g`). -/
theorem exists_leg_minus {g : ℕ} (hg : g.Prime) (hg7 : 7 ≤ g) :
    ∃ j, 1 ≤ j ∧ j < g ∧ g ∣ 30 * j - 1 := by
  have := Fact.mk hg
  have : NeZero g := ⟨by omega⟩
  have h30 : (30 : ZMod g) ≠ 0 := by
    intro h
    have h' : ((30 : ℕ) : ZMod g) = 0 := by exact_mod_cast h
    rw [ZMod.natCast_eq_zero_iff] at h'
    exact not_dvd_thirty hg hg7 h'
  have hc0 : (30 : ZMod g)⁻¹ ≠ 0 := inv_ne_zero h30
  have hpos : 1 ≤ ((30 : ZMod g)⁻¹).val := ZMod.val_pos.mpr hc0
  refine ⟨((30 : ZMod g)⁻¹).val, hpos, ZMod.val_lt _, ?_⟩
  rw [← ZMod.natCast_eq_zero_iff, Nat.cast_sub (by omega), Nat.cast_mul,
    ZMod.natCast_zmod_val, Nat.cast_one, Nat.cast_ofNat, mul_inv_cancel₀ h30, sub_self]

/-- Every prime `g ≥ 7` divides the plus leg of some copy `j` with `1 ≤ j < g`
(reflect the minus-leg copy `m` to `g - m`). -/
theorem exists_leg_plus {g : ℕ} (hg : g.Prime) (hg7 : 7 ≤ g) :
    ∃ j, 1 ≤ j ∧ j < g ∧ g ∣ 30 * j + 1 := by
  obtain ⟨m, hm1, hmg, hm⟩ := exists_leg_minus hg hg7
  refine ⟨g - m, by omega, by omega, ?_⟩
  have hstep : ((30 * (g - m) + 1 : ℕ) : ℤ) = 30 * (g : ℤ) - ((30 * m - 1 : ℕ) : ℤ) := by
    omega
  rw [← Int.natCast_dvd_natCast] at hm
  rw [← Int.natCast_dvd_natCast, hstep]
  exact dvd_sub (dvd_mul_left _ _) hm

/-- (d) The converse of the leg rule, in full: if a prime `g ≥ 7` divides `D`, `15 D - 1` or
`15 D + 1`, then there is a copy `j` with `1 ≤ j < g` such that `g` strikes both copy `j` and
copy `j + D`. -/
theorem copy_leg_rule_converse {g D : ℕ} (hg : g.Prime) (hg7 : 7 ≤ g)
    (h : g ∣ D ∨ g ∣ 15 * D - 1 ∨ g ∣ 15 * D + 1) :
    ∃ j, 1 ≤ j ∧ j < g ∧ (g ∣ 30 * j - 1 ∨ g ∣ 30 * j + 1) ∧
      (g ∣ 30 * (j + D) - 1 ∨ g ∣ 30 * (j + D) + 1) := by
  by_cases hgD : g ∣ D
  · obtain ⟨j, hj1, hjg, hj⟩ := exists_leg_minus hg hg7
    exact ⟨j, hj1, hjg, Or.inl hj, Or.inl ((leg_minus_minus hg hg7 hj1 hj).mpr hgD)⟩
  · have hD : 1 ≤ D := by
      rcases Nat.eq_zero_or_pos D with h0 | h0
      · exact absurd (h0 ▸ dvd_zero g) hgD
      · exact h0
    rcases h with h | h | h
    · exact absurd h hgD
    · obtain ⟨j, hj1, hjg, hj⟩ := exists_leg_plus hg hg7
      exact ⟨j, hj1, hjg, Or.inr hj, Or.inl ((leg_plus_minus hg hg7 hD hj).mpr h)⟩
    · obtain ⟨j, hj1, hjg, hj⟩ := exists_leg_minus hg hg7
      exact ⟨j, hj1, hjg, Or.inl hj, Or.inr ((leg_minus_plus hg hg7 hj1 hj).mpr h)⟩

/-- The leg rule and its converse together: a prime `g ≥ 7` strikes some pair of copies at
distance `D` (copy `j` and copy `j + D`, `j ≥ 1`) exactly when `g ∣ D`, `g ∣ 15 D - 1` or
`g ∣ 15 D + 1`. -/
theorem copy_pair_iff {g D : ℕ} (hg : g.Prime) (hg7 : 7 ≤ g) :
    (∃ j, 1 ≤ j ∧ (g ∣ 30 * j - 1 ∨ g ∣ 30 * j + 1) ∧
      (g ∣ 30 * (j + D) - 1 ∨ g ∣ 30 * (j + D) + 1)) ↔
    (g ∣ D ∨ g ∣ 15 * D - 1 ∨ g ∣ 15 * D + 1) := by
  constructor
  · rintro ⟨j, hj, h1, h2⟩
    exact copy_leg_rule hg hg7 hj h1 h2
  · intro h
    obtain ⟨j, hj1, _, h1, h2⟩ := copy_leg_rule_converse hg hg7 h
    exact ⟨j, hj1, h1, h2⟩

end RangeLine
