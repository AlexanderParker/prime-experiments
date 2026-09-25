import Mathlib.Data.ZMod.Basic
import Mathlib.Tactic

/-!
# The centre law for one gear on two copies

Copies are indexed by naturals `k ≥ 1`; copy `k` carries the pair `(30k - 1, 30k + 1)`.
A gear `g` *strikes* copy `k` when `g ∣ (30k - 1)(30k + 1)`.

For a prime `g ≥ 7`, `g` strikes copy `k` exactly when `30k ≡ 1` or `30k ≡ -1 (mod g)`.
Two copies `x < y` are then both struck in one of two ways:

* same sign: `30x ≡ 30y`, so `g ∣ y - x` (as `g ∤ 30`), and the strike on `x` is
  `g ∣ 900x² - 1`;
* opposite signs: `30x ≡ -30y`, so `g ∣ x + y`, and `30(y - x) ≡ ±2`, i.e.
  `900(y - x)² ≡ 4`, i.e. `g ∣ 225(y - x)² - 1` (as `g` is odd).

Conversely each of the two conditions forces both strikes: `g ∣ y - x` carries the strike on `x`
to `y`, and `y ≡ -x` turns `225(y - x)² - 1` into `225·4x² - 1 = 900x² - 1` (and likewise for
`y`), so the second disjunct needs no separate strike hypothesis.

This file proves:

* `strikesCopy_iff_sq`: `g` strikes copy `k` iff `g ∣ 900k² - 1` (the natural-number identity
  `(30k - 1)(30k + 1) = 900k² - 1` holds for every `k`, including `k = 0` where both sides are `0`).
* `strikesCopy_iff_zmod`: for `k ≥ 1`, the strike read in `ZMod g`.
* `centre_law_field`: the centre law as an identity in any field with `30 ≠ 0` and `4 ≠ 0`.
* `centre_law`: for a prime `g ≥ 7` and `1 ≤ x < y`,
  `g` strikes both `x` and `y` iff
  `(g ∣ y - x ∧ g ∣ 900x² - 1) ∨ (g ∣ x + y ∧ g ∣ 225(y - x)² - 1)`.
-/

namespace RangeLine

/-- The gear `g` strikes copy `k`: `g` divides `(30k - 1)(30k + 1)`. -/
def StrikesCopy (g k : ℕ) : Prop := g ∣ (30 * k - 1) * (30 * k + 1)

/-- The strike on copy `k` is divisibility of `900k² - 1`: in `ℕ`,
`(30k - 1)(30k + 1) = 900k² - 1` for every `k` (both sides are `0` at `k = 0`). -/
theorem strikesCopy_iff_sq (g k : ℕ) : StrikesCopy g k ↔ g ∣ 900 * k ^ 2 - 1 := by
  unfold StrikesCopy
  have e : (30 * k - 1) * (30 * k + 1) = 900 * k ^ 2 - 1 := by
    rcases Nat.eq_zero_or_pos k with rfl | hk
    · simp
    · have h1 : 1 ≤ 30 * k := by omega
      have h2 : 1 ≤ 900 * k ^ 2 := by nlinarith
      zify [h1, h2]
      ring
  rw [e]

/-- For `k ≥ 1`, `g` strikes copy `k` iff `(30k - 1)(30k + 1) = 0` in `ZMod g`. -/
theorem strikesCopy_iff_zmod {g k : ℕ} (hk : 1 ≤ k) :
    StrikesCopy g k ↔ (30 * (k : ZMod g) - 1) * (30 * (k : ZMod g) + 1) = 0 := by
  unfold StrikesCopy
  rw [← ZMod.natCast_eq_zero_iff]
  push_cast [Nat.cast_sub (show 1 ≤ 30 * k by omega)]
  rfl

/-- The centre law in a field `F` with `30 ≠ 0` and `4 ≠ 0`: `a` and `b` both satisfy
`(30t - 1)(30t + 1) = 0` iff either `b = a` and `900a² = 1`, or `a + b = 0` and
`225(b - a)² = 1`. -/
theorem centre_law_field {F : Type*} [Field F] (h30 : (30 : F) ≠ 0) (h4 : (4 : F) ≠ 0)
    (a b : F) :
    ((30 * a - 1) * (30 * a + 1) = 0 ∧ (30 * b - 1) * (30 * b + 1) = 0) ↔
      (b - a = 0 ∧ 900 * a ^ 2 - 1 = 0) ∨ (a + b = 0 ∧ 225 * (b - a) ^ 2 - 1 = 0) := by
  constructor
  · rintro ⟨hA, hB⟩
    rcases mul_eq_zero.1 hA with ha | ha <;> rcases mul_eq_zero.1 hB with hb | hb
    · -- `30a ≡ 1`, `30b ≡ 1`: same sign
      left
      refine ⟨?_, ?_⟩
      · have h : (30 : F) * (b - a) = 0 := by linear_combination hb - ha
        exact (mul_eq_zero.1 h).resolve_left h30
      · linear_combination (30 * a + 1) * ha
    · -- `30a ≡ 1`, `30b ≡ -1`: opposite signs
      right
      refine ⟨?_, ?_⟩
      · have h : (30 : F) * (a + b) = 0 := by linear_combination ha + hb
        exact (mul_eq_zero.1 h).resolve_left h30
      · have h : (4 : F) * (225 * (b - a) ^ 2 - 1) = 0 := by
          linear_combination (30 * (b - a) - 2) * (hb - ha)
        exact (mul_eq_zero.1 h).resolve_left h4
    · -- `30a ≡ -1`, `30b ≡ 1`: opposite signs
      right
      refine ⟨?_, ?_⟩
      · have h : (30 : F) * (a + b) = 0 := by linear_combination ha + hb
        exact (mul_eq_zero.1 h).resolve_left h30
      · have h : (4 : F) * (225 * (b - a) ^ 2 - 1) = 0 := by
          linear_combination (30 * (b - a) + 2) * (hb - ha)
        exact (mul_eq_zero.1 h).resolve_left h4
    · -- `30a ≡ -1`, `30b ≡ -1`: same sign
      left
      refine ⟨?_, ?_⟩
      · have h : (30 : F) * (b - a) = 0 := by linear_combination hb - ha
        exact (mul_eq_zero.1 h).resolve_left h30
      · linear_combination (30 * a - 1) * ha
  · rintro (⟨h1, h2⟩ | ⟨h1, h2⟩)
    · exact ⟨by linear_combination h2, by linear_combination h2 + 900 * (b + a) * h1⟩
    · exact ⟨by linear_combination h2 + 225 * (3 * a - b) * h1,
        by linear_combination h2 + 225 * (3 * b - a) * h1⟩

/-- **Centre law.** For a prime `g ≥ 7` and copies `1 ≤ x < y`, `g` strikes both copies iff
either `g ∣ y - x` and `g ∣ 900x² - 1` (same sign), or `g ∣ x + y` and `g ∣ 225(y - x)² - 1`
(opposite signs). -/
theorem centre_law (g x y : ℕ) (hg : g.Prime) (hg7 : 7 ≤ g) (hx : 1 ≤ x) (hxy : x < y) :
    (StrikesCopy g x ∧ StrikesCopy g y) ↔
      (g ∣ y - x ∧ g ∣ 900 * x ^ 2 - 1) ∨ (g ∣ x + y ∧ g ∣ 225 * (y - x) ^ 2 - 1) := by
  have hn30 : ¬ g ∣ 30 := by
    intro h
    have hle := Nat.le_of_dvd (by norm_num) h
    interval_cases g <;> first | omega | norm_num at hg
  have hn4 : ¬ g ∣ 4 := by
    intro h
    have hle := Nat.le_of_dvd (by norm_num) h
    omega
  have : Fact g.Prime := ⟨hg⟩
  have h30 : (30 : ZMod g) ≠ 0 := by
    intro h
    apply hn30
    rw [← ZMod.natCast_eq_zero_iff]
    exact_mod_cast h
  have h4 : (4 : ZMod g) ≠ 0 := by
    intro h
    apply hn4
    rw [← ZMod.natCast_eq_zero_iff]
    exact_mod_cast h
  have hq : 1 ≤ 900 * x ^ 2 := by nlinarith
  have hd : 1 ≤ (y - x) ^ 2 := Nat.one_le_pow _ _ (by omega)
  have hr : 1 ≤ 225 * (y - x) ^ 2 := by linarith
  have eD : g ∣ y - x ↔ (y : ZMod g) - x = 0 := by
    rw [← ZMod.natCast_eq_zero_iff, Nat.cast_sub hxy.le]
  have eS : g ∣ x + y ↔ (x : ZMod g) + y = 0 := by
    rw [← ZMod.natCast_eq_zero_iff, Nat.cast_add]
  have eQ : g ∣ 900 * x ^ 2 - 1 ↔ 900 * (x : ZMod g) ^ 2 - 1 = 0 := by
    rw [← ZMod.natCast_eq_zero_iff]
    push_cast [Nat.cast_sub hq]
    rfl
  have eR : g ∣ 225 * (y - x) ^ 2 - 1 ↔ 225 * ((y : ZMod g) - x) ^ 2 - 1 = 0 := by
    rw [← ZMod.natCast_eq_zero_iff]
    push_cast [Nat.cast_sub hr, Nat.cast_sub hxy.le]
    rfl
  rw [strikesCopy_iff_zmod hx, strikesCopy_iff_zmod (show 1 ≤ y by omega), eD, eS, eQ, eR]
  exact centre_law_field h30 h4 (x : ZMod g) (y : ZMod g)

end RangeLine
