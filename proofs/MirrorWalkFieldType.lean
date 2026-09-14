/-
MirrorWalkFieldType (round 52, 2026-09-14): ONE field type on the landing line, alone.

The field type is (gear `g`, left member, order 2) of the final step up, `L = E + 6h`.
Its rule on the high-gear line, in three parts, each its own lemma:
  1. the class: `g ∣ E + 6h` iff `h ≡ a (mod g)`, where `a` is the class with `6a ≡ -E (mod g)`
     (needs `g` coprime to 6, true for every gear from 5 up);
  2. the line: inside the class, `h = a + g t` and `E + 6h = g (c₀ + 6 t)` with `g c₀ = E + 6a`,
     so the cofactor runs along the column line `c₀ + 6t`;
  3. the shape: the type takes `h` iff the cofactor `c₀ + 6t` is prime (order 2 = `g` times one
     prime); a cofactor divisible by a smaller gear `p` is not this type's kill, and that happens
     exactly on the classes of `t` with `6t ≡ -c₀ (mod p)`.
Nothing here is about the band as a group; the same three lemmas apply to any gear's type.
-/
import MirrorWalkColumn

namespace MirrorWalk

/-- **Part 1, the class.**  With `6a ≡ -E (mod g)` and `g` coprime to 6, gear `g` divides the
left member `E + 6h` iff `h ≡ a (mod g)`. -/
theorem left_class_iff {g : ℕ} {E a h : ℤ} (hco : IsCoprime (g : ℤ) 6)
    (ha : 6 * a ≡ -E [ZMOD g]) : (g : ℤ) ∣ E + 6 * h ↔ h ≡ a [ZMOD g] := by
  constructor
  · intro hd
    have h1 : 6 * h ≡ -E [ZMOD g] := by
      rw [Int.modEq_iff_dvd]
      have : -E - 6 * h = -(E + 6 * h) := by ring
      rw [this, dvd_neg]; exact hd
    have h2 : 6 * h ≡ 6 * a [ZMOD g] := h1.trans ha.symm
    rw [Int.modEq_iff_dvd] at h2 ⊢
    have h3 : (g : ℤ) ∣ (a - h) * 6 := by
      have : (a - h) * 6 = 6 * a - 6 * h := by ring
      rw [this]; exact h2
    exact hco.dvd_of_dvd_mul_right h3
  · intro hh
    have h1 : 6 * h ≡ -E [ZMOD g] := (hh.mul_left 6).trans ha
    rw [Int.modEq_iff_dvd] at h1
    have : -E - 6 * h = -(E + 6 * h) := by ring
    rwa [this, dvd_neg] at h1

/-- **Part 2, the line.**  Inside the class `h = a + g t` the left member is `g (c₀ + 6 t)`. -/
theorem left_member_on_line {g : ℕ} {E a c₀ t : ℤ} (hc : (g : ℤ) * c₀ = E + 6 * a) :
    E + 6 * (a + g * t) = (g : ℤ) * (c₀ + 6 * t) := by
  linear_combination (-1 : ℤ) * hc

/-- **Part 3, the shape.**  The type (g, left, order 2) takes the column `h = a + g t` iff the
cofactor `c₀ + 6 t` is prime: then the left member is `g` times exactly one prime. -/
theorem left_order_two_iff {g : ℕ} {E a c₀ t : ℤ} (hc : (g : ℤ) * c₀ = E + 6 * a) :
    (∃ p : ℤ, Prime p ∧ E + 6 * (a + g * t) = (g : ℤ) * p ∧ p = c₀ + 6 * t) ↔ Prime (c₀ + 6 * t) := by
  constructor
  · rintro ⟨p, hp, -, rfl⟩; exact hp
  · intro hp; exact ⟨c₀ + 6 * t, hp, left_member_on_line hc, rfl⟩

/-- **Part 3, the smaller gear.**  A smaller gear `p` takes the cofactor exactly on the classes
`6 t ≡ -c₀ (mod p)`; on those `t` the kill is `p`'s, not this type's. -/
theorem cofactor_taken_by_iff {p : ℕ} {c₀ t : ℤ} :
    (p : ℤ) ∣ c₀ + 6 * t ↔ 6 * t ≡ -c₀ [ZMOD p] := by
  rw [Int.modEq_iff_dvd]
  have : -c₀ - 6 * t = -(c₀ + 6 * t) := by ring
  rw [this, dvd_neg]

/-- The order-2 shape and a smaller divisor exclude each other: a prime cofactor at least `g`
has no divisor `p` with `2 ≤ p < g`. -/
theorem order_two_excludes_smaller {g p : ℕ} {c : ℤ} (hp : 2 ≤ p) (hpg : p < g) (hgc : (g : ℤ) ≤ c)
    (hc : Prime c) : ¬ (p : ℤ) ∣ c := by
  rintro ⟨k, hk⟩
  have hpos : (0 : ℤ) < p := by exact_mod_cast (by omega : 0 < p)
  have hpc : (p : ℤ) < c := lt_of_lt_of_le (by exact_mod_cast hpg) hgc
  have hcd : c ∣ (p : ℤ) * k := ⟨1, by rw [← hk]; ring⟩
  rcases hc.dvd_or_dvd hcd with h | h
  · exact absurd (Int.le_of_dvd hpos h) (not_le.mpr hpc)
  · obtain ⟨m, hm⟩ := h
    have hc0 : c ≠ 0 := hc.ne_zero
    have e : c * 1 = c * ((p : ℤ) * m) := by
      calc c * 1 = c := mul_one c
        _ = (p : ℤ) * k := hk
        _ = (p : ℤ) * (c * m) := by rw [hm]
        _ = c * ((p : ℤ) * m) := by ring
    have h1 : (p : ℤ) * m = 1 := (mul_left_cancel₀ hc0 e).symm
    have hp1 : (p : ℤ) ∣ 1 := ⟨m, h1.symm⟩
    have h2 := Int.le_of_dvd one_pos hp1
    have h3 : (2 : ℤ) ≤ p := by exact_mod_cast hp
    omega

end MirrorWalk
